"""Performance optimization utilities for AWS Cost CLI."""

import asyncio
import concurrent.futures
import time
import gzip
import json
import threading
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path

from .models import QueryParameters, CostData, TimePeriod, TimePeriodGranularity
from .exceptions import AWSAPIError, CacheError


@dataclass
class QueryMetrics:
    """Performance metrics for a query."""

    query_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    cache_hit: bool = False
    api_calls_made: int = 0
    data_points_returned: int = 0
    compressed_cache_size: Optional[int] = None
    uncompressed_cache_size: Optional[int] = None
    compression_ratio: Optional[float] = None
    parallel_requests: int = 1
    error: Optional[str] = None


class PerformanceMonitor:
    """Monitor and track query performance metrics."""

    def __init__(self, metrics_file: Optional[str] = None):
        """
        Initialize performance monitor.

        Args:
            metrics_file: Optional file to persist metrics
        """
        self.metrics_file = metrics_file
        self.active_queries: Dict[str, QueryMetrics] = {}
        self._lock = threading.Lock()

    def start_query(self, query_id: str) -> QueryMetrics:
        """Start tracking a query."""
        with self._lock:
            metrics = QueryMetrics(query_id=query_id, start_time=datetime.now())
            self.active_queries[query_id] = metrics
            return metrics

    def end_query(self, query_id: str, **kwargs) -> Optional[QueryMetrics]:
        """End tracking a query and record final metrics."""
        with self._lock:
            if query_id not in self.active_queries:
                return None

            metrics = self.active_queries[query_id]
            metrics.end_time = datetime.now()
            metrics.duration_ms = (
                metrics.end_time - metrics.start_time
            ).total_seconds() * 1000

            # Update metrics with provided values
            for key, value in kwargs.items():
                if hasattr(metrics, key):
                    setattr(metrics, key, value)

            # Calculate compression ratio if both sizes are available
            if metrics.compressed_cache_size and metrics.uncompressed_cache_size:
                metrics.compression_ratio = (
                    metrics.compressed_cache_size / metrics.uncompressed_cache_size
                )

            # Remove from active queries
            del self.active_queries[query_id]

            # Persist metrics if file is configured
            if self.metrics_file:
                self._persist_metrics(metrics)

            return metrics

    def update_query_metrics(self, query_id: str, **kwargs):
        """Update metrics for an active query."""
        with self._lock:
            if query_id in self.active_queries:
                metrics = self.active_queries[query_id]
                for key, value in kwargs.items():
                    if hasattr(metrics, key):
                        setattr(metrics, key, value)

    def _persist_metrics(self, metrics: QueryMetrics):
        """Persist metrics to file."""
        try:
            metrics_data = asdict(metrics)
            # Convert datetime objects to ISO strings
            for key, value in metrics_data.items():
                if isinstance(value, datetime):
                    metrics_data[key] = value.isoformat()

            # Append to metrics file
            with open(self.metrics_file, "a") as f:
                f.write(json.dumps(metrics_data) + "\n")
        except Exception:
            # Don't fail the query if metrics persistence fails
            pass

    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for the last N hours."""
        if not self.metrics_file or not Path(self.metrics_file).exists():
            return {"error": "No metrics data available"}

        cutoff_time = datetime.now() - timedelta(hours=hours)
        metrics_list = []

        try:
            with open(self.metrics_file, "r") as f:
                for line in f:
                    try:
                        metrics_data = json.loads(line.strip())
                        start_time = datetime.fromisoformat(metrics_data["start_time"])
                        if start_time >= cutoff_time:
                            metrics_list.append(metrics_data)
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
        except FileNotFoundError:
            return {"error": "Metrics file not found"}

        if not metrics_list:
            return {"message": f"No queries in the last {hours} hours"}

        # Calculate summary statistics
        durations = [m["duration_ms"] for m in metrics_list if m.get("duration_ms")]
        cache_hits = sum(1 for m in metrics_list if m.get("cache_hit"))
        api_calls = sum(m.get("api_calls_made", 0) for m in metrics_list)
        errors = sum(1 for m in metrics_list if m.get("error"))

        summary = {
            "period_hours": hours,
            "total_queries": len(metrics_list),
            "cache_hit_rate": cache_hits / len(metrics_list) if metrics_list else 0,
            "total_api_calls": api_calls,
            "error_rate": errors / len(metrics_list) if metrics_list else 0,
            "performance": {},
        }

        if durations:
            summary["performance"] = {
                "avg_duration_ms": sum(durations) / len(durations),
                "min_duration_ms": min(durations),
                "max_duration_ms": max(durations),
                "p95_duration_ms": (
                    sorted(durations)[int(len(durations) * 0.95)]
                    if len(durations) > 1
                    else durations[0]
                ),
            }

        return summary


class ParallelQueryExecutor:
    """Execute multiple AWS API queries in parallel."""

    def __init__(self, max_workers: int = 5, timeout: int = 60):
        """
        Initialize parallel query executor.

        Args:
            max_workers: Maximum number of parallel workers
            timeout: Timeout for individual queries in seconds
        """
        self.max_workers = max_workers
        self.timeout = timeout

    def execute_parallel_queries(
        self, queries: List[Dict[str, Any]], aws_client, use_cache: bool = True
    ) -> List[CostData]:
        """
        Execute multiple queries in parallel.

        Args:
            queries: List of query dictionaries with 'params' and optional 'id'
            aws_client: AWS client instance
            use_cache: Whether to use caching

        Returns:
            List of CostData results in the same order as input queries
        """
        results = [None] * len(queries)

        def execute_single_query(index_and_query):
            index, query = index_and_query
            try:
                params = query["params"]
                return index, aws_client.get_cost_and_usage(params, use_cache=use_cache)
            except Exception as e:
                return index, e

        # Use ThreadPoolExecutor for I/O-bound AWS API calls
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            # Submit all queries
            future_to_index = {
                executor.submit(execute_single_query, (i, query)): i
                for i, query in enumerate(queries)
            }

            # Collect results
            for future in concurrent.futures.as_completed(
                future_to_index, timeout=self.timeout
            ):
                try:
                    index, result = future.result()
                    results[index] = result
                except concurrent.futures.TimeoutError:
                    index = future_to_index[future]
                    results[index] = AWSAPIError(
                        f"Query {index} timed out after {self.timeout} seconds"
                    )
                except Exception as e:
                    index = future_to_index[future]
                    results[index] = e

        return results

    def split_large_query(
        self, params: QueryParameters, max_days_per_chunk: int = 90
    ) -> List[QueryParameters]:
        """
        Split a large time range query into smaller chunks for parallel execution.

        Args:
            params: Original query parameters
            max_days_per_chunk: Maximum days per chunk

        Returns:
            List of query parameters for smaller time ranges
        """
        if not params.time_period:
            return [params]

        total_days = (params.time_period.end - params.time_period.start).days

        if total_days <= max_days_per_chunk:
            return [params]

        chunks = []
        current_start = params.time_period.start

        while current_start < params.time_period.end:
            chunk_end = min(
                current_start + timedelta(days=max_days_per_chunk),
                params.time_period.end,
            )

            chunk_params = QueryParameters(
                service=params.service,
                time_period=TimePeriod(start=current_start, end=chunk_end),
                granularity=params.granularity,
                metrics=params.metrics,
                group_by=params.group_by,
                date_range_type=params.date_range_type,
                fiscal_year_start_month=params.fiscal_year_start_month,
                trend_analysis=params.trend_analysis,
                comparison_period=params.comparison_period,
                include_forecast=params.include_forecast,
                forecast_months=params.forecast_months,
                cost_allocation_tags=params.cost_allocation_tags,
            )

            chunks.append(chunk_params)
            current_start = chunk_end

        return chunks

    def merge_cost_data(self, cost_data_list: List[CostData]) -> CostData:
        """
        Merge multiple CostData objects into a single result.

        Args:
            cost_data_list: List of CostData objects to merge

        Returns:
            Merged CostData object
        """
        if not cost_data_list:
            raise ValueError("Cannot merge empty list of cost data")

        if len(cost_data_list) == 1:
            return cost_data_list[0]

        # Filter out any error results
        valid_data = [data for data in cost_data_list if isinstance(data, CostData)]

        if not valid_data:
            raise AWSAPIError("All parallel queries failed")

        # Merge results
        merged_results = []
        total_amount = 0.0

        for cost_data in valid_data:
            merged_results.extend(cost_data.results)
            total_amount += float(cost_data.total_cost.amount)

        # Create overall time period
        start_times = [data.time_period.start for data in valid_data]
        end_times = [data.time_period.end for data in valid_data]

        merged_time_period = TimePeriod(start=min(start_times), end=max(end_times))

        # Use first valid data as template for other fields
        template = valid_data[0]

        return CostData(
            results=merged_results,
            time_period=merged_time_period,
            total_cost=template.total_cost.__class__(
                amount=total_amount, unit=template.total_cost.unit
            ),
            currency=template.currency,
            group_definitions=template.group_definitions,
            trend_data=template.trend_data,  # Note: trend data merging would need special handling
            forecast_data=template.forecast_data,  # Note: forecast data merging would need special handling
        )


class QueryPaginator:
    """Handle pagination for large query results."""

    def __init__(self, page_size: int = 1000):
        """
        Initialize query paginator.

        Args:
            page_size: Number of data points per page
        """
        self.page_size = page_size

    def paginate_time_series(
        self, params: QueryParameters, aws_client, page_size: Optional[int] = None
    ) -> List[CostData]:
        """
        Paginate a time series query by breaking it into smaller time chunks.

        Args:
            params: Query parameters
            aws_client: AWS client instance
            page_size: Override default page size

        Returns:
            List of CostData objects, one per page
        """
        if page_size is None:
            page_size = self.page_size

        if not params.time_period:
            # Single query if no time period specified
            return [aws_client.get_cost_and_usage(params)]

        # Calculate chunk size based on granularity and page size
        if params.granularity == TimePeriodGranularity.HOURLY:
            chunk_hours = page_size
            chunk_delta = timedelta(hours=chunk_hours)
        elif params.granularity == TimePeriodGranularity.DAILY:
            chunk_days = page_size
            chunk_delta = timedelta(days=chunk_days)
        else:  # MONTHLY
            chunk_months = page_size
            chunk_delta = timedelta(days=chunk_months * 30)  # Approximate

        pages = []
        current_start = params.time_period.start

        while current_start < params.time_period.end:
            chunk_end = min(current_start + chunk_delta, params.time_period.end)

            chunk_params = QueryParameters(
                service=params.service,
                time_period=TimePeriod(start=current_start, end=chunk_end),
                granularity=params.granularity,
                metrics=params.metrics,
                group_by=params.group_by,
                date_range_type=params.date_range_type,
                fiscal_year_start_month=params.fiscal_year_start_month,
                trend_analysis=params.trend_analysis,
                comparison_period=params.comparison_period,
                include_forecast=params.include_forecast,
                forecast_months=params.forecast_months,
                cost_allocation_tags=params.cost_allocation_tags,
            )

            try:
                page_data = aws_client.get_cost_and_usage(chunk_params)
                pages.append(page_data)
            except Exception as e:
                # Include error information in the page
                error_data = CostData(
                    results=[],
                    time_period=TimePeriod(start=current_start, end=chunk_end),
                    total_cost=params.metrics[0].__class__(amount=0.0),
                    currency="USD",
                    group_definitions=[],
                )
                error_data.error = str(e)
                pages.append(error_data)

            current_start = chunk_end

        return pages

    def get_page_summary(self, pages: List[CostData]) -> Dict[str, Any]:
        """
        Get summary information about paginated results.

        Args:
            pages: List of paginated CostData objects

        Returns:
            Summary dictionary
        """
        total_pages = len(pages)
        total_data_points = sum(len(page.results) for page in pages)
        total_cost = sum(
            float(page.total_cost.amount)
            for page in pages
            if not hasattr(page, "error")
        )
        error_pages = sum(1 for page in pages if hasattr(page, "error"))

        return {
            "total_pages": total_pages,
            "total_data_points": total_data_points,
            "total_cost": total_cost,
            "error_pages": error_pages,
            "success_rate": (
                (total_pages - error_pages) / total_pages if total_pages > 0 else 0
            ),
        }


class CompressedCacheManager:
    """Enhanced cache manager with compression support."""

    def __init__(self, base_cache_manager, compression_level: int = 6):
        """
        Initialize compressed cache manager.

        Args:
            base_cache_manager: Base cache manager instance
            compression_level: Gzip compression level (1-9)
        """
        self.base_cache = base_cache_manager
        self.compression_level = compression_level

    def _compress_data(self, data: str) -> bytes:
        """Compress string data using gzip."""
        return gzip.compress(data.encode("utf-8"), compresslevel=self.compression_level)

    def _decompress_data(self, compressed_data: bytes) -> str:
        """Decompress gzip data to string."""
        return gzip.decompress(compressed_data).decode("utf-8")

    def cache_data_compressed(
        self, cache_key: str, data: CostData, ttl: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Cache data with compression and return compression metrics.

        Args:
            cache_key: Cache key
            data: Data to cache
            ttl: Time to live

        Returns:
            Dictionary with compression metrics
        """
        # Serialize data to JSON
        serialized_data = json.dumps(
            self.base_cache._serialize_cost_data(data), default=str
        )
        uncompressed_size = len(serialized_data.encode("utf-8"))

        # Compress data
        compressed_data = self._compress_data(serialized_data)
        compressed_size = len(compressed_data)

        # Calculate compression ratio
        compression_ratio = (
            compressed_size / uncompressed_size if uncompressed_size > 0 else 1.0
        )

        # Store compressed data
        cache_file = self.base_cache._get_cache_file_path(cache_key + ".gz")

        try:
            cache_entry = {
                "compressed": True,
                "cached_at": datetime.now().isoformat(),
                "ttl": ttl or self.base_cache.default_ttl,
                "query_hash": cache_key,
                "uncompressed_size": uncompressed_size,
                "compressed_size": compressed_size,
                "compression_ratio": compression_ratio,
            }

            # Write compressed data and metadata
            with open(cache_file, "wb") as f:
                # Write metadata as JSON header
                metadata_json = json.dumps(cache_entry) + "\n"
                f.write(metadata_json.encode("utf-8"))
                # Write compressed data
                f.write(compressed_data)

            return {
                "success": True,
                "uncompressed_size": uncompressed_size,
                "compressed_size": compressed_size,
                "compression_ratio": compression_ratio,
                "space_saved_bytes": uncompressed_size - compressed_size,
                "space_saved_percent": (1 - compression_ratio) * 100,
            }

        except Exception as e:
            raise CacheError(f"Failed to write compressed cache file: {e}")

    def get_cached_data_compressed(self, cache_key: str) -> Optional[CostData]:
        """
        Retrieve compressed cached data.

        Args:
            cache_key: Cache key

        Returns:
            Cached CostData if available and valid
        """
        cache_file = self.base_cache._get_cache_file_path(cache_key + ".gz")

        if not self.base_cache._is_cache_valid(cache_file, self.base_cache.default_ttl):
            return None

        try:
            with open(cache_file, "rb") as f:
                # Read metadata line
                metadata_line = f.readline().decode("utf-8").strip()
                metadata = json.loads(metadata_line)

                # Read compressed data
                compressed_data = f.read()

                # Decompress data
                decompressed_data = self._decompress_data(compressed_data)

                # Parse JSON and reconstruct CostData
                data_dict = json.loads(decompressed_data)
                return self.base_cache._deserialize_cost_data(data_dict)

        except Exception:
            # If decompression fails, remove corrupted cache file
            try:
                cache_file.unlink()
            except OSError:
                pass
            return None

    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics for all cached files."""
        stats = {
            "total_files": 0,
            "compressed_files": 0,
            "total_uncompressed_size": 0,
            "total_compressed_size": 0,
            "total_space_saved": 0,
            "average_compression_ratio": 0.0,
        }

        try:
            compression_ratios = []

            for cache_file in self.base_cache.cache_dir.glob("*.gz"):
                try:
                    stats["total_files"] += 1

                    with open(cache_file, "rb") as f:
                        # Read metadata
                        metadata_line = f.readline().decode("utf-8").strip()
                        metadata = json.loads(metadata_line)

                        if metadata.get("compressed"):
                            stats["compressed_files"] += 1
                            uncompressed_size = metadata.get("uncompressed_size", 0)
                            compressed_size = metadata.get("compressed_size", 0)

                            stats["total_uncompressed_size"] += uncompressed_size
                            stats["total_compressed_size"] += compressed_size

                            if uncompressed_size > 0:
                                ratio = compressed_size / uncompressed_size
                                compression_ratios.append(ratio)

                except (json.JSONDecodeError, KeyError, OSError):
                    continue

            stats["total_space_saved"] = (
                stats["total_uncompressed_size"] - stats["total_compressed_size"]
            )

            if compression_ratios:
                stats["average_compression_ratio"] = sum(compression_ratios) / len(
                    compression_ratios
                )
                stats["space_saved_percent"] = (
                    1 - stats["average_compression_ratio"]
                ) * 100

        except OSError:
            pass

        return stats


class PerformanceOptimizedClient:
    """AWS client wrapper with performance optimizations."""

    def __init__(
        self,
        aws_client,
        cache_manager=None,
        enable_parallel: bool = True,
        enable_compression: bool = True,
        enable_monitoring: bool = True,
    ):
        """
        Initialize performance optimized client.

        Args:
            aws_client: Base AWS client
            cache_manager: Cache manager instance
            enable_parallel: Enable parallel query execution
            enable_compression: Enable cache compression
            enable_monitoring: Enable performance monitoring
        """
        self.aws_client = aws_client
        self.enable_parallel = enable_parallel
        self.enable_compression = enable_compression
        self.enable_monitoring = enable_monitoring

        # Initialize performance components
        if enable_monitoring:
            self.monitor = PerformanceMonitor()

        if enable_parallel:
            self.parallel_executor = ParallelQueryExecutor()
            self.paginator = QueryPaginator()

        if enable_compression and cache_manager:
            self.compressed_cache = CompressedCacheManager(cache_manager)
        else:
            self.compressed_cache = None

    def get_cost_and_usage_optimized(
        self,
        params: QueryParameters,
        use_cache: bool = True,
        force_parallel: bool = False,
        max_chunk_days: int = 90,
    ) -> CostData:
        """
        Get cost data with performance optimizations.

        Args:
            params: Query parameters
            use_cache: Whether to use caching
            force_parallel: Force parallel execution even for small queries
            max_chunk_days: Maximum days per parallel chunk

        Returns:
            CostData with performance optimizations applied
        """
        query_id = f"query_{int(time.time() * 1000)}"

        # Start monitoring
        if self.enable_monitoring:
            metrics = self.monitor.start_query(query_id)

        try:
            # Check if query should be parallelized
            should_parallelize = force_parallel
            if not should_parallelize and params.time_period:
                total_days = (params.time_period.end - params.time_period.start).days
                should_parallelize = total_days > max_chunk_days

            # Try compressed cache first
            if use_cache and self.compressed_cache:
                cache_key = self.aws_client.cache_manager.generate_cache_key(
                    params, self.aws_client.profile or "default"
                )
                cached_data = self.compressed_cache.get_cached_data_compressed(
                    cache_key
                )
                if cached_data:
                    if self.enable_monitoring:
                        self.monitor.end_query(
                            query_id, cache_hit=True, api_calls_made=0
                        )
                    return cached_data

            # Execute query (parallel or single)
            if should_parallelize and self.enable_parallel:
                result = self._execute_parallel_query(params, use_cache, query_id)
            else:
                result = self.aws_client.get_cost_and_usage(params, use_cache=use_cache)
                if self.enable_monitoring:
                    self.monitor.update_query_metrics(query_id, api_calls_made=1)

            # Cache with compression if enabled
            if use_cache and self.compressed_cache:
                cache_key = self.aws_client.cache_manager.generate_cache_key(
                    params, self.aws_client.profile or "default"
                )
                compression_stats = self.compressed_cache.cache_data_compressed(
                    cache_key, result
                )

                if self.enable_monitoring:
                    self.monitor.update_query_metrics(
                        query_id,
                        compressed_cache_size=compression_stats.get("compressed_size"),
                        uncompressed_cache_size=compression_stats.get(
                            "uncompressed_size"
                        ),
                    )

            # End monitoring
            if self.enable_monitoring:
                data_points = len(result.results) if result.results else 0
                self.monitor.end_query(
                    query_id, data_points_returned=data_points, cache_hit=False
                )

            return result

        except Exception as e:
            if self.enable_monitoring:
                self.monitor.end_query(query_id, error=str(e))
            raise

    def _execute_parallel_query(
        self, params: QueryParameters, use_cache: bool, query_id: str
    ) -> CostData:
        """Execute query using parallel processing."""
        # Split query into chunks
        chunks = self.parallel_executor.split_large_query(params)

        if len(chunks) == 1:
            # No need for parallel execution
            result = self.aws_client.get_cost_and_usage(chunks[0], use_cache=use_cache)
            if self.enable_monitoring:
                self.monitor.update_query_metrics(
                    query_id, api_calls_made=1, parallel_requests=1
                )
            return result

        # Execute chunks in parallel
        query_dicts = [{"params": chunk} for chunk in chunks]
        results = self.parallel_executor.execute_parallel_queries(
            query_dicts, self.aws_client, use_cache=use_cache
        )

        # Filter out errors and merge results
        valid_results = [r for r in results if isinstance(r, CostData)]

        if not valid_results:
            # All queries failed
            errors = [str(r) for r in results if isinstance(r, Exception)]
            raise AWSAPIError(f"All parallel queries failed: {'; '.join(errors[:3])}")

        # Update monitoring
        if self.enable_monitoring:
            self.monitor.update_query_metrics(
                query_id, api_calls_made=len(chunks), parallel_requests=len(chunks)
            )

        # Merge results
        return self.parallel_executor.merge_cost_data(valid_results)

    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary."""
        summary = {}

        if self.enable_monitoring:
            summary["query_performance"] = self.monitor.get_performance_summary(hours)

        if self.compressed_cache:
            summary["compression_stats"] = self.compressed_cache.get_compression_stats()

        return summary
