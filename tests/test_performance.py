"""Tests for performance optimization features."""

import pytest
import json
import gzip
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import Future

from src.aws_cost_cli.performance import (
    PerformanceMonitor,
    ParallelQueryExecutor,
    QueryPaginator,
    CompressedCacheManager,
    PerformanceOptimizedClient,
    QueryMetrics
)
from src.aws_cost_cli.models import (
    QueryParameters,
    CostData,
    CostResult,
    CostAmount,
    TimePeriod,
    TimePeriodGranularity,
    MetricType
)
from src.aws_cost_cli.exceptions import AWSAPIError, CacheError


class TestPerformanceMonitor:
    """Test performance monitoring functionality."""
    
    def test_start_and_end_query(self):
        """Test starting and ending query tracking."""
        monitor = PerformanceMonitor()
        
        # Start query
        metrics = monitor.start_query("test_query")
        assert metrics.query_id == "test_query"
        assert metrics.start_time is not None
        assert metrics.end_time is None
        
        # End query
        final_metrics = monitor.end_query("test_query", api_calls_made=2, cache_hit=True)
        assert final_metrics is not None
        assert final_metrics.end_time is not None
        assert final_metrics.duration_ms is not None
        assert final_metrics.api_calls_made == 2
        assert final_metrics.cache_hit is True
    
    def test_update_query_metrics(self):
        """Test updating metrics for active query."""
        monitor = PerformanceMonitor()
        
        # Start query
        monitor.start_query("test_query")
        
        # Update metrics
        monitor.update_query_metrics("test_query", data_points_returned=100)
        
        # Check metrics were updated
        metrics = monitor.active_queries["test_query"]
        assert metrics.data_points_returned == 100
    
    def test_compression_ratio_calculation(self):
        """Test compression ratio calculation."""
        monitor = PerformanceMonitor()
        
        # Start and end query with compression stats
        monitor.start_query("test_query")
        final_metrics = monitor.end_query(
            "test_query",
            compressed_cache_size=500,
            uncompressed_cache_size=1000
        )
        
        assert final_metrics.compression_ratio == 0.5
    
    def test_performance_summary_no_data(self):
        """Test performance summary with no data."""
        monitor = PerformanceMonitor()
        summary = monitor.get_performance_summary(24)
        
        assert "error" in summary or "message" in summary


class TestParallelQueryExecutor:
    """Test parallel query execution."""
    
    def test_split_large_query(self):
        """Test splitting large time range queries."""
        executor = ParallelQueryExecutor()
        
        # Create query with large time range
        start_date = datetime(2025, 1, 1)
        end_date = datetime(2025, 12, 31)  # 365 days
        params = QueryParameters(
            time_period=TimePeriod(start=start_date, end=end_date),
            granularity=TimePeriodGranularity.DAILY
        )
        
        # Split into 90-day chunks
        chunks = executor.split_large_query(params, max_days_per_chunk=90)
        
        # Should create multiple chunks
        assert len(chunks) > 1
        
        # Each chunk should be <= 90 days
        for chunk in chunks:
            days = (chunk.time_period.end - chunk.time_period.start).days
            assert days <= 90
    
    def test_split_small_query(self):
        """Test that small queries are not split."""
        executor = ParallelQueryExecutor()
        
        # Create query with small time range
        start_date = datetime(2025, 7, 1)
        end_date = datetime(2025, 7, 31)  # 30 days
        params = QueryParameters(
            time_period=TimePeriod(start=start_date, end=end_date),
            granularity=TimePeriodGranularity.DAILY
        )
        
        # Should not split
        chunks = executor.split_large_query(params, max_days_per_chunk=90)
        assert len(chunks) == 1
        assert chunks[0] == params
    
    @patch('concurrent.futures.ThreadPoolExecutor')
    def test_execute_parallel_queries(self, mock_executor_class):
        """Test parallel query execution."""
        # Mock the executor and futures
        mock_executor = MagicMock()
        mock_executor_class.return_value.__enter__.return_value = mock_executor
        
        # Create mock futures
        mock_future1 = MagicMock()
        mock_future2 = MagicMock()
        mock_executor.submit.side_effect = [mock_future1, mock_future2]
        
        # Mock as_completed to return futures
        with patch('concurrent.futures.as_completed') as mock_as_completed:
            mock_as_completed.return_value = [mock_future1, mock_future2]
            
            # Mock future results
            mock_cost_data1 = Mock(spec=CostData)
            mock_cost_data2 = Mock(spec=CostData)
            mock_future1.result.return_value = (0, mock_cost_data1)
            mock_future2.result.return_value = (1, mock_cost_data2)
            
            executor = ParallelQueryExecutor()
            mock_aws_client = Mock()
            
            queries = [
                {'params': Mock()},
                {'params': Mock()}
            ]
            
            results = executor.execute_parallel_queries(queries, mock_aws_client)
            
            assert len(results) == 2
            assert results[0] == mock_cost_data1
            assert results[1] == mock_cost_data2
    
    def test_merge_cost_data(self):
        """Test merging multiple CostData objects."""
        executor = ParallelQueryExecutor()
        
        # Create test cost data
        cost_data1 = CostData(
            results=[],
            time_period=TimePeriod(
                start=datetime(2025, 1, 1),
                end=datetime(2025, 1, 31)
            ),
            total_cost=CostAmount(amount=100.0),
            currency="USD"
        )
        
        cost_data2 = CostData(
            results=[],
            time_period=TimePeriod(
                start=datetime(2025, 2, 1),
                end=datetime(2025, 2, 28)
            ),
            total_cost=CostAmount(amount=200.0),
            currency="USD"
        )
        
        merged = executor.merge_cost_data([cost_data1, cost_data2])
        
        assert float(merged.total_cost.amount) == 300.0
        assert merged.time_period.start == datetime(2025, 1, 1)
        assert merged.time_period.end == datetime(2025, 2, 28)
    
    def test_merge_empty_list_raises_error(self):
        """Test that merging empty list raises error."""
        executor = ParallelQueryExecutor()
        
        with pytest.raises(ValueError, match="Cannot merge empty list"):
            executor.merge_cost_data([])
    
    def test_merge_all_errors_raises_error(self):
        """Test that merging all errors raises error."""
        executor = ParallelQueryExecutor()
        
        errors = [Exception("Error 1"), Exception("Error 2")]
        
        with pytest.raises(AWSAPIError, match="All parallel queries failed"):
            executor.merge_cost_data(errors)


class TestQueryPaginator:
    """Test query pagination functionality."""
    
    def test_paginate_time_series_daily(self):
        """Test paginating daily time series."""
        paginator = QueryPaginator(page_size=7)  # 7 days per page
        
        # Create query for 20 days
        start_date = datetime(2025, 1, 1)
        end_date = datetime(2025, 1, 21)
        params = QueryParameters(
            time_period=TimePeriod(start=start_date, end=end_date),
            granularity=TimePeriodGranularity.DAILY
        )
        
        # Mock AWS client
        mock_aws_client = Mock()
        mock_cost_data = Mock(spec=CostData)
        mock_aws_client.get_cost_and_usage.return_value = mock_cost_data
        
        pages = paginator.paginate_time_series(params, mock_aws_client)
        
        # Should create multiple pages
        assert len(pages) > 1
        
        # Each page should be the mock cost data
        for page in pages:
            assert page == mock_cost_data
    
    def test_paginate_no_time_period(self):
        """Test pagination with no time period."""
        paginator = QueryPaginator()
        
        params = QueryParameters()  # No time period
        
        mock_aws_client = Mock()
        mock_cost_data = Mock(spec=CostData)
        mock_aws_client.get_cost_and_usage.return_value = mock_cost_data
        
        pages = paginator.paginate_time_series(params, mock_aws_client)
        
        # Should return single page
        assert len(pages) == 1
        assert pages[0] == mock_cost_data
    
    def test_get_page_summary(self):
        """Test getting page summary."""
        paginator = QueryPaginator()
        
        # Create mock pages
        page1 = Mock(spec=CostData)
        page1.results = [Mock(), Mock()]  # 2 results
        page1.total_cost = Mock()
        page1.total_cost.amount = 100.0
        
        page2 = Mock(spec=CostData)
        page2.results = [Mock()]  # 1 result
        page2.total_cost = Mock()
        page2.total_cost.amount = 50.0
        
        # Add error page
        error_page = Mock(spec=CostData)
        error_page.results = []
        error_page.total_cost = Mock()
        error_page.total_cost.amount = 0.0
        error_page.error = "Test error"
        
        pages = [page1, page2, error_page]
        summary = paginator.get_page_summary(pages)
        
        assert summary["total_pages"] == 3
        assert summary["total_data_points"] == 3
        assert summary["total_cost"] == 150.0
        assert summary["error_pages"] == 1
        assert summary["success_rate"] == 2/3


class TestCompressedCacheManager:
    """Test compressed cache management."""
    
    def test_compress_and_decompress_data(self):
        """Test data compression and decompression."""
        mock_base_cache = Mock()
        compressed_cache = CompressedCacheManager(mock_base_cache)
        
        # Use larger test data that will actually compress well
        test_data = "This is test data that should be compressed. " * 100
        
        # Test compression
        compressed = compressed_cache._compress_data(test_data)
        assert isinstance(compressed, bytes)
        assert len(compressed) < len(test_data.encode('utf-8'))
        
        # Test decompression
        decompressed = compressed_cache._decompress_data(compressed)
        assert decompressed == test_data
    
    def test_cache_data_compressed(self, tmp_path):
        """Test caching data with compression."""
        # Create mock base cache manager
        mock_base_cache = Mock()
        mock_base_cache.cache_dir = tmp_path
        mock_base_cache.default_ttl = 3600
        mock_base_cache._get_cache_file_path.return_value = tmp_path / "test.gz"
        
        # Create larger serialized data that will compress well
        large_data = {"test": "data" * 1000, "results": [{"item": i} for i in range(100)]}
        mock_base_cache._serialize_cost_data.return_value = large_data
        
        compressed_cache = CompressedCacheManager(mock_base_cache)
        
        # Create test cost data
        cost_data = Mock(spec=CostData)
        
        # Cache data
        result = compressed_cache.cache_data_compressed("test_key", cost_data)
        
        assert result["success"] is True
        assert "compression_ratio" in result
        assert "space_saved_percent" in result
        assert result["compression_ratio"] < 1.0  # Should be compressed
    
    def test_get_cached_data_compressed(self, tmp_path):
        """Test retrieving compressed cached data."""
        # Create mock base cache manager
        mock_base_cache = Mock()
        mock_base_cache.cache_dir = tmp_path
        mock_base_cache.default_ttl = 3600
        
        cache_file = tmp_path / "test.gz"
        mock_base_cache._get_cache_file_path.return_value = cache_file
        mock_base_cache._is_cache_valid.return_value = True
        
        # Mock deserialization
        mock_cost_data = Mock(spec=CostData)
        mock_base_cache._deserialize_cost_data.return_value = mock_cost_data
        
        compressed_cache = CompressedCacheManager(mock_base_cache)
        
        # Create test compressed cache file
        test_data = {"test": "data"}
        metadata = {
            "compressed": True,
            "cached_at": datetime.now().isoformat(),
            "ttl": 3600,
            "query_hash": "test_key"
        }
        
        with open(cache_file, 'wb') as f:
            # Write metadata
            f.write((json.dumps(metadata) + '\n').encode('utf-8'))
            # Write compressed data
            compressed_data = gzip.compress(json.dumps(test_data).encode('utf-8'))
            f.write(compressed_data)
        
        # Retrieve cached data
        result = compressed_cache.get_cached_data_compressed("test_key")
        
        assert result == mock_cost_data
    
    def test_get_compression_stats(self, tmp_path):
        """Test getting compression statistics."""
        # Create mock base cache manager
        mock_base_cache = Mock()
        mock_base_cache.cache_dir = tmp_path
        
        compressed_cache = CompressedCacheManager(mock_base_cache)
        
        # Create test compressed cache files
        for i in range(3):
            cache_file = tmp_path / f"test{i}.gz"
            metadata = {
                "compressed": True,
                "uncompressed_size": 1000,
                "compressed_size": 500
            }
            
            with open(cache_file, 'wb') as f:
                f.write((json.dumps(metadata) + '\n').encode('utf-8'))
                f.write(b"compressed_data")
        
        stats = compressed_cache.get_compression_stats()
        
        assert stats["total_files"] == 3
        assert stats["compressed_files"] == 3
        assert stats["total_uncompressed_size"] == 3000
        assert stats["total_compressed_size"] == 1500
        assert stats["average_compression_ratio"] == 0.5


class TestPerformanceOptimizedClient:
    """Test performance optimized client."""
    
    def test_initialization(self):
        """Test client initialization."""
        mock_aws_client = Mock()
        mock_cache_manager = Mock()
        
        client = PerformanceOptimizedClient(
            aws_client=mock_aws_client,
            cache_manager=mock_cache_manager,
            enable_parallel=True,
            enable_compression=True,
            enable_monitoring=True
        )
        
        assert client.aws_client == mock_aws_client
        assert client.enable_parallel is True
        assert client.enable_compression is True
        assert client.enable_monitoring is True
        assert hasattr(client, 'monitor')
        assert hasattr(client, 'parallel_executor')
        assert hasattr(client, 'compressed_cache')
    
    def test_get_cost_and_usage_optimized_cache_hit(self):
        """Test optimized query with cache hit."""
        mock_aws_client = Mock()
        mock_cache_manager = Mock()
        
        client = PerformanceOptimizedClient(
            aws_client=mock_aws_client,
            cache_manager=mock_cache_manager,
            enable_compression=True,
            enable_monitoring=True
        )
        
        # Mock cache hit
        mock_cost_data = Mock(spec=CostData)
        with patch.object(client.compressed_cache, 'get_cached_data_compressed', return_value=mock_cost_data):
            params = QueryParameters()
            result = client.get_cost_and_usage_optimized(params)
            
            assert result == mock_cost_data
    
    def test_get_cost_and_usage_optimized_no_cache(self):
        """Test optimized query without cache."""
        mock_aws_client = Mock()
        mock_cache_manager = Mock()
        mock_aws_client.cache_manager = mock_cache_manager
        mock_aws_client.profile = None
        
        client = PerformanceOptimizedClient(
            aws_client=mock_aws_client,
            cache_manager=mock_cache_manager,
            enable_compression=True,
            enable_monitoring=True
        )
        
        # Mock AWS client response
        mock_cost_data = Mock(spec=CostData)
        mock_cost_data.results = [Mock(), Mock()]
        mock_aws_client.get_cost_and_usage.return_value = mock_cost_data
        
        # Mock cache operations
        mock_cache_manager.generate_cache_key.return_value = "test_key"
        
        with patch.object(client.compressed_cache, 'get_cached_data_compressed', return_value=None), \
             patch.object(client.compressed_cache, 'cache_data_compressed', return_value={
                 "compressed_size": 500,
                 "uncompressed_size": 1000
             }):
            
            params = QueryParameters()
            result = client.get_cost_and_usage_optimized(params, force_parallel=False)
            
            assert result == mock_cost_data
    
    def test_execute_parallel_query(self):
        """Test parallel query execution."""
        mock_aws_client = Mock()
        mock_cache_manager = Mock()
        
        client = PerformanceOptimizedClient(
            aws_client=mock_aws_client,
            cache_manager=mock_cache_manager,
            enable_parallel=True,
            enable_monitoring=True
        )
        
        # Mock parallel executor methods
        mock_chunks = [Mock(), Mock()]
        mock_cost_data1 = Mock(spec=CostData)
        mock_cost_data2 = Mock(spec=CostData)
        merged_data = Mock(spec=CostData)
        
        with patch.object(client.parallel_executor, 'split_large_query', return_value=mock_chunks), \
             patch.object(client.parallel_executor, 'execute_parallel_queries', return_value=[mock_cost_data1, mock_cost_data2]), \
             patch.object(client.parallel_executor, 'merge_cost_data', return_value=merged_data):
            
            params = QueryParameters()
            result = client._execute_parallel_query(params, True, "test_query")
            
            assert result == merged_data
    
    def test_get_performance_summary(self):
        """Test getting performance summary."""
        mock_aws_client = Mock()
        mock_cache_manager = Mock()
        
        client = PerformanceOptimizedClient(
            aws_client=mock_aws_client,
            cache_manager=mock_cache_manager,
            enable_monitoring=True,
            enable_compression=True
        )
        
        # Mock performance data
        with patch.object(client.monitor, 'get_performance_summary', return_value={"test": "data"}), \
             patch.object(client.compressed_cache, 'get_compression_stats', return_value={"compression": "stats"}):
            
            summary = client.get_performance_summary()
            
            assert "query_performance" in summary
            assert "compression_stats" in summary
            assert summary["query_performance"] == {"test": "data"}
            assert summary["compression_stats"] == {"compression": "stats"}


class TestIntegration:
    """Integration tests for performance optimizations."""
    
    def test_end_to_end_performance_optimization(self):
        """Test end-to-end performance optimization flow."""
        # This would be a more complex integration test
        # that tests the full flow from CLI to optimized execution
        pass
    
    def test_performance_with_real_cache_manager(self, tmp_path):
        """Test performance optimizations with real cache manager."""
        from src.aws_cost_cli.cache_manager import CacheManager
        
        # Create real cache manager
        cache_manager = CacheManager(cache_dir=str(tmp_path), default_ttl=3600)
        
        # Create mock AWS client
        mock_aws_client = Mock()
        mock_aws_client.cache_manager = cache_manager
        mock_aws_client.profile = None
        
        # Create performance optimized client
        client = PerformanceOptimizedClient(
            aws_client=mock_aws_client,
            cache_manager=cache_manager,
            enable_compression=True,
            enable_monitoring=True
        )
        
        # Test that it initializes correctly
        assert client.compressed_cache is not None
        assert client.monitor is not None