"""AWS client management for Cost Explorer API."""

import boto3
import time
from botocore.exceptions import (
    ClientError,
    NoCredentialsError,
    PartialCredentialsError,
    ProfileNotFound,
    EndpointConnectionError,
    ConnectTimeoutError,
    ReadTimeoutError,
)
from botocore.config import Config
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta, timezone

from .models import (
    QueryParameters,
    CostData,
    CostResult,
    CostAmount,
    TimePeriod,
    Group,
    TimePeriodGranularity,
    MetricType,
    TrendAnalysisType,
)
from .exceptions import (
    AWSCredentialsError,
    AWSAPIError,
    CacheError,
    handle_aws_client_error,
    handle_network_error,
    is_retryable_error,
    get_retry_delay,
)
from .performance import PerformanceOptimizedClient


class CredentialManager:
    """Manages AWS credentials and profiles."""

    def __init__(self):
        self.session = boto3.Session()

    def get_available_profiles(self) -> List[str]:
        """Get list of available AWS profiles."""
        try:
            return self.session.available_profiles
        except Exception:
            return []

    def validate_credentials(self, profile: Optional[str] = None) -> bool:
        """Validate AWS credentials for the specified profile."""
        try:
            session = (
                boto3.Session(profile_name=profile) if profile else boto3.Session()
            )
            sts_client = session.client("sts")
            sts_client.get_caller_identity()
            return True
        except (NoCredentialsError, PartialCredentialsError):
            return False
        except ProfileNotFound:
            return False
        except ClientError as e:
            # If we get a client error but not credential-related, credentials are valid
            error_code = e.response.get("Error", {}).get("Code", "")
            return error_code not in ["InvalidUserID.NotFound", "AccessDenied"]
        except (EndpointConnectionError, ConnectTimeoutError, ReadTimeoutError):
            # Network errors don't indicate invalid credentials
            return True
        except Exception:
            return False

    def get_caller_identity(self, profile: Optional[str] = None) -> Dict[str, Any]:
        """Get AWS caller identity information."""
        try:
            session = (
                boto3.Session(profile_name=profile) if profile else boto3.Session()
            )
            sts_client = session.client("sts")
            return sts_client.get_caller_identity()
        except (NoCredentialsError, PartialCredentialsError):
            raise AWSCredentialsError(profile=profile)
        except ProfileNotFound:
            raise AWSCredentialsError(
                f"AWS profile '{profile}' not found", profile=profile
            )
        except ClientError as e:
            raise handle_aws_client_error(e, "getting caller identity")
        except (EndpointConnectionError, ConnectTimeoutError, ReadTimeoutError) as e:
            raise handle_network_error(e, "getting caller identity")
        except Exception as e:
            raise AWSAPIError(f"Failed to get caller identity: {e}")


class AWSCostClient:
    """Client for AWS Cost Explorer API operations."""

    REQUIRED_PERMISSIONS = [
        "ce:GetCostAndUsage",
        "ce:GetDimensionValues",
        "ce:GetReservationCoverage",
        "ce:GetReservationPurchaseRecommendation",
        "ce:GetReservationUtilization",
        "ce:GetUsageReport",
    ]

    def __init__(
        self,
        profile: Optional[str] = None,
        region: str = "us-east-1",
        cache_manager=None,
    ):
        """Initialize AWS Cost Explorer client."""
        self.profile = profile
        self.region = region
        self.session = (
            boto3.Session(profile_name=profile) if profile else boto3.Session()
        )
        self.client = self._create_optimized_client(region)
        self.credential_manager = CredentialManager()
        self.cache_manager = cache_manager

    def _create_optimized_client(self, region: str):
        """Create boto3 client with optimized configuration for production."""
        config = Config(
            # Connection pooling and retry configuration
            retries={"max_attempts": 3, "mode": "adaptive"},
            # Connection pooling - reuse connections for better performance
            max_pool_connections=50,
            # Timeout configuration
            connect_timeout=60,
            read_timeout=60,
            # Enable TCP keepalive for long-running connections
            tcp_keepalive=True,
        )
        return self.session.client("ce", region_name=region, config=config)

    def validate_permissions(self) -> bool:
        """Validate that the current credentials have necessary permissions."""
        try:
            # Test with a minimal API call
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=1)

            self.client.get_cost_and_usage(
                TimePeriod={
                    "Start": start_date.strftime("%Y-%m-%d"),
                    "End": end_date.strftime("%Y-%m-%d"),
                },
                Granularity="DAILY",
                Metrics=["BlendedCost"],
            )
            return True
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in ["AccessDenied", "UnauthorizedOperation"]:
                return False
            # Other errors might be due to data availability, not permissions
            return True
        except (EndpointConnectionError, ConnectTimeoutError, ReadTimeoutError):
            # Network errors don't indicate permission issues
            return True
        except Exception:
            return False

    def get_cost_and_usage(
        self, params: QueryParameters, use_cache: bool = True
    ) -> CostData:
        """Retrieve cost and usage data from AWS Cost Explorer."""
        # Try cache first if enabled and cache manager is available
        if use_cache and self.cache_manager:
            try:
                cache_key = self.cache_manager.generate_cache_key(
                    params, self.profile or "default"
                )
                cached_data = self.cache_manager.get_cached_data(cache_key)
                if cached_data:
                    return cached_data
            except CacheError:
                # Continue without cache if cache fails
                pass

        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                # Build the API request
                request_params = self._build_cost_request(params)

                # Make the API call
                response = self.client.get_cost_and_usage(**request_params)

                # Parse the response
                cost_data = self._parse_cost_response(response, params)

                # Cache the result if cache manager is available
                if self.cache_manager:
                    try:
                        cache_key = self.cache_manager.generate_cache_key(
                            params, self.profile or "default"
                        )
                        self.cache_manager.cache_data(cache_key, cost_data)
                    except CacheError:
                        # Continue without caching if cache fails
                        pass

                return cost_data

            except ClientError as e:
                aws_error = handle_aws_client_error(e, "retrieving cost data")
                last_error = aws_error

                # Check if error is retryable
                if is_retryable_error(aws_error) and attempt < max_retries - 1:
                    delay = get_retry_delay(attempt)
                    time.sleep(delay)
                    continue
                else:
                    raise aws_error

            except (
                EndpointConnectionError,
                ConnectTimeoutError,
                ReadTimeoutError,
            ) as e:
                network_error = handle_network_error(e, "retrieving cost data")
                last_error = network_error

                if attempt < max_retries - 1:
                    delay = get_retry_delay(attempt)
                    time.sleep(delay)
                    continue
                else:
                    raise network_error

            except Exception as e:
                last_error = AWSAPIError(f"Failed to retrieve cost data: {e}")

                if attempt < max_retries - 1:
                    delay = get_retry_delay(attempt)
                    time.sleep(delay)
                    continue
                else:
                    raise last_error

        # This should never be reached, but just in case
        if last_error:
            raise last_error
        else:
            raise AWSAPIError("Failed to retrieve cost data after multiple retries")

    def get_dimension_values(
        self, dimension: str, time_period: Optional[TimePeriod] = None
    ) -> List[str]:
        """Get available values for a specific dimension (e.g., SERVICE, INSTANCE_TYPE)."""
        max_retries = 3

        for attempt in range(max_retries):
            try:
                # Default to last 30 days if no time period specified
                if not time_period:
                    end_date = datetime.now().date()
                    start_date = end_date - timedelta(days=30)
                    time_period = TimePeriod(
                        start=datetime.combine(start_date, datetime.min.time()),
                        end=datetime.combine(end_date, datetime.min.time()),
                    )

                response = self.client.get_dimension_values(
                    TimePeriod={
                        "Start": time_period.start.strftime("%Y-%m-%d"),
                        "End": time_period.end.strftime("%Y-%m-%d"),
                    },
                    Dimension=dimension,
                )

                return [item["Value"] for item in response.get("DimensionValues", [])]

            except ClientError as e:
                aws_error = handle_aws_client_error(
                    e, f"getting dimension values for {dimension}"
                )

                if is_retryable_error(aws_error) and attempt < max_retries - 1:
                    delay = get_retry_delay(attempt)
                    time.sleep(delay)
                    continue
                else:
                    raise aws_error

            except (
                EndpointConnectionError,
                ConnectTimeoutError,
                ReadTimeoutError,
            ) as e:
                network_error = handle_network_error(
                    e, f"getting dimension values for {dimension}"
                )

                if attempt < max_retries - 1:
                    delay = get_retry_delay(attempt)
                    time.sleep(delay)
                    continue
                else:
                    raise network_error

            except Exception as e:
                if attempt < max_retries - 1:
                    delay = get_retry_delay(attempt)
                    time.sleep(delay)
                    continue
                else:
                    raise AWSAPIError(
                        f"Failed to get dimension values for {dimension}: {e}"
                    )

    def _build_cost_request(self, params: QueryParameters) -> Dict[str, Any]:
        """Build the Cost Explorer API request from query parameters."""
        # Default time period if not specified (last 30 days)
        if not params.time_period:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=30)
            time_period = TimePeriod(
                start=datetime.combine(start_date, datetime.min.time()),
                end=datetime.combine(end_date, datetime.min.time()),
            )
        else:
            time_period = params.time_period

        request = {
            "TimePeriod": {
                "Start": time_period.start.strftime("%Y-%m-%d"),
                "End": time_period.end.strftime("%Y-%m-%d"),
            },
            "Granularity": params.granularity.value,
            "Metrics": [metric.value for metric in params.metrics],
        }

        # Add service filter if specified
        if params.service:
            request["Filter"] = {
                "Dimensions": {"Key": "SERVICE", "Values": [params.service]}
            }

        # Add group by if specified
        if params.group_by:
            request["GroupBy"] = [
                {"Type": "DIMENSION", "Key": group} for group in params.group_by
            ]

        return request

    def _parse_cost_response(
        self, response: Dict[str, Any], params: QueryParameters
    ) -> CostData:
        """Parse AWS Cost Explorer API response into CostData model."""
        results = []
        total_amount = 0.0

        for result_item in response.get("ResultsByTime", []):
            # Parse time period
            time_period = TimePeriod(
                start=datetime.strptime(result_item["TimePeriod"]["Start"], "%Y-%m-%d"),
                end=datetime.strptime(result_item["TimePeriod"]["End"], "%Y-%m-%d"),
            )

            # Parse total cost
            total_cost_data = result_item.get("Total", {})
            total_cost = CostAmount(amount=0.0)

            if total_cost_data:
                # Use the first metric for total cost
                metric_key = list(total_cost_data.keys())[0]
                amount_str = total_cost_data[metric_key]["Amount"]
                unit = total_cost_data[metric_key]["Unit"]
                total_cost = CostAmount(amount=float(amount_str), unit=unit)
                total_amount += float(amount_str)

            # Parse groups
            groups = []
            for group_item in result_item.get("Groups", []):
                group_keys = group_item.get("Keys", [])
                group_metrics = {}

                for metric_name, metric_data in group_item.get("Metrics", {}).items():
                    amount = float(metric_data["Amount"])
                    unit = metric_data["Unit"]
                    group_metrics[metric_name] = CostAmount(amount=amount, unit=unit)

                groups.append(Group(keys=group_keys, metrics=group_metrics))

            # Create cost result
            cost_result = CostResult(
                time_period=time_period,
                total=total_cost,
                groups=groups,
                estimated=result_item.get("Estimated", False),
            )
            results.append(cost_result)

        # Create overall time period
        if results:
            overall_period = TimePeriod(
                start=results[0].time_period.start, end=results[-1].time_period.end
            )
        else:
            overall_period = params.time_period or TimePeriod(
                start=datetime.now() - timedelta(days=30), end=datetime.now()
            )

        return CostData(
            results=results,
            time_period=overall_period,
            total_cost=CostAmount(amount=total_amount),
            group_definitions=params.group_by or [],
        )

    def check_service_availability(self) -> Dict[str, Any]:
        """Check if AWS Cost Explorer service is available and responsive."""
        try:
            # Make a minimal API call to check service health
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=1)

            start_time = time.time()
            _response = self.client.get_cost_and_usage(
                TimePeriod={
                    "Start": start_date.strftime("%Y-%m-%d"),
                    "End": end_date.strftime("%Y-%m-%d"),
                },
                Granularity="DAILY",
                Metrics=["BlendedCost"],
            )
            response_time = time.time() - start_time

            return {
                "available": True,
                "response_time_ms": round(response_time * 1000, 2),
                "status": "healthy",
            }

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")

            if error_code == "AccessDenied":
                return {
                    "available": False,
                    "status": "access_denied",
                    "error": "Insufficient permissions for Cost Explorer API",
                }
            elif error_code == "ThrottlingException":
                return {
                    "available": True,
                    "status": "throttled",
                    "error": "API rate limit exceeded",
                }
            elif error_code == "ServiceUnavailable":
                return {
                    "available": False,
                    "status": "service_unavailable",
                    "error": "Cost Explorer service is temporarily unavailable",
                }
            else:
                return {
                    "available": False,
                    "status": "error",
                    "error": f"API error ({error_code}): {e.response.get('Error', {}).get('Message', '')}",
                }

        except Exception as e:
            return {
                "available": False,
                "status": "error",
                "error": f"Unexpected error: {str(e)}",
            }

    def get_supported_services(self) -> List[str]:
        """Get list of AWS services that have cost data available."""
        try:
            # Get services from the last 30 days
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=30)

            response = self.client.get_dimension_values(
                TimePeriod={
                    "Start": start_date.strftime("%Y-%m-%d"),
                    "End": end_date.strftime("%Y-%m-%d"),
                },
                Dimension="SERVICE",
            )

            services = [item["Value"] for item in response.get("DimensionValues", [])]
            return sorted(services)

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            error_message = e.response.get("Error", {}).get("Message", "")
            raise RuntimeError(
                f"Failed to get supported services ({error_code}): {error_message}"
            )

    def estimate_query_cost(self, params: QueryParameters) -> Dict[str, Any]:
        """Estimate the complexity and potential cost of a query."""
        complexity_score = 0
        warnings = []

        # Time period complexity
        if params.time_period:
            days = (params.time_period.end - params.time_period.start).days
            if days > 365:
                complexity_score += 3
                warnings.append("Query spans more than 1 year - may be slow")
            elif days > 90:
                complexity_score += 2
                warnings.append("Query spans more than 3 months")
            elif days > 30:
                complexity_score += 1

        # Granularity complexity
        if params.granularity == TimePeriodGranularity.HOURLY:
            complexity_score += 3
            warnings.append("Hourly granularity increases query complexity")
        elif params.granularity == TimePeriodGranularity.DAILY:
            complexity_score += 1

        # Group by complexity
        if params.group_by:
            complexity_score += len(params.group_by)
            if len(params.group_by) > 2:
                warnings.append("Multiple group-by dimensions increase complexity")

        # Determine complexity level
        if complexity_score <= 2:
            complexity_level = "low"
        elif complexity_score <= 5:
            complexity_level = "medium"
        else:
            complexity_level = "high"
            warnings.append("High complexity query may take longer to execute")

        return {
            "complexity_score": complexity_score,
            "complexity_level": complexity_level,
            "warnings": warnings,
            "estimated_response_time": self._estimate_response_time(complexity_score),
        }

    def _estimate_response_time(self, complexity_score: int) -> str:
        """Estimate response time based on complexity score."""
        if complexity_score <= 2:
            return "< 5 seconds"
        elif complexity_score <= 5:
            return "5-15 seconds"
        else:
            return "15-60 seconds"

    def warm_cache_for_common_queries(self) -> Dict[str, Any]:
        """
        Warm the cache with common queries to improve response times.

        Returns:
            Dictionary with warming results
        """
        if not self.cache_manager:
            return {"error": "No cache manager available"}

        warming_results = {"queries_warmed": 0, "queries_failed": 0, "errors": []}

        # Define common query patterns
        common_queries = [
            # Current month total costs
            QueryParameters(
                time_period=self._get_current_month_period(),
                granularity=TimePeriodGranularity.MONTHLY,
                metrics=[MetricType.BLENDED_COST],
            ),
            # Last month total costs
            QueryParameters(
                time_period=self._get_last_month_period(),
                granularity=TimePeriodGranularity.MONTHLY,
                metrics=[MetricType.BLENDED_COST],
            ),
            # Current month by service
            QueryParameters(
                time_period=self._get_current_month_period(),
                granularity=TimePeriodGranularity.MONTHLY,
                metrics=[MetricType.BLENDED_COST],
                group_by=["SERVICE"],
            ),
            # Last 7 days daily costs
            QueryParameters(
                time_period=self._get_last_week_period(),
                granularity=TimePeriodGranularity.DAILY,
                metrics=[MetricType.BLENDED_COST],
            ),
        ]

        for query_params in common_queries:
            try:
                # Check if already cached
                cache_key = self.cache_manager.generate_cache_key(
                    query_params, self.profile or "default"
                )
                if not self.cache_manager.get_cached_data(cache_key):
                    # Not cached, fetch and cache
                    self.get_cost_and_usage(query_params, use_cache=False)
                    warming_results["queries_warmed"] += 1
            except Exception as e:
                warming_results["queries_failed"] += 1
                warming_results["errors"].append(str(e))

        return warming_results

    def _get_current_month_period(self) -> TimePeriod:
        """Get current month time period."""
        now = datetime.now(timezone.utc)
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return TimePeriod(start=start, end=now)

    def _get_last_month_period(self) -> TimePeriod:
        """Get last month time period."""
        now = datetime.now(timezone.utc)
        first_day_this_month = now.replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )
        last_day_last_month = first_day_this_month - timedelta(days=1)
        first_day_last_month = last_day_last_month.replace(day=1)
        return TimePeriod(start=first_day_last_month, end=first_day_this_month)

    def _get_last_week_period(self) -> TimePeriod:
        """Get last 7 days time period."""
        now = datetime.now(timezone.utc)
        week_ago = now - timedelta(days=7)
        return TimePeriod(start=week_ago, end=now)

    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Get cache statistics for this client.

        Returns:
            Dictionary with cache statistics
        """
        if not self.cache_manager:
            return {"error": "No cache manager available"}

        return self.cache_manager.get_cache_stats()

    def clear_cache(self, pattern: Optional[str] = None) -> int:
        """
        Clear cache entries for this client.

        Args:
            pattern: Optional pattern to match cache files

        Returns:
            Number of cache files removed
        """
        if not self.cache_manager:
            return 0

        return self.cache_manager.invalidate_cache(pattern)

    def get_cost_with_trend_analysis(self, params: QueryParameters) -> CostData:
        """
        Get cost data with trend analysis if requested.

        Args:
            params: Query parameters including trend analysis settings

        Returns:
            CostData with trend analysis included
        """
        # Get current period data
        current_data = self.get_cost_and_usage(params)

        # If trend analysis is requested, get comparison data
        if params.trend_analysis:
            from .date_utils import DateRangeCalculator
            from .trend_analysis import TrendAnalyzer

            calculator = DateRangeCalculator(params.fiscal_year_start_month)
            analyzer = TrendAnalyzer()

            # Calculate comparison period
            if params.trend_analysis == TrendAnalysisType.YEAR_OVER_YEAR:
                comparison_period = calculator.get_previous_period(
                    params.time_period, "year_ago"
                )
            elif params.trend_analysis == TrendAnalysisType.MONTH_OVER_MONTH:
                comparison_period = calculator.get_previous_period(
                    params.time_period, "month_ago"
                )
            elif params.trend_analysis == TrendAnalysisType.QUARTER_OVER_QUARTER:
                comparison_period = calculator.get_previous_period(
                    params.time_period, "quarter_ago"
                )
            else:  # PERIOD_OVER_PERIOD
                comparison_period = calculator.get_previous_period(
                    params.time_period, "same_length"
                )

            # Create comparison query parameters
            comparison_params = QueryParameters(
                service=params.service,
                time_period=comparison_period,
                granularity=params.granularity,
                metrics=params.metrics,
                group_by=params.group_by,
                fiscal_year_start_month=params.fiscal_year_start_month,
            )

            # Get comparison data
            comparison_data = self.get_cost_and_usage(comparison_params)

            # Analyze trend
            trend_data = analyzer.analyze_trend(
                current_data, comparison_data, params.trend_analysis
            )

            # Add trend data to result
            current_data.trend_data = trend_data

        return current_data

    def get_cost_with_forecast(self, params: QueryParameters) -> CostData:
        """
        Get cost data with forecasting if requested.

        Args:
            params: Query parameters including forecast settings

        Returns:
            CostData with forecast data included
        """
        # Get current data
        cost_data = self.get_cost_and_usage(params)

        # If forecast is requested, generate forecast
        if params.include_forecast:
            from .trend_analysis import CostForecaster

            forecaster = CostForecaster()

            # Get historical data for forecasting (last 12 months)
            historical_periods = self._get_historical_periods(
                params.time_period.end, 12
            )

            historical_data = []
            for period in historical_periods:
                historical_params = QueryParameters(
                    service=params.service,
                    time_period=period,
                    granularity=TimePeriodGranularity.MONTHLY,
                    metrics=params.metrics,
                    group_by=params.group_by,
                    fiscal_year_start_month=params.fiscal_year_start_month,
                )

                try:
                    historical_cost_data = self.get_cost_and_usage(historical_params)
                    historical_data.append(historical_cost_data)
                except Exception:
                    # Skip periods with no data or errors
                    continue

            # Generate forecast if we have enough historical data
            if len(historical_data) >= 3:
                try:
                    forecast_data = forecaster.forecast_costs(
                        historical_data, params.forecast_months
                    )
                    cost_data.forecast_data = forecast_data
                except Exception:
                    # Forecasting failed, continue without forecast
                    pass

        return cost_data

    def get_advanced_cost_data(self, params: QueryParameters) -> CostData:
        """
        Get cost data with all advanced features (trend analysis, forecasting).

        Args:
            params: Query parameters with advanced features

        Returns:
            CostData with all requested advanced features
        """
        # Start with basic cost data
        cost_data = self.get_cost_and_usage(params)

        # Add trend analysis if requested
        if params.trend_analysis:
            trend_data = self.get_cost_with_trend_analysis(params)
            cost_data.trend_data = trend_data.trend_data

        # Add forecast if requested
        if params.include_forecast:
            forecast_data = self.get_cost_with_forecast(params)
            cost_data.forecast_data = forecast_data.forecast_data

        return cost_data

    def create_performance_optimized_client(
        self,
        enable_parallel: bool = True,
        enable_compression: bool = True,
        enable_monitoring: bool = True,
    ) -> "PerformanceOptimizedClient":
        """
        Create a performance-optimized version of this client.

        Args:
            enable_parallel: Enable parallel query execution
            enable_compression: Enable cache compression
            enable_monitoring: Enable performance monitoring

        Returns:
            PerformanceOptimizedClient instance
        """
        return PerformanceOptimizedClient(
            aws_client=self,
            cache_manager=self.cache_manager,
            enable_parallel=enable_parallel,
            enable_compression=enable_compression,
            enable_monitoring=enable_monitoring,
        )

    def get_cost_and_usage_with_performance_optimization(
        self,
        params: QueryParameters,
        use_cache: bool = True,
        enable_parallel: bool = True,
        enable_compression: bool = True,
        max_chunk_days: int = 90,
    ) -> CostData:
        """
        Get cost data with automatic performance optimizations.

        Args:
            params: Query parameters
            use_cache: Whether to use caching
            enable_parallel: Enable parallel execution for large queries
            enable_compression: Enable cache compression
            max_chunk_days: Maximum days per parallel chunk

        Returns:
            CostData with performance optimizations applied
        """
        # Create temporary performance-optimized client
        perf_client = self.create_performance_optimized_client(
            enable_parallel=enable_parallel,
            enable_compression=enable_compression,
            enable_monitoring=True,
        )

        return perf_client.get_cost_and_usage_optimized(
            params=params, use_cache=use_cache, max_chunk_days=max_chunk_days
        )

    def _get_historical_periods(
        self, end_date: datetime, months: int
    ) -> List[TimePeriod]:
        """
        Get list of historical monthly periods.

        Args:
            end_date: End date to work backwards from
            months: Number of months to go back

        Returns:
            List of TimePeriod objects for historical months
        """
        periods = []
        current_date = end_date

        for _ in range(months):
            # Go back one month
            if current_date.month == 1:
                month_start = current_date.replace(
                    year=current_date.year - 1, month=12, day=1
                )
            else:
                month_start = current_date.replace(month=current_date.month - 1, day=1)

            # End of month is start of next month
            month_end = current_date.replace(day=1)

            periods.append(TimePeriod(start=month_start, end=month_end))
            current_date = month_start

        return list(reversed(periods))  # Return in chronological order

    def prefetch_data(self, queries: List[QueryParameters]) -> Dict[str, Any]:
        """
        Prefetch data for multiple queries to warm the cache.

        Args:
            queries: List of query parameters to prefetch

        Returns:
            Dictionary with prefetch results
        """
        if not self.cache_manager:
            return {"error": "No cache manager available"}

        results = {"prefetched": 0, "already_cached": 0, "failed": 0, "errors": []}

        for query_params in queries:
            try:
                cache_key = self.cache_manager.generate_cache_key(
                    query_params, self.profile or "default"
                )

                # Check if already cached
                if self.cache_manager.get_cached_data(cache_key):
                    results["already_cached"] += 1
                else:
                    # Fetch and cache
                    self.get_cost_and_usage(query_params, use_cache=False)
                    results["prefetched"] += 1

            except Exception as e:
                results["failed"] += 1
                results["errors"].append(f"Query failed: {str(e)}")

        return results
