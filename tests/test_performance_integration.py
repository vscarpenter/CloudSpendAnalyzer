"""Integration tests for performance optimizations."""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from src.aws_cost_cli.query_pipeline import QueryPipeline, QueryContext
from src.aws_cost_cli.models import (
    QueryParameters,
    CostData,
    CostResult,
    CostAmount,
    TimePeriod,
    TimePeriodGranularity,
    MetricType,
    Config,
)


class TestPerformanceIntegration:
    """Integration tests for performance optimizations."""

    @patch("src.aws_cost_cli.query_pipeline.AWSCostClient")
    @patch("src.aws_cost_cli.query_pipeline.QueryParser")
    def test_query_with_performance_optimizations(
        self, mock_query_parser, mock_aws_client_class
    ):
        """Test end-to-end query with performance optimizations enabled."""
        # Mock query parser
        mock_parser = Mock()
        mock_query_parser.return_value = mock_parser

        # Create query parameters for a large time range (should trigger parallel execution)
        start_date = datetime(2025, 1, 1)
        end_date = datetime(2025, 12, 31)  # 365 days
        query_params = QueryParameters(
            service="Amazon Elastic Compute Cloud - Compute",
            time_period=TimePeriod(start=start_date, end=end_date),
            granularity=TimePeriodGranularity.DAILY,
            metrics=[MetricType.BLENDED_COST],
        )
        mock_parser.parse_query.return_value = query_params

        # Mock AWS client
        mock_aws_client = Mock()
        mock_aws_client_class.return_value = mock_aws_client

        # Mock cost data response
        mock_cost_data = CostData(
            results=[
                CostResult(
                    time_period=TimePeriod(start=start_date, end=end_date),
                    total=CostAmount(amount=1000.0),
                    groups=[],
                    estimated=False,
                )
            ],
            time_period=TimePeriod(start=start_date, end=end_date),
            total_cost=CostAmount(amount=1000.0),
            currency="USD",
        )

        # Mock the performance-optimized method
        mock_aws_client.get_cost_and_usage_with_performance_optimization.return_value = (
            mock_cost_data
        )

        # Mock credential validation
        with patch(
            "src.aws_cost_cli.query_pipeline.CredentialManager"
        ) as mock_cred_manager:
            mock_cred_manager.return_value.validate_credentials.return_value = True
            mock_aws_client.validate_permissions.return_value = True

            # Mock response generator
            with patch(
                "src.aws_cost_cli.query_pipeline.ResponseGenerator"
            ) as mock_response_gen:
                mock_response_gen.return_value.format_response.return_value = (
                    "Test response"
                )

                # Create pipeline with test config
                config = Config(llm_provider="test", cache_ttl=3600)
                pipeline = QueryPipeline(config=config)

                # Create query context with performance optimizations enabled
                context = QueryContext(
                    original_query="Show me EC2 costs for 2025",
                    enable_parallel=True,
                    enable_compression=True,
                    max_chunk_days=90,
                    show_performance_metrics=True,
                )

                # Process query
                result = pipeline.process_query(context)

                # Verify result
                assert result.success is True
                assert result.cost_data == mock_cost_data
                assert result.formatted_response == "Test response"

                # Verify performance optimizations were used
                assert result.metadata.get("performance_optimizations_used") is True
                assert result.metadata.get("parallel_enabled") is True
                assert result.metadata.get("compression_enabled") is True

    @patch("src.aws_cost_cli.query_pipeline.AWSCostClient")
    @patch("src.aws_cost_cli.query_pipeline.QueryParser")
    def test_query_without_performance_optimizations(
        self, mock_query_parser, mock_aws_client_class
    ):
        """Test query with performance optimizations disabled."""
        # Mock query parser
        mock_parser = Mock()
        mock_query_parser.return_value = mock_parser

        # Create simple query parameters
        query_params = QueryParameters(
            service="Amazon Simple Storage Service",
            granularity=TimePeriodGranularity.MONTHLY,
            metrics=[MetricType.BLENDED_COST],
        )
        mock_parser.parse_query.return_value = query_params

        # Mock AWS client
        mock_aws_client = Mock()
        mock_aws_client_class.return_value = mock_aws_client

        # Mock cost data response
        mock_cost_data = CostData(
            results=[],
            time_period=TimePeriod(
                start=datetime(2025, 7, 1), end=datetime(2025, 8, 1)
            ),
            total_cost=CostAmount(amount=100.0),
            currency="USD",
        )
        mock_aws_client.get_cost_and_usage.return_value = mock_cost_data

        # Mock credential validation
        with patch(
            "src.aws_cost_cli.query_pipeline.CredentialManager"
        ) as mock_cred_manager:
            mock_cred_manager.return_value.validate_credentials.return_value = True
            mock_aws_client.validate_permissions.return_value = True

            # Mock response generator
            with patch(
                "src.aws_cost_cli.query_pipeline.ResponseGenerator"
            ) as mock_response_gen:
                mock_response_gen.return_value.format_response.return_value = (
                    "Test response"
                )

                # Create pipeline
                config = Config(llm_provider="test", cache_ttl=3600)
                pipeline = QueryPipeline(config=config)

                # Create query context with performance optimizations disabled
                context = QueryContext(
                    original_query="Show me S3 costs",
                    enable_parallel=False,
                    enable_compression=False,
                    show_performance_metrics=False,
                )

                # Process query
                result = pipeline.process_query(context)

                # Verify result
                assert result.success is True
                assert result.cost_data == mock_cost_data

                # Verify standard method was used
                mock_aws_client.get_cost_and_usage.assert_called_once()

    def test_performance_metrics_display(self):
        """Test that performance metrics are properly formatted for display."""
        from src.aws_cost_cli.query_pipeline import QueryResult

        # Create result with performance metrics
        result = QueryResult(
            success=True,
            processing_time_ms=1500.5,
            api_calls_made=3,
            parallel_requests=3,
            cache_hit=False,
            performance_metrics={
                "query_performance": {
                    "cache_hit_rate": 0.75,
                    "performance": {"avg_duration_ms": 1200.0},
                }
            },
            compression_stats={
                "average_compression_ratio": 0.6,
                "space_saved_percent": 40.0,
            },
        )

        # Verify metrics are accessible
        assert result.processing_time_ms == 1500.5
        assert result.api_calls_made == 3
        assert result.parallel_requests == 3
        assert result.performance_metrics["query_performance"]["cache_hit_rate"] == 0.75
        assert result.compression_stats["space_saved_percent"] == 40.0

    def test_large_query_chunking_logic(self):
        """Test that large queries are properly identified for parallel processing."""
        from src.aws_cost_cli.performance import ParallelQueryExecutor

        executor = ParallelQueryExecutor()

        # Test large query (should be chunked)
        large_params = QueryParameters(
            time_period=TimePeriod(
                start=datetime(2025, 1, 1), end=datetime(2025, 12, 31)  # 365 days
            ),
            granularity=TimePeriodGranularity.DAILY,
        )

        chunks = executor.split_large_query(large_params, max_days_per_chunk=90)
        assert len(chunks) > 1

        # Test small query (should not be chunked)
        small_params = QueryParameters(
            time_period=TimePeriod(
                start=datetime(2025, 7, 1), end=datetime(2025, 7, 31)  # 30 days
            ),
            granularity=TimePeriodGranularity.DAILY,
        )

        chunks = executor.split_large_query(small_params, max_days_per_chunk=90)
        assert len(chunks) == 1

    def test_compression_effectiveness(self):
        """Test that compression actually reduces data size for realistic data."""
        from src.aws_cost_cli.performance import CompressedCacheManager
        from src.aws_cost_cli.cache_manager import CacheManager
        import tempfile
        import os

        # Create temporary cache directory
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = CacheManager(cache_dir=temp_dir)
            compressed_cache = CompressedCacheManager(cache_manager)

            # Create realistic large data
            large_text = "This is cost data that repeats many times. " * 1000

            # Test compression
            compressed = compressed_cache._compress_data(large_text)
            decompressed = compressed_cache._decompress_data(compressed)

            # Verify compression worked
            assert len(compressed) < len(large_text.encode("utf-8"))
            assert decompressed == large_text

            # Calculate compression ratio
            compression_ratio = len(compressed) / len(large_text.encode("utf-8"))
            assert compression_ratio < 0.5  # Should achieve at least 50% compression
