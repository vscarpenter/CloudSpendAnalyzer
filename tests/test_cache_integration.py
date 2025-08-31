"""Tests for cache integration with AWS API calls."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone, timedelta

from src.aws_cost_cli.aws_client import AWSCostClient
from src.aws_cost_cli.cache_manager import CacheManager
from src.aws_cost_cli.models import (
    QueryParameters,
    CostData,
    CostAmount,
    TimePeriod,
    TimePeriodGranularity,
    MetricType,
)


class TestCacheIntegration:
    """Test cache integration with AWS client."""

    @patch("boto3.Session")
    def test_aws_client_with_cache_manager(self, mock_session):
        """Test AWS client initialization with cache manager."""
        mock_cache_manager = Mock()

        aws_client = AWSCostClient(cache_manager=mock_cache_manager)

        assert aws_client.cache_manager == mock_cache_manager

    @patch("boto3.Session")
    def test_get_cost_and_usage_cache_hit(self, mock_session):
        """Test get_cost_and_usage with cache hit."""
        # Setup mocks
        mock_cache_manager = Mock()
        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client

        # Create test data
        test_cost_data = CostData(
            results=[],
            time_period=TimePeriod(
                start=datetime.now(timezone.utc), end=datetime.now(timezone.utc)
            ),
            total_cost=CostAmount(amount=100.0),
            group_definitions=[],
        )

        # Mock cache hit
        mock_cache_manager.generate_cache_key.return_value = "test_key"
        mock_cache_manager.get_cached_data.return_value = test_cost_data

        aws_client = AWSCostClient(cache_manager=mock_cache_manager)
        params = QueryParameters()

        result = aws_client.get_cost_and_usage(params)

        # Verify cache was checked
        mock_cache_manager.generate_cache_key.assert_called_once()
        mock_cache_manager.get_cached_data.assert_called_once_with("test_key")

        # Verify API was not called
        mock_client.get_cost_and_usage.assert_not_called()

        # Verify result
        assert result == test_cost_data

    @patch("boto3.Session")
    def test_get_cost_and_usage_cache_miss(self, mock_session):
        """Test get_cost_and_usage with cache miss."""
        # Setup mocks
        mock_cache_manager = Mock()
        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client

        # Mock cache miss
        mock_cache_manager.generate_cache_key.return_value = "test_key"
        mock_cache_manager.get_cached_data.return_value = None

        # Mock API response
        mock_client.get_cost_and_usage.return_value = {
            "ResultsByTime": [],
            "DimensionKey": {},
            "GroupDefinitions": [],
        }

        aws_client = AWSCostClient(cache_manager=mock_cache_manager)
        params = QueryParameters()

        result = aws_client.get_cost_and_usage(params)

        # Verify cache was checked
        mock_cache_manager.get_cached_data.assert_called_once_with("test_key")

        # Verify API was called
        mock_client.get_cost_and_usage.assert_called_once()

        # Verify result was cached
        mock_cache_manager.cache_data.assert_called_once()

        # Verify result
        assert isinstance(result, CostData)

    @patch("boto3.Session")
    def test_get_cost_and_usage_use_cache_false(self, mock_session):
        """Test get_cost_and_usage with use_cache=False."""
        # Setup mocks
        mock_cache_manager = Mock()
        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client

        # Mock API response
        mock_client.get_cost_and_usage.return_value = {
            "ResultsByTime": [],
            "DimensionKey": {},
            "GroupDefinitions": [],
        }

        aws_client = AWSCostClient(cache_manager=mock_cache_manager)
        params = QueryParameters()

        result = aws_client.get_cost_and_usage(params, use_cache=False)

        # Verify cache was not checked
        mock_cache_manager.get_cached_data.assert_not_called()

        # Verify API was called
        mock_client.get_cost_and_usage.assert_called_once()

        # Verify result was still cached
        mock_cache_manager.cache_data.assert_called_once()

        # Verify result
        assert isinstance(result, CostData)

    @patch("boto3.Session")
    def test_get_cost_and_usage_no_cache_manager(self, mock_session):
        """Test get_cost_and_usage without cache manager."""
        # Setup mocks
        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client

        # Mock API response
        mock_client.get_cost_and_usage.return_value = {
            "ResultsByTime": [],
            "DimensionKey": {},
            "GroupDefinitions": [],
        }

        aws_client = AWSCostClient()  # No cache manager
        params = QueryParameters()

        result = aws_client.get_cost_and_usage(params)

        # Verify API was called
        mock_client.get_cost_and_usage.assert_called_once()

        # Verify result
        assert isinstance(result, CostData)

    @patch("boto3.Session")
    def test_warm_cache_for_common_queries(self, mock_session):
        """Test cache warming functionality."""
        # Setup mocks
        mock_cache_manager = Mock()
        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client

        # Mock cache misses for all queries
        mock_cache_manager.generate_cache_key.return_value = "test_key"
        mock_cache_manager.get_cached_data.return_value = None

        # Mock API responses
        mock_client.get_cost_and_usage.return_value = {
            "ResultsByTime": [],
            "DimensionKey": {},
            "GroupDefinitions": [],
        }

        aws_client = AWSCostClient(cache_manager=mock_cache_manager)

        results = aws_client.warm_cache_for_common_queries()

        # Verify results
        assert "queries_warmed" in results
        assert "queries_failed" in results
        assert "errors" in results
        assert results["queries_warmed"] > 0

        # Verify API was called multiple times
        assert mock_client.get_cost_and_usage.call_count > 0

    @patch("boto3.Session")
    def test_warm_cache_no_cache_manager(self, mock_session):
        """Test cache warming without cache manager."""
        aws_client = AWSCostClient()  # No cache manager

        results = aws_client.warm_cache_for_common_queries()

        assert "error" in results
        assert "No cache manager available" in results["error"]

    @patch("boto3.Session")
    def test_prefetch_data(self, mock_session):
        """Test data prefetching functionality."""
        # Setup mocks
        mock_cache_manager = Mock()
        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client

        # Mock cache miss
        mock_cache_manager.generate_cache_key.return_value = "test_key"
        mock_cache_manager.get_cached_data.return_value = None

        # Mock API response
        mock_client.get_cost_and_usage.return_value = {
            "ResultsByTime": [],
            "DimensionKey": {},
            "GroupDefinitions": [],
        }

        aws_client = AWSCostClient(cache_manager=mock_cache_manager)

        # Test queries
        queries = [
            QueryParameters(granularity=TimePeriodGranularity.MONTHLY),
            QueryParameters(granularity=TimePeriodGranularity.DAILY),
        ]

        results = aws_client.prefetch_data(queries)

        # Verify results
        assert "prefetched" in results
        assert "already_cached" in results
        assert "failed" in results
        assert "errors" in results
        assert results["prefetched"] == 2

        # Verify API was called for each query
        assert mock_client.get_cost_and_usage.call_count == 2

    @patch("boto3.Session")
    def test_prefetch_data_already_cached(self, mock_session):
        """Test data prefetching with already cached data."""
        # Setup mocks
        mock_cache_manager = Mock()
        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client

        # Create test data
        test_cost_data = CostData(
            results=[],
            time_period=TimePeriod(
                start=datetime.now(timezone.utc), end=datetime.now(timezone.utc)
            ),
            total_cost=CostAmount(amount=100.0),
            group_definitions=[],
        )

        # Mock cache hit
        mock_cache_manager.generate_cache_key.return_value = "test_key"
        mock_cache_manager.get_cached_data.return_value = test_cost_data

        aws_client = AWSCostClient(cache_manager=mock_cache_manager)

        # Test queries
        queries = [QueryParameters(granularity=TimePeriodGranularity.MONTHLY)]

        results = aws_client.prefetch_data(queries)

        # Verify results
        assert results["already_cached"] == 1
        assert results["prefetched"] == 0

        # Verify API was not called
        mock_client.get_cost_and_usage.assert_not_called()

    @patch("boto3.Session")
    def test_get_cache_statistics(self, mock_session):
        """Test getting cache statistics."""
        mock_cache_manager = Mock()
        mock_cache_manager.get_cache_stats.return_value = {
            "total_entries": 5,
            "valid_entries": 3,
            "expired_entries": 2,
        }

        aws_client = AWSCostClient(cache_manager=mock_cache_manager)

        stats = aws_client.get_cache_statistics()

        assert stats["total_entries"] == 5
        assert stats["valid_entries"] == 3
        assert stats["expired_entries"] == 2

        mock_cache_manager.get_cache_stats.assert_called_once()

    @patch("boto3.Session")
    def test_clear_cache(self, mock_session):
        """Test clearing cache."""
        mock_cache_manager = Mock()
        mock_cache_manager.invalidate_cache.return_value = 3

        aws_client = AWSCostClient(cache_manager=mock_cache_manager)

        removed_count = aws_client.clear_cache()

        assert removed_count == 3
        mock_cache_manager.invalidate_cache.assert_called_once_with(None)

    @patch("boto3.Session")
    def test_clear_cache_with_pattern(self, mock_session):
        """Test clearing cache with pattern."""
        mock_cache_manager = Mock()
        mock_cache_manager.invalidate_cache.return_value = 2

        aws_client = AWSCostClient(cache_manager=mock_cache_manager)

        removed_count = aws_client.clear_cache("test_pattern")

        assert removed_count == 2
        mock_cache_manager.invalidate_cache.assert_called_once_with("test_pattern")


class TestCacheIntegrationEdgeCases:
    """Test edge cases for cache integration."""

    @patch("boto3.Session")
    def test_cache_error_during_get(self, mock_session):
        """Test handling cache error during get operation."""
        from src.aws_cost_cli.exceptions import CacheError

        # Setup mocks
        mock_cache_manager = Mock()
        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client

        # Mock cache error
        mock_cache_manager.generate_cache_key.return_value = "test_key"
        mock_cache_manager.get_cached_data.side_effect = CacheError("Cache read failed")

        # Mock API response
        mock_client.get_cost_and_usage.return_value = {
            "ResultsByTime": [],
            "DimensionKey": {},
            "GroupDefinitions": [],
        }

        aws_client = AWSCostClient(cache_manager=mock_cache_manager)
        params = QueryParameters()

        # Should continue without cache
        result = aws_client.get_cost_and_usage(params)

        # Verify API was called despite cache error
        mock_client.get_cost_and_usage.assert_called_once()
        assert isinstance(result, CostData)

    @patch("boto3.Session")
    def test_cache_error_during_set(self, mock_session):
        """Test handling cache error during set operation."""
        from src.aws_cost_cli.exceptions import CacheError

        # Setup mocks
        mock_cache_manager = Mock()
        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client

        # Mock cache miss and cache write error
        mock_cache_manager.generate_cache_key.return_value = "test_key"
        mock_cache_manager.get_cached_data.return_value = None
        mock_cache_manager.cache_data.side_effect = CacheError("Cache write failed")

        # Mock API response
        mock_client.get_cost_and_usage.return_value = {
            "ResultsByTime": [],
            "DimensionKey": {},
            "GroupDefinitions": [],
        }

        aws_client = AWSCostClient(cache_manager=mock_cache_manager)
        params = QueryParameters()

        # Should continue despite cache write error
        result = aws_client.get_cost_and_usage(params)

        # Verify API was called and result returned
        mock_client.get_cost_and_usage.assert_called_once()
        assert isinstance(result, CostData)

    @patch("boto3.Session")
    def test_warm_cache_with_api_errors(self, mock_session):
        """Test cache warming with API errors."""
        from botocore.exceptions import ClientError

        # Setup mocks
        mock_cache_manager = Mock()
        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client

        # Mock cache miss
        mock_cache_manager.generate_cache_key.return_value = "test_key"
        mock_cache_manager.get_cached_data.return_value = None

        # Mock API error
        mock_client.get_cost_and_usage.side_effect = ClientError(
            error_response={
                "Error": {"Code": "AccessDenied", "Message": "Access denied"}
            },
            operation_name="GetCostAndUsage",
        )

        aws_client = AWSCostClient(cache_manager=mock_cache_manager)

        results = aws_client.warm_cache_for_common_queries()

        # Verify errors were handled
        assert results["queries_failed"] > 0
        assert len(results["errors"]) > 0


if __name__ == "__main__":
    pytest.main([__file__])
