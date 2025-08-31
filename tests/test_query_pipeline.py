"""Tests for end-to-end query processing pipeline."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone

from src.aws_cost_cli.query_pipeline import QueryPipeline, QueryContext, QueryResult
from src.aws_cost_cli.models import (
    Config,
    QueryParameters,
    CostData,
    CostAmount,
    TimePeriod,
    TimePeriodGranularity,
    MetricType,
)
from src.aws_cost_cli.exceptions import (
    AWSCredentialsError,
    AWSPermissionsError,
    QueryParsingError,
    ValidationError,
)


class TestQueryPipeline:
    """Test query pipeline functionality."""

    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        config = Config()
        pipeline = QueryPipeline(config=config)

        assert pipeline.config == config
        assert pipeline.credential_manager is not None
        assert pipeline.cache_manager is not None
        assert pipeline.query_parser is not None

    @patch("src.aws_cost_cli.config.ConfigManager.load_config")
    def test_pipeline_initialization_with_config_file(self, mock_load_config):
        """Test pipeline initialization with config file."""
        mock_config = Config()
        mock_load_config.return_value = mock_config

        pipeline = QueryPipeline(config_path="test_config.yaml")

        assert pipeline.config == mock_config
        mock_load_config.assert_called_once_with("test_config.yaml")

    @patch("src.aws_cost_cli.config.ConfigManager.load_config")
    def test_pipeline_initialization_default_config(self, mock_load_config):
        """Test pipeline initialization with default config when file not found."""
        mock_load_config.side_effect = FileNotFoundError()

        pipeline = QueryPipeline()

        assert isinstance(pipeline.config, Config)

    @patch("src.aws_cost_cli.aws_client.CredentialManager.validate_credentials")
    @patch("src.aws_cost_cli.aws_client.AWSCostClient.validate_permissions")
    @patch("src.aws_cost_cli.query_processor.QueryParser.parse_query")
    @patch("src.aws_cost_cli.query_processor.QueryParser.validate_parameters")
    @patch("src.aws_cost_cli.aws_client.AWSCostClient.get_cost_and_usage")
    @patch("src.aws_cost_cli.response_formatter.ResponseGenerator.format_response")
    def test_successful_query_processing(
        self,
        mock_format_response,
        mock_get_cost_data,
        mock_validate_params,
        mock_parse_query,
        mock_validate_permissions,
        mock_validate_credentials,
    ):
        """Test successful end-to-end query processing."""
        # Setup mocks
        mock_validate_credentials.return_value = True
        mock_validate_permissions.return_value = True

        mock_query_params = QueryParameters(
            granularity=TimePeriodGranularity.MONTHLY, metrics=[MetricType.BLENDED_COST]
        )
        mock_parse_query.return_value = mock_query_params
        mock_validate_params.return_value = True

        mock_cost_data = CostData(
            results=[],
            time_period=TimePeriod(
                start=datetime.now(timezone.utc), end=datetime.now(timezone.utc)
            ),
            total_cost=CostAmount(amount=100.0),
            group_definitions=[],
        )
        mock_get_cost_data.return_value = mock_cost_data

        mock_format_response.return_value = "Test response"

        # Create pipeline and context
        config = Config()
        pipeline = QueryPipeline(config=config)
        context = QueryContext(original_query="How much did I spend on EC2 last month?")

        # Process query
        result = pipeline.process_query(context)

        # Verify result
        assert result.success is True
        assert result.cost_data == mock_cost_data
        assert result.formatted_response == "Test response"
        assert result.processing_time_ms is not None
        assert result.processing_time_ms > 0

        # Verify all steps were called
        mock_validate_credentials.assert_called_once()
        mock_validate_permissions.assert_called_once()
        mock_parse_query.assert_called_once_with(
            "How much did I spend on EC2 last month?"
        )
        mock_validate_params.assert_called_once_with(mock_query_params)
        mock_get_cost_data.assert_called_once()
        mock_format_response.assert_called_once()

    @patch("src.aws_cost_cli.aws_client.CredentialManager.validate_credentials")
    def test_credentials_validation_failure(self, mock_validate_credentials):
        """Test pipeline handling of credentials validation failure."""
        mock_validate_credentials.return_value = False

        config = Config()
        pipeline = QueryPipeline(config=config)
        context = QueryContext(original_query="test query", profile="test-profile")

        result = pipeline.process_query(context)

        assert result.success is False
        assert isinstance(result.error, AWSCredentialsError)
        assert "test-profile" in result.error.message

    @patch("src.aws_cost_cli.aws_client.CredentialManager.validate_credentials")
    @patch("src.aws_cost_cli.aws_client.AWSCostClient.validate_permissions")
    def test_permissions_validation_failure(
        self, mock_validate_permissions, mock_validate_credentials
    ):
        """Test pipeline handling of permissions validation failure."""
        mock_validate_credentials.return_value = True
        mock_validate_permissions.return_value = False

        config = Config()
        pipeline = QueryPipeline(config=config)
        context = QueryContext(original_query="test query")

        result = pipeline.process_query(context)

        assert result.success is False
        assert isinstance(result.error, AWSPermissionsError)

    @patch("src.aws_cost_cli.aws_client.CredentialManager.validate_credentials")
    @patch("src.aws_cost_cli.aws_client.AWSCostClient.validate_permissions")
    @patch("src.aws_cost_cli.query_processor.QueryParser.parse_query")
    def test_query_parsing_failure(
        self, mock_parse_query, mock_validate_permissions, mock_validate_credentials
    ):
        """Test pipeline handling of query parsing failure."""
        mock_validate_credentials.return_value = True
        mock_validate_permissions.return_value = True
        mock_parse_query.side_effect = QueryParsingError("Failed to parse query")

        config = Config()
        pipeline = QueryPipeline(config=config)
        context = QueryContext(original_query="invalid query")

        result = pipeline.process_query(context)

        assert result.success is False
        assert isinstance(result.error, QueryParsingError)

    @patch("src.aws_cost_cli.aws_client.CredentialManager.validate_credentials")
    @patch("src.aws_cost_cli.aws_client.AWSCostClient.validate_permissions")
    @patch("src.aws_cost_cli.query_processor.QueryParser.parse_query")
    @patch("src.aws_cost_cli.query_processor.QueryParser.validate_parameters")
    def test_parameter_validation_failure(
        self,
        mock_validate_params,
        mock_parse_query,
        mock_validate_permissions,
        mock_validate_credentials,
    ):
        """Test pipeline handling of parameter validation failure."""
        mock_validate_credentials.return_value = True
        mock_validate_permissions.return_value = True
        mock_parse_query.return_value = QueryParameters()
        mock_validate_params.side_effect = ValidationError("Invalid parameters")

        config = Config()
        pipeline = QueryPipeline(config=config)
        context = QueryContext(original_query="test query")

        result = pipeline.process_query(context)

        assert result.success is False
        assert isinstance(result.error, ValidationError)

    @patch("src.aws_cost_cli.aws_client.CredentialManager.validate_credentials")
    @patch("src.aws_cost_cli.aws_client.AWSCostClient.validate_permissions")
    @patch("src.aws_cost_cli.query_processor.QueryParser.parse_query")
    @patch("src.aws_cost_cli.query_processor.QueryParser.validate_parameters")
    @patch("src.aws_cost_cli.cache_manager.CacheManager.get_cached_data")
    @patch("src.aws_cost_cli.aws_client.AWSCostClient.get_cost_and_usage")
    @patch("src.aws_cost_cli.response_formatter.ResponseGenerator.format_response")
    def test_cache_hit(
        self,
        mock_format_response,
        mock_get_cost_data,
        mock_get_cached_data,
        mock_validate_params,
        mock_parse_query,
        mock_validate_permissions,
        mock_validate_credentials,
    ):
        """Test pipeline with cache hit."""
        # Setup mocks
        mock_validate_credentials.return_value = True
        mock_validate_permissions.return_value = True
        mock_parse_query.return_value = QueryParameters()
        mock_validate_params.return_value = True

        # Mock cache hit
        mock_cost_data = CostData(
            results=[],
            time_period=TimePeriod(
                start=datetime.now(timezone.utc), end=datetime.now(timezone.utc)
            ),
            total_cost=CostAmount(amount=100.0),
            group_definitions=[],
        )
        mock_get_cached_data.return_value = mock_cost_data
        mock_format_response.return_value = "Test response"

        # Create pipeline and context
        config = Config()
        pipeline = QueryPipeline(config=config)
        context = QueryContext(original_query="test query", fresh_data=False)

        # Process query
        result = pipeline.process_query(context)

        # Verify cache was used
        assert result.success is True
        assert result.cache_hit is True
        assert result.metadata["data_source"] == "cache"

        # Verify AWS API was not called
        mock_get_cost_data.assert_not_called()

    @patch("src.aws_cost_cli.aws_client.CredentialManager.validate_credentials")
    @patch("src.aws_cost_cli.aws_client.AWSCostClient.validate_permissions")
    @patch("src.aws_cost_cli.query_processor.QueryParser.parse_query")
    @patch("src.aws_cost_cli.query_processor.QueryParser.validate_parameters")
    @patch("src.aws_cost_cli.cache_manager.CacheManager.get_cached_data")
    @patch("src.aws_cost_cli.aws_client.AWSCostClient.get_cost_and_usage")
    @patch("src.aws_cost_cli.response_formatter.ResponseGenerator.format_response")
    def test_fresh_data_request(
        self,
        mock_format_response,
        mock_get_cost_data,
        mock_get_cached_data,
        mock_validate_params,
        mock_parse_query,
        mock_validate_permissions,
        mock_validate_credentials,
    ):
        """Test pipeline with fresh data request."""
        # Setup mocks
        mock_validate_credentials.return_value = True
        mock_validate_permissions.return_value = True
        mock_parse_query.return_value = QueryParameters()
        mock_validate_params.return_value = True

        mock_cost_data = CostData(
            results=[],
            time_period=TimePeriod(
                start=datetime.now(timezone.utc), end=datetime.now(timezone.utc)
            ),
            total_cost=CostAmount(amount=100.0),
            group_definitions=[],
        )
        mock_get_cost_data.return_value = mock_cost_data
        mock_format_response.return_value = "Test response"

        # Create pipeline and context
        config = Config()
        pipeline = QueryPipeline(config=config)
        context = QueryContext(original_query="test query", fresh_data=True)

        # Process query
        result = pipeline.process_query(context)

        # Verify fresh data was fetched
        assert result.success is True
        assert result.cache_hit is False
        assert result.metadata["data_source"] == "aws_api"

        # Verify cache was not checked
        mock_get_cached_data.assert_not_called()
        mock_get_cost_data.assert_called_once()


class TestQueryPipelineHelpers:
    """Test query pipeline helper methods."""

    def test_handle_ambiguous_query_time_period(self):
        """Test handling of ambiguous time period queries."""
        config = Config()
        pipeline = QueryPipeline(config=config)
        context = QueryContext(original_query="show me costs for last")

        suggestions = pipeline.handle_ambiguous_query(context)

        assert len(suggestions) > 0
        assert any("time period" in suggestion.lower() for suggestion in suggestions)

    def test_handle_ambiguous_query_service(self):
        """Test handling of ambiguous service queries."""
        config = Config()
        pipeline = QueryPipeline(config=config)
        context = QueryContext(original_query="show me compute costs")

        suggestions = pipeline.handle_ambiguous_query(context)

        assert len(suggestions) > 0
        assert any("EC2" in suggestion for suggestion in suggestions)

    def test_handle_ambiguous_query_cost_type(self):
        """Test handling of ambiguous cost type queries."""
        config = Config()
        pipeline = QueryPipeline(config=config)
        context = QueryContext(original_query="show me cost")

        suggestions = pipeline.handle_ambiguous_query(context)

        assert len(suggestions) > 0
        assert any(
            "total" in suggestion.lower() or "breakdown" in suggestion.lower()
            for suggestion in suggestions
        )

    def test_get_query_suggestions_partial(self):
        """Test getting query suggestions with partial input."""
        config = Config()
        pipeline = QueryPipeline(config=config)

        suggestions = pipeline.get_query_suggestions("EC2")

        assert len(suggestions) > 0
        assert all("EC2" in suggestion for suggestion in suggestions)

    def test_get_query_suggestions_empty(self):
        """Test getting query suggestions with empty input."""
        config = Config()
        pipeline = QueryPipeline(config=config)

        suggestions = pipeline.get_query_suggestions("")

        assert len(suggestions) > 0
        assert len(suggestions) <= 5  # Should return top 5

    @patch("src.aws_cost_cli.aws_client.AWSCostClient.estimate_query_cost")
    def test_validate_query_complexity(self, mock_estimate_cost):
        """Test query complexity validation."""
        mock_estimate_cost.return_value = {
            "complexity_level": "medium",
            "warnings": ["Query spans more than 3 months"],
        }

        config = Config()
        pipeline = QueryPipeline(config=config)

        # Initialize AWS client manually for testing
        from src.aws_cost_cli.aws_client import AWSCostClient

        pipeline.aws_client = AWSCostClient(cache_manager=pipeline.cache_manager)

        query_params = QueryParameters()
        complexity = pipeline.validate_query_complexity(query_params)

        assert complexity["complexity_level"] == "medium"
        assert len(complexity["warnings"]) > 0
        mock_estimate_cost.assert_called_once_with(query_params)

    def test_validate_query_complexity_no_aws_client(self):
        """Test query complexity validation without AWS client."""
        config = Config()
        pipeline = QueryPipeline(config=config)

        query_params = QueryParameters()
        complexity = pipeline.validate_query_complexity(query_params)

        assert complexity["complexity_level"] == "unknown"
        assert "AWS client not initialized" in complexity["warnings"][0]

    @patch("src.aws_cost_cli.cache_manager.CacheManager.get_cache_stats")
    @patch("src.aws_cost_cli.aws_client.AWSCostClient.check_service_availability")
    def test_get_pipeline_status(self, mock_check_service, mock_get_cache_stats):
        """Test getting pipeline status."""
        mock_get_cache_stats.return_value = {"total_entries": 5}
        mock_check_service.return_value = {"available": True, "response_time_ms": 150.5}

        config = Config()
        pipeline = QueryPipeline(config=config)

        # Initialize AWS client manually for testing
        from src.aws_cost_cli.aws_client import AWSCostClient

        pipeline.aws_client = AWSCostClient(cache_manager=pipeline.cache_manager)

        status = pipeline.get_pipeline_status()

        assert status["config_loaded"] is True
        assert status["cache_manager_initialized"] is True
        assert status["query_parser_initialized"] is True
        assert status["aws_client_initialized"] is True
        assert status["cache_healthy"] is True
        assert status["cache_entries"] == 5
        assert status["aws_service_healthy"] is True
        assert status["aws_response_time_ms"] == 150.5

    @patch("src.aws_cost_cli.cache_manager.CacheManager.cleanup_expired_cache")
    def test_cleanup(self, mock_cleanup_cache):
        """Test pipeline cleanup."""
        mock_cleanup_cache.return_value = 3

        config = Config()
        pipeline = QueryPipeline(config=config)

        # Should not raise any exceptions
        pipeline.cleanup()

        mock_cleanup_cache.assert_called_once()


class TestQueryPipelineFallback:
    """Test query pipeline fallback functionality."""

    @patch("src.aws_cost_cli.aws_client.CredentialManager.validate_credentials")
    @patch("src.aws_cost_cli.aws_client.AWSCostClient.validate_permissions")
    @patch("src.aws_cost_cli.query_processor.QueryParser.parse_query")
    @patch("src.aws_cost_cli.query_processor.FallbackParser.parse_query")
    @patch("src.aws_cost_cli.query_processor.QueryParser._convert_to_query_parameters")
    @patch("src.aws_cost_cli.query_processor.QueryParser.validate_parameters")
    @patch("src.aws_cost_cli.aws_client.AWSCostClient.get_cost_and_usage")
    @patch("src.aws_cost_cli.response_formatter.ResponseGenerator.format_response")
    def test_llm_fallback_to_pattern_matching(
        self,
        mock_format_response,
        mock_get_cost_data,
        mock_validate_params,
        mock_convert_params,
        mock_fallback_parse,
        mock_parse_query,
        mock_validate_permissions,
        mock_validate_credentials,
    ):
        """Test fallback from LLM to pattern matching."""
        from src.aws_cost_cli.exceptions import LLMProviderError

        # Setup mocks
        mock_validate_credentials.return_value = True
        mock_validate_permissions.return_value = True

        # Mock LLM failure
        mock_parse_query.side_effect = LLMProviderError(
            "LLM not available", provider="openai"
        )

        # Mock fallback success
        mock_fallback_parse.return_value = {"service": "EC2", "granularity": "MONTHLY"}
        mock_query_params = QueryParameters()
        mock_convert_params.return_value = mock_query_params
        mock_validate_params.return_value = True

        mock_cost_data = CostData(
            results=[],
            time_period=TimePeriod(
                start=datetime.now(timezone.utc), end=datetime.now(timezone.utc)
            ),
            total_cost=CostAmount(amount=100.0),
            group_definitions=[],
        )
        mock_get_cost_data.return_value = mock_cost_data
        mock_format_response.return_value = "Test response"

        # Create pipeline and context
        config = Config()
        pipeline = QueryPipeline(config=config)
        context = QueryContext(original_query="EC2 costs last month")

        # Process query
        result = pipeline.process_query(context)

        # Verify fallback was used
        assert result.success is True
        assert result.fallback_used is True
        assert result.metadata["parsing_method"] == "fallback"
        assert "llm_error" in result.metadata

        # Verify both parsers were called
        mock_parse_query.assert_called_once()
        mock_fallback_parse.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
