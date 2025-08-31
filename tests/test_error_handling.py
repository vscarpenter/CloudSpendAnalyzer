"""Tests for comprehensive error handling."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from botocore.exceptions import ClientError, NoCredentialsError, EndpointConnectionError
import requests

from src.aws_cost_cli.exceptions import (
    AWSCredentialsError,
    AWSPermissionsError,
    AWSAPIError,
    NetworkError,
    QueryParsingError,
    LLMProviderError,
    CacheError,
    ConfigurationError,
    ValidationError,
    format_error_message,
    handle_aws_client_error,
    handle_network_error,
    is_retryable_error,
    get_retry_delay,
)
from src.aws_cost_cli.aws_client import AWSCostClient, CredentialManager
from src.aws_cost_cli.query_processor import (
    QueryParser,
    OpenAIProvider,
    AnthropicProvider,
    OllamaProvider,
)
from src.aws_cost_cli.models import (
    QueryParameters,
    TimePeriod,
    TimePeriodGranularity,
    MetricType,
)
from datetime import datetime, timezone, timedelta


class TestCustomExceptions:
    """Test custom exception classes."""

    def test_aws_credentials_error(self):
        """Test AWSCredentialsError creation and properties."""
        error = AWSCredentialsError()
        assert "AWS credentials not found or invalid" in error.message
        assert error.error_code == "CREDENTIALS_ERROR"
        assert len(error.suggestions) > 0

        # Test with profile
        error_with_profile = AWSCredentialsError(profile="test-profile")
        assert "test-profile" in error_with_profile.message
        assert error_with_profile.profile == "test-profile"

    def test_aws_permissions_error(self):
        """Test AWSPermissionsError creation and properties."""
        error = AWSPermissionsError()
        assert "Insufficient AWS permissions" in error.message
        assert error.error_code == "PERMISSIONS_ERROR"
        assert "ce:GetCostAndUsage" in str(error.required_permissions)
        assert len(error.suggestions) > 0

    def test_aws_api_error(self):
        """Test AWSAPIError creation and properties."""
        error = AWSAPIError("Test error", "ThrottlingException", "Rate limit exceeded")
        assert "Test error" in error.message
        assert error.aws_error_code == "ThrottlingException"
        assert error.aws_error_message == "Rate limit exceeded"
        assert "wait a few minutes" in str(error.suggestions).lower()

    def test_query_parsing_error(self):
        """Test QueryParsingError creation and properties."""
        error = QueryParsingError(original_query="test query")
        assert "test query" in error.message
        assert error.original_query == "test query"
        assert len(error.suggestions) > 0

    def test_llm_provider_error(self):
        """Test LLMProviderError creation and properties."""
        error = LLMProviderError(
            "Test error", provider="openai", fallback_available=True
        )
        assert "Test error" in error.message
        assert error.provider == "openai"
        assert error.fallback_available is True
        assert "fallback" in str(error.suggestions).lower()


class TestErrorHandlers:
    """Test error handling utility functions."""

    def test_handle_aws_client_error_access_denied(self):
        """Test handling of AccessDenied errors."""
        client_error = ClientError(
            error_response={
                "Error": {"Code": "AccessDenied", "Message": "Access denied"}
            },
            operation_name="GetCostAndUsage",
        )

        result = handle_aws_client_error(client_error, "test operation")
        assert isinstance(result, AWSPermissionsError)
        assert "Access denied" in result.message

    def test_handle_aws_client_error_throttling(self):
        """Test handling of ThrottlingException errors."""
        client_error = ClientError(
            error_response={
                "Error": {
                    "Code": "ThrottlingException",
                    "Message": "Rate limit exceeded",
                }
            },
            operation_name="GetCostAndUsage",
        )

        result = handle_aws_client_error(client_error, "test operation")
        assert isinstance(result, AWSAPIError)
        assert result.aws_error_code == "ThrottlingException"
        assert "rate limit" in result.message.lower()

    def test_handle_network_error(self):
        """Test handling of network errors."""
        network_error = ConnectionError("Connection failed")

        result = handle_network_error(network_error, "test operation")
        assert isinstance(result, NetworkError)
        assert "Connection failed" in result.message

    def test_is_retryable_error(self):
        """Test retryable error detection."""
        # Retryable errors
        throttling_error = AWSAPIError("Test", "ThrottlingException")
        assert is_retryable_error(throttling_error) is True

        service_unavailable_error = AWSAPIError("Test", "ServiceUnavailable")
        assert is_retryable_error(service_unavailable_error) is True

        network_error = NetworkError("Test")
        assert is_retryable_error(network_error) is True

        # Non-retryable errors
        credentials_error = AWSCredentialsError()
        assert is_retryable_error(credentials_error) is False

        permissions_error = AWSPermissionsError()
        assert is_retryable_error(permissions_error) is False

    def test_get_retry_delay(self):
        """Test retry delay calculation."""
        # Test exponential backoff
        delay_0 = get_retry_delay(0)
        delay_1 = get_retry_delay(1)
        delay_2 = get_retry_delay(2)

        assert delay_0 < delay_1 < delay_2
        assert delay_0 >= 1.0  # Base delay
        assert delay_2 <= 60.0  # Max delay

    def test_format_error_message(self):
        """Test error message formatting."""
        error = AWSCredentialsError()

        # With suggestions
        message_with_suggestions = format_error_message(error, include_suggestions=True)
        assert "❌" in message_with_suggestions
        assert "💡 Suggestions:" in message_with_suggestions
        assert len(message_with_suggestions.split("\n")) > 2

        # Without suggestions
        message_without_suggestions = format_error_message(
            error, include_suggestions=False
        )
        assert "❌" in message_without_suggestions
        assert "💡 Suggestions:" not in message_without_suggestions


class TestAWSClientErrorHandling:
    """Test error handling in AWS client."""

    @patch("boto3.Session")
    def test_credential_manager_validate_credentials_no_credentials(self, mock_session):
        """Test credential validation with no credentials."""
        mock_session.return_value.client.return_value.get_caller_identity.side_effect = (
            NoCredentialsError()
        )

        credential_manager = CredentialManager()
        result = credential_manager.validate_credentials()
        assert result is False

    @patch("boto3.Session")
    def test_credential_manager_get_caller_identity_error(self, mock_session):
        """Test get_caller_identity with error handling."""
        mock_session.return_value.client.return_value.get_caller_identity.side_effect = (
            NoCredentialsError()
        )

        credential_manager = CredentialManager()
        with pytest.raises(AWSCredentialsError):
            credential_manager.get_caller_identity()

    @patch("boto3.Session")
    def test_aws_client_get_cost_and_usage_throttling(self, mock_session):
        """Test get_cost_and_usage with throttling error and retry."""
        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client

        # First call raises throttling, second succeeds
        mock_client.get_cost_and_usage.side_effect = [
            ClientError(
                error_response={
                    "Error": {"Code": "ThrottlingException", "Message": "Rate limit"}
                },
                operation_name="GetCostAndUsage",
            ),
            {"ResultsByTime": [], "DimensionKey": {}, "GroupDefinitions": []},
        ]

        aws_client = AWSCostClient()
        params = QueryParameters()

        # Should succeed after retry
        with patch("time.sleep"):  # Mock sleep to speed up test
            result = aws_client.get_cost_and_usage(params)
            assert result is not None

    @patch("boto3.Session")
    def test_aws_client_get_cost_and_usage_access_denied(self, mock_session):
        """Test get_cost_and_usage with access denied error."""
        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client

        mock_client.get_cost_and_usage.side_effect = ClientError(
            error_response={
                "Error": {"Code": "AccessDenied", "Message": "Access denied"}
            },
            operation_name="GetCostAndUsage",
        )

        aws_client = AWSCostClient()
        params = QueryParameters()

        with pytest.raises(AWSPermissionsError):
            aws_client.get_cost_and_usage(params)

    @patch("boto3.Session")
    def test_aws_client_network_error(self, mock_session):
        """Test AWS client with network error."""
        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client

        mock_client.get_cost_and_usage.side_effect = EndpointConnectionError(
            endpoint_url="https://ce.us-east-1.amazonaws.com"
        )

        aws_client = AWSCostClient()
        params = QueryParameters()

        with pytest.raises(NetworkError):
            aws_client.get_cost_and_usage(params)


class TestQueryProcessorErrorHandling:
    """Test error handling in query processor."""

    def test_openai_provider_not_available(self):
        """Test OpenAI provider when not available."""
        provider = OpenAIProvider(api_key="")

        with pytest.raises(LLMProviderError) as exc_info:
            provider.parse_query("test query")

        assert exc_info.value.provider == "openai"

    @patch("openai.OpenAI")
    def test_openai_provider_api_error(self, mock_openai):
        """Test OpenAI provider with API error."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API key invalid")

        provider = OpenAIProvider(api_key="test-key")

        with pytest.raises(LLMProviderError) as exc_info:
            provider.parse_query("test query")

        assert "API key" in str(exc_info.value.message)

    def test_anthropic_provider_not_available(self):
        """Test Anthropic provider when not available."""
        provider = AnthropicProvider(api_key="")

        with pytest.raises(LLMProviderError) as exc_info:
            provider.parse_query("test query")

        assert exc_info.value.provider == "anthropic"

    @patch("requests.get")
    def test_ollama_provider_not_available(self, mock_get):
        """Test Ollama provider when not available."""
        mock_get.side_effect = requests.exceptions.ConnectionError()

        provider = OllamaProvider()
        assert provider.is_available() is False

        with pytest.raises(LLMProviderError) as exc_info:
            provider.parse_query("test query")

        assert exc_info.value.provider == "ollama"

    @patch("requests.post")
    @patch("requests.get")
    def test_ollama_provider_connection_error(self, mock_get, mock_post):
        """Test Ollama provider with connection error."""
        mock_get.return_value.status_code = 200  # Available
        mock_post.side_effect = requests.exceptions.ConnectionError()

        provider = OllamaProvider()

        with pytest.raises(LLMProviderError) as exc_info:
            provider.parse_query("test query")

        assert "connect" in str(exc_info.value.message).lower()

    def test_query_parser_empty_query(self):
        """Test query parser with empty query."""
        parser = QueryParser({})

        with pytest.raises(QueryParsingError) as exc_info:
            parser.parse_query("")

        assert "Empty query" in str(exc_info.value.message)

    def test_query_parser_validation_error(self):
        """Test query parser with validation error."""
        parser = QueryParser({})

        # Create invalid parameters (start date after end date)
        invalid_params = QueryParameters(
            time_period=TimePeriod(
                start=datetime(2024, 2, 1, tzinfo=timezone.utc),
                end=datetime(2024, 1, 1, tzinfo=timezone.utc),  # End before start
            )
        )

        with pytest.raises(ValidationError) as exc_info:
            parser.validate_parameters(invalid_params)

        assert "Start date must be before end date" in str(exc_info.value.message)

    def test_query_parser_invalid_granularity(self):
        """Test query parser with invalid granularity."""
        parser = QueryParser({})

        invalid_params = QueryParameters(granularity="INVALID_GRANULARITY")

        with pytest.raises(ValidationError) as exc_info:
            parser.validate_parameters(invalid_params)

        assert "Invalid granularity" in str(exc_info.value.message)

    def test_query_parser_invalid_metric(self):
        """Test query parser with invalid metric."""
        parser = QueryParser({})

        invalid_params = QueryParameters(metrics=["INVALID_METRIC"])

        with pytest.raises(ValidationError) as exc_info:
            parser.validate_parameters(invalid_params)

        assert "Invalid metric" in str(exc_info.value.message)

    def test_query_parser_future_date(self):
        """Test query parser with future date."""
        parser = QueryParser({})

        future_date = datetime.now(timezone.utc) + timedelta(days=30)
        invalid_params = QueryParameters(
            time_period=TimePeriod(
                start=future_date, end=future_date + timedelta(days=1)
            )
        )

        with pytest.raises(ValidationError) as exc_info:
            parser.validate_parameters(invalid_params)

        assert "cannot be in the future" in str(exc_info.value.message)

    def test_query_parser_date_range_too_large(self):
        """Test query parser with date range too large."""
        parser = QueryParser({})

        start_date = datetime(2010, 1, 1, tzinfo=timezone.utc)
        end_date = datetime(2024, 1, 1, tzinfo=timezone.utc)  # More than 5 years

        invalid_params = QueryParameters(
            time_period=TimePeriod(start=start_date, end=end_date)
        )

        with pytest.raises(ValidationError) as exc_info:
            parser.validate_parameters(invalid_params)

        assert "cannot exceed 5 years" in str(exc_info.value.message)


class TestCacheErrorHandling:
    """Test error handling in cache manager."""

    @patch("pathlib.Path.mkdir")
    def test_cache_manager_directory_creation_error(self, mock_mkdir):
        """Test cache manager with directory creation error."""
        from src.aws_cost_cli.cache_manager import CacheManager

        mock_mkdir.side_effect = PermissionError("Permission denied")

        # Should raise CacheError during initialization
        with pytest.raises(CacheError):
            CacheManager()

    @patch("builtins.open")
    @patch("pathlib.Path.mkdir")
    def test_cache_manager_file_write_error(self, mock_mkdir, mock_open):
        """Test cache manager with file write error."""
        from src.aws_cost_cli.cache_manager import CacheManager
        from src.aws_cost_cli.models import CostData, CostAmount, TimePeriod
        from datetime import datetime, timezone

        # Allow directory creation to succeed
        mock_mkdir.return_value = None

        # Make file write fail
        mock_open.side_effect = IOError("Disk full")

        cache_manager = CacheManager()

        # Create a simple CostData object for testing
        cost_data = CostData(
            results=[],
            time_period=TimePeriod(
                start=datetime.now(timezone.utc), end=datetime.now(timezone.utc)
            ),
            total_cost=CostAmount(amount=0.0),
            group_definitions=[],
        )

        with pytest.raises(CacheError):
            cache_manager.cache_data("test_key", cost_data)


class TestConfigurationErrorHandling:
    """Test error handling in configuration manager."""

    @patch("builtins.open")
    def test_config_manager_file_not_found(self, mock_open):
        """Test config manager with file not found."""
        from src.aws_cost_cli.config import ConfigManager

        mock_open.side_effect = FileNotFoundError("Config file not found")

        config_manager = ConfigManager()

        with pytest.raises(FileNotFoundError):
            config_manager.load_config("nonexistent.yaml")

    @patch("pathlib.Path.exists")
    @patch("builtins.open")
    @patch("yaml.safe_load")
    def test_config_manager_invalid_yaml(self, mock_yaml_load, mock_open, mock_exists):
        """Test config manager with invalid YAML."""
        from src.aws_cost_cli.config import ConfigManager
        import yaml

        # Mock file exists
        mock_exists.return_value = True

        # Mock file opening
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file

        # Mock YAML loading to fail
        mock_yaml_load.side_effect = yaml.YAMLError("Invalid YAML")

        config_manager = ConfigManager()

        with pytest.raises(ConfigurationError):
            config_manager._load_config_file("invalid.yaml")


if __name__ == "__main__":
    pytest.main([__file__])
