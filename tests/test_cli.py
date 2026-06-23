"""Tests for CLI interface."""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from click.testing import CliRunner
from datetime import datetime, timezone
from decimal import Decimal

from src.aws_cost_cli.cli import cli
from src.aws_cost_cli.query_pipeline import QueryResult
from src.aws_cost_cli.exceptions import (
    AWSCredentialsError,
    AWSPermissionsError,
    AWSAPIError,
    QueryParsingError,
)
from src.aws_cost_cli.models import (
    CostData,
    CostResult,
    CostAmount,
    TimePeriod,
    QueryParameters,
    Config,
)


class TestCLI:
    """Test cases for CLI interface."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

        # Create test data
        self.time_period = TimePeriod(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 31, tzinfo=timezone.utc),
        )

        self.cost_data = CostData(
            results=[
                CostResult(
                    time_period=self.time_period,
                    total=CostAmount(Decimal("123.45"), "USD"),
                    groups=[],
                    estimated=False,
                )
            ],
            time_period=self.time_period,
            total_cost=CostAmount(Decimal("123.45"), "USD"),
            currency="USD",
        )

        self.query_params = QueryParameters(service="EC2", time_period=self.time_period)

        self.config = Config(
            llm_provider="openai",
            llm_config={"provider": "openai", "api_key": "test-key"},
            output_format="simple",
        )

    # The ``query`` command delegates the full flow (credentials, parsing,
    # fetching, formatting) to ``QueryPipeline``. These tests therefore mock the
    # pipeline and assert how the CLI renders the resulting ``QueryResult`` --
    # matching the provider-override tests below. (They previously mocked the
    # pre-pipeline orchestration directly, including a ``cli.ResponseGenerator``
    # symbol that no longer exists.)

    def _success_result(self, formatted_response="Test response", **kwargs):
        """Build a successful QueryResult for the pipeline mock."""
        return QueryResult(
            success=True,
            cost_data=self.cost_data,
            formatted_response=formatted_response,
            metadata={"original_query": "test"},
            processing_time_ms=12.3,
            **kwargs,
        )

    @patch("src.aws_cost_cli.cli.QueryPipeline")
    def test_query_command_success(self, mock_pipeline):
        """Test successful query command execution."""
        mock_pipeline.return_value.config = self.config
        mock_pipeline.return_value.process_query.return_value = self._success_result(
            "Test response"
        )

        result = self.runner.invoke(cli, ["query", "How much did I spend on EC2?"])

        assert result.exit_code == 0
        assert "Test response" in result.output
        mock_pipeline.return_value.process_query.assert_called_once()

    @patch("src.aws_cost_cli.cli.QueryPipeline")
    def test_query_command_invalid_credentials(self, mock_pipeline):
        """Test query command with invalid credentials."""
        mock_pipeline.return_value.config = self.config
        mock_pipeline.return_value.process_query.return_value = QueryResult(
            success=False, error=AWSCredentialsError(), metadata={}
        )

        result = self.runner.invoke(cli, ["query", "How much did I spend on EC2?"])

        assert result.exit_code == 1
        assert "AWS credentials not found or invalid" in result.output

    @patch("src.aws_cost_cli.cli.QueryPipeline")
    def test_query_command_insufficient_permissions(self, mock_pipeline):
        """Test query command with insufficient permissions."""
        mock_pipeline.return_value.config = self.config
        mock_pipeline.return_value.process_query.return_value = QueryResult(
            success=False, error=AWSPermissionsError(), metadata={}
        )

        result = self.runner.invoke(cli, ["query", "How much did I spend on EC2?"])

        assert result.exit_code == 1
        assert "Insufficient AWS permissions" in result.output

    @patch("src.aws_cost_cli.cli.QueryPipeline")
    def test_query_command_parse_error(self, mock_pipeline):
        """Test query command with query parsing error."""
        mock_pipeline.return_value.config = self.config
        mock_pipeline.return_value.process_query.return_value = QueryResult(
            success=False,
            error=QueryParsingError(original_query="Invalid query"),
            metadata={},
        )
        # handle_ambiguous_query is called for QueryParsingError to print tips
        mock_pipeline.return_value.handle_ambiguous_query.return_value = []

        result = self.runner.invoke(cli, ["query", "Invalid query"])

        assert result.exit_code == 1
        assert "Failed to parse query" in result.output

    @patch("src.aws_cost_cli.cli.QueryPipeline")
    def test_query_command_aws_api_error(self, mock_pipeline):
        """Test query command with AWS API error."""
        mock_pipeline.return_value.config = self.config
        mock_pipeline.return_value.process_query.return_value = QueryResult(
            success=False,
            error=AWSAPIError("Failed to fetch cost data: AWS API Error"),
            metadata={},
        )

        result = self.runner.invoke(cli, ["query", "How much did I spend on EC2?"])

        assert result.exit_code == 1
        assert "Failed to fetch cost data" in result.output

    @patch("src.aws_cost_cli.cli.QueryPipeline")
    def test_query_command_with_cache(self, mock_pipeline):
        """Test query command using cached data."""
        mock_pipeline.return_value.config = self.config
        mock_pipeline.return_value.process_query.return_value = self._success_result(
            "Cached response", cache_hit=True
        )

        result = self.runner.invoke(cli, ["query", "How much did I spend on EC2?"])

        assert result.exit_code == 0
        assert "Cached response" in result.output
        # Cache is enabled (not fresh) so the pipeline receives fresh_data=False.
        context = mock_pipeline.return_value.process_query.call_args[0][0]
        assert context.fresh_data is False

    @patch("src.aws_cost_cli.cli.QueryPipeline")
    def test_query_command_fresh_flag(self, mock_pipeline):
        """Test query command with fresh flag bypassing cache."""
        mock_pipeline.return_value.config = self.config
        mock_pipeline.return_value.process_query.return_value = self._success_result(
            "Fresh response", cache_hit=False
        )

        result = self.runner.invoke(
            cli, ["query", "How much did I spend on EC2?", "--fresh"]
        )

        assert result.exit_code == 0
        assert "Fresh response" in result.output
        # The --fresh flag must be propagated to the pipeline as fresh_data=True.
        context = mock_pipeline.return_value.process_query.call_args[0][0]
        assert context.fresh_data is True

    @patch("src.aws_cost_cli.cli.QueryPipeline")
    def test_query_command_json_output(self, mock_pipeline):
        """Test query command with JSON output format."""
        config = Config(
            llm_provider="openai",
            llm_config={"provider": "openai", "api_key": "test-key"},
            output_format="json",
        )
        mock_pipeline.return_value.config = config
        mock_pipeline.return_value.process_query.return_value = self._success_result()

        result = self.runner.invoke(cli, ["query", "How much did I spend on EC2?"])

        assert result.exit_code == 0
        # Should output valid JSON (no rich panels/headers around it)
        output_data = json.loads(result.output)
        assert "query" in output_data
        assert "total_cost" in output_data
        assert output_data["total_cost"]["amount"] == 123.45

    @patch("src.aws_cost_cli.cli.ConfigManager")
    def test_configure_command_openai(self, mock_config_manager):
        """Test configure command for OpenAI provider."""
        mock_config_manager.return_value.load_config.side_effect = FileNotFoundError()
        mock_config_manager.return_value.get_default_config_path.return_value = (
            "/test/config.yaml"
        )

        # Mock QueryParser to simulate successful configuration test
        with patch("src.aws_cost_cli.cli.QueryParser") as mock_query_parser:
            mock_query_parser.return_value.parse_query.return_value = self.query_params

            result = self.runner.invoke(
                cli,
                [
                    "configure",
                    "--provider",
                    "openai",
                    "--api-key",
                    "sk-test123",
                    "--model",
                    "gpt-4",
                ],
            )

        assert result.exit_code == 0
        assert "Configuration saved successfully" in result.output
        assert "Provider: openai" in result.output
        assert "Model: gpt-4" in result.output
        mock_config_manager.return_value.save_config.assert_called_once()

    @patch("src.aws_cost_cli.cli.ConfigManager")
    def test_configure_command_anthropic(self, mock_config_manager):
        """Test configure command for Anthropic provider."""
        mock_config_manager.return_value.load_config.side_effect = FileNotFoundError()
        mock_config_manager.return_value.get_default_config_path.return_value = (
            "/test/config.yaml"
        )

        with patch("src.aws_cost_cli.cli.QueryParser") as mock_query_parser:
            mock_query_parser.return_value.parse_query.return_value = self.query_params

            result = self.runner.invoke(
                cli,
                ["configure", "--provider", "anthropic", "--api-key", "sk-ant-test123"],
            )

        assert result.exit_code == 0
        assert "Configuration saved successfully" in result.output
        assert "Provider: anthropic" in result.output
        mock_config_manager.return_value.save_config.assert_called_once()

    @patch("src.aws_cost_cli.cli.ConfigManager")
    def test_configure_command_ollama(self, mock_config_manager):
        """Test configure command for Ollama provider."""
        mock_config_manager.return_value.load_config.side_effect = FileNotFoundError()
        mock_config_manager.return_value.get_default_config_path.return_value = (
            "/test/config.yaml"
        )

        with patch("src.aws_cost_cli.cli.QueryParser") as mock_query_parser:
            mock_query_parser.return_value.parse_query.return_value = self.query_params

            result = self.runner.invoke(
                cli,
                [
                    "configure",
                    "--provider",
                    "ollama",
                    "--model",
                    "llama2",
                    "--base-url",
                    "http://localhost:11434",
                ],
            )

        assert result.exit_code == 0
        assert "Configuration saved successfully" in result.output
        assert "Provider: ollama" in result.output
        assert "Model: llama2" in result.output
        mock_config_manager.return_value.save_config.assert_called_once()

    @patch("src.aws_cost_cli.cli.ConfigManager")
    def test_configure_command_bedrock(self, mock_config_manager):
        """Test configure command for Bedrock provider."""
        mock_config_manager.return_value.load_config.side_effect = FileNotFoundError()
        mock_config_manager.return_value.get_default_config_path.return_value = (
            "/test/config.yaml"
        )

        with patch("src.aws_cost_cli.cli.QueryParser") as mock_query_parser:
            mock_query_parser.return_value.parse_query.return_value = self.query_params

            result = self.runner.invoke(
                cli,
                [
                    "configure",
                    "--provider",
                    "bedrock",
                    "--model",
                    "anthropic.claude-3-haiku-20240307-v1:0",
                    "--region",
                    "us-west-2",
                    "--profile",
                    "production",
                ],
            )

        assert result.exit_code == 0
        assert "Configuration saved successfully" in result.output
        assert "Provider: bedrock" in result.output
        assert "Model: anthropic.claude-3-haiku-20240307-v1:0" in result.output
        assert "Region: us-west-2" in result.output
        assert "AWS Profile: production" in result.output
        mock_config_manager.return_value.save_config.assert_called_once()

    @patch("src.aws_cost_cli.cli.CredentialManager")
    def test_list_profiles_command(self, mock_credential_manager):
        """Test list-profiles command."""
        mock_credential_manager.return_value.get_available_profiles.return_value = [
            "default",
            "production",
            "staging",
        ]
        mock_credential_manager.return_value.validate_credentials.side_effect = [
            True,
            True,
            False,  # default and production valid, staging invalid
        ]

        result = self.runner.invoke(cli, ["list-profiles"])

        assert result.exit_code == 0
        assert "Available AWS Profiles" in result.output
        assert "1. default ✅" in result.output
        assert "2. production ✅" in result.output
        assert "3. staging ❌" in result.output

    @patch("src.aws_cost_cli.cli.CredentialManager")
    def test_list_profiles_command_no_profiles(self, mock_credential_manager):
        """Test list-profiles command with no profiles."""
        mock_credential_manager.return_value.get_available_profiles.return_value = []

        result = self.runner.invoke(cli, ["list-profiles"])

        assert result.exit_code == 0
        assert "No AWS profiles found" in result.output

    @patch("src.aws_cost_cli.cli.ConfigManager")
    def test_show_config_command(self, mock_config_manager):
        """Test show-config command."""
        mock_config_manager.return_value.load_config.return_value = self.config
        mock_config_manager.return_value.get_default_config_path.return_value = (
            "/test/config.yaml"
        )

        result = self.runner.invoke(cli, ["show-config"])

        assert result.exit_code == 0
        assert "Current Configuration" in result.output
        assert "LLM Provider: openai" in result.output
        assert "Output Format: simple" in result.output

    @patch("src.aws_cost_cli.cli.ConfigManager")
    def test_show_config_masks_nested_api_key(self, mock_config_manager):
        """show-config must never print an API key verbatim, even when nested."""
        nested_config = Config(
            llm_provider="openai",
            llm_config={"openai": {"api_key": "sk-supersecret-1234567890"}},
            output_format="simple",
        )
        mock_config_manager.return_value.load_config.return_value = nested_config
        mock_config_manager.return_value.get_default_config_path.return_value = (
            "/test/config.yaml"
        )

        result = self.runner.invoke(cli, ["show-config"])

        assert result.exit_code == 0
        assert "sk-supersecret-1234567890" not in result.output

    @patch("src.aws_cost_cli.cli.ConfigManager")
    def test_show_config_command_no_config(self, mock_config_manager):
        """Test show-config command with no configuration file."""
        mock_config_manager.return_value.load_config.side_effect = FileNotFoundError()
        mock_config_manager.return_value.get_default_config_path.return_value = (
            "/test/config.yaml"
        )

        result = self.runner.invoke(cli, ["show-config"])

        assert result.exit_code == 0
        assert "No configuration file found" in result.output

    @patch("src.aws_cost_cli.cli.CacheManager")
    def test_clear_cache_command(self, mock_cache_manager):
        """Test clear-cache command."""
        mock_cache_manager.return_value.clear_cache.return_value = 5

        result = self.runner.invoke(cli, ["clear-cache"], input="y\n")

        assert result.exit_code == 0
        assert "Cache cleared successfully" in result.output
        mock_cache_manager.return_value.clear_cache.assert_called_once()

    @patch("src.aws_cost_cli.cli.ConfigManager")
    @patch("src.aws_cost_cli.cli.CredentialManager")
    @patch("src.aws_cost_cli.cli.AWSCostClient")
    @patch("src.aws_cost_cli.cli.CacheManager")
    @patch("src.aws_cost_cli.cli.QueryParser")
    def test_test_command(
        self,
        mock_query_parser,
        mock_cache_manager,
        mock_aws_client,
        mock_credential_manager,
        mock_config_manager,
    ):
        """Test the test command."""
        mock_config_manager.return_value.load_config.return_value = self.config
        mock_credential_manager.return_value.get_available_profiles.return_value = [
            "default"
        ]
        mock_credential_manager.return_value.validate_credentials.return_value = True
        mock_aws_client.return_value.validate_permissions.return_value = True
        mock_query_parser.return_value.parse_query.return_value = self.query_params
        mock_cache_manager.return_value.cache_data.return_value = True
        mock_cache_manager.return_value.get_cached_data.return_value = {"test": "data"}

        result = self.runner.invoke(cli, ["test"])

        assert result.exit_code == 0
        assert "System test completed" in result.output
        assert "Configuration loaded successfully" in result.output
        assert "AWS Cost Explorer permissions are valid" in result.output

    def test_cli_version(self):
        """Test CLI version option."""
        result = self.runner.invoke(cli, ["--version"])

        assert result.exit_code == 0
        assert "1.0.0" in result.output

    def test_cli_help(self):
        """Test CLI help."""
        result = self.runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "AWS Cost Explorer CLI" in result.output
        assert "query" in result.output
        assert "configure" in result.output

    def test_query_help(self):
        """Test query command help."""
        result = self.runner.invoke(cli, ["query", "--help"])

        assert result.exit_code == 0
        assert "Query AWS costs using natural language" in result.output
        assert "--profile" in result.output
        assert "--fresh" in result.output
        assert "--format" in result.output
        assert "--llm-provider" in result.output

    @patch("src.aws_cost_cli.cli.ConfigManager")
    @patch("src.aws_cost_cli.cli.CredentialManager")
    @patch("src.aws_cost_cli.cli.QueryPipeline")
    def test_query_command_with_provider_override(
        self, mock_pipeline, mock_credential_manager, mock_config_manager
    ):
        """Test query command with LLM provider override."""
        # Mock configuration
        mock_config_manager.return_value.load_config.return_value = self.config
        mock_credential_manager.return_value.validate_credentials.return_value = True

        # Mock pipeline
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.process_query.return_value = Mock(
            cost_data=self.cost_data,
            formatted_response="Test response with Gemini override"
        )
        mock_pipeline.return_value = mock_pipeline_instance

        # Run query with provider override
        result = self.runner.invoke(cli, [
            "query", 
            "EC2 costs last month",
            "--llm-provider", "gemini"
        ])

        assert result.exit_code == 0
        assert "Test response with Gemini override" in result.output

        # Verify pipeline was called with provider override
        mock_pipeline_instance.process_query.assert_called_once()
        call_args = mock_pipeline_instance.process_query.call_args[0][0]
        assert call_args.llm_provider_override == "gemini"

    @patch("src.aws_cost_cli.cli.ConfigManager")
    @patch("src.aws_cost_cli.cli.CredentialManager")
    def test_query_command_invalid_provider_override(
        self, mock_credential_manager, mock_config_manager
    ):
        """Test query command with invalid provider override."""
        # Mock configuration
        mock_config_manager.return_value.load_config.return_value = self.config
        mock_credential_manager.return_value.validate_credentials.return_value = True

        # Run query with invalid provider
        result = self.runner.invoke(cli, [
            "query", 
            "EC2 costs last month",
            "--llm-provider", "invalid-provider"
        ])

        assert result.exit_code != 0
        assert "Invalid value for '--llm-provider'" in result.output

    @patch("src.aws_cost_cli.cli.ConfigManager")
    @patch("src.aws_cost_cli.cli.CredentialManager")
    @patch("src.aws_cost_cli.cli.QueryPipeline")
    def test_query_command_provider_override_validation_error(
        self, mock_pipeline, mock_credential_manager, mock_config_manager
    ):
        """Test query command when provider override is not configured."""
        from src.aws_cost_cli.exceptions import ValidationError
        
        # Mock configuration
        mock_config_manager.return_value.load_config.return_value = self.config
        mock_credential_manager.return_value.validate_credentials.return_value = True

        # Mock pipeline to raise validation error for unconfigured provider
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.process_query.side_effect = ValidationError(
            "Gemini provider is not configured. Please set GEMINI_API_KEY environment variable"
        )
        mock_pipeline.return_value = mock_pipeline_instance

        # Run query with unconfigured provider
        result = self.runner.invoke(cli, [
            "query", 
            "EC2 costs last month",
            "--llm-provider", "gemini"
        ])

        assert result.exit_code != 0
        assert "Gemini provider is not configured" in result.output
        assert "GEMINI_API_KEY" in result.output

    @patch("src.aws_cost_cli.cli.ConfigManager")
    @patch("src.aws_cost_cli.cli.CredentialManager")
    @patch("src.aws_cost_cli.cli.QueryPipeline")
    def test_query_command_provider_override_preserves_config(
        self, mock_pipeline, mock_credential_manager, mock_config_manager
    ):
        """Test that provider override doesn't modify the loaded configuration."""
        # Mock configuration with OpenAI as default
        config_with_openai = Config(
            llm_provider="openai",
            llm_config={"openai": {"api_key": "sk-test"}}
        )
        mock_config_manager.return_value.load_config.return_value = config_with_openai
        mock_credential_manager.return_value.validate_credentials.return_value = True

        # Mock pipeline
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.process_query.return_value = Mock(
            cost_data=self.cost_data,
            formatted_response="Test response"
        )
        mock_pipeline.return_value = mock_pipeline_instance

        # Run query with Gemini override
        result = self.runner.invoke(cli, [
            "query", 
            "EC2 costs last month",
            "--llm-provider", "gemini"
        ])

        assert result.exit_code == 0

        # Verify the original config still has OpenAI as default
        loaded_config = mock_config_manager.return_value.load_config.return_value
        assert loaded_config.llm_provider == "openai"

        # But the query context should have the override
        call_args = mock_pipeline_instance.process_query.call_args[0][0]
        assert call_args.llm_provider_override == "gemini"


if __name__ == "__main__":
    pytest.main([__file__])
