"""Integration tests for provider switching functionality."""

import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from click.testing import CliRunner

import pytest

from src.aws_cost_cli.cli import cli
from src.aws_cost_cli.query_pipeline import QueryPipeline, QueryContext
from src.aws_cost_cli.query_processor import QueryParser
from src.aws_cost_cli.models import Config
from src.aws_cost_cli.config import ConfigManager
from src.aws_cost_cli.exceptions import LLMProviderError, NetworkError


class TestProviderSwitchingIntegration:
    """Integration tests for provider switching functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, "test_config.yaml")

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("src.aws_cost_cli.query_pipeline.AWSCostClient")
    @patch("src.aws_cost_cli.query_pipeline.ResponseGenerator")
    def test_cli_provider_override_openai_to_gemini(
        self, mock_response_gen, mock_aws_client
    ):
        """Test CLI --llm-provider flag overrides configuration from OpenAI to Gemini."""
        # Create config with OpenAI as default
        config_content = """
llm_provider: openai
llm_config:
  openai:
    api_key: sk-test-openai-key
    model: gpt-3.5-turbo
  gemini:
    api_key: test-gemini-key
    model: gemini-1.5-flash
"""
        with open(self.config_file, "w") as f:
            f.write(config_content)

        # Mock successful AWS and response generation
        mock_aws_client.return_value.get_cost_and_usage.return_value = {
            "ResultsByTime": [
                {"Total": {"BlendedCost": {"Amount": "100.50", "Unit": "USD"}}}
            ]
        }
        mock_response_gen.return_value.generate_response.return_value = "Test response"

        with patch("google.generativeai.configure"), patch(
            "google.generativeai.GenerativeModel"
        ) as mock_model, patch("openai.OpenAI"):

            # Mock Gemini response
            mock_gemini_response = Mock()
            mock_gemini_response.text = '{"service": "EC2", "granularity": "MONTHLY"}'
            mock_gemini_client = Mock()
            mock_gemini_client.generate_content.return_value = mock_gemini_response
            mock_model.return_value = mock_gemini_client

            # Run query with Gemini override
            result = self.runner.invoke(
                cli,
                [
                    "query",
                    "EC2 costs last month",
                    "--llm-provider",
                    "gemini",
                    "--config-file",
                    self.config_file,
                ],
            )

            assert result.exit_code == 0
            # Verify Gemini was used (not OpenAI)
            mock_gemini_client.generate_content.assert_called_once()

    @patch("src.aws_cost_cli.query_pipeline.AWSCostClient")
    @patch("src.aws_cost_cli.query_pipeline.ResponseGenerator")
    def test_cli_provider_override_invalid_provider(
        self, mock_response_gen, mock_aws_client
    ):
        """Test CLI with invalid provider override."""
        # Create basic config
        config_content = """
llm_provider: openai
llm_config:
  openai:
    api_key: sk-test-key
"""
        with open(self.config_file, "w") as f:
            f.write(config_content)

        # Run query with invalid provider
        result = self.runner.invoke(
            cli,
            [
                "query",
                "EC2 costs last month",
                "--llm-provider",
                "invalid-provider",
                "--config-file",
                self.config_file,
            ],
        )

        assert result.exit_code != 0
        assert "Invalid value for '--llm-provider'" in result.output

    @patch("src.aws_cost_cli.query_pipeline.AWSCostClient")
    @patch("src.aws_cost_cli.query_pipeline.ResponseGenerator")
    def test_cli_provider_override_unconfigured_provider(
        self, mock_response_gen, mock_aws_client
    ):
        """Test CLI with unconfigured provider override."""
        # Create config with only OpenAI configured
        config_content = """
llm_provider: openai
llm_config:
  openai:
    api_key: sk-test-key
"""
        with open(self.config_file, "w") as f:
            f.write(config_content)

        # Run query with unconfigured Gemini provider
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            result = self.runner.invoke(
                cli,
                [
                    "query",
                    "EC2 costs last month",
                    "--llm-provider",
                    "gemini",
                    "--config-file",
                    self.config_file,
                ],
            )

        assert result.exit_code != 0
        assert "Gemini provider is not configured" in result.output
        assert "GEMINI_API_KEY" in result.output

    def test_automatic_provider_fallback_network_error(self):
        """Test automatic fallback when primary provider has network error."""
        config = Config(
            llm_provider="gemini",
            llm_config={
                "gemini": {"api_key": "test-gemini-key"},
                "ollama": {
                    "model": "gpt-oss:20b",
                    "base_url": "http://localhost:11434",
                },
            },
            fallback_providers=["ollama", "openai"],
        )

        with patch("google.generativeai.configure"), patch(
            "google.generativeai.GenerativeModel"
        ) as mock_gemini_model, patch("requests.get") as mock_ollama_get, patch(
            "requests.post"
        ) as mock_ollama_post:

            # Mock Gemini network error
            mock_gemini_client = Mock()
            mock_gemini_client.generate_content.side_effect = NetworkError(
                "Network error"
            )
            mock_gemini_model.return_value = mock_gemini_client

            # Mock Ollama success
            mock_ollama_get.return_value = Mock(status_code=200)
            mock_ollama_response = Mock()
            mock_ollama_response.status_code = 200
            mock_ollama_response.json.return_value = {
                "response": '{"service": "Lambda", "granularity": "DAILY"}'
            }
            mock_ollama_post.return_value = mock_ollama_response

            # Create query parser with fallback
            from dataclasses import asdict

            query_parser = QueryParser(config.llm_config, asdict(config))

            # Should fallback to Ollama
            result = query_parser.parse_query_with_fallback(
                "Lambda costs daily", "gemini"
            )

            assert result.service == "Lambda"
            assert result.granularity.value == "DAILY"

            # Verify Gemini was tried first, then Ollama succeeded
            mock_gemini_client.generate_content.assert_called_once()
            mock_ollama_post.assert_called_once()

    def test_automatic_provider_fallback_all_fail(self):
        """Test behavior when all providers fail."""
        config = Config(
            llm_provider="gemini",
            llm_config={
                "gemini": {"api_key": "test-gemini-key"},
                "ollama": {"model": "gpt-oss:20b"},
            },
            fallback_providers=["ollama"],
        )

        with patch("google.generativeai.configure"), patch(
            "google.generativeai.GenerativeModel"
        ) as mock_gemini_model, patch("requests.get") as mock_ollama_get:

            # Mock both providers failing
            mock_gemini_client = Mock()
            mock_gemini_client.generate_content.side_effect = LLMProviderError(
                "API error"
            )
            mock_gemini_model.return_value = mock_gemini_client

            mock_ollama_get.side_effect = Exception("Connection error")

            # Create query parser
            from dataclasses import asdict

            query_parser = QueryParser(config.llm_config, asdict(config))

            # Should fall back to the pattern parser after providers fail.
            result = query_parser.parse_query_with_fallback("test query", "gemini")

            assert result.service is None
            assert result.granularity.value == "MONTHLY"

    def test_provider_availability_detection(self):
        """Test provider availability detection."""
        config = {
            "openai": {"api_key": "sk-test"},
            "anthropic": {"api_key": "sk-ant-test"},
            "gemini": {"api_key": "gemini-test"},
            "ollama": {"model": "gpt-oss:20b"},
            "bedrock": {"model": "anthropic.claude-3-haiku-20240307-v1:0"},
        }

        with patch("openai.OpenAI"), patch("anthropic.Anthropic"), patch(
            "google.generativeai.configure"
        ), patch("google.generativeai.GenerativeModel"), patch(
            "requests.get"
        ) as mock_ollama_get, patch(
            "boto3.client"
        ):

            # Mock Ollama as unavailable (server not running)
            mock_ollama_get.side_effect = Exception("Connection refused")

            # Mock other providers as available
            with patch(
                "src.aws_cost_cli.query_processor.OpenAIProvider.is_available",
                return_value=True,
            ), patch(
                "src.aws_cost_cli.query_processor.AnthropicProvider.is_available",
                return_value=True,
            ), patch(
                "src.aws_cost_cli.query_processor.GeminiProvider.is_available",
                return_value=True,
            ), patch(
                "src.aws_cost_cli.query_processor.OllamaProvider.is_available",
                return_value=False,
            ), patch(
                "src.aws_cost_cli.query_processor.BedrockProvider.is_available",
                return_value=True,
            ):

                from src.aws_cost_cli.provider_factory import ProviderFactory

                available = ProviderFactory.get_available_providers(config)

                assert "openai" in available
                assert "anthropic" in available
                assert "gemini" in available
                assert "bedrock" in available
                assert "ollama" not in available

    @patch("src.aws_cost_cli.query_pipeline.AWSCostClient")
    @patch("src.aws_cost_cli.query_pipeline.ResponseGenerator")
    def test_end_to_end_provider_switching_same_query(
        self, mock_response_gen, mock_aws_client
    ):
        """Test end-to-end provider switching with same query produces consistent results."""
        # Mock AWS client response
        mock_aws_client.return_value.get_cost_and_usage.return_value = {
            "ResultsByTime": [
                {"Total": {"BlendedCost": {"Amount": "150.75", "Unit": "USD"}}}
            ]
        }
        mock_response_gen.return_value.generate_response.return_value = (
            "Cost analysis response"
        )

        # Create config with multiple providers
        config_content = """
llm_provider: openai
llm_config:
  openai:
    api_key: sk-test-openai-key
    model: gpt-3.5-turbo
  gemini:
    api_key: test-gemini-key
    model: gemini-1.5-flash
"""
        with open(self.config_file, "w") as f:
            f.write(config_content)

        # Standard query parameters that both providers should produce
        standard_response = {
            "service": "Amazon Simple Storage Service",
            "start_date": "2024-12-01",
            "end_date": "2024-12-31",
            "granularity": "MONTHLY",
            "metrics": ["BlendedCost"],
            "group_by": None,
        }

        with patch("openai.OpenAI") as mock_openai, patch(
            "google.generativeai.configure"
        ), patch("google.generativeai.GenerativeModel") as mock_gemini_model:

            # Mock OpenAI response
            mock_openai_response = Mock()
            mock_openai_response.choices = [Mock()]
            mock_openai_response.choices[0].message.content = str(
                standard_response
            ).replace("'", '"')
            mock_openai_client = Mock()
            mock_openai_client.chat.completions.create.return_value = (
                mock_openai_response
            )
            mock_openai.return_value = mock_openai_client

            # Mock Gemini response
            mock_gemini_response = Mock()
            mock_gemini_response.text = str(standard_response).replace("'", '"')
            mock_gemini_client = Mock()
            mock_gemini_client.generate_content.return_value = mock_gemini_response
            mock_gemini_model.return_value = mock_gemini_client

            query = "S3 costs in December 2024"

            # Test with OpenAI (default)
            result_openai = self.runner.invoke(
                cli,
                [
                    "query",
                    query,
                    "--config-file",
                    self.config_file,
                ],
            )

            # Test with Gemini override
            result_gemini = self.runner.invoke(
                cli,
                [
                    "query",
                    query,
                    "--llm-provider",
                    "gemini",
                    "--config-file",
                    self.config_file,
                ],
            )

            # Both should succeed
            assert result_openai.exit_code == 0
            assert result_gemini.exit_code == 0

            # Both should call AWS client with same parameters
            assert mock_aws_client.return_value.get_cost_and_usage.call_count == 2

    def test_provider_configuration_persistence(self):
        """Test that provider override doesn't modify saved configuration."""
        # Create initial config
        config_content = """
llm_provider: openai
llm_config:
  openai:
    api_key: sk-test-key
  gemini:
    api_key: gemini-test-key
"""
        with open(self.config_file, "w") as f:
            f.write(config_content)

        # Load config before override
        config_manager = ConfigManager()
        config_before = config_manager.load_config(self.config_file)
        assert config_before.llm_provider == "openai"

        # Simulate provider override (this would happen in CLI)
        context = QueryContext(
            original_query="test query", llm_provider_override="gemini"
        )

        # Verify override doesn't change the context's base config
        assert context.llm_provider_override == "gemini"

        # Load config after - should be unchanged
        config_after = config_manager.load_config(self.config_file)
        assert config_after.llm_provider == "openai"  # Should still be OpenAI

    def test_provider_fallback_configuration_loading(self):
        """Test that fallback provider configuration is properly loaded."""
        config_content = """
llm_provider: gemini
llm_config:
  gemini:
    api_key: test-gemini-key
  ollama:
    model: gpt-oss:20b
    base_url: http://localhost:11434
fallback_providers:
  - ollama
  - openai
"""
        with open(self.config_file, "w") as f:
            f.write(config_content)

        config_manager = ConfigManager()
        config = config_manager.load_config(self.config_file)

        assert config.llm_provider == "gemini"
        assert config.fallback_providers == ["ollama", "openai"]
        assert "gemini" in config.llm_config
        assert "ollama" in config.llm_config

    @patch("src.aws_cost_cli.query_pipeline.AWSCostClient")
    @patch("src.aws_cost_cli.query_pipeline.ResponseGenerator")
    def test_provider_error_handling_with_helpful_messages(
        self, mock_response_gen, mock_aws_client
    ):
        """Test that provider errors include helpful configuration messages."""
        # Create config without Gemini API key
        config_content = """
llm_provider: openai
llm_config:
  openai:
    api_key: sk-test-key
"""
        with open(self.config_file, "w") as f:
            f.write(config_content)

        # Try to use unconfigured Gemini provider
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            result = self.runner.invoke(
                cli,
                [
                    "query",
                    "EC2 costs last month",
                    "--llm-provider",
                    "gemini",
                    "--config-file",
                    self.config_file,
                ],
            )

        assert result.exit_code != 0
        # Should include helpful configuration instructions
        assert "GEMINI_API_KEY environment variable" in result.output
        assert "aws-cost-cli configure --provider gemini" in result.output

    def test_multiple_provider_fallback_chain(self):
        """Test complex fallback chain with multiple providers."""
        config = Config(
            llm_provider="gemini",
            llm_config={
                "gemini": {"api_key": "test-gemini-key"},
                "openai": {"api_key": "sk-test-openai-key"},
                "ollama": {"model": "gpt-oss:20b"},
            },
            fallback_providers=["openai", "ollama"],
        )

        with patch("google.generativeai.configure"), patch(
            "google.generativeai.GenerativeModel"
        ) as mock_gemini_model, patch("openai.OpenAI") as mock_openai, patch(
            "requests.get"
        ) as mock_ollama_get, patch(
            "requests.post"
        ) as mock_ollama_post:

            # Mock Gemini failure
            mock_gemini_client = Mock()
            mock_gemini_client.generate_content.side_effect = LLMProviderError(
                "Quota exceeded"
            )
            mock_gemini_model.return_value = mock_gemini_client

            # Mock OpenAI failure
            mock_openai_client = Mock()
            mock_openai_client.chat.completions.create.side_effect = LLMProviderError(
                "API error"
            )
            mock_openai.return_value = mock_openai_client

            # Mock Ollama success
            mock_ollama_get.return_value = Mock(status_code=200)
            mock_ollama_response = Mock()
            mock_ollama_response.status_code = 200
            mock_ollama_response.json.return_value = {
                "response": '{"service": "RDS", "granularity": "DAILY"}'
            }
            mock_ollama_post.return_value = mock_ollama_response

            # Create query parser
            from dataclasses import asdict

            query_parser = QueryParser(config.llm_config, asdict(config))

            # Should fallback through the chain: Gemini -> OpenAI -> Ollama
            result = query_parser.parse_query_with_fallback(
                "RDS costs weekly", "gemini"
            )

            assert result.service == "RDS"
            assert result.granularity.value == "DAILY"

            # Verify all providers were tried in order
            mock_gemini_client.generate_content.assert_called_once()
            mock_openai_client.chat.completions.create.assert_called_once()
            mock_ollama_post.assert_called_once()
