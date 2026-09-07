"""Unit tests for provider factory."""

from unittest.mock import Mock, patch
import pytest

from src.aws_cost_cli.provider_factory import ProviderFactory
from src.aws_cost_cli.query_processor import (
    OpenAIProvider,
    AnthropicProvider,
    BedrockProvider,
    OllamaProvider,
    GeminiProvider,
)
from src.aws_cost_cli.exceptions import ConfigurationError, LLMProviderError


class TestProviderFactory:
    """Test cases for ProviderFactory class."""

    def test_get_all_provider_names(self):
        """Test getting all supported provider names."""
        providers = ProviderFactory.get_all_provider_names()
        expected = ["openai", "anthropic", "bedrock", "ollama", "gemini"]
        assert providers == expected

    def test_create_openai_provider_success(self):
        """Test successful OpenAI provider creation."""
        config = {"openai": {"api_key": "sk-test-key", "model": "gpt-3.5-turbo"}}

        with patch("openai.OpenAI"):
            provider = ProviderFactory.create_provider("openai", config)
            assert isinstance(provider, OpenAIProvider)
            assert provider.api_key == "sk-test-key"
            assert provider.model == "gpt-3.5-turbo"

    def test_create_openai_provider_missing_api_key(self):
        """Test OpenAI provider creation without API key."""
        config = {"openai": {}}

        with pytest.raises(ConfigurationError, match="OpenAI API key is required"):
            ProviderFactory.create_provider("openai", config)

    def test_create_anthropic_provider_success(self):
        """Test successful Anthropic provider creation."""
        config = {
            "anthropic": {
                "api_key": "sk-ant-test-key",
                "model": "claude-3-haiku-20240307",
            }
        }

        with patch("anthropic.Anthropic"):
            provider = ProviderFactory.create_provider("anthropic", config)
            assert isinstance(provider, AnthropicProvider)
            assert provider.api_key == "sk-ant-test-key"
            assert provider.model == "claude-3-haiku-20240307"

    def test_create_anthropic_provider_missing_api_key(self):
        """Test Anthropic provider creation without API key."""
        config = {"anthropic": {}}

        with pytest.raises(ConfigurationError, match="Anthropic API key is required"):
            ProviderFactory.create_provider("anthropic", config)

    def test_create_bedrock_provider_success(self):
        """Test successful Bedrock provider creation."""
        config = {
            "bedrock": {
                "model": "anthropic.claude-3-haiku-20240307-v1:0",
                "region": "us-east-1",
            }
        }

        with patch("boto3.client"):
            provider = ProviderFactory.create_provider("bedrock", config)
            assert isinstance(provider, BedrockProvider)
            assert provider.model == "anthropic.claude-3-haiku-20240307-v1:0"
            assert provider.region == "us-east-1"

    def test_create_ollama_provider_success(self):
        """Test successful Ollama provider creation."""
        config = {
            "ollama": {
                "model": "gpt-oss:20b",
                "base_url": "http://localhost:11434",
                "timeout": 60,
            }
        }

        provider = ProviderFactory.create_provider("ollama", config)
        assert isinstance(provider, OllamaProvider)
        assert provider.model == "gpt-oss:20b"
        assert provider.base_url == "http://localhost:11434"
        assert provider.timeout == 60

    def test_create_gemini_provider_success(self):
        """Test successful Gemini provider creation."""
        config = {"gemini": {"api_key": "test-gemini-key", "model": "gemini-1.5-flash"}}

        with patch("google.generativeai.configure"), patch(
            "google.generativeai.GenerativeModel"
        ):
            provider = ProviderFactory.create_provider("gemini", config)
            assert isinstance(provider, GeminiProvider)
            assert provider.api_key == "test-gemini-key"
            assert provider.model == "gemini-1.5-flash"

    def test_create_gemini_provider_missing_api_key(self):
        """Test Gemini provider creation without API key."""
        config = {"gemini": {}}

        with pytest.raises(ConfigurationError, match="Gemini API key is required"):
            ProviderFactory.create_provider("gemini", config)

    def test_create_gemini_provider_with_defaults(self):
        """Test Gemini provider creation with default model."""
        config = {"gemini": {"api_key": "test-gemini-key"}}

        with patch("google.generativeai.configure"), patch(
            "google.generativeai.GenerativeModel"
        ):
            provider = ProviderFactory.create_provider("gemini", config)
            assert isinstance(provider, GeminiProvider)
            assert provider.api_key == "test-gemini-key"
            assert provider.model == "gemini-1.5-flash"  # Default model

    def test_create_gemini_provider_fallback_format(self):
        """Test Gemini provider creation with old flat config format."""
        config = {"api_key": "test-gemini-key", "model": "gemini-1.5-pro"}

        with patch("google.generativeai.configure"), patch(
            "google.generativeai.GenerativeModel"
        ):
            provider = ProviderFactory.create_provider("gemini", config)
            assert isinstance(provider, GeminiProvider)
            assert provider.api_key == "test-gemini-key"
            assert provider.model == "gemini-1.5-pro"

    def test_create_unknown_provider(self):
        """Test creation of unknown provider."""
        config = {}

        with pytest.raises(LLMProviderError, match="Unknown provider: unknown"):
            ProviderFactory.create_provider("unknown", config)

    def test_get_available_providers_all_configured(self):
        """Test getting available providers when all are configured."""
        config = {
            "openai": {"api_key": "sk-test"},
            "anthropic": {"api_key": "sk-ant-test"},
            "bedrock": {"model": "anthropic.claude-3-haiku-20240307-v1:0"},
            "ollama": {"model": "gpt-oss:20b"},
            "gemini": {"api_key": "gemini-test"},
        }

        with patch("openai.OpenAI"), patch("anthropic.Anthropic"), patch(
            "boto3.client"
        ), patch("requests.get") as mock_get, patch(
            "google.generativeai.configure"
        ), patch(
            "google.generativeai.GenerativeModel"
        ):

            # Mock Ollama availability check
            mock_get.return_value = Mock(status_code=200)

            # Mock all providers as available
            with patch.object(
                OpenAIProvider, "is_available", return_value=True
            ), patch.object(
                AnthropicProvider, "is_available", return_value=True
            ), patch.object(
                BedrockProvider, "is_available", return_value=True
            ), patch.object(
                OllamaProvider, "is_available", return_value=True
            ), patch.object(
                GeminiProvider, "is_available", return_value=True
            ):

                available = ProviderFactory.get_available_providers(config)
                expected = ["openai", "anthropic", "bedrock", "ollama", "gemini"]
                assert set(available) == set(expected)

    def test_get_available_providers_none_configured(self):
        """Test getting available providers when none are configured."""
        config = {}

        available = ProviderFactory.get_available_providers(config)
        assert available == []

    def test_get_available_providers_partial_configuration(self):
        """Test getting available providers with partial configuration."""
        config = {
            "openai": {"api_key": "sk-test"},
            "gemini": {"api_key": "gemini-test"},
            "ollama": {},  # Missing required config
        }

        with patch("openai.OpenAI"), patch("google.generativeai.configure"), patch(
            "google.generativeai.GenerativeModel"
        ):

            # Mock only OpenAI and Gemini as available
            with patch.object(
                OpenAIProvider, "is_available", return_value=True
            ), patch.object(GeminiProvider, "is_available", return_value=True):

                available = ProviderFactory.get_available_providers(config)
                assert "openai" in available
                assert "gemini" in available
                assert len(available) == 2

    def test_get_provider_configuration_status_all_providers(self):
        """Test getting configuration status for all providers."""
        config = {
            "openai": {"api_key": "sk-test"},
            "anthropic": {"api_key": "sk-ant-test"},
            "bedrock": {"model": "anthropic.claude-3-haiku-20240307-v1:0"},
            "ollama": {"model": "gpt-oss:20b"},
            "gemini": {"api_key": "gemini-test"},
        }

        with patch("openai.OpenAI"), patch("anthropic.Anthropic"), patch(
            "boto3.client"
        ), patch("requests.get") as mock_get, patch(
            "google.generativeai.configure"
        ), patch(
            "google.generativeai.GenerativeModel"
        ):

            # Mock Ollama availability check
            mock_get.return_value = Mock(status_code=200)

            # Mock all providers as available
            with patch.object(
                OpenAIProvider, "is_available", return_value=True
            ), patch.object(
                AnthropicProvider, "is_available", return_value=True
            ), patch.object(
                BedrockProvider, "is_available", return_value=True
            ), patch.object(
                OllamaProvider, "is_available", return_value=True
            ), patch.object(
                GeminiProvider, "is_available", return_value=True
            ):

                status = ProviderFactory.get_provider_configuration_status(config)

                # All providers should be configured and available
                for provider in ["openai", "anthropic", "bedrock", "ollama", "gemini"]:
                    assert status[provider]["configured"] is True
                    assert status[provider]["available"] is True
                    assert status[provider]["error"] is None

    def test_get_provider_configuration_status_with_errors(self):
        """Test getting configuration status with some providers having errors."""
        config = {"openai": {}, "gemini": {"api_key": "gemini-test"}}  # Missing API key

        with patch("google.generativeai.configure"), patch(
            "google.generativeai.GenerativeModel"
        ):

            with patch.object(GeminiProvider, "is_available", return_value=True):
                status = ProviderFactory.get_provider_configuration_status(config)

                # OpenAI should have configuration error
                assert status["openai"]["configured"] is False
                assert status["openai"]["available"] is False
                assert "API key is required" in status["openai"]["error"]

                # Gemini should be properly configured
                assert status["gemini"]["configured"] is True
                assert status["gemini"]["available"] is True
                assert status["gemini"]["error"] is None
