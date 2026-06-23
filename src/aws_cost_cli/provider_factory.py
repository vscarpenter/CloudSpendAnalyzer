"""Provider factory for creating and managing LLM provider instances."""

from typing import Dict, Any, List, Optional, Type
from .query_processor import (
    LLMProvider,
    OpenAIProvider,
    AnthropicProvider,
    BedrockProvider,
    OllamaProvider,
    GeminiProvider,
)
from .exceptions import LLMProviderError, ConfigurationError


class ProviderFactory:
    """Factory for creating LLM provider instances."""

    # Registry of available providers
    _PROVIDERS: Dict[str, Type[LLMProvider]] = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "bedrock": BedrockProvider,
        "ollama": OllamaProvider,
        "gemini": GeminiProvider,
    }

    @staticmethod
    def create_provider(provider_name: str, config: Dict[str, Any]) -> LLMProvider:
        """
        Create provider instance based on configuration.

        Args:
            provider_name: Name of the provider to create
            config: Configuration dictionary for the provider

        Returns:
            LLMProvider instance

        Raises:
            LLMProviderError: If provider is unknown or configuration is invalid
            ConfigurationError: If required configuration is missing
        """
        provider_name = provider_name.lower()

        if provider_name not in ProviderFactory._PROVIDERS:
            available_providers = ", ".join(ProviderFactory._PROVIDERS.keys())
            raise LLMProviderError(
                f"Unknown provider: {provider_name}. Available providers: {available_providers}",
                provider=provider_name,
            )

        provider_class = ProviderFactory._PROVIDERS[provider_name]

        try:
            if provider_name == "openai":
                return ProviderFactory._create_openai_provider(config)
            elif provider_name == "anthropic":
                return ProviderFactory._create_anthropic_provider(config)
            elif provider_name == "bedrock":
                return ProviderFactory._create_bedrock_provider(config)
            elif provider_name == "ollama":
                return ProviderFactory._create_ollama_provider(config)
            elif provider_name == "gemini":
                return ProviderFactory._create_gemini_provider(config)
            else:
                # This should never happen due to the check above, but just in case
                raise LLMProviderError(
                    f"Provider creation not implemented: {provider_name}"
                )

        except Exception as e:
            if isinstance(e, (LLMProviderError, ConfigurationError)):
                raise
            else:
                raise LLMProviderError(
                    f"Failed to create {provider_name} provider: {str(e)}",
                    provider=provider_name,
                )

    @staticmethod
    def _create_openai_provider(config: Dict[str, Any]) -> OpenAIProvider:
        """Create OpenAI provider instance."""
        # Check for provider-specific config first
        if "openai" in config:
            openai_config = config["openai"]
            api_key = openai_config.get("api_key")
            model = openai_config.get("model", "gpt-3.5-turbo")
            timeout = openai_config.get("timeout")
        else:
            # Fallback to old format
            api_key = config.get("api_key")
            model = config.get("model", "gpt-3.5-turbo")
            timeout = config.get("timeout")

        if not api_key:
            raise ConfigurationError(
                "OpenAI provider is not configured. OpenAI API key is required. "
                "Set OPENAI_API_KEY environment variable or run:\n"
                "aws-cost-cli configure --provider openai --api-key YOUR_API_KEY"
            )

        return OpenAIProvider(api_key=api_key, model=model, timeout=timeout)

    @staticmethod
    def _create_anthropic_provider(config: Dict[str, Any]) -> AnthropicProvider:
        """Create Anthropic provider instance."""
        # Check for provider-specific config first
        if "anthropic" in config:
            anthropic_config = config["anthropic"]
            api_key = anthropic_config.get("api_key")
            model = anthropic_config.get("model", "claude-3-haiku-20240307")
            timeout = anthropic_config.get("timeout")
        else:
            # Fallback to old format
            api_key = config.get("api_key")
            model = config.get("model", "claude-3-haiku-20240307")
            timeout = config.get("timeout")

        if not api_key:
            raise ConfigurationError(
                "Anthropic provider is not configured. Anthropic API key is required. "
                "Set ANTHROPIC_API_KEY environment variable or run:\n"
                "aws-cost-cli configure --provider anthropic --api-key YOUR_API_KEY"
            )

        return AnthropicProvider(api_key=api_key, model=model, timeout=timeout)

    @staticmethod
    def _create_bedrock_provider(config: Dict[str, Any]) -> BedrockProvider:
        """Create Bedrock provider instance."""
        # Check for provider-specific config first
        if "bedrock" in config:
            bedrock_config = config["bedrock"]
            model = bedrock_config.get(
                "model", "anthropic.claude-3-haiku-20240307-v1:0"
            )
            region = bedrock_config.get("region", "us-east-1")
            profile = bedrock_config.get("profile")
            timeout = bedrock_config.get("timeout")
        elif config.get("provider", "").lower() == "bedrock":
            # Fallback to old format
            model = config.get("model", "anthropic.claude-3-haiku-20240307-v1:0")
            region = config.get("region", "us-east-1")
            profile = config.get("profile")
            timeout = config.get("timeout")
        else:
            raise ConfigurationError(
                "Bedrock provider is not configured. Add a bedrock section to "
                "the config file or set llm_config.provider to bedrock."
            )

        return BedrockProvider(
            model=model, region=region, profile=profile, timeout=timeout
        )

    @staticmethod
    def _create_ollama_provider(config: Dict[str, Any]) -> OllamaProvider:
        """Create Ollama provider instance."""
        # Check for provider-specific config first
        if "ollama" in config:
            ollama_config = config["ollama"]
            model = ollama_config.get("model", "gpt-oss:20b")
            base_url = ollama_config.get("base_url", "http://localhost:11434")
            timeout = ollama_config.get("timeout", 60)
            options = ollama_config.get("options", {})
        else:
            # Fallback to old format
            model = config.get("model", "gpt-oss:20b")
            base_url = config.get("base_url", "http://localhost:11434")
            timeout = config.get("timeout", 60)
            options = config.get("options", {})

        return OllamaProvider(
            model=model, base_url=base_url, timeout=timeout, options=options
        )

    @staticmethod
    def _create_gemini_provider(config: Dict[str, Any]) -> GeminiProvider:
        """Create Gemini provider instance."""
        # Check for provider-specific config first
        if "gemini" in config:
            gemini_config = config["gemini"]
            api_key = gemini_config.get("api_key")
            model = gemini_config.get("model", "gemini-1.5-flash")
            timeout = gemini_config.get("timeout")
        else:
            # Fallback to old format
            api_key = config.get("api_key")
            model = config.get("model", "gemini-1.5-flash")
            timeout = config.get("timeout")

        if not api_key:
            raise ConfigurationError(
                "Gemini provider is not configured. Gemini API key is required. "
                "Set GEMINI_API_KEY environment variable or run:\n"
                "aws-cost-cli configure --provider gemini --api-key YOUR_API_KEY"
            )

        return GeminiProvider(api_key=api_key, model=model, timeout=timeout)

    @staticmethod
    def get_available_providers(config: Dict[str, Any]) -> List[str]:
        """
        Get list of properly configured providers.

        Args:
            config: Configuration dictionary

        Returns:
            List of provider names that are properly configured and available
        """
        available = []

        for provider_name in ProviderFactory._PROVIDERS.keys():
            try:
                provider = ProviderFactory.create_provider(provider_name, config)
                if provider.is_available():
                    available.append(provider_name)
            except Exception:
                # Provider is not configured or not available
                continue

        return available

    @staticmethod
    def get_all_provider_names() -> List[str]:
        """
        Get list of all supported provider names.

        Returns:
            List of all supported provider names
        """
        return list(ProviderFactory._PROVIDERS.keys())

    @staticmethod
    def is_provider_supported(provider_name: str) -> bool:
        """
        Check if a provider is supported.

        Args:
            provider_name: Name of the provider to check

        Returns:
            True if provider is supported, False otherwise
        """
        return provider_name.lower() in ProviderFactory._PROVIDERS

    @staticmethod
    def get_provider_configuration_status(
        config: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get configuration status for all providers.

        Args:
            config: Configuration dictionary

        Returns:
            Dictionary mapping provider names to their configuration status
        """
        status = {}

        for provider_name in ProviderFactory._PROVIDERS.keys():
            try:
                provider = ProviderFactory.create_provider(provider_name, config)
                is_available = provider.is_available()
                status[provider_name] = {
                    "configured": True,
                    "available": is_available,
                    "error": None,
                }
            except ConfigurationError as e:
                status[provider_name] = {
                    "configured": False,
                    "available": False,
                    "error": str(e),
                }
            except Exception as e:
                status[provider_name] = {
                    "configured": False,
                    "available": False,
                    "error": f"Configuration error: {str(e)}",
                }

        return status
