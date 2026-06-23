"""Configuration management for AWS Cost CLI."""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import asdict

from .models import Config, DateFormattingConfig, DateFormatStyle
from .exceptions import ConfigurationError


class ConfigManager:
    """Manages configuration loading and validation."""

    DEFAULT_CONFIG_PATHS = [
        "~/.aws-cost-cli/config.yaml",
        "~/.aws-cost-cli/config.yml",
        "~/.aws-cost-cli/config.json",
        ".aws-cost-cli.yaml",
        ".aws-cost-cli.yml",
        ".aws-cost-cli.json",
    ]

    def __init__(self):
        self._config: Optional[Config] = None
        self._config_path: Optional[Path] = None

    def load_config(self, config_path: Optional[str] = None) -> Config:
        """Load configuration from file, environment, and defaults."""
        config_data = self._load_default_config()

        # Load from file
        if config_path:
            file_config = self._load_config_file(config_path)
            config_data = self._merge_config(config_data, file_config)
        else:
            # Try to find config file automatically
            for path in self.DEFAULT_CONFIG_PATHS:
                expanded_path = Path(path).expanduser()
                if expanded_path.exists():
                    file_config = self._load_config_file(str(expanded_path))
                    config_data = self._merge_config(config_data, file_config)
                    self._config_path = expanded_path
                    break

        # Override with environment variables
        env_config = self._load_env_config()
        config_data = self._merge_config(config_data, env_config)

        # Create and validate config
        self._config = self._create_config_from_dict(config_data)
        return self._config

    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration values."""
        return {
            "llm_provider": "ollama",
            "llm_config": {},
            "default_profile": None,
            "cache_ttl": 3600,
            "output_format": "simple",
            "default_currency": "USD",
            "fallback_providers": ["ollama", "openai", "anthropic", "gemini"],
            "date_formatting": {
                "enabled": True,
                "format_style": "smart",
                "fiscal_year_start_month": 1,
                "locale": "en_US",
                "fallback_to_iso": True
            }
        }

    def _load_config_file(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file."""
        path = Path(config_path).expanduser()

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                if path.suffix.lower() in [".yaml", ".yml"]:
                    return yaml.safe_load(f) or {}
                elif path.suffix.lower() == ".json":
                    return json.load(f) or {}
                else:
                    # Try YAML first, then JSON
                    content = f.read()
                    try:
                        return yaml.safe_load(content) or {}
                    except yaml.YAMLError:
                        return json.loads(content) or {}
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise ConfigurationError(
                f"Invalid configuration file format: {e}", config_file=config_path
            )
        except Exception as e:
            raise ConfigurationError(
                f"Error reading configuration file: {e}", config_file=config_path
            )

    def _load_env_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        env_config = {}

        # Map environment variables to config keys
        env_mappings = {
            "AWS_COST_CLI_LLM_PROVIDER": "llm_provider",
            "AWS_COST_CLI_DEFAULT_PROFILE": "default_profile",
            "AWS_COST_CLI_CACHE_TTL": "cache_ttl",
            "AWS_COST_CLI_OUTPUT_FORMAT": "output_format",
            "AWS_COST_CLI_DEFAULT_CURRENCY": "default_currency",
            "AWS_COST_CLI_DATE_FORMAT_ENABLED": ("date_formatting", "enabled"),
            "AWS_COST_CLI_DATE_FORMAT_STYLE": ("date_formatting", "format_style"),
            "AWS_COST_CLI_FISCAL_YEAR_START": ("date_formatting", "fiscal_year_start_month"),
            "AWS_COST_CLI_DATE_LOCALE": ("date_formatting", "locale"),
            "AWS_COST_CLI_DATE_FALLBACK_ISO": ("date_formatting", "fallback_to_iso"),
            "OPENAI_API_KEY": ("llm_config", "openai", "api_key"),
            "OPENAI_TIMEOUT": ("llm_config", "openai", "timeout"),
            "ANTHROPIC_API_KEY": ("llm_config", "anthropic", "api_key"),
            "ANTHROPIC_TIMEOUT": ("llm_config", "anthropic", "timeout"),
            "AWS_BEDROCK_REGION": ("llm_config", "bedrock", "region"),
            "AWS_BEDROCK_MODEL": ("llm_config", "bedrock", "model"),
            "AWS_BEDROCK_PROFILE": ("llm_config", "bedrock", "profile"),
            "AWS_BEDROCK_TIMEOUT": ("llm_config", "bedrock", "timeout"),
            "OLLAMA_MODEL": ("llm_config", "ollama", "model"),
            "OLLAMA_BASE_URL": ("llm_config", "ollama", "base_url"),
            "OLLAMA_TIMEOUT": ("llm_config", "ollama", "timeout"),
            "GEMINI_API_KEY": ("llm_config", "gemini", "api_key"),
            "GEMINI_MODEL": ("llm_config", "gemini", "model"),
            "GEMINI_TIMEOUT": ("llm_config", "gemini", "timeout"),
        }

        for env_var, config_key in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                if isinstance(config_key, tuple):
                    # Handle nested configurations with type conversion
                    if config_key[-1] == "timeout":
                        try:
                            # Convert timeout to float
                            timeout_value = float(value)
                            self._set_nested_config(env_config, config_key, timeout_value)
                        except ValueError:
                            pass  # Keep default value
                    elif config_key[-1] == "enabled" or config_key[-1] == "fallback_to_iso":
                        # Convert boolean values
                        bool_value = value.lower() in ("true", "1", "yes", "on")
                        self._set_nested_config(env_config, config_key, bool_value)
                    elif config_key[-1] == "fiscal_year_start_month":
                        try:
                            # Convert to integer
                            int_value = int(value)
                            self._set_nested_config(env_config, config_key, int_value)
                        except ValueError:
                            pass  # Keep default value
                    else:
                        # Nested configuration (string values)
                        self._set_nested_config(env_config, config_key, value)
                else:
                    # Convert string values to appropriate types
                    if config_key == "cache_ttl":
                        try:
                            env_config[config_key] = int(value)
                        except ValueError:
                            pass  # Keep default value
                    else:
                        env_config[config_key] = value

        return env_config

    def _set_nested_config(self, config: Dict[str, Any], keys: tuple, value: Any):
        """Set nested configuration value."""
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value

    def _merge_config(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configuration dictionaries, handling nested structures."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                result[key] = self._merge_config(result[key], value)
            else:
                # Override the value
                result[key] = value
        
        return result

    def _create_config_from_dict(self, config_data: Dict[str, Any]) -> Config:
        """Create Config object from dictionary, handling nested structures."""
        # Extract date formatting config if present
        date_formatting_data = config_data.pop("date_formatting", {})
        date_formatting_config = DateFormattingConfig(**date_formatting_data)
        
        # Create main config
        config = Config(**config_data)
        config.date_formatting = date_formatting_config
        
        return config

    def save_config(self, config: Config, config_path: Optional[str] = None) -> None:
        """Save configuration to file."""
        if config_path:
            path = Path(config_path).expanduser()
        elif self._config_path:
            path = self._config_path
        else:
            # Default to user config directory
            path = Path("~/.aws-cost-cli/config.yaml").expanduser()

        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert config to dict and save
        config_dict = self._config_to_serializable_dict(config)

        try:
            with open(path, "w", encoding="utf-8") as f:
                if path.suffix.lower() in [".yaml", ".yml"]:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                else:
                    json.dump(config_dict, f, indent=2)
        except Exception as e:
            raise ConfigurationError(
                f"Error saving configuration file: {e}", config_file=str(path)
            )

    def get_config(self) -> Optional[Config]:
        """Get the currently loaded configuration."""
        return self._config

    def get_default_config_path(self) -> Path:
        """Get the default configuration file path."""
        return Path("~/.aws-cost-cli/config.yaml").expanduser()

    def validate_config(self, config: Config) -> bool:
        """Validate configuration values."""
        # Validate LLM provider
        valid_providers = ["openai", "anthropic", "bedrock", "ollama", "gemini"]
        if config.llm_provider not in valid_providers:
            raise ConfigurationError(f"Invalid LLM provider: {config.llm_provider}")

        # Validate output format
        valid_formats = ["simple", "detailed", "json"]
        if config.output_format not in valid_formats:
            raise ConfigurationError(f"Invalid output format: {config.output_format}")

        # Validate cache TTL
        if config.cache_ttl < 0:
            raise ConfigurationError("Cache TTL must be non-negative")

        # Validate date formatting configuration
        if config.date_formatting:
            self._validate_date_formatting_config(config.date_formatting)

        # Validate Gemini-specific configuration
        if config.llm_provider == "gemini":
            gemini_config = config.llm_config.get("gemini", {})
            if not gemini_config.get("api_key"):
                raise ConfigurationError(
                    "Gemini API key is required. Set GEMINI_API_KEY environment variable or configure in config file."
                )
            
            # Validate Gemini model if specified
            model = gemini_config.get("model", "gemini-1.5-flash")
            valid_gemini_models = [
                "gemini-1.5-flash",
                "gemini-1.5-pro",
                "gemini-1.0-pro",
                "gemini-pro",
                "gemini-pro-vision"
            ]
            if model not in valid_gemini_models:
                raise ConfigurationError(
                    f"Invalid Gemini model: {model}. Valid models: {', '.join(valid_gemini_models)}"
                )

        return True

    def _validate_date_formatting_config(self, date_config: DateFormattingConfig) -> None:
        """Validate date formatting configuration."""
        # Validate fiscal year start month
        if not 1 <= date_config.fiscal_year_start_month <= 12:
            raise ConfigurationError(
                f"Invalid fiscal_year_start_month: {date_config.fiscal_year_start_month}. Must be between 1 and 12."
            )

        # Validate format style
        valid_styles = [style.value for style in DateFormatStyle]
        if isinstance(date_config.format_style, str):
            if date_config.format_style.lower() not in valid_styles:
                raise ConfigurationError(
                    f"Invalid format_style: {date_config.format_style}. Valid styles: {', '.join(valid_styles)}"
                )
        elif not isinstance(date_config.format_style, DateFormatStyle):
            raise ConfigurationError(
                f"Invalid format_style type: {type(date_config.format_style)}. Must be DateFormatStyle enum or string."
            )

        # Validate locale format (basic check)
        if not isinstance(date_config.locale, str) or len(date_config.locale) < 2:
            raise ConfigurationError(f"Invalid locale: {date_config.locale}. Must be a valid locale string.")

        # Validate boolean fields
        if not isinstance(date_config.enabled, bool):
            raise ConfigurationError(f"Invalid enabled value: {date_config.enabled}. Must be boolean.")
        
        if not isinstance(date_config.fallback_to_iso, bool):
            raise ConfigurationError(f"Invalid fallback_to_iso value: {date_config.fallback_to_iso}. Must be boolean.")

    def _config_to_serializable_dict(self, config: Config) -> Dict[str, Any]:
        """Convert Config object to a serializable dictionary."""
        config_dict = asdict(config)
        
        # Convert enum to string for serialization
        if config_dict.get("date_formatting") and "format_style" in config_dict["date_formatting"]:
            format_style = config_dict["date_formatting"]["format_style"]
            if hasattr(format_style, 'value'):
                config_dict["date_formatting"]["format_style"] = format_style.value
        
        return config_dict
