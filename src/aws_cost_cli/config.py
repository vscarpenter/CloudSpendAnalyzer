"""Configuration management for AWS Cost CLI."""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import asdict

from .models import Config


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
            config_data.update(file_config)
        else:
            # Try to find config file automatically
            for path in self.DEFAULT_CONFIG_PATHS:
                expanded_path = Path(path).expanduser()
                if expanded_path.exists():
                    file_config = self._load_config_file(str(expanded_path))
                    config_data.update(file_config)
                    self._config_path = expanded_path
                    break
        
        # Override with environment variables
        env_config = self._load_env_config()
        config_data.update(env_config)
        
        # Create and validate config
        self._config = Config(**config_data)
        return self._config
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration values."""
        return {
            "llm_provider": "openai",
            "llm_config": {},
            "default_profile": None,
            "cache_ttl": 3600,
            "output_format": "simple",
            "default_currency": "USD",
        }
    
    def _load_config_file(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file."""
        path = Path(config_path).expanduser()
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                if path.suffix.lower() in ['.yaml', '.yml']:
                    return yaml.safe_load(f) or {}
                elif path.suffix.lower() == '.json':
                    return json.load(f) or {}
                else:
                    # Try YAML first, then JSON
                    content = f.read()
                    try:
                        return yaml.safe_load(content) or {}
                    except yaml.YAMLError:
                        return json.loads(content) or {}
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise ValueError(f"Invalid configuration file format: {e}")
        except Exception as e:
            raise RuntimeError(f"Error reading configuration file: {e}")
    
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
            "OPENAI_API_KEY": ("llm_config", "openai", "api_key"),
            "ANTHROPIC_API_KEY": ("llm_config", "anthropic", "api_key"),
        }
        
        for env_var, config_key in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                if isinstance(config_key, tuple):
                    # Nested configuration
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
        config_dict = asdict(config)
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                if path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                else:
                    json.dump(config_dict, f, indent=2)
        except Exception as e:
            raise RuntimeError(f"Error saving configuration file: {e}")
    
    def get_config(self) -> Optional[Config]:
        """Get the currently loaded configuration."""
        return self._config
    
    def validate_config(self, config: Config) -> bool:
        """Validate configuration values."""
        # Validate LLM provider
        valid_providers = ["openai", "anthropic", "bedrock", "ollama"]
        if config.llm_provider not in valid_providers:
            raise ValueError(f"Invalid LLM provider: {config.llm_provider}")
        
        # Validate output format
        valid_formats = ["simple", "detailed", "json"]
        if config.output_format not in valid_formats:
            raise ValueError(f"Invalid output format: {config.output_format}")
        
        # Validate cache TTL
        if config.cache_ttl < 0:
            raise ValueError("Cache TTL must be non-negative")
        
        return True