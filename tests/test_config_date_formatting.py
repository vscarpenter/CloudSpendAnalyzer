"""Tests for date formatting configuration management."""

import pytest
import tempfile
import yaml
import json
import os
from pathlib import Path
from unittest.mock import patch

from src.aws_cost_cli.config import ConfigManager
from src.aws_cost_cli.models import Config, DateFormattingConfig, DateFormatStyle
from src.aws_cost_cli.exceptions import ConfigurationError


class TestDateFormattingConfigInConfigManager:
    """Test date formatting configuration in ConfigManager."""

    def test_default_config_includes_date_formatting(self):
        """Test that default configuration includes date formatting settings."""
        manager = ConfigManager()
        default_config = manager._load_default_config()
        
        assert "date_formatting" in default_config
        date_config = default_config["date_formatting"]
        assert date_config["enabled"] is True
        assert date_config["format_style"] == "smart"
        assert date_config["fiscal_year_start_month"] == 1
        assert date_config["locale"] == "en_US"
        assert date_config["fallback_to_iso"] is True

    def test_load_config_with_date_formatting_from_file(self):
        """Test loading configuration with date formatting from file."""
        config_data = {
            "llm_provider": "openai",
            "date_formatting": {
                "enabled": False,
                "format_style": "verbose",
                "fiscal_year_start_month": 7,
                "locale": "en_GB",
                "fallback_to_iso": False
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            manager = ConfigManager()
            config = manager.load_config(config_path)
            
            assert config.llm_provider == "openai"
            assert config.date_formatting.enabled is False
            assert config.date_formatting.format_style == DateFormatStyle.VERBOSE
            assert config.date_formatting.fiscal_year_start_month == 7
            assert config.date_formatting.locale == "en_GB"
            assert config.date_formatting.fallback_to_iso is False
        finally:
            os.unlink(config_path)

    def test_load_config_with_partial_date_formatting(self):
        """Test loading configuration with partial date formatting settings."""
        config_data = {
            "llm_provider": "anthropic",
            "date_formatting": {
                "format_style": "compact",
                "fiscal_year_start_month": 4
                # Other settings should use defaults
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name
        
        try:
            manager = ConfigManager()
            config = manager.load_config(config_path)
            
            assert config.date_formatting.enabled is True  # Default
            assert config.date_formatting.format_style == DateFormatStyle.COMPACT
            assert config.date_formatting.fiscal_year_start_month == 4
            assert config.date_formatting.locale == "en_US"  # Default
            assert config.date_formatting.fallback_to_iso is True  # Default
        finally:
            os.unlink(config_path)

    def test_environment_variable_overrides(self):
        """Test that environment variables override date formatting settings."""
        env_vars = {
            "AWS_COST_CLI_DATE_FORMAT_ENABLED": "false",
            "AWS_COST_CLI_DATE_FORMAT_STYLE": "verbose",
            "AWS_COST_CLI_FISCAL_YEAR_START": "10",
            "AWS_COST_CLI_DATE_LOCALE": "fr_FR",
            "AWS_COST_CLI_DATE_FALLBACK_ISO": "false"
        }
        
        with patch.dict(os.environ, env_vars):
            manager = ConfigManager()
            config = manager.load_config()
            
            assert config.date_formatting.enabled is False
            assert config.date_formatting.format_style == DateFormatStyle.VERBOSE
            assert config.date_formatting.fiscal_year_start_month == 10
            assert config.date_formatting.locale == "fr_FR"
            assert config.date_formatting.fallback_to_iso is False

    def test_environment_variable_boolean_parsing(self):
        """Test parsing of boolean environment variables."""
        test_cases = [
            ("true", True),
            ("True", True),
            ("1", True),
            ("yes", True),
            ("on", True),
            ("false", False),
            ("False", False),
            ("0", False),
            ("no", False),
            ("off", False),
            ("invalid", False)  # Invalid values default to False
        ]
        
        for env_value, expected in test_cases:
            with patch.dict(os.environ, {"AWS_COST_CLI_DATE_FORMAT_ENABLED": env_value}):
                manager = ConfigManager()
                config = manager.load_config()
                assert config.date_formatting.enabled == expected, f"Failed for {env_value}"

    def test_environment_variable_integer_parsing(self):
        """Test parsing of integer environment variables."""
        with patch.dict(os.environ, {"AWS_COST_CLI_FISCAL_YEAR_START": "7"}):
            manager = ConfigManager()
            config = manager.load_config()
            assert config.date_formatting.fiscal_year_start_month == 7

        # Invalid integer should be ignored (keep default)
        with patch.dict(os.environ, {"AWS_COST_CLI_FISCAL_YEAR_START": "invalid"}):
            manager = ConfigManager()
            config = manager.load_config()
            assert config.date_formatting.fiscal_year_start_month == 1  # Default

    def test_save_config_with_date_formatting(self):
        """Test saving configuration with date formatting settings."""
        config = Config(
            llm_provider="ollama",
            date_formatting=DateFormattingConfig(
                enabled=False,
                format_style=DateFormatStyle.COMPACT,
                fiscal_year_start_month=7,
                locale="en_GB",
                fallback_to_iso=False
            )
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_path = f.name
        
        try:
            manager = ConfigManager()
            manager.save_config(config, config_path)
            
            # Load the saved config and verify
            with open(config_path, 'r') as f:
                saved_data = yaml.safe_load(f)
            
            assert saved_data["llm_provider"] == "ollama"
            date_config = saved_data["date_formatting"]
            assert date_config["enabled"] is False
            assert date_config["format_style"] == "compact"
            assert date_config["fiscal_year_start_month"] == 7
            assert date_config["locale"] == "en_GB"
            assert date_config["fallback_to_iso"] is False
        finally:
            os.unlink(config_path)

    def test_config_merge_with_date_formatting(self):
        """Test configuration merging with date formatting settings."""
        manager = ConfigManager()
        
        base_config = {
            "llm_provider": "openai",
            "date_formatting": {
                "enabled": True,
                "format_style": "smart",
                "fiscal_year_start_month": 1
            }
        }
        
        override_config = {
            "date_formatting": {
                "format_style": "verbose",
                "fiscal_year_start_month": 7,
                "locale": "fr_FR"
            }
        }
        
        merged = manager._merge_config(base_config, override_config)
        
        assert merged["llm_provider"] == "openai"
        date_config = merged["date_formatting"]
        assert date_config["enabled"] is True  # From base
        assert date_config["format_style"] == "verbose"  # Overridden
        assert date_config["fiscal_year_start_month"] == 7  # Overridden
        assert date_config["locale"] == "fr_FR"  # Added


class TestDateFormattingConfigValidation:
    """Test validation of date formatting configuration."""

    def test_valid_date_formatting_config(self):
        """Test validation of valid date formatting configuration."""
        config = Config(
            date_formatting=DateFormattingConfig(
                enabled=True,
                format_style=DateFormatStyle.VERBOSE,
                fiscal_year_start_month=7,
                locale="en_GB",
                fallback_to_iso=False
            )
        )
        
        manager = ConfigManager()
        assert manager.validate_config(config) is True

    def test_invalid_fiscal_year_start_month(self):
        """Test validation fails for invalid fiscal year start month."""
        # Create config with valid values first, then modify for testing
        config = Config()
        config.date_formatting.fiscal_year_start_month = 0
        
        manager = ConfigManager()
        with pytest.raises(ConfigurationError, match="Invalid fiscal_year_start_month"):
            manager.validate_config(config)

        config.date_formatting.fiscal_year_start_month = 13
        with pytest.raises(ConfigurationError, match="Invalid fiscal_year_start_month"):
            manager.validate_config(config)

    def test_invalid_format_style_string(self):
        """Test validation fails for invalid format style string."""
        # Create config with invalid format style
        config_data = {
            "date_formatting": {
                "format_style": "invalid_style"
            }
        }
        
        manager = ConfigManager()
        
        # This should not raise during creation due to automatic conversion to SMART
        config = manager._create_config_from_dict(config_data)
        assert config.date_formatting.format_style == DateFormatStyle.SMART

    def test_invalid_locale(self):
        """Test validation fails for invalid locale."""
        config = Config(
            date_formatting=DateFormattingConfig(locale="")
        )
        
        manager = ConfigManager()
        with pytest.raises(ConfigurationError, match="Invalid locale"):
            manager.validate_config(config)

        config.date_formatting.locale = "x"  # Too short
        with pytest.raises(ConfigurationError, match="Invalid locale"):
            manager.validate_config(config)

    def test_invalid_boolean_values(self):
        """Test validation fails for invalid boolean values."""
        # Test invalid enabled value
        config = Config()
        config.date_formatting.enabled = "not_a_boolean"
        
        manager = ConfigManager()
        with pytest.raises(ConfigurationError, match="Invalid enabled value"):
            manager.validate_config(config)

        # Test invalid fallback_to_iso value
        config.date_formatting.enabled = True
        config.date_formatting.fallback_to_iso = "not_a_boolean"
        
        with pytest.raises(ConfigurationError, match="Invalid fallback_to_iso value"):
            manager.validate_config(config)


class TestConfigCreationFromDict:
    """Test creation of Config objects from dictionaries."""

    def test_create_config_with_date_formatting_dict(self):
        """Test creating Config from dictionary with date formatting."""
        config_data = {
            "llm_provider": "anthropic",
            "cache_ttl": 7200,
            "date_formatting": {
                "enabled": False,
                "format_style": "compact",
                "fiscal_year_start_month": 4,
                "locale": "de_DE",
                "fallback_to_iso": False
            }
        }
        
        manager = ConfigManager()
        config = manager._create_config_from_dict(config_data)
        
        assert config.llm_provider == "anthropic"
        assert config.cache_ttl == 7200
        assert config.date_formatting.enabled is False
        assert config.date_formatting.format_style == DateFormatStyle.COMPACT
        assert config.date_formatting.fiscal_year_start_month == 4
        assert config.date_formatting.locale == "de_DE"
        assert config.date_formatting.fallback_to_iso is False

    def test_create_config_without_date_formatting_dict(self):
        """Test creating Config from dictionary without date formatting uses defaults."""
        config_data = {
            "llm_provider": "ollama",
            "cache_ttl": 1800
        }
        
        manager = ConfigManager()
        config = manager._create_config_from_dict(config_data)
        
        assert config.llm_provider == "ollama"
        assert config.cache_ttl == 1800
        # Should have default date formatting config
        assert config.date_formatting.enabled is True
        assert config.date_formatting.format_style == DateFormatStyle.SMART
        assert config.date_formatting.fiscal_year_start_month == 1
        assert config.date_formatting.locale == "en_US"
        assert config.date_formatting.fallback_to_iso is True

    def test_create_config_with_empty_date_formatting_dict(self):
        """Test creating Config with empty date formatting dictionary uses defaults."""
        config_data = {
            "llm_provider": "bedrock",
            "date_formatting": {}
        }
        
        manager = ConfigManager()
        config = manager._create_config_from_dict(config_data)
        
        assert config.llm_provider == "bedrock"
        # Should have default date formatting config
        assert config.date_formatting.enabled is True
        assert config.date_formatting.format_style == DateFormatStyle.SMART
        assert config.date_formatting.fiscal_year_start_month == 1


class TestConfigurationFileFormats:
    """Test different configuration file formats with date formatting."""

    def test_yaml_config_with_date_formatting(self):
        """Test loading YAML configuration with date formatting."""
        config_content = """
llm_provider: openai
cache_ttl: 3600
date_formatting:
  enabled: true
  format_style: verbose
  fiscal_year_start_month: 7
  locale: en_AU
  fallback_to_iso: false
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            config_path = f.name
        
        try:
            manager = ConfigManager()
            config = manager.load_config(config_path)
            
            assert config.date_formatting.enabled is True
            assert config.date_formatting.format_style == DateFormatStyle.VERBOSE
            assert config.date_formatting.fiscal_year_start_month == 7
            assert config.date_formatting.locale == "en_AU"
            assert config.date_formatting.fallback_to_iso is False
        finally:
            os.unlink(config_path)

    def test_json_config_with_date_formatting(self):
        """Test loading JSON configuration with date formatting."""
        config_data = {
            "llm_provider": "gemini",
            "date_formatting": {
                "enabled": False,
                "format_style": "compact",
                "fiscal_year_start_month": 10,
                "locale": "ja_JP",
                "fallback_to_iso": True
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name
        
        try:
            manager = ConfigManager()
            config = manager.load_config(config_path)
            
            assert config.date_formatting.enabled is False
            assert config.date_formatting.format_style == DateFormatStyle.COMPACT
            assert config.date_formatting.fiscal_year_start_month == 10
            assert config.date_formatting.locale == "ja_JP"
            assert config.date_formatting.fallback_to_iso is True
        finally:
            os.unlink(config_path)