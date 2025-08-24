"""Tests for configuration management."""

import os
import json
import yaml
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch

from aws_cost_cli.config import ConfigManager
from aws_cost_cli.models import Config


class TestConfigManager:
    """Test cases for ConfigManager."""
    
    def test_load_default_config(self):
        """Test loading default configuration."""
        manager = ConfigManager()
        config = manager.load_config()
        
        assert config.llm_provider == "openai"
        assert config.cache_ttl == 3600
        assert config.output_format == "simple"
        assert config.default_currency == "USD"
        assert config.default_profile is None
    
    def test_load_yaml_config_file(self):
        """Test loading configuration from YAML file."""
        config_data = {
            "llm_provider": "anthropic",
            "cache_ttl": 7200,
            "output_format": "detailed",
            "llm_config": {
                "anthropic": {
                    "api_key": "test-key"
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            manager = ConfigManager()
            config = manager.load_config(config_path)
            
            assert config.llm_provider == "anthropic"
            assert config.cache_ttl == 7200
            assert config.output_format == "detailed"
            assert config.llm_config["anthropic"]["api_key"] == "test-key"
        finally:
            os.unlink(config_path)
    
    def test_load_json_config_file(self):
        """Test loading configuration from JSON file."""
        config_data = {
            "llm_provider": "bedrock",
            "default_profile": "production",
            "llm_config": {
                "bedrock": {
                    "region": "us-west-2"
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name
        
        try:
            manager = ConfigManager()
            config = manager.load_config(config_path)
            
            assert config.llm_provider == "bedrock"
            assert config.default_profile == "production"
            assert config.llm_config["bedrock"]["region"] == "us-west-2"
        finally:
            os.unlink(config_path)
    
    def test_config_file_not_found(self):
        """Test handling of missing configuration file."""
        manager = ConfigManager()
        
        with pytest.raises(FileNotFoundError):
            manager.load_config("/nonexistent/config.yaml")
    
    def test_invalid_yaml_config(self):
        """Test handling of invalid YAML configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            config_path = f.name
        
        try:
            manager = ConfigManager()
            with pytest.raises(ValueError, match="Invalid configuration file format"):
                manager.load_config(config_path)
        finally:
            os.unlink(config_path)
    
    @patch.dict(os.environ, {
        "AWS_COST_CLI_LLM_PROVIDER": "ollama",
        "AWS_COST_CLI_CACHE_TTL": "1800",
        "AWS_COST_CLI_OUTPUT_FORMAT": "json",
        "OPENAI_API_KEY": "test-openai-key"
    })
    def test_load_env_config(self):
        """Test loading configuration from environment variables."""
        manager = ConfigManager()
        config = manager.load_config()
        
        assert config.llm_provider == "ollama"
        assert config.cache_ttl == 1800
        assert config.output_format == "json"
        assert config.llm_config["openai"]["api_key"] == "test-openai-key"
    
    @patch.dict(os.environ, {"AWS_COST_CLI_CACHE_TTL": "invalid"})
    def test_invalid_env_config(self):
        """Test handling of invalid environment variable values."""
        manager = ConfigManager()
        config = manager.load_config()
        
        # Should fall back to default value
        assert config.cache_ttl == 3600
    
    def test_save_yaml_config(self):
        """Test saving configuration to YAML file."""
        config = Config(
            llm_provider="anthropic",
            cache_ttl=7200,
            output_format="detailed"
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_path = f.name
        
        try:
            manager = ConfigManager()
            manager.save_config(config, config_path)
            
            # Load and verify
            loaded_config = manager.load_config(config_path)
            assert loaded_config.llm_provider == "anthropic"
            assert loaded_config.cache_ttl == 7200
            assert loaded_config.output_format == "detailed"
        finally:
            os.unlink(config_path)
    
    def test_validate_config_valid(self):
        """Test validation of valid configuration."""
        config = Config(
            llm_provider="openai",
            output_format="simple",
            cache_ttl=3600
        )
        
        manager = ConfigManager()
        assert manager.validate_config(config) is True
    
    def test_validate_config_invalid_provider(self):
        """Test validation of invalid LLM provider."""
        config = Config(llm_provider="invalid_provider")
        
        manager = ConfigManager()
        with pytest.raises(ValueError, match="Invalid LLM provider"):
            manager.validate_config(config)
    
    def test_validate_config_invalid_format(self):
        """Test validation of invalid output format."""
        config = Config(output_format="invalid_format")
        
        manager = ConfigManager()
        with pytest.raises(ValueError, match="Invalid output format"):
            manager.validate_config(config)
    
    def test_validate_config_negative_ttl(self):
        """Test validation of negative cache TTL."""
        config = Config(cache_ttl=-100)
        
        manager = ConfigManager()
        with pytest.raises(ValueError, match="Cache TTL must be non-negative"):
            manager.validate_config(config)
    
    def test_auto_discover_config_file(self):
        """Test automatic discovery of configuration files."""
        config_data = {"llm_provider": "bedrock"}
        
        # Create config in current directory
        with tempfile.NamedTemporaryFile(mode='w', suffix='.aws-cost-cli.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            # Temporarily rename to match expected pattern
            expected_path = ".aws-cost-cli.yaml"
            os.rename(config_path, expected_path)
            
            manager = ConfigManager()
            config = manager.load_config()
            
            assert config.llm_provider == "bedrock"
        finally:
            if os.path.exists(expected_path):
                os.unlink(expected_path)