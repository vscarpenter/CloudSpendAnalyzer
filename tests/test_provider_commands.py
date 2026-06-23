"""Tests for provider listing and testing CLI commands."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from click.testing import CliRunner

from src.aws_cost_cli.cli import cli
from src.aws_cost_cli.models import Config
from src.aws_cost_cli.exceptions import ConfigurationError, LLMProviderError


class TestListProvidersCommand:
    """Test the list-providers CLI command."""

    def test_list_providers_success(self):
        """Test successful provider listing."""
        runner = CliRunner()
        
        with patch('src.aws_cost_cli.cli.ConfigManager') as mock_config_manager, \
             patch('src.aws_cost_cli.cli.ProviderFactory') as mock_factory:
            
            # Mock configuration
            mock_config = Config()
            mock_config.llm_provider = "ollama"
            mock_config.llm_config = {
                "ollama": {"model": "gpt-oss:20b", "base_url": "http://localhost:11434"},
                "openai": {"model": "gpt-3.5-turbo"}
            }
            
            mock_config_manager.return_value.load_config.return_value = mock_config
            
            # Mock provider factory
            mock_factory.get_all_provider_names.return_value = ["openai", "anthropic", "bedrock", "ollama", "gemini"]
            mock_factory.get_provider_configuration_status.return_value = {
                "openai": {"configured": True, "available": True, "error": None},
                "anthropic": {"configured": False, "available": False, "error": "API key required"},
                "bedrock": {"configured": True, "available": True, "error": None},
                "ollama": {"configured": True, "available": False, "error": "Server not running"},
                "gemini": {"configured": True, "available": True, "error": None}
            }
            mock_factory.get_available_providers.return_value = ["openai", "bedrock", "gemini"]
            
            result = runner.invoke(cli, ['list-providers'])
            
            assert result.exit_code == 0
            assert "LLM Provider Status" in result.output
            assert "Openai" in result.output
            assert "✅ Ready" in result.output
            assert "❌ Not" in result.output  # "❌ Not configured" is split across lines in table
            assert "Current default provider: ollama" in result.output
            assert "Available providers: openai, bedrock, gemini" in result.output

    def test_list_providers_no_config_file(self):
        """Test provider listing when no config file exists."""
        runner = CliRunner()
        
        with patch('src.aws_cost_cli.cli.ConfigManager') as mock_config_manager, \
             patch('src.aws_cost_cli.cli.ProviderFactory') as mock_factory:
            
            # Mock FileNotFoundError for config loading
            mock_config_manager.return_value.load_config.side_effect = FileNotFoundError()
            
            # Mock provider factory
            mock_factory.get_all_provider_names.return_value = ["openai", "anthropic", "bedrock", "ollama", "gemini"]
            mock_factory.get_provider_configuration_status.return_value = {
                "openai": {"configured": False, "available": False, "error": "API key required"},
                "anthropic": {"configured": False, "available": False, "error": "API key required"},
                "bedrock": {"configured": False, "available": False, "error": "AWS credentials required"},
                "ollama": {"configured": False, "available": False, "error": "Server not running"},
                "gemini": {"configured": False, "available": False, "error": "API key required"}
            }
            mock_factory.get_available_providers.return_value = []
            
            result = runner.invoke(cli, ['list-providers'])
            
            assert result.exit_code == 0
            assert "No providers are currently available" in result.output

    def test_list_providers_with_config_file(self):
        """Test provider listing with custom config file."""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            # Create a temporary config file
            with open('test_config.yaml', 'w') as f:
                f.write('llm_provider: openai\n')
            
            with patch('src.aws_cost_cli.cli.ConfigManager') as mock_config_manager, \
                 patch('src.aws_cost_cli.cli.ProviderFactory') as mock_factory:
                
                mock_config = Config()
                mock_config_manager.return_value.load_config.return_value = mock_config
                mock_factory.get_all_provider_names.return_value = ["openai"]
                mock_factory.get_provider_configuration_status.return_value = {
                    "openai": {"configured": True, "available": True, "error": None}
                }
                mock_factory.get_available_providers.return_value = ["openai"]
                
                result = runner.invoke(cli, ['list-providers', '--config-file', 'test_config.yaml'])
                
                assert result.exit_code == 0
                mock_config_manager.return_value.load_config.assert_called_with('test_config.yaml')

    def test_list_providers_error_handling(self):
        """Test error handling in list-providers command."""
        runner = CliRunner()
        
        with patch('src.aws_cost_cli.cli.ConfigManager') as mock_config_manager:
            mock_config_manager.return_value.load_config.side_effect = Exception("Test error")
            
            result = runner.invoke(cli, ['list-providers'])
            
            assert result.exit_code == 1
            assert "Failed to list providers" in result.output


class TestTestProviderCommand:
    """Test the test-provider CLI command."""

    def test_test_provider_success(self):
        """Test successful provider testing."""
        runner = CliRunner()
        
        with patch('src.aws_cost_cli.cli.ConfigManager') as mock_config_manager, \
             patch('src.aws_cost_cli.cli.ProviderFactory') as mock_factory:
            
            # Mock configuration
            mock_config = Config()
            mock_config.llm_config = {"openai": {"api_key": "test-key", "model": "gpt-3.5-turbo"}}
            mock_config_manager.return_value.load_config.return_value = mock_config
            
            # Mock provider instance
            mock_provider = Mock()
            mock_provider.is_available.return_value = True
            mock_provider.parse_query.return_value = {
                "service": "EC2",
                "date_range": {"start": "2024-01-01", "end": "2024-01-31"},
                "granularity": "MONTHLY"
            }
            mock_factory.create_provider.return_value = mock_provider
            
            result = runner.invoke(cli, ['test-provider', 'openai'])
            
            assert result.exit_code == 0
            assert "Testing Openai provider configuration" in result.output
            assert "Provider instance created successfully" in result.output
            assert "Provider is available and ready" in result.output
            assert "Query parsing successful" in result.output
            assert "Openai provider is working correctly!" in result.output

    def test_test_provider_creation_failure(self):
        """Test provider testing when provider creation fails."""
        runner = CliRunner()
        
        with patch('src.aws_cost_cli.cli.ConfigManager') as mock_config_manager, \
             patch('src.aws_cost_cli.cli.ProviderFactory') as mock_factory:
            
            mock_config = Config()
            mock_config_manager.return_value.load_config.return_value = mock_config
            
            # Mock provider creation failure
            mock_factory.create_provider.side_effect = ConfigurationError("API key required")
            
            result = runner.invoke(cli, ['test-provider', 'openai'])
            
            assert result.exit_code == 1
            assert "Failed to create provider" in result.output
            assert "API key required" in result.output
            assert "Set OPENAI_API_KEY environment variable" in result.output

    def test_test_provider_not_available(self):
        """Test provider testing when provider is not available."""
        runner = CliRunner()
        
        with patch('src.aws_cost_cli.cli.ConfigManager') as mock_config_manager, \
             patch('src.aws_cost_cli.cli.ProviderFactory') as mock_factory:
            
            mock_config = Config()
            mock_config_manager.return_value.load_config.return_value = mock_config
            
            # Mock provider instance that's not available
            mock_provider = Mock()
            mock_provider.is_available.return_value = False
            mock_factory.create_provider.return_value = mock_provider
            
            result = runner.invoke(cli, ['test-provider', 'ollama'])
            
            assert result.exit_code == 1
            assert "Provider created but not available" in result.output

    def test_test_provider_query_parsing_failure(self):
        """Test provider testing when query parsing fails."""
        runner = CliRunner()
        
        with patch('src.aws_cost_cli.cli.ConfigManager') as mock_config_manager, \
             patch('src.aws_cost_cli.cli.ProviderFactory') as mock_factory:
            
            mock_config = Config()
            mock_config.llm_config = {"ollama": {"model": "gpt-oss:20b"}}
            mock_config_manager.return_value.load_config.return_value = mock_config
            
            # Mock provider instance
            mock_provider = Mock()
            mock_provider.is_available.return_value = True
            mock_provider.parse_query.side_effect = Exception("Connection refused")
            mock_factory.create_provider.return_value = mock_provider
            
            result = runner.invoke(cli, ['test-provider', 'ollama'])
            
            assert result.exit_code == 1
            assert "Query parsing failed" in result.output
            assert "Connection refused" in result.output
            assert "Ollama troubleshooting" in result.output

    def test_test_provider_with_config_file(self):
        """Test provider testing with custom config file."""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            # Create a temporary config file
            with open('test_config.yaml', 'w') as f:
                f.write('llm_provider: gemini\n')
            
            with patch('src.aws_cost_cli.cli.ConfigManager') as mock_config_manager, \
                 patch('src.aws_cost_cli.cli.ProviderFactory') as mock_factory:
                
                mock_config = Config()
                mock_config_manager.return_value.load_config.return_value = mock_config
                
                mock_provider = Mock()
                mock_provider.is_available.return_value = True
                mock_provider.parse_query.return_value = {"service": "EC2"}
                mock_factory.create_provider.return_value = mock_provider
                
                result = runner.invoke(cli, ['test-provider', 'gemini', '--config-file', 'test_config.yaml'])
                
                assert result.exit_code == 0
                mock_config_manager.return_value.load_config.assert_called_with('test_config.yaml')

    def test_test_provider_api_key_masking(self):
        """Test that API keys are properly masked in output."""
        runner = CliRunner()
        
        with patch('src.aws_cost_cli.cli.ConfigManager') as mock_config_manager, \
             patch('src.aws_cost_cli.cli.ProviderFactory') as mock_factory:
            
            mock_config = Config()
            mock_config.llm_config = {
                "openai": {
                    "api_key": "sk-1234567890abcdef1234567890abcdef",
                    "model": "gpt-3.5-turbo"
                }
            }
            mock_config_manager.return_value.load_config.return_value = mock_config
            
            mock_provider = Mock()
            mock_provider.is_available.return_value = True
            mock_provider.parse_query.return_value = {"service": "EC2"}
            mock_factory.create_provider.return_value = mock_provider
            
            result = runner.invoke(cli, ['test-provider', 'openai'])
            
            assert result.exit_code == 0
            assert "sk-12345..." in result.output  # API key should be masked
            assert "sk-1234567890abcdef1234567890abcdef" not in result.output  # Full key should not appear

    def test_test_provider_all_providers(self):
        """Test that all supported providers can be tested."""
        runner = CliRunner()
        providers = ["openai", "anthropic", "bedrock", "ollama", "gemini"]
        
        for provider in providers:
            with patch('src.aws_cost_cli.cli.ConfigManager') as mock_config_manager, \
                 patch('src.aws_cost_cli.cli.ProviderFactory') as mock_factory:
                
                mock_config = Config()
                mock_config_manager.return_value.load_config.return_value = mock_config
                
                # Mock provider creation failure to test error handling
                mock_factory.create_provider.side_effect = ConfigurationError("Not configured")
                
                result = runner.invoke(cli, ['test-provider', provider])
                
                assert result.exit_code == 1
                assert f"Testing {provider.title()} provider configuration" in result.output

    def test_test_provider_error_handling(self):
        """Test error handling in test-provider command."""
        runner = CliRunner()
        
        with patch('src.aws_cost_cli.cli.ConfigManager') as mock_config_manager:
            mock_config_manager.return_value.load_config.side_effect = Exception("Test error")
            
            result = runner.invoke(cli, ['test-provider', 'openai'])
            
            assert result.exit_code == 1
            assert "Provider test failed" in result.output