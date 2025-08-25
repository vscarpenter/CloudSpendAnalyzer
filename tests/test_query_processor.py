"""Unit tests for query processor and LLM integration."""

import json
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, MagicMock

import pytest

from src.aws_cost_cli.query_processor import (
    QueryParser, FallbackParser, OpenAIProvider, AnthropicProvider, BedrockProvider, OllamaProvider
)
from src.aws_cost_cli.models import QueryParameters, TimePeriod, TimePeriodGranularity, MetricType
from src.aws_cost_cli.exceptions import LLMProviderError, QueryParsingError, ValidationError


class TestFallbackParser:
    """Test cases for FallbackParser class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = FallbackParser()
    
    def test_extract_service_ec2(self):
        """Test extracting EC2 service from query."""
        query = "How much did EC2 cost last month?"
        result = self.parser.parse_query(query)
        assert result["service"] == "Amazon Elastic Compute Cloud - Compute"
    
    def test_extract_service_s3(self):
        """Test extracting S3 service from query."""
        query = "What's my S3 spending this year?"
        result = self.parser.parse_query(query)
        assert result["service"] == "Amazon Simple Storage Service"
    
    def test_extract_service_none(self):
        """Test when no service is mentioned."""
        query = "What's my total bill?"
        result = self.parser.parse_query(query)
        assert result["service"] is None
    
    def test_extract_granularity_daily(self):
        """Test extracting daily granularity."""
        query = "Show me daily costs for EC2"
        result = self.parser.parse_query(query)
        assert result["granularity"] == "DAILY"
    
    def test_extract_granularity_monthly_default(self):
        """Test default monthly granularity."""
        query = "Show me EC2 costs"
        result = self.parser.parse_query(query)
        assert result["granularity"] == "MONTHLY"
    
    def test_extract_time_period_last_month(self):
        """Test extracting last month time period."""
        query = "EC2 costs last month"
        result = self.parser.parse_query(query)
        
        assert result["start_date"] is not None
        assert result["end_date"] is not None
        
        # Verify it's actually last month
        start_date = datetime.fromisoformat(result["start_date"])
        end_date = datetime.fromisoformat(result["end_date"])
        
        today = datetime.now(timezone.utc)
        expected_start = today.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        expected_start = (expected_start - timedelta(days=1)).replace(day=1)
        
        assert start_date.month == expected_start.month
        assert start_date.year == expected_start.year
    
    def test_extract_time_period_this_year(self):
        """Test extracting this year time period."""
        query = "Total costs this year"
        result = self.parser.parse_query(query)
        
        assert result["start_date"] is not None
        assert result["end_date"] is not None
        
        start_date = datetime.fromisoformat(result["start_date"])
        current_year = datetime.now(timezone.utc).year
        
        assert start_date.year == current_year
        assert start_date.month == 1
        assert start_date.day == 1
    
    def test_extract_specific_date(self):
        """Test extracting specific date from query."""
        query = "Costs on 2024-01-15"
        result = self.parser.parse_query(query)
        
        assert result["start_date"] == "2024-01-15"
        assert result["end_date"] == "2024-01-16"  # Next day
    
    def test_no_time_period(self):
        """Test when no time period is specified."""
        query = "EC2 costs"
        result = self.parser.parse_query(query)
        
        assert result["start_date"] is None
        assert result["end_date"] is None
    
    def test_extract_service_breakdown(self):
        """Test extracting service breakdown request."""
        queries = [
            "What services did I use last month?",
            "List the services that cost money",
            "Show me service breakdown",
            "Which services did I spend money on?"
        ]
        
        for query in queries:
            result = self.parser.parse_query(query)
            assert result["group_by"] == ["SERVICE"], f"Failed for query: {query}"


class TestOpenAIProvider:
    """Test cases for OpenAIProvider class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.provider = OpenAIProvider(api_key="test-key")
    
    def test_is_available_with_key(self):
        """Test availability check with API key."""
        with patch('openai.OpenAI'):
            assert self.provider.is_available() is True
    
    def test_is_available_without_key(self):
        """Test availability check without API key."""
        provider = OpenAIProvider(api_key="")
        with patch('openai.OpenAI'):
            assert provider.is_available() is False
    
    def test_is_available_import_error(self):
        """Test availability check when openai package is not installed."""
        with patch('openai.OpenAI', side_effect=ImportError()):
            assert self.provider.is_available() is False
    
    @patch('openai.OpenAI')
    def test_parse_query_success(self, mock_openai):
        """Test successful query parsing."""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '''
        {
            "service": "EC2",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "granularity": "MONTHLY",
            "metrics": ["BlendedCost"],
            "group_by": ["SERVICE"]
        }
        '''
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        result = self.provider.parse_query("EC2 costs last month")
        
        assert result["service"] == "EC2"
        assert result["start_date"] == "2024-01-01"
        assert result["granularity"] == "MONTHLY"
    
    @patch('openai.OpenAI')
    def test_parse_query_api_error(self, mock_openai):
        """Test handling of OpenAI API errors."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai.return_value = mock_client
        
        with pytest.raises(LLMProviderError, match="OpenAI API error"):
            self.provider.parse_query("test query")
    
    def test_parse_llm_response_valid_json(self):
        """Test parsing valid JSON response."""
        content = '{"service": "EC2", "granularity": "MONTHLY"}'
        result = self.provider._parse_llm_response(content)
        
        assert result["service"] == "EC2"
        assert result["granularity"] == "MONTHLY"
    
    def test_parse_llm_response_json_in_text(self):
        """Test parsing JSON embedded in text."""
        content = 'Here is the result: {"service": "S3", "granularity": "DAILY"} as requested.'
        result = self.provider._parse_llm_response(content)
        
        assert result["service"] == "S3"
        assert result["granularity"] == "DAILY"
    
    def test_parse_llm_response_invalid_json(self):
        """Test handling of invalid JSON response."""
        content = "This is not valid JSON"
        
        with pytest.raises(QueryParsingError, match="Could not parse LLM response as JSON"):
            self.provider._parse_llm_response(content)


class TestBedrockProvider:
    """Test cases for BedrockProvider class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.provider = BedrockProvider(
            model="anthropic.claude-3-haiku-20240307-v1:0",
            region="us-east-1"
        )
    
    def test_is_available_with_credentials(self):
        """Test availability check with AWS credentials."""
        with patch('boto3.Session') as mock_session:
            mock_client = Mock()
            mock_session.return_value.client.return_value = mock_client
            assert self.provider.is_available() is True
    
    def test_is_available_import_error(self):
        """Test availability check when boto3 package is not installed."""
        with patch('boto3.Session', side_effect=ImportError()):
            assert self.provider.is_available() is False
    
    def test_is_available_credentials_error(self):
        """Test availability check with credential errors."""
        with patch('boto3.Session', side_effect=Exception("Credentials error")):
            assert self.provider.is_available() is False
    
    @patch('boto3.Session')
    def test_parse_query_claude_success(self, mock_session):
        """Test successful query parsing with Claude model."""
        # Mock Bedrock response for Claude
        mock_response = {
            'body': Mock(),
            'ResponseMetadata': {'HTTPStatusCode': 200}
        }
        mock_response['body'].read.return_value = json.dumps({
            'content': [{
                'text': '''
                {
                    "service": "Amazon Elastic Compute Cloud - Compute",
                    "start_date": "2025-07-01",
                    "end_date": "2025-08-01",
                    "granularity": "MONTHLY",
                    "metrics": ["BlendedCost"],
                    "group_by": null
                }
                '''
            }]
        }).encode('utf-8')
        
        mock_client = Mock()
        mock_client.invoke_model.return_value = mock_response
        mock_session.return_value.client.return_value = mock_client
        
        result = self.provider.parse_query("EC2 costs last month")
        
        assert result["service"] == "Amazon Elastic Compute Cloud - Compute"
        assert result["start_date"] == "2025-07-01"
        assert result["granularity"] == "MONTHLY"
        
        # Verify the correct model was called
        mock_client.invoke_model.assert_called_once()
        call_args = mock_client.invoke_model.call_args
        assert call_args[1]['modelId'] == "anthropic.claude-3-haiku-20240307-v1:0"
    
    @patch('boto3.Session')
    def test_parse_query_titan_success(self, mock_session):
        """Test successful query parsing with Titan model."""
        provider = BedrockProvider(model="amazon.titan-text-express-v1", region="us-east-1")
        
        # Mock Bedrock response for Titan
        mock_response = {
            'body': Mock(),
            'ResponseMetadata': {'HTTPStatusCode': 200}
        }
        mock_response['body'].read.return_value = json.dumps({
            'results': [{
                'outputText': '''
                {
                    "service": "Amazon Simple Storage Service",
                    "start_date": "2025-08-01",
                    "end_date": "2025-08-24",
                    "granularity": "MONTHLY",
                    "metrics": ["BlendedCost"],
                    "group_by": null
                }
                '''
            }]
        }).encode('utf-8')
        
        mock_client = Mock()
        mock_client.invoke_model.return_value = mock_response
        mock_session.return_value.client.return_value = mock_client
        
        result = provider.parse_query("S3 costs this month")
        
        assert result["service"] == "Amazon Simple Storage Service"
        assert result["start_date"] == "2025-08-01"
        assert result["granularity"] == "MONTHLY"
    
    @patch('boto3.Session')
    def test_parse_query_credentials_error(self, mock_session):
        """Test handling of AWS credentials errors."""
        mock_client = Mock()
        mock_client.invoke_model.side_effect = Exception("UnauthorizedOperation")
        mock_session.return_value.client.return_value = mock_client
        
        with pytest.raises(LLMProviderError, match="Invalid AWS credentials"):
            self.provider.parse_query("test query")
    
    @patch('boto3.Session')
    def test_parse_query_model_not_found(self, mock_session):
        """Test handling of model not found errors."""
        mock_client = Mock()
        mock_client.invoke_model.side_effect = Exception("model not found")
        mock_session.return_value.client.return_value = mock_client
        
        with pytest.raises(LLMProviderError, match="Bedrock model not found"):
            self.provider.parse_query("test query")
    
    @patch('boto3.Session')
    def test_parse_query_api_error(self, mock_session):
        """Test handling of general Bedrock API errors."""
        mock_client = Mock()
        mock_client.invoke_model.side_effect = Exception("Service error")
        mock_session.return_value.client.return_value = mock_client
        
        with pytest.raises(LLMProviderError, match="Bedrock API error"):
            self.provider.parse_query("test query")
    
    def test_parse_llm_response_valid_json(self):
        """Test parsing valid JSON response."""
        content = '{"service": "EC2", "granularity": "MONTHLY"}'
        result = self.provider._parse_llm_response(content)
        
        assert result["service"] == "EC2"
        assert result["granularity"] == "MONTHLY"
    
    def test_parse_llm_response_json_in_text(self):
        """Test parsing JSON embedded in text."""
        content = 'Here is the result: {"service": "S3", "granularity": "DAILY"} as requested.'
        result = self.provider._parse_llm_response(content)
        
        assert result["service"] == "S3"
        assert result["granularity"] == "DAILY"
    
    def test_parse_llm_response_invalid_json(self):
        """Test handling of invalid JSON response."""
        content = "This is not valid JSON"
        
        with pytest.raises(QueryParsingError, match="Could not parse LLM response as JSON"):
            self.provider._parse_llm_response(content)
    
    def test_init_with_profile(self):
        """Test initialization with AWS profile."""
        provider = BedrockProvider(
            model="anthropic.claude-3-haiku-20240307-v1:0",
            region="us-west-2",
            profile="production"
        )
        
        assert provider.model == "anthropic.claude-3-haiku-20240307-v1:0"
        assert provider.region == "us-west-2"
        assert provider.profile == "production"


class TestAnthropicProvider:
    """Test cases for AnthropicProvider class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.provider = AnthropicProvider(api_key="test-key")
    
    def test_is_available_with_key(self):
        """Test availability check with API key."""
        with patch('anthropic.Anthropic'):
            assert self.provider.is_available() is True
    
    def test_is_available_without_key(self):
        """Test availability check without API key."""
        provider = AnthropicProvider(api_key="")
        with patch('anthropic.Anthropic'):
            assert provider.is_available() is False
    
    @patch('anthropic.Anthropic')
    def test_parse_query_success(self, mock_anthropic):
        """Test successful query parsing."""
        # Mock Anthropic response
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = '''
        {
            "service": "RDS",
            "start_date": "2024-02-01",
            "end_date": "2024-02-29",
            "granularity": "DAILY",
            "metrics": ["BlendedCost"],
            "group_by": null
        }
        '''
        
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        result = self.provider.parse_query("RDS costs in February")
        
        assert result["service"] == "RDS"
        assert result["start_date"] == "2024-02-01"
        assert result["granularity"] == "DAILY"


class TestOllamaProvider:
    """Test cases for OllamaProvider class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.provider = OllamaProvider()
    
    @patch('requests.get')
    def test_is_available_success(self, mock_get):
        """Test availability check when Ollama is running."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        assert self.provider.is_available() is True
    
    @patch('requests.get')
    def test_is_available_failure(self, mock_get):
        """Test availability check when Ollama is not running."""
        mock_get.side_effect = Exception("Connection error")
        
        assert self.provider.is_available() is False
    
    @patch('requests.post')
    @patch('requests.get')
    def test_parse_query_success(self, mock_get, mock_post):
        """Test successful query parsing."""
        # Mock availability check
        mock_get.return_value = Mock(status_code=200)
        
        # Mock Ollama response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": '{"service": "Lambda", "granularity": "HOURLY"}'
        }
        mock_post.return_value = mock_response
        
        result = self.provider.parse_query("Lambda costs hourly")
        
        assert result["service"] == "Lambda"
        assert result["granularity"] == "HOURLY"
    
    @patch('requests.post')
    @patch('requests.get')
    def test_parse_query_api_error(self, mock_get, mock_post):
        """Test handling of Ollama API errors."""
        # Mock availability check
        mock_get.return_value = Mock(status_code=200)
        
        # Mock API error
        mock_response = Mock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response
        
        with pytest.raises(LLMProviderError, match="Ollama API error"):
            self.provider.parse_query("test query")


class TestQueryParser:
    """Test cases for QueryParser class."""
    
    def test_init_openai_provider(self):
        """Test initialization with OpenAI provider."""
        config = {
            'provider': 'openai',
            'api_key': 'test-key',
            'model': 'gpt-4'
        }
        
        with patch('src.aws_cost_cli.query_processor.OpenAIProvider') as mock_provider:
            parser = QueryParser(config)
            mock_provider.assert_called_once_with('test-key', 'gpt-4')
    
    def test_init_anthropic_provider(self):
        """Test initialization with Anthropic provider."""
        config = {
            'provider': 'anthropic',
            'api_key': 'test-key',
            'model': 'claude-3-sonnet-20240229'
        }
        
        with patch('src.aws_cost_cli.query_processor.AnthropicProvider') as mock_provider:
            parser = QueryParser(config)
            mock_provider.assert_called_once_with('test-key', 'claude-3-sonnet-20240229')
    
    def test_init_bedrock_provider(self):
        """Test initialization with Bedrock provider."""
        config = {
            'provider': 'bedrock',
            'model': 'anthropic.claude-3-haiku-20240307-v1:0',
            'region': 'us-west-2',
            'profile': 'production'
        }
        
        with patch('src.aws_cost_cli.query_processor.BedrockProvider') as mock_provider:
            parser = QueryParser(config)
            mock_provider.assert_called_once_with('anthropic.claude-3-haiku-20240307-v1:0', 'us-west-2', 'production')
    
    def test_init_bedrock_provider_defaults(self):
        """Test initialization with Bedrock provider using defaults."""
        config = {
            'provider': 'bedrock'
        }
        
        with patch('src.aws_cost_cli.query_processor.BedrockProvider') as mock_provider:
            parser = QueryParser(config)
            mock_provider.assert_called_once_with('anthropic.claude-3-haiku-20240307-v1:0', 'us-east-1', None)
    
    def test_init_ollama_provider(self):
        """Test initialization with Ollama provider."""
        config = {
            'provider': 'ollama',
            'model': 'llama2',
            'base_url': 'http://localhost:11434'
        }
        
        with patch('src.aws_cost_cli.query_processor.OllamaProvider') as mock_provider:
            parser = QueryParser(config)
            mock_provider.assert_called_once_with('llama2', 'http://localhost:11434')
    
    def test_parse_query_with_llm_success(self):
        """Test successful query parsing with LLM provider."""
        config = {'provider': 'openai', 'api_key': 'test-key'}
        
        # Mock LLM provider
        mock_provider = Mock()
        mock_provider.is_available.return_value = True
        mock_provider.parse_query.return_value = {
            'service': 'EC2',
            'start_date': '2024-01-01',
            'end_date': '2024-01-31',
            'granularity': 'MONTHLY',
            'metrics': ['BlendedCost'],
            'group_by': ['SERVICE']
        }
        
        with patch('src.aws_cost_cli.query_processor.OpenAIProvider', return_value=mock_provider):
            parser = QueryParser(config)
            result = parser.parse_query("EC2 costs last month")
        
        assert isinstance(result, QueryParameters)
        assert result.service == 'EC2'
        assert result.granularity == TimePeriodGranularity.MONTHLY
        assert result.metrics == [MetricType.BLENDED_COST]
    
    def test_parse_query_fallback_to_pattern_matching(self):
        """Test fallback to pattern matching when LLM fails."""
        config = {'provider': 'openai', 'api_key': 'test-key'}
        
        # Mock LLM provider that fails
        mock_provider = Mock()
        mock_provider.is_available.return_value = True
        mock_provider.parse_query.side_effect = Exception("API Error")
        
        with patch('src.aws_cost_cli.query_processor.OpenAIProvider', return_value=mock_provider):
            parser = QueryParser(config)
            result = parser.parse_query("EC2 costs last month")
        
        assert isinstance(result, QueryParameters)
        assert result.service == 'Amazon Elastic Compute Cloud - Compute'
        # Should have extracted time period from "last month"
        assert result.time_period is not None
    
    def test_parse_query_no_providers_available(self):
        """Test parsing when no LLM providers are available."""
        config = {'provider': 'openai', 'api_key': ''}  # No API key
        
        parser = QueryParser(config)
        result = parser.parse_query("S3 costs this year")
        
        assert isinstance(result, QueryParameters)
        assert result.service == 'Amazon Simple Storage Service'
        # Should have extracted time period from "this year"
        assert result.time_period is not None
    
    def test_parse_query_complete_failure(self):
        """Test parsing when everything fails - should raise exception."""
        config = {}
        
        with patch.object(FallbackParser, 'parse_query', side_effect=Exception("Parse error")):
            parser = QueryParser(config)
            with pytest.raises(QueryParsingError, match="Failed to parse query"):
                parser.parse_query("invalid query")
    
    def test_validate_parameters_valid(self):
        """Test validation of valid parameters."""
        config = {}
        parser = QueryParser(config)
        
        params = QueryParameters(
            service="EC2",
            time_period=TimePeriod(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 1, 31, tzinfo=timezone.utc)
            ),
            granularity=TimePeriodGranularity.MONTHLY,
            metrics=[MetricType.BLENDED_COST]
        )
        
        assert parser.validate_parameters(params) is True
    
    def test_validate_parameters_invalid_time_period(self):
        """Test validation with invalid time period."""
        config = {}
        parser = QueryParser(config)
        
        params = QueryParameters(
            time_period=TimePeriod(
                start=datetime(2024, 1, 31, tzinfo=timezone.utc),
                end=datetime(2024, 1, 1, tzinfo=timezone.utc)  # End before start
            )
        )
        
        with pytest.raises(ValidationError, match="Start date must be before end date"):
            parser.validate_parameters(params)
    
    def test_convert_to_query_parameters_complete(self):
        """Test conversion of complete result dictionary."""
        config = {}
        parser = QueryParser(config)
        
        result = {
            'service': 'RDS',
            'start_date': '2024-02-01',
            'end_date': '2024-02-29',
            'granularity': 'DAILY',
            'metrics': ['UnblendedCost'],
            'group_by': ['SERVICE', 'INSTANCE_TYPE']
        }
        
        params = parser._convert_to_query_parameters(result)
        
        assert params.service == 'RDS'
        assert params.time_period.start.day == 1
        assert params.time_period.start.month == 2
        assert params.granularity == TimePeriodGranularity.DAILY
        assert params.metrics == [MetricType.UNBLENDED_COST]
        assert params.group_by == ['SERVICE', 'INSTANCE_TYPE']
    
    def test_convert_to_query_parameters_minimal(self):
        """Test conversion of minimal result dictionary."""
        config = {}
        parser = QueryParser(config)
        
        result = {'service': 'S3'}
        
        params = parser._convert_to_query_parameters(result)
        
        assert params.service == 'S3'
        assert params.time_period is None
        assert params.granularity == TimePeriodGranularity.MONTHLY
        assert params.metrics == [MetricType.BLENDED_COST]
        assert params.group_by is None
    
    def test_convert_to_query_parameters_invalid_values(self):
        """Test conversion with invalid values - should use defaults."""
        config = {}
        parser = QueryParser(config)
        
        result = {
            'service': 'EC2',
            'granularity': 'INVALID_GRANULARITY',
            'metrics': ['InvalidMetric', 'BlendedCost']
        }
        
        params = parser._convert_to_query_parameters(result)
        
        assert params.service == 'EC2'
        assert params.granularity == TimePeriodGranularity.MONTHLY  # Default
        assert params.metrics == [MetricType.BLENDED_COST]  # Only valid metric kept
    
    def test_multiple_providers_fallback(self):
        """Test that parser tries multiple providers before fallback."""
        config = {'provider': 'openai', 'api_key': 'test-key'}
        
        # Create parser and manually add multiple providers
        parser = QueryParser(config)
        
        # Mock multiple providers - first fails, second succeeds
        mock_provider1 = Mock()
        mock_provider1.is_available.return_value = True
        mock_provider1.parse_query.side_effect = Exception("Provider 1 failed")
        
        mock_provider2 = Mock()
        mock_provider2.is_available.return_value = True
        mock_provider2.parse_query.return_value = {
            'service': 'Lambda',
            'granularity': 'HOURLY'
        }
        
        parser._providers = {
            'provider1': mock_provider1,
            'provider2': mock_provider2
        }
        
        result = parser.parse_query("Lambda costs")
        
        assert result.service == 'Lambda'
        assert result.granularity == TimePeriodGranularity.HOURLY
        
        # Verify both providers were tried
        mock_provider1.parse_query.assert_called_once()
        mock_provider2.parse_query.assert_called_once()