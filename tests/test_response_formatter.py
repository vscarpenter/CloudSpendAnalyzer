"""Tests for response formatting system."""

import pytest
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock
import io

from src.aws_cost_cli.response_formatter import (
    ResponseGenerator,
    LLMResponseFormatter,
    SimpleResponseFormatter,
    RichResponseFormatter
)
from src.aws_cost_cli.models import (
    CostData,
    CostResult,
    CostAmount,
    TimePeriod,
    QueryParameters,
    TimePeriodGranularity,
    MetricType,
    Group
)


class TestSimpleResponseFormatter:
    """Test cases for SimpleResponseFormatter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = SimpleResponseFormatter()
        
        # Create test data
        self.time_period = TimePeriod(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 31, tzinfo=timezone.utc)
        )
        
        self.cost_data = CostData(
            results=[
                CostResult(
                    time_period=self.time_period,
                    total=CostAmount(Decimal('123.45'), 'USD'),
                    groups=[],
                    estimated=False
                )
            ],
            time_period=self.time_period,
            total_cost=CostAmount(Decimal('123.45'), 'USD'),
            currency='USD'
        )
        
        self.query_params = QueryParameters(
            service='EC2',
            time_period=self.time_period,
            granularity=TimePeriodGranularity.MONTHLY
        )
    
    def test_format_response_basic(self):
        """Test basic response formatting."""
        response = self.formatter.format_response(
            self.cost_data, 
            "What did I spend on EC2 last month?", 
            self.query_params
        )
        
        assert "AWS Cost Summary for EC2" in response
        assert "$123.45" in response
        assert "January 2024" in response
    
    def test_format_response_no_service(self):
        """Test response formatting without specific service."""
        query_params = QueryParameters(
            service=None,
            time_period=self.time_period,
            granularity=TimePeriodGranularity.MONTHLY
        )
        
        response = self.formatter.format_response(
            self.cost_data,
            "What did I spend last month?",
            query_params
        )
        
        assert "AWS Cost Summary (" in response
        assert "for EC2" not in response
    
    def test_format_currency_zero(self):
        """Test currency formatting for zero amount."""
        cost_amount = CostAmount(Decimal('0'), 'USD')
        result = self.formatter._format_currency(cost_amount)
        assert result == "$0.00"
    
    def test_format_currency_small_amount(self):
        """Test currency formatting for very small amounts."""
        cost_amount = CostAmount(Decimal('0.0012'), 'USD')
        result = self.formatter._format_currency(cost_amount)
        assert result == "$0.0012"
    
    def test_format_currency_normal_amount(self):
        """Test currency formatting for normal amounts."""
        cost_amount = CostAmount(Decimal('123.456'), 'USD')
        result = self.formatter._format_currency(cost_amount)
        assert result == "$123.46"
    
    def test_format_time_period_same_day(self):
        """Test time period formatting for same day."""
        period = TimePeriod(
            start=datetime(2024, 1, 15, tzinfo=timezone.utc),
            end=datetime(2024, 1, 15, tzinfo=timezone.utc)
        )
        result = self.formatter._format_time_period(period)
        assert result == "2024-01-15"
    
    def test_format_time_period_same_month(self):
        """Test time period formatting for same month."""
        period = TimePeriod(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 31, tzinfo=timezone.utc)
        )
        result = self.formatter._format_time_period(period)
        assert result == "January 2024"
    
    def test_format_time_period_different_months(self):
        """Test time period formatting for different months."""
        period = TimePeriod(
            start=datetime(2024, 1, 15, tzinfo=timezone.utc),
            end=datetime(2024, 2, 15, tzinfo=timezone.utc)
        )
        result = self.formatter._format_time_period(period)
        assert result == "2024-01-15 to 2024-02-15"
    
    def test_generate_insights_estimated_costs(self):
        """Test insight generation for estimated costs."""
        cost_data = CostData(
            results=[
                CostResult(
                    time_period=self.time_period,
                    total=CostAmount(Decimal('100'), 'USD'),
                    groups=[],
                    estimated=True
                )
            ],
            time_period=self.time_period,
            total_cost=CostAmount(Decimal('100'), 'USD')
        )
        
        insights = self.formatter._generate_simple_insights(cost_data, self.query_params)
        assert any("estimated costs" in insight for insight in insights)
    
    def test_generate_insights_zero_costs(self):
        """Test insight generation for zero costs."""
        cost_data = CostData(
            results=[],
            time_period=self.time_period,
            total_cost=CostAmount(Decimal('0'), 'USD')
        )
        
        insights = self.formatter._generate_simple_insights(cost_data, self.query_params)
        assert any("No costs found" in insight for insight in insights)
    
    def test_generate_insights_high_costs(self):
        """Test insight generation for high costs."""
        cost_data = CostData(
            results=[
                CostResult(
                    time_period=self.time_period,
                    total=CostAmount(Decimal('1500'), 'USD'),
                    groups=[],
                    estimated=False
                )
            ],
            time_period=self.time_period,
            total_cost=CostAmount(Decimal('1500'), 'USD')
        )
        
        insights = self.formatter._generate_simple_insights(cost_data, self.query_params)
        assert any("exceed $1,000" in insight for insight in insights)
    
    def test_format_response_with_groups(self):
        """Test response formatting with group data."""
        groups = [
            Group(
                keys=['EC2-Instance'],
                metrics={'BlendedCost': CostAmount(Decimal('50.00'), 'USD')}
            ),
            Group(
                keys=['EC2-Other'],
                metrics={'BlendedCost': CostAmount(Decimal('73.45'), 'USD')}
            )
        ]
        
        cost_data = CostData(
            results=[
                CostResult(
                    time_period=self.time_period,
                    total=CostAmount(Decimal('123.45'), 'USD'),
                    groups=groups,
                    estimated=False
                )
            ],
            time_period=self.time_period,
            total_cost=CostAmount(Decimal('123.45'), 'USD')
        )
        
        response = self.formatter.format_response(
            cost_data,
            "What did I spend on EC2?",
            self.query_params
        )
        
        assert "EC2-Instance" in response
        assert "EC2-Other" in response
        assert "$50.00" in response
        assert "$73.45" in response


class TestRichResponseFormatter:
    """Test cases for RichResponseFormatter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = RichResponseFormatter()
        
        self.time_period = TimePeriod(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 31, tzinfo=timezone.utc)
        )
        
        self.cost_data = CostData(
            results=[
                CostResult(
                    time_period=self.time_period,
                    total=CostAmount(Decimal('123.45'), 'USD'),
                    groups=[],
                    estimated=False
                )
            ],
            time_period=self.time_period,
            total_cost=CostAmount(Decimal('123.45'), 'USD')
        )
        
        self.query_params = QueryParameters(
            service='EC2',
            time_period=self.time_period,
            granularity=TimePeriodGranularity.MONTHLY
        )
    
    @patch('src.aws_cost_cli.response_formatter.RichResponseFormatter._check_rich_availability')
    def test_format_response_rich_available(self, mock_check_rich):
        """Test response formatting when Rich is available."""
        mock_check_rich.return_value = True
        
        with patch('rich.console.Console') as mock_console_class:
            mock_console = Mock()
            mock_console_class.return_value = mock_console
            
            # Mock string IO
            mock_string_io = Mock()
            mock_string_io.getvalue.return_value = "Rich formatted output"
            
            with patch('io.StringIO', return_value=mock_string_io):
                response = self.formatter.format_response(
                    self.cost_data,
                    "What did I spend on EC2?",
                    self.query_params
                )
            
            assert response == "Rich formatted output"
            assert mock_console.print.called
    
    def test_format_response_rich_not_available(self):
        """Test response formatting when Rich is not available."""
        # Create a new formatter instance that will check Rich availability
        formatter = RichResponseFormatter()
        formatter._rich_available = False  # Force Rich to be unavailable
        
        response = formatter.format_response(
            self.cost_data,
            "What did I spend on EC2?",
            self.query_params
        )
        
        # Should fall back to simple formatter
        assert "AWS Cost Summary for EC2" in response
        assert "$123.45" in response
    
    def test_check_rich_availability_available(self):
        """Test Rich availability check when Rich is available."""
        with patch('builtins.__import__'):
            result = self.formatter._check_rich_availability()
            # This will depend on whether rich is actually installed
            assert isinstance(result, bool)
    
    def test_check_rich_availability_not_available(self):
        """Test Rich availability check when Rich is not available."""
        with patch('builtins.__import__', side_effect=ImportError):
            result = self.formatter._check_rich_availability()
            assert result is False


class TestLLMResponseFormatter:
    """Test cases for LLMResponseFormatter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm_provider = Mock()
        self.mock_llm_provider.is_available.return_value = True
        
        self.formatter = LLMResponseFormatter(self.mock_llm_provider)
        
        self.time_period = TimePeriod(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 31, tzinfo=timezone.utc)
        )
        
        self.cost_data = CostData(
            results=[
                CostResult(
                    time_period=self.time_period,
                    total=CostAmount(Decimal('123.45'), 'USD'),
                    groups=[],
                    estimated=False
                )
            ],
            time_period=self.time_period,
            total_cost=CostAmount(Decimal('123.45'), 'USD')
        )
        
        self.query_params = QueryParameters(
            service='EC2',
            time_period=self.time_period,
            granularity=TimePeriodGranularity.MONTHLY
        )
    
    def test_format_response_llm_available(self):
        """Test response formatting when LLM is available."""
        # Mock the LLM provider to simulate OpenAI
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Your EC2 costs for January 2024 were $123.45."
        mock_client.chat.completions.create.return_value = mock_response
        
        self.mock_llm_provider._get_client.return_value = mock_client
        self.mock_llm_provider.model = 'gpt-3.5-turbo'
        
        response = self.formatter.format_response(
            self.cost_data,
            "What did I spend on EC2 last month?",
            self.query_params
        )
        
        assert "Your EC2 costs for January 2024 were $123.45." in response
    
    def test_format_response_llm_not_available(self):
        """Test response formatting when LLM is not available."""
        self.mock_llm_provider.is_available.return_value = False
        
        response = self.formatter.format_response(
            self.cost_data,
            "What did I spend on EC2 last month?",
            self.query_params
        )
        
        # Should fall back to simple formatter
        assert "AWS Cost Summary for EC2" in response
        assert "$123.45" in response
    
    def test_format_response_llm_error(self):
        """Test response formatting when LLM throws an error."""
        self.mock_llm_provider._get_client.side_effect = Exception("API Error")
        
        response = self.formatter.format_response(
            self.cost_data,
            "What did I spend on EC2 last month?",
            self.query_params
        )
        
        # Should fall back to simple formatter
        assert "AWS Cost Summary for EC2" in response
        assert "$123.45" in response
    
    def test_prepare_cost_summary(self):
        """Test cost summary preparation for LLM."""
        summary = self.formatter._prepare_cost_summary(self.cost_data, self.query_params)
        
        assert summary['total_cost']['amount'] == 123.45
        assert summary['total_cost']['currency'] == 'USD'
        assert summary['service'] == 'EC2'
        assert summary['granularity'] == 'MONTHLY'
        assert len(summary['results']) == 1
        assert summary['results'][0]['total']['amount'] == 123.45
    
    def test_prepare_cost_summary_with_groups(self):
        """Test cost summary preparation with group data."""
        groups = [
            Group(
                keys=['EC2-Instance'],
                metrics={'BlendedCost': CostAmount(Decimal('50.00'), 'USD')}
            )
        ]
        
        cost_data = CostData(
            results=[
                CostResult(
                    time_period=self.time_period,
                    total=CostAmount(Decimal('123.45'), 'USD'),
                    groups=groups,
                    estimated=False
                )
            ],
            time_period=self.time_period,
            total_cost=CostAmount(Decimal('123.45'), 'USD')
        )
        
        summary = self.formatter._prepare_cost_summary(cost_data, self.query_params)
        
        assert len(summary['results'][0]['groups']) == 1
        assert summary['results'][0]['groups'][0]['keys'] == ['EC2-Instance']
        assert summary['results'][0]['groups'][0]['metrics']['BlendedCost']['amount'] == 50.0
    
    def test_prepare_cost_summary_limits_results(self):
        """Test that cost summary limits results to avoid token limits."""
        # Create 15 results (more than the 10 limit)
        results = []
        for i in range(15):
            results.append(
                CostResult(
                    time_period=self.time_period,
                    total=CostAmount(Decimal('10.00'), 'USD'),
                    groups=[],
                    estimated=False
                )
            )
        
        cost_data = CostData(
            results=results,
            time_period=self.time_period,
            total_cost=CostAmount(Decimal('150.00'), 'USD')
        )
        
        summary = self.formatter._prepare_cost_summary(cost_data, self.query_params)
        
        # Should be limited to 10 results
        assert len(summary['results']) == 10


class TestResponseGenerator:
    """Test cases for ResponseGenerator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm_provider = Mock()
        self.mock_llm_provider.is_available.return_value = True
        
        self.time_period = TimePeriod(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 31, tzinfo=timezone.utc)
        )
        
        self.cost_data = CostData(
            results=[
                CostResult(
                    time_period=self.time_period,
                    total=CostAmount(Decimal('123.45'), 'USD'),
                    groups=[],
                    estimated=False
                )
            ],
            time_period=self.time_period,
            total_cost=CostAmount(Decimal('123.45'), 'USD')
        )
        
        self.query_params = QueryParameters(
            service='EC2',
            time_period=self.time_period,
            granularity=TimePeriodGranularity.MONTHLY
        )
    
    def test_init_with_llm_provider(self):
        """Test initialization with LLM provider."""
        generator = ResponseGenerator(
            llm_provider=self.mock_llm_provider,
            output_format="llm"
        )
        
        assert generator.llm_provider == self.mock_llm_provider
        assert generator.output_format == "llm"
        assert generator.llm_formatter is not None
    
    def test_init_without_llm_provider(self):
        """Test initialization without LLM provider."""
        generator = ResponseGenerator(output_format="simple")
        
        assert generator.llm_provider is None
        assert generator.output_format == "simple"
        assert generator.llm_formatter is None
    
    def test_format_response_simple(self):
        """Test response formatting with simple format."""
        generator = ResponseGenerator(output_format="simple")
        
        response = generator.format_response(
            self.cost_data,
            "What did I spend on EC2?",
            self.query_params
        )
        
        assert "AWS Cost Summary for EC2" in response
        assert "$123.45" in response
    
    @patch('src.aws_cost_cli.response_formatter.RichResponseFormatter._check_rich_availability')
    def test_format_response_rich(self, mock_check_rich):
        """Test response formatting with rich format."""
        mock_check_rich.return_value = False  # Force fallback to simple
        
        generator = ResponseGenerator(output_format="rich")
        
        response = generator.format_response(
            self.cost_data,
            "What did I spend on EC2?",
            self.query_params
        )
        
        # Should fall back to simple formatter when Rich not available
        assert "AWS Cost Summary for EC2" in response
        assert "$123.45" in response
    
    def test_format_response_llm(self):
        """Test response formatting with LLM format."""
        # Mock the LLM provider to simulate OpenAI
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Your EC2 costs were $123.45."
        mock_client.chat.completions.create.return_value = mock_response
        
        self.mock_llm_provider._get_client.return_value = mock_client
        self.mock_llm_provider.model = 'gpt-3.5-turbo'
        
        generator = ResponseGenerator(
            llm_provider=self.mock_llm_provider,
            output_format="llm"
        )
        
        response = generator.format_response(
            self.cost_data,
            "What did I spend on EC2?",
            self.query_params
        )
        
        assert "Your EC2 costs were $123.45." in response
    
    def test_format_response_error_fallback(self):
        """Test that errors fall back to simple formatter."""
        # Create a generator that will cause an error
        generator = ResponseGenerator(output_format="invalid_format")
        
        response = generator.format_response(
            self.cost_data,
            "What did I spend on EC2?",
            self.query_params
        )
        
        # Should fall back to simple formatter
        assert "AWS Cost Summary for EC2" in response
        assert "$123.45" in response
    
    def test_format_response_case_insensitive(self):
        """Test that output format is case insensitive."""
        generator = ResponseGenerator(output_format="SIMPLE")
        
        response = generator.format_response(
            self.cost_data,
            "What did I spend on EC2?",
            self.query_params
        )
        
        assert "AWS Cost Summary for EC2" in response
        assert "$123.45" in response


if __name__ == "__main__":
    pytest.main([__file__])