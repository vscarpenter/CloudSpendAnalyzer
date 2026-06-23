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
    RichResponseFormatter,
)
from src.aws_cost_cli.models import (
    CostData,
    CostResult,
    CostAmount,
    TimePeriod,
    QueryParameters,
    TimePeriodGranularity,
    MetricType,
    Group,
    DateFormattingConfig,
)


class TestSimpleResponseFormatter:
    """Test cases for SimpleResponseFormatter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = SimpleResponseFormatter()

        # Create test data
        self.time_period = TimePeriod(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(
                2024, 2, 1, tzinfo=timezone.utc
            ),  # Exclusive end date for full month
        )

        self.cost_data = CostData(
            results=[
                CostResult(
                    time_period=self.time_period,
                    total=CostAmount(Decimal("123.45"), "USD"),
                    groups=[],
                    estimated=False,
                )
            ],
            time_period=self.time_period,
            total_cost=CostAmount(Decimal("123.45"), "USD"),
            currency="USD",
        )

        self.query_params = QueryParameters(
            service="EC2",
            time_period=self.time_period,
            granularity=TimePeriodGranularity.MONTHLY,
        )

    def test_format_response_basic(self):
        """Test basic response formatting."""
        response = self.formatter.format_response(
            self.cost_data, "What did I spend on EC2 last month?", self.query_params
        )

        assert "AWS Cost Summary for EC2" in response
        assert "$123.45" in response
        assert "January 2024" in response

    def test_format_response_no_service(self):
        """Test response formatting without specific service."""
        query_params = QueryParameters(
            service=None,
            time_period=self.time_period,
            granularity=TimePeriodGranularity.MONTHLY,
        )

        response = self.formatter.format_response(
            self.cost_data, "What did I spend last month?", query_params
        )

        assert "AWS Cost Summary (" in response
        assert "for EC2" not in response

    def test_format_currency_zero(self):
        """Test currency formatting for zero amount."""
        cost_amount = CostAmount(Decimal("0"), "USD")
        result = self.formatter._format_currency(cost_amount)
        assert result == "$0.00"

    def test_format_currency_small_amount(self):
        """Test currency formatting for very small amounts."""
        cost_amount = CostAmount(Decimal("0.0012"), "USD")
        result = self.formatter._format_currency(cost_amount)
        assert result == "$0.0012"

    def test_format_currency_normal_amount(self):
        """Test currency formatting for normal amounts."""
        cost_amount = CostAmount(Decimal("123.456"), "USD")
        result = self.formatter._format_currency(cost_amount)
        assert result == "$123.46"

    def test_format_time_period_same_day(self):
        """Test time period formatting for same day."""
        period = TimePeriod(
            start=datetime(2024, 1, 15, tzinfo=timezone.utc),
            end=datetime(
                2024, 1, 16, tzinfo=timezone.utc
            ),  # Exclusive end date for single day
        )
        result = self.formatter._format_time_period(period)
        assert result == "January 15, 2024"

    def test_format_time_period_same_month(self):
        """Test time period formatting for same month."""
        period = TimePeriod(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(
                2024, 2, 1, tzinfo=timezone.utc
            ),  # Exclusive end date for full month
        )
        result = self.formatter._format_time_period(period)
        assert result == "January 2024"

    def test_format_time_period_different_months(self):
        """Test time period formatting for different months."""
        period = TimePeriod(
            start=datetime(2024, 1, 15, tzinfo=timezone.utc),
            end=datetime(2024, 2, 15, tzinfo=timezone.utc),
        )
        result = self.formatter._format_time_period(period)
        assert result == "January 15 - February 14, 2024"

    def test_generate_insights_estimated_costs(self):
        """Test insight generation for estimated costs."""
        cost_data = CostData(
            results=[
                CostResult(
                    time_period=self.time_period,
                    total=CostAmount(Decimal("100"), "USD"),
                    groups=[],
                    estimated=True,
                )
            ],
            time_period=self.time_period,
            total_cost=CostAmount(Decimal("100"), "USD"),
        )

        insights = self.formatter._generate_simple_insights(
            cost_data, self.query_params
        )
        assert any("estimated costs" in insight for insight in insights)

    def test_generate_insights_zero_costs(self):
        """Test insight generation for zero costs."""
        cost_data = CostData(
            results=[],
            time_period=self.time_period,
            total_cost=CostAmount(Decimal("0"), "USD"),
        )

        insights = self.formatter._generate_simple_insights(
            cost_data, self.query_params
        )
        assert any("No costs found" in insight for insight in insights)

    def test_generate_insights_high_costs(self):
        """Test insight generation for high costs."""
        cost_data = CostData(
            results=[
                CostResult(
                    time_period=self.time_period,
                    total=CostAmount(Decimal("1500"), "USD"),
                    groups=[],
                    estimated=False,
                )
            ],
            time_period=self.time_period,
            total_cost=CostAmount(Decimal("1500"), "USD"),
        )

        insights = self.formatter._generate_simple_insights(
            cost_data, self.query_params
        )
        assert any("exceed $1,000" in insight for insight in insights)

    def test_format_response_with_groups(self):
        """Test response formatting with group data."""
        groups = [
            Group(
                keys=["EC2-Instance"],
                metrics={"BlendedCost": CostAmount(Decimal("50.00"), "USD")},
            ),
            Group(
                keys=["EC2-Other"],
                metrics={"BlendedCost": CostAmount(Decimal("73.45"), "USD")},
            ),
        ]

        cost_data = CostData(
            results=[
                CostResult(
                    time_period=self.time_period,
                    total=CostAmount(Decimal("123.45"), "USD"),
                    groups=groups,
                    estimated=False,
                )
            ],
            time_period=self.time_period,
            total_cost=CostAmount(Decimal("123.45"), "USD"),
        )

        response = self.formatter.format_response(
            cost_data, "What did I spend on EC2?", self.query_params
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
            end=datetime(
                2024, 2, 1, tzinfo=timezone.utc
            ),  # Exclusive end date for full month
        )

        self.cost_data = CostData(
            results=[
                CostResult(
                    time_period=self.time_period,
                    total=CostAmount(Decimal("123.45"), "USD"),
                    groups=[],
                    estimated=False,
                )
            ],
            time_period=self.time_period,
            total_cost=CostAmount(Decimal("123.45"), "USD"),
        )

        self.query_params = QueryParameters(
            service="EC2",
            time_period=self.time_period,
            granularity=TimePeriodGranularity.MONTHLY,
        )

    @patch(
        "src.aws_cost_cli.response_formatter.RichResponseFormatter._check_rich_availability"
    )
    def test_format_response_rich_available(self, mock_check_rich):
        """Test response formatting when Rich is available."""
        mock_check_rich.return_value = True

        with patch("rich.console.Console") as mock_console_class:
            mock_console = Mock()
            mock_console_class.return_value = mock_console

            # Mock string IO
            mock_string_io = Mock()
            mock_string_io.getvalue.return_value = "Rich formatted output"

            with patch("io.StringIO", return_value=mock_string_io):
                response = self.formatter.format_response(
                    self.cost_data, "What did I spend on EC2?", self.query_params
                )

            assert response == "Rich formatted output"
            assert mock_console.print.called

    def test_format_response_rich_not_available(self):
        """Test response formatting when Rich is not available."""
        # Create a new formatter instance that will check Rich availability
        formatter = RichResponseFormatter()
        formatter._rich_available = False  # Force Rich to be unavailable

        response = formatter.format_response(
            self.cost_data, "What did I spend on EC2?", self.query_params
        )

        # Should fall back to simple formatter
        assert "AWS Cost Summary for EC2" in response
        assert "$123.45" in response

    def test_check_rich_availability_available(self):
        """Test Rich availability check when Rich is available."""
        with patch("builtins.__import__"):
            result = self.formatter._check_rich_availability()
            # This will depend on whether rich is actually installed
            assert isinstance(result, bool)

    def test_check_rich_availability_not_available(self):
        """Test Rich availability check when Rich is not available."""
        with patch("builtins.__import__", side_effect=ImportError):
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
            end=datetime(
                2024, 2, 1, tzinfo=timezone.utc
            ),  # Exclusive end date for full month
        )

        self.cost_data = CostData(
            results=[
                CostResult(
                    time_period=self.time_period,
                    total=CostAmount(Decimal("123.45"), "USD"),
                    groups=[],
                    estimated=False,
                )
            ],
            time_period=self.time_period,
            total_cost=CostAmount(Decimal("123.45"), "USD"),
        )

        self.query_params = QueryParameters(
            service="EC2",
            time_period=self.time_period,
            granularity=TimePeriodGranularity.MONTHLY,
        )

    def test_format_response_llm_available(self):
        """Test response formatting when LLM is available."""
        # Mock the LLM provider to simulate OpenAI
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = (
            "Your EC2 costs for January 2024 were $123.45."
        )
        mock_client.chat.completions.create.return_value = mock_response

        self.mock_llm_provider._get_client.return_value = mock_client
        self.mock_llm_provider.model = "gpt-3.5-turbo"

        response = self.formatter.format_response(
            self.cost_data, "What did I spend on EC2 last month?", self.query_params
        )

        assert "Your EC2 costs for January 2024 were $123.45." in response

    def test_format_response_llm_not_available(self):
        """Test response formatting when LLM is not available."""
        self.mock_llm_provider.is_available.return_value = False

        response = self.formatter.format_response(
            self.cost_data, "What did I spend on EC2 last month?", self.query_params
        )

        # Should fall back to simple formatter
        assert "AWS Cost Summary for EC2" in response
        assert "$123.45" in response

    def test_format_response_llm_error(self):
        """Test response formatting when LLM throws an error."""
        self.mock_llm_provider._get_client.side_effect = Exception("API Error")

        response = self.formatter.format_response(
            self.cost_data, "What did I spend on EC2 last month?", self.query_params
        )

        # Should fall back to simple formatter
        assert "AWS Cost Summary for EC2" in response
        assert "$123.45" in response

    def test_prepare_cost_summary(self):
        """Test cost summary preparation for LLM."""
        summary = self.formatter._prepare_cost_summary(
            self.cost_data, self.query_params
        )

        assert summary["total_cost"]["amount"] == 123.45
        assert summary["total_cost"]["currency"] == "USD"
        assert summary["service"] == "EC2"
        assert summary["granularity"] == "MONTHLY"
        assert len(summary["results"]) == 1
        assert summary["results"][0]["total"]["amount"] == 123.45

    def test_prepare_cost_summary_with_groups(self):
        """Test cost summary preparation with group data."""
        groups = [
            Group(
                keys=["EC2-Instance"],
                metrics={"BlendedCost": CostAmount(Decimal("50.00"), "USD")},
            )
        ]

        cost_data = CostData(
            results=[
                CostResult(
                    time_period=self.time_period,
                    total=CostAmount(Decimal("123.45"), "USD"),
                    groups=groups,
                    estimated=False,
                )
            ],
            time_period=self.time_period,
            total_cost=CostAmount(Decimal("123.45"), "USD"),
        )

        summary = self.formatter._prepare_cost_summary(cost_data, self.query_params)

        assert len(summary["results"][0]["groups"]) == 1
        assert summary["results"][0]["groups"][0]["keys"] == ["EC2-Instance"]
        assert (
            summary["results"][0]["groups"][0]["metrics"]["BlendedCost"]["amount"]
            == 50.0
        )

    def test_prepare_cost_summary_limits_results(self):
        """Test that cost summary limits results to avoid token limits."""
        # Create 15 results (more than the 10 limit)
        results = []
        for i in range(15):
            results.append(
                CostResult(
                    time_period=self.time_period,
                    total=CostAmount(Decimal("10.00"), "USD"),
                    groups=[],
                    estimated=False,
                )
            )

        cost_data = CostData(
            results=results,
            time_period=self.time_period,
            total_cost=CostAmount(Decimal("150.00"), "USD"),
        )

        summary = self.formatter._prepare_cost_summary(cost_data, self.query_params)

        # Should be limited to 10 results
        assert len(summary["results"]) == 10

    def test_prepare_cost_summary_includes_formatted_dates(self):
        """Test that cost summary includes formatted dates alongside raw dates."""
        summary = self.formatter._prepare_cost_summary(
            self.cost_data, self.query_params
        )

        # Check main time period has formatted date
        assert "formatted" in summary["time_period"]
        # The current test period (Jan 1-31) will be formatted as a custom range
        assert "January" in summary["time_period"]["formatted"]
        assert "2024" in summary["time_period"]["formatted"]
        assert summary["time_period"]["start"] == "2024-01-01"
        assert summary["time_period"]["end"] == "2024-02-01"

        # Check results have formatted dates
        assert len(summary["results"]) == 1
        result = summary["results"][0]
        assert "formatted" in result["period"]
        assert "January" in result["period"]["formatted"]
        assert "2024" in result["period"]["formatted"]
        assert result["period"]["start"] == "2024-01-01"
        assert result["period"]["end"] == "2024-02-01"

    def test_prepare_cost_summary_formatted_dates_single_day(self):
        """Test formatted dates for single day periods."""
        single_day_period = TimePeriod(
            start=datetime(2024, 1, 15, tzinfo=timezone.utc),
            end=datetime(2024, 1, 16, tzinfo=timezone.utc),  # Exclusive end date
        )

        cost_data = CostData(
            results=[
                CostResult(
                    time_period=single_day_period,
                    total=CostAmount(Decimal("50.00"), "USD"),
                    groups=[],
                    estimated=False,
                )
            ],
            time_period=single_day_period,
            total_cost=CostAmount(Decimal("50.00"), "USD"),
        )

        query_params = QueryParameters(
            service="EC2",
            time_period=single_day_period,
            granularity=TimePeriodGranularity.DAILY,
        )

        summary = self.formatter._prepare_cost_summary(cost_data, query_params)

        # Check formatted date for single day
        assert summary["time_period"]["formatted"] == "January 15, 2024"
        assert summary["results"][0]["period"]["formatted"] == "January 15, 2024"

    def test_prepare_cost_summary_formatted_dates_quarter(self):
        """Test formatted dates for quarterly periods."""
        quarter_period = TimePeriod(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 4, 1, tzinfo=timezone.utc),  # Q1 2024
        )

        cost_data = CostData(
            results=[
                CostResult(
                    time_period=quarter_period,
                    total=CostAmount(Decimal("300.00"), "USD"),
                    groups=[],
                    estimated=False,
                )
            ],
            time_period=quarter_period,
            total_cost=CostAmount(Decimal("300.00"), "USD"),
        )

        query_params = QueryParameters(
            service="EC2",
            time_period=quarter_period,
            granularity=TimePeriodGranularity.MONTHLY,
        )

        summary = self.formatter._prepare_cost_summary(cost_data, query_params)

        # Check formatted date for quarter (now correctly detected as quarter)
        assert summary["time_period"]["formatted"] == "Q1 2024"
        assert summary["results"][0]["period"]["formatted"] == "Q1 2024"

    def test_prepare_cost_summary_formatted_dates_with_forecast(self):
        """Test that forecast data includes formatted dates."""
        from src.aws_cost_cli.models import ForecastData

        forecast_period = TimePeriod(
            start=datetime(2024, 2, 1, tzinfo=timezone.utc),
            end=datetime(2024, 3, 1, tzinfo=timezone.utc),
        )

        forecast_data = [
            ForecastData(
                forecast_period=forecast_period,
                forecasted_amount=CostAmount(Decimal("150.00"), "USD"),
                confidence_interval_lower=CostAmount(Decimal("120.00"), "USD"),
                confidence_interval_upper=CostAmount(Decimal("180.00"), "USD"),
                prediction_accuracy=0.85,
            )
        ]

        cost_data = CostData(
            results=[
                CostResult(
                    time_period=self.time_period,
                    total=CostAmount(Decimal("123.45"), "USD"),
                    groups=[],
                    estimated=False,
                )
            ],
            time_period=self.time_period,
            total_cost=CostAmount(Decimal("123.45"), "USD"),
            forecast_data=forecast_data,
        )

        summary = self.formatter._prepare_cost_summary(cost_data, self.query_params)

        # Check forecast has formatted dates
        assert len(summary["forecast"]) == 1
        forecast = summary["forecast"][0]
        assert "formatted" in forecast["period"]
        assert "February" in forecast["period"]["formatted"]
        assert "2024" in forecast["period"]["formatted"]
        assert forecast["period"]["start"] == "2024-02-01"
        assert forecast["period"]["end"] == "2024-03-01"

    def test_system_prompt_mentions_formatted_dates(self):
        """Test that the system prompt instructs LLM to use formatted dates."""
        system_prompt = self.formatter._get_response_system_prompt()

        assert "formatted time periods" in system_prompt
        assert "January 2025" in system_prompt
        assert "Q1 2025" in system_prompt
        assert "formatted periods for better readability" in system_prompt

    def test_llm_response_with_formatted_dates_integration(self):
        """Test integration of formatted dates in LLM responses."""
        # Mock the LLM provider to return a response that uses formatted dates
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = (
            "Your EC2 costs for January 2024 were $123.45. This is a significant "
            "improvement from the previous month."
        )
        mock_client.chat.completions.create.return_value = mock_response

        self.mock_llm_provider._get_client.return_value = mock_client
        self.mock_llm_provider.model = "gpt-3.5-turbo"

        response = self.formatter.format_response(
            self.cost_data, "What did I spend on EC2 last month?", self.query_params
        )

        # Verify the response uses formatted dates
        assert "January 2024" in response
        assert "$123.45" in response

        # Verify that the LLM was called with formatted date information
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        user_message = call_args[1]["messages"][1]["content"]

        # The user message should contain the cost summary with formatted dates
        assert "formatted" in user_message
        assert "January" in user_message


class TestResponseGenerator:
    """Test cases for ResponseGenerator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm_provider = Mock()
        self.mock_llm_provider.is_available.return_value = True

        self.time_period = TimePeriod(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(
                2024, 2, 1, tzinfo=timezone.utc
            ),  # Exclusive end date for full month
        )

        self.cost_data = CostData(
            results=[
                CostResult(
                    time_period=self.time_period,
                    total=CostAmount(Decimal("123.45"), "USD"),
                    groups=[],
                    estimated=False,
                )
            ],
            time_period=self.time_period,
            total_cost=CostAmount(Decimal("123.45"), "USD"),
        )

        self.query_params = QueryParameters(
            service="EC2",
            time_period=self.time_period,
            granularity=TimePeriodGranularity.MONTHLY,
        )

    def test_init_with_llm_provider(self):
        """Test initialization with LLM provider."""
        generator = ResponseGenerator(
            llm_provider=self.mock_llm_provider, output_format="llm"
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
            self.cost_data, "What did I spend on EC2?", self.query_params
        )

        assert "AWS Cost Summary for EC2" in response
        assert "$123.45" in response

    def test_format_response_uses_date_formatting_config(self):
        """Test response generator passes date formatting config to formatters."""
        generator = ResponseGenerator(
            output_format="simple",
            date_formatting_config=DateFormattingConfig(enabled=False),
        )

        response = generator.format_response(
            self.cost_data, "What did I spend on EC2?", self.query_params
        )

        assert "2024-01-01 to 2024-02-01" in response
        assert "January 2024" not in response

    @patch(
        "src.aws_cost_cli.response_formatter.RichResponseFormatter._check_rich_availability"
    )
    def test_format_response_rich(self, mock_check_rich):
        """Test response formatting with rich format."""
        mock_check_rich.return_value = False  # Force fallback to simple

        generator = ResponseGenerator(output_format="rich")

        response = generator.format_response(
            self.cost_data, "What did I spend on EC2?", self.query_params
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
        self.mock_llm_provider.model = "gpt-3.5-turbo"

        generator = ResponseGenerator(
            llm_provider=self.mock_llm_provider, output_format="llm"
        )

        response = generator.format_response(
            self.cost_data, "What did I spend on EC2?", self.query_params
        )

        assert "Your EC2 costs were $123.45." in response

    def test_format_response_error_fallback(self):
        """Test that errors fall back to simple formatter."""
        # Create a generator that will cause an error
        generator = ResponseGenerator(output_format="invalid_format")

        response = generator.format_response(
            self.cost_data, "What did I spend on EC2?", self.query_params
        )

        # Should fall back to simple formatter
        assert "AWS Cost Summary for EC2" in response
        assert "$123.45" in response

    def test_format_response_case_insensitive(self):
        """Test that output format is case insensitive."""
        generator = ResponseGenerator(output_format="SIMPLE")

        response = generator.format_response(
            self.cost_data, "What did I spend on EC2?", self.query_params
        )

        assert "AWS Cost Summary for EC2" in response
        assert "$123.45" in response


if __name__ == "__main__":
    pytest.main([__file__])
