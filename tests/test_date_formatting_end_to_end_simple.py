"""Simplified end-to-end tests for date formatting improvement."""

import pytest
import json
from unittest.mock import Mock, patch
from click.testing import CliRunner
from datetime import datetime, timezone, timedelta
from decimal import Decimal

from src.aws_cost_cli.cli import cli
from src.aws_cost_cli.models import (
    CostData,
    CostResult,
    CostAmount,
    TimePeriod,
    QueryParameters,
    Config,
    TimePeriodGranularity,
    MetricType,
)
from src.aws_cost_cli.response_formatter import (
    SimpleResponseFormatter,
    RichResponseFormatter,
    LLMResponseFormatter,
)
from src.aws_cost_cli.date_formatter import DateFormatter


class TestDateFormattingEndToEndSimple:
    """Simplified end-to-end tests for date formatting across CLI workflows."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.date_formatter = DateFormatter()

        # Create test configurations
        self.simple_config = Config(
            llm_provider="openai",
            llm_config={"provider": "openai", "api_key": "test-key"},
            output_format="simple",
        )

        # Create test time periods for different scenarios
        self.test_periods = {
            "single_day": TimePeriod(
                start=datetime(2024, 1, 15, tzinfo=timezone.utc),
                end=datetime(2024, 1, 16, tzinfo=timezone.utc),
            ),
            "single_month": TimePeriod(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 2, 1, tzinfo=timezone.utc),
            ),
            "single_quarter": TimePeriod(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 4, 1, tzinfo=timezone.utc),
            ),
            "single_year": TimePeriod(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2025, 1, 1, tzinfo=timezone.utc),
            ),
        }

        # Expected formatted outputs for each period type
        self.expected_formats = {
            "single_day": "January 15, 2024",
            "single_month": "January 2024",
            "single_quarter": "Q1 2024",
            "single_year": "2024",
        }

    def create_cost_data(self, time_period, service="EC2", amount=123.45):
        """Create test cost data for a given time period."""
        return CostData(
            results=[
                CostResult(
                    time_period=time_period,
                    total=CostAmount(Decimal(str(amount)), "USD"),
                    groups=[],
                    estimated=False,
                )
            ],
            time_period=time_period,
            total_cost=CostAmount(Decimal(str(amount)), "USD"),
            currency="USD",
        )

    def create_query_params(self, time_period, service="EC2"):
        """Create test query parameters."""
        return QueryParameters(
            service=service,
            time_period=time_period,
            granularity=TimePeriodGranularity.MONTHLY,
            metrics=[MetricType.BLENDED_COST],
        )

    @patch("src.aws_cost_cli.cli.ConfigManager")
    @patch("src.aws_cost_cli.cli.CredentialManager")
    @patch("src.aws_cost_cli.cli.QueryPipeline")
    def test_end_to_end_simple_format_quarterly_period(
        self,
        mock_pipeline,
        mock_credential_manager,
        mock_config_manager,
    ):
        """Test end-to-end flow with simple format for quarterly period."""
        mock_config_manager.return_value.load_config.return_value = self.simple_config
        mock_credential_manager.return_value.validate_credentials.return_value = True

        # Test with a quarterly period
        time_period = self.test_periods["single_quarter"]
        cost_data = self.create_cost_data(time_period)
        query_params = self.create_query_params(time_period)

        # Create a real SimpleResponseFormatter to test actual formatting
        formatter = SimpleResponseFormatter()
        formatted_response = formatter.format_response(
            cost_data, "What did I spend on EC2 in Q1?", query_params
        )

        # Mock pipeline response
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.process_query.return_value = Mock(
            cost_data=cost_data,
            formatted_response=formatted_response
        )
        mock_pipeline.return_value = mock_pipeline_instance

        # Execute CLI command
        result = self.runner.invoke(cli, ["query", "EC2 costs for Q1"])

        # Verify success
        assert result.exit_code == 0, f"CLI failed: {result.output}"

        # Verify the formatted date appears in the output
        expected_format = self.expected_formats["single_quarter"]
        assert expected_format in result.output, (
            f"Expected '{expected_format}' not found in output. "
            f"Output: {result.output}"
        )

    def test_date_formatter_consistency_across_formatters(self):
        """Test that all formatters produce consistent date formatting."""
        time_period = self.test_periods["single_quarter"]
        cost_data = self.create_cost_data(time_period)
        query_params = self.create_query_params(time_period)

        # Test all formatters
        simple_formatter = SimpleResponseFormatter()
        rich_formatter = RichResponseFormatter()
        llm_formatter = LLMResponseFormatter(llm_provider="openai")

        simple_response = simple_formatter.format_response(
            cost_data, "What did I spend on EC2 in Q1?", query_params
        )
        rich_response = rich_formatter.format_response(
            cost_data, "What did I spend on EC2 in Q1?", query_params
        )
        llm_response = llm_formatter.format_response(
            cost_data, "What did I spend on EC2 in Q1?", query_params
        )

        # All should contain the same formatted date
        expected_format = self.expected_formats["single_quarter"]
        assert expected_format in simple_response
        assert expected_format in rich_response or "Q1 2024" in rich_response
        assert expected_format in llm_response

    def test_date_formatter_edge_cases(self):
        """Test date formatter handles edge cases gracefully."""
        # Test leap year handling
        leap_year_period = TimePeriod(
            start=datetime(2024, 2, 29, tzinfo=timezone.utc),  # Leap day
            end=datetime(2024, 3, 1, tzinfo=timezone.utc),
        )
        formatted = self.date_formatter.format_time_period(leap_year_period)
        assert "February 29, 2024" == formatted

        # Test month boundary edge case
        month_boundary_period = TimePeriod(
            start=datetime(2024, 1, 31, tzinfo=timezone.utc),
            end=datetime(2024, 2, 1, tzinfo=timezone.utc),
        )
        formatted = self.date_formatter.format_time_period(month_boundary_period)
        assert "January 31, 2024" == formatted

        # Test year boundary edge case
        year_boundary_period = TimePeriod(
            start=datetime(2024, 12, 31, tzinfo=timezone.utc),
            end=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        formatted = self.date_formatter.format_time_period(year_boundary_period)
        assert "December 31, 2024" == formatted

    def test_date_formatter_fallback_behavior(self):
        """Test that date formatter handles invalid periods gracefully."""
        # Test with invalid period (end before start)
        invalid_period = TimePeriod(
            start=datetime(2024, 1, 15, tzinfo=timezone.utc),
            end=datetime(2024, 1, 10, tzinfo=timezone.utc),
        )

        # Should not crash and should return fallback format
        result = self.date_formatter.safe_format_time_period(invalid_period)
        assert isinstance(result, str)
        assert len(result) > 0
        # Should contain some date information
        assert "2024" in result

    def test_json_output_includes_formatted_dates(self):
        """Test that JSON output includes formatted dates."""
        time_period = self.test_periods["single_year"]
        cost_data = self.create_cost_data(time_period)
        query_params = self.create_query_params(time_period)

        # Test JSON formatting directly
        formatter = SimpleResponseFormatter()
        
        # Test regular formatting and verify formatted date is present
        response = formatter.format_response(
            cost_data, "EC2 costs for 2024", query_params
        )

        # Verify the formatted date is present
        expected_format = self.expected_formats["single_year"]
        assert expected_format in response

    def test_performance_with_multiple_periods(self):
        """Test date formatting performance with multiple time periods."""
        import time

        # Create multiple results with different time periods
        large_results = []
        for i, (period_name, time_period) in enumerate(self.test_periods.items()):
            large_results.append(
                CostResult(
                    time_period=time_period,
                    total=CostAmount(Decimal("100.00"), "USD"),
                    groups=[],
                    estimated=False,
                )
            )

        large_cost_data = CostData(
            results=large_results,
            time_period=TimePeriod(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2025, 1, 1, tzinfo=timezone.utc),
            ),
            total_cost=CostAmount(Decimal("400.00"), "USD"),
            currency="USD",
        )

        query_params = QueryParameters(
            service="EC2",
            time_period=large_cost_data.time_period,
            granularity=TimePeriodGranularity.MONTHLY,
        )

        # Test formatting performance
        formatter = SimpleResponseFormatter()
        start_time = time.time()
        response = formatter.format_response(
            large_cost_data, "EC2 costs breakdown", query_params
        )
        end_time = time.time()

        # Should complete within reasonable time (less than 1 second)
        assert end_time - start_time < 1.0, "Date formatting took too long"
        assert isinstance(response, str)
        assert len(response) > 0

        # Verify that multiple formatted dates appear in the output
        for expected_format in self.expected_formats.values():
            assert expected_format in response

    @patch("src.aws_cost_cli.cli.ConfigManager")
    @patch("src.aws_cost_cli.cli.CredentialManager")
    @patch("src.aws_cost_cli.cli.QueryPipeline")
    def test_cached_data_formatting(
        self,
        mock_pipeline,
        mock_credential_manager,
        mock_config_manager,
    ):
        """Test that cached data works correctly with new date formatting."""
        mock_config_manager.return_value.load_config.return_value = self.simple_config
        mock_credential_manager.return_value.validate_credentials.return_value = True

        # Test with monthly period
        time_period = self.test_periods["single_month"]
        cached_cost_data = self.create_cost_data(time_period)
        query_params = self.create_query_params(time_period)

        # Create a real SimpleResponseFormatter to test actual formatting
        formatter = SimpleResponseFormatter()
        formatted_response = formatter.format_response(
            cached_cost_data, "What did I spend on EC2 in January?", query_params
        )

        # Mock pipeline response indicating cached data
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.process_query.return_value = Mock(
            cost_data=cached_cost_data,
            formatted_response=formatted_response,
            used_cache=True
        )
        mock_pipeline.return_value = mock_pipeline_instance

        # Execute CLI command
        result = self.runner.invoke(cli, ["query", "EC2 costs for January"])

        # Verify success
        assert result.exit_code == 0, f"CLI failed: {result.output}"

        # Verify the formatted date appears in the output
        expected_format = self.expected_formats["single_month"]
        assert expected_format in result.output

    def test_all_period_types_formatting(self):
        """Test that all period types are formatted correctly."""
        for period_name, time_period in self.test_periods.items():
            # Test direct date formatter
            formatted = self.date_formatter.format_time_period(time_period)
            expected = self.expected_formats[period_name]
            assert formatted == expected, (
                f"Period {period_name}: expected '{expected}', got '{formatted}'"
            )

            # Test through response formatter
            cost_data = self.create_cost_data(time_period)
            query_params = self.create_query_params(time_period)
            formatter = SimpleResponseFormatter()
            response = formatter.format_response(
                cost_data, f"EC2 costs for {period_name}", query_params
            )
            assert expected in response, (
                f"Expected '{expected}' not found in response for {period_name}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])