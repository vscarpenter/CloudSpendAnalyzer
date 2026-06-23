"""End-to-end tests for date formatting improvement across the entire CLI workflow."""

import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
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


class TestDateFormattingEndToEnd:
    """End-to-end tests for date formatting across all CLI workflows."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.date_formatter = DateFormatter()

        # Create test configurations for different output formats
        self.simple_config = Config(
            llm_provider="openai",
            llm_config={"provider": "openai", "api_key": "test-key"},
            output_format="simple",
        )

        self.rich_config = Config(
            llm_provider="openai",
            llm_config={"provider": "openai", "api_key": "test-key"},
            output_format="rich",
        )

        self.llm_config = Config(
            llm_provider="openai",
            llm_config={"provider": "openai", "api_key": "test-key"},
            output_format="llm",
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
            "multi_month": TimePeriod(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 3, 1, tzinfo=timezone.utc),
            ),
            "cross_year": TimePeriod(
                start=datetime(2024, 12, 1, tzinfo=timezone.utc),
                end=datetime(2025, 2, 1, tzinfo=timezone.utc),
            ),
            "custom_range": TimePeriod(
                start=datetime(2024, 1, 15, tzinfo=timezone.utc),
                end=datetime(2024, 2, 10, tzinfo=timezone.utc),
            ),
        }

        # Expected formatted outputs for each period type
        self.expected_formats = {
            "single_day": "January 15, 2024",
            "single_month": "January 2024",
            "single_quarter": "Q1 2024",
            "single_year": "2024",
            "multi_month": "January 2024 - February 2024",
            "cross_year": "December 2024 - January 2025",
            "custom_range": "January 15, 2024 - February 10, 2024",
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
    def test_end_to_end_simple_format_all_period_types(
        self,
        mock_pipeline,
        mock_credential_manager,
        mock_config_manager,
    ):
        """Test end-to-end flow with simple format for all period types."""
        mock_config_manager.return_value.load_config.return_value = self.simple_config
        mock_credential_manager.return_value.validate_credentials.return_value = True

        for period_name, time_period in self.test_periods.items():
            with self.subTest(period_type=period_name):
                # Setup test data
                cost_data = self.create_cost_data(time_period)
                query_params = self.create_query_params(time_period)

                # Create a real SimpleResponseFormatter to test actual formatting
                formatter = SimpleResponseFormatter()
                formatted_response = formatter.format_response(
                    cost_data, f"What did I spend on EC2 in {period_name}?", query_params
                )

                # Mock pipeline response
                mock_pipeline_instance = Mock()
                mock_pipeline_instance.process_query.return_value = Mock(
                    cost_data=cost_data,
                    formatted_response=formatted_response
                )
                mock_pipeline.return_value = mock_pipeline_instance

                # Execute CLI command
                result = self.runner.invoke(cli, ["query", f"EC2 costs for {period_name}"])

                # Verify success
                assert result.exit_code == 0, f"CLI failed for {period_name}: {result.output}"

                # Verify the formatted date appears in the output
                expected_format = self.expected_formats[period_name]
                assert expected_format in result.output, (
                    f"Expected '{expected_format}' not found in output for {period_name}. "
                    f"Output: {result.output}"
                )

    @patch("src.aws_cost_cli.cli.ConfigManager")
    @patch("src.aws_cost_cli.cli.CredentialManager")
    @patch("src.aws_cost_cli.cli.AWSCostClient")
    @patch("src.aws_cost_cli.cli.CacheManager")
    @patch("src.aws_cost_cli.cli.QueryParser")
    @patch("src.aws_cost_cli.cli.ResponseGenerator")
    def test_end_to_end_rich_format_consistency(
        self,
        mock_response_gen,
        mock_query_parser,
        mock_cache_manager,
        mock_aws_client,
        mock_credential_manager,
        mock_config_manager,
    ):
        """Test end-to-end flow with rich format maintains date formatting consistency."""
        mock_config_manager.return_value.load_config.return_value = self.rich_config
        mock_credential_manager.return_value.validate_credentials.return_value = True
        mock_aws_client.return_value.validate_permissions.return_value = True
        mock_cache_manager.return_value.get_cached_data.return_value = None
        mock_cache_manager.return_value.generate_cache_key.return_value = "test-key"

        # Test with a quarterly period
        time_period = self.test_periods["single_quarter"]
        cost_data = self.create_cost_data(time_period)
        query_params = self.create_query_params(time_period)

        mock_aws_client.return_value.get_cost_and_usage.return_value = cost_data
        mock_query_parser.return_value.parse_query.return_value = query_params
        mock_query_parser.return_value.validate_parameters.return_value = True

        # Create a real RichResponseFormatter to test actual formatting
        formatter = RichResponseFormatter()
        formatted_response = formatter.format_response(
            cost_data, "What did I spend on EC2 in Q1?", query_params
        )
        mock_response_gen.return_value.format_response.return_value = formatted_response

        # Execute CLI command
        result = self.runner.invoke(cli, ["query", "EC2 costs for Q1"])

        # Verify success
        assert result.exit_code == 0, f"CLI failed: {result.output}"

        # Verify the formatted date appears in the output (Q1 2024)
        expected_format = self.expected_formats["single_quarter"]
        # Rich format might include ANSI codes, so check if the text is present
        assert expected_format in result.output or "Q1 2024" in result.output

    @patch("src.aws_cost_cli.cli.ConfigManager")
    @patch("src.aws_cost_cli.cli.CredentialManager")
    @patch("src.aws_cost_cli.cli.AWSCostClient")
    @patch("src.aws_cost_cli.cli.CacheManager")
    @patch("src.aws_cost_cli.cli.QueryParser")
    @patch("src.aws_cost_cli.cli.ResponseGenerator")
    def test_end_to_end_llm_format_consistency(
        self,
        mock_response_gen,
        mock_query_parser,
        mock_cache_manager,
        mock_aws_client,
        mock_credential_manager,
        mock_config_manager,
    ):
        """Test end-to-end flow with LLM format uses formatted dates."""
        mock_config_manager.return_value.load_config.return_value = self.llm_config
        mock_credential_manager.return_value.validate_credentials.return_value = True
        mock_aws_client.return_value.validate_permissions.return_value = True
        mock_cache_manager.return_value.get_cached_data.return_value = None
        mock_cache_manager.return_value.generate_cache_key.return_value = "test-key"

        # Test with a monthly period
        time_period = self.test_periods["single_month"]
        cost_data = self.create_cost_data(time_period)
        query_params = self.create_query_params(time_period)

        mock_aws_client.return_value.get_cost_and_usage.return_value = cost_data
        mock_query_parser.return_value.parse_query.return_value = query_params
        mock_query_parser.return_value.validate_parameters.return_value = True

        # Create a real LLMResponseFormatter to test actual formatting
        formatter = LLMResponseFormatter()
        formatted_response = formatter.format_response(
            cost_data, "What did I spend on EC2 in January?", query_params
        )
        mock_response_gen.return_value.format_response.return_value = formatted_response

        # Execute CLI command
        result = self.runner.invoke(cli, ["query", "EC2 costs for January"])

        # Verify success
        assert result.exit_code == 0, f"CLI failed: {result.output}"

        # Verify the formatted date appears in the output
        expected_format = self.expected_formats["single_month"]
        assert expected_format in result.output

    @patch("src.aws_cost_cli.cli.ConfigManager")
    @patch("src.aws_cost_cli.cli.CredentialManager")
    @patch("src.aws_cost_cli.cli.AWSCostClient")
    @patch("src.aws_cost_cli.cli.CacheManager")
    @patch("src.aws_cost_cli.cli.QueryParser")
    @patch("src.aws_cost_cli.cli.ResponseGenerator")
    def test_end_to_end_json_output_includes_formatted_dates(
        self,
        mock_response_gen,
        mock_query_parser,
        mock_cache_manager,
        mock_aws_client,
        mock_credential_manager,
        mock_config_manager,
    ):
        """Test that JSON output includes both raw and formatted dates."""
        json_config = Config(
            llm_provider="openai",
            llm_config={"provider": "openai", "api_key": "test-key"},
            output_format="json",
        )

        mock_config_manager.return_value.load_config.return_value = json_config
        mock_credential_manager.return_value.validate_credentials.return_value = True
        mock_aws_client.return_value.validate_permissions.return_value = True
        mock_cache_manager.return_value.get_cached_data.return_value = None
        mock_cache_manager.return_value.generate_cache_key.return_value = "test-key"

        # Test with a yearly period
        time_period = self.test_periods["single_year"]
        cost_data = self.create_cost_data(time_period)
        query_params = self.create_query_params(time_period)

        mock_aws_client.return_value.get_cost_and_usage.return_value = cost_data
        mock_query_parser.return_value.parse_query.return_value = query_params
        mock_query_parser.return_value.validate_parameters.return_value = True

        # Execute CLI command
        result = self.runner.invoke(cli, ["query", "EC2 costs for 2024"])

        # Verify success
        assert result.exit_code == 0, f"CLI failed: {result.output}"

        # Parse JSON output
        output_data = json.loads(result.output)

        # Verify JSON structure includes formatted dates
        assert "time_period" in output_data
        assert "formatted_time_period" in output_data
        assert output_data["formatted_time_period"] == self.expected_formats["single_year"]

        # Verify results also include formatted dates
        if "results" in output_data and output_data["results"]:
            for result_item in output_data["results"]:
                assert "formatted_time_period" in result_item

    @patch("src.aws_cost_cli.cli.ConfigManager")
    @patch("src.aws_cost_cli.cli.CredentialManager")
    @patch("src.aws_cost_cli.cli.AWSCostClient")
    @patch("src.aws_cost_cli.cli.CacheManager")
    @patch("src.aws_cost_cli.cli.QueryParser")
    @patch("src.aws_cost_cli.cli.ResponseGenerator")
    def test_end_to_end_cached_data_formatting(
        self,
        mock_response_gen,
        mock_query_parser,
        mock_cache_manager,
        mock_aws_client,
        mock_credential_manager,
        mock_config_manager,
    ):
        """Test that cached data works correctly with new date formatting."""
        mock_config_manager.return_value.load_config.return_value = self.simple_config
        mock_credential_manager.return_value.validate_credentials.return_value = True
        mock_aws_client.return_value.validate_permissions.return_value = True

        # Test with cached data
        time_period = self.test_periods["single_month"]
        cached_cost_data = self.create_cost_data(time_period)
        query_params = self.create_query_params(time_period)

        mock_cache_manager.return_value.get_cached_data.return_value = cached_cost_data
        mock_cache_manager.return_value.generate_cache_key.return_value = "test-key"
        mock_query_parser.return_value.parse_query.return_value = query_params
        mock_query_parser.return_value.validate_parameters.return_value = True

        # Create a real SimpleResponseFormatter to test actual formatting
        formatter = SimpleResponseFormatter()
        formatted_response = formatter.format_response(
            cached_cost_data, "What did I spend on EC2 in January?", query_params
        )
        mock_response_gen.return_value.format_response.return_value = formatted_response

        # Execute CLI command
        result = self.runner.invoke(cli, ["query", "EC2 costs for January"])

        # Verify success
        assert result.exit_code == 0, f"CLI failed: {result.output}"

        # Verify cached data message appears
        assert "Using cached data" in result.output

        # Verify the formatted date appears in the output
        expected_format = self.expected_formats["single_month"]
        assert expected_format in result.output

        # Verify AWS client was not called (using cache)
        mock_aws_client.return_value.get_cost_and_usage.assert_not_called()

    def test_edge_case_unusual_time_periods(self):
        """Test edge cases with unusual time periods."""
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

    @patch("src.aws_cost_cli.cli.ConfigManager")
    @patch("src.aws_cost_cli.cli.CredentialManager")
    @patch("src.aws_cost_cli.cli.AWSCostClient")
    @patch("src.aws_cost_cli.cli.CacheManager")
    @patch("src.aws_cost_cli.cli.QueryParser")
    @patch("src.aws_cost_cli.cli.ResponseGenerator")
    def test_end_to_end_multiple_results_formatting(
        self,
        mock_response_gen,
        mock_query_parser,
        mock_cache_manager,
        mock_aws_client,
        mock_credential_manager,
        mock_config_manager,
    ):
        """Test end-to-end flow with multiple results having different time periods."""
        mock_config_manager.return_value.load_config.return_value = self.simple_config
        mock_credential_manager.return_value.validate_credentials.return_value = True
        mock_aws_client.return_value.validate_permissions.return_value = True
        mock_cache_manager.return_value.get_cached_data.return_value = None
        mock_cache_manager.return_value.generate_cache_key.return_value = "test-key"

        # Create multiple results with different time periods
        results = []
        for i, (period_name, time_period) in enumerate(list(self.test_periods.items())[:3]):
            results.append(
                CostResult(
                    time_period=time_period,
                    total=CostAmount(Decimal(str(100.0 + i * 50)), "USD"),
                    groups=[],
                    estimated=False,
                )
            )

        # Overall time period spans all results
        overall_period = TimePeriod(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )

        cost_data = CostData(
            results=results,
            time_period=overall_period,
            total_cost=CostAmount(Decimal("300.00"), "USD"),
            currency="USD",
        )

        query_params = self.create_query_params(overall_period)

        mock_aws_client.return_value.get_cost_and_usage.return_value = cost_data
        mock_query_parser.return_value.parse_query.return_value = query_params
        mock_query_parser.return_value.validate_parameters.return_value = True

        # Create a real SimpleResponseFormatter to test actual formatting
        formatter = SimpleResponseFormatter()
        formatted_response = formatter.format_response(
            cost_data, "What did I spend on EC2 over time?", query_params
        )
        mock_response_gen.return_value.format_response.return_value = formatted_response

        # Execute CLI command
        result = self.runner.invoke(cli, ["query", "EC2 costs breakdown"])

        # Verify success
        assert result.exit_code == 0, f"CLI failed: {result.output}"

        # Verify that multiple formatted dates appear in the output
        for period_name in list(self.expected_formats.keys())[:3]:
            expected_format = self.expected_formats[period_name]
            assert expected_format in result.output, (
                f"Expected '{expected_format}' not found in output. "
                f"Output: {result.output}"
            )

    @patch("src.aws_cost_cli.cli.ConfigManager")
    @patch("src.aws_cost_cli.cli.CredentialManager")
    @patch("src.aws_cost_cli.cli.AWSCostClient")
    @patch("src.aws_cost_cli.cli.CacheManager")
    @patch("src.aws_cost_cli.cli.QueryParser")
    @patch("src.aws_cost_cli.cli.ResponseGenerator")
    def test_end_to_end_format_override_consistency(
        self,
        mock_response_gen,
        mock_query_parser,
        mock_cache_manager,
        mock_aws_client,
        mock_credential_manager,
        mock_config_manager,
    ):
        """Test that format override maintains date formatting consistency."""
        # Start with simple config but override to rich
        mock_config_manager.return_value.load_config.return_value = self.simple_config
        mock_credential_manager.return_value.validate_credentials.return_value = True
        mock_aws_client.return_value.validate_permissions.return_value = True
        mock_cache_manager.return_value.get_cached_data.return_value = None
        mock_cache_manager.return_value.generate_cache_key.return_value = "test-key"

        time_period = self.test_periods["single_quarter"]
        cost_data = self.create_cost_data(time_period)
        query_params = self.create_query_params(time_period)

        mock_aws_client.return_value.get_cost_and_usage.return_value = cost_data
        mock_query_parser.return_value.parse_query.return_value = query_params
        mock_query_parser.return_value.validate_parameters.return_value = True

        # Create a real RichResponseFormatter to test actual formatting
        formatter = RichResponseFormatter()
        formatted_response = formatter.format_response(
            cost_data, "What did I spend on EC2 in Q1?", query_params
        )
        mock_response_gen.return_value.format_response.return_value = formatted_response

        # Execute CLI command with format override
        result = self.runner.invoke(cli, ["query", "EC2 costs for Q1", "--format", "rich"])

        # Verify success
        assert result.exit_code == 0, f"CLI failed: {result.output}"

        # Verify the formatted date appears in the output
        expected_format = self.expected_formats["single_quarter"]
        assert expected_format in result.output or "Q1 2024" in result.output

    def test_date_formatter_fallback_behavior(self):
        """Test that date formatter handles edge cases gracefully."""
        # Test with invalid period (end before start)
        invalid_period = TimePeriod(
            start=datetime(2024, 1, 15, tzinfo=timezone.utc),
            end=datetime(2024, 1, 10, tzinfo=timezone.utc),
        )

        # Should not crash and should return fallback format
        result = self.date_formatter.safe_format_time_period(invalid_period)
        assert isinstance(result, str)
        assert len(result) > 0
        # Should fall back to ISO format
        assert "2024-01-15" in result and "2024-01-10" in result

    @patch("src.aws_cost_cli.cli.ConfigManager")
    @patch("src.aws_cost_cli.cli.CredentialManager")
    @patch("src.aws_cost_cli.cli.AWSCostClient")
    @patch("src.aws_cost_cli.cli.CacheManager")
    @patch("src.aws_cost_cli.cli.QueryParser")
    @patch("src.aws_cost_cli.cli.ResponseGenerator")
    def test_end_to_end_fresh_flag_with_formatting(
        self,
        mock_response_gen,
        mock_query_parser,
        mock_cache_manager,
        mock_aws_client,
        mock_credential_manager,
        mock_config_manager,
    ):
        """Test that fresh flag bypasses cache but maintains date formatting."""
        mock_config_manager.return_value.load_config.return_value = self.simple_config
        mock_credential_manager.return_value.validate_credentials.return_value = True
        mock_aws_client.return_value.validate_permissions.return_value = True
        mock_cache_manager.return_value.generate_cache_key.return_value = "test-key"

        time_period = self.test_periods["single_month"]
        cost_data = self.create_cost_data(time_period)
        query_params = self.create_query_params(time_period)

        mock_aws_client.return_value.get_cost_and_usage.return_value = cost_data
        mock_query_parser.return_value.parse_query.return_value = query_params
        mock_query_parser.return_value.validate_parameters.return_value = True

        # Create a real SimpleResponseFormatter to test actual formatting
        formatter = SimpleResponseFormatter()
        formatted_response = formatter.format_response(
            cost_data, "What did I spend on EC2 in January?", query_params
        )
        mock_response_gen.return_value.format_response.return_value = formatted_response

        # Execute CLI command with fresh flag
        result = self.runner.invoke(cli, ["query", "EC2 costs for January", "--fresh"])

        # Verify success
        assert result.exit_code == 0, f"CLI failed: {result.output}"

        # Verify cache was not checked
        mock_cache_manager.return_value.get_cached_data.assert_not_called()

        # Verify AWS client was called
        mock_aws_client.return_value.get_cost_and_usage.assert_called_once()

        # Verify the formatted date appears in the output
        expected_format = self.expected_formats["single_month"]
        assert expected_format in result.output

    def test_cross_format_consistency(self):
        """Test that all formatters produce consistent date formatting."""
        time_period = self.test_periods["single_quarter"]
        cost_data = self.create_cost_data(time_period)
        query_params = self.create_query_params(time_period)

        # Test all formatters
        simple_formatter = SimpleResponseFormatter()
        rich_formatter = RichResponseFormatter()
        llm_formatter = LLMResponseFormatter()

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

    def test_performance_with_large_datasets(self):
        """Test date formatting performance with large datasets."""
        import time

        # Create a large dataset with many time periods
        large_results = []
        for i in range(100):
            period = TimePeriod(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=i),
                end=datetime(2024, 1, 2, tzinfo=timezone.utc) + timedelta(days=i),
            )
            large_results.append(
                CostResult(
                    time_period=period,
                    total=CostAmount(Decimal("10.00"), "USD"),
                    groups=[],
                    estimated=False,
                )
            )

        large_cost_data = CostData(
            results=large_results,
            time_period=TimePeriod(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 4, 10, tzinfo=timezone.utc),
            ),
            total_cost=CostAmount(Decimal("1000.00"), "USD"),
            currency="USD",
        )

        query_params = QueryParameters(
            service="EC2",
            time_period=large_cost_data.time_period,
            granularity=TimePeriodGranularity.DAILY,
        )

        # Test formatting performance
        formatter = SimpleResponseFormatter()
        start_time = time.time()
        response = formatter.format_response(
            large_cost_data, "Daily EC2 costs", query_params
        )
        end_time = time.time()

        # Should complete within reasonable time (less than 1 second)
        assert end_time - start_time < 1.0, "Date formatting took too long"
        assert isinstance(response, str)
        assert len(response) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])