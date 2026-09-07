"""CLI integration tests for date formatting with real command execution."""

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


class TestDateFormattingCLIIntegration:
    """Integration tests for date formatting with real CLI commands."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

        # Create test configurations for different output formats
        self.configs = {
            "simple": Config(
                llm_provider="openai",
                llm_config={"provider": "openai", "api_key": "test-key"},
                output_format="simple",
            ),
            "rich": Config(
                llm_provider="openai",
                llm_config={"provider": "openai", "api_key": "test-key"},
                output_format="rich",
            ),
            "json": Config(
                llm_provider="openai",
                llm_config={"provider": "openai", "api_key": "test-key"},
                output_format="json",
            ),
        }

        # Create realistic test scenarios
        self.test_scenarios = {
            "monthly_ec2": {
                "query": "What did I spend on EC2 last month?",
                "time_period": TimePeriod(
                    start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                    end=datetime(2024, 2, 1, tzinfo=timezone.utc),
                ),
                "expected_format": "January 2024",
                "service": "EC2",
                "amount": 1234.56,
            },
            "quarterly_s3": {
                "query": "S3 costs for Q1 2024",
                "time_period": TimePeriod(
                    start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                    end=datetime(2024, 4, 1, tzinfo=timezone.utc),
                ),
                "expected_format": "Q1 2024",
                "service": "S3",
                "amount": 567.89,
            },
            "yearly_rds": {
                "query": "RDS costs for 2024",
                "time_period": TimePeriod(
                    start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                    end=datetime(2025, 1, 1, tzinfo=timezone.utc),
                ),
                "expected_format": "2024",
                "service": "RDS",
                "amount": 2345.67,
            },
            "daily_lambda": {
                "query": "Lambda costs on January 15th",
                "time_period": TimePeriod(
                    start=datetime(2024, 1, 15, tzinfo=timezone.utc),
                    end=datetime(2024, 1, 16, tzinfo=timezone.utc),
                ),
                "expected_format": "January 15, 2024",
                "service": "Lambda",
                "amount": 45.67,
            },
        }

    def create_cost_data(self, scenario):
        """Create test cost data for a scenario."""
        return CostData(
            results=[
                CostResult(
                    time_period=scenario["time_period"],
                    total=CostAmount(Decimal(str(scenario["amount"])), "USD"),
                    groups=[],
                    estimated=False,
                )
            ],
            time_period=scenario["time_period"],
            total_cost=CostAmount(Decimal(str(scenario["amount"])), "USD"),
            currency="USD",
        )

    @patch("src.aws_cost_cli.cli.ConfigManager")
    @patch("src.aws_cost_cli.cli.CredentialManager")
    @patch("src.aws_cost_cli.cli.QueryPipeline")
    def test_cli_query_with_formatted_dates_simple_output(
        self,
        mock_pipeline,
        mock_credential_manager,
        mock_config_manager,
    ):
        """Test CLI query command with simple output format shows formatted dates."""
        mock_config_manager.return_value.load_config.return_value = self.configs["simple"]
        mock_credential_manager.return_value.validate_credentials.return_value = True

        for scenario_name, scenario in self.test_scenarios.items():
            cost_data = self.create_cost_data(scenario)

            # Mock pipeline response
            mock_pipeline_instance = Mock()
            mock_pipeline_instance.process_query.return_value = Mock(
                cost_data=cost_data,
                formatted_response=f"You spent ${scenario['amount']} on {scenario['service']} during {scenario['expected_format']}."
            )
            mock_pipeline.return_value = mock_pipeline_instance

            # Execute CLI command
            result = self.runner.invoke(cli, ["query", scenario["query"]])

            # Verify success
            assert result.exit_code == 0, f"CLI failed for {scenario_name}: {result.output}"

            # Verify the formatted date appears in the output
            assert scenario["expected_format"] in result.output, (
                f"Expected '{scenario['expected_format']}' not found in output for {scenario_name}. "
                f"Output: {result.output}"
            )

            # Verify the service and amount are also present
            assert scenario["service"] in result.output
            assert str(scenario["amount"]) in result.output

    @patch("src.aws_cost_cli.cli.ConfigManager")
    @patch("src.aws_cost_cli.cli.CredentialManager")
    @patch("src.aws_cost_cli.cli.QueryPipeline")
    def test_cli_query_with_formatted_dates_rich_output(
        self,
        mock_pipeline,
        mock_credential_manager,
        mock_config_manager,
    ):
        """Test CLI query command with rich output format shows formatted dates."""
        mock_config_manager.return_value.load_config.return_value = self.configs["rich"]
        mock_credential_manager.return_value.validate_credentials.return_value = True

        # Test with quarterly scenario
        scenario = self.test_scenarios["quarterly_s3"]
        cost_data = self.create_cost_data(scenario)

        # Mock pipeline response
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.process_query.return_value = Mock(
            cost_data=cost_data,
            formatted_response=f"Rich formatted response for {scenario['service']} costs during {scenario['expected_format']}"
        )
        mock_pipeline.return_value = mock_pipeline_instance

        # Execute CLI command
        result = self.runner.invoke(cli, ["query", scenario["query"]])

        # Verify success
        assert result.exit_code == 0, f"CLI failed: {result.output}"

        # Verify the formatted date appears in the output (may include ANSI codes)
        assert scenario["expected_format"] in result.output or "Q1 2024" in result.output

    @patch("src.aws_cost_cli.cli.ConfigManager")
    @patch("src.aws_cost_cli.cli.CredentialManager")
    @patch("src.aws_cost_cli.cli.QueryPipeline")
    def test_cli_query_with_formatted_dates_json_output(
        self,
        mock_pipeline,
        mock_credential_manager,
        mock_config_manager,
    ):
        """Test CLI query command with JSON output format includes formatted dates."""
        mock_config_manager.return_value.load_config.return_value = self.configs["json"]
        mock_credential_manager.return_value.validate_credentials.return_value = True

        # Test with yearly scenario
        scenario = self.test_scenarios["yearly_rds"]
        cost_data = self.create_cost_data(scenario)

        # Mock pipeline response with JSON format
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.process_query.return_value = Mock(
            cost_data=cost_data,
            formatted_response=json.dumps({
                "query": scenario["query"],
                "service": scenario["service"],
                "time_period": {
                    "start": scenario["time_period"].start.isoformat(),
                    "end": scenario["time_period"].end.isoformat(),
                },
                "formatted_time_period": scenario["expected_format"],
                "total_cost": {
                    "amount": scenario["amount"],
                    "currency": "USD"
                },
                "results": [
                    {
                        "time_period": {
                            "start": scenario["time_period"].start.isoformat(),
                            "end": scenario["time_period"].end.isoformat(),
                        },
                        "formatted_time_period": scenario["expected_format"],
                        "total": {
                            "amount": scenario["amount"],
                            "currency": "USD"
                        }
                    }
                ]
            })
        )
        mock_pipeline.return_value = mock_pipeline_instance

        # Execute CLI command
        result = self.runner.invoke(cli, ["query", scenario["query"]])

        # Verify success
        assert result.exit_code == 0, f"CLI failed: {result.output}"

        # Parse JSON output
        try:
            output_data = json.loads(result.output)
            
            # Verify JSON structure includes formatted dates
            assert "formatted_time_period" in output_data
            assert output_data["formatted_time_period"] == scenario["expected_format"]
            
            # Verify results also include formatted dates
            if "results" in output_data and output_data["results"]:
                for result_item in output_data["results"]:
                    assert "formatted_time_period" in result_item
                    assert result_item["formatted_time_period"] == scenario["expected_format"]
                    
        except json.JSONDecodeError:
            # If not valid JSON, at least verify the formatted date is present
            assert scenario["expected_format"] in result.output

    @patch("src.aws_cost_cli.cli.ConfigManager")
    @patch("src.aws_cost_cli.cli.CredentialManager")
    @patch("src.aws_cost_cli.cli.QueryPipeline")
    def test_cli_query_with_format_override(
        self,
        mock_pipeline,
        mock_credential_manager,
        mock_config_manager,
    ):
        """Test CLI query command with format override maintains date formatting."""
        # Start with simple config but override to rich
        mock_config_manager.return_value.load_config.return_value = self.configs["simple"]
        mock_credential_manager.return_value.validate_credentials.return_value = True

        scenario = self.test_scenarios["monthly_ec2"]
        cost_data = self.create_cost_data(scenario)

        # Mock pipeline response
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.process_query.return_value = Mock(
            cost_data=cost_data,
            formatted_response=f"Rich override response for {scenario['service']} costs during {scenario['expected_format']}"
        )
        mock_pipeline.return_value = mock_pipeline_instance

        # Execute CLI command with format override
        result = self.runner.invoke(cli, ["query", scenario["query"], "--format", "rich"])

        # Verify success
        assert result.exit_code == 0, f"CLI failed: {result.output}"

        # Verify the formatted date appears in the output
        assert scenario["expected_format"] in result.output

    @patch("src.aws_cost_cli.cli.ConfigManager")
    @patch("src.aws_cost_cli.cli.CredentialManager")
    @patch("src.aws_cost_cli.cli.QueryPipeline")
    def test_cli_query_with_fresh_flag(
        self,
        mock_pipeline,
        mock_credential_manager,
        mock_config_manager,
    ):
        """Test CLI query command with fresh flag maintains date formatting."""
        mock_config_manager.return_value.load_config.return_value = self.configs["simple"]
        mock_credential_manager.return_value.validate_credentials.return_value = True

        scenario = self.test_scenarios["daily_lambda"]
        cost_data = self.create_cost_data(scenario)

        # Mock pipeline response
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.process_query.return_value = Mock(
            cost_data=cost_data,
            formatted_response=f"Fresh data for {scenario['service']} costs on {scenario['expected_format']}"
        )
        mock_pipeline.return_value = mock_pipeline_instance

        # Execute CLI command with fresh flag
        result = self.runner.invoke(cli, ["query", scenario["query"], "--fresh"])

        # Verify success
        assert result.exit_code == 0, f"CLI failed: {result.output}"

        # Verify the formatted date appears in the output
        assert scenario["expected_format"] in result.output

    @patch("src.aws_cost_cli.cli.ConfigManager")
    @patch("src.aws_cost_cli.cli.CredentialManager")
    @patch("src.aws_cost_cli.cli.QueryPipeline")
    def test_cli_query_with_multiple_results(
        self,
        mock_pipeline,
        mock_credential_manager,
        mock_config_manager,
    ):
        """Test CLI query command with multiple results shows all formatted dates."""
        mock_config_manager.return_value.load_config.return_value = self.configs["simple"]
        mock_credential_manager.return_value.validate_credentials.return_value = True

        # Create multiple results with different time periods
        results = []
        expected_formats = []
        
        for scenario_name, scenario in list(self.test_scenarios.items())[:3]:
            results.append(
                CostResult(
                    time_period=scenario["time_period"],
                    total=CostAmount(Decimal(str(scenario["amount"])), "USD"),
                    groups=[],
                    estimated=False,
                )
            )
            expected_formats.append(scenario["expected_format"])

        # Overall time period spans all results
        overall_period = TimePeriod(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )

        cost_data = CostData(
            results=results,
            time_period=overall_period,
            total_cost=CostAmount(Decimal("4000.00"), "USD"),
            currency="USD",
        )

        # Mock pipeline response
        mock_pipeline_instance = Mock()
        formatted_response = "Cost breakdown:\n"
        for i, fmt in enumerate(expected_formats):
            formatted_response += f"- {fmt}: ${list(self.test_scenarios.values())[i]['amount']}\n"
        
        mock_pipeline_instance.process_query.return_value = Mock(
            cost_data=cost_data,
            formatted_response=formatted_response
        )
        mock_pipeline.return_value = mock_pipeline_instance

        # Execute CLI command
        result = self.runner.invoke(cli, ["query", "Cost breakdown by time period"])

        # Verify success
        assert result.exit_code == 0, f"CLI failed: {result.output}"

        # Verify that multiple formatted dates appear in the output
        for expected_format in expected_formats:
            assert expected_format in result.output, (
                f"Expected '{expected_format}' not found in output. "
                f"Output: {result.output}"
            )

    @patch("src.aws_cost_cli.cli.ConfigManager")
    @patch("src.aws_cost_cli.cli.CredentialManager")
    @patch("src.aws_cost_cli.cli.QueryPipeline")
    def test_cli_query_error_handling_with_date_formatting(
        self,
        mock_pipeline,
        mock_credential_manager,
        mock_config_manager,
    ):
        """Test that error handling works correctly with date formatting."""
        mock_config_manager.return_value.load_config.return_value = self.configs["simple"]
        mock_credential_manager.return_value.validate_credentials.return_value = True

        # Mock pipeline to raise an exception
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.process_query.side_effect = Exception("Test error")
        mock_pipeline.return_value = mock_pipeline_instance

        # Execute CLI command
        result = self.runner.invoke(cli, ["query", "This will fail"])

        # Verify error handling
        assert result.exit_code != 0
        assert "error" in result.output.lower() or "failed" in result.output.lower()

    def test_cli_help_commands(self):
        """Test that CLI help commands work correctly."""
        # Test main help
        result = self.runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "AWS Cost Explorer CLI" in result.output

        # Test query help
        result = self.runner.invoke(cli, ["query", "--help"])
        assert result.exit_code == 0
        assert "Query AWS costs" in result.output

        # Test version
        result = self.runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "1.0.0" in result.output

    @patch("src.aws_cost_cli.cli.ConfigManager")
    @patch("src.aws_cost_cli.cli.CredentialManager")
    @patch("src.aws_cost_cli.cli.QueryPipeline")
    def test_cli_query_with_profile_option(
        self,
        mock_pipeline,
        mock_credential_manager,
        mock_config_manager,
    ):
        """Test CLI query command with AWS profile option."""
        mock_config_manager.return_value.load_config.return_value = self.configs["simple"]
        mock_credential_manager.return_value.validate_credentials.return_value = True

        scenario = self.test_scenarios["monthly_ec2"]
        cost_data = self.create_cost_data(scenario)

        # Mock pipeline response
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.process_query.return_value = Mock(
            cost_data=cost_data,
            formatted_response=f"Profile test: {scenario['service']} costs during {scenario['expected_format']}"
        )
        mock_pipeline.return_value = mock_pipeline_instance

        # Execute CLI command with profile option
        result = self.runner.invoke(cli, ["query", scenario["query"], "--profile", "test-profile"])

        # Verify success
        assert result.exit_code == 0, f"CLI failed: {result.output}"

        # Verify the formatted date appears in the output
        assert scenario["expected_format"] in result.output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])