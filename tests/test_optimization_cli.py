"""Integration tests for cost optimization CLI commands."""

import pytest
import json
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock
from click.testing import CliRunner

from src.aws_cost_cli.cli import cli
from src.aws_cost_cli.cost_optimizer import (
    OptimizationReport,
    OptimizationRecommendation,
    CostAnomaly,
    OptimizationType,
    SeverityLevel,
)
from src.aws_cost_cli.models import CostAmount, TimePeriod


class TestOptimizationCLI:
    """Test cases for optimization CLI commands."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def mock_optimizer_report(self):
        """Create a mock optimization report."""
        period = TimePeriod(
            start=datetime.now() - timedelta(days=30), end=datetime.now()
        )

        recommendations = [
            OptimizationRecommendation(
                type=OptimizationType.UNUSED_RESOURCES,
                severity=SeverityLevel.HIGH,
                title="Unused EBS Volume: vol-12345",
                description="EBS volume vol-12345 is not attached to any instance",
                potential_savings=CostAmount(amount=Decimal("50.0")),
                confidence_level=0.9,
                resource_id="vol-12345",
                service="Amazon Elastic Block Store",
                action_required="Delete unused volume",
                estimated_effort="low",
            ),
            OptimizationRecommendation(
                type=OptimizationType.RIGHTSIZING,
                severity=SeverityLevel.MEDIUM,
                title="Rightsize EC2 Instance: i-67890",
                description="Instance can be downsized from m5.large to m5.medium",
                potential_savings=CostAmount(amount=Decimal("75.0")),
                confidence_level=0.85,
                resource_id="i-67890",
                service="Amazon EC2",
                action_required="Resize instance to m5.medium",
                estimated_effort="medium",
            ),
        ]

        anomalies = [
            CostAnomaly(
                service="Amazon EC2",
                anomaly_date=datetime.now() - timedelta(days=5),
                expected_cost=CostAmount(amount=Decimal("100.0")),
                actual_cost=CostAmount(amount=Decimal("150.0")),
                variance_percentage=50.0,
                severity=SeverityLevel.HIGH,
                description="Cost spike detected",
            )
        ]

        return OptimizationReport(
            recommendations=recommendations,
            anomalies=anomalies,
            budget_variances=[],
            total_potential_savings=CostAmount(amount=Decimal("125.0")),
            report_date=datetime.now(),
            analysis_period=period,
        )

    @patch("src.aws_cost_cli.cli.CredentialManager")
    @patch("src.aws_cost_cli.cost_optimizer.CostOptimizer")
    def test_optimize_command_default(
        self,
        mock_optimizer_class,
        mock_credential_manager,
        runner,
        mock_optimizer_report,
    ):
        """Test optimize command with default parameters."""
        # Mock credential validation
        mock_credential_manager.return_value.validate_credentials.return_value = True

        # Mock optimizer
        mock_optimizer = Mock()
        mock_optimizer.generate_optimization_report.return_value = mock_optimizer_report
        mock_optimizer_class.return_value = mock_optimizer

        result = runner.invoke(cli, ["optimize"])

        assert result.exit_code == 0
        assert "Analyzing AWS costs for optimization opportunities" in result.output
        assert (
            "Analysis period: Last" in result.output
            and "30" in result.output
            and "days" in result.output
        )
        assert "Total Monthly Savings Potential: $125.00" in result.output

        # Verify optimizer was called with correct parameters
        mock_optimizer_class.assert_called_once_with(profile=None)
        mock_optimizer.generate_optimization_report.assert_called_once()

    @patch("src.aws_cost_cli.cli.CredentialManager")
    @patch("src.aws_cost_cli.cost_optimizer.CostOptimizer")
    def test_optimize_command_with_profile_and_days(
        self,
        mock_optimizer_class,
        mock_credential_manager,
        runner,
        mock_optimizer_report,
    ):
        """Test optimize command with custom profile and days."""
        # Mock credential validation
        mock_credential_manager.return_value.validate_credentials.return_value = True

        # Mock optimizer
        mock_optimizer = Mock()
        mock_optimizer.generate_optimization_report.return_value = mock_optimizer_report
        mock_optimizer_class.return_value = mock_optimizer

        result = runner.invoke(
            cli, ["optimize", "--profile", "production", "--days", "60"]
        )

        assert result.exit_code == 0
        assert (
            "Analysis period: Last" in result.output
            and "60" in result.output
            and "days" in result.output
        )
        assert "AWS Profile: production" in result.output

        # Verify optimizer was called with correct profile
        mock_optimizer_class.assert_called_once_with(profile="production")

    @patch("src.aws_cost_cli.cli.CredentialManager")
    @patch("src.aws_cost_cli.cost_optimizer.CostOptimizer")
    def test_optimize_command_json_output(
        self,
        mock_optimizer_class,
        mock_credential_manager,
        runner,
        mock_optimizer_report,
    ):
        """Test optimize command with JSON output format."""
        # Mock credential validation
        mock_credential_manager.return_value.validate_credentials.return_value = True

        # Mock optimizer
        mock_optimizer = Mock()
        mock_optimizer.generate_optimization_report.return_value = mock_optimizer_report
        mock_optimizer_class.return_value = mock_optimizer

        result = runner.invoke(cli, ["optimize", "--format", "json"])

        assert result.exit_code == 0

        # Parse JSON output
        output_data = json.loads(result.output)

        assert "report_date" in output_data
        assert "analysis_period" in output_data
        assert "total_potential_savings" in output_data
        assert output_data["total_potential_savings"]["amount"] == 125.0
        assert "recommendations" in output_data
        assert "anomalies" in output_data
        assert len(output_data["recommendations"]) == 2
        assert len(output_data["anomalies"]) == 1

        # Check recommendation details
        rec = output_data["recommendations"][0]
        assert rec["type"] == "unused_resources"
        assert rec["severity"] == "high"
        assert rec["resource_id"] == "vol-12345"
        assert rec["potential_savings"]["amount"] == 50.0

    @patch("src.aws_cost_cli.cli.CredentialManager")
    def test_optimize_command_invalid_credentials(
        self, mock_credential_manager, runner
    ):
        """Test optimize command with invalid AWS credentials."""
        # Mock credential validation failure
        mock_credential_manager.return_value.validate_credentials.return_value = False

        result = runner.invoke(cli, ["optimize"])

        assert result.exit_code == 1
        assert (
            "credentials" in result.output.lower() or "error" in result.output.lower()
        )

    @patch("src.aws_cost_cli.cli.CredentialManager")
    @patch("src.aws_cost_cli.cost_optimizer.CostOptimizer")
    def test_optimize_command_no_recommendations(
        self, mock_optimizer_class, mock_credential_manager, runner
    ):
        """Test optimize command when no optimization opportunities are found."""
        # Mock credential validation
        mock_credential_manager.return_value.validate_credentials.return_value = True

        # Create empty report
        period = TimePeriod(
            start=datetime.now() - timedelta(days=30), end=datetime.now()
        )
        empty_report = OptimizationReport(
            recommendations=[],
            anomalies=[],
            budget_variances=[],
            total_potential_savings=CostAmount(amount=Decimal("0.0")),
            report_date=datetime.now(),
            analysis_period=period,
        )

        # Mock optimizer
        mock_optimizer = Mock()
        mock_optimizer.generate_optimization_report.return_value = empty_report
        mock_optimizer_class.return_value = mock_optimizer

        result = runner.invoke(cli, ["optimize"])

        assert result.exit_code == 0
        assert "No significant optimization opportunities found" in result.output

    @patch("src.aws_cost_cli.cli.CredentialManager")
    @patch("src.aws_cost_cli.cost_optimizer.CostOptimizer")
    def test_detect_anomalies_command_default(
        self, mock_optimizer_class, mock_credential_manager, runner
    ):
        """Test detect-anomalies command with default parameters."""
        # Mock credential validation
        mock_credential_manager.return_value.validate_credentials.return_value = True

        # Mock optimizer with anomalies
        mock_optimizer = Mock()
        anomalies = [
            CostAnomaly(
                service="Amazon EC2",
                anomaly_date=datetime.now() - timedelta(days=2),
                expected_cost=CostAmount(amount=Decimal("100.0")),
                actual_cost=CostAmount(amount=Decimal("200.0")),
                variance_percentage=100.0,
                severity=SeverityLevel.HIGH,
                description="Significant cost spike detected",
            ),
            CostAnomaly(
                service="Amazon S3",
                anomaly_date=datetime.now() - timedelta(days=1),
                expected_cost=CostAmount(amount=Decimal("50.0")),
                actual_cost=CostAmount(amount=Decimal("75.0")),
                variance_percentage=50.0,
                severity=SeverityLevel.MEDIUM,
                description="Moderate cost increase",
            ),
        ]
        mock_optimizer._detect_cost_anomalies.return_value = anomalies
        mock_optimizer_class.return_value = mock_optimizer

        result = runner.invoke(cli, ["detect-anomalies"])

        assert result.exit_code == 0
        assert "Detecting cost anomalies" in result.output
        assert (
            "Analysis period: Last" in result.output
            and "7" in result.output
            and "days" in result.output
        )
        assert "Variance threshold: 20.0%" in result.output
        assert "2 Cost Anomalies Detected" in result.output
        assert "Amazon EC2" in result.output
        assert "Amazon S3" in result.output
        assert "$200.00" in result.output  # EC2 anomaly cost
        assert "+100.0%" in result.output  # EC2 variance
        assert "Total Anomaly Impact: $275.00" in result.output

    @patch("src.aws_cost_cli.cli.CredentialManager")
    @patch("src.aws_cost_cli.cost_optimizer.CostOptimizer")
    def test_detect_anomalies_command_with_filters(
        self, mock_optimizer_class, mock_credential_manager, runner
    ):
        """Test detect-anomalies command with service filter and custom parameters."""
        # Mock credential validation
        mock_credential_manager.return_value.validate_credentials.return_value = True

        # Mock optimizer
        mock_optimizer = Mock()
        anomalies = [
            CostAnomaly(
                service="Amazon EC2",
                anomaly_date=datetime.now() - timedelta(days=1),
                expected_cost=CostAmount(amount=Decimal("100.0")),
                actual_cost=CostAmount(amount=Decimal("160.0")),
                variance_percentage=60.0,
                severity=SeverityLevel.HIGH,
                description="EC2 cost spike",
            )
        ]
        mock_optimizer._detect_cost_anomalies.return_value = anomalies
        mock_optimizer_class.return_value = mock_optimizer

        result = runner.invoke(
            cli,
            [
                "detect-anomalies",
                "--service",
                "Amazon EC2",
                "--days",
                "14",
                "--threshold",
                "50.0",
                "--profile",
                "production",
            ],
        )

        assert result.exit_code == 0
        assert (
            "Analysis period: Last" in result.output
            and "14" in result.output
            and "days" in result.output
        )
        assert "Variance threshold: 50.0%" in result.output
        assert "Service filter: Amazon EC2" in result.output
        assert "AWS Profile: production" in result.output
        assert "1 Cost Anomalies Detected" in result.output

        # Verify optimizer was called with correct profile
        mock_optimizer_class.assert_called_once_with(profile="production")

    @patch("src.aws_cost_cli.cli.CredentialManager")
    @patch("src.aws_cost_cli.cost_optimizer.CostOptimizer")
    def test_detect_anomalies_command_no_anomalies(
        self, mock_optimizer_class, mock_credential_manager, runner
    ):
        """Test detect-anomalies command when no anomalies are found."""
        # Mock credential validation
        mock_credential_manager.return_value.validate_credentials.return_value = True

        # Mock optimizer with no anomalies
        mock_optimizer = Mock()
        mock_optimizer._detect_cost_anomalies.return_value = []
        mock_optimizer_class.return_value = mock_optimizer

        result = runner.invoke(cli, ["detect-anomalies"])

        assert result.exit_code == 0
        assert "No significant cost anomalies detected" in result.output

    @patch("src.aws_cost_cli.cli.CredentialManager")
    @patch("src.aws_cost_cli.cost_optimizer.CostOptimizer")
    def test_detect_anomalies_command_filtered_by_threshold(
        self, mock_optimizer_class, mock_credential_manager, runner
    ):
        """Test detect-anomalies command filtering by variance threshold."""
        # Mock credential validation
        mock_credential_manager.return_value.validate_credentials.return_value = True

        # Mock optimizer with anomalies below threshold
        mock_optimizer = Mock()
        anomalies = [
            CostAnomaly(
                service="Amazon S3",
                anomaly_date=datetime.now() - timedelta(days=1),
                expected_cost=CostAmount(amount=Decimal("100.0")),
                actual_cost=CostAmount(amount=Decimal("110.0")),
                variance_percentage=10.0,  # Below 20% threshold
                severity=SeverityLevel.LOW,
                description="Minor cost increase",
            )
        ]
        mock_optimizer._detect_cost_anomalies.return_value = anomalies
        mock_optimizer_class.return_value = mock_optimizer

        result = runner.invoke(cli, ["detect-anomalies", "--threshold", "20.0"])

        assert result.exit_code == 0
        assert "No significant cost anomalies detected" in result.output

    @patch("src.aws_cost_cli.cli.CredentialManager")
    @patch("src.aws_cost_cli.cost_optimizer.CostOptimizer")
    def test_detect_anomalies_command_service_filter(
        self, mock_optimizer_class, mock_credential_manager, runner
    ):
        """Test detect-anomalies command with service filtering."""
        # Mock credential validation
        mock_credential_manager.return_value.validate_credentials.return_value = True

        # Mock optimizer with mixed service anomalies
        mock_optimizer = Mock()
        anomalies = [
            CostAnomaly(
                service="Amazon EC2",
                anomaly_date=datetime.now() - timedelta(days=1),
                expected_cost=CostAmount(amount=Decimal("100.0")),
                actual_cost=CostAmount(amount=Decimal("150.0")),
                variance_percentage=50.0,
                severity=SeverityLevel.HIGH,
                description="EC2 cost spike",
            ),
            CostAnomaly(
                service="Amazon S3",
                anomaly_date=datetime.now() - timedelta(days=1),
                expected_cost=CostAmount(amount=Decimal("50.0")),
                actual_cost=CostAmount(amount=Decimal("75.0")),
                variance_percentage=50.0,
                severity=SeverityLevel.MEDIUM,
                description="S3 cost increase",
            ),
        ]
        mock_optimizer._detect_cost_anomalies.return_value = anomalies
        mock_optimizer_class.return_value = mock_optimizer

        result = runner.invoke(cli, ["detect-anomalies", "--service", "EC2"])

        assert result.exit_code == 0
        assert "1 Cost Anomalies Detected" in result.output
        assert "Amazon EC2" in result.output
        assert "Amazon S3" not in result.output  # Should be filtered out

    @patch("src.aws_cost_cli.cli.CredentialManager")
    def test_detect_anomalies_command_invalid_credentials(
        self, mock_credential_manager, runner
    ):
        """Test detect-anomalies command with invalid AWS credentials."""
        # Mock credential validation failure
        mock_credential_manager.return_value.validate_credentials.return_value = False

        result = runner.invoke(cli, ["detect-anomalies"])

        assert result.exit_code == 1
        assert (
            "credentials" in result.output.lower() or "error" in result.output.lower()
        )

    @patch("src.aws_cost_cli.cli.CredentialManager")
    @patch("src.aws_cost_cli.cost_optimizer.CostOptimizer")
    def test_optimize_command_exception_handling(
        self, mock_optimizer_class, mock_credential_manager, runner
    ):
        """Test optimize command exception handling."""
        # Mock credential validation
        mock_credential_manager.return_value.validate_credentials.return_value = True

        # Mock optimizer to raise exception
        mock_optimizer = Mock()
        mock_optimizer.generate_optimization_report.side_effect = Exception("API Error")
        mock_optimizer_class.return_value = mock_optimizer

        result = runner.invoke(cli, ["optimize"])

        assert result.exit_code == 1
        assert "Optimization analysis failed" in result.output
        assert "API Error" in result.output

    @patch("src.aws_cost_cli.cli.CredentialManager")
    @patch("src.aws_cost_cli.cost_optimizer.CostOptimizer")
    def test_detect_anomalies_command_exception_handling(
        self, mock_optimizer_class, mock_credential_manager, runner
    ):
        """Test detect-anomalies command exception handling."""
        # Mock credential validation
        mock_credential_manager.return_value.validate_credentials.return_value = True

        # Mock optimizer to raise exception
        mock_optimizer = Mock()
        mock_optimizer._detect_cost_anomalies.side_effect = Exception("Detection Error")
        mock_optimizer_class.return_value = mock_optimizer

        result = runner.invoke(cli, ["detect-anomalies"])

        assert result.exit_code == 1
        assert "Anomaly detection failed" in result.output
        assert "Detection Error" in result.output

    def test_optimize_command_help(self, runner):
        """Test optimize command help text."""
        result = runner.invoke(cli, ["optimize", "--help"])

        assert result.exit_code == 0
        assert "Generate cost optimization recommendations" in result.output
        assert "Unused resources" in result.output
        assert "Rightsizing opportunities" in result.output
        assert "Reserved Instance and Savings Plan recommendations" in result.output
        assert "Cost anomaly" in result.output and "detection" in result.output
        assert "Budget variance analysis" in result.output
        assert "--profile" in result.output
        assert "--days" in result.output
        assert "--format" in result.output

    def test_detect_anomalies_command_help(self, runner):
        """Test detect-anomalies command help text."""
        result = runner.invoke(cli, ["detect-anomalies", "--help"])

        assert result.exit_code == 0
        assert "Detect cost anomalies in your AWS spending" in result.output
        assert "unusual cost spikes" in result.output
        assert "--service" in result.output
        assert "--days" in result.output
        assert "--threshold" in result.output
        assert "--profile" in result.output

    @patch("src.aws_cost_cli.cli.CredentialManager")
    @patch("src.aws_cost_cli.cost_optimizer.CostOptimizer")
    def test_optimize_command_recommendations_display(
        self,
        mock_optimizer_class,
        mock_credential_manager,
        runner,
        mock_optimizer_report,
    ):
        """Test that optimize command properly displays different types of recommendations."""
        # Mock credential validation
        mock_credential_manager.return_value.validate_credentials.return_value = True

        # Mock optimizer
        mock_optimizer = Mock()
        mock_optimizer.generate_optimization_report.return_value = mock_optimizer_report
        mock_optimizer_class.return_value = mock_optimizer

        result = runner.invoke(cli, ["optimize"])

        assert result.exit_code == 0

        # Check that different recommendation types are displayed
        assert "Unused Resources" in result.output or "vol-12345" in result.output
        assert "Rightsizing" in result.output or "i-67890" in result.output

        # Check that savings amounts are displayed
        assert "$50.00" in result.output  # Unused resource savings
        assert "$75.00" in result.output  # Rightsizing savings
        assert "$125.00" in result.output  # Total savings

    @patch("src.aws_cost_cli.cli.CredentialManager")
    @patch("src.aws_cost_cli.cost_optimizer.CostOptimizer")
    def test_detect_anomalies_recommendations_display(
        self, mock_optimizer_class, mock_credential_manager, runner
    ):
        """Test that detect-anomalies command displays recommendations for high-impact anomalies."""
        # Mock credential validation
        mock_credential_manager.return_value.validate_credentials.return_value = True

        # Mock optimizer with high-impact anomaly
        mock_optimizer = Mock()
        anomalies = [
            CostAnomaly(
                service="Amazon EC2",
                anomaly_date=datetime.now() - timedelta(days=1),
                expected_cost=CostAmount(amount=Decimal("100.0")),
                actual_cost=CostAmount(amount=Decimal("250.0")),  # High impact > $100
                variance_percentage=150.0,
                severity=SeverityLevel.CRITICAL,
                description="Critical cost spike",
            )
        ]
        mock_optimizer._detect_cost_anomalies.return_value = anomalies
        mock_optimizer_class.return_value = mock_optimizer

        result = runner.invoke(cli, ["detect-anomalies"])

        assert result.exit_code == 0
        assert "Recommendations:" in result.output
        assert "Investigate high-impact anomalies" in result.output
        assert "AWS Budgets alerts" in result.output
        assert "Cost Anomaly Detection" in result.output
