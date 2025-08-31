"""Tests for data export functionality."""

import csv
import json
import os
import tempfile
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

from src.aws_cost_cli.data_exporter import (
    CSVExporter,
    JSONExporter,
    ExcelExporter,
    EmailReporter,
    ExportManager,
)
from src.aws_cost_cli.models import (
    CostData,
    CostResult,
    CostAmount,
    TimePeriod,
    QueryParameters,
    Group,
    TrendData,
    ForecastData,
    TimePeriodGranularity,
)


@pytest.fixture
def sample_cost_data():
    """Create sample cost data for testing."""
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 31)

    # Create cost results
    results = []
    for i in range(3):
        period_start = start_date + timedelta(days=i * 10)
        period_end = period_start + timedelta(days=9)

        # Create groups
        groups = [
            Group(
                keys=["EC2-Instance"],
                metrics={"BlendedCost": CostAmount(Decimal(f"{100 + i*50}"), "USD")},
            ),
            Group(
                keys=["S3"],
                metrics={"BlendedCost": CostAmount(Decimal(f"{50 + i*25}"), "USD")},
            ),
        ]

        result = CostResult(
            time_period=TimePeriod(period_start, period_end),
            total=CostAmount(Decimal(f"{150 + i*75}"), "USD"),
            groups=groups,
            estimated=(i == 2),  # Last result is estimated
        )
        results.append(result)

    # Create trend data
    trend_data = TrendData(
        current_period=CostAmount(Decimal("300"), "USD"),
        comparison_period=CostAmount(Decimal("250"), "USD"),
        change_amount=CostAmount(Decimal("50"), "USD"),
        change_percentage=20.0,
        trend_direction="up",
    )

    # Create forecast data
    forecast_data = [
        ForecastData(
            forecasted_amount=CostAmount(Decimal("320"), "USD"),
            confidence_interval_lower=CostAmount(Decimal("300"), "USD"),
            confidence_interval_upper=CostAmount(Decimal("340"), "USD"),
            forecast_period=TimePeriod(datetime(2024, 2, 1), datetime(2024, 2, 29)),
            prediction_accuracy=0.85,
        )
    ]

    return CostData(
        results=results,
        time_period=TimePeriod(start_date, end_date),
        total_cost=CostAmount(Decimal("525"), "USD"),
        currency="USD",
        group_definitions=["SERVICE"],
        trend_data=trend_data,
        forecast_data=forecast_data,
    )


@pytest.fixture
def sample_query_params():
    """Create sample query parameters."""
    params = QueryParameters(service="EC2", granularity=TimePeriodGranularity.DAILY)
    # Add original_query as an attribute for export functionality
    params.original_query = "EC2 costs for January"
    return params


class TestCSVExporter:
    """Test CSV export functionality."""

    def test_csv_export_basic(self, sample_cost_data, sample_query_params):
        """Test basic CSV export."""
        exporter = CSVExporter()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            output_path = f.name

        try:
            result_path = exporter.export(
                sample_cost_data, sample_query_params, output_path
            )
            assert result_path == output_path
            assert os.path.exists(output_path)

            # Read and verify CSV content
            with open(output_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Check for metadata
            assert "# AWS Cost Data Export" in content
            assert "# Query:,EC2 costs for January" in content
            assert "# Service:,EC2" in content
            assert "# Total Cost:,525 USD" in content

            # Parse CSV data
            lines = content.split("\n")
            data_start = None
            for i, line in enumerate(lines):
                if line.startswith("Period Start"):
                    data_start = i
                    break

            assert data_start is not None

            # Read data rows
            reader = csv.reader(lines[data_start:])
            headers = next(reader)
            assert "Period Start" in headers
            assert "Total Cost" in headers
            assert "Group Keys" in headers

            # Count data rows (should have 6 rows: 3 periods × 2 groups each)
            data_rows = list(reader)
            data_rows = [row for row in data_rows if row and not row[0].startswith("#")]
            assert len(data_rows) >= 6

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_csv_export_with_trend_and_forecast(
        self, sample_cost_data, sample_query_params
    ):
        """Test CSV export includes trend and forecast data."""
        exporter = CSVExporter()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            output_path = f.name

        try:
            exporter.export(sample_cost_data, sample_query_params, output_path)

            with open(output_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Check for trend analysis
            assert "# Trend Analysis" in content
            assert "Current Period Cost:,300.0" in content
            assert "Change Percentage:,20.0" in content

            # Check for forecast data
            assert "# Forecast Data" in content
            assert "Forecast Period Start" in content

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestJSONExporter:
    """Test JSON export functionality."""

    def test_json_export_basic(self, sample_cost_data, sample_query_params):
        """Test basic JSON export."""
        exporter = JSONExporter()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            result_path = exporter.export(
                sample_cost_data, sample_query_params, output_path
            )
            assert result_path == output_path
            assert os.path.exists(output_path)

            # Read and verify JSON content
            with open(output_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Check metadata
            assert "metadata" in data
            assert data["metadata"]["query"] == "EC2 costs for January"
            assert data["metadata"]["service"] == "EC2"

            # Check summary
            assert "summary" in data
            assert data["summary"]["total_cost"]["amount"] == 525.0
            assert data["summary"]["total_cost"]["currency"] == "USD"

            # Check results
            assert "results" in data
            assert len(data["results"]) == 3

            # Check first result
            first_result = data["results"][0]
            assert "time_period" in first_result
            assert "total" in first_result
            assert "groups" in first_result
            assert len(first_result["groups"]) == 2

            # Check group data
            first_group = first_result["groups"][0]
            assert "keys" in first_group
            assert "metrics" in first_group
            assert first_group["keys"] == ["EC2-Instance"]

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_json_export_with_trend_and_forecast(
        self, sample_cost_data, sample_query_params
    ):
        """Test JSON export includes trend and forecast data."""
        exporter = JSONExporter()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            exporter.export(sample_cost_data, sample_query_params, output_path)

            with open(output_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Check trend analysis
            assert "trend_analysis" in data
            trend = data["trend_analysis"]
            assert trend["current_period"]["amount"] == 300.0
            assert trend["change"]["percentage"] == 20.0
            assert trend["change"]["direction"] == "up"

            # Check forecast data
            assert "forecast" in data
            assert len(data["forecast"]) == 1
            forecast = data["forecast"][0]
            assert forecast["forecasted_amount"]["amount"] == 320.0
            assert forecast["prediction_accuracy"] == 0.85

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestExcelExporter:
    """Test Excel export functionality."""

    def test_excel_exporter_initialization(self):
        """Test Excel exporter initialization."""
        try:
            exporter = ExcelExporter()
            assert hasattr(exporter, "openpyxl")
        except ImportError:
            # openpyxl not available, skip test
            pytest.skip("openpyxl not available")

    def test_excel_export_dependency_error(self, sample_cost_data, sample_query_params):
        """Test Excel export fails gracefully when openpyxl is not available."""
        with patch(
            "src.aws_cost_cli.data_exporter.ExcelExporter._check_dependencies"
        ) as mock_check:
            mock_check.side_effect = ImportError(
                "openpyxl is required for Excel export"
            )

            with pytest.raises(ImportError) as exc_info:
                ExcelExporter()

            assert "openpyxl is required for Excel export" in str(exc_info.value)


class TestEmailReporter:
    """Test email reporting functionality."""

    def test_email_reporter_initialization(self):
        """Test email reporter initialization."""
        smtp_config = {
            "host": "smtp.example.com",
            "port": 587,
            "username": "user@example.com",
            "password": "password",
            "use_tls": True,
        }

        reporter = EmailReporter(smtp_config)
        assert reporter.smtp_config == smtp_config

    @patch("smtplib.SMTP")
    @patch("src.aws_cost_cli.data_exporter.EmailReporter._create_attachments")
    def test_send_report_basic(
        self, mock_create_attachments, mock_smtp, sample_cost_data, sample_query_params
    ):
        """Test basic email report sending."""
        # Mock SMTP
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server

        # Mock attachments
        mock_create_attachments.return_value = []

        smtp_config = {
            "host": "smtp.example.com",
            "port": 587,
            "username": "user@example.com",
            "password": "password",
            "use_tls": True,
        }

        reporter = EmailReporter(smtp_config)
        recipients = ["test@example.com"]

        result = reporter.send_report(
            sample_cost_data, sample_query_params, recipients, include_attachments=False
        )

        assert result is True
        mock_smtp.assert_called_with("smtp.example.com", 587)
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_with("user@example.com", "password")
        mock_server.send_message.assert_called_once()

    def test_create_email_body(self, sample_cost_data, sample_query_params):
        """Test email body creation."""
        smtp_config = {
            "host": "smtp.example.com",
            "port": 587,
            "username": "user@example.com",
            "password": "password",
        }

        reporter = EmailReporter(smtp_config)
        body = reporter._create_email_body(sample_cost_data, sample_query_params)

        assert "<html>" in body
        assert "AWS Cost Report for EC2" in body
        assert "$525.00" in body
        assert "Trend Analysis" in body
        assert "📈" in body  # Trend up symbol
        assert "Cost Forecast" in body


class TestExportManager:
    """Test export manager functionality."""

    def test_export_manager_initialization(self):
        """Test export manager initialization."""
        manager = ExportManager()

        # Should always have CSV and JSON
        assert "csv" in manager.exporters
        assert "json" in manager.exporters

        # Excel may or may not be available
        available_formats = manager.get_available_formats()
        assert "csv" in available_formats
        assert "json" in available_formats

    def test_export_data_csv(self, sample_cost_data, sample_query_params):
        """Test data export via manager - CSV format."""
        manager = ExportManager()

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            output_path = f.name

        try:
            result_path = manager.export_data(
                sample_cost_data, sample_query_params, "csv", output_path
            )

            assert result_path == output_path
            assert os.path.exists(output_path)

            # Verify it's a valid CSV
            with open(output_path, "r") as f:
                content = f.read()
                assert "# AWS Cost Data Export" in content

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_export_data_json(self, sample_cost_data, sample_query_params):
        """Test data export via manager - JSON format."""
        manager = ExportManager()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            result_path = manager.export_data(
                sample_cost_data, sample_query_params, "json", output_path
            )

            assert result_path == output_path
            assert os.path.exists(output_path)

            # Verify it's valid JSON
            with open(output_path, "r") as f:
                data = json.load(f)
                assert "metadata" in data
                assert "summary" in data

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_export_data_unsupported_format(
        self, sample_cost_data, sample_query_params
    ):
        """Test export with unsupported format."""
        manager = ExportManager()

        with pytest.raises(ValueError) as exc_info:
            manager.export_data(
                sample_cost_data,
                sample_query_params,
                "pdf",  # Unsupported format
                "output.pdf",
            )

        assert "Unsupported export format 'pdf'" in str(exc_info.value)

    @patch("src.aws_cost_cli.data_exporter.EmailReporter.send_report")
    def test_send_email_report(
        self, mock_send_report, sample_cost_data, sample_query_params
    ):
        """Test email report sending via manager."""
        mock_send_report.return_value = True

        manager = ExportManager()
        smtp_config = {
            "host": "smtp.example.com",
            "port": 587,
            "username": "user@example.com",
            "password": "password",
        }

        result = manager.send_email_report(
            sample_cost_data,
            sample_query_params,
            smtp_config,
            ["test@example.com"],
            attachment_formats=["csv"],
        )

        assert result is True
        mock_send_report.assert_called_once()


class TestIntegration:
    """Integration tests for export functionality."""

    def test_full_export_workflow_csv(self, sample_cost_data, sample_query_params):
        """Test complete export workflow for CSV."""
        manager = ExportManager()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test_export.csv")

            # Export data
            result_path = manager.export_data(
                sample_cost_data, sample_query_params, "csv", output_path
            )

            # Verify file exists and has content
            assert os.path.exists(result_path)
            assert os.path.getsize(result_path) > 0

            # Verify CSV structure
            with open(result_path, "r") as f:
                reader = csv.reader(f)
                rows = list(reader)

            # Should have metadata rows, headers, and data rows
            assert len(rows) > 10

            # Find data section
            header_row = None
            for i, row in enumerate(rows):
                if row and row[0] == "Period Start":
                    header_row = i
                    break

            assert header_row is not None
            assert "Total Cost" in rows[header_row]

    def test_full_export_workflow_json(self, sample_cost_data, sample_query_params):
        """Test complete export workflow for JSON."""
        manager = ExportManager()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test_export.json")

            # Export data
            result_path = manager.export_data(
                sample_cost_data, sample_query_params, "json", output_path
            )

            # Verify file exists and has content
            assert os.path.exists(result_path)
            assert os.path.getsize(result_path) > 0

            # Verify JSON structure
            with open(result_path, "r") as f:
                data = json.load(f)

            # Verify complete structure
            assert "metadata" in data
            assert "summary" in data
            assert "results" in data
            assert "trend_analysis" in data
            assert "forecast" in data

            # Verify data integrity
            assert data["summary"]["total_cost"]["amount"] == 525.0
            assert len(data["results"]) == 3
            assert data["trend_analysis"]["change"]["direction"] == "up"
