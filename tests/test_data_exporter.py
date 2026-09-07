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
    ExportManager,
)
from src.aws_cost_cli.models import (
    CostData,
    CostResult,
    CostAmount,
    TimePeriod,
    QueryParameters,
    Group,
    TimePeriodGranularity,
    DateFormattingConfig,
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

    return CostData(
        results=results,
        time_period=TimePeriod(start_date, end_date),
        total_cost=CostAmount(Decimal("525"), "USD"),
        currency="USD",
        group_definitions=["SERVICE"],
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
            assert "Period End" in headers
            assert "Formatted Period" in headers
            assert "Total Cost" in headers
            assert "Group Keys" in headers

            # Count data rows (should have 6 rows: 3 periods × 2 groups each)
            data_rows = list(reader)
            data_rows = [row for row in data_rows if row and not row[0].startswith("#")]
            assert len(data_rows) >= 6

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_csv_export_with_formatted_dates(
        self, sample_cost_data, sample_query_params
    ):
        """Test CSV export includes formatted dates."""
        exporter = CSVExporter()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            output_path = f.name

        try:
            exporter.export(sample_cost_data, sample_query_params, output_path)

            with open(output_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Check that formatted period is in metadata
            assert "# Period:" in content
            # Should contain formatted date, not just raw dates
            lines = content.split("\n")
            period_line = [line for line in lines if line.startswith("# Period:")][0]
            # Should not be just ISO format
            assert (
                "to" not in period_line
                or "January" in period_line
                or "2024" in period_line
            )

            # Parse CSV data to check formatted period column
            data_start = None
            for i, line in enumerate(lines):
                if line.startswith("Period Start"):
                    data_start = i
                    break

            assert data_start is not None

            reader = csv.reader(lines[data_start:])
            headers = next(reader)

            # Find the formatted period column index
            formatted_period_idx = headers.index("Formatted Period")

            # Check that data rows have formatted periods
            data_rows = list(reader)
            data_rows = [
                row
                for row in data_rows
                if row
                and len(row) > formatted_period_idx
                and not row[0].startswith("#")
            ]

            for row in data_rows[:3]:  # Check first few rows
                formatted_period = row[formatted_period_idx]
                # Should not be empty and should be human-readable
                assert formatted_period
                assert formatted_period != "Invalid date range"

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

    def test_json_export_with_formatted_dates(
        self, sample_cost_data, sample_query_params
    ):
        """Test JSON export includes formatted dates."""
        exporter = JSONExporter()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            exporter.export(sample_cost_data, sample_query_params, output_path)

            with open(output_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Check metadata has formatted time period
            assert "metadata" in data
            assert "time_period" in data["metadata"]
            time_period = data["metadata"]["time_period"]
            assert "start" in time_period
            assert "end" in time_period
            assert "formatted" in time_period

            # Formatted period should not be empty and should be human-readable
            formatted_period = time_period["formatted"]
            assert formatted_period
            assert formatted_period != "Invalid date range"

            # Check results have formatted time periods
            assert "results" in data
            for result in data["results"]:
                assert "time_period" in result
                result_time_period = result["time_period"]
                assert "start" in result_time_period
                assert "end" in result_time_period
                assert "formatted" in result_time_period

                # Formatted period should not be empty and should be human-readable
                result_formatted = result_time_period["formatted"]
                assert result_formatted
                assert result_formatted != "Invalid date range"

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


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

    def test_export_manager_uses_date_formatting_config(
        self, sample_cost_data, sample_query_params
    ):
        """Test export manager passes date formatting config to exporters."""
        manager = ExportManager(
            date_formatting_config=DateFormattingConfig(enabled=False)
        )

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            output_path = f.name

        try:
            manager.export_data(
                sample_cost_data, sample_query_params, "csv", output_path
            )

            with open(output_path, "r") as f:
                content = f.read()
                assert "2024-01-01 to 2024-01-31" in content
                assert "January 1-31, 2024" not in content

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


class TestFormattedDateExport:
    """Test formatted date functionality in exports."""

    def test_csv_formatted_dates_single_month(self):
        """Test CSV export with single month period formatting."""
        # Create single month cost data
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 2, 1)  # Exclusive end date for single month

        result = CostResult(
            time_period=TimePeriod(start_date, end_date),
            total=CostAmount(Decimal("100"), "USD"),
            groups=[],
            estimated=False,
        )

        cost_data = CostData(
            results=[result],
            time_period=TimePeriod(start_date, end_date),
            total_cost=CostAmount(Decimal("100"), "USD"),
            currency="USD",
            group_definitions=[],
        )

        query_params = QueryParameters(service="EC2")
        query_params.original_query = "EC2 costs for January"

        exporter = CSVExporter()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            output_path = f.name

        try:
            exporter.export(cost_data, query_params, output_path)

            with open(output_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Should contain formatted month name
            assert "January 2024" in content

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_json_formatted_dates_single_day(self):
        """Test JSON export with single day period formatting."""
        # Create single day cost data
        start_date = datetime(2024, 1, 15)
        end_date = datetime(2024, 1, 16)  # Exclusive end date for single day

        result = CostResult(
            time_period=TimePeriod(start_date, end_date),
            total=CostAmount(Decimal("50"), "USD"),
            groups=[],
            estimated=False,
        )

        cost_data = CostData(
            results=[result],
            time_period=TimePeriod(start_date, end_date),
            total_cost=CostAmount(Decimal("50"), "USD"),
            currency="USD",
            group_definitions=[],
        )

        query_params = QueryParameters(service="S3")
        query_params.original_query = "S3 costs for January 15"

        exporter = JSONExporter()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            exporter.export(cost_data, query_params, output_path)

            with open(output_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Check metadata formatted period
            formatted_period = data["metadata"]["time_period"]["formatted"]
            assert "January 15, 2024" in formatted_period

            # Check result formatted period
            result_formatted = data["results"][0]["time_period"]["formatted"]
            assert "January 15, 2024" in result_formatted

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_formatted_dates_fallback_handling(self):
        """Test that exports handle date formatting failures gracefully."""
        # Create invalid time period that might cause formatting issues
        start_date = datetime(2024, 1, 15)  # Mid-month start
        end_date = datetime(2024, 2, 10)  # Mid-month end (custom range)

        result = CostResult(
            time_period=TimePeriod(start_date, end_date),
            total=CostAmount(Decimal("75"), "USD"),
            groups=[],
            estimated=False,
        )

        cost_data = CostData(
            results=[result],
            time_period=TimePeriod(start_date, end_date),
            total_cost=CostAmount(Decimal("75"), "USD"),
            currency="USD",
            group_definitions=[],
        )

        query_params = QueryParameters(service="RDS")
        query_params.original_query = "RDS costs for custom period"

        # Test CSV export
        csv_exporter = CSVExporter()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            csv_output_path = f.name

        try:
            csv_exporter.export(cost_data, query_params, csv_output_path)

            with open(csv_output_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Should contain some formatted date (custom range format)
            assert "January" in content and "February" in content

        finally:
            if os.path.exists(csv_output_path):
                os.unlink(csv_output_path)

        # Test JSON export
        json_exporter = JSONExporter()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json_output_path = f.name

        try:
            json_exporter.export(cost_data, query_params, json_output_path)

            with open(json_output_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Should have formatted dates that are not error messages
            formatted_period = data["metadata"]["time_period"]["formatted"]
            assert formatted_period != "Invalid date range"
            assert "January" in formatted_period and "February" in formatted_period

        finally:
            if os.path.exists(json_output_path):
                os.unlink(json_output_path)


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

            # Verify data integrity
            assert data["summary"]["total_cost"]["amount"] == 525.0
            assert len(data["results"]) == 3
