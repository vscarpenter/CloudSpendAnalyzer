"""Tests for advanced query features (date ranges, quarters, fiscal years)."""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from src.aws_cost_cli.models import (
    DateRangeType,
    TimePeriod,
)
from src.aws_cost_cli.date_utils import (
    DateRangeCalculator,
    Quarter,
    parse_advanced_date_range,
)
from src.aws_cost_cli.query_processor import QueryParser


class TestDateRangeCalculator:
    """Test advanced date range calculations."""

    def test_quarter_range_calculation(self):
        """Test quarter date range calculation."""
        calculator = DateRangeCalculator()

        # Test Q1 2025
        q1_range = calculator.get_quarter_range(2025, Quarter.Q1)
        assert q1_range.start == datetime(2025, 1, 1, tzinfo=timezone.utc)
        assert q1_range.end == datetime(2025, 4, 1, tzinfo=timezone.utc)

        # Test Q4 2025 (crosses year boundary)
        q4_range = calculator.get_quarter_range(2025, Quarter.Q4)
        assert q4_range.start == datetime(2025, 10, 1, tzinfo=timezone.utc)
        assert q4_range.end == datetime(2026, 1, 1, tzinfo=timezone.utc)

    def test_fiscal_year_range(self):
        """Test fiscal year range calculation."""
        # Test calendar year fiscal year (January start)
        calculator = DateRangeCalculator(fiscal_year_start_month=1)
        fy_range = calculator.get_fiscal_year_range(2025)
        assert fy_range.start == datetime(2025, 1, 1, tzinfo=timezone.utc)
        assert fy_range.end == datetime(2026, 1, 1, tzinfo=timezone.utc)

        # Test October fiscal year start
        calculator = DateRangeCalculator(fiscal_year_start_month=10)
        fy_range = calculator.get_fiscal_year_range(2025)
        assert fy_range.start == datetime(2024, 10, 1, tzinfo=timezone.utc)
        assert fy_range.end == datetime(2025, 10, 1, tzinfo=timezone.utc)

    def test_fiscal_quarter_range(self):
        """Test fiscal quarter range calculation."""
        # Test with October fiscal year start
        calculator = DateRangeCalculator(fiscal_year_start_month=10)

        # FY2025 Q1 should be Oct-Dec 2024
        fq1_range = calculator.get_fiscal_quarter_range(2025, Quarter.Q1)
        assert fq1_range.start == datetime(2024, 10, 1, tzinfo=timezone.utc)
        assert fq1_range.end == datetime(2025, 1, 1, tzinfo=timezone.utc)

    def test_current_quarter_detection(self):
        """Test current quarter detection."""
        calculator = DateRangeCalculator()

        # Mock current date to August 24, 2025
        with patch("src.aws_cost_cli.date_utils.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2025, 8, 24, tzinfo=timezone.utc)
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            year, quarter = calculator.get_current_quarter()
            assert year == 2025
            assert quarter == Quarter.Q3

    def test_previous_period_calculation(self):
        """Test previous period calculation."""
        calculator = DateRangeCalculator()

        # Test same length previous period
        current_period = TimePeriod(
            start=datetime(2025, 7, 1, tzinfo=timezone.utc),
            end=datetime(2025, 8, 1, tzinfo=timezone.utc),
        )

        previous_period = calculator.get_previous_period(current_period, "same_length")
        # The period is 31 days long (July 1 to Aug 1), so previous period should be 31 days before
        expected_start = datetime(
            2025, 5, 31, tzinfo=timezone.utc
        )  # 31 days before July 1
        expected_end = datetime(2025, 7, 1, tzinfo=timezone.utc)  # July 1
        assert previous_period.start == expected_start
        assert previous_period.end == expected_end

        # Test year ago period
        year_ago_period = calculator.get_previous_period(current_period, "year_ago")
        assert year_ago_period.start == datetime(2024, 7, 1, tzinfo=timezone.utc)
        assert year_ago_period.end == datetime(2024, 8, 1, tzinfo=timezone.utc)

    def test_quarter_string_parsing(self):
        """Test quarter string parsing."""
        calculator = DateRangeCalculator()

        # Test various formats
        year, quarter = calculator.parse_quarter_string("Q1 2025")
        assert year == 2025
        assert quarter == Quarter.Q1

        year, quarter = calculator.parse_quarter_string("2025 Q3")
        assert year == 2025
        assert quarter == Quarter.Q3

        # Test with current year default
        with patch("src.aws_cost_cli.date_utils.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2025, 8, 24)
            year, quarter = calculator.parse_quarter_string("Q2")
            assert year == 2025
            assert quarter == Quarter.Q2


class TestAdvancedDateRangeParsing:
    """Test advanced date range parsing."""

    def test_quarter_parsing(self):
        """Test quarter string parsing."""
        # Test calendar quarter
        period = parse_advanced_date_range("Q1 2025")
        assert period.start == datetime(2025, 1, 1, tzinfo=timezone.utc)
        assert period.end == datetime(2025, 4, 1, tzinfo=timezone.utc)

        # Test fiscal quarter
        period = parse_advanced_date_range("FY Q2 2025", fiscal_year_start_month=10)
        # FY2025 Q2 with Oct start should be Jan-Mar 2025
        assert period.start == datetime(2025, 1, 1, tzinfo=timezone.utc)
        assert period.end == datetime(2025, 4, 1, tzinfo=timezone.utc)

    def test_fiscal_year_parsing(self):
        """Test fiscal year parsing."""
        period = parse_advanced_date_range("FY2025")
        assert period.start == datetime(2025, 1, 1, tzinfo=timezone.utc)
        assert period.end == datetime(2026, 1, 1, tzinfo=timezone.utc)

        # Test with different fiscal year start
        period = parse_advanced_date_range("FY2025", fiscal_year_start_month=7)
        assert period.start == datetime(2024, 7, 1, tzinfo=timezone.utc)
        assert period.end == datetime(2025, 7, 1, tzinfo=timezone.utc)

    def test_calendar_year_parsing(self):
        """Test calendar year parsing."""
        period = parse_advanced_date_range("2025")
        assert period.start == datetime(2025, 1, 1, tzinfo=timezone.utc)
        assert period.end == datetime(2026, 1, 1, tzinfo=timezone.utc)

    def test_invalid_date_range(self):
        """Test invalid date range handling."""
        with pytest.raises(ValueError):
            parse_advanced_date_range("invalid date")


class TestAdvancedQueryProcessing:
    """Test advanced query processing with LLM integration."""

    def setup_method(self):
        """Set up test environment."""
        self.llm_config = {
            "provider": "openai",
            "api_key": "test-key",
            "model": "gpt-3.5-turbo",
        }

    @patch("src.aws_cost_cli.query_processor.OpenAIProvider")
    def test_quarter_query_parsing(self, mock_provider_class):
        """Test parsing of quarter-based queries."""
        mock_provider = Mock()
        mock_provider.is_available.return_value = True
        mock_provider.parse_query.return_value = {
            "service": None,
            "start_date": "2025-07-01",
            "end_date": "2025-10-01",
            "granularity": "MONTHLY",
            "metrics": ["BlendedCost"],
            "group_by": ["SERVICE"],
            "date_range_type": "QUARTER",
        }
        mock_provider_class.return_value = mock_provider

        parser = QueryParser(self.llm_config)
        params = parser.parse_query("Show me costs by service for Q3 2025")

        assert params.date_range_type == DateRangeType.QUARTER
        assert params.group_by == ["SERVICE"]


if __name__ == "__main__":
    pytest.main([__file__])
