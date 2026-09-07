"""Regression tests for date formatting changes.

This module contains tests to ensure that the new date formatting
functionality doesn't break existing behavior and maintains backward
compatibility where expected.
"""

import pytest
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import Mock

from src.aws_cost_cli.models import (
    TimePeriod,
    CostData,
    CostResult,
    CostAmount,
    QueryParameters,
    TimePeriodGranularity,
)
from src.aws_cost_cli.response_formatter import (
    SimpleResponseFormatter,
    RichResponseFormatter,
    LLMResponseFormatter,
)
from src.aws_cost_cli.date_formatter import DateFormatter


class TestDateFormattingRegression:
    """Regression tests for date formatting functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.simple_formatter = SimpleResponseFormatter()
        self.rich_formatter = RichResponseFormatter()
        
        # Mock LLM provider for LLM formatter
        self.mock_llm_provider = Mock()
        self.mock_llm_provider.is_available.return_value = True
        self.llm_formatter = LLMResponseFormatter(self.mock_llm_provider)
        
        self.date_formatter = DateFormatter()

    def test_single_day_formatting_consistency(self):
        """Test that single day periods are formatted consistently across formatters."""
        period = TimePeriod(
            start=datetime(2024, 3, 15, tzinfo=timezone.utc),
            end=datetime(2024, 3, 16, tzinfo=timezone.utc),  # Exclusive end
        )
        
        simple_result = self.simple_formatter._format_time_period(period)
        rich_result = self.rich_formatter._format_time_period(period)
        direct_result = self.date_formatter.format_time_period(period)
        
        expected = "March 15, 2024"
        assert simple_result == expected
        assert rich_result == expected
        assert direct_result == expected

    def test_single_month_formatting_consistency(self):
        """Test that single month periods are formatted consistently."""
        period = TimePeriod(
            start=datetime(2024, 6, 1, tzinfo=timezone.utc),
            end=datetime(2024, 7, 1, tzinfo=timezone.utc),  # Exclusive end
        )
        
        simple_result = self.simple_formatter._format_time_period(period)
        rich_result = self.rich_formatter._format_time_period(period)
        direct_result = self.date_formatter.format_time_period(period)
        
        expected = "June 2024"
        assert simple_result == expected
        assert rich_result == expected
        assert direct_result == expected

    def test_quarter_formatting_consistency(self):
        """Test that quarterly periods are formatted consistently."""
        # Q3 2024 (July 1 - October 1)
        period = TimePeriod(
            start=datetime(2024, 7, 1, tzinfo=timezone.utc),
            end=datetime(2024, 10, 1, tzinfo=timezone.utc),  # Exclusive end
        )
        
        simple_result = self.simple_formatter._format_time_period(period)
        rich_result = self.rich_formatter._format_time_period(period)
        direct_result = self.date_formatter.format_time_period(period)
        
        expected = "Q3 2024"
        assert simple_result == expected
        assert rich_result == expected
        assert direct_result == expected

    def test_year_formatting_consistency(self):
        """Test that yearly periods are formatted consistently."""
        period = TimePeriod(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2025, 1, 1, tzinfo=timezone.utc),  # Exclusive end
        )
        
        simple_result = self.simple_formatter._format_time_period(period)
        rich_result = self.rich_formatter._format_time_period(period)
        direct_result = self.date_formatter.format_time_period(period)
        
        expected = "2024"
        assert simple_result == expected
        assert rich_result == expected
        assert direct_result == expected

    def test_custom_range_formatting_consistency(self):
        """Test that custom ranges are formatted consistently."""
        # Mid-month to mid-month range
        period = TimePeriod(
            start=datetime(2024, 2, 15, tzinfo=timezone.utc),
            end=datetime(2024, 3, 20, tzinfo=timezone.utc),
        )
        
        simple_result = self.simple_formatter._format_time_period(period)
        rich_result = self.rich_formatter._format_time_period(period)
        direct_result = self.date_formatter.format_time_period(period)
        
        expected = "February 15 - March 19, 2024"
        assert simple_result == expected
        assert rich_result == expected
        assert direct_result == expected

    def test_cross_year_range_formatting(self):
        """Test that cross-year ranges are formatted correctly."""
        period = TimePeriod(
            start=datetime(2024, 12, 15, tzinfo=timezone.utc),
            end=datetime(2025, 1, 15, tzinfo=timezone.utc),
        )
        
        simple_result = self.simple_formatter._format_time_period(period)
        rich_result = self.rich_formatter._format_time_period(period)
        direct_result = self.date_formatter.format_time_period(period)
        
        expected = "December 15, 2024 - January 14, 2025"
        assert simple_result == expected
        assert rich_result == expected
        assert direct_result == expected

    def test_full_response_includes_formatted_dates(self):
        """Test that full responses include properly formatted dates."""
        time_period = TimePeriod(
            start=datetime(2024, 4, 1, tzinfo=timezone.utc),
            end=datetime(2024, 5, 1, tzinfo=timezone.utc),  # April 2024
        )
        
        cost_data = CostData(
            results=[
                CostResult(
                    time_period=time_period,
                    total=CostAmount(Decimal("250.75"), "USD"),
                    groups=[],
                    estimated=False,
                )
            ],
            time_period=time_period,
            total_cost=CostAmount(Decimal("250.75"), "USD"),
            currency="USD",
        )
        
        query_params = QueryParameters(
            service="EC2",
            time_period=time_period,
            granularity=TimePeriodGranularity.MONTHLY,
        )
        
        # Test simple formatter response
        simple_response = self.simple_formatter.format_response(
            cost_data, "EC2 costs in April", query_params
        )
        assert "April 2024" in simple_response
        assert "$250.75" in simple_response

    def test_llm_formatter_includes_formatted_dates_in_summary(self):
        """Test that LLM formatter includes formatted dates in cost summary."""
        time_period = TimePeriod(
            start=datetime(2024, 9, 1, tzinfo=timezone.utc),
            end=datetime(2024, 10, 1, tzinfo=timezone.utc),  # September 2024
        )
        
        cost_data = CostData(
            results=[
                CostResult(
                    time_period=time_period,
                    total=CostAmount(Decimal("180.25"), "USD"),
                    groups=[],
                    estimated=False,
                )
            ],
            time_period=time_period,
            total_cost=CostAmount(Decimal("180.25"), "USD"),
            currency="USD",
        )
        
        query_params = QueryParameters(
            service="Lambda",
            time_period=time_period,
            granularity=TimePeriodGranularity.MONTHLY,
        )
        
        # Test cost summary preparation
        summary = self.llm_formatter._prepare_cost_summary(cost_data, query_params)
        
        # Check that formatted dates are included
        assert "formatted" in summary["time_period"]
        assert summary["time_period"]["formatted"] == "September 2024"
        assert summary["time_period"]["start"] == "2024-09-01"
        assert summary["time_period"]["end"] == "2024-10-01"
        
        # Check results also have formatted dates
        assert len(summary["results"]) == 1
        result = summary["results"][0]
        assert "formatted" in result["period"]
        assert result["period"]["formatted"] == "September 2024"

    def test_error_handling_preserves_functionality(self):
        """Test that error handling doesn't break existing functionality."""
        # Test with invalid time period (same start and end)
        invalid_period = TimePeriod(
            start=datetime(2024, 5, 15, tzinfo=timezone.utc),
            end=datetime(2024, 5, 15, tzinfo=timezone.utc),  # Same date
        )
        
        # Should not crash, should return fallback format
        simple_result = self.simple_formatter._format_time_period(invalid_period)
        rich_result = self.rich_formatter._format_time_period(invalid_period)
        direct_result = self.date_formatter.safe_format_time_period(invalid_period)
        
        # All should return some valid string (fallback behavior)
        assert isinstance(simple_result, str)
        assert isinstance(rich_result, str)
        assert isinstance(direct_result, str)
        assert len(simple_result) > 0
        assert len(rich_result) > 0
        assert len(direct_result) > 0

    def test_timezone_handling_consistency(self):
        """Test that timezone handling is consistent across formatters."""
        # Test with different timezone
        from datetime import timezone as tz, timedelta
        
        # UTC+5 timezone
        custom_tz = tz(timedelta(hours=5))
        
        period = TimePeriod(
            start=datetime(2024, 8, 1, tzinfo=custom_tz),
            end=datetime(2024, 9, 1, tzinfo=custom_tz),  # August 2024
        )
        
        simple_result = self.simple_formatter._format_time_period(period)
        rich_result = self.rich_formatter._format_time_period(period)
        direct_result = self.date_formatter.format_time_period(period)
        
        expected = "August 2024"
        assert simple_result == expected
        assert rich_result == expected
        assert direct_result == expected

    def test_leap_year_handling(self):
        """Test that leap year dates are handled correctly."""
        # February 2024 (leap year)
        period = TimePeriod(
            start=datetime(2024, 2, 1, tzinfo=timezone.utc),
            end=datetime(2024, 3, 1, tzinfo=timezone.utc),  # February 2024 (29 days)
        )
        
        simple_result = self.simple_formatter._format_time_period(period)
        rich_result = self.rich_formatter._format_time_period(period)
        direct_result = self.date_formatter.format_time_period(period)
        
        expected = "February 2024"
        assert simple_result == expected
        assert rich_result == expected
        assert direct_result == expected

    def test_legacy_format_still_available(self):
        """Test that legacy formatting methods are still available for compatibility."""
        period = TimePeriod(
            start=datetime(2024, 11, 15, tzinfo=timezone.utc),
            end=datetime(2024, 12, 15, tzinfo=timezone.utc),
        )
        
        # Legacy methods should still exist and work
        legacy_simple = self.simple_formatter._legacy_format_time_period(period)
        legacy_rich = self.rich_formatter._legacy_format_time_period(period)
        
        # Should return ISO-style format
        assert "2024-11-15" in legacy_simple
        assert "2024-12-15" in legacy_simple
        assert "2024-11-15" in legacy_rich
        assert "2024-12-15" in legacy_rich

    def test_enhanced_format_methods_available(self):
        """Test that enhanced formatting methods are available."""
        period = TimePeriod(
            start=datetime(2024, 10, 1, tzinfo=timezone.utc),
            end=datetime(2024, 11, 1, tzinfo=timezone.utc),  # October 2024
        )
        
        # Enhanced methods should exist and work
        enhanced_simple = self.simple_formatter._format_time_period_enhanced(period)
        enhanced_rich = self.rich_formatter._format_time_period_enhanced(period)
        
        expected = "October 2024"
        assert enhanced_simple == expected
        assert enhanced_rich == expected


if __name__ == "__main__":
    pytest.main([__file__])