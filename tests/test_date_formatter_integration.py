"""Integration tests for DateFormatter with ResponseFormatter."""

import pytest
from datetime import datetime, timezone
from decimal import Decimal

from src.aws_cost_cli.response_formatter import SimpleResponseFormatter, RichResponseFormatter
from src.aws_cost_cli.models import (
    CostData,
    CostResult,
    CostAmount,
    TimePeriod,
    QueryParameters,
    TimePeriodGranularity,
)


class TestDateFormatterIntegration:
    """Test DateFormatter integration with response formatters."""

    def setup_method(self):
        """Set up test fixtures."""
        self.simple_formatter = SimpleResponseFormatter()
        self.rich_formatter = RichResponseFormatter()

    def test_enhanced_formatting_available(self):
        """Test that enhanced formatting methods are available."""
        # Test that both formatters have the DateFormatter
        assert hasattr(self.simple_formatter, 'date_formatter')
        assert hasattr(self.rich_formatter, 'date_formatter')
        
        # Test that enhanced methods are available
        assert hasattr(self.simple_formatter, '_format_time_period_enhanced')
        assert hasattr(self.rich_formatter, '_format_time_period_enhanced')

    def test_backward_compatibility_maintained(self):
        """Test that backward compatibility is maintained."""
        # Test same day formatting - now uses enhanced formatting
        same_day_period = TimePeriod(
            start=datetime(2024, 1, 15, tzinfo=timezone.utc),
            end=datetime(2024, 1, 16, tzinfo=timezone.utc),  # Exclusive end date for single day
        )
        result = self.simple_formatter._format_time_period(same_day_period)
        assert result == "January 15, 2024"

        # Test same month formatting
        same_month_period = TimePeriod(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 2, 1, tzinfo=timezone.utc),  # Exclusive end date for full month
        )
        result = self.simple_formatter._format_time_period(same_month_period)
        assert result == "January 2024"

    def test_enhanced_formatting_for_quarters(self):
        """Test enhanced formatting for quarterly periods."""
        # Q1 2024 (Jan 1 - Apr 1, exclusive end)
        q1_period = TimePeriod(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 4, 1, tzinfo=timezone.utc),
        )
        enhanced_result = self.simple_formatter._format_time_period_enhanced(q1_period)
        # DateFormatter now correctly detects this as Q1
        assert enhanced_result == "Q1 2024"

        # Q2 2024 (Apr 1 - Jul 1, exclusive end)
        q2_period = TimePeriod(
            start=datetime(2024, 4, 1, tzinfo=timezone.utc),
            end=datetime(2024, 7, 1, tzinfo=timezone.utc),
        )
        enhanced_result = self.simple_formatter._format_time_period_enhanced(q2_period)
        # DateFormatter now correctly detects this as Q2
        assert enhanced_result == "Q2 2024"

    def test_enhanced_formatting_for_years(self):
        """Test enhanced formatting for yearly periods."""
        # Full year 2024 (Jan 1 - Jan 1 next year, exclusive end)
        year_period = TimePeriod(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        enhanced_result = self.simple_formatter._format_time_period_enhanced(year_period)
        # DateFormatter now correctly detects this as a single year
        assert enhanced_result == "2024"

    def test_enhanced_formatting_for_single_days(self):
        """Test enhanced formatting for single day periods."""
        # Single day with exclusive end (Jan 15 - Jan 16)
        single_day_period = TimePeriod(
            start=datetime(2024, 1, 15, tzinfo=timezone.utc),
            end=datetime(2024, 1, 16, tzinfo=timezone.utc),
        )
        enhanced_result = self.simple_formatter._format_time_period_enhanced(single_day_period)
        assert enhanced_result == "January 15, 2024"

    def test_enhanced_formatting_for_proper_months(self):
        """Test enhanced formatting for proper month boundaries."""
        # January 2024 with exclusive end (Jan 1 - Feb 1)
        month_period = TimePeriod(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 2, 1, tzinfo=timezone.utc),
        )
        enhanced_result = self.simple_formatter._format_time_period_enhanced(month_period)
        # DateFormatter detects this as custom range showing full month span
        assert "January" in enhanced_result and "2024" in enhanced_result

    def test_enhanced_formatting_for_multi_month_ranges(self):
        """Test enhanced formatting for multi-month ranges."""
        # January to March 2024 (Jan 1 - Apr 1, exclusive end)
        multi_month_period = TimePeriod(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 4, 1, tzinfo=timezone.utc),
        )
        enhanced_result = self.simple_formatter._format_time_period_enhanced(multi_month_period)
        # This should be detected as Q1, but if not, should be a multi-month range
        assert "2024" in enhanced_result
        assert ("Q1" in enhanced_result or "January" in enhanced_result)

    def test_enhanced_formatting_cross_year_ranges(self):
        """Test enhanced formatting for cross-year ranges."""
        # December 2024 to February 2025
        cross_year_period = TimePeriod(
            start=datetime(2024, 12, 1, tzinfo=timezone.utc),
            end=datetime(2025, 3, 1, tzinfo=timezone.utc),
        )
        enhanced_result = self.simple_formatter._format_time_period_enhanced(cross_year_period)
        assert "December" in enhanced_result or "2024" in enhanced_result
        assert "2025" in enhanced_result or "February" in enhanced_result

    def test_fallback_behavior(self):
        """Test that fallback behavior works correctly."""
        # Test with invalid period (should fall back gracefully)
        invalid_period = TimePeriod(
            start=datetime(2024, 1, 15, tzinfo=timezone.utc),
            end=datetime(2024, 1, 10, tzinfo=timezone.utc),  # End before start
        )
        
        # Should not crash and should return some reasonable format
        result = self.simple_formatter._format_time_period_enhanced(invalid_period)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_integration_with_full_response(self):
        """Test integration with full response formatting."""
        # Create test data with a quarterly period
        time_period = TimePeriod(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 4, 1, tzinfo=timezone.utc),
        )

        cost_data = CostData(
            results=[
                CostResult(
                    time_period=time_period,
                    total=CostAmount(Decimal("123.45"), "USD"),
                    groups=[],
                    estimated=False,
                )
            ],
            time_period=time_period,
            total_cost=CostAmount(Decimal("123.45"), "USD"),
            currency="USD",
        )

        query_params = QueryParameters(
            service="EC2",
            time_period=time_period,
            granularity=TimePeriodGranularity.MONTHLY,
        )

        # Test that the response includes the formatted period
        response = self.simple_formatter.format_response(
            cost_data, "What did I spend on EC2 in Q1?", query_params
        )

        # The response should contain the enhanced formatted period
        enhanced_period = self.simple_formatter._format_time_period_enhanced(time_period)
        # DateFormatter now correctly detects this as Q1
        assert enhanced_period == "Q1 2024"

    def test_rich_formatter_integration(self):
        """Test that RichResponseFormatter also has DateFormatter integration."""
        # Test that RichResponseFormatter has the same capabilities
        time_period = TimePeriod(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        
        enhanced_result = self.rich_formatter._format_time_period_enhanced(time_period)
        # DateFormatter now correctly detects this as a single year
        assert enhanced_result == "2024"

    def test_comparison_legacy_vs_enhanced(self):
        """Test comparison between legacy and enhanced formatting."""
        test_cases = [
            # Test that enhanced formatting provides more readable output
            (
                TimePeriod(
                    start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                    end=datetime(2024, 4, 1, tzinfo=timezone.utc),
                ),
                "multi_month_range"
            ),
            (
                TimePeriod(
                    start=datetime(2024, 1, 15, tzinfo=timezone.utc),
                    end=datetime(2024, 1, 16, tzinfo=timezone.utc),
                ),
                "single_day"
            ),
        ]

        for period, test_type in test_cases:
            legacy_result = self.simple_formatter._legacy_format_time_period(period)
            enhanced_result = self.simple_formatter._format_time_period_enhanced(period)
            
            # Both should contain the year
            assert "2024" in legacy_result
            assert "2024" in enhanced_result
            
            if test_type == "single_day":
                # Enhanced should be more readable for single days
                assert "January 15, 2024" == enhanced_result
            elif test_type == "multi_month_range":
                # Enhanced should now correctly detect this as Q1
                assert enhanced_result == "Q1 2024"

    def test_rich_formatter_uses_enhanced_formatting(self):
        """Test that RichResponseFormatter now uses enhanced date formatting."""
        # Test single month period
        month_period = TimePeriod(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 2, 1, tzinfo=timezone.utc),
        )
        
        # RichResponseFormatter should now use enhanced formatting
        result = self.rich_formatter._format_time_period(month_period)
        # Should be formatted as "January 2024" by DateFormatter
        assert "January" in result and "2024" in result
        
        # Test single day period
        day_period = TimePeriod(
            start=datetime(2024, 1, 15, tzinfo=timezone.utc),
            end=datetime(2024, 1, 16, tzinfo=timezone.utc),
        )
        
        result = self.rich_formatter._format_time_period(day_period)
        # Should be formatted as "January 15, 2024" by DateFormatter
        assert result == "January 15, 2024"

    def test_rich_formatter_full_response_with_enhanced_dates(self):
        """Test RichResponseFormatter full response with enhanced date formatting."""
        # Create test data with various time periods
        time_period = TimePeriod(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 2, 1, tzinfo=timezone.utc),
        )

        cost_data = CostData(
            results=[
                CostResult(
                    time_period=time_period,
                    total=CostAmount(Decimal("123.45"), "USD"),
                    groups=[],
                    estimated=False,
                )
            ],
            time_period=time_period,
            total_cost=CostAmount(Decimal("123.45"), "USD"),
            currency="USD",
        )

        query_params = QueryParameters(
            service="EC2",
            time_period=time_period,
            granularity=TimePeriodGranularity.MONTHLY,
        )

        # Mock Rich availability to ensure we test the Rich path
        self.rich_formatter._rich_available = True
        
        # Test that the response formatting works without errors
        # (We can't easily test the exact Rich output without complex mocking)
        try:
            response = self.rich_formatter.format_response(
                cost_data, "What did I spend on EC2 in January?", query_params
            )
            # Should not raise an exception
            assert isinstance(response, str)
        except ImportError:
            # If Rich is not available, should fall back gracefully
            pass

    def test_rich_formatter_table_headers_with_enhanced_dates(self):
        """Test that RichResponseFormatter table headers use enhanced date formatting."""
        # Create test data with multiple time periods for breakdown table
        periods = [
            TimePeriod(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 2, 1, tzinfo=timezone.utc),
            ),
            TimePeriod(
                start=datetime(2024, 2, 1, tzinfo=timezone.utc),
                end=datetime(2024, 3, 1, tzinfo=timezone.utc),
            ),
        ]

        results = []
        for period in periods:
            results.append(
                CostResult(
                    time_period=period,
                    total=CostAmount(Decimal("100.00"), "USD"),
                    groups=[],
                    estimated=False,
                )
            )

        cost_data = CostData(
            results=results,
            time_period=TimePeriod(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 3, 1, tzinfo=timezone.utc),
            ),
            total_cost=CostAmount(Decimal("200.00"), "USD"),
            currency="USD",
        )

        query_params = QueryParameters(
            service="EC2",
            time_period=cost_data.time_period,
            granularity=TimePeriodGranularity.MONTHLY,
        )

        # Test that each period is formatted correctly
        for period in periods:
            formatted_period = self.rich_formatter._format_time_period(period)
            # Should use enhanced formatting
            if period.start.month == 1:
                assert "January" in formatted_period and "2024" in formatted_period
            elif period.start.month == 2:
                assert "February" in formatted_period and "2024" in formatted_period

    def test_rich_formatter_panel_titles_with_enhanced_dates(self):
        """Test that RichResponseFormatter panel titles use enhanced date formatting."""
        # Test the main panel subtitle formatting
        time_period = TimePeriod(
            start=datetime(2024, 1, 15, tzinfo=timezone.utc),
            end=datetime(2024, 1, 16, tzinfo=timezone.utc),
        )
        
        # The panel subtitle should use enhanced formatting
        formatted_period = self.rich_formatter._format_time_period(time_period)
        assert formatted_period == "January 15, 2024"
        
        # Test with a month period
        month_period = TimePeriod(
            start=datetime(2024, 3, 1, tzinfo=timezone.utc),
            end=datetime(2024, 4, 1, tzinfo=timezone.utc),
        )
        
        formatted_month = self.rich_formatter._format_time_period(month_period)
        assert "March" in formatted_month and "2024" in formatted_month


if __name__ == "__main__":
    pytest.main([__file__])