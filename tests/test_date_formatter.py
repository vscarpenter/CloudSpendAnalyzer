"""Tests for DateFormatter (smart human-readable time period formatting)."""

import pytest
from datetime import datetime, timedelta

from src.aws_cost_cli.date_formatter import DateFormatter
from src.aws_cost_cli.models import TimePeriod


class TestDateFormatterIntegration:
    """Tests for the public DateFormatter smart output."""

    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = DateFormatter()

    def test_format_single_day(self):
        """Test formatting of single day periods."""
        time_period = TimePeriod(start=datetime(2025, 1, 15), end=datetime(2025, 1, 16))
        assert self.formatter.format_time_period(time_period) == "January 15, 2025"

    def test_format_single_month(self):
        """Test formatting of single month periods."""
        time_period = TimePeriod(start=datetime(2025, 1, 1), end=datetime(2025, 2, 1))
        assert self.formatter.format_time_period(time_period) == "January 2025"

    def test_format_single_quarter(self):
        """Test formatting of single quarter periods."""
        time_period = TimePeriod(start=datetime(2025, 1, 1), end=datetime(2025, 4, 1))
        assert self.formatter.format_time_period(time_period) == "Q1 2025"

    def test_format_single_year(self):
        """Test formatting of single year periods."""
        time_period = TimePeriod(start=datetime(2025, 1, 1), end=datetime(2026, 1, 1))
        assert self.formatter.format_time_period(time_period) == "2025"

    def test_format_multi_month(self):
        """Test formatting of multi-month periods."""
        time_period = TimePeriod(start=datetime(2024, 11, 1), end=datetime(2025, 3, 1))
        assert self.formatter.format_time_period(time_period) == "November 2024 - February 2025"

    def test_format_custom_range(self):
        """Test formatting of custom range periods (same year, different months)."""
        time_period = TimePeriod(start=datetime(2025, 1, 15), end=datetime(2025, 2, 20))
        assert self.formatter.format_time_period(time_period) == "January 15 - February 19, 2025"

    def test_safe_format_with_error(self):
        """Test safe formatting with error handling."""
        time_period = TimePeriod(start=None, end=None)
        assert self.formatter.safe_format_time_period(time_period) == "Invalid date range"

    def test_format_time_period_comprehensive_scenarios(self):
        """Test comprehensive formatting scenarios including time-component edge cases."""
        test_cases = [
            (datetime(2025, 1, 15), datetime(2025, 1, 16), "January 15, 2025"),  # Single day
            (datetime(2025, 1, 1), datetime(2025, 2, 1), "January 2025"),        # Single month
            (datetime(2025, 1, 1), datetime(2025, 4, 1), "Q1 2025"),             # Single quarter
            (datetime(2025, 1, 1), datetime(2026, 1, 1), "2025"),                # Single year
            # Edge cases with time components
            (datetime(2025, 1, 15, 8, 30), datetime(2025, 1, 16, 9, 45), "January 15, 2025"),
            (datetime(2025, 1, 1, 0, 0), datetime(2025, 2, 1, 0, 0), "January 2025"),
        ]

        for start, end, expected in test_cases:
            time_period = TimePeriod(start=start, end=end)
            result = self.formatter.format_time_period(time_period)
            assert result == expected, f"Failed for {start} to {end}: got {result}, expected {expected}"


class TestDateFormatterErrorHandling:
    """Test error handling and fallback behaviour of the public API."""

    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = DateFormatter()

    def test_safe_format_none_time_period(self):
        """Test safe formatting with None time period."""
        assert self.formatter.safe_format_time_period(None) == "Invalid date range"

    def test_safe_format_none_start_date(self):
        """Test safe formatting with None start date."""
        time_period = TimePeriod(start=None, end=datetime(2025, 1, 1))
        assert self.formatter.safe_format_time_period(time_period) == "Invalid date range"

    def test_safe_format_none_end_date(self):
        """Test safe formatting with None end date."""
        time_period = TimePeriod(start=datetime(2025, 1, 1), end=None)
        assert self.formatter.safe_format_time_period(time_period) == "Invalid date range"

    def test_safe_format_both_dates_none(self):
        """Test safe formatting with both dates None."""
        time_period = TimePeriod(start=None, end=None)
        assert self.formatter.safe_format_time_period(time_period) == "Invalid date range"

    def test_safe_format_missing_attributes(self):
        """Test safe formatting with object missing required attributes."""
        class MockTimePeriod:
            pass

        assert self.formatter.safe_format_time_period(MockTimePeriod()) == "Invalid date range"

    def test_format_time_period_with_none_input(self):
        """Test format_time_period raises appropriate exception for None input."""
        with pytest.raises(ValueError, match="time_period cannot be None"):
            self.formatter.format_time_period(None)

    def test_format_time_period_missing_attributes(self):
        """Test format_time_period raises appropriate exception for missing attributes."""
        class MockTimePeriod:
            pass

        with pytest.raises(AttributeError, match="TimePeriod must have start and end attributes"):
            self.formatter.format_time_period(MockTimePeriod())

    def test_format_time_period_none_dates(self):
        """Test format_time_period raises appropriate exception for None dates."""
        time_period = TimePeriod(start=None, end=None)
        with pytest.raises(ValueError, match="TimePeriod start and end cannot be None"):
            self.formatter.format_time_period(time_period)

    def test_invalid_date_order_falls_back_to_iso(self):
        """A reversed range (start after end) should fall back to an ISO range."""
        time_period = TimePeriod(start=datetime(2025, 2, 1), end=datetime(2025, 1, 1))
        result = self.formatter.safe_format_time_period(time_period)
        assert "2025-02-01 to 2025-01-01" in result

    def test_integration_error_recovery(self):
        """Test that a duck-typed time period still formats correctly."""
        class DuckTimePeriod:
            def __init__(self):
                self.start = datetime(2025, 1, 1)
                self.end = datetime(2025, 2, 1)

        assert self.formatter.safe_format_time_period(DuckTimePeriod()) == "January 2025"

    def test_edge_case_recovery_scenarios(self):
        """Test recovery from various edge case scenarios."""
        # Microseconds and complex time components, still one calendar day
        time_period = TimePeriod(
            start=datetime(2025, 1, 15, 14, 30, 45, 123456),
            end=datetime(2025, 1, 16, 16, 45, 30, 654321),
        )
        assert self.formatter.safe_format_time_period(time_period) == "January 15, 2025"

        # Leap year single day
        time_period = TimePeriod(start=datetime(2024, 2, 29), end=datetime(2024, 3, 1))
        assert self.formatter.safe_format_time_period(time_period) == "February 29, 2024"

        # Year boundary single day
        time_period = TimePeriod(
            start=datetime(2024, 12, 31, 23, 59, 59),
            end=datetime(2025, 1, 1, 0, 0, 1),
        )
        assert self.formatter.safe_format_time_period(time_period) == "December 31, 2024"

    def test_smart_output_for_all_period_types(self):
        """Single source of truth for the smart format across all period types."""
        test_cases = [
            # Monthly periods
            (datetime(2025, 1, 1), datetime(2025, 2, 1), "January 2025"),
            (datetime(2025, 12, 1), datetime(2026, 1, 1), "December 2025"),
            # Yearly periods
            (datetime(2025, 1, 1), datetime(2026, 1, 1), "2025"),
            (datetime(2024, 1, 1), datetime(2025, 1, 1), "2024"),
            # Quarterly periods
            (datetime(2025, 1, 1), datetime(2025, 4, 1), "Q1 2025"),
            (datetime(2025, 4, 1), datetime(2025, 7, 1), "Q2 2025"),
            (datetime(2025, 7, 1), datetime(2025, 10, 1), "Q3 2025"),
            (datetime(2025, 10, 1), datetime(2026, 1, 1), "Q4 2025"),
            # Daily periods
            (datetime(2025, 1, 15), datetime(2025, 1, 16), "January 15, 2025"),
            (datetime(2025, 12, 31), datetime(2026, 1, 1), "December 31, 2025"),
            # Multi-month ranges
            (datetime(2025, 1, 1), datetime(2025, 6, 1), "January - May 2025"),
            (datetime(2024, 11, 1), datetime(2025, 2, 1), "November 2024 - January 2025"),
            # Custom ranges
            (datetime(2025, 4, 10), datetime(2025, 4, 20), "April 10 - 19, 2025"),
            (datetime(2025, 1, 15), datetime(2025, 3, 10), "January 15 - March 9, 2025"),
            (datetime(2024, 12, 20), datetime(2025, 1, 15), "December 20, 2024 - January 14, 2025"),
        ]

        for start, end, expected in test_cases:
            time_period = TimePeriod(start=start, end=end)
            result = self.formatter.format_time_period(time_period)
            assert result == expected, f"Failed for {start} to {end}: got '{result}', expected '{expected}'"

    def test_backward_compatibility_month_names(self):
        """The month_names attribute remains available for older callers."""
        assert hasattr(self.formatter, "month_names")
        assert len(self.formatter.month_names) == 12
        assert self.formatter.month_names[0] == "January"
        assert self.formatter.month_names[11] == "December"


class TestDateFormatterEdgeCases:
    """Test edge cases: month/year boundaries, leap years, timezones."""

    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = DateFormatter()

    def test_month_boundary_edge_cases(self):
        """Test edge cases at month boundaries."""
        # January 31 23:59:59 to February 1 00:00:00 -> single day
        time_period = TimePeriod(
            start=datetime(2025, 1, 31, 23, 59, 59),
            end=datetime(2025, 2, 1, 0, 0, 0),
        )
        assert self.formatter.format_time_period(time_period) == "January 31, 2025"

    def test_year_boundary_edge_cases(self):
        """Test edge cases at year boundaries."""
        # December 31, 2024 to January 1, 2025 -> single day
        time_period = TimePeriod(start=datetime(2024, 12, 31), end=datetime(2025, 1, 1))
        assert self.formatter.format_time_period(time_period) == "December 31, 2024"

        # December 2024 to January 2025 (cross-year multi-month)
        time_period = TimePeriod(start=datetime(2024, 12, 1), end=datetime(2025, 2, 1))
        result = self.formatter.format_time_period(time_period)
        assert "December 2024" in result and "January 2025" in result

    def test_leap_year_edge_cases(self):
        """Test edge cases with leap years."""
        # February 29, 2024 (leap year) single day
        time_period = TimePeriod(start=datetime(2024, 2, 29), end=datetime(2024, 3, 1))
        assert self.formatter.format_time_period(time_period) == "February 29, 2024"

        # February 2024 (leap year month)
        time_period = TimePeriod(start=datetime(2024, 2, 1), end=datetime(2024, 3, 1))
        result = self.formatter.format_time_period(time_period)
        assert "February" in result and "2024" in result

        # February 2025 (non-leap year month)
        time_period = TimePeriod(start=datetime(2025, 2, 1), end=datetime(2025, 3, 1))
        result = self.formatter.format_time_period(time_period)
        assert "February" in result and "2025" in result

    def test_partial_month_edge_cases(self):
        """Test edge cases with partial months."""
        # Mid-month to mid-month (same month) - end adjusted for exclusive end
        time_period = TimePeriod(start=datetime(2025, 1, 15), end=datetime(2025, 1, 25))
        result = self.formatter.format_time_period(time_period)
        assert "January 15" in result and "24, 2025" in result

        # Mid-month to mid-month (different months)
        time_period = TimePeriod(start=datetime(2025, 1, 15), end=datetime(2025, 3, 15))
        result = self.formatter.format_time_period(time_period)
        assert "January 15" in result and "March 14" in result and "2025" in result

    def test_quarter_boundary_edge_cases(self):
        """Test edge cases at quarter boundaries."""
        # Q1 to Q2 boundary (March 31 to April 1) -> single day
        time_period = TimePeriod(start=datetime(2025, 3, 31), end=datetime(2025, 4, 1))
        assert self.formatter.format_time_period(time_period) == "March 31, 2025"

        # Partial quarter (February to May) -> custom/multi-month range
        time_period = TimePeriod(start=datetime(2025, 2, 1), end=datetime(2025, 5, 1))
        result = self.formatter.format_time_period(time_period)
        assert "February" in result and "April" in result and "2025" in result

    def test_extreme_date_ranges(self):
        """Test extreme date ranges."""
        # Very short range (1 second) -> same day
        time_period = TimePeriod(
            start=datetime(2025, 1, 15, 12, 0, 0),
            end=datetime(2025, 1, 15, 12, 0, 1),
        )
        assert self.formatter.format_time_period(time_period) == "January 15, 2025"

        # Very long range (multiple years) -> custom range
        time_period = TimePeriod(start=datetime(2020, 1, 1), end=datetime(2025, 1, 1))
        result = self.formatter.format_time_period(time_period)
        assert "January 1, 2020" in result and "December 31, 2024" in result

    def test_timezone_edge_cases(self):
        """Test edge cases with timezone-aware datetimes."""
        from datetime import timezone

        # Same day in different times, timezone-aware
        time_period = TimePeriod(
            start=datetime(2025, 1, 15, 23, 0, 0, tzinfo=timezone.utc),
            end=datetime(2025, 1, 16, 1, 0, 0, tzinfo=timezone.utc),
        )
        assert self.formatter.format_time_period(time_period) == "January 15, 2025"


class TestDateFormatterPerformance:
    """Test performance characteristics of DateFormatter with large datasets."""

    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = DateFormatter()

    def test_performance_with_many_periods(self):
        """Test performance with many time periods."""
        import time

        periods = []
        for i in range(1000):
            start = datetime(2024, 1, 1) + timedelta(days=i)
            end = start + timedelta(days=1)
            periods.append(TimePeriod(start=start, end=end))

        start_time = time.time()
        results = [self.formatter.format_time_period(p) for p in periods]
        elapsed_time = time.time() - start_time

        assert elapsed_time < 1.0, f"Formatting took too long: {elapsed_time:.2f} seconds"
        assert len(results) == 1000
        assert all(isinstance(result, str) and len(result) > 0 for result in results)

    def test_performance_with_complex_periods(self):
        """Test performance with complex period types."""
        import time

        periods = []
        for month in range(1, 13):  # single days
            start = datetime(2024, month, 15)
            periods.append(TimePeriod(start=start, end=start + timedelta(days=1)))
        for month in range(1, 13):  # single months
            start = datetime(2024, month, 1)
            end = datetime(2025, 1, 1) if month == 12 else datetime(2024, month + 1, 1)
            periods.append(TimePeriod(start=start, end=end))
        for quarter_start in [1, 4, 7, 10]:  # quarters
            start = datetime(2024, quarter_start, 1)
            end = datetime(2025, 1, 1) if quarter_start == 10 else datetime(2024, quarter_start + 3, 1)
            periods.append(TimePeriod(start=start, end=end))
        for i in range(50):  # custom ranges
            start = datetime(2024, 1, 1) + timedelta(days=i * 7)
            periods.append(TimePeriod(start=start, end=start + timedelta(days=10)))

        start_time = time.time()
        results = [self.formatter.format_time_period(p) for p in periods]
        elapsed_time = time.time() - start_time

        assert elapsed_time < 2.0, f"Complex formatting took too long: {elapsed_time:.2f} seconds"
        assert len(results) == len(periods)
        assert all(isinstance(result, str) and len(result) > 0 for result in results)

    def test_caching_behavior(self):
        """Test that repeated formatting stays consistent and fast."""
        import time

        period = TimePeriod(start=datetime(2024, 1, 1), end=datetime(2024, 2, 1))

        start_time = time.time()
        result1 = self.formatter.format_time_period(period)
        first_time = time.time() - start_time

        times = []
        for _ in range(100):
            start_time = time.time()
            result = self.formatter.format_time_period(period)
            times.append(time.time() - start_time)
            assert result == result1

        avg_time = sum(times) / len(times)
        # Allow generous overhead; just guard against pathological slowdown.
        assert avg_time < max(first_time * 5, 0.001)


class TestDateFormatterWithRealAWSData:
    """Test DateFormatter with realistic AWS Cost Explorer API response data."""

    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = DateFormatter()

    def test_aws_daily_granularity_format(self):
        """Test formatting with AWS daily granularity data."""
        results = []
        for day in range(1, 32):
            start = datetime(2024, 1, day)
            results.append(self.formatter.format_time_period(
                TimePeriod(start=start, end=start + timedelta(days=1))
            ))

        assert len(results) == 31
        assert all("January" in result and "2024" in result for result in results)
        assert "January 1, 2024" in results
        assert "January 31, 2024" in results

    def test_aws_monthly_granularity_format(self):
        """Test formatting with AWS monthly granularity data."""
        results = []
        for month in range(1, 13):
            start = datetime(2024, month, 1)
            end = datetime(2025, 1, 1) if month == 12 else datetime(2024, month + 1, 1)
            results.append(self.formatter.format_time_period(TimePeriod(start=start, end=end)))

        assert len(results) == 12
        expected_months = [f"{m} 2024" for m in [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ]]
        for expected in expected_months:
            assert expected in results

    def test_aws_quarterly_data_format(self):
        """Test formatting with quarterly data patterns."""
        quarters = [
            (datetime(2024, 1, 1), datetime(2024, 4, 1)),
            (datetime(2024, 4, 1), datetime(2024, 7, 1)),
            (datetime(2024, 7, 1), datetime(2024, 10, 1)),
            (datetime(2024, 10, 1), datetime(2025, 1, 1)),
        ]
        results = [self.formatter.format_time_period(TimePeriod(start=s, end=e)) for s, e in quarters]

        assert results == ["Q1 2024", "Q2 2024", "Q3 2024", "Q4 2024"]

    def test_aws_yearly_data_format(self):
        """Test formatting with yearly data patterns."""
        years = [
            (datetime(2022, 1, 1), datetime(2023, 1, 1)),
            (datetime(2023, 1, 1), datetime(2024, 1, 1)),
            (datetime(2024, 1, 1), datetime(2025, 1, 1)),
        ]
        results = [self.formatter.format_time_period(TimePeriod(start=s, end=e)) for s, e in years]
        assert results == ["2022", "2023", "2024"]

    def test_aws_custom_date_ranges(self):
        """Test formatting with custom date ranges typical in AWS queries."""
        custom_ranges = [
            (datetime(2024, 1, 15), datetime(2024, 1, 22)),   # Last 7 days
            (datetime(2024, 1, 1), datetime(2024, 1, 31)),    # ~30 days (partial month)
            (datetime(2024, 1, 15), datetime(2024, 2, 15)),   # Billing period (partial month)
            (datetime(2024, 1, 1), datetime(2024, 2, 15)),    # Quarter-to-date
            (datetime(2024, 1, 1), datetime(2024, 6, 15)),    # Year-to-date
        ]
        results = [self.formatter.format_time_period(TimePeriod(start=s, end=e)) for s, e in custom_ranges]

        assert len(results) == 5
        assert all(isinstance(result, str) and len(result) > 0 for result in results)
        assert all("2024" in result for result in results)
        assert "January 15" in results[0] and "21, 2024" in results[0]
        assert "January 15" in results[2] and "February 14" in results[2]

    def test_aws_timezone_handling(self):
        """Test formatting with timezone-aware datetimes from AWS."""
        from datetime import timezone

        utc_periods = [
            TimePeriod(
                start=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
                end=datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc),
            ),
            TimePeriod(
                start=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
                end=datetime(2024, 2, 1, 0, 0, 0, tzinfo=timezone.utc),
            ),
        ]
        results = [self.formatter.format_time_period(p) for p in utc_periods]

        assert results[0] == "January 1, 2024"
        assert results[1] == "January 2024"

    def test_edge_cases_from_aws_data(self):
        """Test edge cases that might occur in real AWS data."""
        # Partial day at month boundary
        partial_day = TimePeriod(
            start=datetime(2024, 1, 31, 12, 0, 0),
            end=datetime(2024, 2, 1, 0, 0, 0),
        )
        result = self.formatter.format_time_period(partial_day)
        assert isinstance(result, str) and len(result) > 0

        # Very short period (less than a day) -> same day
        short_period = TimePeriod(
            start=datetime(2024, 1, 15, 10, 0, 0),
            end=datetime(2024, 1, 15, 14, 0, 0),
        )
        assert self.formatter.format_time_period(short_period) == "January 15, 2024"

        # Period spanning multiple years
        long_period = TimePeriod(start=datetime(2022, 1, 1), end=datetime(2024, 12, 31))
        result = self.formatter.format_time_period(long_period)
        assert "January 1, 2022" in result and "December 30, 2024" in result


if __name__ == "__main__":
    pytest.main([__file__])
