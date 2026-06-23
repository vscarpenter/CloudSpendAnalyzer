"""Tests for DateFormatter, FormatRules, and PeriodTypeDetector."""

import pytest
from datetime import datetime, timedelta
from src.aws_cost_cli.date_formatter import DateFormatter, FormatRules, PeriodTypeDetector, PeriodType
from src.aws_cost_cli.models import TimePeriod


class TestPeriodTypeDetector:
    """Test cases for PeriodTypeDetector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = PeriodTypeDetector()
        self.fiscal_detector = PeriodTypeDetector(fiscal_year_start_month=4)  # April fiscal year
    
    def test_is_single_day_exclusive_end(self):
        """Test single day detection with exclusive end date (AWS typical format)."""
        # January 15, 2025 (exclusive end)
        start = datetime(2025, 1, 15)
        end = datetime(2025, 1, 16)
        time_period = TimePeriod(start=start, end=end)
        
        assert self.detector.is_single_day(time_period)
        assert self.detector.detect_period_type(time_period) == PeriodType.SINGLE_DAY
    
    def test_is_single_day_inclusive_same_day(self):
        """Test single day detection with inclusive same day."""
        # January 15, 2025 (same day, different times)
        start = datetime(2025, 1, 15, 0, 0, 0)
        end = datetime(2025, 1, 15, 23, 59, 59)
        time_period = TimePeriod(start=start, end=end)
        
        assert self.detector.is_single_day(time_period)
        assert self.detector.detect_period_type(time_period) == PeriodType.SINGLE_DAY
    
    def test_is_single_day_with_time_components(self):
        """Test single day detection ignoring time components."""
        # January 15, 2025 with various time components
        start = datetime(2025, 1, 15, 8, 30, 45)
        end = datetime(2025, 1, 16, 2, 15, 30)
        time_period = TimePeriod(start=start, end=end)
        
        assert self.detector.is_single_day(time_period)
    
    def test_is_not_single_day_multiple_days(self):
        """Test that multiple days are not detected as single day."""
        # January 15-17, 2025
        start = datetime(2025, 1, 15)
        end = datetime(2025, 1, 17)
        time_period = TimePeriod(start=start, end=end)
        
        assert not self.detector.is_single_day(time_period)
    
    def test_is_single_month_january(self):
        """Test single month detection for January."""
        # January 2025 (full month)
        start = datetime(2025, 1, 1)
        end = datetime(2025, 2, 1)
        time_period = TimePeriod(start=start, end=end)
        
        assert self.detector.is_single_month(time_period)
        assert self.detector.detect_period_type(time_period) == PeriodType.SINGLE_MONTH
    
    def test_is_single_month_december_year_boundary(self):
        """Test single month detection for December crossing year boundary."""
        # December 2024
        start = datetime(2024, 12, 1)
        end = datetime(2025, 1, 1)
        time_period = TimePeriod(start=start, end=end)
        
        assert self.detector.is_single_month(time_period)
        assert self.detector.detect_period_type(time_period) == PeriodType.SINGLE_MONTH
    
    def test_is_single_month_february_leap_year(self):
        """Test single month detection for February in leap year."""
        # February 2024 (leap year)
        start = datetime(2024, 2, 1)
        end = datetime(2024, 3, 1)
        time_period = TimePeriod(start=start, end=end)
        
        assert self.detector.is_single_month(time_period)
    
    def test_is_single_month_february_non_leap_year(self):
        """Test single month detection for February in non-leap year."""
        # February 2025 (non-leap year)
        start = datetime(2025, 2, 1)
        end = datetime(2025, 3, 1)
        time_period = TimePeriod(start=start, end=end)
        
        assert self.detector.is_single_month(time_period)
    
    def test_is_not_single_month_partial_start(self):
        """Test that partial month at start is not detected as single month."""
        # January 15 - February 1, 2025
        start = datetime(2025, 1, 15)
        end = datetime(2025, 2, 1)
        time_period = TimePeriod(start=start, end=end)
        
        assert not self.detector.is_single_month(time_period)
    
    def test_is_not_single_month_partial_end(self):
        """Test that partial month at end is not detected as single month."""
        # January 1 - January 15, 2025
        start = datetime(2025, 1, 1)
        end = datetime(2025, 1, 15)
        time_period = TimePeriod(start=start, end=end)
        
        assert not self.detector.is_single_month(time_period)
    
    def test_is_single_quarter_q1(self):
        """Test single quarter detection for Q1."""
        # Q1 2025 (Jan-Mar)
        start = datetime(2025, 1, 1)
        end = datetime(2025, 4, 1)
        time_period = TimePeriod(start=start, end=end)
        
        assert self.detector.is_single_quarter(time_period)
        assert self.detector.detect_period_type(time_period) == PeriodType.SINGLE_QUARTER
    
    def test_is_single_quarter_q2(self):
        """Test single quarter detection for Q2."""
        # Q2 2025 (Apr-Jun)
        start = datetime(2025, 4, 1)
        end = datetime(2025, 7, 1)
        time_period = TimePeriod(start=start, end=end)
        
        assert self.detector.is_single_quarter(time_period)
    
    def test_is_single_quarter_q3(self):
        """Test single quarter detection for Q3."""
        # Q3 2025 (Jul-Sep)
        start = datetime(2025, 7, 1)
        end = datetime(2025, 10, 1)
        time_period = TimePeriod(start=start, end=end)
        
        assert self.detector.is_single_quarter(time_period)
    
    def test_is_single_quarter_q4_year_boundary(self):
        """Test single quarter detection for Q4 crossing year boundary."""
        # Q4 2024 (Oct-Dec)
        start = datetime(2024, 10, 1)
        end = datetime(2025, 1, 1)
        time_period = TimePeriod(start=start, end=end)
        
        assert self.detector.is_single_quarter(time_period)
    
    def test_is_single_quarter_fiscal_year(self):
        """Test single quarter detection for fiscal year starting in April."""
        # Fiscal Q1 2025 (Apr-Jun) for April fiscal year
        start = datetime(2025, 4, 1)
        end = datetime(2025, 7, 1)
        time_period = TimePeriod(start=start, end=end)
        
        assert self.fiscal_detector.is_single_quarter(time_period)
    
    def test_is_not_single_quarter_wrong_start_month(self):
        """Test that quarter starting in wrong month is not detected."""
        # February - May (not a quarter)
        start = datetime(2025, 2, 1)
        end = datetime(2025, 5, 1)
        time_period = TimePeriod(start=start, end=end)
        
        assert not self.detector.is_single_quarter(time_period)
    
    def test_is_not_single_quarter_partial_start(self):
        """Test that partial quarter at start is not detected."""
        # January 15 - April 1 (partial Q1)
        start = datetime(2025, 1, 15)
        end = datetime(2025, 4, 1)
        time_period = TimePeriod(start=start, end=end)
        
        assert not self.detector.is_single_quarter(time_period)
    
    def test_is_single_year_calendar(self):
        """Test single year detection for calendar year."""
        # Calendar year 2025
        start = datetime(2025, 1, 1)
        end = datetime(2026, 1, 1)
        time_period = TimePeriod(start=start, end=end)
        
        assert self.detector.is_single_year(time_period)
        assert self.detector.detect_period_type(time_period) == PeriodType.SINGLE_YEAR
    
    def test_is_single_year_fiscal(self):
        """Test single year detection for fiscal year."""
        # Fiscal year 2025 (Apr 2025 - Mar 2026)
        start = datetime(2025, 4, 1)
        end = datetime(2026, 4, 1)
        time_period = TimePeriod(start=start, end=end)
        
        assert self.fiscal_detector.is_single_year(time_period)
    
    def test_is_not_single_year_wrong_start_month(self):
        """Test that year starting in wrong month is not detected."""
        # February - February (not a calendar year)
        start = datetime(2025, 2, 1)
        end = datetime(2026, 2, 1)
        time_period = TimePeriod(start=start, end=end)
        
        assert not self.detector.is_single_year(time_period)
    
    def test_is_not_single_year_partial(self):
        """Test that partial year is not detected."""
        # January 15 - January 15 next year
        start = datetime(2025, 1, 15)
        end = datetime(2026, 1, 15)
        time_period = TimePeriod(start=start, end=end)
        
        assert not self.detector.is_single_year(time_period)
    
    def test_is_multi_month_same_year(self):
        """Test multi-month detection within same year."""
        # January - May 2025 (5 months, not a quarter)
        start = datetime(2025, 1, 1)
        end = datetime(2025, 6, 1)
        time_period = TimePeriod(start=start, end=end)
        
        # This should be detected as multi-month
        assert self.detector.is_multi_month(time_period)
        assert self.detector.detect_period_type(time_period) == PeriodType.MULTI_MONTH
    
    def test_is_multi_month_cross_year(self):
        """Test multi-month detection crossing year boundary."""
        # November 2024 - February 2025 (4 months)
        start = datetime(2024, 11, 1)
        end = datetime(2025, 3, 1)
        time_period = TimePeriod(start=start, end=end)
        
        assert self.detector.is_multi_month(time_period)
        assert self.detector.detect_period_type(time_period) == PeriodType.MULTI_MONTH
    
    def test_is_multi_month_five_months(self):
        """Test multi-month detection for 5 months."""
        # January - June 2025 (6 months)
        start = datetime(2025, 1, 1)
        end = datetime(2025, 7, 1)
        time_period = TimePeriod(start=start, end=end)
        
        assert self.detector.is_multi_month(time_period)
    
    def test_quarter_vs_multi_month_priority(self):
        """Test that quarters are detected before multi-month in detect_period_type."""
        # January - March 2025 (3 months = Q1)
        start = datetime(2025, 1, 1)
        end = datetime(2025, 4, 1)
        time_period = TimePeriod(start=start, end=end)
        
        # Individual methods can both return True, but detect_period_type prioritizes quarter
        assert self.detector.is_single_quarter(time_period)
        assert self.detector.is_multi_month(time_period)  # 3 months is technically multi-month
        # But the overall detection should prioritize quarter
        assert self.detector.detect_period_type(time_period) == PeriodType.SINGLE_QUARTER
    
    def test_is_not_multi_month_single_month(self):
        """Test that single month is not detected as multi-month."""
        # January 2025
        start = datetime(2025, 1, 1)
        end = datetime(2025, 2, 1)
        time_period = TimePeriod(start=start, end=end)
        
        assert not self.detector.is_multi_month(time_period)
    
    def test_is_not_multi_month_full_year(self):
        """Test that full year is not detected as multi-month."""
        # Full year 2025
        start = datetime(2025, 1, 1)
        end = datetime(2026, 1, 1)
        time_period = TimePeriod(start=start, end=end)
        
        assert not self.detector.is_multi_month(time_period)
    
    def test_is_not_multi_month_partial_start(self):
        """Test that partial month at start is not multi-month."""
        # January 15 - April 1
        start = datetime(2025, 1, 15)
        end = datetime(2025, 4, 1)
        time_period = TimePeriod(start=start, end=end)
        
        assert not self.detector.is_multi_month(time_period)
    
    def test_is_custom_range_partial_months(self):
        """Test custom range detection for partial months."""
        # January 15 - February 20, 2025
        start = datetime(2025, 1, 15)
        end = datetime(2025, 2, 20)
        time_period = TimePeriod(start=start, end=end)
        
        assert self.detector.is_custom_range(time_period)
        assert self.detector.detect_period_type(time_period) == PeriodType.CUSTOM_RANGE
    
    def test_is_custom_range_unusual_boundaries(self):
        """Test custom range detection for unusual boundaries."""
        # Mid-month to mid-month
        start = datetime(2025, 1, 15)
        end = datetime(2025, 3, 15)
        time_period = TimePeriod(start=start, end=end)
        
        assert self.detector.is_custom_range(time_period)
    
    def test_get_quarter_number(self):
        """Test quarter number calculation."""
        assert self.detector.get_quarter_number(1) == 1  # January
        assert self.detector.get_quarter_number(3) == 1  # March
        assert self.detector.get_quarter_number(4) == 2  # April
        assert self.detector.get_quarter_number(6) == 2  # June
        assert self.detector.get_quarter_number(7) == 3  # July
        assert self.detector.get_quarter_number(9) == 3  # September
        assert self.detector.get_quarter_number(10) == 4  # October
        assert self.detector.get_quarter_number(12) == 4  # December
    
    def test_get_quarter_number_invalid(self):
        """Test quarter number calculation with invalid input."""
        with pytest.raises(ValueError):
            self.detector.get_quarter_number(0)
        
        with pytest.raises(ValueError):
            self.detector.get_quarter_number(13)
    
    def test_get_fiscal_quarter_number(self):
        """Test fiscal quarter number calculation."""
        # For April fiscal year start
        assert self.fiscal_detector.get_fiscal_quarter_number(4) == 1  # April (FQ1)
        assert self.fiscal_detector.get_fiscal_quarter_number(6) == 1  # June (FQ1)
        assert self.fiscal_detector.get_fiscal_quarter_number(7) == 2  # July (FQ2)
        assert self.fiscal_detector.get_fiscal_quarter_number(9) == 2  # September (FQ2)
        assert self.fiscal_detector.get_fiscal_quarter_number(10) == 3  # October (FQ3)
        assert self.fiscal_detector.get_fiscal_quarter_number(12) == 3  # December (FQ3)
        assert self.fiscal_detector.get_fiscal_quarter_number(1) == 4  # January (FQ4)
        assert self.fiscal_detector.get_fiscal_quarter_number(3) == 4  # March (FQ4)
    
    def test_invalid_time_period_none(self):
        """Test handling of None time period."""
        assert self.detector.detect_period_type(None) == PeriodType.CUSTOM_RANGE
    
    def test_invalid_time_period_start_after_end(self):
        """Test handling of invalid time period where start is after end."""
        start = datetime(2025, 2, 1)
        end = datetime(2025, 1, 1)
        time_period = TimePeriod(start=start, end=end)
        
        assert self.detector.detect_period_type(time_period) == PeriodType.CUSTOM_RANGE
    
    def test_invalid_time_period_extreme_dates(self):
        """Test handling of extreme dates."""
        # Very old date
        start = datetime(1800, 1, 1)
        end = datetime(1800, 2, 1)
        time_period = TimePeriod(start=start, end=end)
        
        assert self.detector.detect_period_type(time_period) == PeriodType.CUSTOM_RANGE
        
        # Very future date
        start = datetime(2100, 1, 1)
        end = datetime(2100, 2, 1)
        time_period = TimePeriod(start=start, end=end)
        
        assert self.detector.detect_period_type(time_period) == PeriodType.CUSTOM_RANGE
    
    def test_edge_case_leap_year_february(self):
        """Test edge case with leap year February."""
        # February 29, 2024 (leap year)
        start = datetime(2024, 2, 29)
        end = datetime(2024, 3, 1)
        time_period = TimePeriod(start=start, end=end)
        
        assert self.detector.is_single_day(time_period)
    
    def test_edge_case_month_boundary_with_time(self):
        """Test edge case with month boundary and time components."""
        # January 31, 2025 23:59:59 to February 1, 2025 00:00:00
        start = datetime(2025, 1, 31, 23, 59, 59)
        end = datetime(2025, 2, 1, 0, 0, 0)
        time_period = TimePeriod(start=start, end=end)
        
        # This should be detected as single day due to date normalization
        assert self.detector.is_single_day(time_period)
    
    def test_edge_case_year_boundary_december_january(self):
        """Test edge case with year boundary crossing."""
        # December 31, 2024 to January 1, 2025
        start = datetime(2024, 12, 31)
        end = datetime(2025, 1, 1)
        time_period = TimePeriod(start=start, end=end)
        
        assert self.detector.is_single_day(time_period)


class TestFormatRules:
    """Test cases for FormatRules template-based formatting."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.format_rules = FormatRules()
    
    def test_format_single_day_template(self):
        """Test single day formatting using template."""
        date = datetime(2025, 1, 15)
        result = self.format_rules.format_single_day(date)
        assert result == "January 15, 2025"
    
    def test_format_single_day_different_months(self):
        """Test single day formatting for different months."""
        # Test various months
        test_cases = [
            (datetime(2025, 1, 1), "January 1, 2025"),
            (datetime(2025, 2, 14), "February 14, 2025"),
            (datetime(2025, 3, 31), "March 31, 2025"),
            (datetime(2025, 12, 25), "December 25, 2025"),
        ]
        
        for date, expected in test_cases:
            result = self.format_rules.format_single_day(date)
            assert result == expected
    
    def test_format_single_month_template(self):
        """Test single month formatting using template."""
        date = datetime(2025, 1, 1)
        result = self.format_rules.format_single_month(date)
        assert result == "January 2025"
    
    def test_format_single_month_different_months(self):
        """Test single month formatting for different months."""
        test_cases = [
            (datetime(2025, 1, 1), "January 2025"),
            (datetime(2025, 6, 1), "June 2025"),
            (datetime(2025, 12, 1), "December 2025"),
            (datetime(2024, 2, 1), "February 2024"),  # Different year
        ]
        
        for date, expected in test_cases:
            result = self.format_rules.format_single_month(date)
            assert result == expected
    
    def test_format_single_quarter_template(self):
        """Test single quarter formatting using template."""
        date = datetime(2025, 1, 1)
        result = self.format_rules.format_single_quarter(date, 1)
        assert result == "Q1 2025"
    
    def test_format_single_quarter_all_quarters(self):
        """Test single quarter formatting for all quarters."""
        test_cases = [
            (datetime(2025, 1, 1), 1, "Q1 2025"),
            (datetime(2025, 4, 1), 2, "Q2 2025"),
            (datetime(2025, 7, 1), 3, "Q3 2025"),
            (datetime(2025, 10, 1), 4, "Q4 2025"),
            (datetime(2024, 1, 1), 1, "Q1 2024"),  # Different year
        ]
        
        for date, quarter, expected in test_cases:
            result = self.format_rules.format_single_quarter(date, quarter)
            assert result == expected
    
    def test_format_single_year_template(self):
        """Test single year formatting using template."""
        date = datetime(2025, 1, 1)
        result = self.format_rules.format_single_year(date)
        assert result == "2025"
    
    def test_format_single_year_different_years(self):
        """Test single year formatting for different years."""
        test_cases = [
            (datetime(2025, 1, 1), "2025"),
            (datetime(2024, 1, 1), "2024"),
            (datetime(2026, 1, 1), "2026"),
            (datetime(2023, 1, 1), "2023"),
        ]
        
        for date, expected in test_cases:
            result = self.format_rules.format_single_year(date)
            assert result == expected
    
    def test_format_multi_month_same_year(self):
        """Test multi-month formatting within same year."""
        start_date = datetime(2025, 1, 1)
        end_date = datetime(2025, 3, 31)  # Last day of March
        result = self.format_rules.format_multi_month(start_date, end_date)
        assert result == "January - March 2025"
    
    def test_format_multi_month_cross_year(self):
        """Test multi-month formatting crossing year boundary."""
        start_date = datetime(2024, 11, 1)
        end_date = datetime(2025, 2, 28)  # Last day of February
        result = self.format_rules.format_multi_month(start_date, end_date)
        assert result == "November 2024 - February 2025"
    
    def test_format_multi_month_various_ranges(self):
        """Test multi-month formatting for various ranges."""
        test_cases = [
            # Same year ranges
            (datetime(2025, 1, 1), datetime(2025, 5, 31), "January - May 2025"),
            (datetime(2025, 6, 1), datetime(2025, 8, 31), "June - August 2025"),
            (datetime(2025, 9, 1), datetime(2025, 11, 30), "September - November 2025"),
            
            # Cross-year ranges
            (datetime(2024, 10, 1), datetime(2025, 1, 31), "October 2024 - January 2025"),
            (datetime(2024, 12, 1), datetime(2025, 3, 31), "December 2024 - March 2025"),
        ]
        
        for start_date, end_date, expected in test_cases:
            result = self.format_rules.format_multi_month(start_date, end_date)
            assert result == expected
    
    def test_format_custom_range_same_month(self):
        """Test custom range formatting within same month."""
        start_date = datetime(2025, 1, 15)
        end_date = datetime(2025, 1, 25)
        result = self.format_rules.format_custom_range(start_date, end_date)
        assert result == "January 15 - 25, 2025"
    
    def test_format_custom_range_same_year(self):
        """Test custom range formatting within same year, different months."""
        start_date = datetime(2025, 1, 15)
        end_date = datetime(2025, 2, 10)
        result = self.format_rules.format_custom_range(start_date, end_date)
        assert result == "January 15 - February 10, 2025"
    
    def test_format_custom_range_cross_year(self):
        """Test custom range formatting crossing year boundary."""
        start_date = datetime(2024, 12, 15)
        end_date = datetime(2025, 1, 10)
        result = self.format_rules.format_custom_range(start_date, end_date)
        assert result == "December 15, 2024 - January 10, 2025"
    
    def test_format_custom_range_various_scenarios(self):
        """Test custom range formatting for various scenarios."""
        test_cases = [
            # Same month, different days
            (datetime(2025, 3, 5), datetime(2025, 3, 20), "March 5 - 20, 2025"),
            (datetime(2025, 12, 1), datetime(2025, 12, 31), "December 1 - 31, 2025"),
            
            # Same year, different months
            (datetime(2025, 2, 14), datetime(2025, 4, 1), "February 14 - April 1, 2025"),
            (datetime(2025, 7, 4), datetime(2025, 9, 15), "July 4 - September 15, 2025"),
            
            # Cross year
            (datetime(2024, 11, 20), datetime(2025, 2, 5), "November 20, 2024 - February 5, 2025"),
            (datetime(2024, 12, 31), datetime(2025, 1, 1), "December 31, 2024 - January 1, 2025"),
        ]
        
        for start_date, end_date, expected in test_cases:
            result = self.format_rules.format_custom_range(start_date, end_date)
            assert result == expected
    
    def test_get_month_name_full(self):
        """Test getting full month names."""
        test_cases = [
            (1, "January"), (2, "February"), (3, "March"), (4, "April"),
            (5, "May"), (6, "June"), (7, "July"), (8, "August"),
            (9, "September"), (10, "October"), (11, "November"), (12, "December")
        ]
        
        for month_num, expected in test_cases:
            result = self.format_rules.get_month_name(month_num)
            assert result == expected
    
    def test_get_month_name_short(self):
        """Test getting short month names."""
        test_cases = [
            (1, "Jan"), (2, "Feb"), (3, "Mar"), (4, "Apr"),
            (5, "May"), (6, "Jun"), (7, "Jul"), (8, "Aug"),
            (9, "Sep"), (10, "Oct"), (11, "Nov"), (12, "Dec")
        ]
        
        for month_num, expected in test_cases:
            result = self.format_rules.get_month_name(month_num, short=True)
            assert result == expected
    
    def test_get_month_name_invalid(self):
        """Test getting month name with invalid input."""
        with pytest.raises(ValueError):
            self.format_rules.get_month_name(0)
        
        with pytest.raises(ValueError):
            self.format_rules.get_month_name(13)
        
        with pytest.raises(ValueError):
            self.format_rules.get_month_name(-1)
    
    def test_template_constants(self):
        """Test that template constants are properly defined."""
        # Verify all template constants exist and are strings
        assert isinstance(self.format_rules.SINGLE_DAY, str)
        assert isinstance(self.format_rules.SINGLE_MONTH, str)
        assert isinstance(self.format_rules.SINGLE_QUARTER, str)
        assert isinstance(self.format_rules.SINGLE_YEAR, str)
        assert isinstance(self.format_rules.MULTI_MONTH_SAME_YEAR, str)
        assert isinstance(self.format_rules.MULTI_MONTH_CROSS_YEAR, str)
        assert isinstance(self.format_rules.CUSTOM_RANGE_SAME_MONTH, str)
        assert isinstance(self.format_rules.CUSTOM_RANGE_SAME_YEAR, str)
        assert isinstance(self.format_rules.CUSTOM_RANGE_CROSS_YEAR, str)
        
        # Verify templates contain expected placeholders
        assert "{month}" in self.format_rules.SINGLE_DAY
        assert "{day}" in self.format_rules.SINGLE_DAY
        assert "{year}" in self.format_rules.SINGLE_DAY
        
        assert "{month}" in self.format_rules.SINGLE_MONTH
        assert "{year}" in self.format_rules.SINGLE_MONTH
        
        assert "{quarter}" in self.format_rules.SINGLE_QUARTER
        assert "{year}" in self.format_rules.SINGLE_QUARTER
        
        assert "{year}" in self.format_rules.SINGLE_YEAR
    
    def test_edge_cases_leap_year(self):
        """Test formatting edge cases with leap year dates."""
        # February 29, 2024 (leap year)
        leap_date = datetime(2024, 2, 29)
        result = self.format_rules.format_single_day(leap_date)
        assert result == "February 29, 2024"
        
        # February month in leap year
        result = self.format_rules.format_single_month(leap_date)
        assert result == "February 2024"
    
    def test_edge_cases_year_boundaries(self):
        """Test formatting edge cases at year boundaries."""
        # December 31 to January 1 custom range
        start_date = datetime(2024, 12, 31)
        end_date = datetime(2025, 1, 1)
        result = self.format_rules.format_custom_range(start_date, end_date)
        assert result == "December 31, 2024 - January 1, 2025"
        
        # December to January multi-month
        start_date = datetime(2024, 12, 1)
        end_date = datetime(2025, 1, 31)
        result = self.format_rules.format_multi_month(start_date, end_date)
        assert result == "December 2024 - January 2025"


class TestDateFormatterIntegration:
    """Integration tests for DateFormatter with PeriodTypeDetector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = DateFormatter()
    
    def test_format_single_day(self):
        """Test formatting of single day periods."""
        start = datetime(2025, 1, 15)
        end = datetime(2025, 1, 16)
        time_period = TimePeriod(start=start, end=end)
        
        result = self.formatter.format_time_period(time_period)
        assert result == "January 15, 2025"
    
    def test_format_single_month(self):
        """Test formatting of single month periods."""
        start = datetime(2025, 1, 1)
        end = datetime(2025, 2, 1)
        time_period = TimePeriod(start=start, end=end)
        
        result = self.formatter.format_time_period(time_period)
        assert result == "January 2025"
    
    def test_format_single_quarter(self):
        """Test formatting of single quarter periods."""
        start = datetime(2025, 1, 1)
        end = datetime(2025, 4, 1)
        time_period = TimePeriod(start=start, end=end)
        
        result = self.formatter.format_time_period(time_period)
        assert result == "Q1 2025"
    
    def test_format_single_year(self):
        """Test formatting of single year periods."""
        start = datetime(2025, 1, 1)
        end = datetime(2026, 1, 1)
        time_period = TimePeriod(start=start, end=end)
        
        result = self.formatter.format_time_period(time_period)
        assert result == "2025"
    
    def test_format_multi_month(self):
        """Test formatting of multi-month periods."""
        start = datetime(2024, 11, 1)
        end = datetime(2025, 3, 1)
        time_period = TimePeriod(start=start, end=end)
        
        result = self.formatter.format_time_period(time_period)
        assert result == "November 2024 - February 2025"
    
    def test_format_custom_range(self):
        """Test formatting of custom range periods."""
        start = datetime(2025, 1, 15)
        end = datetime(2025, 2, 20)
        time_period = TimePeriod(start=start, end=end)
        
        result = self.formatter.format_time_period(time_period)
        assert result == "January 15 - February 19, 2025"  # Same year, different months format
    
    def test_safe_format_with_error(self):
        """Test safe formatting with error handling."""
        # Create an invalid time period that might cause errors
        time_period = TimePeriod(start=None, end=None)
        
        result = self.formatter.safe_format_time_period(time_period)
        assert result == "Invalid date range"
    
    def test_format_time_period_comprehensive_scenarios(self):
        """Test comprehensive formatting scenarios including edge cases."""
        test_cases = [
            # Normal cases
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
    """Test cases specifically for error handling and fallback mechanisms."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = DateFormatter()
    
    def test_safe_format_none_time_period(self):
        """Test safe formatting with None time period."""
        result = self.formatter.safe_format_time_period(None)
        assert result == "Invalid date range"
    
    def test_safe_format_none_start_date(self):
        """Test safe formatting with None start date."""
        time_period = TimePeriod(start=None, end=datetime(2025, 1, 1))
        result = self.formatter.safe_format_time_period(time_period)
        assert result == "Invalid date range"
    
    def test_safe_format_none_end_date(self):
        """Test safe formatting with None end date."""
        time_period = TimePeriod(start=datetime(2025, 1, 1), end=None)
        result = self.formatter.safe_format_time_period(time_period)
        assert result == "Invalid date range"
    
    def test_safe_format_both_dates_none(self):
        """Test safe formatting with both dates None."""
        time_period = TimePeriod(start=None, end=None)
        result = self.formatter.safe_format_time_period(time_period)
        assert result == "Invalid date range"
    
    def test_safe_format_missing_attributes(self):
        """Test safe formatting with object missing required attributes."""
        # Create a mock object without start/end attributes
        class MockTimePeriod:
            pass
        
        mock_period = MockTimePeriod()
        result = self.formatter.safe_format_time_period(mock_period)
        assert result == "Invalid date range"
    
    def test_safe_format_invalid_date_objects(self):
        """Test safe formatting with invalid date objects."""
        # Create a mock time period with non-datetime objects
        class MockDate:
            def __init__(self, should_fail=False):
                self.should_fail = should_fail
            
            def strftime(self, fmt):
                if self.should_fail:
                    raise ValueError("Mock strftime error")
                return "2025-01-01"
            
            def __str__(self):
                if self.should_fail:
                    raise ValueError("Mock str error")
                return "2025-01-01"
        
        # Test with dates that fail strftime but succeed with str
        time_period = TimePeriod(start=MockDate(should_fail=False), end=MockDate(should_fail=False))
        result = self.formatter.safe_format_time_period(time_period)
        assert "2025-01-01 to 2025-01-01" in result
    
    def test_safe_format_extreme_fallback(self):
        """Test safe formatting when all fallback strategies fail."""
        class FailingDate:
            def strftime(self, fmt):
                raise ValueError("strftime failed")
            
            def __str__(self):
                raise ValueError("str failed")
        
        time_period = TimePeriod(start=FailingDate(), end=FailingDate())
        result = self.formatter.safe_format_time_period(time_period)
        assert result == "Invalid date range"
    
    def test_format_time_period_with_none_input(self):
        """Test format_time_period raises appropriate exception for None input."""
        with pytest.raises(ValueError, match="time_period cannot be None"):
            self.formatter.format_time_period(None)
    
    def test_format_time_period_missing_attributes(self):
        """Test format_time_period raises appropriate exception for missing attributes."""
        class MockTimePeriod:
            pass
        
        mock_period = MockTimePeriod()
        with pytest.raises(AttributeError, match="TimePeriod must have start and end attributes"):
            self.formatter.format_time_period(mock_period)
    
    def test_format_time_period_none_dates(self):
        """Test format_time_period raises appropriate exception for None dates."""
        time_period = TimePeriod(start=None, end=None)
        with pytest.raises(ValueError, match="TimePeriod start and end cannot be None"):
            self.formatter.format_time_period(time_period)
    
    def test_fallback_format_with_valid_dates(self):
        """Test fallback format with valid dates."""
        time_period = TimePeriod(start=datetime(2025, 1, 15), end=datetime(2025, 1, 20))
        result = self.formatter._fallback_format(time_period)
        assert result == "2025-01-15 to 2025-01-20"
    
    def test_fallback_format_with_none_time_period(self):
        """Test fallback format with None time period."""
        result = self.formatter._fallback_format(None)
        assert result == "Invalid date range"
    
    def test_fallback_format_with_none_dates(self):
        """Test fallback format with None dates."""
        time_period = TimePeriod(start=None, end=None)
        result = self.formatter._fallback_format(time_period)
        assert result == "Invalid date range"
    
    def test_safe_fallback_format_with_valid_dates(self):
        """Test safe fallback format with valid dates."""
        time_period = TimePeriod(start=datetime(2025, 1, 15), end=datetime(2025, 1, 20))
        result = self.formatter._safe_fallback_format(time_period)
        assert result == "2025-01-15 to 2025-01-20"
    
    def test_safe_fallback_format_with_string_representation(self):
        """Test safe fallback format falls back to string representation."""
        class MockDate:
            def strftime(self, fmt):
                raise ValueError("strftime failed")
            
            def __str__(self):
                return "Mock Date String"
        
        time_period = TimePeriod(start=MockDate(), end=MockDate())
        result = self.formatter._safe_fallback_format(time_period)
        assert result == "Mock Date String to Mock Date String"
    
    def test_safe_fallback_format_complete_failure(self):
        """Test safe fallback format when everything fails."""
        class FailingDate:
            def strftime(self, fmt):
                raise ValueError("strftime failed")
            
            def __str__(self):
                raise ValueError("str failed")
        
        time_period = TimePeriod(start=FailingDate(), end=FailingDate())
        result = self.formatter._safe_fallback_format(time_period)
        assert result == "Invalid date range"
    
    def test_format_rules_error_handling_invalid_month(self):
        """Test FormatRules error handling with invalid month."""
        class MockDate:
            month = 13  # Invalid month
            day = 1
            year = 2025
        
        with pytest.raises(ValueError, match="Invalid month"):
            self.formatter.format_rules.format_single_day(MockDate())
    
    def test_format_rules_error_handling_missing_attributes(self):
        """Test FormatRules error handling with missing attributes."""
        class MockDate:
            pass  # No attributes
        
        with pytest.raises(ValueError, match="Invalid date object"):
            self.formatter.format_rules.format_single_day(MockDate())
    
    def test_format_rules_error_handling_none_date(self):
        """Test FormatRules error handling with None date."""
        with pytest.raises(ValueError, match="Invalid date object"):
            self.formatter.format_rules.format_single_day(None)
    
    def test_format_rules_quarter_invalid_quarter_number(self):
        """Test FormatRules error handling with invalid quarter number."""
        date = datetime(2025, 1, 1)
        
        with pytest.raises(ValueError, match="Invalid quarter"):
            self.formatter.format_rules.format_single_quarter(date, 5)  # Invalid quarter
        
        with pytest.raises(ValueError, match="Invalid quarter"):
            self.formatter.format_rules.format_single_quarter(date, 0)  # Invalid quarter
    
    def test_format_rules_multi_month_none_dates(self):
        """Test FormatRules multi-month formatting with None dates."""
        with pytest.raises(ValueError, match="Start and end dates cannot be None"):
            self.formatter.format_rules.format_multi_month(None, datetime(2025, 1, 1))
        
        with pytest.raises(ValueError, match="Start and end dates cannot be None"):
            self.formatter.format_rules.format_multi_month(datetime(2025, 1, 1), None)
    
    def test_format_rules_custom_range_invalid_months(self):
        """Test FormatRules custom range formatting with invalid months."""
        class MockDate:
            def __init__(self, month, day, year):
                self.month = month
                self.day = day
                self.year = year
        
        start_date = MockDate(13, 1, 2025)  # Invalid month
        end_date = MockDate(1, 15, 2025)
        
        with pytest.raises(ValueError, match="Invalid months"):
            self.formatter.format_rules.format_custom_range(start_date, end_date)
    
    def test_period_type_detector_error_handling(self):
        """Test PeriodTypeDetector error handling."""
        # Test with None
        result = self.formatter.detector.detect_period_type(None)
        assert result == PeriodType.CUSTOM_RANGE
        
        # Test with invalid time period (start after end)
        time_period = TimePeriod(start=datetime(2025, 2, 1), end=datetime(2025, 1, 1))
        result = self.formatter.detector.detect_period_type(time_period)
        assert result == PeriodType.CUSTOM_RANGE
        
        # Test with extreme dates
        time_period = TimePeriod(start=datetime(1800, 1, 1), end=datetime(1800, 2, 1))
        result = self.formatter.detector.detect_period_type(time_period)
        assert result == PeriodType.CUSTOM_RANGE
    
    def test_integration_error_recovery(self):
        """Test end-to-end error recovery in realistic scenarios."""
        # Test with a time period that causes errors in detection but recovers
        class ProblematicTimePeriod:
            def __init__(self):
                self.start = datetime(2025, 1, 1)
                self.end = datetime(2025, 2, 1)
        
        # This should work fine despite being a custom class
        time_period = ProblematicTimePeriod()
        result = self.formatter.safe_format_time_period(time_period)
        assert result == "January 2025"  # Should detect as single month
    
    def test_logging_behavior(self, caplog):
        """Test that appropriate log messages are generated during error handling."""
        import logging
        
        # Set log level to capture warnings
        caplog.set_level(logging.WARNING)
        
        # Test with None time period
        self.formatter.safe_format_time_period(None)
        assert "Received None time_period in safe_format_time_period" in caplog.text
        
        # Clear logs
        caplog.clear()
        
        # Test with invalid time period
        time_period = TimePeriod(start=None, end=None)
        self.formatter.safe_format_time_period(time_period)
        assert "TimePeriod has None start or end dates" in caplog.text
    
    def test_edge_case_recovery_scenarios(self):
        """Test recovery from various edge case scenarios."""
        # Test with microseconds and complex time components
        start = datetime(2025, 1, 15, 14, 30, 45, 123456)
        end = datetime(2025, 1, 16, 16, 45, 30, 654321)
        time_period = TimePeriod(start=start, end=end)
        
        result = self.formatter.safe_format_time_period(time_period)
        assert result == "January 15, 2025"
        
        # Test with leap year edge case
        start = datetime(2024, 2, 29)  # Leap year
        end = datetime(2024, 3, 1)
        time_period = TimePeriod(start=start, end=end)
        
        result = self.formatter.safe_format_time_period(time_period)
        assert result == "February 29, 2024"
        
        # Test with year boundary
        start = datetime(2024, 12, 31, 23, 59, 59)
        end = datetime(2025, 1, 1, 0, 0, 1)
        time_period = TimePeriod(start=start, end=end)
        
        result = self.formatter.safe_format_time_period(time_period)
        assert result == "December 31, 2024"  # Should be detected as single day
        result = self.formatter.safe_format_time_period(time_period)
        # Should fall back to a safe format without crashing
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_format_rules_integration_single_day(self):
        """Test that DateFormatter correctly uses FormatRules for single day."""
        start = datetime(2025, 3, 15)
        end = datetime(2025, 3, 16)
        time_period = TimePeriod(start=start, end=end)
        
        result = self.formatter.format_time_period(time_period)
        expected = self.formatter.format_rules.format_single_day(start)
        assert result == expected
        assert result == "March 15, 2025"
    
    def test_format_rules_integration_single_month(self):
        """Test that DateFormatter correctly uses FormatRules for single month."""
        start = datetime(2025, 6, 1)
        end = datetime(2025, 7, 1)
        time_period = TimePeriod(start=start, end=end)
        
        result = self.formatter.format_time_period(time_period)
        expected = self.formatter.format_rules.format_single_month(start)
        assert result == expected
        assert result == "June 2025"
    
    def test_format_rules_integration_single_quarter(self):
        """Test that DateFormatter correctly uses FormatRules for single quarter."""
        start = datetime(2025, 7, 1)
        end = datetime(2025, 10, 1)
        time_period = TimePeriod(start=start, end=end)
        
        result = self.formatter.format_time_period(time_period)
        # Q3 2025
        expected = self.formatter.format_rules.format_single_quarter(start, 3)
        assert result == expected
        assert result == "Q3 2025"
    
    def test_format_rules_integration_single_year(self):
        """Test that DateFormatter correctly uses FormatRules for single year."""
        start = datetime(2025, 1, 1)
        end = datetime(2026, 1, 1)
        time_period = TimePeriod(start=start, end=end)
        
        result = self.formatter.format_time_period(time_period)
        expected = self.formatter.format_rules.format_single_year(start)
        assert result == expected
        assert result == "2025"
    
    def test_format_rules_integration_multi_month_same_year(self):
        """Test that DateFormatter correctly uses FormatRules for multi-month same year."""
        start = datetime(2025, 2, 1)
        end = datetime(2025, 6, 1)  # Feb-May (exclusive end)
        time_period = TimePeriod(start=start, end=end)
        
        result = self.formatter.format_time_period(time_period)
        # Should format as "February - May 2025"
        assert result == "February - May 2025"
    
    def test_format_rules_integration_multi_month_cross_year(self):
        """Test that DateFormatter correctly uses FormatRules for multi-month cross year."""
        start = datetime(2024, 11, 1)
        end = datetime(2025, 3, 1)  # Nov-Feb (exclusive end)
        time_period = TimePeriod(start=start, end=end)
        
        result = self.formatter.format_time_period(time_period)
        # Should format as "November 2024 - February 2025"
        assert result == "November 2024 - February 2025"
    
    def test_format_rules_integration_custom_range_same_month(self):
        """Test that DateFormatter correctly uses FormatRules for custom range same month."""
        start = datetime(2025, 4, 10)
        end = datetime(2025, 4, 20)  # Exclusive end, so actual end is April 19
        time_period = TimePeriod(start=start, end=end)
        
        result = self.formatter.format_time_period(time_period)
        # Should format as "April 10 - 19, 2025"
        assert result == "April 10 - 19, 2025"
    
    def test_format_rules_integration_custom_range_same_year(self):
        """Test that DateFormatter correctly uses FormatRules for custom range same year."""
        start = datetime(2025, 1, 15)
        end = datetime(2025, 3, 10)  # Exclusive end, so actual end is March 9
        time_period = TimePeriod(start=start, end=end)
        
        result = self.formatter.format_time_period(time_period)
        # Should format as "January 15 - March 9, 2025"
        assert result == "January 15 - March 9, 2025"
    
    def test_format_rules_integration_custom_range_cross_year(self):
        """Test that DateFormatter correctly uses FormatRules for custom range cross year."""
        start = datetime(2024, 12, 20)
        end = datetime(2025, 1, 15)  # Exclusive end, so actual end is January 14
        time_period = TimePeriod(start=start, end=end)
        
        result = self.formatter.format_time_period(time_period)
        # Should format as "December 20, 2024 - January 14, 2025"
        assert result == "December 20, 2024 - January 14, 2025"
    
    def test_backward_compatibility_month_names(self):
        """Test that backward compatibility is maintained for month_names attribute."""
        # Ensure the old month_names attribute still works
        assert hasattr(self.formatter, 'month_names')
        assert len(self.formatter.month_names) == 12
        assert self.formatter.month_names[0] == "January"
        assert self.formatter.month_names[11] == "December"
        
        # Ensure it's the same as format_rules.month_names
        assert self.formatter.month_names == self.formatter.format_rules.month_names
    
    def test_comprehensive_formatting_scenarios(self):
        """Test comprehensive formatting scenarios covering all requirements."""
        test_cases = [
            # Requirement 1.1: Monthly periods as "January 2025"
            (datetime(2025, 1, 1), datetime(2025, 2, 1), "January 2025"),
            (datetime(2025, 12, 1), datetime(2026, 1, 1), "December 2025"),
            
            # Requirement 1.2: Yearly periods as "2025"
            (datetime(2025, 1, 1), datetime(2026, 1, 1), "2025"),
            (datetime(2024, 1, 1), datetime(2025, 1, 1), "2024"),
            
            # Requirement 1.3: Quarterly periods as "Q1 2025"
            (datetime(2025, 1, 1), datetime(2025, 4, 1), "Q1 2025"),
            (datetime(2025, 4, 1), datetime(2025, 7, 1), "Q2 2025"),
            (datetime(2025, 7, 1), datetime(2025, 10, 1), "Q3 2025"),
            (datetime(2025, 10, 1), datetime(2026, 1, 1), "Q4 2025"),
            
            # Requirement 1.4: Daily periods as "January 15, 2025"
            (datetime(2025, 1, 15), datetime(2025, 1, 16), "January 15, 2025"),
            (datetime(2025, 12, 31), datetime(2026, 1, 1), "December 31, 2025"),
            
            # Requirement 2.4: Multi-month ranges
            (datetime(2025, 1, 1), datetime(2025, 4, 1), "Q1 2025"),  # Should be detected as quarter
            (datetime(2025, 1, 1), datetime(2025, 6, 1), "January - May 2025"),  # Multi-month
            (datetime(2024, 11, 1), datetime(2025, 2, 1), "November 2024 - January 2025"),  # Cross-year
            
            # Requirement 2.5: Custom ranges
            (datetime(2025, 1, 15), datetime(2025, 2, 10), "January 15 - February 9, 2025"),
            (datetime(2024, 12, 15), datetime(2025, 1, 10), "December 15, 2024 - January 9, 2025"),
        ]
        
        for start, end, expected in test_cases:
            time_period = TimePeriod(start=start, end=end)
            result = self.formatter.format_time_period(time_period)
            assert result == expected, f"Failed for {start} to {end}: got '{result}', expected '{expected}'"

class TestDateFormatterEdgeCases:
    """Test edge cases for DateFormatter including month boundaries, year boundaries, leap years."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = DateFormatter()
    
    def test_month_boundary_edge_cases(self):
        """Test edge cases at month boundaries."""
        # January 31 to February 1 (month boundary)
        start = datetime(2025, 1, 31, 23, 59, 59)
        end = datetime(2025, 2, 1, 0, 0, 0)
        time_period = TimePeriod(start=start, end=end)
        
        result = self.formatter.format_time_period(time_period)
        assert result == "January 31, 2025"  # Should be detected as single day
    
    def test_year_boundary_edge_cases(self):
        """Test edge cases at year boundaries."""
        # December 31, 2024 to January 1, 2025
        start = datetime(2024, 12, 31)
        end = datetime(2025, 1, 1)
        time_period = TimePeriod(start=start, end=end)
        
        result = self.formatter.format_time_period(time_period)
        assert result == "December 31, 2024"  # Should be detected as single day
        
        # December 2024 to January 2025 (cross-year multi-month)
        start = datetime(2024, 12, 1)
        end = datetime(2025, 2, 1)
        time_period = TimePeriod(start=start, end=end)
        
        result = self.formatter.format_time_period(time_period)
        assert "December 2024" in result and "January 2025" in result
    
    def test_leap_year_edge_cases(self):
        """Test edge cases with leap years."""
        # February 29, 2024 (leap year)
        start = datetime(2024, 2, 29)
        end = datetime(2024, 3, 1)
        time_period = TimePeriod(start=start, end=end)
        
        result = self.formatter.format_time_period(time_period)
        assert result == "February 29, 2024"
        
        # February 2024 (leap year month)
        start = datetime(2024, 2, 1)
        end = datetime(2024, 3, 1)
        time_period = TimePeriod(start=start, end=end)
        
        result = self.formatter.format_time_period(time_period)
        assert "February" in result and "2024" in result
        
        # February 2025 (non-leap year month)
        start = datetime(2025, 2, 1)
        end = datetime(2025, 3, 1)
        time_period = TimePeriod(start=start, end=end)
        
        result = self.formatter.format_time_period(time_period)
        assert "February" in result and "2025" in result
    
    def test_partial_month_edge_cases(self):
        """Test edge cases with partial months."""
        # Mid-month to mid-month (same month)
        start = datetime(2025, 1, 15)
        end = datetime(2025, 1, 25)
        time_period = TimePeriod(start=start, end=end)
        
        result = self.formatter.format_time_period(time_period)
        assert "January 15" in result and "24, 2025" in result  # End date adjusted for exclusive end
        
        # Mid-month to mid-month (different months)
        start = datetime(2025, 1, 15)
        end = datetime(2025, 3, 15)
        time_period = TimePeriod(start=start, end=end)
        
        result = self.formatter.format_time_period(time_period)
        assert "January 15" in result and "March 14" in result and "2025" in result
    
    def test_quarter_boundary_edge_cases(self):
        """Test edge cases at quarter boundaries."""
        # Q1 to Q2 boundary (March 31 to April 1)
        start = datetime(2025, 3, 31)
        end = datetime(2025, 4, 1)
        time_period = TimePeriod(start=start, end=end)
        
        result = self.formatter.format_time_period(time_period)
        assert result == "March 31, 2025"  # Should be single day
        
        # Partial quarter (February to April)
        start = datetime(2025, 2, 1)
        end = datetime(2025, 5, 1)
        time_period = TimePeriod(start=start, end=end)
        
        result = self.formatter.format_time_period(time_period)
        assert "February" in result and "April" in result and "2025" in result
    
    def test_fiscal_year_edge_cases(self):
        """Test edge cases with fiscal years."""
        # Create detector with April fiscal year start
        fiscal_detector = PeriodTypeDetector(fiscal_year_start_month=4)
        formatter_fiscal = DateFormatter()
        formatter_fiscal.detector = fiscal_detector
        
        # Fiscal year 2025 (April 2025 - March 2026)
        start = datetime(2025, 4, 1)
        end = datetime(2026, 4, 1)
        time_period = TimePeriod(start=start, end=end)
        
        result = formatter_fiscal.format_time_period(time_period)
        assert "2025" in result  # Should be detected as fiscal year
        
        # Fiscal quarter (April-June 2025)
        start = datetime(2025, 4, 1)
        end = datetime(2025, 7, 1)
        time_period = TimePeriod(start=start, end=end)
        
        result = formatter_fiscal.format_time_period(time_period)
        assert "Q1 2025" in result or ("April" in result and "June" in result)
    
    def test_extreme_date_ranges(self):
        """Test extreme date ranges."""
        # Very short range (1 second)
        start = datetime(2025, 1, 15, 12, 0, 0)
        end = datetime(2025, 1, 15, 12, 0, 1)
        time_period = TimePeriod(start=start, end=end)
        
        result = self.formatter.format_time_period(time_period)
        assert result == "January 15, 2025"  # Should be detected as same day
        
        # Very long range (multiple years)
        start = datetime(2020, 1, 1)
        end = datetime(2025, 1, 1)
        time_period = TimePeriod(start=start, end=end)
        
        result = self.formatter.format_time_period(time_period)
        assert "January 1, 2020" in result and "December 31, 2024" in result
    
    def test_timezone_edge_cases(self):
        """Test edge cases with timezone-aware datetimes."""
        from datetime import timezone
        
        # Same day in different timezones
        start = datetime(2025, 1, 15, 23, 0, 0, tzinfo=timezone.utc)
        end = datetime(2025, 1, 16, 1, 0, 0, tzinfo=timezone.utc)
        time_period = TimePeriod(start=start, end=end)
        
        result = self.formatter.format_time_period(time_period)
        # Should still be detected as single day (ignores time components)
        assert result == "January 15, 2025"


class TestDateFormatterErrorHandling:
    """Test error handling and fallback mechanisms."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = DateFormatter()
    
    def test_none_time_period(self):
        """Test handling of None time period."""
        result = self.formatter.safe_format_time_period(None)
        assert result == "Invalid date range"
    
    def test_invalid_time_period_structure(self):
        """Test handling of invalid TimePeriod structure."""
        # TimePeriod with None start
        time_period = TimePeriod(start=None, end=datetime(2025, 1, 1))
        result = self.formatter.safe_format_time_period(time_period)
        assert result == "Invalid date range"
        
        # TimePeriod with None end
        time_period = TimePeriod(start=datetime(2025, 1, 1), end=None)
        result = self.formatter.safe_format_time_period(time_period)
        assert result == "Invalid date range"
    
    def test_invalid_date_order(self):
        """Test handling of invalid date order (start after end)."""
        start = datetime(2025, 2, 1)
        end = datetime(2025, 1, 1)
        time_period = TimePeriod(start=start, end=end)
        
        result = self.formatter.safe_format_time_period(time_period)
        # Should fall back to ISO format
        assert "2025-02-01 to 2025-01-01" in result
    
    def test_extreme_dates(self):
        """Test handling of extreme dates."""
        # Very old date
        start = datetime(1800, 1, 1)
        end = datetime(1800, 2, 1)
        time_period = TimePeriod(start=start, end=end)
        
        result = self.formatter.safe_format_time_period(time_period)
        # Should fall back to ISO format for extreme dates
        assert "1800-01-01 to 1800-02-01" in result
        
        # Very future date
        start = datetime(2100, 1, 1)
        end = datetime(2100, 2, 1)
        time_period = TimePeriod(start=start, end=end)
        
        result = self.formatter.safe_format_time_period(time_period)
        assert "2100-01-01 to 2100-02-01" in result
    
    def test_malformed_datetime_objects(self):
        """Test handling of malformed datetime objects."""
        # This test simulates what might happen with corrupted data
        import unittest.mock
        
        # Mock a datetime that raises an exception when accessed
        with unittest.mock.patch.object(datetime, 'strftime', side_effect=ValueError("Invalid date")):
            start = datetime(2025, 1, 1)
            end = datetime(2025, 2, 1)
            time_period = TimePeriod(start=start, end=end)
            
            result = self.formatter.safe_format_time_period(time_period)
            # Should handle the error gracefully
            assert isinstance(result, str)
            assert len(result) > 0
    
    def test_format_rules_error_handling(self):
        """Test error handling in FormatRules methods."""
        format_rules = FormatRules()
        
        # Test with invalid date object
        with pytest.raises(ValueError):
            format_rules.format_single_day(None)
        
        with pytest.raises(ValueError):
            format_rules.format_single_month(None)
        
        with pytest.raises(ValueError):
            format_rules.format_single_quarter(None, 1)
        
        with pytest.raises(ValueError):
            format_rules.format_single_year(None)
        
        # Test with invalid quarter number
        with pytest.raises(ValueError):
            format_rules.format_single_quarter(datetime(2025, 1, 1), 5)
        
        with pytest.raises(ValueError):
            format_rules.format_single_quarter(datetime(2025, 1, 1), 0)
    
    def test_period_detector_error_handling(self):
        """Test error handling in PeriodTypeDetector methods."""
        detector = PeriodTypeDetector()
        
        # Test with invalid month numbers
        with pytest.raises(ValueError):
            detector.get_quarter_number(0)
        
        with pytest.raises(ValueError):
            detector.get_quarter_number(13)
        
        with pytest.raises(ValueError):
            detector.get_fiscal_quarter_number(0)
        
        with pytest.raises(ValueError):
            detector.get_fiscal_quarter_number(13)
    
    def test_graceful_degradation(self):
        """Test graceful degradation when formatting fails."""
        # Create a scenario where formatting might fail
        start = datetime(2025, 1, 1)
        end = datetime(2025, 2, 1)
        time_period = TimePeriod(start=start, end=end)
        
        # Mock the format_time_period to raise an exception
        import unittest.mock
        with unittest.mock.patch.object(self.formatter, 'format_time_period', side_effect=Exception("Formatting error")):
            result = self.formatter.safe_format_time_period(time_period)
            # Should fall back to ISO format
            assert "2025-01-01 to 2025-02-01" in result
    
    def test_fallback_format_edge_cases(self):
        """Test edge cases in fallback formatting."""
        # Test fallback with None time period
        result = self.formatter._fallback_format(None)
        assert result == "Invalid date range"
        
        # Test fallback with invalid time period
        time_period = TimePeriod(start=None, end=None)
        result = self.formatter._fallback_format(time_period)
        assert result == "Invalid date range"
        
        # Test fallback when strftime fails
        import unittest.mock
        start = datetime(2025, 1, 1)
        end = datetime(2025, 2, 1)
        time_period = TimePeriod(start=start, end=end)
        
        with unittest.mock.patch.object(datetime, 'strftime', side_effect=ValueError("strftime error")):
            result = self.formatter._fallback_format(time_period)
            assert result == "Invalid date range"


class TestDateFormatterPerformance:
    """Test performance characteristics of DateFormatter with large datasets."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = DateFormatter()
    
    def test_performance_with_many_periods(self):
        """Test performance with many time periods."""
        import time
        
        # Generate 1000 different time periods
        periods = []
        for i in range(1000):
            start = datetime(2024, 1, 1) + timedelta(days=i)
            end = start + timedelta(days=1)
            periods.append(TimePeriod(start=start, end=end))
        
        # Measure formatting time
        start_time = time.time()
        results = []
        for period in periods:
            result = self.formatter.format_time_period(period)
            results.append(result)
        end_time = time.time()
        
        # Should complete in reasonable time (less than 1 second for 1000 periods)
        elapsed_time = end_time - start_time
        assert elapsed_time < 1.0, f"Formatting took too long: {elapsed_time:.2f} seconds"
        
        # All results should be valid
        assert len(results) == 1000
        assert all(isinstance(result, str) and len(result) > 0 for result in results)
    
    def test_performance_with_complex_periods(self):
        """Test performance with complex period types."""
        import time
        
        # Generate various complex period types
        periods = []
        
        # Add single days
        for month in range(1, 13):
            start = datetime(2024, month, 15)
            end = start + timedelta(days=1)
            periods.append(TimePeriod(start=start, end=end))
        
        # Add single months
        for month in range(1, 13):
            start = datetime(2024, month, 1)
            if month == 12:
                end = datetime(2025, 1, 1)
            else:
                end = datetime(2024, month + 1, 1)
            periods.append(TimePeriod(start=start, end=end))
        
        # Add quarters
        for quarter_start in [1, 4, 7, 10]:
            start = datetime(2024, quarter_start, 1)
            if quarter_start == 10:
                end = datetime(2025, 1, 1)
            else:
                end = datetime(2024, quarter_start + 3, 1)
            periods.append(TimePeriod(start=start, end=end))
        
        # Add custom ranges
        for i in range(50):
            start = datetime(2024, 1, 1) + timedelta(days=i * 7)
            end = start + timedelta(days=10)
            periods.append(TimePeriod(start=start, end=end))
        
        # Measure formatting time
        start_time = time.time()
        results = []
        for period in periods:
            result = self.formatter.format_time_period(period)
            results.append(result)
        end_time = time.time()
        
        # Should complete in reasonable time
        elapsed_time = end_time - start_time
        assert elapsed_time < 2.0, f"Complex formatting took too long: {elapsed_time:.2f} seconds"
        
        # All results should be valid
        assert len(results) == len(periods)
        assert all(isinstance(result, str) and len(result) > 0 for result in results)
    
    def test_memory_usage_with_large_datasets(self):
        """Test memory usage with large datasets."""
        import gc
        import sys
        
        # Force garbage collection before test
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Create and format many periods
        for i in range(1000):
            start = datetime(2024, 1, 1) + timedelta(days=i)
            end = start + timedelta(days=1)
            period = TimePeriod(start=start, end=end)
            result = self.formatter.format_time_period(period)
            # Don't store results to test memory cleanup
        
        # Force garbage collection after test
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Memory usage should not grow significantly
        object_growth = final_objects - initial_objects
        assert object_growth < 100, f"Too many objects created: {object_growth}"
    
    def test_caching_behavior(self):
        """Test that repeated formatting doesn't degrade performance."""
        import time
        
        # Create a single period
        start = datetime(2024, 1, 1)
        end = datetime(2024, 2, 1)
        period = TimePeriod(start=start, end=end)
        
        # Time first formatting
        start_time = time.time()
        result1 = self.formatter.format_time_period(period)
        first_time = time.time() - start_time
        
        # Time repeated formatting
        times = []
        for _ in range(100):
            start_time = time.time()
            result = self.formatter.format_time_period(period)
            times.append(time.time() - start_time)
            assert result == result1  # Should be consistent
        
        # Average time for repeated calls should be reasonable
        avg_time = sum(times) / len(times)
        assert avg_time < first_time * 2, f"Repeated calls too slow: {avg_time:.6f}s vs {first_time:.6f}s"


class TestDateFormatterWithRealAWSData:
    """Test DateFormatter with realistic AWS Cost Explorer API response data."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = DateFormatter()
    
    def test_aws_daily_granularity_format(self):
        """Test formatting with AWS daily granularity data."""
        # AWS returns daily data with exclusive end dates
        periods = []
        
        # January 2024 daily data (31 days)
        for day in range(1, 32):
            start = datetime(2024, 1, day)
            end = datetime(2024, 1, day) + timedelta(days=1)
            periods.append(TimePeriod(start=start, end=end))
        
        # Format all periods
        results = []
        for period in periods:
            result = self.formatter.format_time_period(period)
            results.append(result)
        
        # All should be formatted as single days
        assert len(results) == 31
        assert all("January" in result and "2024" in result for result in results)
        assert "January 1, 2024" in results
        assert "January 31, 2024" in results
    
    def test_aws_monthly_granularity_format(self):
        """Test formatting with AWS monthly granularity data."""
        # AWS returns monthly data with exclusive end dates
        periods = []
        
        # 2024 monthly data (12 months)
        for month in range(1, 13):
            start = datetime(2024, month, 1)
            if month == 12:
                end = datetime(2025, 1, 1)
            else:
                end = datetime(2024, month + 1, 1)
            periods.append(TimePeriod(start=start, end=end))
        
        # Format all periods
        results = []
        for period in periods:
            result = self.formatter.format_time_period(period)
            results.append(result)
        
        # All should be formatted as single months
        expected_months = [
            "January 2024", "February 2024", "March 2024", "April 2024",
            "May 2024", "June 2024", "July 2024", "August 2024",
            "September 2024", "October 2024", "November 2024", "December 2024"
        ]
        
        assert len(results) == 12
        for expected in expected_months:
            assert expected in results or any(expected.split()[0] in result and "2024" in result for result in results)
    
    def test_aws_quarterly_data_format(self):
        """Test formatting with quarterly data patterns."""
        # Simulate AWS quarterly data
        quarters = [
            (datetime(2024, 1, 1), datetime(2024, 4, 1)),  # Q1
            (datetime(2024, 4, 1), datetime(2024, 7, 1)),  # Q2
            (datetime(2024, 7, 1), datetime(2024, 10, 1)), # Q3
            (datetime(2024, 10, 1), datetime(2025, 1, 1)), # Q4
        ]
        
        results = []
        for start, end in quarters:
            period = TimePeriod(start=start, end=end)
            result = self.formatter.format_time_period(period)
            results.append(result)
        
        # Should be formatted as quarters or multi-month ranges
        assert len(results) == 4
        assert all("2024" in result for result in results)
        
        # Check that each quarter is properly formatted
        # Q1: January - March
        assert any("January" in result and "March" in result for result in results) or "Q1 2024" in results
        # Q4: October - December
        assert any("October" in result and "December" in result for result in results) or "Q4 2024" in results
    
    def test_aws_yearly_data_format(self):
        """Test formatting with yearly data patterns."""
        # Simulate AWS yearly data
        years = [
            (datetime(2022, 1, 1), datetime(2023, 1, 1)),
            (datetime(2023, 1, 1), datetime(2024, 1, 1)),
            (datetime(2024, 1, 1), datetime(2025, 1, 1)),
        ]
        
        results = []
        for start, end in years:
            period = TimePeriod(start=start, end=end)
            result = self.formatter.format_time_period(period)
            results.append(result)
        
        # Should be formatted as years or year ranges
        assert len(results) == 3
        expected_years = ["2022", "2023", "2024"]
        for year in expected_years:
            assert any(year in result for result in results)
    
    def test_aws_custom_date_ranges(self):
        """Test formatting with custom date ranges typical in AWS queries."""
        # Common AWS custom ranges
        custom_ranges = [
            # Last 7 days
            (datetime(2024, 1, 15), datetime(2024, 1, 22)),
            # Last 30 days
            (datetime(2024, 1, 1), datetime(2024, 1, 31)),
            # Billing period (partial month)
            (datetime(2024, 1, 15), datetime(2024, 2, 15)),
            # Quarter-to-date
            (datetime(2024, 1, 1), datetime(2024, 2, 15)),
            # Year-to-date
            (datetime(2024, 1, 1), datetime(2024, 6, 15)),
        ]
        
        results = []
        for start, end in custom_ranges:
            period = TimePeriod(start=start, end=end)
            result = self.formatter.format_time_period(period)
            results.append(result)
        
        # All should be formatted appropriately
        assert len(results) == 5
        assert all(isinstance(result, str) and len(result) > 0 for result in results)
        assert all("2024" in result for result in results)
        
        # Check specific formatting
        # Last 7 days should show date range
        assert "January 15" in results[0] and "21, 2024" in results[0]
        # Billing period should show cross-month range
        assert "January 15" in results[2] and "February 14" in results[2]
    
    def test_aws_timezone_handling(self):
        """Test formatting with timezone-aware datetimes from AWS."""
        from datetime import timezone
        
        # AWS typically returns UTC times
        utc_periods = [
            TimePeriod(
                start=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
                end=datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
            ),
            TimePeriod(
                start=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
                end=datetime(2024, 2, 1, 0, 0, 0, tzinfo=timezone.utc)
            ),
        ]
        
        results = []
        for period in utc_periods:
            result = self.formatter.format_time_period(period)
            results.append(result)
        
        # Should handle timezone-aware datetimes correctly
        assert results[0] == "January 1, 2024"  # Single day
        assert "January" in results[1] and "2024" in results[1]  # Single month
    
    def test_aws_cost_explorer_response_simulation(self):
        """Test with simulated AWS Cost Explorer response structure."""
        # Simulate a typical AWS Cost Explorer response with multiple time periods
        response_periods = []
        
        # Monthly breakdown for Q1 2024
        months = [
            (datetime(2024, 1, 1), datetime(2024, 2, 1)),
            (datetime(2024, 2, 1), datetime(2024, 3, 1)),
            (datetime(2024, 3, 1), datetime(2024, 4, 1)),
        ]
        
        for start, end in months:
            response_periods.append(TimePeriod(start=start, end=end))
        
        # Format all periods as they would appear in a response
        formatted_periods = []
        for period in response_periods:
            formatted = self.formatter.format_time_period(period)
            formatted_periods.append(formatted)
        
        # Verify proper formatting
        expected = ["January 2024", "February 2024", "March 2024"]
        for i, expected_format in enumerate(expected):
            assert expected_format.split()[0] in formatted_periods[i]
            assert "2024" in formatted_periods[i]
    
    def test_edge_cases_from_aws_data(self):
        """Test edge cases that might occur in real AWS data."""
        # Partial day at month boundary (AWS sometimes returns these)
        partial_day = TimePeriod(
            start=datetime(2024, 1, 31, 12, 0, 0),
            end=datetime(2024, 2, 1, 0, 0, 0)
        )
        result = self.formatter.format_time_period(partial_day)
        # Should handle gracefully
        assert isinstance(result, str) and len(result) > 0
        
        # Very short period (less than a day)
        short_period = TimePeriod(
            start=datetime(2024, 1, 15, 10, 0, 0),
            end=datetime(2024, 1, 15, 14, 0, 0)
        )
        result = self.formatter.format_time_period(short_period)
        assert result == "January 15, 2024"  # Should be detected as same day
        
        # Period spanning multiple years (for long-term analysis)
        long_period = TimePeriod(
            start=datetime(2022, 1, 1),
            end=datetime(2024, 12, 31)
        )
        result = self.formatter.format_time_period(long_period)
        assert "January 1, 2022" in result and "December 30, 2024" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])