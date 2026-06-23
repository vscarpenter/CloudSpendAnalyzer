"""Tests for DateFormatter configuration support."""

import pytest
from datetime import datetime
from unittest.mock import patch

from src.aws_cost_cli.date_formatter import DateFormatter, PeriodTypeDetector, FormatRules
from src.aws_cost_cli.models import TimePeriod, DateFormattingConfig, DateFormatStyle


class TestDateFormattingConfig:
    """Test DateFormattingConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DateFormattingConfig()
        
        assert config.enabled is True
        assert config.format_style == DateFormatStyle.SMART
        assert config.fiscal_year_start_month == 1
        assert config.locale == "en_US"
        assert config.fallback_to_iso is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = DateFormattingConfig(
            enabled=False,
            format_style=DateFormatStyle.VERBOSE,
            fiscal_year_start_month=7,
            locale="en_GB",
            fallback_to_iso=False
        )
        
        assert config.enabled is False
        assert config.format_style == DateFormatStyle.VERBOSE
        assert config.fiscal_year_start_month == 7
        assert config.locale == "en_GB"
        assert config.fallback_to_iso is False

    def test_string_format_style_conversion(self):
        """Test automatic conversion of string format_style to enum."""
        config = DateFormattingConfig(format_style="compact")
        assert config.format_style == DateFormatStyle.COMPACT
        
        config = DateFormattingConfig(format_style="VERBOSE")
        assert config.format_style == DateFormatStyle.VERBOSE

    def test_invalid_format_style_defaults_to_smart(self):
        """Test that invalid format_style defaults to SMART."""
        config = DateFormattingConfig(format_style="invalid")
        assert config.format_style == DateFormatStyle.SMART

    def test_invalid_fiscal_year_start_month(self):
        """Test validation of fiscal_year_start_month."""
        with pytest.raises(ValueError, match="fiscal_year_start_month must be between 1 and 12"):
            DateFormattingConfig(fiscal_year_start_month=0)
        
        with pytest.raises(ValueError, match="fiscal_year_start_month must be between 1 and 12"):
            DateFormattingConfig(fiscal_year_start_month=13)


class TestDateFormatterWithConfig:
    """Test DateFormatter with different configurations."""

    def test_default_config_initialization(self):
        """Test DateFormatter initialization with default config."""
        formatter = DateFormatter()
        
        assert formatter.config.enabled is True
        assert formatter.config.format_style == DateFormatStyle.SMART
        assert formatter.config.fiscal_year_start_month == 1
        assert formatter.detector.fiscal_year_start_month == 1

    def test_custom_config_initialization(self):
        """Test DateFormatter initialization with custom config."""
        config = DateFormattingConfig(
            format_style=DateFormatStyle.COMPACT,
            fiscal_year_start_month=7
        )
        formatter = DateFormatter(config)
        
        assert formatter.config.format_style == DateFormatStyle.COMPACT
        assert formatter.config.fiscal_year_start_month == 7
        assert formatter.detector.fiscal_year_start_month == 7
        assert formatter.format_rules.format_style == DateFormatStyle.COMPACT

    def test_disabled_formatting(self):
        """Test that disabled formatting returns fallback format."""
        config = DateFormattingConfig(enabled=False)
        formatter = DateFormatter(config)
        
        time_period = TimePeriod(
            start=datetime(2025, 1, 1),
            end=datetime(2025, 2, 1)
        )
        
        result = formatter.format_time_period(time_period)
        assert result == "2025-01-01 to 2025-02-01"

    def test_fallback_to_iso_disabled(self):
        """Test that disabling fallback_to_iso raises exceptions on errors."""
        config = DateFormattingConfig(fallback_to_iso=False)
        formatter = DateFormatter(config)
        
        # Create an invalid time period that will cause formatting to fail
        time_period = TimePeriod(start=None, end=None)
        
        with pytest.raises(ValueError):
            formatter.format_time_period(time_period)

    def test_fallback_to_iso_enabled(self):
        """Test that enabling fallback_to_iso handles errors gracefully."""
        config = DateFormattingConfig(fallback_to_iso=True)
        formatter = DateFormatter(config)
        
        # Create an invalid time period that will cause formatting to fail
        time_period = TimePeriod(start=None, end=None)
        
        result = formatter.safe_format_time_period(time_period)
        assert result == "Invalid date range"


class TestFormatStyleBehavior:
    """Test different format styles produce different outputs."""

    def setup_method(self):
        """Set up test data."""
        self.single_day = TimePeriod(
            start=datetime(2025, 1, 15),
            end=datetime(2025, 1, 16)
        )
        self.single_month = TimePeriod(
            start=datetime(2025, 1, 1),
            end=datetime(2025, 2, 1)
        )
        self.single_quarter = TimePeriod(
            start=datetime(2025, 1, 1),
            end=datetime(2025, 4, 1)
        )
        self.single_year = TimePeriod(
            start=datetime(2025, 1, 1),
            end=datetime(2026, 1, 1)
        )
        self.multi_month = TimePeriod(
            start=datetime(2025, 1, 1),
            end=datetime(2025, 4, 1)
        )

    def test_smart_format_style(self):
        """Test SMART format style outputs."""
        config = DateFormattingConfig(format_style=DateFormatStyle.SMART)
        formatter = DateFormatter(config)
        
        assert formatter.format_time_period(self.single_day) == "January 15, 2025"
        assert formatter.format_time_period(self.single_month) == "January 2025"
        assert formatter.format_time_period(self.single_quarter) == "Q1 2025"
        assert formatter.format_time_period(self.single_year) == "2025"

    def test_verbose_format_style(self):
        """Test VERBOSE format style outputs."""
        config = DateFormattingConfig(format_style=DateFormatStyle.VERBOSE)
        formatter = DateFormatter(config)
        
        assert formatter.format_time_period(self.single_day) == "January 15, 2025"
        assert formatter.format_time_period(self.single_month) == "January 1 - 31, 2025"
        assert formatter.format_time_period(self.single_quarter) == "Q1 2025 (January - March)"
        assert formatter.format_time_period(self.single_year) == "2025 (January - December)"

    def test_compact_format_style(self):
        """Test COMPACT format style outputs."""
        config = DateFormattingConfig(format_style=DateFormatStyle.COMPACT)
        formatter = DateFormatter(config)
        
        assert formatter.format_time_period(self.single_day) == "Jan 15, 2025"
        assert formatter.format_time_period(self.single_month) == "Jan 2025"
        assert formatter.format_time_period(self.single_quarter) == "Q1 2025"
        assert formatter.format_time_period(self.single_year) == "2025"

    def test_multi_month_format_styles(self):
        """Test multi-month formatting across different styles."""
        smart_config = DateFormattingConfig(format_style=DateFormatStyle.SMART)
        verbose_config = DateFormattingConfig(format_style=DateFormatStyle.VERBOSE)
        compact_config = DateFormattingConfig(format_style=DateFormatStyle.COMPACT)
        
        smart_formatter = DateFormatter(smart_config)
        verbose_formatter = DateFormatter(verbose_config)
        compact_formatter = DateFormatter(compact_config)
        
        # Multi-month same year (2 months, not a quarter)
        multi_month = TimePeriod(
            start=datetime(2025, 1, 1),
            end=datetime(2025, 3, 1)  # Jan-Feb only
        )
        
        assert smart_formatter.format_time_period(multi_month) == "January - February 2025"
        assert verbose_formatter.format_time_period(multi_month) == "January - February 2025"
        assert compact_formatter.format_time_period(multi_month) == "Jan - Feb 2025"

    def test_custom_range_format_styles(self):
        """Test custom range formatting across different styles."""
        smart_config = DateFormattingConfig(format_style=DateFormatStyle.SMART)
        compact_config = DateFormattingConfig(format_style=DateFormatStyle.COMPACT)
        
        smart_formatter = DateFormatter(smart_config)
        compact_formatter = DateFormatter(compact_config)
        
        # Custom range same month
        custom_range = TimePeriod(
            start=datetime(2025, 1, 15),
            end=datetime(2025, 1, 25)
        )
        
        assert smart_formatter.format_time_period(custom_range) == "January 15 - 24, 2025"
        assert compact_formatter.format_time_period(custom_range) == "Jan 15 - 24, 2025"


class TestFiscalYearConfiguration:
    """Test fiscal year configuration affects period detection."""

    def test_fiscal_year_start_month_affects_detection(self):
        """Test that fiscal year start month affects period type detection."""
        # July fiscal year start
        config = DateFormattingConfig(fiscal_year_start_month=7)
        formatter = DateFormatter(config)
        
        # July to June should be detected as single fiscal year
        fiscal_year = TimePeriod(
            start=datetime(2024, 7, 1),
            end=datetime(2025, 7, 1)
        )
        
        result = formatter.format_time_period(fiscal_year)
        assert result == "2024"  # Should format as single year

    def test_fiscal_quarter_detection(self):
        """Test fiscal quarter detection with custom fiscal year start."""
        # April fiscal year start (Q1: Apr-Jun, Q2: Jul-Sep, Q3: Oct-Dec, Q4: Jan-Mar)
        config = DateFormattingConfig(fiscal_year_start_month=4)
        formatter = DateFormatter(config)
        
        # April to July should be Q1 (Apr-Jun)
        fiscal_q1 = TimePeriod(
            start=datetime(2025, 4, 1),
            end=datetime(2025, 7, 1)
        )
        
        result = formatter.format_time_period(fiscal_q1)
        assert result == "Q1 2025"

    def test_calendar_vs_fiscal_year_detection(self):
        """Test that calendar and fiscal year detection work correctly."""
        calendar_config = DateFormattingConfig(fiscal_year_start_month=1)
        fiscal_config = DateFormattingConfig(fiscal_year_start_month=7)
        
        calendar_formatter = DateFormatter(calendar_config)
        fiscal_formatter = DateFormatter(fiscal_config)
        
        # January to December (calendar year)
        calendar_year = TimePeriod(
            start=datetime(2025, 1, 1),
            end=datetime(2026, 1, 1)
        )
        
        # July to June (fiscal year)
        fiscal_year = TimePeriod(
            start=datetime(2024, 7, 1),
            end=datetime(2025, 7, 1)
        )
        
        assert calendar_formatter.format_time_period(calendar_year) == "2025"
        assert fiscal_formatter.format_time_period(fiscal_year) == "2024"
        
        # Calendar year is still detected as a year even with fiscal config
        # because it's a valid calendar year (the logic checks calendar first)
        assert fiscal_formatter.format_time_period(calendar_year) == "2025"
        
        # But a period that's NOT a calendar year should be formatted differently
        # For example, a period from July to December (6 months)
        partial_fiscal = TimePeriod(
            start=datetime(2024, 7, 1),
            end=datetime(2025, 1, 1)
        )
        
        # This should be formatted as multi-month, not as a year
        result = fiscal_formatter.format_time_period(partial_fiscal)
        assert "July" in result and "December" in result


class TestPeriodTypeDetectorWithConfig:
    """Test PeriodTypeDetector with different fiscal year configurations."""

    def test_detector_fiscal_year_initialization(self):
        """Test detector initialization with fiscal year configuration."""
        detector = PeriodTypeDetector(fiscal_year_start_month=7)
        assert detector.fiscal_year_start_month == 7

    def test_fiscal_quarter_number_calculation(self):
        """Test fiscal quarter number calculation."""
        detector = PeriodTypeDetector(fiscal_year_start_month=7)
        
        # July fiscal year: Q1=Jul-Sep, Q2=Oct-Dec, Q3=Jan-Mar, Q4=Apr-Jun
        assert detector.get_fiscal_quarter_number(7) == 1  # July = Q1
        assert detector.get_fiscal_quarter_number(10) == 2  # October = Q2
        assert detector.get_fiscal_quarter_number(1) == 3   # January = Q3
        assert detector.get_fiscal_quarter_number(4) == 4   # April = Q4

    def test_fiscal_quarter_detection_edge_cases(self):
        """Test fiscal quarter detection with edge cases."""
        # October fiscal year start
        detector = PeriodTypeDetector(fiscal_year_start_month=10)
        
        # Q1: Oct-Dec
        q1_period = TimePeriod(
            start=datetime(2024, 10, 1),
            end=datetime(2025, 1, 1)
        )
        
        assert detector.is_single_quarter(q1_period) is True


class TestFormatRulesWithStyles:
    """Test FormatRules class with different format styles."""

    def test_format_rules_initialization(self):
        """Test FormatRules initialization with format style."""
        rules = FormatRules(format_style=DateFormatStyle.COMPACT)
        assert rules.format_style == DateFormatStyle.COMPACT

    def test_template_selection(self):
        """Test that correct templates are selected for each style."""
        smart_rules = FormatRules(format_style=DateFormatStyle.SMART)
        verbose_rules = FormatRules(format_style=DateFormatStyle.VERBOSE)
        compact_rules = FormatRules(format_style=DateFormatStyle.COMPACT)
        
        # Test single day templates
        assert smart_rules._get_template("SINGLE_DAY") == "{month} {day}, {year}"
        assert verbose_rules._get_template("SINGLE_DAY") == "{month} {day}, {year}"
        assert compact_rules._get_template("SINGLE_DAY") == "{month_short} {day}, {year}"

    def test_verbose_month_formatting(self):
        """Test verbose month formatting includes day range."""
        rules = FormatRules(format_style=DateFormatStyle.VERBOSE)
        
        # January 2025 (31 days)
        result = rules.format_single_month(datetime(2025, 1, 1))
        assert result == "January 1 - 31, 2025"
        
        # February 2025 (28 days, not leap year)
        result = rules.format_single_month(datetime(2025, 2, 1))
        assert result == "February 1 - 28, 2025"

    def test_verbose_quarter_formatting(self):
        """Test verbose quarter formatting includes month range."""
        rules = FormatRules(format_style=DateFormatStyle.VERBOSE)
        
        result = rules.format_single_quarter(datetime(2025, 1, 1), 1)
        assert result == "Q1 2025 (January - March)"
        
        result = rules.format_single_quarter(datetime(2025, 4, 1), 2)
        assert result == "Q2 2025 (April - June)"

    def test_verbose_year_formatting(self):
        """Test verbose year formatting includes month range."""
        rules = FormatRules(format_style=DateFormatStyle.VERBOSE)
        
        result = rules.format_single_year(datetime(2025, 1, 1))
        assert result == "2025 (January - December)"


class TestConfigurationIntegration:
    """Test integration of configuration with existing functionality."""

    def test_safe_format_time_period_respects_config(self):
        """Test that safe_format_time_period respects configuration."""
        config = DateFormattingConfig(enabled=False)
        formatter = DateFormatter(config)
        
        time_period = TimePeriod(
            start=datetime(2025, 1, 1),
            end=datetime(2025, 2, 1)
        )
        
        result = formatter.safe_format_time_period(time_period)
        assert result == "2025-01-01 to 2025-02-01"

    def test_backward_compatibility(self):
        """Test that existing code without config still works."""
        formatter = DateFormatter()  # No config provided
        
        time_period = TimePeriod(
            start=datetime(2025, 1, 1),
            end=datetime(2025, 2, 1)
        )
        
        result = formatter.format_time_period(time_period)
        assert result == "January 2025"  # Should use default smart formatting

    def test_month_names_backward_compatibility(self):
        """Test that month_names attribute is still available for backward compatibility."""
        formatter = DateFormatter()
        
        assert hasattr(formatter, 'month_names')
        assert len(formatter.month_names) == 12
        assert formatter.month_names[0] == "January"