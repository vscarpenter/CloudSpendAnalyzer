"""Tests for DateFormatter configuration support."""

import pytest
from datetime import datetime

from src.aws_cost_cli.date_formatter import DateFormatter
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

    def test_custom_config_initialization(self):
        """Test DateFormatter initialization with custom config."""
        config = DateFormattingConfig(
            format_style=DateFormatStyle.COMPACT,
            fiscal_year_start_month=7
        )
        formatter = DateFormatter(config)

        assert formatter.config.format_style == DateFormatStyle.COMPACT
        assert formatter.config.fiscal_year_start_month == 7

    def test_disabled_formatting(self):
        """Test that disabled formatting returns the ISO fallback format."""
        config = DateFormattingConfig(enabled=False)
        formatter = DateFormatter(config)

        time_period = TimePeriod(start=datetime(2025, 1, 1), end=datetime(2025, 2, 1))
        assert formatter.format_time_period(time_period) == "2025-01-01 to 2025-02-01"

    def test_fallback_to_iso_disabled(self):
        """Test that disabling fallback_to_iso raises exceptions on errors."""
        config = DateFormattingConfig(fallback_to_iso=False)
        formatter = DateFormatter(config)

        # Invalid time period that will cause formatting to fail
        time_period = TimePeriod(start=None, end=None)

        with pytest.raises(ValueError):
            formatter.format_time_period(time_period)

    def test_fallback_to_iso_enabled(self):
        """Test that enabling fallback_to_iso handles errors gracefully."""
        config = DateFormattingConfig(fallback_to_iso=True)
        formatter = DateFormatter(config)

        time_period = TimePeriod(start=None, end=None)
        assert formatter.safe_format_time_period(time_period) == "Invalid date range"


class TestSmartFormatBehavior:
    """Test that the smart format produces the expected outputs."""

    def setup_method(self):
        """Set up test data."""
        self.single_day = TimePeriod(start=datetime(2025, 1, 15), end=datetime(2025, 1, 16))
        self.single_month = TimePeriod(start=datetime(2025, 1, 1), end=datetime(2025, 2, 1))
        self.single_quarter = TimePeriod(start=datetime(2025, 1, 1), end=datetime(2025, 4, 1))
        self.single_year = TimePeriod(start=datetime(2025, 1, 1), end=datetime(2026, 1, 1))

    def test_smart_format_style(self):
        """Test SMART format style outputs."""
        formatter = DateFormatter(DateFormattingConfig(format_style=DateFormatStyle.SMART))

        assert formatter.format_time_period(self.single_day) == "January 15, 2025"
        assert formatter.format_time_period(self.single_month) == "January 2025"
        assert formatter.format_time_period(self.single_quarter) == "Q1 2025"
        assert formatter.format_time_period(self.single_year) == "2025"

    def test_multi_month_format(self):
        """Test multi-month (same year) formatting."""
        formatter = DateFormatter()
        multi_month = TimePeriod(start=datetime(2025, 1, 1), end=datetime(2025, 3, 1))  # Jan-Feb
        assert formatter.format_time_period(multi_month) == "January - February 2025"

    def test_custom_range_format(self):
        """Test custom range (same month) formatting."""
        formatter = DateFormatter()
        custom_range = TimePeriod(start=datetime(2025, 1, 15), end=datetime(2025, 1, 25))
        assert formatter.format_time_period(custom_range) == "January 15 - 24, 2025"


class TestConfigurationIntegration:
    """Test integration of configuration with existing functionality."""

    def test_safe_format_time_period_respects_config(self):
        """Test that safe_format_time_period respects configuration."""
        formatter = DateFormatter(DateFormattingConfig(enabled=False))

        time_period = TimePeriod(start=datetime(2025, 1, 1), end=datetime(2025, 2, 1))
        assert formatter.safe_format_time_period(time_period) == "2025-01-01 to 2025-02-01"

    def test_backward_compatibility(self):
        """Test that existing code without config still works."""
        formatter = DateFormatter()  # No config provided

        time_period = TimePeriod(start=datetime(2025, 1, 1), end=datetime(2025, 2, 1))
        assert formatter.format_time_period(time_period) == "January 2025"

    def test_month_names_backward_compatibility(self):
        """Test that month_names attribute is still available for backward compatibility."""
        formatter = DateFormatter()

        assert hasattr(formatter, 'month_names')
        assert len(formatter.month_names) == 12
        assert formatter.month_names[0] == "January"
