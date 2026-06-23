"""Date formatting utility for AWS Cost CLI.

This module provides intelligent date formatting for time periods,
automatically detecting period types and applying appropriate formatting rules.
"""

import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional
from calendar import monthrange

from .models import TimePeriod

logger = logging.getLogger(__name__)


class PeriodType(Enum):
    """Types of time periods for intelligent formatting."""
    
    SINGLE_DAY = "single_day"
    SINGLE_MONTH = "single_month"
    SINGLE_QUARTER = "single_quarter"
    SINGLE_YEAR = "single_year"
    MULTI_MONTH = "multi_month"
    CUSTOM_RANGE = "custom_range"


class PeriodTypeDetector:
    """Detects the type of time period for appropriate formatting."""
    
    def __init__(self, fiscal_year_start_month: int = 1):
        """Initialize detector with optional fiscal year configuration.
        
        Args:
            fiscal_year_start_month: Month (1-12) when fiscal year starts
        """
        self.fiscal_year_start_month = fiscal_year_start_month
        self.quarter_months = {
            1: (1, 2, 3),    # Q1: Jan, Feb, Mar
            2: (4, 5, 6),    # Q2: Apr, May, Jun
            3: (7, 8, 9),    # Q3: Jul, Aug, Sep
            4: (10, 11, 12)  # Q4: Oct, Nov, Dec
        }
    
    def detect_period_type(self, time_period: TimePeriod) -> PeriodType:
        """Detect the type of time period based on start and end dates.
        
        Detection is performed in order of specificity:
        1. Single day (most specific)
        2. Single month
        3. Single quarter
        4. Single year
        5. Multi-month (multiple complete months)
        6. Custom range (fallback for everything else)
        """
        try:
            # Validate input
            if not self._is_valid_time_period(time_period):
                logger.warning(f"Invalid time period: {time_period}")
                return PeriodType.CUSTOM_RANGE
            
            # Check in order of specificity
            if self.is_single_day(time_period):
                return PeriodType.SINGLE_DAY
            elif self.is_single_month(time_period):
                return PeriodType.SINGLE_MONTH
            elif self.is_single_quarter(time_period):
                return PeriodType.SINGLE_QUARTER
            elif self.is_single_year(time_period):
                return PeriodType.SINGLE_YEAR
            elif self.is_multi_month(time_period):
                return PeriodType.MULTI_MONTH
            else:
                return PeriodType.CUSTOM_RANGE
                
        except Exception as e:
            logger.warning(f"Error detecting period type: {e}")
            return PeriodType.CUSTOM_RANGE
    
    def _is_valid_time_period(self, time_period: TimePeriod) -> bool:
        """Validate that the time period is valid."""
        if not time_period or not time_period.start or not time_period.end:
            return False
        
        if time_period.start >= time_period.end:
            return False
            
        # Check for reasonable date range (not too far in past/future)
        now = datetime.now()
        if (time_period.start.year < 1900 or time_period.start.year > now.year + 10 or
            time_period.end.year < 1900 or time_period.end.year > now.year + 10):
            return False
            
        return True
    
    def is_single_day(self, time_period: TimePeriod) -> bool:
        """Check if the time period represents a single day.
        
        Handles various AWS Cost Explorer date formats:
        - Exclusive end dates (end = start + 1 day)
        - Inclusive end dates (end = start, same day)
        - Time components (ignores hours/minutes/seconds)
        """
        start = time_period.start
        end = time_period.end
        
        # Normalize to date only (ignore time components)
        start_date = start.date()
        end_date = end.date()
        
        # Case 1: Exclusive end date (AWS typical format)
        # Start: 2025-01-15, End: 2025-01-16
        if end_date == start_date + timedelta(days=1):
            return True
            
        # Case 2: Inclusive same day
        # Start: 2025-01-15 00:00:00, End: 2025-01-15 23:59:59
        if start_date == end_date:
            return True
            
        return False
    
    def is_single_month(self, time_period: TimePeriod) -> bool:
        """Check if the time period represents exactly one calendar month.
        
        Handles edge cases:
        - Partial months at start/end
        - Month boundaries across years
        - Leap years (February variations)
        """
        start = time_period.start
        end = time_period.end
        
        # Must start on first day of month
        if start.day != 1:
            return False
        
        # Calculate expected end date (first day of next month)
        try:
            if start.month == 12:
                expected_end = datetime(start.year + 1, 1, 1, tzinfo=start.tzinfo)
            else:
                expected_end = datetime(start.year, start.month + 1, 1, tzinfo=start.tzinfo)
            
            # Handle time components - end could be midnight of next day or 23:59:59 of last day
            end_normalized = end.replace(hour=0, minute=0, second=0, microsecond=0)
            
            return end_normalized == expected_end
            
        except ValueError as e:
            logger.warning(f"Error calculating month boundaries: {e}")
            return False
    
    def is_single_quarter(self, time_period: TimePeriod) -> bool:
        """Check if the time period represents exactly one calendar quarter.
        
        Handles:
        - Calendar quarters (Q1: Jan-Mar, Q2: Apr-Jun, etc.)
        - Fiscal quarters (based on fiscal_year_start_month)
        - Year boundary crossings
        """
        start = time_period.start
        end = time_period.end
        
        # Must start on first day of month
        if start.day != 1:
            return False
        
        # Check for calendar quarter first
        if self._is_calendar_quarter(start, end):
            return True
            
        # Check for fiscal quarter if different from calendar
        if self.fiscal_year_start_month != 1:
            return self._is_fiscal_quarter(start, end)
            
        return False
    
    def _is_calendar_quarter(self, start: datetime, end: datetime) -> bool:
        """Check if period matches a calendar quarter."""
        quarter_starts = [1, 4, 7, 10]  # Jan, Apr, Jul, Oct
        
        if start.month not in quarter_starts:
            return False
        
        # Calculate expected end date (first day of month after quarter)
        try:
            if start.month == 1:  # Q1: Jan-Mar
                expected_end = datetime(start.year, 4, 1, tzinfo=start.tzinfo)
            elif start.month == 4:  # Q2: Apr-Jun
                expected_end = datetime(start.year, 7, 1, tzinfo=start.tzinfo)
            elif start.month == 7:  # Q3: Jul-Sep
                expected_end = datetime(start.year, 10, 1, tzinfo=start.tzinfo)
            else:  # Q4: Oct-Dec
                expected_end = datetime(start.year + 1, 1, 1, tzinfo=start.tzinfo)
            
            end_normalized = end.replace(hour=0, minute=0, second=0, microsecond=0)
            return end_normalized == expected_end
            
        except ValueError as e:
            logger.warning(f"Error calculating quarter boundaries: {e}")
            return False
    
    def _is_fiscal_quarter(self, start: datetime, end: datetime) -> bool:
        """Check if period matches a fiscal quarter."""
        # Calculate fiscal quarter start months
        fiscal_q1_start = self.fiscal_year_start_month
        fiscal_q2_start = (fiscal_q1_start + 2) % 12 + 1
        fiscal_q3_start = (fiscal_q1_start + 5) % 12 + 1
        fiscal_q4_start = (fiscal_q1_start + 8) % 12 + 1
        
        fiscal_quarter_starts = [fiscal_q1_start, fiscal_q2_start, fiscal_q3_start, fiscal_q4_start]
        
        if start.month not in fiscal_quarter_starts:
            return False
        
        # Calculate expected end date for fiscal quarter
        try:
            quarter_index = fiscal_quarter_starts.index(start.month)
            end_month = fiscal_quarter_starts[(quarter_index + 1) % 4]
            
            if end_month > start.month:
                expected_end = datetime(start.year, end_month, 1, tzinfo=start.tzinfo)
            else:
                expected_end = datetime(start.year + 1, end_month, 1, tzinfo=start.tzinfo)
            
            end_normalized = end.replace(hour=0, minute=0, second=0, microsecond=0)
            return end_normalized == expected_end
            
        except (ValueError, IndexError) as e:
            logger.warning(f"Error calculating fiscal quarter boundaries: {e}")
            return False
    
    def is_single_year(self, time_period: TimePeriod) -> bool:
        """Check if the time period represents exactly one calendar year.
        
        Handles:
        - Calendar years (Jan 1 - Dec 31)
        - Fiscal years (based on fiscal_year_start_month)
        """
        start = time_period.start
        end = time_period.end
        
        # Must start on first day of month
        if start.day != 1:
            return False
        
        # Check for calendar year first
        if self._is_calendar_year(start, end):
            return True
            
        # Check for fiscal year if different from calendar
        if self.fiscal_year_start_month != 1:
            return self._is_fiscal_year(start, end)
            
        return False
    
    def _is_calendar_year(self, start: datetime, end: datetime) -> bool:
        """Check if period matches a calendar year."""
        if start.month != 1:
            return False
        
        try:
            expected_end = datetime(start.year + 1, 1, 1, tzinfo=start.tzinfo)
            end_normalized = end.replace(hour=0, minute=0, second=0, microsecond=0)
            return end_normalized == expected_end
        except ValueError as e:
            logger.warning(f"Error calculating year boundaries: {e}")
            return False
    
    def _is_fiscal_year(self, start: datetime, end: datetime) -> bool:
        """Check if period matches a fiscal year."""
        if start.month != self.fiscal_year_start_month:
            return False
        
        try:
            expected_end = datetime(start.year + 1, self.fiscal_year_start_month, 1, tzinfo=start.tzinfo)
            end_normalized = end.replace(hour=0, minute=0, second=0, microsecond=0)
            return end_normalized == expected_end
        except ValueError as e:
            logger.warning(f"Error calculating fiscal year boundaries: {e}")
            return False
    
    def is_multi_month(self, time_period: TimePeriod) -> bool:
        """Check if the time period spans multiple complete months.
        
        Requirements:
        - Must start on first day of a month
        - Must end on first day of a month
        - Must span more than one month but less than a year
        - Handles year boundary crossings
        """
        start = time_period.start
        end = time_period.end
        
        # Must start on first day of month
        if start.day != 1:
            return False
        
        # Must end on first day of a month (or be normalized to it)
        end_normalized = end.replace(hour=0, minute=0, second=0, microsecond=0)
        if end_normalized.day != 1:
            return False
        
        # Calculate months difference
        months_diff = (end.year - start.year) * 12 + (end.month - start.month)
        
        # Must span more than one month but less than a year
        return 1 < months_diff < 12
    
    def is_custom_range(self, time_period: TimePeriod) -> bool:
        """Check if the time period is a custom range.
        
        This is the fallback for any period that doesn't match
        the other specific period types.
        """
        # If it doesn't match any other type, it's custom
        return not any([
            self.is_single_day(time_period),
            self.is_single_month(time_period),
            self.is_single_quarter(time_period),
            self.is_single_year(time_period),
            self.is_multi_month(time_period)
        ])
    
    def get_quarter_number(self, month: int) -> int:
        """Get quarter number (1-4) for a given month.
        
        Args:
            month: Month number (1-12)
            
        Returns:
            Quarter number (1-4)
        """
        if not 1 <= month <= 12:
            raise ValueError(f"Invalid month: {month}")
            
        return ((month - 1) // 3) + 1
    
    def get_fiscal_quarter_number(self, month: int) -> int:
        """Get fiscal quarter number (1-4) for a given month.
        
        Args:
            month: Month number (1-12)
            
        Returns:
            Fiscal quarter number (1-4)
        """
        if not 1 <= month <= 12:
            raise ValueError(f"Invalid month: {month}")
        
        # Calculate months from fiscal year start
        months_from_fy_start = (month - self.fiscal_year_start_month) % 12
        return (months_from_fy_start // 3) + 1


class FormatRules:
    """Template-based formatting rules for different period types."""
    
    def __init__(self, format_style=None):
        """Initialize format rules with month names and format style.
        
        Args:
            format_style: DateFormatStyle enum or None for smart formatting
        """
        from .models import DateFormatStyle
        
        self.format_style = format_style or DateFormatStyle.SMART
        
        self.month_names = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
        self.month_names_short = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
        ]
        
        # Template strings for each period type and style
        self._templates = {
            DateFormatStyle.SMART: {
                "SINGLE_DAY": "{month} {day}, {year}",  # "January 15, 2025"
                "SINGLE_MONTH": "{month} {year}",       # "January 2025"
                "SINGLE_QUARTER": "Q{quarter} {year}",  # "Q1 2025"
                "SINGLE_YEAR": "{year}",                # "2025"
                "MULTI_MONTH_SAME_YEAR": "{start_month} - {end_month} {year}",  # "January - March 2025"
                "MULTI_MONTH_CROSS_YEAR": "{start_month} {start_year} - {end_month} {end_year}",  # "December 2024 - February 2025"
                "CUSTOM_RANGE_SAME_MONTH": "{month} {start_day} - {end_day}, {year}",  # "January 15 - 25, 2025"
                "CUSTOM_RANGE_SAME_YEAR": "{start_month} {start_day} - {end_month} {end_day}, {year}",  # "January 15 - February 10, 2025"
                "CUSTOM_RANGE_CROSS_YEAR": "{start_month} {start_day}, {start_year} - {end_month} {end_day}, {end_year}",  # "December 15, 2024 - January 10, 2025"
            },
            DateFormatStyle.VERBOSE: {
                "SINGLE_DAY": "{month} {day}, {year}",  # "January 15, 2025"
                "SINGLE_MONTH": "{month} 1 - {month_end}, {year}",  # "January 1 - 31, 2025"
                "SINGLE_QUARTER": "Q{quarter} {year} ({start_month} - {end_month})",  # "Q1 2025 (January - March)"
                "SINGLE_YEAR": "{year} ({start_month} - {end_month})",  # "2025 (January - December)"
                "MULTI_MONTH_SAME_YEAR": "{start_month} - {end_month} {year}",  # "January - March 2025"
                "MULTI_MONTH_CROSS_YEAR": "{start_month} {start_year} - {end_month} {end_year}",  # "December 2024 - February 2025"
                "CUSTOM_RANGE_SAME_MONTH": "{month} {start_day} - {end_day}, {year}",  # "January 15 - 25, 2025"
                "CUSTOM_RANGE_SAME_YEAR": "{start_month} {start_day} - {end_month} {end_day}, {year}",  # "January 15 - February 10, 2025"
                "CUSTOM_RANGE_CROSS_YEAR": "{start_month} {start_day}, {start_year} - {end_month} {end_day}, {end_year}",  # "December 15, 2024 - January 10, 2025"
            },
            DateFormatStyle.COMPACT: {
                "SINGLE_DAY": "{month_short} {day}, {year}",  # "Jan 15, 2025"
                "SINGLE_MONTH": "{month_short} {year}",       # "Jan 2025"
                "SINGLE_QUARTER": "Q{quarter} {year}",       # "Q1 2025"
                "SINGLE_YEAR": "{year}",                     # "2025"
                "MULTI_MONTH_SAME_YEAR": "{start_month_short} - {end_month_short} {year}",  # "Jan - Mar 2025"
                "MULTI_MONTH_CROSS_YEAR": "{start_month_short} {start_year} - {end_month_short} {end_year}",  # "Dec 2024 - Feb 2025"
                "CUSTOM_RANGE_SAME_MONTH": "{month_short} {start_day} - {end_day}, {year}",  # "Jan 15 - 25, 2025"
                "CUSTOM_RANGE_SAME_YEAR": "{start_month_short} {start_day} - {end_month_short} {end_day}, {year}",  # "Jan 15 - Feb 10, 2025"
                "CUSTOM_RANGE_CROSS_YEAR": "{start_month_short} {start_day}, {start_year} - {end_month_short} {end_day}, {end_year}",  # "Dec 15, 2024 - Jan 10, 2025"
            }
        }
    
    def _get_template(self, template_key: str) -> str:
        """Get template string for the current format style."""
        return self._templates[self.format_style][template_key]
    
    def format_single_day(self, date: datetime) -> str:
        """Format single day using template."""
        try:
            if not date or not hasattr(date, 'month') or not hasattr(date, 'day') or not hasattr(date, 'year'):
                raise ValueError("Invalid date object")
            
            if not (1 <= date.month <= 12):
                raise ValueError(f"Invalid month: {date.month}")
            
            template = self._get_template("SINGLE_DAY")
            return template.format(
                month=self.month_names[date.month - 1],
                month_short=self.month_names_short[date.month - 1],
                day=date.day,
                year=date.year
            )
        except (IndexError, AttributeError, ValueError) as e:
            logger.warning(f"Error in format_single_day: {e}")
            raise
    
    def format_single_month(self, date: datetime) -> str:
        """Format single month using template."""
        try:
            if not date or not hasattr(date, 'month') or not hasattr(date, 'year'):
                raise ValueError("Invalid date object")
            
            if not (1 <= date.month <= 12):
                raise ValueError(f"Invalid month: {date.month}")
            
            template = self._get_template("SINGLE_MONTH")
            
            # For verbose style, calculate month end day
            month_end = 31
            if template.find("{month_end}") != -1:
                month_end = monthrange(date.year, date.month)[1]
            
            return template.format(
                month=self.month_names[date.month - 1],
                month_short=self.month_names_short[date.month - 1],
                month_end=month_end,
                year=date.year
            )
        except (IndexError, AttributeError, ValueError) as e:
            logger.warning(f"Error in format_single_month: {e}")
            raise
    
    def format_single_quarter(self, date: datetime, quarter: int) -> str:
        """Format single quarter using template."""
        try:
            if not date or not hasattr(date, 'year'):
                raise ValueError("Invalid date object")
            
            if not (1 <= quarter <= 4):
                raise ValueError(f"Invalid quarter: {quarter}")
            
            template = self._get_template("SINGLE_QUARTER")
            
            # For verbose style, calculate quarter month range
            quarter_months = {
                1: ("January", "March"),
                2: ("April", "June"),
                3: ("July", "September"),
                4: ("October", "December")
            }
            start_month, end_month = quarter_months[quarter]
            
            return template.format(
                quarter=quarter,
                year=date.year,
                start_month=start_month,
                end_month=end_month
            )
        except (AttributeError, ValueError) as e:
            logger.warning(f"Error in format_single_quarter: {e}")
            raise
    
    def format_single_year(self, date: datetime) -> str:
        """Format single year using template."""
        try:
            if not date or not hasattr(date, 'year'):
                raise ValueError("Invalid date object")
            
            template = self._get_template("SINGLE_YEAR")
            
            return template.format(
                year=date.year,
                start_month="January",
                end_month="December"
            )
        except (AttributeError, ValueError) as e:
            logger.warning(f"Error in format_single_year: {e}")
            raise
    
    def format_multi_month(self, start_date: datetime, end_date: datetime) -> str:
        """Format multiple months with proper date boundaries."""
        try:
            if not start_date or not end_date:
                raise ValueError("Start and end dates cannot be None")
            
            if not all(hasattr(date, attr) for date in [start_date, end_date] 
                      for attr in ['month', 'year']):
                raise ValueError("Invalid date objects")
            
            if not (1 <= start_date.month <= 12) or not (1 <= end_date.month <= 12):
                raise ValueError(f"Invalid months: {start_date.month}, {end_date.month}")
            
            start_month = self.month_names[start_date.month - 1]
            end_month = self.month_names[end_date.month - 1]
            start_month_short = self.month_names_short[start_date.month - 1]
            end_month_short = self.month_names_short[end_date.month - 1]
            
            if start_date.year == end_date.year:
                template = self._get_template("MULTI_MONTH_SAME_YEAR")
                return template.format(
                    start_month=start_month,
                    start_month_short=start_month_short,
                    end_month=end_month,
                    end_month_short=end_month_short,
                    year=start_date.year
                )
            else:
                template = self._get_template("MULTI_MONTH_CROSS_YEAR")
                return template.format(
                    start_month=start_month,
                    start_month_short=start_month_short,
                    start_year=start_date.year,
                    end_month=end_month,
                    end_month_short=end_month_short,
                    end_year=end_date.year
                )
        except (IndexError, AttributeError, ValueError) as e:
            logger.warning(f"Error in format_multi_month: {e}")
            raise
    
    def format_custom_range(self, start_date: datetime, end_date: datetime) -> str:
        """Format custom range with proper date boundaries."""
        try:
            if not start_date or not end_date:
                raise ValueError("Start and end dates cannot be None")
            
            if not all(hasattr(date, attr) for date in [start_date, end_date] 
                      for attr in ['month', 'day', 'year']):
                raise ValueError("Invalid date objects")
            
            if not (1 <= start_date.month <= 12) or not (1 <= end_date.month <= 12):
                raise ValueError(f"Invalid months: {start_date.month}, {end_date.month}")
            
            start_month = self.month_names[start_date.month - 1]
            end_month = self.month_names[end_date.month - 1]
            start_month_short = self.month_names_short[start_date.month - 1]
            end_month_short = self.month_names_short[end_date.month - 1]
            
            # Same month
            if start_date.year == end_date.year and start_date.month == end_date.month:
                template = self._get_template("CUSTOM_RANGE_SAME_MONTH")
                return template.format(
                    month=start_month,
                    month_short=start_month_short,
                    start_day=start_date.day,
                    end_day=end_date.day,
                    year=start_date.year
                )
            # Same year, different months
            elif start_date.year == end_date.year:
                template = self._get_template("CUSTOM_RANGE_SAME_YEAR")
                return template.format(
                    start_month=start_month,
                    start_month_short=start_month_short,
                    start_day=start_date.day,
                    end_month=end_month,
                    end_month_short=end_month_short,
                    end_day=end_date.day,
                    year=start_date.year
                )
            # Different years
            else:
                template = self._get_template("CUSTOM_RANGE_CROSS_YEAR")
                return template.format(
                    start_month=start_month,
                    start_month_short=start_month_short,
                    start_day=start_date.day,
                    start_year=start_date.year,
                    end_month=end_month,
                    end_month_short=end_month_short,
                    end_day=end_date.day,
                    end_year=end_date.year
                )
        except (IndexError, AttributeError, ValueError) as e:
            logger.warning(f"Error in format_custom_range: {e}")
            raise
    
    def get_month_name(self, month: int, short: bool = False) -> str:
        """Get month name by number.
        
        Args:
            month: Month number (1-12)
            short: Whether to return short form (Jan vs January)
            
        Returns:
            Month name string
        """
        if not 1 <= month <= 12:
            raise ValueError(f"Invalid month: {month}")
        
        if short:
            return self.month_names_short[month - 1]
        else:
            return self.month_names[month - 1]


class DateFormatter:
    """Intelligent date formatter for time periods."""
    
    def __init__(self, config=None):
        """Initialize DateFormatter with optional configuration.
        
        Args:
            config: DateFormattingConfig object or None for defaults
        """
        from .models import DateFormattingConfig, DateFormatStyle
        
        # Use provided config or create default
        if config is None:
            config = DateFormattingConfig()
        
        self.config = config
        self.detector = PeriodTypeDetector(fiscal_year_start_month=config.fiscal_year_start_month)
        self.format_rules = FormatRules(format_style=config.format_style)
        # Keep backward compatibility
        self.month_names = self.format_rules.month_names
    
    def format_time_period(self, time_period: TimePeriod) -> str:
        """Format a time period with intelligent period type detection.
        
        This method attempts to detect the period type and apply appropriate
        formatting rules. If any step fails, it falls back to ISO format.
        
        Args:
            time_period: The time period to format
            
        Returns:
            Human-readable formatted time period string
            
        Raises:
            Exception: May raise exceptions for invalid input (caught by safe_format_time_period)
        """
        if time_period is None:
            logger.warning("Received None time_period in format_time_period")
            raise ValueError("time_period cannot be None")
        
        if not hasattr(time_period, 'start') or not hasattr(time_period, 'end'):
            logger.warning("TimePeriod missing required attributes")
            raise AttributeError("TimePeriod must have start and end attributes")
        
        if time_period.start is None or time_period.end is None:
            logger.warning("TimePeriod has None start or end dates")
            raise ValueError("TimePeriod start and end cannot be None")
        
        # Check if date formatting is disabled
        if not self.config.enabled:
            logger.debug("Date formatting is disabled, using fallback format")
            return self._fallback_format(time_period)
        
        try:
            # Log the input for debugging
            logger.debug(f"Formatting time period: {time_period.start} to {time_period.end}")
            
            # Detect period type
            period_type = self.detector.detect_period_type(time_period)
            logger.debug(f"Detected period type: {period_type}")
            
            # Apply formatting rules
            formatted_result = self._apply_format_rules(time_period, period_type)
            logger.debug(f"Formatted result: {formatted_result}")
            
            return formatted_result
            
        except Exception as e:
            logger.warning(f"Date formatting failed during processing: {type(e).__name__}: {e}")
            if self.config.fallback_to_iso:
                return self._fallback_format(time_period)
            else:
                raise
    
    def _apply_format_rules(self, time_period: TimePeriod, period_type: PeriodType) -> str:
        """Apply formatting rules based on period type using FormatRules templates.
        
        Args:
            time_period: The time period to format
            period_type: The detected period type
            
        Returns:
            Formatted string based on period type
            
        Raises:
            Exception: May raise exceptions during formatting (handled by caller)
        """
        try:
            if period_type == PeriodType.SINGLE_DAY:
                return self._format_single_day(time_period)
            elif period_type == PeriodType.SINGLE_MONTH:
                return self._format_single_month(time_period)
            elif period_type == PeriodType.SINGLE_QUARTER:
                return self._format_single_quarter(time_period)
            elif period_type == PeriodType.SINGLE_YEAR:
                return self._format_single_year(time_period)
            elif period_type == PeriodType.MULTI_MONTH:
                return self._format_multi_month(time_period)
            else:  # CUSTOM_RANGE
                return self._format_custom_range(time_period)
        except Exception as e:
            logger.warning(f"Error applying format rules for {period_type}: {e}")
            raise
    
    def _format_single_day(self, time_period: TimePeriod) -> str:
        """Format single day using FormatRules template."""
        try:
            return self.format_rules.format_single_day(time_period.start)
        except Exception as e:
            logger.warning(f"Error formatting single day: {e}")
            raise
    
    def _format_single_month(self, time_period: TimePeriod) -> str:
        """Format single month using FormatRules template."""
        try:
            return self.format_rules.format_single_month(time_period.start)
        except Exception as e:
            logger.warning(f"Error formatting single month: {e}")
            raise
    
    def _format_single_quarter(self, time_period: TimePeriod) -> str:
        """Format single quarter using FormatRules template."""
        try:
            # Use fiscal quarter if fiscal year start is not January
            if self.detector.fiscal_year_start_month != 1:
                quarter = self.detector.get_fiscal_quarter_number(time_period.start.month)
            else:
                quarter = self.detector.get_quarter_number(time_period.start.month)
            return self.format_rules.format_single_quarter(time_period.start, quarter)
        except Exception as e:
            logger.warning(f"Error formatting single quarter: {e}")
            raise
    
    def _format_single_year(self, time_period: TimePeriod) -> str:
        """Format single year using FormatRules template."""
        try:
            return self.format_rules.format_single_year(time_period.start)
        except Exception as e:
            logger.warning(f"Error formatting single year: {e}")
            raise
    
    def _format_multi_month(self, time_period: TimePeriod) -> str:
        """Format multiple months using FormatRules template with proper date boundaries."""
        try:
            start = time_period.start
            # End date is first day of month after the range, so subtract 1 day to get last day of actual range
            end = time_period.end - timedelta(days=1)
            
            return self.format_rules.format_multi_month(start, end)
        except Exception as e:
            logger.warning(f"Error formatting multi-month period: {e}")
            raise
    
    def _format_custom_range(self, time_period: TimePeriod) -> str:
        """Format custom range using FormatRules template with proper date boundaries."""
        try:
            start = time_period.start
            # For custom ranges, end might be exclusive, so subtract 1 day if it's midnight
            end = time_period.end
            if end.hour == 0 and end.minute == 0 and end.second == 0:
                end = end - timedelta(days=1)
            
            return self.format_rules.format_custom_range(start, end)
        except Exception as e:
            logger.warning(f"Error formatting custom range: {e}")
            raise
    
    def _fallback_format(self, time_period: TimePeriod) -> str:
        """Fallback to ISO date format when primary formatting fails.
        
        This method is used by format_time_period when the main formatting
        logic encounters an error. It provides a simple, reliable fallback.
        
        Args:
            time_period: The time period to format
            
        Returns:
            ISO formatted date range string
        """
        try:
            if not time_period or not time_period.start or not time_period.end:
                logger.warning("Invalid time_period in fallback format")
                return "Invalid date range"
            
            start_str = time_period.start.strftime('%Y-%m-%d')
            end_str = time_period.end.strftime('%Y-%m-%d')
            logger.info(f"Using fallback ISO format: {start_str} to {end_str}")
            return f"{start_str} to {end_str}"
        except Exception as e:
            logger.error(f"Fallback formatting failed: {e}")
            return "Invalid date range"
    
    def safe_format_time_period(self, time_period: TimePeriod) -> str:
        """Safely format time period with comprehensive error handling.
        
        This method provides robust error handling and graceful degradation
        for production use. It will attempt multiple fallback strategies
        before returning a default error message.
        
        Args:
            time_period: The time period to format
            
        Returns:
            Human-readable formatted time period string, or fallback format
            
        Raises:
            Never raises - always returns a string
        """
        if time_period is None:
            logger.warning("Received None time_period in safe_format_time_period")
            return "Invalid date range"
        
        try:
            # Validate basic structure
            if not hasattr(time_period, 'start') or not hasattr(time_period, 'end'):
                logger.warning("TimePeriod missing start or end attributes")
                return "Invalid date range"
            
            if time_period.start is None or time_period.end is None:
                logger.warning("TimePeriod has None start or end dates")
                return "Invalid date range"
            
            # Attempt primary formatting
            return self.format_time_period(time_period)
            
        except AttributeError as e:
            logger.warning(f"AttributeError in date formatting: {e}")
            return self._safe_fallback_format(time_period)
        except ValueError as e:
            logger.warning(f"ValueError in date formatting (likely invalid date): {e}")
            return self._safe_fallback_format(time_period)
        except TypeError as e:
            logger.warning(f"TypeError in date formatting (likely type mismatch): {e}")
            return self._safe_fallback_format(time_period)
        except Exception as e:
            logger.error(f"Unexpected error in date formatting: {type(e).__name__}: {e}")
            return self._safe_fallback_format(time_period)
    
    def _safe_fallback_format(self, time_period: TimePeriod) -> str:
        """Ultra-safe fallback formatting that handles any edge case.
        
        This method attempts multiple fallback strategies in order:
        1. ISO date format with proper validation
        2. String representation of dates
        3. Generic error message
        
        Args:
            time_period: The time period to format
            
        Returns:
            Fallback formatted string, guaranteed to not raise exceptions
        """
        try:
            # First fallback: Try ISO format with validation
            if (hasattr(time_period, 'start') and hasattr(time_period, 'end') and
                time_period.start is not None and time_period.end is not None):
                
                # Validate that start and end are datetime-like objects
                if hasattr(time_period.start, 'strftime') and hasattr(time_period.end, 'strftime'):
                    try:
                        start_str = time_period.start.strftime('%Y-%m-%d')
                        end_str = time_period.end.strftime('%Y-%m-%d')
                        logger.info(f"Using ISO fallback format for period: {start_str} to {end_str}")
                        return f"{start_str} to {end_str}"
                    except (ValueError, AttributeError) as e:
                        logger.warning(f"ISO format fallback failed: {e}")
                
                # Second fallback: Try string representation
                try:
                    start_str = str(time_period.start)
                    end_str = str(time_period.end)
                    logger.info(f"Using string representation fallback: {start_str} to {end_str}")
                    return f"{start_str} to {end_str}"
                except Exception as e:
                    logger.warning(f"String representation fallback failed: {e}")
            
        except Exception as e:
            logger.error(f"All fallback strategies failed: {e}")
        
        # Final fallback: Generic error message
        logger.error("All date formatting strategies failed, returning generic error message")
        return "Invalid date range"