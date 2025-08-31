"""Advanced date range utilities for complex query specifications."""

from datetime import datetime, timezone
from typing import Tuple, List
from enum import Enum

from .models import TimePeriod


class Quarter(Enum):
    """Quarter definitions."""

    Q1 = 1
    Q2 = 2
    Q3 = 3
    Q4 = 4


class DateRangeCalculator:
    """Calculator for advanced date ranges including quarters and fiscal years."""

    def __init__(self, fiscal_year_start_month: int = 1):
        """
        Initialize date range calculator.

        Args:
            fiscal_year_start_month: Month when fiscal year starts (1-12, default: 1 for January)
        """
        self.fiscal_year_start_month = fiscal_year_start_month

    def get_quarter_range(self, year: int, quarter: Quarter) -> TimePeriod:
        """
        Get date range for a specific quarter.

        Args:
            year: Year for the quarter
            quarter: Quarter enum value

        Returns:
            TimePeriod for the quarter
        """
        quarter_start_months = {
            Quarter.Q1: 1,
            Quarter.Q2: 4,
            Quarter.Q3: 7,
            Quarter.Q4: 10,
        }

        start_month = quarter_start_months[quarter]
        start_date = datetime(year, start_month, 1, tzinfo=timezone.utc)

        # Calculate end date (first day of next quarter)
        if quarter == Quarter.Q4:
            end_date = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            next_quarter_month = quarter_start_months[Quarter(quarter.value + 1)]
            end_date = datetime(year, next_quarter_month, 1, tzinfo=timezone.utc)

        return TimePeriod(start=start_date, end=end_date)

    def get_fiscal_quarter_range(
        self, fiscal_year: int, quarter: Quarter
    ) -> TimePeriod:
        """
        Get date range for a fiscal quarter.

        Args:
            fiscal_year: Fiscal year
            quarter: Quarter within the fiscal year

        Returns:
            TimePeriod for the fiscal quarter
        """
        # Calculate the calendar year and month for fiscal quarter start
        fiscal_start_month = self.fiscal_year_start_month

        # Map fiscal quarters to months offset from fiscal year start
        quarter_offsets = {Quarter.Q1: 0, Quarter.Q2: 3, Quarter.Q3: 6, Quarter.Q4: 9}

        months_offset = quarter_offsets[quarter]
        start_month = fiscal_start_month + months_offset

        # Handle year rollover
        if start_month > 12:
            start_month -= 12
            calendar_year = fiscal_year
        else:
            calendar_year = fiscal_year - 1 if fiscal_start_month > 1 else fiscal_year

        start_date = datetime(calendar_year, start_month, 1, tzinfo=timezone.utc)

        # Calculate end date (3 months later)
        end_month = start_month + 3
        end_year = calendar_year
        if end_month > 12:
            end_month -= 12
            end_year += 1

        end_date = datetime(end_year, end_month, 1, tzinfo=timezone.utc)

        return TimePeriod(start=start_date, end=end_date)

    def get_fiscal_year_range(self, fiscal_year: int) -> TimePeriod:
        """
        Get date range for a fiscal year.

        Args:
            fiscal_year: Fiscal year

        Returns:
            TimePeriod for the fiscal year
        """
        # Fiscal year starts in the specified month
        if self.fiscal_year_start_month == 1:
            # Calendar year = fiscal year
            start_date = datetime(fiscal_year, 1, 1, tzinfo=timezone.utc)
            end_date = datetime(fiscal_year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            # Fiscal year spans two calendar years
            start_date = datetime(
                fiscal_year - 1, self.fiscal_year_start_month, 1, tzinfo=timezone.utc
            )
            end_date = datetime(
                fiscal_year, self.fiscal_year_start_month, 1, tzinfo=timezone.utc
            )

        return TimePeriod(start=start_date, end=end_date)

    def get_calendar_year_range(self, year: int) -> TimePeriod:
        """
        Get date range for a calendar year.

        Args:
            year: Calendar year

        Returns:
            TimePeriod for the calendar year
        """
        start_date = datetime(year, 1, 1, tzinfo=timezone.utc)
        end_date = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        return TimePeriod(start=start_date, end=end_date)

    def get_current_quarter(self) -> Tuple[int, Quarter]:
        """
        Get current quarter and year.

        Returns:
            Tuple of (year, quarter)
        """
        now = datetime.now(timezone.utc)
        month = now.month

        if month <= 3:
            return now.year, Quarter.Q1
        elif month <= 6:
            return now.year, Quarter.Q2
        elif month <= 9:
            return now.year, Quarter.Q3
        else:
            return now.year, Quarter.Q4

    def get_current_fiscal_quarter(self) -> Tuple[int, Quarter]:
        """
        Get current fiscal quarter and fiscal year.

        Returns:
            Tuple of (fiscal_year, quarter)
        """
        now = datetime.now(timezone.utc)

        # Determine fiscal year
        if now.month >= self.fiscal_year_start_month:
            fiscal_year = now.year + 1 if self.fiscal_year_start_month > 1 else now.year
        else:
            fiscal_year = now.year

        # Determine fiscal quarter
        months_into_fiscal_year = (now.month - self.fiscal_year_start_month) % 12

        if months_into_fiscal_year < 3:
            quarter = Quarter.Q1
        elif months_into_fiscal_year < 6:
            quarter = Quarter.Q2
        elif months_into_fiscal_year < 9:
            quarter = Quarter.Q3
        else:
            quarter = Quarter.Q4

        return fiscal_year, quarter

    def get_previous_period(
        self, period: TimePeriod, period_type: str = "same_length"
    ) -> TimePeriod:
        """
        Get the previous period for comparison.

        Args:
            period: Current period
            period_type: Type of previous period ("same_length", "year_ago", "month_ago", "quarter_ago")

        Returns:
            TimePeriod for the previous period
        """
        period_length = period.end - period.start

        if period_type == "same_length":
            # Previous period of same length
            end_date = period.start
            start_date = end_date - period_length
        elif period_type == "year_ago":
            # Same period one year ago
            start_date = period.start.replace(year=period.start.year - 1)
            end_date = period.end.replace(year=period.end.year - 1)
        elif period_type == "month_ago":
            # Same period one month ago
            start_date = self._subtract_months(period.start, 1)
            end_date = self._subtract_months(period.end, 1)
        elif period_type == "quarter_ago":
            # Same period one quarter ago
            start_date = self._subtract_months(period.start, 3)
            end_date = self._subtract_months(period.end, 3)
        else:
            raise ValueError(f"Unknown period_type: {period_type}")

        return TimePeriod(start=start_date, end=end_date)

    def _subtract_months(self, date: datetime, months: int) -> datetime:
        """Subtract months from a date, handling year boundaries."""
        month = date.month - months
        year = date.year

        while month <= 0:
            month += 12
            year -= 1

        # Handle day overflow (e.g., Jan 31 - 1 month should be Dec 31, not Dec 31)
        try:
            return date.replace(year=year, month=month)
        except ValueError:
            # Day doesn't exist in target month (e.g., Feb 30), use last day of month
            import calendar

            last_day = calendar.monthrange(year, month)[1]
            return date.replace(year=year, month=month, day=min(date.day, last_day))

    def parse_quarter_string(self, quarter_str: str) -> Tuple[int, Quarter]:
        """
        Parse quarter string like "Q1 2025", "2025 Q2", "Q3", etc.

        Args:
            quarter_str: Quarter string to parse

        Returns:
            Tuple of (year, quarter)
        """
        quarter_str = quarter_str.upper().strip()

        # Extract quarter
        quarter = None
        for q in Quarter:
            if q.name in quarter_str:
                quarter = q
                break

        if not quarter:
            raise ValueError(f"Could not parse quarter from: {quarter_str}")

        # Extract year
        import re

        year_match = re.search(r"\b(20\d{2})\b", quarter_str)
        if year_match:
            year = int(year_match.group(1))
        else:
            # Default to current year
            year = datetime.now().year

        return year, quarter

    def get_quarters_in_range(
        self, start_date: datetime, end_date: datetime
    ) -> List[Tuple[int, Quarter]]:
        """
        Get all quarters that overlap with the given date range.

        Args:
            start_date: Start of date range
            end_date: End of date range

        Returns:
            List of (year, quarter) tuples
        """
        quarters = []
        current_date = start_date.replace(day=1)  # Start from beginning of month

        while current_date < end_date:
            year = current_date.year
            month = current_date.month

            if month <= 3:
                quarter = Quarter.Q1
            elif month <= 6:
                quarter = Quarter.Q2
            elif month <= 9:
                quarter = Quarter.Q3
            else:
                quarter = Quarter.Q4

            if (year, quarter) not in quarters:
                quarters.append((year, quarter))

            # Move to next quarter
            if quarter == Quarter.Q4:
                current_date = datetime(year + 1, 1, 1, tzinfo=current_date.tzinfo)
            else:
                next_quarter_month = [1, 4, 7, 10][quarter.value]
                current_date = datetime(
                    year, next_quarter_month, 1, tzinfo=current_date.tzinfo
                )

        return quarters


def parse_advanced_date_range(
    date_str: str, fiscal_year_start_month: int = 1
) -> TimePeriod:
    """
    Parse advanced date range strings including quarters and fiscal years.

    Args:
        date_str: Date range string (e.g., "Q1 2025", "FY2025", "2024")
        fiscal_year_start_month: Month when fiscal year starts

    Returns:
        TimePeriod for the parsed date range
    """
    import re

    date_str = date_str.strip().upper()
    calculator = DateRangeCalculator(fiscal_year_start_month)

    # Quarter patterns
    if "Q" in date_str and any(q.name in date_str for q in Quarter):
        year, quarter = calculator.parse_quarter_string(date_str)
        if "FY" in date_str or "FISCAL" in date_str:
            return calculator.get_fiscal_quarter_range(year, quarter)
        else:
            return calculator.get_quarter_range(year, quarter)

    # Fiscal year patterns
    elif "FY" in date_str or "FISCAL" in date_str:
        # Look for patterns like FY2025, FY 2025, fiscal year 2025
        year_match = re.search(r"(?:FY|FISCAL\s*YEAR)\s*(\d{4})", date_str)
        if year_match:
            year = int(year_match.group(1))
            return calculator.get_fiscal_year_range(year)
        else:
            raise ValueError(f"Could not parse fiscal year from: {date_str}")

    # Calendar year pattern
    elif re.match(r"^\d{4}$", date_str):
        year = int(date_str)
        return calculator.get_calendar_year_range(year)

    else:
        raise ValueError(f"Could not parse date range: {date_str}")
