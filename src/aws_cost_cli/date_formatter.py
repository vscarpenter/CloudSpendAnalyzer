"""Human-readable formatting for cost-query time periods.

Turns a :class:`~aws_cost_cli.models.TimePeriod` into a friendly string such as
``"January 2025"``, ``"Q1 2025"`` or ``"2025"``. AWS Cost Explorer reports
periods with an *exclusive* end date (the first instant after the range), so a
single calendar month is ``2025-01-01`` to ``2025-02-01``. Anything that is not
a clean single day/month/quarter/year is rendered as an inclusive date span.
"""

import logging
from datetime import datetime, timedelta

from .models import TimePeriod

logger = logging.getLogger(__name__)

_MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
_QUARTER_STARTS = {1: 1, 4: 2, 7: 3, 10: 4}  # start month -> quarter number


class DateFormatter:
    """Formats time periods using a single "smart" style.

    The optional ``date_formatting_config`` (a ``DateFormattingConfig``) is
    honoured for the ``enabled`` and ``fallback_to_iso`` flags. The ``format_style``
    is accepted for backwards compatibility but only smart output is produced.
    """

    def __init__(self, date_formatting_config=None):
        from .models import DateFormattingConfig

        self.config = date_formatting_config or DateFormattingConfig()
        # Retained for backwards compatibility with older callers/tests.
        self.month_names = _MONTH_NAMES

    def format_time_period(self, time_period: TimePeriod) -> str:
        """Return a smart, human-readable label for ``time_period``.

        Raises ``ValueError``/``AttributeError`` for missing data; callers that
        need a guarantee of no exceptions should use :meth:`safe_format_time_period`.
        When formatting fails and ``config.fallback_to_iso`` is set, an ISO date
        range is returned instead of raising.
        """
        if time_period is None:
            raise ValueError("time_period cannot be None")
        if not hasattr(time_period, "start") or not hasattr(time_period, "end"):
            raise AttributeError("TimePeriod must have start and end attributes")
        if time_period.start is None or time_period.end is None:
            raise ValueError("TimePeriod start and end cannot be None")

        if not self.config.enabled:
            return self._iso(time_period)

        try:
            return self._smart_label(time_period)
        except Exception as error:  # noqa: BLE001 - fall back rather than crash callers
            logger.warning("Date formatting failed: %s: %s", type(error).__name__, error)
            if self.config.fallback_to_iso:
                return self._iso(time_period)
            raise

    def safe_format_time_period(self, time_period: TimePeriod) -> str:
        """Format ``time_period``, never raising; returns a fallback string instead."""
        try:
            return self.format_time_period(time_period)
        except Exception as error:  # noqa: BLE001 - this method must never raise
            logger.warning("Falling back from date formatting: %s", error)
            try:
                return self._iso(time_period)
            except Exception:  # noqa: BLE001
                return "Invalid date range"

    def _smart_label(self, time_period: TimePeriod) -> str:
        start, end = time_period.start, time_period.end

        # Reversed/empty ranges are nonsensical to render as a span.
        if start >= end:
            raise ValueError(f"start {start} is not before end {end}")

        if self._is_single_day(start, end):
            return f"{_MONTH_NAMES[start.month - 1]} {start.day}, {start.year}"

        # Detection below works on clean month boundaries (1st of month at midnight).
        end_midnight = end.replace(hour=0, minute=0, second=0, microsecond=0)
        if start.day == 1 and end_midnight.day == 1:
            months = (end_midnight.year - start.year) * 12 + (end_midnight.month - start.month)
            if months == 1:
                return f"{_MONTH_NAMES[start.month - 1]} {start.year}"
            if months == 3 and start.month in _QUARTER_STARTS:
                return f"Q{_QUARTER_STARTS[start.month]} {start.year}"
            if months == 12 and start.month == 1:
                return str(start.year)
            if 1 < months < 12:
                return self._multi_month_label(start, end_midnight)

        return self._custom_range_label(start, end)

    @staticmethod
    def _is_single_day(start: datetime, end: datetime) -> bool:
        """True when the range covers exactly one calendar day (time ignored)."""
        return end.date() == start.date() + timedelta(days=1) or start.date() == end.date()

    @staticmethod
    def _multi_month_label(start: datetime, end_midnight: datetime) -> str:
        """Render a span of whole months, e.g. ``"November 2024 - February 2025"``."""
        last_month = end_midnight - timedelta(days=1)  # inclusive final month
        start_name = _MONTH_NAMES[start.month - 1]
        end_name = _MONTH_NAMES[last_month.month - 1]
        if start.year == last_month.year:
            return f"{start_name} - {end_name} {start.year}"
        return f"{start_name} {start.year} - {end_name} {last_month.year}"

    @staticmethod
    def _custom_range_label(start: datetime, end: datetime) -> str:
        """Render an arbitrary range using the inclusive last day, e.g.
        ``"January 15 - February 19, 2025"`` or ``"January 15 - 24, 2025"``."""
        last_day = end - timedelta(days=1)  # exclusive end -> inclusive last day
        start_name = _MONTH_NAMES[start.month - 1]
        end_name = _MONTH_NAMES[last_day.month - 1]
        if start.year == last_day.year and start.month == last_day.month:
            return f"{start_name} {start.day} - {last_day.day}, {start.year}"
        if start.year == last_day.year:
            return f"{start_name} {start.day} - {end_name} {last_day.day}, {start.year}"
        return (
            f"{start_name} {start.day}, {start.year} - "
            f"{end_name} {last_day.day}, {last_day.year}"
        )

    @staticmethod
    def _iso(time_period: TimePeriod) -> str:
        """Simple, reliable ISO fallback: ``"YYYY-MM-DD to YYYY-MM-DD"``."""
        return f"{time_period.start:%Y-%m-%d} to {time_period.end:%Y-%m-%d}"
