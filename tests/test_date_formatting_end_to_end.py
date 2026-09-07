"""End-to-end tests for date formatting across the CLI response formatting workflow."""

import pytest
import time
from datetime import datetime, timezone, timedelta
from decimal import Decimal

from src.aws_cost_cli.models import (
    CostData,
    CostResult,
    CostAmount,
    TimePeriod,
    QueryParameters,
    TimePeriodGranularity,
)
from src.aws_cost_cli.response_formatter import SimpleResponseFormatter
from src.aws_cost_cli.date_formatter import DateFormatter


class TestDateFormattingEndToEnd:
    """End-to-end tests for date formatting behaviour."""

    def setup_method(self):
        """Set up test fixtures."""
        self.date_formatter = DateFormatter()

    def test_edge_case_unusual_time_periods(self):
        """Test edge cases with unusual time periods."""
        # Leap day single day
        leap_year_period = TimePeriod(
            start=datetime(2024, 2, 29, tzinfo=timezone.utc),
            end=datetime(2024, 3, 1, tzinfo=timezone.utc),
        )
        assert self.date_formatter.format_time_period(leap_year_period) == "February 29, 2024"

        # Month boundary single day
        month_boundary_period = TimePeriod(
            start=datetime(2024, 1, 31, tzinfo=timezone.utc),
            end=datetime(2024, 2, 1, tzinfo=timezone.utc),
        )
        assert self.date_formatter.format_time_period(month_boundary_period) == "January 31, 2024"

        # Year boundary single day
        year_boundary_period = TimePeriod(
            start=datetime(2024, 12, 31, tzinfo=timezone.utc),
            end=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        assert self.date_formatter.format_time_period(year_boundary_period) == "December 31, 2024"

    def test_date_formatter_fallback_behavior(self):
        """Test that date formatter handles edge cases gracefully."""
        # Invalid period (end before start) should fall back to ISO format
        invalid_period = TimePeriod(
            start=datetime(2024, 1, 15, tzinfo=timezone.utc),
            end=datetime(2024, 1, 10, tzinfo=timezone.utc),
        )

        result = self.date_formatter.safe_format_time_period(invalid_period)
        assert isinstance(result, str)
        assert len(result) > 0
        assert "2024-01-15" in result and "2024-01-10" in result

    def test_performance_with_large_datasets(self):
        """Test date formatting performance with large datasets through the formatter."""
        large_results = []
        for i in range(100):
            period = TimePeriod(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=i),
                end=datetime(2024, 1, 2, tzinfo=timezone.utc) + timedelta(days=i),
            )
            large_results.append(
                CostResult(
                    time_period=period,
                    total=CostAmount(Decimal("10.00"), "USD"),
                    groups=[],
                    estimated=False,
                )
            )

        large_cost_data = CostData(
            results=large_results,
            time_period=TimePeriod(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 4, 10, tzinfo=timezone.utc),
            ),
            total_cost=CostAmount(Decimal("1000.00"), "USD"),
            currency="USD",
        )

        query_params = QueryParameters(
            service="EC2",
            time_period=large_cost_data.time_period,
            granularity=TimePeriodGranularity.DAILY,
        )

        formatter = SimpleResponseFormatter()
        start_time = time.time()
        response = formatter.format_response(large_cost_data, "Daily EC2 costs", query_params)
        elapsed = time.time() - start_time

        assert elapsed < 1.0, "Date formatting took too long"
        assert isinstance(response, str)
        assert len(response) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
