"""Simple utility functions."""

from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple
from models import TimePeriod


def parse_date_range(query: str) -> Optional[TimePeriod]:
    """Parse natural language date range into TimePeriod."""
    query_lower = query.lower()
    now = datetime.now(timezone.utc)
    import re
    
    # Define months for checking
    months = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12
    }
    
    # Handle common patterns
    if "last month" in query_lower or "previous month" in query_lower:
        # First day of last month to last day of last month
        first_day_this_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        last_day_last_month = first_day_this_month - timedelta(days=1)
        first_day_last_month = last_day_last_month.replace(day=1)
        return TimePeriod(
            start=first_day_last_month,
            end=last_day_last_month.replace(hour=23, minute=59, second=59)
        )
    
    elif "this month" in query_lower or "current month" in query_lower:
        # First day of this month to now (or end of month if querying for full month)
        first_day = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        # For current month, use current time as end
        return TimePeriod(start=first_day, end=now)
    
    elif "last year" in query_lower or "previous year" in query_lower:
        # January 1 of last year to December 31 of last year
        last_year = now.year - 1
        return TimePeriod(
            start=datetime(last_year, 1, 1, tzinfo=timezone.utc),
            end=datetime(last_year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
        )
    
    elif "this year" in query_lower or "current year" in query_lower:
        # January 1 of this year to now
        return TimePeriod(
            start=datetime(now.year, 1, 1, tzinfo=timezone.utc),
            end=now
        )
    
    elif "yesterday" in query_lower:
        yesterday = now - timedelta(days=1)
        return TimePeriod(
            start=yesterday.replace(hour=0, minute=0, second=0, microsecond=0),
            end=yesterday.replace(hour=23, minute=59, second=59)
        )
    
    elif "today" in query_lower:
        return TimePeriod(
            start=now.replace(hour=0, minute=0, second=0, microsecond=0),
            end=now
        )
    
    elif "last week" in query_lower:
        # Last 7 days
        return TimePeriod(
            start=now - timedelta(days=7),
            end=now
        )
    
    elif "last 30 days" in query_lower:
        return TimePeriod(
            start=now - timedelta(days=30),
            end=now
        )
    
    elif "last 90 days" in query_lower:
        return TimePeriod(
            start=now - timedelta(days=90),
            end=now
        )
    
    elif "last 3 months" in query_lower:
        return TimePeriod(
            start=now - timedelta(days=90),
            end=now
        )
    
    elif "last 6 months" in query_lower:
        return TimePeriod(
            start=now - timedelta(days=180),
            end=now
        )
    
    # Check for specific month names FIRST (before year-only check)
    year_match = re.search(r'\b(20\d{2})\b', query)
    
    # Check if a month is mentioned
    
    for month_name, month_num in months.items():
        if month_name in query_lower:
            # Month is mentioned, handle month+year combination
            year = now.year
            if year_match:
                year = int(year_match.group(1))
            elif "last" in query_lower:
                # "last January" means January of last year if we're past January
                if now.month > month_num:
                    year = now.year
                else:
                    year = now.year - 1
            
            # Get last day of the month
            if month_num == 12:
                last_day = 31
            elif month_num in [4, 6, 9, 11]:
                last_day = 30
            elif month_num == 2:
                # Check for leap year
                if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
                    last_day = 29
                else:
                    last_day = 28
            else:
                last_day = 31
            
            return TimePeriod(
                start=datetime(year, month_num, 1, tzinfo=timezone.utc),
                end=datetime(year, month_num, last_day, 23, 59, 59, tzinfo=timezone.utc)
            )
    
    # If no month mentioned but year is, treat as full year
    if year_match:
        year = int(year_match.group(1))
        if year == now.year:
            # Current year - from Jan 1 to now
            return TimePeriod(
                start=datetime(year, 1, 1, tzinfo=timezone.utc),
                end=now
            )
        else:
            # Past year - full year
            return TimePeriod(
                start=datetime(year, 1, 1, tzinfo=timezone.utc),
                end=datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
            )
    
    # Default to last 30 days if no pattern matches
    return TimePeriod(
        start=now - timedelta(days=30),
        end=now
    )


def format_currency(amount: float, currency: str = "USD") -> str:
    """Format currency for display."""
    symbols = {
        "USD": "$",
        "EUR": "€",
        "GBP": "£",
        "JPY": "¥"
    }
    symbol = symbols.get(currency, currency + " ")
    return f"{symbol}{amount:,.2f}"
