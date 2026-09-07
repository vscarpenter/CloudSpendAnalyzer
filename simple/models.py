"""Simple data models for AWS Cost CLI."""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any
from decimal import Decimal


@dataclass
class TimePeriod:
    """Time period for cost queries."""
    start: datetime
    end: datetime


@dataclass
class QueryParameters:
    """Parameters extracted from natural language query."""
    service: Optional[str] = None
    time_period: Optional[TimePeriod] = None
    group_by: Optional[List[str]] = None


@dataclass
class CostData:
    """AWS cost data response."""
    total_cost: Decimal
    currency: str
    time_period: TimePeriod
    service_breakdown: Optional[Dict[str, Decimal]] = None
    raw_data: Optional[Dict[str, Any]] = None
