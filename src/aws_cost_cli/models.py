"""Core data models and type definitions for AWS Cost CLI."""

from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum


class TimePeriodGranularity(Enum):
    """Time period granularity options."""
    DAILY = "DAILY"
    MONTHLY = "MONTHLY"
    HOURLY = "HOURLY"


class DateRangeType(Enum):
    """Types of date ranges for advanced queries."""
    QUARTER = "QUARTER"
    FISCAL_YEAR = "FISCAL_YEAR"
    CALENDAR_YEAR = "CALENDAR_YEAR"
    CUSTOM = "CUSTOM"


class TrendAnalysisType(Enum):
    """Types of trend analysis."""
    PERIOD_OVER_PERIOD = "PERIOD_OVER_PERIOD"
    YEAR_OVER_YEAR = "YEAR_OVER_YEAR"
    MONTH_OVER_MONTH = "MONTH_OVER_MONTH"
    QUARTER_OVER_QUARTER = "QUARTER_OVER_QUARTER"


class MetricType(Enum):
    """Cost metric types."""
    BLENDED_COST = "BlendedCost"
    UNBLENDED_COST = "UnblendedCost"
    NET_UNBLENDED_COST = "NetUnblendedCost"
    USAGE_QUANTITY = "UsageQuantity"


@dataclass
class TimePeriod:
    """Represents a time period for cost queries."""
    start: datetime
    end: datetime


@dataclass
class QueryParameters:
    """Parameters extracted from natural language queries."""
    service: Optional[str] = None
    time_period: Optional[TimePeriod] = None
    granularity: TimePeriodGranularity = TimePeriodGranularity.MONTHLY
    metrics: List[MetricType] = None
    group_by: Optional[List[str]] = None
    # Advanced query features
    date_range_type: Optional[DateRangeType] = None
    fiscal_year_start_month: int = 1  # January by default
    trend_analysis: Optional[TrendAnalysisType] = None
    comparison_period: Optional[TimePeriod] = None
    include_forecast: bool = False
    forecast_months: int = 3
    cost_allocation_tags: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = [MetricType.BLENDED_COST]


@dataclass
class CostAmount:
    """Represents a cost amount with currency."""
    amount: Decimal
    unit: str = "USD"


@dataclass
class Group:
    """Represents a grouped cost result."""
    keys: List[str]
    metrics: Dict[str, CostAmount]


@dataclass
class CostResult:
    """Individual cost result for a time period."""
    time_period: TimePeriod
    total: CostAmount
    groups: List[Group]
    estimated: bool = False


@dataclass
class TrendData:
    """Trend analysis data for period-over-period comparisons."""
    current_period: CostAmount
    comparison_period: CostAmount
    change_amount: CostAmount
    change_percentage: float
    trend_direction: str  # "up", "down", "stable"


@dataclass
class ForecastData:
    """Cost forecast data."""
    forecasted_amount: CostAmount
    confidence_interval_lower: CostAmount
    confidence_interval_upper: CostAmount
    forecast_period: TimePeriod
    prediction_accuracy: Optional[float] = None


@dataclass
class CostData:
    """Complete cost data response."""
    results: List[CostResult]
    time_period: TimePeriod
    total_cost: CostAmount
    currency: str = "USD"
    group_definitions: List[str] = None
    # Advanced features
    trend_data: Optional[TrendData] = None
    forecast_data: Optional[List[ForecastData]] = None
    
    def __post_init__(self):
        if self.group_definitions is None:
            self.group_definitions = []


@dataclass
class Config:
    """Application configuration."""
    llm_provider: str = "openai"
    llm_config: Dict[str, Any] = None
    default_profile: Optional[str] = None
    cache_ttl: int = 3600  # 1 hour in seconds
    output_format: str = "simple"
    default_currency: str = "USD"
    
    def __post_init__(self):
        if self.llm_config is None:
            self.llm_config = {}