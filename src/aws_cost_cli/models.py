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


class MetricType(Enum):
    """Cost metric types."""

    BLENDED_COST = "BlendedCost"
    UNBLENDED_COST = "UnblendedCost"
    NET_UNBLENDED_COST = "NetUnblendedCost"
    USAGE_QUANTITY = "UsageQuantity"


class DateFormatStyle(Enum):
    """Date formatting style options."""

    SMART = "smart"      # Automatically choose the most appropriate format
    VERBOSE = "verbose"  # Always include full context (e.g., "January 1-31, 2025")
    COMPACT = "compact"  # Use shortest reasonable format (e.g., "Jan 2025")


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
    cost_allocation_tags: Optional[List[str]] = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = [MetricType.BLENDED_COST]


@dataclass
class CostAmount:
    """Represents a cost amount with currency."""

    amount: Decimal
    unit: str = "USD"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-friendly mapping (amount as float)."""
        return {"amount": float(self.amount), "currency": self.unit}


@dataclass
class Group:
    """Represents a grouped cost result."""

    keys: List[str]
    metrics: Dict[str, CostAmount]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-friendly mapping."""
        return {
            "keys": self.keys,
            "metrics": {
                name: amount.to_dict() for name, amount in self.metrics.items()
            },
        }


@dataclass
class CostResult:
    """Individual cost result for a time period."""

    time_period: TimePeriod
    total: CostAmount
    groups: List[Group]
    estimated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-friendly mapping."""
        return {
            "period": {
                "start": self.time_period.start.isoformat(),
                "end": self.time_period.end.isoformat(),
            },
            "total": self.total.to_dict(),
            "estimated": self.estimated,
            "groups": [group.to_dict() for group in self.groups],
        }


@dataclass
class CostData:
    """Complete cost data response."""

    results: List[CostResult]
    time_period: TimePeriod
    total_cost: CostAmount
    currency: str = "USD"
    group_definitions: List[str] = None

    def __post_init__(self):
        if self.group_definitions is None:
            self.group_definitions = []

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the cost-data portion of a query response to a mapping.

        This produces the ``total_cost``/``time_period``/``results`` structure
        used by the CLI's JSON output. The outer envelope (``query``,
        ``success``, ``metadata``) is assembled by the caller.
        """
        return {
            "total_cost": self.total_cost.to_dict(),
            "time_period": {
                "start": self.time_period.start.isoformat(),
                "end": self.time_period.end.isoformat(),
            },
            "results": [result.to_dict() for result in self.results],
        }


@dataclass
class DateFormattingConfig:
    """Date formatting configuration options."""

    enabled: bool = True
    format_style: DateFormatStyle = DateFormatStyle.SMART
    fiscal_year_start_month: int = 1  # January by default
    locale: str = "en_US"
    fallback_to_iso: bool = True

    def __post_init__(self):
        # Validate fiscal year start month
        if not 1 <= self.fiscal_year_start_month <= 12:
            raise ValueError(f"fiscal_year_start_month must be between 1 and 12, got {self.fiscal_year_start_month}")
        
        # Convert string format_style to enum if needed
        if isinstance(self.format_style, str):
            try:
                self.format_style = DateFormatStyle(self.format_style.lower())
            except ValueError:
                self.format_style = DateFormatStyle.SMART


@dataclass
class Config:
    """Application configuration."""

    llm_provider: str = "ollama"
    llm_config: Dict[str, Any] = None
    default_profile: Optional[str] = None
    cache_ttl: int = 3600  # 1 hour in seconds
    output_format: str = "simple"
    default_currency: str = "USD"
    fallback_providers: List[str] = None
    date_formatting: DateFormattingConfig = None

    def __post_init__(self):
        if self.llm_config is None:
            self.llm_config = {}
        if self.fallback_providers is None:
            self.fallback_providers = ["ollama", "openai", "anthropic", "gemini"]
        if self.date_formatting is None:
            self.date_formatting = DateFormattingConfig()
