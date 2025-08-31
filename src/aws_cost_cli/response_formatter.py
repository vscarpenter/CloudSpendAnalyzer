"""Response formatting system with LLM integration and Rich terminal output."""

import json
from abc import ABC, abstractmethod
from datetime import datetime
from decimal import Decimal
from typing import Dict, Any, Optional, List

from .models import (
    CostData,
    CostResult,
    CostAmount,
    QueryParameters,
    TrendData,
    ForecastData,
)


class ResponseFormatter(ABC):
    """Abstract base class for response formatters."""

    @abstractmethod
    def format_response(
        self, cost_data: CostData, original_query: str, query_params: QueryParameters
    ) -> str:
        """Format cost data into a human-readable response."""
        pass


class LLMResponseFormatter(ResponseFormatter):
    """LLM-powered response formatter for natural language responses."""

    def __init__(
        self, llm_provider, fallback_formatter: Optional[ResponseFormatter] = None
    ):
        """
        Initialize LLM response formatter.

        Args:
            llm_provider: LLM provider instance (OpenAI, Anthropic, etc.)
            fallback_formatter: Fallback formatter if LLM fails
        """
        self.llm_provider = llm_provider
        self.fallback_formatter = fallback_formatter or SimpleResponseFormatter()

    def format_response(
        self, cost_data: CostData, original_query: str, query_params: QueryParameters
    ) -> str:
        """Format response using LLM for natural language generation."""
        try:
            if not self.llm_provider.is_available():
                return self.fallback_formatter.format_response(
                    cost_data, original_query, query_params
                )

            # Prepare cost data summary for LLM
            cost_summary = self._prepare_cost_summary(cost_data, query_params)

            # Generate natural language response
            response = self._generate_llm_response(cost_summary, original_query)

            return response

        except Exception as _e:
            # Fall back to simple formatter on any error
            return self.fallback_formatter.format_response(
                cost_data, original_query, query_params
            )

    def _prepare_cost_summary(
        self, cost_data: CostData, query_params: QueryParameters
    ) -> Dict[str, Any]:
        """Prepare cost data summary for LLM processing."""
        summary = {
            "total_cost": {
                "amount": float(cost_data.total_cost.amount),
                "currency": cost_data.total_cost.unit,
            },
            "time_period": {
                "start": cost_data.time_period.start.strftime("%Y-%m-%d"),
                "end": cost_data.time_period.end.strftime("%Y-%m-%d"),
            },
            "service": query_params.service,
            "granularity": (
                query_params.granularity.value
                if hasattr(query_params.granularity, "value")
                else query_params.granularity
            ),
            "results": [],
        }

        # Add trend analysis data if available
        if cost_data.trend_data:
            summary["trend_analysis"] = {
                "current_period_cost": float(
                    cost_data.trend_data.current_period.amount
                ),
                "comparison_period_cost": float(
                    cost_data.trend_data.comparison_period.amount
                ),
                "change_amount": float(cost_data.trend_data.change_amount.amount),
                "change_percentage": cost_data.trend_data.change_percentage,
                "trend_direction": cost_data.trend_data.trend_direction,
            }

        # Add forecast data if available
        if cost_data.forecast_data:
            summary["forecast"] = []
            for forecast in cost_data.forecast_data:
                summary["forecast"].append(
                    {
                        "period": {
                            "start": forecast.forecast_period.start.strftime(
                                "%Y-%m-%d"
                            ),
                            "end": forecast.forecast_period.end.strftime("%Y-%m-%d"),
                        },
                        "forecasted_amount": float(forecast.forecasted_amount.amount),
                        "confidence_interval": {
                            "lower": float(forecast.confidence_interval_lower.amount),
                            "upper": float(forecast.confidence_interval_upper.amount),
                        },
                        "accuracy": forecast.prediction_accuracy,
                    }
                )

        # Add detailed results (limit to avoid token limits)
        for result in cost_data.results[:10]:  # Limit to 10 results
            result_data = {
                "period": {
                    "start": result.time_period.start.strftime("%Y-%m-%d"),
                    "end": result.time_period.end.strftime("%Y-%m-%d"),
                },
                "total": {
                    "amount": float(result.total.amount),
                    "currency": result.total.unit,
                },
                "estimated": result.estimated,
            }

            # Add group data if available
            if result.groups:
                result_data["groups"] = []
                for group in result.groups[:5]:  # Limit groups
                    group_data = {"keys": group.keys, "metrics": {}}
                    for metric_name, cost_amount in group.metrics.items():
                        group_data["metrics"][metric_name] = {
                            "amount": float(cost_amount.amount),
                            "currency": cost_amount.unit,
                        }
                    result_data["groups"].append(group_data)

            summary["results"].append(result_data)

        return summary

    def _generate_llm_response(
        self, cost_summary: Dict[str, Any], original_query: str
    ) -> str:
        """Generate natural language response using LLM."""
        system_prompt = self._get_response_system_prompt()

        user_prompt = f"""Original user query: "{original_query}"

Cost data summary:
{json.dumps(cost_summary, indent=2)}

Please provide a clear, conversational response that directly answers the user's question about their AWS costs. Include specific amounts, time periods, and any relevant insights from the data."""

        if hasattr(self.llm_provider, "parse_query"):
            # This is one of our query parsing providers, adapt for response generation
            return self._generate_with_query_provider(system_prompt, user_prompt)
        else:
            # Direct LLM provider
            return self.llm_provider.generate_response(system_prompt, user_prompt)

    def _generate_with_query_provider(
        self, system_prompt: str, user_prompt: str
    ) -> str:
        """Generate response using query parsing providers (adapted)."""
        # For OpenAI provider
        if hasattr(self.llm_provider, "_get_client"):
            try:
                client = self.llm_provider._get_client()
                response = client.chat.completions.create(
                    model=getattr(self.llm_provider, "model", "gpt-3.5-turbo"),
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.3,
                    max_tokens=800,
                )
                return response.choices[0].message.content.strip()
            except Exception:
                pass

        # For Anthropic provider
        if (
            hasattr(self.llm_provider, "api_key")
            and "anthropic" in str(type(self.llm_provider)).lower()
        ):
            try:
                client = self.llm_provider._get_client()
                response = client.messages.create(
                    model=getattr(
                        self.llm_provider, "model", "claude-3-haiku-20240307"
                    ),
                    max_tokens=800,
                    temperature=0.3,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                return response.content[0].text.strip()
            except Exception:
                pass

        # For Ollama provider
        if hasattr(self.llm_provider, "base_url"):
            try:
                import requests

                full_prompt = f"{system_prompt}\n\n{user_prompt}"

                response = requests.post(
                    f"{self.llm_provider.base_url}/api/generate",
                    json={
                        "model": self.llm_provider.model,
                        "prompt": full_prompt,
                        "stream": False,
                        "options": {"temperature": 0.3, "num_predict": 800},
                    },
                    timeout=30,
                )

                if response.status_code == 200:
                    result = response.json()
                    return result.get("response", "").strip()
            except Exception:
                pass

        # Fallback if all methods fail
        raise RuntimeError("Could not generate response with LLM provider")

    def _get_response_system_prompt(self) -> str:
        """Get system prompt for response generation."""
        return """You are an AWS cost analysis assistant. Your job is to provide clear, conversational responses about AWS cost data.

Guidelines:
- Answer the user's question directly and conversationally
- Include specific dollar amounts and time periods from the data
- Use natural language, not technical jargon
- Highlight important insights or patterns in the spending
- If costs are estimated, mention that
- Format currency amounts clearly (e.g., "$123.45" not "123.45 USD")
- Keep responses concise but informative
- If there are multiple services or time periods, summarize the key points

Example good response:
"Your EC2 costs for January 2024 were $1,234.56. This represents a 15% increase from December, primarily due to increased usage of m5.large instances."

Do not include JSON, technical details about the API, or implementation details. Focus on answering the user's business question about their AWS spending."""


class SimpleResponseFormatter(ResponseFormatter):
    """Simple text-based response formatter without LLM."""

    def format_response(
        self, cost_data: CostData, original_query: str, query_params: QueryParameters
    ) -> str:
        """Format response using simple text formatting."""
        lines = []

        # Header with query context
        if query_params.service:
            service_text = f" for {query_params.service}"
        else:
            service_text = ""

        period_text = self._format_time_period(cost_data.time_period)
        lines.append(f"AWS Cost Summary{service_text} ({period_text})")
        lines.append("=" * len(lines[0]))
        lines.append("")

        # Total cost
        total_formatted = self._format_currency(cost_data.total_cost)
        lines.append(f"Total Cost: {total_formatted}")

        # Add trend analysis if available
        if cost_data.trend_data:
            lines.append("")
            lines.append("Trend Analysis:")
            lines.append("-" * 15)

            trend = cost_data.trend_data
            comparison_cost = self._format_currency(trend.comparison_period)
            change_amount = self._format_currency(trend.change_amount)

            direction_symbol = (
                "📈"
                if trend.trend_direction == "up"
                else "📉" if trend.trend_direction == "down" else "➡️"
            )

            lines.append(f"  Previous Period: {comparison_cost}")
            lines.append(
                f"  Change: {change_amount} ({trend.change_percentage:+.1f}%) {direction_symbol}"
            )
            lines.append(f"  Trend: {trend.trend_direction.title()}")

        # Add forecast if available
        if cost_data.forecast_data:
            lines.append("")
            lines.append("Cost Forecast:")
            lines.append("-" * 14)

            for i, forecast in enumerate(
                cost_data.forecast_data[:3], 1
            ):  # Show first 3 months
                period = self._format_time_period(forecast.forecast_period)
                forecasted = self._format_currency(forecast.forecasted_amount)
                lower = self._format_currency(forecast.confidence_interval_lower)
                upper = self._format_currency(forecast.confidence_interval_upper)

                accuracy_text = ""
                if forecast.prediction_accuracy:
                    accuracy_text = f" (confidence: {forecast.prediction_accuracy:.0%})"

                lines.append(f"  Month {i} ({period}): {forecasted}")
                lines.append(f"    Range: {lower} - {upper}{accuracy_text}")

        lines.append("")

        # Detailed breakdown if available
        if len(cost_data.results) > 1:
            lines.append("Breakdown by Time Period:")
            lines.append("-" * 30)

            for result in cost_data.results:
                period = self._format_time_period(result.time_period)
                cost = self._format_currency(result.total)
                estimated_text = " (estimated)" if result.estimated else ""
                lines.append(f"  {period}: {cost}{estimated_text}")

                # Show group breakdown if available
                if result.groups:
                    for group in result.groups[:5]:  # Limit to 5 groups
                        if group.keys:
                            group_name = " / ".join(group.keys)
                            for metric_name, cost_amount in group.metrics.items():
                                group_cost = self._format_currency(cost_amount)
                                lines.append(f"    {group_name}: {group_cost}")
            lines.append("")

        # Show group breakdown for single result if available
        elif len(cost_data.results) == 1 and cost_data.results[0].groups:
            lines.append("Service Breakdown:")
            lines.append("-" * 20)

            result = cost_data.results[0]
            for group in result.groups[:10]:  # Limit to 10 groups
                if group.keys:
                    group_name = " / ".join(group.keys)
                    for metric_name, cost_amount in group.metrics.items():
                        group_cost = self._format_currency(cost_amount)
                        lines.append(f"  {group_name}: {group_cost}")
            lines.append("")

        # Additional insights
        if cost_data.results:
            insights = self._generate_simple_insights(cost_data, query_params)
            if insights:
                lines.append("Insights:")
                lines.append("-" * 10)
                for insight in insights:
                    lines.append(f"• {insight}")

        return "\n".join(lines)

    def _format_currency(self, cost_amount: CostAmount) -> str:
        """Format currency amount."""
        if cost_amount.amount == 0:
            return "$0.00"

        # Format with appropriate precision
        if cost_amount.amount < Decimal("0.01"):
            return f"${cost_amount.amount:.4f}"
        else:
            return f"${cost_amount.amount:.2f}"

    def _format_time_period(self, time_period) -> str:
        """Format time period for display."""
        start_str = time_period.start.strftime("%Y-%m-%d")
        end_str = time_period.end.strftime("%Y-%m-%d")

        # If same day, show just the date
        if start_str == end_str:
            return start_str

        # If same month, show month/year
        if (
            time_period.start.year == time_period.end.year
            and time_period.start.month == time_period.end.month
        ):
            return time_period.start.strftime("%B %Y")

        return f"{start_str} to {end_str}"

    def _generate_simple_insights(
        self, cost_data: CostData, query_params: QueryParameters
    ) -> List[str]:
        """Generate simple insights from cost data."""
        insights = []

        # Check if any costs are estimated
        estimated_count = sum(1 for result in cost_data.results if result.estimated)
        if estimated_count > 0:
            insights.append(
                f"{estimated_count} of {len(cost_data.results)} periods contain estimated costs"
            )

        # Check for zero costs
        if cost_data.total_cost.amount == 0:
            insights.append("No costs found for the specified criteria")

        # Check for very small costs
        elif cost_data.total_cost.amount < Decimal("1.00"):
            insights.append("Total costs are less than $1.00")

        # Check for high costs (arbitrary threshold)
        elif cost_data.total_cost.amount > Decimal("1000.00"):
            insights.append("Total costs exceed $1,000")

        return insights


class RichResponseFormatter(ResponseFormatter):
    """Rich terminal output formatter with enhanced formatting."""

    def __init__(self, fallback_formatter: Optional[ResponseFormatter] = None):
        """
        Initialize Rich response formatter.

        Args:
            fallback_formatter: Fallback formatter if Rich is not available
        """
        self.fallback_formatter = fallback_formatter or SimpleResponseFormatter()
        self._rich_available = self._check_rich_availability()

    def _check_rich_availability(self) -> bool:
        """Check if Rich library is available."""
        try:
            import rich

            return True
        except ImportError:
            return False

    def format_response(
        self, cost_data: CostData, original_query: str, query_params: QueryParameters
    ) -> str:
        """Format response using Rich library for enhanced terminal output."""
        if not self._rich_available:
            return self.fallback_formatter.format_response(
                cost_data, original_query, query_params
            )

        try:
            from rich.console import Console
            from rich.table import Table
            from rich.panel import Panel
            from rich.text import Text
            from rich import box
            import io

            # Create console that writes to string
            string_io = io.StringIO()
            console = Console(file=string_io, width=80)

            # Create header
            if query_params.service:
                title = f"AWS {query_params.service} Cost Summary"
            else:
                title = "AWS Cost Summary"

            period_text = self._format_time_period(cost_data.time_period)
            subtitle = f"Period: {period_text}"

            # Total cost panel
            total_cost = self._format_currency(cost_data.total_cost)
            total_text = Text(
                total_cost,
                style="bold green" if cost_data.total_cost.amount > 0 else "bold red",
            )

            console.print(
                Panel(
                    total_text,
                    title=title,
                    subtitle=subtitle,
                    border_style="blue",
                    padding=(1, 2),
                )
            )

            # Add trend analysis panel if available
            if cost_data.trend_data:
                self._add_trend_panel(console, cost_data.trend_data)

            # Add forecast panel if available
            if cost_data.forecast_data:
                self._add_forecast_panel(console, cost_data.forecast_data)

            # Detailed breakdown table
            if len(cost_data.results) > 1:
                table = Table(
                    title="Cost Breakdown",
                    box=box.ROUNDED,
                    show_header=True,
                    header_style="bold blue",
                )

                table.add_column("Period", style="cyan")
                table.add_column("Cost", justify="right", style="green")
                table.add_column("Status", justify="center")

                for result in cost_data.results:
                    period = self._format_time_period(result.time_period)
                    cost = self._format_currency(result.total)
                    status = "📊 Est." if result.estimated else "✅ Final"

                    table.add_row(period, cost, status)

                console.print(table)

            # Group breakdown if available
            if cost_data.results and any(result.groups for result in cost_data.results):
                self._add_group_breakdown(console, cost_data)

            # Insights
            insights = self._generate_rich_insights(cost_data, query_params)
            if insights:
                console.print(
                    Panel(
                        "\n".join(f"• {insight}" for insight in insights),
                        title="💡 Insights",
                        border_style="yellow",
                        padding=(1, 2),
                    )
                )

            return string_io.getvalue()

        except Exception as _e:
            # Fall back to simple formatter on any error
            return self.fallback_formatter.format_response(
                cost_data, original_query, query_params
            )

    def _add_group_breakdown(self, console, cost_data: CostData):
        """Add group breakdown table to console output."""
        try:
            from rich.table import Table
            from rich import box

            # Find the result with the most groups for display
            best_result = max(
                cost_data.results, key=lambda r: len(r.groups) if r.groups else 0
            )

            if not best_result.groups:
                return

            table = Table(
                title="Service Breakdown",
                box=box.SIMPLE,
                show_header=True,
                header_style="bold magenta",
            )

            table.add_column("Service/Resource", style="cyan")
            table.add_column("Cost", justify="right", style="green")

            # Sort groups by cost (descending)
            sorted_groups = sorted(
                best_result.groups,
                key=lambda g: (
                    max(cost.amount for cost in g.metrics.values()) if g.metrics else 0
                ),
                reverse=True,
            )

            for group in sorted_groups[:10]:  # Limit to top 10
                if group.keys:
                    service_name = " / ".join(group.keys)

                    # Get the primary cost metric
                    if group.metrics:
                        primary_cost = next(iter(group.metrics.values()))
                        cost_str = self._format_currency(primary_cost)
                        table.add_row(service_name, cost_str)

            console.print(table)

        except Exception:
            # Skip group breakdown on any error
            pass

    def _add_trend_panel(self, console, trend_data: TrendData):
        """Add trend analysis panel to console output."""
        try:
            from rich.panel import Panel
            from rich.text import Text
            from rich.columns import Columns

            # Format trend information
            current_cost = self._format_currency(trend_data.current_period)
            comparison_cost = self._format_currency(trend_data.comparison_period)
            change_amount = self._format_currency(trend_data.change_amount)

            # Choose colors and symbols based on trend
            if trend_data.trend_direction == "up":
                trend_color = "red"
                trend_symbol = "📈"
            elif trend_data.trend_direction == "down":
                trend_color = "green"
                trend_symbol = "📉"
            else:
                trend_color = "yellow"
                trend_symbol = "➡️"

            # Create trend text
            trend_text = Text()
            trend_text.append(f"Previous: {comparison_cost}\n")
            trend_text.append(f"Current: {current_cost}\n")
            trend_text.append(f"Change: {change_amount} (", style="white")
            trend_text.append(
                f"{trend_data.change_percentage:+.1f}%", style=f"bold {trend_color}"
            )
            trend_text.append(f") {trend_symbol}", style="white")

            console.print(
                Panel(
                    trend_text,
                    title="📊 Trend Analysis",
                    border_style=trend_color,
                    padding=(1, 2),
                )
            )

        except Exception:
            # Skip trend panel on any error
            pass

    def _add_forecast_panel(self, console, forecast_data: List[ForecastData]):
        """Add forecast panel to console output."""
        try:
            from rich.panel import Panel
            from rich.table import Table
            from rich import box

            table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")

            table.add_column("Month", style="cyan")
            table.add_column("Forecast", justify="right", style="green")
            table.add_column("Range", justify="center", style="yellow")
            table.add_column("Confidence", justify="center")

            for i, forecast in enumerate(forecast_data[:3], 1):  # Show first 3 months
                _period = self._format_time_period(forecast.forecast_period)
                forecasted = self._format_currency(forecast.forecasted_amount)
                lower = self._format_currency(forecast.confidence_interval_lower)
                upper = self._format_currency(forecast.confidence_interval_upper)

                confidence = ""
                if forecast.prediction_accuracy:
                    confidence = f"{forecast.prediction_accuracy:.0%}"

                table.add_row(
                    f"Month {i}", forecasted, f"{lower} - {upper}", confidence
                )

            console.print(
                Panel(
                    table,
                    title="🔮 Cost Forecast",
                    border_style="magenta",
                    padding=(1, 2),
                )
            )

        except Exception:
            # Skip forecast panel on any error
            pass

    def _format_currency(self, cost_amount: CostAmount) -> str:
        """Format currency amount."""
        if cost_amount.amount == 0:
            return "$0.00"

        # Format with appropriate precision
        if cost_amount.amount < Decimal("0.01"):
            return f"${cost_amount.amount:.4f}"
        else:
            return f"${cost_amount.amount:.2f}"

    def _format_time_period(self, time_period) -> str:
        """Format time period for display."""
        start_str = time_period.start.strftime("%Y-%m-%d")
        end_str = time_period.end.strftime("%Y-%m-%d")

        # If same day, show just the date
        if start_str == end_str:
            return start_str

        # If same month, show month/year
        if (
            time_period.start.year == time_period.end.year
            and time_period.start.month == time_period.end.month
        ):
            return time_period.start.strftime("%B %Y")

        return f"{start_str} to {end_str}"

    def _generate_rich_insights(
        self, cost_data: CostData, query_params: QueryParameters
    ) -> List[str]:
        """Generate insights for Rich display."""
        insights = []

        # Check if any costs are estimated
        estimated_count = sum(1 for result in cost_data.results if result.estimated)
        if estimated_count > 0:
            insights.append(
                f"{estimated_count} of {len(cost_data.results)} periods contain estimated costs"
            )

        # Check for zero costs
        if cost_data.total_cost.amount == 0:
            insights.append("No costs found for the specified criteria")

        # Check for very small costs
        elif cost_data.total_cost.amount < Decimal("1.00"):
            insights.append("Total costs are less than $1.00")

        # Check for high costs (arbitrary threshold)
        elif cost_data.total_cost.amount > Decimal("1000.00"):
            insights.append("Total costs exceed $1,000 - consider cost optimization")

        # Check for service-specific insights
        if query_params.service:
            if (
                query_params.service.upper() == "EC2"
                and cost_data.total_cost.amount > Decimal("500.00")
            ):
                insights.append(
                    "High EC2 costs detected - review instance types and utilization"
                )
            elif (
                query_params.service.upper() == "S3"
                and cost_data.total_cost.amount > Decimal("100.00")
            ):
                insights.append(
                    "Consider S3 storage class optimization for cost savings"
                )

        return insights


class ResponseGenerator:
    """Main response generator that coordinates different formatters."""

    def __init__(self, llm_provider=None, output_format: str = "simple"):
        """
        Initialize response generator.

        Args:
            llm_provider: LLM provider for natural language responses
            output_format: Output format ("simple", "rich", "llm")
        """
        self.llm_provider = llm_provider
        self.output_format = output_format.lower()

        # Initialize formatters
        self.simple_formatter = SimpleResponseFormatter()
        self.rich_formatter = RichResponseFormatter(
            fallback_formatter=self.simple_formatter
        )

        if llm_provider:
            self.llm_formatter = LLMResponseFormatter(
                llm_provider, fallback_formatter=self.rich_formatter
            )
        else:
            self.llm_formatter = None

    def format_response(
        self, cost_data: CostData, original_query: str, query_params: QueryParameters
    ) -> str:
        """
        Format cost data response based on configuration.

        Args:
            cost_data: Cost data to format
            original_query: Original user query
            query_params: Parsed query parameters

        Returns:
            Formatted response string
        """
        try:
            if self.output_format == "llm" and self.llm_formatter:
                return self.llm_formatter.format_response(
                    cost_data, original_query, query_params
                )
            elif self.output_format == "rich":
                return self.rich_formatter.format_response(
                    cost_data, original_query, query_params
                )
            else:
                return self.simple_formatter.format_response(
                    cost_data, original_query, query_params
                )

        except Exception as _e:
            # Always fall back to simple formatter on any error
            return self.simple_formatter.format_response(
                cost_data, original_query, query_params
            )
