"""Data export functionality for AWS Cost CLI."""

import csv
import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, List

from .models import CostData, QueryParameters, DateFormattingConfig
from .date_formatter import DateFormatter


class DataExporter(ABC):
    """Abstract base class for data exporters."""

    @abstractmethod
    def export(
        self, cost_data: CostData, query_params: QueryParameters, output_path: str
    ) -> str:
        """Export cost data to specified format."""
        pass


class CSVExporter(DataExporter):
    """CSV data exporter."""

    def __init__(self, date_formatting_config: Optional[DateFormattingConfig] = None):
        """Initialize CSV exporter with date formatter."""
        self.date_formatter = DateFormatter(date_formatting_config)

    def export(
        self, cost_data: CostData, query_params: QueryParameters, output_path: str
    ) -> str:
        """Export cost data to CSV format."""
        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)

            # Write header with metadata
            writer.writerow(["# AWS Cost Data Export"])
            writer.writerow(["# Generated:", datetime.now().isoformat()])
            writer.writerow(
                ["# Query:", getattr(query_params, "original_query", "N/A")]
            )
            writer.writerow(["# Service:", query_params.service or "All Services"])

            # Format the overall time period
            formatted_period = self.date_formatter.safe_format_time_period(
                cost_data.time_period
            )
            writer.writerow(
                [
                    "# Period:",
                    formatted_period,
                ]
            )
            writer.writerow(
                [
                    "# Total Cost:",
                    f"{cost_data.total_cost.amount} {cost_data.total_cost.unit}",
                ]
            )
            writer.writerow([])  # Empty row

            # Write main data headers
            headers = [
                "Period Start",
                "Period End",
                "Formatted Period",
                "Total Cost",
                "Currency",
                "Estimated",
            ]

            # Add group headers if available
            if cost_data.results and cost_data.results[0].groups:
                sample_group = cost_data.results[0].groups[0]
                if sample_group.keys:
                    headers.extend(["Group Keys", "Group Cost"])

            writer.writerow(headers)

            # Write data rows
            for result in cost_data.results:
                # Format the time period for this result
                formatted_period = self.date_formatter.safe_format_time_period(
                    result.time_period
                )

                base_row = [
                    result.time_period.start.date().isoformat(),
                    result.time_period.end.date().isoformat(),
                    formatted_period,
                    float(result.total.amount),
                    result.total.unit,
                    result.estimated,
                ]

                if result.groups:
                    # Write a row for each group
                    for group in result.groups:
                        row = base_row.copy()
                        if group.keys:
                            row.append(" / ".join(group.keys))
                            # Get primary cost metric
                            if group.metrics:
                                primary_cost = next(iter(group.metrics.values()))
                                row.append(float(primary_cost.amount))
                            else:
                                row.append(0.0)
                        writer.writerow(row)
                else:
                    writer.writerow(base_row)

        return output_path


class JSONExporter(DataExporter):
    """JSON data exporter."""

    def __init__(self, date_formatting_config: Optional[DateFormattingConfig] = None):
        """Initialize JSON exporter with date formatter."""
        self.date_formatter = DateFormatter(date_formatting_config)

    def export(
        self, cost_data: CostData, query_params: QueryParameters, output_path: str
    ) -> str:
        """Export cost data to JSON format."""
        # Format the overall time period
        formatted_period = self.date_formatter.safe_format_time_period(
            cost_data.time_period
        )

        # Convert cost data to JSON-serializable format
        export_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "query": getattr(query_params, "original_query", "N/A"),
                "service": query_params.service,
                "granularity": (
                    query_params.granularity.value
                    if hasattr(query_params.granularity, "value")
                    else str(query_params.granularity)
                ),
                "time_period": {
                    "start": cost_data.time_period.start.isoformat(),
                    "end": cost_data.time_period.end.isoformat(),
                    "formatted": formatted_period,
                },
            },
            "summary": {
                "total_cost": {
                    "amount": float(cost_data.total_cost.amount),
                    "currency": cost_data.total_cost.unit,
                },
                "currency": cost_data.currency,
                "group_definitions": cost_data.group_definitions,
            },
            "results": [],
        }

        # Add detailed results
        for result in cost_data.results:
            # Format the time period for this result
            formatted_period = self.date_formatter.safe_format_time_period(
                result.time_period
            )

            result_data = {
                "time_period": {
                    "start": result.time_period.start.isoformat(),
                    "end": result.time_period.end.isoformat(),
                    "formatted": formatted_period,
                },
                "total": {
                    "amount": float(result.total.amount),
                    "currency": result.total.unit,
                },
                "estimated": result.estimated,
                "groups": [],
            }

            # Add group data
            for group in result.groups:
                group_data = {"keys": group.keys, "metrics": {}}
                for metric_name, cost_amount in group.metrics.items():
                    group_data["metrics"][metric_name] = {
                        "amount": float(cost_amount.amount),
                        "currency": cost_amount.unit,
                    }
                result_data["groups"].append(group_data)

            export_data["results"].append(result_data)

        # Write JSON file
        with open(output_path, "w", encoding="utf-8") as jsonfile:
            json.dump(export_data, jsonfile, indent=2, ensure_ascii=False)

        return output_path


class ExportManager:
    """Main export manager that coordinates different exporters."""

    def __init__(self, date_formatting_config: Optional[DateFormattingConfig] = None):
        """Initialize export manager."""
        self.date_formatting_config = date_formatting_config
        self.exporters = {
            "csv": CSVExporter(date_formatting_config=date_formatting_config),
            "json": JSONExporter(date_formatting_config=date_formatting_config),
        }

    def export_data(
        self,
        cost_data: CostData,
        query_params: QueryParameters,
        format_type: str,
        output_path: str,
    ) -> str:
        """
        Export cost data to specified format.

        Args:
            cost_data: Cost data to export
            query_params: Query parameters for context
            format_type: Export format ('csv', 'json')
            output_path: Output file path

        Returns:
            str: Path to exported file

        Raises:
            ValueError: If format is not supported
        """
        format_type = format_type.lower()

        if format_type not in self.exporters:
            available_formats = list(self.exporters.keys())
            raise ValueError(
                f"Unsupported export format '{format_type}'. Available formats: {available_formats}"
            )

        exporter = self.exporters[format_type]
        return exporter.export(cost_data, query_params, output_path)

    def get_available_formats(self) -> List[str]:
        """Get list of available export formats."""
        return list(self.exporters.keys())
