"""Trend analysis and forecasting capabilities for cost data."""

import statistics
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from decimal import Decimal

from .models import (
    CostData, CostResult, CostAmount, TimePeriod, TrendData, ForecastData,
    TrendAnalysisType, QueryParameters
)
from .date_utils import DateRangeCalculator


class TrendAnalyzer:
    """Analyzes cost trends and performs period-over-period comparisons."""
    
    def __init__(self):
        self.date_calculator = DateRangeCalculator()
    
    def analyze_trend(
        self, 
        current_data: CostData, 
        comparison_data: CostData,
        analysis_type: TrendAnalysisType
    ) -> TrendData:
        """
        Analyze trend between current and comparison periods.
        
        Args:
            current_data: Cost data for current period
            comparison_data: Cost data for comparison period
            analysis_type: Type of trend analysis
            
        Returns:
            TrendData with comparison results
        """
        current_total = current_data.total_cost
        comparison_total = comparison_data.total_cost
        
        # Calculate change
        change_amount = CostAmount(
            amount=current_total.amount - comparison_total.amount,
            unit=current_total.unit
        )
        
        # Calculate percentage change
        if comparison_total.amount != 0:
            change_percentage = float((change_amount.amount / comparison_total.amount) * 100)
        else:
            change_percentage = 0.0 if change_amount.amount == 0 else float('inf')
        
        # Determine trend direction
        if abs(change_percentage) < 5:  # Less than 5% change is considered stable
            trend_direction = "stable"
        elif change_percentage > 0:
            trend_direction = "up"
        else:
            trend_direction = "down"
        
        return TrendData(
            current_period=current_total,
            comparison_period=comparison_total,
            change_amount=change_amount,
            change_percentage=change_percentage,
            trend_direction=trend_direction
        )
    
    def analyze_service_trends(
        self, 
        current_data: CostData, 
        comparison_data: CostData
    ) -> Dict[str, TrendData]:
        """
        Analyze trends for individual services.
        
        Args:
            current_data: Current period cost data with service breakdown
            comparison_data: Comparison period cost data with service breakdown
            
        Returns:
            Dictionary mapping service names to their trend data
        """
        service_trends = {}
        
        # Extract service costs from current period
        current_services = self._extract_service_costs(current_data)
        comparison_services = self._extract_service_costs(comparison_data)
        
        # Analyze trends for each service
        all_services = set(current_services.keys()) | set(comparison_services.keys())
        
        for service in all_services:
            current_cost = current_services.get(service, CostAmount(amount=Decimal('0')))
            comparison_cost = comparison_services.get(service, CostAmount(amount=Decimal('0')))
            
            # Create mock CostData objects for trend analysis
            current_mock = CostData(
                results=[],
                time_period=current_data.time_period,
                total_cost=current_cost
            )
            comparison_mock = CostData(
                results=[],
                time_period=comparison_data.time_period,
                total_cost=comparison_cost
            )
            
            service_trends[service] = self.analyze_trend(
                current_mock, comparison_mock, TrendAnalysisType.PERIOD_OVER_PERIOD
            )
        
        return service_trends
    
    def _extract_service_costs(self, cost_data: CostData) -> Dict[str, CostAmount]:
        """Extract service costs from cost data with service grouping."""
        service_costs = {}
        
        for result in cost_data.results:
            for group in result.groups:
                if group.keys:  # Assuming first key is service name
                    service_name = group.keys[0]
                    # Sum up all metrics for this service
                    total_amount = Decimal('0')
                    unit = "USD"
                    
                    for metric_name, cost_amount in group.metrics.items():
                        total_amount += cost_amount.amount
                        unit = cost_amount.unit
                    
                    if service_name in service_costs:
                        service_costs[service_name].amount += total_amount
                    else:
                        service_costs[service_name] = CostAmount(amount=total_amount, unit=unit)
        
        return service_costs
    
    def get_top_cost_changes(
        self, 
        service_trends: Dict[str, TrendData], 
        limit: int = 5
    ) -> List[Tuple[str, TrendData]]:
        """
        Get services with the highest cost changes.
        
        Args:
            service_trends: Dictionary of service trends
            limit: Maximum number of services to return
            
        Returns:
            List of (service_name, trend_data) tuples sorted by absolute change amount
        """
        # Sort by absolute change amount
        sorted_trends = sorted(
            service_trends.items(),
            key=lambda x: abs(x[1].change_amount.amount),
            reverse=True
        )
        
        return sorted_trends[:limit]
    
    def get_fastest_growing_services(
        self, 
        service_trends: Dict[str, TrendData], 
        limit: int = 5
    ) -> List[Tuple[str, TrendData]]:
        """
        Get services with the highest percentage growth.
        
        Args:
            service_trends: Dictionary of service trends
            limit: Maximum number of services to return
            
        Returns:
            List of (service_name, trend_data) tuples sorted by percentage change
        """
        # Filter for positive growth and sort by percentage
        growing_services = [
            (service, trend) for service, trend in service_trends.items()
            if trend.change_percentage > 0
        ]
        
        sorted_trends = sorted(
            growing_services,
            key=lambda x: x[1].change_percentage,
            reverse=True
        )
        
        return sorted_trends[:limit]


class CostForecaster:
    """Forecasts future costs based on historical data."""
    
    def __init__(self):
        self.min_data_points = 3  # Minimum historical data points for forecasting
    
    def forecast_costs(
        self, 
        historical_data: List[CostData], 
        forecast_months: int = 3
    ) -> List[ForecastData]:
        """
        Forecast future costs based on historical data.
        
        Args:
            historical_data: List of historical cost data (chronologically ordered)
            forecast_months: Number of months to forecast
            
        Returns:
            List of ForecastData for each forecasted month
        """
        if len(historical_data) < self.min_data_points:
            raise ValueError(f"Need at least {self.min_data_points} historical data points for forecasting")
        
        # Extract time series data
        time_series = self._extract_time_series(historical_data)
        
        if len(time_series) < self.min_data_points:
            raise ValueError("Insufficient time series data for forecasting")
        
        # Generate forecasts
        forecasts = []
        
        for month_offset in range(1, forecast_months + 1):
            forecast_data = self._forecast_single_period(time_series, month_offset)
            forecasts.append(forecast_data)
        
        return forecasts
    
    def _extract_time_series(self, historical_data: List[CostData]) -> List[Tuple[datetime, float]]:
        """Extract time series data from historical cost data."""
        time_series = []
        
        for cost_data in historical_data:
            # Use the start date of the period and total cost
            timestamp = cost_data.time_period.start
            amount = float(cost_data.total_cost.amount)
            time_series.append((timestamp, amount))
        
        # Sort by timestamp
        time_series.sort(key=lambda x: x[0])
        return time_series
    
    def _forecast_single_period(
        self, 
        time_series: List[Tuple[datetime, float]], 
        periods_ahead: int
    ) -> ForecastData:
        """
        Forecast a single period using simple trend analysis.
        
        Args:
            time_series: Historical time series data
            periods_ahead: Number of periods ahead to forecast
            
        Returns:
            ForecastData for the forecasted period
        """
        # Extract values and calculate trend
        values = [point[1] for point in time_series]
        
        # Simple linear trend calculation
        n = len(values)
        x_values = list(range(n))
        
        # Calculate linear regression coefficients
        x_mean = statistics.mean(x_values)
        y_mean = statistics.mean(values)
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        
        if denominator == 0:
            # No trend, use average
            slope = 0
            intercept = y_mean
        else:
            slope = numerator / denominator
            intercept = y_mean - slope * x_mean
        
        # Forecast value
        forecast_x = n + periods_ahead - 1
        forecasted_amount = slope * forecast_x + intercept
        
        # Ensure non-negative forecast
        forecasted_amount = max(0, forecasted_amount)
        
        # Calculate confidence interval based on historical variance
        residuals = [values[i] - (slope * i + intercept) for i in range(n)]
        variance = statistics.variance(residuals) if len(residuals) > 1 else 0
        std_dev = variance ** 0.5
        
        # 95% confidence interval (approximately 2 standard deviations)
        # Ensure minimum confidence margin for meaningful intervals
        confidence_margin = max(2 * std_dev, forecasted_amount * 0.1)  # At least 10% margin
        lower_bound = max(0, forecasted_amount - confidence_margin)
        upper_bound = forecasted_amount + confidence_margin
        
        # Calculate forecast period
        last_date = time_series[-1][0]
        forecast_start = last_date + timedelta(days=30 * periods_ahead)
        forecast_end = forecast_start + timedelta(days=30)
        
        # Calculate prediction accuracy based on recent trend consistency
        accuracy = self._calculate_prediction_accuracy(time_series, slope, intercept)
        
        return ForecastData(
            forecasted_amount=CostAmount(amount=Decimal(str(round(forecasted_amount, 2)))),
            confidence_interval_lower=CostAmount(amount=Decimal(str(round(lower_bound, 2)))),
            confidence_interval_upper=CostAmount(amount=Decimal(str(round(upper_bound, 2)))),
            forecast_period=TimePeriod(start=forecast_start, end=forecast_end),
            prediction_accuracy=accuracy
        )
    
    def _calculate_prediction_accuracy(
        self, 
        time_series: List[Tuple[datetime, float]], 
        slope: float, 
        intercept: float
    ) -> float:
        """
        Calculate prediction accuracy based on how well the trend fits historical data.
        
        Args:
            time_series: Historical time series data
            slope: Linear trend slope
            intercept: Linear trend intercept
            
        Returns:
            Accuracy score between 0 and 1
        """
        if len(time_series) < 2:
            return 0.5  # Default accuracy for insufficient data
        
        values = [point[1] for point in time_series]
        n = len(values)
        
        # Calculate R-squared
        y_mean = statistics.mean(values)
        ss_tot = sum((y - y_mean) ** 2 for y in values)
        
        if ss_tot == 0:
            return 1.0  # Perfect fit if no variance
        
        ss_res = sum((values[i] - (slope * i + intercept)) ** 2 for i in range(n))
        r_squared = 1 - (ss_res / ss_tot)
        
        # Convert R-squared to accuracy (clamp between 0 and 1)
        accuracy = max(0, min(1, r_squared))
        return accuracy
    
    def forecast_by_service(
        self, 
        historical_data: List[CostData], 
        forecast_months: int = 3
    ) -> Dict[str, List[ForecastData]]:
        """
        Forecast costs by individual service.
        
        Args:
            historical_data: List of historical cost data with service breakdown
            forecast_months: Number of months to forecast
            
        Returns:
            Dictionary mapping service names to their forecast data
        """
        # Extract service time series
        service_time_series = self._extract_service_time_series(historical_data)
        
        service_forecasts = {}
        
        for service, time_series in service_time_series.items():
            if len(time_series) >= self.min_data_points:
                try:
                    forecasts = []
                    for month_offset in range(1, forecast_months + 1):
                        forecast = self._forecast_single_period(time_series, month_offset)
                        forecasts.append(forecast)
                    service_forecasts[service] = forecasts
                except Exception:
                    # Skip services with forecasting errors
                    continue
        
        return service_forecasts
    
    def _extract_service_time_series(
        self, 
        historical_data: List[CostData]
    ) -> Dict[str, List[Tuple[datetime, float]]]:
        """Extract time series data for each service."""
        service_series = {}
        
        for cost_data in historical_data:
            timestamp = cost_data.time_period.start
            
            # Extract service costs
            service_costs = {}
            for result in cost_data.results:
                for group in result.groups:
                    if group.keys:  # Assuming first key is service name
                        service_name = group.keys[0]
                        total_amount = sum(
                            float(cost_amount.amount) 
                            for cost_amount in group.metrics.values()
                        )
                        
                        if service_name in service_costs:
                            service_costs[service_name] += total_amount
                        else:
                            service_costs[service_name] = total_amount
            
            # Add to time series
            for service, amount in service_costs.items():
                if service not in service_series:
                    service_series[service] = []
                service_series[service].append((timestamp, amount))
        
        # Sort each service's time series
        for service in service_series:
            service_series[service].sort(key=lambda x: x[0])
        
        return service_series


def calculate_cost_efficiency_metrics(cost_data: CostData) -> Dict[str, Any]:
    """
    Calculate cost efficiency metrics from cost data.
    
    Args:
        cost_data: Cost data to analyze
        
    Returns:
        Dictionary with efficiency metrics
    """
    metrics = {}
    
    # Calculate cost per day
    period_days = (cost_data.time_period.end - cost_data.time_period.start).days
    if period_days > 0:
        metrics['cost_per_day'] = float(cost_data.total_cost.amount) / period_days
    
    # Calculate service concentration (how concentrated costs are across services)
    if cost_data.results and cost_data.results[0].groups:
        service_costs = []
        for result in cost_data.results:
            for group in result.groups:
                total_amount = sum(
                    float(cost_amount.amount) 
                    for cost_amount in group.metrics.values()
                )
                service_costs.append(total_amount)
        
        if service_costs:
            total_cost = sum(service_costs)
            if total_cost > 0:
                # Calculate Gini coefficient for cost concentration
                sorted_costs = sorted(service_costs)
                n = len(sorted_costs)
                cumsum = sum((i + 1) * cost for i, cost in enumerate(sorted_costs))
                gini = (2 * cumsum) / (n * total_cost) - (n + 1) / n
                metrics['cost_concentration'] = gini
                
                # Calculate top service percentage
                max_service_cost = max(service_costs)
                metrics['top_service_percentage'] = (max_service_cost / total_cost) * 100
    
    return metrics