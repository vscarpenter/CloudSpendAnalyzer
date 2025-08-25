"""Tests for advanced query features including trend analysis and forecasting."""

import pytest
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch

from src.aws_cost_cli.models import (
    QueryParameters, CostData, CostResult, CostAmount, TimePeriod, Group,
    TimePeriodGranularity, MetricType, DateRangeType, TrendAnalysisType,
    TrendData, ForecastData
)
from src.aws_cost_cli.date_utils import DateRangeCalculator, Quarter, parse_advanced_date_range
from src.aws_cost_cli.trend_analysis import TrendAnalyzer, CostForecaster, calculate_cost_efficiency_metrics
from src.aws_cost_cli.query_processor import QueryParser


class TestDateRangeCalculator:
    """Test advanced date range calculations."""
    
    def test_quarter_range_calculation(self):
        """Test quarter date range calculation."""
        calculator = DateRangeCalculator()
        
        # Test Q1 2025
        q1_range = calculator.get_quarter_range(2025, Quarter.Q1)
        assert q1_range.start == datetime(2025, 1, 1, tzinfo=timezone.utc)
        assert q1_range.end == datetime(2025, 4, 1, tzinfo=timezone.utc)
        
        # Test Q4 2025 (crosses year boundary)
        q4_range = calculator.get_quarter_range(2025, Quarter.Q4)
        assert q4_range.start == datetime(2025, 10, 1, tzinfo=timezone.utc)
        assert q4_range.end == datetime(2026, 1, 1, tzinfo=timezone.utc)
    
    def test_fiscal_year_range(self):
        """Test fiscal year range calculation."""
        # Test calendar year fiscal year (January start)
        calculator = DateRangeCalculator(fiscal_year_start_month=1)
        fy_range = calculator.get_fiscal_year_range(2025)
        assert fy_range.start == datetime(2025, 1, 1, tzinfo=timezone.utc)
        assert fy_range.end == datetime(2026, 1, 1, tzinfo=timezone.utc)
        
        # Test October fiscal year start
        calculator = DateRangeCalculator(fiscal_year_start_month=10)
        fy_range = calculator.get_fiscal_year_range(2025)
        assert fy_range.start == datetime(2024, 10, 1, tzinfo=timezone.utc)
        assert fy_range.end == datetime(2025, 10, 1, tzinfo=timezone.utc)
    
    def test_fiscal_quarter_range(self):
        """Test fiscal quarter range calculation."""
        # Test with October fiscal year start
        calculator = DateRangeCalculator(fiscal_year_start_month=10)
        
        # FY2025 Q1 should be Oct-Dec 2024
        fq1_range = calculator.get_fiscal_quarter_range(2025, Quarter.Q1)
        assert fq1_range.start == datetime(2024, 10, 1, tzinfo=timezone.utc)
        assert fq1_range.end == datetime(2025, 1, 1, tzinfo=timezone.utc)
    
    def test_current_quarter_detection(self):
        """Test current quarter detection."""
        calculator = DateRangeCalculator()
        
        # Mock current date to August 24, 2025
        with patch('src.aws_cost_cli.date_utils.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2025, 8, 24, tzinfo=timezone.utc)
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
            
            year, quarter = calculator.get_current_quarter()
            assert year == 2025
            assert quarter == Quarter.Q3
    
    def test_previous_period_calculation(self):
        """Test previous period calculation."""
        calculator = DateRangeCalculator()
        
        # Test same length previous period
        current_period = TimePeriod(
            start=datetime(2025, 7, 1, tzinfo=timezone.utc),
            end=datetime(2025, 8, 1, tzinfo=timezone.utc)
        )
        
        previous_period = calculator.get_previous_period(current_period, "same_length")
        # The period is 31 days long (July 1 to Aug 1), so previous period should be 31 days before
        expected_start = datetime(2025, 5, 31, tzinfo=timezone.utc)  # 31 days before July 1
        expected_end = datetime(2025, 7, 1, tzinfo=timezone.utc)     # July 1
        assert previous_period.start == expected_start
        assert previous_period.end == expected_end
        
        # Test year ago period
        year_ago_period = calculator.get_previous_period(current_period, "year_ago")
        assert year_ago_period.start == datetime(2024, 7, 1, tzinfo=timezone.utc)
        assert year_ago_period.end == datetime(2024, 8, 1, tzinfo=timezone.utc)
    
    def test_quarter_string_parsing(self):
        """Test quarter string parsing."""
        calculator = DateRangeCalculator()
        
        # Test various formats
        year, quarter = calculator.parse_quarter_string("Q1 2025")
        assert year == 2025
        assert quarter == Quarter.Q1
        
        year, quarter = calculator.parse_quarter_string("2025 Q3")
        assert year == 2025
        assert quarter == Quarter.Q3
        
        # Test with current year default
        with patch('src.aws_cost_cli.date_utils.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2025, 8, 24)
            year, quarter = calculator.parse_quarter_string("Q2")
            assert year == 2025
            assert quarter == Quarter.Q2


class TestAdvancedDateRangeParsing:
    """Test advanced date range parsing."""
    
    def test_quarter_parsing(self):
        """Test quarter string parsing."""
        # Test calendar quarter
        period = parse_advanced_date_range("Q1 2025")
        assert period.start == datetime(2025, 1, 1, tzinfo=timezone.utc)
        assert period.end == datetime(2025, 4, 1, tzinfo=timezone.utc)
        
        # Test fiscal quarter
        period = parse_advanced_date_range("FY Q2 2025", fiscal_year_start_month=10)
        # FY2025 Q2 with Oct start should be Jan-Mar 2025
        assert period.start == datetime(2025, 1, 1, tzinfo=timezone.utc)
        assert period.end == datetime(2025, 4, 1, tzinfo=timezone.utc)
    
    def test_fiscal_year_parsing(self):
        """Test fiscal year parsing."""
        period = parse_advanced_date_range("FY2025")
        assert period.start == datetime(2025, 1, 1, tzinfo=timezone.utc)
        assert period.end == datetime(2026, 1, 1, tzinfo=timezone.utc)
        
        # Test with different fiscal year start
        period = parse_advanced_date_range("FY2025", fiscal_year_start_month=7)
        assert period.start == datetime(2024, 7, 1, tzinfo=timezone.utc)
        assert period.end == datetime(2025, 7, 1, tzinfo=timezone.utc)
    
    def test_calendar_year_parsing(self):
        """Test calendar year parsing."""
        period = parse_advanced_date_range("2025")
        assert period.start == datetime(2025, 1, 1, tzinfo=timezone.utc)
        assert period.end == datetime(2026, 1, 1, tzinfo=timezone.utc)
    
    def test_invalid_date_range(self):
        """Test invalid date range handling."""
        with pytest.raises(ValueError):
            parse_advanced_date_range("invalid date")


class TestTrendAnalyzer:
    """Test trend analysis functionality."""
    
    def setup_method(self):
        """Set up test data."""
        self.analyzer = TrendAnalyzer()
        
        # Create sample cost data
        self.current_data = CostData(
            results=[],
            time_period=TimePeriod(
                start=datetime(2025, 7, 1, tzinfo=timezone.utc),
                end=datetime(2025, 8, 1, tzinfo=timezone.utc)
            ),
            total_cost=CostAmount(amount=Decimal('1000.00'))
        )
        
        self.comparison_data = CostData(
            results=[],
            time_period=TimePeriod(
                start=datetime(2025, 6, 1, tzinfo=timezone.utc),
                end=datetime(2025, 7, 1, tzinfo=timezone.utc)
            ),
            total_cost=CostAmount(amount=Decimal('800.00'))
        )
    
    def test_trend_analysis_increase(self):
        """Test trend analysis with cost increase."""
        trend = self.analyzer.analyze_trend(
            self.current_data, 
            self.comparison_data, 
            TrendAnalysisType.PERIOD_OVER_PERIOD
        )
        
        assert trend.current_period.amount == Decimal('1000.00')
        assert trend.comparison_period.amount == Decimal('800.00')
        assert trend.change_amount.amount == Decimal('200.00')
        assert trend.change_percentage == 25.0
        assert trend.trend_direction == "up"
    
    def test_trend_analysis_decrease(self):
        """Test trend analysis with cost decrease."""
        # Swap the data to test decrease
        trend = self.analyzer.analyze_trend(
            self.comparison_data,  # Lower cost as current
            self.current_data,     # Higher cost as comparison
            TrendAnalysisType.PERIOD_OVER_PERIOD
        )
        
        assert trend.change_amount.amount == Decimal('-200.00')
        assert trend.change_percentage == -20.0
        assert trend.trend_direction == "down"
    
    def test_trend_analysis_stable(self):
        """Test trend analysis with stable costs."""
        stable_data = CostData(
            results=[],
            time_period=self.current_data.time_period,
            total_cost=CostAmount(amount=Decimal('820.00'))  # ~2.5% increase
        )
        
        trend = self.analyzer.analyze_trend(
            stable_data,
            self.comparison_data,
            TrendAnalysisType.PERIOD_OVER_PERIOD
        )
        
        assert trend.trend_direction == "stable"  # Less than 5% change
    
    def test_service_trends_analysis(self):
        """Test service-level trend analysis."""
        # Create cost data with service groups
        current_with_services = CostData(
            results=[
                CostResult(
                    time_period=self.current_data.time_period,
                    total=CostAmount(amount=Decimal('1000.00')),
                    groups=[
                        Group(
                            keys=["Amazon Elastic Compute Cloud - Compute"],
                            metrics={"BlendedCost": CostAmount(amount=Decimal('600.00'))}
                        ),
                        Group(
                            keys=["Amazon Simple Storage Service"],
                            metrics={"BlendedCost": CostAmount(amount=Decimal('400.00'))}
                        )
                    ]
                )
            ],
            time_period=self.current_data.time_period,
            total_cost=CostAmount(amount=Decimal('1000.00'))
        )
        
        comparison_with_services = CostData(
            results=[
                CostResult(
                    time_period=self.comparison_data.time_period,
                    total=CostAmount(amount=Decimal('800.00')),
                    groups=[
                        Group(
                            keys=["Amazon Elastic Compute Cloud - Compute"],
                            metrics={"BlendedCost": CostAmount(amount=Decimal('500.00'))}
                        ),
                        Group(
                            keys=["Amazon Simple Storage Service"],
                            metrics={"BlendedCost": CostAmount(amount=Decimal('300.00'))}
                        )
                    ]
                )
            ],
            time_period=self.comparison_data.time_period,
            total_cost=CostAmount(amount=Decimal('800.00'))
        )
        
        service_trends = self.analyzer.analyze_service_trends(
            current_with_services, comparison_with_services
        )
        
        assert "Amazon Elastic Compute Cloud - Compute" in service_trends
        assert "Amazon Simple Storage Service" in service_trends
        
        ec2_trend = service_trends["Amazon Elastic Compute Cloud - Compute"]
        assert ec2_trend.change_amount.amount == Decimal('100.00')
        assert ec2_trend.change_percentage == 20.0
    
    def test_top_cost_changes(self):
        """Test identification of top cost changes."""
        service_trends = {
            "EC2": TrendData(
                current_period=CostAmount(amount=Decimal('600.00')),
                comparison_period=CostAmount(amount=Decimal('500.00')),
                change_amount=CostAmount(amount=Decimal('100.00')),
                change_percentage=20.0,
                trend_direction="up"
            ),
            "S3": TrendData(
                current_period=CostAmount(amount=Decimal('200.00')),
                comparison_period=CostAmount(amount=Decimal('250.00')),
                change_amount=CostAmount(amount=Decimal('-50.00')),
                change_percentage=-20.0,
                trend_direction="down"
            )
        }
        
        top_changes = self.analyzer.get_top_cost_changes(service_trends, limit=2)
        
        assert len(top_changes) == 2
        # Should be sorted by absolute change amount
        assert top_changes[0][0] == "EC2"  # $100 change
        assert top_changes[1][0] == "S3"   # $50 change (absolute)


class TestCostForecaster:
    """Test cost forecasting functionality."""
    
    def setup_method(self):
        """Set up test data."""
        self.forecaster = CostForecaster()
        
        # Create historical data with upward trend
        self.historical_data = []
        for i in range(6):  # 6 months of data
            month_start = datetime(2025, i + 1, 1, tzinfo=timezone.utc)
            month_end = datetime(2025, i + 2, 1, tzinfo=timezone.utc) if i < 5 else datetime(2025, 7, 1, tzinfo=timezone.utc)
            
            cost_data = CostData(
                results=[],
                time_period=TimePeriod(start=month_start, end=month_end),
                total_cost=CostAmount(amount=Decimal(str(1000 + i * 100)))  # Increasing trend
            )
            self.historical_data.append(cost_data)
    
    def test_forecast_generation(self):
        """Test basic forecast generation."""
        forecasts = self.forecaster.forecast_costs(self.historical_data, forecast_months=3)
        
        assert len(forecasts) == 3
        
        # Check that forecasts are reasonable (should continue upward trend)
        for i, forecast in enumerate(forecasts):
            assert forecast.forecasted_amount.amount > Decimal('1500')  # Should be higher than last historical
            assert forecast.confidence_interval_lower.amount >= 0
            assert forecast.confidence_interval_upper.amount > forecast.forecasted_amount.amount
            assert forecast.prediction_accuracy is not None
    
    def test_insufficient_data_error(self):
        """Test error handling with insufficient historical data."""
        with pytest.raises(ValueError, match="Need at least 3 historical data points"):
            self.forecaster.forecast_costs(self.historical_data[:2], forecast_months=1)
    
    def test_forecast_by_service(self):
        """Test service-level forecasting."""
        # Create historical data with service breakdown
        historical_with_services = []
        for i in range(4):
            month_start = datetime(2025, i + 1, 1, tzinfo=timezone.utc)
            month_end = datetime(2025, i + 2, 1, tzinfo=timezone.utc)
            
            cost_data = CostData(
                results=[
                    CostResult(
                        time_period=TimePeriod(start=month_start, end=month_end),
                        total=CostAmount(amount=Decimal(str(1000 + i * 100))),
                        groups=[
                            Group(
                                keys=["EC2"],
                                metrics={"BlendedCost": CostAmount(amount=Decimal(str(600 + i * 50)))}
                            ),
                            Group(
                                keys=["S3"],
                                metrics={"BlendedCost": CostAmount(amount=Decimal(str(400 + i * 50)))}
                            )
                        ]
                    )
                ],
                time_period=TimePeriod(start=month_start, end=month_end),
                total_cost=CostAmount(amount=Decimal(str(1000 + i * 100)))
            )
            historical_with_services.append(cost_data)
        
        service_forecasts = self.forecaster.forecast_by_service(historical_with_services, forecast_months=2)
        
        assert "EC2" in service_forecasts
        assert "S3" in service_forecasts
        assert len(service_forecasts["EC2"]) == 2
        assert len(service_forecasts["S3"]) == 2


class TestCostEfficiencyMetrics:
    """Test cost efficiency metrics calculation."""
    
    def test_cost_per_day_calculation(self):
        """Test cost per day calculation."""
        cost_data = CostData(
            results=[],
            time_period=TimePeriod(
                start=datetime(2025, 7, 1, tzinfo=timezone.utc),
                end=datetime(2025, 8, 1, tzinfo=timezone.utc)  # 31 days
            ),
            total_cost=CostAmount(amount=Decimal('3100.00'))
        )
        
        metrics = calculate_cost_efficiency_metrics(cost_data)
        
        assert 'cost_per_day' in metrics
        assert metrics['cost_per_day'] == 100.0  # $3100 / 31 days
    
    def test_cost_concentration_metrics(self):
        """Test cost concentration metrics."""
        cost_data = CostData(
            results=[
                CostResult(
                    time_period=TimePeriod(
                        start=datetime(2025, 7, 1, tzinfo=timezone.utc),
                        end=datetime(2025, 8, 1, tzinfo=timezone.utc)
                    ),
                    total=CostAmount(amount=Decimal('1000.00')),
                    groups=[
                        Group(
                            keys=["EC2"],
                            metrics={"BlendedCost": CostAmount(amount=Decimal('800.00'))}
                        ),
                        Group(
                            keys=["S3"],
                            metrics={"BlendedCost": CostAmount(amount=Decimal('200.00'))}
                        )
                    ]
                )
            ],
            time_period=TimePeriod(
                start=datetime(2025, 7, 1, tzinfo=timezone.utc),
                end=datetime(2025, 8, 1, tzinfo=timezone.utc)
            ),
            total_cost=CostAmount(amount=Decimal('1000.00'))
        )
        
        metrics = calculate_cost_efficiency_metrics(cost_data)
        
        assert 'cost_concentration' in metrics
        assert 'top_service_percentage' in metrics
        assert metrics['top_service_percentage'] == 80.0  # EC2 is 80% of total


class TestAdvancedQueryProcessing:
    """Test advanced query processing with LLM integration."""
    
    def setup_method(self):
        """Set up test environment."""
        self.llm_config = {
            'provider': 'openai',
            'api_key': 'test-key',
            'model': 'gpt-3.5-turbo'
        }
    
    @patch('src.aws_cost_cli.query_processor.OpenAIProvider')
    def test_trend_analysis_query_parsing(self, mock_provider_class):
        """Test parsing of trend analysis queries."""
        mock_provider = Mock()
        mock_provider.is_available.return_value = True
        mock_provider.parse_query.return_value = {
            'service': 'Amazon Elastic Compute Cloud - Compute',
            'start_date': '2025-07-01',
            'end_date': '2025-08-01',
            'granularity': 'MONTHLY',
            'metrics': ['BlendedCost'],
            'group_by': None,
            'trend_analysis': 'MONTH_OVER_MONTH',
            'include_forecast': False,
            'forecast_months': 3
        }
        mock_provider_class.return_value = mock_provider
        
        parser = QueryParser(self.llm_config)
        params = parser.parse_query("EC2 costs this month compared to last month")
        
        assert params.service == 'Amazon Elastic Compute Cloud - Compute'
        assert params.trend_analysis == TrendAnalysisType.MONTH_OVER_MONTH
        assert not params.include_forecast
    
    @patch('src.aws_cost_cli.query_processor.OpenAIProvider')
    def test_forecast_query_parsing(self, mock_provider_class):
        """Test parsing of forecast queries."""
        mock_provider = Mock()
        mock_provider.is_available.return_value = True
        mock_provider.parse_query.return_value = {
            'service': None,
            'start_date': '2025-07-01',
            'end_date': '2025-08-01',
            'granularity': 'MONTHLY',
            'metrics': ['BlendedCost'],
            'group_by': None,
            'trend_analysis': None,
            'include_forecast': True,
            'forecast_months': 6
        }
        mock_provider_class.return_value = mock_provider
        
        parser = QueryParser(self.llm_config)
        params = parser.parse_query("Forecast my AWS costs for the next 6 months")
        
        assert params.include_forecast
        assert params.forecast_months == 6
    
    @patch('src.aws_cost_cli.query_processor.OpenAIProvider')
    def test_quarter_query_parsing(self, mock_provider_class):
        """Test parsing of quarter-based queries."""
        mock_provider = Mock()
        mock_provider.is_available.return_value = True
        mock_provider.parse_query.return_value = {
            'service': None,
            'start_date': '2025-07-01',
            'end_date': '2025-10-01',
            'granularity': 'MONTHLY',
            'metrics': ['BlendedCost'],
            'group_by': ['SERVICE'],
            'date_range_type': 'QUARTER',
            'trend_analysis': None,
            'include_forecast': False,
            'forecast_months': 3
        }
        mock_provider_class.return_value = mock_provider
        
        parser = QueryParser(self.llm_config)
        params = parser.parse_query("Show me costs by service for Q3 2025")
        
        assert params.date_range_type == DateRangeType.QUARTER
        assert params.group_by == ['SERVICE']


class TestAdvancedResponseFormatting:
    """Test response formatting for advanced features."""
    
    def setup_method(self):
        """Set up test data."""
        self.cost_data_with_trend = CostData(
            results=[],
            time_period=TimePeriod(
                start=datetime(2025, 7, 1, tzinfo=timezone.utc),
                end=datetime(2025, 8, 1, tzinfo=timezone.utc)
            ),
            total_cost=CostAmount(amount=Decimal('1000.00')),
            trend_data=TrendData(
                current_period=CostAmount(amount=Decimal('1000.00')),
                comparison_period=CostAmount(amount=Decimal('800.00')),
                change_amount=CostAmount(amount=Decimal('200.00')),
                change_percentage=25.0,
                trend_direction="up"
            )
        )
        
        self.cost_data_with_forecast = CostData(
            results=[],
            time_period=TimePeriod(
                start=datetime(2025, 7, 1, tzinfo=timezone.utc),
                end=datetime(2025, 8, 1, tzinfo=timezone.utc)
            ),
            total_cost=CostAmount(amount=Decimal('1000.00')),
            forecast_data=[
                ForecastData(
                    forecasted_amount=CostAmount(amount=Decimal('1100.00')),
                    confidence_interval_lower=CostAmount(amount=Decimal('1000.00')),
                    confidence_interval_upper=CostAmount(amount=Decimal('1200.00')),
                    forecast_period=TimePeriod(
                        start=datetime(2025, 8, 1, tzinfo=timezone.utc),
                        end=datetime(2025, 9, 1, tzinfo=timezone.utc)
                    ),
                    prediction_accuracy=0.85
                )
            ]
        )
    
    def test_simple_formatter_with_trend(self):
        """Test simple formatter with trend data."""
        from src.aws_cost_cli.response_formatter import SimpleResponseFormatter
        
        formatter = SimpleResponseFormatter()
        query_params = QueryParameters(trend_analysis=TrendAnalysisType.MONTH_OVER_MONTH)
        
        response = formatter.format_response(
            self.cost_data_with_trend, 
            "test query", 
            query_params
        )
        
        assert "Trend Analysis:" in response
        assert "$800.00" in response  # Previous period cost
        assert "+25.0%" in response   # Change percentage
        assert "Up" in response       # Trend direction
    
    def test_simple_formatter_with_forecast(self):
        """Test simple formatter with forecast data."""
        from src.aws_cost_cli.response_formatter import SimpleResponseFormatter
        
        formatter = SimpleResponseFormatter()
        query_params = QueryParameters(include_forecast=True)
        
        response = formatter.format_response(
            self.cost_data_with_forecast,
            "test query",
            query_params
        )
        
        assert "Cost Forecast:" in response
        assert "$1100.00" in response  # Forecasted amount
        assert "confidence: 85%" in response  # Prediction accuracy
    
    def test_rich_formatter_with_advanced_features(self):
        """Test Rich formatter with advanced features."""
        from src.aws_cost_cli.response_formatter import RichResponseFormatter
        
        formatter = RichResponseFormatter()
        query_params = QueryParameters(
            trend_analysis=TrendAnalysisType.MONTH_OVER_MONTH,
            include_forecast=True
        )
        
        # Create cost data with both trend and forecast
        cost_data = CostData(
            results=[],
            time_period=TimePeriod(
                start=datetime(2025, 7, 1, tzinfo=timezone.utc),
                end=datetime(2025, 8, 1, tzinfo=timezone.utc)
            ),
            total_cost=CostAmount(amount=Decimal('1000.00')),
            trend_data=self.cost_data_with_trend.trend_data,
            forecast_data=self.cost_data_with_forecast.forecast_data
        )
        
        response = formatter.format_response(cost_data, "test query", query_params)
        
        # Should fall back to simple formatter if Rich is not available
        assert isinstance(response, str)
        assert len(response) > 0


if __name__ == '__main__':
    pytest.main([__file__])