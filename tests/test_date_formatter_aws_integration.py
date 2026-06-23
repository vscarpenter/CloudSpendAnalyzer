"""Integration tests for DateFormatter with realistic AWS Cost Explorer API response data."""

import pytest
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from src.aws_cost_cli.date_formatter import DateFormatter
from src.aws_cost_cli.models import TimePeriod, CostData, CostResult, CostAmount, QueryParameters, TimePeriodGranularity
from src.aws_cost_cli.response_formatter import SimpleResponseFormatter, RichResponseFormatter


class TestDateFormatterAWSIntegration:
    """Test DateFormatter with realistic AWS Cost Explorer response patterns."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = DateFormatter()
        self.simple_formatter = SimpleResponseFormatter()
        self.rich_formatter = RichResponseFormatter()
    
    def test_aws_get_cost_and_usage_daily_response(self):
        """Test with realistic AWS GetCostAndUsage daily response."""
        # Simulate AWS response for daily granularity over a week
        daily_periods = []
        base_date = datetime(2024, 1, 15, tzinfo=timezone.utc)
        
        for i in range(7):
            start = base_date + timedelta(days=i)
            end = start + timedelta(days=1)
            daily_periods.append(TimePeriod(start=start, end=end))
        
        # Test formatting each daily period
        results = []
        for period in daily_periods:
            result = self.formatter.format_time_period(period)
            results.append(result)
        
        # Verify daily formatting
        expected_results = [
            "January 15, 2024",
            "January 16, 2024", 
            "January 17, 2024",
            "January 18, 2024",
            "January 19, 2024",
            "January 20, 2024",
            "January 21, 2024"
        ]
        
        assert results == expected_results
    
    def test_aws_get_cost_and_usage_monthly_response(self):
        """Test with realistic AWS GetCostAndUsage monthly response."""
        # Simulate AWS response for monthly granularity over 6 months
        monthly_periods = []
        
        months_data = [
            (datetime(2024, 1, 1), datetime(2024, 2, 1)),
            (datetime(2024, 2, 1), datetime(2024, 3, 1)),
            (datetime(2024, 3, 1), datetime(2024, 4, 1)),
            (datetime(2024, 4, 1), datetime(2024, 5, 1)),
            (datetime(2024, 5, 1), datetime(2024, 6, 1)),
            (datetime(2024, 6, 1), datetime(2024, 7, 1)),
        ]
        
        for start, end in months_data:
            start_utc = start.replace(tzinfo=timezone.utc)
            end_utc = end.replace(tzinfo=timezone.utc)
            monthly_periods.append(TimePeriod(start=start_utc, end=end_utc))
        
        # Test formatting each monthly period
        results = []
        for period in monthly_periods:
            result = self.formatter.format_time_period(period)
            results.append(result)
        
        # Verify monthly formatting
        expected_months = ["January", "February", "March", "April", "May", "June"]
        for i, expected_month in enumerate(expected_months):
            assert expected_month in results[i]
            assert "2024" in results[i]
    
    def test_aws_dimension_breakdown_response(self):
        """Test with AWS response that includes dimension breakdowns."""
        # Simulate AWS response with service dimension breakdown for a month
        time_period = TimePeriod(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 2, 1, tzinfo=timezone.utc)
        )
        
        # Create cost data with multiple services
        cost_results = [
            CostResult(
                time_period=time_period,
                total=CostAmount(Decimal("150.00"), "USD"),
                groups=[],
                estimated=False
            )
        ]
        
        cost_data = CostData(
            results=cost_results,
            time_period=time_period,
            total_cost=CostAmount(Decimal("150.00"), "USD"),
            currency="USD"
        )
        
        query_params = QueryParameters(
            service="EC2",
            time_period=time_period,
            granularity=TimePeriodGranularity.MONTHLY
        )
        
        # Test formatting in response
        response = self.simple_formatter.format_response(
            cost_data, "What did I spend on EC2 in January?", query_params
        )
        
        # Should contain properly formatted time period
        assert "January" in response and "2024" in response
    
    def test_aws_get_rightsizing_recommendation_response(self):
        """Test with AWS GetRightsizingRecommendation response patterns."""
        # Rightsizing recommendations typically use 14-day lookback periods
        lookback_period = TimePeriod(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 15, tzinfo=timezone.utc)
        )
        
        result = self.formatter.format_time_period(lookback_period)
        
        # Should format as custom range
        assert "January 1" in result and "14, 2024" in result
    
    def test_aws_get_cost_forecast_response(self):
        """Test with AWS GetCostForecast response patterns."""
        # Cost forecasts typically project 1-3 months into the future
        forecast_periods = [
            # Current month (partial)
            TimePeriod(
                start=datetime(2024, 1, 15, tzinfo=timezone.utc),
                end=datetime(2024, 2, 1, tzinfo=timezone.utc)
            ),
            # Next full month
            TimePeriod(
                start=datetime(2024, 2, 1, tzinfo=timezone.utc),
                end=datetime(2024, 3, 1, tzinfo=timezone.utc)
            ),
            # Month after that
            TimePeriod(
                start=datetime(2024, 3, 1, tzinfo=timezone.utc),
                end=datetime(2024, 4, 1, tzinfo=timezone.utc)
            ),
        ]
        
        results = []
        for period in forecast_periods:
            result = self.formatter.format_time_period(period)
            results.append(result)
        
        # First should be partial month
        assert "January 15" in results[0] and "31, 2024" in results[0]
        # Others should be full months
        assert "February" in results[1] and "2024" in results[1]
        assert "March" in results[2] and "2024" in results[2]
    
    def test_aws_billing_period_patterns(self):
        """Test with AWS billing period patterns."""
        # AWS billing periods can have various patterns
        billing_periods = [
            # Standard monthly billing (1st to 1st)
            TimePeriod(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 2, 1, tzinfo=timezone.utc)
            ),
            # Mid-month billing cycle (15th to 15th)
            TimePeriod(
                start=datetime(2024, 1, 15, tzinfo=timezone.utc),
                end=datetime(2024, 2, 15, tzinfo=timezone.utc)
            ),
            # Quarterly billing
            TimePeriod(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 4, 1, tzinfo=timezone.utc)
            ),
            # Annual billing
            TimePeriod(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2025, 1, 1, tzinfo=timezone.utc)
            ),
        ]
        
        results = []
        for period in billing_periods:
            result = self.formatter.format_time_period(period)
            results.append(result)
        
        # Verify appropriate formatting for each billing pattern
        assert "January" in results[0] and "2024" in results[0]  # Monthly
        assert "January 15" in results[1] and "February 14" in results[1]  # Mid-month
        assert ("Q1 2024" in results[2] or 
                ("January" in results[2] and "March" in results[2]))  # Quarterly
        assert ("2024" == results[3] or 
                ("January" in results[3] and "December" in results[3]))  # Annual
    
    def test_aws_cost_anomaly_detection_periods(self):
        """Test with AWS Cost Anomaly Detection time periods."""
        # Anomaly detection typically looks at recent periods
        anomaly_periods = [
            # Yesterday
            TimePeriod(
                start=datetime(2024, 1, 14, tzinfo=timezone.utc),
                end=datetime(2024, 1, 15, tzinfo=timezone.utc)
            ),
            # Last 7 days
            TimePeriod(
                start=datetime(2024, 1, 8, tzinfo=timezone.utc),
                end=datetime(2024, 1, 15, tzinfo=timezone.utc)
            ),
            # Last 30 days
            TimePeriod(
                start=datetime(2023, 12, 16, tzinfo=timezone.utc),
                end=datetime(2024, 1, 15, tzinfo=timezone.utc)
            ),
        ]
        
        results = []
        for period in anomaly_periods:
            result = self.formatter.format_time_period(period)
            results.append(result)
        
        # Verify formatting
        assert results[0] == "January 14, 2024"  # Single day
        assert "January 8" in results[1] and "14, 2024" in results[1]  # 7-day range
        assert "December 16, 2023" in results[2] and "January 14, 2024" in results[2]  # 30-day range
    
    def test_aws_reserved_instance_utilization_periods(self):
        """Test with AWS Reserved Instance utilization report periods."""
        # RI utilization reports typically use monthly periods
        ri_periods = []
        
        # Last 12 months of RI utilization
        for month_offset in range(12):
            if month_offset == 0:
                # Current month (partial)
                start = datetime(2024, 1, 1, tzinfo=timezone.utc)
                end = datetime(2024, 1, 15, tzinfo=timezone.utc)
            else:
                # Previous full months
                year = 2024 if month_offset <= 12 else 2023
                month = 1 - month_offset
                if month <= 0:
                    month += 12
                    year -= 1
                
                start = datetime(year, month, 1, tzinfo=timezone.utc)
                if month == 12:
                    end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
                else:
                    end = datetime(year, month + 1, 1, tzinfo=timezone.utc)
            
            ri_periods.append(TimePeriod(start=start, end=end))
        
        # Test formatting first few periods
        results = []
        for period in ri_periods[:3]:
            result = self.formatter.format_time_period(period)
            results.append(result)
        
        # First should be partial current month
        assert "January 1" in results[0] and "14, 2024" in results[0]
        # Others should be full months
        assert all("2024" in result or "2023" in result for result in results[1:])
    
    def test_aws_savings_plans_utilization_periods(self):
        """Test with AWS Savings Plans utilization periods."""
        # Savings Plans typically have 1-year or 3-year terms
        sp_periods = [
            # 1-year Savings Plan
            TimePeriod(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2025, 1, 1, tzinfo=timezone.utc)
            ),
            # 3-year Savings Plan
            TimePeriod(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2027, 1, 1, tzinfo=timezone.utc)
            ),
            # Monthly utilization within Savings Plan
            TimePeriod(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 2, 1, tzinfo=timezone.utc)
            ),
        ]
        
        results = []
        for period in sp_periods:
            result = self.formatter.format_time_period(period)
            results.append(result)
        
        # Verify long-term period formatting
        assert "2024" in results[0]  # 1-year plan
        assert "January 1, 2024" in results[1] and "December 31, 2026" in results[1]  # 3-year plan
        assert "January" in results[2] and "2024" in results[2]  # Monthly utilization
    
    def test_aws_cost_budget_periods(self):
        """Test with AWS Cost Budget time periods."""
        # Budgets can have various time periods
        budget_periods = [
            # Monthly budget
            TimePeriod(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 2, 1, tzinfo=timezone.utc)
            ),
            # Quarterly budget
            TimePeriod(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 4, 1, tzinfo=timezone.utc)
            ),
            # Annual budget
            TimePeriod(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2025, 1, 1, tzinfo=timezone.utc)
            ),
            # Custom budget period (e.g., project duration)
            TimePeriod(
                start=datetime(2024, 1, 15, tzinfo=timezone.utc),
                end=datetime(2024, 4, 30, tzinfo=timezone.utc)
            ),
        ]
        
        results = []
        for period in budget_periods:
            result = self.formatter.format_time_period(period)
            results.append(result)
        
        # Verify budget period formatting
        assert "January" in results[0] and "2024" in results[0]  # Monthly
        assert ("Q1 2024" in results[1] or 
                ("January" in results[1] and "March" in results[1]))  # Quarterly
        assert ("2024" == results[2] or 
                ("January" in results[2] and "December" in results[2]))  # Annual
        assert "January 15" in results[3] and "April 29" in results[3]  # Custom
    
    def test_integration_with_response_formatters(self):
        """Test integration with response formatters using AWS-like data."""
        # Create realistic cost data
        time_period = TimePeriod(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 4, 1, tzinfo=timezone.utc)
        )
        
        # Monthly breakdown within the quarter
        monthly_results = []
        months = [
            (datetime(2024, 1, 1), datetime(2024, 2, 1)),
            (datetime(2024, 2, 1), datetime(2024, 3, 1)),
            (datetime(2024, 3, 1), datetime(2024, 4, 1)),
        ]
        
        for start, end in months:
            start_utc = start.replace(tzinfo=timezone.utc)
            end_utc = end.replace(tzinfo=timezone.utc)
            monthly_results.append(
                CostResult(
                    time_period=TimePeriod(start=start_utc, end=end_utc),
                    total=CostAmount(Decimal("100.00"), "USD"),
                    groups=[],
                    estimated=False
                )
            )
        
        cost_data = CostData(
            results=monthly_results,
            time_period=time_period,
            total_cost=CostAmount(Decimal("300.00"), "USD"),
            currency="USD"
        )
        
        query_params = QueryParameters(
            service="EC2",
            time_period=time_period,
            granularity=TimePeriodGranularity.MONTHLY
        )
        
        # Test with SimpleResponseFormatter
        simple_response = self.simple_formatter.format_response(
            cost_data, "What did I spend on EC2 in Q1?", query_params
        )
        
        # Should contain formatted time periods
        assert "January" in simple_response or "Q1" in simple_response
        assert "2024" in simple_response
        
        # Test with RichResponseFormatter (if available)
        try:
            rich_response = self.rich_formatter.format_response(
                cost_data, "What did I spend on EC2 in Q1?", query_params
            )
            # Should not raise an exception
            assert isinstance(rich_response, str)
        except ImportError:
            # Rich not available, skip this part
            pass
    
    def test_aws_cross_account_billing_periods(self):
        """Test with AWS cross-account billing periods."""
        # Cross-account billing often involves consolidated periods
        consolidated_periods = [
            # Master account monthly view
            TimePeriod(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 2, 1, tzinfo=timezone.utc)
            ),
            # Linked account daily breakdown
            TimePeriod(
                start=datetime(2024, 1, 15, tzinfo=timezone.utc),
                end=datetime(2024, 1, 16, tzinfo=timezone.utc)
            ),
            # Organization-wide quarterly view
            TimePeriod(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 4, 1, tzinfo=timezone.utc)
            ),
        ]
        
        results = []
        for period in consolidated_periods:
            result = self.formatter.format_time_period(period)
            results.append(result)
        
        # Verify cross-account period formatting
        assert "January" in results[0] and "2024" in results[0]  # Master monthly
        assert results[1] == "January 15, 2024"  # Linked daily
        assert ("Q1 2024" in results[2] or 
                ("January" in results[2] and "March" in results[2]))  # Org quarterly
    
    def test_aws_marketplace_subscription_periods(self):
        """Test with AWS Marketplace subscription periods."""
        # Marketplace subscriptions can have various billing cycles
        marketplace_periods = [
            # Monthly subscription
            TimePeriod(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 2, 1, tzinfo=timezone.utc)
            ),
            # Annual subscription
            TimePeriod(
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2025, 1, 1, tzinfo=timezone.utc)
            ),
            # Usage-based billing (daily)
            TimePeriod(
                start=datetime(2024, 1, 15, tzinfo=timezone.utc),
                end=datetime(2024, 1, 16, tzinfo=timezone.utc)
            ),
        ]
        
        results = []
        for period in marketplace_periods:
            result = self.formatter.format_time_period(period)
            results.append(result)
        
        # Verify marketplace period formatting
        assert "January" in results[0] and "2024" in results[0]  # Monthly
        assert ("2024" == results[1] or 
                ("January" in results[1] and "December" in results[1]))  # Annual
        assert results[2] == "January 15, 2024"  # Daily usage


if __name__ == "__main__":
    pytest.main([__file__, "-v"])