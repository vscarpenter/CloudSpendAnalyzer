"""Tests for cost optimization functionality."""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock

from src.aws_cost_cli.cost_optimizer import (
    CostOptimizer,
    OptimizationReport,
    OptimizationRecommendation,
    CostAnomaly,
    BudgetVariance,
    OptimizationType,
    SeverityLevel,
)
from src.aws_cost_cli.models import CostAmount, TimePeriod


class TestCostOptimizer:
    """Test cases for CostOptimizer class."""

    @pytest.fixture
    def optimizer(self):
        """Create a CostOptimizer instance for testing."""
        with patch("boto3.Session"):
            return CostOptimizer(profile="test-profile")

    @pytest.fixture
    def analysis_period(self):
        """Create a test analysis period."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        return TimePeriod(start=start_date, end=end_date)

    def test_initialization(self):
        """Test CostOptimizer initialization."""
        with patch("boto3.Session") as mock_session:
            mock_session.return_value.client.return_value = Mock()

            optimizer = CostOptimizer(profile="test-profile", region="us-west-2")

            assert optimizer.profile == "test-profile"
            assert optimizer.region == "us-west-2"
            mock_session.assert_called_once_with(profile_name="test-profile")

    def test_generate_optimization_report(self, optimizer, analysis_period):
        """Test generating a complete optimization report."""
        # Mock all the analysis methods
        optimizer._analyze_unused_resources = Mock(
            return_value=[
                OptimizationRecommendation(
                    type=OptimizationType.UNUSED_RESOURCES,
                    severity=SeverityLevel.MEDIUM,
                    title="Test unused resource",
                    description="Test description",
                    potential_savings=CostAmount(amount=Decimal("50.0")),
                    confidence_level=0.9,
                )
            ]
        )

        optimizer._analyze_rightsizing_opportunities = Mock(return_value=[])
        optimizer._get_reserved_instance_recommendations = Mock(return_value=[])
        optimizer._get_savings_plan_recommendations = Mock(return_value=[])
        optimizer._detect_cost_anomalies = Mock(return_value=[])
        optimizer._analyze_budget_variances = Mock(return_value=[])

        report = optimizer.generate_optimization_report(analysis_period)

        assert isinstance(report, OptimizationReport)
        assert len(report.recommendations) == 1
        assert report.total_potential_savings.amount == Decimal("50.0")
        assert report.analysis_period == analysis_period

        # Verify all analysis methods were called
        optimizer._analyze_unused_resources.assert_called_once_with(analysis_period)
        optimizer._analyze_rightsizing_opportunities.assert_called_once_with(
            analysis_period
        )
        optimizer._get_reserved_instance_recommendations.assert_called_once()
        optimizer._get_savings_plan_recommendations.assert_called_once()
        optimizer._detect_cost_anomalies.assert_called_once_with(analysis_period)
        optimizer._analyze_budget_variances.assert_called_once_with(analysis_period)

    def test_analyze_unused_resources(self, optimizer, analysis_period):
        """Test unused resources analysis."""
        # Mock EC2 client responses
        optimizer.ec2_client.describe_volumes.return_value = {
            "Volumes": [
                {
                    "VolumeId": "vol-12345",
                    "Size": 100,
                    "VolumeType": "gp2",
                    "AvailabilityZone": "us-east-1a",
                }
            ]
        }

        optimizer.ec2_client.describe_addresses.return_value = {
            "Addresses": [{"PublicIp": "1.2.3.4", "AllocationId": "eipalloc-12345"}]
        }

        optimizer.rds_client.describe_db_instances.return_value = {
            "DBInstances": [
                {
                    "DBInstanceIdentifier": "test-db",
                    "DBInstanceStatus": "available",
                    "DBInstanceClass": "db.t3.micro",
                }
            ]
        }

        # Mock CloudWatch response for RDS CPU utilization
        optimizer._get_rds_cpu_utilization = Mock(return_value=2.0)  # Low CPU

        recommendations = optimizer._analyze_unused_resources(analysis_period)

        assert len(recommendations) >= 2  # At least EBS volume and Elastic IP

        # Check EBS volume recommendation
        ebs_rec = next((r for r in recommendations if "vol-12345" in r.title), None)
        assert ebs_rec is not None
        assert ebs_rec.type == OptimizationType.UNUSED_RESOURCES
        assert ebs_rec.resource_id == "vol-12345"
        assert ebs_rec.potential_savings.amount > 0

    def test_analyze_rightsizing_opportunities(self, optimizer, analysis_period):
        """Test rightsizing opportunities analysis."""
        # Mock Cost Explorer rightsizing response
        optimizer.ce_client.get_rightsizing_recommendation.return_value = {
            "RightsizingRecommendations": [
                {
                    "CurrentInstance": {
                        "ResourceId": "i-12345",
                        "InstanceType": "m5.large",
                    },
                    "ModifyRecommendationDetail": {
                        "TargetInstances": [{"InstanceType": "m5.medium"}],
                        "EstimatedMonthlySavings": "75.50",
                    },
                }
            ]
        }

        recommendations = optimizer._analyze_rightsizing_opportunities(analysis_period)

        assert len(recommendations) == 1
        rec = recommendations[0]
        assert rec.type == OptimizationType.RIGHTSIZING
        assert rec.resource_id == "i-12345"
        assert rec.potential_savings.amount == Decimal("75.50")
        assert "m5.medium" in rec.action_required

    def test_get_reserved_instance_recommendations(self, optimizer):
        """Test Reserved Instance recommendations."""
        # Mock Cost Explorer RI recommendation response
        optimizer.ce_client.get_reservation_purchase_recommendation.return_value = {
            "Recommendations": [
                {
                    "RecommendationDetails": {
                        "EstimatedMonthlySavingsAmount": "120.00",
                        "RecommendedNumberOfInstancesToPurchase": "2",
                        "InstanceDetails": {
                            "EC2InstanceDetails": {
                                "InstanceType": "m5.large",
                                "Platform": "Linux/UNIX",
                                "Region": "us-east-1",
                            }
                        },
                    }
                }
            ]
        }

        recommendations = optimizer._get_reserved_instance_recommendations()

        assert len(recommendations) == 1
        rec = recommendations[0]
        assert rec.type == OptimizationType.RESERVED_INSTANCES
        assert rec.potential_savings.amount == Decimal("120.00")
        assert "m5.large" in rec.title

    def test_get_savings_plan_recommendations(self, optimizer):
        """Test Savings Plan recommendations."""
        # Mock Cost Explorer Savings Plan recommendation response
        optimizer.ce_client.get_savings_plans_purchase_recommendation.return_value = {
            "SavingsPlansRecommendations": [
                {
                    "EstimatedMonthlySavings": "200.00",
                    "HourlyCommitment": "0.50",
                    "UpfrontCost": "0.00",
                    "SavingsPlansDetails": {"OfferingId": "sp-12345"},
                }
            ]
        }

        recommendations = optimizer._get_savings_plan_recommendations()

        assert len(recommendations) == 1
        rec = recommendations[0]
        assert rec.type == OptimizationType.SAVINGS_PLANS
        assert rec.potential_savings.amount == Decimal("200.00")
        assert "$0.50/hour" in rec.description

    def test_detect_cost_anomalies(self, optimizer, analysis_period):
        """Test cost anomaly detection."""
        # Mock Cost Explorer anomaly detection response
        optimizer.ce_client.get_anomalies.return_value = {
            "Anomalies": [
                {
                    "DimensionKey": "Amazon EC2",
                    "AnomalyStartDate": "2024-01-15",
                    "Impact": {"MaxImpact": "150.00", "TotalImpact": "25.5"},
                    "AnomalyScore": {"CurrentScore": 85.2},
                }
            ]
        }

        optimizer._analyze_anomaly_root_cause = Mock(
            return_value="Service-level cost spike"
        )

        anomalies = optimizer._detect_cost_anomalies(analysis_period)

        assert len(anomalies) == 1
        anomaly = anomalies[0]
        assert anomaly.service == "Amazon EC2"
        assert anomaly.actual_cost.amount == Decimal("150.00")
        assert anomaly.variance_percentage == 25.5
        assert anomaly.severity == SeverityLevel.HIGH  # > $100

    def test_analyze_budget_variances(self, optimizer, analysis_period):
        """Test budget variance analysis."""
        # Mock STS response for account ID
        with patch("boto3.Session") as mock_session:
            mock_sts = Mock()
            mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
            mock_session.return_value.client.return_value = mock_sts

            optimizer.session.client.return_value = mock_sts

            # Mock Budgets response
            optimizer.budgets_client.describe_budgets.return_value = {
                "Budgets": [
                    {
                        "BudgetName": "Monthly-Budget",
                        "BudgetLimit": {"Amount": "1000.00"},
                    }
                ]
            }

            # Mock actual spending calculation
            optimizer._get_actual_spending_for_budget = Mock(
                return_value=Decimal("1200.00")
            )

            variances = optimizer._analyze_budget_variances(analysis_period)

            assert len(variances) == 1
            variance = variances[0]
            assert variance.budget_name == "Monthly-Budget"
            assert variance.budgeted_amount.amount == Decimal("1000.00")
            assert variance.actual_amount.amount == Decimal("1200.00")
            assert variance.is_over_budget is True
            assert variance.variance_percentage == 20.0

    def test_find_unused_ebs_volumes(self, optimizer):
        """Test finding unused EBS volumes."""
        optimizer.ec2_client.describe_volumes.return_value = {
            "Volumes": [
                {
                    "VolumeId": "vol-12345",
                    "Size": 50,
                    "VolumeType": "gp3",
                    "AvailabilityZone": "us-east-1a",
                },
                {
                    "VolumeId": "vol-67890",
                    "Size": 100,
                    "VolumeType": "io1",
                    "AvailabilityZone": "us-east-1b",
                },
            ]
        }

        volumes = optimizer._find_unused_ebs_volumes()

        assert len(volumes) == 2

        # Check cost estimation
        gp3_volume = next(v for v in volumes if v["VolumeId"] == "vol-12345")
        assert gp3_volume["estimated_monthly_cost"] == 50 * 0.08  # gp3 cost

        io1_volume = next(v for v in volumes if v["VolumeId"] == "vol-67890")
        assert io1_volume["estimated_monthly_cost"] == 100 * 0.125  # io1 cost

    def test_find_unused_elastic_ips(self, optimizer):
        """Test finding unused Elastic IPs."""
        optimizer.ec2_client.describe_addresses.return_value = {
            "Addresses": [
                {
                    "PublicIp": "1.2.3.4",
                    "AllocationId": "eipalloc-12345",
                    # No InstanceId or NetworkInterfaceId = unused
                },
                {
                    "PublicIp": "5.6.7.8",
                    "AllocationId": "eipalloc-67890",
                    "InstanceId": "i-12345",  # This one is in use
                },
            ]
        }

        unused_eips = optimizer._find_unused_elastic_ips()

        assert len(unused_eips) == 1
        assert unused_eips[0]["PublicIp"] == "1.2.3.4"

    def test_find_idle_rds_instances(self, optimizer, analysis_period):
        """Test finding idle RDS instances."""
        optimizer.rds_client.describe_db_instances.return_value = {
            "DBInstances": [
                {
                    "DBInstanceIdentifier": "idle-db",
                    "DBInstanceStatus": "available",
                    "DBInstanceClass": "db.t3.small",
                },
                {
                    "DBInstanceIdentifier": "busy-db",
                    "DBInstanceStatus": "available",
                    "DBInstanceClass": "db.m5.large",
                },
            ]
        }

        # Mock CPU utilization - one idle, one busy
        def mock_cpu_utilization(instance_id, period):
            if instance_id == "idle-db":
                return 2.0  # Very low CPU
            else:
                return 75.0  # High CPU

        optimizer._get_rds_cpu_utilization = Mock(side_effect=mock_cpu_utilization)

        idle_instances = optimizer._find_idle_rds_instances(analysis_period)

        assert len(idle_instances) == 1
        assert idle_instances[0]["DBInstanceIdentifier"] == "idle-db"
        assert idle_instances[0]["cpu_utilization"] == 2.0

    def test_get_rds_cpu_utilization(self, optimizer, analysis_period):
        """Test getting RDS CPU utilization from CloudWatch."""
        optimizer.cloudwatch_client.get_metric_statistics.return_value = {
            "Datapoints": [{"Average": 10.5}, {"Average": 15.2}, {"Average": 8.7}]
        }

        cpu_util = optimizer._get_rds_cpu_utilization("test-db", analysis_period)

        expected_avg = (10.5 + 15.2 + 8.7) / 3
        assert abs(cpu_util - expected_avg) < 0.1

    def test_estimate_rds_monthly_cost(self, optimizer):
        """Test RDS monthly cost estimation."""
        # Test known instance types
        assert optimizer._estimate_rds_monthly_cost("db.t3.micro") == 15
        assert optimizer._estimate_rds_monthly_cost("db.m5.large") == 150
        assert optimizer._estimate_rds_monthly_cost("db.r5.xlarge") == 360

        # Test unknown instance type (should return default)
        assert optimizer._estimate_rds_monthly_cost("db.unknown.type") == 100

    def test_analyze_anomaly_root_cause(self, optimizer):
        """Test anomaly root cause analysis."""
        # Test service-level anomaly
        anomaly = {"DimensionKey": "SERVICE", "AnomalyScore": {"CurrentScore": 85.0}}

        root_cause = optimizer._analyze_anomaly_root_cause(anomaly)
        assert "Service-level cost spike" in root_cause
        assert "Highly unusual spending pattern" in root_cause

    def test_get_actual_spending_for_budget(self, optimizer, analysis_period):
        """Test getting actual spending for budget period."""
        optimizer.ce_client.get_cost_and_usage.return_value = {
            "ResultsByTime": [
                {"Total": {"BlendedCost": {"Amount": "500.00"}}},
                {"Total": {"BlendedCost": {"Amount": "600.00"}}},
            ]
        }

        budget = {"BudgetName": "test-budget"}
        actual_spending = optimizer._get_actual_spending_for_budget(
            budget, analysis_period
        )

        assert actual_spending == Decimal("1100.00")  # 500 + 600

    def test_error_handling_in_analysis_methods(self, optimizer, analysis_period):
        """Test that analysis methods handle errors gracefully."""
        # Mock methods to raise exceptions
        optimizer.ec2_client.describe_volumes.side_effect = Exception("API Error")
        optimizer.ce_client.get_rightsizing_recommendation.side_effect = Exception(
            "API Error"
        )

        # These should not raise exceptions, just return empty lists
        unused_resources = optimizer._analyze_unused_resources(analysis_period)
        rightsizing_recs = optimizer._analyze_rightsizing_opportunities(analysis_period)

        assert isinstance(unused_resources, list)
        assert isinstance(rightsizing_recs, list)

    def test_default_analysis_period(self, optimizer):
        """Test that default analysis period is used when none provided."""
        optimizer._analyze_unused_resources = Mock(return_value=[])
        optimizer._analyze_rightsizing_opportunities = Mock(return_value=[])
        optimizer._get_reserved_instance_recommendations = Mock(return_value=[])
        optimizer._get_savings_plan_recommendations = Mock(return_value=[])
        optimizer._detect_cost_anomalies = Mock(return_value=[])
        optimizer._analyze_budget_variances = Mock(return_value=[])

        report = optimizer.generate_optimization_report()  # No period provided

        # Should use default 30-day period
        assert report.analysis_period is not None
        period_days = (report.analysis_period.end - report.analysis_period.start).days
        assert 29 <= period_days <= 31  # Account for partial days


class TestOptimizationDataClasses:
    """Test optimization data classes."""

    def test_optimization_recommendation_creation(self):
        """Test creating OptimizationRecommendation."""
        rec = OptimizationRecommendation(
            type=OptimizationType.UNUSED_RESOURCES,
            severity=SeverityLevel.HIGH,
            title="Test recommendation",
            description="Test description",
            potential_savings=CostAmount(amount=Decimal("100.0")),
            confidence_level=0.9,
            resource_id="test-resource",
            service="Test Service",
            action_required="Delete resource",
        )

        assert rec.type == OptimizationType.UNUSED_RESOURCES
        assert rec.severity == SeverityLevel.HIGH
        assert rec.potential_savings.amount == Decimal("100.0")
        assert rec.confidence_level == 0.9

    def test_cost_anomaly_creation(self):
        """Test creating CostAnomaly."""
        anomaly = CostAnomaly(
            service="Amazon EC2",
            anomaly_date=datetime(2024, 1, 15),
            expected_cost=CostAmount(amount=Decimal("100.0")),
            actual_cost=CostAmount(amount=Decimal("150.0")),
            variance_percentage=50.0,
            severity=SeverityLevel.MEDIUM,
            description="Cost spike detected",
        )

        assert anomaly.service == "Amazon EC2"
        assert anomaly.variance_percentage == 50.0
        assert anomaly.severity == SeverityLevel.MEDIUM

    def test_budget_variance_creation(self):
        """Test creating BudgetVariance."""
        period = TimePeriod(start=datetime(2024, 1, 1), end=datetime(2024, 1, 31))

        variance = BudgetVariance(
            budget_name="Monthly Budget",
            budgeted_amount=CostAmount(amount=Decimal("1000.0")),
            actual_amount=CostAmount(amount=Decimal("1200.0")),
            variance_amount=CostAmount(amount=Decimal("200.0")),
            variance_percentage=20.0,
            time_period=period,
            is_over_budget=True,
        )

        assert variance.budget_name == "Monthly Budget"
        assert variance.is_over_budget is True
        assert variance.variance_percentage == 20.0

    def test_optimization_report_creation(self):
        """Test creating OptimizationReport."""
        period = TimePeriod(start=datetime(2024, 1, 1), end=datetime(2024, 1, 31))

        report = OptimizationReport(
            recommendations=[],
            anomalies=[],
            budget_variances=[],
            total_potential_savings=CostAmount(amount=Decimal("500.0")),
            report_date=datetime.now(),
            analysis_period=period,
        )

        assert report.total_potential_savings.amount == Decimal("500.0")
        assert report.analysis_period == period
        assert isinstance(report.recommendations, list)
        assert isinstance(report.anomalies, list)
        assert isinstance(report.budget_variances, list)
