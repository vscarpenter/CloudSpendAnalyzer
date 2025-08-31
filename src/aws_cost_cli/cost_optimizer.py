"""Cost optimization recommendations and analysis."""

import boto3
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from .models import CostAmount, TimePeriod


class OptimizationType(Enum):
    """Types of cost optimization recommendations."""

    RIGHTSIZING = "rightsizing"
    RESERVED_INSTANCES = "reserved_instances"
    SAVINGS_PLANS = "savings_plans"
    UNUSED_RESOURCES = "unused_resources"
    COST_ANOMALY = "cost_anomaly"
    BUDGET_VARIANCE = "budget_variance"


class SeverityLevel(Enum):
    """Severity levels for optimization recommendations."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class OptimizationRecommendation:
    """Individual cost optimization recommendation."""

    type: OptimizationType
    severity: SeverityLevel
    title: str
    description: str
    potential_savings: CostAmount
    confidence_level: float  # 0.0 to 1.0
    resource_id: Optional[str] = None
    service: Optional[str] = None
    region: Optional[str] = None
    action_required: Optional[str] = None
    estimated_effort: Optional[str] = None  # "low", "medium", "high"
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class CostAnomaly:
    """Cost anomaly detection result."""

    service: str
    anomaly_date: datetime
    expected_cost: CostAmount
    actual_cost: CostAmount
    variance_percentage: float
    severity: SeverityLevel
    description: str
    root_cause_analysis: Optional[str] = None


@dataclass
class BudgetVariance:
    """Budget variance analysis result."""

    budget_name: str
    budgeted_amount: CostAmount
    actual_amount: CostAmount
    variance_amount: CostAmount
    variance_percentage: float
    time_period: TimePeriod
    is_over_budget: bool
    forecast_end_of_period: Optional[CostAmount] = None


@dataclass
class OptimizationReport:
    """Complete cost optimization report."""

    recommendations: List[OptimizationRecommendation]
    anomalies: List[CostAnomaly]
    budget_variances: List[BudgetVariance]
    total_potential_savings: CostAmount
    report_date: datetime
    analysis_period: TimePeriod


class CostOptimizer:
    """Main cost optimization analyzer."""

    def __init__(self, profile: Optional[str] = None, region: str = "us-east-1"):
        """Initialize cost optimizer."""
        self.profile = profile
        self.region = region
        self.session = (
            boto3.Session(profile_name=profile) if profile else boto3.Session()
        )

        # Initialize AWS clients
        self.ce_client = self.session.client("ce", region_name=region)
        self.budgets_client = self.session.client("budgets", region_name=region)
        self.ec2_client = self.session.client("ec2", region_name=region)
        self.rds_client = self.session.client("rds", region_name=region)
        self.cloudwatch_client = self.session.client("cloudwatch", region_name=region)

    def generate_optimization_report(
        self, analysis_period: Optional[TimePeriod] = None
    ) -> OptimizationReport:
        """Generate comprehensive cost optimization report."""
        if not analysis_period:
            # Default to last 30 days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            analysis_period = TimePeriod(start=start_date, end=end_date)

        recommendations = []
        anomalies = []
        budget_variances = []

        # Analyze unused resources
        unused_recommendations = self._analyze_unused_resources(analysis_period)
        recommendations.extend(unused_recommendations)

        # Analyze rightsizing opportunities
        rightsizing_recommendations = self._analyze_rightsizing_opportunities(
            analysis_period
        )
        recommendations.extend(rightsizing_recommendations)

        # Get Reserved Instance recommendations
        ri_recommendations = self._get_reserved_instance_recommendations()
        recommendations.extend(ri_recommendations)

        # Get Savings Plan recommendations
        sp_recommendations = self._get_savings_plan_recommendations()
        recommendations.extend(sp_recommendations)

        # Detect cost anomalies
        detected_anomalies = self._detect_cost_anomalies(analysis_period)
        anomalies.extend(detected_anomalies)

        # Analyze budget variances
        budget_analysis = self._analyze_budget_variances(analysis_period)
        budget_variances.extend(budget_analysis)

        # Calculate total potential savings
        total_savings = sum(
            (rec.potential_savings.amount for rec in recommendations), Decimal("0.0")
        )

        return OptimizationReport(
            recommendations=recommendations,
            anomalies=anomalies,
            budget_variances=budget_variances,
            total_potential_savings=CostAmount(amount=total_savings),
            report_date=datetime.now(),
            analysis_period=analysis_period,
        )

    def _analyze_unused_resources(
        self, period: TimePeriod
    ) -> List[OptimizationRecommendation]:
        """Analyze unused or underutilized resources."""
        recommendations = []

        try:
            # Analyze unused EBS volumes
            unused_volumes = self._find_unused_ebs_volumes()
            for volume in unused_volumes:
                recommendations.append(
                    OptimizationRecommendation(
                        type=OptimizationType.UNUSED_RESOURCES,
                        severity=SeverityLevel.MEDIUM,
                        title=f"Unused EBS Volume: {volume['VolumeId']}",
                        description=f"EBS volume {volume['VolumeId']} is not attached to any instance",
                        potential_savings=CostAmount(
                            amount=Decimal(str(volume["estimated_monthly_cost"]))
                        ),
                        confidence_level=0.9,
                        resource_id=volume["VolumeId"],
                        service="Amazon Elastic Block Store",
                        region=volume.get("AvailabilityZone", "").rstrip("abcdef"),
                        action_required="Delete unused volume or attach to instance",
                        estimated_effort="low",
                        metadata={
                            "volume_size": volume.get("Size"),
                            "volume_type": volume.get("VolumeType"),
                        },
                    )
                )

            # Analyze unused Elastic IPs
            unused_eips = self._find_unused_elastic_ips()
            for eip in unused_eips:
                recommendations.append(
                    OptimizationRecommendation(
                        type=OptimizationType.UNUSED_RESOURCES,
                        severity=SeverityLevel.LOW,
                        title=f"Unused Elastic IP: {eip['PublicIp']}",
                        description=f"Elastic IP {eip['PublicIp']} is not associated with any instance",
                        potential_savings=CostAmount(
                            amount=Decimal("3.65")
                        ),  # ~$0.005/hour * 24 * 30
                        confidence_level=0.95,
                        resource_id=eip.get("AllocationId"),
                        service="Amazon EC2",
                        action_required="Release unused Elastic IP",
                        estimated_effort="low",
                    )
                )

            # Analyze idle RDS instances
            idle_rds = self._find_idle_rds_instances(period)
            for instance in idle_rds:
                recommendations.append(
                    OptimizationRecommendation(
                        type=OptimizationType.UNUSED_RESOURCES,
                        severity=SeverityLevel.HIGH,
                        title=f"Idle RDS Instance: {instance['DBInstanceIdentifier']}",
                        description=f"RDS instance {instance['DBInstanceIdentifier']} shows very low utilization",
                        potential_savings=CostAmount(
                            amount=Decimal(str(instance["estimated_monthly_cost"]))
                        ),
                        confidence_level=0.8,
                        resource_id=instance["DBInstanceIdentifier"],
                        service="Amazon RDS",
                        action_required="Consider stopping, downsizing, or terminating instance",
                        estimated_effort="medium",
                        metadata={"instance_class": instance.get("DBInstanceClass")},
                    )
                )

        except Exception as _e:
            # Log error but don't fail the entire analysis
            pass

        return recommendations

    def _analyze_rightsizing_opportunities(
        self, period: TimePeriod
    ) -> List[OptimizationRecommendation]:
        """Analyze EC2 rightsizing opportunities."""
        recommendations = []

        try:
            response = self.ce_client.get_rightsizing_recommendation(
                Service="AmazonEC2",
                Configuration={
                    "BenefitsConsidered": True,
                    "RecommendationTarget": "SAME_INSTANCE_FAMILY",
                },
            )

            for rec in response.get("RightsizingRecommendations", []):
                current_instance = rec.get("CurrentInstance", {})
                modify_rec = rec.get("ModifyRecommendationDetail", {})

                if modify_rec:
                    target_instances = modify_rec.get("TargetInstances", [])
                    if target_instances:
                        target = target_instances[0]

                        estimated_savings = Decimal("0.0")
                        if modify_rec.get("EstimatedMonthlySavings"):
                            estimated_savings = Decimal(
                                modify_rec["EstimatedMonthlySavings"]
                            )

                        recommendations.append(
                            OptimizationRecommendation(
                                type=OptimizationType.RIGHTSIZING,
                                severity=(
                                    SeverityLevel.MEDIUM
                                    if estimated_savings > 50
                                    else SeverityLevel.LOW
                                ),
                                title=f"Rightsize EC2 Instance: {current_instance.get('ResourceId', 'Unknown')}",
                                description=f"Instance can be downsized from {current_instance.get('InstanceType')} to {target.get('InstanceType')}",
                                potential_savings=CostAmount(amount=estimated_savings),
                                confidence_level=0.85,
                                resource_id=current_instance.get("ResourceId"),
                                service="Amazon EC2",
                                action_required=f"Resize instance to {target.get('InstanceType')}",
                                estimated_effort="medium",
                                metadata={
                                    "current_type": current_instance.get(
                                        "InstanceType"
                                    ),
                                    "recommended_type": target.get("InstanceType"),
                                    "utilization": current_instance.get(
                                        "UtilizationMetrics", {}
                                    ),
                                },
                            )
                        )

        except Exception as _e:
            # Rightsizing API might not be available in all regions
            pass

        return recommendations

    def _get_reserved_instance_recommendations(
        self,
    ) -> List[OptimizationRecommendation]:
        """Get Reserved Instance purchase recommendations."""
        recommendations = []

        try:
            response = self.ce_client.get_reservation_purchase_recommendation(
                Service="AmazonEC2", PaymentOption="NO_UPFRONT", TermInYears="ONE_YEAR"
            )

            for rec in response.get("Recommendations", []):
                details = rec.get("RecommendationDetails", {})

                estimated_savings = Decimal("0.0")
                if details.get("EstimatedMonthlySavingsAmount"):
                    estimated_savings = Decimal(
                        details["EstimatedMonthlySavingsAmount"]
                    )

                instance_details = details.get("InstanceDetails", {}).get(
                    "EC2InstanceDetails", {}
                )

                recommendations.append(
                    OptimizationRecommendation(
                        type=OptimizationType.RESERVED_INSTANCES,
                        severity=(
                            SeverityLevel.MEDIUM
                            if estimated_savings > 100
                            else SeverityLevel.LOW
                        ),
                        title=f"Reserved Instance Opportunity: {instance_details.get('InstanceType', 'EC2')}",
                        description=f"Purchase {details.get('RecommendedNumberOfInstancesToPurchase', 1)} Reserved Instances",
                        potential_savings=CostAmount(amount=estimated_savings),
                        confidence_level=0.9,
                        service="Amazon EC2",
                        action_required="Purchase Reserved Instances",
                        estimated_effort="low",
                        metadata={
                            "instance_type": instance_details.get("InstanceType"),
                            "platform": instance_details.get("Platform"),
                            "region": instance_details.get("Region"),
                            "recommended_quantity": details.get(
                                "RecommendedNumberOfInstancesToPurchase"
                            ),
                        },
                    )
                )

        except Exception as _e:
            # RI recommendations might not be available
            pass

        return recommendations

    def _get_savings_plan_recommendations(self) -> List[OptimizationRecommendation]:
        """Get Savings Plan purchase recommendations."""
        recommendations = []

        try:
            response = self.ce_client.get_savings_plans_purchase_recommendation(
                SavingsPlansType="COMPUTE_SP",
                TermInYears="ONE_YEAR",
                PaymentOption="NO_UPFRONT",
            )

            for rec in response.get("SavingsPlansRecommendations", []):
                details = rec.get("SavingsPlansDetails", {})

                estimated_savings = Decimal("0.0")
                if rec.get("EstimatedMonthlySavings"):
                    estimated_savings = Decimal(rec["EstimatedMonthlySavings"])

                recommendations.append(
                    OptimizationRecommendation(
                        type=OptimizationType.SAVINGS_PLANS,
                        severity=(
                            SeverityLevel.MEDIUM
                            if estimated_savings > 200
                            else SeverityLevel.LOW
                        ),
                        title="Savings Plan Opportunity",
                        description=f"Purchase Compute Savings Plan with ${rec.get('HourlyCommitment', 0)}/hour commitment",
                        potential_savings=CostAmount(amount=estimated_savings),
                        confidence_level=0.85,
                        service="AWS Savings Plans",
                        action_required="Purchase Savings Plan",
                        estimated_effort="low",
                        metadata={
                            "hourly_commitment": rec.get("HourlyCommitment"),
                            "savings_plan_type": details.get("OfferingId"),
                            "upfront_cost": rec.get("UpfrontCost"),
                        },
                    )
                )

        except Exception as _e:
            # Savings Plans recommendations might not be available
            pass

        return recommendations

    def _detect_cost_anomalies(self, period: TimePeriod) -> List[CostAnomaly]:
        """Detect cost anomalies using AWS Cost Anomaly Detection."""
        anomalies = []

        try:
            response = self.ce_client.get_anomalies(
                DateInterval={
                    "StartDate": period.start.strftime("%Y-%m-%d"),
                    "EndDate": period.end.strftime("%Y-%m-%d"),
                }
            )

            for anomaly in response.get("Anomalies", []):
                impact = anomaly.get("Impact", {})
                max_impact = Decimal(impact.get("MaxImpact", "0"))

                if max_impact > 10:  # Only report significant anomalies
                    anomalies.append(
                        CostAnomaly(
                            service=anomaly.get("DimensionKey", "Unknown"),
                            anomaly_date=datetime.strptime(
                                anomaly.get("AnomalyStartDate", ""), "%Y-%m-%d"
                            ),
                            expected_cost=CostAmount(
                                amount=Decimal("0")
                            ),  # AWS doesn't provide expected cost
                            actual_cost=CostAmount(amount=max_impact),
                            variance_percentage=float(impact.get("TotalImpact", 0)),
                            severity=(
                                SeverityLevel.HIGH
                                if max_impact > 100
                                else SeverityLevel.MEDIUM
                            ),
                            description=f"Cost anomaly detected: {anomaly.get('AnomalyScore', {}).get('CurrentScore', 0):.1f} anomaly score",
                            root_cause_analysis=self._analyze_anomaly_root_cause(
                                anomaly
                            ),
                        )
                    )

        except Exception as _e:
            # Cost Anomaly Detection might not be enabled
            pass

        return anomalies

    def _analyze_budget_variances(self, period: TimePeriod) -> List[BudgetVariance]:
        """Analyze budget variances."""
        variances = []

        try:
            # Get account ID
            sts_client = self.session.client("sts")
            account_id = sts_client.get_caller_identity()["Account"]

            # List budgets
            response = self.budgets_client.describe_budgets(AccountId=account_id)

            for budget in response.get("Budgets", []):
                budget_name = budget["BudgetName"]
                budgeted_amount = Decimal(budget["BudgetLimit"]["Amount"])

                # Get actual spending for budget period
                actual_spending = self._get_actual_spending_for_budget(budget, period)

                variance_amount = actual_spending - budgeted_amount
                variance_percentage = (
                    float((variance_amount / budgeted_amount) * 100)
                    if budgeted_amount > 0
                    else 0
                )

                if abs(variance_percentage) > 5:  # Only report significant variances
                    variances.append(
                        BudgetVariance(
                            budget_name=budget_name,
                            budgeted_amount=CostAmount(amount=budgeted_amount),
                            actual_amount=CostAmount(amount=actual_spending),
                            variance_amount=CostAmount(amount=variance_amount),
                            variance_percentage=variance_percentage,
                            time_period=period,
                            is_over_budget=variance_amount > 0,
                        )
                    )

        except Exception as _e:
            # Budgets might not be configured
            pass

        return variances

    def _find_unused_ebs_volumes(self) -> List[Dict[str, Any]]:
        """Find unattached EBS volumes."""
        unused_volumes = []

        try:
            response = self.ec2_client.describe_volumes(
                Filters=[{"Name": "status", "Values": ["available"]}]
            )

            for volume in response.get("Volumes", []):
                # Estimate monthly cost based on volume size and type
                size = volume.get("Size", 0)
                volume_type = volume.get("VolumeType", "gp2")

                # Rough cost estimates per GB per month
                cost_per_gb = {
                    "gp2": 0.10,
                    "gp3": 0.08,
                    "io1": 0.125,
                    "io2": 0.125,
                    "st1": 0.045,
                    "sc1": 0.025,
                    "standard": 0.05,
                }

                monthly_cost = size * cost_per_gb.get(volume_type, 0.10)

                volume_info = volume.copy()
                volume_info["estimated_monthly_cost"] = monthly_cost
                unused_volumes.append(volume_info)

        except Exception:
            pass

        return unused_volumes

    def _find_unused_elastic_ips(self) -> List[Dict[str, Any]]:
        """Find unassociated Elastic IPs."""
        unused_eips = []

        try:
            response = self.ec2_client.describe_addresses()

            for address in response.get("Addresses", []):
                if "InstanceId" not in address and "NetworkInterfaceId" not in address:
                    unused_eips.append(address)

        except Exception:
            pass

        return unused_eips

    def _find_idle_rds_instances(self, period: TimePeriod) -> List[Dict[str, Any]]:
        """Find idle RDS instances based on CloudWatch metrics."""
        idle_instances = []

        try:
            response = self.rds_client.describe_db_instances()

            for instance in response.get("DBInstances", []):
                if instance["DBInstanceStatus"] == "available":
                    # Check CPU utilization
                    cpu_utilization = self._get_rds_cpu_utilization(
                        instance["DBInstanceIdentifier"], period
                    )

                    if cpu_utilization < 5.0:  # Less than 5% average CPU
                        # Estimate monthly cost (rough approximation)
                        instance_class = instance.get("DBInstanceClass", "")
                        estimated_cost = self._estimate_rds_monthly_cost(instance_class)

                        instance_info = {
                            "DBInstanceIdentifier": instance["DBInstanceIdentifier"],
                            "DBInstanceClass": instance_class,
                            "estimated_monthly_cost": estimated_cost,
                            "cpu_utilization": cpu_utilization,
                        }
                        idle_instances.append(instance_info)

        except Exception:
            pass

        return idle_instances

    def _get_rds_cpu_utilization(self, instance_id: str, period: TimePeriod) -> float:
        """Get average CPU utilization for RDS instance."""
        try:
            response = self.cloudwatch_client.get_metric_statistics(
                Namespace="AWS/RDS",
                MetricName="CPUUtilization",
                Dimensions=[{"Name": "DBInstanceIdentifier", "Value": instance_id}],
                StartTime=period.start,
                EndTime=period.end,
                Period=3600,  # 1 hour
                Statistics=["Average"],
            )

            if response.get("Datapoints"):
                return sum(dp["Average"] for dp in response["Datapoints"]) / len(
                    response["Datapoints"]
                )

        except Exception:
            pass

        return 100.0  # Assume high utilization if we can't get metrics

    def _estimate_rds_monthly_cost(self, instance_class: str) -> float:
        """Estimate monthly cost for RDS instance class."""
        # Rough cost estimates for common instance types (per month)
        cost_estimates = {
            "db.t3.micro": 15,
            "db.t3.small": 30,
            "db.t3.medium": 60,
            "db.t3.large": 120,
            "db.t3.xlarge": 240,
            "db.t3.2xlarge": 480,
            "db.m5.large": 150,
            "db.m5.xlarge": 300,
            "db.m5.2xlarge": 600,
            "db.r5.large": 180,
            "db.r5.xlarge": 360,
            "db.r5.2xlarge": 720,
        }

        return cost_estimates.get(instance_class, 100)  # Default estimate

    def _analyze_anomaly_root_cause(self, anomaly: Dict[str, Any]) -> str:
        """Analyze potential root cause of cost anomaly."""
        root_causes = []

        # Check for common patterns
        if anomaly.get("DimensionKey") == "SERVICE":
            root_causes.append("Service-level cost spike")

        if anomaly.get("AnomalyScore", {}).get("CurrentScore", 0) > 80:
            root_causes.append("Highly unusual spending pattern")

        if not root_causes:
            root_causes.append("Unknown cause - manual investigation recommended")

        return "; ".join(root_causes)

    def _get_actual_spending_for_budget(
        self, budget: Dict[str, Any], period: TimePeriod
    ) -> Decimal:
        """Get actual spending for a budget period."""
        try:
            # This is a simplified implementation
            # In practice, you'd need to match the budget's filters and time period
            response = self.ce_client.get_cost_and_usage(
                TimePeriod={
                    "Start": period.start.strftime("%Y-%m-%d"),
                    "End": period.end.strftime("%Y-%m-%d"),
                },
                Granularity="MONTHLY",
                Metrics=["BlendedCost"],
            )

            total_cost = Decimal("0")
            for result in response.get("ResultsByTime", []):
                for metric_name, metric_data in result.get("Total", {}).items():
                    total_cost += Decimal(metric_data["Amount"])

            return total_cost

        except Exception:
            return Decimal("0")
