"""Query validation middleware for AWS Cost CLI."""

import re
from datetime import datetime, timedelta, date
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from .models import QueryParameters, TimePeriod, MetricType
from .exceptions import ValidationError


@dataclass
class ValidationResult:
    """Result of query validation."""

    is_valid: bool
    errors: List[str]
    warnings: List[str]


class QueryValidator:
    """Validates queries to prevent malformed requests and catch issues early."""

    # Maximum date ranges for different granularities (in days)
    MAX_DATE_RANGES = {
        "DAILY": 365,  # 1 year for daily data
        "MONTHLY": 1095,  # 3 years for monthly data
        "HOURLY": 7,  # 1 week for hourly data (if supported in future)
    }

    # Valid AWS service names (subset of common services)
    VALID_SERVICE_NAMES = {
        "ec2",
        "amazon elastic compute cloud - compute",
        "s3",
        "amazon simple storage service",
        "rds",
        "amazon relational database service",
        "lambda",
        "aws lambda",
        "cloudfront",
        "amazon cloudfront",
        "route53",
        "amazon route 53",
        "ebs",
        "amazon elastic block store",
        "elb",
        "elastic load balancing",
        "vpc",
        "amazon virtual private cloud",
        "cloudwatch",
        "amazon cloudwatch",
        "iam",
        "aws identity and access management",
        "sns",
        "amazon simple notification service",
        "sqs",
        "amazon simple queue service",
        "elasticache",
        "amazon elasticache",
        "redshift",
        "amazon redshift",
        "dynamodb",
        "amazon dynamodb",
        "kinesis",
        "amazon kinesis",
        "emr",
        "amazon elastic mapreduce",
        "sagemaker",
        "amazon sagemaker",
        "eks",
        "amazon elastic kubernetes service",
        "ecs",
        "amazon elastic container service",
        "fargate",
        "aws fargate",
        "batch",
        "aws batch",
        "glue",
        "aws glue",
        "athena",
        "amazon athena",
        "quicksight",
        "amazon quicksight",
        "workspaces",
        "amazon workspaces",
        "connect",
        "amazon connect",
        "chime",
        "amazon chime",
        "api gateway",
        "amazon api gateway",
        "elastic search",
        "amazon elasticsearch service",
        "opensearch",
        "amazon opensearch service",
    }

    # SQL injection patterns to detect
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE)\b)",
        r"(\bunion\s+select\b)",
        r"(\bor\s+1\s*=\s*1\b)",
        r"(\band\s+1\s*=\s*1\b)",
        r"(--|#|/\*|\*/)",
        r"(\bxp_cmdshell\b)",
    ]

    def __init__(self):
        """Initialize the query validator."""
        self.sql_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.SQL_INJECTION_PATTERNS
        ]

    def validate_query_parameters(self, params: QueryParameters) -> ValidationResult:
        """Validate query parameters for correctness and safety."""
        errors = []
        warnings = []

        # Validate date range
        date_errors, date_warnings = self._validate_date_range(
            params.time_period, params.granularity
        )
        errors.extend(date_errors)
        warnings.extend(date_warnings)

        # Validate services
        if params.services:
            service_errors = self._validate_services(params.services)
            errors.extend(service_errors)

        # Validate metrics
        if params.metrics:
            metric_errors = self._validate_metrics(params.metrics)
            errors.extend(metric_errors)

        # Validate dimensions and filters
        if params.group_by:
            group_errors = self._validate_group_by(params.group_by)
            errors.extend(group_errors)

        # Check for potential security issues
        security_errors = self._validate_security(params)
        errors.extend(security_errors)

        # Performance warnings
        perf_warnings = self._check_performance_implications(params)
        warnings.extend(perf_warnings)

        return ValidationResult(
            is_valid=len(errors) == 0, errors=errors, warnings=warnings
        )

    def validate_raw_query(self, query: str) -> ValidationResult:
        """Validate raw query string for basic issues."""
        errors = []
        warnings = []

        # Check query length
        if len(query.strip()) == 0:
            errors.append("Query cannot be empty")
        elif len(query) > 1000:
            warnings.append("Very long query detected - consider simplifying")

        # Check for potential SQL injection attempts
        for pattern in self.sql_patterns:
            if pattern.search(query):
                errors.append(
                    f"Potentially malicious pattern detected in query: {pattern.pattern}"
                )

        # Check for suspicious characters
        if any(char in query for char in ["<", ">", "{", "}", "`"]):
            warnings.append(
                "Query contains unusual characters that might cause parsing issues"
            )

        return ValidationResult(
            is_valid=len(errors) == 0, errors=errors, warnings=warnings
        )

    def _validate_date_range(
        self, time_period: Optional[TimePeriod], granularity: Optional[str]
    ) -> tuple[List[str], List[str]]:
        """Validate date range and granularity."""
        errors = []
        warnings = []

        if not time_period:
            return errors, warnings

        # Check that end date is after start date
        if time_period.end <= time_period.start:
            errors.append("End date must be after start date")
            return errors, warnings

        # Calculate date range in days
        date_range_days = (time_period.end - time_period.start).days

        # Check against maximum ranges
        granularity = granularity or "MONTHLY"
        max_days = self.MAX_DATE_RANGES.get(granularity.upper(), 365)

        if date_range_days > max_days:
            errors.append(
                f"Date range too large for {granularity} granularity. Maximum: {max_days} days, requested: {date_range_days} days"
            )

        # Check for future dates
        today = datetime.now().date()
        if isinstance(time_period.end, datetime):
            end_date = time_period.end.date()
        else:
            end_date = time_period.end

        if end_date > today:
            errors.append("End date cannot be in the future")

        # Check for very old dates (AWS Cost Explorer has limited historical data)
        oldest_supported = today - timedelta(days=365 * 2)  # 2 years
        if isinstance(time_period.start, datetime):
            start_date = time_period.start.date()
        else:
            start_date = time_period.start

        if start_date < oldest_supported:
            warnings.append(
                f"Start date is very old ({start_date}). AWS Cost Explorer may have limited data availability"
            )

        # Warn about large date ranges that might be slow
        if date_range_days > 90:
            warnings.append(
                f"Large date range ({date_range_days} days) may result in slow queries"
            )

        return errors, warnings

    def _validate_services(self, services: List[str]) -> List[str]:
        """Validate AWS service names."""
        errors = []

        for service in services:
            service_lower = service.lower().strip()
            # Check if it's a known service (partial matching)
            is_valid = any(
                service_lower in valid_service or valid_service in service_lower
                for valid_service in self.VALID_SERVICE_NAMES
            )

            if not is_valid and len(service) > 100:
                errors.append(f"Service name too long: {service[:50]}...")
            elif not is_valid:
                # This is a warning rather than error since AWS adds new services
                pass  # We'll be permissive with service names

        if len(services) > 20:
            errors.append(
                f"Too many services specified ({len(services)}). Consider using fewer services for better performance"
            )

        return errors

    def _validate_metrics(self, metrics: List[str]) -> List[str]:
        """Validate metric types."""
        errors = []
        valid_metrics = {member.value for member in MetricType}

        for metric in metrics:
            if metric not in valid_metrics:
                errors.append(
                    f"Invalid metric type: {metric}. Valid options: {', '.join(valid_metrics)}"
                )

        return errors

    def _validate_group_by(self, group_by: List[str]) -> List[str]:
        """Validate group by dimensions."""
        errors = []

        # Common valid dimensions
        valid_dimensions = {
            "SERVICE",
            "AZ",
            "INSTANCE_TYPE",
            "REGION",
            "USAGE_TYPE",
            "OPERATION",
            "PURCHASE_TYPE",
            "RECORD_TYPE",
            "RESOURCE_ID",
            "LINKED_ACCOUNT",
            "TENANCY",
            "PLATFORM",
            "DATABASE_ENGINE",
        }

        for dimension in group_by:
            if dimension.upper() not in valid_dimensions:
                # Warning only - AWS may have new dimensions
                pass

        if len(group_by) > 5:
            errors.append(
                f"Too many group by dimensions ({len(group_by)}). AWS Cost Explorer supports up to 2 group by dimensions"
            )

        return errors

    def _validate_security(self, params: QueryParameters) -> List[str]:
        """Check for potential security issues in query parameters."""
        errors = []

        # Check all string fields for suspicious content
        string_fields = []

        if params.services:
            string_fields.extend(params.services)
        if params.group_by:
            string_fields.extend(params.group_by)
        if hasattr(params, "raw_query") and params.raw_query:
            string_fields.append(params.raw_query)

        for field_value in string_fields:
            if isinstance(field_value, str):
                for pattern in self.sql_patterns:
                    if pattern.search(field_value):
                        errors.append(
                            f"Potentially malicious content detected: {field_value[:50]}"
                        )
                        break

        return errors

    def _check_performance_implications(self, params: QueryParameters) -> List[str]:
        """Check for query patterns that might impact performance."""
        warnings = []

        # Large date range + daily granularity
        if params.time_period:
            date_range_days = (params.time_period.end - params.time_period.start).days
            if date_range_days > 90 and params.granularity == "DAILY":
                warnings.append(
                    "Daily granularity with large date range may be slow. Consider using MONTHLY granularity"
                )

        # Many services with grouping
        if params.services and len(params.services) > 10 and params.group_by:
            warnings.append(
                "Querying many services with grouping may result in large result sets"
            )

        # Multiple group by dimensions
        if params.group_by and len(params.group_by) > 1:
            warnings.append(
                "Multiple group by dimensions may result in large result sets and slower queries"
            )

        return warnings


class ValidationMiddleware:
    """Middleware to validate queries before processing."""

    def __init__(self, validator: Optional[QueryValidator] = None):
        """Initialize validation middleware."""
        self.validator = validator or QueryValidator()

    def validate_and_process(
        self, query: str, params: Optional[QueryParameters] = None
    ) -> ValidationResult:
        """Validate query and parameters, returning combined results."""
        # Validate raw query
        query_result = self.validator.validate_raw_query(query)

        # If we have parsed parameters, validate them too
        if params:
            params_result = self.validator.validate_query_parameters(params)

            # Combine results
            return ValidationResult(
                is_valid=query_result.is_valid and params_result.is_valid,
                errors=query_result.errors + params_result.errors,
                warnings=query_result.warnings + params_result.warnings,
            )

        return query_result

    def raise_on_errors(self, result: ValidationResult):
        """Raise ValidationError if validation failed."""
        if not result.is_valid:
            error_msg = "Query validation failed:\n" + "\n".join(
                f"- {error}" for error in result.errors
            )
            raise ValidationError(error_msg)
