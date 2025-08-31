"""Tests for AWS client management."""

import pytest
from unittest.mock import Mock, patch, MagicMock, PropertyMock
from datetime import datetime, timedelta
from botocore.exceptions import ClientError, NoCredentialsError, ProfileNotFound

from src.aws_cost_cli.aws_client import CredentialManager, AWSCostClient
from src.aws_cost_cli.models import (
    QueryParameters,
    TimePeriod,
    MetricType,
    TimePeriodGranularity,
)


class TestCredentialManager:
    """Test cases for CredentialManager."""

    @patch("boto3.Session")
    def test_get_available_profiles(self, mock_session_class):
        """Test getting available AWS profiles."""
        mock_session = Mock()
        mock_session.available_profiles = ["default", "production", "staging"]
        mock_session_class.return_value = mock_session

        manager = CredentialManager()
        profiles = manager.get_available_profiles()

        assert profiles == ["default", "production", "staging"]

    @patch("boto3.Session")
    def test_get_available_profiles_exception(self, mock_session_class):
        """Test handling exception when getting profiles."""
        mock_session = Mock()
        # Make available_profiles property raise an exception when accessed
        type(mock_session).available_profiles = PropertyMock(
            side_effect=Exception("Error")
        )
        mock_session_class.return_value = mock_session

        manager = CredentialManager()
        profiles = manager.get_available_profiles()

        assert profiles == []

    @patch("boto3.Session")
    def test_validate_credentials_success(self, mock_session_class):
        """Test successful credential validation."""
        mock_session = Mock()
        mock_sts_client = Mock()
        mock_sts_client.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_session.client.return_value = mock_sts_client
        mock_session_class.return_value = mock_session

        manager = CredentialManager()
        assert manager.validate_credentials() is True

    @patch("boto3.Session")
    def test_validate_credentials_no_credentials(self, mock_session_class):
        """Test credential validation with no credentials."""
        mock_session = Mock()
        mock_sts_client = Mock()
        mock_sts_client.get_caller_identity.side_effect = NoCredentialsError()
        mock_session.client.return_value = mock_sts_client
        mock_session_class.return_value = mock_session

        manager = CredentialManager()
        assert manager.validate_credentials() is False

    def test_validate_credentials_profile_not_found(self):
        """Test credential validation with invalid profile."""
        manager = CredentialManager()

        with patch("boto3.Session") as mock_session_class:
            mock_session_class.side_effect = ProfileNotFound(profile="invalid")
            assert manager.validate_credentials("invalid") is False

    @patch("boto3.Session")
    def test_get_caller_identity_success(self, mock_session_class):
        """Test successful caller identity retrieval."""
        mock_session = Mock()
        mock_sts_client = Mock()
        expected_identity = {
            "UserId": "AIDACKCEVSQ6C2EXAMPLE",
            "Account": "123456789012",
            "Arn": "arn:aws:iam::123456789012:user/testuser",
        }
        mock_sts_client.get_caller_identity.return_value = expected_identity
        mock_session.client.return_value = mock_sts_client
        mock_session_class.return_value = mock_session

        manager = CredentialManager()
        identity = manager.get_caller_identity()

        assert identity == expected_identity

    @patch("boto3.Session")
    def test_get_caller_identity_failure(self, mock_session_class):
        """Test caller identity retrieval failure."""
        mock_session = Mock()
        mock_sts_client = Mock()
        mock_sts_client.get_caller_identity.side_effect = NoCredentialsError()
        mock_session.client.return_value = mock_sts_client
        mock_session_class.return_value = mock_session

        manager = CredentialManager()
        with pytest.raises(RuntimeError, match="Failed to get caller identity"):
            manager.get_caller_identity()


class TestAWSCostClient:
    """Test cases for AWSCostClient."""

    @patch("boto3.Session")
    def test_init_with_profile(self, mock_session_class):
        """Test client initialization with profile."""
        mock_session = Mock()
        mock_client = Mock()
        mock_session.client.return_value = mock_client

        # Configure the mock to return different instances for different calls
        def session_side_effect(*args, **kwargs):
            if kwargs.get("profile_name") == "production":
                return mock_session
            return Mock()

        mock_session_class.side_effect = session_side_effect

        client = AWSCostClient(profile="production", region="us-west-2")

        assert client.profile == "production"
        assert client.region == "us-west-2"
        mock_session.client.assert_called_with("ce", region_name="us-west-2")

    @patch("boto3.Session")
    def test_validate_permissions_success(self, mock_session_class):
        """Test successful permission validation."""
        mock_session = Mock()
        mock_client = Mock()
        mock_client.get_cost_and_usage.return_value = {"ResultsByTime": []}
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        client = AWSCostClient()
        assert client.validate_permissions() is True

    @patch("boto3.Session")
    def test_validate_permissions_access_denied(self, mock_session_class):
        """Test permission validation with access denied."""
        mock_session = Mock()
        mock_client = Mock()
        error_response = {"Error": {"Code": "AccessDenied", "Message": "Access denied"}}
        mock_client.get_cost_and_usage.side_effect = ClientError(
            error_response, "GetCostAndUsage"
        )
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        client = AWSCostClient()
        assert client.validate_permissions() is False

    @patch("boto3.Session")
    def test_get_cost_and_usage_success(self, mock_session_class):
        """Test successful cost and usage retrieval."""
        mock_session = Mock()
        mock_client = Mock()

        # Mock API response
        mock_response = {
            "ResultsByTime": [
                {
                    "TimePeriod": {"Start": "2024-01-01", "End": "2024-01-02"},
                    "Total": {"BlendedCost": {"Amount": "10.50", "Unit": "USD"}},
                    "Groups": [],
                    "Estimated": False,
                }
            ]
        }
        mock_client.get_cost_and_usage.return_value = mock_response
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        client = AWSCostClient()

        # Create query parameters
        time_period = TimePeriod(start=datetime(2024, 1, 1), end=datetime(2024, 1, 2))
        params = QueryParameters(
            time_period=time_period,
            granularity=TimePeriodGranularity.DAILY,
            metrics=[MetricType.BLENDED_COST],
        )

        cost_data = client.get_cost_and_usage(params)

        assert len(cost_data.results) == 1
        assert cost_data.results[0].total.amount == 10.50
        assert cost_data.results[0].total.unit == "USD"
        assert cost_data.total_cost.amount == 10.50

    @patch("boto3.Session")
    def test_get_cost_and_usage_access_denied(self, mock_session_class):
        """Test cost retrieval with access denied error."""
        mock_session = Mock()
        mock_client = Mock()
        error_response = {"Error": {"Code": "AccessDenied", "Message": "Access denied"}}
        mock_client.get_cost_and_usage.side_effect = ClientError(
            error_response, "GetCostAndUsage"
        )
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        client = AWSCostClient()
        params = QueryParameters()

        with pytest.raises(PermissionError, match="Access denied to Cost Explorer API"):
            client.get_cost_and_usage(params)

    @patch("boto3.Session")
    def test_get_cost_and_usage_throttling(self, mock_session_class):
        """Test cost retrieval with throttling error."""
        mock_session = Mock()
        mock_client = Mock()
        error_response = {
            "Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}
        }
        mock_client.get_cost_and_usage.side_effect = ClientError(
            error_response, "GetCostAndUsage"
        )
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        client = AWSCostClient()
        params = QueryParameters()

        with pytest.raises(RuntimeError, match="AWS API rate limit exceeded"):
            client.get_cost_and_usage(params)

    @patch("boto3.Session")
    def test_get_dimension_values_success(self, mock_session_class):
        """Test successful dimension values retrieval."""
        mock_session = Mock()
        mock_client = Mock()
        mock_response = {
            "DimensionValues": [
                {"Value": "Amazon Elastic Compute Cloud - Compute"},
                {"Value": "Amazon Simple Storage Service"},
                {"Value": "Amazon Relational Database Service"},
            ]
        }
        mock_client.get_dimension_values.return_value = mock_response
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        client = AWSCostClient()
        values = client.get_dimension_values("SERVICE")

        expected_values = [
            "Amazon Elastic Compute Cloud - Compute",
            "Amazon Simple Storage Service",
            "Amazon Relational Database Service",
        ]
        assert values == expected_values

    @patch("boto3.Session")
    def test_build_cost_request_with_service_filter(self, mock_session_class):
        """Test building cost request with service filter."""
        mock_session = Mock()
        mock_session.client.return_value = Mock()
        mock_session_class.return_value = mock_session

        client = AWSCostClient()

        time_period = TimePeriod(start=datetime(2024, 1, 1), end=datetime(2024, 1, 31))
        params = QueryParameters(
            service="Amazon Elastic Compute Cloud - Compute",
            time_period=time_period,
            granularity=TimePeriodGranularity.MONTHLY,
            metrics=[MetricType.BLENDED_COST],
        )

        request = client._build_cost_request(params)

        assert request["TimePeriod"]["Start"] == "2024-01-01"
        assert request["TimePeriod"]["End"] == "2024-01-31"
        assert request["Granularity"] == "MONTHLY"
        assert request["Metrics"] == ["BlendedCost"]
        assert request["Filter"]["Dimensions"]["Key"] == "SERVICE"
        assert request["Filter"]["Dimensions"]["Values"] == [
            "Amazon Elastic Compute Cloud - Compute"
        ]

    @patch("boto3.Session")
    def test_build_cost_request_with_group_by(self, mock_session_class):
        """Test building cost request with group by."""
        mock_session = Mock()
        mock_session.client.return_value = Mock()
        mock_session_class.return_value = mock_session

        client = AWSCostClient()

        params = QueryParameters(
            group_by=["SERVICE", "INSTANCE_TYPE"],
            granularity=TimePeriodGranularity.DAILY,
            metrics=[MetricType.UNBLENDED_COST],
        )

        request = client._build_cost_request(params)

        assert request["GroupBy"] == [
            {"Type": "DIMENSION", "Key": "SERVICE"},
            {"Type": "DIMENSION", "Key": "INSTANCE_TYPE"},
        ]

    @patch("boto3.Session")
    def test_get_cost_and_usage_with_retry_success(self, mock_session_class):
        """Test cost retrieval with retry on throttling."""
        mock_session = Mock()
        mock_client = Mock()

        # First call fails with throttling, second succeeds
        error_response = {
            "Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}
        }
        success_response = {
            "ResultsByTime": [
                {
                    "TimePeriod": {"Start": "2024-01-01", "End": "2024-01-02"},
                    "Total": {"BlendedCost": {"Amount": "5.25", "Unit": "USD"}},
                    "Groups": [],
                    "Estimated": False,
                }
            ]
        }

        mock_client.get_cost_and_usage.side_effect = [
            ClientError(error_response, "GetCostAndUsage"),
            success_response,
        ]
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        client = AWSCostClient()
        params = QueryParameters()

        with patch("time.sleep"):  # Mock sleep to speed up test
            cost_data = client.get_cost_and_usage(params)

        assert len(cost_data.results) == 1
        assert cost_data.results[0].total.amount == 5.25
        assert mock_client.get_cost_and_usage.call_count == 2

    @patch("boto3.Session")
    def test_get_cost_and_usage_retry_exhausted(self, mock_session_class):
        """Test cost retrieval when retries are exhausted."""
        mock_session = Mock()
        mock_client = Mock()

        error_response = {
            "Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}
        }
        mock_client.get_cost_and_usage.side_effect = ClientError(
            error_response, "GetCostAndUsage"
        )
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        client = AWSCostClient()
        params = QueryParameters()

        with patch("time.sleep"):  # Mock sleep to speed up test
            with pytest.raises(
                RuntimeError, match="AWS API rate limit exceeded after multiple retries"
            ):
                client.get_cost_and_usage(params)

        assert mock_client.get_cost_and_usage.call_count == 3  # max_retries

    @patch("boto3.Session")
    def test_check_service_availability_healthy(self, mock_session_class):
        """Test service availability check when healthy."""
        mock_session = Mock()
        mock_client = Mock()
        mock_client.get_cost_and_usage.return_value = {"ResultsByTime": []}
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        client = AWSCostClient()

        with patch("time.time", side_effect=[0, 0.1]):  # Mock timing
            status = client.check_service_availability()

        assert status["available"] is True
        assert status["status"] == "healthy"
        assert status["response_time_ms"] == 100.0

    @patch("boto3.Session")
    def test_check_service_availability_access_denied(self, mock_session_class):
        """Test service availability check with access denied."""
        mock_session = Mock()
        mock_client = Mock()
        error_response = {"Error": {"Code": "AccessDenied", "Message": "Access denied"}}
        mock_client.get_cost_and_usage.side_effect = ClientError(
            error_response, "GetCostAndUsage"
        )
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        client = AWSCostClient()
        status = client.check_service_availability()

        assert status["available"] is False
        assert status["status"] == "access_denied"
        assert "Insufficient permissions" in status["error"]

    @patch("boto3.Session")
    def test_check_service_availability_throttled(self, mock_session_class):
        """Test service availability check when throttled."""
        mock_session = Mock()
        mock_client = Mock()
        error_response = {
            "Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}
        }
        mock_client.get_cost_and_usage.side_effect = ClientError(
            error_response, "GetCostAndUsage"
        )
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        client = AWSCostClient()
        status = client.check_service_availability()

        assert status["available"] is True
        assert status["status"] == "throttled"
        assert "rate limit" in status["error"]

    @patch("boto3.Session")
    def test_get_supported_services_success(self, mock_session_class):
        """Test getting supported services."""
        mock_session = Mock()
        mock_client = Mock()
        mock_response = {
            "DimensionValues": [
                {"Value": "Amazon Simple Storage Service"},
                {"Value": "Amazon Elastic Compute Cloud - Compute"},
                {"Value": "Amazon Relational Database Service"},
            ]
        }
        mock_client.get_dimension_values.return_value = mock_response
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        client = AWSCostClient()
        services = client.get_supported_services()

        expected_services = [
            "Amazon Elastic Compute Cloud - Compute",
            "Amazon Relational Database Service",
            "Amazon Simple Storage Service",
        ]
        assert services == expected_services

    @patch("boto3.Session")
    def test_estimate_query_cost_low_complexity(self, mock_session_class):
        """Test query complexity estimation for low complexity query."""
        mock_session = Mock()
        mock_session.client.return_value = Mock()
        mock_session_class.return_value = mock_session

        client = AWSCostClient()

        # Simple query - last 7 days, monthly granularity, no grouping
        time_period = TimePeriod(
            start=datetime.now() - timedelta(days=7), end=datetime.now()
        )
        params = QueryParameters(
            time_period=time_period,
            granularity=TimePeriodGranularity.MONTHLY,
            metrics=[MetricType.BLENDED_COST],
        )

        estimate = client.estimate_query_cost(params)

        assert estimate["complexity_level"] == "low"
        assert estimate["complexity_score"] <= 2
        assert estimate["estimated_response_time"] == "< 5 seconds"

    @patch("boto3.Session")
    def test_estimate_query_cost_high_complexity(self, mock_session_class):
        """Test query complexity estimation for high complexity query."""
        mock_session = Mock()
        mock_session.client.return_value = Mock()
        mock_session_class.return_value = mock_session

        client = AWSCostClient()

        # Complex query - 2 years, hourly granularity, multiple groupings
        time_period = TimePeriod(
            start=datetime.now() - timedelta(days=730), end=datetime.now()
        )
        params = QueryParameters(
            time_period=time_period,
            granularity=TimePeriodGranularity.HOURLY,
            metrics=[MetricType.BLENDED_COST],
            group_by=["SERVICE", "INSTANCE_TYPE", "REGION"],
        )

        estimate = client.estimate_query_cost(params)

        assert estimate["complexity_level"] == "high"
        assert estimate["complexity_score"] > 5
        assert len(estimate["warnings"]) > 0
        assert estimate["estimated_response_time"] == "15-60 seconds"

    @patch("boto3.Session")
    def test_parse_cost_response_with_groups(self, mock_session_class):
        """Test parsing cost response with grouped data."""
        mock_session = Mock()
        mock_session.client.return_value = Mock()
        mock_session_class.return_value = mock_session

        client = AWSCostClient()

        # Mock response with grouped data
        response = {
            "ResultsByTime": [
                {
                    "TimePeriod": {"Start": "2024-01-01", "End": "2024-01-02"},
                    "Total": {"BlendedCost": {"Amount": "100.00", "Unit": "USD"}},
                    "Groups": [
                        {
                            "Keys": ["Amazon Elastic Compute Cloud - Compute"],
                            "Metrics": {
                                "BlendedCost": {"Amount": "75.00", "Unit": "USD"}
                            },
                        },
                        {
                            "Keys": ["Amazon Simple Storage Service"],
                            "Metrics": {
                                "BlendedCost": {"Amount": "25.00", "Unit": "USD"}
                            },
                        },
                    ],
                    "Estimated": False,
                }
            ]
        }

        params = QueryParameters(group_by=["SERVICE"])
        cost_data = client._parse_cost_response(response, params)

        assert len(cost_data.results) == 1
        assert cost_data.results[0].total.amount == 100.0
        assert len(cost_data.results[0].groups) == 2

        # Check first group
        ec2_group = cost_data.results[0].groups[0]
        assert ec2_group.keys == ["Amazon Elastic Compute Cloud - Compute"]
        assert ec2_group.metrics["BlendedCost"].amount == 75.0

        # Check second group
        s3_group = cost_data.results[0].groups[1]
        assert s3_group.keys == ["Amazon Simple Storage Service"]
        assert s3_group.metrics["BlendedCost"].amount == 25.0

    @patch("boto3.Session")
    def test_build_cost_request_default_time_period(self, mock_session_class):
        """Test building cost request with default time period."""
        mock_session = Mock()
        mock_session.client.return_value = Mock()
        mock_session_class.return_value = mock_session

        client = AWSCostClient()

        # Parameters without time period
        params = QueryParameters(
            granularity=TimePeriodGranularity.DAILY, metrics=[MetricType.BLENDED_COST]
        )

        request = client._build_cost_request(params)

        # Should have default 30-day period
        start_date = datetime.strptime(
            request["TimePeriod"]["Start"], "%Y-%m-%d"
        ).date()
        end_date = datetime.strptime(request["TimePeriod"]["End"], "%Y-%m-%d").date()

        assert (end_date - start_date).days == 30
        assert request["Granularity"] == "DAILY"
        assert request["Metrics"] == ["BlendedCost"]
