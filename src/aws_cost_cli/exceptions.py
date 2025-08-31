"""Custom exceptions and error handling for AWS Cost CLI."""

from typing import Optional, List


class AWSCostCLIError(Exception):
    """Base exception for AWS Cost CLI errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        suggestions: Optional[List[str]] = None,
    ):
        """
        Initialize base exception.

        Args:
            message: Error message
            error_code: Optional error code for programmatic handling
            suggestions: Optional list of suggestions to fix the error
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.suggestions = suggestions or []


class ValidationError(AWSCostCLIError):
    """Exception raised when query validation fails."""

    def __init__(self, message: str, validation_errors: Optional[List[str]] = None):
        suggestions = [
            "Check your query syntax and parameters",
            "Ensure date ranges are valid and not too large",
            "Verify service names are correct",
            "Use supported metric types and dimensions",
        ]
        super().__init__(message, "VALIDATION_ERROR", suggestions)
        self.validation_errors = validation_errors or []


class AWSCredentialsError(AWSCostCLIError):
    """Exception raised when AWS credentials are invalid or missing."""

    def __init__(
        self,
        message: str = "AWS credentials not found or invalid",
        profile: Optional[str] = None,
    ):
        suggestions = [
            "Run 'aws configure' to set up credentials",
            "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables",
            "Use IAM roles if running on EC2",
            "Check that your AWS profile exists and is configured correctly",
        ]

        if profile:
            message = f"AWS credentials for profile '{profile}' not found or invalid"
            suggestions.append(
                f"Run 'aws configure --profile {profile}' to set up the profile"
            )

        super().__init__(message, "CREDENTIALS_ERROR", suggestions)
        self.profile = profile


class AWSPermissionsError(AWSCostCLIError):
    """Exception raised when AWS permissions are insufficient."""

    def __init__(
        self,
        message: str = "Insufficient AWS permissions",
        required_permissions: Optional[List[str]] = None,
    ):
        self.required_permissions = required_permissions or [
            "ce:GetCostAndUsage",
            "ce:GetDimensionValues",
            "ce:GetUsageReport",
        ]

        suggestions = [
            "Ensure your AWS user/role has the required Cost Explorer permissions",
            "Check that Cost Explorer is enabled in your AWS account",
            "Verify you have access to billing information",
            f"Required permissions: {', '.join(self.required_permissions)}",
        ]

        super().__init__(message, "PERMISSIONS_ERROR", suggestions)


class AWSAPIError(AWSCostCLIError):
    """Exception raised when AWS API calls fail."""

    def __init__(
        self,
        message: str,
        aws_error_code: Optional[str] = None,
        aws_error_message: Optional[str] = None,
    ):
        self.aws_error_code = aws_error_code
        self.aws_error_message = aws_error_message

        suggestions = []

        if aws_error_code == "ThrottlingException":
            suggestions = [
                "Wait a few minutes and try again",
                "Consider using cached data with the --fresh flag omitted",
                "Reduce the complexity of your query (shorter time periods, fewer dimensions)",
            ]
        elif aws_error_code == "InvalidParameterValue":
            suggestions = [
                "Check that your query parameters are valid",
                "Verify service names are correct (e.g., 'EC2', 'S3', 'RDS')",
                "Ensure date ranges are valid and not in the future",
            ]
        elif aws_error_code == "ServiceUnavailable":
            suggestions = [
                "AWS Cost Explorer service is temporarily unavailable",
                "Try again in a few minutes",
                "Check AWS service health dashboard",
            ]
        else:
            suggestions = [
                "Check your AWS credentials and permissions",
                "Verify your query parameters are correct",
                "Try a simpler query to test connectivity",
            ]

        super().__init__(message, "AWS_API_ERROR", suggestions)


class NetworkError(AWSCostCLIError):
    """Exception raised when network connectivity issues occur."""

    def __init__(self, message: str = "Network connectivity error"):
        suggestions = [
            "Check your internet connection",
            "Verify you can reach AWS services",
            "Check if you're behind a corporate firewall or proxy",
            "Try again in a few minutes",
        ]

        super().__init__(message, "NETWORK_ERROR", suggestions)


class QueryParsingError(AWSCostCLIError):
    """Exception raised when query parsing fails."""

    def __init__(
        self,
        message: str = "Failed to parse natural language query",
        original_query: Optional[str] = None,
    ):
        self.original_query = original_query

        suggestions = [
            "Try rephrasing your query more clearly",
            "Use specific service names (e.g., 'EC2', 'S3', 'RDS')",
            "Include clear time periods (e.g., 'last month', 'this year')",
            "Examples: 'How much did I spend on EC2 last month?', 'What are my total AWS costs this year?'",
        ]

        if original_query:
            message = f"Failed to parse query: '{original_query}'"

        super().__init__(message, "QUERY_PARSING_ERROR", suggestions)


class LLMProviderError(AWSCostCLIError):
    """Exception raised when LLM provider fails."""

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        fallback_available: bool = True,
    ):
        self.provider = provider
        self.fallback_available = fallback_available

        suggestions = []

        if provider == "openai":
            suggestions = [
                "Check your OpenAI API key is valid and has credits",
                "Verify you have internet connectivity",
                "Try configuring a different LLM provider",
            ]
        elif provider == "anthropic":
            suggestions = [
                "Check your Anthropic API key is valid",
                "Verify you have internet connectivity",
                "Try configuring a different LLM provider",
            ]
        elif provider == "ollama":
            suggestions = [
                "Ensure Ollama is running locally",
                "Check that the specified model is available",
                "Verify Ollama is accessible at the configured URL",
            ]
        else:
            suggestions = [
                "Check your LLM provider configuration",
                "Verify API keys and connectivity",
                "Try using a different provider",
            ]

        if fallback_available:
            suggestions.append("Basic query parsing will be used as fallback")

        super().__init__(message, "LLM_PROVIDER_ERROR", suggestions)


class CacheError(AWSCostCLIError):
    """Exception raised when cache operations fail."""

    def __init__(self, message: str = "Cache operation failed"):
        suggestions = [
            "Check disk space and permissions",
            "Try clearing the cache with 'aws-cost-cli clear-cache'",
            "The CLI will continue without caching",
        ]

        super().__init__(message, "CACHE_ERROR", suggestions)


class ConfigurationError(AWSCostCLIError):
    """Exception raised when configuration is invalid."""

    def __init__(
        self, message: str = "Configuration error", config_file: Optional[str] = None
    ):
        self.config_file = config_file

        suggestions = [
            "Check your configuration file syntax",
            "Run 'aws-cost-cli configure' to set up configuration",
            "Verify all required fields are present",
        ]

        if config_file:
            suggestions.append(f"Configuration file: {config_file}")

        super().__init__(message, "CONFIGURATION_ERROR", suggestions)


class ParameterValidationError(AWSCostCLIError):
    """Exception raised when parameter validation fails."""

    def __init__(
        self, message: str = "Parameter validation failed", field: Optional[str] = None
    ):
        self.field = field

        suggestions = [
            "Check that all required parameters are provided",
            "Verify date formats are correct (YYYY-MM-DD)",
            "Ensure service names are valid AWS service names",
        ]

        if field:
            suggestions.append(f"Issue with field: {field}")

        super().__init__(message, "VALIDATION_ERROR", suggestions)


def format_error_message(
    error: AWSCostCLIError, include_suggestions: bool = True
) -> str:
    """
    Format an error message for display to the user.

    Args:
        error: The error to format
        include_suggestions: Whether to include suggestions in the output

    Returns:
        Formatted error message string
    """
    message = f"❌ {error.message}"

    if include_suggestions and error.suggestions:
        message += "\n\n💡 Suggestions:"
        for suggestion in error.suggestions:
            message += f"\n   • {suggestion}"

    return message


def handle_aws_client_error(
    client_error, operation: str = "AWS operation"
) -> AWSCostCLIError:
    """
    Convert boto3 ClientError to appropriate AWSCostCLIError.

    Args:
        client_error: The boto3 ClientError
        operation: Description of the operation that failed

    Returns:
        Appropriate AWSCostCLIError subclass
    """
    error_code = client_error.response.get("Error", {}).get("Code", "")
    error_message = client_error.response.get("Error", {}).get("Message", "")

    if error_code in ["AccessDenied", "UnauthorizedOperation"]:
        return AWSPermissionsError(f"Access denied during {operation}: {error_message}")
    elif error_code == "InvalidUserID.NotFound":
        return AWSCredentialsError(f"Invalid AWS credentials: {error_message}")
    elif error_code in ["ThrottlingException", "Throttling"]:
        return AWSAPIError(
            f"AWS API rate limit exceeded during {operation}",
            aws_error_code=error_code,
            aws_error_message=error_message,
        )
    elif error_code == "InvalidParameterValue":
        return AWSAPIError(
            f"Invalid parameter in {operation}: {error_message}",
            aws_error_code=error_code,
            aws_error_message=error_message,
        )
    elif error_code == "ServiceUnavailable":
        return AWSAPIError(
            f"AWS service temporarily unavailable during {operation}",
            aws_error_code=error_code,
            aws_error_message=error_message,
        )
    else:
        return AWSAPIError(
            f"AWS API error during {operation} ({error_code}): {error_message}",
            aws_error_code=error_code,
            aws_error_message=error_message,
        )


def handle_network_error(
    network_error, operation: str = "network operation"
) -> NetworkError:
    """
    Convert network-related exceptions to NetworkError.

    Args:
        network_error: The network-related exception
        operation: Description of the operation that failed

    Returns:
        NetworkError instance
    """
    return NetworkError(f"Network error during {operation}: {str(network_error)}")


def is_retryable_error(error: AWSCostCLIError) -> bool:
    """
    Determine if an error is retryable.

    Args:
        error: The error to check

    Returns:
        True if the error is retryable, False otherwise
    """
    if isinstance(error, AWSAPIError):
        return error.aws_error_code in [
            "ThrottlingException",
            "ServiceUnavailable",
            "Throttling",
        ]
    elif isinstance(error, NetworkError):
        return True
    else:
        return False


def get_retry_delay(
    attempt: int, base_delay: float = 1.0, max_delay: float = 60.0
) -> float:
    """
    Calculate retry delay using exponential backoff with jitter.

    Args:
        attempt: Current attempt number (0-based)
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds

    Returns:
        Delay in seconds
    """
    import random

    delay = min(base_delay * (2**attempt), max_delay)
    # Add jitter to avoid thundering herd
    jitter = random.uniform(0, delay * 0.1)
    return delay + jitter
