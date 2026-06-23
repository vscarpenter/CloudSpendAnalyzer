"""Query processing with LLM integration for natural language parsing."""

import json
import re
import time
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path

from .models import QueryParameters, TimePeriod, TimePeriodGranularity, MetricType
from .date_utils import DateRangeCalculator, Quarter
from .exceptions import (
    LLMProviderError,
    QueryParsingError,
    NetworkError,
    ValidationError,
    ParameterValidationError,
)


@dataclass
class ProviderMetrics:
    """Performance metrics for a provider."""
    
    provider_name: str
    request_count: int = 0
    success_count: int = 0
    error_count: int = 0
    total_response_time: float = 0.0
    min_response_time: Optional[float] = None
    max_response_time: Optional[float] = None
    last_success: Optional[datetime] = None
    last_error: Optional[datetime] = None
    last_error_message: Optional[str] = None
    consecutive_errors: int = 0
    health_status: str = "unknown"  # healthy, degraded, unhealthy, unknown
    timeout_count: int = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.request_count == 0:
            return 0.0
        return (self.success_count / self.request_count) * 100
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate as percentage."""
        if self.request_count == 0:
            return 0.0
        return (self.error_count / self.request_count) * 100
    
    @property
    def average_response_time(self) -> float:
        """Calculate average response time in milliseconds."""
        if self.success_count == 0:
            return 0.0
        return self.total_response_time / self.success_count


@dataclass
class ProviderHealthCheck:
    """Health check result for a provider."""
    
    provider_name: str
    is_healthy: bool
    response_time_ms: Optional[float]
    error_message: Optional[str]
    checked_at: datetime
    availability_status: str  # available, unavailable, timeout, error


class ProviderPerformanceMonitor:
    """Monitor performance and reliability of LLM providers."""
    
    def __init__(self, metrics_file: Optional[str] = None):
        """
        Initialize provider performance monitor.
        
        Args:
            metrics_file: Optional file to persist metrics
        """
        self.metrics_file = metrics_file
        self.provider_metrics: Dict[str, ProviderMetrics] = {}
        self._lock = threading.Lock()
        
        # Load existing metrics if file exists
        if self.metrics_file and Path(self.metrics_file).exists():
            self._load_metrics()
    
    def record_request_start(self, provider_name: str) -> float:
        """Record the start of a provider request."""
        return time.time()
    
    def record_request_success(self, provider_name: str, start_time: float, response_time_ms: Optional[float] = None):
        """Record a successful provider request."""
        if response_time_ms is None:
            response_time_ms = (time.time() - start_time) * 1000
        
        with self._lock:
            if provider_name not in self.provider_metrics:
                self.provider_metrics[provider_name] = ProviderMetrics(provider_name=provider_name)
            
            metrics = self.provider_metrics[provider_name]
            metrics.request_count += 1
            metrics.success_count += 1
            metrics.total_response_time += response_time_ms
            metrics.last_success = datetime.now()
            metrics.consecutive_errors = 0
            
            # Update min/max response times
            if metrics.min_response_time is None or response_time_ms < metrics.min_response_time:
                metrics.min_response_time = response_time_ms
            if metrics.max_response_time is None or response_time_ms > metrics.max_response_time:
                metrics.max_response_time = response_time_ms
            
            # Update health status
            self._update_health_status(provider_name)
            
            # Persist metrics
            if self.metrics_file:
                self._persist_metrics()
    
    def record_request_error(self, provider_name: str, start_time: float, error_message: str, is_timeout: bool = False):
        """Record a failed provider request."""
        response_time_ms = (time.time() - start_time) * 1000
        
        with self._lock:
            if provider_name not in self.provider_metrics:
                self.provider_metrics[provider_name] = ProviderMetrics(provider_name=provider_name)
            
            metrics = self.provider_metrics[provider_name]
            metrics.request_count += 1
            metrics.error_count += 1
            metrics.last_error = datetime.now()
            metrics.last_error_message = error_message
            metrics.consecutive_errors += 1
            
            if is_timeout:
                metrics.timeout_count += 1
            
            # Update health status
            self._update_health_status(provider_name)
            
            # Persist metrics
            if self.metrics_file:
                self._persist_metrics()
    
    def _update_health_status(self, provider_name: str):
        """Update health status based on recent performance."""
        metrics = self.provider_metrics[provider_name]
        
        # Determine health status based on recent performance
        if metrics.consecutive_errors >= 5:
            metrics.health_status = "unhealthy"
        elif metrics.consecutive_errors >= 2 or metrics.error_rate > 20:
            metrics.health_status = "degraded"
        elif metrics.success_count > 0 and metrics.error_rate < 5:
            metrics.health_status = "healthy"
        else:
            metrics.health_status = "unknown"
    
    def get_provider_metrics(self, provider_name: str) -> Optional[ProviderMetrics]:
        """Get metrics for a specific provider."""
        with self._lock:
            return self.provider_metrics.get(provider_name)
    
    def get_all_provider_metrics(self) -> Dict[str, ProviderMetrics]:
        """Get metrics for all providers."""
        with self._lock:
            return self.provider_metrics.copy()
    
    def check_provider_health(self, provider: 'LLMProvider', timeout: float = 10.0) -> ProviderHealthCheck:
        """
        Perform a health check on a provider.
        
        Args:
            provider: LLM provider instance
            timeout: Timeout for health check in seconds
            
        Returns:
            ProviderHealthCheck result
        """
        provider_name = provider.__class__.__name__.replace('Provider', '').lower()
        start_time = time.time()
        
        try:
            # Simple health check query
            test_query = "What is the total cost?"
            
            # Set a timeout for the health check
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Health check timed out")
            
            # Set timeout (only works on Unix systems)
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(timeout))
                
                # Perform the health check
                if provider.is_available():
                    provider.parse_query(test_query)
                    response_time_ms = (time.time() - start_time) * 1000
                    
                    return ProviderHealthCheck(
                        provider_name=provider_name,
                        is_healthy=True,
                        response_time_ms=response_time_ms,
                        error_message=None,
                        checked_at=datetime.now(),
                        availability_status="available"
                    )
                else:
                    return ProviderHealthCheck(
                        provider_name=provider_name,
                        is_healthy=False,
                        response_time_ms=None,
                        error_message="Provider not available",
                        checked_at=datetime.now(),
                        availability_status="unavailable"
                    )
                    
            except TimeoutError:
                return ProviderHealthCheck(
                    provider_name=provider_name,
                    is_healthy=False,
                    response_time_ms=None,
                    error_message=f"Health check timed out after {timeout}s",
                    checked_at=datetime.now(),
                    availability_status="timeout"
                )
            finally:
                # Cancel the alarm
                try:
                    signal.alarm(0)
                except:
                    pass
                    
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            return ProviderHealthCheck(
                provider_name=provider_name,
                is_healthy=False,
                response_time_ms=response_time_ms,
                error_message=str(e),
                checked_at=datetime.now(),
                availability_status="error"
            )
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for all providers."""
        with self._lock:
            summary = {
                "period_hours": hours,
                "providers": {},
                "overall": {
                    "total_requests": 0,
                    "total_successes": 0,
                    "total_errors": 0,
                    "average_success_rate": 0.0,
                    "healthy_providers": 0,
                    "degraded_providers": 0,
                    "unhealthy_providers": 0
                }
            }
            
            total_success_rates = []
            
            for provider_name, metrics in self.provider_metrics.items():
                provider_summary = {
                    "request_count": metrics.request_count,
                    "success_count": metrics.success_count,
                    "error_count": metrics.error_count,
                    "success_rate": metrics.success_rate,
                    "error_rate": metrics.error_rate,
                    "average_response_time_ms": metrics.average_response_time,
                    "min_response_time_ms": metrics.min_response_time,
                    "max_response_time_ms": metrics.max_response_time,
                    "health_status": metrics.health_status,
                    "consecutive_errors": metrics.consecutive_errors,
                    "timeout_count": metrics.timeout_count,
                    "last_success": metrics.last_success.isoformat() if metrics.last_success else None,
                    "last_error": metrics.last_error.isoformat() if metrics.last_error else None,
                    "last_error_message": metrics.last_error_message
                }
                
                summary["providers"][provider_name] = provider_summary
                
                # Update overall stats
                summary["overall"]["total_requests"] += metrics.request_count
                summary["overall"]["total_successes"] += metrics.success_count
                summary["overall"]["total_errors"] += metrics.error_count
                
                if metrics.request_count > 0:
                    total_success_rates.append(metrics.success_rate)
                
                # Count health statuses
                if metrics.health_status == "healthy":
                    summary["overall"]["healthy_providers"] += 1
                elif metrics.health_status == "degraded":
                    summary["overall"]["degraded_providers"] += 1
                elif metrics.health_status == "unhealthy":
                    summary["overall"]["unhealthy_providers"] += 1
            
            # Calculate overall average success rate
            if total_success_rates:
                summary["overall"]["average_success_rate"] = sum(total_success_rates) / len(total_success_rates)
            
            return summary
    
    def reset_metrics(self, provider_name: Optional[str] = None):
        """Reset metrics for a specific provider or all providers."""
        with self._lock:
            if provider_name:
                if provider_name in self.provider_metrics:
                    self.provider_metrics[provider_name] = ProviderMetrics(provider_name=provider_name)
            else:
                self.provider_metrics.clear()
            
            if self.metrics_file:
                self._persist_metrics()
    
    def _load_metrics(self):
        """Load metrics from file."""
        try:
            with open(self.metrics_file, 'r') as f:
                data = json.load(f)
                for provider_name, metrics_data in data.items():
                    # Convert datetime strings back to datetime objects
                    if metrics_data.get('last_success'):
                        metrics_data['last_success'] = datetime.fromisoformat(metrics_data['last_success'])
                    if metrics_data.get('last_error'):
                        metrics_data['last_error'] = datetime.fromisoformat(metrics_data['last_error'])
                    
                    self.provider_metrics[provider_name] = ProviderMetrics(**metrics_data)
        except (FileNotFoundError, json.JSONDecodeError, TypeError):
            # If loading fails, start with empty metrics
            pass
    
    def _persist_metrics(self):
        """Persist metrics to file."""
        try:
            # Convert metrics to serializable format
            data = {}
            for provider_name, metrics in self.provider_metrics.items():
                metrics_dict = asdict(metrics)
                # Convert datetime objects to ISO strings
                if metrics_dict.get('last_success'):
                    metrics_dict['last_success'] = metrics_dict['last_success'].isoformat()
                if metrics_dict.get('last_error'):
                    metrics_dict['last_error'] = metrics_dict['last_error'].isoformat()
                data[provider_name] = metrics_dict
            
            # Ensure directory exists
            if self.metrics_file:
                Path(self.metrics_file).parent.mkdir(parents=True, exist_ok=True)
                
                with open(self.metrics_file, 'w') as f:
                    json.dump(data, f, indent=2)
        except Exception:
            # Don't fail if metrics persistence fails
            pass


# Global performance monitor instance
_performance_monitor = None


def get_performance_monitor() -> ProviderPerformanceMonitor:
    """Get the global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        # Default metrics file location
        metrics_file = Path.home() / ".aws-cost-cli" / "provider_metrics.json"
        _performance_monitor = ProviderPerformanceMonitor(str(metrics_file))
    return _performance_monitor


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, timeout: Optional[float] = None):
        """
        Initialize LLM provider.
        
        Args:
            timeout: Request timeout in seconds (default: 30)
        """
        self.timeout = timeout or 30.0
        self._performance_monitor = get_performance_monitor()

    @abstractmethod
    def parse_query(self, query: str) -> Dict[str, Any]:
        """Parse natural language query and return structured parameters."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM provider is available and configured."""
        pass
    
    def parse_query_with_monitoring(self, query: str) -> Dict[str, Any]:
        """Parse query with performance monitoring."""
        provider_name = self.__class__.__name__.replace('Provider', '').lower()
        start_time = self._performance_monitor.record_request_start(provider_name)
        
        try:
            result = self.parse_query(query)
            self._performance_monitor.record_request_success(provider_name, start_time)
            return result
        except Exception as e:
            is_timeout = "timeout" in str(e).lower() or "timed out" in str(e).lower()
            self._performance_monitor.record_request_error(provider_name, start_time, str(e), is_timeout)
            raise
    
    def get_performance_metrics(self) -> Optional[ProviderMetrics]:
        """Get performance metrics for this provider."""
        provider_name = self.__class__.__name__.replace('Provider', '').lower()
        return self._performance_monitor.get_provider_metrics(provider_name)
    
    def check_health(self) -> ProviderHealthCheck:
        """Perform a health check on this provider."""
        return self._performance_monitor.check_provider_health(self, self.timeout)

    def _get_system_prompt(self) -> str:
        """Get the system prompt for query parsing (shared by all providers)."""
        return self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """Build the canonical system prompt used across all LLM providers.

        The current date is injected at call time so the prompt always reflects
        "today" rather than a hardcoded reference date. The example dates below
        are illustrative of the expected output format only.
        """
        today = datetime.now(timezone.utc).strftime("%B %d, %Y")
        return f"""You are an AWS cost analysis assistant. Parse natural language queries about AWS costs and return structured JSON parameters.

IMPORTANT: Today's date is {today}. Use this as the reference for relative dates.

Extract these parameters from the user query:
- service: AWS service name using EXACT AWS service names (see mapping below) or null if not specified
- start_date: Start date in YYYY-MM-DD format or null
- end_date: End date in YYYY-MM-DD format or null
- granularity: DAILY, MONTHLY, or HOURLY (default: MONTHLY)
- metrics: Array of metric types like ["BlendedCost"] (default: ["BlendedCost"])
- group_by: Array of grouping dimensions like ["SERVICE"] or null
- date_range_type: QUARTER, FISCAL_YEAR, CALENDAR_YEAR, or CUSTOM (null if not specified)
- fiscal_year_start_month: Month when fiscal year starts (1-12, default: 1)
- cost_allocation_tags: Array of tag keys for cost allocation (null if not specified)

AWS Service Name Mapping (use the exact names on the right):
- S3 → "Amazon Simple Storage Service"
- EC2 → "Amazon Elastic Compute Cloud - Compute"
- RDS → "Amazon Relational Database Service"
- Lambda → "AWS Lambda"
- CloudFront → "Amazon CloudFront"
- VPC → "Amazon Virtual Private Cloud"
- Route 53 → "Amazon Route 53"
- KMS → "AWS Key Management Service"
- Secrets Manager → "AWS Secrets Manager"

For relative dates (these examples assume a reference date of August 24, 2025):
- "last month" = July 2025 (2025-07-01 to 2025-08-01)
- "this month" = August 2025 (2025-08-01 to 2025-08-24)
- "this year" = 2025 (2025-01-01 to 2025-08-24)
- "last year" = 2024 (2024-01-01 to 2025-01-01)

For specific years (IMPORTANT - use full year ranges):
- "2025" or "all of 2025" or "for 2025" = Full year 2025 (2025-01-01 to 2026-01-01)
- "2024" or "all of 2024" or "for 2024" = Full year 2024 (2024-01-01 to 2025-01-01)
- "S3 costs for 2025" = Full year 2025 (2025-01-01 to 2026-01-01)

For quarters:
- "Q1 2025" = 2025-01-01 to 2025-04-01
- "Q2 2025" = 2025-04-01 to 2025-07-01
- "Q3 2025" = 2025-07-01 to 2025-10-01
- "Q4 2025" = 2025-10-01 to 2026-01-01
- "this quarter" = Q3 2025 (2025-07-01 to 2025-10-01)
- "last quarter" = Q2 2025 (2025-04-01 to 2025-07-01)

For fiscal years (assuming January start unless specified):
- "FY2025" = 2025-01-01 to 2026-01-01
- "fiscal year 2025" = 2025-01-01 to 2026-01-01

For specific months like "july 2025" or "in july 2025":
- Use the full month range: 2025-07-01 to 2025-08-01 (end date is first day of next month)

For date ranges like "from X to Y", use the full range including both dates.

IMPORTANT: For queries asking about service breakdown or listing services, set group_by to ["SERVICE"]. This includes queries like:
- "What services did I use?"
- "List the services that cost money"
- "Show me service breakdown"
- "Which services did I spend money on?"

Return only valid JSON in this format:
{{
  "service": null,
  "start_date": "2025-07-01",
  "end_date": "2025-08-01",
  "granularity": "MONTHLY",
  "metrics": ["BlendedCost"],
  "group_by": ["SERVICE"],
  "date_range_type": "QUARTER",
  "fiscal_year_start_month": 1,
  "cost_allocation_tags": null
}}"""

    def _parse_llm_response(self, content: str) -> Dict[str, Any]:
        """Parse an LLM response string and extract the JSON object.

        Handles bare JSON, JSON wrapped in ```json ... ``` (or plain ```) code
        fences, and JSON embedded in surrounding prose. Raises QueryParsingError
        if no valid JSON object can be extracted.
        """
        cleaned = content.strip()

        # Strip markdown code fences (```json ... ``` or ``` ... ```).
        if cleaned.startswith("```"):
            fence_match = re.search(
                r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL | re.IGNORECASE
            )
            if fence_match:
                cleaned = fence_match.group(1).strip()

        # First attempt: parse the cleaned content directly.
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Fallback: extract the first balanced {...} object from the content.
        json_str = self._extract_json_object(cleaned)
        if json_str is not None:
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        raise QueryParsingError(f"Could not parse LLM response as JSON: {content}")

    @staticmethod
    def _extract_json_object(content: str) -> Optional[str]:
        """Return the first balanced top-level {...} JSON object found in content."""
        start = content.find("{")
        if start == -1:
            return None

        depth = 0
        for index in range(start, len(content)):
            char = content[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return content[start : index + 1]
        return None


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider for query parsing."""

    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", timeout: Optional[float] = None):
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key
            model: Model to use (default: gpt-3.5-turbo)
            timeout: Request timeout in seconds
        """
        super().__init__(timeout)
        self.api_key = api_key
        self.model = model
        self._client = None

    def _get_client(self):
        """Get OpenAI client, creating it if necessary."""
        if self._client is None:
            try:
                import openai

                self._client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai package is required for OpenAI provider")
        return self._client

    def is_available(self) -> bool:
        """Check if OpenAI is available and configured."""
        # An empty key cannot construct a client (the SDK raises), so short-circuit.
        if not self.api_key:
            return False
        try:
            self._get_client()
            return True
        except ImportError:
            return False

    def parse_query(self, query: str) -> Dict[str, Any]:
        """Parse query using OpenAI."""
        if not self.is_available():
            raise LLMProviderError(
                "OpenAI provider is not available", provider="openai"
            )

        client = self._get_client()

        system_prompt = self._get_system_prompt()

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                temperature=0.1,
                max_tokens=500,
            )

            content = response.choices[0].message.content.strip()
            return self._parse_llm_response(content)

        except ImportError:
            raise LLMProviderError("OpenAI package not installed", provider="openai")
        except Exception as e:
            error_msg = str(e).lower()
            if "api key" in error_msg or "authentication" in error_msg:
                raise LLMProviderError("Invalid OpenAI API key", provider="openai")
            elif "network" in error_msg or "connection" in error_msg:
                raise NetworkError(f"Network error connecting to OpenAI: {e}")
            else:
                raise LLMProviderError(f"OpenAI API error: {str(e)}", provider="openai")


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider for query parsing."""

    def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307", timeout: Optional[float] = None):
        """
        Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key
            model: Model to use (default: claude-3-haiku-20240307)
            timeout: Request timeout in seconds
        """
        super().__init__(timeout)
        self.api_key = api_key
        self.model = model
        self._client = None

    def _get_client(self):
        """Get Anthropic client, creating it if necessary."""
        if self._client is None:
            try:
                import anthropic

                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "anthropic package is required for Anthropic provider"
                )
        return self._client

    def is_available(self) -> bool:
        """Check if Anthropic is available and configured."""
        try:
            _client = self._get_client()
            return bool(self.api_key)
        except ImportError:
            return False

    def parse_query(self, query: str) -> Dict[str, Any]:
        """Parse query using Anthropic Claude."""
        if not self.is_available():
            raise LLMProviderError(
                "Anthropic provider is not available", provider="anthropic"
            )

        client = self._get_client()

        system_prompt = self._get_system_prompt()

        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=500,
                temperature=0.1,
                system=system_prompt,
                messages=[{"role": "user", "content": query}],
            )

            content = response.content[0].text.strip()
            return self._parse_llm_response(content)

        except ImportError:
            raise LLMProviderError(
                "Anthropic package not installed", provider="anthropic"
            )
        except Exception as e:
            error_msg = str(e).lower()
            if "api key" in error_msg or "authentication" in error_msg:
                raise LLMProviderError(
                    "Invalid Anthropic API key", provider="anthropic"
                )
            elif "network" in error_msg or "connection" in error_msg:
                raise NetworkError(f"Network error connecting to Anthropic: {e}")
            else:
                raise LLMProviderError(
                    f"Anthropic API error: {str(e)}", provider="anthropic"
                )


class BedrockProvider(LLMProvider):
    """AWS Bedrock provider for query parsing."""

    def __init__(
        self,
        model: str = "anthropic.claude-3-haiku-20240307-v1:0",
        region: str = "us-east-1",
        profile: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        """
        Initialize Bedrock provider.

        Args:
            model: Bedrock model ID (default: anthropic.claude-3-haiku-20240307-v1:0)
            region: AWS region for Bedrock (default: us-east-1)
            profile: AWS profile to use (optional)
            timeout: Request timeout in seconds
        """
        super().__init__(timeout)
        self.model = model
        self.region = region
        self.profile = profile
        self._client = None

    def _get_client(self):
        """Get Bedrock client, creating it if necessary."""
        if self._client is None:
            try:
                import boto3

                session = (
                    boto3.Session(profile_name=self.profile)
                    if self.profile
                    else boto3.Session()
                )
                self._client = session.client(
                    "bedrock-runtime", region_name=self.region
                )
            except ImportError:
                raise ImportError("boto3 package is required for Bedrock provider")
        return self._client

    def is_available(self) -> bool:
        """Check if Bedrock is available and configured."""
        try:
            _client = self._get_client()
            # Try a simple operation to verify credentials and permissions
            # We'll just check if we can create the client without errors
            return True
        except Exception:
            return False

    def parse_query(self, query: str) -> Dict[str, Any]:
        """Parse query using AWS Bedrock."""
        if not self.is_available():
            raise LLMProviderError(
                "Bedrock provider is not available", provider="bedrock"
            )

        client = self._get_client()

        system_prompt = self._get_system_prompt()

        try:
            # Prepare the request based on model type
            if "anthropic.claude" in self.model:
                # Claude models use the messages format
                body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 500,
                    "temperature": 0.1,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": query}],
                }
            elif "amazon.titan" in self.model:
                # Titan models use a different format
                body = {
                    "inputText": f"{system_prompt}\n\nUser query: {query}\n\nJSON response:",
                    "textGenerationConfig": {
                        "maxTokenCount": 500,
                        "temperature": 0.1,
                        "topP": 0.9,
                    },
                }
            elif "ai21.j2" in self.model:
                # Jurassic models use another format
                body = {
                    "prompt": f"{system_prompt}\n\nUser query: {query}\n\nJSON response:",
                    "maxTokens": 500,
                    "temperature": 0.1,
                    "topP": 0.9,
                }
            else:
                # Default to Claude format for unknown models
                body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 500,
                    "temperature": 0.1,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": query}],
                }

            import json as json_module

            response = client.invoke_model(
                modelId=self.model,
                body=json_module.dumps(body),
                contentType="application/json",
                accept="application/json",
            )

            response_body = json_module.loads(response["body"].read())

            # Extract content based on model type
            if "anthropic.claude" in self.model:
                content = response_body["content"][0]["text"].strip()
            elif "amazon.titan" in self.model:
                content = response_body["results"][0]["outputText"].strip()
            elif "ai21.j2" in self.model:
                content = response_body["completions"][0]["data"]["text"].strip()
            else:
                # Try to extract from common response formats
                if "content" in response_body and isinstance(
                    response_body["content"], list
                ):
                    content = response_body["content"][0]["text"].strip()
                elif "results" in response_body:
                    content = response_body["results"][0]["outputText"].strip()
                elif "completions" in response_body:
                    content = response_body["completions"][0]["data"]["text"].strip()
                else:
                    content = str(response_body).strip()

            return self._parse_llm_response(content)

        except ImportError:
            raise LLMProviderError("boto3 package not installed", provider="bedrock")
        except Exception as e:
            error_msg = str(e).lower()
            if (
                "credentials" in error_msg
                or "access" in error_msg
                or "unauthorized" in error_msg
            ):
                raise LLMProviderError(
                    "Invalid AWS credentials or insufficient permissions for Bedrock",
                    provider="bedrock",
                )
            elif "network" in error_msg or "connection" in error_msg:
                raise NetworkError(f"Network error connecting to Bedrock: {e}")
            elif "model" in error_msg and "not found" in error_msg:
                raise LLMProviderError(
                    f"Bedrock model not found: {self.model}", provider="bedrock"
                )
            else:
                raise LLMProviderError(
                    f"Bedrock API error: {str(e)}", provider="bedrock"
                )


class OllamaProvider(LLMProvider):
    """Ollama local LLM provider for query parsing."""

    def __init__(
        self, 
        model: str = "gpt-oss:20b", 
        base_url: str = "http://localhost:11434",
        timeout: int = 60,
        options: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Ollama provider.

        Args:
            model: Model to use (default: gpt-oss:20b)
            base_url: Ollama server URL (default: http://localhost:11434)
            timeout: Request timeout in seconds (default: 60)
            options: Additional generation options for the model
        """
        super().__init__(timeout)
        self.model = model
        self.base_url = base_url
        self.options = options or {}

    def is_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            import requests

            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def parse_query(self, query: str) -> Dict[str, Any]:
        """Parse query using Ollama."""
        if not self.is_available():
            raise LLMProviderError(
                "Ollama provider is not available", provider="ollama"
            )

        try:
            import requests

            system_prompt = self._get_system_prompt()
            full_prompt = f"{system_prompt}\n\nUser query: {query}\n\nJSON response:"

            # Merge default options with custom options
            generation_options = {
                "temperature": 0.1, 
                "num_predict": 1000,
                "top_k": 40,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                **self.options
            }

            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": generation_options,
                },
                timeout=self.timeout,
            )

            if response.status_code != 200:
                raise LLMProviderError(
                    f"Ollama API error: HTTP {response.status_code}", provider="ollama"
                )

            result = response.json()
            content = result.get("response", "").strip()
            return self._parse_llm_response(content)

        except ImportError:
            raise LLMProviderError(
                "requests package is required for Ollama provider", provider="ollama"
            )
        except requests.exceptions.ConnectionError:
            raise LLMProviderError("Cannot connect to Ollama server", provider="ollama")
        except requests.exceptions.Timeout:
            raise LLMProviderError("Ollama request timed out", provider="ollama")
        except Exception as e:
            raise LLMProviderError(f"Ollama API error: {str(e)}", provider="ollama")


class GeminiProvider(LLMProvider):
    """Google Gemini provider for query parsing."""

    def __init__(self, api_key: str, model: str = "gemini-1.5-flash", timeout: Optional[float] = None):
        """
        Initialize Gemini provider.

        Args:
            api_key: Gemini API key
            model: Model to use (default: gemini-1.5-flash)
            timeout: Request timeout in seconds
        """
        super().__init__(timeout)
        self.api_key = api_key
        self.model = model
        self._client = None

    def _get_client(self):
        """Get Gemini client, creating it if necessary."""
        if self._client is None:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client = genai.GenerativeModel(self.model)
            except ImportError:
                raise ImportError("google-generativeai package is required for Gemini provider")
        return self._client

    def is_available(self) -> bool:
        """Check if Gemini is available and configured."""
        try:
            _client = self._get_client()
            return bool(self.api_key)
        except ImportError:
            return False

    def parse_query(self, query: str) -> Dict[str, Any]:
        """Parse query using Google Gemini."""
        if not self.is_available():
            raise LLMProviderError(
                "Gemini provider is not available", provider="gemini"
            )

        client = self._get_client()
        system_prompt = self._get_system_prompt()
        full_prompt = f"{system_prompt}\n\nUser query: {query}\n\nJSON response:"

        try:
            response = client.generate_content(full_prompt)
            content = response.text.strip()
            return self._parse_llm_response(content)

        except ImportError:
            raise LLMProviderError("google-generativeai package not installed", provider="gemini")
        except Exception as e:
            error_msg = str(e).lower()
            if "api key" in error_msg or "authentication" in error_msg:
                raise LLMProviderError("Invalid Gemini API key", provider="gemini")
            elif "quota" in error_msg or "rate limit" in error_msg:
                raise LLMProviderError("Gemini API quota exceeded", provider="gemini")
            elif "network" in error_msg or "connection" in error_msg:
                raise NetworkError(f"Network error connecting to Gemini: {e}")
            else:
                raise LLMProviderError(f"Gemini API error: {str(e)}", provider="gemini")


class FallbackParser:
    """Fallback parser for when LLM services are unavailable."""

    def __init__(self):
        """Initialize fallback parser with pattern matching rules."""
        self.service_patterns = {
            r"\bec2\b": "Amazon Elastic Compute Cloud - Compute",
            r"\bs3\b": "Amazon Simple Storage Service",
            r"\brds\b": "Amazon Relational Database Service",
            r"\blambda\b": "AWS Lambda",
            r"\bcloudfront\b": "Amazon CloudFront",
            r"\bvpc\b": "Amazon Virtual Private Cloud",
            r"\belb\b": "Amazon Elastic Load Balancing",
            r"\bcloudwatch\b": "AmazonCloudWatch",
            r"\biam\b": "AWS Identity and Access Management",
            r"\broute\s*53\b": "Amazon Route 53",
            r"\bkms\b": "AWS Key Management Service",
            r"\bsecrets\s*manager\b": "AWS Secrets Manager",
            r"\bconfig\b": "AWS Config",
            r"\bglue\b": "AWS Glue",
            r"\bdynamodb\b": "Amazon DynamoDB",
            r"\befs\b": "Amazon Elastic File System",
            r"\bquicksight\b": "Amazon QuickSight",
            r"\bsns\b": "Amazon Simple Notification Service",
            r"\bsqs\b": "Amazon Simple Queue Service",
            r"\bsimpledb\b": "Amazon SimpleDB",
        }

        self._calc = DateRangeCalculator()

        # Relative phrases resolved against the real current date (see
        # _extract_time_period for the full resolution order).
        self.relative_time_patterns = {
            r"\blast\s+month\b": self._last_month,
            r"\bthis\s+month\b": self._this_month,
            r"\blast\s+year\b": self._last_year,
            r"\bthis\s+year\b": self._this_year,
            r"\byesterday\b": self._yesterday,
            r"\btoday\b": self._today,
            r"\blast\s+week\b": self._last_week,
            r"\bthis\s+week\b": self._this_week,
            r"\bthis\s+quarter\b": self._this_quarter,
            r"\blast\s+quarter\b": self._last_quarter,
        }

        # Month-name lookup for "<month> [year]" queries.
        self._month_numbers = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12,
        }

        self.granularity_patterns = {
            r"\bdaily\b": TimePeriodGranularity.DAILY,
            r"\bmonthly\b": TimePeriodGranularity.MONTHLY,
            r"\bhourly\b": TimePeriodGranularity.HOURLY,
            r"\bper\s+day\b": TimePeriodGranularity.DAILY,
            r"\bper\s+month\b": TimePeriodGranularity.MONTHLY,
            r"\bper\s+hour\b": TimePeriodGranularity.HOURLY,
        }

        # Patterns that indicate user wants service breakdown
        self.service_breakdown_patterns = [
            r"\bwhat\s+services?\b",
            r"\blist.*services?\b",
            r"\bwhich\s+services?\b",
            r"\bshow.*services?\b",
            r"\bservices?\s+did\s+i\s+use\b",
            r"\bservices?\s+i\s+used\b",
            r"\bbreakdown\s+by\s+service\b",
            r"\bservice\s+breakdown\b",
            r"\bper\s+service\b",
            r"\bby\s+service\b",
            r"\beach\s+service\b",
        ]

    def parse_query(self, query: str) -> Dict[str, Any]:
        """Parse query using pattern matching fallback."""
        query_lower = query.lower()

        result = {
            "service": self._extract_service(query_lower),
            "start_date": None,
            "end_date": None,
            "granularity": self._extract_granularity(query_lower),
            "metrics": ["BlendedCost"],
            "group_by": self._extract_group_by(query_lower),
            "date_range_type": self._extract_date_range_type(query_lower),
            "fiscal_year_start_month": 1,
            "cost_allocation_tags": None,
        }

        # Extract time period
        start_date, end_date = self._extract_time_period(query_lower)
        if start_date:
            result["start_date"] = start_date.strftime("%Y-%m-%d")
        if end_date:
            result["end_date"] = end_date.strftime("%Y-%m-%d")

        return result

    def _extract_service(self, query: str) -> Optional[str]:
        """Extract AWS service from query."""
        for pattern, service in self.service_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                return service
        return None

    def _extract_granularity(self, query: str) -> str:
        """Extract granularity from query."""
        for pattern, granularity in self.granularity_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                return granularity.value
        return TimePeriodGranularity.MONTHLY.value

    def _extract_group_by(self, query: str) -> Optional[List[str]]:
        """Extract group_by dimensions from query."""
        # Check if user is asking for service breakdown
        for pattern in self.service_breakdown_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return ["SERVICE"]

        # Check for other grouping patterns
        if re.search(r"\bby\s+instance\s+type\b", query, re.IGNORECASE):
            return ["INSTANCE_TYPE"]
        elif re.search(r"\bby\s+region\b", query, re.IGNORECASE):
            return ["REGION"]
        elif re.search(r"\bby\s+availability\s+zone\b", query, re.IGNORECASE):
            return ["AVAILABILITY_ZONE"]

        return None

    def _extract_time_period(
        self, query: str
    ) -> tuple[Optional[datetime], Optional[datetime]]:
        """Extract a (start, end) range from the query.

        Resolution order is most-specific-first so a bare year never swallows a
        more specific phrase (e.g. "2024-01-15" or "March 2026"):
        ISO date -> relative phrase -> quarter -> fiscal year -> month[+year]
        -> calendar year. End dates are exclusive (the Cost Explorer convention).
        """
        # 1. Explicit ISO date: "costs on 2024-01-15"
        iso = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", query)
        if iso:
            try:
                day = datetime.strptime(iso.group(1), "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                )
                return day, day + timedelta(days=1)
            except ValueError:
                pass

        # 2. Relative phrases, anchored to the real current date.
        for pattern, resolver in self.relative_time_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                return resolver()

        # 3. Quarter with explicit year: "Q1 2025", "q3 2026"
        quarter = re.search(r"\bq([1-4])\s*(\d{4})\b", query, re.IGNORECASE)
        if quarter:
            tp = self._calc.get_quarter_range(
                int(quarter.group(2)), Quarter(int(quarter.group(1)))
            )
            return tp.start, tp.end

        # 4. Fiscal year: "FY2025", "fiscal year 2025"
        fiscal = re.search(
            r"\bfy\s*(\d{4})\b|\bfiscal\s+year\s+(\d{4})\b", query, re.IGNORECASE
        )
        if fiscal:
            year = int(fiscal.group(1) or fiscal.group(2))
            tp = self._calc.get_fiscal_year_range(year)
            return tp.start, tp.end

        # 5. Month name, optionally with a year: "July 2025", "in March"
        month_range = self._match_month_year(query)
        if month_range:
            return month_range

        # 6. Bare calendar year: "2024", "for 2026"
        year_match = re.search(r"\b(20\d{2})\b", query)
        if year_match:
            tp = self._calc.get_calendar_year_range(int(year_match.group(1)))
            return tp.start, tp.end

        return None, None

    def _now(self) -> datetime:
        """Current reference time (UTC). Centralized so dates stay testable."""
        return datetime.now(timezone.utc)

    def _start_of_day(self, moment: datetime) -> datetime:
        return moment.replace(hour=0, minute=0, second=0, microsecond=0)

    def _match_month_year(
        self, query: str
    ) -> Optional[tuple[datetime, datetime]]:
        """Resolve "<month> [year]" to a full-month range (exclusive end)."""
        lowered = query.lower()
        for name, month in self._month_numbers.items():
            if re.search(rf"\b{name}\b", lowered):
                year_match = re.search(r"\b(20\d{2})\b", query)
                year = int(year_match.group(1)) if year_match else self._now().year
                start = datetime(year, month, 1, tzinfo=timezone.utc)
                if month == 12:
                    end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
                else:
                    end = datetime(year, month + 1, 1, tzinfo=timezone.utc)
                return start, end
        return None

    def _last_month(self) -> tuple[datetime, datetime]:
        """Last calendar month; exclusive end is the first of this month."""
        first_this_month = self._start_of_day(self._now()).replace(day=1)
        first_last_month = (first_this_month - timedelta(days=1)).replace(day=1)
        return first_last_month, first_this_month

    def _this_month(self) -> tuple[datetime, datetime]:
        """This month so far: first of the month through end of today (exclusive)."""
        now = self._now()
        first_day = self._start_of_day(now).replace(day=1)
        return first_day, self._start_of_day(now) + timedelta(days=1)

    def _last_year(self) -> tuple[datetime, datetime]:
        tp = self._calc.get_calendar_year_range(self._now().year - 1)
        return tp.start, tp.end

    def _this_year(self) -> tuple[datetime, datetime]:
        """This year so far: Jan 1 through end of today (exclusive)."""
        now = self._now()
        start = datetime(now.year, 1, 1, tzinfo=timezone.utc)
        return start, self._start_of_day(now) + timedelta(days=1)

    def _yesterday(self) -> tuple[datetime, datetime]:
        start_today = self._start_of_day(self._now())
        return start_today - timedelta(days=1), start_today

    def _today(self) -> tuple[datetime, datetime]:
        start_today = self._start_of_day(self._now())
        return start_today, start_today + timedelta(days=1)

    def _last_week(self) -> tuple[datetime, datetime]:
        start_today = self._start_of_day(self._now())
        this_monday = start_today - timedelta(days=start_today.weekday())
        return this_monday - timedelta(days=7), this_monday

    def _this_week(self) -> tuple[datetime, datetime]:
        start_today = self._start_of_day(self._now())
        this_monday = start_today - timedelta(days=start_today.weekday())
        return this_monday, start_today + timedelta(days=1)

    def _this_quarter(self) -> tuple[datetime, datetime]:
        """Current quarter from its start through end of today (exclusive)."""
        year, quarter = self._calc.get_current_quarter()
        tp = self._calc.get_quarter_range(year, quarter)
        return tp.start, self._start_of_day(self._now()) + timedelta(days=1)

    def _last_quarter(self) -> tuple[datetime, datetime]:
        year, quarter = self._calc.get_current_quarter()
        if quarter == Quarter.Q1:
            prev_year, prev_quarter = year - 1, Quarter.Q4
        else:
            prev_year, prev_quarter = year, Quarter(quarter.value - 1)
        tp = self._calc.get_quarter_range(prev_year, prev_quarter)
        return tp.start, tp.end

    def _extract_date_range_type(self, query: str) -> Optional[str]:
        """Extract date range type from query."""
        if re.search(r"\bq[1-4]\b", query, re.IGNORECASE):
            return "QUARTER"
        elif re.search(r"\bfy\s*\d{4}\b|\bfiscal\s+year\b", query, re.IGNORECASE):
            return "FISCAL_YEAR"
        elif re.search(r"\b\d{4}\b", query) and not re.search(
            r"\b\d{4}-\d{2}-\d{2}\b", query
        ):
            return "CALENDAR_YEAR"
        return None


class QueryParser:
    """Main query parser that coordinates LLM providers and fallback."""

    def __init__(self, llm_config: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        """
        Initialize query parser with LLM configuration.

        Args:
            llm_config: Configuration for LLM providers
            config: Full application configuration (optional)
        """
        self.llm_config = llm_config
        self.config = config or {}
        self.fallback_parser = FallbackParser()
        self._providers = {}
        self._initialize_providers()

    def _initialize_providers(self):
        """Initialize available LLM providers based on configuration."""
        from .provider_factory import ProviderFactory
        
        # Get all supported provider names
        all_providers = ProviderFactory.get_all_provider_names()
        
        # Try to initialize each provider
        for provider_name in all_providers:
            try:
                provider = ProviderFactory.create_provider(provider_name, self.llm_config)
                self._providers[provider_name] = provider
            except Exception:
                # Provider is not configured or not available, skip it
                continue

    def parse_query(self, query: str) -> QueryParameters:
        """
        Parse natural language query into structured parameters.

        Delegates to ``parse_query_with_fallback`` so the configured provider
        order (``llm_provider`` / ``fallback_providers``) is always respected
        before falling back to pattern matching.

        Args:
            query: Natural language query string

        Returns:
            QueryParameters object with extracted parameters
        """
        return self.parse_query_with_fallback(query)

    def parse_query_with_fallback(self, query: str, preferred_provider: Optional[str] = None) -> QueryParameters:
        """
        Parse natural language query with enhanced fallback strategy.

        This method tries providers in the following order:
        1. Preferred provider (if specified)
        2. Default provider (ollama)
        3. Fallback providers from configuration
        4. Any remaining initialized providers (registry order)
        5. Pattern matching fallback

        Args:
            query: Natural language query string
            preferred_provider: Preferred provider to try first (optional)

        Returns:
            QueryParameters object with extracted parameters

        Raises:
            QueryParsingError: If all parsing methods fail
            ValidationError: If parameters are invalid
        """
        if not query or not query.strip():
            raise QueryParsingError("Empty query provided")

        # Build ordered list of providers to try
        providers_to_try = []
        last_error = None

        # 1. Add preferred provider first (if specified and available)
        if preferred_provider and preferred_provider in self._providers:
            providers_to_try.append(preferred_provider)

        # 2. Add configured default provider (ollama by default)
        default_provider = self.config.get("llm_provider", "ollama")
        if default_provider not in providers_to_try and default_provider in self._providers:
            providers_to_try.append(default_provider)

        # 3. Add fallback providers from configuration
        fallback_providers = self.config.get("fallback_providers", ["ollama", "openai", "anthropic", "gemini"])
        for provider_name in fallback_providers:
            if provider_name not in providers_to_try and provider_name in self._providers:
                providers_to_try.append(provider_name)

        # 4. Add any remaining initialized providers (registry/insertion order) as a
        # last resort, so an available provider is never silently skipped.
        for provider_name in self._providers:
            if provider_name not in providers_to_try:
                providers_to_try.append(provider_name)

        # Try each provider in order
        for provider_name in providers_to_try:
            provider = self._providers[provider_name]
            try:
                if provider.is_available():
                    # Use performance monitoring
                    result = provider.parse_query_with_monitoring(query)
                    params = self._convert_to_query_parameters(result)

                    # Validate the parsed parameters
                    if self.validate_parameters(params):
                        return params
                    else:
                        # Continue to next provider if validation fails
                        continue
            except (LLMProviderError, NetworkError, QueryParsingError) as e:
                last_error = e
                continue
            except Exception as e:
                last_error = LLMProviderError(
                    f"Unexpected error in {provider_name}: {str(e)}",
                    provider=provider_name,
                )
                continue

        # 5. Fall back to pattern matching
        try:
            result = self.fallback_parser.parse_query(query)
            params = self._convert_to_query_parameters(result)

            if self.validate_parameters(params):
                return params
            else:
                raise QueryParsingError(
                    "Fallback parser produced invalid parameters", original_query=query
                )

        except Exception as fallback_error:
            # If fallback also fails, provide enhanced error reporting
            if last_error:
                # Provide more context about what was tried
                provider_list = ", ".join(providers_to_try) if providers_to_try else "none"
                enhanced_error = QueryParsingError(
                    f"Failed to parse query. Tried providers: {provider_list}. "
                    f"Last error: {str(last_error)}. Fallback error: {str(fallback_error)}",
                    original_query=query
                )
                # Preserve the original error as the cause
                enhanced_error.__cause__ = last_error
                raise enhanced_error
            else:
                raise QueryParsingError(
                    f"Failed to parse query: {str(fallback_error)}", original_query=query
                )

    def validate_parameters(self, params: QueryParameters) -> bool:
        """
        Validate query parameters.

        Args:
            params: QueryParameters to validate

        Returns:
            True if parameters are valid, False otherwise

        Raises:
            ValidationError: If parameters are invalid with specific details
        """
        try:
            # Check time period validity
            if params.time_period:
                if params.time_period.start >= params.time_period.end:
                    raise ParameterValidationError(
                        "Start date must be before end date", field="time_period"
                    )

                # Check if dates are too far in the future
                now = datetime.now(timezone.utc)
                if params.time_period.start > now:
                    raise ParameterValidationError(
                        "Start date cannot be in the future", field="start_date"
                    )

                # Check if date range is reasonable (not more than 5 years)
                max_range = timedelta(days=5 * 365)
                if (params.time_period.end - params.time_period.start) > max_range:
                    raise ParameterValidationError(
                        "Date range cannot exceed 5 years", field="time_period"
                    )

            # Check granularity
            if hasattr(params.granularity, "value"):
                granularity_value = params.granularity.value
            else:
                granularity_value = params.granularity

            valid_granularities = [g.value for g in TimePeriodGranularity]
            if granularity_value not in valid_granularities:
                raise ParameterValidationError(
                    f"Invalid granularity '{granularity_value}'. Must be one of: {', '.join(valid_granularities)}",
                    field="granularity",
                )

            # Check metrics
            if not params.metrics:
                raise ParameterValidationError(
                    "At least one metric must be specified", field="metrics"
                )

            valid_metrics = [m.value for m in MetricType]
            for metric in params.metrics:
                metric_value = metric.value if hasattr(metric, "value") else metric
                if metric_value not in valid_metrics:
                    raise ParameterValidationError(
                        f"Invalid metric '{metric_value}'. Must be one of: {', '.join(valid_metrics)}",
                        field="metrics",
                    )

            # Check group_by dimensions if specified
            if params.group_by:
                valid_dimensions = [
                    "SERVICE",
                    "INSTANCE_TYPE",
                    "USAGE_TYPE",
                    "OPERATION",
                    "AVAILABILITY_ZONE",
                    "REGION",
                ]
                for dimension in params.group_by:
                    if dimension not in valid_dimensions:
                        raise ParameterValidationError(
                            f"Invalid group_by dimension '{dimension}'. Must be one of: {', '.join(valid_dimensions)}",
                            field="group_by",
                        )

            return True

        except (ValidationError, ParameterValidationError):
            # Re-raise validation errors
            raise
        except Exception as e:
            # Convert unexpected errors to validation errors
            raise ValidationError(f"Parameter validation failed: {str(e)}")

    def _convert_to_query_parameters(self, result: Dict[str, Any]) -> QueryParameters:
        """Convert parsed result dictionary to QueryParameters object."""
        from .models import DateRangeType

        # Convert time period
        time_period = None
        if result.get("start_date") and result.get("end_date"):
            start = datetime.fromisoformat(result["start_date"]).replace(
                tzinfo=timezone.utc
            )
            end = datetime.fromisoformat(result["end_date"]).replace(
                tzinfo=timezone.utc
            )
            time_period = TimePeriod(start=start, end=end)

        # Convert granularity
        granularity = TimePeriodGranularity.MONTHLY
        if result.get("granularity"):
            try:
                granularity = TimePeriodGranularity(result["granularity"])
            except ValueError:
                pass

        # Convert metrics
        metrics = [MetricType.BLENDED_COST]
        if result.get("metrics"):
            converted_metrics = []
            for metric in result["metrics"]:
                try:
                    converted_metrics.append(MetricType(metric))
                except ValueError:
                    # Skip invalid metrics
                    pass
            if converted_metrics:
                metrics = converted_metrics

        # Convert date range type
        date_range_type = None
        if result.get("date_range_type"):
            try:
                date_range_type = DateRangeType(result["date_range_type"])
            except ValueError:
                pass

        return QueryParameters(
            service=result.get("service"),
            time_period=time_period,
            granularity=granularity,
            metrics=metrics,
            group_by=result.get("group_by"),
            date_range_type=date_range_type,
            fiscal_year_start_month=result.get("fiscal_year_start_month", 1),
            cost_allocation_tags=result.get("cost_allocation_tags"),
        )

    def get_available_providers(self) -> List[str]:
        """
        Get list of available and configured providers.

        Returns:
            List of provider names that are available and configured
        """
        available = []
        for provider_name, provider in self._providers.items():
            if provider.is_available():
                available.append(provider_name)
        return available

    def get_all_providers(self) -> Dict[str, LLMProvider]:
        """
        Get all initialized providers.

        Returns:
            Dictionary mapping provider names to provider instances
        """
        return self._providers.copy()

    def get_provider_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status information for all providers.

        Returns:
            Dictionary mapping provider names to their status information
        """
        from .provider_factory import ProviderFactory
        
        return ProviderFactory.get_provider_configuration_status(self.llm_config)
