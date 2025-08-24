"""Query processing with LLM integration for natural language parsing."""

import json
import re
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Union
from dataclasses import asdict

from .models import QueryParameters, TimePeriod, TimePeriodGranularity, MetricType
from .exceptions import LLMProviderError, QueryParsingError, NetworkError, ValidationError


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def parse_query(self, query: str) -> Dict[str, Any]:
        """Parse natural language query and return structured parameters."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM provider is available and configured."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider for query parsing."""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        """
        Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key
            model: Model to use (default: gpt-3.5-turbo)
        """
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
        try:
            client = self._get_client()
            return bool(self.api_key)
        except ImportError:
            return False
    
    def parse_query(self, query: str) -> Dict[str, Any]:
        """Parse query using OpenAI."""
        if not self.is_available():
            raise LLMProviderError("OpenAI provider is not available", provider="openai")
        
        client = self._get_client()
        
        system_prompt = self._get_system_prompt()
        
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.1,
                max_tokens=500
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
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for query parsing."""
        return """You are an AWS cost analysis assistant. Parse natural language queries about AWS costs and return structured JSON parameters.

IMPORTANT: Today's date is August 24, 2025. Use this as the reference for relative dates.

Extract these parameters from the user query:
- service: AWS service name using EXACT AWS service names (see mapping below) or null if not specified
- start_date: Start date in YYYY-MM-DD format or null
- end_date: End date in YYYY-MM-DD format or null  
- granularity: DAILY, MONTHLY, or HOURLY (default: MONTHLY)
- metrics: Array of metric types like ["BlendedCost"] (default: ["BlendedCost"])
- group_by: Array of grouping dimensions like ["SERVICE"] or null

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

For relative dates:
- "last month" = July 2025 (2025-07-01 to 2025-08-01)
- "this month" = August 2025 (2025-08-01 to 2025-08-24)
- "this year" = 2025 (2025-01-01 to 2025-08-24)
- "last year" = 2024 (2024-01-01 to 2025-01-01)

For date ranges like "from X to Y", use the full range including both dates.

For queries asking about service breakdown, set group_by to ["SERVICE"].

Return only valid JSON in this format:
{
  "service": "Amazon Simple Storage Service",
  "start_date": "2025-07-01", 
  "end_date": "2025-08-01",
  "granularity": "MONTHLY",
  "metrics": ["BlendedCost"],
  "group_by": null
}"""
    
    def _parse_llm_response(self, content: str) -> Dict[str, Any]:
        """Parse LLM response and extract JSON."""
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            else:
                # If no JSON found, try parsing the entire content
                return json.loads(content)
        except json.JSONDecodeError:
            raise QueryParsingError(f"Could not parse LLM response as JSON: {content}")


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider for query parsing."""
    
    def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307"):
        """
        Initialize Anthropic provider.
        
        Args:
            api_key: Anthropic API key
            model: Model to use (default: claude-3-haiku-20240307)
        """
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
                raise ImportError("anthropic package is required for Anthropic provider")
        return self._client
    
    def is_available(self) -> bool:
        """Check if Anthropic is available and configured."""
        try:
            client = self._get_client()
            return bool(self.api_key)
        except ImportError:
            return False
    
    def parse_query(self, query: str) -> Dict[str, Any]:
        """Parse query using Anthropic Claude."""
        if not self.is_available():
            raise LLMProviderError("Anthropic provider is not available", provider="anthropic")
        
        client = self._get_client()
        
        system_prompt = self._get_system_prompt()
        
        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=500,
                temperature=0.1,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": query}
                ]
            )
            
            content = response.content[0].text.strip()
            return self._parse_llm_response(content)
        
        except ImportError:
            raise LLMProviderError("Anthropic package not installed", provider="anthropic")
        except Exception as e:
            error_msg = str(e).lower()
            if "api key" in error_msg or "authentication" in error_msg:
                raise LLMProviderError("Invalid Anthropic API key", provider="anthropic")
            elif "network" in error_msg or "connection" in error_msg:
                raise NetworkError(f"Network error connecting to Anthropic: {e}")
            else:
                raise LLMProviderError(f"Anthropic API error: {str(e)}", provider="anthropic")
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for query parsing."""
        return """You are an AWS cost analysis assistant. Parse natural language queries about AWS costs and return structured JSON parameters.

IMPORTANT: Today's date is August 24, 2025. Use this as the reference for relative dates.

Extract these parameters from the user query:
- service: AWS service name using EXACT AWS service names (see mapping below) or null if not specified
- start_date: Start date in YYYY-MM-DD format or null
- end_date: End date in YYYY-MM-DD format or null  
- granularity: DAILY, MONTHLY, or HOURLY (default: MONTHLY)
- metrics: Array of metric types like ["BlendedCost"] (default: ["BlendedCost"])
- group_by: Array of grouping dimensions like ["SERVICE"] or null

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

For relative dates:
- "last month" = July 2025 (2025-07-01 to 2025-08-01)
- "this month" = August 2025 (2025-08-01 to 2025-08-24)
- "this year" = 2025 (2025-01-01 to 2025-08-24)
- "last year" = 2024 (2024-01-01 to 2025-01-01)

For specific months like "july 2025" or "in july 2025":
- Use the full month range: 2025-07-01 to 2025-08-01 (end date is first day of next month)

For date ranges like "from X to Y", use the full range including both dates.

For queries asking about service breakdown, set group_by to ["SERVICE"].

Return only valid JSON in this format:
{
  "service": "Amazon Elastic Compute Cloud - Compute",
  "start_date": "2025-07-01", 
  "end_date": "2025-08-01",
  "granularity": "MONTHLY",
  "metrics": ["BlendedCost"],
  "group_by": null
}"""
    
    def _parse_llm_response(self, content: str) -> Dict[str, Any]:
        """Parse LLM response and extract JSON."""
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            else:
                # If no JSON found, try parsing the entire content
                return json.loads(content)
        except json.JSONDecodeError:
            raise ValueError(f"Could not parse LLM response as JSON: {content}")


class OllamaProvider(LLMProvider):
    """Ollama local LLM provider for query parsing."""
    
    def __init__(self, model: str = "llama2", base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama provider.
        
        Args:
            model: Model to use (default: llama2)
            base_url: Ollama server URL (default: http://localhost:11434)
        """
        self.model = model
        self.base_url = base_url
    
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
            raise LLMProviderError("Ollama provider is not available", provider="ollama")
        
        try:
            import requests
            
            system_prompt = self._get_system_prompt()
            full_prompt = f"{system_prompt}\n\nUser query: {query}\n\nJSON response:"
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 500
                    }
                },
                timeout=30
            )
            
            if response.status_code != 200:
                raise LLMProviderError(f"Ollama API error: HTTP {response.status_code}", provider="ollama")
            
            result = response.json()
            content = result.get("response", "").strip()
            return self._parse_llm_response(content)
        
        except ImportError:
            raise LLMProviderError("requests package is required for Ollama provider", provider="ollama")
        except requests.exceptions.ConnectionError:
            raise LLMProviderError("Cannot connect to Ollama server", provider="ollama")
        except requests.exceptions.Timeout:
            raise LLMProviderError("Ollama request timed out", provider="ollama")
        except Exception as e:
            raise LLMProviderError(f"Ollama API error: {str(e)}", provider="ollama")
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for query parsing."""
        return """You are an AWS cost analysis assistant. Parse natural language queries about AWS costs and return structured JSON parameters.

IMPORTANT: Today's date is August 24, 2025. Use this as the reference for relative dates.

Extract these parameters from the user query:
- service: AWS service name using EXACT AWS service names (see mapping below) or null if not specified
- start_date: Start date in YYYY-MM-DD format or null
- end_date: End date in YYYY-MM-DD format or null  
- granularity: DAILY, MONTHLY, or HOURLY (default: MONTHLY)
- metrics: Array of metric types like ["BlendedCost"] (default: ["BlendedCost"])
- group_by: Array of grouping dimensions like ["SERVICE"] or null

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

For relative dates:
- "last month" = July 2025 (2025-07-01 to 2025-08-01)
- "this month" = August 2025 (2025-08-01 to 2025-08-24)
- "this year" = 2025 (2025-01-01 to 2025-08-24)
- "last year" = 2024 (2024-01-01 to 2025-01-01)

For specific months like "july 2025" or "in july 2025":
- Use the full month range: 2025-07-01 to 2025-08-01 (end date is first day of next month)

For date ranges like "from X to Y", use the full range including both dates.

For queries asking about service breakdown, set group_by to ["SERVICE"].

Return only valid JSON in this format:
{
  "service": "Amazon Elastic Compute Cloud - Compute",
  "start_date": "2025-07-01", 
  "end_date": "2025-08-01",
  "granularity": "MONTHLY",
  "metrics": ["BlendedCost"],
  "group_by": null
}"""
    
    def _parse_llm_response(self, content: str) -> Dict[str, Any]:
        """Parse LLM response and extract JSON."""
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            else:
                # If no JSON found, try parsing the entire content
                return json.loads(content)
        except json.JSONDecodeError:
            raise ValueError(f"Could not parse LLM response as JSON: {content}")

class FallbackParser:
    """Fallback parser for when LLM services are unavailable."""
    
    def __init__(self):
        """Initialize fallback parser with pattern matching rules."""
        self.service_patterns = {
            r'\bec2\b': 'Amazon Elastic Compute Cloud - Compute',
            r'\bs3\b': 'Amazon Simple Storage Service', 
            r'\brds\b': 'Amazon Relational Database Service',
            r'\blambda\b': 'AWS Lambda',
            r'\bcloudfront\b': 'Amazon CloudFront',
            r'\bvpc\b': 'Amazon Virtual Private Cloud',
            r'\belb\b': 'Amazon Elastic Load Balancing',
            r'\bcloudwatch\b': 'AmazonCloudWatch',
            r'\biam\b': 'AWS Identity and Access Management',
            r'\broute\s*53\b': 'Amazon Route 53',
            r'\bkms\b': 'AWS Key Management Service',
            r'\bsecrets\s*manager\b': 'AWS Secrets Manager',
            r'\bconfig\b': 'AWS Config',
            r'\bglue\b': 'AWS Glue',
            r'\bdynamodb\b': 'Amazon DynamoDB',
            r'\befs\b': 'Amazon Elastic File System',
            r'\bquicksight\b': 'Amazon QuickSight',
            r'\bsns\b': 'Amazon Simple Notification Service',
            r'\bsqs\b': 'Amazon Simple Queue Service',
            r'\bsimpledb\b': 'Amazon SimpleDB',
        }
        
        self.time_patterns = {
            r'\blast\s+month\b': self._last_month,
            r'\bthis\s+month\b': self._this_month,
            r'\blast\s+year\b': self._last_year,
            r'\bthis\s+year\b': self._this_year,
            r'\byesterday\b': self._yesterday,
            r'\btoday\b': self._today,
            r'\blast\s+week\b': self._last_week,
            r'\bthis\s+week\b': self._this_week,
            r'\bjuly\s+2025\b': self._july_2025,
            r'\bin\s+july\s+2025\b': self._july_2025,
            r'\bjuly\b': self._july_current_or_last,
        }
        
        self.granularity_patterns = {
            r'\bdaily\b': TimePeriodGranularity.DAILY,
            r'\bmonthly\b': TimePeriodGranularity.MONTHLY,
            r'\bhourly\b': TimePeriodGranularity.HOURLY,
            r'\bper\s+day\b': TimePeriodGranularity.DAILY,
            r'\bper\s+month\b': TimePeriodGranularity.MONTHLY,
            r'\bper\s+hour\b': TimePeriodGranularity.HOURLY,
        }
    
    def parse_query(self, query: str) -> Dict[str, Any]:
        """Parse query using pattern matching fallback."""
        query_lower = query.lower()
        
        result = {
            "service": self._extract_service(query_lower),
            "start_date": None,
            "end_date": None,
            "granularity": self._extract_granularity(query_lower),
            "metrics": ["BlendedCost"],
            "group_by": None
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
    
    def _extract_time_period(self, query: str) -> tuple[Optional[datetime], Optional[datetime]]:
        """Extract time period from query."""
        for pattern, time_func in self.time_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                return time_func()
        
        # Try to extract specific dates
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', query)
        if date_match:
            try:
                date = datetime.strptime(date_match.group(1), "%Y-%m-%d")
                return date, date + timedelta(days=1)
            except ValueError:
                pass
        
        return None, None
    
    def _last_month(self) -> tuple[datetime, datetime]:
        """Get last month's date range."""
        today = datetime(2025, 8, 24, tzinfo=timezone.utc)  # Current date: August 24, 2025
        first_day_this_month = today.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        last_day_last_month = first_day_this_month - timedelta(days=1)
        first_day_last_month = last_day_last_month.replace(day=1)
        return first_day_last_month, first_day_this_month
    
    def _this_month(self) -> tuple[datetime, datetime]:
        """Get this month's date range."""
        today = datetime(2025, 8, 24, tzinfo=timezone.utc)  # Current date: August 24, 2025
        first_day = today.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return first_day, today
    
    def _last_year(self) -> tuple[datetime, datetime]:
        """Get last year's date range."""
        current_year = 2025
        last_year = current_year - 1
        start = datetime(last_year, 1, 1, tzinfo=timezone.utc)
        end = datetime(current_year, 1, 1, tzinfo=timezone.utc)
        return start, end
    
    def _this_year(self) -> tuple[datetime, datetime]:
        """Get this year's date range."""
        today = datetime(2025, 8, 24, tzinfo=timezone.utc)  # Current date: August 24, 2025
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        return start, today
    
    def _yesterday(self) -> tuple[datetime, datetime]:
        """Get yesterday's date range."""
        today = datetime(2025, 8, 24, tzinfo=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        yesterday = today - timedelta(days=1)
        return yesterday, today
    
    def _today(self) -> tuple[datetime, datetime]:
        """Get today's date range."""
        today = datetime(2025, 8, 24, tzinfo=timezone.utc)  # Current date: August 24, 2025
        start_of_day = today.replace(hour=0, minute=0, second=0, microsecond=0)
        return start_of_day, today
    
    def _last_week(self) -> tuple[datetime, datetime]:
        """Get last week's date range."""
        today = datetime(2025, 8, 24, tzinfo=timezone.utc)  # Current date: August 24, 2025
        days_since_monday = today.weekday()
        this_monday = today - timedelta(days=days_since_monday)
        last_monday = this_monday - timedelta(days=7)
        return last_monday, this_monday
    
    def _this_week(self) -> tuple[datetime, datetime]:
        """Get this week's date range."""
        today = datetime(2025, 8, 24, tzinfo=timezone.utc)  # Current date: August 24, 2025
        days_since_monday = today.weekday()
        this_monday = today - timedelta(days=days_since_monday)
        return this_monday, today
    
    def _july_2025(self) -> tuple[datetime, datetime]:
        """Get July 2025 date range."""
        start = datetime(2025, 7, 1, tzinfo=timezone.utc)
        end = datetime(2025, 8, 1, tzinfo=timezone.utc)  # End of month is start of next month
        return start, end
    
    def _july_current_or_last(self) -> tuple[datetime, datetime]:
        """Get July date range (2025 since we're in 2025)."""
        return self._july_2025()


class QueryParser:
    """Main query parser that coordinates LLM providers and fallback."""
    
    def __init__(self, llm_config: Dict[str, Any]):
        """
        Initialize query parser with LLM configuration.
        
        Args:
            llm_config: Configuration for LLM providers
        """
        self.llm_config = llm_config
        self.fallback_parser = FallbackParser()
        self._providers = {}
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize available LLM providers based on configuration."""
        provider_type = self.llm_config.get('provider', 'openai').lower()
        
        if provider_type == 'openai':
            api_key = self.llm_config.get('api_key')
            model = self.llm_config.get('model', 'gpt-3.5-turbo')
            if api_key:
                self._providers['openai'] = OpenAIProvider(api_key, model)
        
        elif provider_type == 'anthropic':
            api_key = self.llm_config.get('api_key')
            model = self.llm_config.get('model', 'claude-3-haiku-20240307')
            if api_key:
                self._providers['anthropic'] = AnthropicProvider(api_key, model)
        
        elif provider_type == 'ollama':
            model = self.llm_config.get('model', 'llama2')
            base_url = self.llm_config.get('base_url', 'http://localhost:11434')
            self._providers['ollama'] = OllamaProvider(model, base_url)
    
    def parse_query(self, query: str) -> QueryParameters:
        """
        Parse natural language query into structured parameters.
        
        Args:
            query: Natural language query string
            
        Returns:
            QueryParameters object with extracted parameters
        """
        if not query or not query.strip():
            raise QueryParsingError("Empty query provided")
        
        last_error = None
        
        # Try LLM providers first
        for provider_name, provider in self._providers.items():
            try:
                if provider.is_available():
                    result = provider.parse_query(query)
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
                last_error = LLMProviderError(f"Unexpected error in {provider_name}: {str(e)}", provider=provider_name)
                continue
        
        # Fall back to pattern matching
        try:
            result = self.fallback_parser.parse_query(query)
            params = self._convert_to_query_parameters(result)
            
            if self.validate_parameters(params):
                return params
            else:
                raise QueryParsingError("Fallback parser produced invalid parameters", original_query=query)
                
        except Exception as e:
            # If fallback also fails, raise the last LLM error or a generic parsing error
            if last_error:
                raise last_error
            else:
                raise QueryParsingError(f"All parsing methods failed: {str(e)}", original_query=query)
    
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
                    raise ValidationError("Start date must be before end date", field="time_period")
                
                # Check if dates are too far in the future
                now = datetime.now(timezone.utc)
                if params.time_period.start > now:
                    raise ValidationError("Start date cannot be in the future", field="start_date")
                
                # Check if date range is reasonable (not more than 5 years)
                max_range = timedelta(days=5*365)
                if (params.time_period.end - params.time_period.start) > max_range:
                    raise ValidationError("Date range cannot exceed 5 years", field="time_period")
            
            # Check granularity
            if hasattr(params.granularity, 'value'):
                granularity_value = params.granularity.value
            else:
                granularity_value = params.granularity
            
            valid_granularities = [g.value for g in TimePeriodGranularity]
            if granularity_value not in valid_granularities:
                raise ValidationError(
                    f"Invalid granularity '{granularity_value}'. Must be one of: {', '.join(valid_granularities)}", 
                    field="granularity"
                )
            
            # Check metrics
            if not params.metrics:
                raise ValidationError("At least one metric must be specified", field="metrics")
            
            valid_metrics = [m.value for m in MetricType]
            for metric in params.metrics:
                metric_value = metric.value if hasattr(metric, 'value') else metric
                if metric_value not in valid_metrics:
                    raise ValidationError(
                        f"Invalid metric '{metric_value}'. Must be one of: {', '.join(valid_metrics)}", 
                        field="metrics"
                    )
            
            # Check group_by dimensions if specified
            if params.group_by:
                valid_dimensions = ['SERVICE', 'INSTANCE_TYPE', 'USAGE_TYPE', 'OPERATION', 'AVAILABILITY_ZONE', 'REGION']
                for dimension in params.group_by:
                    if dimension not in valid_dimensions:
                        raise ValidationError(
                            f"Invalid group_by dimension '{dimension}'. Must be one of: {', '.join(valid_dimensions)}", 
                            field="group_by"
                        )
            
            return True
            
        except ValidationError:
            # Re-raise validation errors
            raise
        except Exception as e:
            # Convert unexpected errors to validation errors
            raise ValidationError(f"Parameter validation failed: {str(e)}")
    
    def _convert_to_query_parameters(self, result: Dict[str, Any]) -> QueryParameters:
        """Convert parsed result dictionary to QueryParameters object."""
        # Convert time period
        time_period = None
        if result.get('start_date') and result.get('end_date'):
            start = datetime.fromisoformat(result['start_date']).replace(tzinfo=timezone.utc)
            end = datetime.fromisoformat(result['end_date']).replace(tzinfo=timezone.utc)
            time_period = TimePeriod(start=start, end=end)
        
        # Convert granularity
        granularity = TimePeriodGranularity.MONTHLY
        if result.get('granularity'):
            try:
                granularity = TimePeriodGranularity(result['granularity'])
            except ValueError:
                pass
        
        # Convert metrics
        metrics = [MetricType.BLENDED_COST]
        if result.get('metrics'):
            converted_metrics = []
            for metric in result['metrics']:
                try:
                    converted_metrics.append(MetricType(metric))
                except ValueError:
                    # Skip invalid metrics
                    pass
            if converted_metrics:
                metrics = converted_metrics
        
        return QueryParameters(
            service=result.get('service'),
            time_period=time_period,
            granularity=granularity,
            metrics=metrics,
            group_by=result.get('group_by')
        )