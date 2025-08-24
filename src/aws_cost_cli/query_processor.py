"""Query processing with LLM integration for natural language parsing."""

import json
import re
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Union
from dataclasses import asdict

from .models import QueryParameters, TimePeriod, TimePeriodGranularity, MetricType


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
            raise RuntimeError("OpenAI provider is not available")
        
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
        
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}")
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for query parsing."""
        return """You are an AWS cost analysis assistant. Parse natural language queries about AWS costs and return structured JSON parameters.

Extract these parameters from the user query:
- service: AWS service name (EC2, S3, RDS, etc.) or null if not specified
- start_date: Start date in YYYY-MM-DD format or null
- end_date: End date in YYYY-MM-DD format or null  
- granularity: DAILY, MONTHLY, or HOURLY (default: MONTHLY)
- metrics: Array of metric types like ["BlendedCost"] (default: ["BlendedCost"])
- group_by: Array of grouping dimensions like ["SERVICE"] or null

For relative dates like "last month", "this year", calculate actual dates.
For ambiguous queries, make reasonable assumptions.

Return only valid JSON in this format:
{
  "service": "EC2",
  "start_date": "2024-01-01", 
  "end_date": "2024-01-31",
  "granularity": "MONTHLY",
  "metrics": ["BlendedCost"],
  "group_by": ["SERVICE"]
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
            raise RuntimeError("Anthropic provider is not available")
        
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
        
        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {str(e)}")
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for query parsing."""
        return """You are an AWS cost analysis assistant. Parse natural language queries about AWS costs and return structured JSON parameters.

Extract these parameters from the user query:
- service: AWS service name (EC2, S3, RDS, etc.) or null if not specified
- start_date: Start date in YYYY-MM-DD format or null
- end_date: End date in YYYY-MM-DD format or null  
- granularity: DAILY, MONTHLY, or HOURLY (default: MONTHLY)
- metrics: Array of metric types like ["BlendedCost"] (default: ["BlendedCost"])
- group_by: Array of grouping dimensions like ["SERVICE"] or null

For relative dates like "last month", "this year", calculate actual dates.
For ambiguous queries, make reasonable assumptions.

Return only valid JSON in this format:
{
  "service": "EC2",
  "start_date": "2024-01-01", 
  "end_date": "2024-01-31",
  "granularity": "MONTHLY",
  "metrics": ["BlendedCost"],
  "group_by": ["SERVICE"]
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
            raise RuntimeError("Ollama provider is not available")
        
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
                raise RuntimeError(f"Ollama API error: {response.status_code}")
            
            result = response.json()
            content = result.get("response", "").strip()
            return self._parse_llm_response(content)
        
        except ImportError:
            raise ImportError("requests package is required for Ollama provider")
        except Exception as e:
            raise RuntimeError(f"Ollama API error: {str(e)}")
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for query parsing."""
        return """You are an AWS cost analysis assistant. Parse natural language queries about AWS costs and return structured JSON parameters.

Extract these parameters from the user query:
- service: AWS service name (EC2, S3, RDS, etc.) or null if not specified
- start_date: Start date in YYYY-MM-DD format or null
- end_date: End date in YYYY-MM-DD format or null  
- granularity: DAILY, MONTHLY, or HOURLY (default: MONTHLY)
- metrics: Array of metric types like ["BlendedCost"] (default: ["BlendedCost"])
- group_by: Array of grouping dimensions like ["SERVICE"] or null

For relative dates like "last month", "this year", calculate actual dates.
For ambiguous queries, make reasonable assumptions.

Return only valid JSON in this format:
{
  "service": "EC2",
  "start_date": "2024-01-01", 
  "end_date": "2024-01-31",
  "granularity": "MONTHLY",
  "metrics": ["BlendedCost"],
  "group_by": ["SERVICE"]
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
            r'\bec2\b': 'EC2',
            r'\bs3\b': 'S3', 
            r'\brds\b': 'RDS',
            r'\blambda\b': 'Lambda',
            r'\bcloudfront\b': 'CloudFront',
            r'\bvpc\b': 'VPC',
            r'\belb\b': 'ELB',
            r'\bcloudwatch\b': 'CloudWatch',
            r'\biam\b': 'IAM',
            r'\broute\s*53\b': 'Route53',
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
        today = datetime.now(timezone.utc)
        first_day_this_month = today.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        last_day_last_month = first_day_this_month - timedelta(days=1)
        first_day_last_month = last_day_last_month.replace(day=1)
        return first_day_last_month, first_day_this_month
    
    def _this_month(self) -> tuple[datetime, datetime]:
        """Get this month's date range."""
        today = datetime.now(timezone.utc)
        first_day = today.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return first_day, today
    
    def _last_year(self) -> tuple[datetime, datetime]:
        """Get last year's date range."""
        today = datetime.now(timezone.utc)
        last_year = today.year - 1
        start = datetime(last_year, 1, 1, tzinfo=timezone.utc)
        end = datetime(today.year, 1, 1, tzinfo=timezone.utc)
        return start, end
    
    def _this_year(self) -> tuple[datetime, datetime]:
        """Get this year's date range."""
        today = datetime.now(timezone.utc)
        start = datetime(today.year, 1, 1, tzinfo=timezone.utc)
        return start, today
    
    def _yesterday(self) -> tuple[datetime, datetime]:
        """Get yesterday's date range."""
        today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        yesterday = today - timedelta(days=1)
        return yesterday, today
    
    def _today(self) -> tuple[datetime, datetime]:
        """Get today's date range."""
        today = datetime.now(timezone.utc)
        start_of_day = today.replace(hour=0, minute=0, second=0, microsecond=0)
        return start_of_day, today
    
    def _last_week(self) -> tuple[datetime, datetime]:
        """Get last week's date range."""
        today = datetime.now(timezone.utc)
        days_since_monday = today.weekday()
        this_monday = today - timedelta(days=days_since_monday)
        last_monday = this_monday - timedelta(days=7)
        return last_monday, this_monday
    
    def _this_week(self) -> tuple[datetime, datetime]:
        """Get this week's date range."""
        today = datetime.now(timezone.utc)
        days_since_monday = today.weekday()
        this_monday = today - timedelta(days=days_since_monday)
        return this_monday, today


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
        # Try LLM providers first
        for provider_name, provider in self._providers.items():
            try:
                if provider.is_available():
                    result = provider.parse_query(query)
                    return self._convert_to_query_parameters(result)
            except Exception as e:
                # Log error and continue to next provider or fallback
                continue
        
        # Fall back to pattern matching
        try:
            result = self.fallback_parser.parse_query(query)
            return self._convert_to_query_parameters(result)
        except Exception as e:
            # Return default parameters if all parsing fails
            return QueryParameters()
    
    def validate_parameters(self, params: QueryParameters) -> bool:
        """
        Validate query parameters.
        
        Args:
            params: QueryParameters to validate
            
        Returns:
            True if parameters are valid, False otherwise
        """
        # Check time period validity
        if params.time_period:
            if params.time_period.start >= params.time_period.end:
                return False
        
        # Check granularity
        if hasattr(params.granularity, 'value'):
            if params.granularity.value not in [g.value for g in TimePeriodGranularity]:
                return False
        elif params.granularity not in [g.value for g in TimePeriodGranularity]:
            return False
        
        # Check metrics
        valid_metrics = [m.value for m in MetricType]
        for metric in params.metrics:
            if hasattr(metric, 'value'):
                if metric.value not in valid_metrics:
                    return False
            elif metric not in valid_metrics:
                return False
        
        return True
    
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