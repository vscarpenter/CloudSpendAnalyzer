"""Query processing with LLM integration for natural language parsing."""

import json
import re
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Union
from dataclasses import asdict

from .models import QueryParameters, TimePeriod, TimePeriodGranularity, MetricType
from .exceptions import (
    LLMProviderError,
    QueryParsingError,
    NetworkError,
    ValidationError,
)


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
- date_range_type: QUARTER, FISCAL_YEAR, CALENDAR_YEAR, or CUSTOM (null if not specified)
- fiscal_year_start_month: Month when fiscal year starts (1-12, default: 1)
- trend_analysis: PERIOD_OVER_PERIOD, YEAR_OVER_YEAR, MONTH_OVER_MONTH, QUARTER_OVER_QUARTER (null if not requested)
- include_forecast: true if user asks for forecast/prediction, false otherwise
- forecast_months: Number of months to forecast (default: 3)
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

For relative dates:
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

For trend analysis queries:
- "compared to last month" → trend_analysis: "MONTH_OVER_MONTH"
- "vs last year" → trend_analysis: "YEAR_OVER_YEAR"
- "compared to last quarter" → trend_analysis: "QUARTER_OVER_QUARTER"
- "trend analysis" → trend_analysis: "PERIOD_OVER_PERIOD"

For forecast queries:
- "forecast", "predict", "projection" → include_forecast: true
- "next 6 months" → forecast_months: 6

IMPORTANT: For queries asking about service breakdown or listing services, set group_by to ["SERVICE"]. This includes queries like:
- "What services did I use?"
- "List the services that cost money"
- "Show me service breakdown"
- "Which services did I spend money on?"

Return only valid JSON in this format:
{
  "service": null,
  "start_date": "2025-07-01", 
  "end_date": "2025-08-01",
  "granularity": "MONTHLY",
  "metrics": ["BlendedCost"],
  "group_by": ["SERVICE"],
  "date_range_type": "QUARTER",
  "fiscal_year_start_month": 1,
  "trend_analysis": "MONTH_OVER_MONTH",
  "include_forecast": false,
  "forecast_months": 3,
  "cost_allocation_tags": null
}"""

    def _parse_llm_response(self, content: str) -> Dict[str, Any]:
        """Parse LLM response and extract JSON."""
        try:
            # Try to find JSON in the response
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
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
                raise ImportError(
                    "anthropic package is required for Anthropic provider"
                )
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

For trend analysis queries:
- "compared to last month" → trend_analysis: "MONTH_OVER_MONTH"
- "vs last year" → trend_analysis: "YEAR_OVER_YEAR"
- "compared to last quarter" → trend_analysis: "QUARTER_OVER_QUARTER"
- "trend analysis" → trend_analysis: "PERIOD_OVER_PERIOD"

For forecast queries:
- "forecast", "predict", "projection" → include_forecast: true
- "next 6 months" → forecast_months: 6

IMPORTANT: For queries asking about service breakdown or listing services, set group_by to ["SERVICE"]. This includes queries like:
- "What services did I use?"
- "List the services that cost money"
- "Show me service breakdown"
- "Which services did I spend money on?"

For queries asking about service breakdown, set group_by to ["SERVICE"].

Return only valid JSON in this format:
{
  "service": null,
  "start_date": "2025-07-01", 
  "end_date": "2025-08-01",
  "granularity": "MONTHLY",
  "metrics": ["BlendedCost"],
  "group_by": ["SERVICE"],
  "date_range_type": "QUARTER",
  "fiscal_year_start_month": 1,
  "trend_analysis": "MONTH_OVER_MONTH",
  "include_forecast": false,
  "forecast_months": 3,
  "cost_allocation_tags": null
}"""

    def _parse_llm_response(self, content: str) -> Dict[str, Any]:
        """Parse LLM response and extract JSON."""
        try:
            # Try to find JSON in the response
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            else:
                # If no JSON found, try parsing the entire content
                return json.loads(content)
        except json.JSONDecodeError:
            raise ValueError(f"Could not parse LLM response as JSON: {content}")


class BedrockProvider(LLMProvider):
    """AWS Bedrock provider for query parsing."""

    def __init__(
        self,
        model: str = "anthropic.claude-3-haiku-20240307-v1:0",
        region: str = "us-east-1",
        profile: Optional[str] = None,
    ):
        """
        Initialize Bedrock provider.

        Args:
            model: Bedrock model ID (default: anthropic.claude-3-haiku-20240307-v1:0)
            region: AWS region for Bedrock (default: us-east-1)
            profile: AWS profile to use (optional)
        """
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
            client = self._get_client()
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

For trend analysis queries:
- "compared to last month" → trend_analysis: "MONTH_OVER_MONTH"
- "vs last year" → trend_analysis: "YEAR_OVER_YEAR"
- "compared to last quarter" → trend_analysis: "QUARTER_OVER_QUARTER"
- "trend analysis" → trend_analysis: "PERIOD_OVER_PERIOD"

For forecast queries:
- "forecast", "predict", "projection" → include_forecast: true
- "next 6 months" → forecast_months: 6

IMPORTANT: For queries asking about service breakdown or listing services, set group_by to ["SERVICE"]. This includes queries like:
- "What services did I use?"
- "List the services that cost money"
- "Show me service breakdown"
- "Which services did I spend money on?"

For queries asking about service breakdown, set group_by to ["SERVICE"].

Return only valid JSON in this format:
{
  "service": null,
  "start_date": "2025-07-01", 
  "end_date": "2025-08-01",
  "granularity": "MONTHLY",
  "metrics": ["BlendedCost"],
  "group_by": ["SERVICE"],
  "date_range_type": "QUARTER",
  "fiscal_year_start_month": 1,
  "trend_analysis": "MONTH_OVER_MONTH",
  "include_forecast": false,
  "forecast_months": 3,
  "cost_allocation_tags": null
}"""

    def _parse_llm_response(self, content: str) -> Dict[str, Any]:
        """Parse LLM response and extract JSON."""
        try:
            # Try to find JSON in the response
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            else:
                # If no JSON found, try parsing the entire content
                return json.loads(content)
        except json.JSONDecodeError:
            raise QueryParsingError(f"Could not parse LLM response as JSON: {content}")


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
            raise LLMProviderError(
                "Ollama provider is not available", provider="ollama"
            )

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
                    "options": {"temperature": 0.1, "num_predict": 500},
                },
                timeout=30,
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

For trend analysis queries:
- "compared to last month" → trend_analysis: "MONTH_OVER_MONTH"
- "vs last year" → trend_analysis: "YEAR_OVER_YEAR"
- "compared to last quarter" → trend_analysis: "QUARTER_OVER_QUARTER"
- "trend analysis" → trend_analysis: "PERIOD_OVER_PERIOD"

For forecast queries:
- "forecast", "predict", "projection" → include_forecast: true
- "next 6 months" → forecast_months: 6

IMPORTANT: For queries asking about service breakdown or listing services, set group_by to ["SERVICE"]. This includes queries like:
- "What services did I use?"
- "List the services that cost money"
- "Show me service breakdown"
- "Which services did I spend money on?"

For queries asking about service breakdown, set group_by to ["SERVICE"].

Return only valid JSON in this format:
{
  "service": null,
  "start_date": "2025-07-01", 
  "end_date": "2025-08-01",
  "granularity": "MONTHLY",
  "metrics": ["BlendedCost"],
  "group_by": ["SERVICE"],
  "date_range_type": "QUARTER",
  "fiscal_year_start_month": 1,
  "trend_analysis": "MONTH_OVER_MONTH",
  "include_forecast": false,
  "forecast_months": 3,
  "cost_allocation_tags": null
}"""

    def _parse_llm_response(self, content: str) -> Dict[str, Any]:
        """Parse LLM response and extract JSON."""
        try:
            # Try to find JSON in the response
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
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

        self.time_patterns = {
            r"\blast\s+month\b": self._last_month,
            r"\bthis\s+month\b": self._this_month,
            r"\blast\s+year\b": self._last_year,
            r"\bthis\s+year\b": self._this_year,
            r"\byesterday\b": self._yesterday,
            r"\btoday\b": self._today,
            r"\blast\s+week\b": self._last_week,
            r"\bthis\s+week\b": self._this_week,
            r"\bjuly\s+2025\b": self._july_2025,
            r"\bin\s+july\s+2025\b": self._july_2025,
            r"\bjuly\b": self._july_current_or_last,
            # Full year patterns
            r"\b2025\b": self._full_year_2025,
            r"\bfor\s+2025\b": self._full_year_2025,
            r"\ball\s+of\s+2025\b": self._full_year_2025,
            r"\b2024\b": self._full_year_2024,
            r"\bfor\s+2024\b": self._full_year_2024,
            r"\ball\s+of\s+2024\b": self._full_year_2024,
            # Quarter patterns
            r"\bq1\s+2025\b": self._q1_2025,
            r"\bq2\s+2025\b": self._q2_2025,
            r"\bq3\s+2025\b": self._q3_2025,
            r"\bq4\s+2025\b": self._q4_2025,
            r"\bthis\s+quarter\b": self._this_quarter,
            r"\blast\s+quarter\b": self._last_quarter,
            # Fiscal year patterns
            r"\bfy\s*2025\b": self._fy_2025,
            r"\bfiscal\s+year\s+2025\b": self._fy_2025,
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

        # Patterns for trend analysis
        self.trend_patterns = {
            r"\bcompared\s+to\s+last\s+month\b": "MONTH_OVER_MONTH",
            r"\bvs\s+last\s+month\b": "MONTH_OVER_MONTH",
            r"\bcompared\s+to\s+last\s+year\b": "YEAR_OVER_YEAR",
            r"\bvs\s+last\s+year\b": "YEAR_OVER_YEAR",
            r"\bcompared\s+to\s+last\s+quarter\b": "QUARTER_OVER_QUARTER",
            r"\bvs\s+last\s+quarter\b": "QUARTER_OVER_QUARTER",
            r"\btrend\s+analysis\b": "PERIOD_OVER_PERIOD",
            r"\bperiod\s+over\s+period\b": "PERIOD_OVER_PERIOD",
        }

        # Patterns for forecasting
        self.forecast_patterns = [
            r"\bforecast\b",
            r"\bpredict\b",
            r"\bprojection\b",
            r"\bwhat\s+will\s+i\s+spend\b",
            r"\bfuture\s+costs?\b",
            r"\bnext\s+\d+\s+months?\b",
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
            "trend_analysis": self._extract_trend_analysis(query_lower),
            "include_forecast": self._extract_forecast_request(query_lower),
            "forecast_months": self._extract_forecast_months(query_lower),
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
        """Extract time period from query."""
        for pattern, time_func in self.time_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                return time_func()

        # Try to extract specific dates
        date_match = re.search(r"(\d{4}-\d{2}-\d{2})", query)
        if date_match:
            try:
                date = datetime.strptime(date_match.group(1), "%Y-%m-%d")
                return date, date + timedelta(days=1)
            except ValueError:
                pass

        return None, None

    def _last_month(self) -> tuple[datetime, datetime]:
        """Get last month's date range."""
        today = datetime(
            2025, 8, 24, tzinfo=timezone.utc
        )  # Current date: August 24, 2025
        first_day_this_month = today.replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )
        last_day_last_month = first_day_this_month - timedelta(days=1)
        first_day_last_month = last_day_last_month.replace(day=1)
        return first_day_last_month, first_day_this_month

    def _this_month(self) -> tuple[datetime, datetime]:
        """Get this month's date range."""
        today = datetime(
            2025, 8, 24, tzinfo=timezone.utc
        )  # Current date: August 24, 2025
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
        today = datetime(
            2025, 8, 24, tzinfo=timezone.utc
        )  # Current date: August 24, 2025
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        return start, today

    def _yesterday(self) -> tuple[datetime, datetime]:
        """Get yesterday's date range."""
        today = datetime(2025, 8, 24, tzinfo=timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        yesterday = today - timedelta(days=1)
        return yesterday, today

    def _today(self) -> tuple[datetime, datetime]:
        """Get today's date range."""
        today = datetime(
            2025, 8, 24, tzinfo=timezone.utc
        )  # Current date: August 24, 2025
        start_of_day = today.replace(hour=0, minute=0, second=0, microsecond=0)
        return start_of_day, today

    def _last_week(self) -> tuple[datetime, datetime]:
        """Get last week's date range."""
        today = datetime(
            2025, 8, 24, tzinfo=timezone.utc
        )  # Current date: August 24, 2025
        days_since_monday = today.weekday()
        this_monday = today - timedelta(days=days_since_monday)
        last_monday = this_monday - timedelta(days=7)
        return last_monday, this_monday

    def _this_week(self) -> tuple[datetime, datetime]:
        """Get this week's date range."""
        today = datetime(
            2025, 8, 24, tzinfo=timezone.utc
        )  # Current date: August 24, 2025
        days_since_monday = today.weekday()
        this_monday = today - timedelta(days=days_since_monday)
        return this_monday, today

    def _july_2025(self) -> tuple[datetime, datetime]:
        """Get July 2025 date range."""
        start = datetime(2025, 7, 1, tzinfo=timezone.utc)
        end = datetime(
            2025, 8, 1, tzinfo=timezone.utc
        )  # End of month is start of next month
        return start, end

    def _july_current_or_last(self) -> tuple[datetime, datetime]:
        """Get July date range (2025 since we're in 2025)."""
        return self._july_2025()

    def _q1_2025(self) -> tuple[datetime, datetime]:
        """Get Q1 2025 date range."""
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 4, 1, tzinfo=timezone.utc)
        return start, end

    def _q2_2025(self) -> tuple[datetime, datetime]:
        """Get Q2 2025 date range."""
        start = datetime(2025, 4, 1, tzinfo=timezone.utc)
        end = datetime(2025, 7, 1, tzinfo=timezone.utc)
        return start, end

    def _q3_2025(self) -> tuple[datetime, datetime]:
        """Get Q3 2025 date range."""
        start = datetime(2025, 7, 1, tzinfo=timezone.utc)
        end = datetime(2025, 10, 1, tzinfo=timezone.utc)
        return start, end

    def _q4_2025(self) -> tuple[datetime, datetime]:
        """Get Q4 2025 date range."""
        start = datetime(2025, 10, 1, tzinfo=timezone.utc)
        end = datetime(2026, 1, 1, tzinfo=timezone.utc)
        return start, end

    def _this_quarter(self) -> tuple[datetime, datetime]:
        """Get current quarter (Q3 2025)."""
        return self._q3_2025()

    def _last_quarter(self) -> tuple[datetime, datetime]:
        """Get last quarter (Q2 2025)."""
        return self._q2_2025()

    def _fy_2025(self) -> tuple[datetime, datetime]:
        """Get fiscal year 2025 (assuming January start)."""
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end = datetime(2026, 1, 1, tzinfo=timezone.utc)
        return start, end

    def _full_year_2025(self) -> tuple[datetime, datetime]:
        """Get full calendar year 2025."""
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end = datetime(2026, 1, 1, tzinfo=timezone.utc)
        return start, end

    def _full_year_2024(self) -> tuple[datetime, datetime]:
        """Get full calendar year 2024."""
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, tzinfo=timezone.utc)
        return start, end

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

    def _extract_trend_analysis(self, query: str) -> Optional[str]:
        """Extract trend analysis type from query."""
        for pattern, trend_type in self.trend_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                return trend_type
        return None

    def _extract_forecast_request(self, query: str) -> bool:
        """Check if query requests forecasting."""
        for pattern in self.forecast_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False

    def _extract_forecast_months(self, query: str) -> int:
        """Extract number of months to forecast."""
        # Look for patterns like "next 6 months"
        match = re.search(r"\bnext\s+(\d+)\s+months?\b", query, re.IGNORECASE)
        if match:
            return int(match.group(1))

        # Look for patterns like "6 month forecast"
        match = re.search(r"\b(\d+)\s+months?\s+forecast\b", query, re.IGNORECASE)
        if match:
            return int(match.group(1))

        return 3  # Default to 3 months


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
        provider_type = self.llm_config.get("provider", "openai").lower()

        if provider_type == "openai":
            api_key = self.llm_config.get("api_key")
            model = self.llm_config.get("model", "gpt-3.5-turbo")
            if api_key:
                self._providers["openai"] = OpenAIProvider(api_key, model)

        elif provider_type == "anthropic":
            api_key = self.llm_config.get("api_key")
            model = self.llm_config.get("model", "claude-3-haiku-20240307")
            if api_key:
                self._providers["anthropic"] = AnthropicProvider(api_key, model)

        elif provider_type == "bedrock":
            model = self.llm_config.get(
                "model", "anthropic.claude-3-haiku-20240307-v1:0"
            )
            region = self.llm_config.get("region", "us-east-1")
            profile = self.llm_config.get("profile")
            self._providers["bedrock"] = BedrockProvider(model, region, profile)

        elif provider_type == "ollama":
            model = self.llm_config.get("model", "llama2")
            base_url = self.llm_config.get("base_url", "http://localhost:11434")
            self._providers["ollama"] = OllamaProvider(model, base_url)

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
                last_error = LLMProviderError(
                    f"Unexpected error in {provider_name}: {str(e)}",
                    provider=provider_name,
                )
                continue

        # Fall back to pattern matching
        try:
            result = self.fallback_parser.parse_query(query)
            params = self._convert_to_query_parameters(result)

            if self.validate_parameters(params):
                return params
            else:
                raise QueryParsingError(
                    "Fallback parser produced invalid parameters", original_query=query
                )

        except Exception as e:
            # If fallback also fails, raise the last LLM error or a generic parsing error
            if last_error:
                raise last_error
            else:
                raise QueryParsingError(
                    f"All parsing methods failed: {str(e)}", original_query=query
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
                    raise ValidationError(
                        "Start date must be before end date", field="time_period"
                    )

                # Check if dates are too far in the future
                now = datetime.now(timezone.utc)
                if params.time_period.start > now:
                    raise ValidationError(
                        "Start date cannot be in the future", field="start_date"
                    )

                # Check if date range is reasonable (not more than 5 years)
                max_range = timedelta(days=5 * 365)
                if (params.time_period.end - params.time_period.start) > max_range:
                    raise ValidationError(
                        "Date range cannot exceed 5 years", field="time_period"
                    )

            # Check granularity
            if hasattr(params.granularity, "value"):
                granularity_value = params.granularity.value
            else:
                granularity_value = params.granularity

            valid_granularities = [g.value for g in TimePeriodGranularity]
            if granularity_value not in valid_granularities:
                raise ValidationError(
                    f"Invalid granularity '{granularity_value}'. Must be one of: {', '.join(valid_granularities)}",
                    field="granularity",
                )

            # Check metrics
            if not params.metrics:
                raise ValidationError(
                    "At least one metric must be specified", field="metrics"
                )

            valid_metrics = [m.value for m in MetricType]
            for metric in params.metrics:
                metric_value = metric.value if hasattr(metric, "value") else metric
                if metric_value not in valid_metrics:
                    raise ValidationError(
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
                        raise ValidationError(
                            f"Invalid group_by dimension '{dimension}'. Must be one of: {', '.join(valid_dimensions)}",
                            field="group_by",
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
        from .models import DateRangeType, TrendAnalysisType

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

        # Convert trend analysis type
        trend_analysis = None
        if result.get("trend_analysis"):
            try:
                trend_analysis = TrendAnalysisType(result["trend_analysis"])
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
            trend_analysis=trend_analysis,
            comparison_period=None,  # Will be calculated later if needed
            include_forecast=result.get("include_forecast", False),
            forecast_months=result.get("forecast_months", 3),
            cost_allocation_tags=result.get("cost_allocation_tags"),
        )
