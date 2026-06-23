"""Minimal LLM providers for query parsing."""

import json
import requests
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from config import Config
from models import QueryParameters, TimePeriod
from utils import parse_date_range


class LLMProvider(ABC):
    """Base class for LLM providers."""
    
    @abstractmethod
    def parse_query(self, query: str) -> QueryParameters:
        """Parse natural language query into parameters."""
        pass
    
    @abstractmethod
    def format_response(self, data: Dict[str, Any], query: str) -> str:
        """Format cost data into natural language response."""
        pass


class OllamaProvider(LLMProvider):
    """Local Ollama provider."""
    
    def __init__(self, config: Config):
        self.base_url = config.ollama_url
        self.model = "llama2"
    
    def parse_query(self, query: str) -> QueryParameters:
        """Parse query using Ollama."""
        prompt = f"""Extract AWS cost query parameters from this natural language query:
        "{query}"
        
        Return JSON with:
        - service: AWS service name (optional)
        - time_period: time range description
        - group_by: list of grouping dimensions (optional)
        
        Example: {{"service": "EC2", "time_period": "last month"}}
        """
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=30
            )
            response.raise_for_status()
            
            # Extract JSON from response
            result = response.json()
            text = result.get("response", "{}")
            
            # Try to parse JSON from the response
            try:
                # Find JSON in the response
                import re
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())
                else:
                    parsed = {}
            except:
                parsed = {}
            
            # Create QueryParameters
            params = QueryParameters()
            params.service = parsed.get("service")
            
            if parsed.get("time_period"):
                params.time_period = parse_date_range(parsed["time_period"])
            
            params.group_by = parsed.get("group_by", ["SERVICE"])
            
            return params
            
        except Exception as e:
            # Fallback to simple parsing
            return self._simple_parse(query)
    
    def format_response(self, data: Dict[str, Any], query: str) -> str:
        """Format response using Ollama."""
        # Convert float costs to 2 decimal places for display
        formatted_data = data.copy()
        if 'total_cost' in formatted_data:
            formatted_data['total_cost'] = round(formatted_data['total_cost'], 2)
        if 'service_breakdown' in formatted_data:
            formatted_data['service_breakdown'] = {
                k: round(v, 2) for k, v in formatted_data['service_breakdown'].items()
            }
        
        prompt = f"""Given this AWS cost data:
        {json.dumps(formatted_data, indent=2)}
        
        Answer this query in a natural, concise way: "{query}"
        Important: Format all dollar amounts with exactly 2 decimal places (e.g., $16.21 not $16.2093534761)
        """
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", str(data))
        except:
            # Fallback to simple formatting
            return self._simple_format(data)
    
    def _simple_parse(self, query: str) -> QueryParameters:
        """Simple fallback parser."""
        params = QueryParameters()
        
        # Extract service names
        services = ["EC2", "S3", "RDS", "Lambda", "CloudFront", "DynamoDB"]
        for service in services:
            if service.lower() in query.lower():
                params.service = service
                break
        
        # Parse time period
        params.time_period = parse_date_range(query)
        params.group_by = ["SERVICE"]
        
        return params
    
    def _simple_format(self, data: Dict[str, Any]) -> str:
        """Simple fallback formatter."""
        total = data.get("total_cost", 0)
        currency = data.get("currency", "USD")
        period = data.get("time_period", {})
        
        # Format with 2 decimal places
        response = f"Total cost: ${total:.2f} {currency}"
        if period:
            response += f" for {period.get('start', '')} to {period.get('end', '')}"
        
        if data.get("service_breakdown"):
            response += "\n\nService breakdown:"
            for service, cost in data["service_breakdown"].items():
                if float(cost) > 0:  # Only show services with costs
                    response += f"\n  {service}: ${float(cost):.2f}"
        
        return response


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""
    
    def __init__(self, config: Config):
        self.api_key = config.openai_api_key
        if not self.api_key:
            raise ValueError("OpenAI API key not configured")
    
    def parse_query(self, query: str) -> QueryParameters:
        """Parse query using OpenAI."""
        try:
            import openai
            openai.api_key = self.api_key
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Extract AWS cost query parameters. Return JSON."},
                    {"role": "user", "content": f"Query: {query}\nExtract: service, time_period, group_by"}
                ],
                temperature=0
            )
            
            text = response.choices[0].message.content
            parsed = json.loads(text)
            
            params = QueryParameters()
            params.service = parsed.get("service")
            if parsed.get("time_period"):
                params.time_period = parse_date_range(parsed["time_period"])
            params.group_by = parsed.get("group_by", ["SERVICE"])
            
            return params
            
        except Exception:
            # Fallback to simple parsing
            return self._simple_parse(query)
    
    def format_response(self, data: Dict[str, Any], query: str) -> str:
        """Format response using OpenAI."""
        try:
            import openai
            openai.api_key = self.api_key
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Format AWS cost data into natural language."},
                    {"role": "user", "content": f"Data: {json.dumps(data)}\nQuery: {query}"}
                ],
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception:
            return self._simple_format(data)
    
    def _simple_parse(self, query: str) -> QueryParameters:
        """Simple fallback parser."""
        params = QueryParameters()
        
        # Extract service names
        services = ["EC2", "S3", "RDS", "Lambda", "CloudFront", "DynamoDB"]
        for service in services:
            if service.lower() in query.lower():
                params.service = service
                break
        
        # Parse time period
        params.time_period = parse_date_range(query)
        params.group_by = ["SERVICE"]
        
        return params
    
    def _simple_format(self, data: Dict[str, Any]) -> str:
        """Simple fallback formatter."""
        total = data.get("total_cost", 0)
        currency = data.get("currency", "USD")
        
        response = f"Your AWS costs totaled ${total:.2f} {currency}"
        
        if data.get("service_breakdown"):
            response += ". Here's the breakdown by service:"
            for service, cost in data["service_breakdown"].items():
                if float(cost) > 0:  # Only show services with costs
                    response += f"\n• {service}: ${float(cost):.2f}"
        
        return response


def get_llm_provider(config: Config) -> LLMProvider:
    """Get the configured LLM provider."""
    if config.llm_provider == "openai":
        return OpenAIProvider(config)
    else:
        return OllamaProvider(config)
