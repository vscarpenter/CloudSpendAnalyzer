"""Simple query processor for natural language cost queries."""

import json
from typing import Dict, Any
from decimal import Decimal
from config import Config
from models import QueryParameters, CostData
from aws import AWSCostClient
from llm import get_llm_provider
from cache import SimpleCache, get_cache_key
from utils import format_currency


def process_query(query: str, config: Config) -> Dict[str, Any]:
    """Process a natural language AWS cost query."""
    
    # Initialize components
    llm = get_llm_provider(config)
    aws_client = AWSCostClient(config)
    cache = SimpleCache(config)
    
    # Generate cache key
    cache_key = get_cache_key(query, config.aws_profile)
    
    # Check cache first
    cached_result = cache.get(cache_key)
    if cached_result:
        return cached_result
    
    # Parse the natural language query
    try:
        params = llm.parse_query(query)
    except Exception as e:
        return {
            'success': False,
            'error': f'Failed to parse query: {str(e)}',
            'query': query
        }
    
    # Get costs from AWS
    try:
        cost_data = aws_client.get_costs(params)
    except Exception as e:
        import traceback
        return {
            'success': False,
            'error': f'Failed to get AWS costs: {str(e)}',
            'details': traceback.format_exc(),
            'query': query
        }
    
    # Check for errors in cost data
    if cost_data.raw_data and 'error' in cost_data.raw_data:
        return {
            'success': False,
            'error': cost_data.raw_data['error'],
            'query': query
        }
    
    # Prepare response data
    response_data = {
        'total_cost': float(cost_data.total_cost),
        'currency': cost_data.currency,
        'time_period': {
            'start': cost_data.time_period.start.isoformat(),
            'end': cost_data.time_period.end.isoformat()
        }
    }
    
    if cost_data.service_breakdown:
        response_data['service_breakdown'] = {
            service: float(cost)
            for service, cost in cost_data.service_breakdown.items()
        }
    
    # Format the response
    if config.output_format == 'json':
        result = {
            'success': True,
            'query': query,
            'data': response_data
        }
    else:
        # Get natural language response
        formatted_response = llm.format_response(response_data, query)
        result = {
            'success': True,
            'query': query,
            'response': formatted_response,
            'data': response_data
        }
    
    # Cache the result
    cache.set(cache_key, result)
    
    return result


def format_simple_response(data: Dict[str, Any]) -> str:
    """Format cost data as simple text response."""
    total = data.get('total_cost', 0)
    currency = data.get('currency', 'USD')
    
    response = format_currency(total, currency)
    
    if data.get('time_period'):
        period = data['time_period']
        response += f" ({period.get('start', '')} to {period.get('end', '')})"
    
    if data.get('service_breakdown'):
        response += "\n\nBy service:"
        for service, cost in sorted(
            data['service_breakdown'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            response += f"\n  {service}: {format_currency(cost, currency)}"
    
    return response
