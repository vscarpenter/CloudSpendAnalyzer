"""Simple AWS Cost Explorer client."""

import boto3
from datetime import datetime
from decimal import Decimal
from typing import Dict, Any, Optional
from config import Config
from models import QueryParameters, CostData, TimePeriod


class AWSCostClient:
    """Simple client for AWS Cost Explorer API."""
    
    def __init__(self, config: Config):
        self.config = config
        self.session = boto3.Session(
            profile_name=config.aws_profile,
            region_name=config.aws_region
        )
        self.client = self.session.client('ce')
    
    def get_costs(self, params: QueryParameters) -> CostData:
        """Get costs from AWS Cost Explorer."""
        
        # Default time period if not specified
        if not params.time_period:
            # Default to last 30 days
            from datetime import timedelta, timezone
            now = datetime.now(timezone.utc)
            params.time_period = TimePeriod(
                start=now - timedelta(days=30),
                end=now
            )
        
        # Format dates for AWS API
        start_date = params.time_period.start.strftime('%Y-%m-%d')
        end_date = params.time_period.end.strftime('%Y-%m-%d')
        
        # Determine granularity based on date range
        from datetime import timedelta
        date_diff = params.time_period.end - params.time_period.start
        if date_diff.days <= 31:
            granularity = 'DAILY'
        else:
            granularity = 'MONTHLY'
        
        # Build the API request
        request = {
            'TimePeriod': {
                'Start': start_date,
                'End': end_date
            },
            'Granularity': granularity,
            'Metrics': ['UnblendedCost'],
            'GroupBy': [
                {'Type': 'DIMENSION', 'Key': 'SERVICE'}
            ]
        }
        
        # Add service filter if specified
        if params.service:
            request['Filter'] = {
                'Dimensions': {
                    'Key': 'SERVICE',
                    'Values': [self._normalize_service_name(params.service)]
                }
            }
        
        try:
            # Make the API call
            response = self.client.get_cost_and_usage(**request)
            
            # Parse the response
            total_cost = Decimal('0')
            service_breakdown = {}
            
            for result in response.get('ResultsByTime', []):
                # Handle grouped costs (by service)
                for group in result.get('Groups', []):
                    service_name = group['Keys'][0]
                    # Check if metrics exist and have the expected structure
                    metrics = group.get('Metrics', {})
                    if 'UnblendedCost' in metrics:
                        cost_data = metrics['UnblendedCost']
                        cost = Decimal(cost_data.get('Amount', '0'))
                    else:
                        cost = Decimal('0')
                    
                    if service_name not in service_breakdown:
                        service_breakdown[service_name] = Decimal('0')
                    service_breakdown[service_name] += cost
                    total_cost += cost
                
                # Handle ungrouped costs (when no grouping is applied)
                if 'Total' in result:
                    total_metrics = result['Total']
                    if 'UnblendedCost' in total_metrics:
                        cost = Decimal(total_metrics['UnblendedCost'].get('Amount', '0'))
                        total_cost += cost
            
            return CostData(
                total_cost=total_cost,
                currency='USD',
                time_period=params.time_period,
                service_breakdown=service_breakdown if service_breakdown else None,
                raw_data=response
            )
            
        except Exception as e:
            # Return error as CostData
            return CostData(
                total_cost=Decimal('0'),
                currency='USD',
                time_period=params.time_period,
                service_breakdown=None,
                raw_data={'error': str(e)}
            )
    
    def _normalize_service_name(self, service: str) -> str:
        """Normalize service name for AWS API."""
        # Map common names to AWS service names
        service_map = {
            'ec2': 'Amazon Elastic Compute Cloud - Compute',
            'elastic compute': 'Amazon Elastic Compute Cloud - Compute',
            's3': 'Amazon Simple Storage Service',
            'storage': 'Amazon Simple Storage Service',
            'rds': 'Amazon Relational Database Service',
            'database': 'Amazon Relational Database Service',
            'lambda': 'AWS Lambda',
            'cloudfront': 'Amazon CloudFront',
            'dynamodb': 'Amazon DynamoDB',
            'elasticache': 'Amazon ElastiCache',
            'redshift': 'Amazon Redshift',
            'eks': 'Amazon Elastic Container Service for Kubernetes',
            'ecs': 'Amazon Elastic Container Service',
            'api gateway': 'Amazon API Gateway',
            'cloudwatch': 'Amazon CloudWatch',
            'sns': 'Amazon Simple Notification Service',
            'sqs': 'Amazon Simple Queue Service',
            'kinesis': 'Amazon Kinesis',
            'glue': 'AWS Glue'
        }
        
        # Try to match the service name
        service_lower = service.lower()
        for key, value in service_map.items():
            if key in service_lower:
                return value
        
        # Return original if no match
        return service
    
    def test_connection(self) -> bool:
        """Test AWS connection and permissions."""
        try:
            # Try a simple API call
            from datetime import timedelta, timezone
            now = datetime.now(timezone.utc)
            response = self.client.get_cost_and_usage(
                TimePeriod={
                    'Start': (now - timedelta(days=1)).strftime('%Y-%m-%d'),
                    'End': now.strftime('%Y-%m-%d')
                },
                Granularity='DAILY',
                Metrics=['UnblendedCost']
            )
            return True
        except Exception:
            return False
