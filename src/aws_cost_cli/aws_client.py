"""AWS client management for Cost Explorer API."""

import boto3
import time
from botocore.exceptions import (
    ClientError, 
    NoCredentialsError, 
    PartialCredentialsError,
    ProfileNotFound
)
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from .models import QueryParameters, CostData, CostResult, CostAmount, TimePeriod, Group, TimePeriodGranularity


class CredentialManager:
    """Manages AWS credentials and profiles."""
    
    def __init__(self):
        self.session = boto3.Session()
    
    def get_available_profiles(self) -> List[str]:
        """Get list of available AWS profiles."""
        try:
            return self.session.available_profiles
        except Exception:
            return []
    
    def validate_credentials(self, profile: Optional[str] = None) -> bool:
        """Validate AWS credentials for the specified profile."""
        try:
            session = boto3.Session(profile_name=profile) if profile else boto3.Session()
            sts_client = session.client('sts')
            sts_client.get_caller_identity()
            return True
        except (NoCredentialsError, PartialCredentialsError, ProfileNotFound):
            return False
        except ClientError as e:
            # If we get a client error but not credential-related, credentials are valid
            error_code = e.response.get('Error', {}).get('Code', '')
            return error_code not in ['InvalidUserID.NotFound', 'AccessDenied']
        except Exception:
            return False
    
    def get_caller_identity(self, profile: Optional[str] = None) -> Dict[str, Any]:
        """Get AWS caller identity information."""
        try:
            session = boto3.Session(profile_name=profile) if profile else boto3.Session()
            sts_client = session.client('sts')
            return sts_client.get_caller_identity()
        except Exception as e:
            raise RuntimeError(f"Failed to get caller identity: {e}")


class AWSCostClient:
    """Client for AWS Cost Explorer API operations."""
    
    REQUIRED_PERMISSIONS = [
        "ce:GetCostAndUsage",
        "ce:GetDimensionValues", 
        "ce:GetReservationCoverage",
        "ce:GetReservationPurchaseRecommendation",
        "ce:GetReservationUtilization",
        "ce:GetUsageReport"
    ]
    
    def __init__(self, profile: Optional[str] = None, region: str = "us-east-1"):
        """Initialize AWS Cost Explorer client."""
        self.profile = profile
        self.region = region
        self.session = boto3.Session(profile_name=profile) if profile else boto3.Session()
        self.client = self.session.client('ce', region_name=region)
        self.credential_manager = CredentialManager()
    
    def validate_permissions(self) -> bool:
        """Validate that the current credentials have necessary permissions."""
        try:
            # Test with a minimal API call
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=1)
            
            self.client.get_cost_and_usage(
                TimePeriod={
                    'Start': start_date.strftime('%Y-%m-%d'),
                    'End': end_date.strftime('%Y-%m-%d')
                },
                Granularity='DAILY',
                Metrics=['BlendedCost']
            )
            return True
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code in ['AccessDenied', 'UnauthorizedOperation']:
                return False
            # Other errors might be due to data availability, not permissions
            return True
        except Exception:
            return False
    
    def get_cost_and_usage(self, params: QueryParameters) -> CostData:
        """Retrieve cost and usage data from AWS Cost Explorer."""
        import time
        import random
        
        max_retries = 3
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                # Build the API request
                request_params = self._build_cost_request(params)
                
                # Make the API call
                response = self.client.get_cost_and_usage(**request_params)
                
                # Parse and return the response
                return self._parse_cost_response(response, params)
                
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', '')
                error_message = e.response.get('Error', {}).get('Message', '')
                
                if error_code == 'AccessDenied':
                    raise PermissionError(
                        f"Access denied to Cost Explorer API. "
                        f"Required permissions: {', '.join(self.REQUIRED_PERMISSIONS)}"
                    )
                elif error_code == 'ThrottlingException':
                    if attempt < max_retries - 1:
                        # Exponential backoff with jitter
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                        time.sleep(delay)
                        continue
                    else:
                        raise RuntimeError(
                            "AWS API rate limit exceeded after multiple retries. "
                            "Please wait a few minutes and try again."
                        )
                elif error_code == 'InvalidParameterValue':
                    raise ValueError(f"Invalid parameter in cost query: {error_message}")
                elif error_code == 'ServiceUnavailable':
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        time.sleep(delay)
                        continue
                    else:
                        raise RuntimeError(
                            "AWS Cost Explorer service is temporarily unavailable. "
                            "Please try again later."
                        )
                else:
                    raise RuntimeError(f"AWS API error ({error_code}): {error_message}")
            
            except Exception as e:
                if attempt < max_retries - 1:
                    # Retry on unexpected errors
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)
                    continue
                else:
                    raise RuntimeError(f"Failed to retrieve cost data: {e}")
    
    def get_dimension_values(self, dimension: str, time_period: Optional[TimePeriod] = None) -> List[str]:
        """Get available values for a specific dimension (e.g., SERVICE, INSTANCE_TYPE)."""
        try:
            # Default to last 30 days if no time period specified
            if not time_period:
                end_date = datetime.now().date()
                start_date = end_date - timedelta(days=30)
                time_period = TimePeriod(
                    start=datetime.combine(start_date, datetime.min.time()),
                    end=datetime.combine(end_date, datetime.min.time())
                )
            
            response = self.client.get_dimension_values(
                TimePeriod={
                    'Start': time_period.start.strftime('%Y-%m-%d'),
                    'End': time_period.end.strftime('%Y-%m-%d')
                },
                Dimension=dimension
            )
            
            return [item['Value'] for item in response.get('DimensionValues', [])]
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            error_message = e.response.get('Error', {}).get('Message', '')
            raise RuntimeError(f"Failed to get dimension values ({error_code}): {error_message}")
    
    def _build_cost_request(self, params: QueryParameters) -> Dict[str, Any]:
        """Build the Cost Explorer API request from query parameters."""
        # Default time period if not specified (last 30 days)
        if not params.time_period:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=30)
            time_period = TimePeriod(
                start=datetime.combine(start_date, datetime.min.time()),
                end=datetime.combine(end_date, datetime.min.time())
            )
        else:
            time_period = params.time_period
        
        request = {
            'TimePeriod': {
                'Start': time_period.start.strftime('%Y-%m-%d'),
                'End': time_period.end.strftime('%Y-%m-%d')
            },
            'Granularity': params.granularity.value,
            'Metrics': [metric.value for metric in params.metrics]
        }
        
        # Add service filter if specified
        if params.service:
            request['Filter'] = {
                'Dimensions': {
                    'Key': 'SERVICE',
                    'Values': [params.service]
                }
            }
        
        # Add group by if specified
        if params.group_by:
            request['GroupBy'] = [
                {'Type': 'DIMENSION', 'Key': group} for group in params.group_by
            ]
        
        return request
    
    def _parse_cost_response(self, response: Dict[str, Any], params: QueryParameters) -> CostData:
        """Parse AWS Cost Explorer API response into CostData model."""
        results = []
        total_amount = 0.0
        
        for result_item in response.get('ResultsByTime', []):
            # Parse time period
            time_period = TimePeriod(
                start=datetime.strptime(result_item['TimePeriod']['Start'], '%Y-%m-%d'),
                end=datetime.strptime(result_item['TimePeriod']['End'], '%Y-%m-%d')
            )
            
            # Parse total cost
            total_cost_data = result_item.get('Total', {})
            total_cost = CostAmount(amount=0.0)
            
            if total_cost_data:
                # Use the first metric for total cost
                metric_key = list(total_cost_data.keys())[0]
                amount_str = total_cost_data[metric_key]['Amount']
                unit = total_cost_data[metric_key]['Unit']
                total_cost = CostAmount(amount=float(amount_str), unit=unit)
                total_amount += float(amount_str)
            
            # Parse groups
            groups = []
            for group_item in result_item.get('Groups', []):
                group_keys = group_item.get('Keys', [])
                group_metrics = {}
                
                for metric_name, metric_data in group_item.get('Metrics', {}).items():
                    amount = float(metric_data['Amount'])
                    unit = metric_data['Unit']
                    group_metrics[metric_name] = CostAmount(amount=amount, unit=unit)
                
                groups.append(Group(keys=group_keys, metrics=group_metrics))
            
            # Create cost result
            cost_result = CostResult(
                time_period=time_period,
                total=total_cost,
                groups=groups,
                estimated=result_item.get('Estimated', False)
            )
            results.append(cost_result)
        
        # Create overall time period
        if results:
            overall_period = TimePeriod(
                start=results[0].time_period.start,
                end=results[-1].time_period.end
            )
        else:
            overall_period = params.time_period or TimePeriod(
                start=datetime.now() - timedelta(days=30),
                end=datetime.now()
            )
        
        return CostData(
            results=results,
            time_period=overall_period,
            total_cost=CostAmount(amount=total_amount),
            group_definitions=params.group_by or []
        )
    
    def check_service_availability(self) -> Dict[str, Any]:
        """Check if AWS Cost Explorer service is available and responsive."""
        try:
            # Make a minimal API call to check service health
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=1)
            
            start_time = time.time()
            response = self.client.get_cost_and_usage(
                TimePeriod={
                    'Start': start_date.strftime('%Y-%m-%d'),
                    'End': end_date.strftime('%Y-%m-%d')
                },
                Granularity='DAILY',
                Metrics=['BlendedCost']
            )
            response_time = time.time() - start_time
            
            return {
                'available': True,
                'response_time_ms': round(response_time * 1000, 2),
                'status': 'healthy'
            }
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            
            if error_code == 'AccessDenied':
                return {
                    'available': False,
                    'status': 'access_denied',
                    'error': 'Insufficient permissions for Cost Explorer API'
                }
            elif error_code == 'ThrottlingException':
                return {
                    'available': True,
                    'status': 'throttled',
                    'error': 'API rate limit exceeded'
                }
            elif error_code == 'ServiceUnavailable':
                return {
                    'available': False,
                    'status': 'service_unavailable',
                    'error': 'Cost Explorer service is temporarily unavailable'
                }
            else:
                return {
                    'available': False,
                    'status': 'error',
                    'error': f"API error ({error_code}): {e.response.get('Error', {}).get('Message', '')}"
                }
        
        except Exception as e:
            return {
                'available': False,
                'status': 'error',
                'error': f"Unexpected error: {str(e)}"
            }
    
    def get_supported_services(self) -> List[str]:
        """Get list of AWS services that have cost data available."""
        try:
            # Get services from the last 30 days
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=30)
            
            response = self.client.get_dimension_values(
                TimePeriod={
                    'Start': start_date.strftime('%Y-%m-%d'),
                    'End': end_date.strftime('%Y-%m-%d')
                },
                Dimension='SERVICE'
            )
            
            services = [item['Value'] for item in response.get('DimensionValues', [])]
            return sorted(services)
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            error_message = e.response.get('Error', {}).get('Message', '')
            raise RuntimeError(f"Failed to get supported services ({error_code}): {error_message}")
    
    def estimate_query_cost(self, params: QueryParameters) -> Dict[str, Any]:
        """Estimate the complexity and potential cost of a query."""
        complexity_score = 0
        warnings = []
        
        # Time period complexity
        if params.time_period:
            days = (params.time_period.end - params.time_period.start).days
            if days > 365:
                complexity_score += 3
                warnings.append("Query spans more than 1 year - may be slow")
            elif days > 90:
                complexity_score += 2
                warnings.append("Query spans more than 3 months")
            elif days > 30:
                complexity_score += 1
        
        # Granularity complexity
        if params.granularity == TimePeriodGranularity.HOURLY:
            complexity_score += 3
            warnings.append("Hourly granularity increases query complexity")
        elif params.granularity == TimePeriodGranularity.DAILY:
            complexity_score += 1
        
        # Group by complexity
        if params.group_by:
            complexity_score += len(params.group_by)
            if len(params.group_by) > 2:
                warnings.append("Multiple group-by dimensions increase complexity")
        
        # Determine complexity level
        if complexity_score <= 2:
            complexity_level = "low"
        elif complexity_score <= 5:
            complexity_level = "medium"
        else:
            complexity_level = "high"
            warnings.append("High complexity query may take longer to execute")
        
        return {
            'complexity_score': complexity_score,
            'complexity_level': complexity_level,
            'warnings': warnings,
            'estimated_response_time': self._estimate_response_time(complexity_score)
        }
    
    def _estimate_response_time(self, complexity_score: int) -> str:
        """Estimate response time based on complexity score."""
        if complexity_score <= 2:
            return "< 5 seconds"
        elif complexity_score <= 5:
            return "5-15 seconds"
        else:
            return "15-60 seconds"