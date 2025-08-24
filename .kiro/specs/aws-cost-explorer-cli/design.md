# Design Document

## Overview

The AWS Cost Explorer CLI is a Python-based command-line tool that provides natural language querying capabilities for AWS cost and billing data. The tool integrates with existing AWS credentials, uses LLM services for query parsing and response generation, and leverages AWS Cost Explorer APIs to retrieve detailed cost information.

## Architecture

The system follows a modular architecture with clear separation of concerns:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   CLI Interface │────│  Query Processor │────│   AWS Client    │
│   (Click/Typer) │    │   (LLM-powered)  │    │    (boto3)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Response Format │    │  Cache Manager   │    │ Config Manager  │
│   (Rich + LLM)  │    │   (File-based)   │    │   (YAML/JSON)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Components and Interfaces

### 1. CLI Interface Layer
**Technology:** Click or Typer framework
**Responsibilities:**
- Command-line argument parsing
- User input validation
- Output formatting coordination
- Error message display

**Key Classes:**
```python
class CostExplorerCLI:
    def query(self, question: str, profile: str = None, fresh: bool = False)
    def configure(self, llm_provider: str, api_key: str)
    def list_profiles(self)
```

### 2. Query Processor
**Technology:** LLM APIs (OpenAI/Anthropic/Bedrock/Ollama)
**Responsibilities:**
- Natural language query parsing
- Parameter extraction (service, date range, aggregation)
- Query intent classification
- Response generation from structured data

**Key Classes:**
```python
class QueryParser:
    def parse_query(self, query: str) -> QueryParameters
    def validate_parameters(self, params: QueryParameters) -> bool

class ResponseGenerator:
    def format_response(self, data: CostData, original_query: str) -> str
```

**Query Parameters Structure:**
```python
@dataclass
class QueryParameters:
    service: Optional[str]  # EC2, S3, RDS, etc.
    time_period: TimePeriod
    aggregation: str  # MONTHLY, DAILY, YEARLY
    metrics: List[str]  # BlendedCost, UnblendedCost, etc.
    group_by: Optional[List[str]]  # SERVICE, INSTANCE_TYPE, etc.
```

### 3. AWS Client Layer
**Technology:** boto3 SDK
**Responsibilities:**
- AWS credential management
- Cost Explorer API interactions
- Billing API interactions
- Error handling for AWS-specific issues

**Key Classes:**
```python
class AWSCostClient:
    def __init__(self, profile: str = None)
    def get_cost_and_usage(self, params: QueryParameters) -> CostData
    def get_dimension_values(self, dimension: str) -> List[str]
    def validate_permissions(self) -> bool

class CredentialManager:
    def get_available_profiles(self) -> List[str]
    def validate_credentials(self, profile: str) -> bool
```

### 4. Cache Manager
**Technology:** File-based caching with JSON/pickle
**Responsibilities:**
- Cost data caching with TTL
- Cache invalidation
- Storage management

**Key Classes:**
```python
class CacheManager:
    def get_cached_data(self, query_hash: str) -> Optional[CostData]
    def cache_data(self, query_hash: str, data: CostData, ttl: int)
    def invalidate_cache(self, pattern: str = None)
```

### 5. Configuration Manager
**Technology:** YAML/JSON configuration files
**Responsibilities:**
- LLM provider configuration
- Default settings management
- User preferences storage

## Data Models

### Cost Data Structure
```python
@dataclass
class CostData:
    results: List[CostResult]
    time_period: TimePeriod
    total_cost: Decimal
    currency: str
    group_definitions: List[GroupDefinition]

@dataclass
class CostResult:
    time_period: TimePeriod
    total: CostAmount
    groups: List[Group]
    estimated: bool

@dataclass
class CostAmount:
    amount: Decimal
    unit: str  # USD, etc.
```

### Configuration Structure
```python
@dataclass
class Config:
    llm_provider: str  # openai, anthropic, bedrock, ollama
    llm_config: Dict[str, Any]
    default_profile: Optional[str]
    cache_ttl: int  # seconds
    output_format: str  # simple, detailed, json
    default_currency: str
```

## Error Handling

### AWS API Errors
- **AccessDenied:** Provide specific IAM permission requirements
- **ThrottlingException:** Implement exponential backoff with user notification
- **InvalidParameterValue:** Parse AWS error and provide user-friendly guidance
- **ServiceUnavailable:** Suggest retry with cached data if available

### LLM API Errors
- **Rate Limiting:** Implement fallback to basic query parsing
- **API Unavailable:** Provide degraded functionality with pattern matching
- **Invalid Response:** Retry with different prompt or fallback to structured output

### Data Processing Errors
- **Invalid Date Ranges:** Suggest valid date formats and ranges
- **Unknown Services:** Provide list of available AWS services
- **Empty Results:** Explain possible reasons (no usage, wrong time period)

## Testing Strategy

### Unit Testing
- **Query Parser:** Test natural language parsing with various query formats
- **AWS Client:** Mock boto3 responses for different scenarios
- **Cache Manager:** Test TTL, invalidation, and storage mechanisms
- **Response Generator:** Verify output formatting and accuracy

### Integration Testing
- **End-to-End Queries:** Test complete flow from CLI input to formatted output
- **AWS API Integration:** Test with real AWS accounts (using test credentials)
- **LLM Integration:** Test with different providers and fallback scenarios
- **Error Scenarios:** Test network failures, permission issues, and invalid inputs

### Performance Testing
- **Cache Effectiveness:** Measure cache hit rates and response time improvements
- **API Rate Limiting:** Test behavior under AWS API throttling
- **Large Dataset Handling:** Test with accounts having extensive cost data

## Security Considerations

### Credential Management
- Never store AWS credentials in cache or logs
- Use AWS credential chain (environment, profile, IAM roles)
- Support MFA-enabled profiles
- Validate minimum required permissions

### LLM Integration
- Sanitize cost data before sending to external LLM APIs
- Support local LLM options (Ollama) for sensitive environments
- Implement request/response logging controls
- Allow opt-out of external LLM services

### Data Privacy
- Cache data locally with appropriate file permissions
- Provide cache clearing functionality
- Support configuration for data retention policies
- Log only non-sensitive query metadata

## Deployment and Distribution

### Package Structure
```
aws-cost-explorer-cli/
├── src/
│   ├── aws_cost_cli/
│   │   ├── __init__.py
│   │   ├── cli.py
│   │   ├── query_processor.py
│   │   ├── aws_client.py
│   │   ├── cache_manager.py
│   │   └── config.py
│   └── tests/
├── setup.py
├── requirements.txt
├── README.md
└── config/
    └── default_config.yaml
```

### Installation Methods
- **PyPI Package:** Standard pip installation
- **Homebrew Formula:** For macOS users
- **Docker Image:** For containerized environments
- **Standalone Binary:** Using PyInstaller for distribution

### Configuration Management
- **Global Config:** `~/.aws-cost-cli/config.yaml`
- **Project Config:** `.aws-cost-cli.yaml` in current directory
- **Environment Variables:** For CI/CD and automation scenarios
- **CLI Arguments:** For one-time overrides