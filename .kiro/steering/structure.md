# Project Structure

## Package Layout

```
aws-cost-explorer-cli/
├── src/
│   └── aws_cost_cli/
│       ├── __init__.py
│       ├── cli.py              # Main CLI interface (Click/Typer)
│       ├── query_processor.py  # LLM-powered query parsing
│       ├── aws_client.py       # boto3 AWS Cost Explorer integration
│       ├── cache_manager.py    # File-based caching system
│       └── config.py           # Configuration management
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── config/
│   └── default_config.yaml
├── setup.py
├── requirements.txt
└── README.md
```

## Core Module Responsibilities

### cli.py
- Command-line argument parsing and validation
- User input handling and output coordination
- Entry point for all CLI commands

### query_processor.py
- Natural language query parsing using LLM APIs
- Parameter extraction (service, date range, aggregation)
- Response generation from structured cost data

### aws_client.py
- AWS credential management and profile handling
- Cost Explorer API interactions via boto3
- AWS-specific error handling and permission validation

### cache_manager.py
- TTL-based file caching for API responses
- Query hash generation and cache invalidation
- Storage management and cleanup

### config.py
- YAML/JSON configuration file loading
- LLM provider configuration management
- User preferences and default settings

## Configuration Files

- **Global**: `~/.aws-cost-cli/config.yaml`
- **Project**: `.aws-cost-cli.yaml` (current directory)
- **Environment**: Environment variables for CI/CD
- **CLI**: Command-line argument overrides

## Testing Organization

- **Unit Tests**: Individual component testing with mocks
- **Integration Tests**: End-to-end workflow testing
- **Fixtures**: Reusable test data and AWS response mocks
- **Performance Tests**: Cache effectiveness and API rate limiting