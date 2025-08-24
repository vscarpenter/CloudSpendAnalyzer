# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AWS Cost Explorer CLI is a Python command-line tool that enables natural language querying of AWS cost and billing data. The application integrates with multiple LLM providers (OpenAI, Anthropic, Bedrock, Ollama) to parse user queries and format responses naturally.

## Common Commands

### Development Setup
```bash
# Install in development mode
pip install -e .

# Install with development dependencies
pip install -e .[dev]
```

### Testing
```bash
# Run all tests
pytest tests/

# Run tests with coverage
pytest tests/ --cov=src/aws_cost_cli

# Run specific test file
pytest tests/test_aws_client.py
```

### Code Quality
```bash
# Format code with Black
black src/

# Lint with flake8
flake8 src/

# Type checking with mypy
mypy src/
```

### Build and Installation
```bash
# Build package
python setup.py sdist bdist_wheel

# Install from source
pip install .
```

## Architecture Overview

The application follows a modular architecture with clear separation of concerns:

### Core Components

- **`models.py`**: Central data models and type definitions using dataclasses and enums
  - `Config`: Application configuration management
  - `QueryParameters`: Natural language query parsing results
  - `CostData`, `CostResult`: AWS cost data representations
  - `TimePeriod`, `MetricType`: Time and metric type definitions

- **`config.py`**: Configuration management system (`ConfigManager`)
  - Hierarchical config loading: defaults → file → environment variables
  - Supports YAML/JSON config files in multiple locations
  - Environment variable overrides with `AWS_COST_CLI_*` prefix

- **`aws_client.py`**: AWS integration layer
  - `CredentialManager`: AWS profile and credential validation
  - `CostExplorerClient`: Direct AWS Cost Explorer API interface
  - Handles multiple AWS profiles and credential validation

- **`query_processor.py`**: LLM integration for natural language processing
  - Abstract `LLMProvider` base class
  - Provider implementations: `OpenAIProvider`, `AnthropicProvider`, `BedrockProvider`, `OllamaProvider`
  - Parses natural language queries into structured `QueryParameters`

- **`cache_manager.py`**: File-based caching system with TTL
  - Hash-based cache keys from query parameters
  - Configurable cache directory and TTL
  - JSON serialization with custom datetime handling

- **`response_formatter.py`**: Response formatting system
  - `LLMResponseFormatter`: Natural language response generation
  - `SimpleFormatter`, `DetailedFormatter`: Structured output formats
  - Rich terminal output integration

### Data Flow

1. User query → `query_processor` (LLM parsing) → `QueryParameters`
2. `QueryParameters` → `cache_manager` (check cache) → cached result or AWS API call
3. AWS API → `aws_client` → raw cost data → `CostData` models
4. `CostData` → `response_formatter` (LLM or structured) → formatted response

### Configuration Hierarchy

Configuration is loaded in order of precedence:
1. Default values (in `ConfigManager`)
2. Config file (YAML/JSON from multiple search paths)
3. Environment variables (`AWS_COST_CLI_*`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`)

### LLM Provider System

The application uses a plugin-style architecture for LLM providers:
- Each provider implements the `LLMProvider` abstract interface
- Providers handle their own authentication and API calls
- Query parsing and response formatting are provider-agnostic
- Fallback mechanisms for provider failures

## Key Dependencies

- **boto3**: AWS SDK for Cost Explorer API integration
- **click**: CLI framework (entry point in setup.py)
- **rich**: Terminal formatting and output
- **pyyaml**: Configuration file parsing
- **openai/anthropic**: LLM provider libraries
- **pytest/black/flake8/mypy**: Development tools