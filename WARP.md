# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

AWS Cost Explorer CLI is a comprehensive Python command-line tool that enables natural language querying of AWS cost and billing data. The application integrates with multiple LLM providers (OpenAI, Anthropic, Bedrock, Ollama, Gemini) to parse user queries into structured AWS Cost Explorer API calls, returning formatted cost analysis with optimization recommendations.

## Common Development Commands

### Development Setup (uv-based)
```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode with all dependencies
uv pip install -e .[dev]

# Install pre-commit hooks
make precommit-install
```

### Testing
```bash
# Run all tests quickly
make test

# Run tests with coverage
make coverage

# Run specific test file
pytest tests/test_aws_client.py -v

# Run tests with specific markers
pytest -m "not integration" -v
```

### Code Quality & Linting
```bash
# Run all linters and formatters (uses pre-commit if available)
make lint

# Individual tools
make format     # Black + isort formatting
make typecheck  # mypy type checking
make security   # bandit + safety checks

# Run comprehensive verification (lint + test)
make verify
```

### Running the CLI
```bash
# Basic usage with Makefile
make run QUERY="EC2 costs last month"
make run QUERY="S3 storage costs" PROFILE=production

# Direct CLI usage
aws-cost-cli query "Show me EC2 costs for last month"
aws-cost-cli query "RDS costs" --profile prod --llm-provider gemini
```

### Build & Package
```bash
# Build Docker image
make docker-build

# Clean build artifacts
make clean

# Build Python package
python -m build
```

## High-Level Architecture

### Core Processing Pipeline

The application follows a modular pipeline architecture centered around the **QueryPipeline** class:

1. **Query Ingestion** (`cli.py`) → User input processing with rich terminal interface
2. **Query Parsing** (`query_processor.py`) → LLM-powered natural language to structured parameters
3. **Data Retrieval** (`aws_client.py` + `cache_manager.py`) → AWS Cost Explorer API with intelligent caching
4. **Response Generation** (`response_formatter.py`) → LLM-generated natural language responses
5. **Output Formatting** → Multiple formats (simple, rich, JSON) with export capabilities

### LLM Provider Architecture

**Multi-Provider System** with unified interface:
- **Abstract Base**: `LLMProvider` class defines consistent interface
- **Provider Factory**: `ProviderFactory` manages provider creation and validation
- **Supported Providers**: OpenAI, Anthropic (Claude), AWS Bedrock, Ollama (local), Google Gemini
- **Fallback Strategy**: Automatic failover between providers with health monitoring
- **Provider Override**: Runtime provider switching via `--llm-provider` flag

### Configuration System

**Hierarchical Configuration** with environment variable support:
- **Precedence**: Default values → Config files (YAML/JSON) → Environment variables
- **Config Manager**: `ConfigManager` handles multi-source configuration loading
- **Environment Variables**: `AWS_COST_CLI_*` prefixed variables for all settings
- **Provider Keys**: Direct environment variable support (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`)

### Data Models & Type System

**Strongly Typed** data flow using dataclasses and enums:
- **Core Models** (`models.py`):
  - `QueryParameters` - Parsed query structure
  - `CostData`, `CostResult` - AWS cost data representations
  - `TimePeriod`, `MetricType` - Time and metric definitions
  - `TrendData`, `ForecastData` - Analysis and forecasting models

### Advanced Features

**Performance Optimization**:
- **Parallel Processing**: Large date range queries split into concurrent chunks
- **Compression**: Cache compression reduces storage by ~35%
- **Performance Monitoring**: Detailed metrics tracking with `--performance-metrics`

**Cost Analysis & Optimization**:
- **Cost Optimizer** (`cost_optimizer.py`) - Rightsizing, Reserved Instances, Savings Plans analysis
- **Trend Analysis** (`trend_analysis.py`) - Period-over-period comparisons and forecasting
- **Optimization Formatter** - Specialized formatting for recommendation reports

**Data Export & Integration**:
- **Export Manager** (`data_exporter.py`) - CSV, JSON, Excel export with email integration
- **Interactive Query Builder** (`interactive_query_builder.py`) - Guided query construction
- **Template System** - Pre-built query templates for common use cases

**System Health & Monitoring**:
- **Health Checker** (`health.py`) - AWS connectivity, provider availability, system metrics
- **Validation** (`validation.py`) - Query validation, security pattern detection, AWS service validation
- **Performance Monitor** - Query optimization and execution tracking

### Key Architectural Patterns

**Error Handling Strategy**:
- Custom exception hierarchy in `exceptions.py`
- Graceful degradation with fallback providers
- User-friendly error messages with actionable guidance

**Caching Strategy**:
- File-based caching with configurable TTL
- Hash-based cache keys from query parameters
- JSON serialization with datetime handling
- Optional compression for storage optimization

**Date Processing**:
- **Date Formatter** (`date_formatter.py`) - Intelligent period detection (single day/month/quarter/year, multi-month, custom ranges)
- **Date Utils** (`date_utils.py`) - Natural language date parsing with business calendar support
- Fiscal year support and configurable formatting

### AWS Integration

**Multi-Profile Support**:
- **Credential Manager** handles AWS profile validation and switching
- **Cost Explorer Client** provides direct AWS API interface
- Comprehensive AWS service name validation and normalization

### CLI Command Groups

The application provides several command interfaces:
- **Basic Queries**: Natural language cost queries with multiple output formats
- **Export Commands**: Multi-format data export (CSV, JSON, Excel) with email integration  
- **Optimization Commands**: Cost optimization analysis and recommendations
- **Interactive Mode**: Guided query building with templates and history
- **Health Commands**: System diagnostics and provider status checking
- **Configuration**: Provider setup and configuration management

### Testing Architecture

**Comprehensive Test Coverage**:
- Unit tests for all core components
- Integration tests for end-to-end query processing
- Performance tests for large dataset queries
- Error handling validation for edge cases
- Mock-based testing for AWS and LLM provider interactions

This architecture enables reliable, performant natural language querying of AWS cost data with extensive customization options and robust error handling.
