# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AWS Cost Explorer CLI is a Python command-line tool that enables natural language querying of AWS cost and billing data. The application integrates with multiple LLM providers (OpenAI, Anthropic, Bedrock, Ollama, Gemini) to parse user queries, provide optimization recommendations, and format responses naturally. It includes features like data export (CSV/JSON), interactive query building, cost optimization analysis, anomaly detection, health checks, and smart date formatting.

## Common Commands

### Development Setup (uv)
```bash
# Create and activate virtualenv
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install in development mode with extras
uv pip install -e .[dev]
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
uv pip install .
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
  - Provider implementations: `OpenAIProvider`, `AnthropicProvider`, `BedrockProvider`, `OllamaProvider`, `GeminiProvider`
  - Parses natural language queries into structured `QueryParameters`
  - Enhanced year parsing for full year queries (e.g., "S3 costs for 2025")
  - Consistent date parsing across all LLM providers with fallback parser support
  - Provider health checks with failover support

- **`cache_manager.py`**: File-based caching system with TTL
  - Hash-based cache keys from query parameters
  - Configurable cache directory and TTL
  - JSON serialization with custom datetime handling
  - Compression support for reduced storage

- **`response_formatter.py`**: Response formatting system
  - `LLMResponseFormatter`: Natural language response generation
  - `SimpleFormatter`, `DetailedFormatter`: Structured output formats
  - Rich terminal output integration

### Advanced Features

- **`query_pipeline.py`**: End-to-end query processing pipeline
  - `QueryPipeline`: Orchestrates the complete query flow
  - `QueryContext`, `QueryResult`: Request/response handling
  - Error handling and retry logic

- **`cost_optimizer.py`**: Cost optimization analysis and recommendations
  - `CostOptimizer`: Analyzes cost data for optimization opportunities
  - `OptimizationRecommendation`: Individual recommendation data structure
  - Support for rightsizing, reserved instances, savings plans analysis
  - Cost anomaly detection and budget variance analysis

- **`data_exporter.py`**: Data export capabilities (CSV and JSON)
  - `ExportManager`: Coordinates the export formats
  - `CSVExporter`, `JSONExporter`: Format-specific exporters
  - Date-formatted output via the shared `DateFormatter`

- **`interactive_query_builder.py`**: Guided query construction interface
  - `InteractiveQueryBuilder`: Step-by-step query building
  - `QueryTemplate`: Pre-built query templates for common use cases
  - Query history and favorites management
  - Real-time query validation and suggestions

### Supporting Components

- **`cli.py`**: Main CLI interface with click framework integration
  - Commands for query, export, optimize, anomaly detection, interactive, cache management, and health
  - `providers` command group with `list`, `test`, `performance`, `health`, and `reset` subcommands
  - Rich terminal output formatting with progress indicators
  - Multi-provider switching support with `--llm-provider` option
  - Comprehensive error handling and user feedback

- **`exceptions.py`**: Centralized exception handling
  - Custom exception types for different error categories
  - Error message formatting and user-friendly output
  - Logging integration for debugging
  - LLM provider-specific error handling

- **`date_utils.py`**: Date parsing and manipulation utilities
  - Natural language date parsing
  - Business calendar support
  - Time zone handling

- **`date_formatter.py`**: Small smart date formatter (~135 lines)
  - `DateFormatter`: Detects the period type from a date range and renders it concisely (e.g. "August 2025", "Q3 2025", "2025", "Jul 1 - Sep 30, 2025")
  - `safe_format_time_period`: Formatting helper that falls back gracefully and never throws

- **`provider_factory.py`**: LLM provider factory and management
  - `ProviderFactory`: Creates and manages LLM provider instances
  - Support for all providers: OpenAI, Anthropic, Bedrock, Ollama, Gemini
  - Provider configuration validation and status checking
  - Unified provider creation with consistent error handling

- **`health.py`**: Health checks and system diagnostics
  - One-shot health check covering AWS credential validity and cache directory access
  - Returns a structured result with per-check status; CLI exits 0 when healthy, 1 otherwise

- **`optimization_formatter.py`**: Specialized formatting for optimization reports
  - Recommendation prioritization and grouping
  - Rich table formatting for optimization results
  - Export formatting for optimization data

### Data Flow

1. User input → `cli.py` → route to appropriate command handler
2. Query command → `query_pipeline.py` → orchestrate complete flow:
   - `query_processor` (LLM parsing) → `QueryParameters`
   - `cache_manager` (check cache) → cached result or AWS API call
   - `aws_client` → raw cost data → `CostData` models
   - `response_formatter` → formatted response
3. Export command → `data_exporter` → CSV or JSON output files
4. Optimization command → `cost_optimizer` → recommendations and analysis
5. Interactive command → `interactive_query_builder` → guided query construction

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
- Consistent system prompts across all providers for reliable date parsing
- Fallback mechanisms for provider failures with robust pattern matching
- Comprehensive error handling for all providers

## Key Dependencies

### Core Dependencies
- **boto3**: AWS SDK for Cost Explorer API integration
- **click**: CLI framework (entry point in setup.py)
- **rich**: Terminal formatting and output with tables, panels, and progress bars
- **pyyaml**: Configuration file parsing (YAML/JSON support)
- **requests**: HTTP client for API calls

### LLM Provider Libraries
- **openai**: OpenAI GPT integration
- **anthropic**: Anthropic Claude integration
- **google-generativeai**: Google Gemini integration
- **boto3** (Bedrock): AWS Bedrock LLM services
- **requests**: HTTP client for Ollama local provider

### Export Dependencies
- Standard library only: **csv**, **json** (CSV and JSON export formats)

### Development Tools
- **pytest**: Testing framework with coverage support
- **black**: Code formatting
- **flake8**: Code linting
- **mypy**: Static type checking

## CLI Commands

The full command surface: `cache-stats`, `cleanup-cache`, `clear-cache`, `configure`, `detect-anomalies`, `export`, `favorites`, `health`, `interactive`, `list-profiles`, `optimize`, `pipeline-status`, `providers`, `query`, `show-config`, `suggest`, `test`, `warm-cache`. The `providers` group has subcommands: `list`, `test`, `performance`, `health`, `reset`.

### Basic Query Commands
```bash
# Basic cost queries
aws-cost-cli query "Show me EC2 costs for last month"
aws-cost-cli query "S3 storage costs for 2025" --format rich

# Query with specific profiles and provider options
aws-cost-cli query "RDS costs" --profile production --fresh --llm-provider gemini
```
Query options: `--profile/-p`, `--fresh/-f`, `--format` (simple, rich, llm, json), `--llm-provider`, `--config-file/-c`.

### Export Commands
```bash
# Export to CSV or JSON (the only supported formats)
aws-cost-cli export "EC2 costs last quarter" --format csv --output costs.csv
aws-cost-cli export "S3 costs" --format json --output costs.json
```

### Optimization and Anomaly Commands
```bash
# Cost optimization analysis
aws-cost-cli optimize --type rightsizing
aws-cost-cli optimize --type reserved_instances --service EC2

# Cost anomaly detection
aws-cost-cli detect-anomalies
```

### Interactive Mode
```bash
# Launch interactive query builder
aws-cost-cli interactive
```

### Health and Provider Commands
```bash
# One-shot health check (AWS credentials + cache directory)
aws-cost-cli health
aws-cost-cli health --json

# Provider management (subcommand group)
aws-cost-cli providers list
aws-cost-cli providers test gemini
aws-cost-cli providers health
aws-cost-cli providers performance
aws-cost-cli providers reset
```

## Recent Improvements

### New LLM Provider Support (Latest)
- **Google Gemini Integration**: Added support for Gemini 1.5 Flash and Pro models with `GeminiProvider`
- **Provider Factory System**: Centralized provider creation and management with `ProviderFactory`
- **Multi-provider Configuration**: Enhanced config support with provider-specific settings and fallback chains
- **Provider Performance Monitoring**: Real-time health checks and performance metrics for all providers

### Smart Date Formatting
- **Period Detection**: Renders a date range concisely based on its type (single month/quarter/year or custom range)
- **Safe Formatting**: Falls back gracefully and never throws exceptions

### Health Checks
- **One-shot Health Command**: Validates AWS credentials and cache directory access, with optional JSON output

### Enhanced Configuration System
- **Multi-provider Support**: Unified configuration for all 5 LLM providers (OpenAI, Anthropic, Bedrock, Ollama, Gemini)
- **Hierarchical Configuration**: Environment variables, config files, and defaults with proper precedence
- **Date Formatting Options**: Configurable formatting settings

### GitHub Actions Integration
- **Automated CI/CD**: GitHub Actions setup for continuous integration and deployment
- **Automated testing**: Test suite execution on pull requests and commits  
- **Code quality checks**: Automated linting, formatting, and type checking

## Documentation

The project includes comprehensive documentation:

- **USER_GUIDE.md**: Comprehensive end-user documentation for all features
- **OLLAMA_SETUP.md**: Local LLM setup guide for Ollama integration

## Quality Assurance

### Testing Strategy
- **Unit tests**: Comprehensive test coverage for all core components
- **Integration tests**: End-to-end testing of query processing pipeline
- **Error handling tests**: Validation of error scenarios and edge cases

### Code Quality
- **Type safety**: Full mypy type checking with strict mode
- **Code formatting**: Black formatter for consistent code style  
- **Linting**: Flake8 for code quality and style enforcement
- **Documentation**: Comprehensive docstrings and inline comments
