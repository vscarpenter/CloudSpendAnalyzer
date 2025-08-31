# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AWS Cost Explorer CLI is a comprehensive Python command-line tool that enables natural language querying of AWS cost and billing data. The application integrates with multiple LLM providers (OpenAI, Anthropic, Bedrock, Ollama) to parse user queries, analyze trends, provide optimization recommendations, and format responses naturally. It includes advanced features like data export, interactive query building, cost optimization analysis, and performance monitoring.

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
  - `TrendData`, `ForecastData`: Trend analysis and forecasting models

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
  - Enhanced year parsing for full year queries (e.g., "S3 costs for 2025")
  - Consistent date parsing across all LLM providers with fallback parser support
  - Comprehensive system prompts with trend analysis and forecasting capabilities

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
  - Performance optimization options

- **`cost_optimizer.py`**: Cost optimization analysis and recommendations
  - `CostOptimizer`: Analyzes cost data for optimization opportunities
  - `OptimizationRecommendation`: Individual recommendation data structure
  - Support for rightsizing, reserved instances, savings plans analysis
  - Cost anomaly detection and budget variance analysis

- **`data_exporter.py`**: Multi-format data export capabilities
  - `ExportManager`: Coordinates different export formats
  - `CSVExporter`, `JSONExporter`, `ExcelExporter`: Format-specific exporters
  - Email integration for automated report distribution
  - Template-based export formatting

- **`interactive_query_builder.py`**: Guided query construction interface
  - `InteractiveQueryBuilder`: Step-by-step query building
  - `QueryTemplate`: Pre-built query templates for common use cases
  - Query history and favorites management
  - Real-time query validation and suggestions

- **`trend_analysis.py`**: Cost trend analysis and forecasting
  - `TrendAnalyzer`: Period-over-period comparison analysis
  - `CostForecaster`: Predictive cost modeling
  - Statistical analysis (moving averages, regression)
  - Seasonal pattern detection

- **`performance.py`**: Performance optimization and monitoring
  - `PerformanceMonitor`: Query performance tracking
  - `QueryOptimizer`: Automatic query optimization
  - Parallel execution for large date ranges
  - Compression and caching optimizations

### Supporting Components

- **`cli.py`**: Main CLI interface with click framework integration
  - Command groups for query, export, optimization, and interactive modes
  - Rich terminal output formatting
  - Comprehensive error handling and user feedback

- **`exceptions.py`**: Centralized exception handling
  - Custom exception types for different error categories
  - Error message formatting and user-friendly output
  - Logging integration for debugging

- **`date_utils.py`**: Date parsing and manipulation utilities
  - Natural language date parsing
  - Business calendar support
  - Time zone handling

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
   - Optional: `trend_analysis` → trend calculations and forecasts
   - `response_formatter` → formatted response
3. Export command → `data_exporter` → multi-format output files
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
- Support for complex queries including trend analysis and forecasting
- Comprehensive error handling and validation for all providers

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
- **boto3** (Bedrock): AWS Bedrock LLM services

### Export Dependencies
- **openpyxl**: Excel file export support (.xlsx format)
- Standard library: **csv**, **json**, **smtplib** (email integration)

### Development Tools
- **pytest**: Testing framework with coverage support
- **black**: Code formatting
- **flake8**: Code linting
- **mypy**: Static type checking

## CLI Commands

The application provides several command groups:

### Basic Query Commands
```bash
# Basic cost queries
aws-cost-cli query "Show me EC2 costs for last month"
aws-cost-cli query "S3 storage costs for 2025" --format detailed

# Query with specific profiles and options
aws-cost-cli query "RDS costs" --profile production --fresh
```

### Export Commands
```bash
# Export to different formats
aws-cost-cli export "EC2 costs last quarter" --format csv --output costs.csv
aws-cost-cli export "All services 2025" --format excel --output report.xlsx
aws-cost-cli export "S3 costs" --format json --email team@company.com
```

### Optimization Commands
```bash
# Cost optimization analysis
aws-cost-cli optimize --type rightsizing
aws-cost-cli optimize --type reserved_instances --service EC2
aws-cost-cli optimize --severity high --format detailed
```

### Interactive Mode
```bash
# Launch interactive query builder
aws-cost-cli interactive

# Use specific templates
aws-cost-cli interactive --template "Monthly Service Breakdown"
```

### Performance and Monitoring
```bash
# Enable performance monitoring
aws-cost-cli query "Large query" --performance --parallel
aws-cost-cli query "EC2 costs 2025" --max-chunk-days 30 --performance
```

## Recent Improvements

### Date Parsing Enhancement (Latest)
- **Fixed full year parsing consistency**: All LLM providers now correctly interpret queries like "S3 costs for 2025" as full calendar year (2025-01-01 to 2026-01-01)
- **Standardized system prompts**: Ensured all providers (OpenAI, Anthropic, Bedrock, Ollama) have consistent date parsing logic
- **Enhanced JSON templates**: Updated response templates to include all required fields for trend analysis and forecasting
- **Improved fallback parser**: Robust pattern matching for date parsing when LLM providers are unavailable

### GitHub Actions Integration
- **Automated CI/CD**: GitHub Actions setup for continuous integration and deployment
- **Automated testing**: Test suite execution on pull requests and commits
- **Code quality checks**: Automated linting, formatting, and type checking

## Documentation

The project includes comprehensive documentation:

- **EXPORT_GUIDE.md**: Complete guide to data export features and formats
- **INTERACTIVE_QUERY_BUILDER.md**: Interactive query building documentation
- **PERFORMANCE_GUIDE.md**: Performance optimization and monitoring guide
- **USER_GUIDE.md**: End-user documentation for all features
- **CONTRIBUTING.md**: Development and contribution guidelines
- **SECURITY.md**: Security practices and vulnerability reporting

## Quality Assurance

### Testing Strategy
- **Unit tests**: Comprehensive test coverage for all core components
- **Integration tests**: End-to-end testing of query processing pipeline
- **Performance tests**: Load testing for large dataset queries
- **Error handling tests**: Validation of error scenarios and edge cases

### Code Quality
- **Type safety**: Full mypy type checking with strict mode
- **Code formatting**: Black formatter for consistent code style  
- **Linting**: Flake8 for code quality and style enforcement
- **Documentation**: Comprehensive docstrings and inline comments
