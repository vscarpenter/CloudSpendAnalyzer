# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AWS Cost Explorer CLI is a comprehensive Python command-line tool that enables natural language querying of AWS cost and billing data. The application integrates with multiple LLM providers (OpenAI, Anthropic, Bedrock, Ollama, Gemini) to parse user queries, analyze trends, provide optimization recommendations, and format responses naturally. It includes advanced features like data export, interactive query building, cost optimization analysis, performance monitoring, health checks, and intelligent date formatting.

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
  - Provider implementations: `OpenAIProvider`, `AnthropicProvider`, `BedrockProvider`, `OllamaProvider`, `GeminiProvider`
  - Parses natural language queries into structured `QueryParameters`
  - Enhanced year parsing for full year queries (e.g., "S3 costs for 2025")
  - Consistent date parsing across all LLM providers with fallback parser support
  - Comprehensive system prompts with trend analysis and forecasting capabilities
  - Provider performance monitoring and health checks with failover support

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
  - Command groups for query, export, optimization, interactive, and health modes
  - Rich terminal output formatting with comprehensive progress indicators
  - Multi-provider switching support with `--llm-provider` option
  - Performance monitoring options (`--performance-metrics`, `--parallel`)
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

- **`date_formatter.py`**: Intelligent date formatting system (NEW)
  - `PeriodTypeDetector`: Automatically detects period types (single day/month/quarter/year, multi-month, custom range)
  - `FormatRules`: Template-based formatting with multiple styles (smart, verbose, compact)
  - `DateFormatter`: Main formatter with comprehensive error handling and fallback strategies
  - Fiscal year support and configurable formatting options
  - Safe formatting methods that never throw exceptions

- **`provider_factory.py`**: LLM provider factory and management (NEW)
  - `ProviderFactory`: Creates and manages LLM provider instances
  - Support for all providers: OpenAI, Anthropic, Bedrock, Ollama, Gemini
  - Provider configuration validation and status checking
  - Unified provider creation with consistent error handling

- **`health.py`**: Health monitoring and system diagnostics (NEW)
  - `HealthChecker`: Comprehensive system health monitoring
  - `SystemMetrics`: Resource utilization tracking (CPU, memory, disk)
  - AWS connectivity, cache system, and database health checks
  - LLM provider availability monitoring
  - HTTP server endpoint for external health monitoring

- **`validation.py`**: Query validation middleware (NEW)
  - `QueryValidator`: Validates queries before processing
  - Date range validation with granularity-specific limits
  - AWS service name validation and normalization
  - SQL injection and security pattern detection
  - Query complexity and cost estimation

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
- **google-generativeai**: Google Gemini integration
- **boto3** (Bedrock): AWS Bedrock LLM services
- **requests**: HTTP client for Ollama local provider

### Export Dependencies
- **openpyxl**: Excel file export support (.xlsx format)
- Standard library: **csv**, **json**, **smtplib** (email integration)

### Development Tools
- **pytest**: Testing framework with coverage support
- **black**: Code formatting
- **flake8**: Code linting
- **mypy**: Static type checking
- **psutil**: System metrics monitoring for health checks

## CLI Commands

The application provides several command groups:

### Basic Query Commands
```bash
# Basic cost queries
aws-cost-cli query "Show me EC2 costs for last month"
aws-cost-cli query "S3 storage costs for 2025" --format detailed

# Query with specific profiles and provider options
aws-cost-cli query "RDS costs" --profile production --fresh --llm-provider gemini
aws-cost-cli query "Large query" --parallel --max-chunk-days 30 --performance-metrics
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

### Health and System Commands
```bash
# System health checks
aws-cost-cli health
aws-cost-cli health --detailed

# Provider status and configuration
aws-cost-cli providers
aws-cost-cli providers --check-availability

# Performance and monitoring
aws-cost-cli query "Large query" --performance-metrics --parallel
aws-cost-cli query "EC2 costs 2025" --max-chunk-days 30 --performance-metrics
```

## Recent Improvements

### New LLM Provider Support (Latest)
- **Google Gemini Integration**: Added support for Gemini 1.5 Flash and Pro models with `GeminiProvider`
- **Provider Factory System**: Centralized provider creation and management with `ProviderFactory`
- **Multi-provider Configuration**: Enhanced config support with provider-specific settings and fallback chains
- **Provider Performance Monitoring**: Real-time health checks and performance metrics for all providers

### Advanced Date Formatting System
- **Intelligent Period Detection**: Automatic detection of period types (single day/month/quarter/year, multi-month, custom ranges)
- **Template-based Formatting**: Multiple format styles (smart, verbose, compact) with comprehensive error handling  
- **Fiscal Year Support**: Configurable fiscal year start months and quarter calculations
- **Safe Formatting**: Robust fallback mechanisms that never throw exceptions

### Health Monitoring and Diagnostics
- **Comprehensive Health Checks**: System metrics monitoring (CPU, memory, disk usage)
- **AWS Connectivity Validation**: Real-time AWS service availability checking
- **LLM Provider Health**: Provider availability and performance monitoring
- **HTTP Health Endpoints**: External monitoring support with JSON status responses

### Query Validation and Security
- **Input Validation Middleware**: Pre-processing validation for all queries
- **Security Pattern Detection**: SQL injection and malicious pattern filtering  
- **AWS Service Validation**: Service name normalization and validation
- **Date Range Optimization**: Granularity-aware date range limits and recommendations

### Enhanced Configuration System
- **Multi-provider Support**: Unified configuration for all 5 LLM providers (OpenAI, Anthropic, Bedrock, Ollama, Gemini)
- **Hierarchical Configuration**: Environment variables, config files, and defaults with proper precedence
- **Date Formatting Options**: Configurable formatting styles and fiscal year settings
- **Performance Tuning**: Parallel execution, compression, and caching configuration

### GitHub Actions Integration
- **Automated CI/CD**: GitHub Actions setup for continuous integration and deployment
- **Automated testing**: Test suite execution on pull requests and commits  
- **Code quality checks**: Automated linting, formatting, and type checking

## Documentation

The project includes comprehensive documentation:

- **USER_GUIDE.md**: Comprehensive end-user documentation for all features
- **EXPORT_GUIDE.md**: Complete guide to data export features and formats  
- **INTERACTIVE_QUERY_BUILDER.md**: Interactive query building documentation
- **PERFORMANCE_GUIDE.md**: Performance optimization and monitoring guide
- **OLLAMA_SETUP.md**: Local LLM setup guide for Ollama integration
- **CONTRIBUTING.md**: Development and contribution guidelines
- **SECURITY.md**: Security practices and vulnerability reporting
- **CHANGELOG.md**: Detailed version history and feature changes
- **OPTIMIZATION_ROADMAP.md**: Future optimization plans and architectural improvements
- **AGENTS.md**: AI agent configuration and integration documentation

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
