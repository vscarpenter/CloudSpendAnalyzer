# AWS Cost Explorer CLI - User Guide

This comprehensive guide will help you master the AWS Cost Explorer CLI tool and get the most value from your AWS cost analysis.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Installation and Setup](#installation-and-setup)
3. [Health Monitoring and System Status](#health-monitoring-and-system-status)
4. [Basic Query Patterns](#basic-query-patterns)
5. [Service-Specific Examples](#service-specific-examples)
6. [Time-Based Analysis](#time-based-analysis)
7. [Advanced Query Techniques](#advanced-query-techniques)
8. [Configuration and Optimization](#configuration-and-optimization)
9. [Enterprise Deployment](#enterprise-deployment)
10. [Troubleshooting](#troubleshooting)
11. [Best Practices](#best-practices)

## Getting Started

### Prerequisites

Before using the AWS Cost Explorer CLI, ensure you have:

1. **AWS CLI configured** with appropriate permissions
2. **Cost Explorer enabled** in your AWS account (may incur charges)
3. **LLM provider configured** (Ollama, OpenAI, Anthropic, Bedrock, or Gemini)
4. **Python 3.8+** installed

### First Query

Start with a simple query to verify everything is working:

```bash
aws-cost-cli query "What did I spend on AWS last month?"
```

This basic query will:
- Test your AWS credentials
- Verify your LLM provider configuration
- Show your overall AWS spending for the previous month

## Installation and Setup

### Standard Installation

For basic usage and development:

```bash
# Install the package with core dependencies
uv pip install -e .

# Or install with development tools
uv pip install -e .[dev]
```

### Production Installation

For production deployments, use the production-optimized dependencies:

```bash
# Install production dependencies for better performance
uv pip install -r requirements-prod.txt

# Then install the main package
uv pip install -e .
```

**Production dependencies include:**
- High-performance ASGI/WSGI servers (uvicorn, gunicorn)
- Enhanced logging and monitoring tools
- Database drivers (PostgreSQL, Redis)
- Security and SSL/TLS support
- Process management tools

## LLM Provider Configuration

The CLI supports five LLM providers with Ollama as the default for local, private processing.

### Provider Overview

| Provider | Type | API Key Required | Best For |
|----------|------|------------------|----------|
| **Ollama** | Local | ❌ No | Privacy, offline use, no costs |
| **Gemini** | Cloud | ✅ Yes | Fast, cost-effective queries |
| **OpenAI** | Cloud | ✅ Yes | Reliable, well-tested |
| **Anthropic** | Cloud | ✅ Yes | Complex analysis, safety |
| **Bedrock** | AWS | ❌ AWS Creds | AWS-native integration |

### 1. Ollama (Default - Recommended)

**Advantages:** No API keys, private, offline capable, no per-query costs
**Setup:**

```bash
# Install Ollama
brew install ollama  # macOS
# or visit https://ollama.ai/download for other platforms

# Start Ollama service
ollama serve

# Pull a model (choose one)
ollama pull llama2        # 7B model, good balance
ollama pull llama2:13b    # 13B model, better quality
ollama pull codellama     # Specialized for code

# Configure (default provider)
aws-cost-cli configure --provider ollama --model llama2
```

### 2. Google Gemini

**Advantages:** Fast responses, cost-effective, good quality, excellent for cost analysis
**Setup:**

```bash
# Get API key from https://makersuite.google.com/app/apikey
export GEMINI_API_KEY="your-api-key-here"

# Configure Gemini with default model
aws-cost-cli configure --provider gemini --model gemini-1.5-flash

# Test configuration
aws-cost-cli test-provider gemini

# Verify API key is working
aws-cost-cli query "test query" --llm-provider gemini
```

**Available Models:**
- `gemini-1.5-flash` - Fast and cost-effective (default, recommended)
- `gemini-1.5-pro` - More capable, higher cost, better for complex analysis

**Configuration Options:**
```bash
# Basic configuration (uses default model)
aws-cost-cli configure --provider gemini

# With specific model
aws-cost-cli configure --provider gemini --model gemini-1.5-pro

# With custom temperature (0.0-1.0, lower = more consistent)
aws-cost-cli configure --provider gemini --temperature 0.1

# Test different models
aws-cost-cli query "EC2 costs last month" --llm-provider gemini
```

**Gemini-Specific Features:**
- **Fast Processing:** Typically 2-3x faster than other cloud providers
- **Cost Effective:** Lower per-query costs compared to OpenAI/Anthropic
- **Good Context:** Handles complex cost queries with multiple parameters
- **JSON Parsing:** Excellent at extracting structured data from natural language

**Best Use Cases:**
- High-volume cost analysis queries
- Real-time cost monitoring dashboards
- Cost optimization analysis
- Budget tracking and alerts

### 3. OpenAI

**Advantages:** Reliable, well-documented, consistent results
**Setup:**

```bash
# Get API key from https://platform.openai.com/api-keys
export OPENAI_API_KEY="sk-your-key-here"

# Configure OpenAI
aws-cost-cli configure --provider openai --model gpt-3.5-turbo

# Test configuration
aws-cost-cli test-provider openai
```

### 4. Anthropic Claude

**Advantages:** Excellent for complex analysis, safety-focused
**Setup:**

```bash
# Get API key from https://console.anthropic.com/
export ANTHROPIC_API_KEY="sk-ant-your-key-here"

# Configure Anthropic
aws-cost-cli configure --provider anthropic --model claude-3-haiku-20240307

# Test configuration
aws-cost-cli test-provider anthropic
```

### 5. AWS Bedrock

**Advantages:** AWS-native, uses existing AWS credentials
**Setup:**

```bash
# Ensure AWS credentials are configured
aws configure

# Configure Bedrock
aws-cost-cli configure --provider bedrock --region us-east-1

# Test configuration
aws-cost-cli test-provider bedrock
```

### Provider Management Commands

```bash
# List all providers and their status
aws-cost-cli list-providers

# Test specific provider
aws-cost-cli test-provider gemini
aws-cost-cli test-provider ollama
aws-cost-cli test-provider openai

# Override provider for single query (doesn't change config)
aws-cost-cli query "EC2 costs" --llm-provider gemini
aws-cost-cli query "EC2 costs" --llm-provider ollama

# Check current configuration
aws-cost-cli config show

# Change default provider permanently
aws-cost-cli configure --provider gemini
aws-cost-cli configure --provider ollama
```

### Advanced Provider Switching

**Scenario-Based Provider Selection:**

```bash
# For privacy-sensitive queries (local processing)
aws-cost-cli query "detailed cost breakdown" --llm-provider ollama

# For fast, cost-effective queries (cloud)
aws-cost-cli query "monthly spending summary" --llm-provider gemini

# For complex analysis requiring reasoning
aws-cost-cli query "cost optimization recommendations" --llm-provider anthropic

# For AWS-native integration
aws-cost-cli query "cross-account cost analysis" --llm-provider bedrock

# For reliable, consistent results
aws-cost-cli query "budget variance analysis" --llm-provider openai
```

**Provider Performance Comparison:**

```bash
# Compare response times across providers
time aws-cost-cli query "EC2 costs last month" --llm-provider ollama
time aws-cost-cli query "EC2 costs last month" --llm-provider gemini
time aws-cost-cli query "EC2 costs last month" --llm-provider openai

# Test provider availability
aws-cost-cli test-provider ollama && echo "Ollama: Available"
aws-cost-cli test-provider gemini && echo "Gemini: Available"
aws-cost-cli test-provider openai && echo "OpenAI: Available"
```

**Batch Provider Testing:**

```bash
# Test all configured providers
for provider in ollama gemini openai anthropic bedrock; do
  echo "Testing $provider..."
  aws-cost-cli test-provider $provider
done

# Find fastest provider for your setup
echo "Provider speed test:"
for provider in ollama gemini openai; do
  echo -n "$provider: "
  time aws-cost-cli query "test" --llm-provider $provider >/dev/null 2>&1
done
```

### Fallback Configuration

Configure automatic fallback when your primary provider fails:

```yaml
# In your config file
llm_provider: "gemini"
fallback_providers:
  - "ollama"    # Local fallback
  - "openai"    # Cloud fallback
  - "anthropic" # Additional fallback
```

## CLI Command Reference

### Core Commands

#### Query Command

The main command for asking cost-related questions:

```bash
# Basic usage
aws-cost-cli query "What did I spend on EC2 last month?"

# With provider override
aws-cost-cli query "S3 costs this year" --llm-provider gemini

# With output format
aws-cost-cli query "RDS costs" --format json

# With AWS profile
aws-cost-cli query "Lambda costs" --profile production

# With performance options
aws-cost-cli query "All costs 2024" --parallel --compression --performance-metrics
```

**Query Command Options:**
- `--llm-provider`: Override LLM provider (ollama, gemini, openai, anthropic, bedrock)
- `--format`: Output format (simple, rich, json, llm)
- `--profile`: AWS profile to use
- `--parallel`: Enable parallel query execution
- `--no-parallel`: Disable parallel execution
- `--compression`: Enable cache compression
- `--no-compression`: Disable cache compression
- `--performance-metrics`: Show performance metrics after query

#### Configuration Commands

```bash
# Configure LLM provider
aws-cost-cli configure --provider gemini --model gemini-1.5-flash

# Configure with API key
aws-cost-cli configure --provider openai --api-key sk-your-key

# Configure AWS profile
aws-cost-cli configure --profile production

# Configure cache settings
aws-cost-cli configure --cache-ttl 3600

# Show current configuration
aws-cost-cli config show

# Validate configuration
aws-cost-cli config validate
```

#### Provider Management Commands

```bash
# List all providers and their status
aws-cost-cli list-providers

# Test specific provider
aws-cost-cli test-provider gemini
aws-cost-cli test-provider ollama

# Test all providers
aws-cost-cli test
```

#### Health and Monitoring Commands

```bash
# Basic health check
aws-cost-cli health check

# Detailed health check
aws-cost-cli health check --detailed

# Readiness check (for containers)
aws-cost-cli health ready

# Start health check server
aws-cost-cli health serve --port 8081

# Performance monitoring
aws-cost-cli performance
aws-cost-cli performance --hours 24 --format json
```

#### Cache Management Commands

```bash
# Show cache status
aws-cost-cli cache status

# Clear cache
aws-cost-cli cache clear

# Clear cache for specific queries
aws-cost-cli cache clear --pattern "EC2*"
```

#### Profile Management Commands

```bash
# List AWS profiles
aws-cost-cli list-profiles

# Set default profile
aws-cost-cli configure --default-profile production
```

### Command Examples by Use Case

#### Daily Cost Monitoring

```bash
# Morning cost check
aws-cost-cli query "How much have I spent on AWS today?"

# Weekly summary
aws-cost-cli query "What was my AWS spending this week compared to last week?"

# Monthly budget check
aws-cost-cli query "Am I on track for my monthly AWS budget of $5000?"
```

#### Cost Analysis and Optimization

```bash
# Service breakdown
aws-cost-cli query "What are my top 10 AWS services by cost this month?"

# Cost trends
aws-cost-cli query "Show me monthly cost trends for the last 6 months"

# Optimization opportunities
aws-cost-cli query "What are my biggest cost optimization opportunities?"

# Regional analysis
aws-cost-cli query "Compare AWS costs between us-east-1 and eu-west-1"
```

#### Provider-Specific Workflows

```bash
# Privacy-focused analysis (local processing)
aws-cost-cli query "Detailed cost breakdown with sensitive data" --llm-provider ollama

# Fast cloud analysis
aws-cost-cli query "Quick monthly summary" --llm-provider gemini

# Complex analysis requiring reasoning
aws-cost-cli query "Cost optimization strategy recommendations" --llm-provider anthropic

# AWS-native integration
aws-cost-cli query "Cross-account cost analysis" --llm-provider bedrock
```

#### Troubleshooting and Testing

```bash
# Test system health
aws-cost-cli health check --detailed

# Test all providers
for provider in ollama gemini openai anthropic bedrock; do
  echo "Testing $provider..."
  aws-cost-cli test-provider $provider
done

# Debug query issues
aws-cost-cli query "test query" --debug --llm-provider gemini

# Performance comparison
time aws-cost-cli query "EC2 costs" --llm-provider ollama
time aws-cost-cli query "EC2 costs" --llm-provider gemini
```

### Environment Variables Reference

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `AWS_COST_CLI_LLM_PROVIDER` | Default LLM provider | `ollama` | `gemini` |
| `AWS_COST_CLI_DEFAULT_PROFILE` | Default AWS profile | None | `production` |
| `AWS_COST_CLI_CACHE_TTL` | Cache TTL in seconds | `3600` | `7200` |
| `AWS_COST_CLI_OUTPUT_FORMAT` | Default output format | `simple` | `rich` |
| `GEMINI_API_KEY` | Google Gemini API key | None | `your-api-key` |
| `OPENAI_API_KEY` | OpenAI API key | None | `sk-your-key` |
| `ANTHROPIC_API_KEY` | Anthropic API key | None | `sk-ant-your-key` |

### Exit Codes

The CLI returns standard exit codes for scripting:

- `0`: Success
- `1`: General error (invalid arguments, configuration issues)
- `2`: AWS API error (credentials, permissions, service issues)
- `3`: LLM provider error (API key, quota, network issues)
- `4`: Cache error (disk space, permissions)
- `5`: Health check failure (system unhealthy)

### Development vs Production Dependencies

| Dependency Type | Development | Production |
|----------------|-------------|------------|
| **Core Features** | ✅ Basic functionality | ✅ All features + optimizations |
| **Performance** | Standard | High-performance servers & pooling |
| **Monitoring** | Basic logging | Structured logging + metrics |
| **Database** | File-based cache | PostgreSQL + Redis support |
| **Security** | Standard | Enhanced SSL/TLS + encryption |
| **Deployment** | Local development | Production-ready servers |

## Health Monitoring and System Status

The CLI includes comprehensive health monitoring capabilities for production deployments and troubleshooting.

### Basic Health Checks

Check the overall system health:

```bash
# Quick health check
aws-cost-cli health check

# Detailed health check with system metrics
aws-cost-cli health check --detailed

# JSON output for monitoring systems
aws-cost-cli health check --json
```

**Example output:**
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ ✅ System Status: HEALTHY                                                      ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Component     ┃ Status     ┃ Details                                                          ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ System        │ ✅ healthy │ OK                                                               │
│ Aws           │ ✅ healthy │ Response: 245ms                                                  │
│ Cache         │ ✅ healthy │ Size: 15.2MB; Response: 2ms                                     │
│ Llm           │ ✅ healthy │ OK                                                               │
└───────────────┴────────────┴──────────────────────────────────────────────────────────────┘

📊 Summary: 4/4 checks healthy
⏱️  Uptime: 45.2 seconds
```

### Readiness Checks

For container orchestration and load balancers:

```bash
# Check if application is ready to serve requests
aws-cost-cli health ready

# JSON output for Kubernetes probes
aws-cost-cli health ready --json
```

### Health Check Server

Start an HTTP server for monitoring endpoints (ideal for containers):

```bash
# Start health check server
aws-cost-cli health serve --port 8081

# Bind to specific host
aws-cost-cli health serve --host 0.0.0.0 --port 8081
```

**Available endpoints:**
- `GET /health` - Basic health check
- `GET /health/detailed` - Detailed health with metrics
- `GET /ready` - Readiness probe
- `GET /metrics` - Prometheus metrics

### Monitoring Integration

#### Docker/Kubernetes Health Checks

```yaml
# Kubernetes deployment example
spec:
  containers:
  - name: aws-cost-cli
    image: your-registry/aws-cost-cli:latest
    ports:
    - containerPort: 8081
    livenessProbe:
      httpGet:
        path: /health
        port: 8081
      initialDelaySeconds: 30
      periodSeconds: 10
    readinessProbe:
      httpGet:
        path: /ready
        port: 8081
      initialDelaySeconds: 5
      periodSeconds: 5
```

#### Prometheus Monitoring

```yaml
# Prometheus scrape config
scrape_configs:
  - job_name: 'aws-cost-cli'
    static_configs:
      - targets: ['your-service:8081']
    metrics_path: '/metrics'
    scrape_interval: 30s
```

### Health Check Components

The health monitoring system checks:

| Component | Description | Healthy State |
|-----------|-------------|---------------|
| **System** | CPU, memory, disk usage | < 80% utilization |
| **AWS** | API connectivity and permissions | < 1s response time |
| **Cache** | File system access and performance | Read/write operations work |
| **LLM** | Provider API key availability | API keys configured |
| **Database** | Connection and query performance | < 100ms queries |

### Status Codes

Health check commands return appropriate exit codes:
- `0` - Healthy
- `1` - Unhealthy (critical issues)
- `2` - Degraded (warnings, but functional)

## Basic Query Patterns

### Understanding Query Structure

The CLI accepts natural language queries, but certain patterns work better:

**Good Query Patterns:**
- "How much did I spend on [SERVICE] [TIME_PERIOD]?"
- "Show me [SERVICE] costs for [TIME_PERIOD]"
- "What are my top spending services [TIME_PERIOD]?"
- "Compare [SERVICE1] vs [SERVICE2] costs [TIME_PERIOD]"

**Time Period Examples:**
- "last month", "this month", "last 3 months"
- "2024", "last year", "this year"
- "January 2024", "Q1 2024"
- "last 30 days", "last week"

### Service Breakdown Queries

Get detailed breakdowns of your AWS spending by service:

```bash
# List all services you used with costs
aws-cost-cli query "What services did I use last month in my AWS account that cost me money?"

# Alternative phrasing for service breakdown
aws-cost-cli query "Can you list the services I used last month that cost me money?"

# Show service breakdown for specific time periods
aws-cost-cli query "Show me all AWS services I used this year with their costs"

# Service breakdown with different time ranges
aws-cost-cli query "Which services did I spend money on last quarter?"
```

**Example Output:**
```
AWS Cost Summary (2025-07-01 to 2025-08-01)
===========================================

Total Cost: $26.32

Service Breakdown:
--------------------
  Amazon Elastic Compute Cloud - Compute: $15.50
  Amazon Simple Storage Service: $8.25
  AWS Lambda: $2.57
```

### Output Formats

Control how results are displayed:

```bash
# Simple format (default) - clean text output
aws-cost-cli query "What did I spend on AWS last month?"

# Rich format - enhanced terminal output with colors and formatting
aws-cost-cli query "What did I spend on AWS last month?" --format rich

# LLM format - natural language response generated by AI
aws-cost-cli query "What did I spend on AWS last month?" --format llm

# JSON format - structured data for scripting and automation
aws-cost-cli query "What did I spend on AWS last month?" --format json
```

**Note:** If a query returns $0.00, it may indicate:
- No costs for the specified service/time period
- The service name needs to be more specific (try "What did I spend on compute services?" instead of "EC2 costs")
- Your AWS account doesn't have Cost Explorer data for that period

## Service-Specific Examples

### 1. Amazon EC2 (Elastic Compute Cloud)

```bash
# Basic EC2 spending
aws-cost-cli query "What did I spend on EC2 last month?"

# Alternative if EC2 doesn't work
aws-cost-cli query "What did I spend on compute services last month?"

# EC2 instance type breakdown
aws-cost-cli query "Show me compute costs by instance type for 2024"

# EC2 regional analysis
aws-cost-cli query "What are my compute costs by region last quarter?"

# EC2 vs EBS comparison
aws-cost-cli query "Compare compute costs vs storage costs this year"
```

### 2. Amazon S3 (Simple Storage Service)

```bash
# S3 storage costs
aws-cost-cli query "What did I spend on S3 storage last month?"

# S3 data transfer costs
aws-cost-cli query "Show me S3 data transfer costs for the last 6 months"

# S3 storage class breakdown
aws-cost-cli query "Break down my S3 costs by storage class this year"

# S3 vs Glacier comparison
aws-cost-cli query "Compare S3 Standard vs Glacier costs annually"
```

### 3. Amazon RDS (Relational Database Service)

```bash
# RDS database costs
aws-cost-cli query "How much am I spending on RDS databases monthly?"

# RDS engine comparison
aws-cost-cli query "Compare MySQL vs PostgreSQL RDS costs this year"

# RDS instance size analysis
aws-cost-cli query "Show RDS costs by instance size last quarter"

# RDS backup and snapshot costs
aws-cost-cli query "What are my RDS backup and snapshot costs this month?"
```

### 4. AWS Lambda

```bash
# Lambda execution costs
aws-cost-cli query "What did I spend on Lambda last month?"

# Lambda vs EC2 cost comparison
aws-cost-cli query "Compare Lambda vs EC2 costs for compute workloads this year"

# Lambda request and duration breakdown
aws-cost-cli query "Break down Lambda costs by requests vs duration last quarter"
```

### 5. Amazon CloudFront

```bash
# CloudFront distribution costs
aws-cost-cli query "Show me CloudFront costs for the last 3 months"

# CloudFront data transfer analysis
aws-cost-cli query "What are my CloudFront data transfer costs by region this year?"

# CloudFront vs direct S3 serving
aws-cost-cli query "Compare CloudFront vs direct S3 serving costs annually"
```

### 6. Amazon EKS (Elastic Kubernetes Service)

```bash
# EKS cluster costs
aws-cost-cli query "How much am I spending on EKS clusters monthly?"

# EKS vs self-managed Kubernetes costs
aws-cost-cli query "What are the cost differences between EKS and EC2-based Kubernetes?"

# EKS node group analysis
aws-cost-cli query "Break down EKS costs by node groups last month"
```

### 7. Amazon Redshift

```bash
# Redshift cluster costs
aws-cost-cli query "What did I spend on Redshift last quarter?"

# Redshift vs RDS cost comparison
aws-cost-cli query "Compare Redshift vs RDS costs for data warehousing this year"

# Redshift reserved vs on-demand
aws-cost-cli query "Show Redshift reserved instance vs on-demand costs"
```

### 8. Amazon ElastiCache

```bash
# ElastiCache costs
aws-cost-cli query "How much am I spending on ElastiCache monthly?"

# Redis vs Memcached comparison
aws-cost-cli query "Compare ElastiCache Redis vs Memcached costs this year"

# ElastiCache node type analysis
aws-cost-cli query "Break down ElastiCache costs by node type last month"
```

### 9. Amazon API Gateway

```bash
# API Gateway costs
aws-cost-cli query "What are my API Gateway costs last month?"

# API Gateway request volume analysis
aws-cost-cli query "Show API Gateway costs by request volume this quarter"

# REST vs HTTP API cost comparison
aws-cost-cli query "Compare REST API vs HTTP API costs in API Gateway"
```

### 10. Amazon DynamoDB

```bash
# DynamoDB costs
aws-cost-cli query "How much did I spend on DynamoDB last month?"

# DynamoDB on-demand vs provisioned
aws-cost-cli query "Compare DynamoDB on-demand vs provisioned capacity costs this year"

# DynamoDB read vs write costs
aws-cost-cli query "Break down DynamoDB costs by read vs write operations"
```

## Time-Based Analysis

### Annual Spending Analysis

```bash
# Total annual spending
aws-cost-cli query "What was my total AWS spending for 2024?"

# Year-over-year comparison
aws-cost-cli query "Compare my AWS spending between 2023 and 2024"

# Annual spending by service
aws-cost-cli query "Show me my top 10 AWS services by annual spending in 2024"

# Annual growth analysis
aws-cost-cli query "Which services had the highest cost growth in 2024?"

# Annual cost optimization opportunities
aws-cost-cli query "What are my biggest cost optimization opportunities based on 2024 spending?"
```

### Monthly Spending Patterns

```bash
# Current month spending
aws-cost-cli query "How much have I spent on AWS this month so far?"

# Month-over-month comparison
aws-cost-cli query "Compare this month's AWS spending to last month"

# Monthly spending trend
aws-cost-cli query "Show my monthly AWS spending trend for the last 12 months"

# Seasonal spending patterns
aws-cost-cli query "What are my seasonal AWS spending patterns over the last 2 years?"

# Monthly budget tracking
aws-cost-cli query "Am I on track to meet my monthly AWS budget of $5000?"
```

### Quarterly Analysis

```bash
# Quarterly spending summary
aws-cost-cli query "What was my AWS spending for Q4 2024?"

# Quarter-over-quarter growth
aws-cost-cli query "Compare Q4 2024 spending to Q3 2024"

# Quarterly service breakdown
aws-cost-cli query "Show me quarterly spending breakdown by service for 2024"

# Quarterly cost trends
aws-cost-cli query "What are the quarterly cost trends for my top 5 AWS services?"
```

### Daily and Weekly Analysis

```bash
# Daily spending patterns
aws-cost-cli query "Show me daily AWS spending for the last 30 days"

# Weekly spending trends
aws-cost-cli query "What are my weekly AWS spending patterns this month?"

# Weekend vs weekday costs
aws-cost-cli query "Compare weekend vs weekday AWS costs last month"

# Daily cost spikes
aws-cost-cli query "Which days had the highest AWS costs last month?"
```

## Advanced Query Techniques

### Multi-Service Comparisons

```bash
# Top spending services
aws-cost-cli query "What are my top 10 AWS services by cost this year?"

# Service cost distribution
aws-cost-cli query "Show me the percentage breakdown of costs by AWS service"

# Compute services comparison
aws-cost-cli query "Compare costs between EC2, Lambda, and EKS this quarter"

# Storage services comparison
aws-cost-cli query "Compare S3, EBS, and EFS storage costs annually"
```

### Regional Analysis

```bash
# Multi-region cost breakdown
aws-cost-cli query "Show me AWS costs by region for the last 6 months"

# Regional cost comparison
aws-cost-cli query "Compare costs between us-east-1 and eu-west-1 this year"

# Data transfer costs by region
aws-cost-cli query "What are my inter-region data transfer costs?"

# Regional optimization opportunities
aws-cost-cli query "Which regions offer the best cost optimization opportunities?"
```

### Cost Optimization Queries

```bash
# Unused resources identification
aws-cost-cli query "Help me identify potentially unused AWS resources based on cost patterns"

# Reserved instance analysis
aws-cost-cli query "Show me potential savings from reserved instances"

# Right-sizing opportunities
aws-cost-cli query "What are my EC2 right-sizing opportunities based on cost analysis?"

# Cost anomaly detection
aws-cost-cli query "Were there any unusual cost spikes last month?"
```

### Budget and Forecasting

```bash
# Cost forecasting
aws-cost-cli query "Based on current trends, what will my AWS costs be next month?"

# Budget variance analysis
aws-cost-cli query "How does my actual spending compare to my $10000 monthly budget?"

# Cost projection
aws-cost-cli query "Project my annual AWS costs based on the last 6 months"

# Service growth forecasting
aws-cost-cli query "Which services are likely to drive cost growth next quarter?"
```

## Enterprise Deployment

The AWS Cost CLI provides enterprise-grade configuration templates and deployment options for large organizations.

### Configuration Templates

Pre-configured templates are available for common enterprise scenarios:

```bash
# List available templates
ls config/templates/

# Available templates:
# - development.yaml    - Development environment
# - production.yaml     - Production deployment  
# - multi-account.yaml  - Multi-account setup
# - organization.yaml   - AWS Organizations
```

### Development Environment Setup

For development and testing:

```bash
# Copy development template
cp config/templates/development.yaml ~/.aws-cost-cli/config.yaml

# Edit for your environment
nano ~/.aws-cost-cli/config.yaml
```

**Development template features:**
- Cost-effective LLM provider settings (GPT-3.5-turbo)
- Longer cache TTL (2 hours) to reduce API calls
- Verbose debug logging
- Relaxed security settings
- Local file-based storage

### Production Environment Setup

For production deployments:

```bash
# Copy production template
cp config/templates/production.yaml ~/.aws-cost-cli/config.yaml

# Install production dependencies
pip install -r requirements-prod.txt

# Set required environment variables
export AWS_COST_CLI_DB_PASSWORD="your-secure-password"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

**Production template features:**
- High-performance connection pooling (100+ connections)
- PostgreSQL database integration
- Comprehensive health checks and monitoring
- Circuit breaker patterns for resilience
- Structured JSON logging
- Resource limits and security controls

### Multi-Account Enterprise Setup

For organizations with multiple AWS accounts:

```bash
# Copy multi-account template
cp config/templates/multi-account.yaml ~/.aws-cost-cli/config.yaml

# Configure account-specific roles
# Edit the accounts section in the config file
```

**Multi-account template example:**
```yaml
accounts:
  - name: "Production"
    profile: "prod-account"  
    account_id: "123456789012"
    role_arn: "arn:aws:iam::123456789012:role/CostExplorerRole"
    
  - name: "Development"
    profile: "dev-account"
    account_id: "123456789013" 
    role_arn: "arn:aws:iam::123456789013:role/CostExplorerRole"

performance:
  parallel_account_queries: true
  max_concurrent_requests: 3
```

### AWS Organizations Setup

For large enterprises using AWS Organizations:

```bash
# Copy organization template
cp config/templates/organization.yaml ~/.aws-cost-cli/config.yaml

# Configure database (required for organization scale)
export AWS_COST_CLI_DB_PASSWORD="your-db-password"
```

**Organization template features:**
- AWS Organizations integration
- Organizational Unit (OU) cost analysis  
- Enterprise-grade PostgreSQL database
- Advanced security and audit logging
- Cost allocation tag support
- Integration with enterprise tools (Slack, JIRA)

### Template Comparison

| Feature | Development | Production | Multi-Account | Organization |
|---------|-------------|------------|---------------|--------------|
| **Database** | File-based | PostgreSQL | File/Optional DB | PostgreSQL |
| **Caching** | 2 hours | 15 minutes | 30 minutes | 1 hour |
| **Security** | Relaxed | Strict | Strict | Enterprise |
| **Monitoring** | Basic | Advanced | Advanced | Enterprise |
| **Scale** | Single user | Team/Dept | Multiple accounts | Enterprise org |

### Configuration Validation

After setting up your configuration:

```bash
# Validate configuration syntax
aws-cost-cli config validate

# Test database connectivity (if enabled)
aws-cost-cli config test-db

# Test LLM provider connectivity  
aws-cost-cli config test-llm

# Comprehensive system test
aws-cost-cli health check --detailed
```

### Security Best Practices

#### Environment Variables

Never commit sensitive information to version control:

```bash
# Database credentials
export AWS_COST_CLI_DB_PASSWORD="secure-password"

# LLM API keys (choose providers you want to use)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="your-gemini-api-key"

# AWS credentials (if not using profiles)
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
```

#### File Permissions

Secure your configuration files:

```bash
# Set restrictive permissions
chmod 600 ~/.aws-cost-cli/config.yaml

# For production deployments
chown app:app /etc/aws-cost-cli/config.yaml
chmod 640 /etc/aws-cost-cli/config.yaml
```

#### Network Security

For production environments:
- Use VPC endpoints for AWS API calls
- Configure firewall rules for database access
- Use TLS/SSL for all external connections
- Enable audit logging for compliance

### Container Deployment

#### Docker Example

```dockerfile
FROM python:3.11-slim

# Install production dependencies
COPY requirements-prod.txt .
RUN pip install -r requirements-prod.txt

# Install application
COPY . /app
WORKDIR /app
RUN pip install -e .

# Health check configuration
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
  CMD aws-cost-cli health ready || exit 1

# Expose health check port
EXPOSE 8081

# Start health check server
CMD ["aws-cost-cli", "health", "serve", "--host", "0.0.0.0", "--port", "8081"]
```

#### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aws-cost-cli
spec:
  replicas: 2
  selector:
    matchLabels:
      app: aws-cost-cli
  template:
    metadata:
      labels:
        app: aws-cost-cli
    spec:
      containers:
      - name: aws-cost-cli
        image: your-registry/aws-cost-cli:latest
        ports:
        - containerPort: 8081
        env:
        - name: AWS_COST_CLI_DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: aws-cost-cli-secrets
              key: db-password
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: aws-cost-cli-secrets  
              key: anthropic-api-key
        - name: GEMINI_API_KEY
          valueFrom:
            secretKeyRef:
              name: aws-cost-cli-secrets  
              key: gemini-api-key
        livenessProbe:
          httpGet:
            path: /health
            port: 8081
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8081
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
```

### Monitoring and Alerting

#### Prometheus Integration

```yaml
# ServiceMonitor for Prometheus Operator
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: aws-cost-cli
spec:
  selector:
    matchLabels:
      app: aws-cost-cli
  endpoints:
  - port: health
    path: /metrics
    interval: 30s
```

#### Grafana Dashboard

Key metrics to monitor:
- Query response times
- Cache hit ratios
- AWS API call frequency
- System resource usage
- Error rates by component

## Configuration and Optimization

### Profile Management

```bash
# List available AWS profiles
aws-cost-cli list-profiles

# Use specific AWS profile
aws-cost-cli query "EC2 costs last month" --profile production

# Set default profile
aws-cost-cli configure --default-profile production
```

### Cache Management

```bash
# Clear cache for fresh data
aws-cost-cli cache clear

# Check cache status
aws-cost-cli cache status

# Set custom cache TTL (in seconds)
aws-cost-cli configure --cache-ttl 7200  # 2 hours
```

### LLM Provider Optimization

```bash
# Default provider (local, no API key required)
aws-cost-cli configure --provider ollama --model llama2

# Switch LLM providers for different use cases
aws-cost-cli configure --provider gemini     # Fast and cost-effective cloud option
aws-cost-cli configure --provider anthropic  # Better for complex analysis
aws-cost-cli configure --provider openai     # Reliable cloud option
aws-cost-cli configure --provider bedrock    # For AWS-native integration

# Test LLM provider performance
aws-cost-cli test-provider ollama    # Test local provider
aws-cost-cli test-provider gemini    # Test Gemini provider
aws-cost-cli test-provider openai    # Test OpenAI provider

# Override provider for single query
aws-cost-cli query "EC2 costs last month" --llm-provider gemini
```

## Troubleshooting

### System Diagnostics

Start troubleshooting with comprehensive health checks:

```bash
# Run comprehensive system health check
aws-cost-cli health check --detailed

# Check if system is ready to serve requests  
aws-cost-cli health ready

# Get system status in JSON format for analysis
aws-cost-cli health check --json
```

### Common Issues and Solutions

#### 1. System Health Issues

```bash
# Check overall system health first
aws-cost-cli health check

# Identify specific component failures
aws-cost-cli health check --detailed --json | jq '.checks'

# Monitor system resources
aws-cost-cli health check --detailed | grep -E "(cpu|memory|disk)"
```

**Common system issues:**
- **High CPU/Memory**: Reduce concurrent queries or increase system resources
- **Cache issues**: Check disk space and permissions in `~/.aws-cost-cli/cache`
- **Network connectivity**: Verify internet connection and firewall settings

#### 2. Authentication Errors

```bash
# Verify AWS credentials
aws sts get-caller-identity

# Check AWS profile configuration
aws-cost-cli list-profiles

# Test with specific profile
aws-cost-cli query "test query" --profile your-profile-name

# Run health check to verify AWS connectivity
aws-cost-cli health check | grep -i aws
```

#### 3. LLM Provider Issues

**General Provider Troubleshooting:**

```bash
# Test LLM connectivity
aws-cost-cli test

# Check API key configuration
aws-cost-cli config show

# List all providers and their status
aws-cost-cli list-providers

# Check LLM provider status in health check
aws-cost-cli health check | grep -i llm
```

**Provider-Specific Troubleshooting:**

##### Ollama (Local Provider - Default)

```bash
# Check if Ollama is running
curl http://localhost:11434/api/version

# Start Ollama if not running
ollama serve

# Check available models
ollama list

# Download a model if none available
ollama pull llama2

# Test Ollama provider
aws-cost-cli test-provider ollama

# Configure Ollama with specific model
aws-cost-cli configure --provider ollama --model llama2

# Troubleshoot Ollama connection issues
ps aux | grep ollama  # Check if process is running
netstat -an | grep 11434  # Check if port is open
```

**Common Ollama Issues:**
- **"Connection refused"**: Ollama service not running → Run `ollama serve`
- **"Model not found"**: No models downloaded → Run `ollama pull llama2`
- **Slow responses**: Model too large for hardware → Use smaller model like `llama2:7b-chat`
- **High memory usage**: Large model loaded → Use smaller model or increase system RAM

##### Google Gemini

```bash
# Check if API key is set
echo $GEMINI_API_KEY

# Set API key if missing
export GEMINI_API_KEY="your-api-key-here"

# Test Gemini provider
aws-cost-cli test-provider gemini

# Configure Gemini
aws-cost-cli configure --provider gemini --model gemini-1.5-flash

# Test with specific query
aws-cost-cli query "test query" --llm-provider gemini
```

**Common Gemini Issues:**
- **"Invalid API key"**: Wrong/expired key → Get new key from [Google AI Studio](https://makersuite.google.com/app/apikey)
- **"Package not installed"**: Missing dependency → Run `pip install google-generativeai>=0.3.0`
- **"Quota exceeded"**: API limits reached → Check usage in Google AI Studio or switch provider
- **"Model not found"**: Invalid model name → Use `gemini-1.5-flash` or `gemini-1.5-pro`

##### OpenAI

```bash
# Check if API key is set
echo $OPENAI_API_KEY

# Set API key if missing
export OPENAI_API_KEY="sk-your-key-here"

# Test OpenAI provider
aws-cost-cli test-provider openai

# Configure OpenAI
aws-cost-cli configure --provider openai --model gpt-3.5-turbo
```

**Common OpenAI Issues:**
- **"Invalid API key"**: Wrong/expired key → Get new key from [OpenAI Platform](https://platform.openai.com/api-keys)
- **"Quota exceeded"**: Usage limits reached → Check billing in OpenAI dashboard
- **"Model not found"**: Invalid model → Use `gpt-3.5-turbo` or `gpt-4`
- **"Rate limit exceeded"**: Too many requests → Wait or upgrade plan

##### Anthropic Claude

```bash
# Check if API key is set
echo $ANTHROPIC_API_KEY

# Set API key if missing
export ANTHROPIC_API_KEY="sk-ant-your-key-here"

# Test Anthropic provider
aws-cost-cli test-provider anthropic

# Configure Anthropic
aws-cost-cli configure --provider anthropic --model claude-3-haiku-20240307
```

**Common Anthropic Issues:**
- **"Invalid API key"**: Wrong/expired key → Get new key from [Anthropic Console](https://console.anthropic.com/)
- **"Usage limit exceeded"**: Monthly limits reached → Check usage in Anthropic console
- **"Model not available"**: Invalid model → Use `claude-3-haiku-20240307` or `claude-3-sonnet-20240229`

##### AWS Bedrock

```bash
# Check AWS credentials
aws sts get-caller-identity

# Test Bedrock provider
aws-cost-cli test-provider bedrock

# Configure Bedrock
aws-cost-cli configure --provider bedrock --region us-east-1

# Check Bedrock permissions
aws bedrock list-foundation-models --region us-east-1
```

**Common Bedrock Issues:**
- **"Access denied"**: Missing IAM permissions → Add `bedrock:InvokeModel` permission
- **"Region not supported"**: Bedrock not available → Use `us-east-1` or `us-west-2`
- **"Model not found"**: Invalid model ID → Use `anthropic.claude-3-haiku-20240307-v1:0`
- **"Credentials not found"**: AWS credentials not configured → Run `aws configure`

**Provider Fallback Troubleshooting:**

```bash
# Test fallback configuration
aws-cost-cli config show | grep -A5 fallback

# Test each fallback provider
aws-cost-cli test-provider ollama
aws-cost-cli test-provider gemini
aws-cost-cli test-provider openai

# Force fallback by disabling primary provider
aws-cost-cli query "test" --llm-provider nonexistent  # Should fallback

# Configure robust fallback chain
aws-cost-cli configure --provider gemini
# Then edit config file to add:
# fallback_providers: ["ollama", "openai", "anthropic"]
```

#### 4. Cost Explorer Access

```bash
# Verify Cost Explorer permissions
aws ce get-cost-and-usage --time-period Start=2024-01-01,End=2024-01-02 --granularity MONTHLY --metrics BlendedCost

# Check if Cost Explorer is enabled
aws-cost-cli query "test access to cost data"
```

#### 5. Query Understanding Issues

```bash
# Use more specific queries with full context
aws-cost-cli query "What did I spend on EC2 last month?"  # Instead of "EC2 costs"

# Try alternative phrasing for better results
aws-cost-cli query "What did I spend on compute services last month?"

# Use complete questions rather than fragments
aws-cost-cli query "What are my S3 costs for last month?" --format rich
```

#### 6. Zero Cost Results ($0.00)

If you're getting $0.00 when you expect costs:

```bash
# Check if you have any AWS costs at all
aws-cost-cli query "What did I spend on AWS last month?"

# Try broader service categories
aws-cost-cli query "What did I spend on compute services last month?"  # Instead of specific "EC2"

# Check different time periods
aws-cost-cli query "What did I spend on AWS this year?"

# Verify with service breakdown
aws-cost-cli query "What services did I use last month that cost me money?"
```

### Debug Mode

```bash
# Enable verbose logging
aws-cost-cli query "EC2 costs" --debug

# Check configuration
aws-cost-cli config show --debug

# Test all components
aws-cost-cli test --debug
```

## Best Practices

### 1. Query Optimization

- **Be specific** with time periods and services
- **Use consistent terminology** (e.g., "last month" vs "previous month")
- **Start simple** and add complexity gradually
- **Cache frequently used queries** by running them regularly

### 2. Cost Analysis Workflow

1. **Start with overview**: "What did I spend on AWS last month?"
2. **Identify top services**: "What are my top 5 AWS services by cost?"
3. **Drill down**: "Show me EC2 costs by instance type"
4. **Compare periods**: "Compare this month to last month"
5. **Look for optimization**: "What are my cost optimization opportunities?"

### 3. Regular Monitoring

#### Cost Monitoring
```bash
# Daily cost check
aws-cost-cli query "How much have I spent on AWS today?"

# Weekly summary
aws-cost-cli query "What was my AWS spending this week?"

# Monthly review
aws-cost-cli query "Show me monthly spending trends and top services"

# Quarterly planning
aws-cost-cli query "What are my quarterly cost trends and forecasts?"
```

#### System Health Monitoring
```bash
# Daily health check (add to cron)
aws-cost-cli health check --json > /var/log/aws-cost-cli-health.log

# Production monitoring with alerting
aws-cost-cli health check || echo "Health check failed" | mail -s "AWS Cost CLI Alert" ops@company.com

# Container readiness monitoring
aws-cost-cli health ready || exit 1
```

### 4. Security and Privacy

- **Use environment variables** for API keys (never commit secrets)
- **Rotate API keys** regularly and use secure storage
- **Use Ollama** for sensitive environments requiring local processing
- **Limit AWS permissions** to Cost Explorer only (principle of least privilege)
- **Review query logs** periodically for suspicious activity
- **Secure configuration files** with appropriate file permissions (600/640)
- **Use enterprise templates** for production environments with enhanced security
- **Enable audit logging** for compliance requirements

### 5. Performance Optimization

- **Use caching** for repeated queries (configure appropriate TTL)
- **Batch similar queries** together to reduce API calls
- **Choose appropriate LLM provider** for your use case:
  - **Ollama** (default): Local processing, no API costs, privacy-focused
  - **Gemini**: Fast cloud processing, cost-effective
  - **OpenAI/Anthropic**: Premium cloud options for complex analysis
- **Monitor API usage** and costs regularly
- **Use production dependencies** for high-performance deployments
- **Configure connection pooling** for better AWS API performance
- **Monitor system health** to identify performance bottlenecks
- **Use appropriate configuration templates** for your environment (dev vs. prod)

### 6. Environment Management

#### Development Environment
```bash
# Use development template for faster iteration
cp config/templates/development.yaml ~/.aws-cost-cli/config.yaml

# Install basic dependencies
pip install -e .[dev]

# Enable debug logging
export AWS_COST_CLI_LOG_LEVEL=DEBUG
```

#### Production Environment  
```bash
# Use production template for reliability
cp config/templates/production.yaml ~/.aws-cost-cli/config.yaml

# Install production dependencies
pip install -r requirements-prod.txt

# Configure health monitoring
aws-cost-cli health serve --host 0.0.0.0 --port 8081 &
```

#### Multi-Account Enterprise
```bash
# Use multi-account template
cp config/templates/multi-account.yaml ~/.aws-cost-cli/config.yaml

# Configure cross-account roles
aws-cost-cli configure --account production --role arn:aws:iam::123456:role/CostExplorer
```

### 7. Team Collaboration

```bash
# Standardize queries across team
aws-cost-cli query "What are our monthly AWS costs by team tag?"

# Create reusable query patterns
aws-cost-cli query "Show development environment costs this month"

# Document common queries
aws-cost-cli query "Production vs staging cost comparison"
```

## Advanced Use Cases

### 1. Cost Allocation and Chargeback

```bash
# Department-based cost allocation
aws-cost-cli query "Show AWS costs by department tag for last quarter"

# Project-based billing
aws-cost-cli query "What are the costs for project Alpha this month?"

# Environment-based analysis
aws-cost-cli query "Compare production vs development environment costs"
```

### 2. Financial Planning

```bash
# Annual budget planning
aws-cost-cli query "Based on growth trends, what should our AWS budget be for 2025?"

# Cost center analysis
aws-cost-cli query "Show cost breakdown by business unit for financial reporting"

# ROI analysis
aws-cost-cli query "What are the cost trends for our revenue-generating services?"
```

### 3. Compliance and Reporting

```bash
# Monthly executive summary
aws-cost-cli query "Create an executive summary of AWS costs and trends this month"

# Audit trail
aws-cost-cli query "Show detailed cost breakdown for compliance reporting"

# Variance reporting
aws-cost-cli query "What are the significant cost variances from our budget this quarter?"
```

This user guide provides comprehensive coverage of the AWS Cost Explorer CLI capabilities. Start with basic queries and gradually explore more advanced features as you become comfortable with the tool. Remember that the natural language interface is flexible, so don't hesitate to experiment with different query phrasings to get the insights you need.
