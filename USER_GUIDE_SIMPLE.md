# AWS Cost Explorer CLI - Simple User Guide

A beginner-friendly guide to get you started with AWS cost analysis using natural language.

## Quick Start

### 1. First-Time Setup

Make sure you have:
- AWS CLI configured (`aws configure`)
- Python 3.8+ installed

### 2. Install the Tool

```bash
# Install the CLI
uv pip install -e .

# Test it works
aws-cost-cli query "What did I spend on AWS last month?"
```

That's it! The tool uses Ollama (local AI) by default, so no API keys are needed.

## Basic Usage

### Simple Cost Queries

Ask questions in plain English:

```bash
# Total spending
aws-cost-cli query "How much did I spend on AWS last month?"

# Service breakdown  
aws-cost-cli query "What services cost me money last month?"

# Specific services
aws-cost-cli query "What did I spend on EC2 last month?"
aws-cost-cli query "Show me S3 storage costs this year"
```

### Time Periods You Can Use

- "last month", "this month"
- "last year", "this year", "2024"
- "last 3 months", "last week"
- "January 2024", "Q1 2024"

## Common Questions

### Getting Started
```bash
# See what you're spending money on
aws-cost-cli query "What services did I use last month that cost me money?"

# Check this month's spending so far
aws-cost-cli query "How much have I spent on AWS this month?"
```

### Service-Specific Costs
```bash
# Compute costs (EC2, Lambda, etc.)
aws-cost-cli query "What did I spend on compute services last month?"

# Storage costs (S3, EBS, etc.)
aws-cost-cli query "What are my storage costs this year?"

# Database costs
aws-cost-cli query "How much am I spending on databases monthly?"
```

### Comparing Costs
```bash
# Month-to-month comparison
aws-cost-cli query "Compare this month's AWS spending to last month"

# Year-over-year comparison
aws-cost-cli query "Compare my 2024 AWS costs to 2023"
```

## Different Output Formats

```bash
# Default simple output
aws-cost-cli query "EC2 costs last month"

# Pretty formatted output
aws-cost-cli query "EC2 costs last month" --format rich

# Get JSON for scripts
aws-cost-cli query "EC2 costs last month" --format json
```

## Using Different AWS Profiles

```bash
# Use a specific AWS profile
aws-cost-cli query "S3 costs last month" --profile production

# List your AWS profiles
aws-cost-cli list-profiles
```

## If You Get $0.00 Results

Try these alternatives:

```bash
# Instead of "EC2 costs"
aws-cost-cli query "What did I spend on compute services last month?"

# Instead of "RDS costs"  
aws-cost-cli query "What did I spend on databases last month?"

# Check if you have any costs at all
aws-cost-cli query "What did I spend on AWS last month?"
```

## Setting Up Cloud AI (Optional)

If you want faster responses, you can use cloud AI providers:

### Google Gemini (Recommended)
```bash
# Get API key from https://makersuite.google.com/app/apikey
export GEMINI_API_KEY="your-api-key-here"

# Configure Gemini
aws-cost-cli configure --provider gemini

# Test it works
aws-cost-cli query "EC2 costs last month" --llm-provider gemini
```

### OpenAI
```bash
# Get API key from https://platform.openai.com/api-keys
export OPENAI_API_KEY="sk-your-key-here"

# Configure OpenAI
aws-cost-cli configure --provider openai

# Test it works
aws-cost-cli query "S3 costs last month" --llm-provider openai
```

## Troubleshooting

### Common Issues

**"No costs found" or getting $0.00:**
- Try broader queries like "What did I spend on AWS last month?"
- Use "compute services" instead of "EC2"
- Check a different time period

**"AWS credentials not configured":**
```bash
aws configure
```

**"LLM provider not available":**
```bash
# Check if default provider works
aws-cost-cli query "test query"

# Try switching providers
aws-cost-cli query "test query" --llm-provider gemini
```

### Getting Help

```bash
# Check system health
aws-cost-cli health check

# Test your setup
aws-cost-cli query "What did I spend on AWS last month?"

# See available commands
aws-cost-cli --help
```

## Tips for Better Results

1. **Be specific with time**: Use "last month" instead of "recently"
2. **Ask complete questions**: "What did I spend on S3 last month?" vs "S3 costs"
3. **Use service categories**: "compute services" instead of specific service names
4. **Start broad, then narrow down**: First ask about total costs, then drill into specific services

## Example Workflow

```bash
# 1. Check overall spending
aws-cost-cli query "What did I spend on AWS last month?"

# 2. See service breakdown
aws-cost-cli query "What services cost me money last month?"

# 3. Focus on expensive services
aws-cost-cli query "What did I spend on compute services last month?"

# 4. Compare to previous period
aws-cost-cli query "Compare this month's spending to last month"
```

That's all you need to get started! The tool is designed to understand natural language, so experiment with different ways of asking your questions.
