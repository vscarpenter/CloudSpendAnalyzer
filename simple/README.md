# AWS Cost CLI (Simple Version)

A dramatically simplified AWS Cost Explorer CLI that queries costs using natural language.

## What's New in v2.0?

- **80% less code** - From 15,700 to ~1,500 lines
- **5 commands instead of 30+** - Just the essentials
- **3 dependencies instead of 10+** - Minimal footprint
- **2 LLM providers instead of 5** - Ollama (local) or OpenAI
- **Zero configuration** - Works out of the box with Ollama

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
pip install -e .
```

### 2. Configure AWS

```bash
aws configure  # If not already done
```

### 3. Start Ollama (Default LLM)

```bash
# Install Ollama if needed
brew install ollama  # macOS
# or visit https://ollama.ai/download

# Start Ollama
ollama serve

# Pull a model
ollama pull llama2
```

### 4. Query Your Costs

```bash
aws-cost-cli query "What did I spend on AWS last month?"
aws-cost-cli query "Show me EC2 costs this year"
aws-cost-cli query "S3 storage costs for January 2024"
```

## Core Commands (Only 5!)

### 1. Query - Natural language cost queries
```bash
aws-cost-cli query "What did I spend on EC2 last month?"
aws-cost-cli query "S3 costs this year" --format json
aws-cost-cli query "RDS costs" --profile production
```

### 2. Configure - Setup the CLI
```bash
aws-cost-cli configure --provider ollama
aws-cost-cli configure --provider openai --api-key sk-...
aws-cost-cli configure --profile production
```

### 3. Cache - Manage cached queries
```bash
aws-cost-cli cache clear
aws-cost-cli cache status
```

### 4. Test - Verify your setup
```bash
aws-cost-cli test
```

### 5. Help - Get help
```bash
aws-cost-cli --help
aws-cost-cli query --help
```

## Configuration

### Environment Variables (Optional)

```bash
# AWS Settings
export AWS_PROFILE=production
export AWS_REGION=us-east-1

# LLM Settings (default: ollama)
export LLM_PROVIDER=ollama  # or "openai"
export OPENAI_API_KEY=sk-... # Only if using OpenAI

# Cache Settings
export CACHE_ENABLED=true
export CACHE_TTL=3600

# Output
export OUTPUT_FORMAT=simple  # or "json"
```

## Using OpenAI Instead of Ollama

```bash
# Set your API key
export OPENAI_API_KEY=sk-your-key-here

# Configure
aws-cost-cli configure --provider openai

# Query
aws-cost-cli query "EC2 costs last month"
```

## Architecture

```
8 files, ~1,500 lines total:

cli.py      (175 lines) - CLI interface with 5 commands
query.py    (117 lines) - Query processing pipeline
aws.py      (152 lines) - AWS Cost Explorer client
llm.py      (236 lines) - LLM providers (Ollama/OpenAI)
cache.py    (103 lines) - Simple file caching
config.py   (44 lines)  - Minimal configuration
models.py   (31 lines)  - Data models
utils.py    (161 lines) - Date parsing utilities
```

## Comparison with Original

| Metric | Original | Simple | Improvement |
|--------|----------|--------|-------------|
| Lines of Code | 15,700 | ~1,500 | -90% |
| Files | 21 | 8 | -62% |
| Dependencies | 10+ | 3 | -70% |
| Commands | 30+ | 5 | -83% |
| LLM Providers | 5 | 2 | -60% |
| Setup Time | 30+ min | 5 min | -83% |

## Common Queries

```bash
# Total spending
aws-cost-cli query "What did I spend on AWS last month?"
aws-cost-cli query "Total AWS costs this year"

# Service-specific
aws-cost-cli query "EC2 costs last month"
aws-cost-cli query "S3 storage costs this year"
aws-cost-cli query "RDS database costs for Q1 2024"

# Comparisons
aws-cost-cli query "Compare this month's costs to last month"
aws-cost-cli query "AWS costs for 2024 vs 2023"

# Time periods
aws-cost-cli query "Costs for January 2024"
aws-cost-cli query "Last 30 days of AWS spending"
aws-cost-cli query "Yesterday's costs"
```

## Troubleshooting

### "Ollama not available"
```bash
# Start Ollama
ollama serve

# Pull a model if needed
ollama pull llama2
```

### "AWS credentials not configured"
```bash
aws configure
```

### "No costs found" or $0.00
```bash
# Try broader queries
aws-cost-cli query "What did I spend on AWS last month?"

# Check if Cost Explorer is enabled in AWS Console
```

## Why Simple?

The original codebase had:
- 5 LLM providers with complex factories
- 30+ CLI commands
- Multiple response formatters
- Complex pipeline architecture
- Extensive configuration options
- Performance monitoring
- Health checks
- Interactive builders
- Data exporters

**This version keeps only what matters:**
- Query costs with natural language
- Cache results
- Simple configuration
- Works out of the box

## License

MIT
