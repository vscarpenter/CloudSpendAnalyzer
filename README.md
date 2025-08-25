# AWS Cost Explorer CLI

A Python-based command-line tool that enables natural language querying of AWS cost and billing data.

## Features

- Natural language query interface for AWS cost data
- Multi-LLM provider support (OpenAI, Anthropic, Bedrock, Ollama)
- AWS credential integration with profile support
- Intelligent caching with TTL for performance
- **Performance optimizations** with parallel query execution and compression
- Rich terminal output formatting
- Comprehensive error handling
- Performance monitoring and metrics

## Installation

```bash
pip install aws-cost-explorer-cli
```

## Quick Start

```bash
# Query your EC2 costs
aws-cost-cli query "How much did I spend on EC2 last month?"

# Configure LLM provider
aws-cost-cli configure --provider openai --api-key your-api-key

# List available AWS profiles
aws-cost-cli list-profiles
```

## Performance Optimizations

The CLI includes several performance optimization features for handling large queries and datasets:

### Parallel Query Execution
Large time range queries are automatically split into chunks and executed in parallel:

```bash
# This query will automatically use parallel execution for the full year
aws-cost-cli query "Show me all AWS costs for 2024" --performance-metrics

# Control parallel execution
aws-cost-cli query "EC2 costs for 2024" --parallel --max-chunk-days 60
aws-cost-cli query "S3 costs last month" --no-parallel
```

### Cache Compression
Reduce cache storage requirements with automatic compression:

```bash
# Enable compression (default)
aws-cost-cli query "RDS costs this year" --compression

# Disable compression
aws-cost-cli query "Lambda costs last month" --no-compression
```

### Performance Monitoring
Track query performance and optimization effectiveness:

```bash
# Show performance metrics after query
aws-cost-cli query "All services last quarter" --performance-metrics

# View comprehensive performance statistics
aws-cost-cli performance

# View performance for specific time period
aws-cost-cli performance --hours 48 --format json
```

**Example performance output:**
```
🚀 Performance Metrics:
   Processing time: 1250.5ms
   API calls made: 4
   Parallel requests: 4
   Cache hit: No
   Compression ratio: 0.65
   Space saved: 35.0%
```

For detailed performance optimization guidance, see [docs/PERFORMANCE_GUIDE.md](docs/PERFORMANCE_GUIDE.md).

## LLM Provider Setup

### OpenAI

1. Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Set your API key:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```
   Or configure via CLI:
   ```bash
   aws-cost-cli configure --provider openai --api-key your-api-key
   ```

### Anthropic Claude

1. Get your API key from [Anthropic Console](https://console.anthropic.com/)
2. Set your API key:
   ```bash
   export ANTHROPIC_API_KEY="your-api-key-here"
   ```
   Or configure via CLI:
   ```bash
   aws-cost-cli configure --provider anthropic --api-key your-api-key
   ```

### AWS Bedrock

1. Ensure you have AWS credentials configured with Bedrock access:
   ```bash
   aws configure  # or use AWS profiles
   ```

2. Verify your AWS credentials have the required permissions:
   - `bedrock:InvokeModel`
   - `bedrock:ListFoundationModels` (optional)

3. Configure Bedrock provider:
   ```bash
   # Basic configuration (uses default region us-east-1)
   aws-cost-cli configure --provider bedrock
   
   # With custom region and model
   aws-cost-cli configure --provider bedrock \
     --region us-west-2 \
     --model anthropic.claude-3-sonnet-20240229-v1:0
   
   # With specific AWS profile
   aws-cost-cli configure --provider bedrock \
     --profile production \
     --region us-east-1
   ```

4. Available Bedrock models:
   - `anthropic.claude-3-haiku-20240307-v1:0` (fast, cost-effective)
   - `anthropic.claude-3-sonnet-20240229-v1:0` (balanced performance)
   - `anthropic.claude-3-opus-20240229-v1:0` (highest capability)
   - `amazon.titan-text-express-v1` (Amazon's model)
   - `ai21.j2-ultra-v1` (AI21 Labs model)

### Ollama (Local)

1. Install Ollama:
   ```bash
   # macOS
   brew install ollama
   
   # Linux
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Windows - Download from https://ollama.ai/download
   ```

2. Start Ollama service:
   ```bash
   ollama serve
   ```

3. Pull a model (recommended: llama2, codellama, or mistral):
   ```bash
   # Pull Llama2 (7B model, good balance of speed/quality)
   ollama pull llama2
   
   # Or pull a smaller/faster model
   ollama pull llama2:7b-chat
   
   # Or pull a more capable model (requires more resources)
   ollama pull codellama:13b
   ```

4. Configure the CLI to use Ollama:
   ```bash
   aws-cost-cli configure --provider ollama --model llama2
   ```

   Optional: Configure custom Ollama URL (if not running on localhost:11434):
   ```bash
   aws-cost-cli configure --provider ollama --base-url http://your-server:11434
   ```

## Environment Variables

The AWS Cost CLI supports configuration through environment variables. These variables take precedence over configuration files and provide a secure way to manage API keys and credentials.

### Core Configuration

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `AWS_COST_CLI_LLM_PROVIDER` | LLM provider to use | `openai` | `openai`, `anthropic`, `bedrock`, `ollama` |
| `AWS_COST_CLI_DEFAULT_PROFILE` | Default AWS profile | None | `my-profile` |
| `AWS_COST_CLI_CACHE_TTL` | Cache TTL in seconds | `3600` | `1800` |
| `AWS_COST_CLI_OUTPUT_FORMAT` | Default output format | `simple` | `simple`, `detailed`, `json` |
| `AWS_COST_CLI_DEFAULT_CURRENCY` | Currency for cost display | `USD` | `USD`, `EUR`, `GBP` |

### LLM Provider API Keys

| Variable | Description | Required For |
|----------|-------------|--------------|
| `OPENAI_API_KEY` | OpenAI API key | OpenAI provider |
| `ANTHROPIC_API_KEY` | Anthropic Claude API key | Anthropic provider |

### AWS Credentials

| Variable | Description | Notes |
|----------|-------------|-------|
| `AWS_ACCESS_KEY_ID` | AWS access key ID | Standard AWS credential |
| `AWS_SECRET_ACCESS_KEY` | AWS secret access key | Standard AWS credential |
| `AWS_SESSION_TOKEN` | AWS session token | For temporary credentials |
| `AWS_PROFILE` | AWS profile to use | Alternative to access key/secret |
| `AWS_REGION` | AWS region | Defaults to `us-east-1` |

### Setting Environment Variables

#### Linux/macOS (Bash/Zsh)

```bash
# Add to ~/.bashrc, ~/.zshrc, or ~/.profile
export OPENAI_API_KEY="sk-your-openai-key-here"
export ANTHROPIC_API_KEY="sk-ant-your-anthropic-key-here"
export AWS_COST_CLI_LLM_PROVIDER="openai"
export AWS_COST_CLI_CACHE_TTL="7200"

# Apply changes
source ~/.bashrc  # or ~/.zshrc
```

#### Windows (PowerShell)

```powershell
# Set for current session
$env:OPENAI_API_KEY = "sk-your-openai-key-here"
$env:ANTHROPIC_API_KEY = "sk-ant-your-anthropic-key-here"
$env:AWS_COST_CLI_LLM_PROVIDER = "openai"

# Set permanently (requires restart)
[Environment]::SetEnvironmentVariable("OPENAI_API_KEY", "sk-your-openai-key-here", "User")
[Environment]::SetEnvironmentVariable("ANTHROPIC_API_KEY", "sk-ant-your-anthropic-key-here", "User")
```

#### Windows (Command Prompt)

```cmd
rem Set for current session
set OPENAI_API_KEY=sk-your-openai-key-here
set ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
set AWS_COST_CLI_LLM_PROVIDER=openai

rem Set permanently
setx OPENAI_API_KEY "sk-your-openai-key-here"
setx ANTHROPIC_API_KEY "sk-ant-your-anthropic-key-here"
```

### Configuration Precedence

The CLI loads configuration in the following order (higher numbers override lower):

1. **Default values** (built into the application)
2. **Configuration files** (`~/.aws-cost-cli/config.yaml`, `./config.yaml`)
3. **Environment variables** (highest precedence)

This means environment variables will always override configuration file settings.

### Security Best Practices

- **Never commit API keys** to version control
- **Use environment variables** for sensitive data like API keys
- **Rotate API keys regularly** and update environment variables
- **Use AWS profiles** instead of hardcoded credentials when possible
- **Set appropriate permissions** on configuration files (`chmod 600`)

### Verification

Check your environment variables:

```bash
# Verify API keys are set (masked output for security)
aws-cost-cli config show

# Test configuration
aws-cost-cli test
```

## Requirements

- Python 3.8+
- AWS CLI configured with appropriate permissions
- One of the following LLM providers:
  - OpenAI API key
  - Anthropic API key  
  - AWS Bedrock access
  - Ollama running locally

## Development

```bash
# Install in development mode
pip install -e .

# Run tests
pytest tests/

# Format code
black src/
```

## Author

Created by [Vinny Carpenter](https://vinny.dev/)

## License

MIT License