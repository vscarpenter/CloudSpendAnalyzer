# AWS Cost Explorer CLI

A Python-based command-line tool that enables natural language querying of AWS cost and billing data.

## Features

- Natural language query interface for AWS cost data
- Multi-LLM provider support (OpenAI, Anthropic, Bedrock, Ollama)
- AWS credential integration with profile support
- Intelligent caching with TTL for performance
- Rich terminal output formatting
- Comprehensive error handling

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

1. Ensure you have AWS credentials configured with Bedrock access
2. Configure Bedrock region (optional, defaults to us-east-1):
   ```bash
   aws-cost-cli configure --provider bedrock --region us-west-2
   ```

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