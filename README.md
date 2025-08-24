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

## Requirements

- Python 3.8+
- AWS CLI configured with appropriate permissions
- LLM API key (OpenAI, Anthropic, or AWS Bedrock access)

## Development

```bash
# Install in development mode
pip install -e .

# Run tests
pytest tests/

# Format code
black src/
```

## License

MIT License