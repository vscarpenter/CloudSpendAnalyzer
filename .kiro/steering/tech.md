# Technology Stack

## Core Technologies

- **Language**: Python 3.8+
- **CLI Framework**: Click or Typer
- **AWS Integration**: boto3 SDK
- **Output Formatting**: Rich library for terminal enhancement
- **Configuration**: YAML/JSON file support
- **Caching**: File-based JSON/pickle serialization
- **Testing**: pytest with mocking capabilities

## LLM Integration

- **Providers**: OpenAI, Anthropic, AWS Bedrock, Ollama
- **Fallback**: Pattern matching for offline scenarios
- **Local Option**: Ollama support for sensitive environments

## Build System & Dependencies

- **Package Management**: pip with requirements.txt
- **Distribution**: PyPI package with setup.py
- **Alternative Installs**: Homebrew, Docker, PyInstaller binaries

## Common Commands

```bash
# Development setup
pip install -r requirements.txt
pip install -e .

# Testing
pytest tests/
pytest tests/ -v --cov=aws_cost_cli

# Package building
python setup.py sdist bdist_wheel

# Installation
pip install aws-cost-explorer-cli

# Usage
aws-cost-cli query "What did I spend on EC2 last month?"
aws-cost-cli configure --provider openai --api-key <key>
aws-cost-cli list-profiles
```

## Architecture Patterns

- **Modular Design**: Clear separation between CLI, query processing, AWS client, and response formatting
- **Dependency Injection**: Configurable LLM providers and AWS profiles
- **Error Handling**: Graceful degradation with user-friendly messages
- **Caching Strategy**: TTL-based file caching for API responses