# Setting up AWS Cost CLI with GPT-OSS 20B on Ollama

This guide walks you through configuring the AWS Cost Explorer CLI to use the gpt-oss:20b model running locally via Ollama.

## Prerequisites

1. **System Requirements**:
   - 16GB+ RAM (recommended for 20B model)
   - 50GB+ free disk space
   - macOS, Linux, or Windows with WSL2

2. **Install Ollama**:
   ```bash
   # macOS (Homebrew)
   brew install ollama
   
   # Or download from https://ollama.ai/
   ```

3. **Install the gpt-oss:20b model**:
   ```bash
   ollama pull gpt-oss:20b
   ```

## Configuration Steps

### 1. Start Ollama with GPT-OSS 20B

```bash
# Start Ollama server with the model
./start-ollama.sh

# Check status
./status-ollama.sh
```

### 2. Configure AWS Cost CLI

Choose one of these configuration methods:

#### Option A: Use the pre-configured file
```bash
# Copy the Ollama configuration
cp config/ollama_gpt_oss_config.yaml ~/.aws-cost-cli/config.yaml
```

#### Option B: Configure manually
Create `~/.aws-cost-cli/config.yaml`:

```yaml
llm_provider: ollama
llm_config:
  ollama:
    model: "gpt-oss:20b"
    base_url: "http://localhost:11434"
    temperature: 0.1
    max_tokens: 1000
    timeout: 60
    options:
      num_predict: 1000
      top_k: 40
      top_p: 0.9
      repeat_penalty: 1.1

# AWS Configuration
default_profile: null
aws_region: "us-east-1"

# Cache Configuration (longer for local models)
cache_ttl: 7200  # 2 hours
cache_directory: "~/.aws-cost-cli/cache"

# Output Configuration
output_format: "simple"
default_currency: "USD"
```

#### Option C: Use environment variables
```bash
export AWS_COST_CLI_LLM_PROVIDER=ollama
export OLLAMA_MODEL=gpt-oss:20b
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_TIMEOUT=60
```

### 3. Test the Configuration

```bash
# Test basic query
aws-cost-cli query "What did I spend on EC2 last month?"

# Test with verbose output
aws-cost-cli query "Show me S3 costs for 2024" --format rich

# Check configuration
aws-cost-cli configure --show
```

## Performance Optimization

### Model-Specific Settings

The gpt-oss:20b model benefits from these optimizations:

```yaml
llm_config:
  ollama:
    model: "gpt-oss:20b"
    timeout: 60  # Increased for large model
    options:
      num_predict: 1000      # Max tokens to generate
      top_k: 40             # Limit vocabulary for consistency
      top_p: 0.9            # Nucleus sampling
      repeat_penalty: 1.1    # Reduce repetition
      temperature: 0.1       # Low for consistent parsing
```

### System Optimization

1. **Memory Management**:
   ```bash
   # Check available memory
   free -h  # Linux
   vm_stat  # macOS
   ```

2. **Ollama Settings** (in start-ollama.sh):
   ```bash
   export OLLAMA_NUM_PARALLEL=1        # Single model only
   export OLLAMA_MAX_LOADED_MODELS=1   # Memory efficiency
   export OLLAMA_FLASH_ATTENTION=1     # Performance boost
   ```

3. **Cache Settings**:
   ```yaml
   cache_ttl: 7200  # 2 hours - longer for local models
   ```

## Troubleshooting

### Common Issues

1. **Model not responding**:
   ```bash
   # Check Ollama status
   ./status-ollama.sh
   
   # Restart if needed
   ./stop-ollama.sh
   ./start-ollama.sh
   ```

2. **Timeout errors**:
   - Increase timeout in config: `timeout: 120`
   - Check system resources: `htop` or Activity Monitor

3. **Memory issues**:
   ```bash
   # Check model memory usage
   ollama ps
   
   # Free up memory
   ollama stop gpt-oss:20b
   ./start-ollama.sh
   ```

4. **Connection errors**:
   ```bash
   # Test Ollama API
   curl http://localhost:11434/api/tags
   
   # Check if port is available
   lsof -i :11434
   ```

### Performance Issues

1. **Slow responses**:
   - Reduce `num_predict` to 500-750
   - Increase `temperature` slightly (0.2)
   - Check system load

2. **Inconsistent parsing**:
   - Lower `temperature` to 0.05
   - Adjust `top_k` and `top_p`
   - Use fresh cache: `--fresh` flag

### Debugging

Enable debug mode:
```bash
# Set debug environment
export AWS_COST_CLI_DEBUG=1

# Run with verbose logging
aws-cost-cli query "test query" --debug
```

## Switching Back to Cloud Providers

To switch back to OpenAI/Anthropic/Bedrock:

```bash
# Update config
aws-cost-cli configure --provider openai --api-key YOUR_KEY

# Or edit config file
vim ~/.aws-cost-cli/config.yaml
```

## Performance Comparison

| Provider | Speed | Cost | Privacy | Offline |
|----------|-------|------|---------|---------|
| GPT-OSS 20B (Local) | Slow | Free | High | Yes |
| OpenAI GPT-3.5 | Fast | Low | Medium | No |
| Claude 3 Haiku | Fast | Low | Medium | No |
| AWS Bedrock | Medium | Medium | High | No |

## Next Steps

1. **Monitor Performance**: Use `--performance-metrics` flag
2. **Optimize Queries**: Cache frequently used queries
3. **Scale Up**: Consider GPU acceleration for faster inference
4. **Backup Config**: Keep your working configuration backed up

For more help, see the main [USER_GUIDE.md](USER_GUIDE.md) or run:
```bash
aws-cost-cli --help
```