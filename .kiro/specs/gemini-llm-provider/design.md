# Design Document

## Overview

This design extends the existing AWS Cost Explorer CLI's LLM provider architecture to include Google Gemini support and enhances the dynamic LLM switching capabilities. The system currently has a well-established provider pattern with OpenAI, Anthropic, Bedrock, and Ollama providers. This enhancement adds Gemini as a fifth provider option and improves the user experience for switching between providers.

## Architecture

The existing architecture already supports multiple LLM providers through an abstract base class pattern. The enhancement will integrate seamlessly into this existing structure:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   CLI Interface │────│  Query Processor │────│   AWS Client    │
│   (Click/Typer) │    │   (LLM-powered)  │    │    (boto3)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Response Format │    │  Cache Manager   │    │ Config Manager  │
│   (Rich + LLM)  │    │   (File-based)   │    │   (YAML/JSON)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### LLM Provider Architecture Enhancement

```
                    ┌─────────────────┐
                    │  LLMProvider    │
                    │  (Abstract)     │
                    └─────────────────┘
                            │
        ┌───────────────────┼───────────────────┬───────────────────┐
        │                   │                   │                   │
┌───────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ OpenAIProvider│  │AnthropicProvider│  │ BedrockProvider │  │ OllamaProvider  │
│   (existing)  │  │   (existing)    │  │   (existing)    │  │   (existing)    │
└───────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘
        │
┌───────────────┐
│ GeminiProvider│  ← NEW
│     (new)     │
└───────────────┘
```

## Components and Interfaces

### 1. GeminiProvider Class

**Technology:** Google AI Python SDK (google-generativeai)
**Responsibilities:**
- Integrate with Google's Gemini API
- Parse natural language queries using Gemini models
- Handle Gemini-specific authentication and error scenarios
- Normalize responses to match existing provider interface

**Implementation:**
```python
class GeminiProvider(LLMProvider):
    """Google Gemini provider for query parsing."""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        self.api_key = api_key
        self.model = model
        self._client = None
    
    def _get_client(self):
        """Get Gemini client, creating it if necessary."""
        if self._client is None:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client = genai.GenerativeModel(self.model)
            except ImportError:
                raise ImportError("google-generativeai package is required")
        return self._client
    
    def is_available(self) -> bool:
        """Check if Gemini is available and configured."""
        try:
            _client = self._get_client()
            return bool(self.api_key)
        except ImportError:
            return False
    
    def parse_query(self, query: str) -> Dict[str, Any]:
        """Parse query using Google Gemini."""
        # Implementation details follow existing pattern
```

### 2. Enhanced Configuration Management

**Enhancements to ConfigManager:**
- Add GEMINI_API_KEY environment variable support
- Add gemini provider validation
- Support for Gemini-specific configuration options

**Configuration Structure Updates:**
```python
# Environment variable mappings (addition to existing)
env_mappings = {
    # ... existing mappings ...
    "GEMINI_API_KEY": ("llm_config", "gemini", "api_key"),
    "GEMINI_MODEL": ("llm_config", "gemini", "model"),
}

# Provider validation (addition to existing)
valid_providers = ["openai", "anthropic", "bedrock", "ollama", "gemini"]
```

### 3. Enhanced CLI Interface

**New Command Line Options:**
```python
@click.option(
    "--llm-provider",
    type=click.Choice(["openai", "anthropic", "bedrock", "ollama", "gemini"]),
    help="Override configured LLM provider for this query"
)
def query(question: str, llm_provider: Optional[str] = None, ...):
    """Enhanced query command with provider override."""
```

**New Provider Management Commands:**
```python
@cli.command()
def list_providers():
    """List available LLM providers and their configuration status."""

@cli.command()
@click.argument("provider", type=click.Choice(["openai", "anthropic", "bedrock", "ollama", "gemini"]))
def test_provider(provider: str):
    """Test a specific LLM provider configuration."""
```

### 4. Provider Factory Enhancement

**Enhanced Provider Creation:**
```python
class ProviderFactory:
    """Factory for creating LLM provider instances."""
    
    @staticmethod
    def create_provider(provider_name: str, config: Dict[str, Any]) -> LLMProvider:
        """Create provider instance based on configuration."""
        providers = {
            "openai": OpenAIProvider,
            "anthropic": AnthropicProvider,
            "bedrock": BedrockProvider,
            "ollama": OllamaProvider,
            "gemini": GeminiProvider,  # NEW
        }
        
        if provider_name not in providers:
            raise ValueError(f"Unknown provider: {provider_name}")
        
        return providers[provider_name](**config)
    
    @staticmethod
    def get_available_providers(config: Dict[str, Any]) -> List[str]:
        """Get list of properly configured providers."""
        available = []
        for provider_name in ["openai", "anthropic", "bedrock", "ollama", "gemini"]:
            try:
                provider = ProviderFactory.create_provider(provider_name, config)
                if provider.is_available():
                    available.append(provider_name)
            except Exception:
                pass
        return available
```

## Data Models

### Configuration Model Updates

**Enhanced Config Class:**
```python
@dataclass
class Config:
    llm_provider: str = "ollama"  # Default provider (local, no API key required)
    llm_config: Dict[str, Any] = field(default_factory=dict)
    default_profile: Optional[str] = None
    cache_ttl: int = 3600
    output_format: str = "simple"
    default_currency: str = "USD"
    fallback_providers: List[str] = field(default_factory=lambda: ["ollama", "openai", "anthropic", "gemini"])  # NEW
```

### Provider Configuration Structure

**Gemini-specific Configuration:**
```python
gemini_config = {
    "api_key": "your-gemini-api-key",
    "model": "gemini-1.5-flash",  # or "gemini-1.5-pro"
    "temperature": 0.1,
    "max_tokens": 500,
}
```

## Error Handling

### Gemini-Specific Error Handling

**API Error Mapping:**
```python
def parse_query(self, query: str) -> Dict[str, Any]:
    try:
        # Gemini API call
        response = self._client.generate_content(prompt)
        return self._parse_llm_response(response.text)
    except ImportError:
        raise LLMProviderError("google-generativeai package not installed", provider="gemini")
    except Exception as e:
        error_msg = str(e).lower()
        if "api key" in error_msg or "authentication" in error_msg:
            raise LLMProviderError("Invalid Gemini API key", provider="gemini")
        elif "quota" in error_msg or "rate limit" in error_msg:
            raise LLMProviderError("Gemini API quota exceeded", provider="gemini")
        elif "network" in error_msg or "connection" in error_msg:
            raise NetworkError(f"Network error connecting to Gemini: {e}")
        else:
            raise LLMProviderError(f"Gemini API error: {str(e)}", provider="gemini")
```

### Enhanced Fallback Strategy

**Provider Fallback Logic:**
```python
class QueryParser:
    def parse_query_with_fallback(self, query: str, preferred_provider: str = None) -> Dict[str, Any]:
        """Parse query with automatic fallback to alternative providers."""
        providers_to_try = []
        
        # Add preferred provider first
        if preferred_provider:
            providers_to_try.append(preferred_provider)
        
        # Add configured default provider
        if self.config.llm_provider not in providers_to_try:
            providers_to_try.append(self.config.llm_provider)
        
        # Add fallback providers
        for provider in self.config.fallback_providers:
            if provider not in providers_to_try:
                providers_to_try.append(provider)
        
        last_error = None
        for provider_name in providers_to_try:
            try:
                provider = ProviderFactory.create_provider(provider_name, self.config.llm_config)
                if provider.is_available():
                    return provider.parse_query(query)
            except Exception as e:
                last_error = e
                continue
        
        # If all providers fail, raise the last error
        if last_error:
            raise last_error
        else:
            raise LLMProviderError("No available LLM providers configured")
```

## Testing Strategy

### Unit Testing for Gemini Provider

**Test Coverage:**
```python
class TestGeminiProvider:
    def test_gemini_provider_initialization(self):
        """Test Gemini provider can be initialized with API key."""
    
    def test_gemini_query_parsing(self):
        """Test Gemini can parse various query formats."""
    
    def test_gemini_error_handling(self):
        """Test Gemini-specific error scenarios."""
    
    def test_gemini_response_normalization(self):
        """Test Gemini responses match expected format."""
```

### Integration Testing

**Provider Switching Tests:**
```python
class TestProviderSwitching:
    def test_cli_provider_override(self):
        """Test --llm-provider flag overrides configuration."""
    
    def test_provider_fallback(self):
        """Test automatic fallback when primary provider fails."""
    
    def test_provider_availability_detection(self):
        """Test system correctly detects available providers."""
```

### Configuration Testing

**Enhanced Configuration Tests:**
```python
class TestEnhancedConfiguration:
    def test_gemini_env_var_loading(self):
        """Test GEMINI_API_KEY environment variable is loaded."""
    
    def test_provider_validation(self):
        """Test configuration validation includes Gemini."""
    
    def test_fallback_provider_configuration(self):
        """Test fallback provider list configuration."""
```

## Security Considerations

### API Key Management

**Environment Variable Security:**
- API keys should only be stored in environment variables, never in configuration files
- Support for multiple API key sources (environment, AWS Secrets Manager, etc.)
- Clear documentation on secure API key management practices

**Gemini-Specific Security:**
- Validate Gemini API key format before making requests
- Implement proper request/response logging controls for Gemini
- Support for Gemini API key rotation

### Data Privacy

**Query Data Handling:**
- Ensure cost data sent to Gemini API is sanitized
- Implement opt-out mechanisms for external API usage
- Support for local-only processing when required

## Deployment and Distribution

### Package Dependencies

**New Dependencies:**
```python
# requirements.txt additions
google-generativeai>=0.3.0  # For Gemini support
```

### Configuration Examples

**Example Configuration File:**
```yaml
# ~/.aws-cost-cli/config.yaml
llm_provider: ollama  # Default to local provider
llm_config:
  ollama:
    model: gpt-oss:20b
    base_url: http://localhost:11434
  gemini:
    model: gemini-1.5-flash
    temperature: 0.1
  openai:
    model: gpt-3.5-turbo
  anthropic:
    model: claude-3-haiku-20240307
fallback_providers:
  - ollama
  - gemini
  - openai
  - anthropic
```

**Environment Variable Setup:**
```bash
# API Keys (choose one or more)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="your-gemini-api-key"

# Optional: Override default provider (default is ollama)
export AWS_COST_CLI_LLM_PROVIDER="gemini"
```

### Migration Strategy

**Backward Compatibility:**
- All existing configurations continue to work unchanged
- New Gemini provider is opt-in
- Existing provider behavior remains identical
- Configuration file format is backward compatible

**Upgrade Path:**
1. Install updated package with Gemini support
2. Set GEMINI_API_KEY environment variable (optional)
3. Update configuration to use Gemini (optional)
4. Test provider switching functionality