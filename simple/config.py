"""Minimal configuration for AWS Cost CLI."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Simple configuration with only essential settings."""
    
    # AWS settings
    aws_profile: str = "default"
    aws_region: str = "us-east-1"
    
    # LLM settings
    llm_provider: str = "ollama"  # "ollama" or "openai"
    openai_api_key: Optional[str] = None
    ollama_url: str = "http://localhost:11434"
    
    # Cache settings
    cache_enabled: bool = True
    cache_ttl: int = 3600  # 1 hour
    cache_dir: str = "~/.aws-cost-cli/cache"
    
    # Output settings
    output_format: str = "simple"  # "simple" or "json"


def load_config() -> Config:
    """Load configuration from environment variables."""
    config = Config()
    
    # Override from environment variables
    config.aws_profile = os.getenv("AWS_PROFILE", config.aws_profile)
    config.aws_region = os.getenv("AWS_REGION", config.aws_region)
    config.llm_provider = os.getenv("LLM_PROVIDER", config.llm_provider)
    config.openai_api_key = os.getenv("OPENAI_API_KEY")
    config.ollama_url = os.getenv("OLLAMA_URL", config.ollama_url)
    config.cache_enabled = os.getenv("CACHE_ENABLED", "true").lower() == "true"
    config.cache_ttl = int(os.getenv("CACHE_TTL", str(config.cache_ttl)))
    config.output_format = os.getenv("OUTPUT_FORMAT", config.output_format)
    
    return config
