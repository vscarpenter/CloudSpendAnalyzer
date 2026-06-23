#!/usr/bin/env python3
"""Minimal CLI for AWS Cost Explorer."""

import sys
import json
import click
from config import load_config, Config
from query import process_query, format_simple_response
from cache import SimpleCache
from aws import AWSCostClient


@click.group()
@click.version_option(version="2.0.0", prog_name="aws-cost-cli")
def cli():
    """Simple AWS Cost Explorer CLI - Query AWS costs using natural language."""
    pass


@cli.command()
@click.argument("query")
@click.option("--profile", "-p", help="AWS profile to use")
@click.option("--format", "-f", "output_format", type=click.Choice(["simple", "json"]), help="Output format")
@click.option("--no-cache", is_flag=True, help="Bypass cache")
def query(query: str, profile: str, output_format: str, no_cache: bool):
    """Query AWS costs using natural language.
    
    Examples:
        aws-cost-cli query "What did I spend on EC2 last month?"
        aws-cost-cli query "Show me S3 costs this year" --format json
        aws-cost-cli query "RDS costs" --profile production
    """
    # Load configuration
    config = load_config()
    
    # Override settings from command line
    if profile:
        config.aws_profile = profile
    if output_format:
        config.output_format = output_format
    if no_cache:
        config.cache_enabled = False
    
    # Process the query
    result = process_query(query, config)
    
    # Output the result
    if config.output_format == "json":
        print(json.dumps(result, indent=2))
    else:
        if result['success']:
            if 'response' in result:
                print(result['response'])
            else:
                # Fallback to simple formatting
                print(format_simple_response(result['data']))
        else:
            print(f"Error: {result['error']}")
            if 'details' in result:
                print(f"\nDetails:\n{result['details']}")
            sys.exit(1)


@cli.command()
@click.option("--provider", type=click.Choice(["ollama", "openai"]), help="LLM provider to use")
@click.option("--api-key", help="API key for OpenAI")
@click.option("--profile", help="Default AWS profile")
def configure(provider: str, api_key: str, profile: str):
    """Configure AWS Cost CLI settings.
    
    Examples:
        aws-cost-cli configure --provider ollama
        aws-cost-cli configure --provider openai --api-key sk-...
        aws-cost-cli configure --profile production
    """
    import os
    
    if provider:
        os.environ["LLM_PROVIDER"] = provider
        print(f"Set LLM provider to: {provider}")
    
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        print("OpenAI API key configured")
    
    if profile:
        os.environ["AWS_PROFILE"] = profile
        print(f"Set default AWS profile to: {profile}")
    
    print("\nCurrent configuration:")
    config = load_config()
    print(f"  LLM Provider: {config.llm_provider}")
    print(f"  AWS Profile: {config.aws_profile}")
    print(f"  Cache Enabled: {config.cache_enabled}")
    print(f"  Output Format: {config.output_format}")


@cli.group()
def cache():
    """Manage query cache."""
    pass


@cache.command(name="clear")
def cache_clear():
    """Clear all cached queries."""
    config = load_config()
    cache = SimpleCache(config)
    cache.clear()
    print("Cache cleared successfully")


@cache.command(name="status")
def cache_status():
    """Show cache status."""
    config = load_config()
    cache_dir = config.cache_dir
    
    from pathlib import Path
    cache_path = Path(cache_dir).expanduser()
    
    if not cache_path.exists():
        print("Cache directory does not exist")
        return
    
    cache_files = list(cache_path.glob("*.json"))
    total_size = sum(f.stat().st_size for f in cache_files)
    
    print(f"Cache Status:")
    print(f"  Directory: {cache_path}")
    print(f"  Files: {len(cache_files)}")
    print(f"  Total Size: {total_size / 1024:.1f} KB")
    print(f"  TTL: {config.cache_ttl} seconds")
    print(f"  Enabled: {config.cache_enabled}")


@cli.command()
def test():
    """Test AWS connection and configuration."""
    config = load_config()
    
    print("Testing configuration...")
    print(f"  AWS Profile: {config.aws_profile}")
    print(f"  AWS Region: {config.aws_region}")
    print(f"  LLM Provider: {config.llm_provider}")
    
    # Test AWS connection
    try:
        aws_client = AWSCostClient(config)
        if aws_client.test_connection():
            print("  ✓ AWS connection successful")
        else:
            print("  ✗ AWS connection failed")
            sys.exit(1)
    except Exception as e:
        print(f"  ✗ AWS error: {e}")
        sys.exit(1)
    
    # Test LLM provider
    try:
        from llm import get_llm_provider
        llm = get_llm_provider(config)
        print(f"  ✓ LLM provider configured")
    except Exception as e:
        print(f"  ✗ LLM error: {e}")
        sys.exit(1)
    
    print("\nAll tests passed!")


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
