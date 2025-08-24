"""Main CLI interface for AWS Cost Explorer CLI."""

import sys
import json
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .config import ConfigManager
from .query_processor import QueryParser
from .aws_client import AWSCostClient, CredentialManager
from .cache_manager import CacheManager
from .response_formatter import ResponseGenerator
from .query_pipeline import QueryPipeline, QueryContext, QueryResult
from .models import Config
from .exceptions import (
    AWSCostCLIError,
    AWSCredentialsError,
    AWSPermissionsError,
    AWSAPIError,
    NetworkError,
    QueryParsingError,
    LLMProviderError,
    CacheError,
    ConfigurationError,
    ValidationError,
    format_error_message
)


# Global console for rich output
console = Console()


@click.group()
@click.version_option(version="1.0.0", prog_name="aws-cost-cli")
@click.pass_context
def cli(ctx):
    """AWS Cost Explorer CLI - Query your AWS costs using natural language."""
    ctx.ensure_object(dict)


@cli.command()
@click.argument('query', required=True)
@click.option(
    '--profile', '-p',
    help='AWS profile to use (defaults to default profile or AWS_PROFILE env var)'
)
@click.option(
    '--fresh', '-f',
    is_flag=True,
    help='Force fresh data retrieval, bypassing cache'
)
@click.option(
    '--format', 'output_format',
    type=click.Choice(['simple', 'rich', 'llm', 'json'], case_sensitive=False),
    help='Output format (defaults to config setting)'
)
@click.option(
    '--config-file', '-c',
    type=click.Path(exists=True),
    help='Path to configuration file'
)
@click.pass_context
def query(ctx, query: str, profile: Optional[str], fresh: bool, output_format: Optional[str], config_file: Optional[str]):
    """Query AWS costs using natural language.
    
    Examples:
        aws-cost-cli query "How much did I spend on EC2 last month?"
        aws-cost-cli query "What are my S3 costs this year?" --profile production
        aws-cost-cli query "Show me RDS spending for Q1" --fresh --format rich
    """
    try:
        # Create query context
        context = QueryContext(
            original_query=query,
            profile=profile,
            fresh_data=fresh,
            output_format=output_format.lower() if output_format else None,
            debug=ctx.obj.get('debug', False)
        )
        
        # Initialize pipeline
        pipeline = QueryPipeline(config_path=config_file)
        
        # Override output format if specified
        if output_format:
            pipeline.config.output_format = output_format.lower()
            context.output_format = output_format.lower()
        else:
            context.output_format = pipeline.config.output_format
        
        # Process query through pipeline
        if context.output_format != 'json':
            console.print(f"🔍 Processing query: '{query}'")
        
        result = pipeline.process_query(context)
        
        # Handle result
        if result.success:
            if context.output_format == 'json':
                # JSON output for programmatic use
                output = {
                    'query': query,
                    'success': True,
                    'total_cost': {
                        'amount': float(result.cost_data.total_cost.amount),
                        'currency': result.cost_data.total_cost.unit
                    },
                    'time_period': {
                        'start': result.cost_data.time_period.start.isoformat(),
                        'end': result.cost_data.time_period.end.isoformat()
                    },
                    'metadata': result.metadata,
                    'results': []
                }
                
                for cost_result in result.cost_data.results:
                    result_data = {
                        'period': {
                            'start': cost_result.time_period.start.isoformat(),
                            'end': cost_result.time_period.end.isoformat()
                        },
                        'total': {
                            'amount': float(cost_result.total.amount),
                            'currency': cost_result.total.unit
                        },
                        'estimated': cost_result.estimated,
                        'groups': []
                    }
                    
                    for group in cost_result.groups:
                        group_data = {
                            'keys': group.keys,
                            'metrics': {}
                        }
                        for metric_name, cost_amount in group.metrics.items():
                            group_data['metrics'][metric_name] = {
                                'amount': float(cost_amount.amount),
                                'currency': cost_amount.unit
                            }
                        result_data['groups'].append(group_data)
                    
                    output['results'].append(result_data)
                
                click.echo(json.dumps(output, indent=2))
            else:
                # Human-readable output
                console.print(result.formatted_response)
                
                # Show processing info if debug mode
                if context.debug:
                    console.print(f"\n📊 Processing time: {result.processing_time_ms:.1f}ms")
                    if result.cache_hit:
                        console.print("📋 Data source: Cache")
                    else:
                        console.print("☁️  Data source: AWS API")
                    
                    if result.llm_used:
                        console.print("🤖 Query parsing: LLM")
                    elif result.fallback_used:
                        console.print("🔧 Query parsing: Fallback")
        else:
            # Handle error
            error = result.error
            
            if context.output_format == 'json':
                output = {
                    'query': query,
                    'success': False,
                    'error': {
                        'type': error.__class__.__name__,
                        'message': error.message,
                        'code': getattr(error, 'error_code', None)
                    },
                    'metadata': result.metadata
                }
                click.echo(json.dumps(output, indent=2))
            else:
                console.print(Panel(
                    Text(error.message, style="bold red"),
                    title="Error",
                    border_style="red"
                ))
                console.print(format_error_message(error, include_suggestions=True))
                
                # Show suggestions for ambiguous queries
                if isinstance(error, QueryParsingError):
                    suggestions = pipeline.handle_ambiguous_query(context)
                    if suggestions:
                        console.print("\n💡 Try these suggestions:")
                        for suggestion in suggestions:
                            console.print(f"   • {suggestion}")
            
            sys.exit(1)
        
    except KeyboardInterrupt:
        console.print("\n👋 Query cancelled by user")
        sys.exit(0)
    except Exception as e:
        console.print(Panel(
            Text(f"❌ Unexpected error: {str(e)}", style="bold red"),
            title="Error",
            border_style="red"
        ))
        if ctx.obj.get('debug'):
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)


@cli.command()
@click.argument('partial_query', required=False)
@click.option(
    '--config-file', '-c',
    type=click.Path(exists=True),
    help='Path to configuration file'
)
def suggest(partial_query: Optional[str], config_file: Optional[str]):
    """Get query suggestions based on partial input.
    
    Examples:
        aws-cost-cli suggest "EC2"
        aws-cost-cli suggest "last month"
        aws-cost-cli suggest
    """
    try:
        # Initialize pipeline
        pipeline = QueryPipeline(config_path=config_file)
        
        # Get suggestions
        suggestions = pipeline.get_query_suggestions(partial_query or "")
        
        console.print(Panel(
            Text("Query Suggestions", style="bold blue"),
            border_style="blue"
        ))
        
        if partial_query:
            console.print(f"💡 Suggestions for '{partial_query}':")
        else:
            console.print("💡 Common query examples:")
        
        for i, suggestion in enumerate(suggestions, 1):
            console.print(f"   {i}. {suggestion}")
        
        console.print("\n🔍 Use these examples as templates for your own queries!")
        
    except Exception as e:
        console.print(Panel(
            Text(f"❌ Failed to get suggestions: {str(e)}", style="bold red"),
            title="Error",
            border_style="red"
        ))
        sys.exit(1)


@cli.command()
@click.option(
    '--config-file', '-c',
    type=click.Path(exists=True),
    help='Path to configuration file'
)
def pipeline_status(config_file: Optional[str]):
    """Show pipeline status and health."""
    try:
        # Initialize pipeline
        pipeline = QueryPipeline(config_path=config_file)
        
        # Get status
        status = pipeline.get_pipeline_status()
        
        console.print(Panel(
            Text("Pipeline Status", style="bold blue"),
            border_style="blue"
        ))
        
        # Component status
        console.print("🔧 Components:")
        console.print(f"   Config loaded: {'✅' if status['config_loaded'] else '❌'}")
        console.print(f"   Cache manager: {'✅' if status['cache_manager_initialized'] else '❌'}")
        console.print(f"   Query parser: {'✅' if status['query_parser_initialized'] else '❌'}")
        console.print(f"   AWS client: {'✅' if status['aws_client_initialized'] else '❌'}")
        console.print(f"   Response generator: {'✅' if status['response_generator_initialized'] else '❌'}")
        
        # Health status
        console.print("\n🏥 Health:")
        if 'cache_healthy' in status:
            console.print(f"   Cache: {'✅' if status['cache_healthy'] else '❌'}")
            if status.get('cache_entries'):
                console.print(f"   Cache entries: {status['cache_entries']}")
        
        if 'aws_service_healthy' in status:
            console.print(f"   AWS service: {'✅' if status['aws_service_healthy'] else '❌'}")
            if status.get('aws_response_time_ms'):
                console.print(f"   AWS response time: {status['aws_response_time_ms']:.1f}ms")
        
    except Exception as e:
        console.print(Panel(
            Text(f"❌ Failed to get pipeline status: {str(e)}", style="bold red"),
            title="Error",
            border_style="red"
        ))
        sys.exit(1)


@cli.command()
@click.option(
    '--provider',
    type=click.Choice(['openai', 'anthropic', 'ollama'], case_sensitive=False),
    required=True,
    help='LLM provider to configure'
)
@click.option(
    '--api-key',
    help='API key for the LLM provider (not needed for Ollama)'
)
@click.option(
    '--model',
    help='Model to use (e.g., gpt-3.5-turbo, claude-3-haiku-20240307, llama2)'
)
@click.option(
    '--base-url',
    help='Base URL for Ollama (default: http://localhost:11434)'
)
@click.option(
    '--config-file', '-c',
    type=click.Path(),
    help='Path to configuration file (will be created if it doesn\'t exist)'
)
def configure(provider: str, api_key: Optional[str], model: Optional[str], base_url: Optional[str], config_file: Optional[str]):
    """Configure LLM provider settings.
    
    Examples:
        aws-cost-cli configure --provider openai --api-key sk-...
        aws-cost-cli configure --provider anthropic --api-key sk-ant-...
        aws-cost-cli configure --provider ollama --model llama2
    """
    try:
        config_manager = ConfigManager()
        
        # Load existing config or create new one
        if config_file:
            config_path = Path(config_file)
        else:
            config_path = config_manager.get_default_config_path()
        
        try:
            config = config_manager.load_config(str(config_path))
        except FileNotFoundError:
            config = Config()
        
        # Update LLM configuration
        config.llm_provider = provider.lower()
        
        if not config.llm_config:
            config.llm_config = {}
        
        config.llm_config['provider'] = provider.lower()
        
        if api_key:
            config.llm_config['api_key'] = api_key
        
        if model:
            config.llm_config['model'] = model
        elif provider.lower() == 'openai' and 'model' not in config.llm_config:
            config.llm_config['model'] = 'gpt-3.5-turbo'
        elif provider.lower() == 'anthropic' and 'model' not in config.llm_config:
            config.llm_config['model'] = 'claude-3-haiku-20240307'
        elif provider.lower() == 'ollama' and 'model' not in config.llm_config:
            config.llm_config['model'] = 'llama2'
        
        if base_url:
            config.llm_config['base_url'] = base_url
        elif provider.lower() == 'ollama' and 'base_url' not in config.llm_config:
            config.llm_config['base_url'] = 'http://localhost:11434'
        
        # Save configuration
        config_manager.save_config(config, str(config_path))
        
        console.print(Panel(
            Text(f"✅ Configuration saved successfully", style="bold green"),
            title="Configuration Updated",
            border_style="green"
        ))
        
        console.print(f"📁 Config file: {config_path}")
        console.print(f"🤖 Provider: {provider}")
        if model:
            console.print(f"🧠 Model: {model}")
        if base_url:
            console.print(f"🌐 Base URL: {base_url}")
        
        # Test the configuration
        console.print("\n🧪 Testing configuration...")
        
        try:
            query_parser = QueryParser(config.llm_config)
            test_result = query_parser.parse_query("test query for configuration")
            console.print("✅ LLM provider configuration is working")
        except Exception as e:
            console.print(f"⚠️  Configuration test failed: {e}")
            console.print("   The configuration was saved but may not work correctly")
        
    except Exception as e:
        console.print(Panel(
            Text(f"❌ Configuration failed: {str(e)}", style="bold red"),
            title="Configuration Error",
            border_style="red"
        ))
        sys.exit(1)


@cli.command()
@click.option(
    '--config-file', '-c',
    type=click.Path(exists=True),
    help='Path to configuration file'
)
def list_profiles(config_file: Optional[str]):
    """List available AWS profiles."""
    try:
        credential_manager = CredentialManager()
        profiles = credential_manager.get_available_profiles()
        
        if not profiles:
            console.print(Panel(
                Text("No AWS profiles found", style="bold yellow"),
                title="AWS Profiles",
                border_style="yellow"
            ))
            console.print("\n💡 To create AWS profiles:")
            console.print("   1. Run: aws configure --profile <profile-name>")
            console.print("   2. Or edit ~/.aws/credentials manually")
            return
        
        console.print(Panel(
            Text("Available AWS Profiles", style="bold blue"),
            border_style="blue"
        ))
        
        for i, profile in enumerate(profiles, 1):
            # Check if profile has valid credentials
            is_valid = credential_manager.validate_credentials(profile)
            status = "✅" if is_valid else "❌"
            console.print(f"  {i}. {profile} {status}")
        
        console.print("\n💡 Use --profile <name> to specify a profile for queries")
        
    except Exception as e:
        console.print(Panel(
            Text(f"❌ Failed to list profiles: {str(e)}", style="bold red"),
            title="Error",
            border_style="red"
        ))
        sys.exit(1)


@cli.command()
@click.option(
    '--config-file', '-c',
    type=click.Path(exists=True),
    help='Path to configuration file'
)
def show_config(config_file: Optional[str]):
    """Show current configuration."""
    try:
        config_manager = ConfigManager()
        
        if config_file:
            config = config_manager.load_config(config_file)
            config_path = config_file
        else:
            config_path = config_manager.get_default_config_path()
            try:
                config = config_manager.load_config()
            except FileNotFoundError:
                console.print(Panel(
                    Text("No configuration file found", style="bold yellow"),
                    title="Configuration",
                    border_style="yellow"
                ))
                console.print(f"\n💡 Default config location: {config_path}")
                console.print("   Run 'aws-cost-cli configure' to create a configuration")
                return
        
        console.print(Panel(
            Text("Current Configuration", style="bold blue"),
            border_style="blue"
        ))
        
        console.print(f"📁 Config file: {config_path}")
        console.print(f"🤖 LLM Provider: {config.llm_provider}")
        console.print(f"📊 Output Format: {config.output_format}")
        console.print(f"⏰ Cache TTL: {config.cache_ttl} seconds")
        console.print(f"💰 Default Currency: {config.default_currency}")
        
        if config.default_profile:
            console.print(f"👤 Default Profile: {config.default_profile}")
        
        if config.llm_config:
            console.print("\n🧠 LLM Configuration:")
            for key, value in config.llm_config.items():
                if key == 'api_key':
                    # Mask API key for security
                    masked_value = f"{value[:8]}..." if len(value) > 8 else "***"
                    console.print(f"   {key}: {masked_value}")
                else:
                    console.print(f"   {key}: {value}")
        
    except Exception as e:
        console.print(Panel(
            Text(f"❌ Failed to show configuration: {str(e)}", style="bold red"),
            title="Error",
            border_style="red"
        ))
        sys.exit(1)


@cli.command()
@click.confirmation_option(prompt='Are you sure you want to clear the cache?')
@click.option(
    '--pattern',
    help='Pattern to match cache files (clears all if not specified)'
)
def clear_cache(pattern: Optional[str]):
    """Clear the cost data cache."""
    try:
        cache_manager = CacheManager()
        removed_count = cache_manager.clear_cache() if not pattern else cache_manager.invalidate_cache(pattern)
        
        console.print(Panel(
            Text(f"✅ Cache cleared successfully ({removed_count} files removed)", style="bold green"),
            title="Cache Management",
            border_style="green"
        ))
        
    except Exception as e:
        console.print(Panel(
            Text(f"❌ Failed to clear cache: {str(e)}", style="bold red"),
            title="Error",
            border_style="red"
        ))
        sys.exit(1)


@cli.command()
@click.option(
    '--profile', '-p',
    help='AWS profile to use'
)
def cache_stats(profile: Optional[str]):
    """Show cache statistics."""
    try:
        cache_manager = CacheManager()
        stats = cache_manager.get_cache_stats()
        
        console.print(Panel(
            Text("Cache Statistics", style="bold blue"),
            border_style="blue"
        ))
        
        console.print(f"📊 Total entries: {stats['total_entries']}")
        console.print(f"✅ Valid entries: {stats['valid_entries']}")
        console.print(f"⏰ Expired entries: {stats['expired_entries']}")
        console.print(f"💾 Cache size: {stats['cache_size_bytes']:,} bytes")
        
        if stats['oldest_entry']:
            console.print(f"📅 Oldest entry: {stats['oldest_entry']}")
        if stats['newest_entry']:
            console.print(f"🆕 Newest entry: {stats['newest_entry']}")
        
    except Exception as e:
        console.print(Panel(
            Text(f"❌ Failed to get cache statistics: {str(e)}", style="bold red"),
            title="Error",
            border_style="red"
        ))
        sys.exit(1)


@cli.command()
@click.option(
    '--profile', '-p',
    help='AWS profile to use'
)
@click.option(
    '--config-file', '-c',
    type=click.Path(exists=True),
    help='Path to configuration file'
)
def warm_cache(profile: Optional[str], config_file: Optional[str]):
    """Warm the cache with common queries."""
    try:
        # Load configuration
        config_manager = ConfigManager()
        if config_file:
            config = config_manager.load_config(config_file)
        else:
            config = config_manager.load_config()
        
        # Initialize components
        credential_manager = CredentialManager()
        
        # Validate AWS credentials
        if not credential_manager.validate_credentials(profile):
            raise AWSCredentialsError(profile=profile)
        
        # Initialize cache manager and AWS client
        cache_manager = CacheManager(default_ttl=config.cache_ttl)
        aws_client = AWSCostClient(profile=profile, cache_manager=cache_manager)
        
        # Check permissions
        if not aws_client.validate_permissions():
            raise AWSPermissionsError()
        
        console.print("🔥 Warming cache with common queries...")
        
        # Warm the cache
        results = aws_client.warm_cache_for_common_queries()
        
        if "error" in results:
            console.print(Panel(
                Text(f"❌ {results['error']}", style="bold red"),
                title="Cache Warming Error",
                border_style="red"
            ))
            sys.exit(1)
        
        console.print(Panel(
            Text("✅ Cache warming completed", style="bold green"),
            title="Cache Management",
            border_style="green"
        ))
        
        console.print(f"🔥 Queries warmed: {results['queries_warmed']}")
        console.print(f"❌ Queries failed: {results['queries_failed']}")
        
        if results['errors']:
            console.print("\n⚠️  Errors encountered:")
            for error in results['errors']:
                console.print(f"   • {error}")
        
    except (AWSCredentialsError, AWSPermissionsError, AWSAPIError, NetworkError) as e:
        console.print(Panel(
            Text(e.message, style="bold red"),
            title="Error",
            border_style="red"
        ))
        console.print(format_error_message(e, include_suggestions=True))
        sys.exit(1)
    except Exception as e:
        console.print(Panel(
            Text(f"❌ Failed to warm cache: {str(e)}", style="bold red"),
            title="Error",
            border_style="red"
        ))
        sys.exit(1)


@cli.command()
def cleanup_cache():
    """Clean up expired cache entries."""
    try:
        cache_manager = CacheManager()
        removed_count = cache_manager.cleanup_expired_cache()
        
        console.print(Panel(
            Text(f"✅ Cache cleanup completed ({removed_count} expired entries removed)", style="bold green"),
            title="Cache Management",
            border_style="green"
        ))
        
    except Exception as e:
        console.print(Panel(
            Text(f"❌ Failed to cleanup cache: {str(e)}", style="bold red"),
            title="Error",
            border_style="red"
        ))
        sys.exit(1)


@cli.command()
@click.option(
    '--debug',
    is_flag=True,
    help='Enable debug output'
)
@click.pass_context
def test(ctx, debug: bool):
    """Test the CLI configuration and AWS connectivity."""
    ctx.obj['debug'] = debug
    
    console.print(Panel(
        Text("AWS Cost CLI - System Test", style="bold blue"),
        border_style="blue"
    ))
    
    # Test 1: Configuration
    console.print("\n1️⃣  Testing configuration...")
    try:
        config_manager = ConfigManager()
        config = config_manager.load_config()
        console.print("   ✅ Configuration loaded successfully")
    except FileNotFoundError:
        console.print("   ⚠️  No configuration file found (using defaults)")
        config = Config()
    except Exception as e:
        console.print(f"   ❌ Configuration error: {e}")
        return
    
    # Test 2: AWS Credentials
    console.print("\n2️⃣  Testing AWS credentials...")
    try:
        credential_manager = CredentialManager()
        profiles = credential_manager.get_available_profiles()
        if profiles:
            console.print(f"   ✅ Found {len(profiles)} AWS profile(s)")
            
            # Test default profile
            if credential_manager.validate_credentials():
                console.print("   ✅ Default profile credentials are valid")
            else:
                console.print("   ❌ Default profile credentials are invalid")
        else:
            console.print("   ❌ No AWS profiles found")
            return
    except Exception as e:
        console.print(f"   ❌ AWS credential error: {e}")
        return
    
    # Test 3: AWS Permissions
    console.print("\n3️⃣  Testing AWS permissions...")
    try:
        aws_client = AWSCostClient()
        if aws_client.validate_permissions():
            console.print("   ✅ AWS Cost Explorer permissions are valid")
        else:
            console.print("   ❌ Insufficient AWS permissions")
    except Exception as e:
        console.print(f"   ❌ AWS permission error: {e}")
    
    # Test 4: LLM Provider
    console.print("\n4️⃣  Testing LLM provider...")
    try:
        query_parser = QueryParser(config.llm_config)
        # Try a simple test query
        test_result = query_parser.parse_query("test")
        console.print("   ✅ LLM provider is working")
    except Exception as e:
        console.print(f"   ⚠️  LLM provider error: {e}")
        console.print("   💡 Fallback parsing will be used")
    
    # Test 5: Cache
    console.print("\n5️⃣  Testing cache system...")
    try:
        cache_manager = CacheManager()
        # Test cache directory creation
        cache_manager.cache_data("test_key", {"test": "data"})
        cached_data = cache_manager.get_cached_data("test_key")
        if cached_data:
            console.print("   ✅ Cache system is working")
            cache_manager.invalidate_cache("test_key")
        else:
            console.print("   ❌ Cache system failed")
    except Exception as e:
        console.print(f"   ❌ Cache error: {e}")
    
    console.print("\n🎉 System test completed!")


def main():
    """Entry point for the CLI application."""
    cli()


if __name__ == '__main__':
    main()