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
from .models import Config


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
        # Load configuration
        config_manager = ConfigManager()
        if config_file:
            config = config_manager.load_config(config_file)
        else:
            config = config_manager.load_config()
        
        # Override format if specified
        if output_format:
            config.output_format = output_format.lower()
        
        # Initialize components
        credential_manager = CredentialManager()
        
        # Validate AWS credentials
        if not credential_manager.validate_credentials(profile):
            console.print(Panel(
                Text("❌ AWS credentials not found or invalid", style="bold red"),
                title="Authentication Error",
                border_style="red"
            ))
            console.print("\n💡 To set up AWS credentials:")
            console.print("   1. Run: aws configure")
            console.print("   2. Or set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables")
            console.print("   3. Or use IAM roles if running on EC2")
            sys.exit(1)
        
        # Initialize AWS client
        aws_client = AWSCostClient(profile=profile)
        
        # Check permissions
        if not aws_client.validate_permissions():
            console.print(Panel(
                Text("❌ Insufficient AWS permissions", style="bold red"),
                title="Permission Error",
                border_style="red"
            ))
            console.print("\n💡 Required IAM permissions:")
            console.print("   - ce:GetCostAndUsage")
            console.print("   - ce:GetDimensionValues")
            console.print("   - ce:GetUsageReport")
            console.print("   - organizations:ListAccounts (if using consolidated billing)")
            sys.exit(1)
        
        # Initialize cache manager
        cache_manager = CacheManager(ttl=config.cache_ttl)
        
        # Initialize query parser
        query_parser = QueryParser(config.llm_config)
        
        # Parse the natural language query
        if config.output_format != 'json':
            console.print(f"🔍 Analyzing query: '{query}'")
        
        try:
            query_params = query_parser.parse_query(query)
        except Exception as e:
            console.print(Panel(
                Text(f"❌ Failed to parse query: {str(e)}", style="bold red"),
                title="Query Parsing Error",
                border_style="red"
            ))
            console.print("\n💡 Try rephrasing your query. Examples:")
            console.print("   - 'How much did I spend on EC2 last month?'")
            console.print("   - 'What are my total AWS costs this year?'")
            console.print("   - 'Show me S3 spending for the last 3 months'")
            sys.exit(1)
        
        # Validate parsed parameters
        if not query_parser.validate_parameters(query_params):
            console.print(Panel(
                Text("❌ Invalid query parameters extracted", style="bold red"),
                title="Validation Error",
                border_style="red"
            ))
            sys.exit(1)
        
        # Generate cache key
        cache_key = cache_manager.generate_cache_key(query_params, profile or 'default')
        
        # Try to get cached data first (unless fresh is requested)
        cost_data = None
        if not fresh:
            cost_data = cache_manager.get_cached_data(cache_key)
            if cost_data and config.output_format != 'json':
                console.print("📋 Using cached data")
        
        # Fetch fresh data if needed
        if cost_data is None:
            if config.output_format != 'json':
                console.print("☁️  Fetching data from AWS...")
            
            try:
                cost_data = aws_client.get_cost_and_usage(query_params)
                
                # Cache the results
                cache_manager.cache_data(cache_key, cost_data)
                
            except Exception as e:
                console.print(Panel(
                    Text(f"❌ Failed to fetch AWS cost data: {str(e)}", style="bold red"),
                    title="AWS API Error",
                    border_style="red"
                ))
                console.print("\n💡 Common issues:")
                console.print("   - Check your AWS credentials and permissions")
                console.print("   - Verify the time period is valid")
                console.print("   - Ensure Cost Explorer is enabled in your AWS account")
                sys.exit(1)
        
        # Initialize response generator
        llm_provider = None
        if config.llm_config.get('provider') and config.output_format == 'llm':
            try:
                # Initialize LLM provider based on config
                if config.llm_config['provider'] == 'openai':
                    from .query_processor import OpenAIProvider
                    llm_provider = OpenAIProvider(
                        config.llm_config.get('api_key', ''),
                        config.llm_config.get('model', 'gpt-3.5-turbo')
                    )
                elif config.llm_config['provider'] == 'anthropic':
                    from .query_processor import AnthropicProvider
                    llm_provider = AnthropicProvider(
                        config.llm_config.get('api_key', ''),
                        config.llm_config.get('model', 'claude-3-haiku-20240307')
                    )
                elif config.llm_config['provider'] == 'ollama':
                    from .query_processor import OllamaProvider
                    llm_provider = OllamaProvider(
                        config.llm_config.get('model', 'llama2'),
                        config.llm_config.get('base_url', 'http://localhost:11434')
                    )
            except Exception as e:
                if config.output_format != 'json':
                    console.print(f"⚠️  LLM provider initialization failed: {e}")
                    console.print("   Falling back to simple formatting")
        
        response_generator = ResponseGenerator(
            llm_provider=llm_provider,
            output_format=config.output_format
        )
        
        # Generate and display response
        if config.output_format == 'json':
            # JSON output for programmatic use
            output = {
                'query': query,
                'total_cost': {
                    'amount': float(cost_data.total_cost.amount),
                    'currency': cost_data.total_cost.unit
                },
                'time_period': {
                    'start': cost_data.time_period.start.isoformat(),
                    'end': cost_data.time_period.end.isoformat()
                },
                'results': []
            }
            
            for result in cost_data.results:
                result_data = {
                    'period': {
                        'start': result.time_period.start.isoformat(),
                        'end': result.time_period.end.isoformat()
                    },
                    'total': {
                        'amount': float(result.total.amount),
                        'currency': result.total.unit
                    },
                    'estimated': result.estimated,
                    'groups': []
                }
                
                for group in result.groups:
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
            response = response_generator.format_response(cost_data, query, query_params)
            console.print(response)
        
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
def clear_cache():
    """Clear the cost data cache."""
    try:
        cache_manager = CacheManager()
        cache_manager.clear_cache()
        
        console.print(Panel(
            Text("✅ Cache cleared successfully", style="bold green"),
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