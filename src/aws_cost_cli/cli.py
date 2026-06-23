"""Main CLI interface for AWS Cost Explorer CLI."""

import sys
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Confirm, Prompt
from rich.table import Table

from .config import ConfigManager
from .query_processor import QueryParser
from .aws_client import AWSCostClient, CredentialManager
from .cache_manager import CacheManager
from .query_pipeline import QueryPipeline, QueryContext
from .models import Config
from .data_exporter import ExportManager
from .interactive_query_builder import InteractiveQueryBuilder
from .exceptions import (
    AWSCredentialsError,
    AWSPermissionsError,
    AWSAPIError,
    NetworkError,
    QueryParsingError,
    format_error_message,
)
from .health import HealthChecker, create_health_check_server
from .provider_factory import ProviderFactory

# Global console for rich output
console = Console()


@click.group()
@click.version_option(version="1.0.0", prog_name="aws-cost-cli")
@click.pass_context
def cli(ctx):
    """AWS Cost Explorer CLI - Query your AWS costs using natural language."""
    ctx.ensure_object(dict)


@cli.command()
@click.argument("query", required=True)
@click.option(
    "--profile",
    "-p",
    help="AWS profile to use (defaults to default profile or AWS_PROFILE env var)",
)
@click.option(
    "--fresh", "-f", is_flag=True, help="Force fresh data retrieval, bypassing cache"
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["simple", "rich", "llm", "json"], case_sensitive=False),
    help="Output format (defaults to config setting)",
)
@click.option(
    "--llm-provider",
    type=click.Choice(
        ["openai", "anthropic", "bedrock", "ollama", "gemini"], case_sensitive=False
    ),
    help="Override configured LLM provider for this query only (default: ollama)",
)
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True),
    help="Path to configuration file",
)
@click.option(
    "--parallel/--no-parallel",
    default=True,
    help="Enable/disable parallel query execution for large queries",
)
@click.option(
    "--compression/--no-compression",
    default=True,
    help="Enable/disable cache compression",
)
@click.option(
    "--max-chunk-days",
    type=int,
    default=90,
    help="Maximum days per parallel chunk (default: 90)",
)
@click.option(
    "--performance-metrics",
    is_flag=True,
    help="Show performance metrics after query execution",
)
@click.pass_context
def query(
    ctx,
    query: str,
    profile: Optional[str],
    fresh: bool,
    output_format: Optional[str],
    llm_provider: Optional[str],
    config_file: Optional[str],
    parallel: bool,
    compression: bool,
    max_chunk_days: int,
    performance_metrics: bool,
):
    """Query AWS costs using natural language.

    Examples:
        aws-cost-cli query "How much did I spend on EC2 last month?"
        aws-cost-cli query "What are my S3 costs this year?" --profile production
        aws-cost-cli query "Show me RDS spending for Q1" --fresh --format rich
        aws-cost-cli query "EC2 costs last month" --llm-provider gemini
    """
    try:
        # Create query context
        context = QueryContext(
            original_query=query,
            profile=profile,
            fresh_data=fresh,
            output_format=output_format.lower() if output_format else None,
            debug=ctx.obj.get("debug", False),
            enable_parallel=parallel,
            enable_compression=compression,
            max_chunk_days=max_chunk_days,
            show_performance_metrics=performance_metrics,
            llm_provider_override=llm_provider.lower() if llm_provider else None,
        )

        # Validate provider override if specified
        if llm_provider:
            valid_providers = ["openai", "anthropic", "bedrock", "ollama", "gemini"]
            if llm_provider.lower() not in valid_providers:
                console.print(
                    Panel(
                        Text(
                            f"Invalid LLM provider '{llm_provider}'. Available providers: {', '.join(valid_providers)}",
                            style="bold red",
                        ),
                        title="Invalid Provider",
                        border_style="red",
                    )
                )
                sys.exit(1)

        # Initialize pipeline
        pipeline = QueryPipeline(config_path=config_file)

        # Override output format if specified
        if output_format:
            pipeline.config.output_format = output_format.lower()
            context.output_format = output_format.lower()
        else:
            context.output_format = pipeline.config.output_format

        # Process query through pipeline
        if context.output_format != "json":
            console.print(f"🔍 Processing query: '{query}'")

        result = pipeline.process_query(context)

        # Handle result
        if result.success:
            if context.output_format == "json":
                # JSON output for programmatic use
                output = {
                    "query": query,
                    "success": True,
                    "total_cost": {
                        "amount": float(result.cost_data.total_cost.amount),
                        "currency": result.cost_data.total_cost.unit,
                    },
                    "time_period": {
                        "start": result.cost_data.time_period.start.isoformat(),
                        "end": result.cost_data.time_period.end.isoformat(),
                    },
                    "metadata": result.metadata,
                    "results": [],
                }

                for cost_result in result.cost_data.results:
                    result_data = {
                        "period": {
                            "start": cost_result.time_period.start.isoformat(),
                            "end": cost_result.time_period.end.isoformat(),
                        },
                        "total": {
                            "amount": float(cost_result.total.amount),
                            "currency": cost_result.total.unit,
                        },
                        "estimated": cost_result.estimated,
                        "groups": [],
                    }

                    for group in cost_result.groups:
                        group_data = {"keys": group.keys, "metrics": {}}
                        for metric_name, cost_amount in group.metrics.items():
                            group_data["metrics"][metric_name] = {
                                "amount": float(cost_amount.amount),
                                "currency": cost_amount.unit,
                            }
                        result_data["groups"].append(group_data)

                    output["results"].append(result_data)

                click.echo(json.dumps(output, indent=2))
            else:
                # Human-readable output
                console.print(result.formatted_response)

                # Show processing info if debug mode
                if context.debug:
                    console.print(
                        f"\n📊 Processing time: {result.processing_time_ms:.1f}ms"
                    )
                    if result.cache_hit:
                        console.print("📋 Data source: Cache")
                    else:
                        console.print("☁️  Data source: AWS API")

                    if result.llm_used:
                        console.print("🤖 Query parsing: LLM")
                    elif result.fallback_used:
                        console.print("🔧 Query parsing: Fallback")

                # Show performance metrics if requested
                if context.show_performance_metrics:
                    console.print("\n🚀 Performance Metrics:")
                    console.print(
                        f"   Processing time: {result.processing_time_ms:.1f}ms"
                    )
                    console.print(f"   API calls made: {result.api_calls_made}")
                    console.print(f"   Parallel requests: {result.parallel_requests}")
                    console.print(
                        f"   Cache hit: {'Yes' if result.cache_hit else 'No'}"
                    )

                    if result.compression_stats:
                        stats = result.compression_stats
                        console.print(
                            f"   Compression ratio: {stats.get('average_compression_ratio', 0):.2f}"
                        )
                        console.print(
                            f"   Space saved: {stats.get('space_saved_percent', 0):.1f}%"
                        )

                    if result.performance_metrics:
                        perf = result.performance_metrics
                        if "query_performance" in perf:
                            qp = perf["query_performance"]
                            if "cache_hit_rate" in qp:
                                console.print(
                                    f"   Cache hit rate: {qp['cache_hit_rate']:.1%}"
                                )
                            if (
                                "performance" in qp
                                and "avg_duration_ms" in qp["performance"]
                            ):
                                console.print(
                                    f"   Avg query time: {qp['performance']['avg_duration_ms']:.1f}ms"
                                )
        else:
            # Handle error
            error = result.error

            if context.output_format == "json":
                output = {
                    "query": query,
                    "success": False,
                    "error": {
                        "type": error.__class__.__name__,
                        "message": error.message,
                        "code": getattr(error, "error_code", None),
                    },
                    "metadata": result.metadata,
                }
                click.echo(json.dumps(output, indent=2))
            else:
                console.print(
                    Panel(
                        Text(error.message, style="bold red"),
                        title="Error",
                        border_style="red",
                    )
                )
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
        console.print(
            Panel(
                Text(f"❌ Unexpected error: {str(e)}", style="bold red"),
                title="Error",
                border_style="red",
            )
        )
        if ctx.obj.get("debug"):
            import traceback

            console.print(traceback.format_exc())
        sys.exit(1)


@cli.command()
@click.argument("partial_query", required=False)
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True),
    help="Path to configuration file",
)
def suggest(partial_query: Optional[str], config_file: Optional[str]):
    """Get query suggestions based on partial input.

    Examples:
        aws-cost-cli suggest "EC2"
        aws-cost-cli suggest "last month"
        aws-cost-cli suggest
    """
    try:
        # Initialize pipeline and interactive builder
        pipeline = QueryPipeline(config_path=config_file)
        builder = InteractiveQueryBuilder(
            query_parser=pipeline.query_parser, config_path=config_file
        )

        # Get suggestions from both pipeline and interactive builder
        pipeline_suggestions = pipeline.get_query_suggestions(partial_query or "")
        builder_suggestions = builder.get_query_suggestions(partial_query or "")

        # Combine and deduplicate suggestions
        all_suggestions = pipeline_suggestions + builder_suggestions
        seen = set()
        unique_suggestions = []
        for suggestion in all_suggestions:
            if suggestion not in seen:
                seen.add(suggestion)
                unique_suggestions.append(suggestion)

        console.print(
            Panel(Text("Query Suggestions", style="bold blue"), border_style="blue")
        )

        if partial_query:
            console.print(f"💡 Suggestions for '{partial_query}':")
        else:
            console.print("💡 Common query examples:")

        for i, suggestion in enumerate(unique_suggestions[:15], 1):
            console.print(f"   {i}. {suggestion}")

        console.print("\n🔍 Use these examples as templates for your own queries!")
        console.print("💡 Try 'aws-cost-cli interactive' for guided query building!")

    except Exception as e:
        console.print(
            Panel(
                Text(f"❌ Failed to get suggestions: {str(e)}", style="bold red"),
                title="Error",
                border_style="red",
            )
        )
        sys.exit(1)


@cli.command()
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True),
    help="Path to configuration file",
)
def interactive(config_file: Optional[str]):
    """Start interactive query builder for guided query construction.

    The interactive mode helps you build queries step by step with:
    - Guided query construction with prompts
    - Query templates for common use cases
    - Query history and favorites management
    - Query validation and suggestions

    Examples:
        aws-cost-cli interactive
        aws-cost-cli interactive --config-file custom_config.yaml
    """
    try:
        # Initialize pipeline to get query parser
        pipeline = QueryPipeline(config_path=config_file)

        # Create interactive query builder
        builder = InteractiveQueryBuilder(
            query_parser=pipeline.query_parser, config_path=config_file
        )

        # Start interactive session
        query = builder.start_interactive_session()

        if query:
            console.print(f"\n🚀 Executing query: '{query}'")

            # Create query context
            context = QueryContext(
                original_query=query,
                profile=None,
                fresh_data=False,
                output_format=pipeline.config.output_format,
                debug=False,
            )

            # Process the query
            result = pipeline.process_query(context)

            # Add to history
            builder.history_manager.add_to_history(
                query=query,
                success=result.success,
                execution_time_ms=result.processing_time_ms,
                error_message=result.error.message if result.error else None,
            )

            # Show result
            if result.success:
                console.print(result.formatted_response)

                # Ask if user wants to save as favorite
                if Confirm.ask("\n⭐ Save this query as a favorite?", default=False):
                    name = Prompt.ask("Favorite name")
                    description = Prompt.ask("Description (optional)", default="")
                    try:
                        builder.history_manager.add_favorite(
                            name, query, description or None
                        )
                        console.print(f"✅ Saved as favorite '{name}'")
                    except Exception as e:
                        console.print(f"❌ Failed to save favorite: {e}")
            else:
                console.print(
                    Panel(
                        Text(result.error.message, style="bold red"),
                        title="Query Error",
                        border_style="red",
                    )
                )

    except KeyboardInterrupt:
        console.print("\n👋 Interactive session cancelled")
    except Exception as e:
        console.print(
            Panel(
                Text(f"❌ Interactive session failed: {str(e)}", style="bold red"),
                title="Error",
                border_style="red",
            )
        )
        sys.exit(1)


@cli.command()
@click.option(
    "--action",
    type=click.Choice(["list", "add", "remove", "run"], case_sensitive=False),
    default="list",
    help="Action to perform on favorites",
)
@click.option("--name", help="Name of the favorite (for add/remove/run actions)")
@click.option("--query", help="Query to save as favorite (for add action)")
@click.option("--description", help="Description for the favorite (for add action)")
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True),
    help="Path to configuration file",
)
def favorites(
    action: str,
    name: Optional[str],
    query: Optional[str],
    description: Optional[str],
    config_file: Optional[str],
):
    """Manage favorite queries.

    Examples:
        aws-cost-cli favorites --action list
        aws-cost-cli favorites --action add --name "Monthly EC2" --query "EC2 costs last month"
        aws-cost-cli favorites --action remove --name "Monthly EC2"
        aws-cost-cli favorites --action run --name "Monthly EC2"
    """
    try:
        from .interactive_query_builder import QueryHistoryManager

        history_manager = QueryHistoryManager()

        if action == "list":
            favorites_list = history_manager.get_favorites()

            if not favorites_list:
                console.print("📭 No favorites found")
                return

            console.print(
                Panel(
                    Text("⭐ Favorite Queries", style="bold blue"), border_style="blue"
                )
            )

            from rich.table import Table

            table = Table()
            table.add_column("Name", style="cyan")
            table.add_column("Query", style="white")
            table.add_column("Description", style="dim")
            table.add_column("Created", style="dim")

            for favorite in favorites_list:
                table.add_row(
                    favorite.name,
                    (
                        favorite.query[:50] + "..."
                        if len(favorite.query) > 50
                        else favorite.query
                    ),
                    favorite.description or "",
                    favorite.created_at.strftime("%Y-%m-%d"),
                )

            console.print(table)

        elif action == "add":
            if not name or not query:
                console.print("❌ Both --name and --query are required for add action")
                sys.exit(1)

            try:
                history_manager.add_favorite(name, query, description)
                console.print(f"✅ Added favorite '{name}'")
            except Exception as e:
                console.print(f"❌ Failed to add favorite: {e}")
                sys.exit(1)

        elif action == "remove":
            if not name:
                console.print("❌ --name is required for remove action")
                sys.exit(1)

            if history_manager.remove_favorite(name):
                console.print(f"✅ Removed favorite '{name}'")
            else:
                console.print(f"❌ Favorite '{name}' not found")
                sys.exit(1)

        elif action == "run":
            if not name:
                console.print("❌ --name is required for run action")
                sys.exit(1)

            favorite = history_manager.get_favorite_by_name(name)
            if not favorite:
                console.print(f"❌ Favorite '{name}' not found")
                sys.exit(1)

            console.print(f"🚀 Running favorite query: '{favorite.query}'")

            # Initialize pipeline and run the query
            pipeline = QueryPipeline(config_path=config_file)

            context = QueryContext(
                original_query=favorite.query,
                profile=None,
                fresh_data=False,
                output_format=pipeline.config.output_format,
                debug=False,
            )

            result = pipeline.process_query(context)

            # Add to history
            history_manager.add_to_history(
                query=favorite.query,
                success=result.success,
                execution_time_ms=result.processing_time_ms,
                error_message=result.error.message if result.error else None,
            )

            # Show result
            if result.success:
                console.print(result.formatted_response)
            else:
                console.print(
                    Panel(
                        Text(result.error.message, style="bold red"),
                        title="Query Error",
                        border_style="red",
                    )
                )
                sys.exit(1)

    except Exception as e:
        console.print(
            Panel(
                Text(f"❌ Favorites operation failed: {str(e)}", style="bold red"),
                title="Error",
                border_style="red",
            )
        )
        sys.exit(1)


@cli.command()
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True),
    help="Path to configuration file",
)
def pipeline_status(config_file: Optional[str]):
    """Show pipeline status and health."""
    try:
        # Initialize pipeline
        pipeline = QueryPipeline(config_path=config_file)

        # Get status
        status = pipeline.get_pipeline_status()

        console.print(
            Panel(Text("Pipeline Status", style="bold blue"), border_style="blue")
        )

        # Component status
        console.print("🔧 Components:")
        console.print(f"   Config loaded: {'✅' if status['config_loaded'] else '❌'}")
        console.print(
            f"   Cache manager: {'✅' if status['cache_manager_initialized'] else '❌'}"
        )
        console.print(
            f"   Query parser: {'✅' if status['query_parser_initialized'] else '❌'}"
        )
        console.print(
            f"   AWS client: {'✅' if status['aws_client_initialized'] else '❌'}"
        )
        console.print(
            f"   Response generator: {'✅' if status['response_generator_initialized'] else '❌'}"
        )

        # Health status
        console.print("\n🏥 Health:")
        if "cache_healthy" in status:
            console.print(f"   Cache: {'✅' if status['cache_healthy'] else '❌'}")
            if status.get("cache_entries"):
                console.print(f"   Cache entries: {status['cache_entries']}")

        if "aws_service_healthy" in status:
            console.print(
                f"   AWS service: {'✅' if status['aws_service_healthy'] else '❌'}"
            )
            if status.get("aws_response_time_ms"):
                console.print(
                    f"   AWS response time: {status['aws_response_time_ms']:.1f}ms"
                )

    except Exception as e:
        console.print(
            Panel(
                Text(f"❌ Failed to get pipeline status: {str(e)}", style="bold red"),
                title="Error",
                border_style="red",
            )
        )
        sys.exit(1)


@cli.command()
@click.option(
    "--provider",
    type=click.Choice(
        ["openai", "anthropic", "bedrock", "ollama", "gemini"], case_sensitive=False
    ),
    required=True,
    help="LLM provider to configure",
)
@click.option(
    "--api-key", help="API key for the LLM provider (not needed for Ollama or Bedrock)"
)
@click.option(
    "--model",
    help="Model to use (e.g., gpt-3.5-turbo, claude-3-haiku-20240307, anthropic.claude-3-haiku-20240307-v1:0, gpt-oss:20b, gemini-1.5-flash, gemini-1.5-pro)",
)
@click.option(
    "--base-url", help="Base URL for Ollama (default: http://localhost:11434)"
)
@click.option(
    "--timeout", type=int, help="Request timeout for Ollama in seconds (default: 60)"
)
@click.option("--region", help="AWS region for Bedrock (default: us-east-1)")
@click.option(
    "--profile",
    help="AWS profile for Bedrock (uses default AWS credentials if not specified)",
)
@click.option(
    "--temperature",
    type=float,
    help="Temperature for LLM responses (0.0-1.0, default varies by provider)",
)
@click.option(
    "--max-tokens",
    type=int,
    help="Maximum tokens for LLM responses (default varies by provider)",
)
@click.option(
    "--config-file",
    "-c",
    type=click.Path(),
    help="Path to configuration file (will be created if it doesn't exist)",
)
def configure(
    provider: str,
    api_key: Optional[str],
    model: Optional[str],
    base_url: Optional[str],
    timeout: Optional[int],
    region: Optional[str],
    profile: Optional[str],
    temperature: Optional[float],
    max_tokens: Optional[int],
    config_file: Optional[str],
):
    """Configure LLM provider settings.

    Default provider is 'ollama' for local processing without API keys.

    Examples:
        aws-cost-cli configure --provider ollama --model llama2  # Default, local processing
        aws-cost-cli configure --provider openai --api-key sk-...
        aws-cost-cli configure --provider anthropic --api-key sk-ant-...
        aws-cost-cli configure --provider gemini --api-key your-gemini-api-key --model gemini-1.5-flash
        aws-cost-cli configure --provider bedrock --model anthropic.claude-3-haiku-20240307-v1:0 --region us-east-1
        aws-cost-cli configure --provider gemini --api-key your-key --model gemini-1.5-pro --temperature 0.2 --max-tokens 1000
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

        # Initialize provider-specific configuration
        provider_key = provider.lower()
        if provider_key not in config.llm_config:
            config.llm_config[provider_key] = {}

        provider_config = config.llm_config[provider_key]

        # Set API key if provided
        if api_key:
            provider_config["api_key"] = api_key

        # Set model with provider-specific defaults
        if model:
            provider_config["model"] = model
        elif "model" not in provider_config:
            if provider_key == "openai":
                provider_config["model"] = "gpt-3.5-turbo"
            elif provider_key == "anthropic":
                provider_config["model"] = "claude-3-haiku-20240307"
            elif provider_key == "bedrock":
                provider_config["model"] = "anthropic.claude-3-haiku-20240307-v1:0"
            elif provider_key == "ollama":
                provider_config["model"] = "gpt-oss:20b"
            elif provider_key == "gemini":
                provider_config["model"] = "gemini-1.5-flash"

        # Set provider-specific options
        if base_url:
            provider_config["base_url"] = base_url
        elif provider_key == "ollama" and "base_url" not in provider_config:
            provider_config["base_url"] = "http://localhost:11434"

        if timeout:
            provider_config["timeout"] = timeout
        elif provider_key == "ollama" and "timeout" not in provider_config:
            provider_config["timeout"] = 60

        if region:
            provider_config["region"] = region
        elif provider_key == "bedrock" and "region" not in provider_config:
            provider_config["region"] = "us-east-1"

        if profile:
            provider_config["profile"] = profile

        # Set temperature if provided
        if temperature is not None:
            if temperature < 0.0 or temperature > 1.0:
                raise click.BadParameter("Temperature must be between 0.0 and 1.0")
            provider_config["temperature"] = temperature
        elif "temperature" not in provider_config:
            # Set provider-specific default temperatures
            if provider_key == "gemini":
                provider_config["temperature"] = 0.1
            elif provider_key == "openai":
                provider_config["temperature"] = 0.1
            elif provider_key == "anthropic":
                provider_config["temperature"] = 0.1

        # Set max_tokens if provided
        if max_tokens is not None:
            if max_tokens <= 0:
                raise click.BadParameter("Max tokens must be positive")
            provider_config["max_tokens"] = max_tokens
        elif "max_tokens" not in provider_config:
            # Set provider-specific default max_tokens
            if provider_key == "gemini":
                provider_config["max_tokens"] = 500
            elif provider_key == "openai":
                provider_config["max_tokens"] = 500
            elif provider_key == "anthropic":
                provider_config["max_tokens"] = 500

        # Save configuration
        config_manager.save_config(config, str(config_path))

        console.print(
            Panel(
                Text("✅ Configuration saved successfully", style="bold green"),
                title="Configuration Updated",
                border_style="green",
            )
        )

        console.print(f"📁 Config file: {config_path}")
        console.print(f"🤖 Provider: {provider}")
        if model:
            console.print(f"🧠 Model: {model}")
        if base_url:
            console.print(f"🌐 Base URL: {base_url}")
        if region:
            console.print(f"🌍 Region: {region}")
        if profile:
            console.print(f"👤 AWS Profile: {profile}")
        if temperature is not None:
            console.print(f"🌡️  Temperature: {temperature}")
        if max_tokens is not None:
            console.print(f"📏 Max Tokens: {max_tokens}")

        # Test the configuration
        console.print("\n🧪 Testing configuration...")

        try:
            from dataclasses import asdict

            config_dict = asdict(config)
            query_parser = QueryParser(config.llm_config, config_dict)
            _test_result = query_parser.parse_query("test query for configuration")
            console.print("✅ LLM provider configuration is working")
        except Exception as e:
            console.print(f"⚠️  Configuration test failed: {e}")
            console.print("   The configuration was saved but may not work correctly")

    except Exception as e:
        console.print(
            Panel(
                Text(f"❌ Configuration failed: {str(e)}", style="bold red"),
                title="Configuration Error",
                border_style="red",
            )
        )
        sys.exit(1)


@cli.command()
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True),
    help="Path to configuration file",
)
def list_profiles(config_file: Optional[str]):
    """List available AWS profiles."""
    try:
        credential_manager = CredentialManager()
        profiles = credential_manager.get_available_profiles()

        if not profiles:
            console.print(
                Panel(
                    Text("No AWS profiles found", style="bold yellow"),
                    title="AWS Profiles",
                    border_style="yellow",
                )
            )
            console.print("\n💡 To create AWS profiles:")
            console.print("   1. Run: aws configure --profile <profile-name>")
            console.print("   2. Or edit ~/.aws/credentials manually")
            return

        console.print(
            Panel(
                Text("Available AWS Profiles", style="bold blue"), border_style="blue"
            )
        )

        for i, profile in enumerate(profiles, 1):
            # Check if profile has valid credentials
            is_valid = credential_manager.validate_credentials(profile)
            status = "✅" if is_valid else "❌"
            console.print(f"  {i}. {profile} {status}")

        console.print("\n💡 Use --profile <name> to specify a profile for queries")

    except Exception as e:
        console.print(
            Panel(
                Text(f"❌ Failed to list profiles: {str(e)}", style="bold red"),
                title="Error",
                border_style="red",
            )
        )
        sys.exit(1)


_SECRET_KEY_HINTS = ("api_key", "apikey", "secret", "token", "password")


def _mask_secret(value: Any) -> str:
    """Mask a secret value, showing only a short prefix."""
    text = str(value)
    return f"{text[:8]}..." if len(text) > 8 else "***"


def _print_masked_config(data: dict, indent: str = "   ") -> None:
    """Print a config mapping, masking any secret value at any nesting depth."""
    for key, value in data.items():
        if isinstance(value, dict):
            console.print(f"{indent}{key}:")
            _print_masked_config(value, indent + "   ")
        elif any(hint in str(key).lower() for hint in _SECRET_KEY_HINTS):
            console.print(f"{indent}{key}: {_mask_secret(value)}")
        else:
            console.print(f"{indent}{key}: {value}")


@cli.command()
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True),
    help="Path to configuration file",
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
                console.print(
                    Panel(
                        Text("No configuration file found", style="bold yellow"),
                        title="Configuration",
                        border_style="yellow",
                    )
                )
                console.print(f"\n💡 Default config location: {config_path}")
                console.print(
                    "   Run 'aws-cost-cli configure' to create a configuration"
                )
                return

        console.print(
            Panel(Text("Current Configuration", style="bold blue"), border_style="blue")
        )

        console.print(f"📁 Config file: {config_path}")
        console.print(f"🤖 LLM Provider: {config.llm_provider}")
        console.print(f"📊 Output Format: {config.output_format}")
        console.print(f"⏰ Cache TTL: {config.cache_ttl} seconds")
        console.print(f"💰 Default Currency: {config.default_currency}")

        if config.default_profile:
            console.print(f"👤 Default Profile: {config.default_profile}")

        if config.llm_config:
            console.print("\n🧠 LLM Configuration:")
            _print_masked_config(config.llm_config)

    except Exception as e:
        console.print(
            Panel(
                Text(f"❌ Failed to show configuration: {str(e)}", style="bold red"),
                title="Error",
                border_style="red",
            )
        )
        sys.exit(1)


@cli.command()
@click.confirmation_option(prompt="Are you sure you want to clear the cache?")
@click.option(
    "--pattern", help="Pattern to match cache files (clears all if not specified)"
)
def clear_cache(pattern: Optional[str]):
    """Clear the cost data cache."""
    try:
        cache_manager = CacheManager()
        removed_count = (
            cache_manager.clear_cache()
            if not pattern
            else cache_manager.invalidate_cache(pattern)
        )

        console.print(
            Panel(
                Text(
                    f"✅ Cache cleared successfully ({removed_count} files removed)",
                    style="bold green",
                ),
                title="Cache Management",
                border_style="green",
            )
        )

    except Exception as e:
        console.print(
            Panel(
                Text(f"❌ Failed to clear cache: {str(e)}", style="bold red"),
                title="Error",
                border_style="red",
            )
        )
        sys.exit(1)


@cli.command()
@click.option("--profile", "-p", help="AWS profile to use")
def cache_stats(profile: Optional[str]):
    """Show cache statistics."""
    try:
        cache_manager = CacheManager()
        stats = cache_manager.get_cache_stats()

        console.print(
            Panel(Text("Cache Statistics", style="bold blue"), border_style="blue")
        )

        console.print(f"📊 Total entries: {stats['total_entries']}")
        console.print(f"✅ Valid entries: {stats['valid_entries']}")
        console.print(f"⏰ Expired entries: {stats['expired_entries']}")
        console.print(f"💾 Cache size: {stats['cache_size_bytes']:,} bytes")

        if stats["oldest_entry"]:
            console.print(f"📅 Oldest entry: {stats['oldest_entry']}")
        if stats["newest_entry"]:
            console.print(f"🆕 Newest entry: {stats['newest_entry']}")

    except Exception as e:
        console.print(
            Panel(
                Text(f"❌ Failed to get cache statistics: {str(e)}", style="bold red"),
                title="Error",
                border_style="red",
            )
        )
        sys.exit(1)


@cli.command()
@click.option("--profile", "-p", help="AWS profile to use")
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True),
    help="Path to configuration file",
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
            console.print(
                Panel(
                    Text(f"❌ {results['error']}", style="bold red"),
                    title="Cache Warming Error",
                    border_style="red",
                )
            )
            sys.exit(1)

        console.print(
            Panel(
                Text("✅ Cache warming completed", style="bold green"),
                title="Cache Management",
                border_style="green",
            )
        )

        console.print(f"🔥 Queries warmed: {results['queries_warmed']}")
        console.print(f"❌ Queries failed: {results['queries_failed']}")

        if results["errors"]:
            console.print("\n⚠️  Errors encountered:")
            for error in results["errors"]:
                console.print(f"   • {error}")

    except (AWSCredentialsError, AWSPermissionsError, AWSAPIError, NetworkError) as e:
        console.print(
            Panel(Text(e.message, style="bold red"), title="Error", border_style="red")
        )
        console.print(format_error_message(e, include_suggestions=True))
        sys.exit(1)
    except Exception as e:
        console.print(
            Panel(
                Text(f"❌ Failed to warm cache: {str(e)}", style="bold red"),
                title="Error",
                border_style="red",
            )
        )
        sys.exit(1)


@cli.command()
def cleanup_cache():
    """Clean up expired cache entries."""
    try:
        cache_manager = CacheManager()
        removed_count = cache_manager.cleanup_expired_cache()

        console.print(
            Panel(
                Text(
                    f"✅ Cache cleanup completed ({removed_count} expired entries removed)",
                    style="bold green",
                ),
                title="Cache Management",
                border_style="green",
            )
        )

    except Exception as e:
        console.print(
            Panel(
                Text(f"❌ Failed to cleanup cache: {str(e)}", style="bold red"),
                title="Error",
                border_style="red",
            )
        )
        sys.exit(1)


@cli.command()
@click.option(
    "--hours",
    type=int,
    default=24,
    help="Hours to look back for performance metrics (default: 24)",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["simple", "json"], case_sensitive=False),
    default="simple",
    help="Output format",
)
def performance(hours: int, output_format: str):
    """Show performance metrics and cache statistics."""
    try:
        from .performance import PerformanceMonitor, CompressedCacheManager

        # Initialize components
        cache_manager = CacheManager()
        monitor = PerformanceMonitor()
        compressed_cache = CompressedCacheManager(cache_manager)

        # Get performance summary
        perf_summary = monitor.get_performance_summary(hours)
        compression_stats = compressed_cache.get_compression_stats()
        cache_stats = cache_manager.get_cache_stats()

        if output_format == "json":
            output = {
                "performance_summary": perf_summary,
                "compression_stats": compression_stats,
                "cache_stats": cache_stats,
            }
            click.echo(json.dumps(output, indent=2))
        else:
            console.print(f"\n📊 Performance Summary (Last {hours} hours)")

            if "total_queries" in perf_summary:
                console.print(f"   Total queries: {perf_summary['total_queries']}")
                console.print(
                    f"   Cache hit rate: {perf_summary.get('cache_hit_rate', 0):.1%}"
                )
                console.print(f"   Error rate: {perf_summary.get('error_rate', 0):.1%}")
                console.print(
                    f"   Total API calls: {perf_summary.get('total_api_calls', 0)}"
                )

                if "performance" in perf_summary:
                    perf = perf_summary["performance"]
                    console.print(
                        f"   Avg response time: {perf.get('avg_duration_ms', 0):.1f}ms"
                    )
                    console.print(
                        f"   95th percentile: {perf.get('p95_duration_ms', 0):.1f}ms"
                    )
            else:
                console.print("   No performance data available")

            console.print("\n💾 Cache Statistics")
            console.print(f"   Total entries: {cache_stats.get('total_entries', 0)}")
            console.print(f"   Valid entries: {cache_stats.get('valid_entries', 0)}")
            console.print(
                f"   Expired entries: {cache_stats.get('expired_entries', 0)}"
            )
            console.print(
                f"   Cache size: {cache_stats.get('cache_size_bytes', 0) / 1024 / 1024:.1f} MB"
            )

            console.print("\n🗜️  Compression Statistics")
            console.print(
                f"   Compressed files: {compression_stats.get('compressed_files', 0)}"
            )
            console.print(
                f"   Avg compression ratio: {compression_stats.get('average_compression_ratio', 0):.2f}"
            )
            console.print(
                f"   Space saved: {compression_stats.get('space_saved_percent', 0):.1f}%"
            )
            console.print(
                f"   Total space saved: {compression_stats.get('total_space_saved', 0) / 1024 / 1024:.1f} MB"
            )

    except Exception as e:
        console.print(
            Panel(
                Text(f"Failed to get performance metrics: {str(e)}", style="bold red"),
                title="Error",
                border_style="red",
            )
        )
        sys.exit(1)


@cli.command()
@click.argument("query", required=True)
@click.option(
    "--format",
    "export_format",
    type=click.Choice(["csv", "json", "excel"], case_sensitive=False),
    default="csv",
    help="Export format (default: csv)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file path (auto-generated if not specified)",
)
@click.option("--profile", "-p", help="AWS profile to use")
@click.option(
    "--fresh", "-f", is_flag=True, help="Force fresh data retrieval, bypassing cache"
)
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True),
    help="Path to configuration file",
)
@click.pass_context
def export(
    ctx,
    query: str,
    export_format: str,
    output: Optional[str],
    profile: Optional[str],
    fresh: bool,
    config_file: Optional[str],
):
    """Export cost data to file.

    Examples:
        aws-cost-cli export "EC2 costs last month" --format csv
        aws-cost-cli export "S3 spending this year" --format excel --output s3_costs.xlsx
        aws-cost-cli export "Total costs Q1" --format json --profile production
    """
    try:
        # Create query context
        context = QueryContext(
            original_query=query,
            profile=profile,
            fresh_data=fresh,
            output_format="json",  # Use JSON internally for data processing
            debug=ctx.obj.get("debug", False),
        )

        # Initialize pipeline
        pipeline = QueryPipeline(config_path=config_file)

        # Process query
        console.print(f"🔍 Processing query: '{query}'")
        result = pipeline.process_query(context)

        if not result.success:
            console.print(
                Panel(
                    Text(result.error.message, style="bold red"),
                    title="Query Error",
                    border_style="red",
                )
            )
            sys.exit(1)

        # Generate output filename if not provided
        if not output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            service_part = (
                f"_{result.query_params.service.lower()}"
                if result.query_params.service
                else ""
            )
            output = (
                f"aws_cost_export{service_part}_{timestamp}.{export_format.lower()}"
            )

        # Export data
        console.print(f"📊 Exporting data to {export_format.upper()} format...")

        export_manager = ExportManager(
            date_formatting_config=pipeline.config.date_formatting
        )

        # Check if format is available
        if export_format.lower() not in export_manager.get_available_formats():
            available = ", ".join(export_manager.get_available_formats())
            console.print(
                Panel(
                    Text(
                        f"Export format '{export_format}' is not available. Available formats: {available}",
                        style="bold red",
                    ),
                    title="Export Error",
                    border_style="red",
                )
            )

            if export_format.lower() == "excel":
                console.print(
                    "💡 To enable Excel export, install openpyxl: pip install openpyxl"
                )

            sys.exit(1)

        # Perform export
        exported_path = export_manager.export_data(
            result.cost_data, result.query_params, export_format.lower(), output
        )

        console.print(
            Panel(
                Text("✅ Data exported successfully", style="bold green"),
                title="Export Complete",
                border_style="green",
            )
        )

        console.print(f"📁 File: {exported_path}")
        console.print(f"📊 Format: {export_format.upper()}")
        console.print(f"💰 Total Cost: ${result.cost_data.total_cost.amount:,.2f}")
        console.print(
            f"📅 Period: {result.cost_data.time_period.start.date()} to {result.cost_data.time_period.end.date()}"
        )

        # Show file size
        try:
            file_size = Path(exported_path).stat().st_size
            if file_size < 1024:
                size_str = f"{file_size} bytes"
            elif file_size < 1024 * 1024:
                size_str = f"{file_size / 1024:.1f} KB"
            else:
                size_str = f"{file_size / (1024 * 1024):.1f} MB"
            console.print(f"📏 File Size: {size_str}")
        except Exception:
            pass

    except KeyboardInterrupt:
        console.print("\n👋 Export cancelled by user")
        sys.exit(0)
    except Exception as e:
        console.print(
            Panel(
                Text(f"❌ Export failed: {str(e)}", style="bold red"),
                title="Export Error",
                border_style="red",
            )
        )
        if ctx.obj.get("debug"):
            import traceback

            console.print(traceback.format_exc())
        sys.exit(1)


@cli.command()
@click.option("--profile", "-p", help="AWS profile to use")
@click.option(
    "--days", "-d", type=int, default=30, help="Number of days to analyze (default: 30)"
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["rich", "json"], case_sensitive=False),
    default="rich",
    help="Output format (default: rich)",
)
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True),
    help="Path to configuration file",
)
@click.pass_context
def optimize(
    ctx,
    profile: Optional[str],
    days: int,
    output_format: str,
    config_file: Optional[str],
):
    """Generate cost optimization recommendations.

    Analyzes your AWS costs and provides recommendations for:
    - Unused resources that can be terminated
    - Rightsizing opportunities for over-provisioned resources
    - Reserved Instance and Savings Plan recommendations
    - Cost anomaly detection
    - Budget variance analysis

    Examples:
        aws-cost-cli optimize
        aws-cost-cli optimize --days 60 --profile production
        aws-cost-cli optimize --format json > optimization_report.json
    """
    try:
        from .cost_optimizer import CostOptimizer, TimePeriod
        from .optimization_formatter import OptimizationFormatter

        # Validate AWS credentials
        credential_manager = CredentialManager()
        if not credential_manager.validate_credentials(profile):
            raise AWSCredentialsError(profile=profile)

        console.print("🔍 Analyzing AWS costs for optimization opportunities...")
        console.print(f"📅 Analysis period: Last {days} days")
        if profile:
            console.print(f"👤 AWS Profile: {profile}")

        # Initialize optimizer
        optimizer = CostOptimizer(profile=profile)

        # Define analysis period
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        analysis_period = TimePeriod(start=start_date, end=end_date)

        # Generate optimization report
        with console.status("[bold green]Generating optimization report..."):
            report = optimizer.generate_optimization_report(analysis_period)

        if output_format.lower() == "json":
            # JSON output for programmatic use
            output = {
                "report_date": report.report_date.isoformat(),
                "analysis_period": {
                    "start": report.analysis_period.start.isoformat(),
                    "end": report.analysis_period.end.isoformat(),
                },
                "total_potential_savings": {
                    "amount": float(report.total_potential_savings.amount),
                    "currency": report.total_potential_savings.unit,
                },
                "summary": {
                    "total_recommendations": len(report.recommendations),
                    "high_priority_recommendations": len(
                        [
                            r
                            for r in report.recommendations
                            if r.severity.value in ["high", "critical"]
                        ]
                    ),
                    "cost_anomalies": len(report.anomalies),
                    "budget_variances": len(report.budget_variances),
                },
                "recommendations": [],
                "anomalies": [],
                "budget_variances": [],
            }

            # Add recommendations
            for rec in report.recommendations:
                rec_data = {
                    "type": rec.type.value,
                    "severity": rec.severity.value,
                    "title": rec.title,
                    "description": rec.description,
                    "potential_savings": {
                        "amount": float(rec.potential_savings.amount),
                        "currency": rec.potential_savings.unit,
                    },
                    "confidence_level": rec.confidence_level,
                    "resource_id": rec.resource_id,
                    "service": rec.service,
                    "region": rec.region,
                    "action_required": rec.action_required,
                    "estimated_effort": rec.estimated_effort,
                    "metadata": rec.metadata,
                }
                output["recommendations"].append(rec_data)

            # Add anomalies
            for anomaly in report.anomalies:
                anomaly_data = {
                    "service": anomaly.service,
                    "anomaly_date": anomaly.anomaly_date.isoformat(),
                    "expected_cost": {
                        "amount": float(anomaly.expected_cost.amount),
                        "currency": anomaly.expected_cost.unit,
                    },
                    "actual_cost": {
                        "amount": float(anomaly.actual_cost.amount),
                        "currency": anomaly.actual_cost.unit,
                    },
                    "variance_percentage": anomaly.variance_percentage,
                    "severity": anomaly.severity.value,
                    "description": anomaly.description,
                    "root_cause_analysis": anomaly.root_cause_analysis,
                }
                output["anomalies"].append(anomaly_data)

            # Add budget variances
            for variance in report.budget_variances:
                variance_data = {
                    "budget_name": variance.budget_name,
                    "budgeted_amount": {
                        "amount": float(variance.budgeted_amount.amount),
                        "currency": variance.budgeted_amount.unit,
                    },
                    "actual_amount": {
                        "amount": float(variance.actual_amount.amount),
                        "currency": variance.actual_amount.unit,
                    },
                    "variance_amount": {
                        "amount": float(variance.variance_amount.amount),
                        "currency": variance.variance_amount.unit,
                    },
                    "variance_percentage": variance.variance_percentage,
                    "is_over_budget": variance.is_over_budget,
                    "time_period": {
                        "start": variance.time_period.start.isoformat(),
                        "end": variance.time_period.end.isoformat(),
                    },
                }
                output["budget_variances"].append(variance_data)

            click.echo(json.dumps(output, indent=2))
        else:
            # Rich formatted output
            formatter = OptimizationFormatter(console)
            formatted_report = formatter.format_optimization_report(report)
            console.print(formatted_report)

            # Show summary at the end
            if report.total_potential_savings.amount > 0:
                console.print(
                    Panel(
                        Text(
                            f"💰 Total Monthly Savings Potential: ${report.total_potential_savings.amount:,.2f}",
                            style="bold green",
                            justify="center",
                        ),
                        border_style="green",
                        padding=(1, 2),
                    )
                )
            else:
                console.print(
                    Panel(
                        Text(
                            "✅ No significant optimization opportunities found",
                            style="bold green",
                            justify="center",
                        ),
                        border_style="green",
                        padding=(1, 2),
                    )
                )

    except (AWSCredentialsError, AWSPermissionsError, AWSAPIError, NetworkError) as e:
        console.print(
            Panel(Text(e.message, style="bold red"), title="Error", border_style="red")
        )
        console.print(format_error_message(e, include_suggestions=True))
        sys.exit(1)
    except Exception as e:
        console.print(
            Panel(
                Text(f"❌ Optimization analysis failed: {str(e)}", style="bold red"),
                title="Error",
                border_style="red",
            )
        )
        if ctx.obj.get("debug"):
            import traceback

            console.print(traceback.format_exc())
        sys.exit(1)


@cli.command()
@click.option("--profile", "-p", help="AWS profile to use")
@click.option("--service", "-s", help="Filter by specific AWS service")
@click.option(
    "--days",
    "-d",
    type=int,
    default=7,
    help="Number of days to check for anomalies (default: 7)",
)
@click.option(
    "--threshold",
    type=float,
    default=20.0,
    help="Minimum variance percentage to report (default: 20.0)",
)
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True),
    help="Path to configuration file",
)
@click.pass_context
def detect_anomalies(
    ctx,
    profile: Optional[str],
    service: Optional[str],
    days: int,
    threshold: float,
    config_file: Optional[str],
):
    """Detect cost anomalies in your AWS spending.

    Analyzes recent spending patterns to identify unusual cost spikes
    or unexpected changes in your AWS bill.

    Examples:
        aws-cost-cli detect-anomalies
        aws-cost-cli detect-anomalies --service "Amazon EC2" --days 14
        aws-cost-cli detect-anomalies --threshold 50.0 --profile production
    """
    try:
        from .cost_optimizer import CostOptimizer, TimePeriod

        # Validate AWS credentials
        credential_manager = CredentialManager()
        if not credential_manager.validate_credentials(profile):
            raise AWSCredentialsError(profile=profile)

        console.print("🔍 Detecting cost anomalies...")
        console.print(f"📅 Analysis period: Last {days} days")
        console.print(f"📊 Variance threshold: {threshold}%")
        if service:
            console.print(f"🔧 Service filter: {service}")
        if profile:
            console.print(f"👤 AWS Profile: {profile}")

        # Initialize optimizer
        optimizer = CostOptimizer(profile=profile)

        # Define analysis period
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        analysis_period = TimePeriod(start=start_date, end=end_date)

        # Detect anomalies
        with console.status("[bold yellow]Analyzing cost patterns..."):
            anomalies = optimizer._detect_cost_anomalies(analysis_period)

        # Filter by service if specified
        if service:
            anomalies = [a for a in anomalies if service.lower() in a.service.lower()]

        # Filter by threshold
        anomalies = [a for a in anomalies if abs(a.variance_percentage) >= threshold]

        if not anomalies:
            console.print(
                Panel(
                    Text(
                        "✅ No significant cost anomalies detected", style="bold green"
                    ),
                    title="Anomaly Detection Results",
                    border_style="green",
                )
            )
            return

        # Display results
        console.print(
            Panel(
                Text(f"⚠️  {len(anomalies)} Cost Anomalies Detected", style="bold red"),
                title="Anomaly Detection Results",
                border_style="red",
            )
        )

        # Create anomalies table
        anomaly_table = Table()
        anomaly_table.add_column("Date", style="cyan")
        anomaly_table.add_column("Service", style="green")
        anomaly_table.add_column("Cost Impact", style="red")
        anomaly_table.add_column("Variance", style="yellow")
        anomaly_table.add_column("Severity", style="bold")
        anomaly_table.add_column("Description", style="white")

        # Sort by cost impact
        sorted_anomalies = sorted(
            anomalies, key=lambda x: x.actual_cost.amount, reverse=True
        )

        for anomaly in sorted_anomalies:
            severity_style = (
                "red" if anomaly.severity.value in ["high", "critical"] else "yellow"
            )
            anomaly_table.add_row(
                anomaly.anomaly_date.strftime("%Y-%m-%d"),
                anomaly.service,
                f"${anomaly.actual_cost.amount:,.2f}",
                f"{anomaly.variance_percentage:+.1f}%",
                Text(anomaly.severity.value.upper(), style=severity_style),
                anomaly.description,
            )

        console.print(anomaly_table)

        # Show total impact
        total_impact = sum(a.actual_cost.amount for a in anomalies)
        console.print(f"\n💰 Total Anomaly Impact: ${total_impact:,.2f}")

        # Show recommendations
        high_impact_anomalies = [a for a in anomalies if a.actual_cost.amount > 100]
        if high_impact_anomalies:
            console.print("\n💡 Recommendations:")
            console.print("   • Investigate high-impact anomalies immediately")
            console.print("   • Review resource usage patterns for affected services")
            console.print(
                "   • Consider setting up AWS Budgets alerts for early detection"
            )
            console.print(
                "   • Enable AWS Cost Anomaly Detection for automated monitoring"
            )

    except (AWSCredentialsError, AWSPermissionsError, AWSAPIError, NetworkError) as e:
        console.print(
            Panel(Text(e.message, style="bold red"), title="Error", border_style="red")
        )
        console.print(format_error_message(e, include_suggestions=True))
        sys.exit(1)
    except Exception as e:
        console.print(
            Panel(
                Text(f"❌ Anomaly detection failed: {str(e)}", style="bold red"),
                title="Error",
                border_style="red",
            )
        )
        if ctx.obj.get("debug"):
            import traceback

            console.print(traceback.format_exc())
        sys.exit(1)


@cli.command()
@click.argument("query", required=True)
@click.option(
    "--recipients", "-r", required=True, help="Comma-separated list of email recipients"
)
@click.option("--subject", "-s", help="Email subject (auto-generated if not specified)")
@click.option(
    "--attachments",
    type=click.Choice(["csv", "json", "excel"], case_sensitive=False),
    multiple=True,
    default=["csv"],
    help="Attachment formats to include (can specify multiple)",
)
@click.option("--smtp-host", required=True, help="SMTP server host")
@click.option(
    "--smtp-port", type=int, default=587, help="SMTP server port (default: 587)"
)
@click.option("--smtp-username", required=True, help="SMTP username")
@click.option("--smtp-password", required=True, help="SMTP password")
@click.option("--no-tls", is_flag=True, help="Disable TLS encryption")
@click.option("--profile", "-p", help="AWS profile to use")
@click.option(
    "--fresh", "-f", is_flag=True, help="Force fresh data retrieval, bypassing cache"
)
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True),
    help="Path to configuration file",
)
@click.pass_context
def email_report(
    ctx,
    query: str,
    recipients: str,
    subject: Optional[str],
    attachments: tuple,
    smtp_host: str,
    smtp_port: int,
    smtp_username: str,
    smtp_password: str,
    no_tls: bool,
    profile: Optional[str],
    fresh: bool,
    config_file: Optional[str],
):
    """Send cost report via email.

    Examples:
        aws-cost-cli email-report "EC2 costs last month" \\
            --recipients "admin@company.com,finance@company.com" \\
            --smtp-host smtp.gmail.com --smtp-username user@gmail.com --smtp-password password

        aws-cost-cli email-report "Monthly AWS costs" \\
            --recipients "team@company.com" --subject "Monthly Cost Report" \\
            --attachments csv --attachments excel \\
            --smtp-host mail.company.com --smtp-username reports --smtp-password secret
    """
    try:
        # Parse recipients
        recipient_list = [email.strip() for email in recipients.split(",")]

        # Create query context
        context = QueryContext(
            original_query=query,
            profile=profile,
            fresh_data=fresh,
            output_format="json",  # Use JSON internally
            debug=ctx.obj.get("debug", False),
        )

        # Initialize pipeline
        pipeline = QueryPipeline(config_path=config_file)

        # Process query
        console.print(f"🔍 Processing query: '{query}'")
        result = pipeline.process_query(context)

        if not result.success:
            console.print(
                Panel(
                    Text(result.error.message, style="bold red"),
                    title="Query Error",
                    border_style="red",
                )
            )
            sys.exit(1)

        # Prepare SMTP configuration
        smtp_config = {
            "host": smtp_host,
            "port": smtp_port,
            "username": smtp_username,
            "password": smtp_password,
            "use_tls": not no_tls,
        }

        # Send email report
        console.print(
            f"📧 Sending email report to {len(recipient_list)} recipient(s)..."
        )

        export_manager = ExportManager(
            date_formatting_config=pipeline.config.date_formatting
        )

        # Filter available attachment formats
        available_formats = export_manager.get_available_formats()
        attachment_formats = [
            fmt for fmt in attachments if fmt.lower() in available_formats
        ]

        if len(attachment_formats) != len(attachments):
            missing = [
                fmt for fmt in attachments if fmt.lower() not in available_formats
            ]
            console.print(f"⚠️  Skipping unavailable formats: {', '.join(missing)}")
            if "excel" in missing:
                console.print(
                    "💡 To enable Excel attachments, install openpyxl: pip install openpyxl"
                )

        success = export_manager.send_email_report(
            result.cost_data,
            result.query_params,
            smtp_config,
            recipient_list,
            subject,
            attachment_formats,
        )

        if success:
            console.print(
                Panel(
                    Text("✅ Email report sent successfully", style="bold green"),
                    title="Email Sent",
                    border_style="green",
                )
            )

            console.print(f"📧 Recipients: {', '.join(recipient_list)}")
            console.print(f"📊 Total Cost: ${result.cost_data.total_cost.amount:,.2f}")
            console.print(
                f"📅 Period: {result.cost_data.time_period.start.date()} to {result.cost_data.time_period.end.date()}"
            )

            if attachment_formats:
                console.print(
                    f"📎 Attachments: {', '.join(attachment_formats).upper()}"
                )
        else:
            console.print(
                Panel(
                    Text("❌ Failed to send email report", style="bold red"),
                    title="Email Error",
                    border_style="red",
                )
            )
            sys.exit(1)

    except KeyboardInterrupt:
        console.print("\n👋 Email sending cancelled by user")
        sys.exit(0)
    except Exception as e:
        console.print(
            Panel(
                Text(f"❌ Email sending failed: {str(e)}", style="bold red"),
                title="Email Error",
                border_style="red",
            )
        )
        if ctx.obj.get("debug"):
            import traceback

            console.print(traceback.format_exc())
        sys.exit(1)


@cli.command()
@click.option("--debug", is_flag=True, help="Enable debug output")
@click.pass_context
def test(ctx, debug: bool):
    """Test the CLI configuration and AWS connectivity."""
    ctx.obj["debug"] = debug

    console.print(
        Panel(
            Text("AWS Cost CLI - System Test", style="bold blue"), border_style="blue"
        )
    )

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
        from dataclasses import asdict

        config_dict = asdict(config)
        query_parser = QueryParser(config.llm_config, config_dict)
        # Try a simple test query
        _test_result = query_parser.parse_query("test")
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


@cli.group()
def health():
    """Health check and monitoring commands."""
    pass


@health.command("check")
@click.option(
    "--detailed",
    "-d",
    is_flag=True,
    help="Include detailed system metrics in health check",
)
@click.option(
    "--json", "output_json", is_flag=True, help="Output results in JSON format"
)
@click.option(
    "--config-file", "-c", type=click.Path(exists=True), help="Configuration file path"
)
def health_check(detailed: bool, output_json: bool, config_file: Optional[str]):
    """Perform comprehensive health check of the system."""
    try:
        # Load configuration if provided
        config = None
        if config_file:
            config_manager = ConfigManager()
            config = config_manager.load_config(config_file)

        health_checker = HealthChecker(config)
        health_status = health_checker.check_health(detailed=detailed)

        if output_json:
            # JSON output for programmatic use
            import json
            from dataclasses import asdict

            result = asdict(health_status)
            result["timestamp"] = result["timestamp"].isoformat()
            print(json.dumps(result, indent=2))
        else:
            # Rich formatted output for humans
            _display_health_status(health_status)

        # Exit with appropriate code
        if health_status.status == "unhealthy":
            sys.exit(1)
        elif health_status.status == "degraded":
            sys.exit(2)
        else:
            sys.exit(0)

    except Exception as e:
        console.print(
            Panel(
                Text(f"❌ Health check failed: {str(e)}", style="bold red"),
                title="Health Check Error",
                border_style="red",
            )
        )
        sys.exit(1)


@health.command("ready")
@click.option(
    "--json", "output_json", is_flag=True, help="Output results in JSON format"
)
def readiness_check(output_json: bool):
    """Check if the application is ready to serve requests."""
    try:
        health_checker = HealthChecker()
        readiness = health_checker.check_readiness()

        if output_json:
            import json

            print(json.dumps(readiness, indent=2))
        else:
            if readiness["ready"]:
                console.print("✅ Application is ready", style="bold green")
            else:
                console.print("❌ Application is not ready", style="bold red")

            # Show check details
            for check_name, check_result in readiness["checks"].items():
                status_emoji = "✅" if check_result["status"] == "ready" else "❌"
                console.print(
                    f"  {status_emoji} {check_name}: {check_result['message']}"
                )

        # Exit with appropriate code
        sys.exit(0 if readiness["ready"] else 1)

    except Exception as e:
        console.print(
            Panel(
                Text(f"❌ Readiness check failed: {str(e)}", style="bold red"),
                title="Readiness Check Error",
                border_style="red",
            )
        )
        sys.exit(1)


@health.command("serve")
@click.option(
    "--host",
    default="0.0.0.0",
    help="Host to bind the health check server (default: 0.0.0.0)",
)
@click.option(
    "--port",
    "-p",
    default=8081,
    type=int,
    help="Port to bind the health check server (default: 8081)",
)
@click.option(
    "--config-file", "-c", type=click.Path(exists=True), help="Configuration file path"
)
def serve_health_checks(host: str, port: int, config_file: Optional[str]):
    """Start HTTP server for health check endpoints.

    Endpoints:
    - GET /health - Basic health check
    - GET /health/detailed - Detailed health check with metrics
    - GET /ready - Readiness check
    - GET /metrics - Prometheus metrics
    """
    try:
        # Load configuration if provided
        config = None
        if config_file:
            config_manager = ConfigManager()
            config = config_manager.load_config(config_file)

        console.print(f"🚀 Starting health check server on {host}:{port}")
        console.print("Available endpoints:")
        console.print(f"  • http://{host}:{port}/health")
        console.print(f"  • http://{host}:{port}/health/detailed")
        console.print(f"  • http://{host}:{port}/ready")
        console.print(f"  • http://{host}:{port}/metrics")
        console.print("\nPress Ctrl+C to stop the server")

        server = create_health_check_server(host, port, config)
        server.serve_forever()

    except KeyboardInterrupt:
        console.print("\n👋 Health check server stopped")
    except Exception as e:
        console.print(
            Panel(
                Text(
                    f"❌ Failed to start health check server: {str(e)}",
                    style="bold red",
                ),
                title="Server Error",
                border_style="red",
            )
        )
        sys.exit(1)


def _display_health_status(health_status):
    """Display health status in a user-friendly format."""
    from rich.table import Table

    # Overall status
    status_color = {"healthy": "green", "degraded": "yellow", "unhealthy": "red"}

    status_emoji = {"healthy": "✅", "degraded": "⚠️", "unhealthy": "❌"}

    console.print(
        Panel(
            Text(
                f"{status_emoji[health_status.status]} System Status: {health_status.status.upper()}",
                style=f"bold {status_color[health_status.status]}",
            ),
            title="Health Check Results",
            border_style=status_color[health_status.status],
        )
    )

    # Create table for check results
    table = Table(show_header=True, header_style="bold blue")
    table.add_column("Component", style="cyan", width=15)
    table.add_column("Status", width=10)
    table.add_column("Details", style="dim")

    for component, check_result in health_status.checks.items():
        if not isinstance(check_result, dict):
            continue

        check_status = check_result.get("status", "unknown")
        check_emoji = status_emoji.get(check_status, "❓")

        # Format details
        details = []
        if "error" in check_result:
            details.append(f"Error: {check_result['error']}")
        if "warnings" in check_result and check_result["warnings"]:
            details.append(f"Warnings: {', '.join(check_result['warnings'])}")
        if "response_time_ms" in check_result:
            details.append(f"Response: {check_result['response_time_ms']}ms")
        if "size_mb" in check_result:
            details.append(f"Size: {check_result['size_mb']}MB")

        details_text = "; ".join(details) if details else "OK"

        table.add_row(component.title(), f"{check_emoji} {check_status}", details_text)

    console.print(table)

    # Show summary
    summary = health_status.summary
    console.print(
        f"\n📊 Summary: {summary['healthy_checks']}/{summary['total_checks']} checks healthy"
    )
    if summary["degraded_checks"] > 0:
        console.print(f"⚠️  {summary['degraded_checks']} checks degraded")
    if summary["unhealthy_checks"] > 0:
        console.print(f"❌ {summary['unhealthy_checks']} checks unhealthy")

    console.print(f"⏱️  Uptime: {summary['uptime_seconds']:.1f} seconds")


@cli.command()
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True),
    help="Path to configuration file",
)
def list_providers(config_file: Optional[str]):
    """List all available LLM providers and their configuration status.

    Shows which providers are configured and ready to use, and what's needed
    to configure providers that aren't set up yet.

    Examples:
        aws-cost-cli list-providers
        aws-cost-cli list-providers --config-file custom_config.yaml
    """
    try:
        # Load configuration
        config_manager = ConfigManager()
        try:
            config = config_manager.load_config(config_file)
        except FileNotFoundError:
            config = Config()

        # Get provider status
        provider_status = ProviderFactory.get_provider_configuration_status(
            config.llm_config
        )
        all_providers = ProviderFactory.get_all_provider_names()

        console.print(
            Panel(
                Text("🤖 LLM Provider Status", style="bold blue"), border_style="blue"
            )
        )

        # Create table for provider status
        table = Table(show_header=True, header_style="bold blue")
        table.add_column("Provider", style="cyan", width=12)
        table.add_column("Status", width=12)
        table.add_column("Configuration", style="dim")
        table.add_column("Requirements", style="dim")

        # Provider requirements mapping
        provider_requirements = {
            "openai": "OPENAI_API_KEY environment variable or api_key in config",
            "anthropic": "ANTHROPIC_API_KEY environment variable or api_key in config",
            "bedrock": "AWS credentials configured (uses default AWS profile)",
            "ollama": "Ollama server running (default: http://localhost:11434)",
            "gemini": "GEMINI_API_KEY environment variable or api_key in config",
        }

        for provider_name in all_providers:
            status_info = provider_status.get(provider_name, {})
            configured = status_info.get("configured", False)
            available = status_info.get("available", False)
            error = status_info.get("error")

            # Status display
            if available:
                status_display = "✅ Ready"
                status_style = "green"
            elif configured:
                status_display = "⚠️  Configured"
                status_style = "yellow"
            else:
                status_display = "❌ Not configured"
                status_style = "red"

            # Configuration details
            config_details = []
            if provider_name in config.llm_config:
                provider_config = config.llm_config[provider_name]
                if "model" in provider_config:
                    config_details.append(f"Model: {provider_config['model']}")
                if "base_url" in provider_config:
                    config_details.append(f"URL: {provider_config['base_url']}")
                if "region" in provider_config:
                    config_details.append(f"Region: {provider_config['region']}")

            config_text = (
                "; ".join(config_details) if config_details else "Default settings"
            )

            # Show error if present
            if error and not available:
                config_text = f"❌ {error}"

            # Requirements
            requirements = provider_requirements.get(
                provider_name, "No special requirements"
            )

            table.add_row(
                provider_name.title(),
                f"[{status_style}]{status_display}[/{status_style}]",
                config_text,
                requirements,
            )

        console.print(table)

        # Show current provider
        current_provider = config.llm_provider
        console.print(f"\n🎯 Current default provider: {current_provider}")

        # Show available providers
        available_providers = ProviderFactory.get_available_providers(config.llm_config)
        if available_providers:
            console.print(f"✅ Available providers: {', '.join(available_providers)}")
        else:
            console.print("❌ No providers are currently available")

        # Show configuration tips
        console.print("\n💡 Configuration tips:")
        console.print("   • Set environment variables for API keys (recommended)")
        console.print(
            "   • Use 'aws-cost-cli configure --provider <name>' to set up a provider"
        )
        console.print("   • Use 'aws-cost-cli test-provider <name>' to test a provider")
        console.print("   • Ollama is recommended for local/offline usage")

    except Exception as e:
        console.print(
            Panel(
                Text(f"❌ Failed to list providers: {str(e)}", style="bold red"),
                title="Error",
                border_style="red",
            )
        )
        sys.exit(1)


@cli.command()
@click.argument(
    "provider",
    type=click.Choice(
        ["openai", "anthropic", "bedrock", "ollama", "gemini"], case_sensitive=False
    ),
)
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True),
    help="Path to configuration file",
)
def test_provider(provider: str, config_file: Optional[str]):
    """Test a specific LLM provider configuration.

    Verifies that the provider is properly configured and can successfully
    process queries. This helps troubleshoot configuration issues.

    Examples:
        aws-cost-cli test-provider openai
        aws-cost-cli test-provider gemini --config-file custom_config.yaml
        aws-cost-cli test-provider ollama
    """
    try:
        provider = provider.lower()

        # Load configuration
        config_manager = ConfigManager()
        try:
            config = config_manager.load_config(config_file)
        except FileNotFoundError:
            config = Config()

        console.print(f"🧪 Testing {provider.title()} provider configuration...")

        # Test provider creation
        try:
            provider_instance = ProviderFactory.create_provider(
                provider, config.llm_config
            )
            console.print("✅ Provider instance created successfully")
        except Exception as e:
            console.print(
                Panel(
                    Text(f"❌ Failed to create provider: {str(e)}", style="bold red"),
                    title="Configuration Error",
                    border_style="red",
                )
            )

            # Show configuration help
            console.print("\n💡 Configuration help:")
            if provider == "openai":
                console.print("   • Set OPENAI_API_KEY environment variable")
                console.print(
                    "   • Or run: aws-cost-cli configure --provider openai --api-key <your-key>"
                )
            elif provider == "anthropic":
                console.print("   • Set ANTHROPIC_API_KEY environment variable")
                console.print(
                    "   • Or run: aws-cost-cli configure --provider anthropic --api-key <your-key>"
                )
            elif provider == "gemini":
                console.print("   • Set GEMINI_API_KEY environment variable")
                console.print(
                    "   • Or run: aws-cost-cli configure --provider gemini --api-key <your-key>"
                )
            elif provider == "bedrock":
                console.print("   • Ensure AWS credentials are configured")
                console.print("   • Run: aws configure (or set AWS_PROFILE)")
                console.print(
                    "   • Or run: aws-cost-cli configure --provider bedrock --region <region>"
                )
            elif provider == "ollama":
                console.print("   • Ensure Ollama server is running")
                console.print("   • Default URL: http://localhost:11434")
                console.print(
                    "   • Or run: aws-cost-cli configure --provider ollama --base-url <url>"
                )

            sys.exit(1)

        # Test provider availability
        try:
            is_available = provider_instance.is_available()
            if is_available:
                console.print("✅ Provider is available and ready")
            else:
                console.print("⚠️  Provider created but not available")
                sys.exit(1)
        except Exception as e:
            console.print(f"⚠️  Provider availability check failed: {str(e)}")

        # Test actual query parsing
        console.print("🔍 Testing query parsing...")
        test_query = "What did I spend on EC2 last month?"

        try:
            result = provider_instance.parse_query(test_query)
            console.print("✅ Query parsing successful")

            # Show parsed result details
            console.print("\n📊 Parsed query details:")
            if isinstance(result, dict):
                for key, value in result.items():
                    if key == "date_range" and isinstance(value, dict):
                        console.print(
                            f"   • {key}: {value.get('start', 'N/A')} to {value.get('end', 'N/A')}"
                        )
                    else:
                        console.print(f"   • {key}: {value}")
            else:
                console.print(f"   • Result: {result}")

        except Exception as e:
            console.print(f"❌ Query parsing failed: {str(e)}")

            # Provider-specific troubleshooting
            if provider == "ollama":
                console.print("\n🔧 Ollama troubleshooting:")
                console.print(
                    "   • Check if Ollama server is running: curl http://localhost:11434"
                )
                console.print("   • Check if model is available: ollama list")
                console.print("   • Pull model if needed: ollama pull gpt-oss:20b")
            elif provider in ["openai", "anthropic", "gemini"]:
                console.print(f"\n🔧 {provider.title()} troubleshooting:")
                console.print(
                    "   • Verify API key is correct and has sufficient credits"
                )
                console.print("   • Check network connectivity")
                console.print("   • Verify model name is correct")
            elif provider == "bedrock":
                console.print("\n🔧 Bedrock troubleshooting:")
                console.print("   • Verify AWS credentials have Bedrock permissions")
                console.print("   • Check if model is available in your region")
                console.print("   • Verify region configuration")

            sys.exit(1)

        # Show provider configuration details
        console.print(f"\n⚙️  {provider.title()} configuration:")
        if provider in config.llm_config:
            provider_config = config.llm_config[provider]
            for key, value in provider_config.items():
                if "api_key" in key.lower():
                    # Mask API keys for security
                    masked_value = f"{value[:8]}..." if len(value) > 8 else "***"
                    console.print(f"   • {key}: {masked_value}")
                else:
                    console.print(f"   • {key}: {value}")
        else:
            console.print("   • Using default configuration")

        console.print(f"\n🎉 {provider.title()} provider is working correctly!")
        console.print(
            f'💡 You can now use: aws-cost-cli query "your question" --llm-provider {provider}'
        )

    except Exception as e:
        console.print(
            Panel(
                Text(f"❌ Provider test failed: {str(e)}", style="bold red"),
                title="Test Error",
                border_style="red",
            )
        )
        sys.exit(1)


@cli.command()
@click.option(
    "--hours",
    "-h",
    type=int,
    default=24,
    help="Number of hours to include in performance summary (default: 24)",
)
@click.option(
    "--provider",
    "-p",
    type=click.Choice(
        ["openai", "anthropic", "bedrock", "ollama", "gemini"], case_sensitive=False
    ),
    help="Show performance for specific provider only",
)
def provider_performance(hours: int, provider: Optional[str]):
    """Show LLM provider performance metrics and statistics.

    Displays performance metrics including response times, success rates,
    error rates, and health status for all configured LLM providers.

    Examples:
        aws-cost-cli provider-performance
        aws-cost-cli provider-performance --hours 48
        aws-cost-cli provider-performance --provider openai
    """
    try:
        from .query_processor import get_performance_monitor

        monitor = get_performance_monitor()
        summary = monitor.get_performance_summary(hours)

        console.print(f"📊 LLM Provider Performance Summary (Last {hours} hours)")
        console.print()

        if not summary["providers"]:
            console.print("ℹ️  No provider performance data available yet.")
            console.print("💡 Run some queries to generate performance metrics.")
            return

        # Overall statistics
        overall = summary["overall"]
        if overall["total_requests"] > 0:
            console.print("🌐 Overall Statistics:")
            console.print(f"   • Total Requests: {overall['total_requests']}")
            console.print(f"   • Success Rate: {overall['average_success_rate']:.1f}%")
            console.print(f"   • Healthy Providers: {overall['healthy_providers']}")
            console.print(f"   • Degraded Providers: {overall['degraded_providers']}")
            console.print(f"   • Unhealthy Providers: {overall['unhealthy_providers']}")
            console.print()

        # Provider-specific metrics
        providers_to_show = (
            [provider.lower()] if provider else summary["providers"].keys()
        )

        for provider_name in providers_to_show:
            if provider_name not in summary["providers"]:
                console.print(f"⚠️  No performance data for provider: {provider_name}")
                continue

            metrics = summary["providers"][provider_name]

            # Health status emoji
            health_emoji = {
                "healthy": "🟢",
                "degraded": "🟡",
                "unhealthy": "🔴",
                "unknown": "⚪",
            }.get(metrics["health_status"], "⚪")

            console.print(f"{health_emoji} {provider_name.title()} Provider:")
            console.print(f"   • Status: {metrics['health_status'].title()}")
            console.print(f"   • Requests: {metrics['request_count']}")
            console.print(f"   • Success Rate: {metrics['success_rate']:.1f}%")
            console.print(f"   • Error Rate: {metrics['error_rate']:.1f}%")

            if metrics["average_response_time_ms"] > 0:
                console.print(
                    f"   • Avg Response Time: {metrics['average_response_time_ms']:.0f}ms"
                )
                if metrics["min_response_time_ms"] and metrics["max_response_time_ms"]:
                    console.print(
                        f"   • Response Time Range: {metrics['min_response_time_ms']:.0f}ms - {metrics['max_response_time_ms']:.0f}ms"
                    )

            if metrics["consecutive_errors"] > 0:
                console.print(
                    f"   • Consecutive Errors: {metrics['consecutive_errors']}"
                )

            if metrics["timeout_count"] > 0:
                console.print(f"   • Timeouts: {metrics['timeout_count']}")

            if metrics["last_success"]:
                last_success = datetime.fromisoformat(metrics["last_success"])
                console.print(
                    f"   • Last Success: {last_success.strftime('%Y-%m-%d %H:%M:%S')}"
                )

            if metrics["last_error"]:
                last_error = datetime.fromisoformat(metrics["last_error"])
                console.print(
                    f"   • Last Error: {last_error.strftime('%Y-%m-%d %H:%M:%S')}"
                )
                if metrics["last_error_message"]:
                    console.print(
                        f"   • Last Error Message: {metrics['last_error_message'][:100]}..."
                    )

            console.print()

    except Exception as e:
        console.print(
            Panel(
                Text(
                    f"❌ Failed to get performance metrics: {str(e)}", style="bold red"
                ),
                title="Performance Error",
                border_style="red",
            )
        )
        sys.exit(1)


@cli.command()
@click.option(
    "--provider",
    "-p",
    type=click.Choice(
        ["openai", "anthropic", "bedrock", "ollama", "gemini"], case_sensitive=False
    ),
    help="Check health of specific provider only",
)
@click.option(
    "--timeout",
    "-t",
    type=float,
    default=10.0,
    help="Timeout for health checks in seconds (default: 10)",
)
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True),
    help="Path to configuration file",
)
def provider_health(
    provider: Optional[str], timeout: float, config_file: Optional[str]
):
    """Check health status of LLM providers.

    Performs health checks on configured LLM providers to verify they are
    responding correctly and measure response times.

    Examples:
        aws-cost-cli provider-health
        aws-cost-cli provider-health --provider openai
        aws-cost-cli provider-health --timeout 5
    """
    try:
        # Load configuration
        config_manager = ConfigManager()
        try:
            config = config_manager.load_config(config_file)
        except FileNotFoundError:
            config = Config()

        console.print("🏥 Checking LLM Provider Health...")
        console.print()

        # Get available providers
        available_providers = ProviderFactory.get_available_providers(config.llm_config)

        if not available_providers:
            console.print("⚠️  No providers are configured and available.")
            console.print(
                "💡 Run 'aws-cost-cli list-providers' to see configuration status."
            )
            return

        providers_to_check = [provider.lower()] if provider else available_providers

        health_results = []

        for provider_name in providers_to_check:
            if provider_name not in available_providers:
                console.print(
                    f"⚠️  Provider {provider_name} is not available or configured"
                )
                continue

            console.print(f"🔍 Checking {provider_name.title()}...")

            try:
                # Create provider instance
                provider_instance = ProviderFactory.create_provider(
                    provider_name, config.llm_config
                )

                # Perform health check
                health_check = provider_instance.check_health()
                health_results.append(health_check)

                # Display result
                if health_check.is_healthy:
                    status_emoji = "🟢"
                    status_text = "Healthy"
                else:
                    status_emoji = "🔴"
                    status_text = "Unhealthy"

                console.print(f"   {status_emoji} Status: {status_text}")

                if health_check.response_time_ms:
                    console.print(
                        f"   ⏱️  Response Time: {health_check.response_time_ms:.0f}ms"
                    )

                if health_check.error_message:
                    console.print(f"   ❌ Error: {health_check.error_message}")

                console.print(
                    f"   📅 Checked: {health_check.checked_at.strftime('%Y-%m-%d %H:%M:%S')}"
                )
                console.print()

            except Exception as e:
                console.print(f"   ❌ Health check failed: {str(e)}")
                console.print()

        # Summary
        if health_results:
            healthy_count = sum(1 for result in health_results if result.is_healthy)
            total_count = len(health_results)

            console.print("📋 Health Check Summary:")
            console.print(f"   • Healthy Providers: {healthy_count}/{total_count}")

            if healthy_count == total_count:
                console.print("   🎉 All providers are healthy!")
            elif healthy_count == 0:
                console.print("   ⚠️  No providers are healthy")
            else:
                console.print("   ⚠️  Some providers have issues")

    except Exception as e:
        console.print(
            Panel(
                Text(f"❌ Health check failed: {str(e)}", style="bold red"),
                title="Health Check Error",
                border_style="red",
            )
        )
        sys.exit(1)


@cli.command()
@click.option(
    "--provider",
    "-p",
    type=click.Choice(
        ["openai", "anthropic", "bedrock", "ollama", "gemini"], case_sensitive=False
    ),
    help="Reset metrics for specific provider only",
)
@click.confirmation_option(
    prompt="Are you sure you want to reset provider performance metrics?"
)
def reset_provider_metrics(provider: Optional[str]):
    """Reset LLM provider performance metrics.

    Clears all stored performance metrics and statistics for providers.
    This is useful for starting fresh after configuration changes.

    Examples:
        aws-cost-cli reset-provider-metrics
        aws-cost-cli reset-provider-metrics --provider openai
    """
    try:
        from .query_processor import get_performance_monitor

        monitor = get_performance_monitor()

        if provider:
            monitor.reset_metrics(provider.lower())
            console.print(
                f"✅ Reset performance metrics for {provider.title()} provider"
            )
        else:
            monitor.reset_metrics()
            console.print("✅ Reset performance metrics for all providers")

        console.print("💡 New metrics will be collected as you use the providers.")

    except Exception as e:
        console.print(
            Panel(
                Text(f"❌ Failed to reset metrics: {str(e)}", style="bold red"),
                title="Reset Error",
                border_style="red",
            )
        )
        sys.exit(1)


def main():
    """Entry point for the CLI application."""
    cli()


if __name__ == "__main__":
    main()
