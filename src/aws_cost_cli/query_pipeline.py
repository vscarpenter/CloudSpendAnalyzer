"""End-to-end query processing pipeline for AWS Cost CLI."""

import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime, timezone

from .models import Config, QueryParameters, CostData
from .config import ConfigManager
from .query_processor import QueryParser
from .aws_client import AWSCostClient, CredentialManager
from .cache_manager import CacheManager
from .response_formatter import ResponseGenerator
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
)


@dataclass
class QueryContext:
    """Context information for query processing."""

    original_query: str
    profile: Optional[str] = None
    fresh_data: bool = False
    output_format: str = "simple"
    debug: bool = False
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    # Performance optimization options
    enable_parallel: bool = True
    enable_compression: bool = True
    max_chunk_days: int = 90
    show_performance_metrics: bool = False


@dataclass
class QueryResult:
    """Result of query processing."""

    success: bool
    cost_data: Optional[CostData] = None
    formatted_response: Optional[str] = None
    error: Optional[AWSCostCLIError] = None
    metadata: Optional[Dict[str, Any]] = None
    processing_time_ms: Optional[float] = None
    cache_hit: bool = False
    llm_used: bool = False
    # Performance metrics
    performance_metrics: Optional[Dict[str, Any]] = None
    api_calls_made: int = 0
    parallel_requests: int = 1
    compression_stats: Optional[Dict[str, Any]] = None
    fallback_used: bool = False


class QueryPipeline:
    """End-to-end query processing pipeline."""

    def __init__(
        self, config: Optional[Config] = None, config_path: Optional[str] = None
    ):
        """
        Initialize query pipeline.

        Args:
            config: Configuration object (if None, will load from file)
            config_path: Path to configuration file
        """
        self.logger = logging.getLogger(__name__)

        # Load configuration
        if config:
            self.config = config
        else:
            config_manager = ConfigManager()
            if config_path:
                self.config = config_manager.load_config(config_path)
            else:
                try:
                    self.config = config_manager.load_config()
                except FileNotFoundError:
                    # Use default config if no file found
                    self.config = Config()

        # Initialize components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all pipeline components."""
        # Initialize credential manager
        self.credential_manager = CredentialManager()

        # Initialize cache manager
        self.cache_manager = CacheManager(default_ttl=self.config.cache_ttl)

        # Initialize query parser
        self.query_parser = QueryParser(self.config.llm_config)

        # AWS client will be initialized per query with profile
        self.aws_client = None

        # Response generator will be initialized per query with LLM provider
        self.response_generator = None

    def process_query(self, context: QueryContext) -> QueryResult:
        """
        Process a complete query from natural language to formatted response.

        Args:
            context: Query context with all necessary information

        Returns:
            QueryResult with processing outcome
        """
        start_time = datetime.now(timezone.utc)
        context.start_time = start_time

        result = QueryResult(success=False)
        result.metadata = {
            "original_query": context.original_query,
            "profile": context.profile,
            "fresh_data": context.fresh_data,
            "output_format": context.output_format,
        }

        try:
            # Step 1: Validate AWS credentials
            self._validate_credentials(context, result)

            # Step 2: Initialize AWS client
            self._initialize_aws_client(context, result)

            # Step 3: Validate AWS permissions
            self._validate_permissions(context, result)

            # Step 4: Parse natural language query
            query_params = self._parse_query(context, result)

            # Step 5: Validate parsed parameters
            self._validate_parameters(query_params, context, result)

            # Step 6: Fetch cost data (with caching)
            cost_data = self._fetch_cost_data(query_params, context, result)

            # Step 7: Generate formatted response
            formatted_response = self._generate_response(
                cost_data, query_params, context, result
            )

            # Step 8: Finalize successful result
            result.success = True
            result.cost_data = cost_data
            result.formatted_response = formatted_response

        except AWSCostCLIError as e:
            result.error = e
            self.logger.error(f"Query processing failed: {e.message}")
        except Exception as e:
            result.error = AWSCostCLIError(f"Unexpected error: {str(e)}")
            self.logger.exception("Unexpected error during query processing")

        finally:
            # Calculate processing time
            end_time = datetime.now(timezone.utc)
            context.end_time = end_time
            result.processing_time_ms = (end_time - start_time).total_seconds() * 1000

            # Add timing metadata
            result.metadata["processing_time_ms"] = result.processing_time_ms
            result.metadata["start_time"] = start_time.isoformat()
            result.metadata["end_time"] = end_time.isoformat()

        return result

    def _validate_credentials(self, context: QueryContext, result: QueryResult):
        """Validate AWS credentials."""
        self.logger.debug(f"Validating AWS credentials for profile: {context.profile}")

        if not self.credential_manager.validate_credentials(context.profile):
            raise AWSCredentialsError(profile=context.profile)

        result.metadata["credentials_valid"] = True

    def _initialize_aws_client(self, context: QueryContext, result: QueryResult):
        """Initialize AWS client with cache manager."""
        self.logger.debug("Initializing AWS client")

        self.aws_client = AWSCostClient(
            profile=context.profile, cache_manager=self.cache_manager
        )

        result.metadata["aws_client_initialized"] = True

    def _validate_permissions(self, context: QueryContext, result: QueryResult):
        """Validate AWS permissions."""
        self.logger.debug("Validating AWS permissions")

        if not self.aws_client.validate_permissions():
            raise AWSPermissionsError()

        result.metadata["permissions_valid"] = True

    def _parse_query(
        self, context: QueryContext, result: QueryResult
    ) -> QueryParameters:
        """Parse natural language query."""
        self.logger.debug(f"Parsing query: {context.original_query}")

        try:
            query_params = self.query_parser.parse_query(context.original_query)
            result.llm_used = True
            result.metadata["parsing_method"] = "llm"
            return query_params

        except LLMProviderError as e:
            self.logger.warning(f"LLM parsing failed, using fallback: {e.message}")

            # Try fallback parser
            try:
                from .query_processor import FallbackParser

                fallback = FallbackParser()
                parsed_result = fallback.parse_query(context.original_query)
                query_params = self.query_parser._convert_to_query_parameters(
                    parsed_result
                )

                result.fallback_used = True
                result.metadata["parsing_method"] = "fallback"
                result.metadata["llm_error"] = e.message

                return query_params

            except Exception as fallback_error:
                raise QueryParsingError(
                    f"Both LLM and fallback parsing failed. LLM error: {e.message}, "
                    f"Fallback error: {str(fallback_error)}",
                    original_query=context.original_query,
                )

        except (QueryParsingError, ValidationError) as e:
            raise e

        except Exception as e:
            raise QueryParsingError(
                f"Query parsing failed: {str(e)}", original_query=context.original_query
            )

    def _validate_parameters(
        self, query_params: QueryParameters, context: QueryContext, result: QueryResult
    ):
        """Validate parsed query parameters."""
        self.logger.debug("Validating query parameters")

        try:
            self.query_parser.validate_parameters(query_params)
            result.metadata["parameters_valid"] = True
        except ValidationError as e:
            raise e

    def _fetch_cost_data(
        self, query_params: QueryParameters, context: QueryContext, result: QueryResult
    ) -> CostData:
        """Fetch cost data from AWS with caching and performance optimizations."""
        self.logger.debug("Fetching cost data with performance optimizations")

        # Check if we should use cache
        use_cache = not context.fresh_data

        # Use performance optimizations if enabled
        if context.enable_parallel or context.enable_compression:
            try:
                # Use performance-optimized client
                cost_data = (
                    self.aws_client.get_cost_and_usage_with_performance_optimization(
                        params=query_params,
                        use_cache=use_cache,
                        enable_parallel=context.enable_parallel,
                        enable_compression=context.enable_compression,
                        max_chunk_days=context.max_chunk_days,
                    )
                )

                # Get performance metrics if available
                perf_client = self.aws_client.create_performance_optimized_client(
                    enable_parallel=context.enable_parallel,
                    enable_compression=context.enable_compression,
                    enable_monitoring=True,
                )

                if hasattr(perf_client, "monitor"):
                    perf_summary = perf_client.get_performance_summary(hours=1)
                    result.performance_metrics = perf_summary

                if hasattr(perf_client, "compressed_cache"):
                    compression_stats = (
                        perf_client.compressed_cache.get_compression_stats()
                    )
                    result.compression_stats = compression_stats

                result.metadata["performance_optimizations_used"] = True
                result.metadata["parallel_enabled"] = context.enable_parallel
                result.metadata["compression_enabled"] = context.enable_compression

                self.logger.debug("Used performance-optimized data fetching")
                return cost_data

            except Exception as e:
                self.logger.warning(
                    f"Performance optimization failed, falling back to standard method: {e}"
                )
                # Fall back to standard method

        # Standard data fetching (fallback or when optimizations disabled)
        # Check cache first if enabled
        if use_cache:
            try:
                cache_key = self.cache_manager.generate_cache_key(
                    query_params, context.profile or "default"
                )
                cached_data = self.cache_manager.get_cached_data(cache_key)
                if cached_data:
                    result.cache_hit = True
                    result.metadata["data_source"] = "cache"
                    self.logger.debug("Using cached data")
                    return cached_data
            except CacheError as e:
                self.logger.warning(f"Cache error: {e.message}")

        # Fetch from AWS (with advanced features if requested)
        try:
            # Check if advanced features are requested
            if query_params.trend_analysis or query_params.include_forecast:
                cost_data = self.aws_client.get_advanced_cost_data(query_params)
                result.metadata["advanced_features_used"] = True
            else:
                cost_data = self.aws_client.get_cost_and_usage(
                    query_params, use_cache=use_cache
                )
                result.api_calls_made = 1

            result.metadata["data_source"] = "aws_api"
            self.logger.debug("Fetched data from AWS API")
            return cost_data

        except (
            AWSCredentialsError,
            AWSPermissionsError,
            AWSAPIError,
            NetworkError,
        ) as e:
            raise e
        except Exception as e:
            raise AWSAPIError(f"Failed to fetch cost data: {str(e)}")

    def _generate_response(
        self,
        cost_data: CostData,
        query_params: QueryParameters,
        context: QueryContext,
        result: QueryResult,
    ) -> str:
        """Generate formatted response."""
        self.logger.debug(f"Generating response in format: {context.output_format}")

        # Initialize response generator if needed
        if not self.response_generator:
            llm_provider = None

            # Initialize LLM provider for response generation if needed
            if context.output_format == "llm" and self.config.llm_config.get(
                "provider"
            ):
                try:
                    provider_type = self.config.llm_config["provider"].lower()

                    if provider_type == "openai":
                        from .query_processor import OpenAIProvider

                        llm_provider = OpenAIProvider(
                            self.config.llm_config.get("api_key", ""),
                            self.config.llm_config.get("model", "gpt-3.5-turbo"),
                        )
                    elif provider_type == "anthropic":
                        from .query_processor import AnthropicProvider

                        llm_provider = AnthropicProvider(
                            self.config.llm_config.get("api_key", ""),
                            self.config.llm_config.get(
                                "model", "claude-3-haiku-20240307"
                            ),
                        )
                    elif provider_type == "ollama":
                        from .query_processor import OllamaProvider

                        llm_provider = OllamaProvider(
                            self.config.llm_config.get("model", "llama2"),
                            self.config.llm_config.get(
                                "base_url", "http://localhost:11434"
                            ),
                        )

                except Exception as e:
                    self.logger.warning(f"Failed to initialize LLM provider: {e}")

            self.response_generator = ResponseGenerator(
                llm_provider=llm_provider, output_format=context.output_format
            )

        try:
            formatted_response = self.response_generator.format_response(
                cost_data, context.original_query, query_params
            )

            result.metadata["response_generated"] = True
            return formatted_response

        except Exception as e:
            raise AWSCostCLIError(f"Failed to generate response: {str(e)}")

    def handle_ambiguous_query(self, context: QueryContext) -> List[str]:
        """
        Handle ambiguous queries by providing clarification suggestions.

        Args:
            context: Query context

        Returns:
            List of clarification suggestions
        """
        suggestions = []

        # Analyze the query for common ambiguities
        query_lower = context.original_query.lower()

        # Time period ambiguities
        if any(word in query_lower for word in ["last", "this", "current", "recent"]):
            if (
                "month" not in query_lower
                and "year" not in query_lower
                and "week" not in query_lower
            ):
                suggestions.append(
                    "Specify a time period (e.g., 'last month', 'this year', 'last week')"
                )

        # Service ambiguities
        if any(word in query_lower for word in ["compute", "storage", "database"]):
            suggestions.extend(
                [
                    "For compute costs, try: 'EC2 costs'",
                    "For storage costs, try: 'S3 costs'",
                    "For database costs, try: 'RDS costs'",
                ]
            )

        # Cost type ambiguities
        if "cost" in query_lower and not any(
            word in query_lower for word in ["total", "breakdown", "by service"]
        ):
            suggestions.extend(
                [
                    "For total costs: 'total costs for [time period]'",
                    "For cost breakdown: 'costs by service for [time period]'",
                ]
            )

        # Generic suggestions if no specific ambiguities found
        if not suggestions:
            suggestions.extend(
                [
                    "Try being more specific about the time period",
                    "Specify which AWS service you're interested in",
                    "Include whether you want total costs or a breakdown",
                ]
            )

        return suggestions

    def get_query_suggestions(self, partial_query: str) -> List[str]:
        """
        Get query suggestions based on partial input.

        Args:
            partial_query: Partial query string

        Returns:
            List of query suggestions
        """
        suggestions = []
        partial_lower = partial_query.lower()

        # Common query patterns
        common_patterns = [
            "How much did I spend on EC2 last month?",
            "What are my total AWS costs this year?",
            "Show me S3 costs for the last 3 months",
            "What did I spend on RDS yesterday?",
            "Break down my costs by service for this month",
            "How much am I spending on Lambda daily?",
            "What are my CloudFront costs this quarter?",
            "Show me VPC costs for last week",
        ]

        # Filter suggestions based on partial input
        if partial_query:
            for pattern in common_patterns:
                if any(word in pattern.lower() for word in partial_lower.split()):
                    suggestions.append(pattern)
        else:
            suggestions = common_patterns

        return suggestions[:5]  # Return top 5 suggestions

    def validate_query_complexity(
        self, query_params: QueryParameters
    ) -> Dict[str, Any]:
        """
        Validate and assess query complexity.

        Args:
            query_params: Parsed query parameters

        Returns:
            Dictionary with complexity assessment
        """
        if self.aws_client:
            return self.aws_client.estimate_query_cost(query_params)
        else:
            return {
                "complexity_level": "unknown",
                "warnings": ["AWS client not initialized"],
            }

    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get current pipeline status and health.

        Returns:
            Dictionary with pipeline status
        """
        status = {
            "config_loaded": bool(self.config),
            "cache_manager_initialized": bool(self.cache_manager),
            "query_parser_initialized": bool(self.query_parser),
            "aws_client_initialized": bool(self.aws_client),
            "response_generator_initialized": bool(self.response_generator),
        }

        # Check component health
        if self.cache_manager:
            try:
                cache_stats = self.cache_manager.get_cache_stats()
                status["cache_healthy"] = True
                status["cache_entries"] = cache_stats.get("total_entries", 0)
            except Exception:
                status["cache_healthy"] = False

        if self.aws_client:
            try:
                service_status = self.aws_client.check_service_availability()
                status["aws_service_healthy"] = service_status.get("available", False)
                status["aws_response_time_ms"] = service_status.get("response_time_ms")
            except Exception:
                status["aws_service_healthy"] = False

        return status

    def cleanup(self):
        """Clean up pipeline resources."""
        self.logger.debug("Cleaning up pipeline resources")

        # Clean up any resources if needed
        if self.cache_manager:
            try:
                self.cache_manager.cleanup_expired_cache()
            except Exception as e:
                self.logger.warning(f"Failed to cleanup expired cache: {e}")
