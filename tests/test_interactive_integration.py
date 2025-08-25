"""Integration tests for interactive query builder with the main pipeline."""

import tempfile
from unittest.mock import Mock, patch, MagicMock

import pytest

from src.aws_cost_cli.interactive_query_builder import InteractiveQueryBuilder, QueryHistoryManager
from src.aws_cost_cli.query_pipeline import QueryPipeline, QueryContext
from src.aws_cost_cli.models import Config


class TestInteractiveIntegration:
    """Test integration between interactive query builder and main pipeline."""
    
    @patch('src.aws_cost_cli.query_pipeline.QueryParser')
    @patch('src.aws_cost_cli.query_pipeline.AWSCostClient')
    @patch('src.aws_cost_cli.query_pipeline.ResponseGenerator')
    def test_interactive_with_pipeline(self, mock_response_gen, mock_aws_client, mock_query_parser):
        """Test interactive query builder integration with pipeline."""
        # Mock pipeline components
        mock_query_parser.return_value.parse_query.return_value = {
            'service': 'Amazon Elastic Compute Cloud - Compute',
            'start_date': '2025-07-01',
            'end_date': '2025-08-01',
            'granularity': 'MONTHLY',
            'metrics': ['BlendedCost'],
            'group_by': None
        }
        
        # Create interactive builder with mocked pipeline
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = f"{temp_dir}/test_config.yaml"
            
            # Create a test config
            config = Config(
                llm_provider="openai",
                llm_config={"api_key": "test-key", "model": "gpt-3.5-turbo"},
                output_format="simple"
            )
            
            # Initialize builder
            builder = InteractiveQueryBuilder(config_path=config_path)
            
            # Test query suggestions
            suggestions = builder.get_query_suggestions("EC2")
            assert len(suggestions) > 0
            assert any("EC2" in suggestion for suggestion in suggestions)
            
            # Test query validation
            is_valid, warnings, suggestions = builder.validator.validate_query("What did I spend on EC2 last month?")
            assert is_valid is True
            assert len(warnings) == 0  # Should be a good query
    
    def test_history_integration(self):
        """Test history integration with query execution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            history_file = f"{temp_dir}/test_history.json"
            history_manager = QueryHistoryManager(history_file=history_file)
            
            # Add some test history
            history_manager.add_to_history("EC2 costs last month", True, 150.0)
            history_manager.add_to_history("S3 costs this year", True, 200.0)
            history_manager.add_to_history("Invalid query", False, error_message="Parse error")
            
            # Test getting successful queries
            successful = history_manager.get_successful_queries(5)
            assert len(successful) == 2
            assert all(entry.success for entry in successful)
            
            # Test getting recent queries
            recent = history_manager.get_recent_queries(5)
            assert len(recent) == 3
            assert recent[0].query == "Invalid query"  # Most recent first
            assert recent[0].success is False
    
    def test_template_parameter_substitution(self):
        """Test template parameter substitution."""
        builder = InteractiveQueryBuilder()
        
        # Get a template with parameters
        template = builder.template_manager.get_template_by_name("EC2 Monthly Costs")
        assert template is not None
        assert "{month}" in template.template
        assert "{year}" in template.template
        
        # Test parameter substitution
        query = template.template
        query = query.replace("{month}", "July")
        query = query.replace("{year}", "2025")
        
        expected = "How much did I spend on EC2 in July 2025?"
        assert query == expected
    
    def test_favorites_with_validation(self):
        """Test favorites with query validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            history_file = f"{temp_dir}/test_history.json"
            builder = InteractiveQueryBuilder()
            builder.history_manager = QueryHistoryManager(history_file=history_file)
            
            # Add a good favorite
            good_query = "What did I spend on EC2 last month?"
            builder.history_manager.add_favorite("Good Query", good_query, "A well-formed query")
            
            # Validate the favorite
            is_valid, warnings, suggestions = builder.validator.validate_query(good_query)
            assert is_valid is True
            assert len(warnings) == 0
            
            # Add a potentially problematic favorite
            ambiguous_query = "storage costs recently"
            builder.history_manager.add_favorite("Ambiguous Query", ambiguous_query, "Needs improvement")
            
            # Validate the ambiguous favorite
            is_valid, warnings, suggestions = builder.validator.validate_query(ambiguous_query)
            assert len(warnings) > 0  # Should have warnings
            assert len(suggestions) > 0  # Should have suggestions
    
    def test_suggestion_deduplication_across_sources(self):
        """Test that suggestions are deduplicated across different sources."""
        with tempfile.TemporaryDirectory() as temp_dir:
            history_file = f"{temp_dir}/test_history.json"
            builder = InteractiveQueryBuilder()
            builder.history_manager = QueryHistoryManager(history_file=history_file)
            
            # Add a query that might match a template
            duplicate_query = "What did I spend on EC2 last month?"
            builder.history_manager.add_to_history(duplicate_query, True)
            
            # Get suggestions
            suggestions = builder.get_query_suggestions("EC2")
            
            # Should not have duplicates
            assert len(suggestions) == len(set(suggestions))
            
            # Should contain the query from history
            assert any(duplicate_query in suggestions for _ in [True])
    
    def test_query_categories_coverage(self):
        """Test that all template categories are covered."""
        builder = InteractiveQueryBuilder()
        
        categories = builder.template_manager.get_categories()
        expected_categories = ["service", "time", "comparison", "analysis", "budget", "optimization"]
        
        for expected in expected_categories:
            assert expected in categories, f"Missing category: {expected}"
        
        # Test that each category has templates
        for category in categories:
            templates = builder.template_manager.get_templates_by_category(category)
            assert len(templates) > 0, f"No templates in category: {category}"
    
    def test_validation_with_common_patterns(self):
        """Test validation with common query patterns."""
        builder = InteractiveQueryBuilder()
        
        # Test various query patterns
        test_queries = [
            ("What did I spend on EC2 last month?", True, 0),  # Good query
            ("EC2", False, 1),  # Too short, missing context
            ("storage costs", False, 1),  # Ambiguous service, missing time
            ("How much did compute cost recently?", False, 1),  # Ambiguous service and time
            ("Show me my AWS bill for July 2025", True, 1),  # Good query but may have minor warnings
            ("Lambda costs this year broken down by service", True, 0),  # Good detailed query
        ]
        
        for query, expected_valid, expected_min_warnings in test_queries:
            is_valid, warnings, suggestions = builder.validator.validate_query(query)
            
            if expected_valid:
                assert is_valid, f"Query should be valid: {query}"
                assert len(warnings) <= expected_min_warnings, f"Too many warnings for: {query}"
            else:
                assert len(warnings) >= expected_min_warnings, f"Not enough warnings for: {query}"
    
    def test_template_coverage_for_requirements(self):
        """Test that templates cover the main requirements from the spec."""
        builder = InteractiveQueryBuilder()
        
        # Check for templates that cover key requirements
        template_names = [t.name for t in builder.template_manager.templates]
        
        # Should have service-specific templates (Requirement 1, 2)
        assert any("EC2" in name for name in template_names)
        assert any("S3" in name for name in template_names)
        assert any("RDS" in name for name in template_names)
        
        # Should have time-based templates (Requirement 3)
        assert any("Monthly" in name or "Quarterly" in name or "Year" in name for name in template_names)
        
        # Should have comparison templates (advanced features)
        assert any("Comparison" in name or "Trend" in name for name in template_names)
        
        # Should have analysis templates
        assert any("Breakdown" in name or "Analysis" in name for name in template_names)
    
    def test_error_handling_in_interactive_flow(self):
        """Test error handling in interactive components."""
        with tempfile.TemporaryDirectory() as temp_dir:
            history_file = f"{temp_dir}/test_history.json"
            builder = InteractiveQueryBuilder()
            builder.history_manager = QueryHistoryManager(history_file=history_file)
            
            # Test adding duplicate favorite
            builder.history_manager.add_favorite("Test", "Test query")
            
            with pytest.raises(Exception):  # Should raise ValidationError
                builder.history_manager.add_favorite("Test", "Another query")
            
            # Test removing non-existent favorite
            result = builder.history_manager.remove_favorite("Non-existent")
            assert result is False
            
            # Test getting non-existent favorite
            favorite = builder.history_manager.get_favorite_by_name("Non-existent")
            assert favorite is None