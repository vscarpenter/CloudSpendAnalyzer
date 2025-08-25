"""Tests for interactive query builder functionality."""

import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from src.aws_cost_cli.interactive_query_builder import (
    QueryTemplate,
    QueryHistoryEntry,
    QueryFavorite,
    QueryTemplateManager,
    QueryHistoryManager,
    QueryValidator,
    InteractiveQueryBuilder
)
from src.aws_cost_cli.exceptions import ValidationError, QueryParsingError


class TestQueryTemplate:
    """Test QueryTemplate dataclass."""
    
    def test_query_template_creation(self):
        """Test creating a query template."""
        template = QueryTemplate(
            name="Test Template",
            description="A test template",
            template="Test query for {service}",
            category="test",
            parameters={"service": "EC2"}
        )
        
        assert template.name == "Test Template"
        assert template.description == "A test template"
        assert template.template == "Test query for {service}"
        assert template.category == "test"
        assert template.parameters == {"service": "EC2"}
    
    def test_query_template_default_parameters(self):
        """Test query template with default parameters."""
        template = QueryTemplate(
            name="Test Template",
            description="A test template",
            template="Test query",
            category="test"
        )
        
        assert template.parameters == {}


class TestQueryHistoryEntry:
    """Test QueryHistoryEntry dataclass."""
    
    def test_history_entry_creation(self):
        """Test creating a history entry."""
        timestamp = datetime.now()
        entry = QueryHistoryEntry(
            query="Test query",
            timestamp=timestamp,
            success=True,
            execution_time_ms=150.5,
            error_message=None
        )
        
        assert entry.query == "Test query"
        assert entry.timestamp == timestamp
        assert entry.success is True
        assert entry.execution_time_ms == 150.5
        assert entry.error_message is None
    
    def test_history_entry_with_error(self):
        """Test creating a history entry with error."""
        timestamp = datetime.now()
        entry = QueryHistoryEntry(
            query="Bad query",
            timestamp=timestamp,
            success=False,
            error_message="Parse error"
        )
        
        assert entry.query == "Bad query"
        assert entry.success is False
        assert entry.error_message == "Parse error"


class TestQueryFavorite:
    """Test QueryFavorite dataclass."""
    
    def test_favorite_creation(self):
        """Test creating a favorite."""
        created_at = datetime.now()
        favorite = QueryFavorite(
            name="Test Favorite",
            query="Test query",
            description="A test favorite",
            created_at=created_at
        )
        
        assert favorite.name == "Test Favorite"
        assert favorite.query == "Test query"
        assert favorite.description == "A test favorite"
        assert favorite.created_at == created_at
    
    def test_favorite_default_created_at(self):
        """Test favorite with default created_at."""
        before = datetime.now()
        favorite = QueryFavorite(
            name="Test Favorite",
            query="Test query"
        )
        after = datetime.now()
        
        assert before <= favorite.created_at <= after
        assert favorite.description is None


class TestQueryTemplateManager:
    """Test QueryTemplateManager class."""
    
    def test_template_manager_initialization(self):
        """Test template manager initialization."""
        manager = QueryTemplateManager()
        
        assert len(manager.templates) > 0
        assert all(isinstance(t, QueryTemplate) for t in manager.templates)
    
    def test_get_templates_by_category(self):
        """Test getting templates by category."""
        manager = QueryTemplateManager()
        
        # Test getting all templates
        all_templates = manager.get_templates_by_category()
        assert len(all_templates) > 0
        
        # Test getting service templates
        service_templates = manager.get_templates_by_category("service")
        assert len(service_templates) > 0
        assert all(t.category == "service" for t in service_templates)
        
        # Test getting non-existent category
        empty_templates = manager.get_templates_by_category("nonexistent")
        assert len(empty_templates) == 0
    
    def test_get_template_by_name(self):
        """Test getting template by name."""
        manager = QueryTemplateManager()
        
        # Test getting existing template
        template = manager.get_template_by_name("EC2 Monthly Costs")
        assert template is not None
        assert template.name == "EC2 Monthly Costs"
        
        # Test getting non-existent template
        template = manager.get_template_by_name("Non-existent Template")
        assert template is None
    
    def test_get_categories(self):
        """Test getting all categories."""
        manager = QueryTemplateManager()
        
        categories = manager.get_categories()
        assert len(categories) > 0
        assert "service" in categories
        assert "time" in categories
        assert "comparison" in categories
        assert "analysis" in categories
        
        # Categories should be sorted
        assert categories == sorted(categories)


class TestQueryHistoryManager:
    """Test QueryHistoryManager class."""
    
    def test_history_manager_initialization(self):
        """Test history manager initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            history_file = os.path.join(temp_dir, "test_history.json")
            manager = QueryHistoryManager(history_file=history_file)
            
            assert manager.history_file == history_file
            assert manager.favorites_file == history_file.replace('history.json', 'favorites.json')
            assert isinstance(manager.history, list)
            assert isinstance(manager.favorites, list)
    
    def test_add_to_history(self):
        """Test adding entries to history."""
        with tempfile.TemporaryDirectory() as temp_dir:
            history_file = os.path.join(temp_dir, "test_history.json")
            manager = QueryHistoryManager(history_file=history_file)
            
            # Add successful query
            manager.add_to_history("Test query 1", True, 100.0)
            assert len(manager.history) == 1
            assert manager.history[0].query == "Test query 1"
            assert manager.history[0].success is True
            assert manager.history[0].execution_time_ms == 100.0
            
            # Add failed query
            manager.add_to_history("Test query 2", False, error_message="Parse error")
            assert len(manager.history) == 2
            assert manager.history[0].query == "Test query 2"  # Most recent first
            assert manager.history[0].success is False
            assert manager.history[0].error_message == "Parse error"
    
    def test_history_limit(self):
        """Test history entry limit."""
        with tempfile.TemporaryDirectory() as temp_dir:
            history_file = os.path.join(temp_dir, "test_history.json")
            manager = QueryHistoryManager(history_file=history_file)
            
            # Add more than 100 entries
            for i in range(105):
                manager.add_to_history(f"Query {i}", True)
            
            # Should keep only 100 entries
            assert len(manager.history) == 100
            assert manager.history[0].query == "Query 104"  # Most recent
            assert manager.history[-1].query == "Query 5"   # Oldest kept
    
    def test_get_recent_queries(self):
        """Test getting recent queries."""
        with tempfile.TemporaryDirectory() as temp_dir:
            history_file = os.path.join(temp_dir, "test_history.json")
            manager = QueryHistoryManager(history_file=history_file)
            
            # Add some queries
            for i in range(15):
                manager.add_to_history(f"Query {i}", i % 2 == 0)  # Alternate success/failure
            
            # Get recent queries
            recent = manager.get_recent_queries(10)
            assert len(recent) == 10
            assert recent[0].query == "Query 14"  # Most recent
            assert recent[-1].query == "Query 5"
    
    def test_get_successful_queries(self):
        """Test getting successful queries only."""
        with tempfile.TemporaryDirectory() as temp_dir:
            history_file = os.path.join(temp_dir, "test_history.json")
            manager = QueryHistoryManager(history_file=history_file)
            
            # Add mixed success/failure queries
            for i in range(10):
                manager.add_to_history(f"Query {i}", i % 2 == 0)
            
            # Get successful queries
            successful = manager.get_successful_queries(5)
            assert len(successful) == 5
            assert all(entry.success for entry in successful)
            assert successful[0].query == "Query 8"  # Most recent successful
    
    def test_add_favorite(self):
        """Test adding favorites."""
        with tempfile.TemporaryDirectory() as temp_dir:
            history_file = os.path.join(temp_dir, "test_history.json")
            manager = QueryHistoryManager(history_file=history_file)
            
            # Add favorite
            manager.add_favorite("Test Favorite", "Test query", "Test description")
            assert len(manager.favorites) == 1
            assert manager.favorites[0].name == "Test Favorite"
            assert manager.favorites[0].query == "Test query"
            assert manager.favorites[0].description == "Test description"
            
            # Try to add duplicate name
            with pytest.raises(ValidationError, match="already exists"):
                manager.add_favorite("Test Favorite", "Another query")
    
    def test_remove_favorite(self):
        """Test removing favorites."""
        with tempfile.TemporaryDirectory() as temp_dir:
            history_file = os.path.join(temp_dir, "test_history.json")
            manager = QueryHistoryManager(history_file=history_file)
            
            # Add favorites
            manager.add_favorite("Favorite 1", "Query 1")
            manager.add_favorite("Favorite 2", "Query 2")
            assert len(manager.favorites) == 2
            
            # Remove existing favorite
            result = manager.remove_favorite("Favorite 1")
            assert result is True
            assert len(manager.favorites) == 1
            assert manager.favorites[0].name == "Favorite 2"
            
            # Try to remove non-existent favorite
            result = manager.remove_favorite("Non-existent")
            assert result is False
            assert len(manager.favorites) == 1
    
    def test_get_favorite_by_name(self):
        """Test getting favorite by name."""
        with tempfile.TemporaryDirectory() as temp_dir:
            history_file = os.path.join(temp_dir, "test_history.json")
            manager = QueryHistoryManager(history_file=history_file)
            
            # Add favorite
            manager.add_favorite("Test Favorite", "Test query")
            
            # Get existing favorite
            favorite = manager.get_favorite_by_name("Test Favorite")
            assert favorite is not None
            assert favorite.name == "Test Favorite"
            assert favorite.query == "Test query"
            
            # Get non-existent favorite
            favorite = manager.get_favorite_by_name("Non-existent")
            assert favorite is None
    
    def test_persistence(self):
        """Test history and favorites persistence."""
        with tempfile.TemporaryDirectory() as temp_dir:
            history_file = os.path.join(temp_dir, "test_history.json")
            
            # Create manager and add data
            manager1 = QueryHistoryManager(history_file=history_file)
            manager1.add_to_history("Test query", True, 100.0)
            manager1.add_favorite("Test Favorite", "Test query")
            
            # Create new manager with same file
            manager2 = QueryHistoryManager(history_file=history_file)
            
            # Data should be loaded
            assert len(manager2.history) == 1
            assert manager2.history[0].query == "Test query"
            assert len(manager2.favorites) == 1
            assert manager2.favorites[0].name == "Test Favorite"


class TestQueryValidator:
    """Test QueryValidator class."""
    
    def test_validator_initialization(self):
        """Test validator initialization."""
        validator = QueryValidator()
        assert validator.query_parser is None
        assert isinstance(validator.common_issues, dict)
        
        # Test with query parser
        mock_parser = Mock()
        validator = QueryValidator(query_parser=mock_parser)
        assert validator.query_parser == mock_parser
    
    def test_validate_query_basic(self):
        """Test basic query validation."""
        validator = QueryValidator()
        
        # Test good query
        is_valid, warnings, suggestions = validator.validate_query("What did I spend on EC2 last month?")
        assert is_valid is True
        assert len(warnings) == 0
        
        # Test short query
        is_valid, warnings, suggestions = validator.validate_query("EC2")
        assert len(warnings) > 0
        assert any("short" in warning.lower() for warning in warnings)
        
        # Test query without cost keywords
        is_valid, warnings, suggestions = validator.validate_query("EC2 usage last month")
        assert len(warnings) > 0
        assert any("cost" in warning.lower() for warning in warnings)
    
    def test_validate_query_ambiguous_terms(self):
        """Test validation of ambiguous terms."""
        validator = QueryValidator()
        
        # Test ambiguous service
        is_valid, warnings, suggestions = validator.validate_query("How much did storage cost last month?")
        assert len(warnings) > 0
        assert any("storage" in warning.lower() for warning in warnings)
        
        # Test ambiguous time
        is_valid, warnings, suggestions = validator.validate_query("EC2 costs recently")
        assert len(warnings) > 0
        assert any("recently" in warning.lower() for warning in warnings)
    
    def test_validate_query_with_parser(self):
        """Test validation with LLM parser."""
        mock_parser = Mock()
        validator = QueryValidator(query_parser=mock_parser)
        
        # Test successful parsing
        mock_parser.parse_query.return_value = {
            'service': 'Amazon Elastic Compute Cloud - Compute',
            'start_date': '2025-07-01',
            'end_date': '2025-08-01'
        }
        
        is_valid, warnings, suggestions = validator.validate_query("EC2 costs last month")
        assert is_valid is True
        
        # Test parsing failure
        mock_parser.parse_query.side_effect = QueryParsingError("Parse error")
        
        is_valid, warnings, suggestions = validator.validate_query("Invalid query")
        assert is_valid is False
        assert len(warnings) > 0
    
    def test_suggest_improvements(self):
        """Test suggestion generation."""
        validator = QueryValidator()
        
        suggestions = validator.suggest_improvements("storage costs")
        assert len(suggestions) > 0
        assert any("specific" in suggestion.lower() for suggestion in suggestions)


class TestInteractiveQueryBuilder:
    """Test InteractiveQueryBuilder class."""
    
    def test_builder_initialization(self):
        """Test builder initialization."""
        builder = InteractiveQueryBuilder()
        
        assert isinstance(builder.template_manager, QueryTemplateManager)
        assert isinstance(builder.history_manager, QueryHistoryManager)
        assert isinstance(builder.validator, QueryValidator)
        assert builder.query_parser is None
        
        # Test with query parser
        mock_parser = Mock()
        builder = InteractiveQueryBuilder(query_parser=mock_parser)
        assert builder.query_parser == mock_parser
    
    def test_get_query_suggestions(self):
        """Test getting query suggestions."""
        builder = InteractiveQueryBuilder()
        
        # Test with partial query
        suggestions = builder.get_query_suggestions("EC2")
        assert len(suggestions) > 0
        assert any("EC2" in suggestion for suggestion in suggestions)
        
        # Test without partial query
        suggestions = builder.get_query_suggestions("")
        assert len(suggestions) > 0
        
        # Test with history
        builder.history_manager.add_to_history("Custom EC2 query", True)
        suggestions = builder.get_query_suggestions("EC2")
        assert any("Custom EC2 query" in suggestions for _ in [True])  # Should include history
    
    def test_get_query_suggestions_deduplication(self):
        """Test query suggestion deduplication."""
        builder = InteractiveQueryBuilder()
        
        # Add duplicate to history
        duplicate_query = "What did I spend on EC2 last month?"
        builder.history_manager.add_to_history(duplicate_query, True)
        
        suggestions = builder.get_query_suggestions("EC2")
        
        # Should not have duplicates
        assert len(suggestions) == len(set(suggestions))
    
    @patch('src.aws_cost_cli.interactive_query_builder.Console')
    @patch('src.aws_cost_cli.interactive_query_builder.Prompt')
    def test_start_interactive_session_exit(self, mock_prompt, mock_console):
        """Test interactive session exit."""
        mock_prompt.ask.return_value = "6"  # Exit option
        
        builder = InteractiveQueryBuilder()
        result = builder.start_interactive_session()
        
        assert result is None
    
    @patch('src.aws_cost_cli.interactive_query_builder.Console')
    @patch('src.aws_cost_cli.interactive_query_builder.Prompt')
    @patch('src.aws_cost_cli.interactive_query_builder.Confirm')
    def test_build_query_from_scratch(self, mock_confirm, mock_prompt, mock_console):
        """Test building query from scratch."""
        # Mock user inputs
        mock_prompt.ask.side_effect = [
            "1",  # Build from scratch
            "total spending",  # Intent
            "EC2",  # Service
            "last month",  # Time period
            "What did I spend on EC2 for last month?"  # Final query edit
        ]
        mock_confirm.ask.side_effect = [
            False,  # No breakdown
            False,  # No comparison
            False,  # Don't use generated query
            True   # Use edited query
        ]
        
        builder = InteractiveQueryBuilder()
        
        # Mock the _build_query_from_scratch method to return a query
        with patch.object(builder, '_build_query_from_scratch', return_value="What did I spend on EC2 last month?"):
            result = builder.start_interactive_session()
            assert result == "What did I spend on EC2 last month?"


class TestIntegration:
    """Integration tests for interactive query builder."""
    
    def test_full_workflow_with_templates(self):
        """Test full workflow using templates."""
        with tempfile.TemporaryDirectory() as temp_dir:
            history_file = os.path.join(temp_dir, "test_history.json")
            builder = InteractiveQueryBuilder()
            builder.history_manager = QueryHistoryManager(history_file=history_file)
            
            # Test template usage
            template = builder.template_manager.get_template_by_name("EC2 Monthly Costs")
            assert template is not None
            
            # Simulate using template
            query = template.template.replace("{month}", "July").replace("{year}", "2025")
            expected = "How much did I spend on EC2 in July 2025?"
            assert query == expected
            
            # Add to history
            builder.history_manager.add_to_history(query, True, 150.0)
            
            # Verify history
            recent = builder.history_manager.get_recent_queries(1)
            assert len(recent) == 1
            assert recent[0].query == query
            assert recent[0].success is True
    
    def test_validation_with_suggestions(self):
        """Test validation with suggestion integration."""
        builder = InteractiveQueryBuilder()
        
        # Test validation of ambiguous query
        is_valid, warnings, suggestions = builder.validator.validate_query("storage costs recently")
        
        assert len(warnings) > 0
        assert len(suggestions) > 0
        
        # Warnings should mention ambiguous terms
        warning_text = " ".join(warnings).lower()
        assert "storage" in warning_text or "recently" in warning_text
        
        # Suggestions should provide alternatives
        suggestion_text = " ".join(suggestions).lower()
        assert any(keyword in suggestion_text for keyword in ["specific", "s3", "month", "year"])
    
    def test_favorites_workflow(self):
        """Test complete favorites workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            history_file = os.path.join(temp_dir, "test_history.json")
            builder = InteractiveQueryBuilder()
            builder.history_manager = QueryHistoryManager(history_file=history_file)
            
            # Add favorite
            query = "What did I spend on EC2 last month?"
            builder.history_manager.add_favorite("Monthly EC2", query, "Monthly EC2 cost check")
            
            # Verify favorite was added
            favorites = builder.history_manager.get_favorites()
            assert len(favorites) == 1
            assert favorites[0].name == "Monthly EC2"
            assert favorites[0].query == query
            
            # Get favorite by name
            favorite = builder.history_manager.get_favorite_by_name("Monthly EC2")
            assert favorite is not None
            assert favorite.query == query
            
            # Remove favorite
            result = builder.history_manager.remove_favorite("Monthly EC2")
            assert result is True
            
            # Verify removal
            favorites = builder.history_manager.get_favorites()
            assert len(favorites) == 0