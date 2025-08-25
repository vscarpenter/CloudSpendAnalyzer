"""Interactive query builder for guided cost query construction."""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict

from rich.console import Console
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from .models import TimePeriodGranularity, MetricType, DateRangeType, TrendAnalysisType
from .query_processor import QueryParser
from .exceptions import ValidationError, QueryParsingError


@dataclass
class QueryTemplate:
    """Template for common query patterns."""
    name: str
    description: str
    template: str
    category: str
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


@dataclass
class QueryHistoryEntry:
    """Entry in query history."""
    query: str
    timestamp: datetime
    success: bool
    execution_time_ms: Optional[float] = None
    error_message: Optional[str] = None


@dataclass
class QueryFavorite:
    """Favorite query entry."""
    name: str
    query: str
    description: Optional[str] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class QueryTemplateManager:
    """Manages query templates for common use cases."""
    
    def __init__(self):
        self.templates = self._load_default_templates()
    
    def _load_default_templates(self) -> List[QueryTemplate]:
        """Load default query templates."""
        return [
            # Service-specific queries
            QueryTemplate(
                name="EC2 Monthly Costs",
                description="Get EC2 costs for a specific month",
                template="How much did I spend on EC2 in {month} {year}?",
                category="service",
                parameters={"service": "EC2", "month": "last month", "year": "2025"}
            ),
            QueryTemplate(
                name="S3 Storage Costs",
                description="Get S3 storage costs for a time period",
                template="What are my S3 costs for {period}?",
                category="service",
                parameters={"service": "S3", "period": "this month"}
            ),
            QueryTemplate(
                name="RDS Database Costs",
                description="Get RDS costs with breakdown",
                template="Show me RDS spending for {period} broken down by service",
                category="service",
                parameters={"service": "RDS", "period": "last month"}
            ),
            QueryTemplate(
                name="Lambda Function Costs",
                description="Get Lambda execution costs",
                template="How much did Lambda cost me in {period}?",
                category="service",
                parameters={"service": "Lambda", "period": "this year"}
            ),
            
            # Time-based queries
            QueryTemplate(
                name="Monthly Total Costs",
                description="Get total costs for a specific month",
                template="What was my total AWS bill for {month} {year}?",
                category="time",
                parameters={"month": "last month", "year": "2025"}
            ),
            QueryTemplate(
                name="Quarterly Costs",
                description="Get costs for a specific quarter",
                template="What did I spend in {quarter} {year}?",
                category="time",
                parameters={"quarter": "Q3", "year": "2025"}
            ),
            QueryTemplate(
                name="Year-to-Date Costs",
                description="Get year-to-date spending",
                template="What have I spent so far this year?",
                category="time"
            ),
            QueryTemplate(
                name="Daily Costs This Month",
                description="Get daily cost breakdown for current month",
                template="Show me daily costs for this month",
                category="time"
            ),
            
            # Comparison queries
            QueryTemplate(
                name="Month-over-Month Comparison",
                description="Compare current month to previous month",
                template="How do this month's costs compare to last month?",
                category="comparison"
            ),
            QueryTemplate(
                name="Year-over-Year Comparison",
                description="Compare current year to previous year",
                template="How do this year's costs compare to last year?",
                category="comparison"
            ),
            QueryTemplate(
                name="Service Cost Trends",
                description="Get cost trends for a specific service",
                template="Show me {service} cost trends for the last 6 months",
                category="comparison",
                parameters={"service": "EC2"}
            ),
            
            # Analysis queries
            QueryTemplate(
                name="Service Breakdown",
                description="Get costs broken down by service",
                template="What services did I use {period} and how much did each cost?",
                category="analysis",
                parameters={"period": "last month"}
            ),
            QueryTemplate(
                name="Top Spending Services",
                description="Find the most expensive services",
                template="What are my top 5 most expensive services {period}?",
                category="analysis",
                parameters={"period": "this year"}
            ),
            QueryTemplate(
                name="Cost Forecast",
                description="Get cost forecast for upcoming months",
                template="What will my costs be for the next 3 months?",
                category="analysis"
            ),
            
            # Budget and optimization
            QueryTemplate(
                name="Budget Tracking",
                description="Track spending against budget",
                template="How much have I spent this month and how does it compare to my budget?",
                category="budget"
            ),
            QueryTemplate(
                name="Unused Resources",
                description="Find potentially unused resources",
                template="Show me services with zero usage {period}",
                category="optimization",
                parameters={"period": "last month"}
            ),
        ]
    
    def get_templates_by_category(self, category: str = None) -> List[QueryTemplate]:
        """Get templates filtered by category."""
        if category is None:
            return self.templates
        return [t for t in self.templates if t.category == category]
    
    def get_template_by_name(self, name: str) -> Optional[QueryTemplate]:
        """Get a specific template by name."""
        for template in self.templates:
            if template.name == name:
                return template
        return None
    
    def get_categories(self) -> List[str]:
        """Get all available template categories."""
        categories = set(t.category for t in self.templates)
        return sorted(list(categories))


class QueryHistoryManager:
    """Manages query history and favorites."""
    
    def __init__(self, history_file: str = None):
        self.history_file = history_file or self._get_default_history_file()
        self.favorites_file = self.history_file.replace('history.json', 'favorites.json')
        self.history = self._load_history()
        self.favorites = self._load_favorites()
    
    def _get_default_history_file(self) -> str:
        """Get default history file path."""
        home_dir = Path.home()
        config_dir = home_dir / '.aws-cost-cli'
        config_dir.mkdir(exist_ok=True)
        return str(config_dir / 'query_history.json')
    
    def _load_history(self) -> List[QueryHistoryEntry]:
        """Load query history from file."""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    return [
                        QueryHistoryEntry(
                            query=entry['query'],
                            timestamp=datetime.fromisoformat(entry['timestamp']),
                            success=entry['success'],
                            execution_time_ms=entry.get('execution_time_ms'),
                            error_message=entry.get('error_message')
                        )
                        for entry in data
                    ]
        except Exception:
            pass
        return []
    
    def _save_history(self):
        """Save query history to file."""
        try:
            os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
            data = []
            for entry in self.history:
                data.append({
                    'query': entry.query,
                    'timestamp': entry.timestamp.isoformat(),
                    'success': entry.success,
                    'execution_time_ms': entry.execution_time_ms,
                    'error_message': entry.error_message
                })
            with open(self.history_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass
    
    def _load_favorites(self) -> List[QueryFavorite]:
        """Load favorites from file."""
        try:
            if os.path.exists(self.favorites_file):
                with open(self.favorites_file, 'r') as f:
                    data = json.load(f)
                    return [
                        QueryFavorite(
                            name=entry['name'],
                            query=entry['query'],
                            description=entry.get('description'),
                            created_at=datetime.fromisoformat(entry['created_at'])
                        )
                        for entry in data
                    ]
        except Exception:
            pass
        return []
    
    def _save_favorites(self):
        """Save favorites to file."""
        try:
            os.makedirs(os.path.dirname(self.favorites_file), exist_ok=True)
            data = []
            for favorite in self.favorites:
                data.append({
                    'name': favorite.name,
                    'query': favorite.query,
                    'description': favorite.description,
                    'created_at': favorite.created_at.isoformat()
                })
            with open(self.favorites_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass
    
    def add_to_history(self, query: str, success: bool, execution_time_ms: float = None, error_message: str = None):
        """Add query to history."""
        entry = QueryHistoryEntry(
            query=query,
            timestamp=datetime.now(),
            success=success,
            execution_time_ms=execution_time_ms,
            error_message=error_message
        )
        self.history.insert(0, entry)  # Add to beginning
        
        # Keep only last 100 entries
        if len(self.history) > 100:
            self.history = self.history[:100]
        
        self._save_history()
    
    def get_recent_queries(self, limit: int = 10) -> List[QueryHistoryEntry]:
        """Get recent queries."""
        return self.history[:limit]
    
    def get_successful_queries(self, limit: int = 10) -> List[QueryHistoryEntry]:
        """Get recent successful queries."""
        successful = [entry for entry in self.history if entry.success]
        return successful[:limit]
    
    def add_favorite(self, name: str, query: str, description: str = None):
        """Add query to favorites."""
        # Check if name already exists
        for favorite in self.favorites:
            if favorite.name == name:
                raise ValidationError(f"Favorite with name '{name}' already exists")
        
        favorite = QueryFavorite(
            name=name,
            query=query,
            description=description
        )
        self.favorites.append(favorite)
        self._save_favorites()
    
    def remove_favorite(self, name: str) -> bool:
        """Remove favorite by name."""
        for i, favorite in enumerate(self.favorites):
            if favorite.name == name:
                del self.favorites[i]
                self._save_favorites()
                return True
        return False
    
    def get_favorites(self) -> List[QueryFavorite]:
        """Get all favorites."""
        return self.favorites
    
    def get_favorite_by_name(self, name: str) -> Optional[QueryFavorite]:
        """Get favorite by name."""
        for favorite in self.favorites:
            if favorite.name == name:
                return favorite
        return None


class QueryValidator:
    """Validates and suggests improvements for queries."""
    
    def __init__(self, query_parser: QueryParser = None):
        self.query_parser = query_parser
        self.common_issues = {
            'ambiguous_service': [
                'storage', 'compute', 'database', 'networking'
            ],
            'ambiguous_time': [
                'recently', 'a while ago', 'some time ago'
            ],
            'missing_context': [
                'costs', 'spending', 'bill'
            ]
        }
    
    def validate_query(self, query: str) -> Tuple[bool, List[str], List[str]]:
        """
        Validate a query and return validation results.
        
        Returns:
            Tuple of (is_valid, warnings, suggestions)
        """
        warnings = []
        suggestions = []
        is_valid = True
        
        query_lower = query.lower()
        
        # Check for ambiguous services
        for ambiguous in self.common_issues['ambiguous_service']:
            if ambiguous in query_lower:
                warnings.append(f"'{ambiguous}' is ambiguous - consider specifying exact service (EC2, S3, RDS, etc.)")
                suggestions.append(f"Try: '{query.replace(ambiguous, 'EC2')}' or similar with specific service")
        
        # Check for ambiguous time references
        for ambiguous in self.common_issues['ambiguous_time']:
            if ambiguous in query_lower:
                warnings.append(f"'{ambiguous}' is vague - consider specific time periods")
                suggestions.append("Try: 'last month', 'this year', 'Q3 2025', etc.")
        
        # Check if query is too short or lacks context
        if len(query.split()) < 3:
            warnings.append("Query seems very short - consider adding more context")
            suggestions.append("Try: 'What did I spend on EC2 last month?'")
        
        # Check for common missing elements
        has_service = any(service in query_lower for service in ['ec2', 's3', 'rds', 'lambda', 'cloudfront'])
        has_time = any(time_ref in query_lower for time_ref in ['month', 'year', 'day', 'quarter', 'week', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december', 'q1', 'q2', 'q3', 'q4', '2024', '2025', '2026'])
        has_cost_keyword = any(keyword in query_lower for keyword in ['cost', 'spend', 'bill', 'charge', 'price'])
        
        if not has_cost_keyword:
            warnings.append("Query doesn't mention costs - consider adding 'cost', 'spending', or 'bill'")
            suggestions.append(f"Try: '{query} costs' or 'How much did I spend on {query}?'")
        
        if not has_time and not has_service:
            warnings.append("Query lacks both service and time context")
            suggestions.append("Try: 'EC2 costs last month' or 'Total spending this year'")
        elif not has_time and has_service:
            warnings.append("Query lacks time context - consider specifying when")
            suggestions.append("Try adding: 'last month', 'this year', 'Q3 2025', etc.")
        elif has_time and not has_service and not any(word in query_lower for word in ['total', 'all', 'overall', 'entire']):
            warnings.append("Query lacks service context - consider specifying which service or use 'total'")
            suggestions.append("Try: 'EC2 costs last month' or 'Total AWS costs last month'")
        
        # Try parsing with LLM if available
        if self.query_parser:
            try:
                parsed = self.query_parser.parse_query(query)
                if not parsed.get('service') and not parsed.get('start_date'):
                    warnings.append("Query may be too vague for accurate parsing")
                    suggestions.append("Consider being more specific about service or time period")
            except Exception:
                is_valid = False
                warnings.append("Query could not be parsed - may be too complex or unclear")
                suggestions.append("Try simplifying the query or using a template")
        
        return is_valid, warnings, suggestions
    
    def suggest_improvements(self, query: str) -> List[str]:
        """Suggest improvements for a query."""
        _, _, suggestions = self.validate_query(query)
        return suggestions


class InteractiveQueryBuilder:
    """Interactive query builder with guided construction."""
    
    def __init__(self, query_parser: QueryParser = None, config_path: str = None):
        self.console = Console()
        self.template_manager = QueryTemplateManager()
        self.history_manager = QueryHistoryManager()
        self.validator = QueryValidator(query_parser)
        self.query_parser = query_parser
    
    def start_interactive_session(self) -> Optional[str]:
        """Start an interactive query building session."""
        self.console.print(Panel(
            Text("🔍 Interactive Query Builder", style="bold blue"),
            subtitle="Build AWS cost queries step by step",
            border_style="blue"
        ))
        
        while True:
            self.console.print("\n📋 What would you like to do?")
            self.console.print("1. Build a new query from scratch")
            self.console.print("2. Use a query template")
            self.console.print("3. Browse query history")
            self.console.print("4. Manage favorites")
            self.console.print("5. Validate a query")
            self.console.print("6. Exit")
            
            choice = Prompt.ask("Choose an option", choices=["1", "2", "3", "4", "5", "6"], default="1")
            
            if choice == "1":
                query = self._build_query_from_scratch()
                if query:
                    return query
            elif choice == "2":
                query = self._use_template()
                if query:
                    return query
            elif choice == "3":
                query = self._browse_history()
                if query:
                    return query
            elif choice == "4":
                self._manage_favorites()
            elif choice == "5":
                self._validate_query()
            elif choice == "6":
                self.console.print("👋 Goodbye!")
                return None
    
    def _build_query_from_scratch(self) -> Optional[str]:
        """Build a query from scratch with guided prompts."""
        self.console.print(Panel(
            Text("🏗️  Building Query from Scratch", style="bold green"),
            border_style="green"
        ))
        
        # Step 1: What do you want to know?
        self.console.print("\n🎯 What do you want to know about your AWS costs?")
        intent = Prompt.ask("Describe what you're looking for", default="total spending")
        
        # Step 2: Service selection
        self.console.print("\n🔧 Which AWS service are you interested in?")
        self.console.print("Leave blank for all services, or specify: EC2, S3, RDS, Lambda, etc.")
        service = Prompt.ask("Service (optional)").strip()
        
        # Step 3: Time period
        self.console.print("\n📅 What time period?")
        self.console.print("Examples: 'last month', 'this year', 'Q3 2025', 'July 2025'")
        time_period = Prompt.ask("Time period", default="last month")
        
        # Step 4: Additional options
        breakdown = Confirm.ask("\n📊 Do you want a breakdown by service?", default=False)
        comparison = Confirm.ask("📈 Do you want to compare to a previous period?", default=False)
        
        # Build the query
        query_parts = []
        
        if "total" in intent.lower() or "bill" in intent.lower():
            if service:
                query_parts.append(f"What did I spend on {service}")
            else:
                query_parts.append("What was my total AWS bill")
        elif "cost" in intent.lower() or "spend" in intent.lower():
            if service:
                query_parts.append(f"How much did {service} cost me")
            else:
                query_parts.append("How much did I spend")
        else:
            if service:
                query_parts.append(f"Show me {service} costs")
            else:
                query_parts.append("Show me my AWS costs")
        
        query_parts.append(f"for {time_period}")
        
        if breakdown:
            query_parts.append("broken down by service")
        
        if comparison:
            if "month" in time_period.lower():
                query_parts.append("compared to the previous month")
            elif "year" in time_period.lower():
                query_parts.append("compared to the previous year")
            else:
                query_parts.append("compared to the previous period")
        
        query = " ".join(query_parts) + "?"
        
        # Show the built query
        self.console.print(Panel(
            Text(f"Built Query: {query}", style="bold yellow"),
            title="Generated Query",
            border_style="yellow"
        ))
        
        # Validate the query
        is_valid, warnings, suggestions = self.validator.validate_query(query)
        
        if warnings:
            self.console.print("\n⚠️  Validation Warnings:")
            for warning in warnings:
                self.console.print(f"   • {warning}")
        
        if suggestions:
            self.console.print("\n💡 Suggestions:")
            for suggestion in suggestions:
                self.console.print(f"   • {suggestion}")
        
        # Ask for confirmation
        if Confirm.ask("\n✅ Use this query?", default=True):
            return query
        
        # Allow manual editing
        if Confirm.ask("📝 Would you like to edit the query manually?", default=True):
            edited_query = Prompt.ask("Enter your query", default=query)
            return edited_query
        
        return None
    
    def _use_template(self) -> Optional[str]:
        """Use a query template."""
        self.console.print(Panel(
            Text("📋 Query Templates", style="bold green"),
            border_style="green"
        ))
        
        # Show categories
        categories = self.template_manager.get_categories()
        self.console.print("\n📂 Template Categories:")
        for i, category in enumerate(categories, 1):
            self.console.print(f"  {i}. {category.title()}")
        
        category_choice = Prompt.ask(
            "Choose a category",
            choices=[str(i) for i in range(1, len(categories) + 1)],
            default="1"
        )
        
        selected_category = categories[int(category_choice) - 1]
        templates = self.template_manager.get_templates_by_category(selected_category)
        
        # Show templates in category
        self.console.print(f"\n📋 {selected_category.title()} Templates:")
        for i, template in enumerate(templates, 1):
            self.console.print(f"  {i}. {template.name}")
            self.console.print(f"     {template.description}")
        
        template_choice = Prompt.ask(
            "Choose a template",
            choices=[str(i) for i in range(1, len(templates) + 1)],
            default="1"
        )
        
        selected_template = templates[int(template_choice) - 1]
        
        # Fill in template parameters
        query = selected_template.template
        if selected_template.parameters:
            self.console.print(f"\n🔧 Customizing template: {selected_template.name}")
            for param, default_value in selected_template.parameters.items():
                value = Prompt.ask(f"Enter {param}", default=str(default_value))
                query = query.replace(f"{{{param}}}", value)
        
        self.console.print(Panel(
            Text(f"Template Query: {query}", style="bold yellow"),
            title="Generated Query",
            border_style="yellow"
        ))
        
        if Confirm.ask("✅ Use this query?", default=True):
            return query
        
        return None
    
    def _browse_history(self) -> Optional[str]:
        """Browse query history."""
        self.console.print(Panel(
            Text("📚 Query History", style="bold green"),
            border_style="green"
        ))
        
        recent_queries = self.history_manager.get_recent_queries(20)
        
        if not recent_queries:
            self.console.print("📭 No query history found")
            return None
        
        # Create table
        table = Table(title="Recent Queries")
        table.add_column("#", style="cyan", width=3)
        table.add_column("Query", style="white")
        table.add_column("Status", width=8)
        table.add_column("Time", style="dim", width=12)
        
        for i, entry in enumerate(recent_queries, 1):
            status = "✅ Success" if entry.success else "❌ Failed"
            time_str = entry.timestamp.strftime("%m/%d %H:%M")
            table.add_row(str(i), entry.query[:60] + "..." if len(entry.query) > 60 else entry.query, status, time_str)
        
        self.console.print(table)
        
        choice = Prompt.ask(
            "Select a query to reuse (or 'q' to go back)",
            choices=[str(i) for i in range(1, len(recent_queries) + 1)] + ["q"],
            default="q"
        )
        
        if choice != "q":
            selected_query = recent_queries[int(choice) - 1]
            self.console.print(f"\n📋 Selected: {selected_query.query}")
            
            if Confirm.ask("✅ Use this query?", default=True):
                return selected_query.query
        
        return None
    
    def _manage_favorites(self):
        """Manage favorite queries."""
        while True:
            self.console.print(Panel(
                Text("⭐ Favorite Queries", style="bold green"),
                border_style="green"
            ))
            
            favorites = self.history_manager.get_favorites()
            
            self.console.print("\n📋 What would you like to do?")
            self.console.print("1. View favorites")
            self.console.print("2. Add new favorite")
            self.console.print("3. Remove favorite")
            self.console.print("4. Back to main menu")
            
            choice = Prompt.ask("Choose an option", choices=["1", "2", "3", "4"], default="1")
            
            if choice == "1":
                self._view_favorites()
            elif choice == "2":
                self._add_favorite()
            elif choice == "3":
                self._remove_favorite()
            elif choice == "4":
                break
    
    def _view_favorites(self):
        """View favorite queries."""
        favorites = self.history_manager.get_favorites()
        
        if not favorites:
            self.console.print("📭 No favorites found")
            return
        
        table = Table(title="Favorite Queries")
        table.add_column("Name", style="cyan")
        table.add_column("Query", style="white")
        table.add_column("Description", style="dim")
        
        for favorite in favorites:
            table.add_row(
                favorite.name,
                favorite.query[:50] + "..." if len(favorite.query) > 50 else favorite.query,
                favorite.description or ""
            )
        
        self.console.print(table)
    
    def _add_favorite(self):
        """Add a new favorite query."""
        self.console.print("\n⭐ Add New Favorite")
        
        name = Prompt.ask("Favorite name")
        query = Prompt.ask("Query")
        description = Prompt.ask("Description (optional)", default="")
        
        try:
            self.history_manager.add_favorite(name, query, description or None)
            self.console.print(f"✅ Added favorite '{name}'")
        except ValidationError as e:
            self.console.print(f"❌ Error: {e}")
    
    def _remove_favorite(self):
        """Remove a favorite query."""
        favorites = self.history_manager.get_favorites()
        
        if not favorites:
            self.console.print("📭 No favorites to remove")
            return
        
        self.console.print("\n🗑️  Select favorite to remove:")
        for i, favorite in enumerate(favorites, 1):
            self.console.print(f"  {i}. {favorite.name}")
        
        choice = Prompt.ask(
            "Choose favorite to remove",
            choices=[str(i) for i in range(1, len(favorites) + 1)]
        )
        
        selected_favorite = favorites[int(choice) - 1]
        
        if Confirm.ask(f"Remove '{selected_favorite.name}'?", default=False):
            if self.history_manager.remove_favorite(selected_favorite.name):
                self.console.print(f"✅ Removed favorite '{selected_favorite.name}'")
            else:
                self.console.print("❌ Failed to remove favorite")
    
    def _validate_query(self):
        """Validate a user-entered query."""
        self.console.print(Panel(
            Text("🔍 Query Validation", style="bold green"),
            border_style="green"
        ))
        
        query = Prompt.ask("\n📝 Enter query to validate")
        
        is_valid, warnings, suggestions = self.validator.validate_query(query)
        
        # Show validation results
        if is_valid:
            self.console.print("✅ Query appears valid")
        else:
            self.console.print("❌ Query has issues")
        
        if warnings:
            self.console.print("\n⚠️  Warnings:")
            for warning in warnings:
                self.console.print(f"   • {warning}")
        
        if suggestions:
            self.console.print("\n💡 Suggestions:")
            for suggestion in suggestions:
                self.console.print(f"   • {suggestion}")
        
        # Try parsing if parser available
        if self.query_parser:
            try:
                self.console.print("\n🤖 Testing with LLM parser...")
                parsed = self.query_parser.parse_query(query)
                self.console.print("✅ Query parsed successfully")
                
                # Show parsed parameters
                if parsed.get('service'):
                    self.console.print(f"   Service: {parsed['service']}")
                if parsed.get('start_date'):
                    self.console.print(f"   Start Date: {parsed['start_date']}")
                if parsed.get('end_date'):
                    self.console.print(f"   End Date: {parsed['end_date']}")
                if parsed.get('granularity'):
                    self.console.print(f"   Granularity: {parsed['granularity']}")
                
            except Exception as e:
                self.console.print(f"❌ Parser error: {str(e)}")
    
    def get_query_suggestions(self, partial_query: str = "") -> List[str]:
        """Get query suggestions based on partial input."""
        suggestions = []
        
        # Template-based suggestions
        if partial_query:
            partial_lower = partial_query.lower()
            for template in self.template_manager.templates:
                if (partial_lower in template.name.lower() or 
                    partial_lower in template.description.lower() or
                    partial_lower in template.template.lower()):
                    suggestions.append(template.template)
        else:
            # Return popular templates
            popular_templates = [
                "What did I spend on EC2 last month?",
                "Show me my total AWS bill for this year",
                "What are my S3 costs this month?",
                "How do this month's costs compare to last month?",
                "What services did I use last month and how much did each cost?"
            ]
            suggestions.extend(popular_templates)
        
        # History-based suggestions
        successful_queries = self.history_manager.get_successful_queries(10)
        for entry in successful_queries:
            if partial_query.lower() in entry.query.lower():
                suggestions.append(entry.query)
        
        # Remove duplicates and limit
        seen = set()
        unique_suggestions = []
        for suggestion in suggestions:
            if suggestion not in seen:
                seen.add(suggestion)
                unique_suggestions.append(suggestion)
        
        return unique_suggestions[:10]