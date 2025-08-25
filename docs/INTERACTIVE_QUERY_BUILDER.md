# Interactive Query Builder

The AWS Cost Explorer CLI includes an interactive query builder that helps users construct cost queries through guided prompts, templates, and validation.

## Features

### 1. Guided Query Construction
Build queries step-by-step with interactive prompts:
- Intent specification (what you want to know)
- Service selection (EC2, S3, RDS, etc.)
- Time period selection (last month, this year, Q3 2025, etc.)
- Additional options (breakdowns, comparisons)

### 2. Query Templates
Pre-built templates for common use cases:
- **Service Templates**: EC2 Monthly Costs, S3 Storage Costs, RDS Database Costs
- **Time Templates**: Monthly Total Costs, Quarterly Costs, Year-to-Date Costs
- **Comparison Templates**: Month-over-Month, Year-over-Year comparisons
- **Analysis Templates**: Service Breakdown, Top Spending Services, Cost Forecast
- **Budget Templates**: Budget Tracking, Unused Resources

### 3. Query History and Favorites
- Automatic query history tracking (last 100 queries)
- Save frequently used queries as favorites
- Browse and reuse successful queries
- Track query execution times and success rates

### 4. Query Validation and Suggestions
- Real-time query validation with warnings and suggestions
- Detection of ambiguous terms (e.g., "storage" instead of "S3")
- Suggestions for improving query clarity
- Integration with LLM parser for advanced validation

## Usage

### Interactive Mode
Start the interactive query builder:
```bash
aws-cost-cli interactive
```

This opens a menu with options:
1. Build a new query from scratch
2. Use a query template
3. Browse query history
4. Manage favorites
5. Validate a query
6. Exit

### Favorites Management
Manage favorite queries from the command line:

```bash
# List all favorites
aws-cost-cli favorites --action list

# Add a new favorite
aws-cost-cli favorites --action add --name "Monthly EC2" --query "What did I spend on EC2 last month?" --description "Monthly EC2 cost check"

# Remove a favorite
aws-cost-cli favorites --action remove --name "Monthly EC2"

# Run a favorite query
aws-cost-cli favorites --action run --name "Monthly EC2"
```

### Enhanced Suggestions
Get improved query suggestions:
```bash
# Get suggestions for partial input
aws-cost-cli suggest "EC2"

# Get general suggestions
aws-cost-cli suggest
```

## Query Templates

### Service-Specific Templates
- **EC2 Monthly Costs**: `"How much did I spend on EC2 in {month} {year}?"`
- **S3 Storage Costs**: `"What are my S3 costs for {period}?"`
- **RDS Database Costs**: `"Show me RDS spending for {period} broken down by service"`
- **Lambda Function Costs**: `"How much did Lambda cost me in {period}?"`

### Time-Based Templates
- **Monthly Total Costs**: `"What was my total AWS bill for {month} {year}?"`
- **Quarterly Costs**: `"What did I spend in {quarter} {year}?"`
- **Year-to-Date Costs**: `"What have I spent so far this year?"`
- **Daily Costs**: `"Show me daily costs for this month"`

### Comparison Templates
- **Month-over-Month**: `"How do this month's costs compare to last month?"`
- **Year-over-Year**: `"How do this year's costs compare to last year?"`
- **Service Trends**: `"Show me {service} cost trends for the last 6 months"`

### Analysis Templates
- **Service Breakdown**: `"What services did I use {period} and how much did each cost?"`
- **Top Spending**: `"What are my top 5 most expensive services {period}?"`
- **Cost Forecast**: `"What will my costs be for the next 3 months?"`

## Query Validation

The validator checks for common issues:

### Warnings
- **Ambiguous Services**: "storage" → suggest "S3", "EBS", etc.
- **Ambiguous Time**: "recently" → suggest "last month", "this year"
- **Missing Context**: Short queries without service or time context
- **Missing Cost Keywords**: Queries without "cost", "spend", "bill"

### Suggestions
- Specific service names instead of generic terms
- Concrete time periods instead of vague references
- Complete query examples based on detected intent

## Data Storage

### History Storage
- Location: `~/.aws-cost-cli/query_history.json`
- Retention: Last 100 queries
- Includes: Query text, timestamp, success status, execution time, error messages

### Favorites Storage
- Location: `~/.aws-cost-cli/favorites.json`
- Includes: Name, query, description, creation date
- No limit on number of favorites

## Integration

The interactive query builder integrates seamlessly with:
- **Query Pipeline**: Executes queries through the main pipeline
- **LLM Providers**: Uses configured LLM for validation and parsing
- **Cache System**: Benefits from query caching
- **Response Formatting**: Uses configured output format

## Examples

### Building a Query from Scratch
1. Start interactive mode: `aws-cost-cli interactive`
2. Choose "Build a new query from scratch"
3. Describe intent: "total spending"
4. Select service: "EC2"
5. Choose time period: "last month"
6. Add breakdown: Yes
7. Generated query: "What did I spend on EC2 for last month broken down by service?"

### Using Templates
1. Start interactive mode: `aws-cost-cli interactive`
2. Choose "Use a query template"
3. Select category: "service"
4. Choose template: "EC2 Monthly Costs"
5. Fill parameters: month="July", year="2025"
6. Final query: "How much did I spend on EC2 in July 2025?"

### Managing Favorites
```bash
# Add a complex query as favorite
aws-cost-cli favorites --action add \
  --name "Quarterly Analysis" \
  --query "Show me service breakdown for Q3 2025 with cost trends" \
  --description "Quarterly cost analysis with trends"

# Run the favorite later
aws-cost-cli favorites --action run --name "Quarterly Analysis"
```

## Benefits

1. **Reduced Learning Curve**: Guided construction helps new users
2. **Consistency**: Templates ensure well-formed queries
3. **Efficiency**: Favorites and history speed up repeated queries
4. **Quality**: Validation catches common mistakes before execution
5. **Discovery**: Templates showcase available query capabilities