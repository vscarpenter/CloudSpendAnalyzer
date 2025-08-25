# Data Export Guide

AWS Cost Explorer CLI provides comprehensive data export capabilities to help you share, analyze, and archive your cost data in various formats.

## Available Export Formats

### CSV Export
- **Format**: Comma-separated values
- **Use case**: Spreadsheet analysis, data processing
- **Features**: 
  - Metadata headers with query information
  - Detailed cost breakdown by time period
  - Service/resource grouping
  - Trend analysis data
  - Forecast data (if available)

### JSON Export
- **Format**: Structured JSON
- **Use case**: API integration, programmatic analysis
- **Features**:
  - Complete data structure preservation
  - Nested grouping and metrics
  - Metadata and timestamps
  - Trend and forecast data

### Excel Export
- **Format**: Microsoft Excel (.xlsx)
- **Use case**: Business reporting, presentations
- **Features**:
  - Multiple worksheets (Summary, Detailed Data)
  - Professional formatting and styling
  - Charts and visualizations
  - Currency formatting
  - Conditional formatting for trends
- **Requirements**: `pip install openpyxl`

## Command Line Usage

### Basic Export

```bash
# Export to CSV (default format)
aws-cost-cli export "EC2 costs last month"

# Export to specific format
aws-cost-cli export "S3 spending this year" --format excel

# Specify output file
aws-cost-cli export "Total costs Q1" --format json --output quarterly_costs.json
```

### Advanced Export Options

```bash
# Use specific AWS profile
aws-cost-cli export "RDS costs" --profile production --format csv

# Force fresh data (bypass cache)
aws-cost-cli export "Lambda costs today" --fresh --format json

# Use custom configuration
aws-cost-cli export "All services this month" --config-file /path/to/config.yaml
```

## Email Reports

Send cost reports directly via email with attachments in multiple formats.

### Basic Email Report

```bash
aws-cost-cli email-report "Monthly AWS costs" \
  --recipients "finance@company.com,admin@company.com" \
  --smtp-host smtp.gmail.com \
  --smtp-username your-email@gmail.com \
  --smtp-password your-app-password
```

### Advanced Email Options

```bash
aws-cost-cli email-report "Quarterly cost analysis" \
  --recipients "team@company.com" \
  --subject "Q1 2024 AWS Cost Report" \
  --attachments csv --attachments excel \
  --smtp-host mail.company.com \
  --smtp-port 587 \
  --smtp-username reports \
  --smtp-password secret123
```

### Email Configuration

The email report feature supports various SMTP configurations:

- **Gmail**: Use app passwords for authentication
- **Office 365**: Standard SMTP settings
- **Corporate SMTP**: Custom host and port settings
- **TLS/SSL**: Enabled by default, use `--no-tls` to disable

## Export Data Structure

### CSV Format Structure

```csv
# AWS Cost Data Export
# Generated: 2024-01-15T10:30:00
# Query: EC2 costs last month
# Service: EC2
# Period: 2024-01-01 to 2024-01-31
# Total Cost: 1234.56 USD

Period Start,Period End,Total Cost,Currency,Estimated,Group Keys,Group Cost
2024-01-01,2024-01-07,150.00,USD,False,EC2-Instance,150.00
2024-01-08,2024-01-14,175.50,USD,False,EC2-Instance,175.50
...

# Trend Analysis
Current Period Cost:,1234.56
Comparison Period Cost:,1100.00
Change Amount:,134.56
Change Percentage:,12.2
Trend Direction:,up

# Forecast Data
Forecast Period Start,Forecast Period End,Forecasted Amount,Lower Bound,Upper Bound,Accuracy
2024-02-01,2024-02-29,1300.00,1200.00,1400.00,0.85
```

### JSON Format Structure

```json
{
  "metadata": {
    "generated_at": "2024-01-15T10:30:00",
    "query": "EC2 costs last month",
    "service": "EC2",
    "granularity": "DAILY",
    "time_period": {
      "start": "2024-01-01T00:00:00",
      "end": "2024-01-31T23:59:59"
    }
  },
  "summary": {
    "total_cost": {
      "amount": 1234.56,
      "currency": "USD"
    },
    "currency": "USD",
    "group_definitions": ["SERVICE"]
  },
  "results": [
    {
      "time_period": {
        "start": "2024-01-01T00:00:00",
        "end": "2024-01-07T23:59:59"
      },
      "total": {
        "amount": 150.00,
        "currency": "USD"
      },
      "estimated": false,
      "groups": [
        {
          "keys": ["EC2-Instance"],
          "metrics": {
            "BlendedCost": {
              "amount": 150.00,
              "currency": "USD"
            }
          }
        }
      ]
    }
  ],
  "trend_analysis": {
    "current_period": {
      "amount": 1234.56,
      "currency": "USD"
    },
    "comparison_period": {
      "amount": 1100.00,
      "currency": "USD"
    },
    "change": {
      "amount": 134.56,
      "currency": "USD",
      "percentage": 12.2,
      "direction": "up"
    }
  },
  "forecast": [
    {
      "period": {
        "start": "2024-02-01T00:00:00",
        "end": "2024-02-29T23:59:59"
      },
      "forecasted_amount": {
        "amount": 1300.00,
        "currency": "USD"
      },
      "confidence_interval": {
        "lower": {
          "amount": 1200.00,
          "currency": "USD"
        },
        "upper": {
          "amount": 1400.00,
          "currency": "USD"
        }
      },
      "prediction_accuracy": 0.85
    }
  ]
}
```

## Programmatic Usage

You can also use the export functionality programmatically:

```python
from aws_cost_cli.data_exporter import ExportManager
from aws_cost_cli.models import CostData, QueryParameters

# Initialize export manager
export_manager = ExportManager()

# Export data
export_manager.export_data(
    cost_data=your_cost_data,
    query_params=your_query_params,
    format_type='csv',
    output_path='output.csv'
)

# Send email report
export_manager.send_email_report(
    cost_data=your_cost_data,
    query_params=your_query_params,
    smtp_config={
        'host': 'smtp.gmail.com',
        'port': 587,
        'username': 'user@gmail.com',
        'password': 'password',
        'use_tls': True
    },
    recipients=['recipient@company.com'],
    attachment_formats=['csv', 'json']
)
```

## Best Practices

### File Naming
- Use descriptive names that include service and time period
- Include timestamps for regular exports
- Use consistent naming conventions for automated exports

### Email Reports
- Use app passwords for Gmail authentication
- Test SMTP settings before scheduling automated reports
- Include multiple recipients for important reports
- Use meaningful subject lines

### Data Security
- Be cautious when emailing cost data
- Use secure SMTP connections (TLS/SSL)
- Consider data retention policies for exported files
- Sanitize sensitive information if needed

### Performance
- Use caching for repeated exports of the same data
- Export large datasets in smaller time periods
- Consider using JSON format for programmatic processing
- Use CSV format for spreadsheet analysis

## Troubleshooting

### Common Issues

1. **Excel export not available**
   - Install openpyxl: `pip install openpyxl`

2. **Email sending fails**
   - Check SMTP settings and credentials
   - Verify network connectivity
   - Use app passwords for Gmail

3. **Large file exports**
   - Break down queries into smaller time periods
   - Use appropriate granularity (MONTHLY vs DAILY)

4. **Permission errors**
   - Ensure write permissions for output directory
   - Check AWS credentials and permissions

### Getting Help

For additional support:
- Check the main CLI help: `aws-cost-cli --help`
- View command-specific help: `aws-cost-cli export --help`
- Review error messages for specific guidance
- Check AWS credentials and permissions