# AWS Cost CLI Configuration Templates

This directory contains pre-configured templates for common enterprise deployment scenarios. Each template is optimized for specific use cases and environments.

## Available Templates

### 1. Multi-Account Configuration (`multi-account.yaml`)
**Use Case**: Organizations with multiple AWS accounts that need consolidated cost analysis
- Configured for cross-account cost analysis
- Optimized caching for multiple account queries
- Parallel account processing
- Account-specific role assumptions

**Key Features**:
- Multi-account role-based access
- Consolidated billing support
- Account-specific cost breakdowns
- Enhanced security for cross-account access

### 2. Organization Configuration (`organization.yaml`)
**Use Case**: Large enterprises using AWS Organizations with consolidated billing
- Full organizational unit (OU) support
- Enterprise-grade database integration
- Comprehensive monitoring and alerting
- Advanced security and compliance features

**Key Features**:
- AWS Organizations integration
- PostgreSQL database support
- Advanced cost allocation and tagging
- Comprehensive audit logging
- Enterprise security controls

### 3. Production Configuration (`production.yaml`)
**Use Case**: Production environments requiring high reliability and performance
- Optimized for high-availability deployments
- Advanced monitoring and alerting
- Circuit breaker patterns for resilience
- Health check endpoints

**Key Features**:
- High-performance connection pooling
- Database-backed persistence
- Comprehensive health checks
- Production-grade logging and monitoring
- Resource limits and circuit breakers

### 4. Development Configuration (`development.yaml`)
**Use Case**: Development and testing environments
- Cost-optimized for development workflows
- Verbose logging for debugging
- Relaxed security for development ease
- Local file-based storage

**Key Features**:
- Debug-friendly logging
- Local cache and storage
- Cost-effective LLM provider settings
- Development-specific debugging features

## How to Use These Templates

### 1. Copy Template
```bash
# Copy the desired template to your config location
cp config/templates/production.yaml ~/.aws-cost-cli/config.yaml
```

### 2. Customize for Your Environment
Edit the copied configuration file to match your specific requirements:

```bash
# Edit the configuration
nano ~/.aws-cost-cli/config.yaml
```

### 3. Set Environment Variables
Many templates reference environment variables for sensitive information:

```bash
# Database password (for production/organization templates)
export AWS_COST_CLI_DB_PASSWORD="your-db-password"

# API keys for LLM providers
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"

# AWS credentials (if not using profiles)
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
```

### 4. Validate Configuration
```bash
# Test your configuration
aws-cost-cli config validate

# Test AWS connectivity
aws-cost-cli config test-connection
```

## Template Comparison

| Feature | Development | Production | Multi-Account | Organization |
|---------|-------------|------------|---------------|--------------|
| **Database** | File-based | PostgreSQL | File/Optional DB | PostgreSQL |
| **Caching** | 2 hours | 15 minutes | 30 minutes | 1 hour |
| **Logging** | DEBUG | WARN | INFO | INFO |
| **Monitoring** | Basic | Advanced | Advanced | Enterprise |
| **Security** | Relaxed | Strict | Strict | Enterprise |
| **Performance** | Basic | Optimized | Optimized | Enterprise |

## Environment-Specific Customizations

### Development Environment
- Uses cheaper LLM models (GPT-3.5-turbo)
- Longer cache TTL to reduce API calls
- Verbose logging for debugging
- No database requirements

### Production Environment  
- High-performance connection pooling
- Circuit breaker patterns for resilience
- Comprehensive health checks
- Structured JSON logging
- Rate limiting and security controls

### Multi-Account Environment
- Cross-account role assumption
- Parallel account processing
- Account-specific caching strategies
- Consolidated reporting features

### Enterprise/Organization Environment
- AWS Organizations integration
- Enterprise-grade database
- Comprehensive audit trails
- Advanced security controls
- Integration with enterprise tools (Slack, JIRA, etc.)

## Security Considerations

### Sensitive Information
Never commit sensitive information to version control:
- Database passwords
- API keys
- AWS credentials
- SMTP passwords

Use environment variables or secure secret management systems instead.

### File Permissions
Ensure configuration files have appropriate permissions:
```bash
chmod 600 ~/.aws-cost-cli/config.yaml  # Owner read/write only
```

### Network Security
For production deployments:
- Use TLS/SSL for all database connections
- Configure firewall rules appropriately
- Use VPC endpoints for AWS API calls when possible

## Common Customizations

### Adding New LLM Providers
```yaml
llm_provider: "custom"
llm_config:
  endpoint: "https://your-custom-llm-endpoint"
  api_key: "${CUSTOM_LLM_API_KEY}"
  model: "your-model-name"
```

### Custom Email Configuration
```yaml
email:
  enabled: true
  smtp_server: "your-smtp-server.com"
  smtp_port: 587
  sender: "costs@yourcompany.com"
  recipients:
    - "team@yourcompany.com"
```

### Custom Database Configuration
```yaml
database:
  enabled: true
  type: "mysql"  # or "sqlite", "postgresql"
  host: "your-db-host"
  port: 3306
  database: "cost_analytics"
  username: "cost_user"
```

## Troubleshooting

### Configuration Validation
```bash
# Validate configuration syntax
aws-cost-cli config validate

# Test database connectivity (if enabled)
aws-cost-cli config test-db

# Test LLM provider connectivity
aws-cost-cli config test-llm
```

### Common Issues

1. **Database Connection Failed**
   - Check database credentials and connectivity
   - Verify firewall rules
   - Ensure database server is running

2. **LLM API Errors**
   - Verify API keys are set correctly
   - Check API quotas and limits
   - Validate model names and parameters

3. **AWS Permission Errors**
   - Verify Cost Explorer permissions
   - Check cross-account role assumptions
   - Validate AWS profile configuration

### Performance Tuning

For better performance:
- Increase connection pool sizes for high-volume usage
- Adjust cache TTL based on data freshness requirements  
- Enable parallel processing for multi-account setups
- Use database backend for large-scale deployments

## Support

For additional help with configuration:
1. Check the main documentation in `USER_GUIDE.md`
2. Review example configurations in this directory
3. Use the built-in validation tools
4. Consult the troubleshooting section above