# Security Policy

## Supported Versions

We provide security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in the AWS Cost Explorer CLI, please report it responsibly:

### How to Report

1. **Do NOT create a public GitHub issue** for security vulnerabilities
2. **Email security concerns** to: vinny@vinny.dev
3. **Include the following information:**
   - Description of the vulnerability
   - Steps to reproduce the issue
   - Potential impact assessment
   - Suggested fix (if available)

### What to Expect

- **Acknowledgment**: We will acknowledge receipt within 48 hours
- **Assessment**: We will assess the vulnerability within 5 business days
- **Updates**: We will provide regular updates on our progress
- **Resolution**: We aim to resolve critical vulnerabilities within 30 days

### Security Considerations

The AWS Cost Explorer CLI handles sensitive information including:

- **AWS Credentials**: Access keys, session tokens, and profile information
- **Cost Data**: Financial information about AWS usage and spending
- **API Keys**: LLM provider API keys for query processing

### Security Best Practices

When using the AWS Cost Explorer CLI:

#### Credential Management
- **Use AWS profiles** instead of hardcoded credentials
- **Rotate API keys regularly** for LLM providers
- **Use environment variables** for sensitive configuration
- **Enable MFA** on AWS accounts when possible
- **Follow least privilege principle** for IAM permissions

#### Data Protection
- **Review query logs** periodically for sensitive information
- **Use local LLM providers** (Ollama) for sensitive environments
- **Clear cache regularly** if handling sensitive cost data
- **Set appropriate file permissions** on configuration files

#### Network Security
- **Use HTTPS** for all API communications
- **Validate SSL certificates** for external API calls
- **Consider network restrictions** for LLM API access
- **Monitor API usage** for unusual patterns

### Required AWS Permissions

The CLI requires minimal AWS permissions:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "ce:GetCostAndUsage",
                "ce:GetDimensionValues",
                "ce:GetReservationCoverage",
                "ce:GetReservationPurchaseRecommendation",
                "ce:GetReservationUtilization",
                "ce:GetUsageReport"
            ],
            "Resource": "*"
        }
    ]
}
```

### LLM Provider Security

#### OpenAI
- Store API keys securely using environment variables
- Monitor API usage and billing
- Review OpenAI's data usage policies

#### Anthropic
- Use environment variables for API key storage
- Monitor Claude API usage
- Review Anthropic's privacy policies

#### AWS Bedrock
- Use IAM roles when possible instead of access keys
- Enable CloudTrail logging for Bedrock API calls
- Review AWS Bedrock security best practices

#### Ollama (Local)
- Keep Ollama updated to the latest version
- Secure the Ollama service endpoint
- Monitor local resource usage

### Data Privacy

The CLI processes cost data that may be sensitive:

- **Cost information** is cached locally with TTL expiration
- **Query patterns** may reveal business information
- **LLM interactions** may include cost data in prompts

#### Privacy Controls
- **Local caching only** - no data sent to third parties except LLM APIs
- **Configurable cache TTL** to limit data retention
- **Cache clearing commands** for immediate data removal
- **Opt-out options** for external LLM services

### Vulnerability Disclosure Timeline

1. **Day 0**: Vulnerability reported
2. **Day 1-2**: Acknowledgment sent to reporter
3. **Day 3-7**: Initial assessment and triage
4. **Day 8-30**: Development and testing of fix
5. **Day 31**: Public disclosure and release of patched version

### Security Updates

Security updates will be:

- **Released as patch versions** (e.g., 0.1.1)
- **Documented in CHANGELOG.md** with security advisory
- **Announced via GitHub releases** with security labels
- **Communicated to users** through appropriate channels

### Acknowledgments

We appreciate security researchers who help improve the security of our project. Responsible disclosure helps protect all users.

### Contact

For security-related questions or concerns:
- **Email**: vinny@vinny.dev
- **Subject**: [SECURITY] AWS Cost Explorer CLI

Thank you for helping keep the AWS Cost Explorer CLI secure!