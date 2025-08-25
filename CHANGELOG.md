# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-24

### Added
- Initial release of AWS Cost Explorer CLI
- Natural language query interface for AWS cost data
- Multi-LLM provider support (OpenAI, Anthropic, AWS Bedrock, Ollama)
- AWS credential integration with profile support
- Intelligent caching with TTL for performance optimization
- Rich terminal output formatting with multiple format options
- Comprehensive error handling and user guidance
- Performance monitoring and metrics
- Data export capabilities (CSV, JSON, Excel)
- Cost optimization recommendations and analysis
- Interactive query builder with templates
- Trend analysis and forecasting capabilities
- Advanced query features with parallel execution
- Comprehensive test suite with 95%+ coverage
- Detailed documentation and user guides

### Features
- **Query Processing**: Natural language parsing with LLM integration
- **AWS Integration**: Full Cost Explorer API support with boto3
- **Caching**: File-based caching with compression and TTL management
- **Performance**: Parallel query execution and optimization metrics
- **Export**: Multiple export formats for cost data analysis
- **Optimization**: Cost analysis and recommendation engine
- **Interactive**: Guided query building and template system
- **Monitoring**: Performance tracking and query analytics

### Supported AWS Services
- Amazon EC2 (Elastic Compute Cloud)
- Amazon S3 (Simple Storage Service)
- Amazon RDS (Relational Database Service)
- AWS Lambda
- Amazon CloudFront
- Amazon EKS (Elastic Kubernetes Service)
- Amazon Redshift
- Amazon ElastiCache
- Amazon API Gateway
- Amazon DynamoDB
- And many more AWS services

### LLM Providers
- OpenAI GPT models
- Anthropic Claude models
- AWS Bedrock (Claude, Titan, AI21)
- Ollama (local models)

[0.1.0]: https://github.com/vinnycarpenter/aws-cost-explorer-cli/releases/tag/v0.1.0