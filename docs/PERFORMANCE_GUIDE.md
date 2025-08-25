# Performance Optimization Guide

This guide covers the performance optimization features available in the AWS Cost Explorer CLI.

## Overview

The AWS Cost Explorer CLI includes several performance optimization features designed to improve query speed, reduce API calls, and minimize storage usage:

1. **Parallel Query Execution** - Automatically splits large queries into smaller chunks that run in parallel
2. **Query Result Pagination** - Handles large datasets by breaking them into manageable pages
3. **Compressed Caching** - Uses gzip compression to reduce cache storage requirements
4. **Performance Monitoring** - Tracks query performance metrics and provides insights

## Features

### Parallel Query Execution

Large time range queries are automatically split into smaller chunks and executed in parallel to improve performance.

**How it works:**
- Queries spanning more than 90 days (configurable) are automatically chunked
- Each chunk is executed in parallel using ThreadPoolExecutor
- Results are merged back into a single response
- Reduces total query time for large date ranges

**Configuration:**
```bash
# Enable/disable parallel execution
aws-cost-cli query "EC2 costs for 2025" --parallel
aws-cost-cli query "EC2 costs for 2025" --no-parallel

# Configure chunk size (default: 90 days)
aws-cost-cli query "EC2 costs for 2025" --max-chunk-days 60
```

### Query Result Pagination

For queries that return large datasets, results can be paginated to improve memory usage and response times.

**Features:**
- Automatic pagination based on data volume
- Configurable page sizes
- Page summary statistics
- Error handling for individual pages

### Compressed Caching

Cache compression reduces storage requirements while maintaining fast access to cached data.

**Benefits:**
- Reduces cache storage by 40-80% depending on data
- Maintains cache performance
- Automatic compression/decompression
- Compression statistics tracking

**Configuration:**
```bash
# Enable/disable compression
aws-cost-cli query "S3 costs last month" --compression
aws-cost-cli query "S3 costs last month" --no-compression
```

### Performance Monitoring

Track query performance and get insights into optimization effectiveness.

**Metrics tracked:**
- Query execution time
- API calls made
- Cache hit rates
- Compression ratios
- Parallel request counts

**Usage:**
```bash
# Show performance metrics after query
aws-cost-cli query "RDS costs this year" --performance-metrics

# View performance summary
aws-cost-cli performance

# View performance for last 48 hours
aws-cost-cli performance --hours 48

# Get JSON output
aws-cost-cli performance --format json
```

## Performance Optimization Examples

### Large Time Range Query
```bash
# This query will automatically use parallel execution
aws-cost-cli query "Show me all AWS costs for 2024" --performance-metrics

# Output will show:
# 🚀 Performance Metrics:
#    Processing time: 2500.0ms
#    API calls made: 4
#    Parallel requests: 4
#    Cache hit: No
#    Compression ratio: 0.65
#    Space saved: 35.0%
```

### Optimized Caching
```bash
# First run - fetches from API and caches with compression
aws-cost-cli query "EC2 costs last month" --compression --performance-metrics

# Second run - uses compressed cache
aws-cost-cli query "EC2 costs last month" --compression --performance-metrics
```

### Performance Monitoring
```bash
# View comprehensive performance statistics
aws-cost-cli performance

# Example output:
# 📊 Performance Summary (Last 24 hours)
#    Total queries: 15
#    Cache hit rate: 60.0%
#    Error rate: 0.0%
#    Total API calls: 25
#    Avg response time: 1250.5ms
#    95th percentile: 3200.0ms
#
# 💾 Cache Statistics
#    Total entries: 12
#    Valid entries: 10
#    Expired entries: 2
#    Cache size: 15.3 MB
#
# 🗜️  Compression Statistics
#    Compressed files: 8
#    Avg compression ratio: 0.62
#    Space saved: 38.0%
#    Total space saved: 9.2 MB
```

## Configuration Options

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--parallel` / `--no-parallel` | Enable/disable parallel execution | `--parallel` |
| `--compression` / `--no-compression` | Enable/disable cache compression | `--compression` |
| `--max-chunk-days INTEGER` | Maximum days per parallel chunk | `90` |
| `--performance-metrics` | Show performance metrics after query | `False` |

### Environment Variables

```bash
# Set default parallel execution
export AWS_COST_CLI_PARALLEL=true

# Set default compression
export AWS_COST_CLI_COMPRESSION=true

# Set default chunk size
export AWS_COST_CLI_MAX_CHUNK_DAYS=60
```

### Configuration File

```yaml
# ~/.aws-cost-cli/config.yaml
performance:
  parallel_execution: true
  cache_compression: true
  max_chunk_days: 90
  enable_monitoring: true
  max_parallel_workers: 5
```

## Best Practices

### When to Use Parallel Execution

**Use parallel execution for:**
- Queries spanning more than 3 months
- Annual or multi-year cost analysis
- Detailed daily granularity over long periods

**Avoid parallel execution for:**
- Simple monthly queries
- Real-time cost monitoring
- Queries with complex grouping that may hit API limits

### Cache Compression Guidelines

**Enable compression when:**
- Storage space is limited
- Working with large datasets
- Long-term cache retention is needed

**Consider disabling compression when:**
- CPU resources are limited
- Very frequent cache access patterns
- Small datasets where compression overhead isn't worth it

### Performance Monitoring

**Monitor performance to:**
- Identify slow queries that could benefit from optimization
- Track cache effectiveness
- Optimize chunk sizes for your usage patterns
- Identify API rate limiting issues

## Troubleshooting

### Common Issues

**Parallel execution fails:**
```bash
# Check AWS API rate limits
aws-cost-cli test --profile your-profile

# Reduce parallel workers
aws-cost-cli query "large query" --max-chunk-days 30
```

**Cache compression errors:**
```bash
# Check disk space
df -h ~/.aws-cost-cli/cache

# Clear corrupted cache
aws-cost-cli clear-cache
```

**Performance degradation:**
```bash
# Check performance metrics
aws-cost-cli performance

# Clean up expired cache
aws-cost-cli cleanup-cache
```

### Performance Tuning

**For faster queries:**
1. Enable parallel execution for large time ranges
2. Use appropriate cache TTL settings
3. Enable compression for better cache efficiency
4. Monitor and optimize chunk sizes

**For lower resource usage:**
1. Reduce max parallel workers
2. Increase chunk sizes to reduce API calls
3. Use shorter cache TTL to reduce storage
4. Disable compression if CPU is limited

## API Integration

### Using Performance Features Programmatically

```python
from aws_cost_cli.performance import PerformanceOptimizedClient
from aws_cost_cli.aws_client import AWSCostClient
from aws_cost_cli.cache_manager import CacheManager

# Create optimized client
cache_manager = CacheManager()
aws_client = AWSCostClient(cache_manager=cache_manager)
perf_client = aws_client.create_performance_optimized_client(
    enable_parallel=True,
    enable_compression=True,
    enable_monitoring=True
)

# Execute optimized query
result = perf_client.get_cost_and_usage_optimized(
    params=query_params,
    max_chunk_days=60
)

# Get performance summary
summary = perf_client.get_performance_summary()
```

## Metrics and Monitoring

### Available Metrics

- **Query Performance**: Execution time, API calls, cache hits
- **Compression Stats**: Ratios, space saved, file counts
- **Cache Statistics**: Hit rates, storage usage, entry counts
- **Error Tracking**: Failed queries, retry counts, error types

### Metric Export

```bash
# Export metrics to JSON
aws-cost-cli performance --format json > metrics.json

# Use in monitoring systems
aws-cost-cli performance --format json | jq '.performance_summary.cache_hit_rate'
```

## Advanced Configuration

### Custom Performance Profiles

Create performance profiles for different use cases:

```yaml
# ~/.aws-cost-cli/profiles/fast.yaml
performance:
  parallel_execution: true
  max_parallel_workers: 10
  max_chunk_days: 30
  cache_compression: true

# ~/.aws-cost-cli/profiles/conservative.yaml
performance:
  parallel_execution: false
  max_parallel_workers: 2
  max_chunk_days: 180
  cache_compression: false
```

### Integration with CI/CD

```bash
#!/bin/bash
# Performance monitoring script for CI/CD

# Run cost analysis with performance tracking
aws-cost-cli query "Monthly costs by service" --performance-metrics --format json > results.json

# Check performance thresholds
RESPONSE_TIME=$(jq '.metadata.processing_time_ms' results.json)
if (( $(echo "$RESPONSE_TIME > 5000" | bc -l) )); then
    echo "Warning: Query took ${RESPONSE_TIME}ms (threshold: 5000ms)"
fi

# Export metrics for monitoring
aws-cost-cli performance --format json > performance_metrics.json
```

This performance optimization system provides comprehensive tools for improving query speed, reducing resource usage, and monitoring system performance.