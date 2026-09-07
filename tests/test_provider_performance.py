"""Tests for provider performance monitoring functionality."""

import json
import time
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from src.aws_cost_cli.query_processor import (
    ProviderMetrics,
    ProviderHealthCheck,
    ProviderPerformanceMonitor,
    get_performance_monitor,
    LLMProvider,
)
from src.aws_cost_cli.exceptions import LLMProviderError, NetworkError


class MockProvider(LLMProvider):
    """Mock LLM provider for testing."""
    
    def __init__(self, should_fail=False, timeout=None):
        super().__init__(timeout)
        self.should_fail = should_fail
        self.call_count = 0
    
    def parse_query(self, query: str):
        self.call_count += 1
        if self.should_fail:
            raise LLMProviderError("Mock provider error", provider="mock")
        return {"service": "EC2", "start_date": "2025-01-01"}
    
    def is_available(self):
        return not self.should_fail


class TestProviderMetrics:
    """Test ProviderMetrics dataclass."""
    
    def test_provider_metrics_initialization(self):
        """Test ProviderMetrics initialization with defaults."""
        metrics = ProviderMetrics(provider_name="test")
        
        assert metrics.provider_name == "test"
        assert metrics.request_count == 0
        assert metrics.success_count == 0
        assert metrics.error_count == 0
        assert metrics.total_response_time == 0.0
        assert metrics.min_response_time is None
        assert metrics.max_response_time is None
        assert metrics.last_success is None
        assert metrics.last_error is None
        assert metrics.last_error_message is None
        assert metrics.consecutive_errors == 0
        assert metrics.health_status == "unknown"
        assert metrics.timeout_count == 0
    
    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        metrics = ProviderMetrics(
            provider_name="test",
            request_count=10,
            success_count=8,
            error_count=2
        )
        
        assert metrics.success_rate == 80.0
    
    def test_error_rate_calculation(self):
        """Test error rate calculation."""
        metrics = ProviderMetrics(
            provider_name="test",
            request_count=10,
            success_count=8,
            error_count=2
        )
        
        assert metrics.error_rate == 20.0
    
    def test_average_response_time_calculation(self):
        """Test average response time calculation."""
        metrics = ProviderMetrics(
            provider_name="test",
            success_count=4,
            total_response_time=400.0
        )
        
        assert metrics.average_response_time == 100.0
    
    def test_zero_division_handling(self):
        """Test handling of zero division in calculations."""
        metrics = ProviderMetrics(provider_name="test")
        
        assert metrics.success_rate == 0.0
        assert metrics.error_rate == 0.0
        assert metrics.average_response_time == 0.0


class TestProviderHealthCheck:
    """Test ProviderHealthCheck dataclass."""
    
    def test_health_check_initialization(self):
        """Test ProviderHealthCheck initialization."""
        now = datetime.now()
        health_check = ProviderHealthCheck(
            provider_name="test",
            is_healthy=True,
            response_time_ms=150.0,
            error_message=None,
            checked_at=now,
            availability_status="available"
        )
        
        assert health_check.provider_name == "test"
        assert health_check.is_healthy is True
        assert health_check.response_time_ms == 150.0
        assert health_check.error_message is None
        assert health_check.checked_at == now
        assert health_check.availability_status == "available"


class TestProviderPerformanceMonitor:
    """Test ProviderPerformanceMonitor class."""
    
    def test_monitor_initialization(self):
        """Test monitor initialization."""
        monitor = ProviderPerformanceMonitor()
        
        assert monitor.metrics_file is None
        assert monitor.provider_metrics == {}
    
    def test_monitor_initialization_with_file(self):
        """Test monitor initialization with metrics file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            metrics_file = f.name
        
        try:
            monitor = ProviderPerformanceMonitor(metrics_file)
            assert monitor.metrics_file == metrics_file
        finally:
            Path(metrics_file).unlink(missing_ok=True)
    
    def test_record_request_success(self):
        """Test recording successful requests."""
        monitor = ProviderPerformanceMonitor()
        start_time = time.time() - 0.1  # 100ms ago
        
        monitor.record_request_success("openai", start_time, 150.0)
        
        metrics = monitor.get_provider_metrics("openai")
        assert metrics is not None
        assert metrics.provider_name == "openai"
        assert metrics.request_count == 1
        assert metrics.success_count == 1
        assert metrics.error_count == 0
        assert metrics.total_response_time == 150.0
        assert metrics.min_response_time == 150.0
        assert metrics.max_response_time == 150.0
        assert metrics.consecutive_errors == 0
        assert metrics.last_success is not None
        assert metrics.health_status == "healthy"
    
    def test_record_request_error(self):
        """Test recording failed requests."""
        monitor = ProviderPerformanceMonitor()
        start_time = time.time() - 0.1  # 100ms ago
        
        monitor.record_request_error("openai", start_time, "API error", False)
        
        metrics = monitor.get_provider_metrics("openai")
        assert metrics is not None
        assert metrics.provider_name == "openai"
        assert metrics.request_count == 1
        assert metrics.success_count == 0
        assert metrics.error_count == 1
        assert metrics.consecutive_errors == 1
        assert metrics.last_error is not None
        assert metrics.last_error_message == "API error"
        assert metrics.timeout_count == 0
        assert metrics.health_status == "degraded"  # Single error makes it degraded
    
    def test_record_timeout_error(self):
        """Test recording timeout errors."""
        monitor = ProviderPerformanceMonitor()
        start_time = time.time() - 0.1
        
        monitor.record_request_error("openai", start_time, "Request timed out", True)
        
        metrics = monitor.get_provider_metrics("openai")
        assert metrics is not None
        assert metrics.timeout_count == 1
    
    def test_health_status_updates(self):
        """Test health status updates based on performance."""
        monitor = ProviderPerformanceMonitor()
        start_time = time.time() - 0.1
        
        # Record multiple successes - should be healthy
        for _ in range(10):
            monitor.record_request_success("openai", start_time, 100.0)
        
        metrics = monitor.get_provider_metrics("openai")
        assert metrics.health_status == "healthy"
        
        # Record some errors - should be degraded
        for _ in range(3):
            monitor.record_request_error("openai", start_time, "Error", False)
        
        metrics = monitor.get_provider_metrics("openai")
        assert metrics.health_status == "degraded"
        
        # Record many consecutive errors - should be unhealthy
        for _ in range(5):
            monitor.record_request_error("openai", start_time, "Error", False)
        
        metrics = monitor.get_provider_metrics("openai")
        assert metrics.health_status == "unhealthy"
    
    def test_min_max_response_times(self):
        """Test min/max response time tracking."""
        monitor = ProviderPerformanceMonitor()
        start_time = time.time() - 0.1
        
        # Record requests with different response times
        monitor.record_request_success("openai", start_time, 100.0)
        monitor.record_request_success("openai", start_time, 200.0)
        monitor.record_request_success("openai", start_time, 50.0)
        
        metrics = monitor.get_provider_metrics("openai")
        assert metrics.min_response_time == 50.0
        assert metrics.max_response_time == 200.0
        assert metrics.average_response_time == (100.0 + 200.0 + 50.0) / 3
    
    def test_get_all_provider_metrics(self):
        """Test getting metrics for all providers."""
        monitor = ProviderPerformanceMonitor()
        start_time = time.time() - 0.1
        
        monitor.record_request_success("openai", start_time, 100.0)
        monitor.record_request_success("anthropic", start_time, 150.0)
        
        all_metrics = monitor.get_all_provider_metrics()
        assert len(all_metrics) == 2
        assert "openai" in all_metrics
        assert "anthropic" in all_metrics
    
    def test_performance_summary(self):
        """Test performance summary generation."""
        monitor = ProviderPerformanceMonitor()
        start_time = time.time() - 0.1
        
        # Add some test data
        monitor.record_request_success("openai", start_time, 100.0)
        monitor.record_request_success("openai", start_time, 200.0)
        monitor.record_request_error("anthropic", start_time, "Error", False)
        
        summary = monitor.get_performance_summary(24)
        
        assert summary["period_hours"] == 24
        assert "providers" in summary
        assert "overall" in summary
        
        # Check provider-specific data
        assert "openai" in summary["providers"]
        assert "anthropic" in summary["providers"]
        
        openai_metrics = summary["providers"]["openai"]
        assert openai_metrics["request_count"] == 2
        assert openai_metrics["success_count"] == 2
        assert openai_metrics["error_count"] == 0
        assert openai_metrics["success_rate"] == 100.0
        
        anthropic_metrics = summary["providers"]["anthropic"]
        assert anthropic_metrics["request_count"] == 1
        assert anthropic_metrics["success_count"] == 0
        assert anthropic_metrics["error_count"] == 1
        assert anthropic_metrics["success_rate"] == 0.0
        
        # Check overall stats
        overall = summary["overall"]
        assert overall["total_requests"] == 3
        assert overall["total_successes"] == 2
        assert overall["total_errors"] == 1
        assert overall["healthy_providers"] == 1  # openai
        assert overall["unhealthy_providers"] == 0
    
    def test_reset_metrics(self):
        """Test resetting metrics."""
        monitor = ProviderPerformanceMonitor()
        start_time = time.time() - 0.1
        
        # Add some data
        monitor.record_request_success("openai", start_time, 100.0)
        monitor.record_request_success("anthropic", start_time, 150.0)
        
        # Reset specific provider
        monitor.reset_metrics("openai")
        
        openai_metrics = monitor.get_provider_metrics("openai")
        anthropic_metrics = monitor.get_provider_metrics("anthropic")
        
        assert openai_metrics.request_count == 0
        assert anthropic_metrics.request_count == 1
        
        # Reset all providers
        monitor.reset_metrics()
        
        all_metrics = monitor.get_all_provider_metrics()
        assert len(all_metrics) == 0
    
    def test_metrics_persistence(self):
        """Test metrics persistence to file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            metrics_file = f.name
        
        try:
            monitor = ProviderPerformanceMonitor(metrics_file)
            start_time = time.time() - 0.1
            
            # Record some data
            monitor.record_request_success("openai", start_time, 100.0)
            
            # Check file was created and contains data
            assert Path(metrics_file).exists()
            
            with open(metrics_file, 'r') as f:
                data = json.load(f)
                assert "openai" in data
                assert data["openai"]["request_count"] == 1
        
        finally:
            Path(metrics_file).unlink(missing_ok=True)
    
    def test_metrics_loading(self):
        """Test loading metrics from file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            metrics_file = f.name
            
            # Write test data
            test_data = {
                "openai": {
                    "provider_name": "openai",
                    "request_count": 5,
                    "success_count": 4,
                    "error_count": 1,
                    "total_response_time": 500.0,
                    "min_response_time": 80.0,
                    "max_response_time": 150.0,
                    "last_success": "2025-01-01T12:00:00",
                    "last_error": None,
                    "last_error_message": None,
                    "consecutive_errors": 0,
                    "health_status": "healthy",
                    "timeout_count": 0
                }
            }
            json.dump(test_data, f)
        
        try:
            monitor = ProviderPerformanceMonitor(metrics_file)
            
            metrics = monitor.get_provider_metrics("openai")
            assert metrics is not None
            assert metrics.request_count == 5
            assert metrics.success_count == 4
            assert metrics.error_count == 1
            assert metrics.health_status == "healthy"
            assert metrics.last_success is not None
        
        finally:
            Path(metrics_file).unlink(missing_ok=True)


class TestProviderHealthChecks:
    """Test provider health check functionality."""
    
    def test_health_check_healthy_provider(self):
        """Test health check on healthy provider."""
        monitor = ProviderPerformanceMonitor()
        provider = MockProvider(should_fail=False)
        
        health_check = monitor.check_provider_health(provider, timeout=5.0)
        
        assert health_check.provider_name == "mock"
        assert health_check.is_healthy is True
        assert health_check.response_time_ms is not None
        assert health_check.response_time_ms > 0
        assert health_check.error_message is None
        assert health_check.availability_status == "available"
    
    def test_health_check_unhealthy_provider(self):
        """Test health check on unhealthy provider."""
        monitor = ProviderPerformanceMonitor()
        provider = MockProvider(should_fail=True)
        
        health_check = monitor.check_provider_health(provider, timeout=5.0)
        
        assert health_check.provider_name == "mock"
        assert health_check.is_healthy is False
        assert health_check.error_message == "Provider not available"
        assert health_check.availability_status == "unavailable"
    
    def test_health_check_timeout(self):
        """Test health check timeout handling."""
        monitor = ProviderPerformanceMonitor()
        provider = MockProvider(should_fail=False)
        
        # Test normal health check (should be healthy)
        health_check = monitor.check_provider_health(provider, timeout=5.0)
        assert health_check.is_healthy is True


class TestLLMProviderEnhancements:
    """Test LLM provider performance monitoring integration."""
    
    def test_provider_with_monitoring(self):
        """Test provider with performance monitoring."""
        # Create a fresh provider to avoid interference from other tests
        provider = MockProvider(should_fail=False, timeout=30.0)
        
        # Reset the global monitor to avoid interference
        from src.aws_cost_cli.query_processor import ProviderPerformanceMonitor
        provider._performance_monitor = ProviderPerformanceMonitor()
        
        # Test successful query with monitoring
        result = provider.parse_query_with_monitoring("test query")
        assert result == {"service": "EC2", "start_date": "2025-01-01"}
        
        # Check metrics were recorded
        metrics = provider.get_performance_metrics()
        assert metrics is not None
        assert metrics.provider_name == "mock"
        assert metrics.success_count == 1
        assert metrics.error_count == 0
    
    def test_provider_error_monitoring(self):
        """Test provider error monitoring."""
        # Create a fresh provider to avoid interference from other tests
        provider = MockProvider(should_fail=True, timeout=30.0)
        
        # Reset the global monitor to avoid interference
        from src.aws_cost_cli.query_processor import ProviderPerformanceMonitor
        provider._performance_monitor = ProviderPerformanceMonitor()
        
        # Test failed query with monitoring
        with pytest.raises(LLMProviderError):
            provider.parse_query_with_monitoring("test query")
        
        # Check error metrics were recorded
        metrics = provider.get_performance_metrics()
        assert metrics is not None
        assert metrics.success_count == 0
        assert metrics.error_count == 1
        assert metrics.consecutive_errors == 1
    
    def test_provider_health_check_method(self):
        """Test provider health check method."""
        provider = MockProvider(should_fail=False, timeout=30.0)
        
        health_check = provider.check_health()
        
        assert health_check.provider_name == "mock"
        assert health_check.is_healthy is True
        assert health_check.response_time_ms is not None


class TestGlobalPerformanceMonitor:
    """Test global performance monitor instance."""
    
    def test_get_performance_monitor_singleton(self):
        """Test global performance monitor singleton."""
        monitor1 = get_performance_monitor()
        monitor2 = get_performance_monitor()
        
        assert monitor1 is monitor2
        assert isinstance(monitor1, ProviderPerformanceMonitor)
    
    @patch('src.aws_cost_cli.query_processor.Path.home')
    def test_performance_monitor_default_location(self, mock_home):
        """Test performance monitor default file location."""
        mock_home.return_value = Path("/home/user")
        
        # Reset the global monitor
        import src.aws_cost_cli.query_processor
        src.aws_cost_cli.query_processor._performance_monitor = None
        
        monitor = get_performance_monitor()
        expected_path = "/home/user/.aws-cost-cli/provider_metrics.json"
        assert monitor.metrics_file == expected_path


class TestProviderTimeoutConfiguration:
    """Test provider timeout configuration."""
    
    def test_provider_timeout_initialization(self):
        """Test provider timeout initialization."""
        provider = MockProvider(timeout=45.0)
        assert provider.timeout == 45.0
    
    def test_provider_default_timeout(self):
        """Test provider default timeout."""
        provider = MockProvider()
        assert provider.timeout == 30.0