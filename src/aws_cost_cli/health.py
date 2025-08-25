"""Health check and monitoring endpoints for AWS Cost CLI."""

import json
import time
import psutil
import platform
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path

from .models import Config
from .exceptions import AWSCostCLIError


@dataclass
class HealthStatus:
    """Health check status result."""
    status: str  # "healthy", "unhealthy", "degraded"
    timestamp: datetime
    checks: Dict[str, Dict[str, Any]]
    summary: Dict[str, Any]


@dataclass
class SystemMetrics:
    """System resource metrics."""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    memory_available_gb: float
    disk_available_gb: float
    uptime_seconds: float


class HealthChecker:
    """Comprehensive health checking for the AWS Cost CLI application."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize health checker.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.start_time = time.time()
        
    def check_health(self, detailed: bool = False) -> HealthStatus:
        """
        Perform comprehensive health check.
        
        Args:
            detailed: Whether to include detailed system metrics
            
        Returns:
            HealthStatus with overall health and individual check results
        """
        checks = {}
        overall_status = "healthy"
        
        # Core system checks
        checks["system"] = self._check_system_health()
        if checks["system"]["status"] != "healthy":
            overall_status = "degraded"
            
        # AWS connectivity check
        checks["aws"] = self._check_aws_connectivity()
        if checks["aws"]["status"] == "unhealthy":
            overall_status = "unhealthy"
        elif checks["aws"]["status"] == "degraded":
            overall_status = "degraded"
            
        # Cache system check
        checks["cache"] = self._check_cache_system()
        if checks["cache"]["status"] == "unhealthy":
            overall_status = "degraded"  # Cache issues are degraded, not unhealthy
            
        # Database check (if enabled)
        if self.config and getattr(self.config, 'database', {}).get('enabled', False):
            checks["database"] = self._check_database_connectivity()
            if checks["database"]["status"] == "unhealthy":
                overall_status = "unhealthy"
        
        # LLM provider check
        checks["llm"] = self._check_llm_connectivity()
        if checks["llm"]["status"] == "unhealthy":
            overall_status = "degraded"  # LLM issues are degraded, not critical
            
        # Detailed system metrics (if requested)
        if detailed:
            checks["metrics"] = self._get_system_metrics()
            
        # Generate summary
        summary = self._generate_summary(checks, overall_status)
        
        return HealthStatus(
            status=overall_status,
            timestamp=datetime.now(timezone.utc),
            checks=checks,
            summary=summary
        )
    
    def check_readiness(self) -> Dict[str, Any]:
        """
        Check if the application is ready to serve requests.
        This is a lighter check than full health check.
        """
        ready = True
        checks = {}
        
        # Check if AWS credentials are available
        try:
            from .aws_client import CredentialManager
            cred_manager = CredentialManager()
            if not cred_manager.validate_credentials():
                ready = False
                checks["aws_credentials"] = {
                    "status": "not_ready",
                    "message": "AWS credentials not available"
                }
            else:
                checks["aws_credentials"] = {
                    "status": "ready",
                    "message": "AWS credentials available"
                }
        except Exception as e:
            ready = False
            checks["aws_credentials"] = {
                "status": "error",
                "message": f"Error checking AWS credentials: {str(e)}"
            }
            
        # Check if cache directory is accessible
        try:
            cache_dir = Path("~/.aws-cost-cli/cache").expanduser()
            cache_dir.mkdir(parents=True, exist_ok=True)
            test_file = cache_dir / "readiness_test"
            test_file.write_text("test")
            test_file.unlink()
            checks["cache_access"] = {
                "status": "ready",
                "message": "Cache directory accessible"
            }
        except Exception as e:
            ready = False
            checks["cache_access"] = {
                "status": "not_ready",
                "message": f"Cache directory not accessible: {str(e)}"
            }
        
        return {
            "ready": ready,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": checks
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get Prometheus-style metrics for monitoring.
        """
        metrics = {}
        
        # System metrics
        system_metrics = self._get_system_metrics_dict()
        for metric_name, value in system_metrics.items():
            metrics[f"aws_cost_cli_system_{metric_name}"] = value
            
        # Application metrics
        metrics["aws_cost_cli_uptime_seconds"] = time.time() - self.start_time
        
        # Health check metrics
        health = self.check_health()
        metrics["aws_cost_cli_health_status"] = 1 if health.status == "healthy" else 0
        metrics["aws_cost_cli_degraded_status"] = 1 if health.status == "degraded" else 0
        
        # Component health metrics
        for component, check in health.checks.items():
            if isinstance(check, dict) and "status" in check:
                status_value = 1 if check["status"] == "healthy" else 0
                metrics[f"aws_cost_cli_{component}_health"] = status_value
        
        return metrics
    
    def _check_system_health(self) -> Dict[str, Any]:
        """Check system resource health."""
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Define thresholds
            cpu_warning_threshold = 80
            cpu_critical_threshold = 95
            memory_warning_threshold = 80
            memory_critical_threshold = 95
            disk_warning_threshold = 85
            disk_critical_threshold = 95
            
            status = "healthy"
            warnings = []
            errors = []
            
            # Check CPU
            if cpu_percent > cpu_critical_threshold:
                status = "unhealthy"
                errors.append(f"CPU usage critical: {cpu_percent:.1f}%")
            elif cpu_percent > cpu_warning_threshold:
                status = "degraded"
                warnings.append(f"CPU usage high: {cpu_percent:.1f}%")
                
            # Check memory
            if memory.percent > memory_critical_threshold:
                status = "unhealthy"
                errors.append(f"Memory usage critical: {memory.percent:.1f}%")
            elif memory.percent > memory_warning_threshold:
                if status == "healthy":
                    status = "degraded"
                warnings.append(f"Memory usage high: {memory.percent:.1f}%")
                
            # Check disk
            if disk.percent > disk_critical_threshold:
                status = "unhealthy" 
                errors.append(f"Disk usage critical: {disk.percent:.1f}%")
            elif disk.percent > disk_warning_threshold:
                if status == "healthy":
                    status = "degraded"
                warnings.append(f"Disk usage high: {disk.percent:.1f}%")
            
            return {
                "status": status,
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent,
                "warnings": warnings,
                "errors": errors,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": f"Failed to check system health: {str(e)}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    def _check_aws_connectivity(self) -> Dict[str, Any]:
        """Check AWS API connectivity."""
        try:
            from .aws_client import CredentialManager
            
            cred_manager = CredentialManager()
            
            # Check if credentials are valid
            if not cred_manager.validate_credentials():
                return {
                    "status": "unhealthy",
                    "error": "AWS credentials invalid or missing",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            
            # Try to get caller identity
            start_time = time.time()
            identity = cred_manager.get_caller_identity()
            response_time = time.time() - start_time
            
            return {
                "status": "healthy",
                "account_id": identity.get("Account", "unknown"),
                "user_id": identity.get("UserId", "unknown"),
                "response_time_ms": round(response_time * 1000, 2),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy", 
                "error": f"AWS connectivity failed: {str(e)}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    def _check_cache_system(self) -> Dict[str, Any]:
        """Check cache system health."""
        try:
            cache_dir = Path("~/.aws-cost-cli/cache").expanduser()
            
            # Check if cache directory exists and is writable
            if not cache_dir.exists():
                cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Test write/read
            test_file = cache_dir / "health_check_test"
            test_data = {"test": True, "timestamp": time.time()}
            
            start_time = time.time()
            test_file.write_text(json.dumps(test_data))
            read_data = json.loads(test_file.read_text())
            test_file.unlink()
            operation_time = time.time() - start_time
            
            # Check cache directory size
            total_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
            total_size_mb = total_size / (1024 * 1024)
            
            status = "healthy"
            warnings = []
            
            if total_size_mb > 1000:  # > 1GB
                status = "degraded"
                warnings.append(f"Cache size large: {total_size_mb:.1f}MB")
                
            return {
                "status": status,
                "cache_dir": str(cache_dir),
                "size_mb": round(total_size_mb, 2),
                "operation_time_ms": round(operation_time * 1000, 2),
                "warnings": warnings,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": f"Cache system failed: {str(e)}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    def _check_database_connectivity(self) -> Dict[str, Any]:
        """Check database connectivity (if database is configured)."""
        try:
            # This would need to be implemented based on actual database integration
            # For now, return a placeholder
            return {
                "status": "healthy",
                "message": "Database connectivity check not implemented yet",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": f"Database connectivity failed: {str(e)}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    def _check_llm_connectivity(self) -> Dict[str, Any]:
        """Check LLM provider connectivity."""
        try:
            # Basic check for API keys
            import os
            
            providers_available = []
            providers_missing = []
            
            # Check OpenAI
            if os.getenv('OPENAI_API_KEY'):
                providers_available.append('openai')
            else:
                providers_missing.append('openai')
                
            # Check Anthropic
            if os.getenv('ANTHROPIC_API_KEY'):
                providers_available.append('anthropic')
            else:
                providers_missing.append('anthropic')
            
            if not providers_available:
                return {
                    "status": "unhealthy",
                    "error": "No LLM provider API keys available",
                    "missing_providers": providers_missing,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            
            return {
                "status": "healthy",
                "available_providers": providers_available,
                "missing_providers": providers_missing,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": f"LLM connectivity check failed: {str(e)}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get detailed system metrics."""
        try:
            metrics = self._get_system_metrics_dict()
            return {
                "status": "healthy",
                "metrics": metrics,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Failed to get system metrics: {str(e)}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    def _get_system_metrics_dict(self) -> Dict[str, float]:
        """Get system metrics as a dictionary."""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available_gb": memory.available / (1024**3),
            "disk_percent": disk.percent,
            "disk_available_gb": disk.free / (1024**3),
            "uptime_seconds": time.time() - self.start_time
        }
    
    def _generate_summary(self, checks: Dict[str, Dict[str, Any]], overall_status: str) -> Dict[str, Any]:
        """Generate health check summary."""
        healthy_count = sum(1 for check in checks.values() 
                          if isinstance(check, dict) and check.get("status") == "healthy")
        degraded_count = sum(1 for check in checks.values() 
                           if isinstance(check, dict) and check.get("status") == "degraded")
        unhealthy_count = sum(1 for check in checks.values() 
                            if isinstance(check, dict) and check.get("status") == "unhealthy")
        
        return {
            "overall_status": overall_status,
            "total_checks": len(checks),
            "healthy_checks": healthy_count,
            "degraded_checks": degraded_count,
            "unhealthy_checks": unhealthy_count,
            "uptime_seconds": time.time() - self.start_time,
            "version": "0.1.0",  # This should come from package metadata
            "python_version": platform.python_version(),
            "platform": platform.platform()
        }


def create_health_check_server(host: str = "0.0.0.0", port: int = 8081, config: Optional[Config] = None):
    """
    Create a simple HTTP server for health checks.
    This can be used for container health checks and monitoring.
    """
    try:
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import urllib.parse as urlparse
        
        health_checker = HealthChecker(config)
        
        class HealthHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                parsed_path = urlparse.urlparse(self.path)
                path = parsed_path.path
                
                try:
                    if path == "/health":
                        # Basic health check
                        health = health_checker.check_health()
                        status_code = 200 if health.status == "healthy" else 503
                        response = {
                            "status": health.status,
                            "timestamp": health.timestamp.isoformat()
                        }
                        
                    elif path == "/health/detailed":
                        # Detailed health check
                        health = health_checker.check_health(detailed=True)
                        status_code = 200 if health.status == "healthy" else 503
                        response = asdict(health)
                        response["timestamp"] = response["timestamp"].isoformat()
                        
                    elif path == "/ready":
                        # Readiness check
                        readiness = health_checker.check_readiness()
                        status_code = 200 if readiness["ready"] else 503
                        response = readiness
                        
                    elif path == "/metrics":
                        # Prometheus metrics
                        metrics = health_checker.get_metrics()
                        status_code = 200
                        
                        # Convert to Prometheus format
                        prometheus_metrics = []
                        for metric_name, value in metrics.items():
                            prometheus_metrics.append(f"{metric_name} {value}")
                        
                        self.send_response(status_code)
                        self.send_header('Content-type', 'text/plain')
                        self.end_headers()
                        self.wfile.write('\n'.join(prometheus_metrics).encode())
                        return
                        
                    else:
                        status_code = 404
                        response = {"error": "Not found"}
                    
                    self.send_response(status_code)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(response, indent=2).encode())
                    
                except Exception as e:
                    self.send_response(500)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    error_response = {"error": str(e)}
                    self.wfile.write(json.dumps(error_response).encode())
            
            def log_message(self, format, *args):
                # Suppress default logging
                pass
        
        server = HTTPServer((host, port), HealthHandler)
        return server
        
    except ImportError:
        raise AWSCostCLIError("HTTP server dependencies not available")