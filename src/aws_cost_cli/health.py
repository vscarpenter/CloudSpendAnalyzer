"""One-shot health check for the AWS Cost CLI.

Verifies the two things the CLI actually needs to run: that AWS credentials
resolve and that the cache directory is writable. No HTTP server, no metrics.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from .models import Config


def _check_aws_credentials(profile: Optional[str] = None) -> Dict[str, Any]:
    """Verify AWS credentials resolve via STS get_caller_identity."""
    try:
        from .aws_client import CredentialManager

        cred_manager = CredentialManager()
        if not cred_manager.validate_credentials(profile):
            return {"status": "unhealthy", "message": "AWS credentials invalid or missing"}

        identity = cred_manager.get_caller_identity(profile)
        return {
            "status": "healthy",
            "message": f"Authenticated as account {identity.get('Account', 'unknown')}",
        }
    except Exception as e:
        return {"status": "unhealthy", "message": f"AWS credential check failed: {e}"}


def _check_cache() -> Dict[str, Any]:
    """Verify the default cache directory is writable."""
    cache_dir = Path(os.path.expanduser("~/.aws-cost-cli/cache"))
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        test_file = cache_dir / ".health_check"
        test_file.write_text("ok")
        test_file.unlink()
        return {"status": "healthy", "message": f"Cache directory writable: {cache_dir}"}
    except Exception as e:
        return {"status": "unhealthy", "message": f"Cache directory not writable: {e}"}


def run_health_check(config: Optional[Config] = None) -> Dict[str, Any]:
    """Run a one-shot health check of AWS credentials and the cache directory.

    Args:
        config: Optional application configuration (reserved for future use).

    Returns:
        Dict with overall ``status`` ("healthy"/"unhealthy") and per-check
        results under ``checks``.
    """
    checks = {
        "aws_credentials": _check_aws_credentials(),
        "cache": _check_cache(),
    }

    overall = (
        "healthy"
        if all(check["status"] == "healthy" for check in checks.values())
        else "unhealthy"
    )

    return {"status": overall, "checks": checks}
