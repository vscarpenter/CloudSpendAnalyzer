"""Simple file-based cache for AWS cost data."""

import json
import hashlib
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Any, Dict
from config import Config


class SimpleCache:
    """Simple file-based cache with TTL."""
    
    def __init__(self, config: Config):
        self.enabled = config.cache_enabled
        self.ttl = config.cache_ttl
        self.cache_dir = Path(config.cache_dir).expanduser()
        
        # Create cache directory if enabled
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached data if it exists and is not expired."""
        if not self.enabled:
            return None
        
        cache_file = self._get_cache_file(key)
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r') as f:
                cached = json.load(f)
            
            # Check if cache is expired
            cached_time = datetime.fromisoformat(cached['timestamp'])
            if datetime.now() - cached_time > timedelta(seconds=self.ttl):
                # Cache expired, delete it
                cache_file.unlink()
                return None
            
            return cached['data']
            
        except (json.JSONDecodeError, KeyError, ValueError):
            # Invalid cache file, delete it
            cache_file.unlink()
            return None
    
    def set(self, key: str, data: Dict[str, Any]) -> None:
        """Store data in cache."""
        if not self.enabled:
            return
        
        cache_file = self._get_cache_file(key)
        
        # Convert Decimal to float for JSON serialization
        data_json = self._serialize_for_cache(data)
        
        cached = {
            'timestamp': datetime.now().isoformat(),
            'data': data_json
        }
        
        with open(cache_file, 'w') as f:
            json.dump(cached, f, indent=2)
    
    def clear(self) -> None:
        """Clear all cached data."""
        if not self.enabled:
            return
        
        for cache_file in self.cache_dir.glob('*.json'):
            cache_file.unlink()
    
    def _get_cache_file(self, key: str) -> Path:
        """Get the cache file path for a given key."""
        # Create a hash of the key to use as filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.json"
    
    def _serialize_for_cache(self, obj: Any) -> Any:
        """Recursively convert Decimal to float for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._serialize_for_cache(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_for_cache(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            # Handle dataclasses and objects
            return self._serialize_for_cache(obj.__dict__)
        elif type(obj).__name__ == 'Decimal':
            return float(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            return obj


def get_cache_key(query: str, profile: str = "default") -> str:
    """Generate a cache key from query and profile."""
    return f"{profile}:{query}"
