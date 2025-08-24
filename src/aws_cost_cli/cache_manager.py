"""Cache management system for AWS Cost CLI."""

import json
import hashlib
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import asdict
from datetime import datetime, timezone

from .models import CostData, QueryParameters
from .exceptions import CacheError


class CacheManager:
    """Manages file-based caching with TTL for cost data."""
    
    def __init__(self, cache_dir: Optional[str] = None, default_ttl: int = 3600):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory for cache files. Defaults to ~/.aws-cost-cli/cache
            default_ttl: Default TTL in seconds (default: 1 hour)
        """
        self.default_ttl = default_ttl
        
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.aws-cost-cli/cache")
        
        self.cache_dir = Path(cache_dir)
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise CacheError(f"Failed to create cache directory {cache_dir}: {e}")
    
    def _generate_query_hash(self, params: QueryParameters, profile: Optional[str] = None) -> str:
        """
        Generate a unique hash for query parameters.
        
        Args:
            params: Query parameters to hash
            profile: AWS profile name to include in hash
            
        Returns:
            SHA256 hash string
        """
        # Convert params to dict for consistent hashing
        params_dict = asdict(params)
        
        # Handle datetime objects in time_period
        if params_dict.get('time_period'):
            time_period = params_dict['time_period']
            if time_period.get('start'):
                time_period['start'] = time_period['start'].isoformat()
            if time_period.get('end'):
                time_period['end'] = time_period['end'].isoformat()
        
        # Convert enums to strings
        if params_dict.get('granularity'):
            params_dict['granularity'] = params_dict['granularity'].value
        
        if params_dict.get('metrics'):
            params_dict['metrics'] = [m.value if hasattr(m, 'value') else str(m) 
                                    for m in params_dict['metrics']]
        
        # Include profile in hash
        cache_key = {
            'params': params_dict,
            'profile': profile or 'default'
        }
        
        # Create consistent JSON string for hashing
        json_str = json.dumps(cache_key, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()
    
    def _get_cache_file_path(self, query_hash: str) -> Path:
        """Get the file path for a cache entry."""
        return self.cache_dir / f"{query_hash}.json"
    
    def _is_cache_valid(self, cache_file: Path, ttl: int) -> bool:
        """
        Check if cache file is still valid based on TTL.
        
        Args:
            cache_file: Path to cache file
            ttl: TTL in seconds
            
        Returns:
            True if cache is valid, False otherwise
        """
        if not cache_file.exists():
            return False
        
        try:
            file_mtime = cache_file.stat().st_mtime
            current_time = time.time()
            return (current_time - file_mtime) < ttl
        except OSError:
            return False
    
    def generate_cache_key(self, params: QueryParameters, profile: Optional[str] = None) -> str:
        """
        Generate cache key for query parameters.
        
        Args:
            params: Query parameters
            profile: AWS profile name
            
        Returns:
            Cache key string
        """
        return self._generate_query_hash(params, profile)
    
    def get_cached_data(self, cache_key: str) -> Optional[CostData]:
        """
        Retrieve cached cost data by cache key.
        
        Args:
            cache_key: Cache key string
            
        Returns:
            Cached CostData if available and valid, None otherwise
        """
        cache_file = self._get_cache_file_path(cache_key)
        
        if not self._is_cache_valid(cache_file, self.default_ttl):
            return None
        
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Reconstruct CostData from cached JSON
            return self._deserialize_cost_data(cache_data['data'])
        
        except (json.JSONDecodeError, KeyError) as e:
            # If cache file is corrupted, remove it
            try:
                cache_file.unlink()
            except OSError:
                pass
            return None
        except OSError as e:
            raise CacheError(f"Failed to read cache file: {e}")
    
    def get_cached_data_by_params(self, params: QueryParameters, profile: Optional[str] = None, 
                       ttl: Optional[int] = None) -> Optional[CostData]:
        """
        Retrieve cached cost data if available and valid (legacy method).
        
        Args:
            params: Query parameters
            profile: AWS profile name
            ttl: TTL in seconds (uses default if None)
            
        Returns:
            Cached CostData if available and valid, None otherwise
        """
        if ttl is None:
            ttl = self.default_ttl
        
        query_hash = self._generate_query_hash(params, profile)
        cache_file = self._get_cache_file_path(query_hash)
        
        if not self._is_cache_valid(cache_file, ttl):
            return None
        
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Reconstruct CostData from cached JSON
            return self._deserialize_cost_data(cache_data['data'])
        
        except (json.JSONDecodeError, KeyError, OSError) as e:
            # If cache file is corrupted, remove it
            try:
                cache_file.unlink()
            except OSError:
                pass
            return None    

    def cache_data(self, cache_key: str, data: CostData, ttl: Optional[int] = None) -> bool:
        """
        Cache cost data with cache key.
        
        Args:
            cache_key: Cache key string
            data: Cost data to cache
            ttl: TTL in seconds (uses default if None)
            
        Returns:
            True if caching succeeded, False otherwise
        """
        if ttl is None:
            ttl = self.default_ttl
        
        cache_file = self._get_cache_file_path(cache_key)
        
        try:
            cache_entry = {
                'data': self._serialize_cost_data(data),
                'cached_at': datetime.now(timezone.utc).isoformat(),
                'ttl': ttl,
                'query_hash': cache_key
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_entry, f, indent=2, default=str)
            
            return True
        
        except (OSError, TypeError, ValueError) as e:
            raise CacheError(f"Failed to write cache file: {e}")
    
    def cache_data_by_params(self, params: QueryParameters, data: CostData, 
                   profile: Optional[str] = None, ttl: Optional[int] = None) -> bool:
        """
        Cache cost data with TTL (legacy method).
        
        Args:
            params: Query parameters used to generate cache key
            data: Cost data to cache
            profile: AWS profile name
            ttl: TTL in seconds (uses default if None)
            
        Returns:
            True if caching succeeded, False otherwise
        """
        if ttl is None:
            ttl = self.default_ttl
        
        query_hash = self._generate_query_hash(params, profile)
        cache_file = self._get_cache_file_path(query_hash)
        
        try:
            cache_entry = {
                'data': self._serialize_cost_data(data),
                'cached_at': datetime.now(timezone.utc).isoformat(),
                'ttl': ttl,
                'query_hash': query_hash
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_entry, f, indent=2, default=str)
            
            return True
        
        except (OSError, TypeError, ValueError) as e:
            return False
    
    def invalidate_cache(self, pattern: Optional[str] = None) -> int:
        """
        Invalidate cache entries.
        
        Args:
            pattern: Optional pattern to match cache files. If None, clears all cache.
            
        Returns:
            Number of cache files removed
        """
        removed_count = 0
        
        try:
            if pattern is None:
                # Remove all cache files
                for cache_file in self.cache_dir.glob("*.json"):
                    try:
                        cache_file.unlink()
                        removed_count += 1
                    except OSError:
                        pass
            else:
                # Remove cache files matching pattern
                for cache_file in self.cache_dir.glob(f"*{pattern}*.json"):
                    try:
                        cache_file.unlink()
                        removed_count += 1
                    except OSError:
                        pass
        
        except OSError:
            pass
        
        return removed_count
    
    def clear_cache(self) -> int:
        """
        Clear all cache entries.
        
        Returns:
            Number of cache files removed
        """
        return self.invalidate_cache()
    
    def cleanup_expired_cache(self) -> int:
        """
        Remove expired cache entries.
        
        Returns:
            Number of expired cache files removed
        """
        removed_count = 0
        current_time = time.time()
        
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    # Check if file is expired based on modification time and stored TTL
                    with open(cache_file, 'r') as f:
                        cache_entry = json.load(f)
                    
                    ttl = cache_entry.get('ttl', self.default_ttl)
                    file_mtime = cache_file.stat().st_mtime
                    
                    if (current_time - file_mtime) >= ttl:
                        cache_file.unlink()
                        removed_count += 1
                
                except (OSError, json.JSONDecodeError, KeyError):
                    # Remove corrupted cache files
                    try:
                        cache_file.unlink()
                        removed_count += 1
                    except OSError:
                        pass
        
        except OSError:
            pass
        
        return removed_count
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        stats = {
            'total_entries': 0,
            'expired_entries': 0,
            'valid_entries': 0,
            'cache_size_bytes': 0,
            'oldest_entry': None,
            'newest_entry': None
        }
        
        try:
            current_time = time.time()
            oldest_time = None
            newest_time = None
            
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    stats['total_entries'] += 1
                    file_size = cache_file.stat().st_size
                    file_mtime = cache_file.stat().st_mtime
                    
                    stats['cache_size_bytes'] += file_size
                    
                    # Track oldest and newest entries
                    if oldest_time is None or file_mtime < oldest_time:
                        oldest_time = file_mtime
                        stats['oldest_entry'] = datetime.fromtimestamp(file_mtime).isoformat()
                    
                    if newest_time is None or file_mtime > newest_time:
                        newest_time = file_mtime
                        stats['newest_entry'] = datetime.fromtimestamp(file_mtime).isoformat()
                    
                    # Check if entry is expired
                    try:
                        with open(cache_file, 'r') as f:
                            cache_entry = json.load(f)
                        ttl = cache_entry.get('ttl', self.default_ttl)
                        
                        if (current_time - file_mtime) >= ttl:
                            stats['expired_entries'] += 1
                        else:
                            stats['valid_entries'] += 1
                    
                    except (json.JSONDecodeError, KeyError):
                        stats['expired_entries'] += 1
                
                except OSError:
                    pass
        
        except OSError:
            pass
        
        return stats  
  
    def _serialize_cost_data(self, data: CostData) -> Dict[str, Any]:
        """
        Serialize CostData to JSON-compatible dictionary.
        
        Args:
            data: CostData to serialize
            
        Returns:
            JSON-compatible dictionary
        """
        def serialize_obj(obj):
            """Helper to serialize complex objects."""
            if hasattr(obj, '__dict__'):
                return asdict(obj)
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif hasattr(obj, 'value'):  # Enum
                return obj.value
            else:
                return str(obj)
        
        return asdict(data)
    
    def _deserialize_cost_data(self, data_dict: Dict[str, Any]) -> CostData:
        """
        Deserialize dictionary back to CostData.
        
        Args:
            data_dict: Dictionary from cached JSON
            
        Returns:
            Reconstructed CostData object
        """
        from decimal import Decimal
        from .models import CostAmount, CostResult, Group, TimePeriod
        
        # Reconstruct TimePeriod objects
        def reconstruct_time_period(tp_dict):
            if tp_dict is None:
                return None
            return TimePeriod(
                start=datetime.fromisoformat(tp_dict['start']),
                end=datetime.fromisoformat(tp_dict['end'])
            )
        
        # Reconstruct CostAmount objects
        def reconstruct_cost_amount(ca_dict):
            if ca_dict is None:
                return None
            return CostAmount(
                amount=Decimal(str(ca_dict['amount'])),
                unit=ca_dict['unit']
            )
        
        # Reconstruct Group objects
        def reconstruct_group(group_dict):
            metrics = {}
            for key, value in group_dict['metrics'].items():
                metrics[key] = reconstruct_cost_amount(value)
            
            return Group(
                keys=group_dict['keys'],
                metrics=metrics
            )
        
        # Reconstruct CostResult objects
        def reconstruct_cost_result(cr_dict):
            groups = [reconstruct_group(g) for g in cr_dict['groups']]
            
            return CostResult(
                time_period=reconstruct_time_period(cr_dict['time_period']),
                total=reconstruct_cost_amount(cr_dict['total']),
                groups=groups,
                estimated=cr_dict.get('estimated', False)
            )
        
        # Reconstruct main CostData object
        results = [reconstruct_cost_result(r) for r in data_dict['results']]
        
        return CostData(
            results=results,
            time_period=reconstruct_time_period(data_dict['time_period']),
            total_cost=reconstruct_cost_amount(data_dict['total_cost']),
            currency=data_dict.get('currency', 'USD'),
            group_definitions=data_dict.get('group_definitions', [])
        )