"""Unit tests for cache management system."""

import json
import os
import tempfile
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.aws_cost_cli.cache_manager import CacheManager
from src.aws_cost_cli.models import (
    CostData, CostResult, CostAmount, Group, TimePeriod, 
    QueryParameters, TimePeriodGranularity, MetricType
)


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for cache testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def cache_manager(temp_cache_dir):
    """Create a CacheManager instance with temporary directory."""
    return CacheManager(cache_dir=temp_cache_dir, default_ttl=3600)


@pytest.fixture
def sample_query_params():
    """Create sample query parameters for testing."""
    return QueryParameters(
        service="EC2",
        time_period=TimePeriod(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 31, tzinfo=timezone.utc)
        ),
        granularity=TimePeriodGranularity.MONTHLY,
        metrics=[MetricType.BLENDED_COST]
    )


@pytest.fixture
def sample_cost_data():
    """Create sample cost data for testing."""
    time_period = TimePeriod(
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 31, tzinfo=timezone.utc)
    )
    
    cost_amount = CostAmount(amount=Decimal("123.45"), unit="USD")
    
    group = Group(
        keys=["EC2-Instance"],
        metrics={"BlendedCost": cost_amount}
    )
    
    cost_result = CostResult(
        time_period=time_period,
        total=cost_amount,
        groups=[group],
        estimated=False
    )
    
    return CostData(
        results=[cost_result],
        time_period=time_period,
        total_cost=cost_amount,
        currency="USD",
        group_definitions=["SERVICE"]
    )


class TestCacheManager:
    """Test cases for CacheManager class."""
    
    def test_init_default_cache_dir(self):
        """Test CacheManager initialization with default cache directory."""
        with patch('os.path.expanduser') as mock_expanduser, \
             patch('pathlib.Path.mkdir') as mock_mkdir:
            mock_expanduser.return_value = "/home/user/.aws-cost-cli/cache"
            
            cache_manager = CacheManager()
            
            assert cache_manager.default_ttl == 3600
            mock_expanduser.assert_called_once_with("~/.aws-cost-cli/cache")
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    
    def test_init_custom_cache_dir(self, temp_cache_dir):
        """Test CacheManager initialization with custom cache directory."""
        cache_manager = CacheManager(cache_dir=temp_cache_dir, default_ttl=7200)
        
        assert cache_manager.default_ttl == 7200
        assert cache_manager.cache_dir == Path(temp_cache_dir)
        assert cache_manager.cache_dir.exists()
    
    def test_generate_query_hash_consistent(self, cache_manager, sample_query_params):
        """Test that query hash generation is consistent."""
        hash1 = cache_manager._generate_query_hash(sample_query_params, "default")
        hash2 = cache_manager._generate_query_hash(sample_query_params, "default")
        
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length
    
    def test_generate_query_hash_different_params(self, cache_manager, sample_query_params):
        """Test that different parameters generate different hashes."""
        hash1 = cache_manager._generate_query_hash(sample_query_params, "default")
        
        # Modify parameters
        different_params = QueryParameters(
            service="S3",
            time_period=sample_query_params.time_period,
            granularity=sample_query_params.granularity,
            metrics=sample_query_params.metrics
        )
        
        hash2 = cache_manager._generate_query_hash(different_params, "default")
        
        assert hash1 != hash2
    
    def test_generate_query_hash_different_profiles(self, cache_manager, sample_query_params):
        """Test that different profiles generate different hashes."""
        hash1 = cache_manager._generate_query_hash(sample_query_params, "profile1")
        hash2 = cache_manager._generate_query_hash(sample_query_params, "profile2")
        
        assert hash1 != hash2
    
    def test_cache_and_retrieve_data(self, cache_manager, sample_query_params, sample_cost_data):
        """Test caching and retrieving cost data."""
        # Cache the data
        success = cache_manager.cache_data(sample_query_params, sample_cost_data)
        assert success is True
        
        # Retrieve the data
        retrieved_data = cache_manager.get_cached_data(sample_query_params)
        
        assert retrieved_data is not None
        assert retrieved_data.currency == sample_cost_data.currency
        assert len(retrieved_data.results) == len(sample_cost_data.results)
        assert retrieved_data.total_cost.amount == sample_cost_data.total_cost.amount
    
    def test_cache_data_with_profile(self, cache_manager, sample_query_params, sample_cost_data):
        """Test caching data with specific AWS profile."""
        profile = "test-profile"
        
        # Cache with profile
        success = cache_manager.cache_data(sample_query_params, sample_cost_data, profile=profile)
        assert success is True
        
        # Retrieve with same profile
        retrieved_data = cache_manager.get_cached_data(sample_query_params, profile=profile)
        assert retrieved_data is not None
        
        # Retrieve with different profile should return None
        different_profile_data = cache_manager.get_cached_data(sample_query_params, profile="different")
        assert different_profile_data is None 
   
    def test_cache_expiration(self, cache_manager, sample_query_params, sample_cost_data):
        """Test cache expiration based on TTL."""
        # Cache with very short TTL
        short_ttl = 1
        success = cache_manager.cache_data(sample_query_params, sample_cost_data, ttl=short_ttl)
        assert success is True
        
        # Immediately retrieve - should work
        retrieved_data = cache_manager.get_cached_data(sample_query_params, ttl=short_ttl)
        assert retrieved_data is not None
        
        # Wait for expiration
        time.sleep(short_ttl + 0.1)
        
        # Try to retrieve expired data
        expired_data = cache_manager.get_cached_data(sample_query_params, ttl=short_ttl)
        assert expired_data is None
    
    def test_cache_file_corruption_handling(self, cache_manager, sample_query_params, temp_cache_dir):
        """Test handling of corrupted cache files."""
        # Create a corrupted cache file
        query_hash = cache_manager._generate_query_hash(sample_query_params)
        cache_file = Path(temp_cache_dir) / f"{query_hash}.json"
        
        # Write invalid JSON
        with open(cache_file, 'w') as f:
            f.write("invalid json content")
        
        # Try to retrieve - should return None and remove corrupted file
        retrieved_data = cache_manager.get_cached_data(sample_query_params)
        assert retrieved_data is None
        assert not cache_file.exists()
    
    def test_invalidate_all_cache(self, cache_manager, sample_query_params, sample_cost_data):
        """Test invalidating all cache entries."""
        # Cache some data
        cache_manager.cache_data(sample_query_params, sample_cost_data)
        
        # Create another cache entry with different params
        different_params = QueryParameters(service="S3")
        cache_manager.cache_data(different_params, sample_cost_data)
        
        # Verify both entries exist
        assert cache_manager.get_cached_data(sample_query_params) is not None
        assert cache_manager.get_cached_data(different_params) is not None
        
        # Invalidate all cache
        removed_count = cache_manager.invalidate_cache()
        assert removed_count == 2
        
        # Verify both entries are gone
        assert cache_manager.get_cached_data(sample_query_params) is None
        assert cache_manager.get_cached_data(different_params) is None
    
    def test_invalidate_cache_with_pattern(self, cache_manager, sample_query_params, sample_cost_data):
        """Test invalidating cache entries with pattern matching."""
        # Cache data with different profiles
        cache_manager.cache_data(sample_query_params, sample_cost_data, profile="prod")
        cache_manager.cache_data(sample_query_params, sample_cost_data, profile="dev")
        
        # Get the hash for prod profile to use as pattern
        prod_hash = cache_manager._generate_query_hash(sample_query_params, "prod")
        pattern = prod_hash[:8]  # Use first 8 characters as pattern
        
        # Invalidate with pattern
        removed_count = cache_manager.invalidate_cache(pattern=pattern)
        assert removed_count >= 1
    
    def test_cleanup_expired_cache(self, cache_manager, sample_query_params, sample_cost_data):
        """Test cleanup of expired cache entries."""
        # Cache data with short TTL
        short_ttl = 1
        cache_manager.cache_data(sample_query_params, sample_cost_data, ttl=short_ttl)
        
        # Cache data with long TTL
        long_ttl = 3600
        different_params = QueryParameters(service="S3")
        cache_manager.cache_data(different_params, sample_cost_data, ttl=long_ttl)
        
        # Wait for first entry to expire
        time.sleep(short_ttl + 0.1)
        
        # Cleanup expired entries
        removed_count = cache_manager.cleanup_expired_cache()
        assert removed_count == 1
        
        # Verify only expired entry was removed
        assert cache_manager.get_cached_data(sample_query_params, ttl=short_ttl) is None
        assert cache_manager.get_cached_data(different_params, ttl=long_ttl) is not None
    
    def test_get_cache_stats(self, cache_manager, sample_query_params, sample_cost_data):
        """Test cache statistics generation."""
        # Initially empty cache
        stats = cache_manager.get_cache_stats()
        assert stats['total_entries'] == 0
        assert stats['valid_entries'] == 0
        assert stats['expired_entries'] == 0
        
        # Add some cache entries
        cache_manager.cache_data(sample_query_params, sample_cost_data)
        different_params = QueryParameters(service="S3")
        cache_manager.cache_data(different_params, sample_cost_data)
        
        # Check stats
        stats = cache_manager.get_cache_stats()
        assert stats['total_entries'] == 2
        assert stats['valid_entries'] == 2
        assert stats['expired_entries'] == 0
        assert stats['cache_size_bytes'] > 0
        assert stats['oldest_entry'] is not None
        assert stats['newest_entry'] is not None
    
    def test_cache_data_serialization_edge_cases(self, cache_manager, sample_query_params):
        """Test serialization of edge cases in cost data."""
        # Create cost data with edge cases
        time_period = TimePeriod(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 31, tzinfo=timezone.utc)
        )
        
        # Cost data with zero amount
        zero_cost = CostAmount(amount=Decimal("0.00"), unit="USD")
        
        # Cost data with very large amount
        large_cost = CostAmount(amount=Decimal("999999.99"), unit="USD")
        
        cost_result = CostResult(
            time_period=time_period,
            total=large_cost,
            groups=[],  # Empty groups
            estimated=True
        )
        
        edge_case_data = CostData(
            results=[cost_result],
            time_period=time_period,
            total_cost=zero_cost,
            currency="EUR",  # Different currency
            group_definitions=[]  # Empty group definitions
        )
        
        # Cache and retrieve
        success = cache_manager.cache_data(sample_query_params, edge_case_data)
        assert success is True
        
        retrieved_data = cache_manager.get_cached_data(sample_query_params)
        assert retrieved_data is not None
        assert retrieved_data.currency == "EUR"
        assert retrieved_data.total_cost.amount == Decimal("0.00")
        assert retrieved_data.results[0].estimated is True
        assert len(retrieved_data.results[0].groups) == 0
    
    def test_cache_directory_creation_failure(self):
        """Test handling of cache directory creation failure."""
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            mock_mkdir.side_effect = OSError("Permission denied")
            
            # Should not raise exception
            cache_manager = CacheManager(cache_dir="/invalid/path")
            assert cache_manager is not None
            assert cache_manager.cache_dir == Path("/invalid/path")
    
    def test_cache_write_failure(self, cache_manager, sample_query_params, sample_cost_data):
        """Test handling of cache write failures."""
        with patch('builtins.open', side_effect=OSError("Disk full")):
            success = cache_manager.cache_data(sample_query_params, sample_cost_data)
            assert success is False
    
    def test_cache_read_failure(self, cache_manager, sample_query_params, sample_cost_data, temp_cache_dir):
        """Test handling of cache read failures."""
        # First cache the data successfully
        cache_manager.cache_data(sample_query_params, sample_cost_data)
        
        # Mock file read failure
        with patch('builtins.open', side_effect=OSError("File not accessible")):
            retrieved_data = cache_manager.get_cached_data(sample_query_params)
            assert retrieved_data is None