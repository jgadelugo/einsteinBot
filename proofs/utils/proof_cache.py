"""
Proof caching utilities for performance optimization.

This module provides caching mechanisms for proof results to avoid
redundant computations and improve overall system performance.
"""

import hashlib
import logging
import pickle
from pathlib import Path
from typing import Dict, Optional, Union, Any

from ..theorem_generator import Theorem


class ProofCache:
    """
    Advanced caching system for proof results.
    
    Provides both memory and disk-based caching with automatic cleanup
    and corruption handling.
    """
    
    def __init__(self, cache_dir: Union[str, Path] = "cache/proofs", 
                 max_memory_size: int = 1000):
        """
        Initialize the proof cache.
        
        Args:
            cache_dir: Directory for disk cache storage
            max_memory_size: Maximum number of results to keep in memory
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache: Dict[str, Any] = {}
        self.max_memory_cache_size = max_memory_size
        self.logger = logging.getLogger(__name__)
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'disk_reads': 0,
            'disk_writes': 0,
            'corrupted_files': 0
        }
        
    def get_cache_key(self, theorem: Theorem) -> str:
        """
        Generate a unique cache key for a theorem.
        
        Args:
            theorem: The theorem to generate a key for
            
        Returns:
            Unique hash string for the theorem
        """
        # Include both statement and expression for uniqueness
        content = f"{theorem.statement}:{str(theorem.sympy_expression)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_cached_result(self, cache_key: str) -> Optional[Any]:
        """
        Retrieve cached proof result.
        
        Args:
            cache_key: The cache key to look up
            
        Returns:
            Cached result if found, None otherwise
        """
        # Check memory cache first
        if cache_key in self.memory_cache:
            self.stats['hits'] += 1
            self.logger.debug(f"Memory cache hit for key {cache_key[:8]}...")
            return self.memory_cache[cache_key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    result = pickle.load(f)
                
                self.stats['hits'] += 1
                self.stats['disk_reads'] += 1
                self.logger.debug(f"Disk cache hit for key {cache_key[:8]}...")
                
                # Store in memory cache if there's space
                if len(self.memory_cache) < self.max_memory_cache_size:
                    self.memory_cache[cache_key] = result
                
                return result
                
            except Exception as e:
                # If cache file is corrupted, remove it
                self.logger.warning(f"Corrupted cache file {cache_file}: {e}")
                cache_file.unlink(missing_ok=True)
                self.stats['corrupted_files'] += 1
                
        self.stats['misses'] += 1
        return None
    
    def cache_result(self, cache_key: str, result: Any) -> None:
        """
        Cache a proof result both in memory and on disk.
        
        Args:
            cache_key: The cache key
            result: The result to cache
        """
        # Memory cache with LRU-style eviction
        if len(self.memory_cache) >= self.max_memory_cache_size:
            # Remove oldest entry (simple FIFO for now)
            oldest_key = next(iter(self.memory_cache))
            del self.memory_cache[oldest_key]
        
        self.memory_cache[cache_key] = result
        
        # Disk cache
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            self.stats['disk_writes'] += 1
            self.logger.debug(f"Cached result for key {cache_key[:8]}...")
            
        except Exception as e:
            self.logger.warning(f"Failed to cache result to disk: {e}")
    
    def clear_cache(self) -> None:
        """Clear all cached results."""
        self.memory_cache.clear()
        
        # Clear disk cache
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
            except Exception as e:
                self.logger.warning(f"Failed to remove cache file {cache_file}: {e}")
        
        self.logger.info("Cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0.0
        
        return {
            **self.stats,
            'hit_rate': hit_rate,
            'memory_cache_size': len(self.memory_cache),
            'disk_cache_files': len(list(self.cache_dir.glob("*.pkl")))
        }
    
    def cleanup_old_cache(self, max_age_days: int = 30) -> int:
        """
        Remove old cache files.
        
        Args:
            max_age_days: Maximum age of cache files in days
            
        Returns:
            Number of files removed
        """
        import time
        
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 3600
        removed_count = 0
        
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                file_age = current_time - cache_file.stat().st_mtime
                if file_age > max_age_seconds:
                    cache_file.unlink()
                    removed_count += 1
            except Exception as e:
                self.logger.warning(f"Error checking cache file {cache_file}: {e}")
        
        if removed_count > 0:
            self.logger.info(f"Removed {removed_count} old cache files")
        
        return removed_count 