"""
Data loaders for MathBot UI with caching and error handling.

This module provides robust, cached data loaders for theorem, formula, and
validation data with comprehensive error handling and performance metrics.
"""

import json
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from config import logger
from .models import (
    FormulaData,
    GenerationMetadata,
    Theorem,
    TheoremCollection,
    ValidationEvidence,
    ValidationReport,
    SourceLineage,
)
from ..config import UIConfig


class CacheEntry:
    """Cache entry with TTL support."""
    
    def __init__(self, data: Any, ttl_seconds: int):
        self.data = data
        self.timestamp = datetime.now()
        self.ttl_seconds = ttl_seconds
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return datetime.now() > self.timestamp + timedelta(seconds=self.ttl_seconds)
    
    @property
    def age_seconds(self) -> float:
        """Get age of cache entry in seconds."""
        return (datetime.now() - self.timestamp).total_seconds()


class DataLoader:
    """Base class for data loaders with caching and error handling."""
    
    def __init__(self, config: UIConfig):
        self.config = config
        self.logger = logger.getChild(self.__class__.__name__)
        self._cache: Dict[str, CacheEntry] = {}
        self._cache_lock = threading.RLock()
        self._stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "load_times": [],
            "total_loads": 0
        }
        
    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get data from cache if not expired."""
        with self._cache_lock:
            entry = self._cache.get(key)
            if entry and not entry.is_expired:
                self._stats["cache_hits"] += 1
                self.logger.debug(f"Cache hit for {key} (age: {entry.age_seconds:.1f}s)")
                return entry.data
            elif entry:
                self.logger.debug(f"Cache expired for {key} (age: {entry.age_seconds:.1f}s)")
                del self._cache[key]
                
            self._stats["cache_misses"] += 1
            return None
    
    def _set_cache(self, key: str, data: Any) -> None:
        """Set data in cache with TTL and LRU eviction."""
        with self._cache_lock:
            # Implement LRU eviction if cache is full
            if len(self._cache) >= self.config.max_cache_size:
                oldest_key = min(self._cache.keys(), 
                               key=lambda k: self._cache[k].timestamp)
                del self._cache[oldest_key]
                self.logger.debug(f"Evicted cache entry: {oldest_key}")
            
            self._cache[key] = CacheEntry(data, self.config.cache_ttl_seconds)
            self.logger.debug(f"Cached data for {key}")
    
    def _load_json_file(self, file_path: Path) -> Optional[Dict]:
        """Load and validate JSON file with error handling and performance logging."""
        if not file_path.exists():
            self.logger.warning(f"File not found: {file_path}")
            return None
        
        try:
            start_time = time.time()
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            load_time = time.time() - start_time
            self._stats["load_times"].append(load_time)
            self._stats["total_loads"] += 1
            
            file_size = file_path.stat().st_size
            self.logger.info(f"Loaded {file_path.name} ({file_size:,} bytes) in {load_time:.3f}s")
            
            return data
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in {file_path}: {e}")
            return None
        except PermissionError as e:
            self.logger.error(f"Permission denied reading {file_path}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error loading {file_path}: {e}")
            return None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this loader."""
        with self._cache_lock:
            cache_hit_rate = 0.0
            if self._stats["cache_hits"] + self._stats["cache_misses"] > 0:
                cache_hit_rate = self._stats["cache_hits"] / (
                    self._stats["cache_hits"] + self._stats["cache_misses"]
                )
            
            avg_load_time = 0.0
            if self._stats["load_times"]:
                avg_load_time = sum(self._stats["load_times"]) / len(self._stats["load_times"])
            
            return {
                "cache_entries": len(self._cache),
                "cache_hit_rate": cache_hit_rate,
                "total_loads": self._stats["total_loads"],
                "average_load_time": avg_load_time,
                "max_load_time": max(self._stats["load_times"]) if self._stats["load_times"] else 0.0,
                "min_load_time": min(self._stats["load_times"]) if self._stats["load_times"] else 0.0,
            }
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        with self._cache_lock:
            cleared_count = len(self._cache)
            self._cache.clear()
            self.logger.info(f"Cleared {cleared_count} cache entries")


class TheoremLoader(DataLoader):
    """Loader for theorem data with validation and search indexing."""
    
    def load_theorems(self, force_reload: bool = False) -> List[Theorem]:
        """
        Load all theorems from the theorems file.
        
        Args:
            force_reload: Bypass cache and reload from file
            
        Returns:
            List of validated Theorem objects
        """
        cache_key = "all_theorems"
        
        if not force_reload:
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                return cached_data
        
        self.logger.info("Loading theorems from file")
        
        # Load raw data
        raw_data = self._load_json_file(self.config.theorems_file)
        if not raw_data:
            self.logger.warning("No theorem data available")
            return []
        
        # Extract theorems list
        theorems_data = raw_data.get('theorems', [])
        if not theorems_data:
            self.logger.warning("No theorems found in data file")
            return []
        
        # Validate and convert to Theorem objects
        theorems = []
        validation_errors = []
        
        for i, theorem_data in enumerate(theorems_data):
            try:
                theorem = Theorem(**theorem_data)
                theorems.append(theorem)
                
            except Exception as e:
                validation_errors.append(f"Theorem {i}: {str(e)}")
                self.logger.warning(f"Failed to validate theorem {i}: {e}")
        
        if validation_errors:
            self.logger.warning(f"Found {len(validation_errors)} validation errors")
            # Log first few errors for debugging
            for error in validation_errors[:3]:
                self.logger.debug(f"Validation error: {error}")
        
        self.logger.info(f"Successfully loaded {len(theorems)} theorems")
        
        # Cache the results
        self._set_cache(cache_key, theorems)
        
        return theorems
    
    def load_theorem_collection(self, force_reload: bool = False) -> Optional[TheoremCollection]:
        """
        Load complete theorem collection with metadata.
        
        Args:
            force_reload: Bypass cache and reload from file
            
        Returns:
            TheoremCollection or None if loading failed
        """
        cache_key = "theorem_collection"
        
        if not force_reload:
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                return cached_data
        
        self.logger.info("Loading theorem collection")
        
        # Load raw data
        raw_data = self._load_json_file(self.config.theorems_file)
        if not raw_data:
            return None
        
        try:
            collection = TheoremCollection(**raw_data)
            self.logger.info(f"Loaded theorem collection with {collection.theorem_count} theorems")
            
            # Cache the collection
            self._set_cache(cache_key, collection)
            
            return collection
            
        except Exception as e:
            self.logger.error(f"Failed to create theorem collection: {e}")
            return None
    
    def get_theorem_by_id(self, theorem_id: str) -> Optional[Theorem]:
        """
        Get a specific theorem by ID.
        
        Args:
            theorem_id: ID of the theorem to retrieve
            
        Returns:
            Theorem object or None if not found
        """
        theorems = self.load_theorems()
        for theorem in theorems:
            if theorem.id == theorem_id:
                return theorem
        return None
    
    def search_theorems(self, query: str, limit: Optional[int] = None) -> List[Theorem]:
        """
        Search theorems by query string.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching theorems, sorted by relevance
        """
        if len(query) < self.config.search_min_length:
            self.logger.debug(f"Query too short: '{query}' (min length: {self.config.search_min_length})")
            return []
        
        collection = self.load_theorem_collection()
        if not collection:
            return []
        
        # Use the collection's search method
        results = collection.search(query, limit or self.config.max_search_results)
        
        self.logger.info(f"Found {len(results)} theorems matching '{query}'")
        return results
    
    def get_theorems_by_type(self, theorem_type: str) -> List[Theorem]:
        """
        Get all theorems of a specific type.
        
        Args:
            theorem_type: Type of theorems to retrieve
            
        Returns:
            List of theorems of the specified type
        """
        theorems = self.load_theorems()
        results = [t for t in theorems if t.theorem_type == theorem_type]
        
        self.logger.debug(f"Found {len(results)} theorems of type '{theorem_type}'")
        return results
    
    def get_validation_summary(self) -> Dict[str, Union[int, float]]:
        """
        Get validation summary statistics.
        
        Returns:
            Dictionary with validation statistics
        """
        theorems = self.load_theorems()
        
        if not theorems:
            return {"total": 0, "validated": 0, "pass_rate": 0.0}
        
        validated_count = sum(1 for t in theorems if t.is_validated)
        
        return {
            "total": len(theorems),
            "validated": validated_count,
            "pass_rate": validated_count / len(theorems),
            "avg_confidence": sum(t.source_lineage.confidence for t in theorems) / len(theorems),
            "type_distribution": {
                theorem_type: len(self.get_theorems_by_type(theorem_type))
                for theorem_type in set(t.theorem_type for t in theorems)
            }
        }


class FormulaLoader(DataLoader):
    """Loader for formula data."""
    
    def load_formulas(self, force_reload: bool = False) -> List[FormulaData]:
        """
        Load all formulas from the formulas file.
        
        Args:
            force_reload: Bypass cache and reload from file
            
        Returns:
            List of FormulaData objects
        """
        cache_key = "all_formulas"
        
        if not force_reload:
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                return cached_data
        
        self.logger.info("Loading formulas from file")
        
        raw_data = self._load_json_file(self.config.formulas_file)
        if not raw_data:
            self.logger.warning("No formula data available")
            return []
        
        # Handle different possible formula data structures
        formulas = []
        
        if isinstance(raw_data, list):
            formulas_data = raw_data
        elif 'formulas' in raw_data:
            formulas_data = raw_data['formulas']
        else:
            self.logger.warning("Unexpected formula data structure")
            return []
        
        for i, formula_data in enumerate(formulas_data):
            try:
                if isinstance(formula_data, str):
                    # Handle simple string formulas
                    formula = FormulaData(
                        id=f"FORMULA_{i:04d}",
                        expression=formula_data,
                        source="processed_data"
                    )
                else:
                    formula = FormulaData(**formula_data)
                
                formulas.append(formula)
                
            except Exception as e:
                self.logger.warning(f"Failed to load formula {i}: {e}")
        
        self.logger.info(f"Loaded {len(formulas)} formulas")
        self._set_cache(cache_key, formulas)
        
        return formulas
    
    def search_formulas(self, query: str, limit: Optional[int] = None) -> List[FormulaData]:
        """
        Search formulas by query string.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching formulas
        """
        if len(query) < self.config.search_min_length:
            return []
        
        formulas = self.load_formulas()
        query_lower = query.lower()
        
        matching_formulas = []
        for formula in formulas:
            if (query_lower in formula.expression.lower() or
                (formula.latex_form and query_lower in formula.latex_form.lower()) or
                query_lower in formula.source.lower()):
                matching_formulas.append(formula)
        
        if limit:
            matching_formulas = matching_formulas[:limit]
        
        self.logger.info(f"Found {len(matching_formulas)} formulas matching '{query}'")
        return matching_formulas


class ValidationLoader(DataLoader):
    """Loader for validation report data."""
    
    def load_validation_report(self, force_reload: bool = False) -> Optional[ValidationReport]:
        """
        Load the latest validation report.
        
        Args:
            force_reload: Bypass cache and reload from file
            
        Returns:
            ValidationReport or None if loading failed
        """
        cache_key = "validation_report"
        
        if not force_reload:
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                return cached_data
        
        self.logger.info("Loading validation report")
        
        raw_data = self._load_json_file(self.config.validation_file)
        if not raw_data:
            self.logger.warning("No validation report available")
            return None
        
        try:
            report = ValidationReport(**raw_data)
            self.logger.info(f"Loaded validation report: {report.validation_summary}")
            
            self._set_cache(cache_key, report)
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to parse validation report: {e}")
            return None


# Factory function for easy initialization
def create_data_loaders(config: Optional[UIConfig] = None) -> Tuple[TheoremLoader, FormulaLoader, ValidationLoader]:
    """
    Create all data loaders with the given configuration.
    
    Args:
        config: UI configuration (uses default if None)
        
    Returns:
        Tuple of (theorem_loader, formula_loader, validation_loader)
    """
    if config is None:
        from ..config import get_ui_config
        config = get_ui_config()
    
    theorem_loader = TheoremLoader(config)
    formula_loader = FormulaLoader(config)
    validation_loader = ValidationLoader(config)
    
    return theorem_loader, formula_loader, validation_loader


def get_all_performance_stats(config: Optional[UIConfig] = None) -> Dict[str, Dict[str, Any]]:
    """
    Get performance statistics for all loaders.
    
    Args:
        config: UI configuration (uses default if None)
        
    Returns:
        Dictionary with performance stats for each loader
    """
    theorem_loader, formula_loader, validation_loader = create_data_loaders(config)
    
    return {
        "theorem_loader": theorem_loader.get_performance_stats(),
        "formula_loader": formula_loader.get_performance_stats(),
        "validation_loader": validation_loader.get_performance_stats(),
    } 