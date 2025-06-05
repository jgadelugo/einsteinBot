"""
Unit tests for UI data loaders.
"""

import json
import time
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from datetime import datetime, timedelta

from ui.data.loaders import (
    CacheEntry,
    DataLoader,
    TheoremLoader,
    FormulaLoader,
    ValidationLoader,
    create_data_loaders
)
from ui.data.models import Theorem, FormulaData, ValidationReport
from ui.config import UIConfig


class TestCacheEntry:
    """Test CacheEntry class."""
    
    def test_cache_entry_creation(self):
        """Test creating cache entry."""
        data = {"test": "value"}
        entry = CacheEntry(data, 300)
        
        assert entry.data == data
        assert entry.ttl_seconds == 300
        assert isinstance(entry.timestamp, datetime)
        assert not entry.is_expired
    
    def test_cache_expiration(self):
        """Test cache expiration logic."""
        data = {"test": "value"}
        entry = CacheEntry(data, 0)  # 0 second TTL
        
        # Should be expired immediately
        time.sleep(0.1)
        assert entry.is_expired
    
    def test_cache_not_expired(self):
        """Test cache not expired."""
        data = {"test": "value"}
        entry = CacheEntry(data, 3600)  # 1 hour TTL
        
        assert not entry.is_expired


class TestDataLoader:
    """Test base DataLoader class."""
    
    def create_mock_config(self) -> UIConfig:
        """Create mock configuration for testing."""
        return UIConfig(
            cache_ttl_seconds=300,
            max_cache_size=100,
            theorems_file=Path("test_theorems.json"),
            formulas_file=Path("test_formulas.json"),
            validation_file=Path("test_validation.json")
        )
    
    def test_loader_initialization(self):
        """Test DataLoader initialization."""
        config = self.create_mock_config()
        loader = DataLoader(config)
        
        assert loader.config == config
        assert loader._cache == {}
        assert loader._cache_lock is not None
    
    def test_cache_get_set(self):
        """Test cache get and set operations."""
        config = self.create_mock_config()
        loader = DataLoader(config)
        
        # Set cache
        test_data = {"test": "value"}
        loader._set_cache("test_key", test_data)
        
        # Get from cache
        cached_data = loader._get_from_cache("test_key")
        assert cached_data == test_data
    
    def test_cache_expiration(self):
        """Test cache expiration."""
        config = UIConfig(cache_ttl_seconds=10)  # 10 second TTL (minimum allowed)
        loader = DataLoader(config)
        
        # Set cache
        test_data = {"test": "value"}
        loader._set_cache("test_key", test_data)
        
        # Should get from cache immediately
        assert loader._get_from_cache("test_key") == test_data
        
        # Manually expire cache by modifying entry timestamp
        with loader._cache_lock:
            for entry in loader._cache.values():
                entry.timestamp = datetime.now() - timedelta(seconds=11)
        
        # Should not get from cache after expiration
        assert loader._get_from_cache("test_key") is None
    
    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        config = UIConfig(cache_ttl_seconds=300, max_cache_size=100)  # Use minimum allowed size
        loader = DataLoader(config)
        
        # Override cache size temporarily for testing
        loader.config = UIConfig(cache_ttl_seconds=300, max_cache_size=100)
        
        # Fill cache beyond capacity by directly modifying the cache
        loader._cache = {}
        for i in range(102):  # Exceed max_cache_size
            loader._set_cache(f"key{i}", f"value{i}")
        
        # Cache should not exceed max size
        assert len(loader._cache) <= loader.config.max_cache_size
    
    def test_load_json_file_success(self):
        """Test successful JSON file loading."""
        config = self.create_mock_config()
        loader = DataLoader(config)
        
        # Test with actual existing file (theorems.json)
        from config import PROJECT_ROOT
        theorems_file = PROJECT_ROOT / "results" / "theorems.json"
        
        if theorems_file.exists():
            result = loader._load_json_file(theorems_file)
            assert result is not None
            assert isinstance(result, dict)
        else:
            # Skip test if file doesn't exist
            pytest.skip("theorems.json not found for testing")
    
    def test_load_json_file_not_found(self):
        """Test JSON file loading when file doesn't exist."""
        config = self.create_mock_config()
        loader = DataLoader(config)
        
        with patch.object(Path, "exists", return_value=False):
            result = loader._load_json_file(Path("nonexistent.json"))
        
        assert result is None
    
    def test_load_json_file_invalid_json(self):
        """Test JSON file loading with invalid JSON."""
        config = self.create_mock_config()
        loader = DataLoader(config)
        
        with patch("builtins.open", mock_open(read_data="invalid json")):
            with patch.object(Path, "exists", return_value=True):
                result = loader._load_json_file(Path("test.json"))
        
        assert result is None
    
    def test_load_json_file_io_error(self):
        """Test JSON file loading with IO error."""
        config = self.create_mock_config()
        loader = DataLoader(config)
        
        with patch("builtins.open", side_effect=IOError("File error")):
            with patch.object(Path, "exists", return_value=True):
                result = loader._load_json_file(Path("test.json"))
        
        assert result is None


class TestTheoremLoader:
    """Test TheoremLoader class."""
    
    def create_mock_config(self) -> UIConfig:
        """Create mock configuration for testing."""
        return UIConfig(
            cache_ttl_seconds=300,
            max_cache_size=100,
            search_min_length=2,
            max_search_results=50,
            theorems_file=Path("test_theorems.json")
        )
    
    def create_sample_theorem_data(self) -> dict:
        """Create sample theorem data for testing."""
        return {
            "generation_metadata": {
                "total_theorems": 2,
                "generation_time": 1.0,
                "validation_passes": 2,
                "type_distribution": {"algebraic": 1, "polynomial": 1},
                "generator_version": "1.0.0"
            },
            "theorems": [
                {
                    "id": "THM_12345678",
                    "statement": "Test theorem 1",
                    "sympy_expression": "x**2",
                    "theorem_type": "algebraic",
                    "assumptions": [],
                    "source_lineage": {
                        "original_formula": "x^2",
                        "hypothesis_id": "test_1",
                        "confidence": 0.95,
                        "validation_score": 1.0,
                        "generation_method": "test",
                        "source_type": "test"
                    },
                    "natural_language": "A test theorem about squares",
                    "symbols": ["x"],
                    "mathematical_context": {
                        "symbols": ["x"],
                        "domain": "algebra"
                    },
                    "validation_evidence": {
                        "validation_status": "PASS",
                        "pass_rate": 1.0,
                        "total_tests": 10,
                        "symbols_tested": ["x"],
                        "validation_time": 0.1
                    },
                    "metadata": {}
                },
                {
                    "id": "THM_87654321",
                    "statement": "Test theorem 2",
                    "sympy_expression": "y**3",
                    "theorem_type": "polynomial",
                    "assumptions": [],
                    "source_lineage": {
                        "original_formula": "y^3",
                        "hypothesis_id": "test_2",
                        "confidence": 0.85,
                        "validation_score": 1.0,
                        "generation_method": "test",
                        "source_type": "test"
                    },
                    "natural_language": "A test theorem about cubes",
                    "symbols": ["y"],
                    "mathematical_context": {
                        "symbols": ["y"],
                        "domain": "algebra"
                    },
                    "validation_evidence": {
                        "validation_status": "PASS",
                        "pass_rate": 0.9,
                        "total_tests": 10,
                        "symbols_tested": ["y"],
                        "validation_time": 0.15
                    },
                    "metadata": {}
                }
            ]
        }
    
    def test_load_theorems_success(self):
        """Test successful theorem loading."""
        config = self.create_mock_config()
        loader = TheoremLoader(config)
        
        sample_data = self.create_sample_theorem_data()
        
        with patch.object(loader, '_load_json_file', return_value=sample_data):
            theorems = loader.load_theorems()
        
        assert len(theorems) == 2
        assert all(isinstance(t, Theorem) for t in theorems)
        assert theorems[0].id == "THM_12345678"
        assert theorems[1].id == "THM_87654321"
    
    def test_load_theorems_empty_file(self):
        """Test loading theorems from empty file."""
        config = self.create_mock_config()
        loader = TheoremLoader(config)
        
        with patch.object(loader, '_load_json_file', return_value=None):
            theorems = loader.load_theorems()
        
        assert theorems == []
    
    def test_load_theorems_no_theorems_key(self):
        """Test loading when theorems key is missing."""
        config = self.create_mock_config()
        loader = TheoremLoader(config)
        
        with patch.object(loader, '_load_json_file', return_value={}):
            theorems = loader.load_theorems()
        
        assert theorems == []
    
    def test_load_theorems_invalid_data(self):
        """Test loading with invalid theorem data."""
        config = self.create_mock_config()
        loader = TheoremLoader(config)
        
        invalid_data = {
            "theorems": [
                {"id": "INVALID_ID", "statement": "test"},  # Missing required fields
                {
                    "id": "THM_12345678",
                    "statement": "Valid theorem",
                    "sympy_expression": "x**2",
                    "theorem_type": "test",
                    "source_lineage": {
                        "original_formula": "x^2",
                        "hypothesis_id": "test",
                        "confidence": 0.95,
                        "validation_score": 1.0,
                        "generation_method": "test",
                        "source_type": "test"
                    },
                    "natural_language": "Valid",
                    "validation_evidence": {
                        "validation_status": "PASS",
                        "pass_rate": 1.0,
                        "total_tests": 10,
                        "validation_time": 0.1
                    }
                }
            ]
        }
        
        with patch.object(loader, '_load_json_file', return_value=invalid_data):
            theorems = loader.load_theorems()
        
        # Should only load valid theorems
        assert len(theorems) == 1
        assert theorems[0].id == "THM_12345678"
    
    def test_theorem_caching(self):
        """Test theorem caching behavior."""
        config = self.create_mock_config()
        loader = TheoremLoader(config)
        
        sample_data = self.create_sample_theorem_data()
        
        with patch.object(loader, '_load_json_file', return_value=sample_data) as mock_load:
            # First call should load from file
            theorems1 = loader.load_theorems()
            assert mock_load.call_count == 1
            
            # Second call should use cache
            theorems2 = loader.load_theorems()
            assert mock_load.call_count == 1  # No additional calls
            
            # Results should be identical
            assert len(theorems1) == len(theorems2)
            assert theorems1[0].id == theorems2[0].id
    
    def test_force_reload(self):
        """Test force reload bypasses cache."""
        config = self.create_mock_config()
        loader = TheoremLoader(config)
        
        sample_data = self.create_sample_theorem_data()
        
        with patch.object(loader, '_load_json_file', return_value=sample_data) as mock_load:
            # First call
            loader.load_theorems()
            assert mock_load.call_count == 1
            
            # Force reload should call file load again
            loader.load_theorems(force_reload=True)
            assert mock_load.call_count == 2
    
    def test_get_theorem_by_id(self):
        """Test getting theorem by ID."""
        config = self.create_mock_config()
        loader = TheoremLoader(config)
        
        sample_data = self.create_sample_theorem_data()
        
        with patch.object(loader, '_load_json_file', return_value=sample_data):
            theorem = loader.get_theorem_by_id("THM_12345678")
            assert theorem is not None
            assert theorem.id == "THM_12345678"
            
            # Non-existent ID
            missing = loader.get_theorem_by_id("THM_99999999")
            assert missing is None
    
    def test_search_theorems(self):
        """Test theorem search functionality."""
        config = self.create_mock_config()
        loader = TheoremLoader(config)
        
        sample_data = self.create_sample_theorem_data()
        
        with patch.object(loader, '_load_json_file', return_value=sample_data):
            # Search for "squares"
            results = loader.search_theorems("squares")
            assert len(results) == 1
            assert "squares" in results[0].natural_language
            
            # Search for "test" (should match both)
            results = loader.search_theorems("test")
            assert len(results) == 2
            
            # Search with limit
            results = loader.search_theorems("test", limit=1)
            assert len(results) == 1
            
            # Short query (below minimum length)
            results = loader.search_theorems("x")
            assert len(results) == 0
    
    def test_get_theorems_by_type(self):
        """Test getting theorems by type."""
        config = self.create_mock_config()
        loader = TheoremLoader(config)
        
        sample_data = self.create_sample_theorem_data()
        
        with patch.object(loader, '_load_json_file', return_value=sample_data):
            algebraic_theorems = loader.get_theorems_by_type("algebraic")
            assert len(algebraic_theorems) == 1
            assert algebraic_theorems[0].theorem_type == "algebraic"
            
            polynomial_theorems = loader.get_theorems_by_type("polynomial")
            assert len(polynomial_theorems) == 1
            assert polynomial_theorems[0].theorem_type == "polynomial"
            
            # Non-existent type
            missing_type = loader.get_theorems_by_type("nonexistent")
            assert len(missing_type) == 0
    
    def test_get_validation_summary(self):
        """Test validation summary statistics."""
        config = self.create_mock_config()
        loader = TheoremLoader(config)
        
        sample_data = self.create_sample_theorem_data()
        
        with patch.object(loader, '_load_json_file', return_value=sample_data):
            summary = loader.get_validation_summary()
            
            assert summary["total"] == 2
            assert summary["validated"] == 2  # Both theorems pass
            assert summary["pass_rate"] == 1.0
            assert 0.8 <= summary["avg_confidence"] <= 1.0


class TestFormulaLoader:
    """Test FormulaLoader class."""
    
    def create_mock_config(self) -> UIConfig:
        """Create mock configuration."""
        return UIConfig(
            cache_ttl_seconds=300,
            formulas_file=Path("test_formulas.json")
        )
    
    def test_load_formulas_list_format(self):
        """Test loading formulas in list format."""
        config = self.create_mock_config()
        loader = FormulaLoader(config)
        
        sample_data = ["x^2 + 1", "y^3 - 2"]
        
        with patch.object(loader, '_load_json_file', return_value=sample_data):
            formulas = loader.load_formulas()
        
        assert len(formulas) == 2
        assert all(isinstance(f, FormulaData) for f in formulas)
        assert formulas[0].expression == "x^2 + 1"
        assert formulas[1].expression == "y^3 - 2"
    
    def test_load_formulas_dict_format(self):
        """Test loading formulas in dict format."""
        config = self.create_mock_config()
        loader = FormulaLoader(config)
        
        sample_data = {
            "formulas": [
                {
                    "id": "F001",
                    "expression": "x^2 + 1",
                    "source": "test"
                }
            ]
        }
        
        with patch.object(loader, '_load_json_file', return_value=sample_data):
            formulas = loader.load_formulas()
        
        assert len(formulas) == 1
        assert formulas[0].id == "F001"
        assert formulas[0].expression == "x^2 + 1"
    
    def test_load_formulas_empty(self):
        """Test loading empty formulas."""
        config = self.create_mock_config()
        loader = FormulaLoader(config)
        
        with patch.object(loader, '_load_json_file', return_value=None):
            formulas = loader.load_formulas()
        
        assert formulas == []


class TestValidationLoader:
    """Test ValidationLoader class."""
    
    def create_mock_config(self) -> UIConfig:
        """Create mock configuration."""
        return UIConfig(
            cache_ttl_seconds=300,
            validation_file=Path("test_validation.json")
        )
    
    def test_load_validation_report(self):
        """Test loading validation report."""
        config = self.create_mock_config()
        loader = ValidationLoader(config)
        
        sample_data = {
            "summary": {
                "total_formulas": 5,
                "validated_formulas": 5,
                "passed_formulas": 4,
                "failed_formulas": 0,
                "error_formulas": 1,
                "partial_formulas": 0,
                "overall_pass_rate": 0.8,
                "average_confidence": 1.0,
                "validation_time": 0.176,
                "timestamp": "2025-01-01 00:00:00"
            },
            "statistics": {"batch_name": "test"},
            "results_by_status": {"PASS": ["formula1"], "ERROR": ["formula2"]},
            "errors_summary": ["formula2: error message"]
        }
        
        with patch.object(loader, '_load_json_file', return_value=sample_data):
            report = loader.load_validation_report()
        
        assert isinstance(report, ValidationReport)
        assert report.summary.total_formulas == 5
        assert report.summary.passed_formulas == 4
    
    def test_load_validation_report_none(self):
        """Test loading when validation file doesn't exist."""
        config = self.create_mock_config()
        loader = ValidationLoader(config)
        
        with patch.object(loader, '_load_json_file', return_value=None):
            report = loader.load_validation_report()
        
        assert report is None
    
    def test_load_validation_report_invalid(self):
        """Test loading invalid validation data."""
        config = self.create_mock_config()
        loader = ValidationLoader(config)
        
        invalid_data = {"invalid": "data"}
        
        with patch.object(loader, '_load_json_file', return_value=invalid_data):
            report = loader.load_validation_report()
        
        assert report is None


class TestCreateDataLoaders:
    """Test create_data_loaders factory function."""
    
    def test_create_with_config(self):
        """Test creating loaders with custom config."""
        config = UIConfig(cache_ttl_seconds=600)
        
        theorem_loader, formula_loader, validation_loader = create_data_loaders(config)
        
        assert isinstance(theorem_loader, TheoremLoader)
        assert isinstance(formula_loader, FormulaLoader)
        assert isinstance(validation_loader, ValidationLoader)
        
        assert theorem_loader.config.cache_ttl_seconds == 600
        assert formula_loader.config.cache_ttl_seconds == 600
        assert validation_loader.config.cache_ttl_seconds == 600
    
    def test_create_with_default_config(self):
        """Test creating loaders with default config."""
        with patch('ui.config.get_ui_config') as mock_get_config:
            mock_config = UIConfig()
            mock_get_config.return_value = mock_config
            
            theorem_loader, formula_loader, validation_loader = create_data_loaders()
            
            assert isinstance(theorem_loader, TheoremLoader)
            assert isinstance(formula_loader, FormulaLoader)
            assert isinstance(validation_loader, ValidationLoader)
            
            mock_get_config.assert_called_once()


class TestPerformanceMetrics:
    """Test performance and caching metrics."""
    
    def test_cache_hit_rate_calculation(self):
        """Test that cache hits improve performance."""
        config = UIConfig(cache_ttl_seconds=300)
        loader = TheoremLoader(config)
        
        sample_data = {
            "theorems": [{
                "id": "THM_12345678",
                "statement": "Test",
                "sympy_expression": "x",
                "theorem_type": "test",
                "source_lineage": {
                    "original_formula": "x",
                    "hypothesis_id": "test",
                    "confidence": 1.0,
                    "validation_score": 1.0,
                    "generation_method": "test",
                    "source_type": "test"
                },
                "natural_language": "Test",
                "validation_evidence": {
                    "validation_status": "PASS",
                    "pass_rate": 1.0,
                    "total_tests": 1,
                    "validation_time": 0.1
                }
            }]
        }
        
        with patch.object(loader, '_load_json_file', return_value=sample_data):
            # Time first load (cache miss)
            start_time = time.time()
            theorems1 = loader.load_theorems()
            first_load_time = time.time() - start_time
            
            # Time second load (cache hit)
            start_time = time.time()
            theorems2 = loader.load_theorems()
            second_load_time = time.time() - start_time
            
            # Cache hit should be significantly faster
            assert second_load_time < first_load_time
            assert len(theorems1) == len(theorems2)
    
    def test_memory_usage_bounded(self):
        """Test that cache size is bounded."""
        config = UIConfig(cache_ttl_seconds=300, max_cache_size=100)
        loader = DataLoader(config)
        
        # Add cache entries up to limit  
        for i in range(101):  # One over the limit
            loader._set_cache(f"key{i}", f"value{i}")
        
        # Cache should not exceed max size due to LRU eviction
        assert len(loader._cache) <= config.max_cache_size 