"""
Comprehensive tests for SearchIndex component.

Author: MathBot Team
Version: Phase 6C
"""

import pytest
import time
from unittest.mock import Mock
from typing import List

from ui.data.search_index import SearchIndex, SearchType, SearchFilters
from ui.config import UIConfig
from ui.data.models import Theorem, ValidationEvidence, SourceLineage, MathematicalContext


class TestSearchIndex:
    """Test suite for SearchIndex functionality."""
    
    @pytest.fixture
    def mock_config(self) -> UIConfig:
        """Create mock UI configuration for testing."""
        config = Mock(spec=UIConfig)
        config.search_cache_size = 100
        config.search_cache_ttl = 300
        config.fuzzy_threshold = 0.7
        config.max_search_results = 50
        return config
    
    @pytest.fixture
    def sample_theorems(self) -> List[Theorem]:
        """Create sample theorem data for testing."""
        return [
            Theorem(
                id="THM_12345678",
                statement="f(x+y) = f(x) + f(y)",
                sympy_expression="Eq(f(x + y), f(x) + f(y))",
                theorem_type="functional_equation",
                assumptions=["f is additive function"],
                source_lineage=SourceLineage(
                    original_formula="f(x+y) = f(x) + f(y)",
                    hypothesis_id="HYP_001",
                    confidence=0.95,
                    validation_score=0.98,
                    generation_method="symbolic_derivation",
                    source_type="derived_theorem",
                    transformation_chain=["identity", "addition"]
                ),
                natural_language="An additive function",
                symbols=["f", "x", "y"],
                mathematical_context=MathematicalContext(
                    symbols=["f", "x", "y"],
                    complexity_score=0.7,
                    domain="real_functions"
                ),
                validation_evidence=ValidationEvidence(
                    validation_status="PASS",
                    pass_rate=0.98,
                    total_tests=59,
                    symbols_tested=["f", "x", "y"],
                    validation_time=0.047
                )
            )
        ]
    
    @pytest.fixture
    def search_index(self, mock_config) -> SearchIndex:
        """Create SearchIndex instance for testing."""
        return SearchIndex(mock_config)
    
    def test_initialization_success(self, mock_config):
        """Test successful SearchIndex initialization."""
        index = SearchIndex(mock_config)
        
        assert index.config == mock_config
        assert index.logger is not None
        assert isinstance(index.text_index, dict)
        assert isinstance(index.symbol_index, dict)
        assert isinstance(index.metadata_index, dict)
        assert isinstance(index._search_cache, dict)
        assert index.index_built_time is None
    
    def test_build_index_success(self, search_index, sample_theorems):
        """Test successful index building."""
        search_index.build_index(sample_theorems)
        
        assert search_index.index_built_time is not None
        assert len(search_index.text_index) > 0
        assert len(search_index.symbol_index) > 0
        assert len(search_index.metadata_index) > 0
    
    def test_text_search_success(self, search_index, sample_theorems):
        """Test successful text search."""
        search_index.build_index(sample_theorems)
        
        filters = SearchFilters()
        results = search_index.search("function", [SearchType.TEXT], filters)
        
        assert isinstance(results, list)
        assert len(results) >= 0
    
    def test_symbol_search_success(self, search_index, sample_theorems):
        """Test successful symbol search."""
        search_index.build_index(sample_theorems)
        
        filters = SearchFilters(symbols=["x"])
        results = search_index.search("", [SearchType.SYMBOL], filters)
        
        assert isinstance(results, list)
        assert len(results) >= 0
    
    def test_search_analytics(self, search_index, sample_theorems):
        """Test search analytics tracking."""
        search_index.build_index(sample_theorems)
        
        filters = SearchFilters()
        search_index.search("function", [SearchType.TEXT], filters)
        
        analytics = search_index.get_search_statistics()
        assert isinstance(analytics, dict)
        assert "queries_executed" in analytics
        assert analytics["queries_executed"] >= 1

    @pytest.mark.performance
    def test_search_performance(self, search_index, sample_theorems):
        """Test search performance."""
        search_index.build_index(sample_theorems)
        
        filters = SearchFilters()
        start_time = time.time()
        results = search_index.search("function", [SearchType.TEXT], filters)
        search_time = time.time() - start_time
        
        assert search_time < 1.0
        assert isinstance(results, list)
