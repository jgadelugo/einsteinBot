#!/usr/bin/env python3
"""
Integration tests for Phase 6A Foundation Layer.

These tests verify that all components work together correctly
and demonstrate the complete functionality of the data layer.
"""

import pytest
import time
from typing import List

from ui.config import get_ui_config, UIConfig
from ui.data.loaders import create_data_loaders, TheoremLoader, FormulaLoader, ValidationLoader
from ui.data.models import Theorem, FormulaData, ValidationReport
from ui.utils.graph_utils import KnowledgeGraphBuilder


class TestPhase6AIntegration:
    """Integration tests for the complete Phase 6A foundation layer."""
    
    @pytest.fixture
    def config(self) -> UIConfig:
        """Get UI configuration for testing."""
        return get_ui_config()
    
    @pytest.fixture
    def data_loaders(self, config):
        """Create data loaders for testing."""
        return create_data_loaders(config)
    
    def test_configuration_system(self, config):
        """Test that configuration system works correctly."""
        assert config.cache_ttl_seconds > 0
        assert config.max_graph_nodes > 0
        assert config.search_min_length >= 1
        assert 0.0 <= config.fuzzy_search_threshold <= 1.0
        
        # Test that file paths are properly set
        assert config.theorems_file.name == "theorems.json"
        assert "formulas" in config.formulas_file.name
        assert "validation" in config.validation_file.name
    
    def test_data_loader_creation(self, data_loaders):
        """Test that all data loaders are created successfully."""
        theorem_loader, formula_loader, validation_loader = data_loaders
        
        assert isinstance(theorem_loader, TheoremLoader)
        assert isinstance(formula_loader, FormulaLoader)
        assert isinstance(validation_loader, ValidationLoader)
    
    def test_data_loading(self, data_loaders):
        """Test loading real data from files."""
        theorem_loader, formula_loader, validation_loader = data_loaders
        
        # Load data
        theorems = theorem_loader.load_theorems()
        formulas = formula_loader.load_formulas()
        validation_report = validation_loader.load_validation_report()
        
        # Verify data types and basic structure
        assert isinstance(theorems, list)
        assert isinstance(formulas, list)
        
        # Should have some theorems (assuming test data exists)
        if theorems:
            assert all(isinstance(t, Theorem) for t in theorems)
            assert all(t.id.startswith("THM_") for t in theorems)
        
        # Should have some formulas
        if formulas:
            assert all(isinstance(f, FormulaData) for f in formulas)
        
        # Validation report should exist or be None
        if validation_report:
            assert isinstance(validation_report, ValidationReport)
    
    def test_search_functionality(self, data_loaders):
        """Test search functionality across the system."""
        theorem_loader, _, _ = data_loaders
        
        theorems = theorem_loader.load_theorems()
        
        if theorems:
            # Test search with common terms
            all_results = theorem_loader.search_theorems("theorem", limit=10)
            assert isinstance(all_results, list)
            assert len(all_results) <= 10
            
            # Test search with specific terms
            specific_results = theorem_loader.search_theorems("algebraic", limit=5)
            assert isinstance(specific_results, list)
            assert len(specific_results) <= 5
            
            # Test empty search
            empty_results = theorem_loader.search_theorems("nonexistent_term_xyz")
            assert isinstance(empty_results, list)
            assert len(empty_results) == 0
    
    def test_validation_summary(self, data_loaders):
        """Test validation summary generation."""
        theorem_loader, _, _ = data_loaders
        
        summary = theorem_loader.get_validation_summary()
        
        assert isinstance(summary, dict)
        assert "total" in summary
        assert "validated" in summary
        assert "pass_rate" in summary
        
        assert summary["total"] >= 0
        assert summary["validated"] >= 0
        assert summary["validated"] <= summary["total"]
        assert 0.0 <= summary["pass_rate"] <= 1.0
    
    def test_graph_building(self, config, data_loaders):
        """Test knowledge graph construction."""
        theorem_loader, _, _ = data_loaders
        
        theorems = theorem_loader.load_theorems()
        
        if theorems:
            graph_builder = KnowledgeGraphBuilder(config)
            graph = graph_builder.build_theorem_graph(theorems)
            
            # Verify graph structure
            assert graph.number_of_nodes() >= len(theorems)
            assert graph.number_of_edges() >= 0
            
            # Get graph statistics
            stats = graph_builder.get_graph_statistics(graph)
            
            assert isinstance(stats, dict)
            assert "total_nodes" in stats
            assert "total_edges" in stats
            assert "density" in stats
            assert "node_types" in stats
            assert "edge_types" in stats
            
            assert stats["total_nodes"] == graph.number_of_nodes()
            assert stats["total_edges"] == graph.number_of_edges()
            assert 0.0 <= stats["density"] <= 1.0
    
    def test_caching_performance(self, data_loaders):
        """Test that caching improves performance."""
        theorem_loader, _, _ = data_loaders
        
        # First load (cache miss)
        start_time = time.time()
        theorems1 = theorem_loader.load_theorems()
        first_load_time = time.time() - start_time
        
        # Second load (cache hit)
        start_time = time.time()
        theorems2 = theorem_loader.load_theorems()
        second_load_time = time.time() - start_time
        
        # Cache hit should be significantly faster
        assert second_load_time < first_load_time
        
        # Results should be identical
        assert len(theorems1) == len(theorems2)
        if theorems1:
            assert theorems1[0].id == theorems2[0].id
    
    def test_theorem_filtering(self, data_loaders):
        """Test theorem filtering by type and ID."""
        theorem_loader, _, _ = data_loaders
        
        theorems = theorem_loader.load_theorems()
        
        if theorems:
            # Test get by ID
            sample_theorem = theorems[0]
            found_theorem = theorem_loader.get_theorem_by_id(sample_theorem.id)
            assert found_theorem is not None
            assert found_theorem.id == sample_theorem.id
            
            # Test get by non-existent ID
            missing_theorem = theorem_loader.get_theorem_by_id("THM_NONEXISTENT")
            assert missing_theorem is None
            
            # Test get by type
            if theorems:
                theorem_type = theorems[0].theorem_type
                type_theorems = theorem_loader.get_theorems_by_type(theorem_type)
                assert isinstance(type_theorems, list)
                assert all(t.theorem_type == theorem_type for t in type_theorems)
    
    def test_data_model_validation(self, data_loaders):
        """Test that all loaded data passes Pydantic validation."""
        theorem_loader, formula_loader, validation_loader = data_loaders
        
        # Load and validate theorems
        theorems = theorem_loader.load_theorems()
        for theorem in theorems:
            assert isinstance(theorem, Theorem)
            assert theorem.id.startswith("THM_")
            assert len(theorem.id) == 12  # THM_ + 8 chars
            assert theorem.is_validated in [True, False]
            assert theorem.complexity_category in ["Simple", "Moderate", "Complex"]
            assert 0.0 <= theorem.source_lineage.confidence <= 1.0
        
        # Load and validate formulas
        formulas = formula_loader.load_formulas()
        for formula in formulas:
            assert isinstance(formula, FormulaData)
            assert len(formula.id) > 0
            assert len(formula.expression) > 0
        
        # Load and validate validation report
        validation_report = validation_loader.load_validation_report()
        if validation_report:
            assert isinstance(validation_report, ValidationReport)
            assert validation_report.summary.total_formulas >= 0
            assert 0.0 <= validation_report.summary.overall_pass_rate <= 1.0
    
    def test_error_handling(self, config):
        """Test graceful error handling for missing files."""
        # Create loaders with non-existent files
        bad_config = UIConfig(
            theorems_file=config.theorems_file.parent / "nonexistent_theorems.json",
            formulas_file=config.formulas_file.parent / "nonexistent_formulas.json",
            validation_file=config.validation_file.parent / "nonexistent_validation.json"
        )
        
        theorem_loader, formula_loader, validation_loader = create_data_loaders(bad_config)
        
        # Should return empty lists/None, not crash
        theorems = theorem_loader.load_theorems()
        formulas = formula_loader.load_formulas()
        validation_report = validation_loader.load_validation_report()
        
        assert theorems == []
        assert formulas == []
        assert validation_report is None
    
    def test_complete_workflow(self, config, data_loaders):
        """Test a complete workflow from data loading to graph visualization."""
        theorem_loader, formula_loader, validation_loader = data_loaders
        
        # Step 1: Load all data
        theorems = theorem_loader.load_theorems()
        formulas = formula_loader.load_formulas()
        validation_report = validation_loader.load_validation_report()
        
        # Step 2: Perform search
        if theorems:
            search_results = theorem_loader.search_theorems("theorem", limit=5)
            assert len(search_results) <= 5
        
        # Step 3: Get validation summary
        summary = theorem_loader.get_validation_summary()
        assert isinstance(summary, dict)
        
        # Step 4: Build knowledge graph
        if theorems:
            graph_builder = KnowledgeGraphBuilder(config)
            graph = graph_builder.build_theorem_graph(theorems)
            stats = graph_builder.get_graph_statistics(graph)
            
            # Verify the complete workflow succeeded
            assert graph.number_of_nodes() > 0
            assert isinstance(stats, dict)
            
        # Step 5: Test caching works across operations
        cached_theorems = theorem_loader.load_theorems()
        assert len(cached_theorems) == len(theorems)


# Additional integration test for specific scenarios
class TestPhase6APerformance:
    """Performance and scalability tests for Phase 6A."""
    
    def test_large_graph_handling(self):
        """Test that graph building handles reasonable sizes efficiently."""
        config = get_ui_config()
        theorem_loader, _, _ = create_data_loaders(config)
        
        theorems = theorem_loader.load_theorems()
        
        if len(theorems) > 0:
            graph_builder = KnowledgeGraphBuilder(config)
            
            start_time = time.time()
            graph = graph_builder.build_theorem_graph(theorems)
            build_time = time.time() - start_time
            
            # Should complete quickly for reasonable datasets
            assert build_time < 5.0  # Less than 5 seconds
            
            # Graph should respect node limits
            assert graph.number_of_nodes() <= config.max_graph_nodes + 20  # Allow some buffer for symbol nodes
    
    def test_memory_efficiency(self):
        """Test that memory usage is reasonable."""
        config = get_ui_config()
        theorem_loader, _, _ = create_data_loaders(config)
        
        # Load data multiple times to test memory handling
        for _ in range(5):
            theorems = theorem_loader.load_theorems()
            # Should use cache after first load
            assert isinstance(theorems, list)
        
        # Cache should not grow unbounded
        assert len(theorem_loader._cache) <= config.max_cache_size 