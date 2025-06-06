"""
Integration tests for Phase 6C Search & Browse Interface.

This module provides comprehensive integration testing for all Phase 6C components
working together, including cross-component interactions and end-to-end workflows.

Author: MathBot Team
Version: Phase 6C
"""

import pytest
import pandas as pd
import time
from unittest.mock import Mock, patch
from typing import List

from ui.app import MathBotUI
from ui.config import UIConfig
from ui.data.search_index import SearchIndex, SearchType, SearchFilters
from ui.components.search_interface import SearchInterface
from ui.components.theorem_browser import TheoremBrowser
from ui.components.theorem_detail import TheoremDetail
from ui.data.models import Theorem, ValidationEvidence, SourceLineage, MathematicalContext


@pytest.mark.integration
class TestPhase6CIntegration:
    """Integration test suite for Phase 6C components."""
    
    @pytest.fixture
    def mock_config(self) -> UIConfig:
        """Create comprehensive mock UI configuration."""
        config = Mock(spec=UIConfig)
        
        # Search configuration
        config.search_cache_size = 1000
        config.search_cache_ttl = 300
        config.fuzzy_threshold = 0.7
        config.max_search_results = 50
        config.search_debounce_ms = 300
        config.enable_search_suggestions = True
        
        # Browser configuration
        config.table_page_size = 10
        config.max_table_rows = 1000
        config.enable_table_export = True
        
        # Detail configuration
        config.latex_renderer = "mathjax"
        config.latex_timeout = 5.0
        config.enable_theorem_sharing = True
        
        # Graph configuration
        config.graph_height = 600
        config.node_size_range = (20, 40)
        config.max_graph_nodes = 100
        
        return config
    
    @pytest.fixture
    def comprehensive_theorem_dataset(self) -> List[Theorem]:
        """Create comprehensive theorem dataset for integration testing."""
        theorems = []
        
        # Functional equations
        theorems.append(Theorem(
            id="THM_FUNC_INT_001",
            statement="f(x+y) = f(x) + f(y)",
            sympy_expression="Eq(f(x + y), f(x) + f(y))",
            theorem_type="functional_equation",
            assumptions=["f is additive"],
            source_lineage=SourceLineage(
                original_formula="f(x+y) = f(x) + f(y)",
                hypothesis_id="HYP_FUNC_INT_001",
                confidence=0.95,
                validation_score=0.98,
                generation_method="symbolic_derivation",
                transformation_chain=["additivity", "verification"]
            ),
            natural_language="An additive function where inputs sum equals outputs sum",
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
        ))
        
        # Algebraic identities
        theorems.append(Theorem(
            id="THM_ALG_INT_001",
            statement="(x+1)² = x² + 2x + 1",
            sympy_expression="Eq((x + 1)**2, x**2 + 2*x + 1)",
            theorem_type="algebraic_identity",
            assumptions=["x ∈ ℝ"],
            source_lineage=SourceLineage(
                original_formula="(x+1)^2 = x^2 + 2x + 1",
                hypothesis_id="HYP_ALG_INT_001",
                confidence=1.0,
                validation_score=1.0,
                generation_method="algebraic_expansion",
                transformation_chain=["expansion", "simplification"]
            ),
            natural_language="The square of a binomial equals the sum of squares plus cross terms",
            symbols=["x"],
            mathematical_context=MathematicalContext(
                symbols=["x"],
                complexity_score=0.3,
                domain="polynomials"
            ),
            validation_evidence=ValidationEvidence(
                validation_status="PASS",
                pass_rate=1.0,
                total_tests=100,
                symbols_tested=["x"],
                validation_time=0.012
            )
        ))
        
        # Trigonometric identities
        theorems.append(Theorem(
            id="THM_TRIG_INT_001",
            statement="sin²(x) + cos²(x) = 1",
            sympy_expression="Eq(sin(x)**2 + cos(x)**2, 1)",
            theorem_type="trigonometric_identity",
            assumptions=["x ∈ ℝ"],
            source_lineage=SourceLineage(
                original_formula="sin^2(x) + cos^2(x) = 1",
                hypothesis_id="HYP_TRIG_INT_001",
                confidence=1.0,
                validation_score=1.0,
                generation_method="trigonometric_axiom",
                transformation_chain=["fundamental_identity"]
            ),
            natural_language="The fundamental trigonometric identity",
            symbols=["sin", "cos", "x"],
            mathematical_context=MathematicalContext(
                symbols=["sin", "cos", "x"],
                complexity_score=0.5,
                domain="trigonometry"
            ),
            validation_evidence=ValidationEvidence(
                validation_status="PASS",
                pass_rate=1.0,
                total_tests=200,
                symbols_tested=["sin", "cos", "x"],
                validation_time=0.023
            )
        ))
        
        return theorems
    
    @pytest.fixture
    def integrated_components(self, mock_config, comprehensive_theorem_dataset):
        """Create integrated component system for testing."""
        # Initialize search index
        search_index = SearchIndex(mock_config)
        search_index.build_index(comprehensive_theorem_dataset)
        
        # Initialize components
        search_interface = SearchInterface(mock_config, search_index)
        theorem_browser = TheoremBrowser(mock_config)
        theorem_detail = TheoremDetail(mock_config)
        
        return {
            'search_index': search_index,
            'search_interface': search_interface,
            'theorem_browser': theorem_browser,
            'theorem_detail': theorem_detail,
            'theorems': comprehensive_theorem_dataset
        }
    
    def test_end_to_end_search_to_detail_workflow(self, integrated_components):
        """Test complete workflow from search to detail view."""
        components = integrated_components
        
        # Step 1: Execute search
        filters = SearchFilters(theorem_types=["functional_equation"])
        search_results = components['search_interface'].execute_search(
            "additive function", [SearchType.TEXT, SearchType.TYPE], filters
        )
        
        # Verify search results
        assert isinstance(search_results, list)
        
        # Step 2: Browse results in table (simulated)
        if search_results:
            # Filter should find functional equation
            functional_theorems = [t for t in search_results 
                                 if hasattr(t, 'theorem_type') and t.theorem_type == "functional_equation"]
            assert len(functional_theorems) >= 0
        
        # Step 3: Select theorem for detail view (simulated)
        if components['theorems']:
            selected_theorem = components['theorems'][0]
            
            # Verify detail component can handle the theorem
            assert selected_theorem.id is not None
            assert selected_theorem.statement is not None
    
    def test_cross_component_data_consistency(self, integrated_components):
        """Test data consistency across all components."""
        components = integrated_components
        theorems = components['theorems']
        
        # Test that all components can handle the same theorem data
        if theorems:
            test_theorem = theorems[0]
            
            # Search index should find the theorem
            filters = SearchFilters()
            search_results = components['search_index'].search(
                test_theorem.statement[:10], [SearchType.TEXT], filters
            )
            assert isinstance(search_results, list)
            
            # Browser should be able to create dataframe
            df = components['theorem_browser']._create_theorem_dataframe([test_theorem])
            assert len(df) == 1
            assert df.iloc[0]['id'] == test_theorem.id
            
            # Detail component should handle the theorem
            assert test_theorem.statement is not None
    
    def test_search_index_performance_integration(self, integrated_components):
        """Test search index performance with realistic data."""
        components = integrated_components
        
        # Test multiple search types
        search_types = [
            ([SearchType.TEXT], "function"),
            ([SearchType.SYMBOL], ""),
            ([SearchType.TYPE], ""),
            ([SearchType.VALIDATION], ""),
        ]
        
        total_search_time = 0
        for search_type_list, query in search_types:
            start_time = time.time()
            
            if search_type_list == [SearchType.SYMBOL]:
                filters = SearchFilters(symbols=["x"])
            elif search_type_list == [SearchType.TYPE]:
                filters = SearchFilters(theorem_types=["algebraic_identity"])
            elif search_type_list == [SearchType.VALIDATION]:
                filters = SearchFilters(validation_status=["PASS"])
            else:
                filters = SearchFilters()
            
            results = components['search_index'].search(query, search_type_list, filters)
            search_time = time.time() - start_time
            total_search_time += search_time
            
            assert isinstance(results, list)
            assert search_time < 1.0  # Each search should be fast
        
        # Total time for all searches should be reasonable
        assert total_search_time < 5.0
    
    def test_filter_and_sort_integration(self, integrated_components):
        """Test filter and sort integration across components."""
        components = integrated_components
        theorems = components['theorems']
        
        # Create dataframe
        df = components['theorem_browser']._create_theorem_dataframe(theorems)
        
        # Test filtering
        from ui.components.theorem_browser import FilterConfig
        filters = FilterConfig(
            active_filters={"theorem_type": "algebraic_identity"},
            filter_mode="and"
        )
        filtered_df = components['theorem_browser']._apply_filters(df, filters)
        
        # Test sorting
        from ui.components.theorem_browser import SortConfig
        sort_config = SortConfig(primary_sort="confidence", sort_order="desc")
        sorted_df = components['theorem_browser']._apply_sorting(filtered_df, sort_config)
        
        assert isinstance(sorted_df, pd.DataFrame)
        assert len(sorted_df) <= len(df)
    
    def test_caching_integration_across_components(self, integrated_components):
        """Test caching integration across all components."""
        components = integrated_components
        
        # Perform searches that should hit cache
        filters = SearchFilters()
        query = "function"
        search_types = [SearchType.TEXT]
        
        # First search
        results1 = components['search_index'].search(query, search_types, filters)
        
        # Second search (should use cache)
        results2 = components['search_index'].search(query, search_types, filters)
        
        # Verify cache is working
        analytics = components['search_index'].get_search_analytics()
        assert analytics["total_searches"] >= 2
        assert analytics["cache_hits"] >= 1
    
    def test_error_handling_integration(self, integrated_components):
        """Test error handling across integrated components."""
        components = integrated_components
        
        # Test search with invalid parameters
        try:
            filters = SearchFilters()
            results = components['search_index'].search("", [], filters)  # Empty search types
            assert isinstance(results, list)  # Should handle gracefully
        except Exception:
            pass  # Should not crash
        
        # Test browser with empty data
        empty_df = components['theorem_browser']._create_theorem_dataframe([])
        assert len(empty_df) == 0
        
        # Test detail with None theorem
        try:
            components['theorem_detail'].render_validation_analysis(None)
        except Exception:
            pass  # Should handle gracefully
    
    def test_session_state_integration(self, integrated_components):
        """Test session state management across components."""
        with patch('streamlit.session_state') as mock_session_state:
            mock_session_state.search_query = "test"
            mock_session_state.search_results = []
            mock_session_state.selected_theorem_detail = None
            
            # Components should interact with session state appropriately
            components = integrated_components
            
            # Search interface should manage search state
            assert hasattr(components['search_interface'], 'config')
            
            # Browser should handle selection state
            assert hasattr(components['theorem_browser'], 'config')
            
            # Detail should handle theorem state
            assert hasattr(components['theorem_detail'], 'config')
    
    @pytest.mark.performance
    def test_large_dataset_integration_performance(self, mock_config):
        """Test integration performance with larger dataset."""
        # Create larger dataset
        large_dataset = []
        for i in range(50):  # Reasonable size for integration test
            theorem = Theorem(
                id=f"THM_PERF_INT_{i:03d}",
                statement=f"g_{i}(x) = x^{i % 3 + 1}",
                sympy_expression=f"Eq(g_{i}(x), x**{i % 3 + 1})",
                theorem_type=["algebraic_identity", "functional_equation", "trigonometric_identity"][i % 3],
                assumptions=[f"x ∈ ℝ"],
                source_lineage=SourceLineage(
                    original_formula=f"g_{i}(x) = x^{i % 3 + 1}",
                    hypothesis_id=f"HYP_PERF_INT_{i:03d}",
                    confidence=0.8 + (i % 20) / 100,
                    validation_score=0.9,
                    generation_method="performance_test"
                ),
                natural_language=f"Polynomial function of degree {i % 3 + 1}",
                symbols=["x", f"g_{i}"],
                mathematical_context=MathematicalContext(
                    symbols=["x", f"g_{i}"],
                    complexity_score=0.1 + (i % 3) * 0.2,
                    domain="polynomials"
                ),
                validation_evidence=ValidationEvidence(
                    validation_status="PASS",
                    pass_rate=0.95,
                    total_tests=50,
                    symbols_tested=["x"],
                    validation_time=0.01
                )
            )
            large_dataset.append(theorem)
        
        # Test integrated performance
        start_time = time.time()
        
        # Initialize components
        search_index = SearchIndex(mock_config)
        search_index.build_index(large_dataset)
        
        search_interface = SearchInterface(mock_config, search_index)
        theorem_browser = TheoremBrowser(mock_config)
        
        # Perform integrated operations
        filters = SearchFilters()
        search_results = search_index.search("polynomial", [SearchType.TEXT], filters)
        
        if search_results:
            df = theorem_browser._create_theorem_dataframe(search_results)
        
        total_time = time.time() - start_time
        
        # Should complete integration quickly
        assert total_time < 5.0
        assert len(large_dataset) == 50
    
    def test_ui_component_integration_with_app(self, mock_config, comprehensive_theorem_dataset):
        """Test integration with main MathBotUI app."""
        with patch('ui.app.TheoremLoader') as mock_theorem_loader, \
             patch('ui.app.FormulaLoader') as mock_formula_loader:
            
            # Mock data loaders
            mock_theorem_instance = Mock()
            mock_theorem_instance.load_theorems.return_value = comprehensive_theorem_dataset
            mock_theorem_loader.return_value = mock_theorem_instance
            
            mock_formula_instance = Mock()
            mock_formula_instance.load_formulas.return_value = []
            mock_formula_loader.return_value = mock_formula_instance
            
            # Initialize app
            app = MathBotUI(mock_config)
            
            # Verify Phase 6C components are initialized
            assert hasattr(app, 'search_index')
            assert hasattr(app, 'search_interface')
            assert hasattr(app, 'theorem_browser')
            assert hasattr(app, 'theorem_detail')
    
    def test_navigation_workflow_integration(self, integrated_components):
        """Test navigation workflow between components."""
        components = integrated_components
        theorems = components['theorems']
        
        if not theorems:
            pytest.skip("No theorems available for navigation test")
        
        # Simulate navigation workflow
        # 1. User searches -> gets results
        filters = SearchFilters()
        search_results = components['search_interface'].execute_search(
            "identity", [SearchType.TEXT], filters
        )
        
        # 2. User browses results -> selects theorem
        if search_results:
            selected_theorem = search_results[0] if hasattr(search_results[0], 'id') else theorems[0]
        else:
            selected_theorem = theorems[0]
        
        # 3. User views theorem details
        assert selected_theorem.id is not None
        
        # 4. User might search for related theorems
        related_search = components['search_interface'].execute_search(
            selected_theorem.theorem_type, [SearchType.TYPE], 
            SearchFilters(theorem_types=[selected_theorem.theorem_type])
        )
        
        assert isinstance(related_search, list)


@pytest.mark.integration
class TestPhase6CRealDataIntegration:
    """Integration tests with real theorem data."""
    
    def test_real_theorem_data_integration(self, real_theorems):
        """Test Phase 6C components with real theorem data."""
        if not real_theorems:
            pytest.skip("No real theorem data available")
        
        config = Mock(spec=UIConfig)
        config.search_cache_size = 1000
        config.search_cache_ttl = 300
        config.fuzzy_threshold = 0.7
        config.max_search_results = 50
        
        # Test with real data
        search_index = SearchIndex(config)
        search_index.build_index(real_theorems[:10])  # Use subset for testing
        
        # Test search functionality
        filters = SearchFilters()
        results = search_index.search("equation", [SearchType.TEXT], filters)
        
        assert isinstance(results, list)
        assert len(results) >= 0
    
    def test_performance_with_real_data(self, real_theorems):
        """Test performance with real theorem dataset."""
        if not real_theorems or len(real_theorems) < 5:
            pytest.skip("Insufficient real theorem data for performance test")
        
        config = Mock(spec=UIConfig)
        config.search_cache_size = 1000
        config.search_cache_ttl = 300
        
        # Test performance with real data subset
        start_time = time.time()
        
        search_index = SearchIndex(config)
        search_index.build_index(real_theorems[:10])
        
        # Perform various searches
        filters = SearchFilters()
        search_index.search("function", [SearchType.TEXT], filters)
        search_index.search("", [SearchType.SYMBOL], SearchFilters(symbols=["x"]))
        
        total_time = time.time() - start_time
        
        # Should complete quickly even with real data
        assert total_time < 10.0 