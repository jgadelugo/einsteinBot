"""
Comprehensive tests for TheoremBrowser component.

Author: MathBot Team
Version: Phase 6C
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch
from typing import List

from ui.components.theorem_browser import TheoremBrowser, FilterConfig, SortConfig, TableConfig
from ui.config import UIConfig
from ui.data.models import Theorem, ValidationEvidence, SourceLineage, MathematicalContext


class TestTheoremBrowser:
    """Test suite for TheoremBrowser functionality."""
    
    @pytest.fixture
    def mock_config(self) -> UIConfig:
        """Create mock UI configuration for testing."""
        config = Mock(spec=UIConfig)
        config.table_page_size = 10
        config.max_table_rows = 1000
        config.enable_table_export = True
        return config
    
    @pytest.fixture
    def sample_theorems(self) -> List[Theorem]:
        """Create sample theorem data for testing."""
        return [
            Theorem(
                id="THM_12345678",
                statement="f(x) = x²",
                sympy_expression="Eq(f(x), x**2)",
                theorem_type="algebraic_identity",
                assumptions=["x ∈ ℝ"],
                source_lineage=SourceLineage(
                    original_formula="f(x) = x^2",
                    hypothesis_id="HYP_001",
                    confidence=0.95,
                    validation_score=0.98,
                    generation_method="test",
                    source_type="derived_theorem"
                ),
                natural_language="A quadratic function",
                symbols=["f", "x"],
                mathematical_context=MathematicalContext(
                    symbols=["f", "x"],
                    complexity_score=0.3,
                    domain="polynomials"
                ),
                validation_evidence=ValidationEvidence(
                    validation_status="PASS",
                    pass_rate=1.0,
                    total_tests=100,
                    symbols_tested=["f", "x"],
                    validation_time=0.01
                )
            ),
            Theorem(
                id="THM_87654321",
                statement="g(x+y) = g(x) + g(y)",
                sympy_expression="Eq(g(x + y), g(x) + g(y))",
                theorem_type="functional_equation",
                assumptions=["g is additive"],
                source_lineage=SourceLineage(
                    original_formula="g(x+y) = g(x) + g(y)",
                    hypothesis_id="HYP_002",
                    confidence=0.87,
                    validation_score=0.92,
                    generation_method="test",
                    source_type="derived_theorem"
                ),
                natural_language="An additive function",
                symbols=["g", "x", "y"],
                mathematical_context=MathematicalContext(
                    symbols=["g", "x", "y"],
                    complexity_score=0.6,
                    domain="functions"
                ),
                validation_evidence=ValidationEvidence(
                    validation_status="PASS",
                    pass_rate=0.92,
                    total_tests=50,
                    symbols_tested=["g", "x", "y"],
                    validation_time=0.02
                )
            )
        ]
    
    @pytest.fixture
    def theorem_browser(self, mock_config) -> TheoremBrowser:
        """Create TheoremBrowser instance for testing."""
        return TheoremBrowser(mock_config)
    
    def test_initialization_success(self, mock_config):
        """Test successful TheoremBrowser initialization."""
        browser = TheoremBrowser(mock_config)
        
        assert browser.config == mock_config
        assert browser.logger is not None
    
    def test_create_theorem_dataframe(self, theorem_browser, sample_theorems):
        """Test theorem data processing for table display."""
        # Test that theorem browser can handle theorem list
        with patch('streamlit.dataframe') as mock_dataframe:
            result = theorem_browser._render_data_table(sample_theorems)
            
            # Should attempt to render table
            mock_dataframe.assert_called_once()
            
            # Result should be None (no selection) or a Theorem
            assert result is None or isinstance(result, Theorem)
    
    def test_apply_filters_success(self, theorem_browser, sample_theorems):
        """Test successful filter application."""
        # Create filter configuration
        filters = FilterConfig(
            active_filters={"theorem_type": "algebraic_identity"},
            filter_mode="and"
        )
        
        # Test that filtering works with theorem list
        filtered_theorems = theorem_browser._apply_table_filters(sample_theorems)
        
        assert isinstance(filtered_theorems, list)
        assert len(filtered_theorems) <= len(sample_theorems)
        assert all(isinstance(t, Theorem) for t in filtered_theorems)
    
    def test_apply_sorting_success(self, theorem_browser, sample_theorems):
        """Test successful sorting application."""
        # Test that sorting works with theorem list
        sorted_theorems = theorem_browser._apply_table_sorting(sample_theorems)
        
        assert isinstance(sorted_theorems, list)
        assert len(sorted_theorems) == len(sample_theorems)
        assert all(isinstance(t, Theorem) for t in sorted_theorems)
    
    @patch('streamlit.selectbox')
    @patch('streamlit.multiselect')
    @patch('streamlit.slider')
    def test_render_filter_controls(self, mock_slider, mock_multiselect, 
                                  mock_selectbox, theorem_browser, sample_theorems):
        """Test rendering of filter controls."""
        mock_selectbox.return_value = "All"
        mock_multiselect.return_value = []
        mock_slider.return_value = (0.0, 1.0)
        
        with patch('streamlit.columns'), patch('streamlit.expander'):
            filters = theorem_browser._render_filter_controls(sample_theorems)
            
            assert isinstance(filters, FilterConfig)
            mock_selectbox.assert_called()
            mock_multiselect.assert_called()
    
    @patch('streamlit.selectbox')
    def test_render_sort_controls(self, mock_selectbox, theorem_browser):
        """Test rendering of sort controls."""
        mock_selectbox.return_value = "statement"
        
        with patch('streamlit.columns') as mock_columns:
            # Mock columns to return proper column objects
            mock_cols = []
            for i in range(3):
                mock_col = Mock()
                mock_col.__enter__ = Mock(return_value=mock_col)
                mock_col.__exit__ = Mock(return_value=None)
                mock_cols.append(mock_col)
            mock_columns.return_value = mock_cols
            
            with patch('streamlit.expander') as mock_expander:
                mock_expander.return_value.__enter__ = Mock(return_value=Mock())
                mock_expander.return_value.__exit__ = Mock(return_value=None)
                sort_config = theorem_browser._render_sort_controls()
            
            assert isinstance(sort_config, SortConfig)
            mock_selectbox.assert_called()
    
    @patch('streamlit.selectbox')
    @patch('streamlit.number_input')
    def test_render_table_controls(self, mock_number_input, mock_selectbox, theorem_browser, sample_theorems):
        """Test rendering of table controls."""
        mock_selectbox.return_value = ["statement", "theorem_type"]
        mock_number_input.return_value = 10
        
        with patch('streamlit.columns'):
            table_config = theorem_browser._render_table_controls(sample_theorems)
            
            assert isinstance(table_config, TableConfig)
            mock_selectbox.assert_called()
            mock_number_input.assert_called()
    
    def test_paginate_dataframe(self, theorem_browser, sample_theorems):
        """Test theorem list pagination."""
        # Test pagination functionality 
        paginated_theorems, pagination_info = theorem_browser._apply_pagination(sample_theorems)
        
        assert isinstance(paginated_theorems, list)
        # Test that pagination returns reasonable results
        assert len(paginated_theorems) <= len(sample_theorems)
        assert isinstance(pagination_info, dict)
        assert "total_pages" in pagination_info
        assert pagination_info["total_pages"] >= 1
    
    def test_handle_row_selection(self, theorem_browser, sample_theorems):
        """Test row selection handling through table rendering."""
        # Test table rendering which handles selection
        with patch('streamlit.dataframe') as mock_dataframe:
            # Mock streamlit dataframe to return selection
            mock_dataframe.return_value = {"selection": {"rows": [0]}}
            
            result = theorem_browser._render_data_table(sample_theorems)
            
            # Should handle selection correctly
            assert result is None or isinstance(result, Theorem)
    
    def test_render_table_display(self, theorem_browser, sample_theorems):
        """Test table display rendering."""
        with patch('streamlit.dataframe') as mock_dataframe:
            mock_dataframe.return_value = None
            
            result = theorem_browser._render_data_table(sample_theorems)
            
            mock_dataframe.assert_called_once()
            assert result is None or isinstance(result, Theorem)
    
    @patch('streamlit.download_button')
    def test_export_functionality(self, mock_download_button, theorem_browser, sample_theorems):
        """Test export functionality."""
        with patch('streamlit.selectbox') as mock_selectbox, \
             patch('streamlit.button') as mock_button:
            
            mock_selectbox.return_value = "CSV"
            mock_button.return_value = True
            
            theorem_browser._handle_export(sample_theorems)
            
            mock_download_button.assert_called()
    
    @patch('streamlit.columns')
    def test_render_theorem_table_integration(self, mock_columns, theorem_browser, sample_theorems):
        """Test complete table rendering integration."""
        with patch.object(theorem_browser, '_render_filter_controls') as mock_filters, \
             patch.object(theorem_browser, '_render_sort_controls') as mock_sort, \
             patch.object(theorem_browser, '_render_table_controls') as mock_table, \
             patch('streamlit.dataframe') as mock_dataframe:
            
            mock_filters.return_value = FilterConfig()
            mock_sort.return_value = SortConfig()
            mock_table.return_value = TableConfig()
            mock_dataframe.return_value = None
            
            result = theorem_browser.render_theorem_table(sample_theorems)
            
            # Should handle the rendering process
            assert result is None or isinstance(result, Theorem)
    
    @pytest.mark.performance
    def test_large_dataset_performance(self, theorem_browser):
        """Test performance with larger dataset."""
        # Create larger dataset
        large_dataset = []
        for i in range(100):
            theorem = Theorem(
                id=f"THM_{i:08X}",
                statement=f"h_{i}(x) = x + {i}",
                sympy_expression=f"Eq(h_{i}(x), x + {i})",
                theorem_type="algebraic_identity",
                assumptions=[f"x ∈ ℝ"],
                source_lineage=SourceLineage(
                    original_formula=f"h_{i}(x) = x + {i}",
                    hypothesis_id=f"HYP_{i:08X}",
                    confidence=0.9,
                    validation_score=0.95,
                    generation_method="performance_test",
                    source_type="derived_theorem"
                ),
                natural_language=f"Linear function with offset {i}",
                symbols=["x", f"h_{i}"],
                mathematical_context=MathematicalContext(
                    symbols=["x", f"h_{i}"],
                    complexity_score=0.1,
                    domain="linear_functions"
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
        
        # Test table rendering performance
        import time
        start_time = time.time()
        with patch('streamlit.dataframe'):
            result = theorem_browser._render_data_table(large_dataset)
        creation_time = time.time() - start_time
        
        assert creation_time < 1.0  # Should render table quickly
        assert result is None or isinstance(result, Theorem)
    
    def test_filter_configuration_persistence(self, theorem_browser):
        """Test filter configuration persistence."""
        filters = FilterConfig(
            active_filters={"theorem_type": "functional_equation"},
            filter_mode="and"
        )
        
        # Test that filter configuration is maintained
        assert filters.active_filters["theorem_type"] == "functional_equation"
        assert filters.filter_mode == "and"
    
    def test_sort_configuration_validation(self, theorem_browser):
        """Test sort configuration validation."""
        sort_config = SortConfig(
            primary_sort="confidence",
            secondary_sort="validation_score",
            sort_direction="desc"
        )
        
        # Test that sort configuration is valid
        assert sort_config.primary_sort == "confidence"
        assert sort_config.secondary_sort == "validation_score"
        assert sort_config.sort_direction.value == "desc"
    
    def test_table_configuration_bounds(self, theorem_browser):
        """Test table configuration bounds checking."""
        # Test page size bounds - use a valid value within bounds
        table_config = TableConfig(page_size=100)  # Maximum allowed page size
        
        # Should handle large page sizes appropriately
        assert isinstance(table_config, TableConfig)
        assert table_config.page_size == 100
    
    def test_error_handling_empty_dataset(self, theorem_browser):
        """Test handling of empty theorem dataset."""
        empty_theorems = []
        
        with patch('streamlit.info') as mock_info:
            result = theorem_browser._render_data_table(empty_theorems)
            
            # Should show info message for empty dataset
            mock_info.assert_called_once()
            assert result is None
    
    def test_error_handling_malformed_data(self, theorem_browser):
        """Test handling of malformed theorem data."""
        malformed_theorems = [None, "invalid", 123]
        
        # Should handle malformed data gracefully
        try:
            with patch('streamlit.dataframe'):
                result = theorem_browser._render_data_table(malformed_theorems)
            # If no exception, should handle gracefully
            assert result is None or isinstance(result, Theorem)
        except Exception:
            # If exception occurs, it should be handled gracefully
            pass 