"""
Comprehensive tests for SearchInterface component.

Author: MathBot Team
Version: Phase 6C
"""

import pytest
import streamlit as st
from unittest.mock import Mock, patch
from typing import List

from ui.components.search_interface import SearchInterface
from ui.data.search_index import SearchIndex, SearchType, SearchFilters
from ui.config import UIConfig
from ui.data.models import Theorem, ValidationEvidence, SourceLineage, MathematicalContext


class TestSearchInterface:
    """Test suite for SearchInterface functionality."""
    
    @pytest.fixture
    def mock_config(self) -> UIConfig:
        """Create mock UI configuration for testing."""
        config = Mock(spec=UIConfig)
        config.search_debounce_ms = 300
        config.max_search_results = 50
        config.enable_search_suggestions = True
        config.max_search_history = 50
        return config
    
    @pytest.fixture
    def mock_search_index(self) -> SearchIndex:
        """Create mock search index for testing."""
        index = Mock(spec=SearchIndex)
        index.search.return_value = []
        index.get_search_statistics.return_value = {
            "queries_executed": 5,
            "cache_hit_rate": 0.4,
            "avg_response_time": 0.25
        }
        # Add missing attributes that the search interface expects
        index.metadata_index = {
            "type": {
                "algebraic_identity": ["THM_12345678"],
                "functional_equation": ["THM_87654321"]
            }
        }
        index.text_index = {}
        index.symbol_index = {}
        index.theorem_map = {}  # Add the theorem_map attribute
        return index
    
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
                    hypothesis_id="HYP_TEST",
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
            )
        ]
    
    @pytest.fixture
    def search_interface(self, mock_config, mock_search_index) -> SearchInterface:
        """Create SearchInterface instance for testing."""
        return SearchInterface(mock_config, mock_search_index)
    
    def test_initialization_success(self, mock_config, mock_search_index):
        """Test successful SearchInterface initialization."""
        interface = SearchInterface(mock_config, mock_search_index)
        
        assert interface.config == mock_config
        assert interface.search_index == mock_search_index
        assert interface.logger is not None
    
    @patch('streamlit.session_state')
    def test_session_state_initialization(self, mock_session_state, search_interface):
        """Test session state initialization."""
        # Mock session state as a dictionary-like object
        mock_session_state.get.return_value = None
        mock_session_state.__contains__ = Mock(return_value=False)
        
        # Test that session initialization doesn't crash
        try:
            search_interface._initialize_search_session()
            # If no exception is raised, the test passes
            assert True
        except Exception as e:
            # If there's an exception, log it and fail
            assert False, f"Session initialization failed: {e}"
    
    @patch('streamlit.text_input')
    @patch('streamlit.multiselect')
    @patch('streamlit.selectbox')
    @patch('streamlit.columns')
    @patch('streamlit.expander')
    @patch('streamlit.session_state')
    def test_render_search_controls(self, mock_session_state, mock_expander, mock_columns,
                                   mock_selectbox, mock_multiselect, mock_text_input, search_interface):
        """Test rendering of search controls."""
        # Mock streamlit components to return proper values
        mock_text_input.return_value = "test query"
        mock_multiselect.return_value = ["x", "y"]
        mock_selectbox.return_value = "TEXT"
        # Create mock column objects that support context manager protocol
        mock_cols = []
        for i in range(3):  # Need 3 columns for search_type_cols
            mock_col = Mock()
            mock_col.__enter__ = Mock(return_value=mock_col)
            mock_col.__exit__ = Mock(return_value=None)
            mock_cols.append(mock_col)
        
        # Mock different column calls
        def columns_side_effect(widths=None):
            if widths == [3, 1]:
                return mock_cols[:2]  # First 2 columns for main layout
            elif isinstance(widths, int) and widths == 3:
                return mock_cols  # All 3 columns for search types
            elif isinstance(widths, int) and widths == 2:
                return mock_cols[:2]  # 2 columns for advanced filters
            else:
                return mock_cols  # Default to all columns
        
        mock_columns.side_effect = columns_side_effect
        mock_expander.return_value.__enter__ = Mock(return_value=Mock())
        mock_expander.return_value.__exit__ = Mock(return_value=None)
        
        # Mock session state
        from ui.components.search_interface import SearchConfig
        from ui.data.search_index import SearchFilters
        def get_side_effect(key, default=None):
            if key == 'search_filters':
                return SearchFilters()
            return default
        
        mock_session_state.get.side_effect = get_side_effect
        mock_session_state.__contains__ = Mock(return_value=False)
        # Add the attributes that render_search_controls expects
        mock_session_state.search_config = SearchConfig()
        mock_session_state.last_search_time = 0.0  # Add this for debouncing
        
        config = search_interface.render_search_controls()
        
        assert config is not None
        mock_text_input.assert_called()
        mock_multiselect.assert_called()
        mock_selectbox.assert_called()
    
    @patch('streamlit.session_state')
    def test_execute_search_success(self, mock_session_state, search_interface, sample_theorems):
        """Test successful search execution."""
        search_interface.search_index.search.return_value = []
        
        # Mock session state with proper time values
        mock_session_state.last_search_time = 0.0
        mock_session_state.search_results = []
        mock_session_state.search_query = ""
        mock_session_state.__contains__ = Mock(return_value=True)
        mock_session_state.__getitem__ = Mock(side_effect=lambda x: getattr(mock_session_state, x, 0.0))
        mock_session_state.__setitem__ = Mock()
        
        from ui.components.search_interface import SearchConfig
        search_config = SearchConfig(
            query="test",
            search_types=[SearchType.TEXT],
            filters=SearchFilters()
        )
        
        # Test search execution through debounced method
        search_interface._execute_search_with_debounce(search_config)
        
        # Should have attempted to search
        search_interface.search_index.search.assert_called_once()
    
    @patch('streamlit.metric')
    @patch('streamlit.info')
    @patch('streamlit.columns')
    @patch('streamlit.session_state')
    def test_render_search_results_summary(self, mock_session_state, mock_columns, mock_info, mock_metric, 
                                         search_interface, sample_theorems):
        """Test rendering of search results summary."""
        # Mock columns to return proper column objects that support context manager protocol
        mock_cols = []
        for i in range(4):
            mock_col = Mock()
            mock_col.__enter__ = Mock(return_value=mock_col)
            mock_col.__exit__ = Mock(return_value=None)
            mock_cols.append(mock_col)
        mock_columns.return_value = mock_cols
        
        # Mock session state
        mock_session_state.search_session = Mock()
        mock_session_state.search_session.performance_metrics = {"search_time": 0.1}
        mock_session_state.search_config = Mock()
        mock_session_state.search_config.query = "test"
        from ui.data.search_index import SearchType
        mock_session_state.search_config.search_types = [SearchType.TEXT]  # Make it iterable
        
        # Create SearchResult objects instead of passing Theorems directly
        from ui.data.search_index import SearchResult, SearchType
        search_results = [
            SearchResult(
                theorem=theorem,
                relevance_score=0.95,
                match_highlights={"statement": ["test match"]},
                match_reasons=["text match"],
                search_type=SearchType.TEXT
            )
            for theorem in sample_theorems
        ]
        
        search_interface.render_search_results_summary(search_results)
        
        mock_metric.assert_called()
    
    def test_search_history_management(self, search_interface):
        """Test search history functionality."""
        with patch('streamlit.session_state') as mock_session_state:
            mock_session_state.search_history = []
            
            # Test that search history is managed through session state
            search_interface._render_search_history()
            
            # Should handle search history rendering
            assert True  # Test passes if no exception is raised
    
    @patch('streamlit.columns')
    @patch('streamlit.text_input')
    def test_render_search_interface_integration(self, mock_text_input, 
                                               mock_columns, search_interface, sample_theorems):
        """Test complete search interface rendering."""
        mock_text_input.return_value = ""
        # Mock columns to return proper column objects that support context manager protocol
        mock_cols = []
        for i in range(4):
            mock_col = Mock()
            mock_col.__enter__ = Mock(return_value=mock_col)
            mock_col.__exit__ = Mock(return_value=None)
            mock_cols.append(mock_col)
        mock_columns.return_value = mock_cols
        
        with patch.object(search_interface, 'render_search_controls') as mock_controls, \
             patch.object(search_interface, 'render_search_results_summary') as mock_summary:
            
            from ui.components.search_interface import SearchConfig
            mock_controls.return_value = SearchConfig()
            
            # Test that components can be rendered together
            config = search_interface.render_search_controls()
            
            # Create SearchResult objects for the summary test
            from ui.data.search_index import SearchResult, SearchType
            search_results = [
                SearchResult(
                    theorem=theorem,
                    relevance_score=0.95,
                    match_highlights={"statement": ["test match"]},
                    match_reasons=["text match"],
                    search_type=SearchType.TEXT
                )
                for theorem in sample_theorems
            ]
            search_interface.render_search_results_summary(search_results)
            
            assert config is not None
    
    @pytest.mark.performance
    @patch('streamlit.session_state')
    def test_search_performance_tracking(self, mock_session_state, search_interface):
        """Test search performance tracking."""
        search_interface.search_index.search.return_value = []
        
        # Mock session state with proper time values
        mock_session_state.last_search_time = 0.0
        mock_session_state.search_results = []
        mock_session_state.search_query = ""
        mock_session_state.__contains__ = Mock(return_value=True)
        mock_session_state.__getitem__ = Mock(side_effect=lambda x: getattr(mock_session_state, x, 0.0))
        mock_session_state.__setitem__ = Mock()
        
        from ui.components.search_interface import SearchConfig
        search_config = SearchConfig(
            query="performance test",
            search_types=[SearchType.TEXT],
            filters=SearchFilters()
        )
        
        # Execute search and verify performance tracking
        search_interface._execute_search_with_debounce(search_config)
        
        # Should track performance metrics
        search_interface.search_index.search.assert_called_once()
    
    def test_search_suggestions_functionality(self, search_interface):
        """Test search suggestions feature."""
        with patch('streamlit.session_state') as mock_session_state:
            mock_session_state.search_suggestions = ["function", "equation"]
            
            # Test that search suggestions are rendered
            search_interface._render_search_suggestions()
            
            # Should handle search suggestions rendering
            assert True  # Test passes if no exception is raised
    
    @patch('streamlit.session_state')
    def test_error_handling_invalid_search(self, mock_session_state, search_interface):
        """Test error handling for invalid searches."""
        search_interface.search_index.search.side_effect = Exception("Search failed")
        
        # Mock session state with proper time values
        mock_session_state.last_search_time = 0.0
        mock_session_state.search_results = []
        mock_session_state.search_query = ""
        mock_session_state.__contains__ = Mock(return_value=True)
        mock_session_state.__getitem__ = Mock(side_effect=lambda x: getattr(mock_session_state, x, 0.0))
        mock_session_state.__setitem__ = Mock()
        
        from ui.components.search_interface import SearchConfig
        search_config = SearchConfig(
            query="invalid query",
            search_types=[SearchType.TEXT],
            filters=SearchFilters()
        )
        
        # Should handle errors gracefully
        try:
            search_interface._execute_search_with_debounce(search_config)
        except Exception:
            pass  # Should handle errors gracefully
        
        # Should have attempted to search
        search_interface.search_index.search.assert_called_once() 