"""
Comprehensive tests for GraphViewer component.

This module provides extensive test coverage for the interactive graph visualization
component, including edge cases, error handling, and performance scenarios.

Author: MathBot Team
Version: Phase 6B
"""

import pytest
import plotly.graph_objects as go
import networkx as nx
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from ui.components.graph_viewer import GraphViewer
from ui.config import UIConfig
from ui.data.models import Theorem, ValidationEvidence, SourceLineage, MathematicalContext


class TestGraphViewer:
    """Comprehensive test suite for GraphViewer functionality."""
    
    @pytest.fixture
    def mock_config(self) -> UIConfig:
        """Create mock UI configuration for testing."""
        config = Mock(spec=UIConfig)
        config.graph_height = 600
        config.node_size_range = (20, 40)
        config.max_graph_nodes = 100
        config.cache_ttl_seconds = 300
        config.max_cache_size = 1000
        return config
    
    @pytest.fixture
    def sample_theorems(self) -> List[Theorem]:
        """Create sample theorem data for testing."""
        return [
            Theorem(
                id="THM_00000001",
                statement="f(x+y) = f(x) + f(y)",
                sympy_expression="Eq(f(x + y), f(x) + f(y))",
                theorem_type="functional_equation",
                assumptions=["f is a function"],
                source_lineage=SourceLineage(
                    original_formula="f(x+y) = f(x) + f(y)",
                    hypothesis_id="HYP_001",
                    confidence=0.95,
                    validation_score=0.98,
                    generation_method="symbolic_derivation",
                    source_type="mathematical_axiom",
                    transformation_chain=["identity", "addition"]
                ),
                natural_language="A function where the sum of inputs equals the sum of outputs",
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
                ),
                metadata={"difficulty": "intermediate"}
            ),
            Theorem(
                id="THM_00000002",
                statement="g(2*x) = 2*g(x)",
                sympy_expression="Eq(g(2*x), 2*g(x))",
                theorem_type="generalization",
                assumptions=["g is a function"],
                source_lineage=SourceLineage(
                    original_formula="g(2*x) = 2*g(x)",
                    hypothesis_id="HYP_002", 
                    confidence=0.87,
                    validation_score=0.92,
                    generation_method="pattern_generalization",
                    source_type="derived_theorem",
                    transformation_chain=["scaling", "multiplication"]
                ),
                natural_language="A function that scales linearly with input scaling",
                symbols=["g", "x"],
                mathematical_context=MathematicalContext(
                    symbols=["g", "x"],
                    complexity_score=0.5,
                    domain="linear_functions"
                ),
                validation_evidence=ValidationEvidence(
                    validation_status="FAIL",
                    pass_rate=0.73,
                    total_tests=42,
                    symbols_tested=["g", "x"],
                    validation_time=0.035
                ),
                metadata={"difficulty": "basic"}
            )
        ]
    
    @pytest.fixture
    def graph_viewer(self, mock_config) -> GraphViewer:
        """Create GraphViewer instance for testing."""
        with patch('ui.components.graph_viewer.KnowledgeGraphBuilder') as mock_builder:
            # Configure the mock builder
            mock_instance = Mock()
            mock_builder.return_value = mock_instance
            
            viewer = GraphViewer(mock_config)
            viewer.graph_builder = mock_instance
            
            return viewer
    
    def test_initialization_success(self, mock_config):
        """Test successful GraphViewer initialization."""
        with patch('ui.components.graph_viewer.KnowledgeGraphBuilder') as mock_builder:
            viewer = GraphViewer(mock_config)
            
            # Verify initialization
            assert viewer.config == mock_config
            assert viewer.logger is not None
            assert len(viewer.color_schemes) > 0
            assert len(viewer.layout_algorithms) > 0
            assert isinstance(viewer._layout_cache, dict)
            assert isinstance(viewer._graph_cache, dict)
            
            # Verify graph builder was created
            mock_builder.assert_called_once_with(mock_config)
    
    def test_initialization_with_invalid_config(self):
        """Test GraphViewer initialization with invalid configuration."""
        with patch('ui.components.graph_viewer.KnowledgeGraphBuilder') as mock_builder:
            mock_builder.side_effect = ValueError("Invalid configuration")
            
            with pytest.raises(ValueError, match="Invalid configuration"):
                GraphViewer(None)
    
    def test_render_empty_graph(self, graph_viewer):
        """Test rendering with no theorems."""
        fig = graph_viewer.render_interactive_graph([])
        
        assert isinstance(fig, go.Figure)
        assert "No Theorem Data Available" in str(fig.layout.title.text)
        assert len(fig.data) == 0  # No data traces for empty graph
    
    def test_render_with_theorems_success(self, graph_viewer, sample_theorems):
        """Test successful rendering with sample theorems."""
        # Setup mock graph builder
        mock_graph = Mock(spec=nx.Graph)
        mock_graph.number_of_nodes.return_value = 2
        mock_graph.number_of_edges.return_value = 1
        mock_graph.nodes.return_value = [
            ("THM_00000001", {"node_type": "theorem", "theorem_type": "functional_equation"}),
            ("THM_00000002", {"node_type": "theorem", "theorem_type": "generalization"})
        ]
        mock_graph.edges.return_value = [
            ("THM_00000001", "THM_00000002", {"edge_type": "relation"})
        ]
        
        graph_viewer.graph_builder.build_theorem_graph.return_value = mock_graph
        
        # Mock layout calculation
        positions = {
            "THM_00000001": (0.0, 1.0),
            "THM_00000002": (1.0, 0.0)
        }
        
        with patch.object(graph_viewer, '_calculate_layout', return_value=positions):
            fig = graph_viewer.render_interactive_graph(sample_theorems)
        
        # Verify results
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # Edge trace and node trace
        assert fig.layout.title.text == "Mathematical Knowledge Graph"
        
        # Verify graph builder was called
        graph_viewer.graph_builder.build_theorem_graph.assert_called_once_with(sample_theorems)
    
    def test_render_with_graph_builder_failure(self, graph_viewer, sample_theorems):
        """Test handling of graph builder failures."""
        graph_viewer.graph_builder.build_theorem_graph.side_effect = Exception("Graph building failed")
        
        fig = graph_viewer.render_interactive_graph(sample_theorems)
        
        assert isinstance(fig, go.Figure)
        assert "Graph Rendering Error" in str(fig.layout.title.text)
        assert "Graph building failed" in str(fig.layout.annotations[0].text)
    
    def test_caching_functionality(self, graph_viewer, sample_theorems):
        """Test graph caching mechanisms."""
        # Setup mock graph
        mock_graph = Mock(spec=nx.Graph)
        mock_graph.number_of_nodes.return_value = 1
        mock_graph.number_of_edges.return_value = 0
        mock_graph.nodes.return_value = [("THM_00000001", {"node_type": "theorem"})]
        mock_graph.edges.return_value = []
        
        graph_viewer.graph_builder.build_theorem_graph.return_value = mock_graph
        
        # Mock layout calculation
        positions = {"THM_00000001": (0.0, 0.0)}
        
        with patch.object(graph_viewer, '_calculate_layout', return_value=positions):
            # First call should build and cache
            fig1 = graph_viewer.render_interactive_graph(sample_theorems[:1])
            
            # Verify cache is populated
            assert len(graph_viewer._graph_cache) == 1
            
            # Second call should use cache
            fig2 = graph_viewer.render_interactive_graph(sample_theorems[:1])
            
            # Should get same figure from cache
            assert fig1 == fig2
    
    def test_layout_calculation_with_different_algorithms(self, graph_viewer):
        """Test layout calculation with various algorithms."""
        mock_graph = Mock(spec=nx.Graph)
        mock_graph.nodes.return_value = ["A", "B", "C"]
        mock_graph.number_of_nodes.return_value = 3
        
        # Test spring layout
        with patch('networkx.spring_layout') as mock_spring:
            mock_spring.return_value = {"A": (0, 0), "B": (1, 1), "C": (0.5, 0.5)}
            positions = graph_viewer._calculate_layout(mock_graph, "spring")
            
            assert len(positions) == 3
            mock_spring.assert_called_once()
        
        # Test circular layout
        with patch('networkx.circular_layout') as mock_circular:
            mock_circular.return_value = {"A": (1, 0), "B": (0, 1), "C": (-1, 0)}
            positions = graph_viewer._calculate_layout(mock_graph, "circular")
            
            assert len(positions) == 3
            mock_circular.assert_called_once()
        
        # Test invalid layout (should fallback to spring)
        with patch('networkx.spring_layout') as mock_spring_fallback:
            mock_spring_fallback.return_value = {"A": (0, 0), "B": (1, 1), "C": (0.5, 0.5)}
            positions = graph_viewer._calculate_layout(mock_graph, "invalid_layout")
            
            assert len(positions) == 3
            mock_spring_fallback.assert_called_once()
    
    def test_layout_calculation_error_handling(self, graph_viewer):
        """Test layout calculation error handling."""
        mock_graph = Mock(spec=nx.Graph)
        mock_graph.nodes.return_value = ["A"]
        mock_graph.number_of_nodes.return_value = 1
        
        # Test layout algorithm failure
        with patch('networkx.kamada_kawai_layout') as mock_layout:
            mock_layout.side_effect = Exception("Layout calculation failed")
            
            with patch('networkx.spring_layout') as mock_spring_fallback:
                mock_spring_fallback.return_value = {"A": (0, 0)}
                
                positions = graph_viewer._calculate_layout(mock_graph, "kamada_kawai")
                
                # Should fallback to spring layout
                assert len(positions) == 1
                mock_spring_fallback.assert_called_once()
    
    def test_filter_theorems_by_selection(self, graph_viewer, sample_theorems):
        """Test theorem filtering functionality."""
        # Filter by validation status
        validated = graph_viewer.filter_theorems_by_selection(
            sample_theorems, 
            selected_types=["all"],
            validation_filter="validated"
        )
        assert len(validated) == 1
        assert validated[0].id == "THM_00000001"
        
        # Filter by theorem type
        functional = graph_viewer.filter_theorems_by_selection(
            sample_theorems,
            selected_types=["functional_equation"]
        )
        assert len(functional) == 1
        assert functional[0].theorem_type == "functional_equation"
    
    def test_color_scheme_assignment(self, graph_viewer):
        """Test node color assignment based on different schemes."""
        theorem = Mock()
        theorem.theorem_type = "functional_equation"
        theorem.validation_evidence.validation_status = "PASS"
        theorem.source_lineage.confidence = 0.95
        
        node_data = {"node_type": "theorem"}
        
        # Test theorem_type coloring
        color = graph_viewer._get_node_color(node_data, "theorem_type", theorem)
        assert color == "#FF6B6B"  # Expected color for functional_equation
        
        # Test validation_status coloring
        color = graph_viewer._get_node_color(node_data, "validation_status", theorem)
        assert color == "#2ECC71"  # Expected color for PASS status
        
        # Test complexity coloring (high confidence)
        color = graph_viewer._get_node_color(node_data, "complexity", theorem)
        assert color == "#2ECC71"  # Expected color for high confidence
        
        # Test symbol node coloring
        symbol_data = {"node_type": "symbol"}
        color = graph_viewer._get_node_color(symbol_data, "theorem_type", None)
        assert color == "#9B59B6"  # Expected color for symbols
    
    def test_node_size_calculation(self, graph_viewer, sample_theorems):
        """Test node size calculation based on theorem properties."""
        theorem = sample_theorems[0]  # High confidence, validated theorem
        
        size = graph_viewer._calculate_node_size({}, theorem)
        
        # Should be larger than base size due to high confidence and validation
        assert size >= 20  # Base size
        assert size <= 40  # Max size
        
        # Test with unvalidated theorem
        unvalidated_theorem = sample_theorems[1]  # Failed validation
        unvalidated_size = graph_viewer._calculate_node_size({}, unvalidated_theorem)
        
        # Should be smaller than validated theorem
        assert unvalidated_size < size
    
    def test_hover_text_creation(self, graph_viewer, sample_theorems):
        """Test creation of rich hover text."""
        theorem = sample_theorems[0]
        
        hover_text = graph_viewer._create_hover_text("THM_00000001", {"node_type": "theorem"}, theorem)
        
        # Verify essential information is included
        assert theorem.id in hover_text
        assert theorem.theorem_type in hover_text
        assert "PASS" in hover_text
        assert "0.95" in hover_text  # Confidence
        
        # Test symbol hover text
        symbol_hover = graph_viewer._create_hover_text(
            "SYM_x", 
            {"node_type": "symbol", "label": "x", "usage_count": 3}, 
            None
        )
        assert "Symbol: x" in symbol_hover
        assert "3 theorems" in symbol_hover
    
    def test_performance_optimization(self, graph_viewer, sample_theorems):
        """Test performance optimization features."""
        # Test with small dataset (should return unchanged)
        optimized = graph_viewer.optimize_for_performance(sample_theorems, max_nodes=10)
        assert len(optimized) == len(sample_theorems)
        
        # Test memory usage tracking
        usage = graph_viewer.get_memory_usage()
        assert "layout_cache_size" in usage
        assert "graph_cache_size" in usage
    
    def test_memory_usage_tracking(self, graph_viewer):
        """Test memory usage statistics."""
        # Initially empty
        usage = graph_viewer.get_memory_usage()
        assert usage["layout_cache_size"] == 0
        assert usage["graph_cache_size"] == 0
        
        # Add some cache entries
        graph_viewer._layout_cache["test_key"] = {"A": (0, 0)}
        graph_viewer._graph_cache["test_fig"] = go.Figure()
        
        usage = graph_viewer.get_memory_usage()
        assert usage["layout_cache_size"] == 1
        assert usage["graph_cache_size"] == 1
    
    def test_cache_clearing(self, graph_viewer):
        """Test cache clearing functionality."""
        # Add cache entries
        graph_viewer._layout_cache["test_layout"] = {"A": (0, 0)}
        graph_viewer._graph_cache["test_graph"] = go.Figure()
        
        # Clear cache
        graph_viewer.clear_cache()
        
        # Verify caches are empty
        assert len(graph_viewer._layout_cache) == 0
        assert len(graph_viewer._graph_cache) == 0
    
    def test_node_neighbors_exploration(self, graph_viewer):
        """Test node neighbor exploration functionality."""
        # Create mock graph
        mock_graph = Mock(spec=nx.Graph)
        mock_graph.__contains__ = Mock(return_value=True)
        mock_graph.neighbors.side_effect = lambda x: {
            "A": ["B", "C"],
            "B": ["A", "D"],
            "C": ["A"],
            "D": ["B"]
        }.get(x, [])
        
        # Test depth 1 neighbors
        neighbors = graph_viewer.get_node_neighbors(mock_graph, "A", depth=1)
        assert neighbors == {"B", "C"}
        
        # Test depth 2 neighbors
        neighbors = graph_viewer.get_node_neighbors(mock_graph, "A", depth=2)
        assert neighbors == {"B", "C", "D"}
        
        # Test non-existent node
        mock_graph.__contains__ = Mock(return_value=False)
        neighbors = graph_viewer.get_node_neighbors(mock_graph, "X", depth=1)
        assert neighbors == set()
    
    def test_edge_cases_and_error_conditions(self, graph_viewer):
        """Test various edge cases and error conditions."""
        # Test with None theorems
        fig = graph_viewer.render_interactive_graph(None)
        assert isinstance(fig, go.Figure)
        
        # Test with empty graph from builder
        mock_graph = Mock(spec=nx.Graph)
        mock_graph.number_of_nodes.return_value = 0
        graph_viewer.graph_builder.build_theorem_graph.return_value = mock_graph
        
        fig = graph_viewer.render_interactive_graph([])
        assert isinstance(fig, go.Figure)
        
        # Test with malformed theorem data
        malformed_theorem = Mock()
        malformed_theorem.id = None  # Invalid ID
        
        try:
            graph_viewer.filter_theorems_by_selection([malformed_theorem], ["all"])
        except Exception:
            pass  # Should handle gracefully
    
    def test_cache_key_generation(self, graph_viewer, sample_theorems):
        """Test cache key generation for consistent caching."""
        params1 = {"layout_algorithm": "spring", "color_by": "theorem_type"}
        params2 = {"layout_algorithm": "spring", "color_by": "theorem_type"}
        params3 = {"layout_algorithm": "circular", "color_by": "theorem_type"}
        
        key1 = graph_viewer._generate_cache_key(sample_theorems, params1)
        key2 = graph_viewer._generate_cache_key(sample_theorems, params2)
        key3 = graph_viewer._generate_cache_key(sample_theorems, params3)
        
        # Same parameters should generate same key
        assert key1 == key2
        
        # Different parameters should generate different keys
        assert key1 != key3
        
        # Keys should be strings of reasonable length
        assert isinstance(key1, str)
        assert len(key1) > 0
    
    @pytest.mark.parametrize("selected_node", [None, "THM_00000001", "invalid_node"])
    def test_selection_highlighting(self, graph_viewer, sample_theorems, selected_node):
        """Test selection highlighting with various node selections."""
        mock_graph = Mock(spec=nx.Graph)
        mock_graph.number_of_nodes.return_value = 1
        mock_graph.number_of_edges.return_value = 0
        mock_graph.nodes.return_value = [("THM_00000001", {"node_type": "theorem"})]
        mock_graph.edges.return_value = []
        mock_graph.__contains__ = Mock(return_value=selected_node == "THM_00000001")
        
        graph_viewer.graph_builder.build_theorem_graph.return_value = mock_graph
        
        positions = {"THM_00000001": (0.0, 0.0)}
        
        with patch.object(graph_viewer, '_calculate_layout', return_value=positions):
            fig = graph_viewer.render_interactive_graph(
                sample_theorems[:1],
                selected_node=selected_node
            )
        
        assert isinstance(fig, go.Figure)
        
        # Check if selection annotation was added for valid selections
        if selected_node == "THM_00000001":
            assert len(fig.layout.shapes) > 0  # Selection highlight shape
        else:
            assert len(fig.layout.shapes) == 0  # No selection highlight 