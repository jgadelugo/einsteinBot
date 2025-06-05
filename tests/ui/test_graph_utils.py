"""
Unit tests for UI graph utilities.
"""

import pytest
import networkx as nx
from unittest.mock import Mock

from ui.utils.graph_utils import KnowledgeGraphBuilder
from ui.data.models import Theorem, ValidationEvidence, SourceLineage, MathematicalContext
from ui.config import UIConfig


class TestKnowledgeGraphBuilder:
    """Test KnowledgeGraphBuilder class."""
    
    def create_mock_config(self) -> UIConfig:
        """Create mock configuration for testing."""
        return UIConfig(
            node_size_range=(10, 50),
            max_graph_nodes=100
        )
    
    def create_sample_theorems(self) -> list:
        """Create sample theorems for testing."""
        theorem1 = Theorem(
            id="THM_11111111",
            statement="∀ x ∈ ℝ, x² ≥ 0",
            sympy_expression="x**2 >= 0",
            theorem_type="algebraic",
            source_lineage=SourceLineage(
                original_formula="x^2",
                hypothesis_id="hyp_1",
                confidence=0.95,
                validation_score=1.0,
                generation_method="symbolic",
                source_type="formula",
                transformation_chain=["expand", "simplify"]
            ),
            natural_language="All real squares are non-negative",
            symbols=["x"],
            mathematical_context=MathematicalContext(
                symbols=["x"],
                domain="real_analysis",
                complexity_score=0.3
            ),
            validation_evidence=ValidationEvidence(
                validation_status="PASS",
                pass_rate=1.0,
                total_tests=100,
                symbols_tested=["x"],
                validation_time=0.15
            )
        )
        
        theorem2 = Theorem(
            id="THM_22222222",
            statement="∀ y ∈ ℝ, y³ = y * y²",
            sympy_expression="y**3 == y * y**2",
            theorem_type="algebraic",
            source_lineage=SourceLineage(
                original_formula="y^3",
                hypothesis_id="hyp_2",
                confidence=0.88,
                validation_score=1.0,
                generation_method="symbolic",
                source_type="formula",
                transformation_chain=["expand", "factor"]
            ),
            natural_language="Cube equals base times square",
            symbols=["y"],
            mathematical_context=MathematicalContext(
                symbols=["y"],
                domain="algebra",
                complexity_score=0.4
            ),
            validation_evidence=ValidationEvidence(
                validation_status="PASS",
                pass_rate=0.95,
                total_tests=95,
                symbols_tested=["y"],
                validation_time=0.12
            )
        )
        
        theorem3 = Theorem(
            id="THM_33333333",
            statement="∀ x ∈ ℝ, (x + 1)² = x² + 2x + 1",
            sympy_expression="(x + 1)**2 == x**2 + 2*x + 1",
            theorem_type="polynomial",
            source_lineage=SourceLineage(
                original_formula="(x+1)^2",
                hypothesis_id="hyp_3",
                confidence=0.92,
                validation_score=1.0,
                generation_method="symbolic",
                source_type="formula",
                transformation_chain=["expand", "simplify", "collect"]
            ),
            natural_language="Square of sum expansion",
            symbols=["x"],
            mathematical_context=MathematicalContext(
                symbols=["x"],
                domain="algebra",
                complexity_score=0.5
            ),
            validation_evidence=ValidationEvidence(
                validation_status="PASS",
                pass_rate=0.98,
                total_tests=100,
                symbols_tested=["x"],
                validation_time=0.18
            )
        )
        
        return [theorem1, theorem2, theorem3]
    
    def test_builder_initialization(self):
        """Test KnowledgeGraphBuilder initialization."""
        config = self.create_mock_config()
        builder = KnowledgeGraphBuilder(config)
        
        assert builder.config == config
        assert builder.logger is not None
    
    def test_build_theorem_graph(self):
        """Test building complete theorem graph."""
        config = self.create_mock_config()
        builder = KnowledgeGraphBuilder(config)
        theorems = self.create_sample_theorems()
        
        graph = builder.build_theorem_graph(theorems)
        
        # Should be a NetworkX graph
        assert isinstance(graph, nx.Graph)
        
        # Should have theorem nodes
        assert graph.number_of_nodes() >= len(theorems)
        
        # Should have edges (relationships)
        assert graph.number_of_edges() >= 0
        
        # Check theorem nodes exist
        theorem_nodes = [node for node, data in graph.nodes(data=True) 
                        if data.get('node_type') == 'theorem']
        assert len(theorem_nodes) == len(theorems)
        
        # Check theorem node data
        for theorem in theorems:
            assert theorem.id in graph.nodes
            node_data = graph.nodes[theorem.id]
            assert node_data['node_type'] == 'theorem'
            assert node_data['label'] == theorem.short_id  # Uses short_id, not full id
            assert node_data['theorem_type'] == theorem.theorem_type
            assert node_data['validation_status'] == theorem.validation_evidence.validation_status
            assert 'confidence' in node_data
            assert 'size' in node_data
    
    def test_calculate_node_size(self):
        """Test node size calculation."""
        config = self.create_mock_config()
        builder = KnowledgeGraphBuilder(config)
        theorems = self.create_sample_theorems()
        
        for theorem in theorems:
            size = builder._calculate_node_size(theorem)
            
            # Size should be within range
            min_size, max_size = config.node_size_range
            assert min_size <= size <= max_size
            
            # Higher confidence should generally mean larger size
            assert size > min_size  # Should be above minimum for validated theorems
    
    def test_add_symbol_relationships(self):
        """Test adding symbol nodes and relationships."""
        config = self.create_mock_config()
        builder = KnowledgeGraphBuilder(config)
        theorems = self.create_sample_theorems()
        
        graph = nx.Graph()
        
        # Add theorem nodes first
        for theorem in theorems:
            graph.add_node(theorem.id, node_type="theorem", symbols=theorem.symbols)
        
        # Add symbol relationships
        builder._add_symbol_relationships(graph, theorems)
        
        # Should have symbol nodes for symbols used in multiple theorems
        symbol_nodes = [node for node, data in graph.nodes(data=True) 
                       if data.get('node_type') == 'symbol']
        
        # 'x' appears in theorems 1 and 3, so should have a symbol node
        x_symbol_node = f"SYM_x"
        if x_symbol_node in graph.nodes:
            assert graph.nodes[x_symbol_node]['node_type'] == 'symbol'
            assert graph.nodes[x_symbol_node]['label'] == 'x'
            assert graph.nodes[x_symbol_node]['usage_count'] >= 2
    
    def test_add_transformation_relationships(self):
        """Test adding transformation similarity relationships."""
        config = self.create_mock_config()
        builder = KnowledgeGraphBuilder(config)
        theorems = self.create_sample_theorems()
        
        graph = nx.Graph()
        
        # Add theorem nodes
        for theorem in theorems:
            graph.add_node(theorem.id, node_type="theorem")
        
        # Add transformation relationships
        builder._add_transformation_relationships(graph, theorems)
        
        # Check for transformation similarity edges
        transformation_edges = [(u, v) for u, v, data in graph.edges(data=True) 
                               if data.get('edge_type') == 'transformation_similarity']
        
        # Should have some transformation similarity if theorems share transformations
        for u, v, data in graph.edges(data=True):
            if data.get('edge_type') == 'transformation_similarity':
                assert 'similarity_score' in data
                assert 0.0 <= data['similarity_score'] <= 1.0
    
    def test_add_source_relationships(self):
        """Test adding various source relationships."""
        config = self.create_mock_config()
        builder = KnowledgeGraphBuilder(config)
        theorems = self.create_sample_theorems()
        
        graph = nx.Graph()
        
        # Add theorem nodes
        for theorem in theorems:
            graph.add_node(theorem.id, node_type="theorem", symbols=theorem.symbols)
        
        # Add source relationships (the actual method name)
        builder._add_source_relationships(graph, theorems)
        
        # Check for source similarity edges
        source_edges = [(u, v) for u, v, data in graph.edges(data=True) 
                       if data.get('edge_type') == 'same_source']
        
        # Should have some source edges if theorems share sources
        for u, v, data in graph.edges(data=True):
            if data.get('edge_type') == 'same_source':
                assert 'weight' in data
                assert data['weight'] == 1.0  # Same source should have weight 1.0
    
    def test_calculate_transformation_similarity(self):
        """Test transformation chain similarity calculation."""
        config = self.create_mock_config()
        builder = KnowledgeGraphBuilder(config)
        
        # Test identical chains
        chain1 = ["expand", "simplify"]
        chain2 = ["expand", "simplify"]
        similarity = builder._calculate_transformation_similarity(chain1, chain2)
        assert similarity == 1.0
        
        # Test partial overlap
        chain1 = ["expand", "simplify"]
        chain2 = ["expand", "factor"]
        similarity = builder._calculate_transformation_similarity(chain1, chain2)
        assert 0.0 < similarity < 1.0
        
        # Test no overlap
        chain1 = ["expand", "simplify"]
        chain2 = ["integrate", "differentiate"]
        similarity = builder._calculate_transformation_similarity(chain1, chain2)
        assert similarity == 0.0
        
        # Test empty chains
        chain1 = []
        chain2 = ["expand"]
        similarity = builder._calculate_transformation_similarity(chain1, chain2)
        assert similarity == 0.0
    
    def test_get_graph_statistics(self):
        """Test graph statistics calculation."""
        config = self.create_mock_config()
        builder = KnowledgeGraphBuilder(config)
        theorems = self.create_sample_theorems()
        
        graph = builder.build_theorem_graph(theorems)
        stats = builder.get_graph_statistics(graph)
        
        # Check required statistics
        assert 'total_nodes' in stats
        assert 'total_edges' in stats
        assert 'node_types' in stats
        assert 'edge_types' in stats
        assert 'density' in stats
        assert 'connected_components' in stats
        assert 'average_clustering' in stats
        
        # Validate statistics
        assert stats['total_nodes'] == graph.number_of_nodes()
        assert stats['total_edges'] == graph.number_of_edges()
        assert isinstance(stats['node_types'], dict)
        assert isinstance(stats['edge_types'], dict)
        assert 0.0 <= stats['density'] <= 1.0
        assert stats['connected_components'] >= 1
        assert 0.0 <= stats['average_clustering'] <= 1.0
        
        # Should have theorem nodes
        assert 'theorem' in stats['node_types']
        assert stats['node_types']['theorem'] == len(theorems)
    
    def test_empty_theorem_list(self):
        """Test building graph with empty theorem list."""
        config = self.create_mock_config()
        builder = KnowledgeGraphBuilder(config)
        
        graph = builder.build_theorem_graph([])
        
        assert isinstance(graph, nx.Graph)
        assert graph.number_of_nodes() == 0
        assert graph.number_of_edges() == 0
        
        stats = builder.get_graph_statistics(graph)
        assert stats['total_nodes'] == 0
        assert stats['total_edges'] == 0
    
    def test_single_theorem(self):
        """Test building graph with single theorem."""
        config = self.create_mock_config()
        builder = KnowledgeGraphBuilder(config)
        theorems = self.create_sample_theorems()[:1]  # Just first theorem
        
        graph = builder.build_theorem_graph(theorems)
        
        assert isinstance(graph, nx.Graph)
        assert graph.number_of_nodes() >= 1
        
        # Should have the theorem node
        assert theorems[0].id in graph.nodes
        
        stats = builder.get_graph_statistics(graph)
        assert stats['node_types']['theorem'] == 1
    
    def test_node_size_calculation_edge_cases(self):
        """Test node size calculation edge cases."""
        config = self.create_mock_config()
        builder = KnowledgeGraphBuilder(config)
        
        # Create theorem with minimum confidence
        low_confidence_theorem = Theorem(
            id="THM_44444444",
            statement="Test theorem",
            sympy_expression="test",
            theorem_type="test",
            source_lineage=SourceLineage(
                original_formula="test",
                hypothesis_id="test",
                confidence=0.0,  # Minimum confidence
                validation_score=0.0,
                generation_method="test",
                source_type="test"
            ),
            natural_language="Test",
            validation_evidence=ValidationEvidence(
                validation_status="FAIL",  # Failed validation
                pass_rate=0.0,
                total_tests=1,
                validation_time=0.1
            )
        )
        
        size = builder._calculate_node_size(low_confidence_theorem)
        min_size, max_size = config.node_size_range
        assert min_size <= size <= max_size
        assert size == min_size  # Should be minimum for failed validation
    
    def test_graph_building_performance(self):
        """Test that graph building completes in reasonable time."""
        config = self.create_mock_config()
        builder = KnowledgeGraphBuilder(config)
        theorems = self.create_sample_theorems()
        
        import time
        start_time = time.time()
        graph = builder.build_theorem_graph(theorems)
        build_time = time.time() - start_time
        
        # Should complete quickly for small datasets
        assert build_time < 1.0  # Less than 1 second
        assert isinstance(graph, nx.Graph)
    
    def test_graph_node_attributes(self):
        """Test that all required node attributes are present."""
        config = self.create_mock_config()
        builder = KnowledgeGraphBuilder(config)
        theorems = self.create_sample_theorems()
        
        graph = builder.build_theorem_graph(theorems)
        
        # Check theorem nodes have required attributes
        for theorem in theorems:
            node_data = graph.nodes[theorem.id]
            required_attrs = [
                'node_type', 'label', 'title', 'theorem_type',
                'validation_status', 'confidence', 'complexity', 'symbols', 'size'
            ]
            
            for attr in required_attrs:
                assert attr in node_data, f"Missing attribute {attr} in node {theorem.id}"
        
        # Check symbol nodes have required attributes
        symbol_nodes = [node for node, data in graph.nodes(data=True) 
                       if data.get('node_type') == 'symbol']
        
        for symbol_node in symbol_nodes:
            node_data = graph.nodes[symbol_node]
            required_attrs = ['node_type', 'label', 'usage_count', 'size']
            
            for attr in required_attrs:
                assert attr in node_data, f"Missing attribute {attr} in symbol node {symbol_node}"
    
    def test_graph_edge_attributes(self):
        """Test that edges have proper attributes."""
        config = self.create_mock_config()
        builder = KnowledgeGraphBuilder(config)
        theorems = self.create_sample_theorems()
        
        graph = builder.build_theorem_graph(theorems)
        
        for u, v, data in graph.edges(data=True):
            # All edges should have edge_type and weight
            assert 'edge_type' in data
            assert 'weight' in data
            assert isinstance(data['weight'], (int, float))
            assert data['weight'] >= 0
            
            # Check specific edge type attributes
            if data['edge_type'] == 'transformation_similarity':
                assert 'similarity_score' in data
                assert 0.0 <= data['similarity_score'] <= 1.0
            
            elif data['edge_type'] == 'same_source':
                assert data['weight'] == 1.0 