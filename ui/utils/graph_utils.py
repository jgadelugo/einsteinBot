"""
Graph utilities for MathBot UI.

This module provides NetworkX graph building functionality for creating
knowledge graphs from theorem relationships and mathematical concepts.
"""

import networkx as nx
from typing import Dict, List, Set, Tuple, Optional, Any, Union

from config import logger
from ui.data.models import Theorem
from ui.config import UIConfig


class KnowledgeGraphBuilder:
    """Builder for mathematical knowledge graphs."""
    
    def __init__(self, config: UIConfig):
        self.config = config
        self.logger = logger.getChild("KnowledgeGraphBuilder")
    
    def build_theorem_graph(self, theorems: List[Theorem]) -> nx.Graph:
        """
        Build a knowledge graph from theorems.
        
        Args:
            theorems: List of theorem objects
            
        Returns:
            NetworkX graph with theorems and relationships
        """
        self.logger.info(f"Building knowledge graph from {len(theorems)} theorems")
        
        graph = nx.Graph()
        
        # Limit number of nodes if too many
        limited_theorems = theorems[:self.config.max_graph_nodes]
        if len(limited_theorems) < len(theorems):
            self.logger.info(f"Limited to {len(limited_theorems)} theorems for visualization")
        
        # Add theorem nodes
        self._add_theorem_nodes(graph, limited_theorems)
        
        # Add symbol nodes and edges
        self._add_symbol_relationships(graph, limited_theorems)
        
        # Add transformation chain relationships
        self._add_transformation_relationships(graph, limited_theorems)
        
        # Add type similarity relationships
        self._add_type_relationships(graph, limited_theorems)
        
        # Add source similarity relationships
        self._add_source_relationships(graph, limited_theorems)
        
        self.logger.info(f"Built graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
        
        return graph
    
    def _add_theorem_nodes(self, graph: nx.Graph, theorems: List[Theorem]) -> None:
        """Add theorem nodes to the graph."""
        for theorem in theorems:
            graph.add_node(
                theorem.id,
                node_type="theorem",
                label=theorem.short_id,
                title=theorem.display_statement[:100] + "..." if len(theorem.display_statement) > 100 else theorem.display_statement,
                theorem_type=theorem.theorem_type,
                theorem_type_display=theorem.theorem_type_display,
                validation_status=theorem.validation_evidence.validation_status,
                confidence=theorem.source_lineage.confidence,
                complexity=theorem.complexity_category,
                symbols=theorem.symbols,
                size=self._calculate_node_size(theorem),
                color=self._get_node_color(theorem),
                shape="dot"
            )
    
    def _calculate_node_size(self, theorem: Theorem) -> int:
        """Calculate node size based on theorem properties."""
        base_size = self.config.node_size_range[0]
        max_size = self.config.node_size_range[1]
        
        # Size based on confidence and validation
        confidence_factor = theorem.source_lineage.confidence
        validation_factor = 1.0 if theorem.is_validated else 0.7
        
        # Boost for complexity
        complexity_boost = {
            "Simple": 1.0,
            "Moderate": 1.2,
            "Complex": 1.4
        }.get(theorem.complexity_category, 1.0)
        
        size = base_size + (max_size - base_size) * confidence_factor * validation_factor * complexity_boost
        return int(min(size, max_size))
    
    def _get_node_color(self, theorem: Theorem) -> str:
        """Get node color based on theorem type and validation."""
        # Base color by theorem type
        type_colors = {
            "algebraic_identity": "#FF6B6B",     # Red
            "functional_equation": "#4ECDC4",    # Teal
            "generalization": "#45B7D1",        # Blue
            "composition": "#96CEB4",           # Green
            "transformation": "#FFEAA7",        # Yellow
        }
        
        base_color = type_colors.get(theorem.theorem_type, "#DDA0DD")  # Plum default
        
        # Darken if not validated
        if not theorem.is_validated:
            # Convert to darker shade (simplified)
            return base_color.replace("FF", "CC").replace("4E", "3A").replace("45", "35").replace("96", "76").replace("FF", "CC")
        
        return base_color
    
    def _add_symbol_relationships(self, graph: nx.Graph, theorems: List[Theorem]) -> None:
        """Add symbol nodes and theorem-symbol relationships."""
        symbol_counts = {}
        
        # Count symbol usage
        for theorem in theorems:
            for symbol in theorem.symbols:
                symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
        
        # Add symbol nodes for frequently used symbols
        for symbol, count in symbol_counts.items():
            if count >= 2:  # Only symbols used in multiple theorems
                graph.add_node(
                    f"SYM_{symbol}",
                    node_type="symbol",
                    label=symbol,
                    title=f"Symbol '{symbol}' used in {count} theorems",
                    usage_count=count,
                    size=min(self.config.node_size_range[1], 15 + count * 5),
                    color="#FFE4E1",  # Misty Rose
                    shape="triangle"
                )
                
                # Add edges to theorems using this symbol
                for theorem in theorems:
                    if symbol in theorem.symbols:
                        graph.add_edge(
                            theorem.id,
                            f"SYM_{symbol}",
                            edge_type="contains_symbol",
                            weight=1.0,
                            color="#E0E0E0",  # Light gray
                            width=1
                        )
    
    def _add_transformation_relationships(self, graph: nx.Graph, theorems: List[Theorem]) -> None:
        """Add edges between theorems with similar transformation chains."""
        for i, theorem1 in enumerate(theorems):
            for theorem2 in theorems[i+1:]:
                similarity = self._calculate_transformation_similarity(
                    theorem1.source_lineage.transformation_chain,
                    theorem2.source_lineage.transformation_chain
                )
                
                if similarity > 0.5:  # Threshold for similarity
                    graph.add_edge(
                        theorem1.id,
                        theorem2.id,
                        edge_type="transformation_similarity",
                        weight=similarity,
                        similarity_score=similarity,
                        color="#FFA500",  # Orange
                        width=int(similarity * 3) + 1,
                        title=f"Transformation similarity: {similarity:.2f}"
                    )
    
    def _add_type_relationships(self, graph: nx.Graph, theorems: List[Theorem]) -> None:
        """Add edges between theorems of the same type."""
        type_groups = {}
        
        # Group theorems by type
        for theorem in theorems:
            theorem_type = theorem.theorem_type
            if theorem_type not in type_groups:
                type_groups[theorem_type] = []
            type_groups[theorem_type].append(theorem)
        
        # Add edges within each type group
        for theorem_type, type_theorems in type_groups.items():
            if len(type_theorems) > 1:
                for i, theorem1 in enumerate(type_theorems):
                    for theorem2 in type_theorems[i+1:]:
                        graph.add_edge(
                            theorem1.id,
                            theorem2.id,
                            edge_type="same_type",
                            weight=0.8,
                            theorem_type=theorem_type,
                            color="#90EE90",  # Light green
                            width=2,
                            title=f"Same type: {theorem1.theorem_type_display}"
                        )
    
    def _add_source_relationships(self, graph: nx.Graph, theorems: List[Theorem]) -> None:
        """Add edges between theorems with related source formulas."""
        for i, theorem1 in enumerate(theorems):
            for theorem2 in theorems[i+1:]:
                # Check if they share the same original formula
                if theorem1.source_lineage.original_formula == theorem2.source_lineage.original_formula:
                    graph.add_edge(
                        theorem1.id,
                        theorem2.id,
                        edge_type="same_source",
                        weight=1.0,
                        original_formula=theorem1.source_lineage.original_formula,
                        color="#FF69B4",  # Hot pink
                        width=3,
                        title=f"Same source: {theorem1.source_lineage.original_formula}"
                    )
                
                # Check symbol overlap
                symbols1 = set(theorem1.symbols)
                symbols2 = set(theorem2.symbols)
                
                if symbols1 and symbols2:
                    overlap = len(symbols1.intersection(symbols2)) / len(symbols1.union(symbols2))
                    
                    if overlap > 0.4:  # Threshold for symbol similarity
                        shared_symbols = list(symbols1.intersection(symbols2))
                        graph.add_edge(
                            theorem1.id,
                            theorem2.id,
                            edge_type="symbol_similarity",
                            weight=overlap,
                            symbol_overlap=overlap,
                            shared_symbols=shared_symbols,
                            color="#87CEEB",  # Sky blue
                            width=int(overlap * 2) + 1,
                            title=f"Shared symbols: {', '.join(shared_symbols)}"
                        )
    
    def _calculate_transformation_similarity(self, chain1: List[str], chain2: List[str]) -> float:
        """Calculate similarity between transformation chains."""
        if not chain1 or not chain2:
            return 0.0
        
        set1, set2 = set(chain1), set(chain2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        # Jaccard similarity
        jaccard = intersection / union if union > 0 else 0.0
        
        # Bonus for sequence similarity
        sequence_bonus = 0.0
        if len(chain1) == len(chain2):
            matches = sum(1 for a, b in zip(chain1, chain2) if a == b)
            sequence_bonus = matches / len(chain1) * 0.3
        
        return min(jaccard + sequence_bonus, 1.0)
    
    def get_graph_statistics(self, graph: nx.Graph) -> Dict[str, Any]:
        """Get comprehensive graph statistics."""
        if graph.number_of_nodes() == 0:
            return {
                "total_nodes": 0,
                "total_edges": 0,
                "node_types": {},
                "edge_types": {},
                "density": 0.0,
                "connected_components": 0,
                "average_clustering": 0.0
            }
        
        node_types = {}
        edge_types = {}
        
        for node, data in graph.nodes(data=True):
            node_type = data.get('node_type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        for _, _, data in graph.edges(data=True):
            edge_type = data.get('edge_type', 'unknown')
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        
        # Calculate clustering only if we have enough nodes
        avg_clustering = 0.0
        if graph.number_of_nodes() >= 3:
            try:
                avg_clustering = nx.average_clustering(graph)
            except Exception:
                avg_clustering = 0.0
        
        return {
            "total_nodes": graph.number_of_nodes(),
            "total_edges": graph.number_of_edges(),
            "node_types": node_types,
            "edge_types": edge_types,
            "density": nx.density(graph),
            "connected_components": nx.number_connected_components(graph),
            "average_clustering": avg_clustering,
            "largest_component_size": len(max(nx.connected_components(graph), key=len)) if graph.number_of_nodes() > 0 else 0
        }
    
    def filter_graph_by_type(self, graph: nx.Graph, theorem_types: List[str]) -> nx.Graph:
        """
        Filter graph to only include theorems of specified types.
        
        Args:
            graph: Original graph
            theorem_types: List of theorem types to include
            
        Returns:
            Filtered graph
        """
        filtered_graph = nx.Graph()
        
        # Add nodes of specified types
        for node, data in graph.nodes(data=True):
            if (data.get('node_type') == 'theorem' and 
                data.get('theorem_type') in theorem_types) or data.get('node_type') == 'symbol':
                filtered_graph.add_node(node, **data)
        
        # Add edges between included nodes
        for u, v, data in graph.edges(data=True):
            if filtered_graph.has_node(u) and filtered_graph.has_node(v):
                filtered_graph.add_edge(u, v, **data)
        
        # Remove isolated symbol nodes
        isolated_symbols = [node for node, data in filtered_graph.nodes(data=True) 
                          if data.get('node_type') == 'symbol' and filtered_graph.degree(node) == 0]
        filtered_graph.remove_nodes_from(isolated_symbols)
        
        self.logger.info(f"Filtered graph to {filtered_graph.number_of_nodes()} nodes "
                        f"({filtered_graph.number_of_edges()} edges) for types: {theorem_types}")
        
        return filtered_graph
    
    def get_node_neighborhood(self, graph: nx.Graph, node_id: str, radius: int = 1) -> nx.Graph:
        """
        Get subgraph containing node and its neighbors within specified radius.
        
        Args:
            graph: Original graph
            node_id: ID of central node
            radius: Radius of neighborhood
            
        Returns:
            Subgraph containing neighborhood
        """
        if not graph.has_node(node_id):
            self.logger.warning(f"Node {node_id} not found in graph")
            return nx.Graph()
        
        # Get nodes within radius
        neighbors = set([node_id])
        current_level = set([node_id])
        
        for _ in range(radius):
            next_level = set()
            for node in current_level:
                next_level.update(graph.neighbors(node))
            neighbors.update(next_level)
            current_level = next_level
        
        # Create subgraph
        subgraph = graph.subgraph(neighbors).copy()
        
        self.logger.info(f"Created neighborhood subgraph with {subgraph.number_of_nodes()} nodes "
                        f"for {node_id} (radius: {radius})")
        
        return subgraph


def create_graph_builder(config: Optional[UIConfig] = None) -> KnowledgeGraphBuilder:
    """
    Create a knowledge graph builder with the given configuration.
    
    Args:
        config: UI configuration (uses default if None)
        
    Returns:
        KnowledgeGraphBuilder instance
    """
    if config is None:
        from ui.config import get_ui_config
        config = get_ui_config()
    
    return KnowledgeGraphBuilder(config)


def build_simple_theorem_graph(theorems: List[Theorem], max_nodes: int = 50) -> nx.Graph:
    """
    Build a simple theorem graph with basic relationships.
    
    Args:
        theorems: List of theorems
        max_nodes: Maximum nodes to include
        
    Returns:
        Simple NetworkX graph
    """
    graph = nx.Graph()
    
    # Limit theorems
    limited_theorems = theorems[:max_nodes]
    
    # Add theorem nodes
    for theorem in limited_theorems:
        graph.add_node(
            theorem.id,
            label=theorem.short_id,
            type=theorem.theorem_type,
            validated=theorem.is_validated
        )
    
    # Add edges for same type
    type_groups = {}
    for theorem in limited_theorems:
        theorem_type = theorem.theorem_type
        if theorem_type not in type_groups:
            type_groups[theorem_type] = []
        type_groups[theorem_type].append(theorem)
    
    for type_theorems in type_groups.values():
        if len(type_theorems) > 1:
            for i, theorem1 in enumerate(type_theorems):
                for theorem2 in type_theorems[i+1:]:
                    graph.add_edge(theorem1.id, theorem2.id, type="same_type")
    
    return graph


def get_graph_layout_positions(graph: nx.Graph, layout: str = "spring") -> Dict[str, Tuple[float, float]]:
    """
    Get node positions for graph layout.
    
    Args:
        graph: NetworkX graph
        layout: Layout algorithm ('spring', 'circular', 'random', 'shell')
        
    Returns:
        Dictionary mapping node IDs to (x, y) positions
    """
    if graph.number_of_nodes() == 0:
        return {}
    
    layout_functions = {
        "spring": nx.spring_layout,
        "circular": nx.circular_layout,
        "random": nx.random_layout,
        "shell": nx.shell_layout,
    }
    
    layout_func = layout_functions.get(layout, nx.spring_layout)
    
    try:
        positions = layout_func(graph, seed=42)  # Fixed seed for reproducibility
        return {str(node): (float(pos[0]), float(pos[1])) for node, pos in positions.items()}
    except Exception as e:
        logger.warning(f"Failed to compute {layout} layout: {e}")
        # Fallback to random layout
        positions = nx.random_layout(graph, seed=42)
        return {str(node): (float(pos[0]), float(pos[1])) for node, pos in positions.items()} 