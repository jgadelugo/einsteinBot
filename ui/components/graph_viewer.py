"""Interactive graph visualization component for mathematical knowledge."""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from typing import Dict, List, Optional, Set, Tuple, Any, Union
import numpy as np
import math
import hashlib
import time
from functools import lru_cache

from ui.data.models import Theorem
from ui.data.loaders import TheoremLoader  
from ui.utils.graph_utils import KnowledgeGraphBuilder
from ui.config import UIConfig, get_ui_config
from ui.utils.ui_logging import get_ui_logger, log_performance, log_ui_error, log_ui_cache, log_ui_interaction

class GraphViewer:
    """Interactive graph visualization component for mathematical knowledge."""
    
    def __init__(self, config: UIConfig):
        """Initialize graph viewer with configuration."""
        self.config = config
        self.logger = get_ui_logger("graph_viewer")
        self.graph_builder = KnowledgeGraphBuilder(config)
        
        # Performance caches
        self._layout_cache: Dict[str, Dict[str, Tuple[float, float]]] = {}
        self._graph_cache: Dict[str, go.Figure] = {}
        
        # Color schemes for theorem visualization
        self.color_schemes = {
            "theorem_type": {
                "functional_equation": "#FF6B6B",
                "generalization": "#4ECDC4", 
                "transformation": "#45B7D1",
                "algebraic_identity": "#96CEB4",
                "composition": "#FFEAA7",
                "default": "#DDA0DD"
            },
            "validation_status": {
                "PASS": "#2ECC71",
                "FAIL": "#E74C3C", 
                "PARTIAL": "#F39C12",
                "ERROR": "#95A5A6"
            }
        }
        
        self.layout_algorithms = {
            "spring": nx.spring_layout,
            "circular": nx.circular_layout,
            "kamada_kawai": nx.kamada_kawai_layout,
            "spectral": nx.spectral_layout,
            "shell": nx.shell_layout
        }
        
        self.logger.info("GraphViewer initialized successfully")
        log_ui_interaction("graph_viewer", "component_initialized", {"config_type": type(config).__name__})
    
    @log_performance("graph_viewer", "render_interactive_graph")
    def render_interactive_graph(
        self, 
        theorems: List[Theorem],
        selected_node: Optional[str] = None,
        layout_algorithm: str = "spring",
        color_by: str = "theorem_type",
        show_labels: bool = True,
        physics_enabled: bool = True,
        node_size_factor: float = 1.0
    ) -> go.Figure:
        """Render interactive Plotly graph with theorem data."""
        start_time = time.time()
        
        try:
            if not theorems:
                self.logger.warning("No theorems provided")
                return self._create_empty_graph()
            
            # Build graph using Phase 6A utilities
            graph = self.graph_builder.build_theorem_graph(theorems)
            
            if graph.number_of_nodes() == 0:
                return self._create_empty_graph()
            
            # Performance optimization with caching
            cache_key = self._generate_cache_key(theorems, {
                'layout_algorithm': layout_algorithm,
                'color_by': color_by,
                'node_size_factor': node_size_factor
            })
            
            if cache_key in self._graph_cache and selected_node is None:
                self.logger.debug("Using cached graph")
                log_ui_cache("graph_viewer", "get", cache_key, True, len(self._graph_cache))
                return self._graph_cache[cache_key]
            
            # Calculate positions
            positions = self._calculate_layout(graph, layout_algorithm)
            
            # Create Plotly components
            edge_trace = self._create_edge_trace(graph, positions)
            node_trace = self._create_node_trace(
                graph, positions, color_by, selected_node, node_size_factor, theorems
            )
            
            # Assemble figure
            fig = go.Figure(
                data=[edge_trace, node_trace],
                layout=self._create_layout(show_labels, physics_enabled)
            )
            
            # Add selection highlighting
            if selected_node and selected_node in graph:
                self._add_selection_annotations(fig, graph, positions, selected_node)
            
            # Cache result
            if selected_node is None:
                self._graph_cache[cache_key] = fig
                log_ui_cache("graph_viewer", "set", cache_key, False, len(self._graph_cache))
            
            render_time = time.time() - start_time
            self.logger.info(f"Rendered graph: {graph.number_of_nodes()} nodes, "
                           f"{graph.number_of_edges()} edges in {render_time:.3f}s")
            
            return fig
            
        except Exception as e:
            log_ui_error("graph_viewer", e, {
                "theorem_count": len(theorems),
                "layout_algorithm": layout_algorithm,
                "color_by": color_by
            }, "Failed to render graph visualization")
            return self._create_error_graph(str(e))
    
    def _generate_cache_key(self, theorems: List[Theorem], params: Dict[str, Any]) -> str:
        """Generate cache key for performance optimization."""
        theorem_ids = sorted([t.id for t in theorems])
        key_data = {'theorem_ids': theorem_ids, 'params': params}
        return hashlib.md5(str(key_data).encode()).hexdigest()
    
    def _calculate_layout(self, graph: nx.Graph, algorithm: str) -> Dict[str, Tuple[float, float]]:
        """Calculate node positions with caching."""
        cache_key = f"{algorithm}_{hash(tuple(sorted(graph.nodes())))}"
        
        if cache_key in self._layout_cache:
            self.logger.debug(f"Using cached {algorithm} layout")
            log_ui_cache("graph_viewer", "layout_get", cache_key, True, len(self._layout_cache))
            return self._layout_cache[cache_key]
        
        try:
            layout_func = self.layout_algorithms.get(algorithm, nx.spring_layout)
            
            if algorithm == "spring":
                positions = layout_func(
                    graph, 
                    k=1/math.sqrt(graph.number_of_nodes()) if graph.number_of_nodes() > 1 else 1,
                    iterations=50,
                    scale=2.0,
                    seed=42
                )
            elif algorithm == "kamada_kawai":
                positions = layout_func(graph, scale=2.0)
            else:
                positions = layout_func(graph)
            
            self._layout_cache[cache_key] = positions
            log_ui_cache("graph_viewer", "layout_set", cache_key, False, len(self._layout_cache))
            self.logger.debug(f"Calculated {algorithm} layout for {len(positions)} nodes")
            return positions
            
        except Exception as e:
            self.logger.warning(f"Layout {algorithm} failed: {e}, using spring fallback")
            fallback = nx.spring_layout(graph, seed=42)
            self._layout_cache[cache_key] = fallback
            return fallback
    
    def _create_edge_trace(self, graph: nx.Graph, positions: Dict) -> go.Scatter:
        """Create edge visualization trace."""
        edge_x, edge_y = [], []
        
        for edge in graph.edges(data=True):
            node1, node2, edge_data = edge
            if node1 not in positions or node2 not in positions:
                continue
                
            x0, y0 = positions[node1]
            x1, y1 = positions[node2]
            
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        return go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.8, color='rgba(136, 136, 136, 0.5)'),
            hoverinfo='none',
            mode='lines',
            showlegend=False,
            name='connections'
        )
    
    def _create_node_trace(
        self, 
        graph: nx.Graph, 
        positions: Dict, 
        color_by: str,
        selected_node: Optional[str],
        size_factor: float,
        theorems: List[Theorem]
    ) -> go.Scatter:
        """Create node visualization with rich data."""
        node_x, node_y = [], []
        node_colors, node_sizes, node_text, hover_text = [], [], [], []
        theorem_lookup = {t.id: t for t in theorems}
        
        for node, node_data in graph.nodes(data=True):
            if node not in positions:
                continue
                
            x, y = positions[node]
            node_x.append(x)
            node_y.append(y)
            
            # Color assignment
            color = self._get_node_color(node_data, color_by, theorem_lookup.get(node))
            if node == selected_node:
                color = '#FFD700'  # Gold for selection
            node_colors.append(color)
            
            # Size calculation
            base_size = self._calculate_node_size(node_data, theorem_lookup.get(node))
            size = base_size * size_factor
            if node == selected_node:
                size *= 1.5
            node_sizes.append(size)
            
            # Labels and hover text
            label = self._format_node_label(node, node_data)
            node_text.append(label)
            
            hover_info = self._create_hover_text(node, node_data, theorem_lookup.get(node))
            hover_text.append(hover_info)
        
        return go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='white'),
                opacity=0.8,
                sizemode='diameter'
            ),
            text=node_text,
            textposition="middle center",
            textfont=dict(size=10, color='white', family="Arial Black"),
            hovertext=hover_text,
            hoverinfo='text',
            name='theorems',
            showlegend=False
        )
    
    def _get_node_color(self, node_data: Dict, color_by: str, theorem: Optional[Theorem]) -> str:
        """Determine node color based on properties."""
        if color_by == "theorem_type" and theorem:
            return self.color_schemes["theorem_type"].get(
                theorem.theorem_type, 
                self.color_schemes["theorem_type"]["default"]
            )
        elif color_by == "validation_status" and theorem:
            status = theorem.validation_evidence.validation_status if theorem.validation_evidence else "ERROR"
            return self.color_schemes["validation_status"].get(status, self.color_schemes["validation_status"]["ERROR"])
        elif color_by == "complexity" and theorem:
            confidence = theorem.source_lineage.confidence if theorem.source_lineage else 0.0
            if confidence > 0.8:
                return "#2ECC71"
            elif confidence > 0.6:
                return "#F39C12"
            else:
                return "#E74C3C"
        else:
            return "#9B59B6" if node_data.get('node_type') == 'symbol' else "#DDA0DD"
    
    def _calculate_node_size(self, node_data: Dict, theorem: Optional[Theorem]) -> float:
        """Calculate node size based on importance."""
        base_size = 20  # Default base size
        max_size = 40   # Maximum size
        
        if theorem:
            confidence = theorem.source_lineage.confidence if theorem.source_lineage else 0.5
            validation_bonus = 1.2 if theorem.is_validated else 0.8
            size = base_size + (max_size - base_size) * confidence * validation_bonus
        else:
            usage_count = node_data.get('usage_count', 1)
            size = base_size + min(usage_count * 2, max_size - base_size)
        
        return max(base_size, min(size, max_size))
    
    def _format_node_label(self, node_id: str, node_data: Dict) -> str:
        """Format node label for display."""
        if node_data.get('node_type') == 'theorem':
            label = node_id.replace('THM_', '').replace('_', ' ')
            return label[:8] + '...' if len(label) > 8 else label
        elif node_data.get('node_type') == 'symbol':
            return node_data.get('label', node_id)
        return node_id
    
    def _create_hover_text(self, node_id: str, node_data: Dict, theorem: Optional[Theorem]) -> str:
        """Create rich hover information."""
        if theorem:
            statement_preview = theorem.display_statement[:150] + "..." if len(theorem.display_statement) > 150 else theorem.display_statement
            return (f"<b>{theorem.id}</b><br>"
                   f"<b>Type:</b> {theorem.theorem_type}<br>"
                   f"<b>Validation:</b> {theorem.validation_evidence.validation_status if theorem.validation_evidence else 'Unknown'}<br>"
                   f"<b>Confidence:</b> {theorem.source_lineage.confidence:.2f}<br>"
                   f"<b>Statement:</b><br>{statement_preview}")
        elif node_data.get('node_type') == 'symbol':
            symbol = node_data.get('label', node_id)
            usage_count = node_data.get('usage_count', 0)
            return (f"<b>Symbol: {symbol}</b><br>"
                   f"Used in {usage_count} theorems<br>"
                   f"Mathematical symbol")
        return f"<b>{node_id}</b><br>Type: {node_data.get('node_type', 'unknown')}"
    
    def _create_layout(self, show_labels: bool, physics_enabled: bool) -> dict:
        """Create Plotly layout configuration."""
        return dict(
            title=dict(
                text="Mathematical Knowledge Graph",
                x=0.5,
                font=dict(size=20, color="#2C3E50")
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=60),
            annotations=[
                dict(
                    text="Hover over nodes for details • Click to select • Use controls to customize",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(color="#7F8C8D", size=12)
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, showspikes=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, showspikes=False),
            plot_bgcolor='white',
            paper_bgcolor='white', 
            height=600,  # Default height
            dragmode='pan'
        )
    
    def _add_selection_annotations(self, fig: go.Figure, graph: nx.Graph, positions: Dict, selected_node: str) -> None:
        """Add visual selection highlighting."""
        if selected_node not in positions:
            return
        
        x, y = positions[selected_node]
        fig.add_shape(
            type="circle",
            xref="x", yref="y",
            x0=x-0.15, y0=y-0.15,
            x1=x+0.15, y1=y+0.15,
            line=dict(color="gold", width=3),
            fillcolor="rgba(255, 215, 0, 0.1)"
        )
    
    def _create_empty_graph(self) -> go.Figure:
        """Create empty state visualization."""
        return go.Figure(
            layout=dict(
                title="No Theorem Data Available",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                annotations=[
                    dict(
                        text="No theorems available for visualization<br><i>Check data loading or adjust filters</i>",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.5, y=0.5,
                        font=dict(size=16, color="#7F8C8D"),
                        align="center"
                    )
                ],
                height=400
            )
        )
    
    def _create_error_graph(self, error_message: str) -> go.Figure:
        """Create error state visualization."""
        return go.Figure(
            layout=dict(
                title="Graph Rendering Error",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                annotations=[
                    dict(
                        text=f"Error rendering graph:<br><i>{error_message}</i>",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.5, y=0.5,
                        font=dict(size=14, color="#E74C3C"),
                        align="center"
                    )
                ],
                height=400
            )
        )
    
    def get_node_neighbors(self, graph: nx.Graph, node_id: str, depth: int = 1) -> Set[str]:
        """Get neighboring nodes for exploration."""
        if node_id not in graph:
            return set()
        
        neighbors = set()
        current_level = {node_id}
        
        for _ in range(depth):
            next_level = set()
            for node in current_level:
                next_level.update(graph.neighbors(node))
            neighbors.update(next_level)
            current_level = next_level
        
        return neighbors - {node_id}
    
    def filter_theorems_by_selection(
        self, 
        theorems: List[Theorem], 
        selected_types: List[str],
        validation_filter: str = "all",
        confidence_range: Tuple[float, float] = (0.0, 1.0)
    ) -> List[Theorem]:
        """Filter theorems based on UI selections."""
        filtered = theorems
        
        if selected_types and "all" not in selected_types:
            filtered = [t for t in filtered if t.theorem_type in selected_types]
        
        if validation_filter != "all":
            if validation_filter == "validated":
                filtered = [t for t in filtered if t.is_validated]
            elif validation_filter == "failed":
                filtered = [t for t in filtered if not t.is_validated]
        
        min_conf, max_conf = confidence_range
        filtered = [t for t in filtered 
                   if min_conf <= (t.source_lineage.confidence if t.source_lineage else 0.0) <= max_conf]
        
        self.logger.info(f"Filtered {len(theorems)} theorems to {len(filtered)}")
        return filtered
    
    def optimize_for_performance(self, theorems: List[Theorem], max_nodes: int = 100) -> List[Theorem]:
        """Optimize dataset for performance."""
        if len(theorems) <= max_nodes:
            return theorems
        
        # Score theorems by importance
        scored_theorems = []
        for theorem in theorems:
            score = 0
            
            # Validation bonus
            if theorem.is_validated:
                score += 10
            
            # Confidence score
            if theorem.source_lineage:
                score += theorem.source_lineage.confidence * 5
            
            # Complexity bonus (more complex theorems are more interesting)
            if len(theorem.display_statement) > 100:
                score += 2
            
            scored_theorems.append((theorem, score))
        
        # Sort by score and take top N
        scored_theorems.sort(key=lambda x: x[1], reverse=True)
        optimized = [t for t, _ in scored_theorems[:max_nodes]]
        
        self.logger.info(f"Optimized dataset from {len(theorems)} to {len(optimized)} theorems")
        return optimized

    def get_memory_usage(self) -> Dict[str, int]:
        """Get current memory usage statistics."""
        return {
            "layout_cache_size": len(self._layout_cache),
            "graph_cache_size": len(self._graph_cache),
            "total_cached_layouts": sum(len(cache) for cache in self._layout_cache.values()) if self._layout_cache else 0,
        }
    
    def clear_cache(self) -> None:
        """Clear caches for memory management."""
        self._layout_cache.clear()
        self._graph_cache.clear()
        self.logger.info("Cleared graph rendering caches") 