"""
Graph control interface for interactive visualization customization.

This module provides a clean, modular interface for graph visualization controls
with comprehensive error handling, logging, and type safety.

Author: MathBot Team
Version: Phase 6B
"""

import streamlit as st
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass
import logging

from ui.data.models import Theorem
from ui.utils.ui_logging import get_ui_logger, log_user_action, log_ui_interaction


@dataclass
class LayoutConfig:
    """Type-safe configuration for graph layout settings."""
    algorithm: str
    color_by: str
    node_size_factor: float
    show_labels: bool
    physics_enabled: bool
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0.1 <= self.node_size_factor <= 3.0:
            raise ValueError(f"node_size_factor must be between 0.1 and 3.0, got {self.node_size_factor}")
        
        if self.algorithm not in ["spring", "circular", "kamada_kawai", "spectral", "shell"]:
            raise ValueError(f"Invalid layout algorithm: {self.algorithm}")
        
        if self.color_by not in ["theorem_type", "validation_status", "complexity"]:
            raise ValueError(f"Invalid color scheme: {self.color_by}")


@dataclass
class FilterConfig:
    """Type-safe configuration for theorem filtering."""
    selected_types: List[str]
    validation_filter: str
    confidence_range: Tuple[float, float]
    
    def __post_init__(self):
        """Validate filter configuration."""
        min_conf, max_conf = self.confidence_range
        if not (0.0 <= min_conf <= max_conf <= 1.0):
            raise ValueError(f"Invalid confidence range: {self.confidence_range}")
        
        if self.validation_filter not in ["all", "validated", "failed"]:
            raise ValueError(f"Invalid validation filter: {self.validation_filter}")


@dataclass
class ExplorationConfig:
    """Type-safe configuration for node exploration."""
    selected_node: Optional[str]
    exploration_depth: int
    highlight_neighbors: bool
    
    def __post_init__(self):
        """Validate exploration configuration."""
        if not 1 <= self.exploration_depth <= 5:
            raise ValueError(f"exploration_depth must be between 1 and 5, got {self.exploration_depth}")


class GraphControls:
    """
    Streamlit controls for graph visualization customization.
    
    This class provides a modular, type-safe interface for all graph control
    components with comprehensive error handling and logging.
    """
    
    def __init__(self):
        """Initialize controls with validated configuration options."""
        self.logger = get_ui_logger("graph_controls")
        
        # Layout algorithm configurations
        self.available_layouts = [
            "spring", "circular", "kamada_kawai", "spectral", "shell"
        ]
        
        self.color_options = ["theorem_type", "validation_status", "complexity"]
        
        # Detailed descriptions for user guidance
        self.layout_descriptions = {
            "spring": "Force-directed layout (good for clusters)",
            "circular": "Nodes arranged in circle", 
            "kamada_kawai": "Energy-based layout (minimal crossings)",
            "spectral": "Eigenvalue-based positioning",
            "shell": "Concentric circles by connectivity"
        }
        
        self.color_descriptions = {
            "theorem_type": "Color by mathematical theorem type",
            "validation_status": "Color by validation pass/fail status",
            "complexity": "Color by confidence/complexity level"
        }
        
        # Performance tracking
        self._render_count = 0
        self._error_count = 0
        
        self.logger.info("GraphControls initialized successfully")
        log_ui_interaction("graph_controls", "component_initialized")
    
    @log_user_action("graph_controls", "render_layout_controls")
    def render_layout_controls(self) -> Dict[str, Any]:
        """
        Render layout and appearance controls with error handling.
        
        Returns:
            Dict containing validated layout configuration
            
        Raises:
            ValueError: If control values are invalid
            RuntimeError: If Streamlit rendering fails
        """
        try:
            self._render_count += 1
            self.logger.debug(f"Rendering layout controls (render #{self._render_count})")
            
            st.subheader("🎨 Graph Appearance")
            
            # Create responsive column layout
            col1, col2, col3 = st.columns(3)
            
            with col1:
                layout = st.selectbox(
                    "Layout Algorithm",
                    self.available_layouts,
                    index=0,
                    help="Algorithm for positioning nodes in the graph visualization",
                    format_func=lambda x: f"{x.title()}"
                )
                
                # Show detailed description
                if layout in self.layout_descriptions:
                    st.caption(self.layout_descriptions[layout])
            
            with col2:
                color_by = st.selectbox(
                    "Color Nodes By",
                    self.color_options,
                    help="Property used to determine node colors",
                    format_func=lambda x: x.replace('_', ' ').title()
                )
                
                # Show detailed description
                if color_by in self.color_descriptions:
                    st.caption(self.color_descriptions[color_by])
            
            with col3:
                node_size = st.slider(
                    "Node Size",
                    min_value=0.5,
                    max_value=2.0,
                    value=1.0,
                    step=0.1,
                    help="Scaling factor for node sizes (1.0 = default size)"
                )
            
            # Additional appearance controls in separate row
            col4, col5 = st.columns(2)
            
            with col4:
                show_labels = st.checkbox(
                    "Show Node Labels", 
                    value=True,
                    help="Display text labels on graph nodes"
                )
            
            with col5:
                physics_enabled = st.checkbox(
                    "Enable Physics", 
                    value=True,
                    help="Enable physics simulation for interactive graph manipulation"
                )
            
            # Validate and create configuration
            config = {
                "layout_algorithm": layout,
                "color_by": color_by,
                "node_size_factor": node_size,
                "show_labels": show_labels,
                "physics_enabled": physics_enabled
            }
            
            # Type-safe validation
            layout_config = LayoutConfig(
                algorithm=layout,
                color_by=color_by,
                node_size_factor=node_size,
                show_labels=show_labels,
                physics_enabled=physics_enabled
            )
            
            self.logger.debug(f"Layout controls rendered successfully: {layout_config}")
            return config
            
        except ValueError as e:
            self._error_count += 1
            self.logger.error(f"Invalid layout control values: {e}")
            st.error(f"Invalid configuration: {e}")
            raise
            
        except Exception as e:
            self._error_count += 1
            self.logger.error(f"Failed to render layout controls: {e}", exc_info=True)
            st.error("Failed to render layout controls. Please refresh the page.")
            raise RuntimeError(f"Layout controls rendering failed: {e}")
    
    def render_filter_controls(self, theorems: List[Theorem]) -> Dict[str, Any]:
        """
        Render theorem filtering controls with comprehensive validation.
        
        Args:
            theorems: List of available theorems for filter options
            
        Returns:
            Dict containing validated filter configuration
            
        Raises:
            ValueError: If theorem data is invalid or empty
            RuntimeError: If control rendering fails
        """
        try:
            if not theorems:
                self.logger.warning("No theorems provided for filter controls")
                st.warning("⚠️ No theorem data available for filtering")
                return self._get_default_filter_config()
            
            self.logger.debug(f"Rendering filter controls for {len(theorems)} theorems")
            
            st.subheader("🔍 Filter Theorems")
            
            # Extract available theorem types safely
            theorem_types = self._extract_theorem_types(theorems)
            
            # Filter controls in responsive layout
            col1, col2 = st.columns(2)
            
            with col1:
                selected_types = st.multiselect(
                    "Theorem Types",
                    ["all"] + sorted(theorem_types),
                    default=["all"],
                    help="Select specific theorem types to display (leave 'all' for no filtering)"
                )
                
                # Provide user feedback
                if not selected_types:
                    st.caption("⚠️ No types selected - showing all theorems")
                    selected_types = ["all"]
            
            with col2:
                validation_filter = st.selectbox(
                    "Validation Status",
                    ["all", "validated", "failed"],
                    help="Filter theorems by their validation status"
                )
            
            # Confidence range control
            confidence_range = st.slider(
                "Confidence Range",
                min_value=0.0,
                max_value=1.0,
                value=(0.0, 1.0),
                step=0.05,
                help="Filter theorems by their confidence level (0.0 = low confidence, 1.0 = high confidence)"
            )
            
            # Provide live filter preview
            filtered_count = self._calculate_filter_preview(
                theorems, selected_types, validation_filter, confidence_range
            )
            
            # User feedback with appropriate styling
            if filtered_count == len(theorems):
                st.info(f"📊 Showing all {filtered_count} theorems")
            elif filtered_count > 0:
                st.success(f"📊 Filters will show {filtered_count} of {len(theorems)} theorems")
            else:
                st.warning(f"⚠️ Current filters exclude all theorems. Consider adjusting your selection.")
            
            # Create validated configuration
            config = {
                "selected_types": selected_types,
                "validation_filter": validation_filter,
                "confidence_range": confidence_range
            }
            
            # Type-safe validation
            filter_config = FilterConfig(
                selected_types=selected_types,
                validation_filter=validation_filter,
                confidence_range=confidence_range
            )
            
            self.logger.debug(f"Filter controls rendered successfully: {filter_config}")
            return config
            
        except ValueError as e:
            self._error_count += 1
            self.logger.error(f"Invalid filter configuration: {e}")
            st.error(f"Filter configuration error: {e}")
            return self._get_default_filter_config()
            
        except Exception as e:
            self._error_count += 1
            self.logger.error(f"Failed to render filter controls: {e}", exc_info=True)
            st.error("Failed to render filter controls. Using default settings.")
            return self._get_default_filter_config()
    
    def render_exploration_controls(self, available_nodes: List[str]) -> Dict[str, Any]:
        """
        Render node exploration controls with validation.
        
        Args:
            available_nodes: List of node IDs available for selection
            
        Returns:
            Dict containing validated exploration configuration
            
        Raises:
            ValueError: If node data is invalid
            RuntimeError: If control rendering fails
        """
        try:
            self.logger.debug(f"Rendering exploration controls for {len(available_nodes)} nodes")
            
            st.subheader("🔭 Explore Connections")
            
            if not available_nodes:
                st.info("No nodes available for exploration")
                return self._get_default_exploration_config()
            
            # Node selection with search functionality
            selected_node = st.selectbox(
                "Focus on Node",
                ["None"] + sorted(available_nodes),
                help="Select a node to highlight and explore its connections"
            )
            
            # Dynamic controls based on selection
            if selected_node != "None":
                col1, col2 = st.columns(2)
                
                with col1:
                    depth = st.slider(
                        "Connection Depth",
                        min_value=1,
                        max_value=3,
                        value=1,
                        help="Number of connection levels to explore (1 = direct neighbors)"
                    )
                
                with col2:
                    highlight_neighbors = st.checkbox(
                        "Highlight Neighbors",
                        value=True,
                        help="Visually highlight connected nodes in the graph"
                    )
                
                # Provide exploration guidance
                if depth == 1:
                    st.caption("🔍 Showing direct connections")
                else:
                    st.caption(f"🔍 Exploring {depth} levels of connections")
                    
            else:
                depth = 1
                highlight_neighbors = False
                st.caption("👆 Select a node above to explore its connections")
            
            # Create validated configuration
            config = {
                "selected_node": selected_node if selected_node != "None" else None,
                "exploration_depth": depth,
                "highlight_neighbors": highlight_neighbors
            }
            
            # Type-safe validation
            exploration_config = ExplorationConfig(
                selected_node=config["selected_node"],
                exploration_depth=depth,
                highlight_neighbors=highlight_neighbors
            )
            
            self.logger.debug(f"Exploration controls rendered successfully: {exploration_config}")
            return config
            
        except ValueError as e:
            self._error_count += 1
            self.logger.error(f"Invalid exploration configuration: {e}")
            st.error(f"Exploration configuration error: {e}")
            return self._get_default_exploration_config()
            
        except Exception as e:
            self._error_count += 1
            self.logger.error(f"Failed to render exploration controls: {e}", exc_info=True)
            st.error("Failed to render exploration controls. Using default settings.")
            return self._get_default_exploration_config()
    
    def render_performance_info(self, node_count: int, edge_count: int) -> None:
        """
        Render performance information and warnings.
        
        Args:
            node_count: Number of nodes in the graph
            edge_count: Number of edges in the graph
            
        Raises:
            ValueError: If counts are negative
            RuntimeError: If rendering fails
        """
        try:
            if node_count < 0 or edge_count < 0:
                raise ValueError(f"Invalid graph metrics: nodes={node_count}, edges={edge_count}")
            
            self.logger.debug(f"Rendering performance info: {node_count} nodes, {edge_count} edges")
            
            st.subheader("📊 Performance Info")
            
            # Metrics in responsive columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Nodes", f"{node_count:,}")
                st.metric("Edges", f"{edge_count:,}")
            
            with col2:
                # Calculate graph density safely
                density = self._calculate_graph_density(node_count, edge_count)
                st.metric("Density", f"{density:.3f}")
                
                # Calculate average degree
                avg_degree = (2 * edge_count) / max(node_count, 1)
                st.metric("Avg Degree", f"{avg_degree:.1f}")
            
            # Performance warnings with appropriate thresholds
            self._render_performance_warnings(node_count, edge_count)
            
            self.logger.debug("Performance info rendered successfully")
            
        except ValueError as e:
            self._error_count += 1
            self.logger.error(f"Invalid performance metrics: {e}")
            st.error(f"Invalid graph metrics: {e}")
            
        except Exception as e:
            self._error_count += 1
            self.logger.error(f"Failed to render performance info: {e}", exc_info=True)
            st.error("Failed to display performance information")
    
    def get_control_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about control component usage.
        
        Returns:
            Dict containing performance and usage statistics
        """
        return {
            "render_count": self._render_count,
            "error_count": self._error_count,
            "error_rate": self._error_count / max(self._render_count, 1),
            "available_layouts": len(self.available_layouts),
            "available_color_schemes": len(self.color_options)
        }
    
    def _extract_theorem_types(self, theorems: List[Theorem]) -> Set[str]:
        """Safely extract unique theorem types from theorem list."""
        try:
            types = set()
            for theorem in theorems:
                if hasattr(theorem, 'theorem_type') and theorem.theorem_type:
                    types.add(theorem.theorem_type)
            
            self.logger.debug(f"Extracted {len(types)} theorem types: {sorted(types)}")
            return types
            
        except Exception as e:
            self.logger.warning(f"Failed to extract theorem types: {e}")
            return set()
    
    def _calculate_filter_preview(
        self, 
        theorems: List[Theorem], 
        selected_types: List[str],
        validation_filter: str,
        confidence_range: Tuple[float, float]
    ) -> int:
        """Calculate how many theorems would pass current filters."""
        try:
            filtered = theorems
            
            # Apply type filter
            if selected_types and "all" not in selected_types:
                filtered = [t for t in filtered if hasattr(t, 'theorem_type') and t.theorem_type in selected_types]
            
            # Apply validation filter
            if validation_filter != "all":
                if validation_filter == "validated":
                    filtered = [t for t in filtered if hasattr(t, 'is_validated') and t.is_validated]
                elif validation_filter == "failed":
                    filtered = [t for t in filtered if hasattr(t, 'is_validated') and not t.is_validated]
            
            # Apply confidence filter
            min_conf, max_conf = confidence_range
            filtered = [
                t for t in filtered 
                if hasattr(t, 'source_lineage') and t.source_lineage and
                min_conf <= t.source_lineage.confidence <= max_conf
            ]
            
            return len(filtered)
            
        except Exception as e:
            self.logger.warning(f"Filter preview calculation failed: {e}")
            return len(theorems)  # Fallback to showing all
    
    def _calculate_graph_density(self, node_count: int, edge_count: int) -> float:
        """Calculate graph density with safe division."""
        if node_count <= 1:
            return 0.0
        
        max_edges = node_count * (node_count - 1) / 2
        return edge_count / max_edges if max_edges > 0 else 0.0
    
    def _render_performance_warnings(self, node_count: int, edge_count: int) -> None:
        """Render appropriate performance warnings based on graph size."""
        if node_count > 100:
            st.error("🚨 Very large graph detected - performance may be severely impacted. Consider filtering.")
        elif node_count > 50:
            st.warning("⚠️ Large graph detected - rendering may be slow. Consider applying filters.")
        elif edge_count > 500:
            st.warning("⚠️ High edge density - interactions may be slower than usual.")
        else:
            st.success("✅ Graph size is optimal for performance")
    
    def _get_default_filter_config(self) -> Dict[str, Any]:
        """Get safe default filter configuration."""
        return {
            "selected_types": ["all"],
            "validation_filter": "all",
            "confidence_range": (0.0, 1.0)
        }
    
    def _get_default_exploration_config(self) -> Dict[str, Any]:
        """Get safe default exploration configuration."""
        return {
            "selected_node": None,
            "exploration_depth": 1,
            "highlight_neighbors": False
        } 