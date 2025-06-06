"""
MathBot Interactive Exploration & Visualization Interface - Main Application

This is the main Streamlit application that provides the user interface for
exploring MathBot's mathematical knowledge base. It orchestrates all the
different components and handles navigation between them.

Usage:
    streamlit run ui/app.py
    or
    python main.py --serve-ui

Author: MathBot Team
Version: 1.0.0
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

import streamlit as st
import pandas as pd

# Add the parent directory to sys.path to import from the main project
sys.path.append(str(Path(__file__).parent.parent))

from ui.utils.ui_logging import get_ui_logger, log_user_action, log_ui_interaction, log_ui_error

# UI-specific imports
from ui.config import UIConfig, get_ui_config
from ui.data.loaders import TheoremLoader, FormulaLoader
from ui.data.models import Theorem, ValidationEvidence
from ui.components.graph_viewer import GraphViewer
from ui.components.graph_controls import GraphControls
# from ui.components.theorem_browser import TheoremBrowser (to be implemented)
# from ui.components.proof_viewer import ProofViewer (to be implemented)
# from ui.components.search_interface import SearchInterface (to be implemented)


class MathBotUI:
    """
    Main application class for the MathBot Interactive Interface.
    
    This class orchestrates all UI components and manages the application state.
    It provides a cohesive interface for exploring mathematical knowledge.
    """
    
    def __init__(self):
        """Initialize the MathBot UI application."""
        self.logger = get_ui_logger("mathbot_ui")
        self.logger.info("Initializing MathBot UI application")
        log_ui_interaction("mathbot_ui", "app_initialization_started")
        
        # Initialize configuration
        self.config = get_ui_config()
        
        # Initialize session state
        self._initialize_session_state()
        
        # Configure page settings
        self._configure_page()
        
        # Initialize data loaders
        self._initialize_data_loaders()
    
    def _initialize_session_state(self) -> None:
        """Initialize Streamlit session state variables."""
        if 'current_page' not in st.session_state:
            st.session_state.current_page = "Overview"
        
        if 'selected_theorem' not in st.session_state:
            st.session_state.selected_theorem = None
        
        if 'search_query' not in st.session_state:
            st.session_state.search_query = ""
        
        if 'graph_filters' not in st.session_state:
            st.session_state.graph_filters = {}
    
    def _configure_page(self) -> None:
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="MathBot - Mathematical Knowledge Explorer",
            page_icon="🧮",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for mathematical content
        st.markdown("""
        <style>
        .math-expression {
            font-family: 'Computer Modern', serif;
            font-size: 1.2em;
            padding: 10px;
            background-color: #f8f9fa;
            border-left: 4px solid #007bff;
            margin: 10px 0;
        }
        
        .theorem-card {
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            background-color: white;
        }
        
        .validation-pass {
            color: #28a745;
            font-weight: bold;
        }
        
        .validation-fail {
            color: #dc3545;
            font-weight: bold;
        }
        
        .proof-step {
            border-left: 3px solid #6c757d;
            padding-left: 15px;
            margin: 5px 0;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _initialize_data_loaders(self) -> None:
        """Initialize data loaders for theorems, formulas, and validation data."""
        try:
            self.logger.info("Initializing data loaders")
            
            # Initialize actual data loaders with error handling
            self.theorem_loader = TheoremLoader(self.config)
            self.formula_loader = FormulaLoader(self.config)
            
            self.logger.info("Data loaders initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize data loaders: {e}", exc_info=True)
            # Set fallback None values to prevent crashes
            self.theorem_loader = None
            self.formula_loader = None
    
    def render_sidebar(self) -> None:
        """Render the main navigation sidebar."""
        st.sidebar.title("🧮 MathBot Explorer")
        st.sidebar.markdown("---")
        
        # Navigation menu
        pages = [
            "Overview",
            "Knowledge Graph", 
            "Theorem Browser",
            "Proof Viewer",
            "Search & Filter"
        ]
        
        selected_page = st.sidebar.selectbox(
            "Navigate to:",
            pages,
            index=pages.index(st.session_state.current_page)
        )
        
        if selected_page != st.session_state.current_page:
            st.session_state.current_page = selected_page
            st.rerun()
        
        st.sidebar.markdown("---")
        
        # Quick stats (placeholder)
        st.sidebar.metric("Total Theorems", "13")
        st.sidebar.metric("Validated Formulas", "---")
        st.sidebar.metric("Proof Traces", "---")
        
        st.sidebar.markdown("---")
        
        # Settings expander
        with st.sidebar.expander("⚙️ Settings"):
            st.checkbox("Show validation details", value=True)
            st.checkbox("Enable graph physics", value=True)
            st.selectbox("LaTeX renderer", ["MathJax", "KaTeX"], index=0)
    
    def render_overview_page(self) -> None:
        """Render the overview/dashboard page."""
        st.title("🧮 MathBot Mathematical Knowledge Explorer")
        st.markdown("*Interactive exploration of validated mathematical theorems and proofs*")
        
        # Welcome message and instructions
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ## Welcome to MathBot Explorer
            
            This interface provides comprehensive access to MathBot's mathematical knowledge base,
            including validated theorems, formula relationships, and proof traces.
            
            ### Features:
            - 📊 **Knowledge Graph**: Visualize relationships between mathematical concepts
            - 📚 **Theorem Browser**: Search and explore validated theorems  
            - 🔍 **Proof Viewer**: Step-by-step proof trace visualization
            - 🔎 **Advanced Search**: Multi-faceted search and filtering
            
            ### Getting Started:
            1. Use the sidebar to navigate between different views
            2. Start with the **Knowledge Graph** for a visual overview
            3. Browse specific theorems in the **Theorem Browser**
            4. Dive deep into proofs with the **Proof Viewer**
            """)
        
        with col2:
            st.info("""
            **Current Data:**
            - 13 Validated Theorems
            - Multiple theorem types
            - Complete validation evidence
            - Transformation chains
            """)
        
        # Quick access buttons
        st.markdown("### Quick Access")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🌐 Knowledge Graph", use_container_width=True):
                st.session_state.current_page = "Knowledge Graph"
                st.rerun()
        
        with col2:
            if st.button("📚 Browse Theorems", use_container_width=True):
                st.session_state.current_page = "Theorem Browser"  
                st.rerun()
        
        with col3:
            if st.button("🔍 View Proofs", use_container_width=True):
                st.session_state.current_page = "Proof Viewer"
                st.rerun()
        
        with col4:
            if st.button("🔎 Advanced Search", use_container_width=True):
                st.session_state.current_page = "Search & Filter"
                st.rerun()
        
        # Recent activity / highlights (placeholder)
        st.markdown("---")
        st.subheader("📈 System Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Functional Equations", "3", "1")
        with col2:
            st.metric("Generalizations", "3", "2") 
        with col3:
            st.metric("Transformations", "3", "1")
        with col4:
            st.metric("Validation Rate", "100%", "0%")
    
    def render_graph_page(self) -> None:
        """Render the interactive knowledge graph visualization page."""
        st.title("🌐 Mathematical Knowledge Graph")
        st.markdown("*Explore relationships between theorems, formulas, and concepts*")
        
        # Initialize components if not in session state
        if not hasattr(st.session_state, 'graph_viewer'):
            try:
                st.session_state.graph_viewer = GraphViewer(self.config)
            except Exception as e:
                self.logger.error(f"Failed to initialize GraphViewer: {e}", exc_info=True)
                st.error("Failed to initialize graph viewer. Please refresh the page.")
                return
                
        if not hasattr(st.session_state, 'graph_controls'):
            try:
                st.session_state.graph_controls = GraphControls()
            except Exception as e:
                self.logger.error(f"Failed to initialize GraphControls: {e}", exc_info=True)
                st.error("Failed to initialize graph controls. Please refresh the page.")
                return
        
        # Load theorem data using Phase 6A loader
        try:
            if self.theorem_loader is None:
                st.error("❌ Data loader not available. Please check system configuration.")
                return
                
            theorems = self.theorem_loader.load_theorems()
            if not theorems:
                st.warning("⚠️ No theorem data available for visualization.")
                st.info("Please ensure data files are available and properly formatted.")
                return
                
        except Exception as e:
            st.error(f"❌ Error loading theorem data: {e}")
            self.logger.error(f"Graph page data loading failed: {e}", exc_info=True)
            return
        
        # Sidebar controls
        with st.sidebar:
            st.header("🎛️ Graph Controls")
            
            try:
                # Layout and appearance controls
                layout_config = st.session_state.graph_controls.render_layout_controls()
                
                st.markdown("---")
                
                # Theorem filtering controls
                filter_config = st.session_state.graph_controls.render_filter_controls(theorems)
                
                st.markdown("---")
                
                # Node exploration controls
                available_nodes = [t.id for t in theorems]
                exploration_config = st.session_state.graph_controls.render_exploration_controls(available_nodes)
                
                st.markdown("---")
                
                # Performance information - will update with actual edge count after building graph
                st.session_state.graph_controls.render_performance_info(len(theorems), 0)
                
            except Exception as e:
                self.logger.error(f"Failed to render sidebar controls: {e}", exc_info=True)
                st.error("Failed to render controls. Using default settings.")
                # Provide fallback configuration
                layout_config = {"layout_algorithm": "spring", "color_by": "theorem_type", "node_size_factor": 1.0, "show_labels": True, "physics_enabled": True}
                filter_config = {"selected_types": ["all"], "validation_filter": "all", "confidence_range": (0.0, 1.0)}
                exploration_config = {"selected_node": None, "exploration_depth": 1, "highlight_neighbors": False}
        
        # Apply filters to theorems
        try:
            filtered_theorems = st.session_state.graph_viewer.filter_theorems_by_selection(
                theorems,
                filter_config["selected_types"],
                filter_config["validation_filter"],
                filter_config["confidence_range"]
            )
        except Exception as e:
            self.logger.error(f"Failed to filter theorems: {e}", exc_info=True)
            st.error("Error applying filters. Showing all theorems.")
            filtered_theorems = theorems
        
        # Main content area
        if filtered_theorems:
            # Display filter results
            st.subheader(f"📈 Interactive Graph ({len(filtered_theorems)} theorems)")
            
            # Create two columns for graph and details
            col1, col2 = st.columns([3, 1])
            
            with col1:
                try:
                    # Render the interactive graph
                    fig = st.session_state.graph_viewer.render_interactive_graph(
                        filtered_theorems,
                        selected_node=exploration_config["selected_node"],
                        **layout_config
                    )
                    
                    # Display graph with interaction
                    st.plotly_chart(
                        fig, 
                        use_container_width=True,
                        key="knowledge_graph",
                        config={'displayModeBar': True, 'displaylogo': False}
                    )
                    
                except Exception as e:
                    self.logger.error(f"Failed to render graph: {e}", exc_info=True)
                    st.error("Failed to render graph visualization. Please try refreshing the page.")
            
            with col2:
                # Selected node information panel
                st.subheader("🎯 Node Details")
                
                try:
                    if exploration_config["selected_node"]:
                        selected_theorem = next(
                            (t for t in filtered_theorems if t.id == exploration_config["selected_node"]), 
                            None
                        )
                        
                        if selected_theorem:
                            # Display theorem information
                            st.markdown(f"**ID:** `{selected_theorem.id}`")
                            st.markdown(f"**Type:** {selected_theorem.theorem_type_display}")
                            
                            # Validation status with color
                            status = selected_theorem.validation_evidence.validation_status if selected_theorem.validation_evidence else "Unknown"
                            if status == "PASS":
                                st.success(f"✅ Validated: {status}")
                            elif status == "FAIL":
                                st.error(f"❌ Validation: {status}")
                            else:
                                st.warning(f"⚠️ Validation: {status}")
                            
                            # Confidence score
                            confidence = selected_theorem.source_lineage.confidence if selected_theorem.source_lineage else 0.0
                            st.metric("Confidence", f"{confidence:.2f}")
                            
                            # Expandable sections
                            with st.expander("📝 View Statement"):
                                st.latex(selected_theorem.display_statement)
                            
                            with st.expander("🔗 View Connections"):
                                try:
                                    # Build graph to get connections
                                    graph = st.session_state.graph_viewer.graph_builder.build_theorem_graph(filtered_theorems)
                                    neighbors = st.session_state.graph_viewer.get_node_neighbors(
                                        graph, 
                                        exploration_config["selected_node"],
                                        exploration_config["exploration_depth"]
                                    )
                                    
                                    if neighbors:
                                        st.write(f"**Connected to {len(neighbors)} nodes:**")
                                        for neighbor in sorted(list(neighbors))[:10]:  # Show first 10
                                            st.write(f"• {neighbor}")
                                        if len(neighbors) > 10:
                                            st.write(f"... and {len(neighbors) - 10} more")
                                    else:
                                        st.info("No connections found at current depth")
                                        
                                except Exception as e:
                                    self.logger.error(f"Failed to analyze connections: {e}")
                                    st.error("Failed to analyze node connections")
                            
                            with st.expander("📊 Validation Details"):
                                if selected_theorem.validation_evidence:
                                    st.json(selected_theorem.validation_evidence.model_dump())
                                else:
                                    st.info("No validation evidence available")
                        
                        else:
                            st.error("Selected theorem not found in filtered results")
                    else:
                        st.info("👆 Select a node in the graph to view detailed information")
                        
                except Exception as e:
                    self.logger.error(f"Failed to render node details: {e}", exc_info=True)
                    st.error("Failed to load node details")
            
            # Graph statistics below
            st.markdown("---")
            st.subheader("📊 Graph Statistics")
            
            try:
                # Build graph for statistics
                graph = st.session_state.graph_viewer.graph_builder.build_theorem_graph(filtered_theorems)
                stats = st.session_state.graph_viewer.graph_builder.get_graph_statistics(graph)
                
                # Display statistics in columns
                stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                
                with stat_col1:
                    st.metric("Total Nodes", stats.get("total_nodes", 0))
                
                with stat_col2:
                    st.metric("Total Edges", stats.get("total_edges", 0))
                
                with stat_col3:
                    st.metric("Graph Density", f"{stats.get('density', 0):.3f}")
                
                with stat_col4:
                    st.metric("Components", stats.get("connected_components", 0))
                
                # Additional insights
                if stats.get("total_nodes", 0) > 0:
                    avg_degree = (2 * stats.get("total_edges", 0)) / stats.get("total_nodes", 1)
                    st.info(f"📈 Average node degree: {avg_degree:.2f}")
                    
            except Exception as e:
                self.logger.error(f"Failed to generate graph statistics: {e}", exc_info=True)
                st.error("Failed to generate graph statistics")
        
        else:
            # No theorems after filtering
            st.info("🔍 No theorems match the current filters.")
            st.markdown("**Try adjusting your selection:**")
            st.markdown("- Include more theorem types")
            st.markdown("- Expand confidence range")
            st.markdown("- Change validation filter")
            
            # Show available options
            with st.expander("View Available Data"):
                st.write(f"**Total theorems:** {len(theorems)}")
                available_types = list(set(t.theorem_type for t in theorems))
                st.write(f"**Available types:** {', '.join(available_types)}")
                validated_count = sum(1 for t in theorems if t.is_validated)
                st.write(f"**Validated theorems:** {validated_count}")
    
    def render_theorem_browser_page(self) -> None:
        """Render the theorem browser and search page."""
        st.title("📚 Theorem Browser")
        st.markdown("*Search, filter, and explore validated mathematical theorems*")
        
        # Search interface
        search_col, filter_col = st.columns([2, 1])
        
        with search_col:
            search_query = st.text_input(
                "Search theorems:",
                value=st.session_state.search_query,
                placeholder="Enter theorem statement, symbol, or description..."
            )
            st.session_state.search_query = search_query
        
        with filter_col:
            theorem_type_filter = st.selectbox(
                "Filter by type:",
                ["All", "Functional Equation", "Generalization", "Transformation", "Algebraic Identity"]
            )
        
        # Advanced filters
        with st.expander("🔧 Advanced Filters"):
            col1, col2, col3 = st.columns(3)
            with col1:
                validation_filter = st.selectbox("Validation Status", ["All", "Pass", "Fail"])
            with col2:
                confidence_range = st.slider("Confidence Range", 0.0, 1.0, (0.0, 1.0))
            with col3:
                symbols_filter = st.multiselect("Contains Symbols", ["x", "f", "a", "π", "e"])
        
        # Placeholder theorem list
        st.markdown("### Theorem Results")
        st.info("🚧 **Under Development**: Theorem browser will display searchable, filterable list of all validated theorems.")
        
        # Placeholder theorem cards
        for i in range(3):
            with st.container():
                st.markdown(f"""
                <div class="theorem-card">
                    <h4>Theorem THM_EXAMPLE_{i+1}</h4>
                    <div class="math-expression">∀x ∈ ℝ, f(x) = x² + 2x + 1</div>
                    <p><strong>Type:</strong> Functional Equation</p>
                    <p><strong>Validation:</strong> <span class="validation-pass">PASS</span> (100% confidence)</p>
                    <p><strong>Description:</strong> Example theorem demonstrating the interface structure.</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"View Details", key=f"theorem_{i}"):
                    st.session_state.selected_theorem = f"THM_EXAMPLE_{i+1}"
                    st.session_state.current_page = "Proof Viewer"
                    st.rerun()
    
    def render_proof_viewer_page(self) -> None:
        """Render the proof trace viewer page."""
        st.title("🔍 Proof Trace Viewer")
        st.markdown("*Step-by-step visualization of theorem validation and proof traces*")
        
        # Theorem selection
        if st.session_state.selected_theorem:
            st.info(f"Viewing proof for: **{st.session_state.selected_theorem}**")
        else:
            st.warning("No theorem selected. Go to Theorem Browser to select a theorem.")
            return
        
        # Proof trace controls
        col1, col2 = st.columns([1, 1])
        with col1:
            st.selectbox("Proof Type", ["Validation Trace", "Transformation Chain", "Symbol Resolution"])
        with col2:
            st.checkbox("Show intermediate steps", value=True)
        
        # Placeholder proof steps
        st.markdown("### Proof Steps")
        st.info("🚧 **Under Development**: Proof viewer will show step-by-step validation and transformation traces.")
        
        # Example proof steps
        proof_steps = [
            "Original formula: x² + 2x + 1",
            "Apply functional transformation: f(2x)",
            "Substitute and expand: 4x² + 4x + 1", 
            "Validate with test cases: 59 tests passed",
            "Generate natural language description"
        ]
        
        for i, step in enumerate(proof_steps):
            st.markdown(f"""
            <div class="proof-step">
                <strong>Step {i+1}:</strong> {step}
            </div>
            """, unsafe_allow_html=True)
        
        # Validation details
        st.markdown("### Validation Evidence")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Test Cases", "59")
        with col2:
            st.metric("Pass Rate", "100%")
        with col3:
            st.metric("Validation Time", "0.047s")
    
    def render_search_page(self) -> None:
        """Render the advanced search and filter page."""
        st.title("🔎 Advanced Search & Analytics")
        st.markdown("*Multi-faceted search with advanced filtering and analytics*")
        
        # Search tabs
        tab1, tab2, tab3 = st.tabs(["Text Search", "Symbol Search", "Pattern Search"])
        
        with tab1:
            st.text_input("Natural language search", placeholder="Find theorems about quadratic functions...")
            st.info("🚧 **Under Development**: Natural language search across theorem descriptions and mathematical content.")
        
        with tab2:
            st.multiselect("Search by symbols", ["x", "f", "a", "π", "e", "∀", "∃", "∈"])
            st.info("🚧 **Under Development**: Symbol-based search to find theorems containing specific mathematical symbols.")
        
        with tab3:
            st.selectbox("Pattern type", ["Functional patterns", "Algebraic patterns", "Transformation patterns"])
            st.info("🚧 **Under Development**: Pattern-based search using mathematical structure analysis.")
        
        # Analytics section
        st.markdown("---")
        st.subheader("📊 Knowledge Analytics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Theorem Distribution**")
            # Placeholder chart
            st.bar_chart({"Functional Equations": 3, "Generalizations": 3, "Transformations": 3, "Identities": 1})
        
        with col2:
            st.markdown("**Validation Metrics**")
            st.write("- Average confidence: 100%")
            st.write("- Total test cases: 180+")
            st.write("- Average validation time: 0.05s")
    
    def run(self) -> None:
        """Run the main application loop."""
        try:
            # Render sidebar navigation
            self.render_sidebar()
            
            # Render current page based on session state
            page = st.session_state.current_page
            
            if page == "Overview":
                self.render_overview_page()
            elif page == "Knowledge Graph":
                self.render_graph_page()
            elif page == "Theorem Browser":
                self.render_theorem_browser_page()
            elif page == "Proof Viewer":
                self.render_proof_viewer_page()
            elif page == "Search & Filter":
                self.render_search_page()
            else:
                st.error(f"Unknown page: {page}")
            
        except Exception as e:
            self.logger.error(f"Application error: {str(e)}")
            st.error(f"An error occurred: {str(e)}")
            st.info("Please refresh the page or contact support if the problem persists.")


def main():
    """Main entry point for the Streamlit application."""
    app = MathBotUI()
    app.run()


if __name__ == "__main__":
    main() 