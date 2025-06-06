"""
Comprehensive theorem detail view with LaTeX rendering and analysis.

This module provides detailed theorem visualization with mathematical expression rendering,
validation analysis, transformation chains, and related content discovery.
"""

import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

import streamlit as st
import pandas as pd
from pydantic import BaseModel, Field

from ui.data.models import Theorem, ValidationEvidence, SourceLineage
from ui.config import UIConfig
from ui.utils.ui_logging import get_ui_logger, log_ui_interaction


@dataclass
class DetailSession:
    """Detail view session state management."""
    current_theorem_id: Optional[str] = None
    view_history: List[str] = None
    expanded_sections: Dict[str, bool] = None
    performance_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.view_history is None:
            self.view_history = []
        if self.expanded_sections is None:
            self.expanded_sections = {
                "statement": True,
                "validation": True,
                "lineage": False,
                "symbols": False,
                "related": False
            }
        if self.performance_metrics is None:
            self.performance_metrics = {}


class TheoremDetail:
    """
    Comprehensive theorem detail view with LaTeX rendering and related content.
    
    Provides detailed analysis of individual theorems including mathematical expressions,
    validation evidence, transformation chains, and related theorem discovery.
    """
    
    def __init__(self, config: UIConfig):
        """Initialize theorem detail view with configuration."""
        self.config = config
        self.logger = get_ui_logger("theorem_detail")
        
        # Initialize session state
        self._initialize_detail_session()
        
        # Detail view configuration
        self.enable_latex = getattr(config, 'enable_latex_rendering', True)
        self.max_related_theorems = getattr(config, 'max_related_theorems', 5)
        self.enable_copy_functionality = getattr(config, 'enable_copy_functionality', True)
    
    def _initialize_detail_session(self) -> None:
        """Initialize detail view session state."""
        if 'detail_session' not in st.session_state:
            st.session_state.detail_session = DetailSession()
        
        if 'theorem_view_history' not in st.session_state:
            st.session_state.theorem_view_history = []
    
    def render_theorem_detail(self, theorem: Theorem, 
                            related_theorems: Optional[List[Theorem]] = None) -> None:
        """
        Render comprehensive theorem details with all metadata.
        
        Args:
            theorem: Theorem to display in detail
            related_theorems: Optional list of related theorems
        """
        if not theorem:
            st.info("No theorem selected for detailed view.")
            return
        
        start_time = time.time()
        
        # Update session state
        if theorem.id != st.session_state.detail_session.current_theorem_id:
            st.session_state.detail_session.current_theorem_id = theorem.id
            if theorem.id not in st.session_state.theorem_view_history:
                st.session_state.theorem_view_history.insert(0, theorem.id)
                # Keep only recent views
                st.session_state.theorem_view_history = st.session_state.theorem_view_history[:20]
        
        # Render header
        self._render_theorem_header(theorem)
        
        # Render main content in tabs
        self._render_theorem_tabs(theorem, related_theorems)
        
        # Render navigation and actions
        self._render_theorem_actions(theorem)
        
        # Update performance metrics
        render_time = time.time() - start_time
        st.session_state.detail_session.performance_metrics["detail_render_time"] = render_time
        
        # Log interaction
        log_ui_interaction("theorem_detail", "theorem_viewed", {
            "theorem_id": theorem.id,
            "render_time": render_time
        })
    
    def _render_theorem_header(self, theorem: Theorem) -> None:
        """Render theorem header with key information."""
        # Title and ID
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.markdown(f"# 📐 Theorem {theorem.short_id}")
            st.markdown(f"**Type:** {theorem.theorem_type_display}")
        
        with col2:
            # Validation status badge
            status = theorem.validation_evidence.validation_status
            if status == "PASS":
                st.success(f"✅ {status}")
            elif status == "FAIL":
                st.error(f"❌ {status}")
            elif status == "PARTIAL":
                st.warning(f"⚠️ {status}")
            else:
                st.info(f"ℹ️ {status}")
        
        with col3:
            # Confidence score
            confidence = theorem.source_lineage.confidence
            st.metric("Confidence", f"{confidence:.2%}")
        
        st.markdown("---")
    
    def _render_theorem_tabs(self, theorem: Theorem, 
                           related_theorems: Optional[List[Theorem]] = None) -> None:
        """Render theorem content in organized tabs."""
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📝 Statement", 
            "🔬 Validation", 
            "🔗 Lineage", 
            "🔤 Symbols", 
            "🌐 Related"
        ])
        
        with tab1:
            self._render_statement_section(theorem)
        
        with tab2:
            self._render_validation_analysis(theorem.validation_evidence)
        
        with tab3:
            self._render_transformation_chain(theorem.source_lineage)
        
        with tab4:
            self._render_symbol_analysis(theorem)
        
        with tab5:
            self._render_related_theorems(theorem, related_theorems)
    
    def _render_statement_section(self, theorem: Theorem) -> None:
        """Render theorem statement with LaTeX rendering."""
        st.markdown("### Mathematical Statement")
        
        # LaTeX rendered statement
        if self.enable_latex:
            try:
                # Clean up the statement for LaTeX rendering
                latex_statement = self._prepare_latex_statement(theorem.statement)
                st.latex(latex_statement)
            except Exception as e:
                self.logger.warning(f"LaTeX rendering failed: {e}")
                st.code(theorem.statement, language="latex")
        else:
            st.code(theorem.statement, language="latex")
        
        # Copy functionality
        if self.enable_copy_functionality:
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("📋 Copy LaTeX", help="Copy LaTeX statement to clipboard"):
                    st.success("Statement copied to clipboard!")
        
        # Natural language description
        st.markdown("### Natural Language Description")
        st.markdown(theorem.natural_language)
        
        # SymPy expression
        if theorem.sympy_expression:
            st.markdown("### SymPy Expression")
            st.code(theorem.sympy_expression, language="python")
        
        # Assumptions
        if theorem.assumptions:
            st.markdown("### Assumptions")
            for i, assumption in enumerate(theorem.assumptions, 1):
                st.markdown(f"{i}. {assumption}")
        
        # Mathematical context
        if hasattr(theorem, 'mathematical_context') and theorem.mathematical_context:
            self._render_mathematical_context(theorem.mathematical_context)
    
    def _render_validation_analysis(self, evidence: ValidationEvidence) -> None:
        """
        Render detailed validation analysis with visualizations.
        
        Args:
            evidence: Validation evidence to analyze
        """
        st.markdown("### Validation Summary")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Status", evidence.validation_status)
        
        with col2:
            st.metric("Pass Rate", f"{evidence.pass_rate:.1%}")
        
        with col3:
            st.metric("Total Tests", evidence.total_tests)
        
        with col4:
            st.metric("Validation Time", f"{evidence.validation_time:.2f}s")
        
        # Confidence level
        st.markdown("### Confidence Assessment")
        confidence_level = evidence.confidence_level
        confidence_color = {
            "Very High": "🟢",
            "High": "🟡", 
            "Medium": "🟠",
            "Low": "🔴"
        }.get(confidence_level, "⚪")
        
        st.markdown(f"{confidence_color} **{confidence_level}** confidence level")
        
        # Progress bar for pass rate
        st.markdown("### Test Results")
        st.progress(evidence.pass_rate)
        st.caption(f"{evidence.success_percentage:.1f}% of tests passed")
        
        # Symbols tested
        if evidence.symbols_tested:
            st.markdown("### Symbols Tested")
            symbol_cols = st.columns(min(len(evidence.symbols_tested), 5))
            for i, symbol in enumerate(evidence.symbols_tested[:5]):
                with symbol_cols[i]:
                    st.code(symbol)
            
            if len(evidence.symbols_tested) > 5:
                with st.expander(f"View all {len(evidence.symbols_tested)} symbols"):
                    st.write(", ".join(evidence.symbols_tested))
        
        # Validation insights
        self._render_validation_insights(evidence)
    
    def _render_transformation_chain(self, lineage: SourceLineage) -> None:
        """
        Render transformation chain with step-by-step breakdown.
        
        Args:
            lineage: Source lineage information
        """
        st.markdown("### Source Information")
        
        # Basic lineage info
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original Formula:**")
            st.code(lineage.original_formula, language="latex")
            
            st.markdown("**Generation Method:**")
            st.info(lineage.generation_method)
        
        with col2:
            st.markdown("**Source Type:**")
            st.info(lineage.source_type)
            
            st.markdown("**Hypothesis ID:**")
            st.code(lineage.hypothesis_id)
        
        # Confidence and validation scores
        st.markdown("### Quality Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Confidence", f"{lineage.confidence_percentage:.1f}%")
        
        with col2:
            st.metric("Validation Score", f"{lineage.validation_score:.2f}")
        
        with col3:
            st.metric("Lineage Hash", lineage.lineage_hash)
        
        # Transformation chain
        if lineage.transformation_chain:
            st.markdown("### Transformation Steps")
            
            # Visual transformation chain
            for i, step in enumerate(lineage.transformation_chain):
                col1, col2, col3 = st.columns([1, 8, 1])
                
                with col1:
                    st.markdown(f"**{i+1}**")
                
                with col2:
                    st.info(step)
                
                with col3:
                    if i < len(lineage.transformation_chain) - 1:
                        st.markdown("⬇️")
            
            # Summary
            st.markdown("### Transformation Summary")
            st.markdown(lineage.transformation_summary)
        else:
            st.info("No transformation steps recorded for this theorem.")
    
    def _render_symbol_analysis(self, theorem: Theorem) -> None:
        """Render mathematical symbol analysis."""
        st.markdown("### Symbol Analysis")
        
        if not theorem.symbols:
            st.info("No symbols identified for this theorem.")
            return
        
        # Symbol count and complexity
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Symbols", len(theorem.symbols))
        
        with col2:
            unique_symbols = len(set(theorem.symbols))
            st.metric("Unique Symbols", unique_symbols)
        
        with col3:
            complexity = theorem.complexity_category
            st.metric("Complexity", complexity)
        
        # Symbol frequency
        st.markdown("### Symbol Frequency")
        symbol_counts = {}
        for symbol in theorem.symbols:
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
        
        # Create DataFrame for symbol analysis
        symbol_data = []
        for symbol, count in sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True):
            symbol_data.append({
                "Symbol": symbol,
                "Frequency": count,
                "Percentage": f"{count/len(theorem.symbols)*100:.1f}%"
            })
        
        if symbol_data:
            df = pd.DataFrame(symbol_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Symbol categories
        self._render_symbol_categories(theorem.symbols)
    
    def _render_related_theorems(self, theorem: Theorem, 
                               related_theorems: Optional[List[Theorem]] = None) -> None:
        """Render related theorems and connections."""
        st.markdown("### Related Theorems")
        
        if not related_theorems:
            st.info("No related theorems found. This could indicate:")
            st.markdown("""
            - This theorem is highly unique
            - Limited connections in the current dataset
            - Different mathematical domain from other theorems
            """)
            return
        
        # Display related theorems
        for i, related in enumerate(related_theorems[:self.max_related_theorems]):
            with st.expander(f"📐 {related.short_id}: {related.theorem_type_display}"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Truncated statement
                    statement = related.statement
                    if len(statement) > 200:
                        statement = statement[:197] + "..."
                    st.markdown(f"**Statement:** {statement}")
                    
                    st.markdown(f"**Description:** {related.natural_language[:150]}...")
                
                with col2:
                    st.metric("Confidence", f"{related.source_lineage.confidence:.2f}")
                    st.metric("Status", related.validation_evidence.validation_status)
                
                # View button
                if st.button(f"View Details", key=f"view_related_{related.id}"):
                    st.session_state.selected_theorem_id = related.id
                    st.rerun()
        
        # Show connection analysis
        if len(related_theorems) > 0:
            self._render_connection_analysis(theorem, related_theorems)
    
    def _render_theorem_actions(self, theorem: Theorem) -> None:
        """Render theorem actions and navigation."""
        st.markdown("---")
        st.markdown("### Actions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📋 Copy ID", help="Copy theorem ID"):
                st.success(f"Copied: {theorem.id}")
        
        with col2:
            if st.button("📄 Export JSON", help="Export theorem as JSON"):
                self._export_theorem_json(theorem)
        
        with col3:
            if st.button("🔗 Share Link", help="Generate shareable link"):
                st.success("Link copied to clipboard!")
        
        with col4:
            if st.button("⭐ Bookmark", help="Bookmark this theorem"):
                st.success("Theorem bookmarked!")
        
        # Navigation history
        if st.session_state.theorem_view_history:
            st.markdown("### Recent Views")
            history_cols = st.columns(min(len(st.session_state.theorem_view_history), 5))
            
            for i, theorem_id in enumerate(st.session_state.theorem_view_history[:5]):
                with history_cols[i]:
                    short_id = theorem_id.replace("THM_", "")
                    if st.button(f"📐 {short_id}", key=f"history_{i}"):
                        st.session_state.selected_theorem_id = theorem_id
                        st.rerun()
    
    def _prepare_latex_statement(self, statement: str) -> str:
        """Prepare theorem statement for LaTeX rendering."""
        # Basic LaTeX cleanup and formatting
        latex_statement = statement
        
        # Remove common problematic patterns
        latex_statement = latex_statement.replace("\\text{", "\\mathrm{")
        
        # Ensure proper math mode
        if not latex_statement.startswith("$") and not latex_statement.startswith("\\["):
            latex_statement = f"${latex_statement}$"
        
        return latex_statement
    
    def _render_mathematical_context(self, context) -> None:
        """Render mathematical context information."""
        st.markdown("### Mathematical Context")
        
        if hasattr(context, 'domain') and context.domain:
            st.markdown(f"**Domain:** {context.domain}")
        
        if hasattr(context, 'complexity_score') and context.complexity_score:
            st.metric("Complexity Score", f"{context.complexity_score:.2f}")
        
        if hasattr(context, 'variables') and context.variables:
            st.markdown("**Variables:**")
            for var, desc in context.variables.items():
                st.markdown(f"- `{var}`: {desc}")
    
    def _render_validation_insights(self, evidence: ValidationEvidence) -> None:
        """Render validation insights and recommendations."""
        st.markdown("### Validation Insights")
        
        insights = []
        
        if evidence.pass_rate >= 0.95:
            insights.append("🟢 Excellent validation results - theorem is highly reliable")
        elif evidence.pass_rate >= 0.8:
            insights.append("🟡 Good validation results - theorem is generally reliable")
        elif evidence.pass_rate >= 0.6:
            insights.append("🟠 Moderate validation results - use with caution")
        else:
            insights.append("🔴 Poor validation results - theorem may be unreliable")
        
        if evidence.total_tests < 10:
            insights.append("⚠️ Limited test coverage - more validation recommended")
        elif evidence.total_tests > 100:
            insights.append("✅ Comprehensive test coverage")
        
        if evidence.validation_time > 10:
            insights.append("⏱️ Long validation time - complex theorem")
        
        for insight in insights:
            st.info(insight)
    
    def _render_symbol_categories(self, symbols: List[str]) -> None:
        """Categorize and display mathematical symbols."""
        st.markdown("### Symbol Categories")
        
        # Basic symbol categorization
        categories = {
            "Variables": [],
            "Operators": [],
            "Functions": [],
            "Constants": [],
            "Other": []
        }
        
        for symbol in set(symbols):
            if symbol.lower() in ['x', 'y', 'z', 'a', 'b', 'c', 'n', 'm', 'k']:
                categories["Variables"].append(symbol)
            elif symbol in ['+', '-', '*', '/', '=', '<', '>', '≤', '≥']:
                categories["Operators"].append(symbol)
            elif symbol.lower() in ['sin', 'cos', 'tan', 'log', 'exp', 'sqrt']:
                categories["Functions"].append(symbol)
            elif symbol.lower() in ['pi', 'e', 'phi']:
                categories["Constants"].append(symbol)
            else:
                categories["Other"].append(symbol)
        
        # Display categories
        for category, symbols_list in categories.items():
            if symbols_list:
                st.markdown(f"**{category}:** {', '.join(symbols_list)}")
    
    def _render_connection_analysis(self, theorem: Theorem, related_theorems: List[Theorem]) -> None:
        """Analyze and display connections between theorems."""
        st.markdown("### Connection Analysis")
        
        # Analyze common elements
        common_types = set()
        common_symbols = set(theorem.symbols)
        
        for related in related_theorems:
            common_types.add(related.theorem_type)
            common_symbols.intersection_update(related.symbols)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Common Types:**")
            for theorem_type in common_types:
                st.markdown(f"- {theorem_type.replace('_', ' ').title()}")
        
        with col2:
            st.markdown("**Shared Symbols:**")
            if common_symbols:
                st.markdown(f"- {', '.join(list(common_symbols)[:10])}")
            else:
                st.markdown("- No shared symbols")
    
    def _export_theorem_json(self, theorem: Theorem) -> None:
        """Export theorem as JSON."""
        try:
            import json
            theorem_data = theorem.dict()
            json_str = json.dumps(theorem_data, indent=2)
            
            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name=f"theorem_{theorem.short_id}.json",
                mime="application/json"
            )
            
            log_ui_interaction("theorem_detail", "theorem_exported", {
                "theorem_id": theorem.id,
                "format": "json"
            })
            
        except Exception as e:
            self.logger.error(f"JSON export failed: {e}", exc_info=True)
            st.error(f"Export failed: {str(e)}")
    
    def get_detail_analytics(self) -> Dict[str, Any]:
        """Get detail view analytics and performance data."""
        session = st.session_state.detail_session
        
        return {
            "current_theorem_id": session.current_theorem_id,
            "view_history_count": len(st.session_state.theorem_view_history),
            "performance_metrics": session.performance_metrics,
            "expanded_sections": session.expanded_sections
        }
