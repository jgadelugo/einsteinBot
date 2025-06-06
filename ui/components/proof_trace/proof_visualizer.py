import streamlit as st
import logging
from typing import List, Optional
from ui.data.proof_models import ProofVisualizationSession, ProofMethodType, ProofStep
from ui.services.proof_service import ProofVisualizationService
from ui.config import UIConfig

class ProofVisualizer:
    """Main proof trace visualization component."""
    
    def __init__(self, config: UIConfig):
        self.config = config
        self.proof_service = ProofVisualizationService(config)
        self.logger = logging.getLogger(__name__)
        
    def render_proof_visualization(self, theorem, session: ProofVisualizationSession) -> None:
        """Main proof visualization interface."""
        
        # Load proof data if not available
        if session.proof_data is None:
            with st.spinner("Loading proof data..."):
                session.proof_data = self.proof_service.load_proof_data(theorem)
        
        # Header with theorem information  
        self._render_theorem_header(theorem)
        
        # Proof method selector
        self._render_method_selector(session)
        
        # Main visualization area
        self._render_proof_content(session)
        
        # Export controls
        self._render_export_controls(session)
    
    def _render_theorem_header(self, theorem) -> None:
        """Render theorem information header."""
        st.markdown("### 🧮 Theorem Information")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"**Statement:** {theorem.statement}")
            if hasattr(theorem, 'theorem_type'):
                st.markdown(f"**Type:** {theorem.theorem_type_display}")
            if hasattr(theorem, 'short_id'):
                st.markdown(f"**ID:** {theorem.short_id}")
        
        with col2:
            if hasattr(theorem, 'validation_evidence') and theorem.validation_evidence:
                st.metric("Validation", f"{theorem.validation_evidence.pass_rate:.1%}")
                st.metric("Tests", theorem.validation_evidence.total_tests)
                st.metric("Confidence", theorem.validation_evidence.confidence_level)
    
    def _render_method_selector(self, session: ProofVisualizationSession) -> None:
        """Render proof method selection interface."""
        st.markdown("### 🔍 Proof Method")
        
        # Available methods based on data
        available_methods = []
        if session.proof_data:
            if session.proof_data.symbolic_steps:
                available_methods.append(("🔬 Symbolic Proof", ProofMethodType.SYMBOLIC))
            if session.proof_data.rule_steps:
                available_methods.append(("🔄 Rule-Based", ProofMethodType.RULE_BASED))
            if session.proof_data.validation_steps:
                available_methods.append(("✅ Validation Trace", ProofMethodType.VALIDATION))
        
        if not available_methods:
            st.warning("No proof methods available for this theorem.")
            return
        
        # Method selector
        method_options = [label for label, _ in available_methods]
        method_values = [method for _, method in available_methods]
        
        # Find current method index
        current_index = 0
        try:
            current_index = method_values.index(session.current_method)
        except ValueError:
            # Current method not available, use first available
            session.current_method = method_values[0]
        
        selected_index = st.selectbox(
            "Select proof method to visualize:",
            range(len(method_options)),
            index=current_index,
            format_func=lambda i: method_options[i],
            key="proof_method_selector"
        )
        
        session.current_method = method_values[selected_index]
    
    def _render_proof_content(self, session: ProofVisualizationSession) -> None:
        """Render main proof content based on selected method."""
        if not session.proof_data:
            st.error("No proof data available.")
            return
        
        if session.current_method == ProofMethodType.SYMBOLIC:
            self._render_symbolic_proof(session.proof_data.symbolic_steps, session)
        elif session.current_method == ProofMethodType.RULE_BASED:
            self._render_rule_proof(session.proof_data.rule_steps, session)
        elif session.current_method == ProofMethodType.VALIDATION:
            self._render_validation_proof(session.proof_data.validation_steps, session)
    
    def _render_symbolic_proof(self, steps: List[ProofStep], session: ProofVisualizationSession) -> None:
        """Render symbolic proof visualization."""
        st.markdown("### 🔬 Symbolic Proof Steps")
        
        if not steps:
            st.info("No symbolic proof steps available for this theorem.")
            st.markdown("**Note:** This may indicate that the theorem doesn't require symbolic manipulation or that the proof engine hasn't been run yet.")
            return
        
        # Step navigation
        current_step = self._render_step_navigation(steps, session)
        
        # Current step visualization
        if 0 <= current_step < len(steps):
            self._render_proof_step(steps[current_step])
    
    def _render_rule_proof(self, steps: List[ProofStep], session: ProofVisualizationSession) -> None:
        """Render rule-based proof visualization."""
        st.markdown("### 🔄 Rule-Based Transformations")
        
        if not steps:
            st.info("No rule-based transformations available for this theorem.")
            st.markdown("**Note:** This theorem may not have transformation chains in its source lineage.")
            return
        
        # Step navigation
        current_step = self._render_step_navigation(steps, session)
        
        # Current step visualization
        if 0 <= current_step < len(steps):
            self._render_proof_step(steps[current_step])
    
    def _render_validation_proof(self, steps: List[ProofStep], session: ProofVisualizationSession) -> None:
        """Render validation proof visualization."""
        st.markdown("### ✅ Validation Evidence")
        
        if not steps:
            st.info("No validation steps available for this theorem.")
            st.markdown("**Note:** This theorem may not have been validated yet.")
            return
        
        # Step navigation
        current_step = self._render_step_navigation(steps, session)
        
        # Current step visualization
        if 0 <= current_step < len(steps):
            self._render_proof_step(steps[current_step])
    
    def _render_step_navigation(self, steps: List[ProofStep], session: ProofVisualizationSession) -> int:
        """Render step navigation controls."""
        if not steps:
            return 0
        
        # Navigation controls
        col1, col2, col3, col4 = st.columns([1, 3, 1, 1])
        
        with col1:
            if st.button("⏮️ First", disabled=session.current_step == 0, key="first_step"):
                session.current_step = 0
                st.rerun()
                
        with col2:
            # Progress bar
            progress = (session.current_step + 1) / len(steps)
            st.progress(progress)
            st.markdown(f"**Step {session.current_step + 1} of {len(steps)}**")
            
        with col3:
            if st.button("⏭️ Next", disabled=session.current_step >= len(steps) - 1, key="next_step"):
                session.current_step = min(session.current_step + 1, len(steps) - 1)
                st.rerun()
                
        with col4:
            if st.button("⏭️ Last", disabled=session.current_step >= len(steps) - 1, key="last_step"):
                session.current_step = len(steps) - 1
                st.rerun()
        
        # Step selector
        selected_step = st.select_slider(
            "Jump to step:",
            options=list(range(len(steps))),
            value=session.current_step,
            format_func=lambda x: f"Step {x + 1}: {steps[x].title}",
            key=f"step_slider_{session.current_method.value}"
        )
        
        if selected_step != session.current_step:
            session.current_step = selected_step
            
        return session.current_step
    
    def _render_proof_step(self, step: ProofStep) -> None:
        """Render individual proof step."""
        # Step header with styling
        st.markdown(f"#### Step {step.step_number}: {step.title}")
        
        # Expression transformation
        col1, col2, col3 = st.columns([5, 1, 5])
        
        with col1:
            st.markdown("**From:**")
            self._render_expression(step.expression_from)
            
        with col2:
            st.markdown("<div style='text-align: center; padding-top: 20px; font-size: 24px; color: #1f77b4;'>→</div>", 
                       unsafe_allow_html=True)
            
        with col3:
            st.markdown("**To:**")
            self._render_expression(step.expression_to)
        
        # Rule and justification
        if step.rule_applied:
            st.markdown(f"**Rule Applied:** `{step.rule_applied}`")
            
        if step.justification:
            st.markdown(f"**Justification:** {step.justification}")
        
        # Metrics row
        metrics_cols = st.columns(4)
        with metrics_cols[0]:
            st.metric("Confidence", f"{step.confidence:.2%}")
        with metrics_cols[1]:
            if step.execution_time > 0:
                st.metric("Time", f"{step.execution_time:.3f}s")
            else:
                st.metric("Time", "N/A")
        with metrics_cols[2]:
            st.metric("Method", step.method_type.value.title())
        with metrics_cols[3]:
            if step.metadata:
                st.metric("Details", f"{len(step.metadata)} items")
        
        # Metadata details
        if step.metadata:
            with st.expander("📋 Step Details", expanded=False):
                for key, value in step.metadata.items():
                    if isinstance(value, (list, dict)):
                        st.json({key: value})
                    else:
                        st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
    
    def _render_expression(self, expression: str) -> None:
        """Render mathematical expression with LaTeX if possible."""
        if not expression or expression.strip() == "":
            st.markdown("*(empty)*")
            return
            
        try:
            # Try LaTeX rendering first for mathematical expressions
            if self._is_latex_compatible(expression):
                # Clean up for LaTeX
                latex_expr = self._convert_to_latex(expression)
                st.latex(latex_expr)
            else:
                # Fallback to formatted code block
                st.code(expression, language="text")
        except Exception as e:
            # Final fallback to plain text
            st.code(expression, language="text")
            if self.config.ui_settings.get("show_debug", False):
                st.caption(f"LaTeX rendering failed: {e}")
    
    def _is_latex_compatible(self, expression: str) -> bool:
        """Check if expression can be rendered as LaTeX."""
        if not expression:
            return False
            
        # Simple heuristic - improve based on actual expressions
        latex_indicators = ['\\', '^', '_', '{', '}', 'frac', 'sqrt', 'sum', 'int', '=', '+', '-', '*', '/', '(', ')']
        mathematical_symbols = ['sin', 'cos', 'tan', 'log', 'ln', 'exp', 'pi', 'theta', 'alpha', 'beta', 'gamma']
        
        # If it contains mathematical operations or symbols, try LaTeX
        return any(indicator in expression.lower() for indicator in latex_indicators + mathematical_symbols)
    
    def _convert_to_latex(self, expression: str) -> str:
        """Convert expression to LaTeX format."""
        if not expression:
            return expression
            
        # Basic LaTeX cleanup
        latex_expr = expression.strip()
        
        # Remove common Python/SymPy syntax that doesn't work in LaTeX
        replacements = {
            '**': '^',
            '*': r' \cdot ',
            'sqrt(': r'\sqrt{',
            'sin(': r'\sin(',
            'cos(': r'\cos(',
            'tan(': r'\tan(',
            'log(': r'\log(',
            'ln(': r'\ln(',
            'exp(': r'\exp(',
        }
        
        for old, new in replacements.items():
            latex_expr = latex_expr.replace(old, new)
        
        return latex_expr
    
    def _render_export_controls(self, session: ProofVisualizationSession) -> None:
        """Render export and sharing controls."""
        st.markdown("---")
        st.markdown("### 📄 Export & Share")
        
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        with col1:
            if st.button("📋 Copy Steps", help="Copy proof steps to clipboard"):
                self._export_steps_text(session)
        
        with col2:
            if st.button("📁 Export JSON", help="Export proof data as JSON"):
                self._export_json(session)
        
        with col3:
            if st.button("📄 Generate LaTeX", help="Generate LaTeX representation"):
                self._export_latex(session)
        
        with col4:
            if st.button("📊 Proof Report", help="Generate comprehensive proof report"):
                self._generate_report(session)
    
    def _export_steps_text(self, session: ProofVisualizationSession) -> None:
        """Export proof steps as formatted text."""
        if not session.proof_data:
            st.warning("No proof data to export.")
            return
            
        # Get current method steps
        steps = []
        if session.current_method == ProofMethodType.SYMBOLIC:
            steps = session.proof_data.symbolic_steps
        elif session.current_method == ProofMethodType.RULE_BASED:
            steps = session.proof_data.rule_steps
        elif session.current_method == ProofMethodType.VALIDATION:
            steps = session.proof_data.validation_steps
            
        if not steps:
            st.warning("No steps available for the selected method.")
            return
            
        # Format steps as text
        text_output = f"Proof Steps ({session.current_method.value.title()})\n"
        text_output += "=" * 50 + "\n\n"
        
        for step in steps:
            text_output += f"Step {step.step_number}: {step.title}\n"
            text_output += f"From: {step.expression_from}\n"
            text_output += f"To: {step.expression_to}\n"
            if step.rule_applied:
                text_output += f"Rule: {step.rule_applied}\n"
            if step.justification:
                text_output += f"Justification: {step.justification}\n"
            text_output += f"Confidence: {step.confidence:.2%}\n"
            text_output += "-" * 30 + "\n\n"
        
        st.text_area("Proof Steps (Copy with Ctrl+A, Ctrl+C):", text_output, height=200)
        st.success("Proof steps formatted for copying!")
    
    def _export_json(self, session: ProofVisualizationSession) -> None:
        """Export proof data as JSON."""
        if not session.proof_data:
            st.warning("No proof data to export.")
            return
            
        try:
            import json
            from dataclasses import asdict
            
            # Convert to dict for JSON serialization
            proof_dict = asdict(session.proof_data)
            
            # Convert datetime objects to strings
            def convert_datetime(obj):
                if hasattr(obj, 'isoformat'):
                    return obj.isoformat()
                return obj
            
            json_output = json.dumps(proof_dict, indent=2, default=convert_datetime)
            st.text_area("JSON Export (Copy with Ctrl+A, Ctrl+C):", json_output, height=200)
            st.success("Proof data exported as JSON!")
            
        except Exception as e:
            st.error(f"Failed to export JSON: {e}")
    
    def _export_latex(self, session: ProofVisualizationSession) -> None:
        """Generate LaTeX representation of proof."""
        if not session.proof_data:
            st.warning("No proof data to export.")
            return
            
        latex_output = "\\documentclass{article}\n"
        latex_output += "\\usepackage{amsmath}\n"
        latex_output += "\\usepackage{amssymb}\n"
        latex_output += "\\begin{document}\n\n"
        latex_output += f"\\section{{Theorem {session.theorem_id}}}\n\n"
        
        # Add current method steps
        steps = []
        if session.current_method == ProofMethodType.SYMBOLIC:
            steps = session.proof_data.symbolic_steps
            latex_output += "\\subsection{Symbolic Proof}\n\n"
        elif session.current_method == ProofMethodType.RULE_BASED:
            steps = session.proof_data.rule_steps
            latex_output += "\\subsection{Rule-Based Transformations}\n\n"
        elif session.current_method == ProofMethodType.VALIDATION:
            steps = session.proof_data.validation_steps
            latex_output += "\\subsection{Validation Evidence}\n\n"
        
        for step in steps:
            latex_output += f"\\textbf{{Step {step.step_number}: {step.title}}}\n\n"
            latex_output += f"From: {step.expression_from}\n\n"
            latex_output += f"To: {step.expression_to}\n\n"
            if step.justification:
                latex_output += f"Justification: {step.justification}\n\n"
        
        latex_output += "\\end{document}"
        
        st.text_area("LaTeX Export (Copy with Ctrl+A, Ctrl+C):", latex_output, height=200)
        st.success("LaTeX document generated!")
    
    def _generate_report(self, session: ProofVisualizationSession) -> None:
        """Generate comprehensive proof report."""
        if not session.proof_data:
            st.warning("No proof data to generate report.")
            return
            
        st.markdown("### 📊 Proof Analysis Report")
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_steps = (len(session.proof_data.symbolic_steps) + 
                          len(session.proof_data.rule_steps) + 
                          len(session.proof_data.validation_steps))
            st.metric("Total Steps", total_steps)
            
        with col2:
            available_methods = len(session.proof_data.success_methods)
            st.metric("Available Methods", available_methods)
            
        with col3:
            if session.proof_data.total_execution_time > 0:
                st.metric("Total Time", f"{session.proof_data.total_execution_time:.3f}s")
            else:
                st.metric("Total Time", "N/A")
        
        # Method breakdown
        st.markdown("#### Method Breakdown")
        method_data = []
        if session.proof_data.symbolic_steps:
            method_data.append(["Symbolic", len(session.proof_data.symbolic_steps)])
        if session.proof_data.rule_steps:
            method_data.append(["Rule-Based", len(session.proof_data.rule_steps)])
        if session.proof_data.validation_steps:
            method_data.append(["Validation", len(session.proof_data.validation_steps)])
            
        if method_data:
            import pandas as pd
            df = pd.DataFrame(method_data, columns=["Method", "Steps"])
            st.dataframe(df, use_container_width=True)
        
        st.success("Proof analysis report generated!") 