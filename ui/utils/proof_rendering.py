import streamlit as st
import re
from typing import Optional
import logging

class MathematicalRenderer:
    """Enhanced mathematical expression rendering for proofs."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def render_mathematical_step(self, step_title: str, from_expr: str, to_expr: str, 
                                rule: Optional[str] = None, justification: str = "") -> None:
        """Render a complete mathematical transformation step."""
        
        st.markdown(f"#### {step_title}")
        
        # Main transformation display
        col1, col2, col3 = st.columns([5, 1, 5])
        
        with col1:
            st.markdown("**From:**")
            self.render_expression(from_expr)
            
        with col2:
            # Arrow with rule annotation
            arrow_html = """
            <div style='text-align: center; padding-top: 20px;'>
                <div style='font-size: 24px; color: #1f77b4;'>→</div>
            """
            if rule:
                arrow_html += f"<div style='font-size: 10px; color: #666;'>{rule}</div>"
            arrow_html += "</div>"
            
            st.markdown(arrow_html, unsafe_allow_html=True)
            
        with col3:
            st.markdown("**To:**")
            self.render_expression(to_expr)
        
        # Justification
        if justification:
            st.markdown(f"**Explanation:** {justification}")
    
    def render_expression(self, expression: str) -> None:
        """Render mathematical expression with best available method."""
        if not expression:
            st.markdown("*(empty)*")
            return
        
        try:
            # Try LaTeX rendering
            latex_expr = self.convert_to_latex(expression)
            if latex_expr != expression:  # Conversion occurred
                st.latex(latex_expr)
            else:
                # Fallback to formatted code
                st.code(expression, language="text")
                
        except Exception as e:
            self.logger.debug(f"Expression rendering failed for '{expression}': {e}")
            st.code(expression, language="text")
    
    def convert_to_latex(self, expression: str) -> str:
        """Convert expression to LaTeX format."""
        if not expression or not isinstance(expression, str):
            return expression
        
        # Basic conversions
        latex_expr = expression
        
        # Common mathematical symbols
        replacements = {
            '**': '^',
            '*': r' \cdot ',
            '>=': r' \geq ',
            '<=': r' \leq ',
            '!=': r' \neq ',
            'sqrt': r'\sqrt',
            'sin': r'\sin',
            'cos': r'\cos',
            'tan': r'\tan',
            'pi': r'\pi',
            'infinity': r'\infty',
            'alpha': r'\alpha',
            'beta': r'\beta',
            'gamma': r'\gamma',
            'theta': r'\theta',
            'lambda': r'\lambda',
            'sum': r'\sum',
            'integral': r'\int'
        }
        
        for old, new in replacements.items():
            latex_expr = latex_expr.replace(old, new)
        
        # Handle fractions: a/b -> \frac{a}{b}
        latex_expr = re.sub(r'(\w+)/(\w+)', r'\\frac{\1}{\2}', latex_expr)
        
        # Handle subscripts: a_b -> a_{b}
        latex_expr = re.sub(r'(\w)_(\w)', r'\1_{\2}', latex_expr)
        
        return latex_expr
    
    def render_proof_overview(self, steps_count: int, success_rate: float, 
                            total_time: float) -> None:
        """Render proof overview statistics."""
        st.markdown("### 📊 Proof Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Steps", steps_count)
        with col2:
            st.metric("Success Rate", f"{success_rate:.1%}")
        with col3:
            st.metric("Total Time", f"{total_time:.3f}s")
    
    def render_confidence_visualization(self, confidence_scores: list) -> None:
        """Render visualization of confidence scores across proof steps."""
        if not confidence_scores:
            return
        
        try:
            import plotly.graph_objects as go
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=confidence_scores,
                mode='lines+markers',
                name='Confidence',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                title="Proof Step Confidence",
                xaxis_title="Step Number",
                yaxis_title="Confidence",
                yaxis=dict(range=[0, 1]),
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except ImportError:
            # Fallback to simple text display
            st.markdown("**Confidence Scores by Step:**")
            for i, score in enumerate(confidence_scores):
                st.markdown(f"Step {i+1}: {score:.2%}")
    
    def render_step_complexity_analysis(self, steps) -> None:
        """Render analysis of step complexity across the proof."""
        if not steps:
            return
        
        st.markdown("### 🔍 Step Complexity Analysis")
        
        # Calculate complexity metrics
        avg_confidence = sum(step.confidence for step in steps) / len(steps)
        total_time = sum(step.execution_time for step in steps)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Confidence", f"{avg_confidence:.2%}")
        with col2:
            st.metric("Total Steps", len(steps))
        with col3:
            st.metric("Total Time", f"{total_time:.3f}s")
        
        # Rule usage analysis
        rules_used = [step.rule_applied for step in steps if step.rule_applied]
        if rules_used:
            st.markdown("**Rules Used:**")
            rule_counts = {}
            for rule in rules_used:
                rule_counts[rule] = rule_counts.get(rule, 0) + 1
            
            for rule, count in rule_counts.items():
                st.markdown(f"- {rule}: {count} times")
    
    def render_mathematical_notation_guide(self) -> None:
        """Render a guide for mathematical notation used in proofs."""
        with st.expander("📚 Mathematical Notation Guide"):
            st.markdown("""
            **Common Symbols:**
            - ∀ : For all (universal quantifier)
            - ∃ : There exists (existential quantifier)
            - ∈ : Element of
            - ⊆ : Subset of
            - ∪ : Union
            - ∩ : Intersection
            - → : Implies
            - ↔ : If and only if
            - ℝ : Real numbers
            - ℕ : Natural numbers
            - ℤ : Integers
            - ℚ : Rational numbers
            
            **Proof Steps:**
            - Each step shows a mathematical transformation
            - Confidence indicates the reliability of the step
            - Rules show the mathematical principle applied
            - Justification explains the reasoning
            """)
    
    def format_expression_for_display(self, expression: str, max_length: int = 100) -> str:
        """Format expression for compact display in UI elements."""
        if not expression:
            return "Empty"
        
        # Clean up common formatting issues
        cleaned = expression.strip()
        
        # Truncate if too long
        if len(cleaned) > max_length:
            cleaned = cleaned[:max_length-3] + "..."
        
        # Replace common symbols for better readability
        display_replacements = {
            '**': '^',
            'sqrt(': '√(',
            'pi': 'π',
            'theta': 'θ',
            'alpha': 'α',
            'beta': 'β',
            'gamma': 'γ',
            'lambda': 'λ',
        }
        
        for old, new in display_replacements.items():
            cleaned = cleaned.replace(old, new)
        
        return cleaned 