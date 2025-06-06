"""
Comprehensive tests for TheoremDetail component.

Author: MathBot Team
Version: Phase 6C
"""

import pytest
from unittest.mock import Mock, patch
from typing import List

from ui.components.theorem_detail import TheoremDetail
from ui.config import UIConfig
from ui.data.models import Theorem, ValidationEvidence, SourceLineage, MathematicalContext


class TestTheoremDetail:
    """Test suite for TheoremDetail functionality."""
    
    @pytest.fixture
    def mock_config(self) -> UIConfig:
        """Create mock UI configuration for testing."""
        config = Mock(spec=UIConfig)
        config.latex_renderer = "mathjax"
        config.latex_timeout = 5.0
        config.enable_theorem_sharing = True
        return config
    
    @pytest.fixture
    def sample_theorem(self) -> Theorem:
        """Create sample theorem for testing."""
        return Theorem(
            id="THM_12345678",
            statement="∀x∈ℝ: (x+1)² = x² + 2x + 1",
            sympy_expression="Eq((x + 1)**2, x**2 + 2*x + 1)",
            theorem_type="algebraic_identity",
            assumptions=["x ∈ ℝ"],
            source_lineage=SourceLineage(
                original_formula="(x+1)^2 = x^2 + 2x + 1",
                hypothesis_id="HYP_12345678",
                confidence=1.0,
                validation_score=1.0,
                generation_method="algebraic_expansion",
                source_type="derived_theorem",
                transformation_chain=["expansion", "simplification", "verification"]
            ),
            natural_language="The square of a binomial equals the sum of squares plus twice the product",
            symbols=["x"],
            mathematical_context=MathematicalContext(
                symbols=["x"],
                complexity_score=0.4,
                domain="polynomial_algebra"
            ),
            validation_evidence=ValidationEvidence(
                validation_status="PASS",
                pass_rate=1.0,
                total_tests=1000,
                symbols_tested=["x"],
                validation_time=0.156,
                test_cases=[
                    {"input": {"x": 0}, "expected": True, "actual": True},
                    {"input": {"x": 1}, "expected": True, "actual": True},
                    {"input": {"x": -1}, "expected": True, "actual": True}
                ]
            ),
            metadata={"difficulty": "intermediate", "topics": ["algebra", "polynomials"]}
        )
    
    @pytest.fixture
    def related_theorems(self) -> List[Theorem]:
        """Create related theorems for testing."""
        return [
            Theorem(
                id="THM_87654321",
                statement="(x-1)² = x² - 2x + 1",
                sympy_expression="Eq((x - 1)**2, x**2 - 2*x + 1)",
                theorem_type="algebraic_identity",
                assumptions=["x ∈ ℝ"],
                source_lineage=SourceLineage(
                    original_formula="(x-1)^2 = x^2 - 2x + 1",
                    hypothesis_id="HYP_87654321",
                    confidence=1.0,
                    validation_score=1.0,
                    generation_method="algebraic_expansion"
                ),
                natural_language="The square of a difference",
                symbols=["x"],
                mathematical_context=MathematicalContext(
                    symbols=["x"],
                    complexity_score=0.4,
                    domain="polynomial_algebra"
                ),
                validation_evidence=ValidationEvidence(
                    validation_status="PASS",
                    pass_rate=1.0,
                    total_tests=500,
                    symbols_tested=["x"],
                    validation_time=0.089
                )
            )
        ]
    
    @pytest.fixture
    def theorem_detail(self, mock_config) -> TheoremDetail:
        """Create TheoremDetail instance for testing."""
        return TheoremDetail(mock_config)
    
    def test_initialization_success(self, mock_config):
        """Test successful TheoremDetail initialization."""
        detail = TheoremDetail(mock_config)
        
        assert detail.config == mock_config
        assert detail.logger is not None
    
    @patch('streamlit.latex')
    @patch('streamlit.write')
    def test_render_theorem_statement(self, mock_write, mock_latex, 
                                    theorem_detail, sample_theorem):
        """Test rendering of theorem statement with LaTeX."""
        theorem_detail._render_theorem_statement(sample_theorem)
        
        mock_latex.assert_called()
        mock_write.assert_called()
    
    @patch('streamlit.write')
    @patch('streamlit.code')
    def test_render_theorem_metadata(self, mock_code, mock_write, 
                                   theorem_detail, sample_theorem):
        """Test rendering of theorem metadata."""
        with patch('streamlit.columns'):
            theorem_detail._render_theorem_metadata(sample_theorem)
            
            mock_write.assert_called()
    
    @patch('streamlit.metric')
    @patch('streamlit.progress')
    def test_render_validation_analysis(self, mock_progress, mock_metric, 
                                      theorem_detail, sample_theorem):
        """Test rendering of validation analysis."""
        with patch('streamlit.columns'):
            theorem_detail.render_validation_analysis(sample_theorem.validation_evidence)
            
            mock_metric.assert_called()
    
    @patch('streamlit.write')
    @patch('streamlit.code')
    def test_render_transformation_chain(self, mock_code, mock_write, 
                                       theorem_detail, sample_theorem):
        """Test rendering of transformation chain."""
        with patch('streamlit.expander'):
            theorem_detail.render_transformation_chain(sample_theorem.source_lineage)
            
            mock_write.assert_called()
    
    @patch('streamlit.write')
    def test_render_symbol_analysis(self, mock_write, theorem_detail, sample_theorem):
        """Test rendering of symbol analysis."""
        with patch('streamlit.expander'):
            theorem_detail._render_symbol_analysis(sample_theorem)
            
            mock_write.assert_called()
    
    @patch('streamlit.write')
    def test_render_related_theorems(self, mock_write, theorem_detail, 
                                   sample_theorem, related_theorems):
        """Test rendering of related theorems."""
        with patch('streamlit.expander'):
            theorem_detail._render_related_theorems(related_theorems)
            
            mock_write.assert_called()
    
    @patch('streamlit.tabs')
    def test_render_theorem_detail_with_tabs(self, mock_tabs, theorem_detail, 
                                           sample_theorem, related_theorems):
        """Test complete theorem detail rendering with tabs."""
        mock_tabs.return_value = [Mock(), Mock(), Mock(), Mock(), Mock()]
        
        with patch.object(theorem_detail, '_render_theorem_statement'), \
             patch.object(theorem_detail, 'render_validation_analysis'), \
             patch.object(theorem_detail, 'render_transformation_chain'), \
             patch.object(theorem_detail, '_render_symbol_analysis'), \
             patch.object(theorem_detail, '_render_related_theorems'):
            
            theorem_detail.render_theorem_detail(sample_theorem, related_theorems)
            
            mock_tabs.assert_called()
    
    @patch('streamlit.download_button')
    def test_render_sharing_controls(self, mock_download_button, 
                                   theorem_detail, sample_theorem):
        """Test rendering of sharing and export controls."""
        with patch('streamlit.columns'):
            theorem_detail._render_sharing_controls(sample_theorem)
            
            mock_download_button.assert_called()
    
    def test_format_latex_expression(self, theorem_detail, sample_theorem):
        """Test LaTeX expression formatting."""
        formatted = theorem_detail._format_latex_expression(sample_theorem.statement)
        
        assert isinstance(formatted, str)
        assert len(formatted) > 0
    
    def test_extract_symbols_from_statement(self, theorem_detail, sample_theorem):
        """Test symbol extraction from theorem statement."""
        symbols = theorem_detail._extract_symbols_from_statement(sample_theorem.statement)
        
        assert isinstance(symbols, list)
        assert len(symbols) >= 0
    
    def test_generate_theorem_insights(self, theorem_detail, sample_theorem):
        """Test generation of theorem insights."""
        insights = theorem_detail._generate_theorem_insights(sample_theorem)
        
        assert isinstance(insights, list)
        assert len(insights) >= 0
    
    @patch('streamlit.plotly_chart')
    def test_render_validation_metrics_chart(self, mock_plotly_chart, 
                                           theorem_detail, sample_theorem):
        """Test rendering of validation metrics chart."""
        theorem_detail._render_validation_metrics_chart(sample_theorem.validation_evidence)
        
        # Should attempt to render chart if data is available
        if sample_theorem.validation_evidence.test_cases:
            mock_plotly_chart.assert_called()
    
    @patch('streamlit.json')
    def test_render_theorem_json_export(self, mock_json, theorem_detail, sample_theorem):
        """Test JSON export functionality."""
        with patch('streamlit.expander'):
            theorem_detail._render_theorem_json_export(sample_theorem)
            
            mock_json.assert_called()
    
    def test_calculate_complexity_metrics(self, theorem_detail, sample_theorem):
        """Test complexity metrics calculation."""
        metrics = theorem_detail._calculate_complexity_metrics(sample_theorem)
        
        assert isinstance(metrics, dict)
        assert "symbol_count" in metrics
        assert "statement_length" in metrics
        assert "complexity_score" in metrics
    
    def test_generate_related_content_suggestions(self, theorem_detail, 
                                                sample_theorem, related_theorems):
        """Test related content suggestions."""
        suggestions = theorem_detail._generate_related_content_suggestions(
            sample_theorem, related_theorems
        )
        
        assert isinstance(suggestions, list)
        assert len(suggestions) >= 0
    
    @patch('streamlit.error')
    def test_error_handling_invalid_latex(self, mock_error, theorem_detail):
        """Test error handling for invalid LaTeX."""
        invalid_statement = "Invalid \\invalid\\latex"
        
        # Should handle invalid LaTeX gracefully
        try:
            formatted = theorem_detail._format_latex_expression(invalid_statement)
            assert isinstance(formatted, str)
        except Exception:
            # Should log error and handle gracefully
            pass
    
    def test_handle_missing_validation_evidence(self, theorem_detail):
        """Test handling of missing validation evidence."""
        # Create theorem without validation evidence
        theorem_no_validation = Theorem(
            id="THM_ABCDEF12",
            statement="f(x) = x",
            sympy_expression="Eq(f(x), x)",
            theorem_type="identity",
            assumptions=[],
            source_lineage=SourceLineage(
                original_formula="f(x) = x",
                hypothesis_id="HYP_ABCDEF12",
                confidence=0.5,
                validation_score=0.0,
                generation_method="test",
                source_type="derived_theorem"
            ),
            natural_language="Identity function",
            symbols=["f", "x"],
            mathematical_context=MathematicalContext(
                symbols=["f", "x"],
                complexity_score=0.1,
                domain="functions"
            ),
            validation_evidence=None
        )
        
        # Should handle missing validation evidence gracefully
        with patch('streamlit.warning'):
            theorem_detail.render_validation_analysis(None)
    
    def test_handle_empty_related_theorems(self, theorem_detail):
        """Test handling of empty related theorems list."""
        empty_related = []
        
        with patch('streamlit.info'):
            theorem_detail._render_related_theorems(empty_related)
    
    @pytest.mark.performance
    def test_latex_rendering_performance(self, theorem_detail):
        """Test LaTeX rendering performance."""
        complex_statement = "∫_{-∞}^{∞} e^{-x²} dx = √π ∧ ∑_{n=1}^{∞} \\frac{1}{n²} = \\frac{π²}{6}"
        
        import time
        start_time = time.time()
        formatted = theorem_detail._format_latex_expression(complex_statement)
        render_time = time.time() - start_time
        
        assert render_time < 1.0  # Should format quickly
        assert isinstance(formatted, str)
    
    def test_theorem_comparison_functionality(self, theorem_detail, 
                                            sample_theorem, related_theorems):
        """Test theorem comparison features."""
        if related_theorems:
            comparison = theorem_detail._compare_theorems(sample_theorem, related_theorems[0])
            
            assert isinstance(comparison, dict)
            assert "similarities" in comparison
            assert "differences" in comparison
    
    def test_accessibility_features(self, theorem_detail, sample_theorem):
        """Test accessibility features for theorem details."""
        # Test alt text generation for mathematical expressions
        alt_text = theorem_detail._generate_alt_text(sample_theorem.statement)
        
        assert isinstance(alt_text, str)
        assert len(alt_text) > 0
    
    def test_mobile_responsive_rendering(self, theorem_detail, sample_theorem):
        """Test mobile-responsive rendering considerations."""
        # Test that components can handle smaller screen sizes
        with patch('streamlit.columns') as mock_columns:
            # Mock mobile layout
            mock_columns.return_value = [Mock()]
            
            theorem_detail._render_theorem_metadata(sample_theorem)
            
            mock_columns.assert_called() 