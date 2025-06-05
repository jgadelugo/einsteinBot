"""
Tests for the formula extractor module.

This module contains tests for mathematical expression detection,
extraction, confidence scoring, and various edge cases.
"""

import pytest
from ingestion.formula_extractor import (
    FormulaExtractor,
    extract_math_expressions,
    find_math_patterns,
)


class TestFormulaExtractor:
    """Test suite for FormulaExtractor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = FormulaExtractor(clean_formulas=True)
        self.extractor_no_clean = FormulaExtractor(clean_formulas=False)
    
    def test_init(self):
        """Test extractor initialization."""
        assert self.extractor.clean_formulas is True
        assert self.extractor_no_clean.clean_formulas is False
    
    def test_extract_inline_math(self):
        """Test extraction of inline mathematical expressions."""
        text = "The formula $x = y + z$ is simple, and $E = mc^2$ is famous."
        
        expressions = self.extractor.extract_expressions(text, include_block=False)
        
        assert len(expressions) == 2
        assert any("x = y + z" in expr["expression"] for expr in expressions)
        assert any("E = mc^2" in expr["expression"] for expr in expressions)
        
        for expr in expressions:
            assert expr["type"] == "inline"
            assert expr["confidence"] > 0
    
    def test_extract_block_math(self):
        """Test extraction of block/display mathematical expressions."""
        text = r"""
        Here is a display equation:
        $$\int_0^\infty e^{-x^2} dx = \frac{\sqrt{\pi}}{2}$$
        
        And another one:
        \begin{equation}
        \sum_{n=1}^{\infty} \frac{1}{n^2} = \frac{\pi^2}{6}
        \end{equation}
        """
        
        expressions = self.extractor.extract_expressions(text, include_inline=False)
        
        assert len(expressions) >= 2
        
        for expr in expressions:
            assert expr["type"] == "block"
            assert expr["confidence"] > 0
    
    def test_extract_both_types(self):
        """Test extraction of both inline and block expressions."""
        text = r"""
        The quadratic formula is $x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$.
        
        For the integral:
        $$\int_0^{2\pi} \sin(x) dx = 0$$
        """
        
        expressions = self.extractor.extract_expressions(text)
        
        inline_count = sum(1 for expr in expressions if expr["type"] == "inline")
        block_count = sum(1 for expr in expressions if expr["type"] == "block")
        
        assert inline_count >= 1
        assert block_count >= 1
    
    def test_confidence_scoring(self):
        """Test confidence scoring for mathematical expressions."""
        test_cases = [
            ("x + y", 0.3),  # Simple expression, low confidence
            (r"\alpha + \beta", 0.5),  # Greek letters, higher confidence
            (r"\int_0^1 f(x) dx", 0.7),  # Complex math, high confidence
            ("123", 0.1),  # Just numbers, very low confidence
            ("abc", 0.1),  # Just letters, very low confidence
        ]
        
        for expr, expected_min_confidence in test_cases:
            confidence = self.extractor._calculate_confidence(expr)
            assert confidence >= expected_min_confidence, f"Expression '{expr}' had confidence {confidence}, expected >= {expected_min_confidence}"
    
    def test_deduplication(self):
        """Test deduplication of extracted expressions."""
        text = r"""
        The formula $x = y + z$ appears here.
        Later, the same formula $x = y + z$ appears again.
        """
        
        expressions_with_dup = self.extractor.extract_expressions(text, deduplicate=False)
        expressions_no_dup = self.extractor.extract_expressions(text, deduplicate=True)
        
        assert len(expressions_with_dup) >= 2
        assert len(expressions_no_dup) == 1
    
    def test_expression_filtering(self):
        """Test filtering of low-quality expressions."""
        text = r"""
        Valid math: $\alpha + \beta = \gamma$
        Too short: $x$
        Just number: $123$
        Good complex: $\int_0^1 f(x) dx$
        """
        
        expressions = self.extractor.extract_expressions(text)
        
        # Should filter out very short and number-only expressions
        expression_texts = [expr["expression"] for expr in expressions]
        
        assert any("α + β = γ" in expr for expr in expression_texts)
        assert any("∫" in expr for expr in expression_texts)
        assert not any(expr.strip() == "x" for expr in expression_texts)
        assert not any(expr.strip() == "123" for expr in expression_texts)
    
    def test_extract_by_environment(self):
        """Test extraction from specific LaTeX environments."""
        text = r"""
        \begin{equation}
        E = mc^2
        \end{equation}
        
        Some text here.
        
        \begin{equation*}
        F = ma
        \end{equation*}
        
        \begin{align}
        x &= y + z \\
        a &= b + c
        \end{align}
        """
        
        equation_exprs = self.extractor.extract_by_environment(text, "equation")
        align_exprs = self.extractor.extract_by_environment(text, "align")
        
        assert len(equation_exprs) == 2  # equation and equation*
        assert len(align_exprs) == 1
        
        assert "E = mc^2" in equation_exprs[0]
        assert "F = ma" in equation_exprs[1]
        assert "x &= y + z" in align_exprs[0]
    
    def test_get_math_statistics(self):
        """Test mathematical content statistics."""
        text = r"""
        Inline math: $x = y$ and $\alpha + \beta$.
        
        Display math:
        $$\int_0^1 f(x) dx$$
        
        \begin{equation}
        \sum_{n=1}^{\infty} \frac{1}{n^2}
        \end{equation}
        """
        
        stats = self.extractor.get_math_statistics(text)
        
        assert stats["total_expressions"] >= 3
        assert stats["inline_expressions"] >= 2
        assert stats["block_expressions"] >= 2
        assert stats["equation_environments"] >= 1
    
    def test_clean_vs_no_clean(self):
        """Test difference between cleaning and not cleaning expressions."""
        text = r"The formula is $\alpha + \beta = \gamma$."
        
        clean_expressions = self.extractor.extract_expressions(text)
        no_clean_expressions = self.extractor_no_clean.extract_expressions(text)
        
        clean_expr = clean_expressions[0]["expression"]
        no_clean_expr = no_clean_expressions[0]["expression"]
        
        # Clean version should have Unicode symbols
        assert "α" in clean_expr or "α + β = γ" in clean_expr
        # No-clean version should have LaTeX commands
        assert "\\alpha" in no_clean_expr or "\\beta" in no_clean_expr
    
    def test_empty_text(self):
        """Test extraction from empty text."""
        expressions = self.extractor.extract_expressions("")
        assert len(expressions) == 0
    
    def test_text_without_math(self):
        """Test extraction from text without mathematical expressions."""
        text = "This is just regular text without any mathematical content."
        expressions = self.extractor.extract_expressions(text)
        assert len(expressions) == 0
    
    def test_malformed_math(self):
        """Test handling of malformed mathematical expressions."""
        text = r"""
        Unclosed dollar: $x = y + z
        Mismatched braces: \frac{x}{y
        Empty math: $$$$
        """
        
        # Should not crash and may or may not extract anything
        expressions = self.extractor.extract_expressions(text)
        assert isinstance(expressions, list)
    
    def test_nested_math_delimiters(self):
        """Test handling of nested or complex delimiters."""
        text = r"""
        Nested: $\text{if } x > 0 \text{ then } y = \sqrt{x}$
        Complex: $$\left(\frac{a}{b}\right)^{-1} = \frac{b}{a}$$
        """
        
        expressions = self.extractor.extract_expressions(text)
        assert len(expressions) >= 2
        
        # Should handle complex expressions
        expr_texts = [expr["expression"] for expr in expressions]
        assert any("sqrt" in expr or "√" in expr for expr in expr_texts)
    
    def test_position_tracking(self):
        """Test that expression positions are tracked correctly."""
        text = "Start $x = y$ middle $a = b$ end"
        
        expressions = self.extractor.extract_expressions(text)
        
        assert len(expressions) == 2
        
        # Positions should be in order
        pos1 = expressions[0]["position"]
        pos2 = expressions[1]["position"]
        assert pos1 < pos2
        
        # Positions should be reasonable
        assert 0 <= pos1 < len(text)
        assert 0 <= pos2 < len(text)


class TestConvenienceFunctions:
    """Test suite for convenience functions."""
    
    def test_extract_math_expressions(self):
        """Test extract_math_expressions convenience function."""
        text = r"Simple math: $x = y + z$ and $\alpha = \beta$."
        
        expressions = extract_math_expressions(text)
        
        assert isinstance(expressions, list)
        assert len(expressions) == 2
        assert all(isinstance(expr, str) for expr in expressions)
    
    def test_extract_math_expressions_options(self):
        """Test extract_math_expressions with different options."""
        text = r"Math: $x = y$ and $x = y$."  # Duplicate
        
        # With deduplication (default)
        expr_dedup = extract_math_expressions(text, deduplicate=True)
        
        # Without deduplication
        expr_no_dedup = extract_math_expressions(text, deduplicate=False)
        
        assert len(expr_dedup) == 1
        assert len(expr_no_dedup) == 2
    
    def test_find_math_patterns(self):
        """Test find_math_patterns function."""
        text = r"""
        Greek letters: \alpha, \beta, γ
        Functions: \sin(x), \cos(y), log(z)
        Operators: x + y - z * w / v
        Superscripts: x^2, y^{n+1}
        Subscripts: a_i, b_{j,k}
        Fractions: \frac{a}{b}, \dfrac{x}{y}
        Integrals: \int_0^1 f(x) dx
        Summations: \sum_{i=1}^n x_i
        """
        
        patterns = find_math_patterns(text)
        
        assert len(patterns["greek_letters"]) >= 3
        assert len(patterns["functions"]) >= 3
        assert len(patterns["operators"]) >= 4
        assert len(patterns["superscripts"]) >= 2
        assert len(patterns["subscripts"]) >= 2
        assert len(patterns["fractions"]) >= 2
        assert len(patterns["integrals"]) >= 1
        assert len(patterns["summations"]) >= 1
        
        # Check that positions are returned
        for pattern_matches in patterns.values():
            for match, position in pattern_matches:
                assert isinstance(match, str)
                assert isinstance(position, int)
                assert 0 <= position < len(text)


class TestEdgeCases:
    """Test suite for edge cases and error conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = FormulaExtractor()
    
    def test_very_long_expression(self):
        """Test handling of very long mathematical expressions."""
        # Create a very long expression
        long_expr = "$" + "x + " * 1000 + "y$"
        text = f"Here is a long expression: {long_expr}"
        
        expressions = self.extractor.extract_expressions(text)
        
        # Should be filtered out due to length
        assert len(expressions) == 0
    
    def test_unicode_in_math(self):
        """Test handling of Unicode characters in math."""
        text = "Math with Unicode: $α + β = γ$ and $∫₀¹ f(x)dx$"
        
        expressions = self.extractor.extract_expressions(text)
        
        assert len(expressions) >= 2
        expr_texts = [expr["expression"] for expr in expressions]
        assert any("α" in expr for expr in expr_texts)
    
    def test_multiple_dollar_signs(self):
        """Test handling of multiple consecutive dollar signs."""
        text = r"""
        Regular: $x = y$
        Display: $$z = w$$
        Multiple: $$$$a = b$$$$
        """
        
        expressions = self.extractor.extract_expressions(text)
        
        # Should correctly distinguish between $...$ and $$...$$
        inline_count = sum(1 for expr in expressions if expr["type"] == "inline")
        block_count = sum(1 for expr in expressions if expr["type"] == "block")
        
        assert inline_count >= 1
        assert block_count >= 1
    
    def test_math_in_latex_commands(self):
        """Test math expressions within LaTeX commands."""
        text = r"""
        In text: \textbf{The formula $x = y$ is important}
        In section: \section{Math: $E = mc^2$}
        """
        
        expressions = self.extractor.extract_expressions(text)
        
        # Should extract math even when nested in commands
        assert len(expressions) >= 2
    
    def test_special_characters(self):
        """Test handling of special characters in math."""
        text = r"""
        Special: $x \in \mathbb{R}$
        More: $\forall x \exists y : x < y$
        Sets: $A \cup B \cap C$
        """
        
        expressions = self.extractor.extract_expressions(text)
        
        assert len(expressions) >= 3
        # Should not crash on special LaTeX commands
    
    def test_confidence_edge_cases(self):
        """Test confidence calculation edge cases."""
        edge_cases = [
            "",  # Empty string
            " ",  # Just whitespace
            "x",  # Very short
            "1234567890",  # Only numbers
            "abcdefghij",  # Only letters
        ]
        
        for case in edge_cases:
            confidence = self.extractor._calculate_confidence(case)
            assert 0.0 <= confidence <= 1.0
    
    def test_extraction_modes(self):
        """Test different extraction mode combinations."""
        text = r"Inline: $x = y$ and display: $$z = w$$"
        
        # Inline only
        inline_only = self.extractor.extract_expressions(
            text, include_inline=True, include_block=False
        )
        
        # Block only
        block_only = self.extractor.extract_expressions(
            text, include_inline=False, include_block=True
        )
        
        # Both
        both = self.extractor.extract_expressions(
            text, include_inline=True, include_block=True
        )
        
        assert len(inline_only) >= 1
        assert len(block_only) >= 1
        assert len(both) >= len(inline_only) + len(block_only)
        
        assert all(expr["type"] == "inline" for expr in inline_only)
        assert all(expr["type"] == "block" for expr in block_only)


if __name__ == "__main__":
    pytest.main([__file__]) 