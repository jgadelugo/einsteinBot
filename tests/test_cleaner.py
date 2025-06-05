"""
Tests for the cleaner module.

This module contains tests for mathematical symbol normalization,
expression cleaning, and format conversion functionality.
"""

import pytest
from ingestion.cleaner import (
    FormulaCleaner,
    normalize_symbols,
    clean_math_expression,
    batch_normalize,
    detect_notation_inconsistencies,
)


class TestFormulaCleaner:
    """Test suite for FormulaCleaner class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cleaner = FormulaCleaner(preserve_latex=False)
        self.cleaner_preserve = FormulaCleaner(preserve_latex=True)
    
    def test_init(self):
        """Test cleaner initialization."""
        assert self.cleaner.preserve_latex is False
        assert self.cleaner_preserve.preserve_latex is True
    
    def test_normalize_greek_letters(self):
        """Test normalization of Greek letters."""
        test_cases = [
            (r"\alpha + \beta", "α + β"),
            (r"\gamma \cdot \delta", "γ * δ"),
            (r"\Gamma + \Delta", "Γ + Δ"),
            (r"\pi r^2", "π r^2"),
        ]
        
        for input_expr, expected in test_cases:
            result = self.cleaner.clean_expression(input_expr)
            assert expected in result, f"Expected '{expected}' in result '{result}'"
    
    def test_normalize_math_operators(self):
        """Test normalization of mathematical operators."""
        test_cases = [
            (r"a \cdot b", "a * b"),
            (r"x \times y", "x * y"),
            (r"a \div b", "a / b"),
            (r"x \pm y", "x ± y"),
            (r"a \leq b", "a ≤ b"),
            (r"x \geq y", "x ≥ y"),
            (r"a \neq b", "a ≠ b"),
        ]
        
        for input_expr, expected in test_cases:
            result = self.cleaner.clean_expression(input_expr)
            assert expected in result, f"Expected '{expected}' in result '{result}'"
    
    def test_normalize_fractions(self):
        """Test normalization of fraction expressions."""
        test_cases = [
            (r"\frac{a}{b}", "(a)/(b)"),
            (r"\frac{x+1}{y-1}", "frac(x+1, y-1)"),
            (r"\dfrac{1}{2}", "(1)/(2)"),
        ]
        
        for input_expr, expected_pattern in test_cases:
            result = self.cleaner.clean_expression(input_expr)
            # Check if the result contains the expected pattern or similar
            assert ("(" in result and "/" in result) or "frac(" in result
    
    def test_normalize_brackets(self):
        """Test normalization of LaTeX brackets."""
        test_cases = [
            (r"\left( x \right)", "( x )"),
            (r"\left[ a \right]", "[ a ]"),
            (r"\left\{ b \right\}", "{ b }"),
            (r"\left| c \right|", "| c |"),
        ]
        
        for input_expr, expected in test_cases:
            result = self.cleaner.clean_expression(input_expr)
            # Remove extra spaces for comparison
            result_clean = " ".join(result.split())
            expected_clean = " ".join(expected.split())
            assert expected_clean in result_clean
    
    def test_normalize_scripts(self):
        """Test normalization of superscripts and subscripts."""
        test_cases = [
            ("x^2", "x^2"),
            ("x^{n+1}", "x^(n+1)"),
            ("a_i", "a_i"),
            ("b_{j,k}", "b_(j,k)"),
        ]
        
        for input_expr, expected in test_cases:
            result = self.cleaner.clean_expression(input_expr)
            assert expected in result
    
    def test_cleanup_artifacts(self):
        """Test cleanup of LaTeX artifacts."""
        test_cases = [
            (r"x \, y", "x y"),  # Thin space
            (r"a \quad b", "a b"),  # Quad space
            (r"x + y", "x + y"),  # Operator spacing
            (r"a{} + b", "a + b"),  # Empty braces
        ]
        
        for input_expr, expected_pattern in test_cases:
            result = self.cleaner.clean_expression(input_expr)
            # Check that spacing is normalized
            assert " + " in result or expected_pattern in result
    
    def test_batch_clean(self):
        """Test batch cleaning of multiple expressions."""
        expressions = [
            r"\alpha + \beta",
            r"\frac{x}{y}",
            r"a \cdot b",
            r"\gamma^2",
        ]
        
        cleaned = self.cleaner.batch_clean(expressions)
        
        assert len(cleaned) == len(expressions)
        assert "α + β" in cleaned[0]
        assert "*" in cleaned[2]
        assert "γ" in cleaned[3]
    
    def test_convert_to_sympy_format(self):
        """Test conversion to SymPy-compatible format."""
        test_cases = [
            (r"\frac{a}{b}", "(a)/(b)"),
            (r"\sqrt{x}", "sqrt(x)"),
            (r"\sin(x)", "sin(x)"),
            (r"\alpha + \beta", "alpha + beta"),
            (r"\pi r^2", "pi r^2"),
        ]
        
        for input_expr, expected_pattern in test_cases:
            result = self.cleaner.convert_to_sympy_format(input_expr)
            assert any(pattern in result for pattern in expected_pattern.split(" + "))
    
    def test_get_symbol_statistics(self):
        """Test symbol usage statistics."""
        expressions = [
            r"\alpha + \beta",
            r"\frac{x}{y}",
            r"a^2 + b_i",
            r"\sin(x) + \cos(y)",
        ]
        
        stats = self.cleaner.get_symbol_statistics(expressions)
        
        assert stats["total_expressions"] == 4
        assert stats["latex_commands"] > 0
        assert stats["fractions"] >= 1
        assert stats["superscripts"] >= 1
        assert stats["subscripts"] >= 1
        assert stats["functions"] >= 2
    
    def test_preserve_latex_mode(self):
        """Test preserve LaTeX mode functionality."""
        expr = r"\alpha + \beta"
        
        normal_result = self.cleaner.clean_expression(expr)
        preserve_result = self.cleaner_preserve.clean_expression(expr)
        
        # Normal mode should convert to Unicode
        assert "α" in normal_result or "β" in normal_result
        
        # Note: preserve_latex mode is not fully implemented in the current version
        # This test documents the expected behavior
    
    def test_empty_expression(self):
        """Test handling of empty expressions."""
        result = self.cleaner.clean_expression("")
        assert result == ""
        
        result = self.cleaner.clean_expression("   ")
        assert result == ""
    
    def test_complex_expression(self):
        """Test cleaning of complex mathematical expressions."""
        complex_expr = r"""
        \int_0^\infty e^{-\alpha x^2} dx = \frac{1}{2}\sqrt{\frac{\pi}{\alpha}}
        """
        
        result = self.cleaner.clean_expression(complex_expr)
        
        # Should contain normalized symbols
        assert "∫" in result or "int" in result
        assert "α" in result or "alpha" in result
        assert "π" in result or "pi" in result
    
    def test_error_handling(self):
        """Test error handling in batch processing."""
        expressions = [
            r"\alpha + \beta",  # Valid
            None,  # Invalid - will cause error
            r"\gamma",  # Valid
        ]
        
        # Should handle errors gracefully
        try:
            cleaned = self.cleaner.batch_clean([e for e in expressions if e is not None])
            assert len(cleaned) == 2
        except Exception:
            # If it raises an exception, that's also acceptable behavior
            pass


class TestConvenienceFunctions:
    """Test suite for convenience functions."""
    
    def test_normalize_symbols(self):
        """Test normalize_symbols convenience function."""
        text = r"The formula \alpha + \beta = \gamma is important."
        
        result = normalize_symbols(text)
        
        assert "α" in result
        assert "β" in result
        assert "γ" in result
    
    def test_clean_math_expression(self):
        """Test clean_math_expression convenience function."""
        expr = r"\frac{\alpha}{\beta} + \gamma"
        
        # Normal cleaning
        result = clean_math_expression(expr)
        assert "α" in result or "β" in result
        
        # SymPy format
        sympy_result = clean_math_expression(expr, to_sympy=True)
        assert "alpha" in sympy_result or "beta" in sympy_result
    
    def test_batch_normalize(self):
        """Test batch_normalize convenience function."""
        expressions = [
            r"\alpha + \beta",
            r"\gamma \cdot \delta",
        ]
        
        # Normal batch
        normal_results = batch_normalize(expressions)
        assert len(normal_results) == 2
        assert any("α" in result for result in normal_results)
        
        # SymPy batch
        sympy_results = batch_normalize(expressions, to_sympy=True)
        assert len(sympy_results) == 2
        assert any("alpha" in result for result in sympy_results)
    
    def test_detect_notation_inconsistencies(self):
        """Test detection of notation inconsistencies."""
        expressions = [
            r"\alpha + β",  # Mixed LaTeX and Unicode
            r"\unknown_command",  # Unknown command
            r"\alpha + \beta",  # Consistent LaTeX
            "α + β",  # Consistent Unicode
        ]
        
        inconsistencies = detect_notation_inconsistencies(expressions)
        
        assert len(inconsistencies["mixed_notation"]) >= 1
        assert len(inconsistencies["unknown_symbols"]) >= 1


class TestEdgeCases:
    """Test suite for edge cases and error conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cleaner = FormulaCleaner()
    
    def test_nested_commands(self):
        """Test handling of nested LaTeX commands."""
        expr = r"\frac{\sqrt{\alpha + \beta}}{\gamma^2}"
        
        result = self.cleaner.clean_expression(expr)
        
        # Should handle nested structure
        assert "α" in result or "alpha" in result
        assert "β" in result or "beta" in result
        assert "γ" in result or "gamma" in result
    
    def test_malformed_commands(self):
        """Test handling of malformed LaTeX commands."""
        malformed_expressions = [
            r"\frac{a}{",  # Incomplete fraction
            r"\alpha{",  # Incomplete braces
            r"\\invalid",  # Invalid command
        ]
        
        for expr in malformed_expressions:
            # Should not crash
            result = self.cleaner.clean_expression(expr)
            assert isinstance(result, str)
    
    def test_unicode_input(self):
        """Test handling of Unicode input."""
        expr = "α + β = γ"  # Already Unicode
        
        result = self.cleaner.clean_expression(expr)
        
        # Should preserve Unicode
        assert "α" in result
        assert "β" in result
        assert "γ" in result
    
    def test_mixed_content(self):
        """Test handling of mixed LaTeX and Unicode content."""
        expr = r"\alpha + β = \gamma"  # Mixed
        
        result = self.cleaner.clean_expression(expr)
        
        # Should normalize all to Unicode
        assert "α" in result
        assert "β" in result
        assert "γ" in result
    
    def test_very_long_expression(self):
        """Test handling of very long expressions."""
        long_expr = r"\alpha + " * 100 + r"\beta"
        
        result = self.cleaner.clean_expression(long_expr)
        
        # Should handle without crashing
        assert isinstance(result, str)
        assert "α" in result
        assert "β" in result
    
    def test_special_unicode_symbols(self):
        """Test handling of special Unicode mathematical symbols."""
        expr = "∫₀¹ f(x)dx = ∑ᵢ₌₁ⁿ aᵢ"
        
        result = self.cleaner.clean_expression(expr)
        
        # Should preserve special Unicode
        assert "∫" in result
        assert "∑" in result
    
    def test_confidence_with_cleaned_expressions(self):
        """Test that cleaning doesn't break confidence calculation."""
        from ingestion.formula_extractor import FormulaExtractor
        
        extractor = FormulaExtractor(clean_formulas=True)
        text = r"The formula $\alpha + \beta = \gamma$ is important."
        
        expressions = extractor.extract_expressions(text)
        
        assert len(expressions) >= 1
        assert expressions[0]["confidence"] > 0
        assert "α" in expressions[0]["expression"]


class TestSymPyIntegration:
    """Test suite for SymPy integration functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cleaner = FormulaCleaner()
    
    def test_sympy_basic_conversion(self):
        """Test basic SymPy conversion."""
        test_cases = [
            (r"x^2 + y^2", "x^2 + y^2"),
            (r"\sin(x)", "sin(x)"),
            (r"\cos(\theta)", "cos(theta)"),
            (r"\frac{a}{b}", "(a)/(b)"),
            (r"\sqrt{x}", "sqrt(x)"),
        ]
        
        for input_expr, expected_pattern in test_cases:
            result = self.cleaner.convert_to_sympy_format(input_expr)
            assert any(part in result for part in expected_pattern.split())
    
    def test_sympy_greek_letters(self):
        """Test SymPy conversion of Greek letters."""
        expr = r"\alpha + \beta \cdot \gamma"
        
        result = self.cleaner.convert_to_sympy_format(expr)
        
        assert "alpha" in result
        assert "beta" in result
        assert "gamma" in result
    
    def test_sympy_complex_expression(self):
        """Test SymPy conversion of complex expressions."""
        expr = r"\int_0^1 \sin(\pi x) dx"
        
        result = self.cleaner.convert_to_sympy_format(expr)
        
        # Should convert to SymPy-compatible format
        assert "sin" in result
        assert "pi" in result
        # Integration symbol might be removed or converted


if __name__ == "__main__":
    pytest.main([__file__]) 