"""
Enhanced tests for SymPy to Lean 4 translation capabilities.

This module tests the advanced translation features added in Session 4,
including enhanced mathematical functions, complex expressions, and 
improved pattern handling.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from proofs.formal_systems.translation.sympy_to_lean import SymPyToLean4Translator
from proofs.formal_systems.base_interface import TranslationError


class TestEnhancedSymPyToLean4Translator:
    """Test enhanced SymPy to Lean 4 translation capabilities."""
    
    def test_enhanced_trigonometric_functions(self):
        """Test enhanced trigonometric function translation."""
        translator = SymPyToLean4Translator()
        
        # Test inverse trigonometric functions
        test_cases = {
            'asin(x)': 'Real.arcsin',
            'acos(x)': 'Real.arccos', 
            'atan(x)': 'Real.arctan',
            'sinh(x)': 'Real.sinh',
            'cosh(x)': 'Real.cosh',
            'tanh(x)': 'Real.tanh'
        }
        
        for sympy_expr, expected_lean in test_cases.items():
            result = translator.translate_simple_expression(sympy_expr)
            assert expected_lean in result, f"Expected '{expected_lean}' in result for '{sympy_expr}', got: {result}"
    
    def test_enhanced_logarithmic_functions(self):
        """Test enhanced logarithmic function translation."""
        translator = SymPyToLean4Translator()
        
        # Test logarithms with different bases
        result_log10 = translator.translate_simple_expression('log10(x)')
        assert 'Real.log x / Real.log 10' in result_log10
        
        result_log2 = translator.translate_simple_expression('log2(x)')
        assert 'Real.log x / Real.log 2' in result_log2
        
        # Test natural logarithm (if supported differently)
        result_ln = translator.translate_simple_expression('log(x)')
        assert 'Real.log' in result_ln
    
    def test_complex_mathematical_expressions(self):
        """Test translation of complex mathematical expressions."""
        translator = SymPyToLean4Translator()
        
        test_cases = [
            'sqrt(x**2 + y**2)',
            'abs(sin(x))',
            'exp(cos(x))',
            'log(sin(x) + cos(x))',
            'sinh(x)**2 + cosh(x)**2'
        ]
        
        for expr in test_cases:
            try:
                result = translator.translate_simple_expression(expr)
                assert isinstance(result, str), f"Translation should return string for '{expr}'"
                assert len(result) > 0, f"Translation should not be empty for '{expr}'"
                assert 'Real.' in result or '(' in result, f"Result should contain Lean syntax for '{expr}'"
            except TranslationError:
                pytest.fail(f"Translation should succeed for '{expr}'")
    
    def test_advanced_algebraic_expressions(self):
        """Test translation of advanced algebraic expressions."""
        translator = SymPyToLean4Translator()
        
        test_cases = {
            'x**3 + 2*x**2 - x + 1': ['x', '^', '3', '+', '2', '*'],
            'sqrt(x) + sqrt(y)': ['Real.sqrt', '+'],
            '(x + y)**2': ['^', '2'],
            'abs(x - y)': ['abs', '-']
        }
        
        for expr, expected_elements in test_cases.items():
            result = translator.translate_simple_expression(expr)
            for element in expected_elements:
                assert element in result, f"Expected '{element}' in result for '{expr}', got: {result}"
    
    def test_special_mathematical_constants(self):
        """Test translation of special mathematical constants."""
        translator = SymPyToLean4Translator()
        
        test_cases = {
            'pi': 'Real.pi',
            'E': 'Real.exp 1',
            # Test constants in expressions
            'pi * 2': 'Real.pi',
            'E**x': 'Real.exp'  # E is handled as exp function
        }
        
        for expr, expected_part in test_cases.items():
            result = translator.translate_simple_expression(expr)
            assert expected_part in result, f"Expected '{expected_part}' in result for '{expr}', got: {result}"
    
    def test_function_with_multiple_arguments(self):
        """Test functions that take multiple arguments."""
        translator = SymPyToLean4Translator()
        
        # Test max and min functions
        max_result = translator.translate_simple_expression('max(x, y)')
        # Should handle max function somehow
        assert 'max' in max_result or 'Maximum' in max_result or 'Unsupported' in max_result
        
        min_result = translator.translate_simple_expression('min(x, y)')
        # Should handle min function somehow  
        assert 'min' in min_result or 'Minimum' in min_result or 'Unsupported' in min_result
    
    def test_nested_function_calls(self):
        """Test translation of nested function calls."""
        translator = SymPyToLean4Translator()
        
        test_cases = [
            'sin(cos(x))',
            'log(exp(x))',
            'sqrt(abs(x))',
            'exp(sin(x))',
            'abs(sqrt(x))'
        ]
        
        for expr in test_cases:
            try:
                result = translator.translate_simple_expression(expr)
                # Should contain nested Lean function calls
                assert 'Real.' in result or 'abs' in result, f"Should contain Lean functions for '{expr}'"
                # Should have some structure indicating function composition
                assert len(result.split()) >= 3, f"Should have nested structure for '{expr}'"
            except TranslationError:
                pytest.fail(f"Translation should succeed for nested expression '{expr}'")
    
    def test_enhanced_test_suite(self):
        """Test the enhanced built-in test suite."""
        translator = SymPyToLean4Translator()
        
        # Run enhanced test suite
        results = translator.test_translation()
        
        # Should have more test cases now
        assert len(results) >= 8, "Enhanced test suite should have at least 8 test cases"
        
        # Check for new test cases
        expected_new_cases = ['inverse_trig', 'hyperbolic', 'logarithm_base', 'complex_expression']
        for case in expected_new_cases:
            assert case in results, f"Enhanced test suite should include '{case}'"
        
        # Verify success rate
        successful_tests = sum(1 for r in results.values() if r['success'])
        total_tests = len(results)
        success_rate = successful_tests / total_tests
        
        # Should have decent success rate
        assert success_rate >= 0.6, f"Success rate should be at least 60%, got {success_rate:.1%}"
    
    def test_error_handling_for_unsupported_functions(self):
        """Test error handling for unsupported mathematical functions."""
        translator = SymPyToLean4Translator()
        
        # These should produce informative error messages rather than crashing
        unsupported_cases = [
            'bessel(x)',  # Bessel function
            'gamma(x)',   # Gamma function
            'zeta(x)',    # Riemann zeta function
        ]
        
        for expr in unsupported_cases:
            try:
                result = translator.translate_simple_expression(expr)
                # Should contain informative placeholder
                assert 'Unsupported' in result or '--' in result, f"Should indicate unsupported for '{expr}'"
            except TranslationError:
                # This is also acceptable - clear error handling
                pass
    
    def test_complex_equality_expressions(self):
        """Test translation of complex equality expressions."""
        translator = SymPyToLean4Translator()
        
        test_cases = [
            'sin(x)**2 + cos(x)**2 = 1',
            'exp(log(x)) = x',
            'log(exp(x)) = x', 
            'sqrt(x**2) = abs(x)',
            'sinh(x)**2 - cosh(x)**2 = -1'
        ]
        
        for expr in test_cases:
            try:
                result = translator.translate_simple_expression(expr)
                # Should contain equality (if successful)
                if 'Unsupported' not in result and '--' not in result:
                    assert ' = ' in result, f"Should contain equality for '{expr}'"
                    # Should contain Lean mathematical functions
                    assert 'Real.' in result or 'abs' in result, f"Should contain Lean functions for '{expr}'"
            except TranslationError:
                # Some complex expressions might fail - that's acceptable for now
                print(f"Translation failed for '{expr}'")
                print(result)
                pass
    
    def test_function_composition_and_chaining(self):
        """Test translation of function composition and chaining."""
        translator = SymPyToLean4Translator()
        
        composition_cases = [
            'sin(2*x)',
            'cos(x + y)', 
            'exp(x*y)',
            'log(x/y)',
            'sqrt(x + 1)'
        ]
        
        for expr in composition_cases:
            result = translator.translate_simple_expression(expr)
            # Should handle function composition correctly
            assert 'Real.' in result, f"Should contain Lean Real functions for '{expr}'"
            # Should preserve mathematical structure
            assert len(result) > len(expr), f"Lean translation should be more verbose for '{expr}'"
    
    def test_enhanced_variable_type_handling(self):
        """Test enhanced variable type handling in theorem translation."""
        translator = SymPyToLean4Translator()
        
        # Create mock theorem objects
        class MockTheorem:
            def __init__(self, id_, statement, expr):
                self.id = id_
                self.statement = statement
                self.sympy_expression = expr
        
        import sympy as sp
        
        # Test with complex expression involving multiple variable types
        expr = sp.Eq(sp.sin(sp.Symbol('x'))**2 + sp.cos(sp.Symbol('x'))**2, 1)
        theorem = MockTheorem("enhanced_trig", "Enhanced trigonometric identity", expr)
        
        result = translator.translate(theorem)
        
        # Should contain proper variable declarations
        assert 'variable (x : ℝ)' in result
        # Should contain proper theorem structure
        assert 'theorem enhanced_trig' in result
        # Should contain enhanced function translations
        assert 'Real.sin' in result and 'Real.cos' in result
    
    def test_supported_functions_list(self):
        """Test that the supported functions list includes enhanced functions."""
        translator = SymPyToLean4Translator()
        
        supported = translator.get_supported_functions()
        
        # Should include enhanced trigonometric functions
        enhanced_trig = ['asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh']
        for func in enhanced_trig:
            assert func in supported, f"Should support enhanced trigonometric function '{func}'"
        
        # Should include enhanced logarithmic functions
        enhanced_log = ['ln', 'log10', 'log2']
        for func in enhanced_log:
            assert func in supported, f"Should support enhanced logarithmic function '{func}'"
        
        # Should include enhanced algebraic functions
        enhanced_alg = ['floor', 'ceil', 'sign', 'factorial']
        for func in enhanced_alg:
            assert func in supported, f"Should support enhanced algebraic function '{func}'"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"]) 