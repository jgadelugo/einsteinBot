#!/usr/bin/env python3
"""
Demo script for Phase 5D Session 4: Enhanced SymPy to Lean Translation Engine

This script demonstrates the enhanced translation capabilities including:
- Advanced mathematical functions (trigonometric, logarithmic, hyperbolic)
- Complex mathematical expressions  
- Improved error handling and function composition
- Enhanced test suite with comprehensive coverage

Usage:
    PYTHONPATH=. venv/bin/python demos/demo_phase_5d_session_4.py
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from proofs.formal_systems.translation.sympy_to_lean import SymPyToLean4Translator
from proofs.formal_systems.base_interface import TranslationError
import sympy as sp


def demonstrate_enhanced_trigonometric_functions():
    """Demonstrate enhanced trigonometric function translation."""
    print("🔺 Enhanced Trigonometric Functions")
    print("=" * 50)
    
    translator = SymPyToLean4Translator()
    
    trig_examples = [
        ('sin(x)', 'Basic sine function'),
        ('asin(x)', 'Inverse sine (arcsine)'),
        ('sinh(x)', 'Hyperbolic sine'), 
        ('asin(sin(x))', 'Composed inverse/forward sine'),
        ('sin(x)**2 + cos(x)**2', 'Pythagorean identity'),
        ('cosh(x)**2 - sinh(x)**2', 'Hyperbolic identity'),
        ('tan(x)', 'Tangent function'),
        ('atan(x)', 'Inverse tangent')
    ]
    
    for expr, description in trig_examples:
        try:
            lean_result = translator.translate_simple_expression(expr)
            print(f"  {description}:")
            print(f"    SymPy: {expr}")
            print(f"    Lean:  {lean_result}")
            print()
        except TranslationError as e:
            print(f"  {description}: Translation failed - {e}")
            print()


def demonstrate_enhanced_logarithmic_functions():
    """Demonstrate enhanced logarithmic function translation."""
    print("📊 Enhanced Logarithmic Functions")
    print("=" * 50)
    
    translator = SymPyToLean4Translator()
    
    log_examples = [
        ('log(x)', 'Natural logarithm'),
        ('log10(x)', 'Base-10 logarithm'),
        ('log2(x)', 'Base-2 logarithm'),
        ('ln(x)', 'Natural logarithm (ln)'),
        ('log(x*y)', 'Logarithm of product'),
        ('exp(log(x))', 'Exponential-logarithm composition'),
        ('log10(100)', 'Concrete base-10 example'),
        ('log(exp(x))', 'Logarithm-exponential composition')
    ]
    
    for expr, description in log_examples:
        try:
            lean_result = translator.translate_simple_expression(expr)
            print(f"  {description}:")
            print(f"    SymPy: {expr}")
            print(f"    Lean:  {lean_result}")
            print()
        except TranslationError as e:
            print(f"  {description}: Translation failed - {e}")
            print()


def demonstrate_complex_mathematical_expressions():
    """Demonstrate translation of complex mathematical expressions."""
    print("🧮 Complex Mathematical Expressions")
    print("=" * 50)
    
    translator = SymPyToLean4Translator()
    
    complex_examples = [
        ('sqrt(x**2 + y**2)', 'Euclidean distance'),
        ('abs(sin(x))', 'Absolute value of sine'),
        ('exp(cos(x))', 'Exponential of cosine'),
        ('sin(2*x)', 'Sine with argument multiplication'),
        ('log(sin(x) + cos(x))', 'Logarithm of trigonometric sum'),
        ('(x + 1)**3', 'Cubic expansion'),
        ('sqrt(abs(x - y))', 'Nested function composition'),
        ('sinh(x)**2 + cosh(x)**2', 'Hyperbolic function combination')
    ]
    
    for expr, description in complex_examples:
        try:
            lean_result = translator.translate_simple_expression(expr)
            print(f"  {description}:")
            print(f"    SymPy: {expr}")
            print(f"    Lean:  {lean_result}")
            print()
        except TranslationError as e:
            print(f"  {description}: Translation failed - {e}")
            print()


def demonstrate_mathematical_identities():
    """Demonstrate translation of mathematical identities.""" 
    print("⚖️  Mathematical Identities")
    print("=" * 50)
    
    translator = SymPyToLean4Translator()
    
    identity_examples = [
        ('sin(x)**2 + cos(x)**2 = 1', 'Pythagorean identity'),
        ('exp(log(x)) = x', 'Exponential-logarithm inverse'),
        ('log(exp(x)) = x', 'Logarithm-exponential inverse'), 
        ('asin(sin(x)) = x', 'Inverse trigonometric identity'),
        ('cosh(x)**2 - sinh(x)**2 = 1', 'Hyperbolic identity'),
        ('sqrt(x**2) = abs(x)', 'Square root of square'),
        ('log(x*y) = log(x) + log(y)', 'Logarithm product rule'),
        ('exp(x + y) = exp(x)*exp(y)', 'Exponential sum rule')
    ]
    
    for expr, description in identity_examples:
        try:
            lean_result = translator.translate_simple_expression(expr)
            print(f"  {description}:")
            print(f"    SymPy: {expr}")
            print(f"    Lean:  {lean_result}")
            print()
        except TranslationError as e:
            print(f"  {description}: Translation failed - {e}")
            print()


def demonstrate_enhanced_test_suite():
    """Demonstrate the enhanced built-in test suite."""
    print("🧪 Enhanced Test Suite")
    print("=" * 50)
    
    translator = SymPyToLean4Translator()
    
    # Run the enhanced test suite
    print("Running enhanced test suite...")
    results = translator.test_translation()
    
    print(f"\nTest Results Summary:")
    print(f"  Total test cases: {len(results)}")
    
    successful = sum(1 for r in results.values() if r['success'])
    failed = len(results) - successful
    success_rate = successful / len(results) * 100
    
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Success rate: {success_rate:.1f}%")
    
    print(f"\nDetailed Results:")
    for test_name, result in results.items():
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"  {status} {test_name}: {result['original']}")
        if result['success']:
            print(f"      → {result['translated']}")
        else:
            print(f"      → Error: {result['error']}")
        print()


def demonstrate_supported_functions():
    """Demonstrate the list of supported mathematical functions."""
    print("📚 Supported Mathematical Functions")
    print("=" * 50)
    
    translator = SymPyToLean4Translator()
    supported = translator.get_supported_functions()
    
    # Categorize functions
    categories = {
        'Trigonometric': ['sin', 'cos', 'tan', 'asin', 'acos', 'atan'],
        'Hyperbolic': ['sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh'],
        'Logarithmic': ['log', 'ln', 'log10', 'log2'],
        'Exponential': ['exp'],
        'Algebraic': ['sqrt', 'abs', 'floor', 'ceil', 'sign'],
        'Special': ['factorial', 'pi', 'E', 'I'],
        'Constants': ['oo', 'zoo', 'nan']
    }
    
    for category, functions in categories.items():
        print(f"  {category}:")
        available = [f for f in functions if f in supported]
        if available:
            print(f"    {', '.join(available)}")
        else:
            print(f"    (none available)")
        print()
    
    total_supported = len(supported)
    total_enhanced = sum(len(funcs) for funcs in categories.values())
    coverage = (len([f for funcs in categories.values() for f in funcs if f in supported]) / total_enhanced) * 100
    
    print(f"Total supported functions: {total_supported}")
    print(f"Enhanced coverage: {coverage:.1f}%")


def demonstrate_theorem_translation():
    """Demonstrate complete theorem translation."""
    print("📜 Complete Theorem Translation")
    print("=" * 50)
    
    translator = SymPyToLean4Translator()
    
    # Create mock theorem objects
    class MockTheorem:
        def __init__(self, id_, statement, expr_str):
            self.id = id_
            self.statement = statement
            self.sympy_expression = sp.parse_expr(expr_str) if '=' not in expr_str else sp.Eq(*[sp.parse_expr(part.strip()) for part in expr_str.split('=')])
    
    theorem_examples = [
        MockTheorem(
            "enhanced_trig_identity", 
            "Enhanced trigonometric identity with hyperbolic functions",
            "sinh(x)**2 + cosh(x)**2"
        ),
        MockTheorem(
            "logarithm_composition",
            "Logarithm composition with enhanced base handling", 
            "log10(x*y)"
        ),
        MockTheorem(
            "inverse_trig_composition",
            "Inverse trigonometric function composition",
            "asin(cos(x))"
        )
    ]
    
    for theorem in theorem_examples:
        try:
            lean_theorem = translator.translate(theorem)
            print(f"  Theorem: {theorem.statement}")
            print(f"  ID: {theorem.id}")
            print(f"  Lean 4 Translation:")
            print("    " + "\n    ".join(lean_theorem.split('\n')))
            print()
        except TranslationError as e:
            print(f"  Theorem {theorem.id}: Translation failed - {e}")
            print()


def main():
    """Main demonstration function."""
    print("🚀 Phase 5D Session 4: Enhanced SymPy to Lean Translation Engine")
    print("=" * 80)
    print("Demonstrating advanced mathematical function translation capabilities")
    print("=" * 80)
    print()
    
    try:
        demonstrate_enhanced_trigonometric_functions()
        print()
        
        demonstrate_enhanced_logarithmic_functions()
        print()
        
        demonstrate_complex_mathematical_expressions()
        print()
        
        demonstrate_mathematical_identities()
        print()
        
        demonstrate_enhanced_test_suite()
        print()
        
        demonstrate_supported_functions()
        print()
        
        demonstrate_theorem_translation()
        print()
        
        print("✅ Session 4 Enhancement Demonstration Complete!")
        print("=" * 80)
        print("Key improvements achieved:")
        print("  • Enhanced trigonometric and hyperbolic function support")
        print("  • Advanced logarithmic functions with base conversion")
        print("  • Improved function composition and nesting")
        print("  • Better error handling for unsupported functions")
        print("  • Comprehensive test suite with 13 test categories") 
        print("  • Support for complex mathematical identities")
        print("  • Enhanced theorem translation capabilities")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 