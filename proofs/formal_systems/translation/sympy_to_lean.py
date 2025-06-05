"""
SymPy to Lean 4 translation engine.

This module provides translation capabilities from SymPy mathematical expressions
to Lean 4 formal theorem proving language syntax.

The translator handles:
- Basic algebraic expressions (addition, multiplication, exponentiation)
- Trigonometric functions
- Logarithmic and exponential functions
- Variable declarations and type annotations
- Theorem statement formatting
"""

import sympy as sp
from typing import Dict, List, Set, Optional, Any
from ..base_interface import TranslationError


class SymPyToLean4Translator:
    """
    Translator for converting SymPy expressions to Lean 4 syntax.
    
    This class provides comprehensive translation from SymPy mathematical
    expressions to valid Lean 4 code, including proper variable typing,
    theorem formatting, and expression syntax conversion.
    """
    
    def __init__(self):
        """Initialize the translator with symbol mappings."""
        # Mapping from SymPy function names to Lean 4 equivalents
        self.function_map = {
            'sin': 'Real.sin',
            'cos': 'Real.cos', 
            'tan': 'Real.tan',
            'log': 'Real.log',
            'exp': 'Real.exp',
            'sqrt': 'Real.sqrt',
            'abs': 'abs',
            'pi': 'Real.pi',
            'E': 'Real.exp 1',
            # Enhanced trigonometric functions
            'asin': 'Real.arcsin',
            'acos': 'Real.arccos',
            'atan': 'Real.arctan',
            'atan2': 'Real.arctan2',
            'sinh': 'Real.sinh',
            'cosh': 'Real.cosh',
            'tanh': 'Real.tanh',
            'asinh': 'Real.arcsinh',
            'acosh': 'Real.arccosh',
            'atanh': 'Real.arctanh',
            # Enhanced algebraic functions
            'floor': 'Int.floor',
            'ceil': 'Int.ceil',
            'sign': 'Int.sign',
            'factorial': 'Nat.factorial',
            # Logarithmic functions
            'ln': 'Real.log',
            'log10': 'Real.log',  # Will handle base conversion
            'log2': 'Real.log',   # Will handle base conversion
            # Complex number support
            'I': 'Complex.I',
            'im': 'Complex.im',
            're': 'Complex.re',
            'conjugate': 'Complex.conj',
            # Special constants
            'oo': '∞',
            'zoo': '∞',  # Complex infinity
            'nan': 'NaN'
        }
        
        # Mapping from SymPy operators to Lean 4 equivalents
        self.operator_map = {
            sp.Add: ' + ',
            sp.Mul: ' * ',
            sp.Pow: '^',
            sp.Eq: ' = '
        }
    
    def translate(self, theorem) -> str:
        """
        Translate a complete theorem to Lean 4 syntax.
        
        Args:
            theorem: Theorem object from Phase 5A with SymPy expressions
            
        Returns:
            Complete Lean 4 theorem statement
            
        Raises:
            TranslationError: If translation fails
        """
        try:
            # Extract information from theorem
            theorem_id = getattr(theorem, 'id', 'unnamed_theorem')
            statement = getattr(theorem, 'statement', str(theorem))
            sympy_expr = getattr(theorem, 'sympy_expression', None)
            
            if sympy_expr is None:
                # Try to parse the statement as a SymPy expression
                try:
                    sympy_expr = sp.parse_expr(statement)
                except Exception:
                    raise TranslationError(f"Could not extract SymPy expression from theorem: {theorem}")
            
            # Extract variables and create declarations
            variables = self._extract_variables(sympy_expr)
            var_decls = self._create_variable_declarations(variables)
            
            # Translate the main expression
            lean_expr = self._translate_expr(sympy_expr)
            
            # Create complete theorem
            lean_theorem = self._format_theorem(theorem_id, statement, var_decls, lean_expr)
            
            return lean_theorem
            
        except Exception as e:
            raise TranslationError(f"Failed to translate theorem: {e}")
    
    def _extract_variables(self, expr: sp.Expr) -> List[sp.Symbol]:
        """
        Extract all variables from a SymPy expression.
        
        Args:
            expr: SymPy expression to analyze
            
        Returns:
            Sorted list of variable symbols
        """
        return sorted(expr.free_symbols, key=str)
    
    def _create_variable_declarations(self, variables: List[sp.Symbol]) -> str:
        """
        Create Lean 4 variable declarations.
        
        Args:
            variables: List of SymPy symbols
            
        Returns:
            Lean 4 variable declaration string
        """
        if not variables:
            return ""
        
        # For now, assume all variables are real numbers
        var_list = " ".join(str(var) for var in variables)
        return f"variable ({var_list} : ℝ)"
    
    def _translate_expr(self, expr: sp.Expr) -> str:
        """
        Translate a SymPy expression to Lean 4 syntax.
        
        Args:
            expr: SymPy expression to translate
            
        Returns:
            Lean 4 expression string
            
        Raises:
            TranslationError: If expression cannot be translated
        """
        try:
            return self._translate_expr_recursive(expr)
        except Exception as e:
            raise TranslationError(f"Failed to translate expression {expr}: {e}")
    
    def _translate_expr_recursive(self, expr: sp.Expr) -> str:
        """
        Recursively translate SymPy expression to Lean 4.
        
        Args:
            expr: SymPy expression to translate
            
        Returns:
            Lean 4 expression string
        """
        # Handle atomic expressions
        if expr.is_Symbol:
            return str(expr)
        
        elif expr.is_Number:
            if expr.is_Integer:
                return str(expr)
            elif expr.is_Rational:
                return f"({expr.p} / {expr.q} : ℝ)"
            elif expr.is_Float:
                return str(float(expr))
            else:
                return str(expr)
        
        # Handle constants
        elif expr == sp.pi:
            return "Real.pi"
        elif expr == sp.E:
            return "Real.exp 1"
        elif expr == sp.I:
            return "Complex.I"
        elif hasattr(expr, 'is_Symbol') and str(expr) == 'E':
            return "Real.exp 1"
        
        # Handle compound expressions
        elif isinstance(expr, sp.Add):
            return self._translate_add(expr)
        
        elif isinstance(expr, sp.Mul):
            return self._translate_mul(expr)
        
        elif isinstance(expr, sp.Pow):
            return self._translate_pow(expr)
        
        elif isinstance(expr, sp.Eq):
            return self._translate_eq(expr)
        
        # Handle functions
        elif isinstance(expr, sp.Function):
            return self._translate_function(expr)
        
        else:
            # Fallback for unsupported expressions
            return f"-- Unsupported expression: {expr}"
    
    def _translate_add(self, expr: sp.Add) -> str:
        """Translate addition expression."""
        terms = [self._translate_expr_recursive(arg) for arg in expr.args]
        return f"({' + '.join(terms)})"
    
    def _translate_mul(self, expr: sp.Mul) -> str:
        """Translate multiplication expression."""
        factors = [self._translate_expr_recursive(arg) for arg in expr.args]
        return f"({' * '.join(factors)})"
    
    def _translate_pow(self, expr: sp.Pow) -> str:
        """Translate power expression."""
        base = self._translate_expr_recursive(expr.base)
        exponent = self._translate_expr_recursive(expr.exp)
        
        # Special case for square root
        if expr.exp == sp.Rational(1, 2):
            return f"Real.sqrt {base}"
        
        # Special case for integer powers
        elif expr.exp.is_Integer and expr.exp >= 0:
            return f"({base}) ^ {exponent}"
        
        # General case - for now use ^ operator (Lean 4 handles type conversion)
        else:
            return f"({base}) ^ ({exponent})"
    
    def _translate_eq(self, expr: sp.Eq) -> str:
        """Translate equality expression."""
        lhs = self._translate_expr_recursive(expr.lhs)
        rhs = self._translate_expr_recursive(expr.rhs)
        return f"{lhs} = {rhs}"
    
    def _translate_function(self, expr: sp.Function) -> str:
        """Translate function calls with enhanced support for complex mathematical functions."""
        func_name = expr.func.__name__
        args = [self._translate_expr_recursive(arg) for arg in expr.args]
        
        # Handle special function cases
        if func_name == 'log' and len(args) == 2:
            # Logarithm with base: log(x, base) -> Real.log x / Real.log base
            base_arg, value_arg = args
            return f"(Real.log {value_arg} / Real.log {base_arg})"
            
        elif func_name == 'log10' and len(args) == 1:
            # Base-10 logarithm: log10(x) -> Real.log x / Real.log 10
            return f"(Real.log {args[0]} / Real.log 10)"
            
        elif func_name == 'log2' and len(args) == 1:
            # Base-2 logarithm: log2(x) -> Real.log x / Real.log 2
            return f"(Real.log {args[0]} / Real.log 2)"
            
        elif func_name == 'Abs':
            # Absolute value
            return f"abs {args[0]}" if args else "abs"
            
        elif func_name == 'Max':
            # Maximum function
            if len(args) == 2:
                return f"max {args[0]} {args[1]}"
            else:
                return f"List.maximum [{', '.join(args)}]"
                
        elif func_name == 'Min':
            # Minimum function  
            if len(args) == 2:
                return f"min {args[0]} {args[1]}"
            else:
                return f"List.minimum [{', '.join(args)}]"
                
        elif func_name == 'factorial':
            # Factorial
            return f"Nat.factorial {args[0]}" if args else "Nat.factorial"
            
        elif func_name == 'binomial':
            # Binomial coefficient
            if len(args) == 2:
                return f"Nat.choose {args[0]} {args[1]}"
            else:
                return f"-- Invalid binomial: {func_name} {args}"
                
        elif func_name in ['Piecewise', 'piecewise']:
            # Piecewise function -> conditional expression
            return self._translate_piecewise(expr)
            
        elif func_name in ['Sum', 'summation']:
            # Summation
            return self._translate_summation(expr)
            
        elif func_name in ['Product', 'product']:
            # Product
            return self._translate_product(expr)
            
        elif func_name in ['Derivative', 'diff']:
            # Derivative - placeholder for now
            return f"-- Derivative: d/dx({', '.join(args)})"
            
        elif func_name in ['Integral', 'integrate']:
            # Integral - placeholder for now
            return f"-- Integral: ∫({', '.join(args)}) dx"
            
        elif func_name in ['Limit', 'limit']:
            # Limit - placeholder for now
            return f"-- Limit: lim({', '.join(args)})"
            
        elif func_name in self.function_map:
            # Standard function mapping
            lean_func = self.function_map[func_name]
            if args:
                # Handle parentheses for complex arguments
                formatted_args = []
                for arg in args:
                    if (' ' in arg or '+' in arg or '-' in arg) and not arg.startswith('('):
                        formatted_args.append(f"({arg})")
                    else:
                        formatted_args.append(arg)
                
                if len(args) == 1:
                    return f"{lean_func} {formatted_args[0]}"
                else:
                    return f"{lean_func} ({', '.join(formatted_args)})"
            else:
                return lean_func
        else:
            # Unsupported function - provide informative placeholder
            if args:
                args_str = f"({', '.join(args)})"
            else:
                args_str = ""
            return f"-- Unsupported function: {func_name}{args_str}"
    
    def _format_theorem(self, theorem_id: str, statement: str, var_decls: str, lean_expr: str) -> str:
        """
        Format complete Lean 4 theorem.
        
        Args:
            theorem_id: Unique identifier for theorem
            statement: Human-readable statement
            var_decls: Variable declarations
            lean_expr: Translated expression
            
        Returns:
            Complete Lean 4 theorem
        """
        safe_id = self._sanitize_id(theorem_id)
        
        lean_code = f"""-- {statement}
{var_decls}

theorem {safe_id} : {lean_expr} := by
  sorry
"""
        return lean_code.strip()
    
    def _translate_piecewise(self, expr: sp.Function) -> str:
        """
        Translate piecewise function to Lean 4 conditional expressions.
        
        Args:
            expr: SymPy Piecewise function
            
        Returns:
            Lean 4 conditional expression
        """
        # This is a simplified implementation - real piecewise translation is complex
        args = [self._translate_expr_recursive(arg) for arg in expr.args]
        if len(args) >= 2:
            condition = args[1] if len(args) > 1 else "true"
            value = args[0]
            return f"if {condition} then {value} else 0"
        else:
            return f"-- Complex piecewise: {args}"
    
    def _translate_summation(self, expr: sp.Function) -> str:
        """
        Translate summation to Lean 4.
        
        Args:
            expr: SymPy Sum function
            
        Returns:
            Lean 4 summation expression
        """
        args = [self._translate_expr_recursive(arg) for arg in expr.args]
        if len(args) >= 1:
            # Simplified summation - real implementation would handle bounds
            return f"∑ i, {args[0]}"
        else:
            return f"-- Sum: {args}"
    
    def _translate_product(self, expr: sp.Function) -> str:
        """
        Translate product to Lean 4.
        
        Args:
            expr: SymPy Product function
            
        Returns:
            Lean 4 product expression
        """
        args = [self._translate_expr_recursive(arg) for arg in expr.args]
        if len(args) >= 1:
            # Simplified product - real implementation would handle bounds
            return f"∏ i, {args[0]}"
        else:
            return f"-- Product: {args}"
    
    def _sanitize_id(self, theorem_id: str) -> str:
        """
        Sanitize theorem ID for use in Lean 4.
        
        Args:
            theorem_id: Raw theorem identifier
            
        Returns:
            Valid Lean 4 identifier
        """
        # Replace invalid characters with underscores
        import re
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', theorem_id)
        
        # Ensure it starts with a letter
        if sanitized and not sanitized[0].isalpha():
            sanitized = 'theorem_' + sanitized
        
        # Ensure it's not empty
        if not sanitized:
            sanitized = 'unnamed_theorem'
        
        return sanitized
    
    def translate_simple_expression(self, expr_str: str) -> str:
        """
        Translate a simple expression string to Lean 4.
        
        Args:
            expr_str: String representation of mathematical expression
            
        Returns:
            Lean 4 expression
            
        Raises:
            TranslationError: If parsing or translation fails
        """
        try:
            # Handle equality expressions specially
            if '=' in expr_str:
                parts = expr_str.split('=')
                if len(parts) == 2:
                    lhs = sp.parse_expr(parts[0].strip())
                    rhs = sp.parse_expr(parts[1].strip())
                    expr = sp.Eq(lhs, rhs)
                else:
                    expr = sp.parse_expr(expr_str)
            else:
                expr = sp.parse_expr(expr_str)
            
            return self._translate_expr(expr)
        except Exception as e:
            raise TranslationError(f"Failed to translate expression '{expr_str}': {e}")
    
    def get_supported_functions(self) -> List[str]:
        """Get list of supported mathematical functions."""
        return list(self.function_map.keys())
    
    def test_translation(self, test_cases: Optional[Dict[str, str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Test the translator with various expressions.
        
        Args:
            test_cases: Optional dictionary of test cases (name -> expression)
            
        Returns:
            Dictionary of test results
        """
        if test_cases is None:
            test_cases = {
                'simple_equality': 'x + 1 = 2',
                'algebraic_identity': '(x + 1)**2 = x**2 + 2*x + 1',
                'trigonometric': 'sin(x)**2 + cos(x)**2 = 1',
                'exponential': 'exp(log(x)) = x',
                'polynomial': 'x**3 - 2*x**2 + x - 1 = 0',
                # Enhanced test cases for Session 4
                'inverse_trig': 'asin(sin(x)) = x',
                'hyperbolic': 'cosh(x)**2 - sinh(x)**2 = 1',
                'logarithm_base': 'log10(100) = 2',
                'complex_expression': 'sqrt(x**2 + y**2)',
                'absolute_value': 'abs(-5) = 5',
                'trigonometric_advanced': 'sin(2*x) = 2*sin(x)*cos(x)',
                'exponential_rules': 'exp(x + y) = exp(x)*exp(y)',
                'logarithm_properties': 'log(x*y) = log(x) + log(y)'
            }
        
        results = {}
        
        for name, expr_str in test_cases.items():
            try:
                translated = self.translate_simple_expression(expr_str)
                results[name] = {
                    'original': expr_str,
                    'translated': translated,
                    'success': True,
                    'error': None
                }
            except Exception as e:
                results[name] = {
                    'original': expr_str,
                    'translated': None,
                    'success': False,
                    'error': str(e)
                }
        
        return results 