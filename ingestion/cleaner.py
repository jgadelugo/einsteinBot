"""
Cleaner module for normalizing mathematical symbols and expressions.

This module provides functionality to clean, normalize, and standardize
mathematical expressions by converting LaTeX commands to Unicode symbols,
resolving notational inconsistencies, and applying consistent formatting.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Union

from config import (
    GREEK_LETTERS,
    MATH_OPERATORS,
    FRACTION_PATTERNS,
    logger,
)


class FormulaCleaner:
    """
    Cleaner for normalizing mathematical expressions and symbols.
    
    This class provides methods to convert LaTeX commands to Unicode symbols,
    normalize notation, and apply consistent formatting to mathematical expressions.
    """
    
    def __init__(self, preserve_latex: bool = False):
        """
        Initialize the formula cleaner.
        
        Args:
            preserve_latex: Whether to preserve LaTeX commands alongside Unicode conversion
        """
        self.preserve_latex = preserve_latex
        self.logger = logging.getLogger(f"{__name__}.FormulaCleaner")
        
        # Build comprehensive symbol mapping
        self.symbol_mapping = {**GREEK_LETTERS, **MATH_OPERATORS}
        
        # Build reverse mapping for Unicode to LaTeX
        self.unicode_to_latex = {v: k for k, v in self.symbol_mapping.items()}
        
        # Compile regex patterns for better performance
        self._compile_patterns()
        
        self.logger.info(f"Initialized formula cleaner (preserve_latex={preserve_latex})")
    
    def _compile_patterns(self) -> None:
        """Compile regex patterns for symbol normalization."""
        # Pattern for LaTeX commands
        self.latex_command_pattern = re.compile(r'\\[a-zA-Z]+')
        
        # Pattern for fractions
        self.fraction_patterns = [re.compile(pattern) for pattern in FRACTION_PATTERNS]
        
        # Pattern for superscripts and subscripts
        self.superscript_pattern = re.compile(r'\^(\{[^}]*\}|[^{}\s])')
        self.subscript_pattern = re.compile(r'_(\{[^}]*\}|[^{}\s])')
        
        # Pattern for function names
        self.function_pattern = re.compile(r'\\(sin|cos|tan|sec|csc|cot|sinh|cosh|tanh|log|ln|exp|sqrt|lim)')
        
        # Pattern for brackets and parentheses
        self.bracket_patterns = {
            r'\\left\(': '(',
            r'\\right\)': ')',
            r'\\left\[': '[',
            r'\\right\]': ']',
            r'\\left\{': '{',
            r'\\right\}': '}',
            r'\\left\|': '|',
            r'\\right\|': '|',
            r'\\left\\langle': '⟨',
            r'\\right\\rangle': '⟩',
        }
        
        # Pattern for common LaTeX environments to preserve
        self.preserve_environments = [
            r'\\begin\{[^}]*\}.*?\\end\{[^}]*\}',
            r'\\[a-zA-Z]+\{[^}]*\}',  # Commands with arguments
        ]
    
    def clean_expression(self, expression: str, normalize_unicode: bool = True) -> str:
        """
        Clean and normalize a mathematical expression.
        
        Args:
            expression: Raw mathematical expression
            normalize_unicode: Whether to convert LaTeX to Unicode symbols
            
        Returns:
            Cleaned and normalized expression
        """
        if not expression:
            return expression
        
        # Start with the original expression
        cleaned = expression.strip()
        
        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Normalize symbols if requested
        if normalize_unicode:
            cleaned = self._normalize_symbols(cleaned)
        
        # Normalize fractions
        cleaned = self._normalize_fractions(cleaned)
        
        # Normalize brackets
        cleaned = self._normalize_brackets(cleaned)
        
        # Normalize superscripts and subscripts
        cleaned = self._normalize_scripts(cleaned)
        
        # Clean up remaining artifacts
        cleaned = self._cleanup_artifacts(cleaned)
        
        self.logger.debug(f"Cleaned expression: '{expression}' -> '{cleaned}'")
        return cleaned.strip()
    
    def _normalize_symbols(self, expression: str) -> str:
        """
        Convert LaTeX symbols to Unicode equivalents.
        
        Args:
            expression: Expression containing LaTeX symbols
            
        Returns:
            Expression with Unicode symbols
        """
        normalized = expression
        
        # Sort by length (longest first) to avoid partial replacements
        sorted_symbols = sorted(self.symbol_mapping.items(), key=lambda x: len(x[0]), reverse=True)
        
        for latex_cmd, unicode_symbol in sorted_symbols:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(latex_cmd) + r'\b'
            normalized = re.sub(pattern, unicode_symbol, normalized)
        
        return normalized
    
    def _normalize_fractions(self, expression: str) -> str:
        """
        Convert LaTeX fraction commands to a consistent format.
        
        Args:
            expression: Expression containing LaTeX fractions
            
        Returns:
            Expression with normalized fractions
        """
        normalized = expression
        
        for pattern in self.fraction_patterns:
            def replace_fraction(match):
                numerator = match.group(1).strip()
                denominator = match.group(2).strip()
                
                # If both are simple, use inline format
                if (len(numerator) <= 3 and len(denominator) <= 3 and 
                    not re.search(r'[{}\\]', numerator + denominator)):
                    return f"({numerator})/({denominator})"
                else:
                    # Keep fraction format for complex expressions
                    return f"frac({numerator}, {denominator})"
            
            normalized = pattern.sub(replace_fraction, normalized)
        
        return normalized
    
    def _normalize_brackets(self, expression: str) -> str:
        """
        Normalize LaTeX bracket commands to standard symbols.
        
        Args:
            expression: Expression containing LaTeX brackets
            
        Returns:
            Expression with normalized brackets
        """
        normalized = expression
        
        for latex_bracket, unicode_bracket in self.bracket_patterns.items():
            normalized = re.sub(latex_bracket, unicode_bracket, normalized)
        
        return normalized
    
    def _normalize_scripts(self, expression: str) -> str:
        """
        Normalize superscripts and subscripts for better readability.
        
        Args:
            expression: Expression containing scripts
            
        Returns:
            Expression with normalized scripts
        """
        normalized = expression
        
        # Handle superscripts
        def replace_superscript(match):
            script = match.group(1)
            # Remove braces if present
            if script.startswith('{') and script.endswith('}'):
                script = script[1:-1]
            return f'^({script})' if len(script) > 1 else f'^{script}'
        
        normalized = self.superscript_pattern.sub(replace_superscript, normalized)
        
        # Handle subscripts
        def replace_subscript(match):
            script = match.group(1)
            # Remove braces if present
            if script.startswith('{') and script.endswith('}'):
                script = script[1:-1]
            return f'_({script})' if len(script) > 1 else f'_{script}'
        
        normalized = self.subscript_pattern.sub(replace_subscript, normalized)
        
        return normalized
    
    def _cleanup_artifacts(self, expression: str) -> str:
        """
        Clean up remaining LaTeX artifacts and formatting issues.
        
        Args:
            expression: Expression to clean up
            
        Returns:
            Cleaned expression
        """
        cleaned = expression
        
        # Remove empty braces
        cleaned = re.sub(r'\{\}', '', cleaned)
        
        # Remove standalone backslashes
        cleaned = re.sub(r'\\(?![a-zA-Z])', '', cleaned)
        
        # Normalize spacing around operators
        cleaned = re.sub(r'\s*([+\-*/=<>≤≥≠±])\s*', r' \1 ', cleaned)
        
        # Remove excessive spaces
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Clean up common artifacts
        artifacts = [
            r'\\,',  # Thin space
            r'\\;',  # Medium space
            r'\\!',  # Negative thin space
            r'\\quad',  # Quad space
            r'\\qquad',  # Double quad space
        ]
        
        for artifact in artifacts:
            cleaned = re.sub(artifact, ' ', cleaned)
        
        return cleaned
    
    def normalize_symbols(self, text: str) -> str:
        """
        Normalize mathematical symbols in text (convenience method).
        
        Args:
            text: Text containing mathematical symbols
            
        Returns:
            Text with normalized symbols
        """
        return self.clean_expression(text, normalize_unicode=True)
    
    def batch_clean(self, expressions: List[str]) -> List[str]:
        """
        Clean multiple expressions in batch.
        
        Args:
            expressions: List of expressions to clean
            
        Returns:
            List of cleaned expressions
        """
        cleaned_expressions = []
        
        for expr in expressions:
            try:
                cleaned = self.clean_expression(expr)
                cleaned_expressions.append(cleaned)
            except Exception as e:
                self.logger.warning(f"Failed to clean expression '{expr}': {e}")
                cleaned_expressions.append(expr)  # Keep original if cleaning fails
        
        self.logger.info(f"Batch cleaned {len(cleaned_expressions)} expressions")
        return cleaned_expressions
    
    def get_symbol_statistics(self, expressions: List[str]) -> Dict[str, int]:
        """
        Get statistics about symbol usage in expressions.
        
        Args:
            expressions: List of expressions to analyze
            
        Returns:
            Dictionary with symbol usage statistics
        """
        stats = {
            "total_expressions": len(expressions),
            "latex_commands": 0,
            "unicode_symbols": 0,
            "fractions": 0,
            "superscripts": 0,
            "subscripts": 0,
            "functions": 0,
        }
        
        for expr in expressions:
            # Count LaTeX commands
            latex_matches = self.latex_command_pattern.findall(expr)
            stats["latex_commands"] += len(latex_matches)
            
            # Count Unicode symbols
            for unicode_symbol in self.unicode_to_latex.keys():
                stats["unicode_symbols"] += expr.count(unicode_symbol)
            
            # Count fractions
            for pattern in self.fraction_patterns:
                stats["fractions"] += len(pattern.findall(expr))
            
            # Count scripts
            stats["superscripts"] += len(self.superscript_pattern.findall(expr))
            stats["subscripts"] += len(self.subscript_pattern.findall(expr))
            
            # Count functions
            stats["functions"] += len(self.function_pattern.findall(expr))
        
        return stats
    
    def convert_to_sympy_format(self, expression: str) -> str:
        """
        Convert expression to SymPy-compatible format.
        
        Args:
            expression: Mathematical expression
            
        Returns:
            SymPy-compatible expression string
        """
        # Start with basic cleaning
        sympy_expr = self.clean_expression(expression, normalize_unicode=False)
        
        # SymPy-specific conversions
        sympy_conversions = {
            # Keep common LaTeX commands that SymPy understands
            r'\\frac\{([^{}]+)\}\{([^{}]+)\}': r'(\1)/(\2)',
            r'\\sqrt\{([^{}]+)\}': r'sqrt(\1)',
            r'\\sin\b': 'sin',
            r'\\cos\b': 'cos',
            r'\\tan\b': 'tan',
            r'\\log\b': 'log',
            r'\\ln\b': 'log',
            r'\\exp\b': 'exp',
            r'\\pi\b': 'pi',
            r'\\e\b': 'E',
            # Convert some Greek letters to SymPy symbols
            r'\\alpha\b': 'alpha',
            r'\\beta\b': 'beta',
            r'\\gamma\b': 'gamma',
            r'\\delta\b': 'delta',
            r'\\epsilon\b': 'epsilon',
            r'\\theta\b': 'theta',
            r'\\lambda\b': 'lambda',
            r'\\mu\b': 'mu',
            r'\\sigma\b': 'sigma',
            r'\\phi\b': 'phi',
            r'\\omega\b': 'omega',
        }
        
        for pattern, replacement in sympy_conversions.items():
            sympy_expr = re.sub(pattern, replacement, sympy_expr)
        
        # Clean up remaining LaTeX artifacts
        sympy_expr = re.sub(r'\\[a-zA-Z]+', '', sympy_expr)
        sympy_expr = re.sub(r'[{}]', '', sympy_expr)
        
        return sympy_expr.strip()


# Convenience functions for common operations

def normalize_symbols(text: str, preserve_latex: bool = False) -> str:
    """
    Normalize mathematical symbols in text (convenience function).
    
    Args:
        text: Text containing mathematical symbols
        preserve_latex: Whether to preserve LaTeX commands
        
    Returns:
        Text with normalized symbols
    """
    cleaner = FormulaCleaner(preserve_latex=preserve_latex)
    return cleaner.normalize_symbols(text)


def clean_math_expression(expression: str, to_sympy: bool = False) -> str:
    """
    Clean a mathematical expression (convenience function).
    
    Args:
        expression: Expression to clean
        to_sympy: Whether to format for SymPy compatibility
        
    Returns:
        Cleaned expression
    """
    cleaner = FormulaCleaner()
    
    if to_sympy:
        return cleaner.convert_to_sympy_format(expression)
    else:
        return cleaner.clean_expression(expression)


def batch_normalize(expressions: List[str], to_sympy: bool = False) -> List[str]:
    """
    Normalize multiple expressions in batch (convenience function).
    
    Args:
        expressions: List of expressions to normalize
        to_sympy: Whether to format for SymPy compatibility
        
    Returns:
        List of normalized expressions
    """
    cleaner = FormulaCleaner()
    
    if to_sympy:
        return [cleaner.convert_to_sympy_format(expr) for expr in expressions]
    else:
        return cleaner.batch_clean(expressions)


def create_symbol_mapping() -> Dict[str, str]:
    """
    Create a comprehensive symbol mapping dictionary.
    
    Returns:
        Dictionary mapping LaTeX commands to Unicode symbols
    """
    return {**GREEK_LETTERS, **MATH_OPERATORS}


def detect_notation_inconsistencies(expressions: List[str]) -> Dict[str, List[str]]:
    """
    Detect potential notation inconsistencies in a list of expressions.
    
    Args:
        expressions: List of mathematical expressions
        
    Returns:
        Dictionary mapping inconsistency types to lists of problematic expressions
    """
    inconsistencies = {
        "mixed_notation": [],
        "unknown_symbols": [],
        "malformed_commands": [],
    }
    
    cleaner = FormulaCleaner()
    known_symbols = set(cleaner.symbol_mapping.keys())
    
    for expr in expressions:
        # Check for mixed LaTeX and Unicode
        has_latex = bool(re.search(r'\\[a-zA-Z]+', expr))
        has_unicode = any(symbol in expr for symbol in cleaner.unicode_to_latex.keys())
        
        if has_latex and has_unicode:
            inconsistencies["mixed_notation"].append(expr)
        
        # Check for unknown LaTeX commands
        latex_commands = re.findall(r'\\[a-zA-Z]+', expr)
        for cmd in latex_commands:
            if cmd not in known_symbols:
                inconsistencies["unknown_symbols"].append(f"{expr} (unknown: {cmd})")
        
        # Check for malformed commands
        if re.search(r'\\[a-zA-Z]*[^a-zA-Z\\]', expr):
            inconsistencies["malformed_commands"].append(expr)
    
    return inconsistencies


if __name__ == "__main__":
    # Example usage and testing
    sample_expressions = [
        r"x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}",
        r"\int_0^\infty e^{-x^2} dx = \frac{\sqrt{\pi}}{2}",
        r"\sum_{n=1}^{\infty} \frac{1}{n^2} = \frac{\pi^2}{6}",
        r"\alpha + \beta = \gamma \cdot \delta",
        r"F = ma \quad \text{where } a = \frac{dv}{dt}",
        r"\lim_{x \to 0} \frac{\sin x}{x} = 1",
    ]
    
    cleaner = FormulaCleaner()
    
    print("Original expressions:")
    for i, expr in enumerate(sample_expressions, 1):
        print(f"{i}. {expr}")
    
    print("\nCleaned expressions:")
    cleaned = cleaner.batch_clean(sample_expressions)
    for i, expr in enumerate(cleaned, 1):
        print(f"{i}. {expr}")
    
    print("\nSymPy-compatible format:")
    for i, expr in enumerate(sample_expressions, 1):
        sympy_expr = cleaner.convert_to_sympy_format(expr)
        print(f"{i}. {sympy_expr}")
    
    print("\nSymbol statistics:")
    stats = cleaner.get_symbol_statistics(sample_expressions)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nNotation inconsistencies:")
    inconsistencies = detect_notation_inconsistencies(sample_expressions)
    for issue_type, issues in inconsistencies.items():
        if issues:
            print(f"  {issue_type}: {len(issues)} found")
            for issue in issues[:3]:  # Show first 3
                print(f"    {issue}")
        else:
            print(f"  {issue_type}: None found")