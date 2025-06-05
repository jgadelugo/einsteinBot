"""
Formula extraction module for detecting and extracting mathematical expressions.

This module provides functionality to identify, extract, and clean LaTeX-style
mathematical expressions from text content using sophisticated pattern matching
and heuristics.
"""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple, Union

from config import (
    INLINE_MATH_DELIMITERS,
    BLOCK_MATH_DELIMITERS,
    ALL_MATH_DELIMITERS,
    MIN_FORMULA_LENGTH,
    MAX_FORMULA_LENGTH,
    DEFAULT_DEDUPLICATE,
    DEFAULT_CLEAN_FORMULAS,
    logger,
)


class FormulaExtractor:
    """
    Extractor for identifying and extracting mathematical expressions from text.
    
    This class provides methods to detect LaTeX-style mathematical expressions,
    including inline math, display math, and various equation environments.
    """
    
    def __init__(self, clean_formulas: bool = DEFAULT_CLEAN_FORMULAS):
        """
        Initialize the formula extractor.
        
        Args:
            clean_formulas: Whether to apply basic cleaning to extracted formulas
        """
        self.clean_formulas = clean_formulas
        self.logger = logging.getLogger(f"{__name__}.FormulaExtractor")
        
        # Compile regex patterns for better performance
        self._compile_patterns()
        
        self.logger.info(f"Initialized formula extractor (clean_formulas={clean_formulas})")
    
    def _compile_patterns(self) -> None:
        """Compile regex patterns for mathematical expression detection."""
        # Inline math patterns
        self.inline_patterns = []
        for start, end in INLINE_MATH_DELIMITERS:
            # Handle special cases for $ delimiters
            if start == r'\$' and end == r'\$':
                # Match $...$ but not $$...$$ (which is display math)
                pattern = r'(?<!\$)\$(?!\$)([^$\n]+?)\$(?!\$)'
            else:
                pattern = f'{start}(.*?){end}'
            self.inline_patterns.append(re.compile(pattern, re.DOTALL))
        
        # Block/display math patterns
        self.block_patterns = []
        for start, end in BLOCK_MATH_DELIMITERS:
            pattern = f'{start}(.*?){end}'
            self.block_patterns.append(re.compile(pattern, re.DOTALL))
        
        # Pattern for detecting potential math content (heuristic)
        self.math_indicators = re.compile(
            r'[\\α-ωΑ-Ω]|'  # Backslash or Greek letters
            r'\b(?:sin|cos|tan|log|ln|exp|lim|sum|int|sqrt|frac)\b|'  # Math functions
            r'[∫∑∏∇∂±≤≥≠≈≡∞]|'  # Math symbols
            r'\^[{(]|_[{(]|'  # Superscripts/subscripts
            r'\d+[a-zA-Z]|[a-zA-Z]\d+'  # Variable-number combinations
        )
    
    def extract_expressions(
        self,
        text: str,
        include_inline: bool = True,
        include_block: bool = True,
        deduplicate: bool = DEFAULT_DEDUPLICATE
    ) -> List[Dict[str, Union[str, int, float]]]:
        """
        Extract mathematical expressions from text.
        
        Args:
            text: Input text to search for mathematical expressions
            include_inline: Whether to include inline math expressions
            include_block: Whether to include block/display math expressions
            deduplicate: Whether to remove duplicate expressions
            
        Returns:
            List of dictionaries containing extracted expressions with metadata:
            - "expression": Cleaned mathematical expression
            - "raw_expression": Original expression with delimiters
            - "type": "inline" or "block"
            - "position": Character position in text
            - "confidence": Confidence score (0.0 to 1.0)
        """
        expressions = []
        
        if include_inline:
            expressions.extend(self._extract_inline_math(text))
        
        if include_block:
            expressions.extend(self._extract_block_math(text))
        
        # Apply basic filtering
        expressions = self._filter_expressions(expressions)
        
        # Deduplicate if requested
        if deduplicate:
            expressions = self._deduplicate_expressions(expressions)
        
        # Sort by position in text
        expressions.sort(key=lambda x: x["position"])
        
        self.logger.info(f"Extracted {len(expressions)} mathematical expressions")
        return expressions
    
    def _extract_inline_math(self, text: str) -> List[Dict[str, Union[str, int, float]]]:
        """Extract inline mathematical expressions."""
        expressions = []
        
        for pattern in self.inline_patterns:
            for match in pattern.finditer(text):
                raw_expr = match.group(0)
                expr_content = match.group(1)
                
                # Calculate confidence based on math indicators
                confidence = self._calculate_confidence(expr_content)
                
                expression_data = {
                    "expression": self._clean_expression(expr_content) if self.clean_formulas else expr_content,
                    "raw_expression": raw_expr,
                    "type": "inline",
                    "position": match.start(),
                    "confidence": confidence,
                }
                
                expressions.append(expression_data)
        
        return expressions
    
    def _extract_block_math(self, text: str) -> List[Dict[str, Union[str, int, float]]]:
        """Extract block/display mathematical expressions."""
        expressions = []
        
        for pattern in self.block_patterns:
            for match in pattern.finditer(text):
                raw_expr = match.group(0)
                expr_content = match.group(1)
                
                # Block math generally has higher confidence
                confidence = min(self._calculate_confidence(expr_content) + 0.2, 1.0)
                
                expression_data = {
                    "expression": self._clean_expression(expr_content) if self.clean_formulas else expr_content,
                    "raw_expression": raw_expr,
                    "type": "block",
                    "position": match.start(),
                    "confidence": confidence,
                }
                
                expressions.append(expression_data)
        
        return expressions
    
    def _calculate_confidence(self, expression: str) -> float:
        """
        Calculate confidence score for a mathematical expression.
        
        Args:
            expression: The mathematical expression to evaluate
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not expression or len(expression.strip()) < MIN_FORMULA_LENGTH:
            return 0.0
        
        # Base confidence
        confidence = 0.3
        
        # Boost for math indicators
        math_matches = len(self.math_indicators.findall(expression))
        confidence += min(math_matches * 0.1, 0.4)
        
        # Boost for LaTeX commands
        latex_commands = len(re.findall(r'\\[a-zA-Z]+', expression))
        confidence += min(latex_commands * 0.05, 0.2)
        
        # Boost for mathematical operators
        operators = len(re.findall(r'[+\-*/^_=<>]', expression))
        confidence += min(operators * 0.02, 0.1)
        
        # Penalty for very short expressions
        if len(expression.strip()) < 5:
            confidence *= 0.8
        
        # Penalty for expressions that are mostly numbers or letters
        if re.match(r'^[\d\s]+$', expression) or re.match(r'^[a-zA-Z\s]+$', expression):
            confidence *= 0.5
        
        return min(confidence, 1.0)
    
    def _clean_expression(self, expression: str) -> str:
        """
        Apply basic cleaning to a mathematical expression.
        
        Args:
            expression: Raw mathematical expression
            
        Returns:
            Cleaned expression
        """
        if not expression:
            return expression
        
        # Remove excessive whitespace
        expression = re.sub(r'\s+', ' ', expression.strip())
        
        # Remove LaTeX comments
        expression = re.sub(r'%.*?$', '', expression, flags=re.MULTILINE)
        
        # Remove common LaTeX formatting that doesn't affect math content
        formatting_commands = [
            r'\\text(bf|it|rm|sf|tt)\{([^}]*)\}',
            r'\\(bf|it|rm|sf|tt)\s+',
            r'\\label\{[^}]*\}',
            r'\\tag\{[^}]*\}',
        ]
        
        for cmd_pattern in formatting_commands:
            if r'\{([^}]*)\}' in cmd_pattern:
                expression = re.sub(cmd_pattern, r'\2', expression)
            else:
                expression = re.sub(cmd_pattern, '', expression)
        
        return expression.strip()
    
    def _filter_expressions(
        self,
        expressions: List[Dict[str, Union[str, int, float]]]
    ) -> List[Dict[str, Union[str, int, float]]]:
        """
        Filter expressions based on length and confidence criteria.
        
        Args:
            expressions: List of expression dictionaries
            
        Returns:
            Filtered list of expressions
        """
        filtered = []
        
        for expr in expressions:
            content = expr["expression"]
            
            # Length filters
            if len(content) < MIN_FORMULA_LENGTH or len(content) > MAX_FORMULA_LENGTH:
                continue
            
            # Confidence filter
            if expr["confidence"] < 0.1:
                continue
            
            # Skip expressions that are just numbers or single letters
            if re.match(r'^\d+$', content.strip()) or re.match(r'^[a-zA-Z]$', content.strip()):
                continue
            
            filtered.append(expr)
        
        return filtered
    
    def _deduplicate_expressions(
        self,
        expressions: List[Dict[str, Union[str, int, float]]]
    ) -> List[Dict[str, Union[str, int, float]]]:
        """
        Remove duplicate expressions, keeping the one with highest confidence.
        
        Args:
            expressions: List of expression dictionaries
            
        Returns:
            Deduplicated list of expressions
        """
        seen_expressions: Dict[str, Dict[str, Union[str, int, float]]] = {}
        
        for expr in expressions:
            content = expr["expression"].strip()
            
            if content not in seen_expressions or expr["confidence"] > seen_expressions[content]["confidence"]:
                seen_expressions[content] = expr
        
        return list(seen_expressions.values())
    
    def extract_by_environment(self, text: str, environment: str) -> List[str]:
        """
        Extract expressions from specific LaTeX environments.
        
        Args:
            text: Input text
            environment: LaTeX environment name (e.g., "equation", "align")
            
        Returns:
            List of expressions from the specified environment
        """
        pattern = rf'\\begin\{{{environment}\*?\}}(.*?)\\end\{{{environment}\*?\}}'
        matches = re.findall(pattern, text, re.DOTALL)
        
        expressions = []
        for match in matches:
            if self.clean_formulas:
                match = self._clean_expression(match)
            expressions.append(match)
        
        self.logger.debug(f"Extracted {len(expressions)} expressions from {environment} environments")
        return expressions
    
    def get_math_statistics(self, text: str) -> Dict[str, int]:
        """
        Get statistics about mathematical content in text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with statistics about mathematical content
        """
        expressions = self.extract_expressions(text)
        
        stats = {
            "total_expressions": len(expressions),
            "inline_expressions": len([e for e in expressions if e["type"] == "inline"]),
            "block_expressions": len([e for e in expressions if e["type"] == "block"]),
            "high_confidence": len([e for e in expressions if e["confidence"] > 0.8]),
            "medium_confidence": len([e for e in expressions if 0.5 <= e["confidence"] <= 0.8]),
            "low_confidence": len([e for e in expressions if e["confidence"] < 0.5]),
        }
        
        # Environment-specific counts
        common_environments = ["equation", "align", "gather", "multline"]
        for env in common_environments:
            env_exprs = self.extract_by_environment(text, env)
            stats[f"{env}_environments"] = len(env_exprs)
        
        return stats


# Convenience function for simple extraction
def extract_math_expressions(
    text: str,
    clean_formulas: bool = DEFAULT_CLEAN_FORMULAS,
    deduplicate: bool = DEFAULT_DEDUPLICATE
) -> List[str]:
    """
    Extract mathematical expressions from text (convenience function).
    
    Args:
        text: Input text to search
        clean_formulas: Whether to clean extracted formulas
        deduplicate: Whether to remove duplicates
        
    Returns:
        List of extracted mathematical expressions (strings only)
    """
    extractor = FormulaExtractor(clean_formulas=clean_formulas)
    expressions = extractor.extract_expressions(text, deduplicate=deduplicate)
    return [expr["expression"] for expr in expressions]


def find_math_patterns(text: str) -> Dict[str, List[Tuple[str, int]]]:
    """
    Find various mathematical patterns in text for analysis.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Dictionary mapping pattern types to lists of (match, position) tuples
    """
    patterns = {
        "greek_letters": r'[α-ωΑ-Ω]|\\(?:alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|omicron|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega)',
        "functions": r'\\?(?:sin|cos|tan|sec|csc|cot|sinh|cosh|tanh|log|ln|exp|sqrt|lim|sum|prod|int)',
        "operators": r'[+\-*/=<>]|\\(?:pm|mp|times|div|cdot|leq|geq|neq|approx|equiv)',
        "superscripts": r'\^[{(]?[^}\s)]*[})]?',
        "subscripts": r'_[{(]?[^}\s)]*[})]?',
        "fractions": r'\\(?:frac|dfrac|tfrac)\{[^{}]*\}\{[^{}]*\}',
        "integrals": r'\\int(?:_[^\\]*)?(?:\^[^\\]*)?',
        "summations": r'\\sum(?:_[^\\]*)?(?:\^[^\\]*)?',
    }
    
    results = {}
    for pattern_name, pattern in patterns.items():
        matches = []
        for match in re.finditer(pattern, text, re.IGNORECASE):
            matches.append((match.group(0), match.start()))
        results[pattern_name] = matches
    
    return results


if __name__ == "__main__":
    # Example usage and testing
    sample_text = """
    The quadratic formula is $x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$.
    
    For the integral:
    $$\\int_0^\\infty e^{-x^2} dx = \\frac{\\sqrt{\\pi}}{2}$$
    
    We can also write this as:
    \\begin{equation}
    \\lim_{n \\to \\infty} \\sum_{k=1}^n \\frac{1}{k^2} = \\frac{\\pi^2}{6}
    \\end{equation}
    """
    
    extractor = FormulaExtractor(clean_formulas=True)
    expressions = extractor.extract_expressions(sample_text)
    
    print(f"Found {len(expressions)} mathematical expressions:")
    for i, expr in enumerate(expressions, 1):
        print(f"{i}. {expr['type']}: {expr['expression'][:50]}... (confidence: {expr['confidence']:.2f})")
    
    print("\nMath statistics:")
    stats = extractor.get_math_statistics(sample_text)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nMath patterns:")
    patterns = find_math_patterns(sample_text)
    for pattern_type, matches in patterns.items():
        if matches:
            print(f"  {pattern_type}: {len(matches)} matches")
            for match, pos in matches[:3]:  # Show first 3 matches
                print(f"    '{match}' at position {pos}") 