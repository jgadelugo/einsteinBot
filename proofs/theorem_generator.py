"""
Core theorem generation engine for MathBot.

This module converts validated hypotheses from Phase 4 into formal mathematical
theorems with proper classification, assumptions, and natural language descriptions.
"""

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Union
from pathlib import Path
import json
import uuid

import sympy as sp
from sympy import Symbol, sympify, Eq, simplify, expand, factor
from sympy.parsing.sympy_parser import parse_expr
from sympy.core.sympify import SympifyError

from validation.formula_tester import FormulaValidator, ValidationConfig


class TheoremType(Enum):
    """Classification of different theorem types."""
    ALGEBRAIC_IDENTITY = "algebraic_identity"
    TRIGONOMETRIC = "trigonometric"
    LOGARITHMIC = "logarithmic"
    EXPONENTIAL = "exponential"
    CALCULUS = "calculus"
    FUNCTIONAL_EQUATION = "functional_equation"
    LIMIT_CONJECTURE = "limit_conjecture"
    INEQUALITY = "inequality"
    EQUIVALENCE = "equivalence"
    EXISTENCE = "existence"
    GENERALIZATION = "generalization"
    COMPOSITION = "composition"
    TRANSFORMATION = "transformation"
    SERIES_EXPANSION = "series_expansion"


@dataclass
class SourceLineage:
    """Tracks the lineage of a theorem from its source hypothesis."""
    original_formula: str
    hypothesis_id: str
    confidence: float
    validation_score: float
    generation_method: str
    source_type: str = ""
    transformation_chain: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            'original_formula': self.original_formula,
            'hypothesis_id': self.hypothesis_id,
            'confidence': self.confidence,
            'validation_score': self.validation_score,
            'generation_method': self.generation_method,
            'source_type': self.source_type,
            'transformation_chain': self.transformation_chain
        }


@dataclass
class Theorem:
    """Formal mathematical theorem with metadata and validation."""
    id: str
    statement: str
    sympy_expression: sp.Expr
    theorem_type: TheoremType
    assumptions: List[str]
    source_lineage: SourceLineage
    natural_language: Optional[str] = None
    symbols: Set[str] = field(default_factory=set)
    mathematical_context: Dict[str, Any] = field(default_factory=dict)
    validation_evidence: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert theorem to JSON-serializable dictionary."""
        return {
            'id': self.id,
            'statement': self.statement,
            'sympy_expression': str(self.sympy_expression),
            'theorem_type': self.theorem_type.value,
            'assumptions': self.assumptions,
            'source_lineage': self.source_lineage.to_dict(),
            'natural_language': self.natural_language,
            'symbols': list(self.symbols),
            'mathematical_context': self.mathematical_context,
            'validation_evidence': self.validation_evidence,
            'metadata': self.metadata
        }
    
    def validate_preconditions(self) -> bool:
        """Check if theorem assumptions are consistent."""
        try:
            # Basic consistency check - symbols in statement should match assumptions
            statement_symbols = {str(s) for s in self.sympy_expression.free_symbols}
            
            # Check if all symbols are accounted for in assumptions or are constants
            known_constants = {'pi', 'e', 'I', 'oo', 'zoo', 'nan'}
            undefined_symbols = statement_symbols - self.symbols - known_constants
            
            return len(undefined_symbols) == 0
        except Exception:
            return False


class TheoremGenerator:
    """
    Converts validated hypotheses into formal mathematical theorems.
    
    This class takes hypotheses from Phase 4 and transforms them into
    properly classified theorems with formal statements, assumptions,
    and natural language descriptions.
    """
    
    def __init__(self, validation_engine: Optional[FormulaValidator] = None, 
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the theorem generator.
        
        Args:
            validation_engine: Formula validation engine for consistency checks
            config: Configuration dictionary for generation parameters
        """
        self.validation_engine = validation_engine or FormulaValidator()
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Generation statistics
        self.stats = {
            'theorems_generated': 0,
            'classification_accuracy': 0.0,
            'generation_time': 0.0,
            'validation_passes': 0
        }
    
    def generate_from_hypotheses(self, hypotheses: List[Dict[str, Any]]) -> List[Theorem]:
        """
        Main entry point: convert validated hypotheses to formal theorems.
        
        Args:
            hypotheses: List of hypothesis dictionaries from Phase 4
            
        Returns:
            List of generated theorems
        """
        start_time = time.time()
        self.logger.info(f"Generating theorems from {len(hypotheses)} hypotheses")
        
        theorems = []
        
        for hypothesis in hypotheses:
            try:
                theorem = self._convert_hypothesis_to_theorem(hypothesis)
                if theorem and theorem.validate_preconditions():
                    theorems.append(theorem)
                    self.stats['theorems_generated'] += 1
                    self.stats['validation_passes'] += 1
                else:
                    self.logger.warning(f"Failed to generate valid theorem from hypothesis {hypothesis.get('hypothesis_id', 'unknown')}")
                    
            except Exception as e:
                self.logger.error(f"Error converting hypothesis {hypothesis.get('hypothesis_id', 'unknown')}: {e}")
                continue
        
        self.stats['generation_time'] = time.time() - start_time
        self.logger.info(f"Generated {len(theorems)} theorems in {self.stats['generation_time']:.2f}s")
        
        return theorems
    
    def _convert_hypothesis_to_theorem(self, hypothesis: Dict[str, Any]) -> Optional[Theorem]:
        """
        Convert a single hypothesis to a formal theorem.
        
        Args:
            hypothesis: Hypothesis dictionary from Phase 4
            
        Returns:
            Formal theorem or None if conversion fails
        """
        try:
            # Extract key information from hypothesis
            formula = hypothesis.get('formula', '')
            hypothesis_id = hypothesis.get('hypothesis_id', '')
            hypothesis_type = hypothesis.get('hypothesis_type', '')
            confidence = hypothesis.get('confidence_score', 0.0)
            validation_evidence = hypothesis.get('evidence', {})
            
            # Parse the formula
            sympy_expr = self._parse_formula(formula)
            if sympy_expr is None:
                return None
            
            # Generate theorem ID
            theorem_id = f"THM_{str(uuid.uuid4().hex[:8]).upper()}"
            
            # Classify theorem type
            theorem_type = self._classify_theorem_type(formula, sympy_expr, hypothesis_type)
            
            # Generate formal statement
            formal_statement = self._generate_formal_statement(sympy_expr, theorem_type)
            
            # Extract symbols and generate assumptions
            symbols = {str(s) for s in sympy_expr.free_symbols}
            
            # Add function symbols for functional equations
            if 'f(' in formula or 'g(' in formula or 'h(' in formula:
                import re
                func_matches = re.findall(r'([fgh])\(', formula)
                symbols.update(func_matches)
            assumptions = self._generate_assumptions(symbols, theorem_type)
            
            # Create source lineage
            source_lineage = SourceLineage(
                original_formula=hypothesis.get('source_formulas', [formula])[0] if hypothesis.get('source_formulas') else formula,
                hypothesis_id=hypothesis_id,
                confidence=confidence,
                validation_score=validation_evidence.get('pass_rate', 0.0),
                generation_method='direct_conversion',
                source_type=hypothesis_type,
                transformation_chain=hypothesis.get('transformation_lineage', [])
            )
            
            # Create theorem
            theorem = Theorem(
                id=theorem_id,
                statement=formal_statement,
                sympy_expression=sympy_expr,
                theorem_type=theorem_type,
                assumptions=assumptions,
                source_lineage=source_lineage,
                symbols=symbols,
                mathematical_context=hypothesis.get('mathematical_context', {}),
                validation_evidence=validation_evidence,
                metadata=hypothesis.get('metadata', {})
            )
            
            # Generate natural language description
            theorem.natural_language = self._generate_natural_language(theorem)
            
            return theorem
            
        except Exception as e:
            self.logger.error(f"Failed to convert hypothesis to theorem: {e}")
            return None
    
    def _parse_formula(self, formula: str) -> Optional[sp.Expr]:
        """Parse formula string into SymPy expression."""
        try:
            # Clean formula
            cleaned = self._clean_formula(formula)
            
            # Handle special cases like functional equations
            if 'f(' in cleaned:
                return self._parse_functional_equation(cleaned)
            
            # Standard parsing
            return sympify(cleaned)
        except (SympifyError, Exception) as e:
            self.logger.warning(f"Failed to parse formula '{formula}': {e}")
            return None
    
    def _clean_formula(self, formula: str) -> str:
        """Clean and normalize formula string."""
        # Remove description prefixes
        if '=' in formula and ':' in formula.split('=')[0]:
            formula = formula.split(':', 1)[1].strip()
        
        # Handle common patterns
        formula = formula.replace('sqrt(2*pi)', 'sqrt(2)*sqrt(pi)')
        formula = formula.replace('exp(', 'exp(')
        
        return formula.strip()
    
    def _parse_functional_equation(self, formula: str) -> sp.Expr:
        """Parse functional equations specially."""
        # For expressions like "f(2*x) = 4*x**2 + 4*x + 1"
        if '=' in formula:
            left, right = formula.split('=', 1)
            left_expr = sympify(left.strip())
            right_expr = sympify(right.strip())
            return Eq(left_expr, right_expr)
        else:
            return sympify(formula)
    
    def _classify_theorem_type(self, statement: str, expression: sp.Expr, 
                              hint_type: str = "") -> TheoremType:
        """
        Automatically classify theorem based on mathematical content.
        
        Args:
            statement: Original statement string
            expression: SymPy expression
            hint_type: Hint from original hypothesis type
            
        Returns:
            Classified theorem type
        """
        # Use hint from original hypothesis if available
        type_mapping = {
            'algebraic_identity': TheoremType.ALGEBRAIC_IDENTITY,
            'trigonometric': TheoremType.TRIGONOMETRIC,
            'logarithmic': TheoremType.LOGARITHMIC,
            'exponential': TheoremType.EXPONENTIAL,
            'calculus': TheoremType.CALCULUS,
            'functional_equation': TheoremType.FUNCTIONAL_EQUATION,
            'generalization': TheoremType.GENERALIZATION,
            'composition': TheoremType.COMPOSITION,
            'transformation': TheoremType.TRANSFORMATION,
            'limit_conjecture': TheoremType.LIMIT_CONJECTURE,
            'series_expansion': TheoremType.SERIES_EXPANSION
        }
        
        if hint_type in type_mapping:
            return type_mapping[hint_type]
        
        # Content-based classification
        expr_str = str(expression).lower()
        statement_lower = statement.lower()
        
        # Check for trigonometric functions
        trig_functions = ['sin', 'cos', 'tan', 'sec', 'csc', 'cot', 'asin', 'acos', 'atan']
        if any(func in expr_str for func in trig_functions):
            return TheoremType.TRIGONOMETRIC
        
        # Check for logarithmic/exponential functions
        if any(func in expr_str for func in ['log', 'ln', 'exp']):
            if 'exp' in expr_str or 'e**' in expr_str:
                return TheoremType.EXPONENTIAL
            else:
                return TheoremType.LOGARITHMIC
        
        # Check for calculus operations
        calculus_keywords = ['diff', 'integrate', 'derivative', 'integral', 'limit']
        if any(keyword in statement_lower for keyword in calculus_keywords):
            return TheoremType.CALCULUS
        
        # Check for specific theorem types
        if isinstance(expression, sp.Eq):
            # Check if it's a functional equation
            if any('f(' in str(side) for side in [expression.lhs, expression.rhs]):
                return TheoremType.FUNCTIONAL_EQUATION
            else:
                return TheoremType.ALGEBRAIC_IDENTITY
        
        # Check for inequalities
        if any(op in statement for op in ['<', '>', '<=', '>=']):
            return TheoremType.INEQUALITY
        
        # Check for limits
        if 'limit' in statement_lower or 'lim' in statement_lower:
            return TheoremType.LIMIT_CONJECTURE
        
        # Default to algebraic identity
        return TheoremType.ALGEBRAIC_IDENTITY
    
    def _generate_formal_statement(self, expression: sp.Expr, theorem_type: TheoremType) -> str:
        """Generate formal mathematical statement."""
        try:
            # For equations, format as "For all variables, LHS = RHS"
            if isinstance(expression, sp.Eq):
                symbols = expression.free_symbols
                if symbols:
                    symbol_list = ', '.join(sorted(str(s) for s in symbols))
                    return f"∀{symbol_list} ∈ ℝ, {sp.latex(expression.lhs)} = {sp.latex(expression.rhs)}"
                else:
                    return f"{sp.latex(expression.lhs)} = {sp.latex(expression.rhs)}"
            else:
                # For other expressions, just format nicely
                symbols = expression.free_symbols
                if symbols:
                    symbol_list = ', '.join(sorted(str(s) for s in symbols))
                    return f"∀{symbol_list} ∈ ℝ, {sp.latex(expression)}"
                else:
                    return sp.latex(expression)
        except Exception:
            # Fallback to string representation
            return str(expression)
    
    def _generate_assumptions(self, symbols: Set[str], theorem_type: TheoremType) -> List[str]:
        """Generate mathematical assumptions for the theorem."""
        assumptions = []
        
        # Standard domain assumptions for real variables
        real_symbols = symbols - {'f', 'g', 'h'}  # Exclude function symbols
        if real_symbols:
            for symbol in sorted(real_symbols):
                assumptions.append(f"{symbol} ∈ ℝ")
        
        # Add type-specific assumptions
        if theorem_type == TheoremType.FUNCTIONAL_EQUATION:
            func_symbols = symbols & {'f', 'g', 'h'}
            for func in sorted(func_symbols):
                assumptions.append(f"{func}: ℝ → ℝ")
        
        return assumptions
    
    def _generate_natural_language(self, theorem: Theorem) -> str:
        """Convert formal theorem to natural language description."""
        try:
            theorem_type = theorem.theorem_type
            expr = theorem.sympy_expression
            
            if theorem_type == TheoremType.ALGEBRAIC_IDENTITY:
                if isinstance(expr, sp.Eq):
                    return f"For any real numbers {', '.join(sorted(theorem.symbols))}, {expr.lhs} equals {expr.rhs}"
                else:
                    return f"The expression {expr} represents an algebraic identity"
            
            elif theorem_type == TheoremType.TRIGONOMETRIC:
                return f"This trigonometric identity states that {expr}"
            
            elif theorem_type == TheoremType.LOGARITHMIC:
                return f"This logarithmic relationship shows that {expr}"
            
            elif theorem_type == TheoremType.EXPONENTIAL:
                return f"This exponential relationship demonstrates that {expr}"
            
            elif theorem_type == TheoremType.CALCULUS:
                return f"This calculus theorem establishes that {expr}"
            
            elif theorem_type == TheoremType.FUNCTIONAL_EQUATION:
                return f"There exists a functional relationship defined by {expr}"
            
            elif theorem_type == TheoremType.GENERALIZATION:
                return f"This theorem generalizes the mathematical relationship expressed by {expr}"
            
            elif theorem_type == TheoremType.COMPOSITION:
                return f"The composition of mathematical operations yields {expr}"
            
            elif theorem_type == TheoremType.TRANSFORMATION:
                return f"Under the given transformation, the expression becomes {expr}"
            
            else:
                return f"This mathematical relationship holds: {expr}"
                
        except Exception:
            return f"Mathematical theorem: {theorem.statement}"
    
    def save_theorems(self, theorems: List[Theorem], output_path: Union[str, Path]) -> None:
        """
        Save generated theorems to JSON file.
        
        Args:
            theorems: List of theorems to save
            output_path: Path to output file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare output data
        output_data = {
            'generation_metadata': {
                'total_theorems': len(theorems),
                'generation_time': self.stats['generation_time'],
                'validation_passes': self.stats['validation_passes'],
                'type_distribution': self._get_type_distribution(theorems),
                'generator_version': '1.0.0'
            },
            'theorems': [theorem.to_dict() for theorem in theorems]
        }
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        self.logger.info(f"Saved {len(theorems)} theorems to {output_path}")
    
    def _get_type_distribution(self, theorems: List[Theorem]) -> Dict[str, int]:
        """Get distribution of theorem types."""
        distribution = {}
        for theorem in theorems:
            theorem_type = theorem.theorem_type.value
            distribution[theorem_type] = distribution.get(theorem_type, 0) + 1
        return distribution
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get generation statistics."""
        return self.stats.copy() 