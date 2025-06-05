"""
Hypothesis generation engine for mathematical conjectures.

This module generates new mathematical hypotheses by:
- Composing known formulas in novel ways
- Testing algebraic identities through symbolic transformation
- Creating variations and generalizations of existing equations
- Validating hypotheses using the Phase 3 validation engine
"""

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from collections import defaultdict
import itertools
import random

import sympy as sp
import numpy as np
from sympy import (
    sympify, symbols, diff, integrate, simplify, expand, factor,
    solve, limit, series, apart, together, cancel, trigsimp,
    sin, cos, tan, exp, log, sqrt, pi, E, I, oo
)
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol

# Import validation engine from Phase 3
from validation.formula_tester import FormulaValidator, ValidationConfig, ValidationResult


class HypothesisStatus(Enum):
    """Status of generated hypotheses."""
    GENERATED = "generated"
    VALIDATED = "validated"
    PROMISING = "promising"
    REJECTED = "rejected"
    ERROR = "error"


class HypothesisType(Enum):
    """Types of mathematical hypotheses."""
    ALGEBRAIC_IDENTITY = "algebraic_identity"
    FUNCTIONAL_EQUATION = "functional_equation"
    GENERALIZATION = "generalization"
    COMPOSITION = "composition"
    TRANSFORMATION = "transformation"
    LIMIT_CONJECTURE = "limit_conjecture"
    SERIES_EXPANSION = "series_expansion"


@dataclass
class Hypothesis:
    """Represents a mathematical hypothesis or conjecture."""
    hypothesis_id: str
    hypothesis_type: HypothesisType
    status: HypothesisStatus
    formula: str
    description: str
    confidence_score: float = 0.0
    source_formulas: List[str] = field(default_factory=list)
    transformation_lineage: List[str] = field(default_factory=list)
    validation_result: Optional[ValidationResult] = None
    mathematical_context: Dict[str, Any] = field(default_factory=dict)
    evidence: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class HypothesisGenerator:
    """
    Generates and validates mathematical hypotheses and conjectures.
    
    Creates new mathematical expressions by combining and transforming
    known formulas, then validates them using symbolic and numerical methods.
    """
    
    def __init__(self, max_hypotheses_per_type: int = 10,
                 validation_config: Optional[ValidationConfig] = None):
        """
        Initialize the hypothesis generator.
        
        Args:
            max_hypotheses_per_type: Maximum hypotheses to generate per type
            validation_config: Configuration for hypothesis validation
        """
        self.max_hypotheses_per_type = max_hypotheses_per_type
        self.validation_config = validation_config or ValidationConfig(
            num_random_tests=50,  # Fewer tests for hypothesis screening 
            tolerance=1e-8
        )
        self.validator = FormulaValidator(self.validation_config)
        self.logger = logging.getLogger(__name__)
        
        # Common mathematical constants and functions for generation
        self.constants = {
            'pi': sp.pi, 'e': sp.E, 'I': sp.I, 'oo': sp.oo,
            'sqrt2': sp.sqrt(2), 'sqrt3': sp.sqrt(3)
        }
        
        self.functions = {
            'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
            'exp': sp.exp, 'log': sp.log, 'sqrt': sp.sqrt,
            'sinh': sp.sinh, 'cosh': sp.cosh, 'tanh': sp.tanh
        }
    
    def generate_hypotheses(self, source_formulas: List[str],
                          formula_metadata: Optional[Dict] = None) -> List[Hypothesis]:
        """
        Generate mathematical hypotheses from source formulas.
        
        Args:
            source_formulas: List of validated formulas to use as sources
            formula_metadata: Optional metadata about source formulas
            
        Returns:
            List of generated hypotheses
        """
        self.logger.info(f"Generating hypotheses from {len(source_formulas)} source formulas")
        start_time = time.time()
        
        hypotheses = []
        
        # Parse source formulas
        parsed_formulas = self._parse_source_formulas(source_formulas)
        
        if len(parsed_formulas) < 2:
            self.logger.warning("Need at least 2 valid formulas for hypothesis generation")
            return hypotheses
        
        # Generate different types of hypotheses
        hypotheses.extend(self._generate_algebraic_identities(parsed_formulas))
        hypotheses.extend(self._generate_functional_equations(parsed_formulas))
        hypotheses.extend(self._generate_generalizations(parsed_formulas, formula_metadata))
        hypotheses.extend(self._generate_compositions(parsed_formulas))
        hypotheses.extend(self._generate_transformations(parsed_formulas))
        hypotheses.extend(self._generate_limit_conjectures(parsed_formulas))
        hypotheses.extend(self._generate_series_expansions(parsed_formulas))
        
        # Validate all hypotheses
        hypotheses = self._validate_hypotheses(hypotheses)
        
        # Rank by promise/confidence
        hypotheses = self._rank_hypotheses(hypotheses)
        
        self.logger.info(f"Generated {len(hypotheses)} hypotheses in "
                        f"{time.time() - start_time:.2f}s")
        
        return hypotheses
    
    def _parse_source_formulas(self, formulas: List[str]) -> List[Tuple[str, Expr]]:
        """Parse source formulas into SymPy expressions."""
        parsed = []
        
        for formula in formulas:
            try:
                expr = sympify(formula)
                parsed.append((formula, expr))
            except Exception as e:
                self.logger.debug(f"Failed to parse formula '{formula}': {e}")
        
        self.logger.info(f"Successfully parsed {len(parsed)}/{len(formulas)} formulas")
        return parsed
    
    def _generate_algebraic_identities(self, parsed_formulas: List[Tuple[str, Expr]]) -> List[Hypothesis]:
        """Generate hypotheses about algebraic identities."""
        hypotheses = []
        count = 0
        
        for i, (formula1, expr1) in enumerate(parsed_formulas):
            for formula2, expr2 in parsed_formulas[i+1:]:
                if count >= self.max_hypotheses_per_type:
                    break
                
                # Try to find relationships between expressions
                identity_hypotheses = self._find_algebraic_relationships(
                    formula1, expr1, formula2, expr2
                )
                hypotheses.extend(identity_hypotheses)
                count += len(identity_hypotheses)
        
        return hypotheses[:self.max_hypotheses_per_type]
    
    def _find_algebraic_relationships(self, formula1: str, expr1: Expr,
                                   formula2: str, expr2: Expr) -> List[Hypothesis]:
        """Find potential algebraic relationships between two expressions."""
        relationships = []
        
        try:
            # Check if expressions have same symbols
            if expr1.free_symbols & expr2.free_symbols:
                
                # Try addition/subtraction relationships
                sum_expr = simplify(expr1 + expr2)
                diff_expr = simplify(expr1 - expr2)
                
                # Look for simple relationships
                if sum_expr.is_constant():
                    hypothesis = Hypothesis(
                        hypothesis_id=f"identity_{hash((formula1, formula2))}",
                        hypothesis_type=HypothesisType.ALGEBRAIC_IDENTITY,
                        status=HypothesisStatus.GENERATED,
                        formula=f"({formula1}) + ({formula2})",
                        description=f"Conjecture: {formula1} + {formula2} = {sum_expr}",
                        source_formulas=[formula1, formula2],
                        transformation_lineage=["addition", "simplification"],
                        mathematical_context={
                            "symbols": list(str(s) for s in expr1.free_symbols | expr2.free_symbols),
                            "result": str(sum_expr)
                        }
                    )
                    relationships.append(hypothesis)
                
                # Try multiplicative relationships
                try:
                    ratio = simplify(expr1 / expr2)
                    if ratio.is_constant() and not ratio.has(oo):
                        hypothesis = Hypothesis(
                            hypothesis_id=f"ratio_{hash((formula1, formula2))}",
                            hypothesis_type=HypothesisType.ALGEBRAIC_IDENTITY,
                            status=HypothesisStatus.GENERATED,
                            formula=f"({formula1}) / ({formula2})",
                            description=f"Conjecture: ({formula1}) / ({formula2}) = {ratio}",
                            source_formulas=[formula1, formula2],
                            transformation_lineage=["division", "simplification"],
                            mathematical_context={
                                "symbols": list(str(s) for s in expr1.free_symbols | expr2.free_symbols),
                                "result": str(ratio)
                            }
                        )
                        relationships.append(hypothesis)
                except:
                    pass
        
        except Exception as e:
            self.logger.debug(f"Error finding relationships between {formula1} and {formula2}: {e}")
        
        return relationships
    
    def _generate_functional_equations(self, parsed_formulas: List[Tuple[str, Expr]]) -> List[Hypothesis]:
        """Generate hypotheses about functional equations."""
        hypotheses = []
        
        # Look for patterns that might suggest functional equations
        for formula, expr in parsed_formulas[:self.max_hypotheses_per_type]:
            try:
                symbols_list = list(expr.free_symbols)
                if len(symbols_list) >= 1:
                    x = symbols_list[0]
                    
                    # Generate functional equation variations
                    variations = [
                        (f"f(2*{x})", expr.subs(x, 2*x)),
                        (f"f({x}/2)", expr.subs(x, x/2)),
                        (f"f(-{x})", expr.subs(x, -x)),
                        (f"f({x} + 1)", expr.subs(x, x + 1))
                    ]
                    
                    for var_desc, var_expr in variations:
                        # Create functional equation hypothesis
                        hypothesis = Hypothesis(
                            hypothesis_id=f"func_eq_{hash((formula, var_desc))}",
                            hypothesis_type=HypothesisType.FUNCTIONAL_EQUATION,
                            status=HypothesisStatus.GENERATED,
                            formula=f"{var_desc} = {var_expr}",
                            description=f"Functional equation based on {formula}",
                            source_formulas=[formula],
                            transformation_lineage=["substitution", "functional_transformation"],
                            mathematical_context={
                                "original_formula": formula,
                                "transformation": var_desc,
                                "variable": str(x)
                            }
                        )
                        hypotheses.append(hypothesis)
                        
                        if len(hypotheses) >= self.max_hypotheses_per_type:
                            break
            
            except Exception as e:
                self.logger.debug(f"Error generating functional equations for {formula}: {e}")
        
        return hypotheses[:self.max_hypotheses_per_type]
    
    def _generate_generalizations(self, parsed_formulas: List[Tuple[str, Expr]],
                                formula_metadata: Optional[Dict] = None) -> List[Hypothesis]:
        """Generate generalizations of existing formulas."""
        hypotheses = []
        
        for formula, expr in parsed_formulas[:self.max_hypotheses_per_type]:
            try:
                generalizations = []
                
                # Try parameter generalization
                if expr.is_polynomial():
                    generalizations.extend(self._generalize_polynomial(formula, expr))
                
                # Try adding parameters to trigonometric functions
                if any(func in str(expr) for func in ['sin', 'cos', 'tan']):
                    generalizations.extend(self._generalize_trigonometric(formula, expr))
                
                # Try exponential generalizations
                if any(func in str(expr) for func in ['exp', 'log']):
                    generalizations.extend(self._generalize_exponential(formula, expr))
                
                hypotheses.extend(generalizations)
                
            except Exception as e:
                self.logger.debug(f"Error generating generalizations for {formula}: {e}")
        
        return hypotheses[:self.max_hypotheses_per_type]
    
    def _generalize_polynomial(self, formula: str, expr: Expr) -> List[Hypothesis]:
        """Generate polynomial generalizations."""
        generalizations = []
        
        try:
            # Add a parameter to the polynomial
            a = symbols('a')
            generalized = expr + a
            
            hypothesis = Hypothesis(
                hypothesis_id=f"poly_gen_{hash(formula)}",
                hypothesis_type=HypothesisType.GENERALIZATION,
                status=HypothesisStatus.GENERATED,
                formula=str(generalized),
                description=f"Polynomial generalization of {formula} with parameter a",
                source_formulas=[formula],
                transformation_lineage=["parameter_addition", "polynomial_extension"],
                mathematical_context={
                    "original": formula,
                    "parameter": "a",
                    "type": "polynomial"
                }
            )
            generalizations.append(hypothesis)
            
        except Exception:
            pass
        
        return generalizations
    
    def _generalize_trigonometric(self, formula: str, expr: Expr) -> List[Hypothesis]:
        """Generate trigonometric generalizations."""
        generalizations = []
        
        try:
            # Add phase and amplitude parameters
            symbols_list = list(expr.free_symbols)
            if symbols_list:
                x = symbols_list[0]
                A, phi = symbols('A phi')
                
                # Replace sin(x) with A*sin(x + phi), etc.
                generalized = expr.replace(sp.sin, lambda arg: A * sp.sin(arg + phi))
                generalized = generalized.replace(sp.cos, lambda arg: A * sp.cos(arg + phi))
                
                if generalized != expr:
                    hypothesis = Hypothesis(
                        hypothesis_id=f"trig_gen_{hash(formula)}",
                        hypothesis_type=HypothesisType.GENERALIZATION,
                        status=HypothesisStatus.GENERATED,
                        formula=str(generalized),
                        description=f"Trigonometric generalization with amplitude A and phase φ",
                        source_formulas=[formula],
                        transformation_lineage=["amplitude_scaling", "phase_shift"],
                        mathematical_context={
                            "original": formula,
                            "parameters": ["A", "phi"],
                            "type": "trigonometric"
                        }
                    )
                    generalizations.append(hypothesis)
                    
        except Exception:
            pass
        
        return generalizations
    
    def _generalize_exponential(self, formula: str, expr: Expr) -> List[Hypothesis]:
        """Generate exponential generalizations."""
        generalizations = []
        
        try:
            # Add scaling parameter to exponential
            symbols_list = list(expr.free_symbols)
            if symbols_list:
                x = symbols_list[0]
                k = symbols('k')
                
                # Replace exp(x) with exp(k*x)
                generalized = expr.replace(sp.exp, lambda arg: sp.exp(k * arg))
                
                if generalized != expr:
                    hypothesis = Hypothesis(
                        hypothesis_id=f"exp_gen_{hash(formula)}",
                        hypothesis_type=HypothesisType.GENERALIZATION,
                        status=HypothesisStatus.GENERATED,
                        formula=str(generalized),
                        description=f"Exponential generalization with scaling parameter k",
                        source_formulas=[formula],
                        transformation_lineage=["parameter_scaling", "exponential_extension"],
                        mathematical_context={
                            "original": formula,
                            "parameter": "k",
                            "type": "exponential"
                        }
                    )
                    generalizations.append(hypothesis)
                    
        except Exception:
            pass
        
        return generalizations
    
    def _generate_compositions(self, parsed_formulas: List[Tuple[str, Expr]]) -> List[Hypothesis]:
        """Generate hypotheses by composing formulas."""
        hypotheses = []
        count = 0
        
        for i, (formula1, expr1) in enumerate(parsed_formulas):
            for formula2, expr2 in parsed_formulas[i+1:]:
                if count >= self.max_hypotheses_per_type:
                    break
                
                compositions = self._compose_expressions(formula1, expr1, formula2, expr2)
                hypotheses.extend(compositions)
                count += len(compositions)
        
        return hypotheses[:self.max_hypotheses_per_type]
    
    def _compose_expressions(self, formula1: str, expr1: Expr,
                           formula2: str, expr2: Expr) -> List[Hypothesis]:
        """Compose two expressions in various ways."""
        compositions = []
        
        try:
            # Function composition if possible
            symbols1 = list(expr1.free_symbols)
            symbols2 = list(expr2.free_symbols)
            
            if symbols1 and symbols2:
                x = symbols1[0]
                
                # Substitute expr2 for x in expr1
                try:
                    composed = expr1.subs(x, expr2)
                    hypothesis = Hypothesis(
                        hypothesis_id=f"comp_{hash((formula1, formula2))}",
                        hypothesis_type=HypothesisType.COMPOSITION,
                        status=HypothesisStatus.GENERATED,
                        formula=str(composed),
                        description=f"Composition: f(g(x)) where f = {formula1}, g = {formula2}",
                        source_formulas=[formula1, formula2],
                        transformation_lineage=["function_composition"],
                        mathematical_context={
                            "outer_function": formula1,
                            "inner_function": formula2,
                            "composition_variable": str(x)
                        }
                    )
                    compositions.append(hypothesis)
                except:
                    pass
                
                # Try other combinations
                product = simplify(expr1 * expr2)
                if not product.has(oo):
                    hypothesis = Hypothesis(
                        hypothesis_id=f"prod_{hash((formula1, formula2))}",
                        hypothesis_type=HypothesisType.COMPOSITION,
                        status=HypothesisStatus.GENERATED,
                        formula=str(product),
                        description=f"Product composition: ({formula1}) * ({formula2})",
                        source_formulas=[formula1, formula2],
                        transformation_lineage=["multiplication", "simplification"],
                        mathematical_context={
                            "operation": "product",
                            "operands": [formula1, formula2]
                        }
                    )
                    compositions.append(hypothesis)
        
        except Exception as e:
            self.logger.debug(f"Error composing {formula1} and {formula2}: {e}")
        
        return compositions
    
    def _generate_transformations(self, parsed_formulas: List[Tuple[str, Expr]]) -> List[Hypothesis]:
        """Generate hypotheses through symbolic transformations."""
        hypotheses = []
        
        for formula, expr in parsed_formulas[:self.max_hypotheses_per_type]:
            try:
                transformations = []
                
                # Calculus transformations
                symbols_list = list(expr.free_symbols)
                if symbols_list:
                    x = symbols_list[0]
                    
                    # Derivative hypothesis
                    try:
                        derivative = diff(expr, x)
                        transformations.append(("derivative", derivative, f"d/d{x}[{formula}]"))
                    except:
                        pass
                    
                    # Integral hypothesis (simple cases)
                    try:
                        if len(str(expr)) < 50:  # Avoid complex integrals
                            integral = integrate(expr, x)
                            transformations.append(("integral", integral, f"∫ {formula} d{x}"))
                    except:
                        pass
                
                # Algebraic transformations
                try:
                    expanded = expand(expr)
                    if expanded != expr:
                        transformations.append(("expansion", expanded, f"expand({formula})"))
                except:
                    pass
                
                try:
                    factored = factor(expr)
                    if factored != expr:
                        transformations.append(("factorization", factored, f"factor({formula})"))
                except:
                    pass
                
                # Create hypotheses from transformations
                for trans_type, trans_expr, trans_desc in transformations:
                    hypothesis = Hypothesis(
                        hypothesis_id=f"trans_{trans_type}_{hash(formula)}",
                        hypothesis_type=HypothesisType.TRANSFORMATION,
                        status=HypothesisStatus.GENERATED,
                        formula=str(trans_expr),
                        description=f"Transformation: {trans_desc}",
                        source_formulas=[formula],
                        transformation_lineage=[trans_type],
                        mathematical_context={
                            "original": formula,
                            "transformation_type": trans_type,
                            "operation": trans_desc
                        }
                    )
                    hypotheses.append(hypothesis)
                    
                    if len(hypotheses) >= self.max_hypotheses_per_type:
                        break
                        
            except Exception as e:
                self.logger.debug(f"Error generating transformations for {formula}: {e}")
        
        return hypotheses[:self.max_hypotheses_per_type]
    
    def _generate_limit_conjectures(self, parsed_formulas: List[Tuple[str, Expr]]) -> List[Hypothesis]:
        """Generate limit-based conjectures."""
        hypotheses = []
        
        for formula, expr in parsed_formulas[:self.max_hypotheses_per_type//2]:
            try:
                symbols_list = list(expr.free_symbols)
                if symbols_list:
                    x = symbols_list[0]
                    
                    # Generate limit conjectures
                    limit_points = [0, 1, -1, oo, -oo]
                    
                    for point in limit_points:
                        try:
                            limit_result = limit(expr, x, point)
                            if limit_result is not None and not limit_result.has(sp.Limit):
                                hypothesis = Hypothesis(
                                    hypothesis_id=f"limit_{point}_{hash(formula)}",
                                    hypothesis_type=HypothesisType.LIMIT_CONJECTURE,
                                    status=HypothesisStatus.GENERATED,
                                    formula=f"limit({formula}, {x}, {point}) = {limit_result}",
                                    description=f"Limit of {formula} as {x} approaches {point}",
                                    source_formulas=[formula],
                                    transformation_lineage=["limit_evaluation"],
                                    mathematical_context={
                                        "original": formula,
                                        "variable": str(x),
                                        "limit_point": str(point),
                                        "result": str(limit_result)
                                    }
                                )
                                hypotheses.append(hypothesis)
                        except:
                            pass
                        
                        if len(hypotheses) >= self.max_hypotheses_per_type:
                            break
                            
            except Exception as e:
                self.logger.debug(f"Error generating limits for {formula}: {e}")
        
        return hypotheses[:self.max_hypotheses_per_type]
    
    def _generate_series_expansions(self, parsed_formulas: List[Tuple[str, Expr]]) -> List[Hypothesis]:
        """Generate series expansion conjectures."""
        hypotheses = []
        
        for formula, expr in parsed_formulas[:self.max_hypotheses_per_type//2]:
            try:
                symbols_list = list(expr.free_symbols)
                if symbols_list:
                    x = symbols_list[0]
                    
                    # Generate Taylor series expansion around x=0
                    try:
                        series_expansion = series(expr, x, 0, n=4).removeO()  # First 4 terms
                        if series_expansion != expr:
                            hypothesis = Hypothesis(
                                hypothesis_id=f"series_{hash(formula)}",
                                hypothesis_type=HypothesisType.SERIES_EXPANSION,
                                status=HypothesisStatus.GENERATED,
                                formula=str(series_expansion),
                                description=f"Taylor series expansion of {formula} around {x}=0",
                                source_formulas=[formula],
                                transformation_lineage=["taylor_expansion"],
                                mathematical_context={
                                    "original": formula,
                                    "variable": str(x),
                                    "expansion_point": "0",
                                    "terms": 4
                                }
                            )
                            hypotheses.append(hypothesis)
                        
                    except Exception:
                        pass
                        
            except Exception as e:
                self.logger.debug(f"Error generating series for {formula}: {e}")
        
        return hypotheses[:self.max_hypotheses_per_type]
    
    def _validate_hypotheses(self, hypotheses: List[Hypothesis]) -> List[Hypothesis]:
        """Validate generated hypotheses using the Phase 3 validation engine."""
        self.logger.info(f"Validating {len(hypotheses)} hypotheses")
        
        validated_hypotheses = []
        
        for hypothesis in hypotheses:
            try:
                # Validate the hypothesis formula
                validation_result = self.validator.validate_formula(hypothesis.formula)
                hypothesis.validation_result = validation_result
                
                # Update hypothesis status based on validation
                if validation_result.status.value == "PASS":
                    hypothesis.status = HypothesisStatus.VALIDATED
                    hypothesis.confidence_score = validation_result.confidence_score
                elif validation_result.status.value == "PARTIAL":
                    hypothesis.status = HypothesisStatus.PROMISING
                    hypothesis.confidence_score = validation_result.confidence_score * 0.8
                else:
                    hypothesis.status = HypothesisStatus.REJECTED
                    hypothesis.confidence_score = 0.1
                
                # Add validation evidence
                hypothesis.evidence = {
                    "validation_status": validation_result.status.value,
                    "pass_rate": validation_result.pass_rate,
                    "total_tests": validation_result.total_tests,
                    "symbols_tested": list(validation_result.symbols_found),
                    "validation_time": validation_result.validation_time
                }
                
                validated_hypotheses.append(hypothesis)
                
            except Exception as e:
                self.logger.debug(f"Error validating hypothesis {hypothesis.hypothesis_id}: {e}")
                hypothesis.status = HypothesisStatus.ERROR
                hypothesis.confidence_score = 0.0
                validated_hypotheses.append(hypothesis)
        
        return validated_hypotheses
    
    def _rank_hypotheses(self, hypotheses: List[Hypothesis]) -> List[Hypothesis]:
        """Rank hypotheses by promise and mathematical interest."""
        
        for hypothesis in hypotheses:
            # Base score from validation confidence
            base_score = hypothesis.confidence_score
            
            # Bonus for certain types of hypotheses
            type_bonuses = {
                HypothesisType.ALGEBRAIC_IDENTITY: 0.2,
                HypothesisType.GENERALIZATION: 0.15,
                HypothesisType.FUNCTIONAL_EQUATION: 0.1,
                HypothesisType.COMPOSITION: 0.05,
                HypothesisType.TRANSFORMATION: 0.1,
                HypothesisType.LIMIT_CONJECTURE: 0.05,
                HypothesisType.SERIES_EXPANSION: 0.1
            }
            
            type_bonus = type_bonuses.get(hypothesis.hypothesis_type, 0.0)
            
            # Bonus for novel combinations (multiple source formulas)
            novelty_bonus = 0.1 if len(hypothesis.source_formulas) > 1 else 0.0
            
            # Calculate final score
            final_score = base_score + type_bonus + novelty_bonus
            hypothesis.confidence_score = min(final_score, 1.0)
        
        # Sort by confidence score (descending)
        return sorted(hypotheses, key=lambda h: h.confidence_score, reverse=True)
    
    def save_hypotheses(self, hypotheses: List[Hypothesis], output_path: Union[str, Path]) -> None:
        """
        Save generated hypotheses to a JSON file.
        
        Args:
            hypotheses: List of hypotheses to save  
            output_path: Path to output file
        """
        output_path = Path(output_path)
        
        # Convert hypotheses to serializable format
        hypotheses_data = {
            "generation_metadata": {
                "total_hypotheses": len(hypotheses),
                "status_distribution": {
                    status.value: sum(1 for h in hypotheses if h.status == status)
                    for status in HypothesisStatus
                },
                "type_distribution": {
                    htype.value: sum(1 for h in hypotheses if h.hypothesis_type == htype)
                    for htype in HypothesisType
                },
                "generation_time": time.time(),
                "validation_config": {
                    "num_tests": self.validation_config.num_random_tests,
                    "tolerance": self.validation_config.tolerance
                }
            },
            "hypotheses": [
                {
                    "hypothesis_id": h.hypothesis_id,
                    "hypothesis_type": h.hypothesis_type.value,
                    "status": h.status.value,
                    "formula": h.formula,
                    "description": h.description,
                    "confidence_score": h.confidence_score,
                    "source_formulas": h.source_formulas,
                    "transformation_lineage": h.transformation_lineage,
                    "mathematical_context": h.mathematical_context,
                    "evidence": h.evidence,
                    "metadata": h.metadata,
                    "validation_summary": {
                        "status": h.validation_result.status.value if h.validation_result else "not_validated",
                        "pass_rate": h.validation_result.pass_rate if h.validation_result else 0.0,
                        "confidence": h.validation_result.confidence_score if h.validation_result else 0.0
                    } if h.validation_result else None
                }
                for h in hypotheses
            ]
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(hypotheses_data, f, indent=2, default=str)
        
        self.logger.info(f"Saved {len(hypotheses)} hypotheses to {output_path}")
    
    def get_promising_hypotheses(self, hypotheses: List[Hypothesis], 
                               min_confidence: float = 0.7) -> List[Hypothesis]:
        """
        Filter hypotheses to get the most promising ones.
        
        Args:
            hypotheses: List of all hypotheses
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of promising hypotheses
        """
        promising = [
            h for h in hypotheses 
            if h.confidence_score >= min_confidence 
            and h.status in [HypothesisStatus.VALIDATED, HypothesisStatus.PROMISING]
        ]
        
        return promising 