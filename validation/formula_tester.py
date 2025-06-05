"""
Core formula validation logic for MathBot.

This module provides the FormulaValidator class for comprehensive testing
of mathematical formulas through symbolic analysis and numerical validation.
"""

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from enum import Enum
import traceback

import sympy as sp
import numpy as np
from sympy import Symbol, sympify, lambdify, simplify, expand, factor
from sympy.parsing.sympy_parser import parse_expr
from sympy.core.sympify import SympifyError


class ValidationStatus(Enum):
    """Enumeration of validation statuses."""
    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"
    PARTIAL = "PARTIAL"


class TestType(Enum):
    """Types of validation tests."""
    SYMBOLIC = "symbolic"
    NUMERICAL = "numerical"
    EDGE_CASE = "edge_case"
    ROUND_TRIP = "round_trip"
    DOMAIN = "domain"


@dataclass
class ValidationConfig:
    """Configuration for formula validation."""
    num_random_tests: int = 100
    random_seed: Optional[int] = 42
    test_range: Tuple[float, float] = (-10.0, 10.0)
    tolerance: float = 1e-10
    max_complexity: int = 1000
    timeout_seconds: int = 30
    enable_symbolic: bool = True
    enable_numerical: bool = True
    enable_edge_cases: bool = True
    enable_round_trip: bool = True
    edge_case_values: List[float] = field(default_factory=lambda: [0, 1, -1, 0.5, -0.5, np.pi, np.e])


@dataclass
class TestResult:
    """Result of a single validation test."""
    test_type: TestType
    passed: bool
    error_message: Optional[str] = None
    test_values: Optional[Dict[str, float]] = None
    expected: Optional[float] = None
    actual: Optional[float] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Complete validation result for a formula."""
    formula: str
    status: ValidationStatus
    confidence_score: float
    pass_rate: float
    total_tests: int
    passed_tests: int
    failed_tests: int
    error_tests: int
    test_results: List[TestResult] = field(default_factory=list)
    validation_time: float = 0.0
    symbols_found: Set[str] = field(default_factory=set)
    domain_constraints: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    error_summary: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class FormulaValidator:
    """
    Comprehensive validator for mathematical formulas.
    
    Provides symbolic validation, numerical testing, edge case analysis,
    and round-trip verification of mathematical expressions.
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        """
        Initialize the formula validator.
        
        Args:
            config: Validation configuration (uses defaults if None)
        """
        self.config = config or ValidationConfig()
        self.logger = logging.getLogger(__name__)
        
        # Set random seed for reproducibility
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)
    
    def validate_formula(self, formula: str, known_identity: Optional[str] = None) -> ValidationResult:
        """
        Perform comprehensive validation of a mathematical formula.
        
        Args:
            formula: Mathematical expression as string
            known_identity: Optional known equivalent form for validation
            
        Returns:
            Complete validation results
        """
        start_time = time.time()
        
        result = ValidationResult(
            formula=formula,
            status=ValidationStatus.ERROR,
            confidence_score=0.0,
            pass_rate=0.0,
            total_tests=0,
            passed_tests=0,
            failed_tests=0,
            error_tests=0
        )
        
        try:
            # Parse the formula
            expr = self._parse_formula(formula)
            if expr is None:
                result.error_summary = "Failed to parse formula"
                result.validation_time = time.time() - start_time
                return result
            
            # Extract symbols and analyze structure
            result.symbols_found = {str(s) for s in expr.free_symbols}
            
            # Run validation tests
            test_results = []
            
            if self.config.enable_symbolic:
                test_results.extend(self._run_symbolic_tests(expr, known_identity))
            
            if self.config.enable_numerical and result.symbols_found:
                test_results.extend(self._run_numerical_tests(expr))
            
            if self.config.enable_edge_cases and result.symbols_found:
                test_results.extend(self._run_edge_case_tests(expr))
            
            if self.config.enable_round_trip:
                test_results.extend(self._run_round_trip_tests(expr))
            
            # Compile results
            result.test_results = test_results
            result.total_tests = len(test_results)
            result.passed_tests = sum(1 for t in test_results if t.passed)
            result.failed_tests = sum(1 for t in test_results if not t.passed and t.error_message is None)
            result.error_tests = sum(1 for t in test_results if t.error_message is not None)
            
            # Calculate metrics
            if result.total_tests > 0:
                result.pass_rate = result.passed_tests / result.total_tests
                result.confidence_score = self._calculate_confidence_score(result)
                result.status = self._determine_status(result)
            
            result.validation_time = time.time() - start_time
            
            self.logger.info(f"Validated formula '{formula[:50]}...' - Status: {result.status}, "
                           f"Pass rate: {result.pass_rate:.2%}, Confidence: {result.confidence_score:.2f}")
            
        except Exception as e:
            result.error_summary = str(e)
            result.validation_time = time.time() - start_time
            self.logger.error(f"Validation failed for formula '{formula}': {e}")
            
        return result
    
    def _parse_formula(self, formula: str) -> Optional[sp.Expr]:
        """
        Parse a formula string into a SymPy expression.
        
        Args:
            formula: Mathematical expression as string
            
        Returns:
            Parsed SymPy expression or None if parsing fails
        """
        try:
            # Clean up common LaTeX patterns
            cleaned = self._clean_latex_formula(formula)
            
            # Parse with SymPy
            expr = parse_expr(cleaned, transformations='all')
            
            # Check complexity
            if len(str(expr)) > self.config.max_complexity:
                self.logger.warning(f"Formula too complex: {len(str(expr))} characters")
                return None
            
            return expr
            
        except (SympifyError, ValueError, TypeError) as e:
            self.logger.warning(f"Failed to parse formula '{formula}': {e}")
            return None
    
    def _clean_latex_formula(self, formula: str) -> str:
        """
        Clean LaTeX formatting from formula for SymPy parsing.
        
        Args:
            formula: Raw formula string
            
        Returns:
            Cleaned formula string
        """
        # Remove common LaTeX commands that SymPy doesn't handle
        cleaned = formula.replace('\\', '')
        
        # Replace common patterns
        replacements = {
            'cdot': '*',
            'times': '*',
            'div': '/',
            'frac': '',  # Will be handled by SymPy parser
            'sqrt': 'sqrt',
            'sin': 'sin',
            'cos': 'cos',
            'tan': 'tan',
            'log': 'log',
            'ln': 'log',  # SymPy uses log for natural log
            'exp': 'exp',
            'pi': 'pi',
            'e': 'E',
        }
        
        for latex, sympy_equiv in replacements.items():
            cleaned = cleaned.replace(latex, sympy_equiv)
        
        return cleaned
    
    def _run_symbolic_tests(self, expr: sp.Expr, known_identity: Optional[str] = None) -> List[TestResult]:
        """
        Run symbolic validation tests.
        
        Args:
            expr: SymPy expression to test
            known_identity: Known equivalent expression for comparison
            
        Returns:
            List of symbolic test results
        """
        results = []
        
        # Test 1: Check if expression is well-formed
        start_time = time.time()
        try:
            simplified = simplify(expr)
            results.append(TestResult(
                test_type=TestType.SYMBOLIC,
                passed=True,
                execution_time=time.time() - start_time,
                metadata={"test_name": "well_formed", "simplified": str(simplified)}
            ))
        except Exception as e:
            results.append(TestResult(
                test_type=TestType.SYMBOLIC,
                passed=False,
                error_message=str(e),
                execution_time=time.time() - start_time,
                metadata={"test_name": "well_formed"}
            ))
        
        # Test 2: Identity verification if provided
        if known_identity:
            start_time = time.time()
            try:
                identity_expr = parse_expr(self._clean_latex_formula(known_identity))
                difference = simplify(expr - identity_expr)
                is_equivalent = difference.equals(0)
                
                results.append(TestResult(
                    test_type=TestType.SYMBOLIC,
                    passed=is_equivalent,
                    execution_time=time.time() - start_time,
                    metadata={
                        "test_name": "identity_check",
                        "known_identity": known_identity,
                        "difference": str(difference)
                    }
                ))
            except Exception as e:
                results.append(TestResult(
                    test_type=TestType.SYMBOLIC,
                    passed=False,
                    error_message=str(e),
                    execution_time=time.time() - start_time,
                    metadata={"test_name": "identity_check"}
                ))
        
        return results
    
    def _run_numerical_tests(self, expr: sp.Expr) -> List[TestResult]:
        """
        Run numerical validation tests with random inputs.
        
        Args:
            expr: SymPy expression to test
            
        Returns:
            List of numerical test results
        """
        results = []
        
        try:
            # Convert to numerical function
            symbols = list(expr.free_symbols)
            if not symbols:
                return results
            
            # Create lambdified function
            func = lambdify(symbols, expr, 'numpy')
            
            # Generate random test values
            for i in range(self.config.num_random_tests):
                start_time = time.time()
                
                # Generate random values for each symbol
                test_values = {}
                for sym in symbols:
                    test_values[str(sym)] = random.uniform(*self.config.test_range)
                
                try:
                    # Evaluate function
                    args = [test_values[str(sym)] for sym in symbols]
                    result_val = func(*args)
                    
                    # Check for valid result
                    is_valid = (
                        np.isfinite(result_val) if np.isscalar(result_val) 
                        else np.all(np.isfinite(result_val))
                    )
                    
                    results.append(TestResult(
                        test_type=TestType.NUMERICAL,
                        passed=is_valid,
                        test_values=test_values.copy(),
                        actual=float(result_val) if np.isscalar(result_val) else None,
                        execution_time=time.time() - start_time,
                        metadata={"test_iteration": i, "result_type": type(result_val).__name__}
                    ))
                    
                except Exception as e:
                    results.append(TestResult(
                        test_type=TestType.NUMERICAL,
                        passed=False,
                        error_message=str(e),
                        test_values=test_values.copy(),
                        execution_time=time.time() - start_time,
                        metadata={"test_iteration": i}
                    ))
        
        except Exception as e:
            # If lambdify fails entirely
            results.append(TestResult(
                test_type=TestType.NUMERICAL,
                passed=False,
                error_message=f"Lambdify failed: {str(e)}",
                metadata={"error_type": "lambdify_failure"}
            ))
        
        return results
    
    def _run_edge_case_tests(self, expr: sp.Expr) -> List[TestResult]:
        """
        Test formula with edge case values.
        
        Args:
            expr: SymPy expression to test
            
        Returns:
            List of edge case test results
        """
        results = []
        symbols = list(expr.free_symbols)
        
        if not symbols:
            return results
        
        try:
            func = lambdify(symbols, expr, 'numpy')
            
            # Test each edge case value
            for edge_val in self.config.edge_case_values:
                start_time = time.time()
                
                # Test with all symbols set to edge value
                test_values = {str(sym): edge_val for sym in symbols}
                
                try:
                    args = [edge_val] * len(symbols)
                    result_val = func(*args)
                    
                    # Edge cases may produce inf/nan, which is acceptable
                    passed = not (np.isscalar(result_val) and np.isnan(result_val))
                    
                    results.append(TestResult(
                        test_type=TestType.EDGE_CASE,
                        passed=passed,
                        test_values=test_values.copy(),
                        actual=float(result_val) if np.isscalar(result_val) else None,
                        execution_time=time.time() - start_time,
                        metadata={"edge_value": edge_val}
                    ))
                    
                except Exception as e:
                    results.append(TestResult(
                        test_type=TestType.EDGE_CASE,
                        passed=False,
                        error_message=str(e),
                        test_values=test_values.copy(),
                        execution_time=time.time() - start_time,
                        metadata={"edge_value": edge_val}
                    ))
                    
        except Exception as e:
            results.append(TestResult(
                test_type=TestType.EDGE_CASE,
                passed=False,
                error_message=f"Edge case testing failed: {str(e)}",
                metadata={"error_type": "edge_case_failure"}
            ))
        
        return results
    
    def _run_round_trip_tests(self, expr: sp.Expr) -> List[TestResult]:
        """
        Test round-trip consistency (parse -> simplify -> parse).
        
        Args:
            expr: SymPy expression to test
            
        Returns:
            List of round-trip test results
        """
        results = []
        
        # Test 1: Simplification round-trip
        start_time = time.time()
        try:
            simplified = simplify(expr)
            resimplified = simplify(simplified)
            
            # Check if double simplification gives same result
            passed = simplified.equals(resimplified)
            
            results.append(TestResult(
                test_type=TestType.ROUND_TRIP,
                passed=passed,
                execution_time=time.time() - start_time,
                metadata={
                    "test_name": "simplification_consistency",
                    "original": str(expr),
                    "simplified": str(simplified),
                    "resimplified": str(resimplified)
                }
            ))
            
        except Exception as e:
            results.append(TestResult(
                test_type=TestType.ROUND_TRIP,
                passed=False,
                error_message=str(e),
                execution_time=time.time() - start_time,
                metadata={"test_name": "simplification_consistency"}
            ))
        
        # Test 2: Expansion/factorization round-trip (if applicable)
        start_time = time.time()
        try:
            expanded = expand(expr)
            if expanded != expr:  # Only test if expansion changed something
                contracted = factor(expanded)
                # Note: factor(expand(x)) might not equal x, so we test evaluation equivalence
                
                if expr.free_symbols:
                    # Test equivalence numerically at a few points
                    symbols = list(expr.free_symbols)
                    test_points = [random.uniform(*self.config.test_range) for _ in range(3)]
                    
                    func_original = lambdify(symbols, expr, 'numpy')
                    func_processed = lambdify(symbols, contracted, 'numpy')
                    
                    equivalent = True
                    for test_val in test_points:
                        args = [test_val] * len(symbols)
                        try:
                            orig_val = func_original(*args)
                            proc_val = func_processed(*args)
                            if not np.allclose(orig_val, proc_val, rtol=self.config.tolerance):
                                equivalent = False
                                break
                        except:
                            equivalent = False
                            break
                    
                    results.append(TestResult(
                        test_type=TestType.ROUND_TRIP,
                        passed=equivalent,
                        execution_time=time.time() - start_time,
                        metadata={
                            "test_name": "expand_factor_consistency",
                            "expanded": str(expanded),
                            "factored": str(contracted)
                        }
                    ))
                else:
                    # No free symbols, direct comparison
                    passed = expr.equals(contracted)
                    results.append(TestResult(
                        test_type=TestType.ROUND_TRIP,
                        passed=passed,
                        execution_time=time.time() - start_time,
                        metadata={
                            "test_name": "expand_factor_consistency",
                            "expanded": str(expanded),
                            "factored": str(contracted)
                        }
                    ))
                    
        except Exception as e:
            results.append(TestResult(
                test_type=TestType.ROUND_TRIP,
                passed=False,
                error_message=str(e),
                execution_time=time.time() - start_time,
                metadata={"test_name": "expand_factor_consistency"}
            ))
        
        return results
    
    def _calculate_confidence_score(self, result: ValidationResult) -> float:
        """
        Calculate confidence score based on test results.
        
        Args:
            result: Validation result to score
            
        Returns:
            Confidence score between 0 and 1
        """
        if result.total_tests == 0:
            return 0.0
        
        # Base score from pass rate
        base_score = result.pass_rate
        
        # Penalty for errors
        error_penalty = (result.error_tests / result.total_tests) * 0.5
        
        # Bonus for comprehensive testing
        test_type_bonus = 0.0
        test_types_present = set(tr.test_type for tr in result.test_results)
        if len(test_types_present) >= 3:  # Multiple test types
            test_type_bonus = 0.1
        
        # Bonus for high volume of successful numerical tests
        numerical_tests = [tr for tr in result.test_results if tr.test_type == TestType.NUMERICAL]
        if len(numerical_tests) >= 50 and result.pass_rate > 0.9:
            test_type_bonus += 0.1
        
        confidence = min(1.0, base_score - error_penalty + test_type_bonus)
        return max(0.0, confidence)
    
    def _determine_status(self, result: ValidationResult) -> ValidationStatus:
        """
        Determine overall validation status.
        
        Args:
            result: Validation result to analyze
            
        Returns:
            Overall validation status
        """
        if result.error_tests == result.total_tests:
            return ValidationStatus.ERROR
        
        if result.pass_rate >= 0.95:
            return ValidationStatus.PASS
        elif result.pass_rate >= 0.7:
            return ValidationStatus.PARTIAL
        else:
            return ValidationStatus.FAIL 