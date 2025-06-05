"""
Symbolic proof attempt engine for MathBot.

This module implements a sophisticated proof system that attempts to prove or disprove
mathematical theorems using multiple symbolic methods including SymPy manipulation,
algebraic techniques, and pattern recognition.
"""

import hashlib
import logging
import pickle
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import json

import sympy as sp
from sympy import (
    Symbol, Eq, simplify, expand, factor, collect, cancel, apart,
    trigsimp, powsimp, radsimp, together, solve, limit, diff, integrate,
    pi, E, I, oo, zoo, nan, Float, Rational
)
from sympy.core.sympify import SympifyError
from sympy.solvers import solve as sp_solve

from .theorem_generator import Theorem, TheoremType


class ProofMethod(Enum):
    """Enumeration of proof methods available in the engine."""
    SYMPY_DIRECT = "sympy_direct"
    SYMPY_SIMPLIFY = "sympy_simplify"
    ALGEBRAIC_MANIPULATION = "algebraic_manipulation"
    SUBSTITUTION = "substitution"
    SYMBOLIC_SOLVER = "symbolic_solver"
    PATTERN_MATCHING = "pattern_matching"
    NUMERICAL_VERIFICATION = "numerical_verification"


class ProofStatus(Enum):
    """Status of a proof attempt."""
    PROVED = "proved"
    DISPROVED = "disproved"
    FAILED = "failed"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"
    INCONCLUSIVE = "inconclusive"


class ProofTimeout(Exception):
    """Exception raised when proof attempt times out."""
    pass


@dataclass
class ProofStep:
    """Represents a single step in a mathematical proof."""
    step_number: int
    method: str
    from_expression: str
    to_expression: str
    justification: str
    transformation_rule: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert proof step to JSON-serializable dictionary."""
        return {
            'step_number': self.step_number,
            'method': self.method,
            'from_expression': self.from_expression,
            'to_expression': self.to_expression,
            'justification': self.justification,
            'transformation_rule': self.transformation_rule,
            'success': self.success,
            'error_message': self.error_message
        }


@dataclass
class ProofResult:
    """Complete result of a proof attempt."""
    theorem_id: str
    status: ProofStatus
    method: ProofMethod
    steps: List[ProofStep]
    execution_time: float
    confidence_score: float
    error_message: Optional[str] = None
    additional_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert proof result to JSON-serializable dictionary."""
        return {
            'theorem_id': self.theorem_id,
            'status': self.status.value,
            'method': self.method.value,
            'steps': [step.to_dict() for step in self.steps],
            'execution_time': self.execution_time,
            'confidence_score': self.confidence_score,
            'error_message': self.error_message,
            'additional_info': self.additional_info
        }
    
    def is_successful(self) -> bool:
        """Check if the proof was successful."""
        return self.status == ProofStatus.PROVED
    
    def get_step_count(self) -> int:
        """Get the number of proof steps."""
        return len(self.steps)


class ProofCache:
    """Caching system for proof results to improve performance."""
    
    def __init__(self, cache_dir: Union[str, Path] = "cache/proofs"):
        """Initialize the proof cache."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache: Dict[str, ProofResult] = {}
        self.max_memory_cache_size = 1000
        
    def get_cache_key(self, theorem: Theorem) -> str:
        """Generate a unique cache key for a theorem."""
        content = f"{theorem.statement}:{str(theorem.sympy_expression)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_cached_result(self, cache_key: str) -> Optional[ProofResult]:
        """Retrieve cached proof result."""
        # Check memory cache first
        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    result = pickle.load(f)
                
                # Store in memory cache
                if len(self.memory_cache) < self.max_memory_cache_size:
                    self.memory_cache[cache_key] = result
                
                return result
            except Exception:
                # If cache file is corrupted, remove it
                cache_file.unlink(missing_ok=True)
                
        return None
    
    def cache_result(self, cache_key: str, result: ProofResult) -> None:
        """Cache a proof result both in memory and on disk."""
        # Memory cache
        if len(self.memory_cache) < self.max_memory_cache_size:
            self.memory_cache[cache_key] = result
        
        # Disk cache
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            logging.warning(f"Failed to cache result to disk: {e}")
    
    def clear_cache(self) -> None:
        """Clear all cached results."""
        self.memory_cache.clear()
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink(missing_ok=True)


class ProofAttemptEngine:
    """Main engine for attempting mathematical proofs using symbolic methods."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the proof attempt engine."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Configuration parameters
        self.timeout_seconds = self.config.get('timeout_seconds', 300)  # 5 minutes
        self.max_proof_steps = self.config.get('max_proof_steps', 50)
        self.enable_caching = self.config.get('enable_caching', True)
        self.numerical_test_count = self.config.get('numerical_test_count', 10)
        
        # Initialize cache
        if self.enable_caching:
            cache_dir = self.config.get('cache_dir', 'cache/proofs')
            self.cache = ProofCache(cache_dir)
        else:
            self.cache = None
        
        # Statistics tracking
        self.stats = {
            'total_attempts': 0,
            'successful_proofs': 0,
            'failed_proofs': 0,
            'timeouts': 0,
            'cache_hits': 0,
            'total_time': 0.0,
            'method_success_rates': {}
        }
        
        # Proof method order (priority)
        self.proof_methods = [
            ProofMethod.SYMPY_DIRECT,
            ProofMethod.SYMPY_SIMPLIFY,
            ProofMethod.ALGEBRAIC_MANIPULATION,
            ProofMethod.SUBSTITUTION,
            ProofMethod.SYMBOLIC_SOLVER,
            ProofMethod.PATTERN_MATCHING,
            ProofMethod.NUMERICAL_VERIFICATION
        ]
    
    def attempt_proof(self, theorem: Theorem) -> ProofResult:
        """Main entry point: attempt to prove a theorem using available methods."""
        start_time = time.time()
        self.stats['total_attempts'] += 1
        
        # Check cache first
        if self.cache:
            cache_key = self.cache.get_cache_key(theorem)
            cached_result = self.cache.get_cached_result(cache_key)
            if cached_result:
                self.stats['cache_hits'] += 1
                self.logger.info(f"Using cached result for theorem {theorem.id}")
                return cached_result
        
        # Initialize result
        result = ProofResult(
            theorem_id=theorem.id,
            status=ProofStatus.UNKNOWN,
            method=ProofMethod.SYMPY_DIRECT,
            steps=[],
            execution_time=0.0,
            confidence_score=0.0
        )
        
        try:
            # Try each proof method in order
            for method in self.proof_methods:
                try:
                    method_result = self._try_proof_method(theorem, method)
                    
                    if method_result.status == ProofStatus.PROVED:
                        result = method_result
                        self.stats['successful_proofs'] += 1
                        break
                    elif method_result.status == ProofStatus.DISPROVED:
                        result = method_result
                        break
                    
                    # Keep the best result so far
                    if (method_result.confidence_score > result.confidence_score):
                        result = method_result
                        
                except Exception as e:
                    self.logger.warning(f"Error in method {method.value}: {e}")
                    continue
            
            # Update statistics
            if result.status != ProofStatus.PROVED:
                self.stats['failed_proofs'] += 1
            
            # Track method success rates
            method_key = result.method.value
            if method_key not in self.stats['method_success_rates']:
                self.stats['method_success_rates'][method_key] = {'attempts': 0, 'successes': 0}
            
            self.stats['method_success_rates'][method_key]['attempts'] += 1
            if result.status == ProofStatus.PROVED:
                self.stats['method_success_rates'][method_key]['successes'] += 1
                
        except Exception as e:
            self.logger.error(f"Critical error in proof attempt for {theorem.id}: {e}")
            result.status = ProofStatus.FAILED
            result.error_message = str(e)
        
        # Finalize result
        result.execution_time = time.time() - start_time
        self.stats['total_time'] += result.execution_time
        
        # Cache result
        if self.cache:
            self.cache.cache_result(cache_key, result)
        
        return result

    def _try_proof_method(self, theorem: Theorem, method: ProofMethod) -> ProofResult:
        """Try a specific proof method on a theorem."""
        if method == ProofMethod.SYMPY_DIRECT:
            return self._try_direct_symbolic(theorem)
        elif method == ProofMethod.SYMPY_SIMPLIFY:
            return self._try_simplify_method(theorem)
        elif method == ProofMethod.ALGEBRAIC_MANIPULATION:
            return self._try_algebraic_manipulation(theorem)
        elif method == ProofMethod.SUBSTITUTION:
            return self._try_substitution_method(theorem)
        elif method == ProofMethod.SYMBOLIC_SOLVER:
            return self._try_symbolic_solver(theorem)
        elif method == ProofMethod.PATTERN_MATCHING:
            return self._try_pattern_matching(theorem)
        elif method == ProofMethod.NUMERICAL_VERIFICATION:
            return self._try_numerical_verification(theorem)
        else:
            raise ValueError(f"Unknown proof method: {method}")
    
    def _try_direct_symbolic(self, theorem: Theorem) -> ProofResult:
        """Attempt direct symbolic proof using SymPy's built-in capabilities."""
        steps = []
        
        try:
            expr = theorem.sympy_expression
            
            # Handle equations
            if isinstance(expr, Eq):
                lhs, rhs = expr.lhs, expr.rhs
                
                steps.append(ProofStep(
                    step_number=1,
                    method="equation_analysis",
                    from_expression=str(expr),
                    to_expression=f"{lhs} == {rhs}",
                    justification="Analyzing equation components"
                ))
                
                # Check if LHS - RHS simplifies to 0
                diff = lhs - rhs
                simplified_diff = simplify(diff)
                
                steps.append(ProofStep(
                    step_number=2,
                    method="difference_simplification",
                    from_expression=str(diff),
                    to_expression=str(simplified_diff),
                    justification="Simplifying LHS - RHS"
                ))
                
                if simplified_diff == 0 or simplified_diff.equals(sp.S.Zero):
                    steps.append(ProofStep(
                        step_number=3,
                        method="identity_verification",
                        from_expression=str(simplified_diff),
                        to_expression="0",
                        justification="Difference equals zero, proving equality"
                    ))
                    
                    return ProofResult(
                        theorem_id=theorem.id,
                        status=ProofStatus.PROVED,
                        method=ProofMethod.SYMPY_DIRECT,
                        steps=steps,
                        execution_time=0.0,
                        confidence_score=1.0
                    )
            
            # If we reach here, direct method didn't prove/disprove
            return ProofResult(
                theorem_id=theorem.id,
                status=ProofStatus.INCONCLUSIVE,
                method=ProofMethod.SYMPY_DIRECT,
                steps=steps,
                execution_time=0.0,
                confidence_score=0.3
            )
            
        except Exception as e:
            return ProofResult(
                theorem_id=theorem.id,
                status=ProofStatus.FAILED,
                method=ProofMethod.SYMPY_DIRECT,
                steps=steps,
                execution_time=0.0,
                confidence_score=0.0,
                error_message=str(e)
            )
    
    def _try_simplify_method(self, theorem: Theorem) -> ProofResult:
        """Try comprehensive simplification approaches."""
        steps = []
        expr = theorem.sympy_expression
        
        try:
            if isinstance(expr, Eq):
                lhs, rhs = expr.lhs, expr.rhs
                
                # Try different simplification methods
                simplification_methods = [
                    ("simplify", simplify),
                    ("expand", expand),
                    ("factor", factor),
                ]
                
                for method_name, method_func in simplification_methods:
                    try:
                        simplified_lhs = method_func(lhs)
                        simplified_rhs = method_func(rhs)
                        
                        steps.append(ProofStep(
                            step_number=len(steps) + 1,
                            method=method_name,
                            from_expression=f"{lhs} = {rhs}",
                            to_expression=f"{simplified_lhs} = {simplified_rhs}",
                            justification=f"Applied {method_name} to both sides"
                        ))
                        
                        # Check if simplified forms are equal
                        if simplified_lhs.equals(simplified_rhs):
                            steps.append(ProofStep(
                                step_number=len(steps) + 1,
                                method="equality_verification",
                                from_expression=f"{simplified_lhs} = {simplified_rhs}",
                                to_expression="True",
                                justification="Simplified forms are equal"
                            ))
                            
                            return ProofResult(
                                theorem_id=theorem.id,
                                status=ProofStatus.PROVED,
                                method=ProofMethod.SYMPY_SIMPLIFY,
                                steps=steps,
                                execution_time=0.0,
                                confidence_score=0.9
                            )
                            
                    except Exception:
                        continue
            
            return ProofResult(
                theorem_id=theorem.id,
                status=ProofStatus.INCONCLUSIVE,
                method=ProofMethod.SYMPY_SIMPLIFY,
                steps=steps,
                execution_time=0.0,
                confidence_score=0.2
            )
            
        except Exception as e:
            return ProofResult(
                theorem_id=theorem.id,
                status=ProofStatus.FAILED,
                method=ProofMethod.SYMPY_SIMPLIFY,
                steps=steps,
                execution_time=0.0,
                confidence_score=0.0,
                error_message=str(e)
            )
    
    def _try_algebraic_manipulation(self, theorem: Theorem) -> ProofResult:
        """Try systematic algebraic manipulation techniques."""
        steps = []
        expr = theorem.sympy_expression
        
        try:
            if isinstance(expr, Eq):
                lhs, rhs = expr.lhs, expr.rhs
                
                # Strategy 1: Expand both sides
                expanded_lhs = expand(lhs)
                expanded_rhs = expand(rhs)
                
                steps.append(ProofStep(
                    step_number=1,
                    method="expand",
                    from_expression=f"{lhs} = {rhs}",
                    to_expression=f"{expanded_lhs} = {expanded_rhs}",
                    justification="Algebraic expansion of both sides"
                ))
                
                if expanded_lhs.equals(expanded_rhs):
                    return ProofResult(
                        theorem_id=theorem.id,
                        status=ProofStatus.PROVED,
                        method=ProofMethod.ALGEBRAIC_MANIPULATION,
                        steps=steps,
                        execution_time=0.0,
                        confidence_score=0.95
                    )
                
                # Strategy 2: Factor expanded forms
                try:
                    factored_lhs = factor(expanded_lhs)
                    factored_rhs = factor(expanded_rhs)
                    
                    steps.append(ProofStep(
                        step_number=2,
                        method="factor",
                        from_expression=f"{expanded_lhs} = {expanded_rhs}",
                        to_expression=f"{factored_lhs} = {factored_rhs}",
                        justification="Factorization of expanded forms"
                    ))
                    
                    if factored_lhs.equals(factored_rhs):
                        return ProofResult(
                            theorem_id=theorem.id,
                            status=ProofStatus.PROVED,
                            method=ProofMethod.ALGEBRAIC_MANIPULATION,
                            steps=steps,
                            execution_time=0.0,
                            confidence_score=0.95
                        )
                except Exception:
                    pass
            
            return ProofResult(
                theorem_id=theorem.id,
                status=ProofStatus.INCONCLUSIVE,
                method=ProofMethod.ALGEBRAIC_MANIPULATION,
                steps=steps,
                execution_time=0.0,
                confidence_score=0.3
            )
            
        except Exception as e:
            return ProofResult(
                theorem_id=theorem.id,
                status=ProofStatus.FAILED,
                method=ProofMethod.ALGEBRAIC_MANIPULATION,
                steps=steps,
                execution_time=0.0,
                confidence_score=0.0,
                error_message=str(e)
            )
    
    def _try_substitution_method(self, theorem: Theorem) -> ProofResult:
        """Try variable substitution and symbolic solving."""
        steps = []
        expr = theorem.sympy_expression
        
        try:
            variables = list(expr.free_symbols)
            if not variables:
                return ProofResult(
                    theorem_id=theorem.id,
                    status=ProofStatus.INCONCLUSIVE,
                    method=ProofMethod.SUBSTITUTION,
                    steps=[],
                    execution_time=0.0,
                    confidence_score=0.1,
                    error_message="No variables for substitution"
                )
            
            # Test with specific values
            test_values = [0, 1, -1, 2, -2]
            
            if isinstance(expr, Eq):
                lhs, rhs = expr.lhs, expr.rhs
                
                for val in test_values:
                    try:
                        for var in variables:
                            substituted_lhs = lhs.subs(var, val)
                            substituted_rhs = rhs.subs(var, val)
                            
                            steps.append(ProofStep(
                                step_number=len(steps) + 1,
                                method="substitution",
                                from_expression=f"{lhs} = {rhs}",
                                to_expression=f"{substituted_lhs} = {substituted_rhs}",
                                justification=f"Substituting {var} = {val}"
                            ))
                            
                            # Evaluate numerical result
                            try:
                                lhs_val = float(substituted_lhs.evalf())
                                rhs_val = float(substituted_rhs.evalf())
                                
                                if abs(lhs_val - rhs_val) > 1e-10:
                                    # Found a counterexample
                                    return ProofResult(
                                        theorem_id=theorem.id,
                                        status=ProofStatus.DISPROVED,
                                        method=ProofMethod.SUBSTITUTION,
                                        steps=steps,
                                        execution_time=0.0,
                                        confidence_score=0.8,
                                        additional_info={'counterexample': {str(var): val}}
                                    )
                                    
                            except (ValueError, TypeError, OverflowError):
                                continue
                                
                    except Exception:
                        continue
            
            # If all substitutions passed, it's evidence (but not proof)
            if len(steps) > 0:
                return ProofResult(
                    theorem_id=theorem.id,
                    status=ProofStatus.INCONCLUSIVE,
                    method=ProofMethod.SUBSTITUTION,
                    steps=steps,
                    execution_time=0.0,
                    confidence_score=0.6,
                    additional_info={'test_values_passed': len(steps)}
                )
            
            return ProofResult(
                theorem_id=theorem.id,
                status=ProofStatus.FAILED,
                method=ProofMethod.SUBSTITUTION,
                steps=steps,
                execution_time=0.0,
                confidence_score=0.0
            )
            
        except Exception as e:
            return ProofResult(
                theorem_id=theorem.id,
                status=ProofStatus.FAILED,
                method=ProofMethod.SUBSTITUTION,
                steps=steps,
                execution_time=0.0,
                confidence_score=0.0,
                error_message=str(e)
            )
    
    def _try_symbolic_solver(self, theorem: Theorem) -> ProofResult:
        """Try SymPy's equation solving capabilities."""
        steps = []
        expr = theorem.sympy_expression
        
        try:
            if isinstance(expr, Eq):
                lhs, rhs = expr.lhs, expr.rhs
                
                # Move everything to one side
                equation = lhs - rhs
                steps.append(ProofStep(
                    step_number=1,
                    method="equation_rearrangement",
                    from_expression=f"{lhs} = {rhs}",
                    to_expression=f"{equation} = 0",
                    justification="Rearranging equation to standard form"
                ))
                
                # Check if equation simplifies to 0 = 0
                simplified_eq = simplify(equation)
                if simplified_eq == 0:
                    steps.append(ProofStep(
                        step_number=len(steps) + 1,
                        method="identity_verification",
                        from_expression=str(equation),
                        to_expression="0",
                        justification="Equation simplifies to identity"
                    ))
                    
                    return ProofResult(
                        theorem_id=theorem.id,
                        status=ProofStatus.PROVED,
                        method=ProofMethod.SYMBOLIC_SOLVER,
                        steps=steps,
                        execution_time=0.0,
                        confidence_score=1.0
                    )
            
            return ProofResult(
                theorem_id=theorem.id,
                status=ProofStatus.INCONCLUSIVE,
                method=ProofMethod.SYMBOLIC_SOLVER,
                steps=steps,
                execution_time=0.0,
                confidence_score=0.4
            )
            
        except Exception as e:
            return ProofResult(
                theorem_id=theorem.id,
                status=ProofStatus.FAILED,
                method=ProofMethod.SYMBOLIC_SOLVER,
                steps=steps,
                execution_time=0.0,
                confidence_score=0.0,
                error_message=str(e)
            )
    
    def _try_pattern_matching(self, theorem: Theorem) -> ProofResult:
        """Try pattern matching for common mathematical forms."""
        steps = []
        expr = theorem.sympy_expression
        
        try:
            if isinstance(expr, Eq):
                lhs, rhs = expr.lhs, expr.rhs
                
                # For perfect square patterns like (x+1)^2 = x^2 + 2x + 1
                if str(lhs).find("**2") != -1:
                    expanded_lhs = expand(lhs)
                    if expanded_lhs.equals(rhs):
                        steps.append(ProofStep(
                            step_number=1,
                            method="pattern_recognition",
                            from_expression=str(expr),
                            to_expression="Perfect square expansion",
                            justification="Recognized perfect square pattern"
                        ))
                        
                        return ProofResult(
                            theorem_id=theorem.id,
                            status=ProofStatus.PROVED,
                            method=ProofMethod.PATTERN_MATCHING,
                            steps=steps,
                            execution_time=0.0,
                            confidence_score=0.9
                        )
            
            return ProofResult(
                theorem_id=theorem.id,
                status=ProofStatus.INCONCLUSIVE,
                method=ProofMethod.PATTERN_MATCHING,
                steps=steps,
                execution_time=0.0,
                confidence_score=0.3
            )
            
        except Exception as e:
            return ProofResult(
                theorem_id=theorem.id,
                status=ProofStatus.FAILED,
                method=ProofMethod.PATTERN_MATCHING,
                steps=steps,
                execution_time=0.0,
                confidence_score=0.0,
                error_message=str(e)
            )
    
    def _try_numerical_verification(self, theorem: Theorem) -> ProofResult:
        """Try numerical verification as a last resort."""
        steps = []
        expr = theorem.sympy_expression
        
        try:
            if isinstance(expr, Eq):
                lhs, rhs = expr.lhs, expr.rhs
                variables = list(expr.free_symbols)
                
                if not variables:
                    # No variables, just evaluate
                    try:
                        lhs_val = float(lhs.evalf())
                        rhs_val = float(rhs.evalf())
                        
                        if abs(lhs_val - rhs_val) < 1e-12:
                            return ProofResult(
                                theorem_id=theorem.id,
                                status=ProofStatus.PROVED,
                                method=ProofMethod.NUMERICAL_VERIFICATION,
                                steps=[ProofStep(
                                    step_number=1,
                                    method="numerical_evaluation",
                                    from_expression=f"{lhs} = {rhs}",
                                    to_expression=f"{lhs_val} ≈ {rhs_val}",
                                    justification="Numerical evaluation confirms equality"
                                )],
                                execution_time=0.0,
                                confidence_score=1.0
                            )
                    except Exception:
                        pass
                
                # Test with random values
                import random
                random.seed(42)  # Reproducible results
                
                test_count = min(self.numerical_test_count, 20)
                passed_tests = 0
                
                for i in range(test_count):
                    test_values = {}
                    for var in variables:
                        test_values[var] = random.uniform(-10, 10)
                    
                    try:
                        lhs_val = float(lhs.subs(test_values).evalf())
                        rhs_val = float(rhs.subs(test_values).evalf())
                        
                        if abs(lhs_val - rhs_val) < 1e-10:
                            passed_tests += 1
                        else:
                            # Found counterexample
                            return ProofResult(
                                theorem_id=theorem.id,
                                status=ProofStatus.DISPROVED,
                                method=ProofMethod.NUMERICAL_VERIFICATION,
                                steps=[ProofStep(
                                    step_number=1,
                                    method="counterexample",
                                    from_expression=str(test_values),
                                    to_expression=f"{lhs_val} ≠ {rhs_val}",
                                    justification="Found numerical counterexample"
                                )],
                                execution_time=0.0,
                                confidence_score=0.9,
                                additional_info={'counterexample': test_values}
                            )
                            
                    except Exception:
                        continue
                
                # All tests passed
                confidence = min(0.8, passed_tests / test_count)
                return ProofResult(
                    theorem_id=theorem.id,
                    status=ProofStatus.INCONCLUSIVE,
                    method=ProofMethod.NUMERICAL_VERIFICATION,
                    steps=[ProofStep(
                        step_number=1,
                        method="numerical_testing",
                        from_expression=f"Tested {test_count} random values",
                        to_expression=f"{passed_tests}/{test_count} tests passed",
                        justification="Numerical evidence supports theorem"
                    )],
                    execution_time=0.0,
                    confidence_score=confidence,
                    additional_info={'tests_passed': passed_tests, 'total_tests': test_count}
                )
            
            return ProofResult(
                theorem_id=theorem.id,
                status=ProofStatus.FAILED,
                method=ProofMethod.NUMERICAL_VERIFICATION,
                steps=steps,
                execution_time=0.0,
                confidence_score=0.0
            )
            
        except Exception as e:
            return ProofResult(
                theorem_id=theorem.id,
                status=ProofStatus.FAILED,
                method=ProofMethod.NUMERICAL_VERIFICATION,
                steps=steps,
                execution_time=0.0,
                confidence_score=0.0,
                error_message=str(e)
            )

    def batch_prove_theorems(self, theorems: List[Theorem]) -> List[ProofResult]:
        """Prove multiple theorems with progress tracking."""
        results = []
        total = len(theorems)
        
        self.logger.info(f"Starting batch proof of {total} theorems")
        
        for i, theorem in enumerate(theorems, 1):
            self.logger.info(f"Proving theorem {i}/{total}: {theorem.id}")
            
            try:
                result = self.attempt_proof(theorem)
                results.append(result)
                
                # Progress reporting
                if i % 5 == 0 or i == total:
                    success_rate = sum(1 for r in results if r.is_successful()) / len(results)
                    self.logger.info(f"Progress {i}/{total}, Success rate: {success_rate:.1%}")
                    
            except Exception as e:
                self.logger.error(f"Failed to prove theorem {theorem.id}: {e}")
                results.append(ProofResult(
                    theorem_id=theorem.id,
                    status=ProofStatus.FAILED,
                    method=ProofMethod.SYMPY_DIRECT,
                    steps=[],
                    execution_time=0.0,
                    confidence_score=0.0,
                    error_message=str(e)
                ))
        
        # Final statistics
        successful = sum(1 for r in results if r.is_successful())
        if total > 0:
            success_rate = successful / total
            self.logger.info(f"Batch proof complete: {successful}/{total} theorems proved ({success_rate:.1%})")
        else:
            self.logger.info("Batch proof complete: 0 theorems processed")
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about proof attempts."""
        stats = self.stats.copy()
        
        # Calculate success rates for methods
        for method, data in stats['method_success_rates'].items():
            if data['attempts'] > 0:
                data['success_rate'] = data['successes'] / data['attempts']
            else:
                data['success_rate'] = 0.0
        
        # Overall success rate
        if stats['total_attempts'] > 0:
            stats['overall_success_rate'] = stats['successful_proofs'] / stats['total_attempts']
            stats['average_time_per_attempt'] = stats['total_time'] / stats['total_attempts']
        else:
            stats['overall_success_rate'] = 0.0
            stats['average_time_per_attempt'] = 0.0
        
        return stats
    
    def save_results(self, results: List[ProofResult], output_path: Union[str, Path]) -> None:
        """Save proof results to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'metadata': {
                'total_results': len(results),
                'successful_proofs': sum(1 for r in results if r.is_successful()),
                'timestamp': time.time(),
                'engine_config': self.config,
                'statistics': self.get_statistics()
            },
            'results': [result.to_dict() for result in results]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Proof results saved to {output_path}")
