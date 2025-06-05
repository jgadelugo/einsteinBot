"""
Core proof pipeline for MathBot Phase 5E.

This module implements the ProofPipeline class that orchestrates end-to-end proof attempts
combining all Phase 5 components: theorem generation (5A), symbolic proving (5B),
logic rules (5C), and formal verification (5D).
"""

import hashlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import json

import sympy as sp

# Phase 5A: Theorem Generation
from proofs.theorem_generator import TheoremGenerator, Theorem, TheoremType

# Phase 5B: Proof Attempt Engine
from proofs.proof_attempt import ProofAttemptEngine, ProofResult, ProofStatus

# Phase 5C: Logic Rules (will be implemented)
try:
    from proofs.utils.logic import LogicalRuleEngine, TransformationStep
except ImportError:
    LogicalRuleEngine = None
    TransformationStep = None

# Phase 5D: Formal Systems
from proofs.formal_systems.base_interface import FormalSystemInterface
from proofs.formal_systems.lean4_interface import Lean4Interface


class ProofStrategy(Enum):
    """Available proof strategies in the pipeline."""
    SYMBOLIC_ONLY = "symbolic"
    RULE_BASED = "rule_based"  
    FORMAL_VERIFICATION = "formal"
    HYBRID_SYMBOLIC_RULE = "hybrid_sr"
    HYBRID_SYMBOLIC_FORMAL = "hybrid_sf"
    COMPREHENSIVE = "comprehensive"
    AUTO_SELECT = "auto"


@dataclass
class ComprehensiveResult:
    """Complete result from proof pipeline including all attempted strategies."""
    theorem: Theorem
    strategies_attempted: List[ProofStrategy]
    symbolic_result: Optional[ProofResult] = None
    rule_transformations: List[Any] = field(default_factory=list)  # TransformationStep when available
    formal_result: Optional[Any] = None  # FormalProof when available
    final_status: str = "unknown"  # "proved", "partially_proved", "failed"
    confidence_score: float = 0.0
    execution_time: float = 0.0
    error_messages: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            'theorem_id': self.theorem.id,
            'theorem_statement': self.theorem.statement,
            'strategies_attempted': [s.value for s in self.strategies_attempted],
            'symbolic_result': self.symbolic_result.to_dict() if self.symbolic_result else None,
            'rule_transformations': [str(t) for t in self.rule_transformations],
            'formal_result': str(self.formal_result) if self.formal_result else None,
            'final_status': self.final_status,
            'confidence_score': self.confidence_score,
            'execution_time': self.execution_time,
            'error_messages': self.error_messages,
            'metadata': self.metadata
        }


@dataclass 
class ProofPipeline:
    """
    Orchestrates end-to-end proof attempts combining all Phase 5 components.
    
    This is the main integration class that manages strategy selection, execution flow,
    and result integration across theorem generation, symbolic proving, rule-based 
    transformations, and formal verification.
    """
    theorem_generator: TheoremGenerator      # Phase 5A
    proof_engine: ProofAttemptEngine        # Phase 5B  
    rule_engine: Optional[Any] = None       # Phase 5C (LogicalRuleEngine when available)
    formal_systems: Dict[str, FormalSystemInterface] = field(default_factory=dict)  # Phase 5D
    
    def __post_init__(self):
        """Initialize pipeline components and statistics."""
        self.logger = logging.getLogger(__name__)
        self.statistics = {
            'theorems_attempted': 0,
            'theorems_proved': 0,
            'strategies_used': {},
            'execution_times': [],
            'method_success_rates': {}
        }
        
        # Initialize rule engine if available
        if LogicalRuleEngine is not None and self.rule_engine is None:
            try:
                self.rule_engine = LogicalRuleEngine()
                self.logger.info("LogicalRuleEngine initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize LogicalRuleEngine: {e}")
                self.rule_engine = None
        
        self.logger.info("ProofPipeline initialized with all available components")

    def prove_theorem(self, theorem: Theorem, 
                      strategies: List[Union[str, ProofStrategy]] = None,
                      timeout: int = 30) -> ComprehensiveResult:
        """
        Prove single theorem using specified strategies.
        
        Args:
            theorem: Theorem to prove
            strategies: List of strategies to attempt (defaults to AUTO_SELECT)
            timeout: Timeout in seconds per strategy
            
        Returns:
            ComprehensiveResult with detailed proof information
        """
        start_time = time.time()
        
        if strategies is None:
            strategies = [ProofStrategy.AUTO_SELECT]
        
        # Convert string strategies to enum
        strategy_list = []
        for s in strategies:
            if isinstance(s, str):
                try:
                    strategy_list.append(ProofStrategy(s))
                except ValueError:
                    self.logger.warning(f"Unknown strategy: {s}, skipping")
                    continue
            else:
                strategy_list.append(s)
        
        if not strategy_list:
            strategy_list = [ProofStrategy.AUTO_SELECT]
        
        self.statistics['theorems_attempted'] += 1
        
        try:
            result = self._execute_strategies(theorem, strategy_list, timeout)
            if result.final_status == "proved":
                self.statistics['theorems_proved'] += 1
            
            execution_time = time.time() - start_time
            self.statistics['execution_times'].append(execution_time)
            result.execution_time = execution_time
            
            return result
        except Exception as e:
            self.logger.error(f"Error proving theorem {theorem.id}: {e}")
            return self._create_error_result(theorem, str(e))

    def _execute_strategies(self, theorem: Theorem, 
                           strategies: List[ProofStrategy], 
                           timeout: int) -> ComprehensiveResult:
        """Execute proof strategies for a theorem."""
        results = {}
        
        for strategy in strategies:
            if strategy == ProofStrategy.AUTO_SELECT:
                strategy = self._auto_select_strategy(theorem)
            
            self.statistics['strategies_used'][strategy.value] = \
                self.statistics['strategies_used'].get(strategy.value, 0) + 1
            
            try:
                if strategy == ProofStrategy.SYMBOLIC_ONLY:
                    results[strategy] = self._execute_symbolic_only(theorem, timeout)
                elif strategy == ProofStrategy.RULE_BASED:
                    results[strategy] = self._execute_rule_based(theorem, timeout)
                elif strategy == ProofStrategy.FORMAL_VERIFICATION:
                    results[strategy] = self._execute_formal_verification(theorem, timeout)
                elif strategy == ProofStrategy.HYBRID_SYMBOLIC_RULE:
                    results[strategy] = self._execute_hybrid_sr(theorem, timeout)
                elif strategy == ProofStrategy.HYBRID_SYMBOLIC_FORMAL:
                    results[strategy] = self._execute_hybrid_sf(theorem, timeout)
                elif strategy == ProofStrategy.COMPREHENSIVE:
                    results[strategy] = self._execute_comprehensive(theorem, timeout)
            except Exception as e:
                self.logger.warning(f"Strategy {strategy.value} failed: {e}")
                results[strategy] = {'error': str(e)}
                continue
        
        return self._integrate_results(theorem, results)

    def _execute_symbolic_only(self, theorem: Theorem, timeout: int) -> Dict[str, Any]:
        """Execute symbolic-only proof strategy using Phase 5B."""
        try:
            result = self.proof_engine.attempt_proof(theorem)
            return {
                'symbolic_result': result,
                'strategy': ProofStrategy.SYMBOLIC_ONLY,
                'success': result.status == ProofStatus.PROVED
            }
        except Exception as e:
            return {'error': str(e), 'strategy': ProofStrategy.SYMBOLIC_ONLY}

    def _execute_rule_based(self, theorem: Theorem, timeout: int) -> Dict[str, Any]:
        """Execute rule-based proof strategy using Phase 5C."""
        if self.rule_engine is None:
            return {'error': 'LogicalRuleEngine not available', 'strategy': ProofStrategy.RULE_BASED}
        
        try:
            expr = theorem.sympy_expression
            transformations = self.rule_engine.apply_transformation_sequence(expr, max_steps=5)
            
            return {
                'transformations': transformations,
                'strategy': ProofStrategy.RULE_BASED,
                'success': len(transformations) > 0
            }
        except Exception as e:
            return {'error': str(e), 'strategy': ProofStrategy.RULE_BASED}

    def _execute_formal_verification(self, theorem: Theorem, timeout: int) -> Dict[str, Any]:
        """Execute formal verification strategy using Phase 5D."""
        if 'lean4' not in self.formal_systems:
            return {'error': 'Lean4 interface not available', 'strategy': ProofStrategy.FORMAL_VERIFICATION}
        
        try:
            lean_interface = self.formal_systems['lean4']
            formal_statement = lean_interface.translate_theorem(theorem)
            formal_result = lean_interface.attempt_proof(formal_statement, timeout=timeout)
            
            return {
                'formal_result': formal_result,
                'strategy': ProofStrategy.FORMAL_VERIFICATION,
                'success': formal_result.verification_status == "proved" if formal_result else False
            }
        except Exception as e:
            return {'error': str(e), 'strategy': ProofStrategy.FORMAL_VERIFICATION}

    def _execute_hybrid_sr(self, theorem: Theorem, timeout: int) -> Dict[str, Any]:
        """Execute hybrid symbolic + rule-based strategy."""
        
        # First try rule-based transformations
        transformations = []
        if self.rule_engine is not None:
            try:
                expr = theorem.sympy_expression
                transformations = self.rule_engine.apply_transformation_sequence(
                    expr, max_steps=3
                )
            except Exception as e:
                self.logger.warning(f"Rule transformations failed: {e}")
        
        # Then apply symbolic proving
        symbolic_result = None
        try:
            symbolic_result = self.proof_engine.attempt_proof(theorem)
        except Exception as e:
            self.logger.warning(f"Symbolic proof failed: {e}")
        
        success = (len(transformations) > 0 and 
                  symbolic_result and 
                  symbolic_result.status == ProofStatus.PROVED)
        
        return {
            'transformations': transformations,
            'symbolic_result': symbolic_result,
            'strategy': ProofStrategy.HYBRID_SYMBOLIC_RULE,
            'success': success
        }

    def _execute_hybrid_sf(self, theorem: Theorem, timeout: int) -> Dict[str, Any]:
        """Execute hybrid symbolic + formal strategy."""
        timeout_per_method = timeout // 2
        
        # Symbolic proving
        symbolic_result = None
        try:
            symbolic_result = self.proof_engine.attempt_proof(theorem)
        except Exception as e:
            self.logger.warning(f"Symbolic proof failed: {e}")
        
        # Formal verification
        formal_result = None
        if 'lean4' in self.formal_systems:
            try:
                lean_interface = self.formal_systems['lean4']
                formal_statement = lean_interface.translate_theorem(theorem)
                formal_result = lean_interface.attempt_proof(
                    formal_statement, timeout=timeout_per_method
                )
            except Exception as e:
                self.logger.warning(f"Formal verification failed: {e}")
        
        symbolic_success = symbolic_result and symbolic_result.status == ProofStatus.PROVED
        formal_success = formal_result and formal_result.verification_status == "proved"
        
        return {
            'symbolic_result': symbolic_result,
            'formal_result': formal_result,
            'strategy': ProofStrategy.HYBRID_SYMBOLIC_FORMAL,
            'success': symbolic_success or formal_success
        }

    def _execute_comprehensive(self, theorem: Theorem, timeout: int) -> Dict[str, Any]:
        """Execute comprehensive strategy using all available methods."""
        timeout_per_method = timeout // 3
        results = {}
        
        # Symbolic proving
        try:
            results['symbolic'] = self.proof_engine.attempt_proof(theorem)
        except Exception as e:
            self.logger.warning(f"Symbolic method failed: {e}")
        
        # Rule-based transformations
        if self.rule_engine is not None:
            try:
                expr = theorem.sympy_expression
                results['transformations'] = self.rule_engine.apply_transformation_sequence(
                    expr, max_steps=5
                )
            except Exception as e:
                self.logger.warning(f"Rule method failed: {e}")
        
        # Formal verification (if available)
        if 'lean4' in self.formal_systems:
            try:
                lean_interface = self.formal_systems['lean4']
                formal_statement = lean_interface.translate_theorem(theorem)
                results['formal'] = lean_interface.attempt_proof(
                    formal_statement, timeout=timeout_per_method
                )
            except Exception as e:
                self.logger.warning(f"Formal verification failed: {e}")
        
        # Determine overall success
        symbolic_success = (results.get('symbolic') and 
                           results['symbolic'].status == ProofStatus.PROVED)
        rule_success = len(results.get('transformations', [])) > 0
        formal_success = (results.get('formal') and 
                         results['formal'].verification_status == "proved")
        
        return {
            **results,
            'strategy': ProofStrategy.COMPREHENSIVE,
            'success': symbolic_success or formal_success or rule_success
        }

    def _auto_select_strategy(self, theorem: Theorem) -> ProofStrategy:
        """Intelligent strategy selection based on theorem characteristics."""
        
        # Simple algebraic theorems -> symbolic only
        if theorem.theorem_type == TheoremType.ALGEBRAIC_IDENTITY:
            complexity = self._estimate_complexity(theorem)
            if complexity < 0.3:
                return ProofStrategy.SYMBOLIC_ONLY
        
        # Complex expressions -> rule-based transformations first
        if self._estimate_complexity(theorem) > 0.7:
            return ProofStrategy.HYBRID_SYMBOLIC_RULE
        
        # Functional equations often need comprehensive approach
        if theorem.theorem_type == TheoremType.FUNCTIONAL_EQUATION:
            return ProofStrategy.COMPREHENSIVE
        
        # Default to hybrid symbolic+rule approach
        return ProofStrategy.HYBRID_SYMBOLIC_RULE

    def _estimate_complexity(self, theorem: Theorem) -> float:
        """Estimate theorem complexity score (0.0 to 1.0)."""
        try:
            expr_str = str(theorem.sympy_expression)
            
            # Count operations, variables, functions
            operations = expr_str.count('+') + expr_str.count('*') + expr_str.count('**')
            variables = len(theorem.symbols) if hasattr(theorem, 'symbols') else 0
            functions = expr_str.count('(')
            
            # Normalize to 0-1 scale
            complexity = min(1.0, (operations + variables + functions) / 15.0)
            return complexity
        except:
            return 0.5  # Default complexity

    def _integrate_results(self, theorem: Theorem, 
                          results: Dict[ProofStrategy, Dict]) -> ComprehensiveResult:
        """Integrate results from multiple proof strategies."""
        
        # Determine final status
        final_status = "failed"
        any_success = False
        
        for strategy, result in results.items():
            if result.get('success', False):
                final_status = "proved"
                any_success = True
                break
        
        # Extract best results from each method
        best_symbolic = None
        best_transformations = []
        best_formal = None
        
        for result in results.values():
            # Check for symbolic results (can be stored as 'symbolic_result' or 'symbolic')
            symbolic_res = result.get('symbolic_result') or result.get('symbolic')
            if symbolic_res:
                if best_symbolic is None or symbolic_res.confidence_score > best_symbolic.confidence_score:
                    best_symbolic = symbolic_res
            
            if 'transformations' in result and result['transformations']:
                if len(result['transformations']) > len(best_transformations):
                    best_transformations = result['transformations']
            
            # Check for formal results (can be stored as 'formal_result' or 'formal')
            formal_res = result.get('formal_result') or result.get('formal')
            if formal_res:
                best_formal = formal_res
        
        # Calculate confidence score
        confidence = self._calculate_confidence(results)
        
        # Collect error messages
        error_messages = []
        for result in results.values():
            if 'error' in result:
                error_messages.append(result['error'])
        
        return ComprehensiveResult(
            theorem=theorem,
            strategies_attempted=list(results.keys()),
            symbolic_result=best_symbolic,
            rule_transformations=best_transformations,
            formal_result=best_formal,
            final_status=final_status,
            confidence_score=confidence,
            error_messages=error_messages
        )

    def _calculate_confidence(self, results: Dict[ProofStrategy, Dict]) -> float:
        """Calculate confidence score based on multiple proof attempts."""
        if not results:
            return 0.0
        
        confidence = 0.0
        weight_sum = 0.0
        
        # Weights for different proof methods
        method_weights = {
            ProofStrategy.SYMBOLIC_ONLY: 0.4,
            ProofStrategy.RULE_BASED: 0.3,
            ProofStrategy.FORMAL_VERIFICATION: 0.8,
            ProofStrategy.HYBRID_SYMBOLIC_RULE: 0.6,
            ProofStrategy.HYBRID_SYMBOLIC_FORMAL: 0.7,
            ProofStrategy.COMPREHENSIVE: 0.9
        }
        
        for strategy, result in results.items():
            method_confidence = 0.0
            
            # Success gives high confidence
            if result.get('success', False):
                method_confidence = 0.8
            
            # Symbolic proof confidence
            symbolic_res = result.get('symbolic_result') or result.get('symbolic')
            if symbolic_res:
                method_confidence = max(method_confidence, symbolic_res.confidence_score)
            
            # Rule transformation confidence
            if 'transformations' in result and result['transformations']:
                method_confidence = max(method_confidence, min(0.6, len(result['transformations']) * 0.2))
            
            # Apply strategy weight
            strategy_weight = method_weights.get(strategy, 0.5)
            confidence += method_confidence * strategy_weight
            weight_sum += strategy_weight
        
        return min(1.0, confidence / max(weight_sum, 1.0))

    def _create_error_result(self, theorem: Theorem, error_message: str) -> ComprehensiveResult:
        """Create error result for failed theorem proof."""
        return ComprehensiveResult(
            theorem=theorem,
            strategies_attempted=[],
            final_status="failed", 
            confidence_score=0.0,
            execution_time=0.0,
            error_messages=[error_message]
        )

    def prove_all_theorems(self, max_parallel: int = 3,
                          strategy: ProofStrategy = ProofStrategy.AUTO_SELECT) -> List[ComprehensiveResult]:
        """
        Prove all theorems in parallel.
        
        Args:
            max_parallel: Maximum number of parallel proof attempts
            strategy: Default strategy for all theorems
            
        Returns:
            List of ComprehensiveResult objects
        """
        
        theorems = self.theorem_generator.get_all_theorems() if hasattr(self.theorem_generator, 'get_all_theorems') else []
        
        if not theorems:
            self.logger.warning("No theorems available for proving")
            return []
        
        results = []
        
        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            # Submit all theorem proving tasks
            future_to_theorem = {
                executor.submit(self.prove_theorem, theorem, [strategy]): theorem
                for theorem in theorems
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_theorem):
                theorem = future_to_theorem[future]
                try:
                    result = future.result()
                    results.append(result)
                    self.logger.info(f"Completed theorem {theorem.id}: {result.final_status}")
                except Exception as e:
                    self.logger.error(f"Error with theorem {theorem.id}: {e}")
                    error_result = self._create_error_result(theorem, str(e))
                    results.append(error_result)
        
        return results

    def generate_and_prove(self, hypothesis_data: Dict) -> ComprehensiveResult:
        """
        End-to-end theorem generation and proving workflow.
        
        Args:
            hypothesis_data: Dictionary containing hypothesis information
            
        Returns:
            ComprehensiveResult from proving the generated theorem
        """
        try:
            # Generate theorem from hypothesis data
            hypotheses = [hypothesis_data] if isinstance(hypothesis_data, dict) else hypothesis_data
            theorems = self.theorem_generator.generate_from_hypotheses(hypotheses)
            
            if not theorems:
                raise ValueError("No theorems generated from hypothesis data")
            
            # Prove the first generated theorem
            theorem = theorems[0]
            result = self.prove_theorem(theorem, [ProofStrategy.COMPREHENSIVE])
            
            # Add generation metadata
            result.metadata['generated_from_hypothesis'] = True
            result.metadata['hypothesis_count'] = len(hypotheses)
            result.metadata['theorems_generated'] = len(theorems)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in generate_and_prove: {e}")
            # Create dummy theorem for error result
            dummy_theorem = type('DummyTheorem', (), {
                'id': 'generate_error',
                'statement': 'Failed to generate theorem',
                'sympy_expression': sp.S.Zero
            })()
            return self._create_error_result(dummy_theorem, str(e))

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        stats = self.statistics.copy()
        
        # Calculate success rate
        if stats['theorems_attempted'] > 0:
            stats['success_rate'] = stats['theorems_proved'] / stats['theorems_attempted']
            stats['average_execution_time'] = sum(stats['execution_times']) / len(stats['execution_times'])
        else:
            stats['success_rate'] = 0.0
            stats['average_execution_time'] = 0.0
        
        # Add component availability
        stats['components_available'] = {
            'theorem_generator': self.theorem_generator is not None,
            'proof_engine': self.proof_engine is not None,
            'rule_engine': self.rule_engine is not None,
            'formal_systems': list(self.formal_systems.keys())
        }
        
        return stats 