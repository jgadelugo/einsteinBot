"""
Strategy management for MathBot Phase 5E.

This module implements the StrategyManager class that provides intelligent 
strategy selection and execution for the ProofPipeline. It analyzes theorem
characteristics to select optimal proof strategies.
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

import sympy as sp

from proofs.theorem_generator import Theorem, TheoremType
from proofs.integration.proof_pipeline import ProofStrategy


@dataclass
class TheoremCharacteristics:
    """Characteristics of a theorem used for strategy selection."""
    complexity_score: float
    variable_count: int
    operation_count: int
    function_count: int
    has_trigonometric: bool
    has_logarithmic: bool
    has_exponential: bool
    has_rational: bool
    has_algebraic_structure: bool
    theorem_type: TheoremType
    expression_depth: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert characteristics to dictionary."""
        return {
            'complexity_score': self.complexity_score,
            'variable_count': self.variable_count,
            'operation_count': self.operation_count,
            'function_count': self.function_count,
            'has_trigonometric': self.has_trigonometric,
            'has_logarithmic': self.has_logarithmic,
            'has_exponential': self.has_exponential,
            'has_rational': self.has_rational,
            'has_algebraic_structure': self.has_algebraic_structure,
            'theorem_type': self.theorem_type.value,
            'expression_depth': self.expression_depth
        }


class StrategyScore:
    """Scoring system for proof strategies."""
    
    def __init__(self, strategy: ProofStrategy, base_score: float = 0.0):
        self.strategy = strategy
        self.base_score = base_score
        self.adjustments = {}
        self.final_score = base_score
    
    def add_adjustment(self, reason: str, adjustment: float):
        """Add a score adjustment with reason."""
        self.adjustments[reason] = adjustment
        self.final_score += adjustment
    
    def get_final_score(self) -> float:
        """Get the final calculated score."""
        return max(0.0, min(1.0, self.final_score))  # Clamp to [0, 1]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for analysis."""
        return {
            'strategy': self.strategy.value,
            'base_score': self.base_score,
            'adjustments': self.adjustments,
            'final_score': self.get_final_score()
        }


@dataclass
class StrategyManager:
    """
    Manages intelligent strategy selection and execution for the ProofPipeline.
    
    This class analyzes theorem characteristics and selects optimal proof
    strategies based on mathematical content, complexity, and historical
    performance data.
    """
    
    def __init__(self, performance_history: Optional[Dict] = None):
        """
        Initialize the strategy manager.
        
        Args:
            performance_history: Historical performance data for strategy selection
        """
        self.logger = logging.getLogger(__name__)
        self.performance_history = performance_history or {}
        
        # Strategy performance weights (updated based on experience)
        self.strategy_weights = {
            ProofStrategy.SYMBOLIC_ONLY: 0.7,
            ProofStrategy.RULE_BASED: 0.5,
            ProofStrategy.FORMAL_VERIFICATION: 0.8,
            ProofStrategy.HYBRID_SYMBOLIC_RULE: 0.75,
            ProofStrategy.HYBRID_SYMBOLIC_FORMAL: 0.85,
            ProofStrategy.COMPREHENSIVE: 0.9
        }
        
        # Complexity thresholds for strategy selection
        self.complexity_thresholds = {
            'very_simple': 0.2,
            'simple': 0.4,
            'moderate': 0.6,
            'complex': 0.8
        }
    
    def select_optimal_strategy(self, theorem: Theorem) -> ProofStrategy:
        """
        Select optimal strategy based on theorem characteristics.
        
        Args:
            theorem: Theorem to analyze
            
        Returns:
            Optimal ProofStrategy for the theorem
        """
        start_time = time.time()
        
        # Analyze theorem characteristics
        characteristics = self._analyze_theorem_characteristics(theorem)
        
        # Score all available strategies
        strategy_scores = self._score_all_strategies(characteristics)
        
        # Select best strategy
        best_strategy = max(strategy_scores, key=lambda s: s.get_final_score())
        
        execution_time = time.time() - start_time
        self.logger.info(
            f"Selected strategy {best_strategy.strategy.value} "
            f"(score: {best_strategy.get_final_score():.3f}) "
            f"for theorem {theorem.id} in {execution_time:.3f}s"
        )
        
        return best_strategy.strategy
    
    def _analyze_theorem_characteristics(self, theorem: Theorem) -> TheoremCharacteristics:
        """
        Analyze theorem to extract mathematical characteristics.
        
        Args:
            theorem: Theorem to analyze
            
        Returns:
            TheoremCharacteristics object
        """
        try:
            expr = theorem.sympy_expression
            expr_str = str(expr).lower()
            
            # Count variables, operations, and functions
            if hasattr(expr, 'free_symbols'):
                variables = expr.free_symbols
            elif hasattr(theorem, 'symbols'):
                variables = theorem.symbols
            else:
                variables = set()
            
            variable_count = len(variables)
            
            # Count operations
            operation_count = self._count_operations(expr)
            
            # Count functions
            function_count = len(expr.atoms(sp.Function)) if hasattr(expr, 'atoms') else 0
            
            # Detect mathematical content types
            has_trigonometric = any(func in expr_str for func in [
                'sin', 'cos', 'tan', 'sec', 'csc', 'cot', 
                'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh'
            ])
            
            has_logarithmic = any(func in expr_str for func in ['log', 'ln'])
            has_exponential = any(func in expr_str for func in ['exp', 'e**'])
            has_rational = '/' in expr_str or any(op in expr_str for op in ['rational', 'fraction'])
            
            # Check for algebraic structure
            has_algebraic_structure = any(op in expr_str for op in ['+', '*', '**', 'expand', 'factor'])
            
            # Calculate expression depth
            expression_depth = self._calculate_expression_depth(expr)
            
            # Calculate overall complexity
            complexity_score = self._calculate_complexity_score(
                variable_count, operation_count, function_count, 
                expression_depth, has_trigonometric, has_logarithmic
            )
            
            return TheoremCharacteristics(
                complexity_score=complexity_score,
                variable_count=variable_count,
                operation_count=operation_count,
                function_count=function_count,
                has_trigonometric=has_trigonometric,
                has_logarithmic=has_logarithmic,
                has_exponential=has_exponential,
                has_rational=has_rational,
                has_algebraic_structure=has_algebraic_structure,
                theorem_type=theorem.theorem_type,
                expression_depth=expression_depth
            )
            
        except Exception as e:
            self.logger.warning(f"Error analyzing theorem characteristics: {e}")
            # Return default characteristics
            return TheoremCharacteristics(
                complexity_score=0.5,
                variable_count=1,
                operation_count=1,
                function_count=0,
                has_trigonometric=False,
                has_logarithmic=False,
                has_exponential=False,
                has_rational=False,
                has_algebraic_structure=True,
                theorem_type=theorem.theorem_type,
                expression_depth=2
            )
    
    def _count_operations(self, expr) -> int:
        """Count mathematical operations in expression."""
        try:
            if hasattr(expr, 'atoms'):
                # Count Add, Mul, Pow operations
                operations = expr.atoms(sp.Add, sp.Mul, sp.Pow)
                return len(operations)
            else:
                # Fallback: count operators in string representation
                expr_str = str(expr)
                return expr_str.count('+') + expr_str.count('*') + expr_str.count('**')
        except:
            return 1  # Default value
    
    def _calculate_expression_depth(self, expr) -> int:
        """Calculate the depth of the expression tree."""
        try:
            if hasattr(expr, 'args') and expr.args:
                return 1 + max(self._calculate_expression_depth(arg) for arg in expr.args)
            else:
                return 1
        except:
            return 2  # Default depth
    
    def _calculate_complexity_score(self, variable_count: int, operation_count: int, 
                                   function_count: int, expression_depth: int,
                                   has_trigonometric: bool, has_logarithmic: bool) -> float:
        """
        Calculate overall complexity score for a theorem.
        
        Args:
            variable_count: Number of variables
            operation_count: Number of operations
            function_count: Number of functions
            expression_depth: Depth of expression tree
            has_trigonometric: Whether expression contains trig functions
            has_logarithmic: Whether expression contains log functions
            
        Returns:
            Complexity score between 0.0 and 1.0
        """
        # Base complexity from structural elements
        structural_complexity = (
            variable_count * 0.1 +
            operation_count * 0.15 +
            function_count * 0.2 +
            expression_depth * 0.1
        )
        
        # Bonuses for special function types
        function_complexity = 0.0
        if has_trigonometric:
            function_complexity += 0.2
        if has_logarithmic:
            function_complexity += 0.15
        
        # Combine and normalize
        total_complexity = structural_complexity + function_complexity
        return min(1.0, total_complexity / 2.0)  # Normalize to [0, 1]
    
    def _score_all_strategies(self, characteristics: TheoremCharacteristics) -> List[StrategyScore]:
        """
        Score all available strategies for given theorem characteristics.
        
        Args:
            characteristics: Analyzed theorem characteristics
            
        Returns:
            List of StrategyScore objects for all strategies
        """
        scores = []
        
        for strategy in ProofStrategy:
            if strategy == ProofStrategy.AUTO_SELECT:
                continue  # Skip auto-select in scoring
            
            score = self._score_strategy(strategy, characteristics)
            scores.append(score)
        
        return scores
    
    def _score_strategy(self, strategy: ProofStrategy, 
                       characteristics: TheoremCharacteristics) -> StrategyScore:
        """
        Score a specific strategy for given characteristics.
        
        Args:
            strategy: Strategy to score
            characteristics: Theorem characteristics
            
        Returns:
            StrategyScore object
        """
        # Start with base weight for the strategy
        base_score = self.strategy_weights.get(strategy, 0.5)
        score = StrategyScore(strategy, base_score)
        
        # Adjust based on complexity
        if strategy == ProofStrategy.SYMBOLIC_ONLY:
            self._score_symbolic_only(score, characteristics)
        elif strategy == ProofStrategy.RULE_BASED:
            self._score_rule_based(score, characteristics)
        elif strategy == ProofStrategy.FORMAL_VERIFICATION:
            self._score_formal_verification(score, characteristics)
        elif strategy == ProofStrategy.HYBRID_SYMBOLIC_RULE:
            self._score_hybrid_symbolic_rule(score, characteristics)
        elif strategy == ProofStrategy.HYBRID_SYMBOLIC_FORMAL:
            self._score_hybrid_symbolic_formal(score, characteristics)
        elif strategy == ProofStrategy.COMPREHENSIVE:
            self._score_comprehensive(score, characteristics)
        
        # Apply theorem type specific adjustments
        self._apply_theorem_type_adjustments(score, characteristics)
        
        # Apply performance history adjustments
        self._apply_performance_history_adjustments(score, characteristics)
        
        return score
    
    def _score_symbolic_only(self, score: StrategyScore, 
                            characteristics: TheoremCharacteristics):
        """Score symbolic-only strategy."""
        # Best for simple algebraic expressions
        if characteristics.complexity_score < self.complexity_thresholds['simple']:
            score.add_adjustment("low_complexity", 0.3)
        elif characteristics.complexity_score > self.complexity_thresholds['complex']:
            score.add_adjustment("high_complexity", -0.4)
        
        # Good for pure algebraic content
        if characteristics.has_algebraic_structure and not any([
            characteristics.has_trigonometric,
            characteristics.has_logarithmic,
            characteristics.has_exponential
        ]):
            score.add_adjustment("pure_algebraic", 0.2)
        
        # Penalty for complex functions
        if characteristics.has_trigonometric:
            score.add_adjustment("trigonometric_functions", -0.2)
    
    def _score_rule_based(self, score: StrategyScore, 
                         characteristics: TheoremCharacteristics):
        """Score rule-based strategy."""
        # Good for moderate complexity with algebraic structure
        if (self.complexity_thresholds['simple'] <= characteristics.complexity_score 
            <= self.complexity_thresholds['moderate']):
            score.add_adjustment("moderate_complexity", 0.2)
        
        # Excellent for expressions with clear transformation patterns
        if characteristics.has_algebraic_structure:
            score.add_adjustment("algebraic_structure", 0.3)
        
        # Good for trigonometric identities
        if characteristics.has_trigonometric:
            score.add_adjustment("trigonometric_identities", 0.25)
    
    def _score_formal_verification(self, score: StrategyScore, 
                                  characteristics: TheoremCharacteristics):
        """Score formal verification strategy."""
        # Excellent for well-structured theorems
        if characteristics.theorem_type in [
            TheoremType.ALGEBRAIC_IDENTITY,
            TheoremType.TRIGONOMETRIC
        ]:
            score.add_adjustment("formal_suitable_type", 0.2)
        
        # Good for complex theorems where symbolic methods might fail
        if characteristics.complexity_score > self.complexity_thresholds['moderate']:
            score.add_adjustment("complex_theorem", 0.15)
        
        # Penalty for very simple theorems (overkill)
        if characteristics.complexity_score < self.complexity_thresholds['very_simple']:
            score.add_adjustment("too_simple", -0.3)
    
    def _score_hybrid_symbolic_rule(self, score: StrategyScore, 
                                   characteristics: TheoremCharacteristics):
        """Score hybrid symbolic+rule strategy."""
        # Excellent for moderate to complex algebraic theorems
        if (characteristics.has_algebraic_structure and 
            characteristics.complexity_score >= self.complexity_thresholds['simple']):
            score.add_adjustment("good_hybrid_candidate", 0.25)
        
        # Good balance for most theorem types
        score.add_adjustment("balanced_approach", 0.1)
    
    def _score_hybrid_symbolic_formal(self, score: StrategyScore, 
                                     characteristics: TheoremCharacteristics):
        """Score hybrid symbolic+formal strategy."""
        # Excellent for complex theorems where high confidence is needed
        if characteristics.complexity_score > self.complexity_thresholds['moderate']:
            score.add_adjustment("high_confidence_needed", 0.2)
        
        # Good for formal-suitable theorem types
        if characteristics.theorem_type in [
            TheoremType.ALGEBRAIC_IDENTITY,
            TheoremType.TRIGONOMETRIC,
            TheoremType.FUNCTIONAL_EQUATION
        ]:
            score.add_adjustment("formal_suitable", 0.15)
    
    def _score_comprehensive(self, score: StrategyScore, 
                            characteristics: TheoremCharacteristics):
        """Score comprehensive strategy."""
        # Always gets a good base score, but may be overkill for simple theorems
        if characteristics.complexity_score > self.complexity_thresholds['complex']:
            score.add_adjustment("complex_needs_comprehensive", 0.1)
        elif characteristics.complexity_score < self.complexity_thresholds['simple']:
            score.add_adjustment("simple_overkill", -0.2)
        
        # Good for unknown or challenging theorem types
        if characteristics.theorem_type in [
            TheoremType.FUNCTIONAL_EQUATION,
            TheoremType.GENERALIZATION
        ]:
            score.add_adjustment("challenging_type", 0.15)
    
    def _apply_theorem_type_adjustments(self, score: StrategyScore, 
                                      characteristics: TheoremCharacteristics):
        """Apply theorem type specific adjustments."""
        theorem_type = characteristics.theorem_type
        strategy = score.strategy
        
        # Algebraic identities
        if theorem_type == TheoremType.ALGEBRAIC_IDENTITY:
            if strategy == ProofStrategy.SYMBOLIC_ONLY:
                score.add_adjustment("algebraic_identity_symbolic", 0.2)
            elif strategy == ProofStrategy.RULE_BASED:
                score.add_adjustment("algebraic_identity_rules", 0.15)
        
        # Trigonometric theorems
        elif theorem_type == TheoremType.TRIGONOMETRIC:
            if strategy == ProofStrategy.RULE_BASED:
                score.add_adjustment("trigonometric_rules", 0.25)
            elif strategy == ProofStrategy.FORMAL_VERIFICATION:
                score.add_adjustment("trigonometric_formal", 0.2)
        
        # Functional equations
        elif theorem_type == TheoremType.FUNCTIONAL_EQUATION:
            if strategy in [ProofStrategy.COMPREHENSIVE, ProofStrategy.HYBRID_SYMBOLIC_FORMAL]:
                score.add_adjustment("functional_equation_comprehensive", 0.3)
            elif strategy == ProofStrategy.HYBRID_SYMBOLIC_RULE:
                score.add_adjustment("functional_equation_rule_based", -0.1)  # Slight penalty
    
    def _apply_performance_history_adjustments(self, score: StrategyScore, 
                                             characteristics: TheoremCharacteristics):
        """Apply adjustments based on historical performance."""
        if not self.performance_history:
            return
        
        strategy_key = score.strategy.value
        if strategy_key in self.performance_history:
            history = self.performance_history[strategy_key]
            
            # Adjust based on success rate
            success_rate = history.get('success_rate', 0.5)
            if success_rate > 0.8:
                score.add_adjustment("high_historical_success", 0.1)
            elif success_rate < 0.3:
                score.add_adjustment("low_historical_success", -0.1)
            
            # Adjust based on average execution time
            avg_time = history.get('avg_execution_time', 30.0)
            if avg_time < 10.0:  # Fast execution
                score.add_adjustment("fast_execution", 0.05)
            elif avg_time > 60.0:  # Slow execution
                score.add_adjustment("slow_execution", -0.05)
    
    def get_strategy_recommendation_details(self, theorem: Theorem) -> Dict[str, Any]:
        """
        Get detailed strategy recommendation with analysis.
        
        Args:
            theorem: Theorem to analyze
            
        Returns:
            Dictionary with detailed recommendation analysis
        """
        characteristics = self._analyze_theorem_characteristics(theorem)
        strategy_scores = self._score_all_strategies(characteristics)
        
        # Sort by score
        strategy_scores.sort(key=lambda s: s.get_final_score(), reverse=True)
        
        return {
            'theorem_id': theorem.id,
            'characteristics': characteristics.to_dict(),
            'recommended_strategy': strategy_scores[0].strategy.value,
            'strategy_scores': [score.to_dict() for score in strategy_scores],
            'reasoning': self._generate_reasoning(characteristics, strategy_scores[0])
        }
    
    def _generate_reasoning(self, characteristics: TheoremCharacteristics, 
                          best_score: StrategyScore) -> str:
        """Generate human-readable reasoning for strategy selection."""
        reasons = []
        
        # Complexity-based reasoning
        if characteristics.complexity_score < self.complexity_thresholds['simple']:
            reasons.append("theorem has low complexity")
        elif characteristics.complexity_score > self.complexity_thresholds['complex']:
            reasons.append("theorem has high complexity")
        
        # Content-based reasoning
        if characteristics.has_trigonometric:
            reasons.append("contains trigonometric functions")
        if characteristics.has_logarithmic:
            reasons.append("contains logarithmic functions")
        if characteristics.has_algebraic_structure:
            reasons.append("has clear algebraic structure")
        
        # Strategy-specific reasoning
        strategy = best_score.strategy
        if strategy == ProofStrategy.SYMBOLIC_ONLY:
            reasons.append("symbolic methods should be sufficient")
        elif strategy == ProofStrategy.COMPREHENSIVE:
            reasons.append("may require multiple proof approaches")
        elif "hybrid" in strategy.value:
            reasons.append("benefits from combined approach")
        
        # Top adjustments
        top_adjustments = sorted(
            best_score.adjustments.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )[:2]
        
        for reason, adjustment in top_adjustments:
            if adjustment > 0:
                reasons.append(f"favored due to {reason.replace('_', ' ')}")
        
        return f"Selected {strategy.value} because " + ", ".join(reasons)
    
    def update_performance_history(self, strategy: ProofStrategy, 
                                 success: bool, execution_time: float):
        """
        Update performance history with new results.
        
        Args:
            strategy: Strategy that was executed
            success: Whether the strategy was successful
            execution_time: Time taken for execution
        """
        strategy_key = strategy.value
        
        if strategy_key not in self.performance_history:
            self.performance_history[strategy_key] = {
                'total_attempts': 0,
                'successes': 0,
                'total_time': 0.0,
                'success_rate': 0.0,
                'avg_execution_time': 0.0
            }
        
        history = self.performance_history[strategy_key]
        history['total_attempts'] += 1
        history['total_time'] += execution_time
        
        if success:
            history['successes'] += 1
        
        # Recalculate derived metrics
        history['success_rate'] = history['successes'] / history['total_attempts']
        history['avg_execution_time'] = history['total_time'] / history['total_attempts']
        
        self.logger.debug(
            f"Updated performance history for {strategy_key}: "
            f"success_rate={history['success_rate']:.3f}, "
            f"avg_time={history['avg_execution_time']:.3f}s"
        ) 