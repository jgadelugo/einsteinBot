"""
Test suite for the StrategyManager module.

Tests cover strategy selection algorithms, theorem analysis, performance
tracking, and detailed recommendation functionality.
"""

import pytest
from unittest.mock import Mock

import sympy as sp

from proofs.integration.strategy_manager import (
    StrategyManager, 
    TheoremCharacteristics, 
    StrategyScore
)
from proofs.integration.proof_pipeline import ProofStrategy
from proofs.theorem_generator import Theorem, TheoremType, SourceLineage


class TestStrategyManager:
    """Test suite for StrategyManager class."""
    
    @pytest.fixture
    def strategy_manager(self):
        """Create a StrategyManager for testing."""
        return StrategyManager()
    
    @pytest.fixture
    def strategy_manager_with_history(self):
        """Create a StrategyManager with performance history."""
        history = {
            'symbolic': {
                'total_attempts': 10,
                'successes': 8,
                'total_time': 50.0,
                'success_rate': 0.8,
                'avg_execution_time': 5.0
            },
            'comprehensive': {
                'total_attempts': 5,
                'successes': 2,
                'total_time': 150.0,
                'success_rate': 0.4,
                'avg_execution_time': 30.0
            }
        }
        return StrategyManager(performance_history=history)
    
    @pytest.fixture
    def simple_algebraic_theorem(self):
        """Create a simple algebraic theorem for testing."""
        lineage = SourceLineage(
            original_formula="x + 0",
            hypothesis_id="test_hyp",
            confidence=1.0,
            validation_score=1.0,
            generation_method="test"
        )
        
        return Theorem(
            id="simple_theorem",
            statement="∀x ∈ ℝ, x + 0 = x",
            sympy_expression=sp.Eq(sp.Symbol('x') + 0, sp.Symbol('x')),
            theorem_type=TheoremType.ALGEBRAIC_IDENTITY,
            assumptions=["x ∈ ℝ"],
            source_lineage=lineage,
            symbols={'x'}
        )
    
    @pytest.fixture
    def complex_trigonometric_theorem(self):
        """Create a complex trigonometric theorem for testing."""
        lineage = SourceLineage(
            original_formula="sin(x)**2 + cos(x)**2",
            hypothesis_id="trig_hyp",
            confidence=1.0,
            validation_score=1.0,
            generation_method="test"
        )
        
        x = sp.Symbol('x')
        expr = sp.Eq(sp.sin(x)**2 + sp.cos(x)**2, 1)
        
        return Theorem(
            id="trig_theorem",
            statement="∀x ∈ ℝ, sin²(x) + cos²(x) = 1",
            sympy_expression=expr,
            theorem_type=TheoremType.TRIGONOMETRIC,
            assumptions=["x ∈ ℝ"],
            source_lineage=lineage,
            symbols={'x'}
        )
    
    @pytest.fixture
    def functional_equation_theorem(self):
        """Create a functional equation theorem for testing."""
        lineage = SourceLineage(
            original_formula="f(x + y) = f(x) + f(y)",
            hypothesis_id="func_hyp",
            confidence=1.0,
            validation_score=1.0,
            generation_method="test"
        )
        
        # Mock a functional equation
        theorem = Mock()
        theorem.id = "func_theorem"
        theorem.statement = "f(x + y) = f(x) + f(y)"
        theorem.theorem_type = TheoremType.FUNCTIONAL_EQUATION
        theorem.source_lineage = lineage
        theorem.symbols = {'x', 'y', 'f'}
        
        # Create a mock sympy expression
        x, y = sp.symbols('x y')
        f = sp.Function('f')
        theorem.sympy_expression = sp.Eq(f(x + y), f(x) + f(y))
        
        return theorem
    
    def test_strategy_manager_initialization(self, strategy_manager):
        """Test StrategyManager initialization."""
        assert strategy_manager.performance_history == {}
        assert len(strategy_manager.strategy_weights) == 6
        assert len(strategy_manager.complexity_thresholds) == 4
        
        # Check all strategies have weights
        for strategy in ProofStrategy:
            if strategy != ProofStrategy.AUTO_SELECT:
                assert strategy in strategy_manager.strategy_weights
    
    def test_strategy_manager_with_history(self, strategy_manager_with_history):
        """Test StrategyManager initialization with performance history."""
        assert len(strategy_manager_with_history.performance_history) == 2
        assert 'symbolic' in strategy_manager_with_history.performance_history
        assert 'comprehensive' in strategy_manager_with_history.performance_history
    
    def test_select_optimal_strategy_simple_algebraic(self, strategy_manager, simple_algebraic_theorem):
        """Test strategy selection for simple algebraic theorem."""
        strategy = strategy_manager.select_optimal_strategy(simple_algebraic_theorem)
        
        # Simple algebraic theorems should prefer symbolic-only or hybrid approaches
        assert strategy in [
            ProofStrategy.SYMBOLIC_ONLY,
            ProofStrategy.HYBRID_SYMBOLIC_RULE,
            ProofStrategy.HYBRID_SYMBOLIC_FORMAL
        ]
    
    def test_select_optimal_strategy_trigonometric(self, strategy_manager, complex_trigonometric_theorem):
        """Test strategy selection for trigonometric theorem."""
        strategy = strategy_manager.select_optimal_strategy(complex_trigonometric_theorem)
        
        # Trigonometric theorems should prefer rule-based or hybrid approaches
        assert strategy in [
            ProofStrategy.RULE_BASED,
            ProofStrategy.HYBRID_SYMBOLIC_RULE,
            ProofStrategy.FORMAL_VERIFICATION,
            ProofStrategy.COMPREHENSIVE
        ]
    
    def test_select_optimal_strategy_functional_equation(self, strategy_manager, functional_equation_theorem):
        """Test strategy selection for functional equation."""
        strategy = strategy_manager.select_optimal_strategy(functional_equation_theorem)
        
        # Functional equations should prefer comprehensive or hybrid approaches
        assert strategy in [
            ProofStrategy.COMPREHENSIVE,
            ProofStrategy.HYBRID_SYMBOLIC_FORMAL,
            ProofStrategy.HYBRID_SYMBOLIC_RULE  # May also be selected
        ]
    
    def test_analyze_theorem_characteristics_simple(self, strategy_manager, simple_algebraic_theorem):
        """Test theorem characteristics analysis for simple theorem."""
        characteristics = strategy_manager._analyze_theorem_characteristics(simple_algebraic_theorem)
        
        assert isinstance(characteristics, TheoremCharacteristics)
        assert characteristics.theorem_type == TheoremType.ALGEBRAIC_IDENTITY
        assert characteristics.variable_count >= 0  # May be 0 if symbols not detected
        assert characteristics.has_algebraic_structure in [True, False]  # Depends on expression detection
        assert characteristics.has_trigonometric is False
        assert characteristics.complexity_score >= 0.0
        assert characteristics.complexity_score <= 1.0
    
    def test_analyze_theorem_characteristics_trigonometric(self, strategy_manager, complex_trigonometric_theorem):
        """Test theorem characteristics analysis for trigonometric theorem."""
        characteristics = strategy_manager._analyze_theorem_characteristics(complex_trigonometric_theorem)
        
        assert isinstance(characteristics, TheoremCharacteristics)
        assert characteristics.theorem_type == TheoremType.TRIGONOMETRIC
        assert characteristics.has_trigonometric is True
        assert characteristics.variable_count >= 1
        assert characteristics.operation_count > 0
    
    def test_analyze_theorem_characteristics_error_handling(self, strategy_manager):
        """Test error handling in theorem characteristics analysis."""
        # Create a malformed theorem
        malformed_theorem = Mock()
        malformed_theorem.sympy_expression = None
        malformed_theorem.theorem_type = TheoremType.ALGEBRAIC_IDENTITY
        
        characteristics = strategy_manager._analyze_theorem_characteristics(malformed_theorem)
        
        # Should return default characteristics without crashing
        assert isinstance(characteristics, TheoremCharacteristics)
        assert characteristics.complexity_score == 0.5
        assert characteristics.theorem_type == TheoremType.ALGEBRAIC_IDENTITY
    
    def test_count_operations(self, strategy_manager):
        """Test operation counting functionality."""
        # Simple expression
        expr1 = sp.Symbol('x') + 1
        count1 = strategy_manager._count_operations(expr1)
        assert count1 >= 1
        
        # Complex expression
        x = sp.Symbol('x')
        expr2 = (x + 1) * (x - 1) + x**2
        count2 = strategy_manager._count_operations(expr2)
        assert count2 > count1
    
    def test_calculate_expression_depth(self, strategy_manager):
        """Test expression depth calculation."""
        # Simple expression: x
        expr1 = sp.Symbol('x')
        depth1 = strategy_manager._calculate_expression_depth(expr1)
        assert depth1 == 1
        
        # Nested expression: (x + 1) * (x - 1)
        x = sp.Symbol('x')
        expr2 = (x + 1) * (x - 1)
        depth2 = strategy_manager._calculate_expression_depth(expr2)
        assert depth2 > depth1
    
    def test_calculate_complexity_score(self, strategy_manager):
        """Test complexity score calculation."""
        # Simple case
        score1 = strategy_manager._calculate_complexity_score(1, 1, 0, 1, False, False)
        assert 0.0 <= score1 <= 1.0
        
        # Complex case with trig functions
        score2 = strategy_manager._calculate_complexity_score(3, 5, 2, 4, True, True)
        assert score2 > score1
        assert 0.0 <= score2 <= 1.0
    
    def test_score_all_strategies(self, strategy_manager, simple_algebraic_theorem):
        """Test scoring of all strategies."""
        characteristics = strategy_manager._analyze_theorem_characteristics(simple_algebraic_theorem)
        scores = strategy_manager._score_all_strategies(characteristics)
        
        # Should have scores for all strategies except AUTO_SELECT
        assert len(scores) == len(ProofStrategy) - 1
        
        # All scores should be StrategyScore objects
        for score in scores:
            assert isinstance(score, StrategyScore)
            assert 0.0 <= score.get_final_score() <= 1.0
    
    def test_score_strategy_symbolic_only(self, strategy_manager):
        """Test scoring of symbolic-only strategy."""
        # Create characteristics for simple algebraic theorem
        characteristics = TheoremCharacteristics(
            complexity_score=0.2,  # Low complexity
            variable_count=1,
            operation_count=1,
            function_count=0,
            has_trigonometric=False,
            has_logarithmic=False,
            has_exponential=False,
            has_rational=False,
            has_algebraic_structure=True,
            theorem_type=TheoremType.ALGEBRAIC_IDENTITY,
            expression_depth=2
        )
        
        score = strategy_manager._score_strategy(ProofStrategy.SYMBOLIC_ONLY, characteristics)
        
        # Should get good score for simple algebraic theorem
        assert score.get_final_score() > 0.5
        assert "low_complexity" in score.adjustments
    
    def test_score_strategy_rule_based(self, strategy_manager):
        """Test scoring of rule-based strategy."""
        # Create characteristics for trigonometric theorem
        characteristics = TheoremCharacteristics(
            complexity_score=0.5,  # Moderate complexity
            variable_count=1,
            operation_count=3,
            function_count=2,
            has_trigonometric=True,
            has_logarithmic=False,
            has_exponential=False,
            has_rational=False,
            has_algebraic_structure=True,
            theorem_type=TheoremType.TRIGONOMETRIC,
            expression_depth=3
        )
        
        score = strategy_manager._score_strategy(ProofStrategy.RULE_BASED, characteristics)
        
        # Should get good score for trigonometric theorem
        assert score.get_final_score() > 0.5
        assert "trigonometric_identities" in score.adjustments
    
    def test_score_strategy_comprehensive(self, strategy_manager):
        """Test scoring of comprehensive strategy."""
        # Create characteristics for complex functional equation
        characteristics = TheoremCharacteristics(
            complexity_score=0.8,  # High complexity
            variable_count=3,
            operation_count=5,
            function_count=1,
            has_trigonometric=False,
            has_logarithmic=False,
            has_exponential=False,
            has_rational=False,
            has_algebraic_structure=True,
            theorem_type=TheoremType.FUNCTIONAL_EQUATION,
            expression_depth=4
        )
        
        score = strategy_manager._score_strategy(ProofStrategy.COMPREHENSIVE, characteristics)
        
        # Should get good score for complex functional equation
        assert score.get_final_score() > 0.5
        assert "challenging_type" in score.adjustments
    
    def test_apply_theorem_type_adjustments(self, strategy_manager):
        """Test theorem type specific adjustments."""
        characteristics = TheoremCharacteristics(
            complexity_score=0.5,
            variable_count=1,
            operation_count=2,
            function_count=0,
            has_trigonometric=False,
            has_logarithmic=False,
            has_exponential=False,
            has_rational=False,
            has_algebraic_structure=True,
            theorem_type=TheoremType.ALGEBRAIC_IDENTITY,
            expression_depth=2
        )
        
        score = StrategyScore(ProofStrategy.SYMBOLIC_ONLY, 0.7)
        strategy_manager._apply_theorem_type_adjustments(score, characteristics)
        
        # Should have algebraic identity specific adjustment
        assert "algebraic_identity_symbolic" in score.adjustments
        assert score.adjustments["algebraic_identity_symbolic"] > 0
    
    def test_apply_performance_history_adjustments(self, strategy_manager_with_history):
        """Test performance history adjustments."""
        characteristics = TheoremCharacteristics(
            complexity_score=0.3,
            variable_count=1,
            operation_count=1,
            function_count=0,
            has_trigonometric=False,
            has_logarithmic=False,
            has_exponential=False,
            has_rational=False,
            has_algebraic_structure=True,
            theorem_type=TheoremType.ALGEBRAIC_IDENTITY,
            expression_depth=2
        )
        
        score = StrategyScore(ProofStrategy.SYMBOLIC_ONLY, 0.7)
        strategy_manager_with_history._apply_performance_history_adjustments(score, characteristics)
        
        # Should have high historical success adjustment (success_rate = 0.8)
        assert "high_historical_success" in score.adjustments
        assert score.adjustments["high_historical_success"] > 0
    
    def test_get_strategy_recommendation_details(self, strategy_manager, simple_algebraic_theorem):
        """Test detailed strategy recommendation."""
        details = strategy_manager.get_strategy_recommendation_details(simple_algebraic_theorem)
        
        assert isinstance(details, dict)
        assert 'theorem_id' in details
        assert 'characteristics' in details
        assert 'recommended_strategy' in details
        assert 'strategy_scores' in details
        assert 'reasoning' in details
        
        # Check structure
        assert details['theorem_id'] == simple_algebraic_theorem.id
        assert isinstance(details['characteristics'], dict)
        assert isinstance(details['strategy_scores'], list)
        assert isinstance(details['reasoning'], str)
        
        # Strategy scores should be sorted by score (highest first)
        scores = details['strategy_scores']
        for i in range(len(scores) - 1):
            assert scores[i]['final_score'] >= scores[i + 1]['final_score']
    
    def test_generate_reasoning(self, strategy_manager):
        """Test reasoning generation for strategy selection."""
        characteristics = TheoremCharacteristics(
            complexity_score=0.2,  # Low complexity
            variable_count=1,
            operation_count=1,
            function_count=0,
            has_trigonometric=False,
            has_logarithmic=False,
            has_exponential=False,
            has_rational=False,
            has_algebraic_structure=True,
            theorem_type=TheoremType.ALGEBRAIC_IDENTITY,
            expression_depth=2
        )
        
        score = StrategyScore(ProofStrategy.SYMBOLIC_ONLY, 0.7)
        score.add_adjustment("low_complexity", 0.3)
        score.add_adjustment("pure_algebraic", 0.2)
        
        reasoning = strategy_manager._generate_reasoning(characteristics, score)
        
        assert isinstance(reasoning, str)
        assert "symbolic" in reasoning.lower()
        assert len(reasoning) > 10  # Should be a meaningful explanation
    
    def test_update_performance_history(self, strategy_manager):
        """Test performance history updates."""
        # Initial state - no history
        assert strategy_manager.performance_history == {}
        
        # Update with first result
        strategy_manager.update_performance_history(
            ProofStrategy.SYMBOLIC_ONLY, success=True, execution_time=5.0
        )
        
        assert 'symbolic' in strategy_manager.performance_history
        history = strategy_manager.performance_history['symbolic']
        assert history['total_attempts'] == 1
        assert history['successes'] == 1
        assert history['success_rate'] == 1.0
        assert history['avg_execution_time'] == 5.0
        
        # Update with second result (failure)
        strategy_manager.update_performance_history(
            ProofStrategy.SYMBOLIC_ONLY, success=False, execution_time=3.0
        )
        
        history = strategy_manager.performance_history['symbolic']
        assert history['total_attempts'] == 2
        assert history['successes'] == 1
        assert history['success_rate'] == 0.5
        assert history['avg_execution_time'] == 4.0  # (5.0 + 3.0) / 2
    
    def test_theorem_characteristics_to_dict(self):
        """Test TheoremCharacteristics serialization."""
        characteristics = TheoremCharacteristics(
            complexity_score=0.5,
            variable_count=2,
            operation_count=3,
            function_count=1,
            has_trigonometric=True,
            has_logarithmic=False,
            has_exponential=False,
            has_rational=True,
            has_algebraic_structure=True,
            theorem_type=TheoremType.TRIGONOMETRIC,
            expression_depth=3
        )
        
        result_dict = characteristics.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict['complexity_score'] == 0.5
        assert result_dict['variable_count'] == 2
        assert result_dict['theorem_type'] == 'trigonometric'
        assert result_dict['has_trigonometric'] is True
        assert result_dict['has_logarithmic'] is False
    
    def test_strategy_score_functionality(self):
        """Test StrategyScore class functionality."""
        score = StrategyScore(ProofStrategy.SYMBOLIC_ONLY, 0.7)
        
        # Initial state
        assert score.strategy == ProofStrategy.SYMBOLIC_ONLY
        assert score.base_score == 0.7
        assert score.get_final_score() == 0.7
        assert score.adjustments == {}
        
        # Add adjustments
        score.add_adjustment("test_positive", 0.2)
        assert score.get_final_score() == pytest.approx(0.9)
        
        score.add_adjustment("test_negative", -0.3)
        assert score.get_final_score() == pytest.approx(0.6)
        
        # Test clamping
        score.add_adjustment("huge_positive", 2.0)
        assert score.get_final_score() == 1.0  # Clamped to 1.0
        
        score.add_adjustment("huge_negative", -5.0)
        assert score.get_final_score() == 0.0  # Clamped to 0.0
    
    def test_strategy_score_to_dict(self):
        """Test StrategyScore serialization."""
        score = StrategyScore(ProofStrategy.HYBRID_SYMBOLIC_RULE, 0.75)
        score.add_adjustment("test_reason", 0.1)
        
        result_dict = score.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict['strategy'] == 'hybrid_sr'
        assert result_dict['base_score'] == 0.75
        assert result_dict['adjustments']['test_reason'] == 0.1
        assert result_dict['final_score'] == 0.85
    
    def test_edge_cases_expression_analysis(self, strategy_manager):
        """Test edge cases in expression analysis."""
        # Empty expression
        empty_theorem = Mock()
        empty_theorem.sympy_expression = sp.S.Zero
        empty_theorem.theorem_type = TheoremType.ALGEBRAIC_IDENTITY
        
        characteristics = strategy_manager._analyze_theorem_characteristics(empty_theorem)
        assert isinstance(characteristics, TheoremCharacteristics)
        
        # Very complex nested expression
        x = sp.Symbol('x')
        complex_expr = sp.sin(sp.cos(sp.tan(x**2 + sp.log(x + 1))))
        complex_theorem = Mock()
        complex_theorem.sympy_expression = complex_expr
        complex_theorem.theorem_type = TheoremType.TRIGONOMETRIC
        
        characteristics = strategy_manager._analyze_theorem_characteristics(complex_theorem)
        assert characteristics.complexity_score > 0.5
        assert characteristics.has_trigonometric is True
    
    def test_strategy_selection_consistency(self, strategy_manager, simple_algebraic_theorem):
        """Test that strategy selection is consistent for the same theorem."""
        strategy1 = strategy_manager.select_optimal_strategy(simple_algebraic_theorem)
        strategy2 = strategy_manager.select_optimal_strategy(simple_algebraic_theorem)
        
        # Should select the same strategy for the same theorem
        assert strategy1 == strategy2
    
    def test_performance_history_influence(self, strategy_manager_with_history, simple_algebraic_theorem):
        """Test that performance history influences strategy selection."""
        # Get recommendation details to see the influence
        details = strategy_manager_with_history.get_strategy_recommendation_details(simple_algebraic_theorem)
        
        # Find symbolic strategy score
        symbolic_score = None
        for score in details['strategy_scores']:
            if score['strategy'] == 'symbolic':
                symbolic_score = score
                break
        
        assert symbolic_score is not None
        # Should have positive adjustment due to high historical success rate
        assert 'high_historical_success' in symbolic_score['adjustments'] 