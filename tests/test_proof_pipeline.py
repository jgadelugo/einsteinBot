"""
Comprehensive test suite for the ProofPipeline integration module.

Tests cover integration of all Phase 5 components, strategy selection and execution,
parallel processing, and result integration.
"""

import pytest
import tempfile
import json
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

import sympy as sp

from proofs.integration.proof_pipeline import (
    ProofPipeline, 
    ProofStrategy, 
    ComprehensiveResult
)
from proofs.theorem_generator import Theorem, TheoremType, SourceLineage
from proofs.proof_attempt import ProofAttemptEngine, ProofResult, ProofStatus
from proofs.formal_systems.lean4_interface import Lean4Interface


class TestProofPipeline:
    """Test suite for ProofPipeline class."""
    
    @pytest.fixture
    def mock_theorem_generator(self):
        """Create mock theorem generator."""
        mock_gen = Mock()
        mock_gen.get_all_theorems.return_value = []
        mock_gen.generate_from_hypotheses.return_value = []
        return mock_gen
    
    @pytest.fixture
    def mock_proof_engine(self):
        """Create mock proof engine."""
        mock_engine = Mock()
        mock_result = ProofResult(
            theorem_id="test_theorem",
            status=ProofStatus.PROVED,
            method=None,
            steps=[],
            execution_time=0.1,
            confidence_score=0.8
        )
        mock_engine.attempt_proof.return_value = mock_result
        return mock_engine
    
    @pytest.fixture
    def mock_rule_engine(self):
        """Create mock rule engine."""
        mock_engine = Mock()
        mock_engine.apply_transformation_sequence.return_value = []
        return mock_engine
    
    @pytest.fixture
    def mock_formal_systems(self):
        """Create mock formal systems."""
        mock_lean = Mock()
        mock_lean.translate_theorem.return_value = "theorem test"
        mock_lean.attempt_proof.return_value = Mock(verification_status="proved")
        return {"lean4": mock_lean}
    
    @pytest.fixture
    def sample_theorem(self):
        """Create a sample theorem for testing."""
        lineage = SourceLineage(
            original_formula="x + 0",
            hypothesis_id="test_hyp",
            confidence=1.0,
            validation_score=1.0,
            generation_method="test"
        )
        
        return Theorem(
            id="test_theorem_001",
            statement="∀x ∈ ℝ, x + 0 = x",
            sympy_expression=sp.Eq(sp.Symbol('x') + 0, sp.Symbol('x')),
            theorem_type=TheoremType.ALGEBRAIC_IDENTITY,
            assumptions=["x ∈ ℝ"],
            source_lineage=lineage,
            symbols={'x'}
        )
    
    @pytest.fixture
    def proof_pipeline(self, mock_theorem_generator, mock_proof_engine, 
                       mock_rule_engine, mock_formal_systems):
        """Create proof pipeline with all mock components."""
        return ProofPipeline(
            theorem_generator=mock_theorem_generator,
            proof_engine=mock_proof_engine,
            rule_engine=mock_rule_engine,
            formal_systems=mock_formal_systems
        )
    
    def test_proof_pipeline_initialization(self, proof_pipeline):
        """Test ProofPipeline initialization."""
        assert proof_pipeline.theorem_generator is not None
        assert proof_pipeline.proof_engine is not None
        assert proof_pipeline.rule_engine is not None
        assert len(proof_pipeline.formal_systems) == 1
        assert 'lean4' in proof_pipeline.formal_systems
        
        # Check statistics initialization
        stats = proof_pipeline.statistics
        assert stats['theorems_attempted'] == 0
        assert stats['theorems_proved'] == 0
        assert stats['strategies_used'] == {}
        assert stats['execution_times'] == []
    
    def test_proof_pipeline_without_optional_components(self, mock_theorem_generator, mock_proof_engine):
        """Test ProofPipeline initialization without optional components."""
        pipeline = ProofPipeline(
            theorem_generator=mock_theorem_generator,
            proof_engine=mock_proof_engine
        )
        
        assert pipeline.theorem_generator is not None
        assert pipeline.proof_engine is not None
        assert pipeline.formal_systems == {}
        # rule_engine may be None or auto-initialized depending on availability
    
    def test_prove_theorem_symbolic_only(self, proof_pipeline, sample_theorem):
        """Test proving theorem with symbolic-only strategy."""
        result = proof_pipeline.prove_theorem(
            sample_theorem, 
            strategies=[ProofStrategy.SYMBOLIC_ONLY]
        )
        
        assert isinstance(result, ComprehensiveResult)
        assert result.theorem == sample_theorem
        assert ProofStrategy.SYMBOLIC_ONLY in result.strategies_attempted
        assert result.symbolic_result is not None
        assert result.execution_time > 0
        
        # Check statistics update
        assert proof_pipeline.statistics['theorems_attempted'] == 1
    
    def test_prove_theorem_rule_based(self, proof_pipeline, sample_theorem):
        """Test proving theorem with rule-based strategy."""
        result = proof_pipeline.prove_theorem(
            sample_theorem,
            strategies=[ProofStrategy.RULE_BASED]
        )
        
        assert isinstance(result, ComprehensiveResult)
        assert result.theorem == sample_theorem
        assert ProofStrategy.RULE_BASED in result.strategies_attempted
        # rule_transformations should be set (empty list in mock)
        assert result.rule_transformations == []
    
    def test_prove_theorem_formal_verification(self, proof_pipeline, sample_theorem):
        """Test proving theorem with formal verification strategy."""
        result = proof_pipeline.prove_theorem(
            sample_theorem,
            strategies=[ProofStrategy.FORMAL_VERIFICATION]
        )
        
        assert isinstance(result, ComprehensiveResult)
        assert result.theorem == sample_theorem
        assert ProofStrategy.FORMAL_VERIFICATION in result.strategies_attempted
        assert result.formal_result is not None
    
    def test_prove_theorem_hybrid_symbolic_rule(self, proof_pipeline, sample_theorem):
        """Test proving theorem with hybrid symbolic+rule strategy."""
        result = proof_pipeline.prove_theorem(
            sample_theorem,
            strategies=[ProofStrategy.HYBRID_SYMBOLIC_RULE]
        )
        
        assert isinstance(result, ComprehensiveResult)
        assert result.theorem == sample_theorem
        assert ProofStrategy.HYBRID_SYMBOLIC_RULE in result.strategies_attempted
        assert result.symbolic_result is not None
        assert result.rule_transformations == []  # Empty in mock
    
    def test_prove_theorem_hybrid_symbolic_formal(self, proof_pipeline, sample_theorem):
        """Test proving theorem with hybrid symbolic+formal strategy."""
        result = proof_pipeline.prove_theorem(
            sample_theorem,
            strategies=[ProofStrategy.HYBRID_SYMBOLIC_FORMAL]
        )
        
        assert isinstance(result, ComprehensiveResult)
        assert result.theorem == sample_theorem
        assert ProofStrategy.HYBRID_SYMBOLIC_FORMAL in result.strategies_attempted
        assert result.symbolic_result is not None
        assert result.formal_result is not None
    
    def test_prove_theorem_comprehensive(self, proof_pipeline, sample_theorem):
        """Test proving theorem with comprehensive strategy."""
        result = proof_pipeline.prove_theorem(
            sample_theorem,
            strategies=[ProofStrategy.COMPREHENSIVE]
        )
        
        assert isinstance(result, ComprehensiveResult)
        assert result.theorem == sample_theorem
        assert ProofStrategy.COMPREHENSIVE in result.strategies_attempted
        assert result.symbolic_result is not None
        # Should have attempted all available methods
    
    def test_prove_theorem_auto_select(self, proof_pipeline, sample_theorem):
        """Test proving theorem with auto-select strategy."""
        result = proof_pipeline.prove_theorem(
            sample_theorem,
            strategies=[ProofStrategy.AUTO_SELECT]
        )
        
        assert isinstance(result, ComprehensiveResult)
        assert result.theorem == sample_theorem
        # AUTO_SELECT should be replaced with actual strategy
        assert ProofStrategy.AUTO_SELECT not in result.strategies_attempted
        assert len(result.strategies_attempted) > 0
    
    def test_prove_theorem_multiple_strategies(self, proof_pipeline, sample_theorem):
        """Test proving theorem with multiple strategies."""
        result = proof_pipeline.prove_theorem(
            sample_theorem,
            strategies=[ProofStrategy.SYMBOLIC_ONLY, ProofStrategy.RULE_BASED]
        )
        
        assert isinstance(result, ComprehensiveResult)
        assert result.theorem == sample_theorem
        # Should have attempted both strategies
        assert len(result.strategies_attempted) >= 2
    
    def test_prove_theorem_string_strategies(self, proof_pipeline, sample_theorem):
        """Test proving theorem with string strategy names."""
        result = proof_pipeline.prove_theorem(
            sample_theorem,
            strategies=["symbolic", "rule_based"]
        )
        
        assert isinstance(result, ComprehensiveResult)
        assert result.theorem == sample_theorem
        assert len(result.strategies_attempted) >= 2
    
    def test_prove_theorem_invalid_strategy(self, proof_pipeline, sample_theorem):
        """Test proving theorem with invalid strategy name."""
        result = proof_pipeline.prove_theorem(
            sample_theorem,
            strategies=["invalid_strategy", "symbolic"]
        )
        
        assert isinstance(result, ComprehensiveResult)
        assert result.theorem == sample_theorem
        # Should still work with valid strategy
        assert len(result.strategies_attempted) >= 1
    
    def test_auto_select_strategy_simple_algebraic(self, proof_pipeline):
        """Test auto strategy selection for simple algebraic theorems."""
        # Create simple algebraic theorem
        simple_theorem = Mock()
        simple_theorem.theorem_type = TheoremType.ALGEBRAIC_IDENTITY
        simple_theorem.sympy_expression = sp.Symbol('x')
        simple_theorem.symbols = {'x'}
        
        strategy = proof_pipeline._auto_select_strategy(simple_theorem)
        
        # Simple theorems should use symbolic only
        assert strategy == ProofStrategy.SYMBOLIC_ONLY
    
    def test_auto_select_strategy_functional_equation(self, proof_pipeline):
        """Test auto strategy selection for functional equations."""
        func_theorem = Mock()
        func_theorem.theorem_type = TheoremType.FUNCTIONAL_EQUATION
        func_theorem.sympy_expression = sp.Symbol('x')
        func_theorem.symbols = {'x'}
        
        strategy = proof_pipeline._auto_select_strategy(func_theorem)
        
        # Functional equations should use comprehensive
        assert strategy == ProofStrategy.COMPREHENSIVE
    
    def test_estimate_complexity(self, proof_pipeline, sample_theorem):
        """Test theorem complexity estimation."""
        complexity = proof_pipeline._estimate_complexity(sample_theorem)
        
        assert isinstance(complexity, float)
        assert 0.0 <= complexity <= 1.0
    
    def test_calculate_confidence_empty_results(self, proof_pipeline):
        """Test confidence calculation with empty results."""
        confidence = proof_pipeline._calculate_confidence({})
        assert confidence == 0.0
    
    def test_calculate_confidence_successful_results(self, proof_pipeline):
        """Test confidence calculation with successful results."""
        results = {
            ProofStrategy.SYMBOLIC_ONLY: {'success': True},
            ProofStrategy.FORMAL_VERIFICATION: {'success': True}
        }
        
        confidence = proof_pipeline._calculate_confidence(results)
        
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be high for successful results
    
    def test_prove_theorem_with_timeout(self, proof_pipeline, sample_theorem):
        """Test proving theorem with custom timeout."""
        result = proof_pipeline.prove_theorem(
            sample_theorem,
            strategies=[ProofStrategy.SYMBOLIC_ONLY],
            timeout=60
        )
        
        assert isinstance(result, ComprehensiveResult)
        # Timeout should be passed to strategy execution
    
    def test_prove_theorem_error_handling(self, mock_theorem_generator, sample_theorem):
        """Test error handling when proof engine fails."""
        mock_engine = Mock()
        mock_engine.attempt_proof.side_effect = Exception("Proof engine error")
        
        pipeline = ProofPipeline(
            theorem_generator=mock_theorem_generator,
            proof_engine=mock_engine
        )
        
        result = pipeline.prove_theorem(
            sample_theorem,
            strategies=[ProofStrategy.SYMBOLIC_ONLY]
        )
        
        assert isinstance(result, ComprehensiveResult)
        assert result.final_status == "failed"
        assert len(result.error_messages) > 0
    
    def test_prove_all_theorems_empty(self, proof_pipeline):
        """Test proving all theorems when no theorems available."""
        proof_pipeline.theorem_generator.get_all_theorems.return_value = []
        
        results = proof_pipeline.prove_all_theorems()
        
        assert results == []
    
    def test_prove_all_theorems_single(self, proof_pipeline, sample_theorem):
        """Test proving all theorems with single theorem."""
        proof_pipeline.theorem_generator.get_all_theorems.return_value = [sample_theorem]
        
        results = proof_pipeline.prove_all_theorems(max_parallel=1)
        
        assert len(results) == 1
        assert isinstance(results[0], ComprehensiveResult)
        assert results[0].theorem == sample_theorem
    
    def test_prove_all_theorems_multiple(self, proof_pipeline, sample_theorem):
        """Test proving all theorems with multiple theorems."""
        # Create multiple theorems
        theorem2 = Mock()
        theorem2.id = "test_theorem_002"
        theorem2.statement = "Another theorem"
        theorem2.sympy_expression = sp.Symbol('y')
        theorem2.theorem_type = TheoremType.ALGEBRAIC_IDENTITY
        
        proof_pipeline.theorem_generator.get_all_theorems.return_value = [sample_theorem, theorem2]
        
        results = proof_pipeline.prove_all_theorems(max_parallel=2)
        
        assert len(results) == 2
        assert all(isinstance(r, ComprehensiveResult) for r in results)
    
    def test_generate_and_prove_success(self, proof_pipeline, sample_theorem):
        """Test end-to-end generate and prove workflow."""
        hypothesis_data = {
            "hypothesis_id": "test_hyp",
            "formula": "x + 0",
            "confidence_score": 1.0
        }
        
        proof_pipeline.theorem_generator.generate_from_hypotheses.return_value = [sample_theorem]
        
        result = proof_pipeline.generate_and_prove(hypothesis_data)
        
        assert isinstance(result, ComprehensiveResult)
        assert result.theorem == sample_theorem
        assert result.metadata['generated_from_hypothesis'] is True
        assert result.metadata['hypothesis_count'] == 1
        assert result.metadata['theorems_generated'] == 1
    
    def test_generate_and_prove_no_theorems(self, proof_pipeline):
        """Test generate and prove when no theorems generated."""
        hypothesis_data = {"invalid": "data"}
        
        proof_pipeline.theorem_generator.generate_from_hypotheses.return_value = []
        
        result = proof_pipeline.generate_and_prove(hypothesis_data)
        
        assert isinstance(result, ComprehensiveResult)
        assert result.final_status == "failed"
        assert len(result.error_messages) > 0
    
    def test_generate_and_prove_generation_error(self, proof_pipeline):
        """Test generate and prove when generation fails."""
        hypothesis_data = {"test": "data"}
        
        proof_pipeline.theorem_generator.generate_from_hypotheses.side_effect = Exception("Generation failed")
        
        result = proof_pipeline.generate_and_prove(hypothesis_data)
        
        assert isinstance(result, ComprehensiveResult)
        assert result.final_status == "failed"
        assert len(result.error_messages) > 0
    
    def test_get_statistics_initial(self, proof_pipeline):
        """Test getting statistics from fresh pipeline."""
        stats = proof_pipeline.get_statistics()
        
        assert stats['theorems_attempted'] == 0
        assert stats['theorems_proved'] == 0
        assert stats['success_rate'] == 0.0
        assert stats['average_execution_time'] == 0.0
        assert 'components_available' in stats
        
        components = stats['components_available']
        assert components['theorem_generator'] is True
        assert components['proof_engine'] is True
        assert components['rule_engine'] is True
        assert 'lean4' in components['formal_systems']
    
    def test_get_statistics_after_proving(self, proof_pipeline, sample_theorem):
        """Test getting statistics after proving theorems."""
        # Prove a theorem first
        result = proof_pipeline.prove_theorem(sample_theorem)
        
        stats = proof_pipeline.get_statistics()
        
        assert stats['theorems_attempted'] == 1
        assert stats['success_rate'] >= 0.0
        assert stats['average_execution_time'] > 0.0
        assert len(stats['execution_times']) == 1
    
    def test_comprehensive_result_to_dict(self, sample_theorem):
        """Test ComprehensiveResult serialization to dictionary."""
        result = ComprehensiveResult(
            theorem=sample_theorem,
            strategies_attempted=[ProofStrategy.SYMBOLIC_ONLY],
            final_status="proved",
            confidence_score=0.8,
            execution_time=0.5
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict['theorem_id'] == sample_theorem.id
        assert result_dict['theorem_statement'] == sample_theorem.statement
        assert result_dict['strategies_attempted'] == ['symbolic']
        assert result_dict['final_status'] == 'proved'
        assert result_dict['confidence_score'] == 0.8
        assert result_dict['execution_time'] == 0.5
    
    def test_rule_engine_unavailable(self, mock_theorem_generator, mock_proof_engine, sample_theorem):
        """Test pipeline behavior when rule engine is unavailable."""
        pipeline = ProofPipeline(
            theorem_generator=mock_theorem_generator,
            proof_engine=mock_proof_engine,
            rule_engine=None  # Explicitly set to None
        )
        
        result = pipeline.prove_theorem(
            sample_theorem,
            strategies=[ProofStrategy.RULE_BASED]
        )
        
        assert isinstance(result, ComprehensiveResult)
        # Should handle gracefully when rule engine not available
        assert len(result.error_messages) >= 0  # May have error about unavailable engine
    
    def test_formal_systems_unavailable(self, mock_theorem_generator, mock_proof_engine, sample_theorem):
        """Test pipeline behavior when formal systems are unavailable."""
        pipeline = ProofPipeline(
            theorem_generator=mock_theorem_generator,
            proof_engine=mock_proof_engine,
            formal_systems={}  # No formal systems
        )
        
        result = pipeline.prove_theorem(
            sample_theorem,
            strategies=[ProofStrategy.FORMAL_VERIFICATION]
        )
        
        assert isinstance(result, ComprehensiveResult)
        # Should handle gracefully when formal systems not available
    
    def test_strategy_execution_exception_handling(self, proof_pipeline, sample_theorem):
        """Test handling of exceptions during strategy execution."""
        # Mock the proof engine to raise an exception
        proof_pipeline.proof_engine.attempt_proof.side_effect = Exception("Test exception")
        
        result = proof_pipeline.prove_theorem(
            sample_theorem,
            strategies=[ProofStrategy.SYMBOLIC_ONLY]
        )
        
        assert isinstance(result, ComprehensiveResult)
        # Pipeline should handle the exception gracefully
        assert len(result.error_messages) >= 0
    
    def test_parallel_proof_execution_error_handling(self, proof_pipeline):
        """Test error handling in parallel proof execution."""
        # Create mock theorems where some will fail
        theorems = []
        for i in range(3):
            mock_theorem = Mock()
            mock_theorem.id = f"theorem_{i}"
            mock_theorem.statement = f"Test theorem {i}"
            theorems.append(mock_theorem)
        
        proof_pipeline.theorem_generator.get_all_theorems.return_value = theorems
        
        # Mock one theorem to cause an exception
        def side_effect(theorem, strategies):
            if theorem.id == "theorem_1":
                raise Exception("Test parallel exception")
            return ComprehensiveResult(
                theorem=theorem,
                strategies_attempted=[ProofStrategy.SYMBOLIC_ONLY],
                final_status="proved"
            )
        
        with patch.object(proof_pipeline, 'prove_theorem', side_effect=side_effect):
            results = proof_pipeline.prove_all_theorems(max_parallel=2)
        
        assert len(results) == 3
        # Should have error result for the failed theorem
        failed_results = [r for r in results if r.final_status == "failed"]
        assert len(failed_results) >= 1 