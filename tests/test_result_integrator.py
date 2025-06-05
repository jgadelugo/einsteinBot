"""
Test suite for the ResultIntegrator module.

Tests cover result integration, confidence calculation, analysis generation,
and export functionality.
"""

import pytest
import tempfile
import json
from unittest.mock import Mock, patch
from pathlib import Path

from proofs.integration.result_integrator import (
    ResultIntegrator,
    MethodResult,
    ResultAnalysis,
    ResultConfidenceLevel
)
from proofs.integration.proof_pipeline import ProofStrategy, ComprehensiveResult
from proofs.theorem_generator import Theorem, TheoremType, SourceLineage
from proofs.proof_attempt import ProofResult, ProofStatus


class TestResultIntegrator:
    """Test suite for ResultIntegrator class."""
    
    @pytest.fixture
    def result_integrator(self):
        """Create a ResultIntegrator for testing."""
        return ResultIntegrator()
    
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
            sympy_expression=True,
            theorem_type=TheoremType.ALGEBRAIC_IDENTITY,
            assumptions=["x ∈ ℝ"],
            source_lineage=lineage,
            symbols={'x'}
        )
    
    @pytest.fixture
    def successful_symbolic_result(self):
        """Create a successful symbolic proof result."""
        result = Mock()
        result.confidence_score = 0.85
        result.status = ProofStatus.PROVED
        result.proof_steps = ["step1", "step2", "step3"]
        return result
    
    def test_result_integrator_initialization(self, result_integrator):
        """Test ResultIntegrator initialization."""
        assert result_integrator.method_weights is not None
        assert len(result_integrator.method_weights) >= 6
        assert result_integrator.confidence_thresholds is not None
        assert len(result_integrator.confidence_thresholds) == 5
        assert result_integrator.integration_stats['total_integrations'] == 0
    
    def test_method_result_to_dict(self):
        """Test MethodResult serialization."""
        method_result = MethodResult(
            method_name="symbolic",
            success=True,
            confidence=0.8,
            execution_time=2.5,
            details={'steps': 3},
            error_message=None
        )
        
        result_dict = method_result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict['method_name'] == "symbolic"
        assert result_dict['success'] is True
        assert result_dict['confidence'] == 0.8
        assert result_dict['execution_time'] == 2.5
        assert result_dict['details'] == {'steps': 3}
        assert result_dict['error_message'] is None
    
    def test_result_analysis_to_dict(self):
        """Test ResultAnalysis serialization."""
        analysis = ResultAnalysis(
            total_methods_attempted=3,
            successful_methods=['symbolic', 'rule_based'],
            failed_methods=['formal'],
            best_method='symbolic',
            worst_method='formal',
            average_confidence=0.6,
            confidence_variance=0.1,
            total_execution_time=6.0,
            convergent_results=True
        )
        
        result_dict = analysis.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict['total_methods_attempted'] == 3
        assert result_dict['successful_methods'] == ['symbolic', 'rule_based']
        assert result_dict['failed_methods'] == ['formal']
        assert result_dict['convergent_results'] is True
    
    def test_determine_confidence_level(self, result_integrator):
        """Test confidence level determination."""
        assert result_integrator._determine_confidence_level(0.1) == ResultConfidenceLevel.VERY_LOW
        assert result_integrator._determine_confidence_level(0.3) == ResultConfidenceLevel.LOW
        assert result_integrator._determine_confidence_level(0.5) == ResultConfidenceLevel.MODERATE
        assert result_integrator._determine_confidence_level(0.7) == ResultConfidenceLevel.HIGH
        assert result_integrator._determine_confidence_level(0.9) == ResultConfidenceLevel.VERY_HIGH
        assert result_integrator._determine_confidence_level(1.0) == ResultConfidenceLevel.VERY_HIGH
    
    def test_extract_confidence_from_result_symbolic(self, result_integrator, successful_symbolic_result):
        """Test confidence extraction from symbolic results."""
        result = {
            'symbolic_result': successful_symbolic_result,
            'success': True
        }
        
        confidence = result_integrator._extract_confidence_from_result(
            ProofStrategy.SYMBOLIC_ONLY, result
        )
        
        assert confidence == 0.85  # Should use the symbolic result's confidence
    
    def test_extract_confidence_from_result_rule_based(self, result_integrator):
        """Test confidence extraction from rule-based results."""
        result = {
            'transformations': ['t1', 't2', 't3'],
            'success': True
        }
        
        confidence = result_integrator._extract_confidence_from_result(
            ProofStrategy.RULE_BASED, result
        )
        
        assert confidence > 0.5  # Should be based on number of transformations
    
    def test_determine_final_status_proved(self, result_integrator):
        """Test final status determination for proved theorems."""
        method_results = [
            MethodResult("symbolic", True, 0.85, 1.0),
            MethodResult("rule_based", True, 0.75, 2.0)
        ]
        
        status = result_integrator._determine_final_status(method_results)
        assert status == "proved"
    
    def test_determine_final_status_failed(self, result_integrator):
        """Test final status determination for failed theorems."""
        method_results = [
            MethodResult("symbolic", False, 0.1, 1.0),
            MethodResult("rule_based", False, 0.2, 2.0)
        ]
        
        status = result_integrator._determine_final_status(method_results)
        assert status == "failed"
    
    def test_calculate_integrated_confidence_empty(self, result_integrator):
        """Test integrated confidence calculation with empty results."""
        confidence = result_integrator._calculate_integrated_confidence([])
        assert confidence == 0.0
    
    def test_check_result_convergence_convergent(self, result_integrator):
        """Test convergence detection for convergent results."""
        method_results = [
            MethodResult("symbolic", True, 0.8, 1.0),
            MethodResult("rule_based", True, 0.7, 2.0),
            MethodResult("formal", True, 0.9, 3.0)
        ]
        
        convergent = result_integrator._check_result_convergence(method_results)
        assert convergent is True
    
    def test_check_result_convergence_divergent(self, result_integrator):
        """Test convergence detection for divergent results."""
        method_results = [
            MethodResult("symbolic", True, 0.8, 1.0),
            MethodResult("rule_based", False, 0.2, 2.0),
            MethodResult("formal", False, 0.1, 3.0)
        ]
        
        convergent = result_integrator._check_result_convergence(method_results)
        # With 2/3 methods failing, this is actually convergent (majority agree on failure)
        assert convergent is True
    
    def test_check_result_convergence_truly_divergent(self, result_integrator):
        """Test convergence detection for truly divergent results."""
        method_results = [
            MethodResult("symbolic", True, 0.8, 1.0),
            MethodResult("rule_based", False, 0.2, 2.0)
        ]
        
        convergent = result_integrator._check_result_convergence(method_results)
        # With 1/2 success and 1/2 failure, neither reaches 60% threshold
        assert convergent is False
    
    def test_get_integration_statistics(self, result_integrator):
        """Test integration statistics retrieval."""
        # Perform some integrations first
        method_results = [MethodResult("symbolic", True, 0.8, 1.0)]
        result_integrator._update_statistics(0.8, True, method_results)
        
        stats = result_integrator.get_integration_statistics()
        
        assert isinstance(stats, dict)
        assert 'total_integrations' in stats
        assert 'high_confidence_rate' in stats
        assert 'convergence_rate' in stats
        assert 'method_performance' in stats 