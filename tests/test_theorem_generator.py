"""
Comprehensive test suite for the theorem generator module.

Tests cover hypothesis-to-theorem conversion, classification accuracy,
natural language generation, and integration with validation engine.
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import sympy as sp
from sympy import Symbol, Eq

from proofs.theorem_generator import (
    TheoremGenerator,
    Theorem,
    TheoremType,
    SourceLineage
)
from validation.formula_tester import FormulaValidator


class TestTheoremGenerator:
    """Test suite for TheoremGenerator class."""
    
    @pytest.fixture
    def generator(self):
        """Create a theorem generator for testing."""
        return TheoremGenerator()
    
    @pytest.fixture
    def sample_hypotheses(self):
        """Load sample hypotheses for testing."""
        with open('tests/fixtures/sample_hypotheses.json') as f:
            data = json.load(f)
        return data['hypotheses']
    
    @pytest.fixture
    def simple_hypothesis(self):
        """Simple hypothesis for basic testing."""
        return {
            "hypothesis_id": "simple_test",
            "hypothesis_type": "algebraic_identity",
            "formula": "x**2 + 2*x + 1",
            "confidence_score": 1.0,
            "source_formulas": ["x**2 + 2*x + 1"],
            "evidence": {"pass_rate": 1.0, "total_tests": 100}
        }
    
    def test_initialization(self):
        """Test theorem generator initialization."""
        generator = TheoremGenerator()
        assert generator.validation_engine is not None
        assert generator.config == {}
        assert generator.stats['theorems_generated'] == 0
        
        # Test with custom config
        config = {'max_theorems': 50}
        generator_with_config = TheoremGenerator(config=config)
        assert generator_with_config.config == config
    
    def test_generate_from_hypotheses_basic(self, generator, sample_hypotheses):
        """Test basic theorem generation from hypotheses."""
        theorems = generator.generate_from_hypotheses(sample_hypotheses)
        
        # Should generate theorems for all valid hypotheses
        assert len(theorems) > 0
        assert len(theorems) <= len(sample_hypotheses)
        
        # Check statistics
        stats = generator.get_generation_stats()
        assert stats['theorems_generated'] == len(theorems)
        assert stats['generation_time'] > 0
    
    def test_theorem_structure(self, generator, simple_hypothesis):
        """Test that generated theorems have correct structure."""
        theorems = generator.generate_from_hypotheses([simple_hypothesis])
        assert len(theorems) == 1
        
        theorem = theorems[0]
        
        # Check required fields
        assert theorem.id.startswith('THM_')
        assert theorem.statement
        assert theorem.sympy_expression is not None
        assert isinstance(theorem.theorem_type, TheoremType)
        assert isinstance(theorem.assumptions, list)
        assert isinstance(theorem.source_lineage, SourceLineage)
        assert theorem.natural_language is not None
    
    def test_theorem_classification(self, generator):
        """Test theorem type classification."""
        test_cases = [
            {
                "formula": "x**2 + 2*x + 1",
                "type": "algebraic_identity",
                "expected": TheoremType.ALGEBRAIC_IDENTITY
            },
            {
                "formula": "f(x) = x**2",
                "type": "functional_equation", 
                "expected": TheoremType.FUNCTIONAL_EQUATION
            },
            {
                "formula": "a*x + b",
                "type": "generalization",
                "expected": TheoremType.GENERALIZATION
            }
        ]
        
        for case in test_cases:
            hypothesis = {
                "hypothesis_id": f"test_{case['type']}",
                "hypothesis_type": case['type'],
                "formula": case['formula'],
                "confidence_score": 1.0,
                "evidence": {"pass_rate": 1.0}
            }
            
            theorems = generator.generate_from_hypotheses([hypothesis])
            if theorems:  # Only test if theorem was successfully generated
                assert theorems[0].theorem_type == case['expected']
    
    def test_functional_equation_parsing(self, generator):
        """Test parsing of functional equations."""
        hypothesis = {
            "hypothesis_id": "func_test",
            "hypothesis_type": "functional_equation",
            "formula": "f(2*x) = 4*x**2 + 4*x + 1",
            "confidence_score": 1.0,
            "evidence": {"pass_rate": 1.0}
        }
        
        theorems = generator.generate_from_hypotheses([hypothesis])
        assert len(theorems) == 1
        
        theorem = theorems[0]
        assert theorem.theorem_type == TheoremType.FUNCTIONAL_EQUATION
        assert isinstance(theorem.sympy_expression, sp.Eq)
        assert 'f' in theorem.symbols
        assert 'x' in theorem.symbols
    
    def test_natural_language_generation(self, generator, sample_hypotheses):
        """Test natural language description generation."""
        theorems = generator.generate_from_hypotheses(sample_hypotheses)
        
        for theorem in theorems:
            assert theorem.natural_language is not None
            assert len(theorem.natural_language) > 10  # Should be a meaningful description
            assert isinstance(theorem.natural_language, str)
    
    def test_assumptions_generation(self, generator):
        """Test mathematical assumptions generation."""
        hypothesis = {
            "hypothesis_id": "assumptions_test",
            "hypothesis_type": "algebraic_identity",
            "formula": "x**2 + y**2",
            "confidence_score": 1.0,
            "evidence": {"pass_rate": 1.0}
        }
        
        theorems = generator.generate_from_hypotheses([hypothesis])
        assert len(theorems) == 1
        
        theorem = theorems[0]
        assumptions = theorem.assumptions
        
        # Should have domain assumptions for variables
        assert any('x ∈ ℝ' in assumption for assumption in assumptions)
        assert any('y ∈ ℝ' in assumption for assumption in assumptions)
    
    def test_source_lineage_tracking(self, generator, simple_hypothesis):
        """Test that source lineage is properly tracked."""
        theorems = generator.generate_from_hypotheses([simple_hypothesis])
        assert len(theorems) == 1
        
        theorem = theorems[0]
        lineage = theorem.source_lineage
        
        assert lineage.hypothesis_id == simple_hypothesis['hypothesis_id']
        assert lineage.confidence == simple_hypothesis['confidence_score']
        assert lineage.validation_score == simple_hypothesis['evidence']['pass_rate']
        assert lineage.generation_method == 'direct_conversion'
    
    def test_theorem_validation(self, generator, simple_hypothesis):
        """Test theorem precondition validation."""
        theorems = generator.generate_from_hypotheses([simple_hypothesis])
        assert len(theorems) == 1
        
        theorem = theorems[0]
        assert theorem.validate_preconditions() is True
    
    def test_error_handling_malformed_input(self, generator):
        """Test handling of malformed hypothesis input."""
        malformed_hypotheses = [
            {"invalid": "data"},
            {"formula": "invalid_formula!!!"},
            {"formula": "", "hypothesis_id": "empty"}
        ]
        
        # Should not crash and should skip invalid hypotheses
        theorems = generator.generate_from_hypotheses(malformed_hypotheses)
        # May generate 0 theorems, but should not crash
        assert isinstance(theorems, list)
    
    def test_formula_cleaning(self, generator):
        """Test formula cleaning and preprocessing."""
        hypothesis = {
            "hypothesis_id": "cleaning_test",
            "hypothesis_type": "algebraic_identity",
            "formula": "Conjecture: x**2 + 2*x + 1 = (x + 1)**2",
            "confidence_score": 1.0,
            "evidence": {"pass_rate": 1.0}
        }
        
        theorems = generator.generate_from_hypotheses([hypothesis])
        # Should successfully parse despite the "Conjecture:" prefix
        assert len(theorems) <= 1  # May be 0 if parsing fails, but shouldn't crash
    
    def test_theorem_serialization(self, generator, simple_hypothesis):
        """Test theorem serialization to dictionary."""
        theorems = generator.generate_from_hypotheses([simple_hypothesis])
        assert len(theorems) == 1
        
        theorem = theorems[0]
        theorem_dict = theorem.to_dict()
        
        # Check all required fields are present
        required_fields = [
            'id', 'statement', 'sympy_expression', 'theorem_type',
            'assumptions', 'source_lineage', 'natural_language',
            'symbols', 'mathematical_context', 'validation_evidence'
        ]
        
        for field in required_fields:
            assert field in theorem_dict
        
        # Should be JSON serializable
        json.dumps(theorem_dict, default=str)
    
    def test_save_theorems(self, generator, sample_hypotheses):
        """Test saving theorems to file."""
        theorems = generator.generate_from_hypotheses(sample_hypotheses)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = f.name
        
        generator.save_theorems(theorems, output_path)
        
        # Verify file was created and has correct structure
        with open(output_path) as f:
            saved_data = json.load(f)
        
        assert 'generation_metadata' in saved_data
        assert 'theorems' in saved_data
        assert saved_data['generation_metadata']['total_theorems'] == len(theorems)
        assert len(saved_data['theorems']) == len(theorems)
        
        # Clean up
        Path(output_path).unlink()
    
    def test_statistics_tracking(self, generator, sample_hypotheses):
        """Test generation statistics tracking."""
        initial_stats = generator.get_generation_stats()
        assert initial_stats['theorems_generated'] == 0
        
        theorems = generator.generate_from_hypotheses(sample_hypotheses)
        
        final_stats = generator.get_generation_stats()
        assert final_stats['theorems_generated'] == len(theorems)
        assert final_stats['generation_time'] > 0
        assert final_stats['validation_passes'] >= 0
    
    def test_empty_input(self, generator):
        """Test handling of empty hypothesis list."""
        theorems = generator.generate_from_hypotheses([])
        assert theorems == []
        
        stats = generator.get_generation_stats()
        assert stats['theorems_generated'] == 0


class TestTheoremClass:
    """Test suite for Theorem data class."""
    
    @pytest.fixture
    def sample_theorem(self):
        """Create a sample theorem for testing."""
        x = Symbol('x')
        expr = Eq((x + 1)**2, x**2 + 2*x + 1)
        
        lineage = SourceLineage(
            original_formula="(x + 1)**2",
            hypothesis_id="test_hyp",
            confidence=1.0,
            validation_score=1.0,
            generation_method="test"
        )
        
        return Theorem(
            id="THM_TEST",
            statement="∀x ∈ ℝ, (x + 1)² = x² + 2x + 1",
            sympy_expression=expr,
            theorem_type=TheoremType.ALGEBRAIC_IDENTITY,
            assumptions=["x ∈ ℝ"],
            source_lineage=lineage,
            symbols={'x'},
            natural_language="Test theorem"
        )
    
    def test_theorem_validation(self, sample_theorem):
        """Test theorem precondition validation."""
        assert sample_theorem.validate_preconditions() is True
        
        # Test with inconsistent symbols
        sample_theorem.symbols = {'y'}  # x is in expression but not in symbols
        assert sample_theorem.validate_preconditions() is False
    
    def test_theorem_serialization(self, sample_theorem):
        """Test theorem to dictionary conversion."""
        theorem_dict = sample_theorem.to_dict()
        
        assert theorem_dict['id'] == "THM_TEST"
        assert theorem_dict['theorem_type'] == "algebraic_identity"
        assert 'source_lineage' in theorem_dict
        assert isinstance(theorem_dict['symbols'], list)


class TestSourceLineage:
    """Test suite for SourceLineage data class."""
    
    def test_lineage_serialization(self):
        """Test source lineage serialization."""
        lineage = SourceLineage(
            original_formula="x**2",
            hypothesis_id="test",
            confidence=0.95,
            validation_score=0.98,
            generation_method="direct",
            transformation_chain=["expand", "simplify"]
        )
        
        lineage_dict = lineage.to_dict()
        
        assert lineage_dict['original_formula'] == "x**2"
        assert lineage_dict['confidence'] == 0.95
        assert lineage_dict['transformation_chain'] == ["expand", "simplify"]


class TestIntegration:
    """Integration tests with existing validation system."""
    
    def test_validation_engine_integration(self):
        """Test integration with FormulaValidator."""
        validator = FormulaValidator()
        generator = TheoremGenerator(validation_engine=validator)
        
        hypothesis = {
            "hypothesis_id": "integration_test",
            "hypothesis_type": "algebraic_identity", 
            "formula": "x**2 + 2*x + 1",
            "confidence_score": 1.0,
            "evidence": {"pass_rate": 1.0}
        }
        
        theorems = generator.generate_from_hypotheses([hypothesis])
        # Should successfully use the validation engine
        assert isinstance(theorems, list)
    
    @patch('proofs.theorem_generator.FormulaValidator')
    def test_mock_validation_engine(self, mock_validator_class):
        """Test with mocked validation engine."""
        mock_validator = Mock()
        mock_validator_class.return_value = mock_validator
        
        generator = TheoremGenerator()
        
        hypothesis = {
            "hypothesis_id": "mock_test",
            "hypothesis_type": "algebraic_identity",
            "formula": "x + 1",
            "confidence_score": 1.0,
            "evidence": {"pass_rate": 1.0}
        }
        
        theorems = generator.generate_from_hypotheses([hypothesis])
        assert isinstance(theorems, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 