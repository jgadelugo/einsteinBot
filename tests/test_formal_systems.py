"""
Tests for formal system interfaces and translation frameworks.

This module tests the base formal system interface, translation capabilities,
and integration with SymPy mathematical expressions.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from proofs.formal_systems.base_interface import (
    FormalSystemType, 
    FormalProof, 
    FormalSystemInterface,
    TranslationError,
    ProofError,
    VerificationError
)
from proofs.formal_systems.translation.sympy_to_lean import SymPyToLean4Translator


class TestFormalSystemType:
    """Test FormalSystemType enumeration."""
    
    def test_formal_system_type_values(self):
        """Test that formal system types have correct values."""
        assert FormalSystemType.LEAN4.value == "lean4"
        assert FormalSystemType.COQ.value == "coq"
        assert FormalSystemType.ISABELLE.value == "isabelle"
        assert FormalSystemType.AGDA.value == "agda"
    
    def test_from_string(self):
        """Test creating FormalSystemType from string."""
        assert FormalSystemType.from_string("lean4") == FormalSystemType.LEAN4
        assert FormalSystemType.from_string("LEAN4") == FormalSystemType.LEAN4
        assert FormalSystemType.from_string("Coq") == FormalSystemType.COQ
        
        with pytest.raises(ValueError):
            FormalSystemType.from_string("invalid_system")
    
    def test_string_representation(self):
        """Test string representation of FormalSystemType."""
        assert str(FormalSystemType.LEAN4) == "lean4"
        assert str(FormalSystemType.COQ) == "coq"


class TestFormalProof:
    """Test FormalProof data structure."""
    
    def test_formal_proof_creation(self):
        """Test creating FormalProof objects."""
        proof = FormalProof(
            theorem_id="test_theorem",
            system_type=FormalSystemType.LEAN4,
            formal_statement="theorem test : 1 + 1 = 2",
            formal_proof="by norm_num",
            verification_status="proved"
        )
        
        assert proof.theorem_id == "test_theorem"
        assert proof.system_type == FormalSystemType.LEAN4
        assert proof.is_successful()
        assert not proof.has_errors()
        assert proof.proof_size == len("by norm_num")
    
    def test_formal_proof_with_errors(self):
        """Test FormalProof with errors."""
        proof = FormalProof(
            theorem_id="failing_theorem",
            system_type=FormalSystemType.LEAN4,
            formal_statement="theorem fail : 1 = 2",
            formal_proof="sorry",
            verification_status="failed",
            error_messages=["Type mismatch", "Proof incomplete"]
        )
        
        assert not proof.is_successful()
        assert proof.has_errors()
        assert len(proof.error_messages) == 2
    
    def test_formal_proof_summary(self):
        """Test FormalProof summary generation."""
        proof = FormalProof(
            theorem_id="summary_test",
            system_type=FormalSystemType.LEAN4,
            formal_statement="theorem test : True",
            formal_proof="trivial",
            verification_status="proved",
            compilation_time=0.5
        )
        
        summary = proof.get_summary()
        assert summary['theorem_id'] == "summary_test"
        assert summary['system'] == "lean4"
        assert summary['status'] == "proved"
        assert summary['compilation_time'] == 0.5
        assert not summary['has_errors']
    
    def test_formal_proof_to_dict(self):
        """Test FormalProof dictionary conversion."""
        proof = FormalProof(
            theorem_id="dict_test",
            system_type=FormalSystemType.LEAN4,
            formal_statement="theorem test : True",
            formal_proof="trivial",
            verification_status="proved"
        )
        
        proof_dict = proof.to_dict()
        assert proof_dict['theorem_id'] == "dict_test"
        assert proof_dict['system_type'] == "lean4"
        assert proof_dict['verification_status'] == "proved"


class MockFormalSystemInterface(FormalSystemInterface):
    """Mock implementation of FormalSystemInterface for testing."""
    
    def __init__(self, available: bool = True):
        super().__init__()
        self._available = available
    
    def _get_system_type(self) -> FormalSystemType:
        return FormalSystemType.LEAN4
    
    def is_available(self) -> bool:
        return self._available
    
    def translate_theorem(self, theorem) -> str:
        return f"-- Mock translation of {theorem}"
    
    def attempt_proof(self, formal_statement: str, timeout=None) -> FormalProof:
        return FormalProof(
            theorem_id="mock_theorem",
            system_type=FormalSystemType.LEAN4,
            formal_statement=formal_statement,
            formal_proof="by sorry",
            verification_status="proved" if "simple" in formal_statement else "failed"
        )
    
    def verify_proof(self, formal_statement: str, formal_proof: str) -> bool:
        return "sorry" not in formal_proof


class TestFormalSystemInterface:
    """Test FormalSystemInterface abstract base class."""
    
    def test_interface_initialization(self):
        """Test interface initialization and statistics."""
        interface = MockFormalSystemInterface()
        
        assert interface.system_type == FormalSystemType.LEAN4
        assert interface.is_available()
        assert interface.timeout == 30
        
        stats = interface.get_statistics()
        assert stats['total_attempts'] == 0
        assert stats['successful_proofs'] == 0
        assert stats['success_rate'] == 0.0
    
    def test_interface_proof_attempt(self):
        """Test proof attempt through interface."""
        interface = MockFormalSystemInterface()
        
        # Successful proof
        result = interface.attempt_proof("simple theorem")
        assert result.is_successful()
        
        # Failed proof
        result = interface.attempt_proof("complex theorem")
        assert not result.is_successful()
    
    def test_interface_proof_verification(self):
        """Test proof verification through interface."""
        interface = MockFormalSystemInterface()
        
        # Valid proof (no sorry)
        assert interface.verify_proof("theorem test : True", "trivial")
        
        # Invalid proof (contains sorry)
        assert not interface.verify_proof("theorem test : True", "by sorry")
    
    def test_interface_statistics_tracking(self):
        """Test that interface tracks statistics correctly."""
        interface = MockFormalSystemInterface()
        
        # Make some proof attempts
        proof1 = interface.attempt_proof("simple theorem")
        interface._update_statistics(proof1)
        
        proof2 = interface.attempt_proof("complex theorem")
        interface._update_statistics(proof2)
        
        stats = interface.get_statistics()
        assert stats['total_attempts'] == 2
        assert stats['successful_proofs'] == 1
        assert stats['success_rate'] == 0.5
    
    def test_interface_unavailable_system(self):
        """Test interface with unavailable system."""
        interface = MockFormalSystemInterface(available=False)
        
        assert not interface.is_available()
        
        stats = interface.get_statistics()
        assert not stats['is_available']


class TestSymPyToLean4Translator:
    """Test SymPy to Lean 4 translation functionality."""
    
    def test_translator_initialization(self):
        """Test translator initialization."""
        translator = SymPyToLean4Translator()
        
        assert 'sin' in translator.function_map
        assert 'cos' in translator.function_map
        assert 'log' in translator.function_map
        
        supported = translator.get_supported_functions()
        assert 'sin' in supported
        assert 'exp' in supported
    
    def test_simple_expression_translation(self):
        """Test translation of simple expressions."""
        translator = SymPyToLean4Translator()
        
        # Test basic arithmetic
        result = translator.translate_simple_expression("x + 1")
        assert "x" in result
        assert "+" in result
        
        # Test equality
        result = translator.translate_simple_expression("x = 1")
        assert "=" in result
        
        # Test functions
        result = translator.translate_simple_expression("sin(x)")
        assert "Real.sin" in result
    
    def test_variable_extraction(self):
        """Test variable extraction from expressions."""
        translator = SymPyToLean4Translator()
        
        import sympy as sp
        expr = sp.parse_expr("x + y * z")
        variables = translator._extract_variables(expr)
        
        var_names = [str(var) for var in variables]
        assert "x" in var_names
        assert "y" in var_names
        assert "z" in var_names
    
    def test_variable_declarations(self):
        """Test Lean 4 variable declaration generation."""
        translator = SymPyToLean4Translator()
        
        import sympy as sp
        variables = [sp.Symbol('x'), sp.Symbol('y')]
        
        declarations = translator._create_variable_declarations(variables)
        assert "variable" in declarations
        assert "x y" in declarations
        assert "ℝ" in declarations
    
    def test_function_translation(self):
        """Test translation of mathematical functions."""
        translator = SymPyToLean4Translator()
        
        # Trigonometric functions
        result = translator.translate_simple_expression("sin(x)")
        assert "Real.sin" in result
        
        result = translator.translate_simple_expression("cos(x)")
        assert "Real.cos" in result
        
        # Exponential and logarithmic
        result = translator.translate_simple_expression("exp(x)")
        assert "Real.exp" in result
        
        result = translator.translate_simple_expression("log(x)")
        assert "Real.log" in result
    
    def test_power_translation(self):
        """Test translation of power expressions."""
        translator = SymPyToLean4Translator()
        
        # Square root
        result = translator.translate_simple_expression("sqrt(x)")
        assert "Real.sqrt" in result
        
        # Integer power
        result = translator.translate_simple_expression("x**2")
        assert "^" in result
        
        # General power
        result = translator.translate_simple_expression("x**y")
        # Should use Real.rpow for general powers
        assert "Real.rpow" in result or "^" in result
    
    def test_complex_expression_translation(self):
        """Test translation of complex expressions."""
        translator = SymPyToLean4Translator()
        
        # Algebraic identity
        result = translator.translate_simple_expression("(x + 1)**2")
        assert "x" in result
        assert "+" in result
        
        # Trigonometric identity
        result = translator.translate_simple_expression("sin(x)**2 + cos(x)**2")
        assert "Real.sin" in result
        assert "Real.cos" in result
        assert "+" in result
    
    def test_id_sanitization(self):
        """Test theorem ID sanitization."""
        translator = SymPyToLean4Translator()
        
        # Test with invalid characters
        assert translator._sanitize_id("test-theorem") == "test_theorem"
        assert translator._sanitize_id("123theorem") == "theorem_123theorem"
        assert translator._sanitize_id("") == "unnamed_theorem"
        assert translator._sanitize_id("valid_theorem") == "valid_theorem"
    
    def test_translation_test_suite(self):
        """Test the built-in translation test suite."""
        translator = SymPyToLean4Translator()
        
        results = translator.test_translation()
        
        assert 'simple_equality' in results
        assert 'trigonometric' in results
        
        # Check that some translations were successful
        successful_count = sum(1 for result in results.values() if result['success'])
        assert successful_count > 0
    
    def test_translation_errors(self):
        """Test handling of translation errors."""
        translator = SymPyToLean4Translator()
        
        # Test with invalid expression
        with pytest.raises(TranslationError):
            translator.translate_simple_expression("invalid expression syntax +++")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"]) 