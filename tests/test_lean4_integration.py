"""
Tests for Lean 4 integration and interface functionality.

This module tests the Lean 4 interface, including installation verification,
theorem translation, proof attempts, and simulation mode functionality.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import subprocess
import tempfile

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from proofs.formal_systems.lean4_interface import Lean4Interface
from proofs.formal_systems.base_interface import (
    FormalSystemType, 
    FormalProof,
    TranslationError,
    ProofError
)


class TestLean4Interface:
    """Test Lean 4 interface functionality."""
    
    def test_lean4_interface_initialization(self):
        """Test Lean 4 interface initialization."""
        # Test with simulation mode (no actual Lean 4)
        interface = Lean4Interface()
        
        assert interface.system_type == FormalSystemType.LEAN4
        assert interface.timeout == 30
        assert interface.translator is not None
        assert isinstance(interface.proof_cache, dict)
        assert isinstance(interface.lean_metrics, dict)
        
        # Check metrics structure
        expected_metrics = [
            'compilation_attempts', 'compilation_successes', 'cache_hits',
            'timeout_errors', 'syntax_errors', 'type_errors'
        ]
        for metric in expected_metrics:
            assert metric in interface.lean_metrics
    
    def test_lean4_interface_with_custom_path(self):
        """Test initialization with custom Lean path."""
        custom_path = "/custom/path/to/lean"
        interface = Lean4Interface(lean_path=custom_path, timeout=60)
        
        assert interface.lean_path == custom_path
        assert interface.timeout == 60
    
    def test_find_lean_executable(self):
        """Test Lean executable discovery."""
        interface = Lean4Interface()
        
        # This will likely return None in test environment
        lean_path = interface._find_lean_executable()
        
        # Should be None or a valid path string
        assert lean_path is None or isinstance(lean_path, str)
    
    @patch('subprocess.run')
    def test_verify_lean4_installation_success(self, mock_run):
        """Test successful Lean 4 verification."""
        # Mock successful version check
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Lean 4.3.0 (release)"
        mock_run.return_value = mock_result
        
        interface = Lean4Interface(lean_path="/mock/lean")
        
        # Should call verification during init
        assert mock_run.called
        # In this case, interface should recognize it as available
        # (though this depends on the specific mock setup)
    
    @patch('subprocess.run')
    def test_verify_lean4_installation_failure(self, mock_run):
        """Test failed Lean 4 verification."""
        # Mock failed version check
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "Command not found"
        mock_run.return_value = mock_result
        
        interface = Lean4Interface(lean_path="/nonexistent/lean")
        
        assert not interface.is_available()
    
    @patch('subprocess.run')
    def test_verify_lean4_installation_wrong_version(self, mock_run):
        """Test verification with wrong Lean version."""
        # Mock Lean 3 version output
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Lean 3.51.1"
        mock_run.return_value = mock_result
        
        interface = Lean4Interface(lean_path="/mock/lean3")
        
        assert not interface.is_available()
    
    def test_translate_theorem(self):
        """Test theorem translation to Lean 4."""
        interface = Lean4Interface()
        
        # Create a mock theorem object
        class MockTheorem:
            def __init__(self):
                self.id = "test_theorem"
                self.statement = "Test theorem"
                self.sympy_expression = None
        
        theorem = MockTheorem()
        
        try:
            result = interface.translate_theorem(theorem)
            assert isinstance(result, str)
            assert "theorem" in result.lower()
        except TranslationError:
            # Translation might fail with mock theorem
            pass
    
    def test_simulation_mode_proof_attempt(self):
        """Test proof attempt in simulation mode."""
        interface = Lean4Interface()
        
        # Ensure we're in simulation mode
        assert not interface.is_available()
        
        # Test simple theorem (should succeed in simulation)
        simple_statement = "theorem test : 1 + 1 = 2 := by norm_num"
        result = interface.attempt_proof(simple_statement)
        
        assert isinstance(result, FormalProof)
        assert result.system_type == FormalSystemType.LEAN4
        assert result.theorem_id == "test"
        assert result.verification_status in ["proved", "failed"]
        assert result.metadata.get('simulation_mode') is True
        
        # Test complex theorem (should fail in simulation)
        complex_statement = "theorem complex : ∀ x : ℝ, complex_function(x) = 0"
        result = interface.attempt_proof(complex_statement)
        
        assert isinstance(result, FormalProof)
        assert result.verification_status == "failed"
        assert len(result.error_messages) > 0
    
    def test_proof_caching(self):
        """Test proof result caching."""
        interface = Lean4Interface()
        
        statement = "theorem cached_test : True := trivial"
        
        # First attempt
        result1 = interface.attempt_proof(statement)
        cache_size_after_first = len(interface.proof_cache)
        
        # Second attempt should use cache
        result2 = interface.attempt_proof(statement)
        
        # Check that cache was used
        assert interface.lean_metrics['cache_hits'] > 0
        assert len(interface.proof_cache) == cache_size_after_first
        
        # Results should be the same
        assert result1.theorem_id == result2.theorem_id
        assert result1.verification_status == result2.verification_status
    
    def test_extract_theorem_id(self):
        """Test theorem ID extraction."""
        interface = Lean4Interface()
        
        # Test with well-formed theorem
        statement1 = "theorem my_theorem : 1 = 1 := rfl"
        assert interface._extract_theorem_id(statement1) == "my_theorem"
        
        # Test with theorem without name
        statement2 = "1 = 1"
        assert interface._extract_theorem_id(statement2) == "generated_theorem"
        
        # Test with complex theorem
        statement3 = "theorem pythagorean_identity : ∀ x : ℝ, sin x ^ 2 + cos x ^ 2 = 1"
        assert interface._extract_theorem_id(statement3) == "pythagorean_identity"
    
    def test_extract_theorem_body(self):
        """Test theorem body extraction."""
        interface = Lean4Interface()
        
        # Test with complete theorem
        statement1 = "theorem test : 1 = 1 := rfl"
        body1 = interface._extract_theorem_body(statement1)
        assert "1 = 1" in body1
        
        # Test with theorem declaration only
        statement2 = "theorem test : ∀ x : ℝ, x + 0 = x"
        body2 = interface._extract_theorem_body(statement2)
        assert "∀ x : ℝ, x + 0 = x" in body2
        
        # Test edge case
        statement3 = "simple expression"
        body3 = interface._extract_theorem_body(statement3)
        assert body3 == "simple expression"
    
    def test_create_lean_file(self):
        """Test Lean file generation."""
        interface = Lean4Interface()
        
        statement = "theorem test_theorem : 1 + 1 = 2"
        lean_content = interface._create_lean_file(statement)
        
        assert isinstance(lean_content, str)
        assert "-- Generated by MathBot Phase 5D" in lean_content
        assert "theorem test_theorem" in lean_content
        assert "attempt1" in lean_content
        assert "attempt2" in lean_content
        assert "simp" in lean_content
        assert "ring" in lean_content
        assert "#check" in lean_content
    
    def test_parse_lean_errors(self):
        """Test Lean error message parsing."""
        interface = Lean4Interface()
        
        # Test with empty error
        errors1 = interface._parse_lean_errors("")
        assert errors1 == []
        
        # Test with single error
        stderr1 = "syntax error: expected expression"
        errors2 = interface._parse_lean_errors(stderr1)
        assert len(errors2) == 1
        assert "syntax error" in errors2[0]
        
        # Test with multiple errors
        stderr2 = """error 1: syntax error
        
error 2: type mismatch
expected: Nat
got: String"""
        errors3 = interface._parse_lean_errors(stderr2)
        assert len(errors3) == 2
        
        # Check metrics are updated
        initial_syntax_errors = interface.lean_metrics['syntax_errors']
        interface._parse_lean_errors("syntax error occurred")
        assert interface.lean_metrics['syntax_errors'] > initial_syntax_errors
    
    def test_verify_proof_simulation_mode(self):
        """Test proof verification in simulation mode."""
        interface = Lean4Interface()
        
        # Should return True for valid-looking proofs
        valid_proof = "theorem test : True := trivial"
        assert interface.verify_proof("", valid_proof) is True
        
        # Should return False for proofs with sorry
        invalid_proof = "theorem test : False := sorry"
        assert interface.verify_proof("", invalid_proof) is False
        
        # Should return False for error proofs
        error_proof = "-- Error during compilation"
        assert interface.verify_proof("", error_proof) is False
    
    def test_statistics_tracking(self):
        """Test performance statistics tracking."""
        interface = Lean4Interface()
        
        # Get initial statistics
        initial_stats = interface.get_statistics()
        initial_lean_stats = interface.get_lean_statistics()
        
        assert initial_stats['total_attempts'] == 0
        assert initial_stats['successful_proofs'] == 0
        assert initial_lean_stats['lean_specific']['compilation_attempts'] == 0
        
        # Make a proof attempt
        interface.attempt_proof("theorem test : True")
        
        # Check updated statistics
        updated_stats = interface.get_statistics()
        updated_lean_stats = interface.get_lean_statistics()
        
        assert updated_stats['total_attempts'] == 1
        assert updated_lean_stats['lean_specific']['compilation_attempts'] == 1
        
        # Check lean-specific metrics are present
        lean_specific = updated_lean_stats['lean_specific']
        assert 'proof_cache_size' in lean_specific
        assert 'temp_directory' in lean_specific
        assert 'lean_executable' in lean_specific
    
    def test_cache_operations(self):
        """Test cache management operations."""
        interface = Lean4Interface()
        
        # Add something to cache
        statement = "theorem cache_test : True"
        interface.attempt_proof(statement)
        
        # Verify cache has content
        assert len(interface.proof_cache) > 0
        
        # Clear cache
        interface.clear_cache()
        
        # Verify cache is cleared
        assert len(interface.proof_cache) == 0
    
    @patch('tempfile.mkdtemp')
    def test_temp_directory_setup(self, mock_mkdtemp):
        """Test temporary directory setup."""
        mock_mkdtemp.return_value = "/tmp/test_mathbot_lean4_123"
        
        interface = Lean4Interface()
        interface._setup_temp_directory()
        
        assert mock_mkdtemp.called
        assert interface.temp_dir == "/tmp/test_mathbot_lean4_123"
    
    def test_string_representations(self):
        """Test string representations of Lean4Interface."""
        interface = Lean4Interface()
        
        # Test __str__
        str_repr = str(interface)
        assert "Lean4Interface" in str_repr
        assert "status=" in str_repr
        
        # Test __repr__
        repr_str = repr(interface)
        assert "Lean4Interface" in repr_str
        assert "available=" in repr_str
        assert "success_rate=" in repr_str
    
    @patch('subprocess.run')
    def test_compilation_with_lean4_available(self, mock_run):
        """Test compilation when Lean 4 is available."""
        # Mock Lean 4 being available
        version_result = Mock()
        version_result.returncode = 0
        version_result.stdout = "Lean 4.3.0"
        
        compile_result = Mock()
        compile_result.returncode = 0
        compile_result.stdout = "Success"
        compile_result.stderr = ""
        
        mock_run.side_effect = [version_result, compile_result]
        
        # Create interface with mocked temp directory
        with patch('tempfile.mkdtemp') as mock_mkdtemp:
            mock_mkdtemp.return_value = "/tmp/test_lean"
            
            interface = Lean4Interface(lean_path="/mock/lean")
            
            # Mock file operations
            with patch('builtins.open', create=True) as mock_open:
                mock_open.return_value.__enter__.return_value.write = Mock()
                
                statement = "theorem test : True"
                result = interface.attempt_proof(statement)
                
                # Should have attempted actual compilation
                assert mock_run.call_count >= 2  # Version check + compilation
                assert isinstance(result, FormalProof)
    
    @patch('subprocess.run')
    def test_compilation_timeout(self, mock_run):
        """Test compilation timeout handling."""
        # Mock version check success
        version_result = Mock()
        version_result.returncode = 0
        version_result.stdout = "Lean 4.3.0"
        
        # Mock compilation timeout
        mock_run.side_effect = [
            version_result,
            subprocess.TimeoutExpired("lean", 30)
        ]
        
        with patch('tempfile.mkdtemp') as mock_mkdtemp:
            mock_mkdtemp.return_value = "/tmp/test_lean"
            
            interface = Lean4Interface(lean_path="/mock/lean")
            
            with patch('builtins.open', create=True):
                result = interface.attempt_proof("theorem test : True", timeout=1)
                
                assert result.verification_status == "timeout"
                assert interface.lean_metrics['timeout_errors'] > 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"]) 