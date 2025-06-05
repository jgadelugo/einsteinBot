"""
Tests for FormulaValidator class.

Comprehensive test suite covering symbolic validation, numerical testing,
edge case analysis, and validation with known mathematical formulas.
"""

import pytest
import numpy as np
from unittest.mock import patch, Mock

from validation.formula_tester import (
    FormulaValidator, ValidationConfig, ValidationResult, ValidationStatus, 
    TestType, TestResult
)


class TestFormulaValidator:
    """Test suite for FormulaValidator."""
    
    @pytest.fixture
    def validator(self):
        """Create validator with test configuration."""
        config = ValidationConfig(
            num_random_tests=20,  # Reduce for faster tests
            random_seed=42,
            test_range=(-5.0, 5.0),
            tolerance=1e-8
        )
        return FormulaValidator(config)
    
    @pytest.fixture
    def simple_config(self):
        """Minimal configuration for basic tests."""
        return ValidationConfig(
            num_random_tests=5,
            random_seed=123,
            enable_symbolic=True,
            enable_numerical=True,
            enable_edge_cases=False,
            enable_round_trip=False
        )
    
    def test_validator_initialization(self):
        """Test validator initialization with default and custom configs."""
        # Default config
        validator = FormulaValidator()
        assert validator.config.num_random_tests == 100
        assert validator.config.random_seed == 42
        
        # Custom config
        config = ValidationConfig(num_random_tests=50, random_seed=999)
        validator = FormulaValidator(config)
        assert validator.config.num_random_tests == 50
        assert validator.config.random_seed == 999
    
    def test_parse_formula_success(self, validator):
        """Test successful formula parsing."""
        # Simple polynomial
        expr = validator._parse_formula("x**2 + 2*x + 1")
        assert expr is not None
        assert str(expr) == "x**2 + 2*x + 1"
        
        # Trigonometric function
        expr = validator._parse_formula("sin(x) + cos(x)")
        assert expr is not None
        assert len(expr.free_symbols) == 1
    
    def test_parse_formula_failure(self, validator):
        """Test formula parsing failures."""
        # Invalid syntax
        expr = validator._parse_formula("x + + y")
        assert expr is None
        
        # Empty formula
        expr = validator._parse_formula("")
        assert expr is None
        
        # Extremely complex formula (should be rejected)
        complex_formula = "x**" + "2**" * 100 + "1"
        expr = validator._parse_formula(complex_formula)
        assert expr is None
    
    def test_clean_latex_formula(self, validator):
        """Test LaTeX cleaning functionality."""
        # Basic cleaning
        cleaned = validator._clean_latex_formula("\\frac{x}{y}")
        assert "frac" not in cleaned
        
        # Multiple replacements
        cleaned = validator._clean_latex_formula("\\sin(\\pi) + \\cos(\\theta)")
        assert "sin" in cleaned and "cos" in cleaned
        assert "pi" in cleaned
        
        # No LaTeX (should pass through)
        cleaned = validator._clean_latex_formula("x + y")
        assert cleaned == "x + y"


class TestKnownFormulas:
    """Test validation with real mathematical formulas."""
    
    @pytest.fixture
    def validator(self):
        """Create validator for known formula tests."""
        config = ValidationConfig(
            num_random_tests=50,
            random_seed=42,
            test_range=(-2.0, 2.0),  # Smaller range for stability
            tolerance=1e-6
        )
        return FormulaValidator(config)
    
    def test_quadratic_formula(self, validator):
        """Test the quadratic formula: (-b ± sqrt(b²-4ac)) / 2a"""
        formula = "(-b + sqrt(b**2 - 4*a*c)) / (2*a)"
        result = validator.validate_formula(formula)
        
        assert result.status in [ValidationStatus.PASS, ValidationStatus.PARTIAL]
        assert result.symbols_found == {'a', 'b', 'c'}
        assert result.total_tests > 0
        # Should pass most numerical tests (avoiding discriminant < 0)
        assert result.pass_rate > 0.3  # Some tests may fail due to complex numbers
    
    def test_pythagorean_theorem(self, validator):
        """Test Pythagorean theorem: a² + b² = c²"""
        formula = "a**2 + b**2"
        result = validator.validate_formula(formula, known_identity="c**2")
        
        assert result.status != ValidationStatus.ERROR
        assert result.symbols_found == {'a', 'b'}
        assert result.total_tests > 0
        assert result.pass_rate > 0.8  # Should be very stable
    
    def test_derivative_power_rule(self, validator):
        """Test derivative of x^n: n*x^(n-1)"""
        # Specific case: derivative of x^3
        formula = "3*x**2"
        result = validator.validate_formula(formula)
        
        assert result.status != ValidationStatus.ERROR
        assert result.symbols_found == {'x'}
        assert result.pass_rate > 0.9  # Very stable polynomial
    
    def test_trigonometric_identity(self, validator):
        """Test sin²x + cos²x = 1"""
        formula = "sin(x)**2 + cos(x)**2"
        result = validator.validate_formula(formula, known_identity="1")
        
        assert result.status != ValidationStatus.ERROR
        assert result.symbols_found == {'x'}
        # This should pass most tests due to fundamental trig identity
        assert result.pass_rate > 0.9
    
    def test_exponential_logarithm(self, validator):
        """Test e^(ln(x)) = x for x > 0"""
        formula = "exp(log(x))"
        result = validator.validate_formula(formula, known_identity="x")
        
        assert result.status != ValidationStatus.ERROR
        assert result.symbols_found == {'x'}
        # May have some failures due to domain restrictions
        assert result.pass_rate > 0.3
    
    def test_area_of_circle(self, validator):
        """Test area of circle: π*r²"""
        formula = "pi * r**2"
        result = validator.validate_formula(formula)
        
        assert result.status != ValidationStatus.ERROR
        assert result.symbols_found == {'r'}
        assert result.pass_rate > 0.9  # Very stable
    
    def test_compound_interest(self, validator):
        """Test compound interest: P(1 + r/n)^(nt)"""
        formula = "P * (1 + r/n)**(n*t)"
        result = validator.validate_formula(formula)
        
        assert result.status != ValidationStatus.ERROR
        assert result.symbols_found == {'P', 'r', 'n', 't'}
        # May have domain issues if n=0 or negative values
        assert result.pass_rate > 0.4
    
    def test_gaussian_function(self, validator):
        """Test Gaussian/normal distribution: exp(-x²/2)"""
        formula = "exp(-x**2/2)"
        result = validator.validate_formula(formula)
        
        assert result.status != ValidationStatus.ERROR
        assert result.symbols_found == {'x'}
        assert result.pass_rate > 0.9  # Very stable
    
    def test_distance_formula(self, validator):
        """Test 2D distance formula: sqrt((x2-x1)² + (y2-y1)²)"""
        formula = "sqrt((x2 - x1)**2 + (y2 - y1)**2)"
        result = validator.validate_formula(formula)
        
        assert result.status != ValidationStatus.ERROR
        assert result.symbols_found == {'x1', 'x2', 'y1', 'y2'}
        assert result.pass_rate > 0.9  # Distance is always non-negative
    
    def test_binomial_theorem_simple(self, validator):
        """Test simple binomial expansion: (a + b)²"""
        formula = "a**2 + 2*a*b + b**2"
        result = validator.validate_formula(formula, known_identity="(a + b)**2")
        
        assert result.status != ValidationStatus.ERROR
        assert result.symbols_found == {'a', 'b'}
        assert result.pass_rate > 0.9  # Should be very stable


class TestValidationTypes:
    """Test different types of validation."""
    
    @pytest.fixture
    def validator(self):
        """Create validator for validation type tests."""
        return FormulaValidator(ValidationConfig(
            num_random_tests=10,
            random_seed=42,
            enable_symbolic=True,
            enable_numerical=True,
            enable_edge_cases=True,
            enable_round_trip=True
        ))
    
    def test_symbolic_validation(self, validator):
        """Test symbolic validation tests."""
        formula = "x**2 + 2*x + 1"
        result = validator.validate_formula(formula)
        
        # Check that symbolic tests were run
        symbolic_tests = [t for t in result.test_results if t.test_type == TestType.SYMBOLIC]
        assert len(symbolic_tests) > 0
        
        # Should pass well-formed test
        well_formed_tests = [t for t in symbolic_tests if t.metadata.get('test_name') == 'well_formed']
        assert len(well_formed_tests) > 0
        assert all(t.passed for t in well_formed_tests)
    
    def test_numerical_validation(self, validator):
        """Test numerical validation tests."""
        formula = "x**2 + 1"
        result = validator.validate_formula(formula)
        
        # Check that numerical tests were run
        numerical_tests = [t for t in result.test_results if t.test_type == TestType.NUMERICAL]
        assert len(numerical_tests) == validator.config.num_random_tests
        
        # Most should pass for this stable formula
        passed_tests = [t for t in numerical_tests if t.passed]
        assert len(passed_tests) >= 8  # At least 80% should pass
    
    def test_edge_case_validation(self, validator):
        """Test edge case validation."""
        formula = "x + 1"
        result = validator.validate_formula(formula)
        
        # Check that edge case tests were run
        edge_tests = [t for t in result.test_results if t.test_type == TestType.EDGE_CASE]
        assert len(edge_tests) > 0
        
        # Should handle most edge cases for linear function
        passed_edge_tests = [t for t in edge_tests if t.passed]
        assert len(passed_edge_tests) >= len(edge_tests) * 0.5
    
    def test_round_trip_validation(self, validator):
        """Test round-trip validation."""
        formula = "x**2 + 2*x + 1"
        result = validator.validate_formula(formula)
        
        # Check that round-trip tests were run
        round_trip_tests = [t for t in result.test_results if t.test_type == TestType.ROUND_TRIP]
        assert len(round_trip_tests) > 0
        
        # Simplification consistency should pass
        simplification_tests = [
            t for t in round_trip_tests 
            if t.metadata.get('test_name') == 'simplification_consistency'
        ]
        assert len(simplification_tests) > 0


class TestValidationResult:
    """Test validation result processing."""
    
    @pytest.fixture
    def validator(self):
        """Create validator for result tests."""
        return FormulaValidator(ValidationConfig(
            num_random_tests=20,
            random_seed=42
        ))
    
    def test_confidence_score_calculation(self, validator):
        """Test confidence score calculation."""
        # High-confidence case: simple, stable formula
        result = validator.validate_formula("x + 1")
        assert result.confidence_score > 0.8
        assert result.status in [ValidationStatus.PASS, ValidationStatus.PARTIAL]
        
        # Lower confidence case: potentially problematic formula
        result = validator.validate_formula("1/x")  # Division by zero possible
        assert result.confidence_score < result.pass_rate + 0.1  # Should account for potential issues
    
    def test_status_determination(self, validator):
        """Test validation status determination."""
        # Should pass for stable formula
        result = validator.validate_formula("x**2 + 1")
        assert result.status in [ValidationStatus.PASS, ValidationStatus.PARTIAL]
        
        # Should error for unparseable formula
        result = validator.validate_formula("invalid syntax +++")
        assert result.status == ValidationStatus.ERROR
    
    def test_metadata_collection(self, validator):
        """Test metadata collection in results."""
        result = validator.validate_formula("sin(x) + cos(x)")
        
        # Should have symbols
        assert result.symbols_found == {'x'}
        
        # Should have execution time
        assert result.validation_time > 0
        
        # Should have test results with metadata
        assert len(result.test_results) > 0
        for test_result in result.test_results:
            assert hasattr(test_result, 'metadata')
            assert isinstance(test_result.metadata, dict)


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.fixture
    def validator(self):
        """Create validator for error tests."""
        return FormulaValidator(ValidationConfig(
            num_random_tests=5,
            random_seed=42
        ))
    
    def test_empty_formula(self, validator):
        """Test handling of empty formula."""
        result = validator.validate_formula("")
        assert result.status == ValidationStatus.ERROR
        assert result.total_tests == 0
    
    def test_invalid_formula(self, validator):
        """Test handling of invalid formula syntax."""
        result = validator.validate_formula("x + + y")
        assert result.status == ValidationStatus.ERROR
        assert result.error_summary is not None
    
    def test_complex_formula_rejection(self, validator):
        """Test rejection of overly complex formulas."""
        # Create a very long formula
        complex_formula = " + ".join([f"x**{i}" for i in range(200)])
        result = validator.validate_formula(complex_formula)
        
        # Should either error or handle gracefully
        assert result.status in [ValidationStatus.ERROR, ValidationStatus.FAIL, ValidationStatus.PARTIAL]
    
    def test_domain_errors(self, validator):
        """Test handling of domain errors during evaluation."""
        # Formula with potential domain issues
        result = validator.validate_formula("sqrt(x)")  # Negative x problematic
        
        # Should complete but may have some failures
        assert result.total_tests > 0
        # Some tests may fail due to negative inputs
        assert result.pass_rate < 1.0 or result.error_tests > 0
    
    def test_division_by_zero(self, validator):
        """Test handling of division by zero."""
        result = validator.validate_formula("1/x")
        
        # Should handle gracefully
        assert result.total_tests > 0
        # Should have some errors when x=0 is tested
        if 0 in validator.config.edge_case_values:
            assert result.error_tests > 0 or result.failed_tests > 0


@pytest.mark.parametrize("formula,expected_symbols", [
    ("x + y", {"x", "y"}),
    ("sin(theta)", {"theta"}),
    ("a*b*c", {"a", "b", "c"}),
    ("pi * r**2", {"r"}),  # pi is a constant in SymPy
    ("2 + 3", set()),  # No symbols
])
def test_symbol_extraction(formula, expected_symbols):
    """Test symbol extraction from various formulas."""
    validator = FormulaValidator(ValidationConfig(num_random_tests=1))
    result = validator.validate_formula(formula)
    
    if result.status != ValidationStatus.ERROR:
        assert result.symbols_found == expected_symbols


def test_reproducibility():
    """Test that validation results are reproducible with same seed."""
    config = ValidationConfig(num_random_tests=10, random_seed=12345)
    
    validator1 = FormulaValidator(config)
    result1 = validator1.validate_formula("x**2 + 2*x + 1")
    
    validator2 = FormulaValidator(config)
    result2 = validator2.validate_formula("x**2 + 2*x + 1")
    
    # Results should be identical
    assert result1.pass_rate == result2.pass_rate
    assert result1.total_tests == result2.total_tests
    assert result1.passed_tests == result2.passed_tests
    
    # Test values should be the same (approximately, due to floating point)
    numerical_results1 = [t for t in result1.test_results if t.test_type == TestType.NUMERICAL]
    numerical_results2 = [t for t in result2.test_results if t.test_type == TestType.NUMERICAL]
    
    assert len(numerical_results1) == len(numerical_results2)
    
    for t1, t2 in zip(numerical_results1, numerical_results2):
        if t1.test_values and t2.test_values:
            for key in t1.test_values:
                assert abs(t1.test_values[key] - t2.test_values[key]) < 1e-10


class TestValidationConfig:
    """Test validation configuration options."""
    
    def test_config_defaults(self):
        """Test default configuration values."""
        config = ValidationConfig()
        assert config.num_random_tests == 100
        assert config.random_seed == 42
        assert config.test_range == (-10.0, 10.0)
        assert config.tolerance == 1e-10
        assert config.enable_symbolic is True
        assert config.enable_numerical is True
    
    def test_config_customization(self):
        """Test custom configuration."""
        config = ValidationConfig(
            num_random_tests=50,
            test_range=(-1.0, 1.0),
            tolerance=1e-6,
            enable_edge_cases=False
        )
        
        validator = FormulaValidator(config)
        result = validator.validate_formula("x + 1")
        
        # Should have correct number of numerical tests
        numerical_tests = [t for t in result.test_results if t.test_type == TestType.NUMERICAL]
        assert len(numerical_tests) == 50
        
        # Should not have edge case tests
        edge_tests = [t for t in result.test_results if t.test_type == TestType.EDGE_CASE]
        assert len(edge_tests) == 0 if not config.enable_edge_cases else len(edge_tests) >= 0
    
    def test_disabled_test_types(self):
        """Test disabling specific test types."""
        config = ValidationConfig(
            num_random_tests=5,
            enable_symbolic=False,
            enable_numerical=True,
            enable_edge_cases=False,
            enable_round_trip=False
        )
        
        validator = FormulaValidator(config)
        result = validator.validate_formula("x**2")
        
        # Should only have numerical tests
        test_types = {t.test_type for t in result.test_results}
        assert TestType.NUMERICAL in test_types
        assert TestType.SYMBOLIC not in test_types
        assert TestType.EDGE_CASE not in test_types
        assert TestType.ROUND_TRIP not in test_types