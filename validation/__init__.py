"""
Formula Validation Engine for MathBot.

This module provides comprehensive validation of mathematical formulas through:
- Symbolic validation using SymPy
- Numerical testing with randomized inputs
- Edge case testing and domain validation
- Integration with the knowledge graph system

Main Components:
- FormulaValidator: Core validation logic
- TestRunner: Orchestrates validation workflows
- ValidationResult: Standardized result format
"""

from .formula_tester import FormulaValidator, ValidationResult, ValidationConfig
from .test_runner import TestRunner, ValidationReport

__all__ = [
    "FormulaValidator", 
    "ValidationResult", 
    "ValidationConfig",
    "TestRunner", 
    "ValidationReport"
]

__version__ = "1.0.0" 