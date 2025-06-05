"""
Unit tests for UI data models.
"""

import pytest
from datetime import datetime
from typing import Dict, Any

from ui.data.models import (
    ValidationEvidence,
    SourceLineage,
    MathematicalContext,
    Theorem,
    FormulaData,
    ValidationReport,
    ValidationReportSummary,
    GenerationMetadata,
    TheoremCollection
)


class TestValidationEvidence:
    """Test ValidationEvidence model."""
    
    def test_valid_creation(self):
        """Test creating valid ValidationEvidence."""
        evidence = ValidationEvidence(
            validation_status="PASS",
            pass_rate=1.0,
            total_tests=100,
            symbols_tested=["x", "y"],
            validation_time=0.5
        )
        
        assert evidence.validation_status == "PASS"
        assert evidence.pass_rate == 1.0
        assert evidence.total_tests == 100
        assert evidence.symbols_tested == ["x", "y"]
        assert evidence.validation_time == 0.5
    
    def test_computed_fields(self):
        """Test computed fields."""
        # High confidence evidence
        high_evidence = ValidationEvidence(
            validation_status="PASS",
            pass_rate=0.95,
            total_tests=100,
            validation_time=0.5
        )
        
        assert high_evidence.is_successful is True
        assert high_evidence.confidence_level == "Very High"
        assert high_evidence.success_percentage == 95.0
        
        # Low confidence evidence
        low_evidence = ValidationEvidence(
            validation_status="FAIL",
            pass_rate=0.6,
            total_tests=100,
            validation_time=0.5
        )
        
        assert low_evidence.is_successful is False
        assert low_evidence.confidence_level == "Low"
        assert low_evidence.success_percentage == 60.0
    
    def test_validation_status_pattern(self):
        """Test validation status pattern validation."""
        # Valid statuses
        for status in ["PASS", "FAIL", "PARTIAL", "ERROR"]:
            evidence = ValidationEvidence(
                validation_status=status,
                pass_rate=0.8,
                total_tests=10,
                validation_time=0.1
            )
            assert evidence.validation_status == status
        
        # Invalid status should raise
        with pytest.raises(ValueError):
            ValidationEvidence(
                validation_status="INVALID",
                pass_rate=0.8,
                total_tests=10,
                validation_time=0.1
            )
    
    def test_pass_rate_bounds(self):
        """Test pass rate bounds validation."""
        # Valid pass rates
        for rate in [0.0, 0.5, 1.0]:
            evidence = ValidationEvidence(
                validation_status="PASS",
                pass_rate=rate,
                total_tests=10,
                validation_time=0.1
            )
            assert evidence.pass_rate == rate
        
        # Invalid pass rates should raise
        with pytest.raises(ValueError):
            ValidationEvidence(
                validation_status="PASS",
                pass_rate=-0.1,
                total_tests=10,
                validation_time=0.1
            )
        
        with pytest.raises(ValueError):
            ValidationEvidence(
                validation_status="PASS",
                pass_rate=1.1,
                total_tests=10,
                validation_time=0.1
            )


class TestSourceLineage:
    """Test SourceLineage model."""
    
    def test_valid_creation(self):
        """Test creating valid SourceLineage."""
        lineage = SourceLineage(
            original_formula="x^2",
            hypothesis_id="test_123",
            confidence=0.95,
            validation_score=1.0,
            generation_method="test_method",
            source_type="test_type",
            transformation_chain=["step1", "step2"]
        )
        
        assert lineage.original_formula == "x^2"
        assert lineage.hypothesis_id == "test_123"
        assert lineage.confidence == 0.95
        assert lineage.transformation_chain == ["step1", "step2"]
    
    def test_computed_fields(self):
        """Test computed fields."""
        lineage = SourceLineage(
            original_formula="x^2",
            hypothesis_id="test_123",
            confidence=0.85,
            validation_score=1.0,
            generation_method="test_method",
            source_type="test_type",
            transformation_chain=["substitute", "simplify"]
        )
        
        assert lineage.confidence_percentage == 85.0
        assert lineage.transformation_summary == "substitute → simplify"
        assert len(lineage.lineage_hash) == 8  # MD5 hash truncated to 8 chars
    
    def test_empty_transformation_chain(self):
        """Test handling of empty transformation chain."""
        lineage = SourceLineage(
            original_formula="x^2",
            hypothesis_id="test_123",
            confidence=0.95,
            validation_score=1.0,
            generation_method="test_method",
            source_type="test_type"
        )
        
        assert lineage.transformation_summary == "No transformations"


class TestMathematicalContext:
    """Test MathematicalContext model."""
    
    def test_valid_creation(self):
        """Test creating valid MathematicalContext."""
        context = MathematicalContext(
            symbols=["x", "y"],
            complexity_score=0.7,
            domain="algebra",
            variables={"x": "real", "y": "complex"},
            original_formula="x + y"
        )
        
        assert context.symbols == ["x", "y"]
        assert context.complexity_score == 0.7
        assert context.domain == "algebra"
        assert context.variables == {"x": "real", "y": "complex"}
    
    def test_computed_fields(self):
        """Test computed fields."""
        context = MathematicalContext(
            symbols=["x", "x", "y"],  # Duplicates should be handled
            variables={"x": "real"}
        )
        
        assert context.symbol_count == 2  # Unique symbols
        assert context.has_variables is True
        
        empty_context = MathematicalContext()
        assert empty_context.symbol_count == 0
        assert empty_context.has_variables is False


class TestTheorem:
    """Test Theorem model."""
    
    def create_valid_theorem(self) -> Theorem:
        """Helper to create a valid theorem."""
        return Theorem(
            id="THM_12345678",
            statement="Test theorem",
            sympy_expression="x**2",
            theorem_type="test_type",
            source_lineage=SourceLineage(
                original_formula="x^2",
                hypothesis_id="test_123",
                confidence=0.95,
                validation_score=1.0,
                generation_method="test",
                source_type="test"
            ),
            natural_language="A test theorem",
            validation_evidence=ValidationEvidence(
                validation_status="PASS",
                pass_rate=1.0,
                total_tests=10,
                validation_time=0.1
            )
        )
    
    def test_valid_creation(self):
        """Test creating valid theorem."""
        theorem = self.create_valid_theorem()
        
        assert theorem.id == "THM_12345678"
        assert theorem.statement == "Test theorem"
        assert theorem.theorem_type == "test_type"
        assert isinstance(theorem.source_lineage, SourceLineage)
        assert isinstance(theorem.validation_evidence, ValidationEvidence)
    
    def test_id_pattern_validation(self):
        """Test theorem ID pattern validation."""
        # Valid ID
        theorem = self.create_valid_theorem()
        assert theorem.id == "THM_12345678"
        
        # Invalid IDs should raise
        with pytest.raises(ValueError):
            theorem_data = self.create_valid_theorem().model_dump()
            theorem_data["id"] = "INVALID_ID"
            Theorem(**theorem_data)
    
    def test_computed_fields(self):
        """Test computed fields."""
        theorem = self.create_valid_theorem()
        
        assert theorem.is_validated is True
        assert theorem.short_id == "12345678"
        assert theorem.theorem_type_display == "Test Type"
        assert theorem.complexity_category in ["Simple", "Moderate", "Complex"]
        assert theorem.symbol_summary == "No symbols"  # No symbols in test theorem
    
    def test_mathematical_context_conversion(self):
        """Test automatic conversion of dict to MathematicalContext."""
        theorem_data = self.create_valid_theorem().model_dump()
        theorem_data["mathematical_context"] = {
            "symbols": ["x"],
            "domain": "algebra"
        }
        
        theorem = Theorem(**theorem_data)
        assert isinstance(theorem.mathematical_context, MathematicalContext)
        assert theorem.mathematical_context.symbols == ["x"]
    
    def test_display_statement_formatting(self):
        """Test display statement formatting."""
        theorem_data = self.create_valid_theorem().model_dump()
        theorem_data["statement"] = "\\forall x \\in \\mathbb{R}, x^2 \\geq 0"
        
        theorem = Theorem(**theorem_data)
        display = theorem.display_statement
        
        # Should replace LaTeX with Unicode
        assert "∀" in display
        assert "∈" in display
        assert "ℝ" in display
    
    def test_search_functionality(self):
        """Test search functionality."""
        theorem = self.create_valid_theorem()
        
        # Should match theorem type
        assert theorem.matches_search("test") is True
        assert theorem.matches_search("TEST") is True  # Case insensitive
        
        # Should not match unrelated terms
        assert theorem.matches_search("unrelated") is False
        
        # Test search scoring
        score = theorem.get_search_score("test")
        assert 0.0 <= score <= 1.0
        assert score > theorem.get_search_score("unrelated")


class TestFormulaData:
    """Test FormulaData model."""
    
    def test_valid_creation(self):
        """Test creating valid FormulaData."""
        formula = FormulaData(
            id="FORMULA_001",
            expression="x^2 + 1",
            sympy_form="x**2 + 1",
            latex_form="x^2 + 1",
            source="test_source"
        )
        
        assert formula.id == "FORMULA_001"
        assert formula.expression == "x^2 + 1"
        assert formula.source == "test_source"
    
    def test_computed_fields(self):
        """Test computed fields."""
        # With LaTeX form
        formula = FormulaData(
            id="FORMULA_001",
            expression="x^2 + 1",
            latex_form="x^2 + 1",
            source="test"
        )
        assert formula.display_form == "x^2 + 1"
        assert formula.short_id == "F001"
        
        # Without LaTeX form
        formula_no_latex = FormulaData(
            id="FORMULA_002",
            expression="y^3",
            source="test"
        )
        assert formula_no_latex.display_form == "y^3"


class TestValidationReport:
    """Test ValidationReport and ValidationReportSummary models."""
    
    def test_validation_report_summary(self):
        """Test ValidationReportSummary creation."""
        summary = ValidationReportSummary(
            total_formulas=10,
            validated_formulas=9,
            passed_formulas=8,
            failed_formulas=1,
            error_formulas=1,
            partial_formulas=0,
            overall_pass_rate=0.8,
            average_confidence=0.9,
            validation_time=1.5,
            timestamp="2025-01-01 00:00:00"
        )
        
        assert summary.total_formulas == 10
        assert summary.passed_formulas == 8
        assert summary.overall_pass_rate == 0.8
    
    def test_validation_report(self):
        """Test ValidationReport creation."""
        summary = ValidationReportSummary(
            total_formulas=5,
            validated_formulas=5,
            passed_formulas=4,
            failed_formulas=0,
            error_formulas=1,
            partial_formulas=0,
            overall_pass_rate=0.8,
            average_confidence=1.0,
            validation_time=0.176,
            timestamp="2025-01-01 00:00:00"
        )
        
        report = ValidationReport(
            summary=summary,
            statistics={"batch_name": "test"},
            results_by_status={"PASS": ["formula1"], "ERROR": ["formula2"]},
            errors_summary=["formula2: error message"]
        )
        
        assert report.success_percentage == 80.0
        assert abs(report.failure_percentage - 20.0) < 0.001  # Allow for floating point precision
        assert "4/5 passed" in report.validation_summary
        assert report.error_count == 1  # error_formulas + failed_formulas


class TestTheoremCollection:
    """Test TheoremCollection model."""
    
    def create_test_collection(self) -> TheoremCollection:
        """Helper to create test collection."""
        metadata = GenerationMetadata(
            total_theorems=2,
            generation_time=1.0,
            validation_passes=2,
            type_distribution={"test_type": 2},
            generator_version="1.0.0"
        )
        
        theorem1 = Theorem(
            id="THM_11111111",
            statement="Test theorem 1",
            sympy_expression="x**2",
            theorem_type="test_type",
            source_lineage=SourceLineage(
                original_formula="x^2",
                hypothesis_id="test_1",
                confidence=0.95,
                validation_score=1.0,
                generation_method="test",
                source_type="test"
            ),
            natural_language="First test theorem",
            validation_evidence=ValidationEvidence(
                validation_status="PASS",
                pass_rate=1.0,
                total_tests=10,
                validation_time=0.1
            )
        )
        
        theorem2 = Theorem(
            id="THM_22222222",
            statement="Test theorem 2",
            sympy_expression="y**3",
            theorem_type="test_type",
            source_lineage=SourceLineage(
                original_formula="y^3",
                hypothesis_id="test_2",
                confidence=0.85,
                validation_score=1.0,
                generation_method="test",
                source_type="test"
            ),
            natural_language="Second test theorem",
            validation_evidence=ValidationEvidence(
                validation_status="PASS",
                pass_rate=1.0,
                total_tests=10,
                validation_time=0.1
            )
        )
        
        return TheoremCollection(
            generation_metadata=metadata,
            theorems=[theorem1, theorem2]
        )
    
    def test_collection_creation(self):
        """Test creating theorem collection."""
        collection = self.create_test_collection()
        
        assert collection.theorem_count == 2
        assert collection.validated_count == 2
        assert collection.validation_rate == 1.0
        assert collection.type_summary == {"test_type": 2}
    
    def test_search_functionality(self):
        """Test collection search functionality."""
        collection = self.create_test_collection()
        
        # Search for "first"
        results = collection.search("first", limit=1)
        assert len(results) == 1
        assert "First" in results[0].natural_language
        
        # Search for theorem type
        results = collection.search("test", limit=5)
        assert len(results) == 2  # Both theorems match
    
    def test_filtering_methods(self):
        """Test collection filtering methods."""
        collection = self.create_test_collection()
        
        # Get by type
        type_theorems = collection.get_by_type("test_type")
        assert len(type_theorems) == 2
        
        # Get by ID
        theorem = collection.get_by_id("THM_11111111")
        assert theorem is not None
        assert theorem.statement == "Test theorem 1"
        
        # Get non-existent ID
        missing = collection.get_by_id("THM_99999999")
        assert missing is None 