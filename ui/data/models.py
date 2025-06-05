"""
Pydantic data models for MathBot UI.

This module defines type-safe, validated data models that match the existing
theorem and validation data structures, with computed fields for UI display.
"""

import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, computed_field, field_validator


class ValidationEvidence(BaseModel):
    """Validation evidence for a mathematical theorem."""
    
    validation_status: str = Field(..., pattern="^(PASS|FAIL|PARTIAL|ERROR)$")
    pass_rate: float = Field(..., ge=0.0, le=1.0)
    total_tests: int = Field(..., ge=0)
    symbols_tested: List[str] = Field(default_factory=list)
    validation_time: float = Field(..., ge=0.0)
    
    @computed_field
    @property
    def is_successful(self) -> bool:
        """Check if validation was successful."""
        return self.validation_status == "PASS" and self.pass_rate >= 0.9
    
    @computed_field
    @property
    def confidence_level(self) -> str:
        """Get human-readable confidence level."""
        if self.pass_rate >= 0.95:
            return "Very High"
        elif self.pass_rate >= 0.9:
            return "High" 
        elif self.pass_rate >= 0.75:
            return "Medium"
        else:
            return "Low"
    
    @computed_field
    @property
    def success_percentage(self) -> float:
        """Get success rate as percentage."""
        return self.pass_rate * 100


class SourceLineage(BaseModel):
    """Source lineage tracking for theorem generation."""
    
    original_formula: str = Field(..., min_length=1)
    hypothesis_id: str = Field(..., min_length=1)
    confidence: float = Field(..., ge=0.0, le=1.0)
    validation_score: float = Field(..., ge=0.0, le=1.0)
    generation_method: str = Field(..., min_length=1)
    source_type: str = Field(..., min_length=1)
    transformation_chain: List[str] = Field(default_factory=list)
    
    @computed_field
    @property
    def lineage_hash(self) -> str:
        """Generate unique hash for this lineage."""
        content = f"{self.original_formula}{self.hypothesis_id}{self.generation_method}"
        return hashlib.md5(content.encode()).hexdigest()[:8]
    
    @computed_field
    @property
    def confidence_percentage(self) -> float:
        """Get confidence as percentage."""
        return self.confidence * 100
    
    @computed_field
    @property
    def transformation_summary(self) -> str:
        """Get summary of transformation chain."""
        if not self.transformation_chain:
            return "No transformations"
        return " → ".join(self.transformation_chain)


class MathematicalContext(BaseModel):
    """Mathematical context and metadata."""
    
    symbols: List[str] = Field(default_factory=list)
    complexity_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    domain: Optional[str] = None
    variables: Dict[str, str] = Field(default_factory=dict)
    original_formula: Optional[str] = None
    transformation: Optional[str] = None
    variable: Optional[str] = None
    result: Optional[str] = None
    
    @computed_field
    @property
    def symbol_count(self) -> int:
        """Count of unique symbols."""
        return len(set(self.symbols))
    
    @computed_field
    @property
    def has_variables(self) -> bool:
        """Check if context has variables defined."""
        return bool(self.variables)


class Theorem(BaseModel):
    """Complete theorem with validation and lineage."""
    
    id: str = Field(..., pattern="^THM_[A-F0-9]{8}$")
    statement: str = Field(..., min_length=1)
    sympy_expression: str = Field(..., min_length=1)
    theorem_type: str = Field(..., min_length=1)
    assumptions: List[str] = Field(default_factory=list)
    source_lineage: SourceLineage
    natural_language: str = Field(..., min_length=1)
    symbols: List[str] = Field(default_factory=list)
    mathematical_context: Union[Dict[str, Any], MathematicalContext] = Field(default_factory=dict)
    validation_evidence: ValidationEvidence
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('mathematical_context', mode='before')
    @classmethod
    def validate_mathematical_context(cls, v):
        """Convert dict to MathematicalContext if needed."""
        if isinstance(v, dict):
            return MathematicalContext(**v)
        return v
    
    # Computed fields
    @computed_field
    @property
    def is_validated(self) -> bool:
        """Check if theorem is validated."""
        return self.validation_evidence.is_successful
    
    @computed_field
    @property
    def complexity_category(self) -> str:
        """Categorize theorem complexity."""
        symbol_count = len(set(self.symbols))
        assumption_count = len(self.assumptions)
        
        total_complexity = symbol_count + assumption_count
        
        if total_complexity <= 2:
            return "Simple"
        elif total_complexity <= 5:
            return "Moderate"
        else:
            return "Complex"
    
    @computed_field
    @property
    def display_statement(self) -> str:
        """Get formatted statement for display."""
        # Clean up Unicode symbols for display
        statement = self.statement
        replacements = {
            "\\forall": "∀",
            "\\exists": "∃", 
            "\\in": "∈",
            "\\mathbb{R}": "ℝ",
            "\\mathbb{N}": "ℕ",
            "\\mathbb{Z}": "ℤ",
            "\\mathbb{Q}": "ℚ",
            "\\mathbb{C}": "ℂ",
            "\\frac": "",  # Remove LaTeX fraction command
            "\\left": "",  # Remove LaTeX sizing commands
            "\\right": "",
            "{": "",       # Remove braces for cleaner display
            "}": ""
        }
        
        for latex, unicode_char in replacements.items():
            statement = statement.replace(latex, unicode_char)
        
        return statement.strip()
    
    @computed_field
    @property
    def short_id(self) -> str:
        """Get shortened ID for display."""
        return self.id.replace("THM_", "")
    
    @computed_field
    @property
    def theorem_type_display(self) -> str:
        """Get formatted theorem type for display."""
        return self.theorem_type.replace("_", " ").title()
    
    @computed_field
    @property
    def symbol_summary(self) -> str:
        """Get summary of symbols used."""
        unique_symbols = list(set(self.symbols))
        if not unique_symbols:
            return "No symbols"
        elif len(unique_symbols) <= 3:
            return ", ".join(unique_symbols)
        else:
            return f"{', '.join(unique_symbols[:3])}, ... ({len(unique_symbols)} total)"
    
    def matches_search(self, query: str) -> bool:
        """
        Check if theorem matches a search query.
        
        Args:
            query: Search query string
            
        Returns:
            bool: True if theorem matches query
        """
        query_lower = query.lower()
        searchable_text = (
            f"{self.statement} {self.natural_language} "
            f"{self.theorem_type} {' '.join(self.symbols)} "
            f"{self.source_lineage.original_formula} "
            f"{' '.join(self.assumptions)}"
        ).lower()
        
        return query_lower in searchable_text
    
    def get_search_score(self, query: str) -> float:
        """
        Get relevance score for search query.
        
        Args:
            query: Search query string
            
        Returns:
            float: Relevance score (0.0 to 1.0)
        """
        if not query or len(query) < 2:
            return 0.0
        
        query_lower = query.lower()
        score = 0.0
        
        # Direct matches in statement get highest score
        if query_lower in self.statement.lower():
            score += 0.4
        
        # Matches in natural language description
        if query_lower in self.natural_language.lower():
            score += 0.3
        
        # Matches in theorem type
        if query_lower in self.theorem_type.lower():
            score += 0.2
        
        # Matches in symbols
        for symbol in self.symbols:
            if query_lower in symbol.lower():
                score += 0.1
                break
        
        # Boost for validation status
        if self.is_validated:
            score *= 1.1
        
        # Boost for confidence
        score *= (0.8 + 0.2 * self.source_lineage.confidence)
        
        return min(score, 1.0)


class FormulaData(BaseModel):
    """Mathematical formula data structure."""
    
    id: str = Field(..., min_length=1)
    expression: str = Field(..., min_length=1)
    sympy_form: Optional[str] = None
    latex_form: Optional[str] = None
    source: str = Field(..., min_length=1)
    extraction_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @computed_field
    @property
    def display_form(self) -> str:
        """Get best display form of the formula."""
        return self.latex_form or self.expression
    
    @computed_field
    @property
    def short_id(self) -> str:
        """Get shortened ID for display."""
        return self.id.replace("FORMULA_", "F")


class ValidationReportSummary(BaseModel):
    """Summary section of validation report."""
    
    total_formulas: int = Field(..., ge=0)
    validated_formulas: int = Field(..., ge=0)
    passed_formulas: int = Field(..., ge=0)
    failed_formulas: int = Field(..., ge=0)
    error_formulas: int = Field(..., ge=0)
    partial_formulas: int = Field(..., ge=0)
    overall_pass_rate: float = Field(..., ge=0.0, le=1.0)
    average_confidence: float = Field(..., ge=0.0, le=1.0)
    validation_time: float = Field(..., ge=0.0)
    timestamp: str


class ValidationReport(BaseModel):
    """Validation test report data."""
    
    summary: ValidationReportSummary
    statistics: Dict[str, Any] = Field(default_factory=dict)
    results_by_status: Dict[str, List[str]] = Field(default_factory=dict)
    errors_summary: List[str] = Field(default_factory=list)
    detailed_results: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    
    @computed_field
    @property
    def success_percentage(self) -> float:
        """Get success rate as percentage."""
        return self.summary.overall_pass_rate * 100
    
    @computed_field
    @property
    def failure_percentage(self) -> float:
        """Get failure rate as percentage."""
        return (1.0 - self.summary.overall_pass_rate) * 100
    
    @computed_field
    @property
    def validation_summary(self) -> str:
        """Get human-readable validation summary."""
        return (f"{self.summary.passed_formulas}/{self.summary.total_formulas} passed "
                f"({self.success_percentage:.1f}%) in {self.summary.validation_time:.3f}s")
    
    @computed_field
    @property
    def error_count(self) -> int:
        """Get total number of errors."""
        return self.summary.error_formulas + self.summary.failed_formulas


class GenerationMetadata(BaseModel):
    """Metadata about theorem generation process."""
    
    total_theorems: int = Field(..., ge=0)
    generation_time: float = Field(..., ge=0.0)
    validation_passes: int = Field(..., ge=0)
    type_distribution: Dict[str, int] = Field(default_factory=dict)
    generator_version: str = Field(..., min_length=1)
    
    @computed_field
    @property
    def theorems_per_second(self) -> float:
        """Calculate generation rate."""
        if self.generation_time <= 0:
            return 0.0
        return self.total_theorems / self.generation_time
    
    @computed_field
    @property
    def pass_rate(self) -> float:
        """Calculate validation pass rate."""
        if self.total_theorems <= 0:
            return 0.0
        return self.validation_passes / self.total_theorems


class TheoremCollection(BaseModel):
    """Collection of theorems with metadata."""
    
    generation_metadata: GenerationMetadata
    theorems: List[Theorem] = Field(default_factory=list)
    
    @computed_field
    @property
    def theorem_count(self) -> int:
        """Get count of theorems."""
        return len(self.theorems)
    
    @computed_field
    @property
    def validated_count(self) -> int:
        """Get count of validated theorems."""
        return sum(1 for t in self.theorems if t.is_validated)
    
    @computed_field
    @property
    def validation_rate(self) -> float:
        """Get overall validation rate."""
        if not self.theorems:
            return 0.0
        return self.validated_count / len(self.theorems)
    
    @computed_field
    @property
    def type_summary(self) -> Dict[str, int]:
        """Get summary of theorem types."""
        type_counts = {}
        for theorem in self.theorems:
            type_counts[theorem.theorem_type] = type_counts.get(theorem.theorem_type, 0) + 1
        return type_counts
    
    def search(self, query: str, limit: Optional[int] = None) -> List[Theorem]:
        """
        Search theorems by query.
        
        Args:
            query: Search query string
            limit: Maximum number of results
            
        Returns:
            List of matching theorems, sorted by relevance
        """
        if not query or len(query) < 2:
            return []
        
        # Get matching theorems with scores
        scored_theorems = [
            (theorem, theorem.get_search_score(query))
            for theorem in self.theorems
            if theorem.matches_search(query)
        ]
        
        # Sort by score (descending)
        scored_theorems.sort(key=lambda x: x[1], reverse=True)
        
        # Extract theorems and apply limit
        results = [theorem for theorem, score in scored_theorems]
        
        if limit:
            results = results[:limit]
        
        return results
    
    def get_by_type(self, theorem_type: str) -> List[Theorem]:
        """Get all theorems of a specific type."""
        return [t for t in self.theorems if t.theorem_type == theorem_type]
    
    def get_by_id(self, theorem_id: str) -> Optional[Theorem]:
        """Get theorem by ID."""
        for theorem in self.theorems:
            if theorem.id == theorem_id:
                return theorem
        return None 