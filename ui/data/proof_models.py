from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime

class ProofViewMode(Enum):
    STEP_BY_STEP = "step_by_step"
    OVERVIEW = "overview"
    COMPARISON = "comparison"
    INTERACTIVE = "interactive"

class ProofMethodType(Enum):
    SYMBOLIC = "symbolic"
    RULE_BASED = "rule_based"
    FORMAL = "formal"
    VALIDATION = "validation"

@dataclass
class ProofStep:
    """Individual proof step with visualization data."""
    step_number: int
    method_type: ProofMethodType
    title: str
    expression_from: str
    expression_to: str
    rule_applied: Optional[str] = None
    justification: str = ""
    execution_time: float = 0.0
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProofTraceData:
    """Complete proof trace information."""
    theorem_id: str
    symbolic_steps: List[ProofStep] = field(default_factory=list)
    rule_steps: List[ProofStep] = field(default_factory=list)
    formal_steps: List[ProofStep] = field(default_factory=list)
    validation_steps: List[ProofStep] = field(default_factory=list)
    total_execution_time: float = 0.0
    success_methods: List[ProofMethodType] = field(default_factory=list)

@dataclass
class ProofVisualizationSession:
    """Proof visualization session state."""
    theorem_id: str
    theorem: Any  # Theorem from Phase 6A
    proof_data: Optional[ProofTraceData] = None
    current_step: int = 0
    current_method: ProofMethodType = ProofMethodType.SYMBOLIC
    view_mode: ProofViewMode = ProofViewMode.STEP_BY_STEP
    show_details: bool = True
    created_at: datetime = field(default_factory=datetime.now) 