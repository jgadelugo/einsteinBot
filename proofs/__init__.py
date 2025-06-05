"""
Core theorem generation and formal proof system for MathBot.

This module transforms validated hypotheses from the pattern discovery phase
into formal mathematical theorems with proper classification and metadata,
and provides symbolic proof capabilities.

Key Components:
- TheoremGenerator: Converts hypotheses to formal theorems
- Theorem: Data structure for formal mathematical statements
- TheoremType: Classification system for different theorem types
- ProofAttemptEngine: Symbolic proof attempt system
- ProofResult: Data structure for proof outcomes
- ProofMethod: Enumeration of available proof methods
"""

from .theorem_generator import (
    Theorem,
    TheoremType,
    SourceLineage,
    TheoremGenerator
)

from .proof_attempt import (
    ProofAttemptEngine,
    ProofResult,
    ProofStep,
    ProofMethod,
    ProofStatus,
    ProofTimeout,
    ProofCache
)

__all__ = [
    # Phase 5A: Theorem Generation
    'Theorem',
    'TheoremType', 
    'SourceLineage',
    'TheoremGenerator',
    
    # Phase 5B: Proof Attempt System
    'ProofAttemptEngine',
    'ProofResult',
    'ProofStep',
    'ProofMethod',
    'ProofStatus',
    'ProofTimeout',
    'ProofCache'
]

__version__ = "0.2.0"  # Updated for Phase 5B 