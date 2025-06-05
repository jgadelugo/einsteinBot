"""
Base interfaces for formal theorem proving systems.

This module defines abstract base classes and data structures for integrating
external formal theorem proving systems like Lean 4, Coq, Isabelle/HOL, etc.

The design provides a unified interface for:
- Translating SymPy theorems to formal system syntax
- Attempting automated proof generation
- Verifying formal proofs
- Managing formal proof results and metadata
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import time


class FormalSystemType(Enum):
    """Enumeration of supported formal theorem proving systems."""
    LEAN4 = "lean4"
    COQ = "coq"
    ISABELLE = "isabelle"
    AGDA = "agda"
    
    def __str__(self) -> str:
        """Return string representation."""
        return self.value
    
    @classmethod
    def from_string(cls, value: str) -> 'FormalSystemType':
        """Create FormalSystemType from string value."""
        try:
            return cls(value.lower())
        except ValueError:
            valid_types = [ft.value for ft in cls]
            raise ValueError(f"Invalid formal system type '{value}'. Valid types: {valid_types}")


@dataclass
class FormalProof:
    """
    Data structure representing the result of a formal proof attempt.
    
    This class encapsulates all information about a formal proof attempt,
    including the formal statement, proof code, verification status, and
    performance metrics.
    
    Attributes:
        theorem_id: Unique identifier for the theorem
        system_type: Type of formal system used
        formal_statement: Theorem statement in formal system syntax
        formal_proof: Proof code in formal system syntax
        verification_status: Status of proof verification
        error_messages: List of error messages if proof failed
        proof_size: Size of proof code in characters
        compilation_time: Time taken to compile/verify proof
        metadata: Additional metadata about the proof attempt
    """
    theorem_id: str
    system_type: FormalSystemType
    formal_statement: str
    formal_proof: str
    verification_status: str  # "proved", "failed", "timeout", "error"
    error_messages: List[str] = field(default_factory=list)
    proof_size: int = 0
    compilation_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Initialize computed fields."""
        if self.proof_size == 0:
            self.proof_size = len(self.formal_proof)
    
    def is_successful(self) -> bool:
        """Check if the proof was successfully verified."""
        return self.verification_status == "proved"
    
    def has_errors(self) -> bool:
        """Check if there were errors during proof attempt."""
        return bool(self.error_messages) or self.verification_status == "error"
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the proof attempt."""
        return {
            'theorem_id': self.theorem_id,
            'system': self.system_type.value,
            'status': self.verification_status,
            'proof_size': self.proof_size,
            'compilation_time': self.compilation_time,
            'has_errors': self.has_errors(),
            'error_count': len(self.error_messages)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'theorem_id': self.theorem_id,
            'system_type': self.system_type.value,
            'formal_statement': self.formal_statement,
            'formal_proof': self.formal_proof,
            'verification_status': self.verification_status,
            'error_messages': self.error_messages,
            'proof_size': self.proof_size,
            'compilation_time': self.compilation_time,
            'metadata': self.metadata
        }


class FormalSystemInterface(ABC):
    """
    Abstract base class for formal theorem proving system interfaces.
    
    This class defines the contract that all formal system integrations
    must implement. It provides a unified interface for theorem translation,
    proof attempts, and verification across different formal systems.
    
    Implementations should handle:
    - System-specific installation and setup
    - Theorem translation to system syntax
    - Automated proof attempts with timeouts
    - Proof verification and error handling
    - Performance monitoring and optimization
    """
    
    def __init__(self, system_path: Optional[str] = None, timeout: int = 30):
        """
        Initialize the formal system interface.
        
        Args:
            system_path: Path to the formal system executable
            timeout: Default timeout for proof attempts in seconds
        """
        self.system_path = system_path
        self.timeout = timeout
        self.system_type = self._get_system_type()
        
        # Performance tracking
        self.total_attempts = 0
        self.successful_proofs = 0
        self.total_time = 0.0
    
    @abstractmethod
    def _get_system_type(self) -> FormalSystemType:
        """Get the type of formal system this interface represents."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the formal system is installed and available.
        
        Returns:
            True if system is available, False otherwise
        """
        pass
    
    @abstractmethod
    def translate_theorem(self, theorem) -> str:
        """
        Translate a SymPy theorem to formal system syntax.
        
        Args:
            theorem: Theorem object from Phase 5A with SymPy expressions
            
        Returns:
            Formal statement in system-specific syntax
            
        Raises:
            TranslationError: If translation fails
        """
        pass
    
    @abstractmethod
    def attempt_proof(self, formal_statement: str, timeout: Optional[int] = None) -> FormalProof:
        """
        Attempt to prove a theorem in the formal system.
        
        Args:
            formal_statement: Theorem statement in formal system syntax
            timeout: Timeout in seconds (uses default if None)
            
        Returns:
            FormalProof object with results
            
        Raises:
            ProofError: If proof attempt encounters critical errors
        """
        pass
    
    @abstractmethod
    def verify_proof(self, formal_statement: str, formal_proof: str) -> bool:
        """
        Verify that a formal proof is correct.
        
        Args:
            formal_statement: Theorem statement in formal system syntax
            formal_proof: Proof code in formal system syntax
            
        Returns:
            True if proof is valid, False otherwise
            
        Raises:
            VerificationError: If verification process fails
        """
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get performance statistics for this formal system interface.
        
        Returns:
            Dictionary containing performance metrics
        """
        success_rate = (self.successful_proofs / self.total_attempts) if self.total_attempts > 0 else 0.0
        avg_time = (self.total_time / self.total_attempts) if self.total_attempts > 0 else 0.0
        
        return {
            'system_type': self.system_type.value,
            'total_attempts': self.total_attempts,
            'successful_proofs': self.successful_proofs,
            'success_rate': success_rate,
            'total_time': self.total_time,
            'average_time_per_attempt': avg_time,
            'is_available': self.is_available()
        }
    
    def reset_statistics(self) -> None:
        """Reset performance statistics."""
        self.total_attempts = 0
        self.successful_proofs = 0
        self.total_time = 0.0
    
    def _update_statistics(self, proof_result: FormalProof) -> None:
        """Update performance statistics based on proof result."""
        self.total_attempts += 1
        self.total_time += proof_result.compilation_time
        
        if proof_result.is_successful():
            self.successful_proofs += 1
    
    def __str__(self) -> str:
        """Return string representation."""
        return f"{self.__class__.__name__}(system={self.system_type.value})"
    
    def __repr__(self) -> str:
        """Return detailed string representation."""
        stats = self.get_statistics()
        return (
            f"{self.__class__.__name__}("
            f"system={self.system_type.value}, "
            f"available={stats['is_available']}, "
            f"success_rate={stats['success_rate']:.2%})"
        )


class TranslationError(Exception):
    """Exception raised when theorem translation fails."""
    pass


class ProofError(Exception):
    """Exception raised when proof attempt encounters critical errors."""
    pass


class VerificationError(Exception):
    """Exception raised when proof verification fails."""
    pass


# Utility functions for working with formal systems

def get_available_systems() -> List[FormalSystemType]:
    """
    Get list of available formal systems on the current system.
    
    Returns:
        List of FormalSystemType enums for available systems
    """
    available = []
    
    # This would be implemented to check for actual system availability
    # For now, return empty list as no systems are integrated yet
    
    return available


def create_formal_system_interface(system_type: FormalSystemType, **kwargs) -> FormalSystemInterface:
    """
    Factory function to create formal system interfaces.
    
    Args:
        system_type: Type of formal system to create interface for
        **kwargs: Additional arguments for system initialization
        
    Returns:
        FormalSystemInterface instance
        
    Raises:
        NotImplementedError: If system type is not supported
    """
    if system_type == FormalSystemType.LEAN4:
        from .lean4_interface import Lean4Interface
        return Lean4Interface(**kwargs)
    elif system_type == FormalSystemType.COQ:
        # Future implementation
        raise NotImplementedError(f"Coq interface not yet implemented")
    elif system_type == FormalSystemType.ISABELLE:
        # Future implementation
        raise NotImplementedError(f"Isabelle interface not yet implemented")
    elif system_type == FormalSystemType.AGDA:
        # Future implementation
        raise NotImplementedError(f"Agda interface not yet implemented")
    else:
        raise ValueError(f"Unknown formal system type: {system_type}") 