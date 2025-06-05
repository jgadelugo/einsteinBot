"""
Proof utilities package for MathBot.

This package contains utility functions and classes for proof operations
including caching, timeout handling, and result processing.
"""

from .proof_cache import ProofCache
from .timeout_utils import timeout_handler, ProofTimeout

__all__ = [
    'ProofCache',
    'timeout_handler',
    'ProofTimeout'
]

__version__ = "0.1.0" 