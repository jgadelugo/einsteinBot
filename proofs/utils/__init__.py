"""
Proof utilities package for MathBot.

This package contains utility functions and classes for proof operations
including caching, timeout handling, logical rule systems, and result processing.
"""

from .proof_cache import ProofCache
from .timeout_utils import timeout_handler, ProofTimeout
from .logic import RuleType, LogicalRule, RuleDatabase, LogicalRuleEngine

__all__ = [
    'ProofCache',
    'timeout_handler',
    'ProofTimeout',
    'RuleType',
    'LogicalRule',
    'RuleDatabase',
    'LogicalRuleEngine'
]

__version__ = "0.1.0" 