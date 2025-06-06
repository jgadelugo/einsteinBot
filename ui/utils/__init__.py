"""
MathBot UI Utilities Package

This package contains utility functions and helpers for the MathBot UI:
- graph_utils.py: Graph processing and layout utilities
- ui_logging.py: UI-specific logging utilities
- proof_rendering.py: Mathematical expression and proof rendering utilities

These utilities are designed to be pure functions that can be easily
tested and reused across different UI components.
"""

from .proof_rendering import MathematicalRenderer

__all__ = [
    'MathematicalRenderer'
] 