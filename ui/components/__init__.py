"""
MathBot UI Components Package

This package contains the individual UI components for the MathBot interface:
- GraphViewer: Interactive mathematical knowledge graph visualization
- TheoremBrowser: Searchable theorem exploration interface  
- ProofViewer: Step-by-step proof trace visualization
- SearchInterface: Advanced search and filtering capabilities

Each component is designed to be modular, reusable, and type-safe.
"""

# Component imports
from .graph_viewer import GraphViewer
from .graph_controls import GraphControls
# from .theorem_browser import TheoremBrowser  (to be implemented)
# from .proof_viewer import ProofViewer (to be implemented)
# from .search_interface import SearchInterface (to be implemented)

__all__ = [
    "GraphViewer",
    "GraphControls",
    # Additional components will be added as they are implemented
] 