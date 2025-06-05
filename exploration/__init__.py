"""
Phase 4 - Symbolic Pattern Discovery & Exploration

This package provides tools for discovering patterns, detecting gaps, and generating
hypotheses within the validated mathematical knowledge graph.

Main components:
- PatternFinder: Discovers structural similarities between formulas
- GapDetector: Identifies missing connections and concepts
- HypothesisGenerator: Creates and validates new mathematical conjectures
- EmbeddingUtils: Formula embedding and clustering utilities
"""

from .pattern_finder import PatternFinder, PatternCluster, SimilarityMetric
from .gap_detector import GapDetector, Gap, GapType  
from .hypothesis_generator import HypothesisGenerator, Hypothesis, HypothesisStatus
from .utils.embedding import FormulaEmbedder, ClusterVisualizer

__version__ = "0.1.0"
__all__ = [
    "PatternFinder",
    "PatternCluster", 
    "SimilarityMetric",
    "GapDetector",
    "Gap",
    "GapType",
    "HypothesisGenerator", 
    "Hypothesis",
    "HypothesisStatus",
    "FormulaEmbedder",
    "ClusterVisualizer"
] 