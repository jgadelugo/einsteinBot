"""
Pattern discovery engine for mathematical formulas.

This module identifies structural similarities between formulas through:
- Symbolic fingerprinting using SymPy AST analysis
- Formula embedding for semantic similarity
- Graph neighborhood analysis for contextual relationships
"""

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from collections import defaultdict, Counter
import hashlib

import sympy as sp
import numpy as np
from sympy import sympify, count_ops, latex
from sympy.core.expr import Expr
from sympy.core.function import Function
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


class SimilarityMetric(Enum):
    """Types of similarity measures."""
    STRUCTURAL = "structural"
    SYMBOLIC = "symbolic"  
    SEMANTIC = "semantic"
    TOPOLOGICAL = "topological"


@dataclass
class FormulaFingerprint:
    """Structural fingerprint of a mathematical formula."""
    formula: str
    symbol_count: Dict[str, int] = field(default_factory=dict)
    operation_count: Dict[str, int] = field(default_factory=dict)
    structure_hash: str = ""
    depth: int = 0
    complexity: int = 0
    function_types: Set[str] = field(default_factory=set)
    symmetries: List[str] = field(default_factory=list)
    parsed_expr: Optional[Expr] = None


@dataclass
class PatternCluster:
    """A cluster of similar mathematical formulas."""
    cluster_id: int
    formulas: List[str] = field(default_factory=list)
    similarity_scores: Dict[str, float] = field(default_factory=dict)
    cluster_center: Optional[str] = None
    common_patterns: Dict[str, Any] = field(default_factory=dict)
    topic_tags: Set[str] = field(default_factory=set)
    confidence_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class PatternFinder:
    """
    Discovers structural patterns and similarities in mathematical formulas.
    
    Uses multiple similarity metrics to identify clusters of related formulas
    and extract common mathematical patterns.
    """
    
    def __init__(self, similarity_threshold: float = 0.7, 
                 min_cluster_size: int = 2):
        """
        Initialize the pattern finder.
        
        Args:
            similarity_threshold: Minimum similarity for clustering
            min_cluster_size: Minimum formulas per cluster
        """
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
        self.logger = logging.getLogger(__name__)
        
        # Cache for computed fingerprints and similarities
        self.fingerprint_cache: Dict[str, FormulaFingerprint] = {}
        self.similarity_cache: Dict[Tuple[str, str], float] = {}
        
    def find_patterns(self, formulas: List[str], 
                     source_metadata: Optional[Dict] = None) -> List[PatternCluster]:
        """
        Discover patterns in a collection of formulas.
        
        Args:
            formulas: List of formula strings to analyze
            source_metadata: Optional metadata about formula sources
            
        Returns:
            List of discovered pattern clusters
        """
        self.logger.info(f"Finding patterns in {len(formulas)} formulas")
        start_time = time.time()
        
        # Generate fingerprints for all formulas
        fingerprints = []
        valid_formulas = []
        
        for formula in formulas:
            try:
                fingerprint = self._generate_fingerprint(formula)
                if fingerprint.parsed_expr is not None:
                    fingerprints.append(fingerprint)
                    valid_formulas.append(formula)
            except Exception as e:
                self.logger.warning(f"Failed to fingerprint formula '{formula}': {e}")
        
        self.logger.info(f"Generated {len(fingerprints)} valid fingerprints")
        
        if len(fingerprints) < self.min_cluster_size:
            self.logger.warning("Insufficient valid formulas for clustering")
            return []
        
        # Compute similarity matrix
        similarity_matrix = self._compute_similarity_matrix(fingerprints)
        
        # Perform clustering
        clusters = self._cluster_formulas(valid_formulas, fingerprints, similarity_matrix)
        
        # Analyze patterns within clusters
        pattern_clusters = []
        for i, cluster_formulas in enumerate(clusters):
            if len(cluster_formulas) >= self.min_cluster_size:
                pattern_cluster = self._analyze_cluster_patterns(
                    i, cluster_formulas, fingerprints, source_metadata
                )
                pattern_clusters.append(pattern_cluster)
        
        self.logger.info(f"Discovered {len(pattern_clusters)} pattern clusters in "
                        f"{time.time() - start_time:.2f}s")
        
        return pattern_clusters
    
    def _generate_fingerprint(self, formula: str) -> FormulaFingerprint:
        """
        Generate a structural fingerprint for a formula.
        
        Args:
            formula: Formula string to fingerprint
            
        Returns:
            FormulaFingerprint object
        """
        if formula in self.fingerprint_cache:
            return self.fingerprint_cache[formula]
        
        fingerprint = FormulaFingerprint(formula=formula)
        
        try:
            # Parse the formula
            expr = sympify(formula)
            fingerprint.parsed_expr = expr
            
            # Count symbols
            for symbol in expr.free_symbols:
                fingerprint.symbol_count[str(symbol)] = fingerprint.symbol_count.get(str(symbol), 0) + 1
            
            # Count operations
            operation_counts = self._count_operations(expr)
            fingerprint.operation_count = operation_counts
            
            # Calculate structural properties
            fingerprint.complexity = count_ops(expr)
            fingerprint.depth = self._calculate_depth(expr)
            
            # Identify function types
            fingerprint.function_types = self._extract_function_types(expr)
            
            # Generate structure hash
            structure_repr = f"{sorted(fingerprint.symbol_count.items())}" \
                           f"{sorted(fingerprint.operation_count.items())}" \
                           f"{fingerprint.depth}{fingerprint.complexity}"
            fingerprint.structure_hash = hashlib.md5(structure_repr.encode()).hexdigest()
            
            # Detect symmetries
            fingerprint.symmetries = self._detect_symmetries(expr)
            
        except Exception as e:
            self.logger.debug(f"Error generating fingerprint for '{formula}': {e}")
        
        self.fingerprint_cache[formula] = fingerprint
        return fingerprint
    
    def _count_operations(self, expr: Expr) -> Dict[str, int]:
        """Count different types of operations in an expression."""
        operations = defaultdict(int)
        
        def count_recursive(node):
            if hasattr(node, 'func'):
                op_name = node.func.__name__
                operations[op_name] += 1
            
            for arg in node.args if hasattr(node, 'args') else []:
                count_recursive(arg)
        
        count_recursive(expr)
        return dict(operations)
    
    def _calculate_depth(self, expr: Expr) -> int:
        """Calculate the maximum depth of nested operations."""
        if not hasattr(expr, 'args') or not expr.args:
            return 1
        
        return 1 + max(self._calculate_depth(arg) for arg in expr.args)
    
    def _extract_function_types(self, expr: Expr) -> Set[str]:
        """Extract types of functions used in the expression."""
        functions = set()
        
        def extract_recursive(node):
            if isinstance(node, Function):
                functions.add(type(node).__name__)
            
            for arg in node.args if hasattr(node, 'args') else []:
                extract_recursive(arg)
        
        extract_recursive(expr)
        return functions
    
    def _detect_symmetries(self, expr: Expr) -> List[str]:
        """Detect potential symmetries in the expression."""
        symmetries = []
        
        try:
            # Check for even/odd symmetries
            symbols = list(expr.free_symbols)
            if len(symbols) == 1:
                x = symbols[0]
                if expr.subs(x, -x) == expr:
                    symmetries.append("even")
                elif expr.subs(x, -x) == -expr:
                    symmetries.append("odd")
            
            # Check for periodic patterns (basic)
            if any(func in str(expr) for func in ['sin', 'cos', 'tan']):
                symmetries.append("periodic")
                
        except Exception:
            pass  # Ignore symmetry detection errors
        
        return symmetries
    
    def _compute_similarity_matrix(self, fingerprints: List[FormulaFingerprint]) -> np.ndarray:
        """
        Compute similarity matrix between formula fingerprints.
        
        Args:
            fingerprints: List of formula fingerprints
            
        Returns:
            Similarity matrix as numpy array
        """
        n = len(fingerprints)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarity = 1.0
                else:
                    similarity = self._calculate_similarity(fingerprints[i], fingerprints[j])
                
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
        
        return similarity_matrix
    
    def _calculate_similarity(self, fp1: FormulaFingerprint, fp2: FormulaFingerprint) -> float:
        """
        Calculate similarity between two formula fingerprints.
        
        Args:
            fp1, fp2: Formula fingerprints to compare
            
        Returns:
            Similarity score between 0 and 1
        """
        cache_key = (fp1.formula, fp2.formula)
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        # Structural similarity (hash-based)
        structural_sim = 1.0 if fp1.structure_hash == fp2.structure_hash else 0.0
        
        # Symbol similarity (Jaccard coefficient)
        symbols1 = set(fp1.symbol_count.keys())
        symbols2 = set(fp2.symbol_count.keys())
        symbol_sim = len(symbols1 & symbols2) / len(symbols1 | symbols2) if (symbols1 | symbols2) else 0.0
        
        # Operation similarity
        ops1 = set(fp1.operation_count.keys())
        ops2 = set(fp2.operation_count.keys())
        op_sim = len(ops1 & ops2) / len(ops1 | ops2) if (ops1 | ops2) else 0.0
        
        # Function type similarity
        func_sim = len(fp1.function_types & fp2.function_types) / len(fp1.function_types | fp2.function_types) \
                   if (fp1.function_types | fp2.function_types) else 0.0
        
        # Complexity similarity (normalized difference)
        complexity_diff = abs(fp1.complexity - fp2.complexity)
        max_complexity = max(fp1.complexity, fp2.complexity)
        complexity_sim = 1.0 - (complexity_diff / max_complexity) if max_complexity > 0 else 1.0
        
        # Weighted combination
        similarity = (
            0.3 * structural_sim + 
            0.25 * symbol_sim + 
            0.25 * op_sim + 
            0.1 * func_sim + 
            0.1 * complexity_sim
        )
        
        self.similarity_cache[cache_key] = similarity
        return similarity
    
    def _cluster_formulas(self, formulas: List[str], fingerprints: List[FormulaFingerprint], 
                         similarity_matrix: np.ndarray) -> List[List[str]]:
        """
        Cluster formulas based on similarity matrix.
        
        Args:
            formulas: Original formula strings
            fingerprints: Formula fingerprints
            similarity_matrix: Precomputed similarity matrix
            
        Returns:
            List of formula clusters
        """
        # Convert similarity to distance for clustering
        distance_matrix = 1.0 - similarity_matrix
        
        # Use DBSCAN for density-based clustering
        clustering = DBSCAN(
            metric='precomputed',
            eps=1.0 - self.similarity_threshold,
            min_samples=self.min_cluster_size
        )
        
        cluster_labels = clustering.fit_predict(distance_matrix)
        
        # Group formulas by cluster
        clusters = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            if label != -1:  # -1 indicates noise/outliers
                clusters[label].append(formulas[i])
        
        return list(clusters.values())
    
    def _analyze_cluster_patterns(self, cluster_id: int, formulas: List[str], 
                                fingerprints: List[FormulaFingerprint],
                                source_metadata: Optional[Dict] = None) -> PatternCluster:
        """
        Analyze patterns within a cluster of similar formulas.
        
        Args:
            cluster_id: Unique identifier for the cluster
            formulas: Formulas in the cluster
            fingerprints: All fingerprints (for lookup)
            source_metadata: Optional source metadata
            
        Returns:
            PatternCluster with analyzed patterns
        """
        cluster = PatternCluster(cluster_id=cluster_id, formulas=formulas)
        
        # Find fingerprints for cluster formulas
        cluster_fingerprints = []
        for formula in formulas:
            for fp in fingerprints:
                if fp.formula == formula:
                    cluster_fingerprints.append(fp)
                    break
        
        if not cluster_fingerprints:
            return cluster
        
        # Analyze common patterns
        cluster.common_patterns = self._extract_common_patterns(cluster_fingerprints)
        
        # Calculate cluster center (most representative formula)
        cluster.cluster_center = self._find_cluster_center(cluster_fingerprints)
        
        # Extract topic tags from metadata
        if source_metadata:
            cluster.topic_tags = self._extract_topic_tags(formulas, source_metadata)
        
        # Calculate confidence score
        cluster.confidence_score = self._calculate_cluster_confidence(cluster_fingerprints)
        
        # Add metadata
        cluster.metadata = {
            "avg_complexity": np.mean([fp.complexity for fp in cluster_fingerprints]),
            "avg_depth": np.mean([fp.depth for fp in cluster_fingerprints]),
            "total_unique_symbols": len(set().union(*[fp.symbol_count.keys() for fp in cluster_fingerprints])),
            "creation_time": time.time()
        }
        
        return cluster
    
    def _extract_common_patterns(self, fingerprints: List[FormulaFingerprint]) -> Dict[str, Any]:
        """Extract common patterns across fingerprints in a cluster."""
        if not fingerprints:
            return {}
        
        # Find common symbols
        common_symbols = set(fingerprints[0].symbol_count.keys())
        for fp in fingerprints[1:]:
            common_symbols &= set(fp.symbol_count.keys())
        
        # Find common operations
        common_operations = set(fingerprints[0].operation_count.keys())
        for fp in fingerprints[1:]:
            common_operations &= set(fp.operation_count.keys())
        
        # Find common function types
        common_functions = set(fingerprints[0].function_types)
        for fp in fingerprints[1:]:
            common_functions &= fp.function_types
        
        # Find common symmetries
        common_symmetries = set(fingerprints[0].symmetries)
        for fp in fingerprints[1:]:
            common_symmetries &= set(fp.symmetries)
        
        return {
            "common_symbols": list(common_symbols),
            "common_operations": list(common_operations),
            "common_functions": list(common_functions),
            "common_symmetries": list(common_symmetries),
            "complexity_range": [
                min(fp.complexity for fp in fingerprints),
                max(fp.complexity for fp in fingerprints)
            ],
            "depth_range": [
                min(fp.depth for fp in fingerprints),
                max(fp.depth for fp in fingerprints)
            ]
        }
    
    def _find_cluster_center(self, fingerprints: List[FormulaFingerprint]) -> str:
        """Find the most representative formula in a cluster."""
        if len(fingerprints) == 1:
            return fingerprints[0].formula
        
        # Calculate average similarity of each formula to all others
        best_formula = fingerprints[0].formula
        best_avg_similarity = 0.0
        
        for i, fp1 in enumerate(fingerprints):
            similarities = []
            for j, fp2 in enumerate(fingerprints):
                if i != j:
                    similarities.append(self._calculate_similarity(fp1, fp2))
            
            avg_similarity = np.mean(similarities)
            if avg_similarity > best_avg_similarity:
                best_avg_similarity = avg_similarity
                best_formula = fp1.formula
        
        return best_formula
    
    def _extract_topic_tags(self, formulas: List[str], 
                           source_metadata: Dict) -> Set[str]:
        """Extract topic tags for cluster formulas from source metadata."""
        tags = set()
        
        # Look for topic information in metadata
        if "detailed_formulas" in source_metadata:
            for detailed in source_metadata["detailed_formulas"]:
                if detailed.get("expression") in formulas:
                    if "metadata" in detailed and "topic" in detailed["metadata"]:
                        tags.add(detailed["metadata"]["topic"])
        
        return tags
    
    def _calculate_cluster_confidence(self, fingerprints: List[FormulaFingerprint]) -> float:
        """Calculate confidence score for a cluster based on internal similarity."""
        if len(fingerprints) < 2:
            return 1.0
        
        # Calculate average pairwise similarity within cluster
        similarities = []
        for i in range(len(fingerprints)):
            for j in range(i + 1, len(fingerprints)):
                similarity = self._calculate_similarity(fingerprints[i], fingerprints[j])
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def save_patterns(self, clusters: List[PatternCluster], output_path: Union[str, Path]) -> None:
        """
        Save discovered patterns to a JSON file.
        
        Args:
            clusters: Pattern clusters to save
            output_path: Path to output file
        """
        output_path = Path(output_path)
        
        # Convert clusters to serializable format
        patterns_data = {
            "discovery_metadata": {
                "total_clusters": len(clusters),
                "similarity_threshold": self.similarity_threshold,
                "min_cluster_size": self.min_cluster_size,
                "discovery_time": time.time()
            },
            "clusters": [
                {
                    "cluster_id": cluster.cluster_id,
                    "formulas": cluster.formulas,
                    "cluster_center": cluster.cluster_center,
                    "common_patterns": cluster.common_patterns,
                    "topic_tags": list(cluster.topic_tags),
                    "confidence_score": cluster.confidence_score,
                    "metadata": cluster.metadata
                }
                for cluster in clusters
            ]
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(patterns_data, f, indent=2, default=str)
        
        self.logger.info(f"Saved {len(clusters)} pattern clusters to {output_path}") 