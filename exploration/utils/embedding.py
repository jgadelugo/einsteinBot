"""
Formula embedding and clustering utilities.

This module provides tools for:
- Converting SymPy formulas to embedding vectors
- Clustering formulas in embedding space
- Visualizing formula clusters and relationships
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import umap

import sympy as sp
from sympy import sympify, latex


@dataclass 
class EmbeddingResult:
    """Result of formula embedding process."""
    formulas: List[str]
    embeddings: np.ndarray
    embedding_method: str
    clustering_labels: Optional[np.ndarray] = None
    cluster_method: Optional[str] = None
    reduced_embeddings: Optional[np.ndarray] = None
    reduction_method: Optional[str] = None
    metadata: Dict[str, Any] = None


class FormulaEmbedder:
    """
    Converts mathematical formulas to embedding vectors for similarity analysis.
    
    Supports multiple embedding approaches:
    - Structural feature vectors (symbol counts, operation types, etc.)
    - TF-IDF based text embeddings
    - Custom mathematical property vectors
    """
    
    def __init__(self, embedding_method: str = "structural"):
        """
        Initialize the formula embedder.
        
        Args:
            embedding_method: Method for embedding ("structural", "tfidf", "hybrid")
        """
        self.embedding_method = embedding_method
        self.logger = logging.getLogger(__name__)
        
        # TF-IDF vectorizer for text-based embeddings
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=1000,
            stop_words=None
        )
        
        # Cache for parsed formulas
        self.formula_cache: Dict[str, sp.Expr] = {}
    
    def embed_formulas(self, formulas: List[str]) -> EmbeddingResult:
        """
        Convert formulas to embedding vectors.
        
        Args:
            formulas: List of formula strings
            
        Returns:
            EmbeddingResult with formula embeddings
        """
        self.logger.info(f"Embedding {len(formulas)} formulas using {self.embedding_method} method")
        start_time = time.time()
        
        # Parse formulas
        parsed_formulas = self._parse_formulas(formulas)
        
        # Generate embeddings based on method
        if self.embedding_method == "structural":
            embeddings = self._create_structural_embeddings(parsed_formulas)
        elif self.embedding_method == "tfidf":
            embeddings = self._create_tfidf_embeddings(formulas)
        elif self.embedding_method == "hybrid":
            embeddings = self._create_hybrid_embeddings(formulas, parsed_formulas)
        else:
            raise ValueError(f"Unknown embedding method: {self.embedding_method}")
        
        result = EmbeddingResult(
            formulas=formulas,
            embeddings=embeddings,
            embedding_method=self.embedding_method,
            metadata={
                "embedding_time": time.time() - start_time,
                "embedding_dimensions": embeddings.shape[1],
                "valid_formulas": len(parsed_formulas)
            }
        )
        
        self.logger.info(f"Created {embeddings.shape[1]}D embeddings in "
                        f"{time.time() - start_time:.2f}s")
        
        return result
    
    def _parse_formulas(self, formulas: List[str]) -> List[Tuple[str, Optional[sp.Expr]]]:
        """Parse formulas into SymPy expressions."""
        parsed = []
        
        for formula in formulas:
            if formula in self.formula_cache:
                expr = self.formula_cache[formula]
            else:
                try:
                    expr = sympify(formula)
                    self.formula_cache[formula] = expr
                except Exception as e:
                    self.logger.debug(f"Failed to parse '{formula}': {e}")
                    expr = None
                    self.formula_cache[formula] = None
            
            parsed.append((formula, expr))
        
        return parsed
    
    def _create_structural_embeddings(self, parsed_formulas: List[Tuple[str, Optional[sp.Expr]]]) -> np.ndarray:
        """Create embeddings based on structural features of formulas."""
        
        # Define structural features to extract
        feature_extractors = [
            self._extract_symbol_features,
            self._extract_operation_features,
            self._extract_function_features,
            self._extract_complexity_features,
            self._extract_mathematical_features
        ]
        
        all_features = []
        
        for formula, expr in parsed_formulas:
            if expr is None:
                # Create zero vector for unparseable formulas
                features = np.zeros(100)  # Will be adjusted based on actual feature count
            else:
                features = []
                for extractor in feature_extractors:
                    try:
                        feature_vec = extractor(expr)
                        features.extend(feature_vec)
                    except Exception as e:
                        self.logger.debug(f"Feature extraction failed for {formula}: {e}")
                        features.extend([0] * 20)  # Default feature count per extractor
                
                features = np.array(features)
            
            all_features.append(features)
        
        # Convert to numpy array and handle dimension mismatch
        max_length = max(len(f) for f in all_features)
        embeddings = np.zeros((len(all_features), max_length))
        
        for i, features in enumerate(all_features):
            embeddings[i, :len(features)] = features
        
        return embeddings
    
    def _extract_symbol_features(self, expr: sp.Expr) -> List[float]:
        """Extract symbol-based features."""
        features = []
        
        # Count different types of symbols
        symbols = expr.free_symbols
        features.append(len(symbols))  # Total symbol count
        
        # Count specific symbol types
        symbol_names = [str(s) for s in symbols]
        features.append(sum(1 for s in symbol_names if len(s) == 1))  # Single letter symbols
        features.append(sum(1 for s in symbol_names if 'x' in s.lower()))  # X-like symbols
        features.append(sum(1 for s in symbol_names if 'y' in s.lower()))  # Y-like symbols
        features.append(sum(1 for s in symbol_names if any(c.isdigit() for c in s)))  # Numbered symbols
        
        # Greek letters (common in math)
        greek_symbols = ['alpha', 'beta', 'gamma', 'delta', 'theta', 'pi', 'sigma']
        features.append(sum(1 for s in symbol_names if any(g in s.lower() for g in greek_symbols)))
        
        return features
    
    def _extract_operation_features(self, expr: sp.Expr) -> List[float]:
        """Extract operation-based features."""
        features = []
        
        # Count operations by traversing the expression tree
        operations = {
            'Add': 0, 'Mul': 0, 'Pow': 0, 'Rational': 0,
            'Integer': 0, 'Float': 0, 'Symbol': 0
        }
        
        def count_operations(node):
            if hasattr(node, 'func'):
                op_name = node.func.__name__
                if op_name in operations:
                    operations[op_name] += 1
            
            for arg in getattr(node, 'args', []):
                count_operations(arg)
        
        count_operations(expr)
        
        # Add operation counts as features
        features.extend(operations.values())
        
        # Add ratios
        total_ops = sum(operations.values())
        if total_ops > 0:
            features.extend([count / total_ops for count in operations.values()])
        else:
            features.extend([0] * len(operations))
        
        return features
    
    def _extract_function_features(self, expr: sp.Expr) -> List[float]:
        """Extract function-based features."""
        features = []
        
        # Count mathematical functions
        function_counts = {
            'sin': 0, 'cos': 0, 'tan': 0, 'exp': 0, 'log': 0, 'sqrt': 0,
            'sinh': 0, 'cosh': 0, 'tanh': 0, 'asin': 0, 'acos': 0, 'atan': 0
        }
        
        expr_str = str(expr).lower()
        for func in function_counts:
            function_counts[func] = expr_str.count(func)
        
        features.extend(function_counts.values())
        
        # Function type indicators
        features.append(1 if any(count > 0 for count in function_counts.values()) else 0)  # Has functions
        features.append(1 if any(function_counts[f] > 0 for f in ['sin', 'cos', 'tan']) else 0)  # Trigonometric
        features.append(1 if function_counts['exp'] > 0 or function_counts['log'] > 0 else 0)  # Exponential/log
        
        return features
    
    def _extract_complexity_features(self, expr: sp.Expr) -> List[float]:
        """Extract complexity-based features."""
        features = []
        
        # Basic complexity measures
        features.append(sp.count_ops(expr))  # Operation count
        features.append(len(str(expr)))  # String length
        features.append(self._calculate_depth(expr))  # Tree depth
        features.append(len(expr.atoms()))  # Number of atoms
        
        # Polynomial degree if applicable
        try:
            if expr.is_polynomial():
                symbols = expr.free_symbols
                if symbols:
                    degree = sp.degree(expr, list(symbols)[0])
                    features.append(degree)
                else:
                    features.append(0)
            else:
                features.append(-1)  # Not a polynomial
        except:
            features.append(-1)
        
        return features
    
    def _extract_mathematical_features(self, expr: sp.Expr) -> List[float]:
        """Extract mathematical property features."""
        features = []
        
        # Mathematical properties
        try:
            features.append(1 if expr.is_polynomial() else 0)
            features.append(1 if expr.is_rational_function() else 0)
            features.append(1 if expr.is_algebraic_expr() else 0)
            features.append(1 if expr.has(sp.I) else 0)  # Complex numbers
            features.append(1 if expr.has(sp.pi) else 0)  # Contains pi
            features.append(1 if expr.has(sp.E) else 0)  # Contains e
            features.append(1 if expr.has(sp.oo) else 0)  # Contains infinity
        except:
            features.extend([0] * 7)
        
        # Symmetry properties
        try:
            symbols = list(expr.free_symbols)
            if len(symbols) == 1:
                x = symbols[0]
                features.append(1 if expr.subs(x, -x) == expr else 0)  # Even function
                features.append(1 if expr.subs(x, -x) == -expr else 0)  # Odd function
            else:
                features.extend([0, 0])
        except:
            features.extend([0, 0])
        
        return features
    
    def _calculate_depth(self, expr: sp.Expr) -> int:
        """Calculate the depth of the expression tree."""
        if not hasattr(expr, 'args') or not expr.args:
            return 1
        return 1 + max(self._calculate_depth(arg) for arg in expr.args)
    
    def _create_tfidf_embeddings(self, formulas: List[str]) -> np.ndarray:
        """Create TF-IDF based embeddings from formula strings."""
        
        # Preprocess formulas for TF-IDF
        processed_formulas = []
        for formula in formulas:
            # Convert to a more text-like representation
            processed = self._preprocess_formula_for_tfidf(formula)
            processed_formulas.append(processed)
        
        # Fit TF-IDF vectorizer and transform
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(processed_formulas)
        
        return tfidf_matrix.toarray()
    
    def _preprocess_formula_for_tfidf(self, formula: str) -> str:
        """Preprocess formula for TF-IDF representation."""
        
        # Replace mathematical symbols with words
        replacements = {
            '+': ' plus ',
            '-': ' minus ',
            '*': ' times ',
            '/': ' divide ',
            '**': ' power ',
            '^': ' power ',
            '(': ' lparen ',
            ')': ' rparen ',
            '=': ' equals ',
            'sin': ' sine ',
            'cos': ' cosine ',
            'tan': ' tangent ',
            'exp': ' exponential ',
            'log': ' logarithm ',
            'sqrt': ' squareroot ',
            'pi': ' pi ',
            'e': ' euler '
        }
        
        processed = formula
        for symbol, word in replacements.items():
            processed = processed.replace(symbol, word)
        
        return processed
    
    def _create_hybrid_embeddings(self, formulas: List[str], 
                                parsed_formulas: List[Tuple[str, Optional[sp.Expr]]]) -> np.ndarray:
        """Create hybrid embeddings combining structural and TF-IDF features."""
        
        # Get both types of embeddings
        structural_embeddings = self._create_structural_embeddings(parsed_formulas)
        tfidf_embeddings = self._create_tfidf_embeddings(formulas)
        
        # Normalize both embedding types
        structural_norm = structural_embeddings / (np.linalg.norm(structural_embeddings, axis=1, keepdims=True) + 1e-8)
        tfidf_norm = tfidf_embeddings / (np.linalg.norm(tfidf_embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Concatenate embeddings
        hybrid_embeddings = np.concatenate([structural_norm, tfidf_norm], axis=1)
        
        return hybrid_embeddings
    
    def cluster_embeddings(self, embedding_result: EmbeddingResult,
                         method: str = "dbscan", **kwargs) -> EmbeddingResult:
        """
        Cluster formula embeddings.
        
        Args:
            embedding_result: Result from embed_formulas
            method: Clustering method ("dbscan", "kmeans")
            **kwargs: Additional parameters for clustering algorithm
            
        Returns:
            Updated EmbeddingResult with clustering labels
        """
        self.logger.info(f"Clustering embeddings using {method}")
        
        if method == "dbscan":
            eps = kwargs.get('eps', 0.5)
            min_samples = kwargs.get('min_samples', 3)
            clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        elif method == "kmeans":
            n_clusters = kwargs.get('n_clusters', 5)
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        labels = clusterer.fit_predict(embedding_result.embeddings)
        
        embedding_result.clustering_labels = labels
        embedding_result.cluster_method = method
        
        # Add clustering metadata
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # Exclude noise cluster
        embedding_result.metadata['clustering'] = {
            'method': method,
            'n_clusters': n_clusters,
            'n_noise_points': sum(1 for label in labels if label == -1),
            'parameters': kwargs
        }
        
        self.logger.info(f"Found {n_clusters} clusters")
        
        return embedding_result
    
    def reduce_dimensions(self, embedding_result: EmbeddingResult,
                        method: str = "tsne", n_components: int = 2,
                        **kwargs) -> EmbeddingResult:
        """
        Reduce embedding dimensions for visualization.
        
        Args:
            embedding_result: Result from embed_formulas
            method: Reduction method ("tsne", "umap", "pca")
            n_components: Number of components to keep
            **kwargs: Additional parameters for reduction algorithm
            
        Returns:
            Updated EmbeddingResult with reduced embeddings
        """
        self.logger.info(f"Reducing dimensions using {method} to {n_components}D")
        
        if method == "tsne":
            perplexity = kwargs.get('perplexity', min(30, len(embedding_result.formulas) - 1))
            reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        elif method == "umap":
            n_neighbors = kwargs.get('n_neighbors', 15)
            min_dist = kwargs.get('min_dist', 0.1)
            reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, 
                              min_dist=min_dist, random_state=42)
        elif method == "pca":
            reducer = PCA(n_components=n_components, random_state=42)
        else:
            raise ValueError(f"Unknown reduction method: {method}")
        
        reduced_embeddings = reducer.fit_transform(embedding_result.embeddings)
        
        embedding_result.reduced_embeddings = reduced_embeddings
        embedding_result.reduction_method = method
        
        # Add reduction metadata
        embedding_result.metadata['dimension_reduction'] = {
            'method': method,
            'n_components': n_components,
            'original_dimensions': embedding_result.embeddings.shape[1],
            'parameters': kwargs
        }
        
        return embedding_result


class ClusterVisualizer:
    """
    Visualizes formula clusters and relationships.
    
    Creates plots and diagrams to help understand patterns in
    mathematical formula collections.
    """
    
    def __init__(self):
        """Initialize the cluster visualizer."""
        self.logger = logging.getLogger(__name__)
        
        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_clusters(self, embedding_result: EmbeddingResult,
                     output_path: Optional[Union[str, Path]] = None,
                     show_formulas: bool = True,
                     max_formulas_display: int = 20) -> None:
        """
        Plot formula clusters in 2D space.
        
        Args:
            embedding_result: EmbeddingResult with clustering and dimension reduction
            output_path: Optional path to save plot
            show_formulas: Whether to show formula text on plot
            max_formulas_display: Maximum number of formulas to display
        """
        if embedding_result.reduced_embeddings is None:
            self.logger.error("Need reduced embeddings for plotting. Call reduce_dimensions first.")
            return
        
        if embedding_result.reduced_embeddings.shape[1] != 2:
            self.logger.error("Need 2D reduced embeddings for plotting")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get cluster labels (use dummy labels if clustering wasn't performed)
        if embedding_result.clustering_labels is not None:
            labels = embedding_result.clustering_labels
        else:
            labels = np.zeros(len(embedding_result.formulas))
        
        # Plot points colored by cluster
        unique_labels = set(labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            if label == -1:
                # Noise points in black
                mask = labels == label
                ax.scatter(embedding_result.reduced_embeddings[mask, 0],
                          embedding_result.reduced_embeddings[mask, 1],
                          c='black', marker='x', s=50, alpha=0.7, label='Noise')
            else:
                mask = labels == label
                ax.scatter(embedding_result.reduced_embeddings[mask, 0],
                          embedding_result.reduced_embeddings[mask, 1],
                          c=[color], s=100, alpha=0.7, label=f'Cluster {label}')
        
        # Add formula text for a subset of points
        if show_formulas and len(embedding_result.formulas) <= max_formulas_display:
            for i, formula in enumerate(embedding_result.formulas):
                # Truncate long formulas
                display_formula = formula[:30] + "..." if len(formula) > 30 else formula
                ax.annotate(display_formula, 
                          (embedding_result.reduced_embeddings[i, 0], 
                           embedding_result.reduced_embeddings[i, 1]),
                          xytext=(5, 5), textcoords='offset points',
                          fontsize=8, alpha=0.8)
        
        ax.set_xlabel(f'{embedding_result.reduction_method.upper()} Component 1')
        ax.set_ylabel(f'{embedding_result.reduction_method.upper()} Component 2')
        ax.set_title(f'Formula Clusters ({embedding_result.cluster_method} clustering)')
        
        if len(unique_labels) <= 10:  # Only show legend if not too many clusters
            ax.legend()
        
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved cluster plot to {output_path}")
        
        plt.show()
    
    def plot_similarity_matrix(self, embedding_result: EmbeddingResult,
                             output_path: Optional[Union[str, Path]] = None,
                             max_formulas: int = 50) -> None:
        """
        Plot similarity matrix heatmap.
        
        Args:
            embedding_result: EmbeddingResult with embeddings
            output_path: Optional path to save plot
            max_formulas: Maximum number of formulas to include
        """
        # Limit to manageable size
        n_formulas = min(len(embedding_result.formulas), max_formulas)
        embeddings = embedding_result.embeddings[:n_formulas]
        formulas = embedding_result.formulas[:n_formulas]
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        im = ax.imshow(similarity_matrix, cmap='viridis', aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Cosine Similarity')
        
        # Set ticks and labels
        if n_formulas <= 20:  # Only show formula labels if not too many
            # Truncate formulas for display
            formula_labels = [f[:20] + "..." if len(f) > 20 else f for f in formulas]
            ax.set_xticks(range(n_formulas))
            ax.set_yticks(range(n_formulas))
            ax.set_xticklabels(formula_labels, rotation=45, ha='right')
            ax.set_yticklabels(formula_labels)
        else:
            ax.set_xlabel('Formula Index')
            ax.set_ylabel('Formula Index')
        
        ax.set_title('Formula Similarity Matrix')
        
        plt.tight_layout()
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved similarity matrix to {output_path}")
        
        plt.show()
    
    def create_cluster_summary(self, embedding_result: EmbeddingResult) -> Dict[str, Any]:
        """
        Create a summary of clustering results.
        
        Args:
            embedding_result: EmbeddingResult with clustering
            
        Returns:
            Dictionary with cluster summary statistics
        """
        if embedding_result.clustering_labels is None:
            return {"error": "No clustering performed"}
        
        labels = embedding_result.clustering_labels
        unique_labels = set(labels)
        
        summary = {
            "total_formulas": len(embedding_result.formulas),
            "n_clusters": len(unique_labels) - (1 if -1 in unique_labels else 0),
            "n_noise_points": sum(1 for label in labels if label == -1),
            "cluster_sizes": {},
            "cluster_examples": {}
        }
        
        for label in unique_labels:
            if label == -1:
                continue  # Skip noise
            
            cluster_mask = labels == label
            cluster_formulas = [f for i, f in enumerate(embedding_result.formulas) if cluster_mask[i]]
            
            summary["cluster_sizes"][f"cluster_{label}"] = len(cluster_formulas)
            summary["cluster_examples"][f"cluster_{label}"] = cluster_formulas[:3]  # First 3 examples
        
        return summary
    
    def save_visualization_data(self, embedding_result: EmbeddingResult,
                              output_path: Union[str, Path]) -> None:
        """
        Save visualization data to JSON for external use.
        
        Args:
            embedding_result: EmbeddingResult to save
            output_path: Path to output JSON file
        """
        output_path = Path(output_path)
        
        data = {
            "formulas": embedding_result.formulas,
            "embedding_method": embedding_result.embedding_method,
            "embedding_dimensions": embedding_result.embeddings.shape[1],
            "metadata": embedding_result.metadata
        }
        
        # Add clustering data if available
        if embedding_result.clustering_labels is not None:
            data["clustering"] = {
                "method": embedding_result.cluster_method,
                "labels": embedding_result.clustering_labels.tolist(),
                "summary": self.create_cluster_summary(embedding_result)
            }
        
        # Add dimension reduction data if available
        if embedding_result.reduced_embeddings is not None:
            data["dimension_reduction"] = {
                "method": embedding_result.reduction_method,
                "coordinates": embedding_result.reduced_embeddings.tolist(),
                "n_components": embedding_result.reduced_embeddings.shape[1]
            }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        self.logger.info(f"Saved visualization data to {output_path}") 