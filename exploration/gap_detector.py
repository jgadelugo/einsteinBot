"""
Gap detection engine for mathematical knowledge graphs.

This module identifies potential gaps in mathematical knowledge by analyzing:
- Sparsely connected nodes (isolated concepts)
- Disconnected clusters (missing bridges)
- Missing formula variations and generalizations
- Incomplete transformation paths between related concepts
"""

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from collections import defaultdict, Counter
import itertools

import sympy as sp
import numpy as np
from sympy import sympify, symbols, diff, integrate, simplify, expand, factor
from sympy.core.expr import Expr
import networkx as nx


class GapType(Enum):
    """Types of gaps that can be detected."""
    MISSING_FORMULA = "missing_formula"
    DISCONNECTED_CLUSTER = "disconnected_cluster"
    SPARSE_CONNECTION = "sparse_connection"
    MISSING_VARIATION = "missing_variation"
    INCOMPLETE_TRANSFORMATION = "incomplete_transformation"
    DOMAIN_GAP = "domain_gap"


@dataclass
class Gap:
    """Represents a detected gap in mathematical knowledge."""
    gap_id: str
    gap_type: GapType
    title: str
    description: str
    confidence_score: float
    priority: float
    suggested_formulas: List[str] = field(default_factory=list)
    related_concepts: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConceptNode:
    """Represents a mathematical concept in the knowledge graph."""
    concept_id: str
    formulas: List[str] = field(default_factory=list)  
    topics: Set[str] = field(default_factory=set)
    connections: Set[str] = field(default_factory=set)
    complexity_level: int = 0
    validation_scores: List[float] = field(default_factory=list)


class GapDetector:
    """
    Detects gaps and missing connections in mathematical knowledge graphs.
    
    Analyzes formula collections to identify potential areas for expansion,
    missing intermediate concepts, and disconnected mathematical domains.
    """
    
    def __init__(self, min_connection_threshold: int = 2,
                 isolation_threshold: float = 0.1):
        """
        Initialize the gap detector.
        
        Args:
            min_connection_threshold: Minimum connections to not be considered sparse
            isolation_threshold: Threshold for detecting isolated clusters
        """
        self.min_connection_threshold = min_connection_threshold
        self.isolation_threshold = isolation_threshold
        self.logger = logging.getLogger(__name__)
        
        # Knowledge graph representation
        self.knowledge_graph = nx.Graph()
        self.concept_nodes: Dict[str, ConceptNode] = {}
        
    def detect_gaps(self, formulas_data: List[Dict], 
                   validation_results: Optional[Dict] = None) -> List[Gap]:
        """
        Detect gaps in mathematical knowledge from formula data.
        
        Args:
            formulas_data: List of formula dictionaries with metadata
            validation_results: Optional validation results for formulas
            
        Returns:
            List of detected gaps with suggestions
        """
        self.logger.info(f"Detecting gaps in {len(formulas_data)} formulas")
        start_time = time.time()
        
        # Build knowledge graph from formulas
        self._build_knowledge_graph(formulas_data, validation_results)
        
        gaps = []
        
        # Detect different types of gaps
        gaps.extend(self._detect_sparse_connections())
        gaps.extend(self._detect_disconnected_clusters())
        gaps.extend(self._detect_missing_variations())
        gaps.extend(self._detect_incomplete_transformations())
        gaps.extend(self._detect_domain_gaps())
        
        # Rank gaps by priority
        gaps = self._rank_gaps(gaps)
        
        self.logger.info(f"Detected {len(gaps)} gaps in "
                        f"{time.time() - start_time:.2f}s")
        
        return gaps
    
    def _build_knowledge_graph(self, formulas_data: List[Dict], 
                              validation_results: Optional[Dict] = None) -> None:
        """
        Build a knowledge graph from formula data.
        
        Args:
            formulas_data: Formula data with metadata
            validation_results: Optional validation scores
        """
        self.knowledge_graph.clear()
        self.concept_nodes.clear()
        
        # Extract concepts and formulas
        for data in formulas_data:
            self._process_formula_data(data, validation_results)
        
        # Build connections between related concepts
        self._establish_concept_connections()
        
        self.logger.info(f"Built knowledge graph with {len(self.concept_nodes)} concepts "
                        f"and {self.knowledge_graph.number_of_edges()} connections")
    
    def _process_formula_data(self, data: Dict, validation_results: Optional[Dict] = None) -> None:
        """Process individual formula data and extract concepts."""
        
        # Process simple formulas list
        if "formulas" in data:
            for formula in data["formulas"]:
                concept_id = self._extract_concept_id(formula)
                if concept_id not in self.concept_nodes:
                    self.concept_nodes[concept_id] = ConceptNode(concept_id=concept_id)
                self.concept_nodes[concept_id].formulas.append(formula)
        
        # Process detailed formulas with metadata
        if "detailed_formulas" in data:
            for detailed in data["detailed_formulas"]:
                formula = detailed.get("expression", "")
                if not formula:
                    continue
                    
                concept_id = self._extract_concept_id(formula)
                if concept_id not in self.concept_nodes:
                    self.concept_nodes[concept_id] = ConceptNode(concept_id=concept_id)
                
                node = self.concept_nodes[concept_id]
                node.formulas.append(formula)
                
                # Add topic information
                if "metadata" in detailed:
                    if "topic" in detailed["metadata"]:
                        node.topics.add(detailed["metadata"]["topic"])
                    if "type" in detailed["metadata"]:
                        node.topics.add(detailed["metadata"]["type"])
                
                # Add validation scores if available
                if validation_results and formula in validation_results:
                    node.validation_scores.append(validation_results[formula])
        
        # Add nodes to graph
        for concept_id in self.concept_nodes:
            if not self.knowledge_graph.has_node(concept_id):
                self.knowledge_graph.add_node(concept_id)
    
    def _extract_concept_id(self, formula: str) -> str:
        """Extract a concept identifier from a formula."""
        try:
            expr = sympify(formula)
            
            # Use combination of symbols and main operations as concept ID
            symbols_str = "_".join(sorted(str(s) for s in expr.free_symbols))
            
            # Get main function types
            functions = set()
            def extract_functions(node):
                if hasattr(node, 'func') and hasattr(node.func, '__name__'):
                    functions.add(node.func.__name__)
                for arg in getattr(node, 'args', []):
                    extract_functions(arg)
            
            extract_functions(expr)
            functions_str = "_".join(sorted(functions))
            
            concept_id = f"{symbols_str}_{functions_str}".replace("__", "_").strip("_")
            return concept_id if concept_id else str(hash(formula))[:8]
            
        except Exception:
            # Fallback to hash-based ID
            return str(hash(formula))[:8]
    
    def _establish_concept_connections(self) -> None:
        """Establish connections between related concepts in the knowledge graph."""
        
        concept_list = list(self.concept_nodes.keys())
        
        for i, concept1 in enumerate(concept_list):
            for j, concept2 in enumerate(concept_list[i+1:], i+1):
                
                similarity = self._calculate_concept_similarity(
                    self.concept_nodes[concept1], 
                    self.concept_nodes[concept2]
                )
                
                # Create edge if concepts are sufficiently similar
                if similarity > 0.3:  # Threshold for relatedness
                    self.knowledge_graph.add_edge(concept1, concept2, weight=similarity)
                    self.concept_nodes[concept1].connections.add(concept2)
                    self.concept_nodes[concept2].connections.add(concept1)
    
    def _calculate_concept_similarity(self, node1: ConceptNode, node2: ConceptNode) -> float:
        """Calculate similarity between two concept nodes."""
        
        # Topic similarity
        topic_sim = 0.0
        if node1.topics and node2.topics:
            common_topics = len(node1.topics & node2.topics)
            total_topics = len(node1.topics | node2.topics)
            topic_sim = common_topics / total_topics if total_topics > 0 else 0.0
        
        # Formula symbol similarity
        symbol_sim = 0.0
        try:
            symbols1 = set()
            symbols2 = set()
            
            for formula in node1.formulas:
                try:
                    expr = sympify(formula)
                    symbols1.update(str(s) for s in expr.free_symbols)
                except:
                    pass
            
            for formula in node2.formulas:
                try:
                    expr = sympify(formula)
                    symbols2.update(str(s) for s in expr.free_symbols)
                except:
                    pass
            
            if symbols1 and symbols2:
                common_symbols = len(symbols1 & symbols2)
                total_symbols = len(symbols1 | symbols2)
                symbol_sim = common_symbols / total_symbols if total_symbols > 0 else 0.0
        
        except Exception:
            pass
        
        # Weighted combination
        return 0.6 * topic_sim + 0.4 * symbol_sim
    
    def _detect_sparse_connections(self) -> List[Gap]:
        """Detect concepts with few connections (potentially isolated)."""
        gaps = []
        
        for concept_id, node in self.concept_nodes.items():
            connection_count = len(node.connections)
            
            if connection_count < self.min_connection_threshold:
                # This is a sparsely connected concept
                gap = Gap(
                    gap_id=f"sparse_{concept_id}",
                    gap_type=GapType.SPARSE_CONNECTION,
                    title=f"Sparsely Connected Concept: {concept_id}",
                    description=f"Concept '{concept_id}' has only {connection_count} "
                               f"connections, suggesting it may be isolated or "
                               f"missing important relationships.",
                    confidence_score=0.8,
                    priority=self._calculate_sparse_priority(node),
                    related_concepts=[concept_id],
                    evidence={
                        "connection_count": connection_count,
                        "formulas": node.formulas[:3],  # Show first few formulas
                        "topics": list(node.topics)
                    }
                )
                gaps.append(gap)
        
        return gaps
    
    def _detect_disconnected_clusters(self) -> List[Gap]:
        """Detect disconnected clusters in the knowledge graph."""
        gaps = []
        
        # Find connected components
        components = list(nx.connected_components(self.knowledge_graph))
        
        if len(components) > 1:
            # Multiple disconnected components exist
            component_info = []
            
            for i, component in enumerate(components):
                topics = set()
                for concept_id in component:
                    topics.update(self.concept_nodes[concept_id].topics)
                
                component_info.append({
                    "size": len(component),
                    "topics": topics,
                    "concepts": list(component)
                })
            
            # Look for components that should potentially be connected
            for i, comp1 in enumerate(component_info):
                for j, comp2 in enumerate(component_info[i+1:], i+1):
                    
                    # Check if they share related topics
                    shared_topics = comp1["topics"] & comp2["topics"]
                    if shared_topics or self._topics_are_related(comp1["topics"], comp2["topics"]):
                        
                        gap = Gap(
                            gap_id=f"disconnect_{i}_{j}",
                            gap_type=GapType.DISCONNECTED_CLUSTER,
                            title=f"Disconnected Mathematical Domains",
                            description=f"Two clusters with related topics "
                                       f"({shared_topics}) are not connected, "
                                       f"suggesting missing bridging concepts.",
                            confidence_score=0.7,
                            priority=0.8,
                            related_concepts=comp1["concepts"][:3] + comp2["concepts"][:3],
                            evidence={
                                "cluster1_size": comp1["size"],
                                "cluster2_size": comp2["size"],
                                "shared_topics": list(shared_topics),
                                "cluster1_topics": list(comp1["topics"]),
                                "cluster2_topics": list(comp2["topics"])
                            }
                        )
                        gaps.append(gap)
        
        return gaps
    
    def _detect_missing_variations(self) -> List[Gap]:
        """Detect missing variations of existing formulas."""
        gaps = []
        
        # Look for patterns where variations might be missing
        for concept_id, node in self.concept_nodes.items():
            if len(node.formulas) < 2:  # Concepts with only one formula
                continue
            
            missing_variations = self._suggest_formula_variations(node.formulas)
            
            if missing_variations:
                gap = Gap(
                    gap_id=f"variation_{concept_id}",
                    gap_type=GapType.MISSING_VARIATION,
                    title=f"Missing Formula Variations: {concept_id}",
                    description=f"Concept has {len(node.formulas)} formulas but "
                               f"may be missing common variations or generalizations.",
                    confidence_score=0.6,
                    priority=0.5,
                    suggested_formulas=missing_variations,
                    related_concepts=[concept_id],
                    evidence={
                        "existing_formulas": node.formulas,
                        "topics": list(node.topics)
                    }
                )
                gaps.append(gap)
        
        return gaps
    
    def _detect_incomplete_transformations(self) -> List[Gap]:
        """Detect incomplete transformation paths between related formulas."""
        gaps = []
        
        # Look for formulas that could be transformations of each other
        all_formulas = []
        for node in self.concept_nodes.values():
            all_formulas.extend(node.formulas)
        
        transformation_gaps = self._find_transformation_gaps(all_formulas)
        
        for i, (formula1, formula2, missing_steps) in enumerate(transformation_gaps):
            gap = Gap(
                gap_id=f"transform_{i}",
                gap_type=GapType.INCOMPLETE_TRANSFORMATION,
                title=f"Missing Transformation Steps",
                description=f"Transformation from '{formula1}' to '{formula2}' "
                           f"may be missing intermediate steps.",
                confidence_score=0.5,
                priority=0.4,
                suggested_formulas=missing_steps,
                evidence={
                    "source_formula": formula1,
                    "target_formula": formula2,
                    "suggested_steps": missing_steps
                }
            )
            gaps.append(gap)
        
        return gaps
    
    def _detect_domain_gaps(self) -> List[Gap]:
        """Detect gaps in mathematical domain coverage."""
        gaps = []
        
        # Analyze topic distribution
        topic_counts = Counter()
        for node in self.concept_nodes.values():
            for topic in node.topics:
                topic_counts[topic] += 1
        
        # Identify underrepresented domains
        if topic_counts:
            avg_count = np.mean(list(topic_counts.values()))
            underrepresented = [topic for topic, count in topic_counts.items() 
                              if count < avg_count * 0.5]
            
            for topic in underrepresented:
                gap = Gap(
                    gap_id=f"domain_{topic}",
                    gap_type=GapType.DOMAIN_GAP,
                    title=f"Underrepresented Domain: {topic}",
                    description=f"Domain '{topic}' has only {topic_counts[topic]} "
                               f"formulas compared to average of {avg_count:.1f}.",
                    confidence_score=0.4,
                    priority=0.3,
                    evidence={
                        "formula_count": topic_counts[topic],
                        "average_count": avg_count,
                        "all_topics": dict(topic_counts)
                    }
                )
                gaps.append(gap)
        
        return gaps
    
    def _suggest_formula_variations(self, formulas: List[str]) -> List[str]:
        """Suggest potential variations of existing formulas."""
        variations = []
        
        for formula in formulas[:3]:  # Limit to avoid explosion
            try:
                expr = sympify(formula)
                
                # Try common algebraic variations
                variations.extend(self._generate_algebraic_variations(expr))
                
                # Try calculus variations if applicable
                if expr.free_symbols:
                    variations.extend(self._generate_calculus_variations(expr))
                
            except Exception:
                continue
        
        # Remove duplicates and original formulas
        variations = list(set(variations) - set(formulas))
        return variations[:5]  # Limit suggestions
    
    def _generate_algebraic_variations(self, expr: Expr) -> List[str]:
        """Generate algebraic variations of an expression."""
        variations = []
        
        try:
            # Expand and factor variations
            expanded = expand(expr)
            if expanded != expr:
                variations.append(str(expanded))
            
            factored = factor(expr)
            if factored != expr:
                variations.append(str(factored))
            
            # Simplify
            simplified = simplify(expr)
            if simplified != expr:
                variations.append(str(simplified))
                
        except Exception:
            pass
        
        return variations
    
    def _generate_calculus_variations(self, expr: Expr) -> List[str]:
        """Generate calculus-based variations of an expression."""
        variations = []
        
        try:
            symbols_list = list(expr.free_symbols)
            if not symbols_list:
                return variations
            
            main_symbol = symbols_list[0]  # Use first symbol
            
            # Derivative
            try:
                derivative = diff(expr, main_symbol)
                variations.append(str(derivative))
            except:
                pass
            
            # Integral (simple cases only)
            try:
                # Only for simple expressions to avoid complexity
                if len(str(expr)) < 50:
                    integral = integrate(expr, main_symbol)
                    variations.append(str(integral))
            except:
                pass
                
        except Exception:
            pass
        
        return variations
    
    def _find_transformation_gaps(self, formulas: List[str]) -> List[Tuple[str, str, List[str]]]:
        """Find potential transformation gaps between formulas."""
        gaps = []
        
        # This is a simplified version - in practice, this would be more sophisticated
        for i, formula1 in enumerate(formulas[:10]):  # Limit for performance
            for formula2 in formulas[i+1:i+5]:  # Check nearby formulas
                try:
                    expr1 = sympify(formula1)
                    expr2 = sympify(formula2)
                    
                    # Check if they have same symbols but different structure
                    if (expr1.free_symbols == expr2.free_symbols and 
                        expr1 != expr2 and 
                        len(expr1.free_symbols) > 0):
                        
                        # Suggest a potential intermediate step
                        intermediate = self._suggest_intermediate_step(expr1, expr2)
                        if intermediate:
                            gaps.append((formula1, formula2, [intermediate]))
                
                except Exception:
                    continue
        
        return gaps[:5]  # Limit results
    
    def _suggest_intermediate_step(self, expr1: Expr, expr2: Expr) -> Optional[str]:
        """Suggest an intermediate transformation step between two expressions."""
        try:
            # Try simplifying expr1 to see if it gets closer to expr2
            simplified = simplify(expr1)
            if simplified != expr1 and simplified != expr2:
                return str(simplified)
            
            # Try expanding expr1
            expanded = expand(expr1)
            if expanded != expr1 and expanded != expr2:
                return str(expanded)
            
        except Exception:
            pass
        
        return None
    
    def _topics_are_related(self, topics1: Set[str], topics2: Set[str]) -> bool:
        """Check if two sets of topics are conceptually related."""
        
        # Define related topic groups
        related_groups = [
            {"algebra", "polynomial", "quadratic"},
            {"calculus", "derivative", "integral", "limits"},
            {"trigonometry", "periodic", "sine", "cosine"},
            {"geometry", "circle", "triangle", "area"},
            {"complex_analysis", "complex", "euler"},
            {"statistics", "probability", "distribution"}
        ]
        
        for group in related_groups:
            if (topics1 & group) and (topics2 & group):
                return True
        
        return False
    
    def _calculate_sparse_priority(self, node: ConceptNode) -> float:
        """Calculate priority score for sparse connection gaps."""
        priority = 0.5  # Base priority
        
        # Higher priority for nodes with validation scores
        if node.validation_scores:
            avg_score = np.mean(node.validation_scores)
            priority += 0.3 * avg_score
        
        # Higher priority for nodes with multiple formulas
        if len(node.formulas) > 1:
            priority += 0.2
        
        # Higher priority for nodes with topic information
        if node.topics:
            priority += 0.1
        
        return min(priority, 1.0)
    
    def _rank_gaps(self, gaps: List[Gap]) -> List[Gap]:
        """Rank gaps by priority and confidence."""
        
        # Calculate combined score
        for gap in gaps:
            gap.priority = gap.confidence_score * 0.6 + gap.priority * 0.4
        
        # Sort by priority (descending)
        return sorted(gaps, key=lambda g: g.priority, reverse=True)
    
    def save_gaps(self, gaps: List[Gap], output_path: Union[str, Path]) -> None:
        """
        Save detected gaps to a JSON file.
        
        Args:
            gaps: List of detected gaps
            output_path: Path to output file
        """
        output_path = Path(output_path)
        
        gaps_data = {
            "detection_metadata": {
                "total_gaps": len(gaps),
                "gap_types": dict(Counter(gap.gap_type.value for gap in gaps)),
                "detection_time": time.time(),
                "thresholds": {
                    "min_connection_threshold": self.min_connection_threshold,
                    "isolation_threshold": self.isolation_threshold
                }
            },
            "gaps": [
                {
                    "gap_id": gap.gap_id,
                    "gap_type": gap.gap_type.value,
                    "title": gap.title,
                    "description": gap.description,
                    "confidence_score": gap.confidence_score,
                    "priority": gap.priority,
                    "suggested_formulas": gap.suggested_formulas,
                    "related_concepts": gap.related_concepts,
                    "evidence": gap.evidence,
                    "metadata": gap.metadata
                }
                for gap in gaps
            ]
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(gaps_data, f, indent=2, default=str)
        
        self.logger.info(f"Saved {len(gaps)} detected gaps to {output_path}") 