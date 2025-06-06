"""
High-performance search indexing system for MathBot theorems.

This module provides comprehensive search capabilities including full-text search,
symbol search, metadata filtering, and fuzzy matching with caching optimization.
"""

import hashlib
import re
import time
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict

from pydantic import BaseModel, Field
import Levenshtein

from ui.data.models import Theorem, ValidationEvidence, SourceLineage
from ui.config import UIConfig
from ui.utils.ui_logging import get_ui_logger, log_ui_interaction


class SearchType(str, Enum):
    """Types of search supported by the search engine."""
    TEXT = "text"
    SYMBOL = "symbol"
    TYPE = "type"
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"
    FUZZY = "fuzzy"


class SearchFilters(BaseModel):
    """Search filter configuration."""
    theorem_types: Optional[List[str]] = None
    validation_status: Optional[List[str]] = None
    confidence_range: Optional[Tuple[float, float]] = None
    symbols: Optional[List[str]] = None
    transformation_methods: Optional[List[str]] = None
    min_confidence: Optional[float] = None
    max_confidence: Optional[float] = None


class SearchResult(BaseModel):
    """Search result with relevance scoring and highlighting."""
    theorem: Theorem
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    match_highlights: Dict[str, List[str]] = Field(default_factory=dict)
    match_reasons: List[str] = Field(default_factory=list)
    search_type: SearchType


@dataclass
class IndexEntry:
    """Individual index entry for search optimization."""
    theorem_id: str
    content: str
    field_type: str
    tokens: Set[str]
    weight: float = 1.0


class SearchIndex:
    """
    High-performance search indexing system with caching and optimization.
    
    Provides multiple search modalities with TF-IDF scoring, fuzzy matching,
    and comprehensive filtering capabilities.
    """
    
    def __init__(self, config: UIConfig):
        """Initialize search index with configuration."""
        self.config = config
        self.logger = get_ui_logger("search_index")
        
        # Index storage
        self.text_index: Dict[str, List[IndexEntry]] = defaultdict(list)
        self.symbol_index: Dict[str, List[IndexEntry]] = defaultdict(list)
        self.metadata_index: Dict[str, Dict[str, List[str]]] = defaultdict(dict)
        self.theorem_map: Dict[str, Theorem] = {}
        
        # Search statistics
        self.index_built_time: Optional[float] = None
        self.search_stats = {"queries": 0, "cache_hits": 0, "avg_response_time": 0.0}
        
        # Caching
        self._search_cache: Dict[str, List[SearchResult]] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self.cache_ttl = getattr(config, 'search_cache_ttl', 300)  # 5 minutes
        self.max_cache_size = getattr(config, 'search_cache_size', 1000)
    
    def build_index(self, theorems: List[Theorem]) -> None:
        """
        Build comprehensive search index with multiple field types.
        
        Args:
            theorems: List of theorems to index
        """
        start_time = time.time()
        self.logger.info(f"Building search index for {len(theorems)} theorems")
        
        # Clear existing indices
        self.text_index.clear()
        self.symbol_index.clear()
        self.metadata_index.clear()
        self.theorem_map.clear()
        
        for theorem in theorems:
            self._index_theorem(theorem)
        
        self.index_built_time = time.time() - start_time
        self.logger.info(f"Search index built in {self.index_built_time:.3f}s")
        log_ui_interaction("search_index", "index_built", {
            "theorem_count": len(theorems),
            "build_time": self.index_built_time
        })
    
    def _index_theorem(self, theorem: Theorem) -> None:
        """Index a single theorem across all index types."""
        self.theorem_map[theorem.id] = theorem
        
        # Text indexing
        self._index_text_content(theorem)
        
        # Symbol indexing
        self._index_symbols(theorem)
        
        # Metadata indexing
        self._index_metadata(theorem)
    
    def _index_text_content(self, theorem: Theorem) -> None:
        """Index text content with tokenization and normalization."""
        # Index different text fields with appropriate weights
        text_fields = [
            (theorem.statement, "statement", 3.0),
            (theorem.natural_language, "description", 2.0),
            (" ".join(theorem.assumptions), "assumptions", 1.5),
            (theorem.theorem_type, "type", 1.0)
        ]
        
        for content, field_type, weight in text_fields:
            if content:
                tokens = self._tokenize_text(content)
                entry = IndexEntry(
                    theorem_id=theorem.id,
                    content=content,
                    field_type=field_type,
                    tokens=tokens,
                    weight=weight
                )
                
                for token in tokens:
                    self.text_index[token.lower()].append(entry)
    
    def _index_symbols(self, theorem: Theorem) -> None:
        """Index mathematical symbols with LaTeX normalization."""
        symbols = set(theorem.symbols)
        
        # Extract symbols from statement and expressions
        latex_symbols = self._extract_latex_symbols(theorem.statement)
        symbols.update(latex_symbols)
        
        for symbol in symbols:
            normalized_symbol = self._normalize_symbol(symbol)
            entry = IndexEntry(
                theorem_id=theorem.id,
                content=symbol,
                field_type="symbol",
                tokens={normalized_symbol},
                weight=2.0
            )
            self.symbol_index[normalized_symbol].append(entry)
    
    def _index_metadata(self, theorem: Theorem) -> None:
        """Index metadata for filtering."""
        metadata = {
            "type": [theorem.theorem_type],
            "validation_status": [theorem.validation_evidence.validation_status],
            "confidence": [str(int(theorem.source_lineage.confidence * 100))],
            "generation_method": [theorem.source_lineage.generation_method],
            "transformation_methods": theorem.source_lineage.transformation_chain
        }
        
        for field, values in metadata.items():
            if field not in self.metadata_index:
                self.metadata_index[field] = defaultdict(list)
            
            for value in values:
                if value:  # Skip empty values
                    self.metadata_index[field][value.lower()].append(theorem.id)
    
    def _tokenize_text(self, text: str) -> Set[str]:
        """Tokenize text into searchable terms."""
        # Remove LaTeX commands and normalize
        text = re.sub(r'\\[a-zA-Z]+', ' ', text)
        text = re.sub(r'[{}]', ' ', text)
        
        # Split on non-alphanumeric characters
        tokens = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
        
        # Remove very short tokens and common stop words
        stop_words = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 
                     'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 
                     'that', 'the', 'to', 'was', 'will', 'with'}
        
        return {token for token in tokens if len(token) > 2 and token not in stop_words}
    
    def _extract_latex_symbols(self, text: str) -> Set[str]:
        """Extract mathematical symbols from LaTeX text."""
        symbols = set()
        
        # Common mathematical symbols and their LaTeX representations
        latex_patterns = [
            r'\\([a-zA-Z]+)',  # LaTeX commands
            r'\\([a-zA-Z]+)\{([^}]+)\}',  # LaTeX with arguments
            r'([a-zA-Z])_\{([^}]+)\}',  # Subscripts
            r'([a-zA-Z])\^\{([^}]+)\}',  # Superscripts
        ]
        
        for pattern in latex_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    symbols.update(match)
                else:
                    symbols.add(match)
        
        return {s for s in symbols if s and len(s) > 0}
    
    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize mathematical symbols for consistent matching."""
        # Remove common LaTeX formatting
        symbol = symbol.replace('\\', '').replace('{', '').replace('}', '')
        
        # Normalize common mathematical symbols
        normalizations = {
            'forall': '∀', 'exists': '∃', 'in': '∈',
            'mathbb{R}': 'ℝ', 'mathbb{N}': 'ℕ', 'mathbb{Z}': 'ℤ',
            'mathbb{Q}': 'ℚ', 'mathbb{C}': 'ℂ'
        }
        
        return normalizations.get(symbol.lower(), symbol.lower())
    
    def search(self, query: str, search_types: List[SearchType], 
               filters: Optional[SearchFilters] = None,
               fuzzy_threshold: float = 0.7,
               max_results: int = 50) -> List[SearchResult]:
        """
        Execute optimized search with ranking and relevance scoring.
        
        Args:
            query: Search query string
            search_types: Types of search to perform
            filters: Optional filters to apply
            fuzzy_threshold: Threshold for fuzzy matching (0.0-1.0)
            max_results: Maximum number of results to return
            
        Returns:
            List of search results sorted by relevance
        """
        start_time = time.time()
        
        # Check cache first
        cache_key = self._get_cache_key(query, search_types, filters, fuzzy_threshold)
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            self.search_stats["cache_hits"] += 1
            return cached_result[:max_results]
        
        # Perform search
        results = []
        
        for search_type in search_types:
            type_results = self._search_by_type(query, search_type, fuzzy_threshold)
            results.extend(type_results)
        
        # Apply filters
        if filters:
            results = self._apply_filters(results, filters)
        
        # Merge and rank results
        merged_results = self._merge_and_rank_results(results)
        
        # Cache results
        self._cache_result(cache_key, merged_results)
        
        # Update statistics
        response_time = time.time() - start_time
        self.search_stats["queries"] += 1
        self.search_stats["avg_response_time"] = (
            (self.search_stats["avg_response_time"] * (self.search_stats["queries"] - 1) + response_time) 
            / self.search_stats["queries"]
        )
        
        self.logger.debug(f"Search completed in {response_time:.3f}s, {len(merged_results)} results")
        
        return merged_results[:max_results]
    
    def _search_by_type(self, query: str, search_type: SearchType, 
                       fuzzy_threshold: float) -> List[SearchResult]:
        """Perform search by specific type."""
        if search_type == SearchType.TEXT:
            return self._text_search(query)
        elif search_type == SearchType.SYMBOL:
            return self._symbol_search(query)
        elif search_type == SearchType.TYPE:
            return self._type_search(query)
        elif search_type == SearchType.VALIDATION:
            return self._validation_search(query)
        elif search_type == SearchType.TRANSFORMATION:
            return self._transformation_search(query)
        elif search_type == SearchType.FUZZY:
            return self._fuzzy_search(query, fuzzy_threshold)
        else:
            return []
    
    def _text_search(self, query: str) -> List[SearchResult]:
        """Perform full-text search with TF-IDF scoring."""
        query_tokens = self._tokenize_text(query)
        theorem_scores = defaultdict(float)
        theorem_highlights = defaultdict(list)
        theorem_reasons = defaultdict(list)
        
        for token in query_tokens:
            if token in self.text_index:
                entries = self.text_index[token]
                # Simple TF-IDF approximation
                idf = len(self.theorem_map) / len(entries) if entries else 1
                
                for entry in entries:
                    tf = 1  # Could be improved with actual term frequency
                    score = tf * idf * entry.weight
                    theorem_scores[entry.theorem_id] += score
                    
                    if entry.field_type not in theorem_highlights[entry.theorem_id]:
                        theorem_highlights[entry.theorem_id].append(entry.field_type)
                    
                    theorem_reasons[entry.theorem_id].append(f"Matched '{token}' in {entry.field_type}")
        
        results = []
        for theorem_id, score in theorem_scores.items():
            if theorem_id in self.theorem_map:
                normalized_score = min(score / len(query_tokens), 1.0) if query_tokens else 0.0
                results.append(SearchResult(
                    theorem=self.theorem_map[theorem_id],
                    relevance_score=normalized_score,
                    match_highlights={"matched_fields": theorem_highlights[theorem_id]},
                    match_reasons=theorem_reasons[theorem_id][:3],  # Top 3 reasons
                    search_type=SearchType.TEXT
                ))
        
        return results
    
    def _symbol_search(self, query: str) -> List[SearchResult]:
        """Search for mathematical symbols."""
        normalized_query = self._normalize_symbol(query)
        results = []
        
        if normalized_query in self.symbol_index:
            entries = self.symbol_index[normalized_query]
            for entry in entries:
                if entry.theorem_id in self.theorem_map:
                    results.append(SearchResult(
                        theorem=self.theorem_map[entry.theorem_id],
                        relevance_score=0.9,  # High relevance for exact symbol match
                        match_highlights={"symbols": [entry.content]},
                        match_reasons=[f"Contains symbol '{entry.content}'"],
                        search_type=SearchType.SYMBOL
                    ))
        
        return results
    
    def _type_search(self, query: str) -> List[SearchResult]:
        """Search by theorem type."""
        query_lower = query.lower().replace(" ", "_")
        results = []
        
        if "type" in self.metadata_index and query_lower in self.metadata_index["type"]:
            theorem_ids = self.metadata_index["type"][query_lower]
            for theorem_id in theorem_ids:
                if theorem_id in self.theorem_map:
                    results.append(SearchResult(
                        theorem=self.theorem_map[theorem_id],
                        relevance_score=0.95,  # Very high for exact type match
                        match_highlights={"type": [query]},
                        match_reasons=[f"Theorem type is '{query}'"],
                        search_type=SearchType.TYPE
                    ))
        
        return results
    
    def _validation_search(self, query: str) -> List[SearchResult]:
        """Search by validation status."""
        query_upper = query.upper()
        results = []
        
        if "validation_status" in self.metadata_index and query_upper in self.metadata_index["validation_status"]:
            theorem_ids = self.metadata_index["validation_status"][query_upper]
            for theorem_id in theorem_ids:
                if theorem_id in self.theorem_map:
                    results.append(SearchResult(
                        theorem=self.theorem_map[theorem_id],
                        relevance_score=0.8,
                        match_highlights={"validation": [query_upper]},
                        match_reasons=[f"Validation status is '{query_upper}'"],
                        search_type=SearchType.VALIDATION
                    ))
        
        return results
    
    def _transformation_search(self, query: str) -> List[SearchResult]:
        """Search by transformation methods."""
        query_lower = query.lower()
        results = []
        
        if "transformation_methods" in self.metadata_index:
            for method, theorem_ids in self.metadata_index["transformation_methods"].items():
                if query_lower in method.lower():
                    for theorem_id in theorem_ids:
                        if theorem_id in self.theorem_map:
                            results.append(SearchResult(
                                theorem=self.theorem_map[theorem_id],
                                relevance_score=0.7,
                                match_highlights={"transformation": [method]},
                                match_reasons=[f"Uses transformation method '{method}'"],
                                search_type=SearchType.TRANSFORMATION
                            ))
        
        return results
    
    def _fuzzy_search(self, query: str, threshold: float) -> List[SearchResult]:
        """Perform fuzzy search using simple string matching."""
        results = []
        query_lower = query.lower()
        
        # Simple fuzzy search across theorem statements and descriptions
        for theorem in self.theorem_map.values():
            # Check statement
            statement_similarity = self._simple_similarity(query_lower, theorem.statement.lower())
            description_similarity = self._simple_similarity(query_lower, theorem.natural_language.lower())
            
            max_similarity = max(statement_similarity, description_similarity)
            
            if max_similarity >= threshold:
                match_field = "statement" if statement_similarity >= description_similarity else "description"
                results.append(SearchResult(
                    theorem=theorem,
                    relevance_score=max_similarity,
                    match_highlights={match_field: [query]},
                    match_reasons=[f"Fuzzy match in {match_field} (similarity: {max_similarity:.2f})"],
                    search_type=SearchType.FUZZY
                ))
        
        return results
    
    def _simple_similarity(self, s1: str, s2: str) -> float:
        """Calculate simple similarity based on common words."""
        if not s1 or not s2:
            return 0.0
        
        words1 = set(s1.split())
        words2 = set(s2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _apply_filters(self, results: List[SearchResult], filters: SearchFilters) -> List[SearchResult]:
        """Apply search filters to results."""
        filtered_results = []
        
        for result in results:
            theorem = result.theorem
            
            # Filter by theorem type
            if filters.theorem_types and theorem.theorem_type not in filters.theorem_types:
                continue
            
            # Filter by validation status
            if (filters.validation_status and 
                theorem.validation_evidence.validation_status not in filters.validation_status):
                continue
            
            # Filter by confidence range
            if filters.confidence_range:
                min_conf, max_conf = filters.confidence_range
                theorem_conf = theorem.source_lineage.confidence
                if not (min_conf <= theorem_conf <= max_conf):
                    continue
            
            # Filter by minimum confidence
            if filters.min_confidence and theorem.source_lineage.confidence < filters.min_confidence:
                continue
            
            # Filter by maximum confidence
            if filters.max_confidence and theorem.source_lineage.confidence > filters.max_confidence:
                continue
            
            # Filter by symbols
            if filters.symbols:
                theorem_symbols = set(theorem.symbols)
                if not any(symbol in theorem_symbols for symbol in filters.symbols):
                    continue
            
            # Filter by transformation methods
            if filters.transformation_methods:
                theorem_methods = set(theorem.source_lineage.transformation_chain)
                if not any(method in theorem_methods for method in filters.transformation_methods):
                    continue
            
            filtered_results.append(result)
        
        return filtered_results
    
    def _merge_and_rank_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Merge duplicate results and rank by relevance."""
        # Group results by theorem ID
        theorem_results = defaultdict(list)
        for result in results:
            theorem_results[result.theorem.id].append(result)
        
        # Merge results for each theorem
        merged_results = []
        for theorem_id, theorem_result_list in theorem_results.items():
            if len(theorem_result_list) == 1:
                merged_results.append(theorem_result_list[0])
            else:
                # Combine multiple search results for the same theorem
                combined_score = max(r.relevance_score for r in theorem_result_list)
                combined_highlights = {}
                combined_reasons = []
                
                for result in theorem_result_list:
                    combined_highlights.update(result.match_highlights)
                    combined_reasons.extend(result.match_reasons)
                
                merged_results.append(SearchResult(
                    theorem=theorem_result_list[0].theorem,
                    relevance_score=combined_score,
                    match_highlights=combined_highlights,
                    match_reasons=list(set(combined_reasons))[:5],  # Top 5 unique reasons
                    search_type=SearchType.TEXT  # Default to text for combined results
                ))
        
        # Sort by relevance score
        return sorted(merged_results, key=lambda r: r.relevance_score, reverse=True)
    
    def _get_cache_key(self, query: str, search_types: List[SearchType], 
                      filters: Optional[SearchFilters], fuzzy_threshold: float) -> str:
        """Generate cache key for search parameters."""
        filter_str = ""
        if filters:
            filter_str = str(sorted(filters.model_dump().items()))
        
        key_components = [
            query.lower(),
            str(sorted([st.value for st in search_types])),
            filter_str,
            str(fuzzy_threshold)
        ]
        
        return hashlib.md5("|".join(key_components).encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[List[SearchResult]]:
        """Get cached search result if still valid."""
        if cache_key not in self._search_cache:
            return None
        
        # Check if cache entry is still valid
        cache_time = self._cache_timestamps.get(cache_key, 0)
        if time.time() - cache_time > self.cache_ttl:
            # Remove expired entry
            del self._search_cache[cache_key]
            del self._cache_timestamps[cache_key]
            return None
        
        return self._search_cache[cache_key]
    
    def _cache_result(self, cache_key: str, results: List[SearchResult]) -> None:
        """Cache search results with TTL."""
        # Implement LRU eviction if cache is full
        if len(self._search_cache) >= self.max_cache_size:
            # Remove oldest entry
            oldest_key = min(self._cache_timestamps.keys(), 
                           key=lambda k: self._cache_timestamps[k])
            del self._search_cache[oldest_key]
            del self._cache_timestamps[oldest_key]
        
        self._search_cache[cache_key] = results
        self._cache_timestamps[cache_key] = time.time()
    
    def update_index(self, theorem: Theorem) -> None:
        """Incrementally update index for real-time data changes."""
        self.logger.debug(f"Updating index for theorem {theorem.id}")
        
        # Remove old entries if theorem exists
        if theorem.id in self.theorem_map:
            self._remove_theorem_from_index(theorem.id)
        
        # Add updated theorem
        self._index_theorem(theorem)
        
        # Clear cache to ensure fresh results
        self._search_cache.clear()
        self._cache_timestamps.clear()
        
        log_ui_interaction("search_index", "theorem_updated", {"theorem_id": theorem.id})
    
    def _remove_theorem_from_index(self, theorem_id: str) -> None:
        """Remove theorem from all indices."""
        # Remove from text index
        for token_entries in self.text_index.values():
            token_entries[:] = [e for e in token_entries if e.theorem_id != theorem_id]
        
        # Remove from symbol index
        for symbol_entries in self.symbol_index.values():
            symbol_entries[:] = [e for e in symbol_entries if e.theorem_id != theorem_id]
        
        # Remove from metadata index
        for field_dict in self.metadata_index.values():
            for value_list in field_dict.values():
                value_list[:] = [tid for tid in value_list if tid != theorem_id]
        
        # Remove from theorem map
        if theorem_id in self.theorem_map:
            del self.theorem_map[theorem_id]
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get search performance statistics."""
        cache_hit_rate = 0.0
        if self.search_stats["queries"] > 0:
            cache_hit_rate = self.search_stats["cache_hits"] / self.search_stats["queries"]
        
        return {
            "index_built_time": self.index_built_time,
            "theorem_count": len(self.theorem_map),
            "queries_executed": self.search_stats["queries"],
            "cache_hit_rate": cache_hit_rate,
            "avg_response_time": self.search_stats["avg_response_time"],
            "cache_size": len(self._search_cache),
            "index_size": {
                "text_tokens": len(self.text_index),
                "symbols": len(self.symbol_index),
                "metadata_fields": len(self.metadata_index)
            }
        } 