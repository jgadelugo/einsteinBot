"""
Advanced search interface for MathBot theorems.

This module provides a comprehensive search interface with multiple search modalities,
real-time filtering, and performance optimization.
"""

import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

import streamlit as st
import pandas as pd
from pydantic import BaseModel, Field

from ui.data.models import Theorem
from ui.data.search_index import SearchIndex, SearchType, SearchFilters, SearchResult
from ui.config import UIConfig
from ui.utils.ui_logging import get_ui_logger, log_ui_interaction


class SearchConfig(BaseModel):
    """Complete search configuration."""
    query: str = ""
    search_types: List[SearchType] = Field(default_factory=lambda: [SearchType.TEXT])
    filters: SearchFilters = Field(default_factory=SearchFilters)
    fuzzy_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_results: int = Field(default=50, ge=1, le=1000)
    sort_by: str = "relevance"
    sort_order: str = "desc"


@dataclass
class SearchSession:
    """Search session state management."""
    last_query: str = ""
    last_results: List[SearchResult] = None
    search_history: List[str] = None
    performance_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.last_results is None:
            self.last_results = []
        if self.search_history is None:
            self.search_history = []
        if self.performance_metrics is None:
            self.performance_metrics = {}


class SearchInterface:
    """
    Advanced search interface with multiple modalities and real-time filtering.
    
    Provides comprehensive search capabilities including text search, symbol search,
    type filtering, validation filtering, and fuzzy matching with performance optimization.
    """
    
    def __init__(self, config: UIConfig, search_index: SearchIndex):
        """Initialize search interface with configuration and search index."""
        self.config = config
        self.search_index = search_index
        self.logger = get_ui_logger("search_interface")
        
        # Initialize session state for search
        self._initialize_search_session()
        
        # Search configuration
        self.debounce_delay = getattr(config, 'search_debounce_ms', 300) / 1000.0
        self.max_search_history = getattr(config, 'max_search_history', 50)
        self.enable_suggestions = getattr(config, 'enable_search_suggestions', True)
    
    def _initialize_search_session(self) -> None:
        """Initialize search session state."""
        if 'search_session' not in st.session_state:
            st.session_state.search_session = SearchSession()
        
        if 'search_config' not in st.session_state:
            st.session_state.search_config = SearchConfig()
        
        if 'search_results' not in st.session_state:
            st.session_state.search_results = []
        
        if 'last_search_time' not in st.session_state:
            st.session_state.last_search_time = 0
    
    def render_search_controls(self) -> SearchConfig:
        """
        Render comprehensive search controls with real-time preview.
        
        Returns:
            SearchConfig: Current search configuration
        """
        st.markdown("### 🔍 Search Mathematical Theorems")
        
        # Main search input
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_input(
                "Search query",
                value=st.session_state.search_config.query,
                placeholder="Enter theorem description, symbols, or keywords...",
                help="Search across theorem statements, descriptions, and metadata",
                key="search_query_input"
            )
        
        with col2:
            search_button = st.button("🔍 Search", type="primary", use_container_width=True)
        
        # Search type selection
        st.markdown("#### Search Types")
        search_type_cols = st.columns(3)
        
        with search_type_cols[0]:
            text_search = st.checkbox("Text Search", value=True, help="Search in theorem statements and descriptions")
            symbol_search = st.checkbox("Symbol Search", value=False, help="Search for mathematical symbols")
        
        with search_type_cols[1]:
            type_search = st.checkbox("Type Search", value=False, help="Search by theorem type")
            validation_search = st.checkbox("Validation Search", value=False, help="Search by validation status")
        
        with search_type_cols[2]:
            transformation_search = st.checkbox("Transformation Search", value=False, help="Search by transformation methods")
            fuzzy_search = st.checkbox("Fuzzy Search", value=False, help="Approximate matching")
        
        # Build search types list
        search_types = []
        if text_search:
            search_types.append(SearchType.TEXT)
        if symbol_search:
            search_types.append(SearchType.SYMBOL)
        if type_search:
            search_types.append(SearchType.TYPE)
        if validation_search:
            search_types.append(SearchType.VALIDATION)
        if transformation_search:
            search_types.append(SearchType.TRANSFORMATION)
        if fuzzy_search:
            search_types.append(SearchType.FUZZY)
        
        # Default to text search if none selected
        if not search_types:
            search_types = [SearchType.TEXT]
        
        # Advanced filters in expander
        with st.expander("🎛️ Advanced Filters", expanded=False):
            self._render_advanced_filters()
        
        # Search options
        with st.expander("⚙️ Search Options", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                fuzzy_threshold = st.slider(
                    "Fuzzy Search Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    step=0.1,
                    help="Minimum similarity for fuzzy matching"
                )
                
                max_results = st.number_input(
                    "Maximum Results",
                    min_value=1,
                    max_value=1000,
                    value=50,
                    step=10,
                    help="Maximum number of search results to return"
                )
            
            with col2:
                sort_by = st.selectbox(
                    "Sort By",
                    ["relevance", "confidence", "type", "validation"],
                    index=0,
                    help="Sort search results by"
                )
                
                sort_order = st.selectbox(
                    "Sort Order",
                    ["desc", "asc"],
                    index=0,
                    help="Sort order"
                )
        
        # Build search configuration
        search_config = SearchConfig(
            query=query,
            search_types=search_types,
            filters=st.session_state.get('search_filters', SearchFilters()),
            fuzzy_threshold=fuzzy_threshold,
            max_results=max_results,
            sort_by=sort_by,
            sort_order=sort_order
        )
        
        # Update session state
        st.session_state.search_config = search_config
        
        # Auto-search on query change or manual search
        should_search = (
            search_button or 
            (query != st.session_state.search_session.last_query and query.strip())
        )
        
        if should_search:
            self._execute_search_with_debounce(search_config)
        
        # Render search history
        self._render_search_history()
        
        return search_config
    
    def _render_advanced_filters(self) -> None:
        """Render advanced filtering controls."""
        # Get available filter options from search index
        available_types = self._get_available_theorem_types()
        available_statuses = self._get_available_validation_statuses()
        available_methods = self._get_available_transformation_methods()
        available_symbols = self._get_available_symbols()
        
        # Theorem type filter
        theorem_types = st.multiselect(
            "Theorem Types",
            options=available_types,
            default=[],
            help="Filter by theorem type"
        )
        
        # Validation status filter
        validation_status = st.multiselect(
            "Validation Status",
            options=available_statuses,
            default=[],
            help="Filter by validation status"
        )
        
        # Confidence range filter
        col1, col2 = st.columns(2)
        with col1:
            min_confidence = st.slider(
                "Minimum Confidence",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.1,
                help="Minimum confidence score"
            )
        
        with col2:
            max_confidence = st.slider(
                "Maximum Confidence",
                min_value=0.0,
                max_value=1.0,
                value=1.0,
                step=0.1,
                help="Maximum confidence score"
            )
        
        # Symbol filter
        symbols = st.multiselect(
            "Mathematical Symbols",
            options=available_symbols[:20],  # Limit to top 20 for UI
            default=[],
            help="Filter by mathematical symbols used"
        )
        
        # Transformation methods filter
        transformation_methods = st.multiselect(
            "Transformation Methods",
            options=available_methods,
            default=[],
            help="Filter by transformation methods used"
        )
        
        # Build filters object
        filters = SearchFilters(
            theorem_types=theorem_types if theorem_types else None,
            validation_status=validation_status if validation_status else None,
            confidence_range=(min_confidence, max_confidence) if min_confidence < max_confidence else None,
            symbols=symbols if symbols else None,
            transformation_methods=transformation_methods if transformation_methods else None,
            min_confidence=min_confidence if min_confidence > 0.0 else None,
            max_confidence=max_confidence if max_confidence < 1.0 else None
        )
        
        # Store in session state
        st.session_state.search_filters = filters
    
    def _get_available_theorem_types(self) -> List[str]:
        """Get available theorem types from search index."""
        if "type" in self.search_index.metadata_index:
            return list(self.search_index.metadata_index["type"].keys())
        return []
    
    def _get_available_validation_statuses(self) -> List[str]:
        """Get available validation statuses from search index."""
        if "validation_status" in self.search_index.metadata_index:
            return list(self.search_index.metadata_index["validation_status"].keys())
        return ["PASS", "FAIL", "PARTIAL", "ERROR"]
    
    def _get_available_transformation_methods(self) -> List[str]:
        """Get available transformation methods from search index."""
        if "transformation_methods" in self.search_index.metadata_index:
            return list(self.search_index.metadata_index["transformation_methods"].keys())
        return []
    
    def _get_available_symbols(self) -> List[str]:
        """Get available mathematical symbols from search index."""
        symbols = set()
        for theorem in self.search_index.theorem_map.values():
            symbols.update(theorem.symbols)
        return sorted(list(symbols))[:50]  # Return top 50 most common
    
    def _execute_search_with_debounce(self, search_config: SearchConfig) -> None:
        """Execute search with debouncing to prevent excessive API calls."""
        current_time = time.time()
        
        # Check if enough time has passed since last search
        if current_time - st.session_state.last_search_time < self.debounce_delay:
            return
        
        # Execute search
        try:
            start_time = time.time()
            
            results = self.search_index.search(
                query=search_config.query,
                search_types=search_config.search_types,
                filters=search_config.filters,
                fuzzy_threshold=search_config.fuzzy_threshold,
                max_results=search_config.max_results
            )
            
            # Sort results if needed
            if search_config.sort_by != "relevance":
                results = self._sort_results(results, search_config.sort_by, search_config.sort_order)
            
            search_time = time.time() - start_time
            
            # Update session state
            st.session_state.search_results = results
            st.session_state.last_search_time = current_time
            st.session_state.search_session.last_query = search_config.query
            st.session_state.search_session.last_results = results
            
            # Update search history
            if search_config.query.strip() and search_config.query not in st.session_state.search_session.search_history:
                st.session_state.search_session.search_history.insert(0, search_config.query)
                # Keep only recent searches
                st.session_state.search_session.search_history = st.session_state.search_session.search_history[:self.max_search_history]
            
            # Update performance metrics
            st.session_state.search_session.performance_metrics = {
                "search_time": search_time,
                "result_count": len(results),
                "query_length": len(search_config.query),
                "search_types_count": len(search_config.search_types)
            }
            
            # Log search interaction
            log_ui_interaction("search_interface", "search_executed", {
                "query": search_config.query,
                "search_types": [st.value for st in search_config.search_types],
                "result_count": len(results),
                "search_time": search_time
            })
            
            self.logger.info(f"Search executed: '{search_config.query}' -> {len(results)} results in {search_time:.3f}s")
            
        except Exception as e:
            self.logger.error(f"Search execution failed: {e}", exc_info=True)
            st.error(f"Search failed: {str(e)}")
    
    def _sort_results(self, results: List[SearchResult], sort_by: str, sort_order: str) -> List[SearchResult]:
        """Sort search results by specified criteria."""
        reverse = sort_order == "desc"
        
        if sort_by == "confidence":
            return sorted(results, key=lambda r: r.theorem.source_lineage.confidence, reverse=reverse)
        elif sort_by == "type":
            return sorted(results, key=lambda r: r.theorem.theorem_type, reverse=reverse)
        elif sort_by == "validation":
            return sorted(results, key=lambda r: r.theorem.validation_evidence.pass_rate, reverse=reverse)
        else:  # relevance (default)
            return sorted(results, key=lambda r: r.relevance_score, reverse=reverse)
    
    def _render_search_history(self) -> None:
        """Render search history for quick access."""
        if st.session_state.search_session.search_history:
            with st.expander("📚 Recent Searches", expanded=False):
                for i, historical_query in enumerate(st.session_state.search_session.search_history[:10]):
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        st.text(historical_query)
                    
                    with col2:
                        if st.button("🔄", key=f"rerun_search_{i}", help="Rerun this search"):
                            st.session_state.search_config.query = historical_query
                            st.rerun()
    
    def render_search_results_summary(self, results: List[SearchResult]) -> None:
        """
        Display search results with statistics and performance metrics.
        
        Args:
            results: List of search results to summarize
        """
        if not results:
            st.info("No results found. Try adjusting your search query or filters.")
            return
        
        # Results summary header
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Results Found", len(results))
        
        with col2:
            avg_relevance = sum(r.relevance_score for r in results) / len(results)
            st.metric("Avg Relevance", f"{avg_relevance:.2f}")
        
        with col3:
            search_time = st.session_state.search_session.performance_metrics.get("search_time", 0)
            st.metric("Search Time", f"{search_time:.3f}s")
        
        with col4:
            # Get cache hit rate from search index
            stats = self.search_index.get_search_statistics()
            cache_hit_rate = stats.get("cache_hit_rate", 0)
            st.metric("Cache Hit Rate", f"{cache_hit_rate:.1%}")
        
        # Results breakdown by type
        if len(results) > 1:
            st.markdown("#### Results Breakdown")
            
            # Group results by search type
            type_counts = {}
            for result in results:
                search_type = result.search_type.value
                type_counts[search_type] = type_counts.get(search_type, 0) + 1
            
            # Display type breakdown
            type_cols = st.columns(len(type_counts))
            for i, (search_type, count) in enumerate(type_counts.items()):
                with type_cols[i]:
                    st.metric(f"{search_type.title()} Matches", count)
        
        # Performance insights
        if search_time > 1.0:
            st.warning(f"Search took {search_time:.2f}s. Consider using more specific queries for better performance.")
        elif search_time < 0.1:
            st.success("⚡ Lightning fast search! Results likely served from cache.")
        
        # Search suggestions
        if len(results) < 5 and st.session_state.search_config.query:
            self._render_search_suggestions()
    
    def _render_search_suggestions(self) -> None:
        """Render search suggestions to help users find more results."""
        st.markdown("#### 💡 Search Suggestions")
        
        suggestions = []
        query = st.session_state.search_config.query.lower()
        
        # Suggest enabling more search types
        current_types = st.session_state.search_config.search_types
        if SearchType.FUZZY not in current_types:
            suggestions.append("Try enabling **Fuzzy Search** for approximate matching")
        
        if SearchType.SYMBOL not in current_types and any(c in query for c in "∀∃∈∑∏∫"):
            suggestions.append("Try enabling **Symbol Search** - your query contains mathematical symbols")
        
        # Suggest removing filters
        filters = st.session_state.search_config.filters
        if any([filters.theorem_types, filters.validation_status, filters.symbols]):
            suggestions.append("Try removing some **filters** to broaden your search")
        
        # Suggest alternative queries
        if len(query.split()) > 3:
            suggestions.append("Try using **fewer keywords** for broader results")
        elif len(query.split()) == 1:
            suggestions.append("Try adding **more descriptive terms** to your query")
        
        # Display suggestions
        for suggestion in suggestions[:3]:  # Show top 3 suggestions
            st.info(suggestion)
    
    def get_search_analytics(self) -> Dict[str, Any]:
        """Get search analytics and performance data."""
        session = st.session_state.search_session
        index_stats = self.search_index.get_search_statistics()
        
        return {
            "session_metrics": session.performance_metrics,
            "search_history_count": len(session.search_history),
            "last_query": session.last_query,
            "last_result_count": len(session.last_results),
            "index_statistics": index_stats
        }
    
    def export_search_results(self, results: List[SearchResult], format: str = "csv") -> Optional[str]:
        """
        Export search results to specified format.
        
        Args:
            results: Search results to export
            format: Export format ("csv" or "json")
            
        Returns:
            Exported data as string or None if export fails
        """
        try:
            if format.lower() == "csv":
                # Convert to DataFrame for CSV export
                data = []
                for result in results:
                    theorem = result.theorem
                    data.append({
                        "ID": theorem.id,
                        "Statement": theorem.statement,
                        "Type": theorem.theorem_type,
                        "Confidence": theorem.source_lineage.confidence,
                        "Validation Status": theorem.validation_evidence.validation_status,
                        "Pass Rate": theorem.validation_evidence.pass_rate,
                        "Relevance Score": result.relevance_score,
                        "Match Reasons": "; ".join(result.match_reasons)
                    })
                
                df = pd.DataFrame(data)
                return df.to_csv(index=False)
            
            elif format.lower() == "json":
                # Convert to JSON
                import json
                data = []
                for result in results:
                    data.append({
                        "theorem": result.theorem.dict(),
                        "relevance_score": result.relevance_score,
                        "match_highlights": result.match_highlights,
                        "match_reasons": result.match_reasons,
                        "search_type": result.search_type.value
                    })
                
                return json.dumps(data, indent=2)
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            self.logger.error(f"Export failed: {e}", exc_info=True)
            return None 