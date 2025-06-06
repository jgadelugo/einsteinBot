"""
Professional theorem browser with advanced table features.

This module provides a comprehensive theorem browsing interface with sortable tables,
advanced filtering, pagination, and export functionality.
"""

import time
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import streamlit as st
import pandas as pd
from pydantic import BaseModel, Field

from ui.data.models import Theorem
from ui.data.search_index import SearchResult
from ui.config import UIConfig
from ui.utils.ui_logging import get_ui_logger, log_ui_interaction


class SortDirection(str, Enum):
    """Sort direction options."""
    ASC = "asc"
    DESC = "desc"


class FilterConfig(BaseModel):
    """Table filtering configuration."""
    active_filters: Dict[str, Any] = Field(default_factory=dict)
    available_filters: Dict[str, List[str]] = Field(default_factory=dict)
    filter_mode: str = "and"  # "and" or "or"


class SortConfig(BaseModel):
    """Table sorting configuration."""
    primary_sort: str = "statement"
    secondary_sort: Optional[str] = None
    sort_direction: SortDirection = SortDirection.ASC


class TableConfig(BaseModel):
    """Table display configuration."""
    columns: List[str] = Field(default_factory=lambda: [
        "id", "statement", "type", "validation_status", "confidence", "pass_rate"
    ])
    page_size: int = Field(default=10, ge=5, le=100)
    current_page: int = Field(default=1, ge=1)
    show_expanded: bool = False
    enable_selection: bool = True


@dataclass
class BrowserSession:
    """Browser session state management."""
    selected_theorems: List[str] = None
    last_sort_config: SortConfig = None
    last_filter_config: FilterConfig = None
    performance_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.selected_theorems is None:
            self.selected_theorems = []
        if self.last_sort_config is None:
            self.last_sort_config = SortConfig()
        if self.last_filter_config is None:
            self.last_filter_config = FilterConfig()
        if self.performance_metrics is None:
            self.performance_metrics = {}


class TheoremBrowser:
    """
    Professional theorem browsing interface with advanced table features.
    
    Provides sortable, filterable table with pagination, row selection,
    and export functionality for comprehensive theorem exploration.
    """
    
    def __init__(self, config: UIConfig):
        """Initialize theorem browser with configuration."""
        self.config = config
        self.logger = get_ui_logger("theorem_browser")
        
        # Initialize session state
        self._initialize_browser_session()
        
        # Browser configuration
        self.default_page_size = getattr(config, 'default_page_size', 10)
        self.max_page_size = getattr(config, 'max_page_size', 100)
        self.enable_export = getattr(config, 'enable_export', True)
    
    def _initialize_browser_session(self) -> None:
        """Initialize browser session state."""
        if 'browser_session' not in st.session_state:
            st.session_state.browser_session = BrowserSession()
        
        if 'table_config' not in st.session_state:
            st.session_state.table_config = TableConfig()
        
        if 'selected_theorem_id' not in st.session_state:
            st.session_state.selected_theorem_id = None
    
    def render_theorem_table(self, theorems: List[Theorem], 
                           search_results: Optional[List[SearchResult]] = None) -> Optional[Theorem]:
        """
        Render sortable, filterable theorem table with pagination.
        
        Args:
            theorems: List of theorems to display
            search_results: Optional search results for relevance scoring
            
        Returns:
            Selected theorem if any, None otherwise
        """
        if not theorems:
            st.info("No theorems to display.")
            return None
        
        start_time = time.time()
        
        # Render table controls
        self._render_table_controls(theorems)
        
        # Apply filters and sorting
        filtered_theorems = self._apply_table_filters(theorems)
        sorted_theorems = self._apply_table_sorting(filtered_theorems, search_results)
        
        # Pagination
        paginated_theorems, pagination_info = self._apply_pagination(sorted_theorems)
        
        # Render pagination controls
        self._render_pagination_controls(pagination_info)
        
        # Render the actual table
        selected_theorem = self._render_data_table(paginated_theorems, search_results)
        
        # Render table footer with statistics
        self._render_table_footer(theorems, filtered_theorems, pagination_info)
        
        # Update performance metrics
        render_time = time.time() - start_time
        st.session_state.browser_session.performance_metrics["table_render_time"] = render_time
        
        if render_time > 1.0:
            st.warning(f"Table rendering took {render_time:.2f}s. Consider reducing page size for better performance.")
        
        return selected_theorem
    
    def _render_table_controls(self, theorems: List[Theorem]) -> None:
        """Render table control panel."""
        st.markdown("### 📊 Theorem Browser")
        
        # Control panel
        col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
        
        with col1:
            # Column selection
            available_columns = [
                "id", "statement", "type", "validation_status", 
                "confidence", "pass_rate", "symbols", "assumptions"
            ]
            
            selected_columns = st.multiselect(
                "Display Columns",
                options=available_columns,
                default=st.session_state.table_config.columns,
                help="Select columns to display in the table"
            )
            
            if selected_columns:
                st.session_state.table_config.columns = selected_columns
        
        with col2:
            # Page size selection
            page_size = st.selectbox(
                "Rows per page",
                options=[5, 10, 20, 50, 100],
                index=[5, 10, 20, 50, 100].index(st.session_state.table_config.page_size),
                help="Number of theorems to display per page"
            )
            st.session_state.table_config.page_size = page_size
        
        with col3:
            # Export button
            if self.enable_export and st.button("📥 Export", help="Export current view"):
                self._handle_export(theorems)
        
        with col4:
            # Refresh button
            if st.button("🔄 Refresh", help="Refresh table data"):
                st.rerun()
    
    def _render_filter_controls(self, theorems: List[Theorem]) -> FilterConfig:
        """
        Render advanced filtering controls with real-time preview.
        
        Args:
            theorems: List of theorems for filter options
            
        Returns:
            FilterConfig: Current filter configuration
        """
        with st.expander("🎛️ Table Filters", expanded=False):
            # Get available filter options
            available_types = list(set(t.theorem_type for t in theorems))
            available_statuses = list(set(t.validation_evidence.validation_status for t in theorems))
            available_symbols = list(set(symbol for t in theorems for symbol in t.symbols))
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Theorem type filter
                type_filter = st.multiselect(
                    "Theorem Types",
                    options=sorted(available_types),
                    default=st.session_state.browser_session.last_filter_config.active_filters.get("types", []),
                    help="Filter by theorem type"
                )
                
                # Validation status filter
                status_filter = st.multiselect(
                    "Validation Status",
                    options=sorted(available_statuses),
                    default=st.session_state.browser_session.last_filter_config.active_filters.get("statuses", []),
                    help="Filter by validation status"
                )
            
            with col2:
                # Confidence range filter
                confidence_range = st.slider(
                    "Confidence Range",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.browser_session.last_filter_config.active_filters.get("confidence_range", (0.0, 1.0)),
                    step=0.1,
                    help="Filter by confidence score range"
                )
                
                # Pass rate filter
                pass_rate_range = st.slider(
                    "Pass Rate Range",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.browser_session.last_filter_config.active_filters.get("pass_rate_range", (0.0, 1.0)),
                    step=0.1,
                    help="Filter by validation pass rate range"
                )
            
            # Symbol filter
            symbol_filter = st.multiselect(
                "Mathematical Symbols",
                options=sorted(available_symbols)[:20],  # Limit for UI performance
                default=st.session_state.browser_session.last_filter_config.active_filters.get("symbols", []),
                help="Filter by mathematical symbols used"
            )
            
            # Filter mode
            filter_mode = st.radio(
                "Filter Mode",
                options=["and", "or"],
                index=0 if st.session_state.browser_session.last_filter_config.filter_mode == "and" else 1,
                horizontal=True,
                help="How to combine multiple filters"
            )
            
            # Build filter configuration
            filter_config = FilterConfig(
                active_filters={
                    "types": type_filter,
                    "statuses": status_filter,
                    "confidence_range": confidence_range,
                    "pass_rate_range": pass_rate_range,
                    "symbols": symbol_filter
                },
                available_filters={
                    "types": available_types,
                    "statuses": available_statuses,
                    "symbols": available_symbols
                },
                filter_mode=filter_mode
            )
            
            # Update session state
            st.session_state.browser_session.last_filter_config = filter_config
            
            return filter_config
    
    def _render_sort_controls(self) -> SortConfig:
        """
        Render sorting controls with multiple sort keys.
        
        Returns:
            SortConfig: Current sort configuration
        """
        with st.expander("📈 Sort Options", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                primary_sort = st.selectbox(
                    "Primary Sort",
                    options=["statement", "type", "confidence", "pass_rate", "id"],
                    index=["statement", "type", "confidence", "pass_rate", "id"].index(
                        st.session_state.browser_session.last_sort_config.primary_sort
                    ),
                    help="Primary sort column"
                )
            
            with col2:
                secondary_sort = st.selectbox(
                    "Secondary Sort",
                    options=["None", "statement", "type", "confidence", "pass_rate", "id"],
                    index=0,
                    help="Secondary sort column"
                )
                secondary_sort = None if secondary_sort == "None" else secondary_sort
            
            with col3:
                sort_direction = st.selectbox(
                    "Sort Direction",
                    options=["asc", "desc"],
                    index=0 if st.session_state.browser_session.last_sort_config.sort_direction == SortDirection.ASC else 1,
                    help="Sort direction"
                )
            
            sort_config = SortConfig(
                primary_sort=primary_sort,
                secondary_sort=secondary_sort,
                sort_direction=SortDirection(sort_direction)
            )
            
            # Update session state
            st.session_state.browser_session.last_sort_config = sort_config
            
            return sort_config
    
    def _apply_table_filters(self, theorems: List[Theorem]) -> List[Theorem]:
        """Apply active filters to theorem list."""
        filter_config = self._render_filter_controls(theorems)
        
        if not any(filter_config.active_filters.values()):
            return theorems
        
        filtered_theorems = []
        
        for theorem in theorems:
            matches = []
            
            # Type filter
            if filter_config.active_filters["types"]:
                matches.append(theorem.theorem_type in filter_config.active_filters["types"])
            
            # Status filter
            if filter_config.active_filters["statuses"]:
                matches.append(theorem.validation_evidence.validation_status in filter_config.active_filters["statuses"])
            
            # Confidence range filter
            conf_range = filter_config.active_filters["confidence_range"]
            if conf_range != (0.0, 1.0):
                confidence = theorem.source_lineage.confidence
                matches.append(conf_range[0] <= confidence <= conf_range[1])
            
            # Pass rate range filter
            pass_range = filter_config.active_filters["pass_rate_range"]
            if pass_range != (0.0, 1.0):
                pass_rate = theorem.validation_evidence.pass_rate
                matches.append(pass_range[0] <= pass_rate <= pass_range[1])
            
            # Symbol filter
            if filter_config.active_filters["symbols"]:
                theorem_symbols = set(theorem.symbols)
                filter_symbols = set(filter_config.active_filters["symbols"])
                matches.append(bool(theorem_symbols.intersection(filter_symbols)))
            
            # Apply filter mode
            if matches:
                if filter_config.filter_mode == "and":
                    if all(matches):
                        filtered_theorems.append(theorem)
                else:  # "or"
                    if any(matches):
                        filtered_theorems.append(theorem)
            else:
                # No filters applied
                filtered_theorems.append(theorem)
        
        return filtered_theorems
    
    def _apply_table_sorting(self, theorems: List[Theorem], 
                           search_results: Optional[List[SearchResult]] = None) -> List[Theorem]:
        """Apply sorting to theorem list."""
        sort_config = self._render_sort_controls()
        
        # Create relevance map if search results available
        relevance_map = {}
        if search_results:
            relevance_map = {r.theorem.id: r.relevance_score for r in search_results}
        
        def get_sort_key(theorem: Theorem, field: str):
            """Get sort key for a theorem field."""
            if field == "statement":
                return theorem.statement.lower()
            elif field == "type":
                return theorem.theorem_type
            elif field == "confidence":
                return theorem.source_lineage.confidence
            elif field == "pass_rate":
                return theorem.validation_evidence.pass_rate
            elif field == "id":
                return theorem.id
            elif field == "relevance" and theorem.id in relevance_map:
                return relevance_map[theorem.id]
            else:
                return ""
        
        # Sort by primary key
        sorted_theorems = sorted(
            theorems,
            key=lambda t: get_sort_key(t, sort_config.primary_sort),
            reverse=(sort_config.sort_direction == SortDirection.DESC)
        )
        
        # Sort by secondary key if specified
        if sort_config.secondary_sort:
            sorted_theorems = sorted(
                sorted_theorems,
                key=lambda t: get_sort_key(t, sort_config.secondary_sort),
                reverse=(sort_config.sort_direction == SortDirection.DESC)
            )
        
        return sorted_theorems
    
    def _apply_pagination(self, theorems: List[Theorem]) -> Tuple[List[Theorem], Dict[str, Any]]:
        """Apply pagination to theorem list."""
        total_count = len(theorems)
        page_size = st.session_state.table_config.page_size
        current_page = st.session_state.table_config.current_page
        
        # Calculate pagination
        total_pages = max(1, (total_count + page_size - 1) // page_size)
        current_page = min(current_page, total_pages)
        
        start_idx = (current_page - 1) * page_size
        end_idx = min(start_idx + page_size, total_count)
        
        paginated_theorems = theorems[start_idx:end_idx]
        
        pagination_info = {
            "total_count": total_count,
            "page_size": page_size,
            "current_page": current_page,
            "total_pages": total_pages,
            "start_idx": start_idx,
            "end_idx": end_idx
        }
        
        return paginated_theorems, pagination_info
    
    def _render_pagination_controls(self, pagination_info: Dict[str, Any]) -> None:
        """Render pagination controls."""
        if pagination_info["total_pages"] <= 1:
            return
        
        col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
        
        with col1:
            if st.button("⏮️ First", disabled=pagination_info["current_page"] == 1):
                st.session_state.table_config.current_page = 1
                st.rerun()
        
        with col2:
            if st.button("◀️ Prev", disabled=pagination_info["current_page"] == 1):
                st.session_state.table_config.current_page -= 1
                st.rerun()
        
        with col3:
            st.markdown(
                f"<div style='text-align: center; padding: 8px;'>"
                f"Page {pagination_info['current_page']} of {pagination_info['total_pages']} "
                f"({pagination_info['start_idx'] + 1}-{pagination_info['end_idx']} of {pagination_info['total_count']})"
                f"</div>",
                unsafe_allow_html=True
            )
        
        with col4:
            if st.button("▶️ Next", disabled=pagination_info["current_page"] == pagination_info["total_pages"]):
                st.session_state.table_config.current_page += 1
                st.rerun()
        
        with col5:
            if st.button("⏭️ Last", disabled=pagination_info["current_page"] == pagination_info["total_pages"]):
                st.session_state.table_config.current_page = pagination_info["total_pages"]
                st.rerun()
    
    def _render_data_table(self, theorems: List[Theorem], 
                          search_results: Optional[List[SearchResult]] = None) -> Optional[Theorem]:
        """Render the actual data table."""
        if not theorems:
            st.info("No theorems match the current filters.")
            return None
        
        # Create relevance map
        relevance_map = {}
        if search_results:
            relevance_map = {r.theorem.id: r.relevance_score for r in search_results}
        
        # Prepare table data
        table_data = []
        for theorem in theorems:
            row = {}
            
            # Add columns based on configuration
            if "id" in st.session_state.table_config.columns:
                row["ID"] = theorem.short_id
            
            if "statement" in st.session_state.table_config.columns:
                # Truncate long statements for table display
                statement = theorem.statement
                if len(statement) > 100:
                    statement = statement[:97] + "..."
                row["Statement"] = statement
            
            if "type" in st.session_state.table_config.columns:
                row["Type"] = theorem.theorem_type_display
            
            if "validation_status" in st.session_state.table_config.columns:
                row["Status"] = theorem.validation_evidence.validation_status
            
            if "confidence" in st.session_state.table_config.columns:
                row["Confidence"] = f"{theorem.source_lineage.confidence:.2f}"
            
            if "pass_rate" in st.session_state.table_config.columns:
                row["Pass Rate"] = f"{theorem.validation_evidence.pass_rate:.2f}"
            
            if "symbols" in st.session_state.table_config.columns:
                symbols = ", ".join(theorem.symbols[:3])  # Show first 3 symbols
                if len(theorem.symbols) > 3:
                    symbols += f" (+{len(theorem.symbols) - 3})"
                row["Symbols"] = symbols
            
            if "assumptions" in st.session_state.table_config.columns:
                row["Assumptions"] = len(theorem.assumptions)
            
            # Add relevance if available
            if theorem.id in relevance_map:
                row["Relevance"] = f"{relevance_map[theorem.id]:.2f}"
            
            # Store theorem ID for selection
            row["_theorem_id"] = theorem.id
            table_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(table_data)
        
        # Display table with selection
        selected_indices = st.dataframe(
            df.drop(columns=["_theorem_id"]),
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row"
        )
        
        # Handle selection
        if selected_indices and "selection" in selected_indices and selected_indices["selection"]["rows"]:
            selected_idx = selected_indices["selection"]["rows"][0]
            selected_theorem_id = table_data[selected_idx]["_theorem_id"]
            
            # Find and return selected theorem
            for theorem in theorems:
                if theorem.id == selected_theorem_id:
                    st.session_state.selected_theorem_id = selected_theorem_id
                    return theorem
        
        return None
    
    def _render_table_footer(self, all_theorems: List[Theorem], 
                           filtered_theorems: List[Theorem],
                           pagination_info: Dict[str, Any]) -> None:
        """Render table footer with statistics."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Theorems", len(all_theorems))
        
        with col2:
            st.metric("Filtered Results", len(filtered_theorems))
        
        with col3:
            if filtered_theorems:
                avg_confidence = sum(t.source_lineage.confidence for t in filtered_theorems) / len(filtered_theorems)
                st.metric("Avg Confidence", f"{avg_confidence:.2f}")
        
        with col4:
            if filtered_theorems:
                pass_count = sum(1 for t in filtered_theorems if t.validation_evidence.validation_status == "PASS")
                pass_rate = pass_count / len(filtered_theorems)
                st.metric("Pass Rate", f"{pass_rate:.1%}")
    
    def _handle_export(self, theorems: List[Theorem]) -> None:
        """Handle theorem export functionality."""
        export_format = st.selectbox(
            "Export Format",
            options=["CSV", "JSON"],
            key="export_format_select"
        )
        
        if st.button("Download", key="download_export"):
            try:
                if export_format == "CSV":
                    # Prepare CSV data
                    data = []
                    for theorem in theorems:
                        data.append({
                            "ID": theorem.id,
                            "Statement": theorem.statement,
                            "Type": theorem.theorem_type,
                            "Confidence": theorem.source_lineage.confidence,
                            "Validation Status": theorem.validation_evidence.validation_status,
                            "Pass Rate": theorem.validation_evidence.pass_rate,
                            "Symbols": "; ".join(theorem.symbols),
                            "Assumptions": "; ".join(theorem.assumptions)
                        })
                    
                    df = pd.DataFrame(data)
                    csv_data = df.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_data,
                        file_name=f"mathbot_theorems_{int(time.time())}.csv",
                        mime="text/csv"
                    )
                
                elif export_format == "JSON":
                    import json
                    data = [theorem.dict() for theorem in theorems]
                    json_data = json.dumps(data, indent=2)
                    
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_data,
                        file_name=f"mathbot_theorems_{int(time.time())}.json",
                        mime="application/json"
                    )
                
                log_ui_interaction("theorem_browser", "export_completed", {
                    "format": export_format.lower(),
                    "theorem_count": len(theorems)
                })
                
            except Exception as e:
                self.logger.error(f"Export failed: {e}", exc_info=True)
                st.error(f"Export failed: {str(e)}")
    
    def get_browser_analytics(self) -> Dict[str, Any]:
        """Get browser analytics and performance data."""
        session = st.session_state.browser_session
        
        return {
            "performance_metrics": session.performance_metrics,
            "selected_theorems_count": len(session.selected_theorems),
            "last_sort_config": session.last_sort_config.dict(),
            "last_filter_config": session.last_filter_config.dict(),
            "table_config": st.session_state.table_config.dict()
        }
