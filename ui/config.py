"""
UI Configuration module for MathBot.

This module provides centralized, type-safe configuration for the MathBot UI
components, integrating with the existing MathBot configuration system.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys
import os

# Get the project root directory (two levels up from ui/config.py)
PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_DIR)

# Import from the root config module by explicit path to avoid naming conflicts
import importlib.util
root_config_path = os.path.join(PROJECT_ROOT_DIR, 'config.py')
spec = importlib.util.spec_from_file_location("root_config", root_config_path)
root_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(root_config)

PROJECT_ROOT = root_config.PROJECT_ROOT
logger = root_config.logger


@dataclass(frozen=True)
class UIConfig:
    """Configuration for MathBot UI components."""
    
    # Data Sources
    theorems_file: Path = PROJECT_ROOT / "results" / "theorems.json"
    formulas_file: Path = PROJECT_ROOT / "data" / "processed" / "sample_formulas.json"
    validation_file: Path = PROJECT_ROOT / "validation" / "validation_test_report.json"
    
    # UI Display Settings
    page_title: str = "MathBot - Mathematical Knowledge Explorer"
    page_icon: str = "🧮"
    layout: str = "wide"
    
    # Graph Visualization
    graph_height: int = 600
    graph_physics_enabled: bool = True
    max_graph_nodes: int = 100
    node_size_range: Tuple[int, int] = (10, 50)
    
    # Search & Filtering
    search_min_length: int = 2
    max_search_results: int = 50
    fuzzy_search_threshold: float = 0.7
    
    # Performance Settings
    theorems_per_page: int = 10
    cache_ttl_seconds: int = 300
    max_cache_size: int = 1000
    
    # Display Options
    show_validation_details: bool = True
    latex_renderer: str = "mathjax"
    decimal_precision: int = 3
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate file paths exist (warn if missing, don't fail)
        for field_name in ['theorems_file', 'formulas_file', 'validation_file']:
            file_path = getattr(self, field_name)
            if not file_path.exists():
                logger.warning(f"UI config: {field_name} not found at {file_path}")
        
        # Validate ranges
        if not (1 <= self.search_min_length <= 10):
            raise ValueError("search_min_length must be between 1 and 10")
        
        if not (0.0 <= self.fuzzy_search_threshold <= 1.0):
            raise ValueError("fuzzy_search_threshold must be between 0.0 and 1.0")
        
        if not (1 <= self.theorems_per_page <= 100):
            raise ValueError("theorems_per_page must be between 1 and 100")
        
        if not (1 <= self.max_search_results <= 1000):
            raise ValueError("max_search_results must be between 1 and 1000")
        
        if not (10 <= self.cache_ttl_seconds <= 3600):
            raise ValueError("cache_ttl_seconds must be between 10 and 3600")
        
        if not (100 <= self.max_cache_size <= 10000):
            raise ValueError("max_cache_size must be between 100 and 10000")
        
        # Validate node size range
        min_size, max_size = self.node_size_range
        if not (5 <= min_size < max_size <= 100):
            raise ValueError("node_size_range must have min >= 5, max <= 100, and min < max")


def get_ui_config() -> UIConfig:
    """
    Get UI configuration with environment variable overrides.
    
    Environment variables supported:
    - MATHBOT_UI_CACHE_TTL: Cache TTL in seconds
    - MATHBOT_UI_MAX_RESULTS: Maximum search results
    - MATHBOT_UI_PAGE_SIZE: Theorems per page
    - MATHBOT_UI_GRAPH_HEIGHT: Graph visualization height
    - MATHBOT_UI_MAX_NODES: Maximum nodes in graph
    - MATHBOT_UI_SEARCH_MIN_LEN: Minimum search query length
    - MATHBOT_UI_FUZZY_THRESHOLD: Fuzzy search threshold (0.0-1.0)
    
    Returns:
        UIConfig: Configured UI settings
    """
    kwargs = {}
    
    # Cache settings
    if cache_ttl := os.getenv("MATHBOT_UI_CACHE_TTL"):
        try:
            kwargs["cache_ttl_seconds"] = int(cache_ttl)
        except ValueError:
            logger.warning(f"Invalid MATHBOT_UI_CACHE_TTL value: {cache_ttl}")
    
    # Search settings
    if max_results := os.getenv("MATHBOT_UI_MAX_RESULTS"):
        try:
            kwargs["max_search_results"] = int(max_results)
        except ValueError:
            logger.warning(f"Invalid MATHBOT_UI_MAX_RESULTS value: {max_results}")
    
    if page_size := os.getenv("MATHBOT_UI_PAGE_SIZE"):
        try:
            kwargs["theorems_per_page"] = int(page_size)
        except ValueError:
            logger.warning(f"Invalid MATHBOT_UI_PAGE_SIZE value: {page_size}")
    
    # Graph settings
    if graph_height := os.getenv("MATHBOT_UI_GRAPH_HEIGHT"):
        try:
            kwargs["graph_height"] = int(graph_height)
        except ValueError:
            logger.warning(f"Invalid MATHBOT_UI_GRAPH_HEIGHT value: {graph_height}")
    
    if max_nodes := os.getenv("MATHBOT_UI_MAX_NODES"):
        try:
            kwargs["max_graph_nodes"] = int(max_nodes)
        except ValueError:
            logger.warning(f"Invalid MATHBOT_UI_MAX_NODES value: {max_nodes}")
    
    # Search quality settings
    if search_min_len := os.getenv("MATHBOT_UI_SEARCH_MIN_LEN"):
        try:
            kwargs["search_min_length"] = int(search_min_len)
        except ValueError:
            logger.warning(f"Invalid MATHBOT_UI_SEARCH_MIN_LEN value: {search_min_len}")
    
    if fuzzy_threshold := os.getenv("MATHBOT_UI_FUZZY_THRESHOLD"):
        try:
            kwargs["fuzzy_search_threshold"] = float(fuzzy_threshold)
        except ValueError:
            logger.warning(f"Invalid MATHBOT_UI_FUZZY_THRESHOLD value: {fuzzy_threshold}")
    
    # Custom file paths
    if theorems_file := os.getenv("MATHBOT_THEOREMS_FILE"):
        kwargs["theorems_file"] = Path(theorems_file)
    
    if formulas_file := os.getenv("MATHBOT_FORMULAS_FILE"):
        kwargs["formulas_file"] = Path(formulas_file)
    
    if validation_file := os.getenv("MATHBOT_VALIDATION_FILE"):
        kwargs["validation_file"] = Path(validation_file)
    
    try:
        return UIConfig(**kwargs)
    except ValueError as e:
        logger.error(f"Invalid UI configuration: {e}")
        logger.info("Using default UI configuration")
        return UIConfig()


# Default configuration instance
ui_config = get_ui_config() 