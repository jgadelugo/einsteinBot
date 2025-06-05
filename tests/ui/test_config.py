"""
Unit tests for UI configuration system.
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch

from ui.config import UIConfig, get_ui_config


class TestUIConfig:
    """Test UI configuration dataclass."""
    
    def test_default_config_creation(self):
        """Test creating default configuration."""
        config = UIConfig()
        
        # Test basic fields
        assert config.page_title == "MathBot - Mathematical Knowledge Explorer"
        assert config.page_icon == "🧮"
        assert config.layout == "wide"
        
        # Test search settings
        assert config.search_min_length >= 1
        assert config.max_search_results > 0
        assert 0.0 <= config.fuzzy_search_threshold <= 1.0
        
        # Test performance settings
        assert config.cache_ttl_seconds > 0
        assert config.max_cache_size > 0
        assert config.theorems_per_page > 0
        
        # Test node size range
        min_size, max_size = config.node_size_range
        assert min_size < max_size
        assert min_size >= 5
        assert max_size <= 100
    
    def test_file_paths(self):
        """Test file path configuration."""
        config = UIConfig()
        
        # Test that file paths are Path objects
        assert isinstance(config.theorems_file, Path)
        assert isinstance(config.formulas_file, Path)
        assert isinstance(config.validation_file, Path)
        
        # Test file names
        assert config.theorems_file.name == "theorems.json"
        assert "formulas" in config.formulas_file.name
        assert "validation" in config.validation_file.name
    
    def test_validation_ranges(self):
        """Test configuration validation."""
        # Valid configuration should not raise
        config = UIConfig(
            search_min_length=3,
            fuzzy_search_threshold=0.8,
            theorems_per_page=15,
            max_search_results=25,
            cache_ttl_seconds=600,
            max_cache_size=500
        )
        
        # Test values are set correctly
        assert config.search_min_length == 3
        assert config.fuzzy_search_threshold == 0.8
        assert config.theorems_per_page == 15
    
    def test_validation_errors(self):
        """Test configuration validation errors."""
        # Test search_min_length validation
        with pytest.raises(ValueError, match="search_min_length must be between 1 and 10"):
            UIConfig(search_min_length=0)
        
        with pytest.raises(ValueError, match="search_min_length must be between 1 and 10"):
            UIConfig(search_min_length=11)
        
        # Test fuzzy_search_threshold validation
        with pytest.raises(ValueError, match="fuzzy_search_threshold must be between 0.0 and 1.0"):
            UIConfig(fuzzy_search_threshold=-0.1)
        
        with pytest.raises(ValueError, match="fuzzy_search_threshold must be between 0.0 and 1.0"):
            UIConfig(fuzzy_search_threshold=1.1)
        
        # Test theorems_per_page validation
        with pytest.raises(ValueError, match="theorems_per_page must be between 1 and 100"):
            UIConfig(theorems_per_page=0)
        
        # Test cache_ttl_seconds validation
        with pytest.raises(ValueError, match="cache_ttl_seconds must be between 10 and 3600"):
            UIConfig(cache_ttl_seconds=5)
        
        # Test node_size_range validation
        with pytest.raises(ValueError, match="node_size_range must have min >= 5, max <= 100, and min < max"):
            UIConfig(node_size_range=(60, 50))  # min > max


class TestGetUIConfig:
    """Test get_ui_config function with environment variables."""
    
    def test_default_config(self):
        """Test getting default configuration without env vars."""
        config = get_ui_config()
        assert isinstance(config, UIConfig)
        assert config.cache_ttl_seconds == 300  # default
    
    @patch.dict(os.environ, {'MATHBOT_UI_CACHE_TTL': '600'})
    def test_cache_ttl_override(self):
        """Test cache TTL override from environment."""
        config = get_ui_config()
        assert config.cache_ttl_seconds == 600
    
    @patch.dict(os.environ, {'MATHBOT_UI_MAX_RESULTS': '75'})
    def test_max_results_override(self):
        """Test max results override from environment."""
        config = get_ui_config()
        assert config.max_search_results == 75
    
    @patch.dict(os.environ, {'MATHBOT_UI_PAGE_SIZE': '25'})
    def test_page_size_override(self):
        """Test page size override from environment."""
        config = get_ui_config()
        assert config.theorems_per_page == 25
    
    @patch.dict(os.environ, {'MATHBOT_UI_GRAPH_HEIGHT': '800'})
    def test_graph_height_override(self):
        """Test graph height override from environment."""
        config = get_ui_config()
        assert config.graph_height == 800
    
    @patch.dict(os.environ, {'MATHBOT_UI_MAX_NODES': '150'})
    def test_max_nodes_override(self):
        """Test max nodes override from environment."""
        config = get_ui_config()
        assert config.max_graph_nodes == 150
    
    @patch.dict(os.environ, {'MATHBOT_UI_SEARCH_MIN_LEN': '3'})
    def test_search_min_len_override(self):
        """Test search min length override from environment."""
        config = get_ui_config()
        assert config.search_min_length == 3
    
    @patch.dict(os.environ, {'MATHBOT_UI_FUZZY_THRESHOLD': '0.8'})
    def test_fuzzy_threshold_override(self):
        """Test fuzzy threshold override from environment."""
        config = get_ui_config()
        assert config.fuzzy_search_threshold == 0.8
    
    @patch.dict(os.environ, {
        'MATHBOT_THEOREMS_FILE': '/custom/theorems.json',
        'MATHBOT_FORMULAS_FILE': '/custom/formulas.json',
        'MATHBOT_VALIDATION_FILE': '/custom/validation.json'
    })
    def test_file_path_overrides(self):
        """Test file path overrides from environment."""
        config = get_ui_config()
        assert str(config.theorems_file) == '/custom/theorems.json'
        assert str(config.formulas_file) == '/custom/formulas.json'
        assert str(config.validation_file) == '/custom/validation.json'
    
    @patch.dict(os.environ, {'MATHBOT_UI_CACHE_TTL': 'invalid'})
    def test_invalid_environment_values(self):
        """Test handling of invalid environment variable values."""
        # Should not raise and use defaults
        config = get_ui_config()
        assert config.cache_ttl_seconds == 300  # default value
    
    @patch.dict(os.environ, {'MATHBOT_UI_CACHE_TTL': '5'})  # Too low
    def test_invalid_range_from_env(self):
        """Test handling of out-of-range values from environment."""
        # Should fall back to default configuration
        config = get_ui_config()
        assert config.cache_ttl_seconds == 300  # default value


class TestConfigIntegration:
    """Test configuration integration with other components."""
    
    def test_immutability(self):
        """Test that configuration is immutable."""
        config = UIConfig()
        
        # Should not be able to modify frozen dataclass
        with pytest.raises(AttributeError):
            config.cache_ttl_seconds = 1000
    
    def test_config_consistency(self):
        """Test that configuration values are consistent."""
        config = UIConfig()
        
        # Node size range should be consistent
        min_size, max_size = config.node_size_range
        assert min_size < max_size
        
        # Search settings should be reasonable
        assert config.search_min_length < config.max_search_results
        
        # Performance settings should be reasonable
        assert config.cache_ttl_seconds > 10
        assert config.max_cache_size > config.theorems_per_page 