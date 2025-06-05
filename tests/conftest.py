"""
Shared pytest configuration and fixtures for the test suite.
"""

import pytest
import tempfile
import shutil
import json
import sympy as sp
from pathlib import Path
from typing import List, Dict

from proofs.theorem_generator import Theorem, TheoremType, SourceLineage
from proofs.proof_attempt import ProofAttemptEngine


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp(prefix="mathbot_tests_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_theorem_data():
    """Provide sample theorem data for testing."""
    return {
        "simple_polynomial": {
            "id": "TEST_POLY_001",
            "statement": "(x + 1)² = x² + 2x + 1",
            "expression": "Eq((x + 1)**2, x**2 + 2*x + 1)",
            "type": "algebraic_identity",
            "assumptions": ["x ∈ ℝ"]
        },
        "trigonometric": {
            "id": "TEST_TRIG_001", 
            "statement": "sin²(x) + cos²(x) = 1",
            "expression": "Eq(sin(x)**2 + cos(x)**2, 1)",
            "type": "algebraic_identity",
            "assumptions": ["x ∈ ℝ"]
        },
        "false_statement": {
            "id": "TEST_FALSE_001",
            "statement": "x² = x + 1 for all x",
            "expression": "Eq(x**2, x + 1)",
            "type": "algebraic_identity",
            "assumptions": ["x ∈ ℝ"]
        },
        "functional_equation": {
            "id": "TEST_FUNC_001",
            "statement": "f(2x) = 4x² + 4x + 1",
            "expression": "Eq(f(2*x), 4*x**2 + 4*x + 1)",
            "type": "functional_equation", 
            "assumptions": ["x ∈ ℝ"]
        }
    }


@pytest.fixture
def theorem_factory(sample_theorem_data):
    """Factory function to create Theorem objects from sample data."""
    def _create_theorem(theorem_key: str) -> Theorem:
        data = sample_theorem_data[theorem_key]
        
        return Theorem(
            id=data["id"],
            statement=data["statement"],
            sympy_expression=sp.sympify(data["expression"]),
            theorem_type=TheoremType(data["type"]),
            assumptions=data["assumptions"],
            source_lineage=SourceLineage(
                original_formula=data["expression"].split("Eq(")[1].split(",")[0] if "Eq(" in data["expression"] else data["expression"],
                hypothesis_id=f"test_{theorem_key}",
                confidence=1.0 if "false" not in theorem_key else 0.5,
                validation_score=1.0,
                generation_method="test_factory"
            )
        )
    
    return _create_theorem


@pytest.fixture
def test_config():
    """Provide test configuration for proof engines."""
    return {
        'timeout_seconds': 30,  # Shorter timeout for tests
        'enable_caching': True,
        'max_substitution_values': 5,
        'debug_mode': True
    }


@pytest.fixture
def fast_engine(test_config, test_data_dir):
    """Create a fast proof engine for testing."""
    config = test_config.copy()
    config['cache_dir'] = str(test_data_dir / "cache")
    config['timeout_seconds'] = 10  # Very short timeout
    return ProofAttemptEngine(config)


@pytest.fixture
def slow_engine(test_config, test_data_dir):
    """Create a slower, more thorough proof engine for integration tests."""
    config = test_config.copy()
    config['cache_dir'] = str(test_data_dir / "cache")
    config['timeout_seconds'] = 60  # Longer timeout
    return ProofAttemptEngine(config)


@pytest.fixture(scope="session")
def real_theorems():
    """Load real theorems from the results file if available."""
    try:
        theorem_file = Path("results/theorems.json")
        if not theorem_file.exists():
            return []
        
        with open(theorem_file, 'r') as f:
            data = json.load(f)
        
        theorems = []
        for thm_data in data['theorems']:
            try:
                theorem = Theorem(
                    id=thm_data['id'],
                    statement=thm_data['statement'],
                    sympy_expression=sp.sympify(thm_data['sympy_expression']),
                    theorem_type=TheoremType(thm_data['theorem_type']),
                    assumptions=thm_data.get('assumptions', []),
                    source_lineage=SourceLineage(
                        original_formula=thm_data['source_lineage']['original_formula'],
                        hypothesis_id=thm_data['source_lineage']['hypothesis_id'],
                        confidence=thm_data['source_lineage']['confidence'],
                        validation_score=thm_data['source_lineage']['validation_score'],
                        generation_method=thm_data['source_lineage']['generation_method']
                    )
                )
                theorems.append(theorem)
            except Exception as e:
                # Skip malformed theorems
                continue
        
        return theorems
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return []


# Pytest markers for different test categories
pytest_plugins = []

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_real_data: mark test as requiring real theorem data"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add unit marker to most tests by default
        if not any(marker.name in ['integration', 'performance', 'slow'] for marker in item.iter_markers()):
            item.add_marker(pytest.mark.unit)
        
        # Mark slow tests
        if any(keyword in item.name.lower() for keyword in ['timeout', 'performance', 'batch', 'real_theorem']):
            item.add_marker(pytest.mark.slow)
        
        # Mark tests requiring real data
        if 'real_theorem' in item.name.lower():
            item.add_marker(pytest.mark.requires_real_data)


# Custom pytest options
def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-slow", action="store_true", default=False,
        help="run slow tests"
    )
    parser.addoption(
        "--run-integration", action="store_true", default=False,
        help="run integration tests"
    )
    parser.addoption(
        "--use-real-data", action="store_true", default=False,
        help="run tests that require real theorem data"
    )


def pytest_runtest_setup(item):
    """Setup function called before each test."""
    # Skip slow tests unless explicitly requested
    if item.get_closest_marker("slow") and not item.config.getoption("--run-slow"):
        pytest.skip("need --run-slow option to run")
    
    # Skip integration tests unless explicitly requested
    if item.get_closest_marker("integration") and not item.config.getoption("--run-integration"):
        pytest.skip("need --run-integration option to run")
    
    # Skip tests requiring real data unless available and requested
    if item.get_closest_marker("requires_real_data"):
        if not item.config.getoption("--use-real-data"):
            pytest.skip("need --use-real-data option to run")
        
        # Check if real data is actually available
        theorem_file = Path("results/theorems.json")
        if not theorem_file.exists():
            pytest.skip("real theorem data not available") 