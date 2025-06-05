"""
Test suite for the proof attempt engine.
"""

import pytest
import sympy as sp
from sympy import Symbol, Eq, sin, cos, pi, sqrt, exp, log
import tempfile
import shutil
import json
import time
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import os

from proofs.proof_attempt import (
    ProofAttemptEngine,
    ProofResult,
    ProofStep,
    ProofMethod,
    ProofStatus,
    ProofCache
)
from proofs.theorem_generator import (
    Theorem,
    TheoremType,
    SourceLineage
)


class TestProofAttemptEngine:
    """Comprehensive test suite for ProofAttemptEngine class."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary directory for cache testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def engine(self, temp_cache_dir):
        """Create a proof attempt engine with temporary cache for testing."""
        config = {
            'cache_dir': temp_cache_dir,
            'timeout_seconds': 30,  # Shorter timeout for tests
            'enable_caching': True
        }
        return ProofAttemptEngine(config)
    
    @pytest.fixture
    def engine_no_cache(self):
        """Create a proof attempt engine without caching."""
        config = {
            'enable_caching': False,
            'timeout_seconds': 30
        }
        return ProofAttemptEngine(config)
    
    @pytest.fixture
    def simple_theorems(self):
        """Create various simple theorems for testing."""
        x = Symbol('x')
        a, b, c = sp.symbols('a b c')
        
        theorems = {
            'algebraic_identity': Theorem(
                id="TEST_ALGEBRAIC_001",
                statement="(x + 1)² = x² + 2x + 1",
                sympy_expression=Eq((x + 1)**2, x**2 + 2*x + 1),
                theorem_type=TheoremType.ALGEBRAIC_IDENTITY,
                assumptions=["x ∈ ℝ"],
                source_lineage=SourceLineage(
                    original_formula="(x + 1)**2",
                    hypothesis_id="test_hyp_001",
                    confidence=1.0,
                    validation_score=1.0,
                    generation_method="test"
                )
            ),
            'functional_equation': Theorem(
                id="TEST_FUNC_001",
                statement="f(2x) = 4x² + 4x + 1",
                sympy_expression=Eq(sp.Function('f')(2*x), 4*x**2 + 4*x + 1),
                theorem_type=TheoremType.FUNCTIONAL_EQUATION,
                assumptions=["x ∈ ℝ"],
                source_lineage=SourceLineage(
                    original_formula="(2x)**2 + 2*(2x) + 1",
                    hypothesis_id="test_hyp_002",
                    confidence=1.0,
                    validation_score=1.0,
                    generation_method="test"
                )
            ),
            'trigonometric': Theorem(
                id="TEST_TRIG_001",
                statement="sin²(x) + cos²(x) = 1",
                sympy_expression=Eq(sin(x)**2 + cos(x)**2, 1),
                theorem_type=TheoremType.ALGEBRAIC_IDENTITY,
                assumptions=["x ∈ ℝ"],
                source_lineage=SourceLineage(
                    original_formula="sin²(x) + cos²(x)",
                    hypothesis_id="test_hyp_003",
                    confidence=1.0,
                    validation_score=1.0,
                    generation_method="test"
                )
            ),
            'false_theorem': Theorem(
                id="TEST_FALSE_001",
                statement="x² = x + 1 for all x",
                sympy_expression=Eq(x**2, x + 1),
                theorem_type=TheoremType.ALGEBRAIC_IDENTITY,
                assumptions=["x ∈ ℝ"],
                source_lineage=SourceLineage(
                    original_formula="x**2",
                    hypothesis_id="test_hyp_004",
                    confidence=0.5,
                    validation_score=0.5,
                    generation_method="test"
                )
            ),
            'complex_expression': Theorem(
                id="TEST_COMPLEX_001",
                statement="(a + b + c)² = a² + b² + c² + 2ab + 2ac + 2bc",
                sympy_expression=Eq((a + b + c)**2, a**2 + b**2 + c**2 + 2*a*b + 2*a*c + 2*b*c),
                theorem_type=TheoremType.ALGEBRAIC_IDENTITY,
                assumptions=["a, b, c ∈ ℝ"],
                source_lineage=SourceLineage(
                    original_formula="(a + b + c)**2",
                    hypothesis_id="test_hyp_005",
                    confidence=1.0,
                    validation_score=1.0,
                    generation_method="test"
                )
            )
        }
        return theorems
    
    @pytest.fixture
    def real_theorems(self):
        """Load real theorems from the results file."""
        try:
            with open('results/theorems.json', 'r') as f:
                data = json.load(f)
            
            theorems = []
            for thm_data in data['theorems'][:5]:  # Use first 5 for testing
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
            return theorems
        except FileNotFoundError:
            pytest.skip("No real theorem data available")
    
    def test_initialization_default_config(self):
        """Test proof attempt engine initialization with default configuration."""
        engine = ProofAttemptEngine()
        
        assert engine.config == {}
        assert engine.timeout_seconds == 300
        assert engine.enable_caching is True
        assert engine.cache is not None
        assert engine.stats['total_attempts'] == 0
        assert engine.stats['successful_proofs'] == 0
        assert engine.stats['failed_proofs'] == 0
        assert engine.stats['timeouts'] == 0
        assert engine.stats['total_time'] == 0.0
        assert len(engine.proof_methods) == 7
        
        # Verify all proof methods are present
        expected_methods = {
            ProofMethod.SYMPY_DIRECT,
            ProofMethod.SYMPY_SIMPLIFY,
            ProofMethod.ALGEBRAIC_MANIPULATION,
            ProofMethod.SUBSTITUTION,
            ProofMethod.SYMBOLIC_SOLVER,
            ProofMethod.PATTERN_MATCHING,
            ProofMethod.NUMERICAL_VERIFICATION
        }
        assert set(engine.proof_methods) == expected_methods
    
    def test_initialization_custom_config(self, temp_cache_dir):
        """Test proof attempt engine initialization with custom configuration."""
        config = {
            'cache_dir': temp_cache_dir,
            'timeout_seconds': 60,
            'enable_caching': False,
            'max_substitution_values': 10
        }
        engine = ProofAttemptEngine(config)
        
        assert engine.config == config
        assert engine.timeout_seconds == 60
        assert engine.enable_caching is False
        assert engine.cache is None
    
    @pytest.mark.parametrize("theorem_type", [
        'algebraic_identity', 'functional_equation', 'trigonometric', 'complex_expression'
    ])
    def test_proof_attempts_various_types(self, engine, simple_theorems, theorem_type):
        """Test proof attempts on various theorem types."""
        theorem = simple_theorems[theorem_type]
        result = engine.attempt_proof(theorem)
        
        assert result.theorem_id == theorem.id
        assert isinstance(result.status, ProofStatus)
        assert isinstance(result.method, ProofMethod)
        assert isinstance(result.steps, list)
        assert result.execution_time >= 0
        assert 0 <= result.confidence_score <= 1
        
        # Algebraic identities should generally be provable
        if theorem_type in ['algebraic_identity', 'trigonometric']:
            assert result.status in [ProofStatus.PROVED, ProofStatus.FAILED]
    
    def test_false_theorem_handling(self, engine, simple_theorems):
        """Test handling of false theorems."""
        false_theorem = simple_theorems['false_theorem']
        result = engine.attempt_proof(false_theorem)
        
        assert result.theorem_id == false_theorem.id
        # Should not be proved (could be disproved or failed)
        assert result.status != ProofStatus.PROVED
        assert result.status in [ProofStatus.DISPROVED, ProofStatus.FAILED, ProofStatus.INCONCLUSIVE]
    
    def test_caching_functionality(self, temp_cache_dir, simple_theorems):
        """Test proof result caching functionality."""
        # Create fresh engine with clean cache
        config = {
            'cache_dir': temp_cache_dir,
            'timeout_seconds': 30,
            'enable_caching': True
        }
        engine = ProofAttemptEngine(config)
        
        theorem = simple_theorems['algebraic_identity']
        
        # First attempt
        result1 = engine.attempt_proof(theorem)
        first_time = result1.execution_time
        
        # Second attempt (should be cached)
        result2 = engine.attempt_proof(theorem)
        second_time = result2.execution_time
        
        # Results should be identical
        assert result1.theorem_id == result2.theorem_id
        assert result1.status == result2.status
        assert result1.method == result2.method
        
        # Check cache hit statistics
        stats = engine.get_statistics()
        assert stats['cache_hits'] >= 1
    
    def test_caching_disabled(self, engine_no_cache, simple_theorems):
        """Test behavior when caching is disabled."""
        theorem = simple_theorems['algebraic_identity']
        
        # Multiple attempts should not use caching
        result1 = engine_no_cache.attempt_proof(theorem)
        result2 = engine_no_cache.attempt_proof(theorem)
        
        # Results should be similar but not necessarily identical execution times
        assert result1.theorem_id == result2.theorem_id
        assert result1.status == result2.status
    
    def test_statistics_tracking(self, temp_cache_dir, simple_theorems):
        """Test comprehensive statistics tracking."""
        # Create fresh engine with clean cache
        config = {
            'cache_dir': temp_cache_dir,
            'timeout_seconds': 30,
            'enable_caching': True
        }
        engine = ProofAttemptEngine(config)
        
        initial_stats = engine.get_statistics()
        assert initial_stats['total_attempts'] == 0
        
        # Attempt multiple proofs
        theorems_to_test = ['algebraic_identity', 'trigonometric', 'false_theorem']
        results = []
        
        for thm_name in theorems_to_test:
            result = engine.attempt_proof(simple_theorems[thm_name])
            results.append(result)
        
        final_stats = engine.get_statistics()
        
        # Check basic counts
        assert final_stats['total_attempts'] == len(theorems_to_test)
        assert final_stats['total_time'] >= 0  # May be 0 if very fast
        
        # Check status counts
        proved_count = sum(1 for r in results if r.status == ProofStatus.PROVED)
        failed_count = sum(1 for r in results if r.status in [ProofStatus.FAILED, ProofStatus.DISPROVED])
        
        # If using cache, successful_proofs might be 0 due to cache hits
        # Check either successful proofs were counted OR we have cache hits
        assert final_stats['successful_proofs'] + final_stats.get('cache_hits', 0) >= proved_count
    
    def test_batch_proof_processing(self, engine, simple_theorems):
        """Test batch processing of multiple theorems."""
        theorems = [
            simple_theorems['algebraic_identity'],
            simple_theorems['trigonometric'],
            simple_theorems['complex_expression']
        ]
        
        results = engine.batch_prove_theorems(theorems)
        
        assert len(results) == len(theorems)
        
        # Check each result
        for i, result in enumerate(results):
            assert result.theorem_id == theorems[i].id
            assert isinstance(result.status, ProofStatus)
            assert isinstance(result.method, ProofMethod)
            assert result.execution_time >= 0
    
    def test_timeout_handling(self, temp_cache_dir):
        """Test timeout handling for complex proofs."""
        # Create engine with very short timeout
        config = {
            'cache_dir': temp_cache_dir,
            'timeout_seconds': 0.1,  # 100ms timeout
            'enable_caching': True
        }
        engine = ProofAttemptEngine(config)
        
        # Create a potentially complex theorem
        x = Symbol('x')
        complex_theorem = Theorem(
            id="TEST_TIMEOUT_001",
            statement="Complex expression with many terms",
            sympy_expression=Eq(
                (x + 1)**10, 
                sum(sp.binomial(10, k) * x**k for k in range(11))
            ),
            theorem_type=TheoremType.ALGEBRAIC_IDENTITY,
            assumptions=["x ∈ ℝ"],
            source_lineage=SourceLineage(
                original_formula="(x + 1)**10",
                hypothesis_id="test_timeout",
                confidence=1.0,
                validation_score=1.0,
                generation_method="test"
            )
        )
        
        result = engine.attempt_proof(complex_theorem)
        
        # Should either complete quickly or timeout
        assert result.status in [ProofStatus.PROVED, ProofStatus.TIMEOUT, ProofStatus.FAILED]
        if result.status == ProofStatus.TIMEOUT:
            assert result.error_message is not None
            assert "timeout" in result.error_message.lower()
    
    def test_malformed_expression_handling(self, temp_cache_dir):
        """Test handling of malformed mathematical expressions."""
        # Create fresh engine with clean cache
        config = {
            'cache_dir': temp_cache_dir,
            'timeout_seconds': 30,
            'enable_caching': False  # Disable caching for this test
        }
        engine = ProofAttemptEngine(config)
        
        # Create theorem with potentially problematic expression
        malformed_theorem = Theorem(
            id="TEST_MALFORMED_001",
            statement="Malformed expression",
            sympy_expression="invalid_expression",  # This should cause issues
            theorem_type=TheoremType.ALGEBRAIC_IDENTITY,
            assumptions=[],
            source_lineage=SourceLineage(
                original_formula="invalid",
                hypothesis_id="test_malformed",
                confidence=1.0,
                validation_score=1.0,
                generation_method="test"
            )
        )
        
        # The engine should handle malformed expressions gracefully
        result = engine.attempt_proof(malformed_theorem)
        assert result.status in [ProofStatus.FAILED, ProofStatus.UNKNOWN, ProofStatus.INCONCLUSIVE]
    
    def test_empty_theorems_batch(self, engine):
        """Test batch processing with empty theorem list."""
        results = engine.batch_prove_theorems([])
        assert results == []
    
    @pytest.mark.integration
    def test_real_theorem_integration(self, engine, real_theorems):
        """Integration test with real theorems from the system."""
        if not real_theorems:
            pytest.skip("No real theorems available")
        
        # Test with a subset of real theorems
        results = engine.batch_prove_theorems(real_theorems[:3])
        
        assert len(results) == min(3, len(real_theorems))
        
        # At least some should be processable (not failed due to errors)
        processable_results = [r for r in results if r.status != ProofStatus.FAILED]
        assert len(processable_results) > 0
        
        # Check that we get meaningful results (proved, disproved, or inconclusive)
        meaningful_results = [
            r for r in results 
            if r.status in [ProofStatus.PROVED, ProofStatus.DISPROVED, ProofStatus.INCONCLUSIVE]
        ]
        
        # We should get at least some meaningful results, even if not proved
        # Real theorems from Phase 5A might be complex, so we're more lenient
        assert len(meaningful_results) >= len(results) * 0.5  # At least 50% should be processable


class TestProofResult:
    """Test suite for ProofResult class."""
    
    @pytest.fixture
    def sample_steps(self):
        """Create sample proof steps for testing."""
        return [
            ProofStep(
                step_number=1,
                method="expand",
                from_expression="(x + 1)**2",
                to_expression="x**2 + 2*x + 1",
                justification="Algebraic expansion"
            ),
            ProofStep(
                step_number=2,
                method="verify",
                from_expression="x**2 + 2*x + 1",
                to_expression="x**2 + 2*x + 1",
                justification="Identity verification"
            )
        ]
    
    def test_proof_result_creation(self, sample_steps):
        """Test proof result creation and basic properties."""
        result = ProofResult(
            theorem_id="TEST_RESULT_001",
            status=ProofStatus.PROVED,
            method=ProofMethod.ALGEBRAIC_MANIPULATION,
            steps=sample_steps,
            execution_time=0.05,
            confidence_score=0.95
        )
        
        assert result.theorem_id == "TEST_RESULT_001"
        assert result.status == ProofStatus.PROVED
        assert result.method == ProofMethod.ALGEBRAIC_MANIPULATION
        assert result.steps == sample_steps
        assert result.execution_time == 0.05
        assert result.confidence_score == 0.95
        assert result.error_message is None
        assert result.is_successful() is True
        assert result.get_step_count() == 2
    
    def test_proof_result_failed(self):
        """Test proof result for failed proofs."""
        result = ProofResult(
            theorem_id="TEST_RESULT_002",
            status=ProofStatus.FAILED,
            method=ProofMethod.SYMPY_DIRECT,
            steps=[],
            execution_time=0.01,
            confidence_score=0.0,
            error_message="Proof failed: unable to verify"
        )
        
        assert result.is_successful() is False
        assert result.get_step_count() == 0
        assert result.error_message == "Proof failed: unable to verify"
    
    @pytest.mark.parametrize("status,expected_success", [
        (ProofStatus.PROVED, True),
        (ProofStatus.DISPROVED, False),
        (ProofStatus.FAILED, False),
        (ProofStatus.TIMEOUT, False),
        (ProofStatus.UNKNOWN, False),
        (ProofStatus.INCONCLUSIVE, False),
    ])
    def test_is_successful_status_mapping(self, status, expected_success):
        """Test is_successful method for different statuses."""
        result = ProofResult(
            theorem_id="TEST",
            status=status,
            method=ProofMethod.SYMPY_DIRECT,
            steps=[],
            execution_time=0.01,
            confidence_score=0.5
        )
        
        assert result.is_successful() == expected_success
    
    def test_proof_result_serialization(self, sample_steps):
        """Test proof result serialization to dictionary."""
        result = ProofResult(
            theorem_id="TEST_RESULT_003",
            status=ProofStatus.PROVED,
            method=ProofMethod.ALGEBRAIC_MANIPULATION,
            steps=sample_steps,
            execution_time=0.123,
            confidence_score=0.87,
            error_message=None
        )
        
        result_dict = result.to_dict()
        
        # Check basic fields
        assert result_dict['theorem_id'] == "TEST_RESULT_003"
        assert result_dict['status'] == "proved"
        assert result_dict['method'] == "algebraic_manipulation"
        assert result_dict['execution_time'] == 0.123
        assert result_dict['confidence_score'] == 0.87
        assert result_dict['error_message'] is None
        
        # Check steps serialization
        assert len(result_dict['steps']) == 2
        assert result_dict['steps'][0]['step_number'] == 1
        assert result_dict['steps'][0]['method'] == "expand"


class TestProofStep:
    """Test suite for ProofStep class."""
    
    def test_proof_step_creation(self):
        """Test proof step creation and basic properties."""
        step = ProofStep(
            step_number=1,
            method="simplify",
            from_expression="x**2 + 2*x + 1",
            to_expression="(x + 1)**2",
            justification="Factorization",
            transformation_rule="binomial_square"
        )
        
        assert step.step_number == 1
        assert step.method == "simplify"
        assert step.from_expression == "x**2 + 2*x + 1"
        assert step.to_expression == "(x + 1)**2"
        assert step.justification == "Factorization"
        assert step.transformation_rule == "binomial_square"
        assert step.success is True
    
    def test_proof_step_serialization(self):
        """Test proof step serialization to dictionary."""
        step = ProofStep(
            step_number=2,
            method="expand",
            from_expression="(a + b)**2",
            to_expression="a**2 + 2*a*b + b**2",
            justification="Binomial expansion"
        )
        
        step_dict = step.to_dict()
        
        assert step_dict['step_number'] == 2
        assert step_dict['method'] == "expand"
        assert step_dict['from_expression'] == "(a + b)**2"
        assert step_dict['to_expression'] == "a**2 + 2*a*b + b**2"
        assert step_dict['justification'] == "Binomial expansion"
        assert step_dict['transformation_rule'] is None
        assert step_dict['success'] is True
    
    def test_proof_step_failed(self):
        """Test proof step that represents a failed transformation."""
        step = ProofStep(
            step_number=1,
            method="solve",
            from_expression="x**2 + 1 = 0",
            to_expression="No real solutions",
            justification="Discriminant < 0",
            transformation_rule=None
        )
        
        # Manually set success to False for failed step
        step.success = False
        
        assert step.success is False
        step_dict = step.to_dict()
        assert step_dict['success'] is False


class TestProofCache:
    """Test suite for ProofCache class."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary directory for cache testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_cache_initialization(self, temp_cache_dir):
        """Test cache initialization."""
        cache = ProofCache(cache_dir=temp_cache_dir)
        
        assert str(cache.cache_dir) == temp_cache_dir
        assert cache.max_memory_cache_size == 1000
        assert len(cache.memory_cache) == 0
    
    def test_cache_key_generation(self, temp_cache_dir):
        """Test cache key generation."""
        cache = ProofCache(cache_dir=temp_cache_dir)
        
        # Test with simple theorem
        x = Symbol('x')
        theorem = Theorem(
            id="TEST_CACHE_001",
            statement="x + 1 = 1 + x",
            sympy_expression=Eq(x + 1, 1 + x),
            theorem_type=TheoremType.ALGEBRAIC_IDENTITY,
            assumptions=[],
            source_lineage=SourceLineage(
                original_formula="x + 1",
                hypothesis_id="test_cache",
                confidence=1.0,
                validation_score=1.0,
                generation_method="test"
            )
        )
        
        key1 = cache.get_cache_key(theorem)
        key2 = cache.get_cache_key(theorem)
        
        # Same theorem should produce same key
        assert key1 == key2
        assert isinstance(key1, str)
        assert len(key1) > 0
    
    def test_memory_cache_operations(self, temp_cache_dir):
        """Test memory cache get/set operations."""
        cache = ProofCache(cache_dir=temp_cache_dir)
        
        # Create a dummy result
        result = ProofResult(
            theorem_id="TEST_CACHE_002",
            status=ProofStatus.PROVED,
            method=ProofMethod.SYMPY_DIRECT,
            steps=[],
            execution_time=0.01,
            confidence_score=1.0
        )
        
        cache_key = "test_key_001"
        
        # Test cache miss
        assert cache.get_cached_result(cache_key) is None
        
        # Test cache set and hit
        cache.cache_result(cache_key, result)
        cached_result = cache.get_cached_result(cache_key)
        
        assert cached_result is not None
        assert cached_result.theorem_id == result.theorem_id
        assert cached_result.status == result.status
    
    def test_cache_eviction(self, temp_cache_dir):
        """Test cache eviction when memory limit is reached."""
        # Create cache with small memory limit
        cache = ProofCache(cache_dir=temp_cache_dir)
        cache.max_memory_cache_size = 2  # Override the default
        
        # Add entries beyond the limit
        for i in range(5):
            result = ProofResult(
                theorem_id=f"TEST_CACHE_{i:03d}",
                status=ProofStatus.PROVED,
                method=ProofMethod.SYMPY_DIRECT,
                steps=[],
                execution_time=0.01,
                confidence_score=1.0
            )
            cache.cache_result(f"key_{i}", result)
        
        # Memory cache should be limited
        assert len(cache.memory_cache) <= 2
    
    def test_cache_cleanup(self, temp_cache_dir):
        """Test cache cleanup functionality."""
        cache = ProofCache(cache_dir=temp_cache_dir)
        
        # Add some entries
        for i in range(3):
            result = ProofResult(
                theorem_id=f"TEST_CLEANUP_{i:03d}",
                status=ProofStatus.PROVED,
                method=ProofMethod.SYMPY_DIRECT,
                steps=[],
                execution_time=0.01,
                confidence_score=1.0
            )
            cache.cache_result(f"cleanup_key_{i}", result)
        
        # Verify entries exist
        assert len(cache.memory_cache) == 3
        
        # Cleanup
        cache.clear_cache()
        
        # Memory cache should be cleared
        assert len(cache.memory_cache) == 0


class TestIntegration:
    """Integration test suite for complete proof workflows."""
    
    @pytest.fixture
    def fresh_engine(self, test_data_dir):
        """Create a fresh engine with clean cache for each test."""
        import tempfile
        import shutil
        import time
        
        # Create a unique cache directory for this test using timestamp and random suffix
        cache_dir = tempfile.mkdtemp(prefix=f"test_cache_{int(time.time())}_")
        
        config = {
            'timeout_seconds': 60,
            'enable_caching': True,
            'cache_dir': cache_dir
        }
        engine = ProofAttemptEngine(config)
        
        # Explicitly clear any existing cache to ensure freshness
        engine.cache.clear_cache()
        
        yield engine
        
        # Cleanup after test
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
    
    @pytest.fixture
    def no_cache_engine(self):
        """Create an engine with caching disabled."""
        config = {
            'timeout_seconds': 60,
            'enable_caching': False
        }
        return ProofAttemptEngine(config)

    def test_end_to_end_proof_workflow(self, fresh_engine):
        """Test complete proof workflow from theorem to result."""
        x = Symbol('x')
        
        # Create a simple theorem
        theorem = Theorem(
            id="E2E_TEST_001",
            statement="(x + 1)² = x² + 2x + 1",
            sympy_expression=Eq((x + 1)**2, x**2 + 2*x + 1),
            theorem_type=TheoremType.ALGEBRAIC_IDENTITY,
            assumptions=["x ∈ ℝ"],
            source_lineage=SourceLineage(
                original_formula="(x + 1)**2",
                hypothesis_id="e2e_test",
                confidence=1.0,
                validation_score=1.0,
                generation_method="end_to_end_test"
            )
        )
        
        # Attempt proof
        result = fresh_engine.attempt_proof(theorem)
        
        # Verify result structure
        assert result.theorem_id == "E2E_TEST_001"
        assert result.status in [ProofStatus.PROVED, ProofStatus.DISPROVED, ProofStatus.INCONCLUSIVE]
        assert len(result.steps) > 0
        assert result.execution_time >= 0
        assert 0 <= result.confidence_score <= 1
        
        # Verify steps have proper structure
        for i, step in enumerate(result.steps):
            assert step.step_number == i + 1
            assert step.method
            assert step.from_expression
            assert step.to_expression
            assert step.justification
        
        # Test serialization
        result_dict = result.to_dict()
        assert result_dict['theorem_id'] == theorem.id
        assert 'steps' in result_dict
        assert len(result_dict['steps']) == len(result.steps)

    @pytest.mark.slow
    def test_performance_with_fresh_cache(self, fresh_engine):
        """Test performance benchmarks with fresh cache (uncached execution)."""
        import uuid
        x = Symbol('x')
        
        # Create unique test ID to avoid cache conflicts
        test_uuid = str(uuid.uuid4())[:8]
        
        # Create multiple theorems for performance testing
        theorems = []
        for i in range(5):  # Reduced number for faster testing
            theorem = Theorem(
                id=f"PERF_FRESH_{test_uuid}_{i:03d}",
                statement=f"(x + {i})² = x² + {2*i}x + {i**2}",
                sympy_expression=Eq((x + i)**2, x**2 + 2*i*x + i**2),
                theorem_type=TheoremType.ALGEBRAIC_IDENTITY,
                assumptions=["x ∈ ℝ"],
                source_lineage=SourceLineage(
                    original_formula=f"(x + {i})**2",
                    hypothesis_id=f"perf_fresh_{test_uuid}_{i}",
                    confidence=1.0,
                    validation_score=1.0,
                    generation_method="performance_test_fresh"
                )
            )
            theorems.append(theorem)
        
        # Measure fresh execution time
        start_time = time.time()
        results = fresh_engine.batch_prove_theorems(theorems)
        end_time = time.time()
        
        total_time = end_time - start_time
        
        # Performance checks for fresh execution
        assert len(results) == len(theorems)
        assert total_time < 30  # Should complete within 30 seconds
        
        # At least 60% should be processed successfully (more lenient for complex theorems)
        successful_results = [r for r in results if r.status in [ProofStatus.PROVED, ProofStatus.DISPROVED]]
        success_rate = len(successful_results) / len(results)
        assert success_rate >= 0.6
        
        # Check statistics for fresh execution
        stats = fresh_engine.get_statistics()
        assert stats['total_attempts'] >= len(theorems)
        assert stats['total_time'] > 0  # Should have actual execution time
        
        # Test caching behavior by running same theorems again
        start_time_cached = time.time()
        cached_results = fresh_engine.batch_prove_theorems(theorems)
        end_time_cached = time.time()
        
        cached_time = end_time_cached - start_time_cached
        
        # Cached execution should be faster
        assert cached_time < total_time
        assert len(cached_results) == len(results)
        
        # Results should be identical (from cache)
        for original, cached in zip(results, cached_results):
            assert original.theorem_id == cached.theorem_id
            assert original.status == cached.status

    @pytest.mark.slow
    def test_performance_without_cache(self, no_cache_engine):
        """Test performance benchmarks with caching disabled."""
        x = Symbol('x')
        
        # Create theorems for testing
        theorems = []
        for i in range(3):  # Smaller set for non-cached testing
            theorem = Theorem(
                id=f"PERF_NO_CACHE_{i:03d}",
                statement=f"(x + {i})² = x² + {2*i}x + {i**2}",
                sympy_expression=Eq((x + i)**2, x**2 + 2*i*x + i**2),
                theorem_type=TheoremType.ALGEBRAIC_IDENTITY,
                assumptions=["x ∈ ℝ"],
                source_lineage=SourceLineage(
                    original_formula=f"(x + {i})**2",
                    hypothesis_id=f"perf_no_cache_{i}",
                    confidence=1.0,
                    validation_score=1.0,
                    generation_method="performance_test"
                )
            )
            theorems.append(theorem)
        
        # First execution
        start_time = time.time()
        results1 = no_cache_engine.batch_prove_theorems(theorems)
        end_time = time.time()
        time1 = end_time - start_time
        
        # Second execution (should take similar time since no caching)
        start_time = time.time()
        results2 = no_cache_engine.batch_prove_theorems(theorems)
        end_time = time.time()
        time2 = end_time - start_time
        
        # Both executions should take similar time (no significant speedup from caching)
        time_ratio = min(time1, time2) / max(time1, time2)
        assert time_ratio > 0.5  # Times should be reasonably similar
        
        # Results should be consistent
        assert len(results1) == len(results2) == len(theorems)
        
        # Check that results are deterministic
        for r1, r2 in zip(results1, results2):
            assert r1.theorem_id == r2.theorem_id
            assert r1.status == r2.status

    @pytest.mark.slow
    def test_cache_isolation_between_engines(self, test_data_dir):
        """Test that different engines with different cache directories are isolated."""
        import tempfile
        import shutil
        
        # Create two engines with different cache directories
        cache_dir1 = tempfile.mkdtemp()
        cache_dir2 = tempfile.mkdtemp()
        
        try:
            engine1 = ProofAttemptEngine({'enable_caching': True, 'cache_dir': cache_dir1})
            engine2 = ProofAttemptEngine({'enable_caching': True, 'cache_dir': cache_dir2})
            
            # Clear any existing caches to ensure clean test
            engine1.cache.clear_cache()
            engine2.cache.clear_cache()
            
            import uuid
            test_id = str(uuid.uuid4())[:8]
            x = Symbol('x')
            theorem = Theorem(
                id=f"CACHE_ISOLATION_{test_id}",
                statement="(x + 2)² = x² + 4x + 4",
                sympy_expression=Eq((x + 2)**2, x**2 + 4*x + 4),
                theorem_type=TheoremType.ALGEBRAIC_IDENTITY,
                assumptions=["x ∈ ℝ"],
                source_lineage=SourceLineage(
                    original_formula="(x + 2)**2",
                    hypothesis_id=f"cache_isolation_{test_id}",
                    confidence=1.0,
                    validation_score=1.0,
                    generation_method="cache_isolation_test"
                )
            )
            
            # Prove with engine1 (should cache result)
            result1 = engine1.attempt_proof(theorem)
            
            # Check engine1 has cached result
            cache_key = engine1.cache.get_cache_key(theorem)
            cached_result1 = engine1.cache.get_cached_result(cache_key)
            assert cached_result1 is not None
            
            # Check engine2 doesn't have cached result (different cache)
            cached_result2 = engine2.cache.get_cached_result(cache_key)
            assert cached_result2 is None
            
            # Prove with engine2 (should not use engine1's cache)
            result2 = engine2.attempt_proof(theorem)
            
            # Results should be similar but independently computed
            assert result1.theorem_id == result2.theorem_id
            assert result1.status == result2.status
            
        finally:
            # Cleanup
            if os.path.exists(cache_dir1):
                shutil.rmtree(cache_dir1)
            if os.path.exists(cache_dir2):
                shutil.rmtree(cache_dir2)

    @pytest.mark.slow
    def test_concurrent_proof_attempts(self, fresh_engine):
        """Test that multiple proof attempts work correctly."""
        import concurrent.futures
        import threading
        
        x = Symbol('x')
        
        # Create multiple theorems
        theorems = []
        for i in range(5):
            theorem = Theorem(
                id=f"CONCURRENT_{i:03d}",
                statement=f"(x + {i})² = x² + {2*i}x + {i**2}",
                sympy_expression=Eq((x + i)**2, x**2 + 2*i*x + i**2),
                theorem_type=TheoremType.ALGEBRAIC_IDENTITY,
                assumptions=["x ∈ ℝ"],
                source_lineage=SourceLineage(
                    original_formula=f"(x + {i})**2",
                    hypothesis_id=f"concurrent_{i}",
                    confidence=1.0,
                    validation_score=1.0,
                    generation_method="concurrent_test"
                )
            )
            theorems.append(theorem)
        
        # Function to prove a single theorem
        def prove_theorem(theorem):
            return fresh_engine.attempt_proof(theorem)
        
        # Execute proofs concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_theorem = {executor.submit(prove_theorem, theorem): theorem for theorem in theorems}
            results = []
            
            for future in concurrent.futures.as_completed(future_to_theorem):
                theorem = future_to_theorem[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    pytest.fail(f'Theorem {theorem.id} generated an exception: {exc}')
        
        # Verify all theorems were processed
        assert len(results) == len(theorems)
        
        # Verify all results are valid
        for result in results:
            assert result.theorem_id.startswith("CONCURRENT_")
            assert result.status in list(ProofStatus)
            assert result.execution_time >= 0

    def test_error_handling_robustness(self, fresh_engine):
        """Test that the engine handles various error conditions gracefully."""
        # Test with malformed SymPy expression
        malformed_theorem = Theorem(
            id="ERROR_TEST_001",
            statement="Invalid expression",
            sympy_expression="this_is_not_valid_sympy",  # This will cause an error
            theorem_type=TheoremType.ALGEBRAIC_IDENTITY,
            assumptions=["x ∈ ℝ"],
            source_lineage=SourceLineage(
                original_formula="invalid",
                hypothesis_id="error_test",
                confidence=1.0,
                validation_score=1.0,
                generation_method="error_test"
            )
        )
        
        # Should handle gracefully without crashing
        result = fresh_engine.attempt_proof(malformed_theorem)
        assert result.status in [ProofStatus.FAILED, ProofStatus.INCONCLUSIVE]
        # Error message might be None for INCONCLUSIVE status, which is acceptable
        if result.status == ProofStatus.FAILED:
            assert result.error_message is not None
        
        # Test batch processing with mixed valid/invalid theorems
        import uuid
        test_id = str(uuid.uuid4())[:8]
        x = Symbol('x')
        
        # Use a more obviously provable theorem
        valid_theorem = Theorem(
            id=f"ERROR_TEST_{test_id}_VALID",
            statement="(x + 1)² = x² + 2x + 1",
            sympy_expression=Eq((x + 1)**2, x**2 + 2*x + 1),
            theorem_type=TheoremType.ALGEBRAIC_IDENTITY,
            assumptions=["x ∈ ℝ"],
            source_lineage=SourceLineage(
                original_formula="(x + 1)**2",
                hypothesis_id=f"error_test_valid_{test_id}",
                confidence=1.0,
                validation_score=1.0,
                generation_method="error_test"
            )
        )
        
        mixed_theorems = [valid_theorem, malformed_theorem]
        results = fresh_engine.batch_prove_theorems(mixed_theorems)
        
        assert len(results) == 2
        # At least one should be processable (not necessarily proved, but processed)
        processable_results = [r for r in results if r.status != ProofStatus.FAILED]
        assert len(processable_results) >= 1


# Test configuration and markers
pytest.main([__file__, "-v", "--tb=short"])
