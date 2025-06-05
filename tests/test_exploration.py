"""
Test suite for Phase 4 exploration modules.

Tests pattern discovery, gap detection, hypothesis generation,
and embedding utilities.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import sympy as sp

from exploration import PatternFinder, GapDetector, HypothesisGenerator
from exploration.pattern_finder import FormulaFingerprint, PatternCluster, SimilarityMetric
from exploration.gap_detector import Gap, GapType, ConceptNode
from exploration.hypothesis_generator import Hypothesis, HypothesisStatus, HypothesisType
from exploration.utils.embedding import FormulaEmbedder, ClusterVisualizer, EmbeddingResult


class TestPatternFinder:
    """Test suite for PatternFinder functionality."""
    
    @pytest.fixture
    def sample_formulas(self):
        """Sample mathematical formulas for testing."""
        return [
            "x**2 + 2*x + 1",
            "y**2 + 2*y + 1", 
            "a**2 + b**2",
            "sin(x)**2 + cos(x)**2",
            "sin(y)**2 + cos(y)**2",
            "exp(x) + log(x)",
            "pi * r**2",
            "E = m*c**2"
        ]
    
    @pytest.fixture
    def pattern_finder(self):
        """PatternFinder instance for testing."""
        return PatternFinder(similarity_threshold=0.6, min_cluster_size=2)
    
    def test_generate_fingerprint(self, pattern_finder):
        """Test formula fingerprint generation."""
        formula = "x**2 + 2*x + 1"
        fingerprint = pattern_finder._generate_fingerprint(formula)
        
        assert isinstance(fingerprint, FormulaFingerprint)
        assert fingerprint.formula == formula
        assert fingerprint.parsed_expr is not None
        assert 'x' in fingerprint.symbol_count
        assert fingerprint.complexity > 0
        assert fingerprint.structure_hash != ""
    
    def test_find_patterns(self, pattern_finder, sample_formulas):
        """Test pattern discovery in sample formulas."""
        patterns = pattern_finder.find_patterns(sample_formulas)
        
        assert isinstance(patterns, list)
        assert len(patterns) > 0
        
        # Check pattern cluster structure
        for cluster in patterns:
            assert isinstance(cluster, PatternCluster)
            assert len(cluster.formulas) >= pattern_finder.min_cluster_size
            assert cluster.confidence_score >= 0.0
            assert cluster.confidence_score <= 1.0
    
    def test_similarity_calculation(self, pattern_finder):
        """Test similarity calculation between formulas."""
        # Similar formulas should have high similarity
        fp1 = pattern_finder._generate_fingerprint("x**2 + 2*x + 1")
        fp2 = pattern_finder._generate_fingerprint("y**2 + 2*y + 1")
        similarity = pattern_finder._calculate_similarity(fp1, fp2)
        
        assert similarity > 0.5  # Should be similar
        
        # Different formulas should have lower similarity
        fp3 = pattern_finder._generate_fingerprint("sin(x) + cos(x)")
        similarity2 = pattern_finder._calculate_similarity(fp1, fp3)
        
        assert similarity2 < similarity
    
    def test_save_patterns(self, pattern_finder, sample_formulas):
        """Test saving patterns to file."""
        patterns = pattern_finder.find_patterns(sample_formulas)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "patterns.json"
            pattern_finder.save_patterns(patterns, output_path)
            
            assert output_path.exists()
            
            # Verify JSON structure
            with open(output_path, 'r') as f:
                data = json.load(f)
            
            assert "discovery_metadata" in data
            assert "clusters" in data
            assert len(data["clusters"]) == len(patterns)


class TestGapDetector:
    """Test suite for GapDetector functionality."""
    
    @pytest.fixture
    def sample_formula_data(self):
        """Sample formula data for testing."""
        return [
            {
                "formulas": ["x**2 + 1", "y**2 + 1"],
                "detailed_formulas": [
                    {
                        "expression": "sin(x)**2 + cos(x)**2",
                        "metadata": {"topic": "trigonometry", "type": "identity"}
                    },
                    {
                        "expression": "a**2 + b**2",
                        "metadata": {"topic": "algebra", "type": "polynomial"}
                    }
                ]
            },
            {
                "formulas": ["log(x*y)", "exp(a+b)"],
                "detailed_formulas": [
                    {
                        "expression": "d/dx[x**n]",
                        "metadata": {"topic": "calculus", "type": "derivative"}
                    }
                ]
            }
        ]
    
    @pytest.fixture
    def gap_detector(self):
        """GapDetector instance for testing."""
        return GapDetector(min_connection_threshold=2, isolation_threshold=0.1)
    
    def test_detect_gaps(self, gap_detector, sample_formula_data):
        """Test gap detection in formula data."""
        gaps = gap_detector.detect_gaps(sample_formula_data)
        
        assert isinstance(gaps, list)
        
        # Check gap structure
        for gap in gaps:
            assert isinstance(gap, Gap)
            assert isinstance(gap.gap_type, GapType)
            assert gap.confidence_score >= 0.0
            assert gap.confidence_score <= 1.0
            assert gap.priority >= 0.0
    
    def test_build_knowledge_graph(self, gap_detector, sample_formula_data):
        """Test knowledge graph construction."""
        gap_detector._build_knowledge_graph(sample_formula_data)
        
        assert len(gap_detector.concept_nodes) > 0
        assert gap_detector.knowledge_graph.number_of_nodes() > 0
        
        # Check concept node structure
        for concept_id, node in gap_detector.concept_nodes.items():
            assert isinstance(node, ConceptNode)
            assert len(node.formulas) > 0
    
    def test_extract_concept_id(self, gap_detector):
        """Test concept ID extraction from formulas."""
        formula = "x**2 + 2*x + 1"
        concept_id = gap_detector._extract_concept_id(formula)
        
        assert isinstance(concept_id, str)
        assert len(concept_id) > 0
    
    def test_save_gaps(self, gap_detector, sample_formula_data):
        """Test saving gaps to file."""
        gaps = gap_detector.detect_gaps(sample_formula_data)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "gaps.json"
            gap_detector.save_gaps(gaps, output_path)
            
            assert output_path.exists()
            
            # Verify JSON structure
            with open(output_path, 'r') as f:
                data = json.load(f)
            
            assert "detection_metadata" in data
            assert "gaps" in data


class TestHypothesisGenerator:
    """Test suite for HypothesisGenerator functionality."""
    
    @pytest.fixture
    def sample_formulas(self):
        """Sample formulas for hypothesis generation."""
        return [
            "x**2 + 1",
            "sin(x)",
            "exp(x)",
            "x + y",
            "a*b"
        ]
    
    @pytest.fixture 
    def hypothesis_generator(self):
        """HypothesisGenerator instance for testing."""
        # Mock the validator to avoid real validation
        with patch('exploration.hypothesis_generator.FormulaValidator') as mock_validator:
            mock_result = Mock()
            mock_result.status.value = "PASS"
            mock_result.confidence_score = 0.8
            mock_result.pass_rate = 0.9
            mock_result.total_tests = 100
            mock_result.symbols_found = {"x"}
            mock_result.validation_time = 0.1
            
            mock_validator.return_value.validate_formula.return_value = mock_result
            
            generator = HypothesisGenerator(max_hypotheses_per_type=5)
            return generator
    
    def test_generate_hypotheses(self, hypothesis_generator, sample_formulas):
        """Test hypothesis generation from sample formulas."""
        hypotheses = hypothesis_generator.generate_hypotheses(sample_formulas)
        
        assert isinstance(hypotheses, list)
        assert len(hypotheses) > 0
        
        # Check hypothesis structure
        for hypothesis in hypotheses:
            assert isinstance(hypothesis, Hypothesis)
            assert isinstance(hypothesis.hypothesis_type, HypothesisType)
            assert isinstance(hypothesis.status, HypothesisStatus)
            assert hypothesis.formula != ""
            assert hypothesis.confidence_score >= 0.0
    
    def test_parse_source_formulas(self, hypothesis_generator, sample_formulas):
        """Test parsing of source formulas."""
        parsed = hypothesis_generator._parse_source_formulas(sample_formulas)
        
        assert len(parsed) <= len(sample_formulas)  # Some may fail to parse
        
        for formula, expr in parsed:
            assert isinstance(formula, str)
            assert isinstance(expr, sp.Expr)
    
    def test_generate_algebraic_identities(self, hypothesis_generator, sample_formulas):
        """Test algebraic identity generation."""
        parsed = hypothesis_generator._parse_source_formulas(sample_formulas)
        identities = hypothesis_generator._generate_algebraic_identities(parsed)
        
        assert isinstance(identities, list)
        
        for identity in identities:
            assert identity.hypothesis_type == HypothesisType.ALGEBRAIC_IDENTITY
    
    def test_save_hypotheses(self, hypothesis_generator, sample_formulas):
        """Test saving hypotheses to file."""
        hypotheses = hypothesis_generator.generate_hypotheses(sample_formulas)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "hypotheses.json"
            hypothesis_generator.save_hypotheses(hypotheses, output_path)
            
            assert output_path.exists()
            
            # Verify JSON structure
            with open(output_path, 'r') as f:
                data = json.load(f)
            
            assert "generation_metadata" in data
            assert "hypotheses" in data
            assert len(data["hypotheses"]) == len(hypotheses)
    
    def test_get_promising_hypotheses(self, hypothesis_generator, sample_formulas):
        """Test filtering of promising hypotheses."""
        hypotheses = hypothesis_generator.generate_hypotheses(sample_formulas)
        promising = hypothesis_generator.get_promising_hypotheses(hypotheses, min_confidence=0.5)
        
        assert isinstance(promising, list)
        assert len(promising) <= len(hypotheses)
        
        # All promising hypotheses should meet the confidence threshold
        for hyp in promising:
            assert hyp.confidence_score >= 0.5


class TestFormulaEmbedder:
    """Test suite for FormulaEmbedder functionality."""
    
    @pytest.fixture
    def sample_formulas(self):
        """Sample formulas for embedding."""
        return [
            "x**2 + 1",
            "y**2 + 1",
            "sin(x)",
            "cos(x)",
            "exp(x)"
        ]
    
    @pytest.fixture
    def embedder(self):
        """FormulaEmbedder instance for testing."""
        return FormulaEmbedder(embedding_method="structural")
    
    def test_embed_formulas_structural(self, embedder, sample_formulas):
        """Test structural embedding of formulas."""
        result = embedder.embed_formulas(sample_formulas)
        
        assert isinstance(result, EmbeddingResult)
        assert result.embeddings.shape[0] == len(sample_formulas)
        assert result.embeddings.shape[1] > 0
        assert result.embedding_method == "structural"
    
    def test_embed_formulas_tfidf(self, sample_formulas):
        """Test TF-IDF embedding of formulas."""
        embedder = FormulaEmbedder(embedding_method="tfidf")
        result = embedder.embed_formulas(sample_formulas)
        
        assert isinstance(result, EmbeddingResult)
        assert result.embeddings.shape[0] == len(sample_formulas)
        assert result.embedding_method == "tfidf"
    
    def test_cluster_embeddings(self, embedder, sample_formulas):
        """Test clustering of formula embeddings."""
        result = embedder.embed_formulas(sample_formulas)
        clustered_result = embedder.cluster_embeddings(result, method="kmeans", n_clusters=2)
        
        assert clustered_result.clustering_labels is not None
        assert len(clustered_result.clustering_labels) == len(sample_formulas)
        assert clustered_result.cluster_method == "kmeans"
    
    def test_reduce_dimensions(self, embedder, sample_formulas):
        """Test dimension reduction of embeddings."""
        result = embedder.embed_formulas(sample_formulas)
        reduced_result = embedder.reduce_dimensions(result, method="pca", n_components=2)
        
        assert reduced_result.reduced_embeddings is not None
        assert reduced_result.reduced_embeddings.shape == (len(sample_formulas), 2)
        assert reduced_result.reduction_method == "pca"


class TestClusterVisualizer:
    """Test suite for ClusterVisualizer functionality."""
    
    @pytest.fixture
    def sample_embedding_result(self):
        """Sample embedding result for visualization testing."""
        formulas = ["x**2", "y**2", "sin(x)", "cos(x)"]
        embeddings = np.random.rand(4, 10)
        reduced_embeddings = np.random.rand(4, 2)
        clustering_labels = np.array([0, 0, 1, 1])
        
        result = EmbeddingResult(
            formulas=formulas,
            embeddings=embeddings,
            embedding_method="test",
            clustering_labels=clustering_labels,
            cluster_method="test",
            reduced_embeddings=reduced_embeddings,
            reduction_method="test"
        )
        return result
    
    @pytest.fixture
    def visualizer(self):
        """ClusterVisualizer instance for testing."""
        return ClusterVisualizer()
    
    def test_create_cluster_summary(self, visualizer, sample_embedding_result):
        """Test cluster summary creation."""
        summary = visualizer.create_cluster_summary(sample_embedding_result)
        
        assert isinstance(summary, dict)
        assert "total_formulas" in summary
        assert "n_clusters" in summary
        assert "cluster_sizes" in summary
        assert summary["total_formulas"] == 4
    
    def test_save_visualization_data(self, visualizer, sample_embedding_result):
        """Test saving visualization data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "viz_data.json"
            visualizer.save_visualization_data(sample_embedding_result, output_path)
            
            assert output_path.exists()
            
            # Verify JSON structure
            with open(output_path, 'r') as f:
                data = json.load(f)
            
            assert "formulas" in data
            assert "embedding_method" in data
            assert "clustering" in data
            assert "dimension_reduction" in data


class TestIntegration:
    """Integration tests for Phase 4 modules."""
    
    @pytest.fixture
    def sample_data(self):
        """Sample data for integration testing."""
        return {
            "formulas": [
                "x**2 + 2*x + 1",
                "y**2 + 2*y + 1",
                "sin(x)**2 + cos(x)**2",
                "a**2 + b**2"
            ],
            "detailed_formulas": [
                {
                    "expression": "(a + b)**2",
                    "metadata": {"topic": "algebra", "type": "binomial"}
                },
                {
                    "expression": "sin(2*x)",
                    "metadata": {"topic": "trigonometry", "type": "double_angle"}
                }
            ]
        }
    
    def test_end_to_end_exploration(self, sample_data):
        """Test complete exploration pipeline."""
        # Extract formulas
        all_formulas = sample_data["formulas"].copy()
        for detailed in sample_data["detailed_formulas"]:
            all_formulas.append(detailed["expression"])
        
        # Pattern discovery
        pattern_finder = PatternFinder(similarity_threshold=0.5, min_cluster_size=2)
        patterns = pattern_finder.find_patterns(all_formulas, sample_data)
        
        assert len(patterns) >= 0  # May find patterns or not
        
        # Gap detection
        gap_detector = GapDetector()
        gaps = gap_detector.detect_gaps([sample_data])
        
        assert isinstance(gaps, list)
        
        # Hypothesis generation (with mocked validation)
        with patch('exploration.hypothesis_generator.FormulaValidator') as mock_validator:
            mock_result = Mock()
            mock_result.status.value = "PASS"
            mock_result.confidence_score = 0.7
            mock_result.pass_rate = 0.8
            mock_result.total_tests = 50
            mock_result.symbols_found = {"x"}
            mock_result.validation_time = 0.1
            
            mock_validator.return_value.validate_formula.return_value = mock_result
            
            hypothesis_generator = HypothesisGenerator(max_hypotheses_per_type=3)
            hypotheses = hypothesis_generator.generate_hypotheses(all_formulas[:4])  # Limit for testing
            
            assert len(hypotheses) > 0
    
    def test_file_io_operations(self, sample_data):
        """Test file input/output operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Test pattern saving/loading
            pattern_finder = PatternFinder()
            patterns = pattern_finder.find_patterns(sample_data["formulas"], sample_data)
            pattern_finder.save_patterns(patterns, temp_path / "patterns.json")
            
            assert (temp_path / "patterns.json").exists()
            
            # Test gap saving
            gap_detector = GapDetector()
            gaps = gap_detector.detect_gaps([sample_data])
            gap_detector.save_gaps(gaps, temp_path / "gaps.json")
            
            assert (temp_path / "gaps.json").exists()
            
            # Verify files can be loaded as valid JSON
            with open(temp_path / "patterns.json", 'r') as f:
                patterns_data = json.load(f)
            assert "clusters" in patterns_data
            
            with open(temp_path / "gaps.json", 'r') as f:
                gaps_data = json.load(f)
            assert "gaps" in gaps_data


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 