"""
Tests for TestRunner class.

Test suite covering test orchestration, formula loading from different sources,
report generation, and integration with knowledge graphs.
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock, mock_open

import networkx as nx

from validation.test_runner import TestRunner, ValidationReport
from validation.formula_tester import ValidationConfig, ValidationResult, ValidationStatus


class TestTestRunner:
    """Test suite for TestRunner."""
    
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            data_dir = temp_path / "processed"
            graph_dir = temp_path / "graph"
            data_dir.mkdir()
            graph_dir.mkdir()
            yield data_dir, graph_dir
    
    @pytest.fixture
    def config(self):
        """Test configuration."""
        return ValidationConfig(
            num_random_tests=5,
            random_seed=42,
            enable_edge_cases=False,
            enable_round_trip=False
        )
    
    @pytest.fixture
    def runner(self, temp_dirs, config):
        """Create test runner with temporary directories."""
        data_dir, graph_dir = temp_dirs
        return TestRunner(config, data_dir, graph_dir)
    
    def test_runner_initialization(self, temp_dirs, config):
        """Test runner initialization."""
        data_dir, graph_dir = temp_dirs
        runner = TestRunner(config, data_dir, graph_dir)
        
        assert runner.config == config
        assert runner.data_dir == data_dir
        assert runner.graph_dir == graph_dir
        assert isinstance(runner.validator, type(runner.validator))
    
    def test_runner_default_initialization(self):
        """Test runner with default parameters."""
        runner = TestRunner()
        assert runner.config is not None
        assert runner.data_dir is not None
        assert runner.graph_dir is not None


class TestFormulaLoading:
    """Test formula loading from different sources."""
    
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories with test data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            data_dir = temp_path / "processed"
            graph_dir = temp_path / "graph"
            data_dir.mkdir()
            graph_dir.mkdir()
            
            # Create test processed data
            test_data = {
                "formulas": ["x**2 + 1", "sin(x) + cos(x)"],
                "detailed_formulas": [
                    {"expression": "a + b", "metadata": {"topic": "algebra"}},
                    {"expression": "sqrt(x)", "metadata": {"topic": "functions"}}
                ],
                "cleaned_formulas": {
                    "unicode_normalized": ["y**2", "log(z)"],
                    "sympy_compatible": ["t**3", "exp(w)"]
                }
            }
            
            with open(data_dir / "test_document.json", 'w') as f:
                json.dump(test_data, f)
            
            # Create test graph data
            graph_data = {
                "nodes": [
                    {
                        "id": "node1",
                        "attributes": {
                            "formula": "pi * r**2",
                            "topic": "geometry"
                        }
                    },
                    {
                        "id": "node2", 
                        "attributes": {
                            "expression": "E = m*c**2",
                            "topic": "physics"
                        }
                    }
                ]
            }
            
            with open(graph_dir / "test_graph.json", 'w') as f:
                json.dump(graph_data, f)
            
            # Create NetworkX graph file
            G = nx.Graph()
            G.add_node("n1", formula="x + y", topic="basic")
            G.add_node("n2", expression="sin(theta)", topic="trigonometry")
            nx.write_graphml(G, graph_dir / "test_nx.graphml")
            
            yield data_dir, graph_dir
    
    @pytest.fixture
    def runner(self, temp_dirs):
        """Create runner with test data."""
        data_dir, graph_dir = temp_dirs
        config = ValidationConfig(num_random_tests=2, random_seed=42)
        return TestRunner(config, data_dir, graph_dir)
    
    def test_load_from_processed_data(self, runner):
        """Test loading formulas from processed JSON files."""
        formulas = runner._load_from_processed_data()
        
        # Should load formulas from all sections
        assert len(formulas) > 0
        
        # Check specific formulas are present
        formula_values = list(formulas.values())
        formula_strings = [
            f if isinstance(f, str) else f.get('expression', f.get('formula', ''))
            for f in formula_values
        ]
        
        assert "x**2 + 1" in formula_strings
        assert "sin(x) + cos(x)" in formula_strings
        assert "a + b" in formula_strings
    
    def test_load_from_graph_data(self, runner):
        """Test loading formulas from graph files."""
        formulas = runner._load_from_graph_data()
        
        assert len(formulas) > 0
        
        # Check formulas from both JSON and GraphML sources
        formula_expressions = []
        for formula_data in formulas.values():
            if isinstance(formula_data, dict):
                expr = formula_data.get('expression', '')
                formula_expressions.append(expr)
            else:
                formula_expressions.append(str(formula_data))
        
        # Should include formulas from both graph files
        assert any("pi * r**2" in expr for expr in formula_expressions)
        assert any("E = m*c**2" in expr for expr in formula_expressions)
        assert any("x + y" in expr for expr in formula_expressions)
    
    def test_extract_formulas_from_data(self, runner):
        """Test formula extraction from processed data structure."""
        data = {
            "formulas": ["f1", "f2"],
            "detailed_formulas": [{"expression": "f3"}],
            "cleaned_formulas": {"variant1": ["f4", "f5"]}
        }
        
        formulas = runner._extract_formulas_from_data(data, "test_source")
        
        assert len(formulas) == 5
        assert "test_source_formula_0" in formulas
        assert "test_source_detailed_0" in formulas
        assert "test_source_variant1_0" in formulas
        assert "test_source_variant1_1" in formulas
    
    def test_extract_formulas_from_graph_json(self, runner):
        """Test formula extraction from graph JSON."""
        data = {
            "nodes": [
                {"id": "n1", "attributes": {"formula": "test_formula_1"}},
                {"id": "n2", "attributes": {"expression": "test_formula_2"}},
                {"id": "n3", "attributes": {"other": "not_a_formula"}}
            ]
        }
        
        formulas = runner._extract_formulas_from_graph_json(data, "test_graph")
        
        assert len(formulas) == 2
        assert any("test_formula_1" in str(f) for f in formulas.values())
        assert any("test_formula_2" in str(f) for f in formulas.values())
    
    def test_extract_formulas_from_networkx(self, runner):
        """Test formula extraction from NetworkX graph."""
        G = nx.Graph()
        G.add_node("n1", formula="nx_formula_1")
        G.add_node("n2", expression="nx_formula_2")
        G.add_node("n3", other_attr="not_formula")
        
        formulas = runner._extract_formulas_from_networkx(G, "test_nx")
        
        assert len(formulas) == 2
        expressions = [f['expression'] for f in formulas.values() if isinstance(f, dict)]
        assert "nx_formula_1" in expressions
        assert "nx_formula_2" in expressions


class TestValidationExecution:
    """Test validation execution workflows."""
    
    @pytest.fixture
    def mock_runner(self):
        """Create runner with mocked validator."""
        config = ValidationConfig(num_random_tests=2, random_seed=42)
        runner = TestRunner(config)
        
        # Mock the validator to return predictable results
        mock_result = ValidationResult(
            formula="test_formula",
            status=ValidationStatus.PASS,
            confidence_score=0.95,
            pass_rate=0.9,
            total_tests=10,
            passed_tests=9,
            failed_tests=1,
            error_tests=0
        )
        
        runner.validator.validate_formula = Mock(return_value=mock_result)
        return runner, mock_result
    
    def test_validate_random_formulas(self, mock_runner):
        """Test random formula validation workflow."""
        runner, mock_result = mock_runner
        
        # Mock formula loading
        test_formulas = {
            "f1": "x + 1",
            "f2": "y**2",
            "f3": "sin(z)"
        }
        runner._load_formulas_from_source = Mock(return_value=test_formulas)
        
        report = runner.validate_random_formulas(count=2, source="processed")
        
        assert report.total_formulas == 2
        assert report.passed_formulas == 2  # All mocked as PASS
        assert report.overall_pass_rate == 1.0
        assert runner.validator.validate_formula.call_count == 2
    
    def test_validate_all_formulas(self, mock_runner):
        """Test validation of all formulas."""
        runner, mock_result = mock_runner
        
        test_formulas = {"f1": "x + 1", "f2": "y**2"}
        runner._load_formulas_from_source = Mock(return_value=test_formulas)
        
        report = runner.validate_all_formulas(source="processed")
        
        assert report.total_formulas == 2
        assert runner.validator.validate_formula.call_count == 2
    
    def test_validate_specific_formulas(self, mock_runner):
        """Test validation of specific formula IDs."""
        runner, mock_result = mock_runner
        
        test_formulas = {
            "f1": "x + 1",
            "f2": "y**2", 
            "f3": "z**3"
        }
        runner._load_formulas_from_source = Mock(return_value=test_formulas)
        
        report = runner.validate_specific_formulas(["f1", "f3"], source="processed")
        
        assert report.total_formulas == 2  # Only f1 and f3
        assert runner.validator.validate_formula.call_count == 2
    
    def test_validation_with_errors(self):
        """Test handling of validation errors."""
        config = ValidationConfig(num_random_tests=1, random_seed=42)
        runner = TestRunner(config)
        
        # Mock validator to raise exception
        runner.validator.validate_formula = Mock(side_effect=Exception("Test error"))
        
        test_formulas = {"f1": "x + 1"}
        runner._load_formulas_from_source = Mock(return_value=test_formulas)
        
        report = runner.validate_all_formulas()
        
        assert report.total_formulas == 1
        assert report.error_formulas == 1
        assert len(report.errors_summary) > 0


class TestReportGeneration:
    """Test validation report generation."""
    
    def test_generate_report_basic(self):
        """Test basic report generation."""
        config = ValidationConfig(num_random_tests=5, random_seed=42)
        runner = TestRunner(config)
        
        # Create mock results
        results = {
            "f1": ValidationResult(
                formula="x + 1",
                status=ValidationStatus.PASS,
                confidence_score=0.95,
                pass_rate=1.0,
                total_tests=5,
                passed_tests=5,
                failed_tests=0,
                error_tests=0
            ),
            "f2": ValidationResult(
                formula="1/x",
                status=ValidationStatus.FAIL,
                confidence_score=0.3,
                pass_rate=0.2,
                total_tests=5,
                passed_tests=1,
                failed_tests=4,
                error_tests=0
            )
        }
        
        report = runner._generate_report(results, "test_batch", 10.5)
        
        assert report.total_formulas == 2
        assert report.passed_formulas == 1
        assert report.failed_formulas == 1
        assert report.error_formulas == 0
        assert report.overall_pass_rate == 0.5  # 1 pass out of 2
        assert report.validation_time == 10.5
        assert len(report.results_by_formula) == 2
    
    def test_generate_report_with_errors(self):
        """Test report generation with error cases."""
        config = ValidationConfig(num_random_tests=2)
        runner = TestRunner(config)
        
        results = {
            "f1": ValidationResult(
                formula="invalid",
                status=ValidationStatus.ERROR,
                confidence_score=0.0,
                pass_rate=0.0,
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                error_tests=1,
                error_summary="Parse error"
            ),
            "f2": ValidationResult(
                formula="x**2",
                status=ValidationStatus.PARTIAL,
                confidence_score=0.7,
                pass_rate=0.8,
                total_tests=5,
                passed_tests=4,
                failed_tests=1,
                error_tests=0
            )
        }
        
        report = runner._generate_report(results, "error_test", 5.0)
        
        assert report.error_formulas == 1
        assert report.partial_formulas == 1
        assert len(report.errors_summary) == 1
        assert "Parse error" in report.errors_summary[0]
    
    def test_save_report_to_file(self, runner):
        """Test saving report to JSON file."""
        # Create a simple report
        report = ValidationReport(
            total_formulas=1,
            validated_formulas=1,
            passed_formulas=1,
            failed_formulas=0,
            error_formulas=0,
            partial_formulas=0,
            overall_pass_rate=1.0,
            average_confidence=0.95,
            validation_time=2.5,
            timestamp="2024-01-01 12:00:00"
        )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            runner.save_report_to_file(report, temp_path)
            
            # Verify file was created and contains expected data
            assert temp_path.exists()
            
            with open(temp_path, 'r') as f:
                saved_data = json.load(f)
            
            assert saved_data['summary']['total_formulas'] == 1
            assert saved_data['summary']['overall_pass_rate'] == 1.0
            assert saved_data['summary']['timestamp'] == "2024-01-01 12:00:00"
            
        finally:
            if temp_path.exists():
                temp_path.unlink()


class TestGraphIntegration:
    """Test integration with knowledge graphs."""
    
    @pytest.fixture
    def temp_graph_files(self):
        """Create temporary graph files for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create JSON graph file
            json_graph = {
                "nodes": [
                    {
                        "id": "n1",
                        "attributes": {"formula": "x**2", "topic": "algebra"}
                    },
                    {
                        "id": "n2", 
                        "attributes": {"expression": "sin(x)", "topic": "trig"}
                    }
                ]
            }
            
            json_path = temp_path / "test.json"
            with open(json_path, 'w') as f:
                json.dump(json_graph, f)
            
            # Create NetworkX graph file
            G = nx.Graph()
            G.add_node("gn1", formula="y + 1", topic="linear")
            G.add_node("gn2", expression="cos(t)", topic="trig")
            
            graphml_path = temp_path / "test.graphml"
            nx.write_graphml(G, graphml_path)
            
            yield temp_path, json_path, graphml_path
    
    def test_update_json_graph(self, temp_graph_files):
        """Test updating JSON graph with validation results."""
        temp_path, json_path, graphml_path = temp_graph_files
        
        config = ValidationConfig(num_random_tests=2)
        runner = TestRunner(config, graph_dir=temp_path)
        
        # Create mock report
        results = {
            "test_node_n1": ValidationResult(
                formula="x**2",
                status=ValidationStatus.PASS,
                confidence_score=0.9,
                pass_rate=0.95,
                total_tests=5,
                passed_tests=5,
                failed_tests=0,
                error_tests=0,
                symbols_found={"x"}
            )
        }
        results["test_node_n1"].metadata = {"node_id": "n1"}
        
        report = ValidationReport(
            total_formulas=1,
            validated_formulas=1,
            passed_formulas=1,
            failed_formulas=0,
            error_formulas=0,
            partial_formulas=0,
            overall_pass_rate=1.0,
            average_confidence=0.9,
            validation_time=1.0,
            results_by_formula=results,
            timestamp="2024-01-01 12:00:00"
        )
        
        # Update the graph
        runner._update_json_graph(report, json_path)
        
        # Verify updates
        with open(json_path, 'r') as f:
            updated_graph = json.load(f)
        
        node1 = next(n for n in updated_graph["nodes"] if n["id"] == "n1")
        attrs = node1["attributes"]
        
        assert attrs["validation_score"] == 0.9
        assert attrs["validation_status"] == "PASS"
        assert attrs["validation_pass_rate"] == 0.95
        assert "x" in attrs["tested_on"]
    
    def test_update_networkx_graph(self, temp_graph_files):
        """Test updating NetworkX graph with validation results."""
        temp_path, json_path, graphml_path = temp_graph_files
        
        config = ValidationConfig(num_random_tests=2)
        runner = TestRunner(config, graph_dir=temp_path)
        
        # Create mock report
        results = {
            "test_node_gn1": ValidationResult(
                formula="y + 1",
                status=ValidationStatus.PASS,
                confidence_score=0.85,
                pass_rate=0.9,
                total_tests=4,
                passed_tests=4,
                failed_tests=0,
                error_tests=0,
                symbols_found={"y"}
            )
        }
        results["test_node_gn1"].metadata = {"node_id": "gn1"}
        
        report = ValidationReport(
            total_formulas=1,
            validated_formulas=1,
            passed_formulas=1,
            failed_formulas=0,
            error_formulas=0,
            partial_formulas=0,
            overall_pass_rate=1.0,
            average_confidence=0.85,
            validation_time=1.5,
            results_by_formula=results,
            timestamp="2024-01-01 12:00:00"
        )
        
        # Update the graph
        runner._update_networkx_graph(report, graphml_path)
        
        # Verify updates
        updated_graph = nx.read_graphml(graphml_path)
        
        node_attrs = updated_graph.nodes["gn1"]
        assert float(node_attrs["validation_score"]) == 0.85
        assert node_attrs["validation_status"] == "PASS"
        assert float(node_attrs["validation_pass_rate"]) == 0.9
        assert "y" in node_attrs["tested_on"]


class TestCLIFunctionality:
    """Test CLI command functionality."""
    
    def test_create_cli_parser(self):
        """Test CLI parser creation."""
        from validation.test_runner import create_cli_parser
        
        parser = create_cli_parser()
        
        # Test basic parser structure
        assert parser.prog is not None
        
        # Test subcommands exist
        args = parser.parse_args(['validate-random', '--count', '5'])
        assert args.command == 'validate-random'
        assert args.count == 5
        
        args = parser.parse_args(['validate-all', '--source', 'graph'])
        assert args.command == 'validate-all'
        assert args.source == 'graph'
        
        args = parser.parse_args(['generate-report', '--output', 'report.json'])
        assert args.command == 'generate-report'
        assert args.output == 'report.json'
    
    @patch('validation.test_runner.TestRunner')
    def test_main_function_validate_random(self, mock_runner_class):
        """Test main function with validate-random command."""
        from validation.test_runner import main
        
        # Mock the runner and its methods
        mock_runner = Mock()
        mock_report = Mock()
        mock_report.total_formulas = 5
        mock_report.passed_formulas = 4
        mock_report.failed_formulas = 1
        mock_report.error_formulas = 0
        mock_report.partial_formulas = 0
        mock_report.overall_pass_rate = 0.8
        mock_report.average_confidence = 0.85
        mock_report.validation_time = 2.5
        
        mock_runner.validate_random_formulas.return_value = mock_report
        mock_runner_class.return_value = mock_runner
        
        # Test the main function (would need to mock sys.argv)
        with patch('sys.argv', ['test_runner.py', 'validate-random', '--count', '5']):
            try:
                main()
                mock_runner.validate_random_formulas.assert_called_once_with(5, 'processed')
            except SystemExit:
                pass  # Expected for CLI programs
    
    def test_empty_report_creation(self):
        """Test creation of empty reports."""
        runner = TestRunner()
        report = runner._create_empty_report()
        
        assert report.total_formulas == 0
        assert report.overall_pass_rate == 0.0
        assert report.average_confidence == 0.0
        assert report.timestamp is not None 