"""
Test orchestration and reporting for MathBot formula validation.

This module provides the TestRunner class that coordinates validation workflows,
loads formulas from various sources, and generates comprehensive reports.
"""

import json
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import argparse

import networkx as nx

from .formula_tester import FormulaValidator, ValidationResult, ValidationConfig, ValidationStatus
from config import PROCESSED_DATA_DIR, GRAPH_DATA_DIR, logger


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    total_formulas: int
    validated_formulas: int
    passed_formulas: int
    failed_formulas: int
    error_formulas: int
    partial_formulas: int
    overall_pass_rate: float
    average_confidence: float
    validation_time: float
    results_by_formula: Dict[str, ValidationResult] = field(default_factory=dict)
    results_by_status: Dict[ValidationStatus, List[str]] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)
    errors_summary: List[str] = field(default_factory=list)
    timestamp: str = ""


class TestRunner:
    """
    Orchestrates formula validation workflows.
    
    Loads formulas from processed data or knowledge graphs, runs validation tests,
    and generates comprehensive reports with integration back to the graph system.
    """
    
    def __init__(
        self,
        config: Optional[ValidationConfig] = None,
        data_dir: Optional[Path] = None,
        graph_dir: Optional[Path] = None
    ):
        """
        Initialize the test runner.
        
        Args:
            config: Validation configuration
            data_dir: Directory containing processed formula data
            graph_dir: Directory containing knowledge graph data
        """
        self.config = config or ValidationConfig()
        self.data_dir = data_dir or PROCESSED_DATA_DIR
        self.graph_dir = graph_dir or GRAPH_DATA_DIR
        self.validator = FormulaValidator(self.config)
        self.logger = logger
        
        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.graph_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("TestRunner initialized")
    
    def validate_random_formulas(
        self, 
        count: int = 10, 
        source: str = "processed"
    ) -> ValidationReport:
        """
        Validate a random selection of formulas.
        
        Args:
            count: Number of formulas to validate
            source: Data source ("processed" or "graph")
            
        Returns:
            Validation report
        """
        self.logger.info(f"Starting random validation of {count} formulas from {source}")
        
        # Load available formulas
        formulas = self._load_formulas_from_source(source)
        
        if not formulas:
            self.logger.warning(f"No formulas found in {source} data")
            return self._create_empty_report()
        
        # Select random subset
        if len(formulas) > count:
            selected_formulas = dict(random.sample(list(formulas.items()), count))
        else:
            selected_formulas = formulas
            self.logger.info(f"Only {len(formulas)} formulas available, validating all")
        
        return self._run_validation_batch(selected_formulas, f"random_{count}")
    
    def validate_all_formulas(self, source: str = "processed") -> ValidationReport:
        """
        Validate all available formulas.
        
        Args:
            source: Data source ("processed" or "graph")
            
        Returns:
            Validation report
        """
        self.logger.info(f"Starting validation of all formulas from {source}")
        
        formulas = self._load_formulas_from_source(source)
        
        if not formulas:
            self.logger.warning(f"No formulas found in {source} data")
            return self._create_empty_report()
        
        return self._run_validation_batch(formulas, "all_formulas")
    
    def validate_specific_formulas(
        self, 
        formula_ids: List[str], 
        source: str = "processed"
    ) -> ValidationReport:
        """
        Validate specific formulas by ID.
        
        Args:
            formula_ids: List of formula identifiers
            source: Data source ("processed" or "graph")
            
        Returns:
            Validation report
        """
        self.logger.info(f"Starting validation of {len(formula_ids)} specific formulas")
        
        all_formulas = self._load_formulas_from_source(source)
        selected_formulas = {
            fid: formula for fid, formula in all_formulas.items() 
            if fid in formula_ids
        }
        
        missing_ids = set(formula_ids) - set(selected_formulas.keys())
        if missing_ids:
            self.logger.warning(f"Formula IDs not found: {missing_ids}")
        
        return self._run_validation_batch(selected_formulas, "specific_formulas")
    
    def _run_validation_batch(
        self, 
        formulas: Dict[str, Union[str, Dict]], 
        batch_name: str
    ) -> ValidationReport:
        """
        Run validation on a batch of formulas.
        
        Args:
            formulas: Dictionary of formula_id -> formula_data
            batch_name: Name for this validation batch
            
        Returns:
            Validation report
        """
        start_time = time.time()
        results = {}
        
        self.logger.info(f"Validating {len(formulas)} formulas in batch '{batch_name}'")
        
        for i, (formula_id, formula_data) in enumerate(formulas.items(), 1):
            try:
                # Extract formula string and metadata
                if isinstance(formula_data, str):
                    formula_str = formula_data
                    known_identity = None
                elif isinstance(formula_data, dict):
                    formula_str = formula_data.get('expression', formula_data.get('formula', ''))
                    known_identity = formula_data.get('known_identity', formula_data.get('identity'))
                else:
                    self.logger.error(f"Invalid formula data format for {formula_id}: {type(formula_data)}")
                    continue
                
                if not formula_str:
                    self.logger.warning(f"Empty formula string for {formula_id}")
                    continue
                
                self.logger.debug(f"Validating formula {i}/{len(formulas)}: {formula_id}")
                
                # Run validation
                result = self.validator.validate_formula(formula_str, known_identity)
                result.metadata['formula_id'] = formula_id
                result.metadata['batch_name'] = batch_name
                
                results[formula_id] = result
                
                # Log progress for large batches
                if i % 10 == 0 or i == len(formulas):
                    self.logger.info(f"Progress: {i}/{len(formulas)} formulas validated")
                
            except Exception as e:
                self.logger.error(f"Error validating formula {formula_id}: {e}")
                # Create error result
                error_result = ValidationResult(
                    formula=str(formula_data),
                    status=ValidationStatus.ERROR,
                    confidence_score=0.0,
                    pass_rate=0.0,
                    total_tests=0,
                    passed_tests=0,
                    failed_tests=0,
                    error_tests=1,
                    error_summary=str(e)
                )
                error_result.metadata['formula_id'] = formula_id
                error_result.metadata['batch_name'] = batch_name
                results[formula_id] = error_result
        
        validation_time = time.time() - start_time
        
        # Generate report
        report = self._generate_report(results, batch_name, validation_time)
        
        self.logger.info(f"Batch '{batch_name}' completed: {report.passed_formulas}/{report.total_formulas} passed "
                        f"({report.overall_pass_rate:.1%}) in {validation_time:.2f}s")
        
        return report
    
    def _load_formulas_from_source(self, source: str) -> Dict[str, Union[str, Dict]]:
        """
        Load formulas from specified data source.
        
        Args:
            source: Data source ("processed" or "graph")
            
        Returns:
            Dictionary mapping formula_id to formula data
        """
        formulas = {}
        
        if source == "processed":
            formulas.update(self._load_from_processed_data())
        elif source == "graph":
            formulas.update(self._load_from_graph_data())
        else:
            self.logger.error(f"Unknown source: {source}")
        
        self.logger.info(f"Loaded {len(formulas)} formulas from {source}")
        return formulas
    
    def _load_from_processed_data(self) -> Dict[str, Union[str, Dict]]:
        """Load formulas from processed JSON files."""
        formulas = {}
        
        # Look for JSON files in processed data directory
        json_files = list(self.data_dir.glob("*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract formulas based on expected structure
                file_formulas = self._extract_formulas_from_data(data, json_file.stem)
                formulas.update(file_formulas)
                
            except Exception as e:
                self.logger.error(f"Error loading {json_file}: {e}")
        
        return formulas
    
    def _load_from_graph_data(self) -> Dict[str, Union[str, Dict]]:
        """Load formulas from knowledge graph files."""
        formulas = {}
        
        # Look for graph files (JSON or GraphML)
        graph_files = list(self.graph_dir.glob("*.json")) + list(self.graph_dir.glob("*.graphml"))
        
        for graph_file in graph_files:
            try:
                if graph_file.suffix == ".json":
                    # Load as JSON graph data
                    with open(graph_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Extract formulas from graph structure
                    file_formulas = self._extract_formulas_from_graph_json(data, graph_file.stem)
                    formulas.update(file_formulas)
                
                elif graph_file.suffix == ".graphml":
                    # Load as NetworkX graph
                    graph = nx.read_graphml(graph_file)
                    file_formulas = self._extract_formulas_from_networkx(graph, graph_file.stem)
                    formulas.update(file_formulas)
                
            except Exception as e:
                self.logger.error(f"Error loading graph {graph_file}: {e}")
        
        return formulas
    
    def _extract_formulas_from_data(self, data: Dict, source_id: str) -> Dict[str, Union[str, Dict]]:
        """Extract formulas from processed data structure."""
        formulas = {}
        
        # Handle different data structures from ingestion pipeline
        if 'formulas' in data:
            # Simple list of formulas
            for i, formula in enumerate(data['formulas']):
                formula_id = f"{source_id}_formula_{i}"
                formulas[formula_id] = formula
        
        if 'detailed_formulas' in data:
            # Detailed formula objects
            for i, formula_data in enumerate(data['detailed_formulas']):
                formula_id = f"{source_id}_detailed_{i}"
                formulas[formula_id] = formula_data
        
        if 'cleaned_formulas' in data:
            # Cleaned formula variants
            if isinstance(data['cleaned_formulas'], dict):
                for variant_type, formula_list in data['cleaned_formulas'].items():
                    for i, formula in enumerate(formula_list):
                        formula_id = f"{source_id}_{variant_type}_{i}"
                        formulas[formula_id] = formula
        
        return formulas
    
    def _extract_formulas_from_graph_json(self, data: Dict, source_id: str) -> Dict[str, Union[str, Dict]]:
        """Extract formulas from graph JSON structure."""
        formulas = {}
        
        # Handle NetworkX JSON format
        if 'nodes' in data:
            for node_data in data['nodes']:
                node_id = node_data.get('id', str(len(formulas)))
                attrs = node_data.get('attributes', {})
                
                # Look for formula-related attributes
                formula_attrs = ['formula', 'expression', 'equation', 'math_expression']
                for attr in formula_attrs:
                    if attr in attrs and attrs[attr]:
                        formula_id = f"{source_id}_node_{node_id}"
                        formulas[formula_id] = {
                            'expression': attrs[attr],
                            'node_id': node_id,
                            'metadata': attrs
                        }
                        break
        
        return formulas
    
    def _extract_formulas_from_networkx(self, graph: nx.Graph, source_id: str) -> Dict[str, Union[str, Dict]]:
        """Extract formulas from NetworkX graph."""
        formulas = {}
        
        for node_id, attrs in graph.nodes(data=True):
            # Look for formula-related attributes
            formula_attrs = ['formula', 'expression', 'equation', 'math_expression']
            for attr in formula_attrs:
                if attr in attrs and attrs[attr]:
                    formula_id = f"{source_id}_node_{node_id}"
                    formulas[formula_id] = {
                        'expression': attrs[attr],
                        'node_id': node_id,
                        'metadata': attrs
                    }
                    break
        
        return formulas
    
    def _generate_report(
        self, 
        results: Dict[str, ValidationResult], 
        batch_name: str, 
        validation_time: float
    ) -> ValidationReport:
        """Generate comprehensive validation report."""
        
        # Basic statistics
        total_formulas = len(results)
        status_counts = {status: 0 for status in ValidationStatus}
        confidence_scores = []
        pass_rates = []
        results_by_status = {status: [] for status in ValidationStatus}
        errors_summary = []
        
        for formula_id, result in results.items():
            status_counts[result.status] += 1
            results_by_status[result.status].append(formula_id)
            
            if result.confidence_score > 0:
                confidence_scores.append(result.confidence_score)
            if result.pass_rate > 0:
                pass_rates.append(result.pass_rate)
            
            if result.error_summary:
                errors_summary.append(f"{formula_id}: {result.error_summary}")
        
        # Calculate aggregate metrics
        overall_pass_rate = (
            (status_counts[ValidationStatus.PASS] + status_counts[ValidationStatus.PARTIAL]) 
            / total_formulas if total_formulas > 0 else 0
        )
        average_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        # Additional statistics
        statistics = {
            'batch_name': batch_name,
            'total_tests_run': sum(r.total_tests for r in results.values()),
            'total_passed_tests': sum(r.passed_tests for r in results.values()),
            'average_tests_per_formula': sum(r.total_tests for r in results.values()) / total_formulas if total_formulas > 0 else 0,
            'average_pass_rate': sum(pass_rates) / len(pass_rates) if pass_rates else 0,
            'validation_config': {
                'num_random_tests': self.config.num_random_tests,
                'test_range': self.config.test_range,
                'tolerance': self.config.tolerance,
                'random_seed': self.config.random_seed
            }
        }
        
        report = ValidationReport(
            total_formulas=total_formulas,
            validated_formulas=total_formulas,
            passed_formulas=status_counts[ValidationStatus.PASS],
            failed_formulas=status_counts[ValidationStatus.FAIL],
            error_formulas=status_counts[ValidationStatus.ERROR],
            partial_formulas=status_counts[ValidationStatus.PARTIAL],
            overall_pass_rate=overall_pass_rate,
            average_confidence=average_confidence,
            validation_time=validation_time,
            results_by_formula=results,
            results_by_status=results_by_status,
            statistics=statistics,
            errors_summary=errors_summary[:10],  # Limit to first 10 errors
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        return report
    
    def _create_empty_report(self) -> ValidationReport:
        """Create an empty validation report."""
        return ValidationReport(
            total_formulas=0,
            validated_formulas=0,
            passed_formulas=0,
            failed_formulas=0,
            error_formulas=0,
            partial_formulas=0,
            overall_pass_rate=0.0,
            average_confidence=0.0,
            validation_time=0.0,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
    
    def save_report_to_file(self, report: ValidationReport, output_path: Path) -> None:
        """
        Save validation report to JSON file.
        
        Args:
            report: Validation report to save
            output_path: Path for output file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert report to serializable format
        report_data = {
            'summary': {
                'total_formulas': report.total_formulas,
                'validated_formulas': report.validated_formulas,
                'passed_formulas': report.passed_formulas,
                'failed_formulas': report.failed_formulas,
                'error_formulas': report.error_formulas,
                'partial_formulas': report.partial_formulas,
                'overall_pass_rate': report.overall_pass_rate,
                'average_confidence': report.average_confidence,
                'validation_time': report.validation_time,
                'timestamp': report.timestamp
            },
            'statistics': report.statistics,
            'results_by_status': {
                status.value: formula_ids 
                for status, formula_ids in report.results_by_status.items()
            },
            'errors_summary': report.errors_summary,
            'detailed_results': {}
        }
        
        # Add detailed results (simplified for JSON serialization)
        for formula_id, result in report.results_by_formula.items():
            report_data['detailed_results'][formula_id] = {
                'formula': result.formula,
                'status': result.status.value,
                'confidence_score': result.confidence_score,
                'pass_rate': result.pass_rate,
                'total_tests': result.total_tests,
                'passed_tests': result.passed_tests,
                'failed_tests': result.failed_tests,
                'error_tests': result.error_tests,
                'validation_time': result.validation_time,
                'symbols_found': list(result.symbols_found),
                'error_summary': result.error_summary,
                'metadata': result.metadata
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Validation report saved to {output_path}")
    
    def update_graph_with_results(self, report: ValidationReport, graph_path: Optional[Path] = None) -> None:
        """
        Update knowledge graph with validation results.
        
        Args:
            report: Validation report with results
            graph_path: Path to graph file (optional)
        """
        if not graph_path:
            # Look for existing graph files
            graph_files = list(self.graph_dir.glob("*.json")) + list(self.graph_dir.glob("*.graphml"))
            if not graph_files:
                self.logger.warning("No graph files found for updating")
                return
            graph_path = graph_files[0]  # Use first available
        
        try:
            if graph_path.suffix == ".json":
                self._update_json_graph(report, graph_path)
            elif graph_path.suffix == ".graphml":
                self._update_networkx_graph(report, graph_path)
            
            self.logger.info(f"Updated graph {graph_path} with validation results")
            
        except Exception as e:
            self.logger.error(f"Failed to update graph {graph_path}: {e}")
    
    def _update_json_graph(self, report: ValidationReport, graph_path: Path) -> None:
        """Update JSON graph file with validation results."""
        with open(graph_path, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        
        # Update nodes with validation results
        if 'nodes' in graph_data:
            for node in graph_data['nodes']:
                node_id = node.get('id')
                if not node_id:
                    continue
                
                # Find matching validation results
                matching_results = [
                    (fid, result) for fid, result in report.results_by_formula.items()
                    if result.metadata.get('node_id') == str(node_id)
                ]
                
                if matching_results:
                    formula_id, result = matching_results[0]  # Use first match
                    
                    # Add validation attributes
                    if 'attributes' not in node:
                        node['attributes'] = {}
                    
                    node['attributes'].update({
                        'validation_score': result.confidence_score,
                        'validation_status': result.status.value,
                        'validation_pass_rate': result.pass_rate,
                        'tested_on': list(result.symbols_found),
                        'validation_timestamp': report.timestamp
                    })
        
        # Save updated graph
        with open(graph_path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)
    
    def _update_networkx_graph(self, report: ValidationReport, graph_path: Path) -> None:
        """Update NetworkX graph file with validation results."""
        graph = nx.read_graphml(graph_path)
        
        # Update nodes with validation results
        for node_id in graph.nodes():
            matching_results = [
                (fid, result) for fid, result in report.results_by_formula.items()
                if result.metadata.get('node_id') == str(node_id)
            ]
            
            if matching_results:
                formula_id, result = matching_results[0]  # Use first match
                
                # Add validation attributes to node
                graph.nodes[node_id].update({
                    'validation_score': result.confidence_score,
                    'validation_status': result.status.value,
                    'validation_pass_rate': result.pass_rate,
                    'tested_on': ','.join(result.symbols_found),
                    'validation_timestamp': report.timestamp
                })
        
        # Save updated graph
        nx.write_graphml(graph, graph_path)


def create_cli_parser() -> argparse.ArgumentParser:
    """Create CLI parser for validation commands."""
    parser = argparse.ArgumentParser(
        description="MathBot Formula Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Validation commands')
    
    # Validate random formulas
    random_parser = subparsers.add_parser('validate-random', help='Validate random formulas')
    random_parser.add_argument('--count', type=int, default=10, help='Number of formulas to validate')
    random_parser.add_argument('--source', choices=['processed', 'graph'], default='processed', 
                             help='Data source')
    
    # Validate all formulas
    all_parser = subparsers.add_parser('validate-all', help='Validate all formulas')
    all_parser.add_argument('--source', choices=['processed', 'graph'], default='processed', 
                          help='Data source')
    
    # Generate report
    report_parser = subparsers.add_parser('generate-report', help='Generate validation report')
    report_parser.add_argument('--source', choices=['processed', 'graph'], default='processed', 
                             help='Data source')
    report_parser.add_argument('--output', help='Output file path')
    
    # Common arguments
    for subparser in [random_parser, all_parser, report_parser]:
        subparser.add_argument('--seed', type=int, help='Random seed for reproducibility')
        subparser.add_argument('--tests', type=int, default=100, help='Number of random tests per formula')
        subparser.add_argument('--tolerance', type=float, default=1e-10, help='Numerical tolerance')
        subparser.add_argument('--update-graph', action='store_true', help='Update graph with results')
    
    return parser


def main():
    """Main CLI entry point for validation."""
    parser = create_cli_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Create validation config
    config = ValidationConfig(
        num_random_tests=args.tests,
        random_seed=args.seed,
        tolerance=args.tolerance
    )
    
    # Initialize test runner
    runner = TestRunner(config)
    
    # Execute command
    report = None
    
    if args.command == 'validate-random':
        report = runner.validate_random_formulas(args.count, args.source)
        
    elif args.command == 'validate-all':
        report = runner.validate_all_formulas(args.source)
        
    elif args.command == 'generate-report':
        report = runner.validate_all_formulas(args.source)
    
    if report:
        # Print summary
        print(f"\n=== Validation Report ===")
        print(f"Total formulas: {report.total_formulas}")
        print(f"Passed: {report.passed_formulas}")
        print(f"Failed: {report.failed_formulas}")
        print(f"Errors: {report.error_formulas}")
        print(f"Partial: {report.partial_formulas}")
        print(f"Overall pass rate: {report.overall_pass_rate:.1%}")
        print(f"Average confidence: {report.average_confidence:.3f}")
        print(f"Validation time: {report.validation_time:.2f}s")
        
        # Save report if requested
        if hasattr(args, 'output') and args.output:
            output_path = Path(args.output)
            runner.save_report_to_file(report, output_path)
        
        # Update graph if requested
        if args.update_graph:
            runner.update_graph_with_results(report)


if __name__ == '__main__':
    main() 