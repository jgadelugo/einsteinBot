"""
Main entry point for MathBot ingestion pipeline.

This module provides the command-line interface and orchestrates the complete
ingestion workflow from PDF/LaTeX sources to cleaned mathematical expressions.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

from config import (
    ExtractionMode,
    DEFAULT_EXTRACTION_MODE,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    logger,
    setup_logging,
)
from ingestion.parser import PDFParser, LaTeXParser
from ingestion.formula_extractor import FormulaExtractor
from ingestion.cleaner import FormulaCleaner


class MathBotIngestion:
    """
    Main class for orchestrating the MathBot ingestion pipeline.
    
    This class coordinates the parsing, extraction, and cleaning of mathematical
    content from various sources.
    """
    
    def __init__(
        self,
        pdf_backend: str = "pymupdf",
        clean_formulas: bool = True,
        normalize_symbols: bool = True
    ):
        """
        Initialize the ingestion pipeline.
        
        Args:
            pdf_backend: Backend for PDF parsing ("pymupdf" or "pdfminer")
            clean_formulas: Whether to clean extracted formulas
            normalize_symbols: Whether to normalize mathematical symbols
        """
        self.pdf_parser = PDFParser(backend=pdf_backend)
        self.latex_parser = LaTeXParser()
        self.formula_extractor = FormulaExtractor(clean_formulas=clean_formulas)
        self.formula_cleaner = FormulaCleaner()
        self.normalize_symbols = normalize_symbols
        
        self.logger = logger
        self.logger.info("Initialized MathBot ingestion pipeline")
    
    def process_file(
        self,
        file_path: Union[str, Path],
        extraction_mode: str = DEFAULT_EXTRACTION_MODE,
        max_pages: Optional[int] = None,
        output_format: str = "json"
    ) -> Dict:
        """
        Process a single file through the complete ingestion pipeline.
        
        Args:
            file_path: Path to the file to process
            extraction_mode: What to extract ("text_only", "formulas_only", "both")
            max_pages: Maximum pages to process (PDF only)
            output_format: Output format ("json", "text")
            
        Returns:
            Dictionary containing processed results
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        self.logger.info(f"Processing file: {file_path}")
        
        # Determine file type and parse
        if file_path.suffix.lower() == ".pdf":
            parser_result = self.pdf_parser.extract_text(
                file_path, max_pages, extraction_mode
            )
        else:
            parser_result = self.latex_parser.extract_text(
                file_path, extraction_mode
            )
        
        # Extract additional formula information if needed
        if extraction_mode in [ExtractionMode.FORMULAS_ONLY, ExtractionMode.BOTH]:
            self._enhance_formula_extraction(parser_result)
        
        # Clean and normalize if requested
        if self.normalize_symbols and "formulas" in parser_result:
            parser_result["cleaned_formulas"] = self._clean_formulas(
                parser_result["formulas"]
            )
        
        # Add processing metadata
        parser_result["processing_metadata"] = {
            "pipeline_version": "0.1.0",
            "extraction_mode": extraction_mode,
            "normalize_symbols": self.normalize_symbols,
            "file_type": file_path.suffix.lower(),
        }
        
        self.logger.info(f"Successfully processed {file_path}")
        return parser_result
    
    def _enhance_formula_extraction(self, parser_result: Dict) -> None:
        """
        Enhance formula extraction with additional metadata and analysis.
        
        Args:
            parser_result: Result dictionary from parser
        """
        if "formulas" not in parser_result:
            return
        
        # Get detailed formula information
        if "text_blocks" in parser_result:
            all_text = "\n".join(parser_result["text_blocks"])
        else:
            all_text = ""
        
        if all_text:
            detailed_formulas = self.formula_extractor.extract_expressions(all_text)
            parser_result["detailed_formulas"] = detailed_formulas
            
            # Add statistics
            parser_result["formula_statistics"] = self.formula_extractor.get_math_statistics(all_text)
    
    def _clean_formulas(self, formulas: List[str]) -> Dict[str, List[str]]:
        """
        Clean and normalize formulas using the formula cleaner.
        
        Args:
            formulas: List of raw formulas
            
        Returns:
            Dictionary with different cleaned formats
        """
        return {
            "unicode_normalized": self.formula_cleaner.batch_clean(formulas),
            "sympy_compatible": [
                self.formula_cleaner.convert_to_sympy_format(formula)
                for formula in formulas
            ],
        }
    
    def process_directory(
        self,
        directory_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        file_pattern: str = "*.pdf",
        **kwargs
    ) -> Dict[str, Dict]:
        """
        Process all files in a directory.
        
        Args:
            directory_path: Path to directory containing files
            output_dir: Directory to save results (optional)
            file_pattern: Glob pattern for files to process
            **kwargs: Additional arguments for process_file
            
        Returns:
            Dictionary mapping file paths to processing results
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        files = list(directory_path.glob(file_pattern))
        
        if not files:
            self.logger.warning(f"No files found matching pattern: {file_pattern}")
            return {}
        
        self.logger.info(f"Processing {len(files)} files from {directory_path}")
        
        results = {}
        
        for file_path in files:
            try:
                result = self.process_file(file_path, **kwargs)
                results[str(file_path)] = result
                
                # Save individual result if output directory specified
                if output_dir:
                    output_file = output_dir / f"{file_path.stem}_processed.json"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
                
            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {e}")
                results[str(file_path)] = {"error": str(e)}
        
        self.logger.info(f"Completed processing {len(files)} files")
        return results
    
    def generate_report(self, results: Dict[str, Dict]) -> Dict:
        """
        Generate a summary report from processing results.
        
        Args:
            results: Dictionary of processing results
            
        Returns:
            Summary report dictionary
        """
        report = {
            "summary": {
                "total_files": len(results),
                "successful_files": len([r for r in results.values() if "error" not in r]),
                "failed_files": len([r for r in results.values() if "error" in r]),
            },
            "statistics": {
                "total_formulas": 0,
                "total_text_blocks": 0,
                "files_by_type": {},
                "extraction_modes": {},
            },
            "errors": []
        }
        
        for file_path, result in results.items():
            if "error" in result:
                report["errors"].append({"file": file_path, "error": result["error"]})
                continue
            
            # Aggregate statistics
            if "formulas" in result:
                report["statistics"]["total_formulas"] += len(result["formulas"])
            
            if "text_blocks" in result:
                report["statistics"]["total_text_blocks"] += len(result["text_blocks"])
            
            # File type statistics
            if "processing_metadata" in result:
                file_type = result["processing_metadata"].get("file_type", "unknown")
                report["statistics"]["files_by_type"][file_type] = \
                    report["statistics"]["files_by_type"].get(file_type, 0) + 1
                
                extraction_mode = result["processing_metadata"].get("extraction_mode", "unknown")
                report["statistics"]["extraction_modes"][extraction_mode] = \
                    report["statistics"]["extraction_modes"].get(extraction_mode, 0) + 1
        
        return report


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="MathBot Ingestion Pipeline - Extract and process mathematical content",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single PDF file
  python main.py process-file document.pdf

  # Extract only formulas from a LaTeX file
  python main.py process-file paper.tex --mode formulas_only

  # Process all PDFs in a directory
  python main.py process-directory ./documents --pattern "*.pdf"

  # Process with custom output directory and normalization
  python main.py process-file book.pdf --output results/ --normalize-symbols
  
  # Validate formulas
  python main.py validate-formulas --count 10 --source processed
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Process single file command
    file_parser = subparsers.add_parser(
        "process-file", 
        help="Process a single file"
    )
    file_parser.add_argument("file_path", help="Path to the file to process")
    file_parser.add_argument(
        "--mode", 
        choices=["text_only", "formulas_only", "both"], 
        default="both",
        help="Extraction mode (default: both)"
    )
    file_parser.add_argument(
        "--max-pages", 
        type=int, 
        help="Maximum pages to process (PDF only)"
    )
    file_parser.add_argument(
        "--backend", 
        choices=["pymupdf", "pdfminer"], 
        default="pymupdf",
        help="PDF parsing backend (default: pymupdf)"
    )
    file_parser.add_argument(
        "--output", 
        help="Output file path (default: print to stdout)"
    )
    file_parser.add_argument(
        "--format", 
        choices=["json", "text"], 
        default="json",
        help="Output format (default: json)"
    )
    file_parser.add_argument(
        "--normalize-symbols", 
        action="store_true",
        help="Normalize mathematical symbols"
    )
    
    # Process directory command
    dir_parser = subparsers.add_parser(
        "process-directory", 
        help="Process all files in a directory"
    )
    dir_parser.add_argument("directory_path", help="Path to the directory")
    dir_parser.add_argument(
        "--pattern", 
        default="*.pdf",
        help="File pattern to match (default: *.pdf)"
    )
    dir_parser.add_argument(
        "--output-dir", 
        help="Output directory for results"
    )
    dir_parser.add_argument(
        "--mode", 
        choices=["text_only", "formulas_only", "both"], 
        default="both",
        help="Extraction mode (default: both)"
    )
    dir_parser.add_argument(
        "--backend", 
        choices=["pymupdf", "pdfminer"], 
        default="pymupdf",
        help="PDF parsing backend (default: pymupdf)"
    )
    dir_parser.add_argument(
        "--normalize-symbols", 
        action="store_true",
        help="Normalize mathematical symbols"
    )
    dir_parser.add_argument(
        "--generate-report", 
        action="store_true",
        help="Generate summary report"
    )
    
    # Validate formulas command
    validate_parser = subparsers.add_parser(
        "validate-formulas",
        help="Validate mathematical formulas"
    )
    validate_parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of random formulas to validate (default: 10)"
    )
    validate_parser.add_argument(
        "--source",
        choices=["processed", "graph"],
        default="processed",
        help="Source of formulas to validate (default: processed)"
    )
    validate_parser.add_argument(
        "--all",
        action="store_true",
        help="Validate all available formulas instead of random selection"
    )
    validate_parser.add_argument(
        "--output-report",
        help="Path to save validation report (JSON format)"
    )
    validate_parser.add_argument(
        "--update-graph",
        action="store_true",
        help="Update knowledge graph with validation results"
    )
    validate_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible testing (default: 42)"
    )
    validate_parser.add_argument(
        "--num-tests",
        type=int,
        default=100,
        help="Number of random numerical tests per formula (default: 100)"
    )
    validate_parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-10,
        help="Numerical tolerance for validation (default: 1e-10)"
    )
    
    # Explore patterns command
    explore_parser = subparsers.add_parser(
        "explore-patterns",
        help="Discover patterns in mathematical formulas"
    )
    explore_parser.add_argument(
        "--source",
        choices=["processed", "graph"],
        default="processed",
        help="Source of formulas to analyze (default: processed)"
    )
    explore_parser.add_argument(
        "--output-dir",
        default="results/exploration",
        help="Directory to save exploration results (default: results/exploration)"
    )
    explore_parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.7,
        help="Similarity threshold for pattern clustering (default: 0.7)"
    )
    explore_parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=2,
        help="Minimum formulas per cluster (default: 2)"
    )
    explore_parser.add_argument(
        "--max-hypotheses",
        type=int,
        default=10,
        help="Maximum hypotheses per type (default: 10)"
    )
    explore_parser.add_argument(
        "--embedding-method",
        choices=["structural", "tfidf", "hybrid"],
        default="structural",
        help="Method for formula embeddings (default: structural)"
    )
    explore_parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization plots"
    )
    
    # Generate theorems command (Phase 5A)
    theorem_parser = subparsers.add_parser(
        "generate-theorems",
        help="Generate formal theorems from validated hypotheses (Phase 5A)"
    )
    theorem_parser.add_argument(
        "--input",
        required=True,
        help="Path to hypotheses JSON file from Phase 4"
    )
    theorem_parser.add_argument(
        "--output",
        default="results/theorems.json",
        help="Path to save generated theorems (default: results/theorems.json)"
    )
    theorem_parser.add_argument(
        "--config",
        help="Path to theorem generation configuration file"
    )
    theorem_parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate generated theorems using Phase 3 engine"
    )
    theorem_parser.add_argument(
        "--stats",
        action="store_true",
        help="Print detailed generation statistics"
    )
    theorem_parser.add_argument(
        "--prove",
        action="store_true",
        help="Attempt to prove generated theorems using symbolic methods (Phase 5B)"
    )
    theorem_parser.add_argument(
        "--proof-output",
        default="results/proof_results.json",
        help="Path to save proof results (default: results/proof_results.json)"
    )
    theorem_parser.add_argument(
        "--proof-config",
        help="Path to proof engine configuration file"
    )

    # Global options
    parser.add_argument(
        "--log-level", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
        default="INFO",
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--quiet", 
        action="store_true",
        help="Suppress output (except errors)"
    )
    
    return parser


def main():
    """Main entry point for the application."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Set up logging
    if args.quiet:
        log_level = "ERROR"
    else:
        log_level = args.log_level
    
    setup_logging(level=log_level)
    
    try:
        # Initialize the ingestion pipeline
        pipeline = MathBotIngestion(
            pdf_backend=getattr(args, "backend", "pymupdf"),
            normalize_symbols=getattr(args, "normalize_symbols", False)
        )
        
        if args.command == "process-file":
            # Process single file
            result = pipeline.process_file(
                args.file_path,
                extraction_mode=args.mode,
                max_pages=args.max_pages,
                output_format=args.format
            )
            
            # Output result
            if args.output:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    if args.format == "json":
                        json.dump(result, f, indent=2, ensure_ascii=False)
                    else:
                        # Text format
                        if "text_blocks" in result:
                            f.write("=== TEXT BLOCKS ===\n")
                            for i, block in enumerate(result["text_blocks"], 1):
                                f.write(f"\n--- Block {i} ---\n{block}\n")
                        
                        if "formulas" in result:
                            f.write("\n=== FORMULAS ===\n")
                            for i, formula in enumerate(result["formulas"], 1):
                                f.write(f"{i}. {formula}\n")
                
                print(f"Results saved to: {output_path}")
            else:
                if args.format == "json":
                    print(json.dumps(result, indent=2, ensure_ascii=False))
                else:
                    print(f"Processed: {args.file_path}")
                    if "text_blocks" in result:
                        print(f"Text blocks: {len(result['text_blocks'])}")
                    if "formulas" in result:
                        print(f"Formulas: {len(result['formulas'])}")
                        print("\nFormulas found:")
                        for i, formula in enumerate(result["formulas"][:10], 1):
                            print(f"  {i}. {formula}")
                        if len(result["formulas"]) > 10:
                            print(f"  ... and {len(result['formulas']) - 10} more")
        
        elif args.command == "process-directory":
            # Process directory
            results = pipeline.process_directory(
                args.directory_path,
                output_dir=args.output_dir,
                file_pattern=args.pattern,
                extraction_mode=args.mode
            )
            
            print(f"Processed {len(results)} files")
            
            if args.generate_report:
                report = pipeline.generate_report(results)
                
                report_path = Path(args.output_dir or ".") / "processing_report.json"
                with open(report_path, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
                
                print(f"Summary report saved to: {report_path}")
                
                # Print summary
                print("\n=== Processing Summary ===")
                print(f"Total files: {report['summary']['total_files']}")
                print(f"Successful: {report['summary']['successful_files']}")
                print(f"Failed: {report['summary']['failed_files']}")
                print(f"Total formulas extracted: {report['statistics']['total_formulas']}")
                print(f"Total text blocks: {report['statistics']['total_text_blocks']}")
                
                if report["errors"]:
                    print(f"\nErrors ({len(report['errors'])}):")
                    for error in report["errors"][:5]:
                        print(f"  {error['file']}: {error['error']}")
        
        elif args.command == "validate-formulas":
            # Import validation modules
            try:
                from validation.test_runner import TestRunner
                from validation.formula_tester import ValidationConfig
            except ImportError as e:
                logger.error(f"Validation modules not available: {e}")
                sys.exit(1)
            
            # Create validation configuration
            validation_config = ValidationConfig(
                num_random_tests=args.num_tests,
                random_seed=args.seed,
                tolerance=args.tolerance
            )
            
            # Initialize test runner
            test_runner = TestRunner(validation_config)
            
            # Run validation
            if args.all:
                report = test_runner.validate_all_formulas(source=args.source)
                print(f"Validated all formulas from {args.source}")
            else:
                report = test_runner.validate_random_formulas(
                    count=args.count, 
                    source=args.source
                )
                print(f"Validated {args.count} random formulas from {args.source}")
            
            # Display summary
            print(f"\n=== Validation Summary ===")
            print(f"Total formulas: {report.total_formulas}")
            pass_rate = (report.passed_formulas/report.total_formulas)*100 if report.total_formulas > 0 else 0
            print(f"Passed: {report.passed_formulas} ({pass_rate:.1f}%)")
            print(f"Failed: {report.failed_formulas}")
            print(f"Errors: {report.error_formulas}")
            print(f"Partial: {report.partial_formulas}")
            print(f"Overall pass rate: {report.overall_pass_rate:.1%}")
            print(f"Average confidence: {report.average_confidence:.3f}")
            print(f"Validation time: {report.validation_time:.2f}s")
            
            if report.errors_summary:
                print(f"\nFirst few errors:")
                for error in report.errors_summary[:3]:
                    print(f"  {error}")
            
            # Save report if requested
            if args.output_report:
                report_path = Path(args.output_report)
                test_runner.save_report_to_file(report, report_path)
                print(f"\nDetailed report saved to: {report_path}")
            
            # Update graph if requested
            if args.update_graph:
                test_runner.update_graph_with_results(report)
                print("Knowledge graph updated with validation results")
        
        elif args.command == "explore-patterns":
            # Import exploration modules
            try:
                from exploration import PatternFinder, GapDetector, HypothesisGenerator
                from exploration.utils.embedding import FormulaEmbedder, ClusterVisualizer
                from validation.formula_tester import ValidationConfig
            except ImportError as e:
                logger.error(f"Exploration modules not available: {e}")
                sys.exit(1)
            
            # Load formula data
            formulas_data = []
            validation_results = {}
            
            if args.source == "processed":
                processed_dir = Path("data/processed")
                for json_file in processed_dir.glob("*.json"):
                    try:
                        with open(json_file, 'r') as f:
                            data = json.load(f)
                            formulas_data.append(data)
                    except Exception as e:
                        logger.warning(f"Failed to load {json_file}: {e}")
            else:
                # Load from graph source
                graph_dir = Path("data/graph") 
                for json_file in graph_dir.glob("*.json"):
                    try:
                        with open(json_file, 'r') as f:
                            data = json.load(f)
                            formulas_data.append(data)
                    except Exception as e:
                        logger.warning(f"Failed to load {json_file}: {e}")
            
            if not formulas_data:
                logger.error(f"No formula data found in {args.source} source")
                sys.exit(1)
            
            # Extract all formulas
            all_formulas = []
            formula_metadata = {}
            
            for data in formulas_data:
                if "formulas" in data:
                    all_formulas.extend(data["formulas"])
                if "detailed_formulas" in data:
                    for detailed in data["detailed_formulas"]:
                        formula = detailed.get("expression", "")
                        if formula:
                            all_formulas.append(formula)
                            formula_metadata[formula] = detailed.get("metadata", {})
            
            print(f"Found {len(all_formulas)} formulas from {len(formulas_data)} data sources")
            
            # Create output directory
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. Pattern Discovery
            print("\n=== Phase 4.1: Pattern Discovery ===")
            pattern_finder = PatternFinder(
                similarity_threshold=args.similarity_threshold,
                min_cluster_size=args.min_cluster_size
            )
            
            patterns = pattern_finder.find_patterns(all_formulas, formula_metadata)
            pattern_finder.save_patterns(patterns, output_dir / "patterns.json")
            
            print(f"Discovered {len(patterns)} pattern clusters")
            for i, cluster in enumerate(patterns[:5]):  # Show first 5 clusters
                print(f"  Cluster {cluster.cluster_id}: {len(cluster.formulas)} formulas, "
                      f"confidence: {cluster.confidence_score:.3f}")
                print(f"    Center: {cluster.cluster_center}")
                print(f"    Topics: {list(cluster.topic_tags)}")
            
            # 2. Gap Detection
            print("\n=== Phase 4.2: Gap Detection ===")
            gap_detector = GapDetector()
            gaps = gap_detector.detect_gaps(formulas_data, validation_results)
            gap_detector.save_gaps(gaps, output_dir / "gaps.json")
            
            print(f"Detected {len(gaps)} potential gaps")
            for gap in gaps[:5]:  # Show first 5 gaps
                print(f"  {gap.gap_type.value}: {gap.title}")
                print(f"    Confidence: {gap.confidence_score:.3f}, Priority: {gap.priority:.3f}")
                if gap.suggested_formulas:
                    print(f"    Suggestions: {gap.suggested_formulas[:2]}")
            
            # 3. Hypothesis Generation
            print("\n=== Phase 4.3: Hypothesis Generation ===")
            hypothesis_generator = HypothesisGenerator(
                max_hypotheses_per_type=args.max_hypotheses
            )
            
            # Use validated formulas as sources (fallback to all if none validated)
            source_formulas = all_formulas[:20]  # Limit for performance
            hypotheses = hypothesis_generator.generate_hypotheses(source_formulas, formula_metadata)
            hypothesis_generator.save_hypotheses(hypotheses, output_dir / "conjectures.json")
            
            print(f"Generated {len(hypotheses)} hypotheses")
            
            # Show promising hypotheses
            promising = hypothesis_generator.get_promising_hypotheses(hypotheses, min_confidence=0.6)
            print(f"Found {len(promising)} promising hypotheses:")
            
            for hyp in promising[:5]:  # Show first 5 promising
                print(f"  {hyp.hypothesis_type.value}: {hyp.formula}")
                print(f"    Status: {hyp.status.value}, Confidence: {hyp.confidence_score:.3f}")
                print(f"    Description: {hyp.description}")
            
            # 4. Visualization (if requested)
            if args.visualize:
                print("\n=== Phase 4.4: Visualization ===")
                try:
                    embedder = FormulaEmbedder(embedding_method=args.embedding_method)
                    visualizer = ClusterVisualizer()
                    
                    # Create embeddings
                    embedding_result = embedder.embed_formulas(all_formulas[:50])  # Limit for visualization
                    
                    # Cluster embeddings
                    embedding_result = embedder.cluster_embeddings(embedding_result, method="dbscan")
                    
                    # Reduce dimensions for plotting
                    embedding_result = embedder.reduce_dimensions(embedding_result, method="tsne")
                    
                    # Create visualizations
                    visualizer.plot_clusters(
                        embedding_result,
                        output_path=output_dir / "cluster_plot.png",
                        show_formulas=False
                    )
                    
                    visualizer.plot_similarity_matrix(
                        embedding_result,
                        output_path=output_dir / "similarity_matrix.png"
                    )
                    
                    # Save visualization data
                    visualizer.save_visualization_data(
                        embedding_result,
                        output_dir / "visualization_data.json"
                    )
                    
                    print("Visualizations saved to output directory")
                    
                except Exception as e:
                    logger.warning(f"Visualization failed: {e}")
            
            # Summary
            print(f"\n=== Phase 4 Complete ===")
            print(f"Results saved to: {output_dir}")
            print(f"- patterns.json: {len(patterns)} formula clusters")
            print(f"- gaps.json: {len(gaps)} detected gaps")
            print(f"- conjectures.json: {len(hypotheses)} generated hypotheses")
            print(f"- {len(promising)} promising hypotheses identified")
            
            if args.visualize:
                print("- cluster_plot.png: Formula cluster visualization")
                print("- similarity_matrix.png: Formula similarity heatmap")
                print("- visualization_data.json: Raw visualization data")
        
        elif args.command == "generate-theorems":
            # Import theorem generation modules
            try:
                from proofs.theorem_generator import TheoremGenerator
                from validation.formula_tester import FormulaValidator
            except ImportError as e:
                logger.error(f"Theorem generation modules not available: {e}")
                sys.exit(1)
            
            # Load hypotheses from Phase 4
            hypotheses_path = Path(args.input)
            if not hypotheses_path.exists():
                logger.error(f"Hypotheses file not found: {hypotheses_path}")
                sys.exit(1)
            
            try:
                with open(hypotheses_path, 'r') as f:
                    hypotheses_data = json.load(f)
                hypotheses = hypotheses_data.get('hypotheses', [])
                
                if not hypotheses:
                    logger.error("No hypotheses found in input file")
                    sys.exit(1)
                
                print(f"Loaded {len(hypotheses)} hypotheses from {hypotheses_path}")
            except Exception as e:
                logger.error(f"Failed to load hypotheses: {e}")
                sys.exit(1)
            
            # Load configuration if provided
            config = {}
            if args.config:
                try:
                    with open(args.config, 'r') as f:
                        config = json.load(f)
                    print(f"Loaded configuration from {args.config}")
                except Exception as e:
                    logger.warning(f"Failed to load config: {e}")
            
            # Initialize theorem generator
            validation_engine = FormulaValidator() if args.validate else None
            theorem_generator = TheoremGenerator(
                validation_engine=validation_engine,
                config=config
            )
            
            print("\n=== Phase 5A: Theorem Generation ===")
            print("Converting validated hypotheses to formal theorems...")
            
            # Generate theorems
            theorems = theorem_generator.generate_from_hypotheses(hypotheses)
            
            if not theorems:
                logger.error("No theorems generated from hypotheses")
                sys.exit(1)
            
            print(f"Generated {len(theorems)} formal theorems")
            
            # Save theorems
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            theorem_generator.save_theorems(theorems, output_path)
            
            print(f"Theorems saved to: {output_path}")
            
            # Display statistics
            stats = theorem_generator.get_generation_stats()
            print(f"\n=== Generation Statistics ===")
            print(f"Total theorems generated: {stats['theorems_generated']}")
            print(f"Validation passes: {stats['validation_passes']}")
            print(f"Generation time: {stats['generation_time']:.2f}s")
            
            # Show theorem type distribution
            type_distribution = theorem_generator._get_type_distribution(theorems)
            print(f"\n=== Theorem Type Distribution ===")
            for theorem_type, count in sorted(type_distribution.items()):
                print(f"  {theorem_type}: {count}")
            
            # Show sample theorems
            print(f"\n=== Sample Generated Theorems ===")
            for i, theorem in enumerate(theorems[:3]):  # Show first 3 theorems
                print(f"\nTheorem {i+1} ({theorem.id}):")
                print(f"  Type: {theorem.theorem_type.value}")
                print(f"  Statement: {theorem.statement}")
                print(f"  Natural Language: {theorem.natural_language}")
                print(f"  Source: {theorem.source_lineage.hypothesis_id}")
                print(f"  Confidence: {theorem.source_lineage.confidence:.2f}")
            
            if len(theorems) > 3:
                print(f"\n... and {len(theorems) - 3} more theorems")
            
            # Detailed statistics if requested
            if args.stats:
                print(f"\n=== Detailed Statistics ===")
                print(f"Input hypotheses: {len(hypotheses)}")
                print(f"Successful conversions: {len(theorems)}")
                conversion_rate = (len(theorems) / len(hypotheses)) * 100 if hypotheses else 0
                print(f"Conversion rate: {conversion_rate:.1f}%")
                
                # Symbol analysis
                all_symbols = set()
                for theorem in theorems:
                    all_symbols.update(theorem.symbols)
                print(f"Unique symbols across all theorems: {len(all_symbols)}")
                print(f"Symbols: {sorted(list(all_symbols))}")
                
                # Assumption analysis
                all_assumptions = set()
                for theorem in theorems:
                    all_assumptions.update(theorem.assumptions)
                print(f"Unique assumptions: {len(all_assumptions)}")
                
                # Validation evidence summary
                if args.validate:
                    total_validation_score = sum(
                        theorem.source_lineage.validation_score for theorem in theorems
                    )
                    avg_validation_score = total_validation_score / len(theorems) if theorems else 0
                    print(f"Average validation score: {avg_validation_score:.3f}")
            
            print(f"\n=== Phase 5A Complete ===")
            print(f"Successfully generated {len(theorems)} formal theorems")
            print(f"Results saved to: {output_path}")
            
            # Phase 5B: Proof Attempt (if requested)
            if args.prove:
                try:
                    from proofs.proof_attempt import ProofAttemptEngine
                except ImportError as e:
                    logger.error(f"Proof attempt modules not available: {e}")
                    sys.exit(1)
                
                print(f"\n=== Phase 5B: Symbolic Proof Attempts ===")
                print("Attempting to prove generated theorems using symbolic methods...")
                
                # Load proof configuration if provided
                proof_config = {}
                if args.proof_config:
                    try:
                        with open(args.proof_config, 'r') as f:
                            proof_config = json.load(f)
                        print(f"Loaded proof configuration from {args.proof_config}")
                    except Exception as e:
                        logger.warning(f"Failed to load proof config: {e}")
                
                # Initialize proof engine
                proof_engine = ProofAttemptEngine(config=proof_config)
                
                # Attempt to prove theorems
                proof_results = proof_engine.batch_prove_theorems(theorems)
                
                # Save proof results
                proof_output_path = Path(args.proof_output)
                proof_output_path.parent.mkdir(parents=True, exist_ok=True)
                proof_engine.save_results(proof_results, proof_output_path)
                
                print(f"Proof results saved to: {proof_output_path}")
                
                # Display proof statistics
                proof_stats = proof_engine.get_statistics()
                successful_proofs = sum(1 for r in proof_results if r.is_successful())
                
                print(f"\n=== Proof Statistics ===")
                print(f"Total proof attempts: {proof_stats['total_attempts']}")
                print(f"Successfully proved: {successful_proofs}")
                print(f"Success rate: {proof_stats['overall_success_rate']:.1%}")
                print(f"Total proof time: {proof_stats['total_time']:.2f}s")
                print(f"Average time per proof: {proof_stats['average_time_per_attempt']:.3f}s")
                
                if proof_stats['cache_hits'] > 0:
                    print(f"Cache hits: {proof_stats['cache_hits']}")
                
                # Show method success rates
                if proof_stats['method_success_rates']:
                    print(f"\n=== Method Success Rates ===")
                    for method, data in proof_stats['method_success_rates'].items():
                        if data['attempts'] > 0:
                            success_rate = data['success_rate'] * 100
                            print(f"  {method}: {data['successes']}/{data['attempts']} ({success_rate:.1f}%)")
                
                # Show sample proof results
                print(f"\n=== Sample Proof Results ===")
                proved_results = [r for r in proof_results if r.is_successful()]
                failed_results = [r for r in proof_results if not r.is_successful()]
                
                if proved_results:
                    print(f"\nSuccessfully Proved ({len(proved_results)}):")
                    for i, result in enumerate(proved_results[:3]):  # Show first 3
                        print(f"  {i+1}. {result.theorem_id} - {result.method.value}")
                        print(f"     Confidence: {result.confidence_score:.2f}, Steps: {result.get_step_count()}")
                        print(f"     Time: {result.execution_time:.3f}s")
                
                if failed_results:
                    print(f"\nFailed/Inconclusive ({len(failed_results)}):")
                    for i, result in enumerate(failed_results[:3]):  # Show first 3
                        print(f"  {i+1}. {result.theorem_id} - {result.status.value}")
                        if result.error_message:
                            print(f"     Error: {result.error_message[:100]}...")
                
                print(f"\n=== Phase 5B Complete ===")
                print(f"Attempted proofs for {len(theorems)} theorems")
                print(f"Successfully proved {successful_proofs} theorems ({successful_proofs/len(theorems):.1%})")
                print(f"Proof results saved to: {proof_output_path}")
    
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        if args.log_level == "DEBUG":
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()