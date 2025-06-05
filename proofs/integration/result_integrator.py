"""
Result integration for MathBot Phase 5E.

This module implements the ResultIntegrator class that combines results from
multiple proof strategies, calculates confidence scores, and generates
comprehensive proof summaries.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum

from proofs.integration.proof_pipeline import ProofStrategy, ComprehensiveResult
from proofs.theorem_generator import Theorem
from proofs.proof_attempt import ProofResult, ProofStatus


class ResultConfidenceLevel(Enum):
    """Confidence levels for proof results."""
    VERY_LOW = "very_low"      # 0.0 - 0.2
    LOW = "low"                # 0.2 - 0.4
    MODERATE = "moderate"      # 0.4 - 0.6
    HIGH = "high"              # 0.6 - 0.8
    VERY_HIGH = "very_high"    # 0.8 - 1.0


@dataclass
class MethodResult:
    """Result from a specific proof method."""
    method_name: str
    success: bool
    confidence: float
    execution_time: float
    details: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'method_name': self.method_name,
            'success': self.success,
            'confidence': self.confidence,
            'execution_time': self.execution_time,
            'details': self.details,
            'error_message': self.error_message
        }


@dataclass
class ResultAnalysis:
    """Analysis of proof results across multiple methods."""
    total_methods_attempted: int
    successful_methods: List[str]
    failed_methods: List[str]
    best_method: str
    worst_method: str
    average_confidence: float
    confidence_variance: float
    total_execution_time: float
    convergent_results: bool  # Do multiple methods agree?
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'total_methods_attempted': self.total_methods_attempted,
            'successful_methods': self.successful_methods,
            'failed_methods': self.failed_methods,
            'best_method': self.best_method,
            'worst_method': self.worst_method,
            'average_confidence': self.average_confidence,
            'confidence_variance': self.confidence_variance,
            'total_execution_time': self.total_execution_time,
            'convergent_results': self.convergent_results
        }


@dataclass
class ResultIntegrator:
    """
    Integrates results from multiple proof strategies and methods.
    
    This class combines results from symbolic proving, rule-based transformations,
    and formal verification to create comprehensive proof results with confidence
    scoring and detailed analysis.
    """
    
    def __init__(self):
        """Initialize the result integrator."""
        self.logger = logging.getLogger(__name__)
        
        # Weights for different proof methods in confidence calculation
        self.method_weights = {
            'symbolic': 0.4,
            'rule_based': 0.3,
            'formal_verification': 0.8,
            'hybrid_symbolic_rule': 0.6,
            'hybrid_symbolic_formal': 0.7,
            'comprehensive': 0.9
        }
        
        # Confidence thresholds for different levels
        self.confidence_thresholds = {
            ResultConfidenceLevel.VERY_LOW: (0.0, 0.2),
            ResultConfidenceLevel.LOW: (0.2, 0.4),
            ResultConfidenceLevel.MODERATE: (0.4, 0.6),
            ResultConfidenceLevel.HIGH: (0.6, 0.8),
            ResultConfidenceLevel.VERY_HIGH: (0.8, 1.0)
        }
        
        # Statistics
        self.integration_stats = {
            'total_integrations': 0,
            'high_confidence_results': 0,
            'convergent_results': 0,
            'method_performance': {}
        }
    
    def integrate_proof_results(self, theorem: Theorem, 
                               strategy_results: Dict[ProofStrategy, Dict],
                               execution_time: float = 0.0) -> ComprehensiveResult:
        """
        Integrate results from multiple proof strategies.
        
        Args:
            theorem: The theorem that was being proved
            strategy_results: Results from different strategies
            execution_time: Total execution time
            
        Returns:
            ComprehensiveResult with integrated information
        """
        start_time = time.time()
        
        self.integration_stats['total_integrations'] += 1
        
        # Extract method results
        method_results = self._extract_method_results(strategy_results)
        
        # Determine final status
        final_status = self._determine_final_status(method_results)
        
        # Calculate confidence score
        confidence_score = self._calculate_integrated_confidence(method_results)
        
        # Extract best results from each method type
        best_symbolic = self._extract_best_symbolic_result(strategy_results)
        best_transformations = self._extract_best_transformations(strategy_results)
        best_formal = self._extract_best_formal_result(strategy_results)
        
        # Perform result analysis
        analysis = self._analyze_results(method_results)
        
        # Check for convergent results
        convergent = self._check_result_convergence(method_results)
        
        # Collect error messages
        error_messages = self._collect_error_messages(strategy_results)
        
        # Create metadata
        metadata = self._create_result_metadata(
            analysis, convergent, method_results, execution_time
        )
        
        # Update statistics
        self._update_statistics(confidence_score, convergent, method_results)
        
        integration_time = time.time() - start_time
        self.logger.debug(
            f"Integrated results for theorem {theorem.id} in {integration_time:.3f}s: "
            f"status={final_status}, confidence={confidence_score:.3f}"
        )
        
        return ComprehensiveResult(
            theorem=theorem,
            strategies_attempted=list(strategy_results.keys()),
            symbolic_result=best_symbolic,
            rule_transformations=best_transformations,
            formal_result=best_formal,
            final_status=final_status,
            confidence_score=confidence_score,
            execution_time=execution_time,
            error_messages=error_messages,
            metadata=metadata
        )
    
    def _extract_method_results(self, strategy_results: Dict[ProofStrategy, Dict]) -> List[MethodResult]:
        """Extract standardized method results from strategy results."""
        method_results = []
        
        for strategy, result in strategy_results.items():
            if not result or 'error' in result:
                # Failed method
                method_results.append(MethodResult(
                    method_name=strategy.value,
                    success=False,
                    confidence=0.0,
                    execution_time=0.0,
                    error_message=result.get('error', 'Unknown error') if result else 'No result'
                ))
                continue
            
            # Determine success and confidence based on strategy type
            success = result.get('success', False)
            confidence = self._extract_confidence_from_result(strategy, result)
            
            method_results.append(MethodResult(
                method_name=strategy.value,
                success=success,
                confidence=confidence,
                execution_time=0.0,  # Will be updated if available
                details=self._extract_method_details(strategy, result)
            ))
        
        return method_results
    
    def _extract_confidence_from_result(self, strategy: ProofStrategy, result: Dict) -> float:
        """Extract confidence score from a strategy result."""
        # Check for explicit success first
        if result.get('success', False):
            base_confidence = 0.8
        else:
            base_confidence = 0.2
        
        # Adjust based on specific result contents
        if strategy == ProofStrategy.SYMBOLIC_ONLY:
            symbolic_res = result.get('symbolic_result') or result.get('symbolic')
            if symbolic_res and hasattr(symbolic_res, 'confidence_score'):
                return symbolic_res.confidence_score
            
        elif strategy == ProofStrategy.RULE_BASED:
            transformations = result.get('transformations', [])
            if transformations:
                # Higher confidence with more successful transformations
                return min(0.8, 0.3 + len(transformations) * 0.1)
                
        elif strategy == ProofStrategy.FORMAL_VERIFICATION:
            formal_res = result.get('formal_result') or result.get('formal')
            if formal_res:
                if hasattr(formal_res, 'verification_status'):
                    if formal_res.verification_status == "proved":
                        return 0.95
                    elif formal_res.verification_status == "timeout":
                        return 0.1
                return 0.5
                
        elif strategy in [ProofStrategy.HYBRID_SYMBOLIC_RULE, 
                         ProofStrategy.HYBRID_SYMBOLIC_FORMAL,
                         ProofStrategy.COMPREHENSIVE]:
            # For hybrid strategies, average the confidence of sub-methods
            confidences = []
            
            symbolic_res = result.get('symbolic_result') or result.get('symbolic')
            if symbolic_res and hasattr(symbolic_res, 'confidence_score'):
                confidences.append(symbolic_res.confidence_score)
            
            transformations = result.get('transformations', [])
            if transformations:
                confidences.append(min(0.8, 0.3 + len(transformations) * 0.1))
            
            formal_res = result.get('formal_result') or result.get('formal')
            if formal_res and hasattr(formal_res, 'verification_status'):
                if formal_res.verification_status == "proved":
                    confidences.append(0.95)
                elif formal_res.verification_status == "timeout":
                    confidences.append(0.1)
            
            if confidences:
                return sum(confidences) / len(confidences)
        
        return base_confidence
    
    def _extract_method_details(self, strategy: ProofStrategy, result: Dict) -> Dict[str, Any]:
        """Extract detailed information from a strategy result."""
        details = {}
        
        if 'symbolic_result' in result or 'symbolic' in result:
            symbolic_res = result.get('symbolic_result') or result.get('symbolic')
            if symbolic_res:
                details['symbolic_method'] = str(type(symbolic_res).__name__)
                if hasattr(symbolic_res, 'proof_steps'):
                    details['proof_steps_count'] = len(symbolic_res.proof_steps)
        
        if 'transformations' in result:
            transformations = result['transformations']
            details['transformation_count'] = len(transformations)
            if transformations:
                details['transformation_types'] = [str(t) for t in transformations[:3]]  # First 3
        
        if 'formal_result' in result or 'formal' in result:
            formal_res = result.get('formal_result') or result.get('formal')
            if formal_res:
                details['formal_system'] = 'lean4'
                if hasattr(formal_res, 'verification_status'):
                    details['verification_status'] = formal_res.verification_status
        
        return details
    
    def _determine_final_status(self, method_results: List[MethodResult]) -> str:
        """Determine the final proof status based on method results."""
        successful_methods = [r for r in method_results if r.success]
        
        if not successful_methods:
            return "failed"
        
        # Check for high-confidence successes
        high_confidence_successes = [r for r in successful_methods if r.confidence > 0.7]
        
        if high_confidence_successes:
            return "proved"
        
        # Check for moderate confidence or multiple successes
        moderate_successes = [r for r in successful_methods if r.confidence > 0.4]
        
        if len(moderate_successes) >= 2 or (len(moderate_successes) == 1 and moderate_successes[0].confidence > 0.6):
            return "proved"
        
        return "partially_proved"
    
    def _calculate_integrated_confidence(self, method_results: List[MethodResult]) -> float:
        """Calculate integrated confidence score across all methods."""
        if not method_results:
            return 0.0
        
        # Weight-based confidence calculation
        total_weighted_confidence = 0.0
        total_weights = 0.0
        
        for result in method_results:
            # Get method weight (normalize method name to base form)
            base_method = result.method_name.split('_')[0]  # e.g., 'hybrid' -> 'hybrid'
            if result.method_name in self.method_weights:
                weight = self.method_weights[result.method_name]
            else:
                weight = self.method_weights.get(base_method, 0.5)
            
            total_weighted_confidence += result.confidence * weight
            total_weights += weight
        
        if total_weights == 0:
            return 0.0
        
        base_confidence = total_weighted_confidence / total_weights
        
        # Apply bonuses for convergent results
        successful_results = [r for r in method_results if r.success]
        if len(successful_results) >= 2:
            # Multiple successful methods boost confidence
            convergence_bonus = min(0.1, (len(successful_results) - 1) * 0.05)
            base_confidence += convergence_bonus
        
        # Apply penalty for conflicting results
        failed_results = [r for r in method_results if not r.success and r.confidence > 0.3]
        if failed_results and successful_results:
            conflict_penalty = min(0.1, len(failed_results) * 0.03)
            base_confidence -= conflict_penalty
        
        return max(0.0, min(1.0, base_confidence))
    
    def _extract_best_symbolic_result(self, strategy_results: Dict) -> Optional[Any]:
        """Extract the best symbolic result from strategy results."""
        best_symbolic = None
        best_confidence = 0.0
        
        for result in strategy_results.values():
            if not result or 'error' in result:
                continue
                
            symbolic_res = result.get('symbolic_result') or result.get('symbolic')
            if symbolic_res:
                confidence = getattr(symbolic_res, 'confidence_score', 0.5)
                if confidence > best_confidence:
                    best_symbolic = symbolic_res
                    best_confidence = confidence
        
        return best_symbolic
    
    def _extract_best_transformations(self, strategy_results: Dict) -> List[Any]:
        """Extract the best rule transformations from strategy results."""
        best_transformations = []
        max_transformations = 0
        
        for result in strategy_results.values():
            if not result or 'error' in result:
                continue
                
            transformations = result.get('transformations', [])
            if len(transformations) > max_transformations:
                best_transformations = transformations
                max_transformations = len(transformations)
        
        return best_transformations
    
    def _extract_best_formal_result(self, strategy_results: Dict) -> Optional[Any]:
        """Extract the best formal verification result from strategy results."""
        for result in strategy_results.values():
            if not result or 'error' in result:
                continue
                
            formal_res = result.get('formal_result') or result.get('formal')
            if formal_res:
                # Return the first formal result found (typically there's only one)
                return formal_res
        
        return None
    
    def _analyze_results(self, method_results: List[MethodResult]) -> ResultAnalysis:
        """Perform detailed analysis of method results."""
        if not method_results:
            return ResultAnalysis(
                total_methods_attempted=0,
                successful_methods=[],
                failed_methods=[],
                best_method="none",
                worst_method="none",
                average_confidence=0.0,
                confidence_variance=0.0,
                total_execution_time=0.0,
                convergent_results=False
            )
        
        successful_methods = [r.method_name for r in method_results if r.success]
        failed_methods = [r.method_name for r in method_results if not r.success]
        
        # Find best and worst methods by confidence
        best_method = max(method_results, key=lambda r: r.confidence).method_name
        worst_method = min(method_results, key=lambda r: r.confidence).method_name
        
        # Calculate confidence statistics
        confidences = [r.confidence for r in method_results]
        average_confidence = sum(confidences) / len(confidences)
        
        # Calculate variance
        confidence_variance = sum((c - average_confidence) ** 2 for c in confidences) / len(confidences)
        
        # Total execution time
        total_execution_time = sum(r.execution_time for r in method_results)
        
        # Check for convergent results
        convergent_results = self._check_result_convergence(method_results)
        
        return ResultAnalysis(
            total_methods_attempted=len(method_results),
            successful_methods=successful_methods,
            failed_methods=failed_methods,
            best_method=best_method,
            worst_method=worst_method,
            average_confidence=average_confidence,
            confidence_variance=confidence_variance,
            total_execution_time=total_execution_time,
            convergent_results=convergent_results
        )
    
    def _check_result_convergence(self, method_results: List[MethodResult]) -> bool:
        """Check if multiple methods converge on the same result."""
        if len(method_results) < 2:
            return False
        
        successful_methods = [r for r in method_results if r.success]
        failed_methods = [r for r in method_results if not r.success]
        
        # Convergent if majority agree
        total_methods = len(method_results)
        if len(successful_methods) >= total_methods * 0.6:
            return True
        elif len(failed_methods) >= total_methods * 0.6:
            return True
        
        return False
    
    def _collect_error_messages(self, strategy_results: Dict) -> List[str]:
        """Collect error messages from all strategies."""
        error_messages = []
        
        for strategy, result in strategy_results.items():
            if not result:
                error_messages.append(f"{strategy.value}: No result returned")
            elif 'error' in result:
                error_messages.append(f"{strategy.value}: {result['error']}")
        
        return error_messages
    
    def _create_result_metadata(self, analysis: ResultAnalysis, convergent: bool,
                               method_results: List[MethodResult], execution_time: float) -> Dict[str, Any]:
        """Create comprehensive metadata for the result."""
        confidence_level = self._determine_confidence_level(analysis.average_confidence)
        
        return {
            'analysis': analysis.to_dict(),
            'confidence_level': confidence_level.value,
            'integration_timestamp': time.time(),
            'method_details': [r.to_dict() for r in method_results],
            'convergent_analysis': {
                'methods_agree': convergent,
                'agreement_threshold': 0.6,
                'success_ratio': len(analysis.successful_methods) / max(analysis.total_methods_attempted, 1)
            },
            'performance_metrics': {
                'total_execution_time': execution_time,
                'average_method_time': analysis.total_execution_time / max(len(method_results), 1),
                'fastest_method': min(method_results, key=lambda r: r.execution_time).method_name if method_results else None,
                'slowest_method': max(method_results, key=lambda r: r.execution_time).method_name if method_results else None
            }
        }
    
    def _determine_confidence_level(self, confidence_score: float) -> ResultConfidenceLevel:
        """Determine confidence level from numerical score."""
        for level, (min_val, max_val) in self.confidence_thresholds.items():
            if min_val <= confidence_score < max_val:
                return level
        
        # Handle edge case for perfect confidence (1.0)
        if confidence_score >= 0.8:
            return ResultConfidenceLevel.VERY_HIGH
        
        return ResultConfidenceLevel.VERY_LOW
    
    def _update_statistics(self, confidence_score: float, convergent: bool, 
                          method_results: List[MethodResult]):
        """Update integration statistics."""
        if confidence_score >= 0.6:
            self.integration_stats['high_confidence_results'] += 1
        
        if convergent:
            self.integration_stats['convergent_results'] += 1
        
        # Update method performance statistics
        for result in method_results:
            method = result.method_name
            if method not in self.integration_stats['method_performance']:
                self.integration_stats['method_performance'][method] = {
                    'total_attempts': 0,
                    'successes': 0,
                    'average_confidence': 0.0,
                    'total_confidence': 0.0
                }
            
            perf = self.integration_stats['method_performance'][method]
            perf['total_attempts'] += 1
            perf['total_confidence'] += result.confidence
            
            if result.success:
                perf['successes'] += 1
            
            perf['average_confidence'] = perf['total_confidence'] / perf['total_attempts']
    
    def generate_proof_summary(self, result: ComprehensiveResult) -> str:
        """
        Generate a human-readable proof summary.
        
        Args:
            result: ComprehensiveResult to summarize
            
        Returns:
            Formatted proof summary string
        """
        lines = []
        lines.append("=" * 60)
        lines.append(f"PROOF SUMMARY: {result.theorem.id}")
        lines.append("=" * 60)
        
        # Theorem information
        lines.append(f"Theorem: {result.theorem.statement}")
        lines.append(f"Type: {result.theorem.theorem_type.value}")
        lines.append("")
        
        # Result overview
        lines.append(f"Status: {result.final_status.upper()}")
        lines.append(f"Confidence: {result.confidence_score:.3f}")
        
        confidence_level = self._determine_confidence_level(result.confidence_score)
        lines.append(f"Confidence Level: {confidence_level.value.replace('_', ' ').title()}")
        lines.append(f"Execution Time: {result.execution_time:.3f} seconds")
        lines.append("")
        
        # Strategy information
        lines.append("Strategies Attempted:")
        for strategy in result.strategies_attempted:
            lines.append(f"  • {strategy.value}")
        lines.append("")
        
        # Method results
        if 'method_details' in result.metadata:
            lines.append("Method Results:")
            for method in result.metadata['method_details']:
                status = "✓" if method['success'] else "✗"
                lines.append(f"  {status} {method['method_name']}: confidence={method['confidence']:.3f}")
        lines.append("")
        
        # Detailed results
        if result.symbolic_result:
            lines.append("Symbolic Proof: Available")
        
        if result.rule_transformations:
            lines.append(f"Rule Transformations: {len(result.rule_transformations)} steps")
        
        if result.formal_result:
            lines.append("Formal Verification: Available")
        
        lines.append("")
        
        # Analysis
        if 'analysis' in result.metadata:
            analysis = result.metadata['analysis']
            lines.append("Analysis:")
            lines.append(f"  Methods Successful: {len(analysis['successful_methods'])}/{analysis['total_methods_attempted']}")
            lines.append(f"  Convergent Results: {'Yes' if analysis['convergent_results'] else 'No'}")
            lines.append(f"  Best Method: {analysis['best_method']}")
        
        # Error messages
        if result.error_messages:
            lines.append("")
            lines.append("Issues Encountered:")
            for error in result.error_messages:
                lines.append(f"  • {error}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def get_integration_statistics(self) -> Dict[str, Any]:
        """Get comprehensive integration statistics."""
        stats = self.integration_stats.copy()
        
        # Calculate derived metrics
        total = stats['total_integrations']
        if total > 0:
            stats['high_confidence_rate'] = stats['high_confidence_results'] / total
            stats['convergence_rate'] = stats['convergent_results'] / total
        else:
            stats['high_confidence_rate'] = 0.0
            stats['convergence_rate'] = 0.0
        
        # Add method success rates
        for method, perf in stats['method_performance'].items():
            if perf['total_attempts'] > 0:
                perf['success_rate'] = perf['successes'] / perf['total_attempts']
            else:
                perf['success_rate'] = 0.0
        
        return stats
    
    def export_detailed_results(self, results: List[ComprehensiveResult], 
                               filename: str, format: str = 'json') -> str:
        """
        Export detailed results to file.
        
        Args:
            results: List of ComprehensiveResult objects
            filename: Output filename
            format: Export format ('json', 'csv', 'txt')
            
        Returns:
            Path to exported file
        """
        if format == 'json':
            return self._export_json(results, filename)
        elif format == 'csv':
            return self._export_csv(results, filename)
        elif format == 'txt':
            return self._export_txt(results, filename)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_json(self, results: List[ComprehensiveResult], filename: str) -> str:
        """Export results to JSON format."""
        data = {
            'export_timestamp': time.time(),
            'total_results': len(results),
            'integration_statistics': self.get_integration_statistics(),
            'results': [result.to_dict() for result in results]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        return filename
    
    def _export_csv(self, results: List[ComprehensiveResult], filename: str) -> str:
        """Export results to CSV format."""
        import csv
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'theorem_id', 'status', 'confidence_score', 'execution_time',
                'strategies_attempted', 'successful_methods', 'convergent_results'
            ])
            
            # Write data
            for result in results:
                analysis = result.metadata.get('analysis', {})
                writer.writerow([
                    result.theorem.id,
                    result.final_status,
                    result.confidence_score,
                    result.execution_time,
                    ';'.join([s.value for s in result.strategies_attempted]),
                    ';'.join(analysis.get('successful_methods', [])),
                    analysis.get('convergent_results', False)
                ])
        
        return filename
    
    def _export_txt(self, results: List[ComprehensiveResult], filename: str) -> str:
        """Export results to human-readable text format."""
        with open(filename, 'w') as f:
            f.write("MATHBOT PROOF RESULTS SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            for i, result in enumerate(results, 1):
                f.write(f"RESULT {i}\n")
                f.write(self.generate_proof_summary(result))
                f.write("\n\n")
            
            # Add overall statistics
            f.write("OVERALL STATISTICS\n")
            f.write("=" * 60 + "\n")
            stats = self.get_integration_statistics()
            f.write(f"Total Results: {stats['total_integrations']}\n")
            f.write(f"High Confidence Rate: {stats['high_confidence_rate']:.1%}\n")
            f.write(f"Convergence Rate: {stats['convergence_rate']:.1%}\n")
        
        return filename 