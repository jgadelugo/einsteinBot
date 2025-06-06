import logging
from typing import List, Optional, Dict, Any
from ui.data.proof_models import ProofTraceData, ProofStep, ProofMethodType
from ui.config import UIConfig

class ProofVisualizationService:
    """Service for integrating with Phase 5 proof systems."""
    
    def __init__(self, config: UIConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize proof engines from Phase 5
        self.proof_engine = None
        self.rule_engine = None
        self.formal_systems = {}
        
    def load_proof_data(self, theorem) -> ProofTraceData:
        """Load complete proof trace data for theorem."""
        try:
            trace_data = ProofTraceData(theorem_id=theorem.id)
            
            # Generate symbolic proof steps
            trace_data.symbolic_steps = self._generate_symbolic_steps(theorem)
            
            # Generate rule-based steps  
            trace_data.rule_steps = self._generate_rule_steps(theorem)
            
            # Generate validation steps
            trace_data.validation_steps = self._generate_validation_steps(theorem)
            
            # Determine successful methods
            trace_data.success_methods = self._determine_success_methods(trace_data)
            
            return trace_data
            
        except Exception as e:
            self.logger.error(f"Failed to load proof data for {theorem.id}: {e}")
            return ProofTraceData(theorem_id=theorem.id)
    
    def _generate_symbolic_steps(self, theorem) -> List[ProofStep]:
        """Generate symbolic proof steps using Phase 5B ProofAttemptEngine."""
        steps = []
        
        try:
            # Mock implementation - replace with actual Phase 5B integration
            if hasattr(theorem, 'sympy_expression'):
                steps.append(ProofStep(
                    step_number=1,
                    method_type=ProofMethodType.SYMBOLIC,
                    title="Original Expression",
                    expression_from="Given",
                    expression_to=str(theorem.sympy_expression),
                    justification="Starting expression from theorem statement",
                    confidence=1.0
                ))
                
                # Add symbolic manipulation steps if available
                if hasattr(theorem, 'mathematical_context') and theorem.mathematical_context:
                    context = theorem.mathematical_context
                    if hasattr(context, 'transformation') and context.transformation:
                        steps.append(ProofStep(
                            step_number=2,
                            method_type=ProofMethodType.SYMBOLIC,
                            title="Apply Transformation",
                            expression_from=str(theorem.sympy_expression),
                            expression_to=context.transformation,
                            rule_applied="symbolic_transformation",
                            justification=f"Apply transformation: {context.transformation}",
                            confidence=0.95
                        ))
                
        except Exception as e:
            self.logger.warning(f"Symbolic step generation failed: {e}")
            
        return steps
    
    def _generate_rule_steps(self, theorem) -> List[ProofStep]:
        """Generate rule-based transformation steps using Phase 5C."""
        steps = []
        
        try:
            # Mock implementation - replace with actual Phase 5C integration
            if hasattr(theorem, 'source_lineage'):
                transformation_chain = getattr(theorem.source_lineage, 'transformation_chain', [])
                
                for i, transformation in enumerate(transformation_chain):
                    steps.append(ProofStep(
                        step_number=i + 1,
                        method_type=ProofMethodType.RULE_BASED,
                        title=f"Apply {transformation}",
                        expression_from="Previous step",
                        expression_to="Transformed expression",
                        rule_applied=transformation,
                        justification=f"Applied transformation rule: {transformation}",
                        confidence=0.9
                    ))
                    
        except Exception as e:
            self.logger.warning(f"Rule step generation failed: {e}")
            
        return steps
    
    def _generate_validation_steps(self, theorem) -> List[ProofStep]:
        """Generate validation evidence steps from theorem data."""
        steps = []
        
        try:
            if hasattr(theorem, 'validation_evidence') and theorem.validation_evidence:
                evidence = theorem.validation_evidence
                
                steps.append(ProofStep(
                    step_number=1,
                    method_type=ProofMethodType.VALIDATION,
                    title="Validation Testing",
                    expression_from="Theorem statement",
                    expression_to=f"Validated with {evidence.total_tests} tests",
                    justification=f"Pass rate: {evidence.pass_rate:.2%}",
                    confidence=evidence.pass_rate,
                    metadata={
                        'total_tests': evidence.total_tests,
                        'pass_rate': evidence.pass_rate,
                        'validation_time': evidence.validation_time,
                        'validation_status': evidence.validation_status,
                        'symbols_tested': evidence.symbols_tested
                    }
                ))
                
                # Add individual test steps if symbols were tested
                if evidence.symbols_tested:
                    for i, symbol in enumerate(evidence.symbols_tested[:5]):  # Limit to first 5
                        steps.append(ProofStep(
                            step_number=i + 2,
                            method_type=ProofMethodType.VALIDATION,
                            title=f"Test Symbol: {symbol}",
                            expression_from=f"Symbol {symbol}",
                            expression_to="Validated",
                            justification=f"Symbol validation for {symbol}",
                            confidence=evidence.pass_rate,
                            execution_time=evidence.validation_time / len(evidence.symbols_tested)
                        ))
                
        except Exception as e:
            self.logger.warning(f"Validation step generation failed: {e}")
            
        return steps
    
    def _determine_success_methods(self, trace_data: ProofTraceData) -> List[ProofMethodType]:
        """Determine which proof methods were successful."""
        successful = []
        
        if trace_data.symbolic_steps:
            successful.append(ProofMethodType.SYMBOLIC)
        if trace_data.rule_steps:
            successful.append(ProofMethodType.RULE_BASED)
        if trace_data.validation_steps:
            successful.append(ProofMethodType.VALIDATION)
            
        return successful 