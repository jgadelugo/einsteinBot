#!/usr/bin/env python3
"""
Test complete Phase 5C integration with real theorems.
"""

from proofs.proof_attempt import ProofAttemptEngine
from proofs.theorem_generator import Theorem, TheoremType
import sympy as sp
import json

def main():
    print('=== Testing Phase 5C Integration ===')

    # Initialize proof engine
    engine = ProofAttemptEngine()
    
    # Test with the perfect square theorem from results
    perfect_square_theorem = Theorem(
        id="THM_PHASE5C_TEST",
        statement="∀x ∈ ℝ, (x + 1)² = x² + 2x + 1",
        sympy_expression=sp.Eq((sp.Symbol('x') + 1)**2, sp.Symbol('x')**2 + 2*sp.Symbol('x') + 1),
        theorem_type=TheoremType.ALGEBRAIC_IDENTITY,
        assumptions=["x ∈ ℝ"],
        source_lineage={
            "original_formula": "(x + 1)**2",
            "hypothesis_id": "phase5c_test",
            "confidence": 1.0,
            "validation_score": 1.0,
            "generation_method": "manual_test",
            "source_type": "algebraic_identity"
        },
        natural_language="The square of (x+1) expands to x² + 2x + 1"
    )
    
    print(f'Testing theorem: {perfect_square_theorem.statement}')
    print(f'Expression: {perfect_square_theorem.sympy_expression}')
    
    # Attempt proof
    result = engine.attempt_proof(perfect_square_theorem)
    
    print(f'\n=== Proof Result ===')
    print(f'Status: {result.status.value}')
    print(f'Method: {result.method.value}')
    print(f'Confidence: {result.confidence_score:.2f}')
    print(f'Execution time: {result.execution_time:.4f}s')
    print(f'Steps: {len(result.steps)}')
    
    if result.steps:
        print('\n=== Proof Steps ===')
        for i, step in enumerate(result.steps, 1):
            print(f'Step {i}: {step.method}')
            print(f'  From: {step.from_expression}')
            print(f'  To: {step.to_expression}')
            print(f'  Rule: {step.transformation_rule}')
            print(f'  Justification: {step.justification}')
            print()
    
    # Test with Pythagorean identity
    pythagorean_theorem = Theorem(
        id="THM_PYTHAGOREAN_TEST",
        statement="∀x ∈ ℝ, sin²(x) + cos²(x) = 1",
        sympy_expression=sp.Eq(sp.sin(sp.Symbol('x'))**2 + sp.cos(sp.Symbol('x'))**2, 1),
        theorem_type=TheoremType.TRIGONOMETRIC,
        assumptions=["x ∈ ℝ"],
        source_lineage={
            "original_formula": "sin(x)**2 + cos(x)**2",
            "hypothesis_id": "pythagorean_test",
            "confidence": 1.0,
            "validation_score": 1.0,
            "generation_method": "manual_test",
            "source_type": "trigonometric"
        },
        natural_language="The fundamental Pythagorean identity"
    )
    
    print(f'\n=== Testing Trigonometric Identity ===')
    print(f'Testing: {pythagorean_theorem.statement}')
    
    result2 = engine.attempt_proof(pythagorean_theorem)
    print(f'Status: {result2.status.value}')
    print(f'Method: {result2.method.value}')
    print(f'Confidence: {result2.confidence_score:.2f}')
    
    if result2.steps:
        print('Proof steps found:', len(result2.steps))
    
    # Get engine statistics
    print(f'\n=== Engine Statistics ===')
    stats = engine.get_statistics()
    print(f'Total attempts: {stats["total_attempts"]}')
    print(f'Successful proofs: {stats["successful_proofs"]}')
    print(f'Success rate: {stats.get("overall_success_rate", 0):.1%}')
    
    # Test LogicalRuleEngine directly
    print(f'\n=== Direct Rule Engine Test ===')
    from proofs.utils.logic import LogicalRuleEngine
    
    rule_engine = LogicalRuleEngine()
    rule_stats = rule_engine.get_engine_statistics()
    
    print(f'Total rules loaded: {rule_stats["database_statistics"]["total_rules"]}')
    print(f'Rules by type: {rule_stats["database_statistics"]["rules_by_type"]}')
    
    # Test rule application on square expansion
    expr = sp.parse_expr("(x + 1)**2")
    rules = rule_engine.find_applicable_rules(expr)
    print(f'Applicable rules for {expr}: {len(rules)}')
    
    if rules:
        # Apply the square expansion rule
        square_rule = None
        for rule in rules:
            if 'square' in rule.name.lower() and 'sum' in rule.name.lower():
                square_rule = rule
                break
        
        if square_rule:
            transformed, success, info = rule_engine.apply_rule(expr, square_rule)
            if success:
                print(f'Rule application: {expr} -> {transformed}')
                print(f'Mathematically correct: {transformed == sp.expand(expr)}')

if __name__ == '__main__':
    main() 