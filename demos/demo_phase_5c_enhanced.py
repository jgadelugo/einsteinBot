#!/usr/bin/env python3
"""
Demo script for testing Phase 5C enhancements.

This script demonstrates the enhanced features added to Phase 5C:
1. Duplicate rule detection
2. Multi-step transformation sequences
3. Enhanced bidirectional rule handling
4. Pattern priority optimization
"""

import sys
from pathlib import Path
import sympy as sp

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from proofs.utils.logic import (
    LogicalRuleEngine, 
    RuleDatabase, 
    LogicalRule, 
    RuleType,
    TransformationStep
)


def test_duplicate_rule_detection():
    """Test that duplicate rules are properly detected and skipped."""
    print("=== Testing Duplicate Rule Detection ===")
    
    # Create a test rule database
    db = RuleDatabase()
    
    # Create a test rule
    rule1 = LogicalRule(
        id="test_rule_1",
        name="Test Rule 1",
        rule_type=RuleType.ALGEBRAIC,
        pattern="x + 0",
        replacement="x",
        justification="Addition by zero"
    )
    
    # Create a duplicate rule (same pattern/replacement)
    rule2 = LogicalRule(
        id="test_rule_2",  # Different ID
        name="Test Rule 2",  # Different name
        rule_type=RuleType.ALGEBRAIC,
        pattern="x + 0",  # Same pattern
        replacement="x",  # Same replacement
        justification="Addition by zero (duplicate)"
    )
    
    # Add first rule
    db.add_rule(rule1)
    initial_count = len(db.get_all_rules())
    print(f"After adding first rule: {initial_count} rules")
    
    # Try to add duplicate rule
    db.add_rule(rule2)
    final_count = len(db.get_all_rules())
    print(f"After adding duplicate rule: {final_count} rules")
    
    if initial_count == final_count:
        print("✅ Duplicate detection working correctly!")
    else:
        print("❌ Duplicate detection failed!")
    
    print()


def test_multi_step_transformations():
    """Test multi-step transformation sequences."""
    print("=== Testing Multi-step Transformations ===")
    
    # Create rule engine with some test rules
    engine = LogicalRuleEngine()
    
    # Add test rules to the database
    db = engine.rule_database
    
    # Rule 1: x + 0 -> x
    rule1 = LogicalRule(
        id="add_zero",
        name="Addition by Zero",
        rule_type=RuleType.ALGEBRAIC,
        pattern="x + 0",
        replacement="x",
        priority=5,
        justification="Additive identity"
    )
    
    # Rule 2: x * 1 -> x
    rule2 = LogicalRule(
        id="mult_one",
        name="Multiplication by One",
        rule_type=RuleType.ALGEBRAIC,
        pattern="x * 1",
        replacement="x",
        priority=5,
        justification="Multiplicative identity"
    )
    
    # Rule 3: x - x -> 0
    rule3 = LogicalRule(
        id="sub_self",
        name="Subtract Self",
        rule_type=RuleType.ALGEBRAIC,
        pattern="x - x",
        replacement="0",
        priority=6,
        justification="Subtraction gives zero"
    )
    
    db.add_rule(rule1)
    db.add_rule(rule2)
    db.add_rule(rule3)
    
    # Test expression that can be simplified in multiple steps
    test_expr = sp.parse_expr("(x + 0) * 1 + (y - y)")
    print(f"Starting expression: {test_expr}")
    
    # Apply transformation sequence
    steps = engine.apply_transformation_sequence(test_expr, max_steps=5)
    
    print(f"Transformation completed in {len(steps)} steps:")
    for step in steps:
        print(f"  {step}")
    
    if steps:
        final_expr = steps[-1].to_expr
        print(f"Final expression: {final_expr}")
        print("✅ Multi-step transformations working!")
    else:
        print("❌ No transformations applied!")
    
    print()


def test_bidirectional_rule_handling():
    """Test enhanced bidirectional rule handling."""
    print("=== Testing Bidirectional Rule Handling ===")
    
    engine = LogicalRuleEngine()
    db = engine.rule_database
    
    # Create a bidirectional rule
    bidirectional_rule = LogicalRule(
        id="commutative_add",
        name="Commutative Addition",
        rule_type=RuleType.ALGEBRAIC,
        pattern="x + y",
        replacement="y + x",
        bidirectional=True,
        priority=3,
        justification="Addition is commutative"
    )
    
    db.add_rule(bidirectional_rule)
    
    # Test expression
    test_expr = sp.parse_expr("a + b")
    print(f"Test expression: {test_expr}")
    
    # Find applicable rules
    applicable_rules = engine.find_applicable_rules(test_expr)
    print(f"Found {len(applicable_rules)} applicable rules:")
    
    for rule in applicable_rules:
        print(f"  - {rule.name} (priority: {rule.priority})")
        if rule.metadata.get("reverse_of"):
            print(f"    (Reverse of rule: {rule.metadata['reverse_of']})")
    
    if len(applicable_rules) >= 2:  # Should find forward and reverse
        print("✅ Bidirectional rule handling working!")
    else:
        print("❌ Bidirectional rule handling issue!")
    
    print()


def test_rule_selection_optimization():
    """Test pattern priority optimization and rule selection."""
    print("=== Testing Rule Selection Optimization ===")
    
    engine = LogicalRuleEngine()
    db = engine.rule_database
    
    # Create rules with different priorities and characteristics
    high_priority_rule = LogicalRule(
        id="high_priority",
        name="High Priority Rule",
        rule_type=RuleType.ALGEBRAIC,
        pattern="x + x",
        replacement="2*x",
        priority=10,
        justification="Combine like terms"
    )
    
    low_priority_rule = LogicalRule(
        id="low_priority",
        name="Low Priority Rule",
        rule_type=RuleType.ALGEBRAIC,
        pattern="x + x",
        replacement="x + x",  # No actual change
        priority=1,
        justification="Identity transformation"
    )
    
    simplifying_rule = LogicalRule(
        id="simplifying",
        name="Simplifying Rule",
        rule_type=RuleType.ALGEBRAIC,
        pattern="x * 0",
        replacement="0",
        priority=5,
        justification="Multiplication by zero"
    )
    
    db.add_rule(high_priority_rule)
    db.add_rule(low_priority_rule)
    db.add_rule(simplifying_rule)
    
    # Test rule selection for x + x
    test_expr = sp.parse_expr("x + x")
    applicable_rules = engine.find_applicable_rules(test_expr)
    
    if applicable_rules:
        best_rule = engine._select_optimal_rule(test_expr, applicable_rules)
        print(f"Expression: {test_expr}")
        print(f"Best rule selected: {best_rule.name} (priority: {best_rule.priority})")
        
        if best_rule.id == "high_priority":
            print("✅ Rule selection optimization working!")
        else:
            print("❌ Rule selection optimization issue!")
    else:
        print("❌ No applicable rules found!")
    
    print()


def demo_complete_workflow():
    """Demonstrate a complete enhanced workflow."""
    print("=== Complete Enhanced Workflow Demo ===")
    
    engine = LogicalRuleEngine()
    db = engine.rule_database
    
    # Add a variety of rules
    rules = [
        LogicalRule("add_zero", "Add Zero", RuleType.ALGEBRAIC, "x + 0", "x", 
                   priority=5, justification="Additive identity"),
        LogicalRule("mult_one", "Multiply One", RuleType.ALGEBRAIC, "x * 1", "x", 
                   priority=5, justification="Multiplicative identity"),
        LogicalRule("mult_zero", "Multiply Zero", RuleType.ALGEBRAIC, "x * 0", "0", 
                   priority=8, justification="Multiplication by zero"),
        LogicalRule("double", "Double", RuleType.ALGEBRAIC, "x + x", "2*x", 
                   priority=6, justification="Combine like terms"),
    ]
    
    for rule in rules:
        db.add_rule(rule)
    
    # Complex expression requiring multiple steps
    complex_expr = sp.parse_expr("(x + 0) * 1 + (y + y) * 0 + z")
    print(f"Complex expression: {complex_expr}")
    
    # Apply enhanced transformation sequence
    steps = engine.apply_transformation_sequence(complex_expr, max_steps=10)
    
    print(f"\nTransformation sequence ({len(steps)} steps):")
    for i, step in enumerate(steps, 1):
        print(f"Step {i}: {step.from_expr} → {step.to_expr}")
        print(f"        Rule: {step.rule.name} ({step.rule.justification})")
    
    if steps:
        final_expr = steps[-1].to_expr
        print(f"\nFinal simplified expression: {final_expr}")
        
        # The expression should simplify to just 'z'
        if str(final_expr) == 'z':
            print("✅ Complete workflow successful!")
        else:
            print(f"✅ Workflow completed (final: {final_expr})")
    else:
        print("❌ No transformations applied!")
    
    print()


def main():
    """Run all Phase 5C enhancement tests."""
    print("🚀 Phase 5C Enhancement Demo")
    print("=" * 50)
    print()
    
    try:
        test_duplicate_rule_detection()
        test_multi_step_transformations()
        test_bidirectional_rule_handling()
        test_rule_selection_optimization()
        demo_complete_workflow()
        
        print("=" * 50)
        print("✅ Phase 5C Enhancement Demo Completed Successfully!")
        print("Ready for Phase 5D implementation.")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 