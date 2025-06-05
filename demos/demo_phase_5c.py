#!/usr/bin/env python3
"""
Demo script for Phase 5C: Logic Rule System

This script demonstrates the core functionality of the rule-based system.
"""

from proofs.utils.logic import RuleDatabase, RuleType, LogicalRule
import sympy as sp


def main():
    print("🎯 Phase 5C: Logic Rule System Demo")
    print("=" * 50)
    
    # Initialize rule database
    print("\n1. Loading Rule Database...")
    db = RuleDatabase('proofs/rules')
    rules = db.load_all_rules()
    
    # Show statistics
    stats = db.get_statistics()
    print(f"✅ Loaded {stats['total_rules']} rules successfully!")
    print(f"📊 Rules by type: {stats['rules_by_type']}")
    print(f"🔄 Bidirectional rules: {stats['bidirectional_rules']}")
    
    # Show sample rules by category
    print("\n2. Sample Rules by Category:")
    for rule_type in [RuleType.ALGEBRAIC, RuleType.TRIGONOMETRIC]:
        type_rules = db.get_rules_by_type(rule_type)
        if type_rules:
            print(f"\n{rule_type.value.upper()} ({len(type_rules)} rules):")
            for rule in type_rules[:3]:  # Show first 3 rules
                bidirectional = "⟷" if rule.bidirectional else "→"
                print(f"  {bidirectional} {rule.name}: {rule.pattern} → {rule.replacement}")
                print(f"    📝 {rule.justification}")
    
    # Demonstrate rule creation and validation
    print("\n3. Rule Creation & Validation:")
    try:
        new_rule = LogicalRule(
            id="commutative_addition",
            name="Commutative Addition",
            rule_type=RuleType.ALGEBRAIC,
            pattern="a + b",
            replacement="b + a",
            justification="Addition is commutative",
            priority=5,
            bidirectional=True
        )
        print(f"✅ Created rule: {new_rule}")
        
        # Test SymPy integration
        pattern_expr = new_rule.get_sympy_pattern()
        replacement_expr = new_rule.get_sympy_replacement()
        print(f"🔍 Pattern variables: {new_rule.get_pattern_variables()}")
        
    except Exception as e:
        print(f"❌ Error creating rule: {e}")
    
    # Test pattern matching
    print("\n4. Pattern Matching:")
    test_expr = sp.parse_expr("x + y")
    print(f"Testing expression: {test_expr}")
    
    matching_rules = []
    for rule in db.get_all_rules():
        if rule.is_applicable_to_expression(test_expr):
            matching_rules.append(rule)
    
    print(f"Found {len(matching_rules)} potentially applicable rules:")
    for rule in matching_rules[:5]:  # Show first 5 matches
        print(f"  📋 {rule.name} (priority: {rule.priority})")
    
    # Database validation
    print("\n5. Database Validation:")
    errors = db.validate_database()
    if errors:
        print(f"⚠️  Found {len(errors)} validation errors:")
        for error in errors:
            print(f"    • {error}")
    else:
        print("✅ Database validation passed - all rules are mathematically sound!")
    
    print("\n🎉 Phase 5C Core Data Structures Implementation Complete!")
    print(f"📈 Summary: {stats['total_rules']} rules across {len([t for t, rules in stats['rules_by_type'].items() if rules > 0])} categories")


if __name__ == "__main__":
    main() 