#!/usr/bin/env python3
"""
Test complete rule application with the LogicalRuleEngine.
"""

from proofs.utils.logic import LogicalRuleEngine, RuleType
import sympy as sp

def main():
    print('=== Testing Complete Rule Application ===')

    # Initialize engine
    engine = LogicalRuleEngine()
    
    # Test with square expansion
    expr = sp.parse_expr('(x + 1)**2')
    print(f'Test expression: {expr}')

    # Find applicable rules
    rules = engine.find_applicable_rules(expr, [RuleType.ALGEBRAIC])
    print(f'Found {len(rules)} applicable rules')

    # Find the square expansion rule
    square_rule = None
    for rule in rules:
        if rule.id == 'square_of_sum':
            square_rule = rule
            break
    
    if square_rule:
        print(f'\nTesting square expansion rule: {square_rule.name}')
        
        # Apply the rule
        transformed, success, info = engine.apply_rule(expr, square_rule)
        print(f'Application success: {success}')
        
        if success:
            print(f'Original: {expr}')
            print(f'Transformed: {transformed}')
            
            # Verify correctness
            expected = sp.expand(expr)
            print(f'Expected: {expected}')
            print(f'Correct: {transformed == expected}')
        else:
            print(f'Failure info: {info}')
    
    # Test with other expressions
    test_expressions = [
        "sin(x)**2 + cos(x)**2",  # Should trigger Pythagorean identity
        "x + 0",                   # Additive identity
        "x * 1",                   # Multiplicative identity
        "(a - b)**2"               # Square of difference
    ]
    
    print('\n=== Testing Other Expressions ===')
    for expr_str in test_expressions:
        print(f'\nTesting: {expr_str}')
        expr = sp.parse_expr(expr_str)
        
        # Find and apply first applicable rule
        rules = engine.find_applicable_rules(expr)
        if rules:
            rule = rules[0]  # Take highest priority rule
            transformed, success, info = engine.apply_rule(expr, rule)
            
            if success:
                print(f'  Applied rule: {rule.name}')
                print(f'  {expr} -> {transformed}')
            else:
                print(f'  Rule application failed: {rule.name}')
        else:
            print(f'  No applicable rules found')

if __name__ == '__main__':
    main() 