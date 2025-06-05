#!/usr/bin/env python3
"""
Debug pattern matching issues in LogicalRuleEngine.
"""

import sympy as sp
from proofs.utils.logic import RuleDatabase, RuleType

def main():
    # Load rules and check them
    db = RuleDatabase()
    db.load_all_rules()
    algebraic_rules = db.get_rules_by_type(RuleType.ALGEBRAIC)

    print('Algebraic rules:')
    for rule in algebraic_rules[:5]:
        print(f'  {rule.id}: {rule.pattern} -> {rule.replacement}')

    # Test pattern matching manually
    expr = sp.parse_expr('(x + 1)**2')
    print(f'\nTesting expression: {expr}')

    # Try each rule manually
    for rule in algebraic_rules[:3]:
        print(f'\n--- Testing rule: {rule.name} ---')
        try:
            pattern = rule.get_sympy_pattern()
            print(f'Rule pattern: {pattern}')
            print(f'Pattern variables: {pattern.free_symbols}')
            print(f'Expression variables: {expr.free_symbols}')
            
            # Test match
            match_result = expr.match(pattern)
            print(f'Match result: {match_result}')
            
            # Try reverse match
            reverse_match = pattern.match(expr)
            print(f'Reverse match: {reverse_match}')
            
        except Exception as e:
            print(f'Error testing rule: {e}')

    # Test specific square of sum rule
    print('\n--- Testing specific square expansion ---')
    square_pattern = sp.parse_expr('(a + b)**2')
    print(f'Square pattern: {square_pattern}')
    match_result = expr.match(square_pattern)
    print(f'Square match result: {match_result}')
    
    # Manual substitution test
    print('\n--- Manual substitution test ---')
    a, b = sp.symbols('a b')
    pattern = (a + b)**2
    test_expr = (sp.Symbol('x') + 1)**2
    
    print(f'Pattern: {pattern}')
    print(f'Test expr: {test_expr}')
    
    # Try explicit matching
    match_dict = test_expr.match(pattern)
    print(f'Explicit match: {match_dict}')
    
    # Test with Wild symbols for more flexible matching
    print('\n--- Using Wild symbols ---')
    from sympy import Wild
    w1, w2 = Wild('w1'), Wild('w2')
    wild_pattern = (w1 + w2)**2
    wild_match = test_expr.match(wild_pattern)
    print(f'Wild pattern: {wild_pattern}')
    print(f'Wild match: {wild_match}')

if __name__ == '__main__':
    main() 