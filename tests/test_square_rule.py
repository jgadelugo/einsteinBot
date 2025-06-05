#!/usr/bin/env python3
"""
Test specifically the square expansion rule.
"""

from proofs.utils.logic import LogicalRuleEngine, RuleType
import sympy as sp

def main():
    print('=== Testing Square Expansion Rule ===')

    # Initialize engine
    engine = LogicalRuleEngine()
    
    # Find square expansion rule
    all_rules = engine.rule_database.get_all_rules()
    square_rules = [rule for rule in all_rules if 'square' in rule.name.lower()]
    
    print(f'Found {len(square_rules)} square-related rules:')
    for rule in square_rules:
        print(f'  {rule.id}: {rule.name} | {rule.pattern} -> {rule.replacement}')
    
    # Test with square expansion
    expr = sp.parse_expr('(x + 1)**2')
    print(f'\nTest expression: {expr}')
    
    # Find the specific rule we want
    expansion_rule = None
    for rule in square_rules:
        if rule.pattern == "(a + b)**2":
            expansion_rule = rule
            break
    
    if expansion_rule:
        print(f'\nTesting expansion rule: {expansion_rule.name}')
        print(f'Pattern: {expansion_rule.pattern}')
        print(f'Replacement: {expansion_rule.replacement}')
        
        # Test direct SymPy matching
        pattern = sp.parse_expr('(a + b)**2')
        print(f'SymPy pattern: {pattern}')
        
        # Test with exact variable names
        a, b = sp.symbols('a b')
        test_expr = (a + b)**2
        match1 = test_expr.match(pattern)
        print(f'Exact match test: {test_expr} matches {pattern} = {match1}')
        
        # Test with different variables
        x = sp.Symbol('x')
        test_expr2 = (x + 1)**2
        match2 = test_expr2.match(pattern)
        print(f'Different variables: {test_expr2} matches {pattern} = {match2}')
        
        # Test with Wild symbols
        from sympy import Wild
        w1, w2 = Wild('w1'), Wild('w2')
        wild_pattern = (w1 + w2)**2
        match3 = test_expr2.match(wild_pattern)
        print(f'Wild pattern: {test_expr2} matches {wild_pattern} = {match3}')
        
        if match3:
            # Apply substitution
            replacement = sp.parse_expr('a**2 + 2*a*b + b**2')
            # Map wild symbols to original variables
            substitution = {}
            substitution[sp.Symbol('a')] = match3[w1]
            substitution[sp.Symbol('b')] = match3[w2]
            
            result = replacement.subs(substitution)
            print(f'Substitution result: {result}')
            
            # Expand the original for comparison
            expanded_original = sp.expand(test_expr2)
            print(f'Original expanded: {expanded_original}')
            print(f'Match: {result == expanded_original}')

if __name__ == '__main__':
    main() 