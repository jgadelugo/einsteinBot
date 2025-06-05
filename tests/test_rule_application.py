#!/usr/bin/env python3
"""
Debug rule application issues in LogicalRuleEngine.
"""

from proofs.utils.logic import LogicalRuleEngine, RuleType
import sympy as sp

def main():
    print('=== Testing Rule Application ===')

    # Initialize engine
    engine = LogicalRuleEngine()
    
    # Test with a simple expression
    expr = sp.parse_expr('(x + 1)**2')
    print(f'Test expression: {expr}')

    # Find applicable rules
    rules = engine.find_applicable_rules(expr, [RuleType.ALGEBRAIC])
    print(f'Found {len(rules)} applicable rules')

    if rules:
        # Test the first few rules
        for i, rule in enumerate(rules[:3]):
            print(f'\n--- Testing rule {i+1}: {rule.name} ---')
            print(f'Rule pattern: {rule.pattern}')
            print(f'Rule replacement: {rule.replacement}')
            
            # Apply the rule
            transformed, success, info = engine.apply_rule(expr, rule)
            print(f'Application success: {success}')
            
            if success:
                print(f'Original: {expr}')
                print(f'Transformed: {transformed}')
                print(f'Matches found: {info.get("matches_found", 0)}')
            else:
                print(f'Failure reason: {info.get("error", "No error message")}')
                
            # Test pattern matching directly
            try:
                pattern_expr = rule.get_sympy_pattern()
                wild_pattern = engine._convert_to_wild_pattern(pattern_expr)
                
                print(f'Original pattern: {pattern_expr}')
                print(f'Wild pattern: {wild_pattern}')
                
                # Test wild pattern matching
                wild_match = expr.match(wild_pattern)
                print(f'Wild match result: {wild_match}')
                
                if wild_match:
                    # Try manual substitution
                    replacement_expr = rule.get_sympy_replacement()
                    print(f'Replacement expr: {replacement_expr}')
                    
                    # Apply substitution manually
                    try:
                        manual_result = replacement_expr.subs(wild_match)
                        print(f'Manual substitution result: {manual_result}')
                    except Exception as e:
                        print(f'Manual substitution error: {e}')
                        
            except Exception as e:
                print(f'Pattern analysis error: {e}')

if __name__ == '__main__':
    main() 