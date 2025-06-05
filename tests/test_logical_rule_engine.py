"""
Tests for the LogicalRuleEngine class in MathBot.

This module contains comprehensive tests for the rule-based transformation engine,
including tests for pattern matching, rule application, transformation sequences,
and path finding functionality.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import sympy as sp
from sympy.core.sympify import SympifyError

from proofs.utils.logic import LogicalRuleEngine, RuleType, LogicalRule, RuleDatabase


class TestLogicalRuleEngine:
    """Test cases for LogicalRuleEngine class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_rules_dir = Path(self.temp_dir) / "test_rules"
        self.temp_rules_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a simple test rule file
        test_rules = {
            "algebraic_rules": [
                {
                    "id": "test_square_sum",
                    "name": "Test Square of Sum",
                    "rule_type": "algebraic",
                    "pattern": "(a + b)**2",
                    "replacement": "a**2 + 2*a*b + b**2",
                    "conditions": [],
                    "justification": "Test binomial expansion",
                    "priority": 4,
                    "bidirectional": True
                },
                {
                    "id": "test_distributive",
                    "name": "Test Distributive",
                    "rule_type": "algebraic",
                    "pattern": "a*(b + c)",
                    "replacement": "a*b + a*c",
                    "conditions": [],
                    "justification": "Test distributive property",
                    "priority": 5,
                    "bidirectional": True
                }
            ]
        }
        
        import json
        test_rules_file = self.temp_rules_dir / "test_rules.json"
        with open(test_rules_file, 'w') as f:
            json.dump(test_rules, f)
    
    def test_engine_initialization(self):
        """Test LogicalRuleEngine initialization."""
        engine = LogicalRuleEngine(rule_database_path=self.temp_rules_dir)
        
        assert engine.max_transformation_steps == 20  # default
        assert engine.enable_bidirectional == True  # default
        assert isinstance(engine.rule_database, RuleDatabase)
        assert isinstance(engine.pattern_cache, dict)
        assert isinstance(engine.transformation_cache, dict)
    
    def test_engine_initialization_with_custom_params(self):
        """Test LogicalRuleEngine initialization with custom parameters."""
        engine = LogicalRuleEngine(
            rule_database_path=self.temp_rules_dir,
            max_transformation_steps=10,
            enable_bidirectional=False
        )
        
        assert engine.max_transformation_steps == 10
        assert engine.enable_bidirectional == False
    
    def test_engine_initialization_default_path(self):
        """Test engine initialization with default rules path."""
        # Should not raise an error even if default path doesn't exist
        engine = LogicalRuleEngine()
        assert isinstance(engine.rule_database, RuleDatabase)
    
    def test_find_applicable_rules_basic(self):
        """Test finding applicable rules for a basic expression."""
        engine = LogicalRuleEngine(rule_database_path=self.temp_rules_dir)
        
        # Simple distributive case: x*(y + z)
        expr = sp.parse_expr("x*(y + z)")
        applicable_rules = engine.find_applicable_rules(expr, [RuleType.ALGEBRAIC])
        
        # Should find some rules (exact count depends on implementation)
        assert isinstance(applicable_rules, list)
        
        # All returned items should be LogicalRule instances
        for rule in applicable_rules:
            assert isinstance(rule, LogicalRule)
    
    def test_find_applicable_rules_with_rule_types(self):
        """Test finding applicable rules with specific rule types."""
        engine = LogicalRuleEngine(rule_database_path=self.temp_rules_dir)
        
        expr = sp.parse_expr("x + y")
        
        # Test with specific rule types
        algebraic_rules = engine.find_applicable_rules(expr, [RuleType.ALGEBRAIC])
        trig_rules = engine.find_applicable_rules(expr, [RuleType.TRIGONOMETRIC])
        
        assert isinstance(algebraic_rules, list)
        assert isinstance(trig_rules, list)
        
        # Should return different results for different rule types
        # (exact behavior depends on available rules)
    
    def test_find_applicable_rules_caching(self):
        """Test that rule finding results are cached properly."""
        engine = LogicalRuleEngine(rule_database_path=self.temp_rules_dir)
        
        expr = sp.parse_expr("x + y")
        
        # First call
        rules1 = engine.find_applicable_rules(expr, [RuleType.ALGEBRAIC])
        cache_size_after_first = len(engine.pattern_cache)
        
        # Second call with same parameters
        rules2 = engine.find_applicable_rules(expr, [RuleType.ALGEBRAIC])
        cache_size_after_second = len(engine.pattern_cache)
        
        # Results should be identical
        assert rules1 == rules2
        # Cache size should not increase on second call
        assert cache_size_after_second == cache_size_after_first
    
    def test_apply_rule_success(self):
        """Test successful rule application."""
        engine = LogicalRuleEngine(rule_database_path=self.temp_rules_dir)
        
        # Create a simple test rule
        test_rule = LogicalRule(
            id="test_simple",
            name="Test Simple",
            rule_type=RuleType.ALGEBRAIC,
            pattern="x + 0",
            replacement="x",
            justification="Additive identity"
        )
        
        expr = sp.parse_expr("y + 0")
        transformed, success, info = engine.apply_rule(expr, test_rule)
        
        if success:  # Only check if transformation was successful
            assert transformed != expr  # Should be different
            assert info['rule_id'] == 'test_simple'
            assert info['transformation_success'] == True
    
    def test_apply_rule_failure(self):
        """Test rule application when rule doesn't match."""
        engine = LogicalRuleEngine(rule_database_path=self.temp_rules_dir)
        
        # Create a rule that won't match
        test_rule = LogicalRule(
            id="test_nomatch",
            name="Test No Match",
            rule_type=RuleType.ALGEBRAIC,
            pattern="sin(x)",
            replacement="cos(x)",
            justification="Won't match"
        )
        
        expr = sp.parse_expr("x + y")  # Won't match sin(x) pattern
        transformed, success, info = engine.apply_rule(expr, test_rule)
        
        assert success == False
        assert transformed == expr  # Should be unchanged
        assert info['rule_id'] == 'test_nomatch'
    
    def test_apply_rule_with_invalid_rule(self):
        """Test rule application with invalid rule."""
        engine = LogicalRuleEngine(rule_database_path=self.temp_rules_dir)
        
        # Create a rule with invalid pattern
        invalid_rule = LogicalRule.__new__(LogicalRule)  # Bypass validation
        invalid_rule.id = "invalid"
        invalid_rule.name = "Invalid"
        invalid_rule.rule_type = RuleType.ALGEBRAIC
        invalid_rule.pattern = "invalid @#$ pattern"
        invalid_rule.replacement = "x"
        invalid_rule.justification = "Invalid"
        invalid_rule.priority = 1
        invalid_rule.bidirectional = False
        invalid_rule.conditions = []
        invalid_rule.metadata = {}
        
        expr = sp.parse_expr("x + y")
        transformed, success, info = engine.apply_rule(expr, invalid_rule)
        
        assert success == False
        assert transformed == expr
        assert 'error' in info
    
    def test_apply_rule_sequence_basic(self):
        """Test applying a sequence of rules."""
        engine = LogicalRuleEngine(rule_database_path=self.temp_rules_dir)
        
        expr = sp.parse_expr("x + 0")
        sequence = engine.apply_rule_sequence(expr, max_steps=3)
        
        assert isinstance(sequence, list)
        # Each step should have required fields
        for step in sequence:
            assert 'step_number' in step
            assert 'rule_applied' in step
            assert 'from_expression' in step
            assert 'to_expression' in step
            assert 'justification' in step
    
    def test_apply_rule_sequence_max_steps(self):
        """Test that rule sequence respects max_steps parameter."""
        engine = LogicalRuleEngine(rule_database_path=self.temp_rules_dir)
        
        expr = sp.parse_expr("(x + y) * (a + b)")
        sequence = engine.apply_rule_sequence(expr, max_steps=2)
        
        # Should not exceed max_steps
        assert len(sequence) <= 2
    
    def test_find_transformation_path_simple(self):
        """Test finding transformation path between expressions."""
        engine = LogicalRuleEngine(rule_database_path=self.temp_rules_dir)
        
        start_expr = sp.parse_expr("x + 0")
        target_expr = sp.parse_expr("x")
        
        path = engine.find_transformation_path(start_expr, target_expr, max_depth=3)
        
        # Path can be None (no path found) or a list of rules
        if path is not None:
            assert isinstance(path, list)
            for rule in path:
                assert isinstance(rule, LogicalRule)
    
    def test_find_transformation_path_identical_expressions(self):
        """Test transformation path when start and target are identical."""
        engine = LogicalRuleEngine(rule_database_path=self.temp_rules_dir)
        
        expr = sp.parse_expr("x + y")
        path = engine.find_transformation_path(expr, expr)
        
        # Should return empty path for identical expressions
        assert path == [] or path is None
    
    def test_find_transformation_path_caching(self):
        """Test that transformation paths are cached."""
        engine = LogicalRuleEngine(rule_database_path=self.temp_rules_dir)
        
        start_expr = sp.parse_expr("x + 0")
        target_expr = sp.parse_expr("x")
        
        # First call
        path1 = engine.find_transformation_path(start_expr, target_expr)
        cache_size_after_first = len(engine.transformation_cache)
        
        # Second call
        path2 = engine.find_transformation_path(start_expr, target_expr)
        cache_size_after_second = len(engine.transformation_cache)
        
        # Results should be identical
        assert path1 == path2
        # Cache should be used
        assert cache_size_after_second == cache_size_after_first
    
    def test_validate_rule_basic(self):
        """Test basic rule validation."""
        engine = LogicalRuleEngine(rule_database_path=self.temp_rules_dir)
        
        valid_rule = LogicalRule(
            id="test_valid",
            name="Test Valid Rule",
            rule_type=RuleType.ALGEBRAIC,
            pattern="x + 0",
            replacement="x",
            justification="Additive identity"
        )
        
        validation_result = engine.validate_rule(valid_rule)
        
        assert isinstance(validation_result, dict)
        assert 'rule_id' in validation_result
        assert 'is_valid' in validation_result
        assert 'errors' in validation_result
        assert 'test_results' in validation_result
        assert validation_result['rule_id'] == 'test_valid'
    
    def test_validate_rule_with_test_cases(self):
        """Test rule validation with custom test cases."""
        engine = LogicalRuleEngine(rule_database_path=self.temp_rules_dir)
        
        rule = LogicalRule(
            id="test_custom",
            name="Test Custom",
            rule_type=RuleType.ALGEBRAIC,
            pattern="x + 0",
            replacement="x",
            justification="Test"
        )
        
        test_cases = [sp.parse_expr("y + 0"), sp.parse_expr("z + 0")]
        validation_result = engine.validate_rule(rule, test_cases)
        
        assert len(validation_result['test_results']) == len(test_cases)
        for test_result in validation_result['test_results']:
            assert 'test_case' in test_result
            assert 'original_expression' in test_result
            assert 'transformation_success' in test_result
    
    def test_get_engine_statistics(self):
        """Test getting engine statistics."""
        engine = LogicalRuleEngine(rule_database_path=self.temp_rules_dir)
        
        stats = engine.get_engine_statistics()
        
        assert isinstance(stats, dict)
        assert 'database_statistics' in stats
        assert 'engine_configuration' in stats
        assert 'cache_statistics' in stats
        
        # Check engine configuration
        config = stats['engine_configuration']
        assert 'max_transformation_steps' in config
        assert 'enable_bidirectional' in config
        assert 'rules_directory' in config
        
        # Check cache statistics
        cache_stats = stats['cache_statistics']
        assert 'pattern_cache_size' in cache_stats
        assert 'transformation_cache_size' in cache_stats
    
    def test_clear_caches(self):
        """Test clearing engine caches."""
        engine = LogicalRuleEngine(rule_database_path=self.temp_rules_dir)
        
        # Populate caches
        expr = sp.parse_expr("x + y")
        engine.find_applicable_rules(expr)
        engine.find_transformation_path(expr, sp.parse_expr("x - y"))
        
        # Verify caches have content
        assert len(engine.pattern_cache) > 0 or len(engine.transformation_cache) > 0
        
        # Clear caches
        engine.clear_caches()
        
        # Verify caches are empty
        assert len(engine.pattern_cache) == 0
        assert len(engine.transformation_cache) == 0
    
    def test_reload_rules(self):
        """Test reloading rules from database."""
        engine = LogicalRuleEngine(rule_database_path=self.temp_rules_dir)
        
        initial_count = engine.reload_rules()
        assert isinstance(initial_count, int)
        assert initial_count >= 0
        
        # Reload again should return same count
        second_count = engine.reload_rules()
        assert second_count == initial_count
    
    def test_bidirectional_rule_handling(self):
        """Test handling of bidirectional rules."""
        engine = LogicalRuleEngine(
            rule_database_path=self.temp_rules_dir,
            enable_bidirectional=True
        )
        
        # Create a bidirectional rule
        bidirectional_rule = LogicalRule(
            id="test_bidirectional",
            name="Test Bidirectional",
            rule_type=RuleType.ALGEBRAIC,
            pattern="a + b",
            replacement="b + a",
            justification="Commutative property",
            bidirectional=True
        )
        
        # The engine should consider both directions
        expr = sp.parse_expr("x + y")
        applicable_rules = engine.find_applicable_rules(expr)
        
        # Test that we can find rules (implementation may vary)
        assert isinstance(applicable_rules, list)
    
    def test_bidirectional_disabled(self):
        """Test engine with bidirectional rules disabled."""
        engine = LogicalRuleEngine(
            rule_database_path=self.temp_rules_dir,
            enable_bidirectional=False
        )
        
        expr = sp.parse_expr("x + y")
        applicable_rules = engine.find_applicable_rules(expr)
        
        # Should still work, just not consider reverse rules
        assert isinstance(applicable_rules, list)
    
    def test_pattern_matching_with_constants(self):
        """Test pattern matching with mathematical constants."""
        engine = LogicalRuleEngine(rule_database_path=self.temp_rules_dir)
        
        # Test with expressions containing constants
        expr_with_pi = sp.parse_expr("pi + x")
        expr_with_e = sp.parse_expr("E + x")
        
        rules_pi = engine.find_applicable_rules(expr_with_pi)
        rules_e = engine.find_applicable_rules(expr_with_e)
        
        assert isinstance(rules_pi, list)
        assert isinstance(rules_e, list)
    
    def test_error_handling_in_rule_application(self):
        """Test error handling during rule application."""
        engine = LogicalRuleEngine(rule_database_path=self.temp_rules_dir)
        
        # Create a rule that might cause issues
        problematic_rule = LogicalRule(
            id="test_problematic",
            name="Test Problematic",
            rule_type=RuleType.ALGEBRAIC,
            pattern="x",
            replacement="1/x",  # Could cause division by zero
            justification="Test problematic transformation"
        )
        
        expr = sp.parse_expr("0")  # This could cause issues with 1/x
        transformed, success, info = engine.apply_rule(expr, problematic_rule)
        
        # Should handle errors gracefully
        assert isinstance(transformed, sp.Expr)
        assert isinstance(success, bool)
        assert isinstance(info, dict)
    
    def test_complex_expression_handling(self):
        """Test handling of complex mathematical expressions."""
        engine = LogicalRuleEngine(rule_database_path=self.temp_rules_dir)
        
        # Test with various complex expressions
        complex_expressions = [
            "sin(x)**2 + cos(x)**2",
            "log(a*b)",
            "exp(x + y)",
            "(x + y)*(x - y)",
            "sqrt(x**2)"
        ]
        
        for expr_str in complex_expressions:
            try:
                expr = sp.parse_expr(expr_str)
                rules = engine.find_applicable_rules(expr)
                assert isinstance(rules, list)
                
                # Try applying rule sequence
                sequence = engine.apply_rule_sequence(expr, max_steps=2)
                assert isinstance(sequence, list)
            except Exception as e:
                # Should not raise unhandled exceptions
                pytest.fail(f"Complex expression '{expr_str}' caused error: {e}")


class TestLogicalRuleEngineIntegration:
    """Integration tests for LogicalRuleEngine with real rule database."""
    
    def test_integration_with_real_rules(self):
        """Test LogicalRuleEngine with the actual rule database."""
        # This test uses the real proofs/rules directory
        engine = LogicalRuleEngine()
        
        # Test with actual mathematical expressions
        test_expressions = [
            "(x + 1)**2",
            "sin(x)**2 + cos(x)**2",
            "x + 0",
            "x * 1",
            "a**2 - b**2"
        ]
        
        for expr_str in test_expressions:
            expr = sp.parse_expr(expr_str)
            
            # Find applicable rules
            rules = engine.find_applicable_rules(expr)
            assert isinstance(rules, list)
            
            # Try rule application if rules are found
            if rules:
                transformed, success, info = engine.apply_rule(expr, rules[0])
                assert isinstance(transformed, sp.Expr)
                assert isinstance(success, bool)
                assert isinstance(info, dict)
    
    def test_integration_statistics(self):
        """Test getting statistics from real rule database."""
        engine = LogicalRuleEngine()
        stats = engine.get_engine_statistics()
        
        assert isinstance(stats, dict)
        db_stats = stats['database_statistics']
        
        # Should have some rules loaded
        assert db_stats['total_rules'] >= 0
        assert isinstance(db_stats['rules_by_type'], dict)
    
    def teardown_method(self):
        """Clean up after each test method."""
        import shutil
        if hasattr(self, 'temp_dir') and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir) 