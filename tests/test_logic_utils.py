"""
Tests for the logical rule system in MathBot.

This module contains comprehensive tests for the rule-based transformation system,
including tests for RuleType enum, LogicalRule dataclass, and RuleDatabase manager.
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch
from sympy.core.sympify import SympifyError

from proofs.utils.logic import RuleType, LogicalRule, RuleDatabase


class TestRuleType:
    """Test cases for RuleType enumeration."""
    
    def test_rule_type_values(self):
        """Test that RuleType has correct values."""
        assert RuleType.ALGEBRAIC.value == "algebraic"
        assert RuleType.TRIGONOMETRIC.value == "trigonometric"
        assert RuleType.LOGARITHMIC.value == "logarithmic"
        assert RuleType.EXPONENTIAL.value == "exponential"
        assert RuleType.CALCULUS.value == "calculus"
        assert RuleType.LOGICAL.value == "logical"
        assert RuleType.SUBSTITUTION.value == "substitution"
    
    def test_rule_type_string_representation(self):
        """Test string representation of RuleType."""
        assert str(RuleType.ALGEBRAIC) == "algebraic"
        assert str(RuleType.TRIGONOMETRIC) == "trigonometric"
    
    def test_from_string_valid(self):
        """Test creating RuleType from valid string."""
        assert RuleType.from_string("algebraic") == RuleType.ALGEBRAIC
        assert RuleType.from_string("TRIGONOMETRIC") == RuleType.TRIGONOMETRIC
        assert RuleType.from_string("Logarithmic") == RuleType.LOGARITHMIC
    
    def test_from_string_invalid(self):
        """Test creating RuleType from invalid string."""
        with pytest.raises(ValueError, match="Invalid rule type"):
            RuleType.from_string("invalid_type")
        
        with pytest.raises(ValueError, match="Invalid rule type"):
            RuleType.from_string("")


class TestLogicalRule:
    """Test cases for LogicalRule dataclass."""
    
    def test_valid_rule_creation(self):
        """Test creating a valid LogicalRule."""
        rule = LogicalRule(
            id="test_rule",
            name="Test Rule",
            rule_type=RuleType.ALGEBRAIC,
            pattern="a + b",
            replacement="b + a",
            justification="Commutative property"
        )
        
        assert rule.id == "test_rule"
        assert rule.name == "Test Rule"
        assert rule.rule_type == RuleType.ALGEBRAIC
        assert rule.pattern == "a + b"
        assert rule.replacement == "b + a"
        assert rule.justification == "Commutative property"
        assert rule.priority == 1  # Default
        assert rule.bidirectional == False  # Default
        assert rule.conditions == []  # Default
        assert rule.metadata == {}  # Default
    
    def test_rule_with_all_fields(self):
        """Test creating rule with all optional fields."""
        rule = LogicalRule(
            id="complex_rule",
            name="Complex Rule",
            rule_type=RuleType.TRIGONOMETRIC,
            pattern="sin(x)**2 + cos(x)**2",
            replacement="1",
            conditions=["x is real"],
            justification="Pythagorean identity",
            priority=5,
            bidirectional=True,
            metadata={"category": "identity", "difficulty": "basic"}
        )
        
        assert rule.conditions == ["x is real"]
        assert rule.priority == 5
        assert rule.bidirectional == True
        assert rule.metadata == {"category": "identity", "difficulty": "basic"}
    
    def test_invalid_rule_id(self):
        """Test rule creation with invalid ID."""
        # Empty ID
        with pytest.raises(ValueError, match="Rule ID cannot be empty"):
            LogicalRule(
                id="",
                name="Test",
                rule_type=RuleType.ALGEBRAIC,
                pattern="a",
                replacement="b"
            )
        
        # ID with invalid characters
        with pytest.raises(ValueError, match="Rule ID must start with letter"):
            LogicalRule(
                id="123invalid",
                name="Test",
                rule_type=RuleType.ALGEBRAIC,
                pattern="a",
                replacement="b"
            )
        
        # ID with spaces
        with pytest.raises(ValueError, match="Rule ID must start with letter"):
            LogicalRule(
                id="invalid id",
                name="Test",
                rule_type=RuleType.ALGEBRAIC,
                pattern="a",
                replacement="b"
            )
    
    def test_invalid_rule_name(self):
        """Test rule creation with invalid name."""
        with pytest.raises(ValueError, match="Rule name cannot be empty"):
            LogicalRule(
                id="test_rule",
                name="",
                rule_type=RuleType.ALGEBRAIC,
                pattern="a",
                replacement="b"
            )
    
    def test_invalid_rule_type(self):
        """Test rule creation with invalid rule type."""
        with pytest.raises(ValueError, match="rule_type must be RuleType enum"):
            LogicalRule(
                id="test_rule",
                name="Test",
                rule_type="invalid_type",  # Should be RuleType enum
                pattern="a",
                replacement="b"
            )
    
    def test_invalid_pattern(self):
        """Test rule creation with invalid pattern."""
        with pytest.raises((SympifyError, ValueError), match="Invalid pattern expression|Error from parse_expr"):
            LogicalRule(
                id="test_rule",
                name="Test",
                rule_type=RuleType.ALGEBRAIC,
                pattern="invalid @#$ pattern",
                replacement="b"
            )
    
    def test_invalid_replacement(self):
        """Test rule creation with invalid replacement."""
        with pytest.raises((SympifyError, ValueError), match="Invalid replacement expression|Error from parse_expr"):
            LogicalRule(
                id="test_rule",
                name="Test",
                rule_type=RuleType.ALGEBRAIC,
                pattern="a",
                replacement="invalid @#$ replacement"
            )
    
    def test_negative_priority(self):
        """Test rule creation with negative priority."""
        with pytest.raises(ValueError, match="Rule priority must be non-negative"):
            LogicalRule(
                id="test_rule",
                name="Test",
                rule_type=RuleType.ALGEBRAIC,
                pattern="a",
                replacement="b",
                priority=-1
            )
    
    def test_to_dict(self):
        """Test converting rule to dictionary."""
        rule = LogicalRule(
            id="test_rule",
            name="Test Rule",
            rule_type=RuleType.ALGEBRAIC,
            pattern="a + b",
            replacement="b + a",
            priority=3
        )
        
        rule_dict = rule.to_dict()
        
        assert rule_dict["id"] == "test_rule"
        assert rule_dict["name"] == "Test Rule"
        assert rule_dict["rule_type"] == "algebraic"  # Should be string, not enum
        assert rule_dict["pattern"] == "a + b"
        assert rule_dict["replacement"] == "b + a"
        assert rule_dict["priority"] == 3
    
    def test_from_dict_valid(self):
        """Test creating rule from valid dictionary."""
        rule_data = {
            "id": "test_rule",
            "name": "Test Rule",
            "rule_type": "algebraic",
            "pattern": "a + b",
            "replacement": "b + a",
            "conditions": ["a is real"],
            "justification": "Commutative property",
            "priority": 3,
            "bidirectional": True,
            "metadata": {"category": "basic"}
        }
        
        rule = LogicalRule.from_dict(rule_data)
        
        assert rule.id == "test_rule"
        assert rule.name == "Test Rule"
        assert rule.rule_type == RuleType.ALGEBRAIC
        assert rule.pattern == "a + b"
        assert rule.replacement == "b + a"
        assert rule.conditions == ["a is real"]
        assert rule.justification == "Commutative property"
        assert rule.priority == 3
        assert rule.bidirectional == True
        assert rule.metadata == {"category": "basic"}
    
    def test_from_dict_minimal(self):
        """Test creating rule from minimal dictionary."""
        rule_data = {
            "id": "minimal_rule",
            "name": "Minimal Rule",
            "rule_type": "trigonometric",
            "pattern": "sin(x)",
            "replacement": "sin(x)"
        }
        
        rule = LogicalRule.from_dict(rule_data)
        
        assert rule.id == "minimal_rule"
        assert rule.name == "Minimal Rule"
        assert rule.rule_type == RuleType.TRIGONOMETRIC
        assert rule.conditions == []  # Default
        assert rule.justification == ""  # Default
        assert rule.priority == 1  # Default
        assert rule.bidirectional == False  # Default
        assert rule.metadata == {}  # Default
    
    def test_from_dict_missing_keys(self):
        """Test creating rule from dictionary with missing required keys."""
        rule_data = {
            "id": "test_rule",
            "name": "Test Rule"
            # Missing rule_type, pattern, replacement
        }
        
        with pytest.raises(KeyError, match="Missing required keys"):
            LogicalRule.from_dict(rule_data)
    
    def test_from_dict_invalid_rule_type(self):
        """Test creating rule from dictionary with invalid rule type."""
        rule_data = {
            "id": "test_rule",
            "name": "Test Rule",
            "rule_type": "invalid_type",
            "pattern": "a",
            "replacement": "b"
        }
        
        with pytest.raises(ValueError, match="Invalid rule type"):
            LogicalRule.from_dict(rule_data)
    
    def test_get_sympy_expressions(self):
        """Test getting SymPy expressions from rule."""
        rule = LogicalRule(
            id="test_rule",
            name="Test Rule",
            rule_type=RuleType.ALGEBRAIC,
            pattern="x**2 + 2*x + 1",
            replacement="(x + 1)**2"
        )
        
        pattern_expr = rule.get_sympy_pattern()
        replacement_expr = rule.get_sympy_replacement()
        
        assert str(pattern_expr) == "x**2 + 2*x + 1"
        assert str(replacement_expr) == "(x + 1)**2"
    
    def test_get_pattern_variables(self):
        """Test extracting variables from pattern."""
        rule = LogicalRule(
            id="test_rule",
            name="Test Rule",
            rule_type=RuleType.ALGEBRAIC,
            pattern="a*x + b*y + c",
            replacement="c + b*y + a*x"
        )
        
        variables = rule.get_pattern_variables()
        assert variables == {"a", "b", "c", "x", "y"}
    
    def test_is_applicable_to_expression(self):
        """Test checking if rule is applicable to expression."""
        import sympy as sp
        
        rule = LogicalRule(
            id="test_rule",
            name="Test Rule",
            rule_type=RuleType.ALGEBRAIC,
            pattern="a + b",
            replacement="b + a"
        )
        
        # Should be applicable to expressions with compatible structure
        expr1 = sp.parse_expr("x + y")
        assert rule.is_applicable_to_expression(expr1) == True
        
        # Should not be applicable to constants when pattern has variables
        expr2 = sp.parse_expr("5")
        assert rule.is_applicable_to_expression(expr2) == False
    
    def test_rule_equality_and_hashing(self):
        """Test rule equality and hashing based on ID."""
        rule1 = LogicalRule(
            id="test_rule",
            name="Test Rule 1",
            rule_type=RuleType.ALGEBRAIC,
            pattern="a",
            replacement="b"
        )
        
        rule2 = LogicalRule(
            id="test_rule",  # Same ID
            name="Test Rule 2",  # Different name
            rule_type=RuleType.TRIGONOMETRIC,  # Different type
            pattern="c",  # Different pattern
            replacement="d"  # Different replacement
        )
        
        rule3 = LogicalRule(
            id="different_rule",
            name="Test Rule 3",
            rule_type=RuleType.ALGEBRAIC,
            pattern="a",
            replacement="b"
        )
        
        # Same ID should be equal
        assert rule1 == rule2
        assert hash(rule1) == hash(rule2)
        
        # Different ID should not be equal
        assert rule1 != rule3
        assert hash(rule1) != hash(rule3)
    
    def test_rule_string_representations(self):
        """Test string representations of rules."""
        rule = LogicalRule(
            id="test_rule",
            name="Test Rule",
            rule_type=RuleType.ALGEBRAIC,
            pattern="a + b",
            replacement="b + a",
            priority=3
        )
        
        str_repr = str(rule)
        assert "test_rule" in str_repr
        assert "Test Rule" in str_repr
        assert "algebraic" in str_repr
        
        repr_str = repr(rule)
        assert "test_rule" in repr_str
        assert "Test Rule" in repr_str
        assert "a + b" in repr_str
        assert "b + a" in repr_str
        assert "priority=3" in repr_str


class TestRuleDatabase:
    """Test cases for RuleDatabase class."""
    
    def test_database_initialization(self):
        """Test database initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db = RuleDatabase(temp_dir)
            
            assert db.rules_dir == Path(temp_dir)
            assert len(db.rules) == 7  # One entry for each RuleType
            assert len(db.rule_index) == 0
            
            # Check that all rule types are initialized
            for rule_type in RuleType:
                assert rule_type in db.rules
                assert db.rules[rule_type] == []
    
    def test_add_rule(self):
        """Test adding a rule to the database."""
        db = RuleDatabase()
        
        rule = LogicalRule(
            id="test_rule",
            name="Test Rule",
            rule_type=RuleType.ALGEBRAIC,
            pattern="a + b",
            replacement="b + a"
        )
        
        db.add_rule(rule)
        
        assert len(db.rule_index) == 1
        assert "test_rule" in db.rule_index
        assert db.rule_index["test_rule"] == rule
        assert len(db.rules[RuleType.ALGEBRAIC]) == 1
        assert rule in db.rules[RuleType.ALGEBRAIC]
    
    def test_add_duplicate_rule(self):
        """Test adding a rule with duplicate ID."""
        db = RuleDatabase()
        
        rule1 = LogicalRule(
            id="duplicate_id",
            name="Rule 1",
            rule_type=RuleType.ALGEBRAIC,
            pattern="a",
            replacement="b"
        )
        
        rule2 = LogicalRule(
            id="duplicate_id",  # Same ID
            name="Rule 2",
            rule_type=RuleType.TRIGONOMETRIC,
            pattern="c",
            replacement="d"
        )
        
        db.add_rule(rule1)
        
        with pytest.raises(ValueError, match="Rule with ID 'duplicate_id' already exists"):
            db.add_rule(rule2)
    
    def test_get_rule_by_id(self):
        """Test retrieving rule by ID."""
        db = RuleDatabase()
        
        rule = LogicalRule(
            id="test_rule",
            name="Test Rule",
            rule_type=RuleType.ALGEBRAIC,
            pattern="a",
            replacement="b"
        )
        
        db.add_rule(rule)
        
        retrieved_rule = db.get_rule_by_id("test_rule")
        assert retrieved_rule == rule
        
        non_existent = db.get_rule_by_id("non_existent")
        assert non_existent is None
    
    def test_get_rules_by_type(self):
        """Test retrieving rules by type."""
        db = RuleDatabase()
        
        algebraic_rule = LogicalRule(
            id="algebraic_rule",
            name="Algebraic Rule",
            rule_type=RuleType.ALGEBRAIC,
            pattern="a",
            replacement="b"
        )
        
        trig_rule = LogicalRule(
            id="trig_rule",
            name="Trigonometric Rule",
            rule_type=RuleType.TRIGONOMETRIC,
            pattern="sin(x)",
            replacement="sin(x)"
        )
        
        db.add_rule(algebraic_rule)
        db.add_rule(trig_rule)
        
        algebraic_rules = db.get_rules_by_type(RuleType.ALGEBRAIC)
        assert len(algebraic_rules) == 1
        assert algebraic_rule in algebraic_rules
        
        trig_rules = db.get_rules_by_type(RuleType.TRIGONOMETRIC)
        assert len(trig_rules) == 1
        assert trig_rule in trig_rules
        
        # Test empty type
        empty_rules = db.get_rules_by_type(RuleType.LOGARITHMIC)
        assert len(empty_rules) == 0
    
    def test_get_all_rules(self):
        """Test retrieving all rules."""
        db = RuleDatabase()
        
        rule1 = LogicalRule(
            id="rule1",
            name="Rule 1",
            rule_type=RuleType.ALGEBRAIC,
            pattern="a",
            replacement="b"
        )
        
        rule2 = LogicalRule(
            id="rule2",
            name="Rule 2",
            rule_type=RuleType.TRIGONOMETRIC,
            pattern="sin(x)",
            replacement="sin(x)"
        )
        
        db.add_rule(rule1)
        db.add_rule(rule2)
        
        all_rules = db.get_all_rules()
        assert len(all_rules) == 2
        assert rule1 in all_rules
        assert rule2 in all_rules
    
    def test_remove_rule(self):
        """Test removing a rule from the database."""
        db = RuleDatabase()
        
        rule = LogicalRule(
            id="test_rule",
            name="Test Rule",
            rule_type=RuleType.ALGEBRAIC,
            pattern="a",
            replacement="b"
        )
        
        db.add_rule(rule)
        assert len(db.rule_index) == 1
        
        # Remove existing rule
        success = db.remove_rule("test_rule")
        assert success == True
        assert len(db.rule_index) == 0
        assert len(db.rules[RuleType.ALGEBRAIC]) == 0
        
        # Try to remove non-existent rule
        success = db.remove_rule("non_existent")
        assert success == False
    
    def test_export_rules(self):
        """Test exporting rules to JSON file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db = RuleDatabase()
            
            rule = LogicalRule(
                id="test_rule",
                name="Test Rule",
                rule_type=RuleType.ALGEBRAIC,
                pattern="a + b",
                replacement="b + a"
            )
            
            db.add_rule(rule)
            
            export_path = Path(temp_dir) / "exported_rules.json"
            db.export_rules(export_path)
            
            assert export_path.exists()
            
            # Load and verify exported data
            with open(export_path, 'r') as f:
                exported_data = json.load(f)
            
            assert "algebraic_rules" in exported_data
            assert len(exported_data["algebraic_rules"]) == 1
            assert exported_data["algebraic_rules"][0]["id"] == "test_rule"
    
    def test_load_rules_from_json(self):
        """Test loading rules from JSON file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            rules_dir = Path(temp_dir)
            
            # Create test JSON file
            test_rules = {
                "algebraic_rules": [
                    {
                        "id": "square_of_sum",
                        "name": "Square of Sum",
                        "rule_type": "algebraic",
                        "pattern": "(a + b)**2",
                        "replacement": "a**2 + 2*a*b + b**2",
                        "conditions": [],
                        "justification": "Binomial expansion",
                        "priority": 4,
                        "bidirectional": True
                    }
                ]
            }
            
            json_file = rules_dir / "test_rules.json"
            with open(json_file, 'w') as f:
                json.dump(test_rules, f)
            
            # Load rules
            db = RuleDatabase(rules_dir)
            loaded_rules = db.load_all_rules()
            
            assert len(db.rule_index) == 1
            assert "square_of_sum" in db.rule_index
            
            rule = db.get_rule_by_id("square_of_sum")
            assert rule.name == "Square of Sum"
            assert rule.rule_type == RuleType.ALGEBRAIC
            assert rule.pattern == "(a + b)**2"
            assert rule.bidirectional == True
    
    def test_validate_database(self):
        """Test database validation."""
        db = RuleDatabase()
        
        # Add valid rule
        valid_rule = LogicalRule(
            id="valid_rule",
            name="Valid Rule",
            rule_type=RuleType.ALGEBRAIC,
            pattern="a + b",
            replacement="b + a"
        )
        db.add_rule(valid_rule)
        
        errors = db.validate_database()
        assert len(errors) == 0  # No errors for valid rule
    
    def test_get_statistics(self):
        """Test getting database statistics."""
        db = RuleDatabase()
        
        rule1 = LogicalRule(
            id="rule1",
            name="Rule 1",
            rule_type=RuleType.ALGEBRAIC,
            pattern="a",
            replacement="b",
            priority=3,
            bidirectional=True,
            conditions=["a is real"]
        )
        
        rule2 = LogicalRule(
            id="rule2",
            name="Rule 2",
            rule_type=RuleType.TRIGONOMETRIC,
            pattern="sin(x)",
            replacement="sin(x)",
            priority=3
        )
        
        db.add_rule(rule1)
        db.add_rule(rule2)
        
        stats = db.get_statistics()
        
        assert stats["total_rules"] == 2
        assert stats["rules_by_type"]["algebraic"] == 1
        assert stats["rules_by_type"]["trigonometric"] == 1
        assert stats["priority_distribution"][3] == 2  # Both rules have priority 3
        assert stats["bidirectional_rules"] == 1  # Only rule1 is bidirectional
        assert stats["rules_with_conditions"] == 1  # Only rule1 has conditions
    
    def test_find_rules_by_pattern(self):
        """Test finding rules by pattern substring."""
        db = RuleDatabase()
        
        rule1 = LogicalRule(
            id="rule1",
            name="Rule 1",
            rule_type=RuleType.ALGEBRAIC,
            pattern="a + b",
            replacement="b + a"
        )
        
        rule2 = LogicalRule(
            id="rule2",
            name="Rule 2",
            rule_type=RuleType.TRIGONOMETRIC,
            pattern="sin(x)**2",
            replacement="1 - cos(x)**2"
        )
        
        db.add_rule(rule1)
        db.add_rule(rule2)
        
        # Search for addition patterns
        addition_rules = db.find_rules_by_pattern("+")
        assert len(addition_rules) == 1
        assert rule1 in addition_rules
        
        # Search for sine patterns
        sine_rules = db.find_rules_by_pattern("sin")
        assert len(sine_rules) == 1
        assert rule2 in sine_rules
        
        # Search for non-existent pattern
        empty_results = db.find_rules_by_pattern("xyz")
        assert len(empty_results) == 0
    
    def test_database_magic_methods(self):
        """Test database magic methods (__len__, __contains__, __iter__, etc.)."""
        db = RuleDatabase()
        
        rule1 = LogicalRule(
            id="rule1",
            name="Rule 1",
            rule_type=RuleType.ALGEBRAIC,
            pattern="a",
            replacement="b"
        )
        
        rule2 = LogicalRule(
            id="rule2",
            name="Rule 2",
            rule_type=RuleType.TRIGONOMETRIC,
            pattern="sin(x)",
            replacement="sin(x)"
        )
        
        # Test empty database
        assert len(db) == 0
        assert "rule1" not in db
        
        # Add rules
        db.add_rule(rule1)
        db.add_rule(rule2)
        
        # Test populated database
        assert len(db) == 2
        assert "rule1" in db
        assert "rule2" in db
        assert "non_existent" not in db
        
        # Test iteration
        rules_from_iter = list(db)
        assert len(rules_from_iter) == 2
        assert rule1 in rules_from_iter
        assert rule2 in rules_from_iter
    
    def test_database_string_representations(self):
        """Test string representations of database."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db = RuleDatabase(temp_dir)
            
            rule = LogicalRule(
                id="test_rule",
                name="Test Rule",
                rule_type=RuleType.ALGEBRAIC,
                pattern="a",
                replacement="b"
            )
            
            db.add_rule(rule)
            
            str_repr = str(db)
            assert temp_dir in str_repr
            assert "total_rules=1" in str_repr
            
            repr_str = repr(db)
            assert temp_dir in repr_str
            assert "total_rules=1" in repr_str


if __name__ == "__main__":
    pytest.main([__file__]) 