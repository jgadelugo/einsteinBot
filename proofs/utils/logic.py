"""
Logical rule system for mathematical transformations in MathBot.

This module implements a sophisticated rule-based system for applying mathematical
transformations to symbolic expressions. It provides a comprehensive framework for
pattern matching, rule application, and mathematical reasoning.

Core Components:
- RuleType: Enumeration of mathematical rule categories
- LogicalRule: Data structure representing individual transformation rules
- RuleDatabase: Manager for collections of mathematical rules

The system is designed for extensibility, mathematical soundness, and high performance.
"""

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union, Iterator
import sympy as sp
from sympy.core.sympify import SympifyError


class RuleType(Enum):
    """
    Enumeration of mathematical rule categories.
    
    This enumeration categorizes mathematical transformation rules by their
    mathematical domain and application context.
    """
    ALGEBRAIC = "algebraic"
    TRIGONOMETRIC = "trigonometric"
    LOGARITHMIC = "logarithmic"
    EXPONENTIAL = "exponential"
    CALCULUS = "calculus"
    LOGICAL = "logical"
    SUBSTITUTION = "substitution"
    
    def __str__(self) -> str:
        """Return string representation of rule type."""
        return self.value
    
    @classmethod
    def from_string(cls, value: str) -> 'RuleType':
        """
        Create RuleType from string value.
        
        Args:
            value: String representation of rule type
            
        Returns:
            RuleType enum value
            
        Raises:
            ValueError: If value doesn't match any rule type
        """
        try:
            return cls(value.lower())
        except ValueError:
            valid_types = [rt.value for rt in cls]
            raise ValueError(f"Invalid rule type '{value}'. Valid types: {valid_types}")


@dataclass
class LogicalRule:
    """
    Data structure representing a mathematical transformation rule.
    
    A LogicalRule encapsulates a mathematical transformation that can be applied
    to symbolic expressions. It includes pattern matching information, replacement
    logic, validation conditions, and metadata for rule management.
    
    Attributes:
        id: Unique identifier for the rule
        name: Human-readable name describing the rule
        rule_type: Category of mathematical rule
        pattern: SymPy expression pattern to match
        replacement: SymPy expression to replace matched pattern
        conditions: List of conditions that must be satisfied for rule application
        justification: Mathematical explanation for the transformation
        priority: Rule priority (higher values indicate higher priority)
        bidirectional: Whether rule can be applied in reverse direction
        metadata: Additional rule metadata and configuration
    """
    
    id: str
    name: str
    rule_type: RuleType
    pattern: str
    replacement: str
    conditions: List[str] = field(default_factory=list)
    justification: str = ""
    priority: int = 1
    bidirectional: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """
        Validate rule data after initialization.
        
        Raises:
            ValueError: If rule data is invalid
            SympifyError: If pattern or replacement cannot be parsed
        """
        self._validate_rule_data()
        self._validate_sympy_expressions()
    
    def _validate_rule_data(self) -> None:
        """Validate basic rule data integrity."""
        if not self.id or not self.id.strip():
            raise ValueError("Rule ID cannot be empty")
        
        if not self.name or not self.name.strip():
            raise ValueError("Rule name cannot be empty")
        
        if not isinstance(self.rule_type, RuleType):
            raise ValueError(f"rule_type must be RuleType enum, got {type(self.rule_type)}")
        
        if not self.pattern or not self.pattern.strip():
            raise ValueError("Rule pattern cannot be empty")
        
        if not self.replacement or not self.replacement.strip():
            raise ValueError("Rule replacement cannot be empty")
        
        if self.priority < 0:
            raise ValueError("Rule priority must be non-negative")
        
        # Validate ID format (alphanumeric with underscores)
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', self.id):
            raise ValueError("Rule ID must start with letter and contain only alphanumeric characters and underscores")
    
    def _validate_sympy_expressions(self) -> None:
        """
        Validate that pattern and replacement are valid SymPy expressions.
        
        Raises:
            SympifyError: If expressions cannot be parsed by SymPy
        """
        try:
            # Attempt to parse pattern
            sp.parse_expr(self.pattern)
        except (SympifyError, ValueError, TypeError, SyntaxError) as e:
            raise SympifyError(f"Invalid pattern expression '{self.pattern}': {e}")
        
        try:
            # Attempt to parse replacement
            sp.parse_expr(self.replacement)
        except (SympifyError, ValueError, TypeError, SyntaxError) as e:
            raise SympifyError(f"Invalid replacement expression '{self.replacement}': {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert rule to JSON-serializable dictionary.
        
        Returns:
            Dictionary representation of the rule suitable for JSON serialization
        """
        rule_dict = asdict(self)
        # Convert enum to string for JSON serialization
        rule_dict['rule_type'] = self.rule_type.value
        return rule_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LogicalRule':
        """
        Create LogicalRule from dictionary representation.
        
        Args:
            data: Dictionary containing rule data
            
        Returns:
            LogicalRule instance
            
        Raises:
            ValueError: If dictionary data is invalid
            KeyError: If required keys are missing
        """
        # Validate required keys
        required_keys = {'id', 'name', 'rule_type', 'pattern', 'replacement'}
        missing_keys = required_keys - set(data.keys())
        if missing_keys:
            raise KeyError(f"Missing required keys: {missing_keys}")
        
        # Convert rule_type string to enum
        rule_data = data.copy()
        rule_data['rule_type'] = RuleType.from_string(data['rule_type'])
        
        # Handle optional fields with defaults
        rule_data.setdefault('conditions', [])
        rule_data.setdefault('justification', "")
        rule_data.setdefault('priority', 1)
        rule_data.setdefault('bidirectional', False)
        rule_data.setdefault('metadata', {})
        
        return cls(**rule_data)
    
    def get_sympy_pattern(self) -> sp.Expr:
        """
        Get parsed SymPy expression for the pattern.
        
        Returns:
            SymPy expression representing the pattern
            
        Raises:
            SympifyError: If pattern cannot be parsed
        """
        return sp.parse_expr(self.pattern)
    
    def get_sympy_replacement(self) -> sp.Expr:
        """
        Get parsed SymPy expression for the replacement.
        
        Returns:
            SymPy expression representing the replacement
            
        Raises:
            SympifyError: If replacement cannot be parsed
        """
        return sp.parse_expr(self.replacement)
    
    def get_pattern_variables(self) -> Set[str]:
        """
        Extract variable names from the pattern expression.
        
        Returns:
            Set of variable names used in the pattern
        """
        try:
            pattern_expr = self.get_sympy_pattern()
            return {str(symbol) for symbol in pattern_expr.free_symbols}
        except SympifyError:
            return set()
    
    def is_applicable_to_expression(self, expression: sp.Expr) -> bool:
        """
        Check if this rule could potentially be applied to an expression.
        
        This is a lightweight check that doesn't perform full pattern matching,
        but rather checks basic compatibility.
        
        Args:
            expression: SymPy expression to check
            
        Returns:
            True if rule might be applicable, False otherwise
        """
        try:
            pattern_expr = self.get_sympy_pattern()
            
            # Check if expression contains similar structure
            expr_symbols = expression.free_symbols
            pattern_symbols = pattern_expr.free_symbols
            
            # If pattern has no variables, it should match exactly
            if not pattern_symbols:
                return expression.equals(pattern_expr)
            
            # Check if expression has compatible symbols
            return len(expr_symbols) >= len(pattern_symbols)
            
        except Exception:
            return False
    
    def __str__(self) -> str:
        """Return string representation of the rule."""
        return f"LogicalRule(id='{self.id}', name='{self.name}', type={self.rule_type.value})"
    
    def __repr__(self) -> str:
        """Return detailed string representation of the rule."""
        return (
            f"LogicalRule(id='{self.id}', name='{self.name}', "
            f"rule_type={self.rule_type}, pattern='{self.pattern}', "
            f"replacement='{self.replacement}', priority={self.priority})"
        )
    
    def __eq__(self, other) -> bool:
        """Check equality based on rule ID."""
        if not isinstance(other, LogicalRule):
            return False
        return self.id == other.id
    
    def __hash__(self) -> int:
        """Hash based on rule ID."""
        return hash(self.id)


class RuleDatabase:
    """
    Manager for mathematical transformation rule database.
    
    The RuleDatabase class provides comprehensive management for collections of
    mathematical transformation rules. It supports loading rules from JSON files,
    organizing rules by category, validation, and efficient rule lookup.
    
    Attributes:
        rules_dir: Directory containing rule JSON files
        rules: Dictionary mapping rule types to lists of rules
        rule_index: Dictionary mapping rule IDs to rules for fast lookup
        logger: Logger instance for database operations
    """
    
    def __init__(self, rules_dir: Union[str, Path] = "proofs/rules"):
        """
        Initialize rule database.
        
        Args:
            rules_dir: Directory containing rule JSON files
        """
        self.rules_dir = Path(rules_dir)
        self.rules: Dict[RuleType, List[LogicalRule]] = {}
        self.rule_index: Dict[str, LogicalRule] = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize empty rule lists for all rule types
        for rule_type in RuleType:
            self.rules[rule_type] = []
        
        # Create rules directory if it doesn't exist
        self.rules_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Initialized RuleDatabase with rules directory: {self.rules_dir}")
    
    def load_all_rules(self) -> Dict[RuleType, List[LogicalRule]]:
        """
        Load all rules from JSON files in the rules directory.
        
        This method scans the rules directory for JSON files and loads all
        valid rule definitions found in those files.
        
        Returns:
            Dictionary mapping rule types to lists of loaded rules
            
        Raises:
            FileNotFoundError: If rules directory doesn't exist
            json.JSONDecodeError: If JSON files are malformed
        """
        if not self.rules_dir.exists():
            raise FileNotFoundError(f"Rules directory not found: {self.rules_dir}")
        
        # Clear existing rules
        self.rules.clear()
        self.rule_index.clear()
        for rule_type in RuleType:
            self.rules[rule_type] = []
        
        # Load rules from all JSON files
        json_files = list(self.rules_dir.glob("*.json"))
        if not json_files:
            self.logger.warning(f"No JSON files found in rules directory: {self.rules_dir}")
            return self.rules
        
        total_loaded = 0
        for json_file in json_files:
            try:
                loaded_count = self._load_rules_from_file(json_file)
                total_loaded += loaded_count
                self.logger.info(f"Loaded {loaded_count} rules from {json_file.name}")
            except Exception as e:
                self.logger.error(f"Failed to load rules from {json_file}: {e}")
        
        self.logger.info(f"Total rules loaded: {total_loaded}")
        return self.rules
    
    def _load_rules_from_file(self, file_path: Path) -> int:
        """
        Load rules from a single JSON file.
        
        Args:
            file_path: Path to JSON file containing rules
            
        Returns:
            Number of rules loaded from the file
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        loaded_count = 0
        
        # Handle different JSON structures
        if isinstance(data, list):
            # Direct list of rules
            for rule_data in data:
                if self._add_rule_from_dict(rule_data):
                    loaded_count += 1
        elif isinstance(data, dict):
            # Handle nested structure or single rule
            if 'id' in data and 'name' in data:
                # Single rule
                if self._add_rule_from_dict(data):
                    loaded_count += 1
            else:
                # Nested structure with rule categories
                for key, value in data.items():
                    if isinstance(value, list):
                        for rule_data in value:
                            if self._add_rule_from_dict(rule_data):
                                loaded_count += 1
                    elif isinstance(value, dict) and 'id' in value:
                        if self._add_rule_from_dict(value):
                            loaded_count += 1
        
        return loaded_count
    
    def _add_rule_from_dict(self, rule_data: Dict[str, Any]) -> bool:
        """
        Add a rule from dictionary data.
        
        Args:
            rule_data: Dictionary containing rule information
            
        Returns:
            True if rule was added successfully, False otherwise
        """
        try:
            rule = LogicalRule.from_dict(rule_data)
            self.add_rule(rule)
            return True
        except Exception as e:
            self.logger.warning(f"Failed to create rule from data {rule_data}: {e}")
            return False
    
    def add_rule(self, rule: LogicalRule) -> None:
        """
        Add a new rule to the database.
        
        Args:
            rule: LogicalRule instance to add
            
        Raises:
            ValueError: If rule with same ID already exists
        """
        if rule.id in self.rule_index:
            raise ValueError(f"Rule with ID '{rule.id}' already exists")
        
        # Add to categorized storage
        self.rules[rule.rule_type].append(rule)
        
        # Add to index for fast lookup
        self.rule_index[rule.id] = rule
        
        self.logger.debug(f"Added rule: {rule}")
    
    def get_rule_by_id(self, rule_id: str) -> Optional[LogicalRule]:
        """
        Get a rule by its ID.
        
        Args:
            rule_id: Unique identifier of the rule
            
        Returns:
            LogicalRule instance if found, None otherwise
        """
        return self.rule_index.get(rule_id)
    
    def get_rules_by_type(self, rule_type: RuleType) -> List[LogicalRule]:
        """
        Get all rules of a specific type.
        
        Args:
            rule_type: Type of rules to retrieve
            
        Returns:
            List of rules of the specified type
        """
        return self.rules.get(rule_type, []).copy()
    
    def get_all_rules(self) -> List[LogicalRule]:
        """
        Get all rules in the database.
        
        Returns:
            List of all rules across all categories
        """
        all_rules = []
        for rule_list in self.rules.values():
            all_rules.extend(rule_list)
        return all_rules
    
    def remove_rule(self, rule_id: str) -> bool:
        """
        Remove a rule from the database.
        
        Args:
            rule_id: ID of rule to remove
            
        Returns:
            True if rule was removed, False if not found
        """
        rule = self.rule_index.get(rule_id)
        if not rule:
            return False
        
        # Remove from categorized storage
        self.rules[rule.rule_type].remove(rule)
        
        # Remove from index
        del self.rule_index[rule_id]
        
        self.logger.debug(f"Removed rule: {rule_id}")
        return True
    
    def export_rules(self, output_path: Union[str, Path]) -> None:
        """
        Export all rules to a JSON file.
        
        Args:
            output_path: Path where to save the exported rules
            
        Raises:
            IOError: If file cannot be written
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Organize rules by type for export
        export_data = {}
        for rule_type, rule_list in self.rules.items():
            if rule_list:  # Only export non-empty categories
                category_key = f"{rule_type.value}_rules"
                export_data[category_key] = [rule.to_dict() for rule in rule_list]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        total_rules = sum(len(rules) for rules in self.rules.values())
        self.logger.info(f"Exported {total_rules} rules to {output_path}")
    
    def validate_database(self) -> List[str]:
        """
        Validate all rules in the database.
        
        This method performs comprehensive validation of all rules, checking
        for mathematical soundness, consistency, and proper formatting.
        
        Returns:
            List of validation error messages (empty if all rules are valid)
        """
        errors = []
        
        # Check for duplicate IDs (should not happen with proper add_rule usage)
        all_ids = [rule.id for rule in self.get_all_rules()]
        duplicate_ids = [id_ for id_ in set(all_ids) if all_ids.count(id_) > 1]
        if duplicate_ids:
            errors.append(f"Duplicate rule IDs found: {duplicate_ids}")
        
        # Validate individual rules
        for rule in self.get_all_rules():
            try:
                # Test SymPy expression parsing
                pattern_expr = rule.get_sympy_pattern()
                replacement_expr = rule.get_sympy_replacement()
                
                # Check that pattern and replacement have compatible variables
                pattern_vars = rule.get_pattern_variables()
                replacement_vars = {str(s) for s in replacement_expr.free_symbols}
                
                # Replacement should not introduce new variables not in pattern
                # (unless it's a constant or well-defined function)
                extra_vars = replacement_vars - pattern_vars
                if extra_vars:
                    # Allow common mathematical constants and functions
                    allowed_extras = {'pi', 'e', 'I', 'oo', 'zoo', 'nan'}
                    problematic_vars = extra_vars - allowed_extras
                    if problematic_vars:
                        errors.append(
                            f"Rule '{rule.id}': replacement introduces new variables {problematic_vars} "
                            f"not present in pattern"
                        )
                
            except Exception as e:
                errors.append(f"Rule '{rule.id}': validation failed - {e}")
        
        if errors:
            self.logger.warning(f"Database validation found {len(errors)} errors")
        else:
            self.logger.info("Database validation passed")
        
        return errors
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the rule database.
        
        Returns:
            Dictionary containing database statistics
        """
        stats = {
            'total_rules': len(self.rule_index),
            'rules_by_type': {},
            'priority_distribution': {},
            'bidirectional_rules': 0,
            'rules_with_conditions': 0
        }
        
        # Rules by type
        for rule_type, rule_list in self.rules.items():
            stats['rules_by_type'][rule_type.value] = len(rule_list)
        
        # Priority distribution and other metrics
        all_rules = self.get_all_rules()
        priorities = [rule.priority for rule in all_rules]
        
        if priorities:
            from collections import Counter
            priority_counts = Counter(priorities)
            stats['priority_distribution'] = dict(priority_counts)
        
        stats['bidirectional_rules'] = sum(1 for rule in all_rules if rule.bidirectional)
        stats['rules_with_conditions'] = sum(1 for rule in all_rules if rule.conditions)
        
        return stats
    
    def find_rules_by_pattern(self, pattern_substring: str) -> List[LogicalRule]:
        """
        Find rules containing a specific pattern substring.
        
        Args:
            pattern_substring: Substring to search for in rule patterns
            
        Returns:
            List of rules whose patterns contain the substring
        """
        matching_rules = []
        for rule in self.get_all_rules():
            if pattern_substring.lower() in rule.pattern.lower():
                matching_rules.append(rule)
        return matching_rules
    
    def __len__(self) -> int:
        """Return total number of rules in database."""
        return len(self.rule_index)
    
    def __contains__(self, rule_id: str) -> bool:
        """Check if rule ID exists in database."""
        return rule_id in self.rule_index
    
    def __iter__(self) -> Iterator[LogicalRule]:
        """Iterate over all rules in database."""
        return iter(self.get_all_rules())
    
    def __str__(self) -> str:
        """Return string representation of database."""
        return f"RuleDatabase(rules_dir='{self.rules_dir}', total_rules={len(self)})"
    
    def __repr__(self) -> str:
        """Return detailed string representation of database."""
        stats = self.get_statistics()
        return (
            f"RuleDatabase(rules_dir='{self.rules_dir}', "
            f"total_rules={stats['total_rules']}, "
            f"rules_by_type={stats['rules_by_type']})"
        )


class LogicalRuleEngine:
    """
    Engine for applying logical transformation rules to mathematical expressions.
    
    The LogicalRuleEngine provides sophisticated pattern matching and rule application
    capabilities for mathematical expressions. It can find applicable rules, apply
    individual rules, execute rule sequences, and find transformation paths between
    expressions.
    
    Key Features:
    - Intelligent pattern matching using SymPy's pattern system
    - Rule prioritization and conflict resolution
    - Bidirectional rule application support
    - Multi-step transformation sequences
    - Transformation path finding between expressions
    - Comprehensive validation and error handling
    
    Attributes:
        rule_database: RuleDatabase instance containing transformation rules
        logger: Logger instance for engine operations
        max_transformation_steps: Maximum steps allowed in transformation sequences
        enable_bidirectional: Whether to consider bidirectional rules in reverse
        pattern_cache: Cache for pattern matching results
    """
    
    def __init__(self, 
                 rule_database_path: Optional[Union[str, Path]] = None,
                 max_transformation_steps: int = 20,
                 enable_bidirectional: bool = True):
        """
        Initialize the logical rule engine.
        
        Args:
            rule_database_path: Path to rule database directory (defaults to "proofs/rules")
            max_transformation_steps: Maximum steps allowed in transformation sequences
            enable_bidirectional: Whether to consider bidirectional rules in reverse
        """
        self.logger = logging.getLogger(__name__)
        self.max_transformation_steps = max_transformation_steps
        self.enable_bidirectional = enable_bidirectional
        
        # Initialize rule database
        if rule_database_path is None:
            rule_database_path = "proofs/rules"
        
        self.rule_database = RuleDatabase(rule_database_path)
        
        # Performance optimization caches
        self.pattern_cache: Dict[str, List[LogicalRule]] = {}
        self.transformation_cache: Dict[Tuple[str, str], Optional[List[LogicalRule]]] = {}
        
        # Load rules on initialization
        try:
            loaded_rules = self.rule_database.load_all_rules()
            total_rules = sum(len(rules) for rules in loaded_rules.values())
            self.logger.info(f"LogicalRuleEngine initialized with {total_rules} rules")
        except Exception as e:
            self.logger.warning(f"Failed to load rules during initialization: {e}")
    
    def find_applicable_rules(self, expression: sp.Expr, rule_types: Optional[List[RuleType]] = None) -> List[LogicalRule]:
        """
        Find all rules that could be applied to the given expression.
        
        This method performs comprehensive pattern matching to identify rules that
        could potentially transform the expression or its subexpressions.
        
        Args:
            expression: SymPy expression to analyze
            rule_types: Optional list of rule types to consider (defaults to all types)
            
        Returns:
            List of applicable rules sorted by priority (highest first)
        """
        if rule_types is None:
            rule_types = list(RuleType)
        
        applicable_rules = []
        
        # Check cache first
        expr_str = str(expression)
        cache_key = f"{expr_str}_{','.join(rt.value for rt in rule_types)}"
        if cache_key in self.pattern_cache:
            return self.pattern_cache[cache_key]
        
        # Get rules to check
        rules_to_check = []
        for rule_type in rule_types:
            rules_to_check.extend(self.rule_database.get_rules_by_type(rule_type))
        
        # Check each rule for applicability
        for rule in rules_to_check:
            try:
                if self._is_rule_applicable(expression, rule):
                    applicable_rules.append(rule)
                    
                # Also check bidirectional rules in reverse
                if self.enable_bidirectional and rule.bidirectional:
                    reverse_rule = self._create_reverse_rule(rule)
                    if self._is_rule_applicable(expression, reverse_rule):
                        applicable_rules.append(reverse_rule)
                        
            except Exception as e:
                self.logger.debug(f"Error checking rule {rule.id} applicability: {e}")
                continue
        
        # Sort by priority (higher priority first)
        applicable_rules.sort(key=lambda r: r.priority, reverse=True)
        
        # Cache result
        self.pattern_cache[cache_key] = applicable_rules
        
        self.logger.debug(f"Found {len(applicable_rules)} applicable rules for expression: {expr_str}")
        return applicable_rules
    
    def _is_rule_applicable(self, expression: sp.Expr, rule: LogicalRule) -> bool:
        """
        Check if a specific rule is applicable to an expression.
        
        Args:
            expression: SymPy expression to check
            rule: Rule to test for applicability
            
        Returns:
            True if rule can be applied, False otherwise
        """
        try:
            pattern_expr = rule.get_sympy_pattern()
            
            # Check direct pattern match
            if self._pattern_matches(expression, pattern_expr):
                return True
            
            # Check subexpression matches
            for subexpr in expression.atoms() | {expression}:
                if self._pattern_matches(subexpr, pattern_expr):
                    return True
                    
            return False
            
        except Exception:
            return False
    
    def _pattern_matches(self, expression: sp.Expr, pattern: sp.Expr) -> bool:
        """
        Check if an expression matches a pattern using SymPy's matching.
        
        Args:
            expression: Expression to match
            pattern: Pattern to match against
            
        Returns:
            True if pattern matches, False otherwise
        """
        try:
            # Direct match using SymPy's standard matching
            match_result = expression.match(pattern)
            if match_result is not None:
                return True
            
            # Try with pattern and expression swapped (for symmetry)
            match_result = pattern.match(expression)
            if match_result is not None:
                return True
            
            # Enhanced pattern matching using Wild symbols
            # Convert pattern to use Wild symbols for more flexible matching
            wild_pattern = self._convert_to_wild_pattern(pattern)
            if wild_pattern != pattern:
                wild_match = expression.match(wild_pattern)
                if wild_match is not None:
                    return True
            
            # Try structural equivalence
            try:
                if sp.simplify(expression - pattern) == 0:
                    return True
            except Exception:
                pass
                
            return False
            
        except Exception:
            return False
    
    def _convert_to_wild_pattern(self, pattern: sp.Expr) -> sp.Expr:
        """
        Convert a pattern to use Wild symbols for more flexible matching.
        
        Args:
            pattern: Original pattern expression
            
        Returns:
            Pattern with variables converted to Wild symbols
        """
        try:
            from sympy import Wild
            
            # Get all free symbols in the pattern
            free_symbols = pattern.free_symbols
            if not free_symbols:
                return pattern
            
            # Create Wild symbols and substitution mapping
            substitutions = {}
            for i, symbol in enumerate(sorted(free_symbols, key=str)):
                wild_symbol = Wild(f'w{i}')
                substitutions[symbol] = wild_symbol
            
            # Apply substitutions to create wild pattern
            wild_pattern = pattern.subs(substitutions)
            return wild_pattern
            
        except Exception:
            return pattern
    
    def _create_reverse_rule(self, rule: LogicalRule) -> LogicalRule:
        """
        Create a reverse version of a bidirectional rule.
        
        Args:
            rule: Original rule to reverse
            
        Returns:
            New rule with pattern and replacement swapped
        """
        return LogicalRule(
            id=f"{rule.id}_reverse",
            name=f"{rule.name} (Reverse)",
            rule_type=rule.rule_type,
            pattern=rule.replacement,
            replacement=rule.pattern,
            conditions=rule.conditions,
            justification=f"Reverse of: {rule.justification}",
            priority=rule.priority,
            bidirectional=False,  # Prevent infinite recursion
            metadata={**rule.metadata, 'reverse_of': rule.id}
        )
    
    def apply_rule(self, expression: sp.Expr, rule: LogicalRule) -> Tuple[sp.Expr, bool, Dict[str, Any]]:
        """
        Apply a specific rule to an expression.
        
        Args:
            expression: SymPy expression to transform
            rule: Rule to apply
            
        Returns:
            Tuple of (transformed_expression, success_flag, application_info)
        """
        application_info = {
            'rule_id': rule.id,
            'rule_name': rule.name,
            'original_expression': str(expression),
            'pattern_used': rule.pattern,
            'replacement_used': rule.replacement,
            'matches_found': 0,
            'subexpression_matches': []
        }
        
        try:
            pattern_expr = rule.get_sympy_pattern()
            replacement_expr = rule.get_sympy_replacement()
            
            # Try direct substitution first
            new_expr, direct_success = self._apply_direct_substitution(
                expression, pattern_expr, replacement_expr, rule
            )
            
            if direct_success:
                application_info['matches_found'] = 1
                application_info['transformed_expression'] = str(new_expr)
                return new_expr, True, application_info
            
            # Try subexpression substitution
            new_expr, sub_success, sub_info = self._apply_subexpression_substitution(
                expression, pattern_expr, replacement_expr, rule
            )
            
            if sub_success:
                application_info.update(sub_info)
                application_info['transformed_expression'] = str(new_expr)
                return new_expr, True, application_info
            
            # Rule not applicable
            application_info['transformed_expression'] = str(expression)
            return expression, False, application_info
            
        except Exception as e:
            self.logger.error(f"Error applying rule {rule.id}: {e}")
            application_info['error'] = str(e)
            application_info['transformed_expression'] = str(expression)
            return expression, False, application_info
    
    def _apply_direct_substitution(self, expression: sp.Expr, pattern: sp.Expr, 
                                   replacement: sp.Expr, rule: LogicalRule) -> Tuple[sp.Expr, bool]:
        """
        Apply direct pattern substitution to an expression.
        
        Args:
            expression: Expression to transform
            pattern: Pattern to match
            replacement: Replacement expression
            rule: Rule being applied
            
        Returns:
            Tuple of (transformed_expression, success_flag)
        """
        try:
            # Try direct match first
            match_dict = expression.match(pattern)
            if match_dict is not None:
                # Apply substitution
                new_expr = replacement.subs(match_dict)
                
                # Validate transformation
                if self._validate_transformation(expression, new_expr, rule):
                    return new_expr, True
            
            # Try enhanced matching with Wild symbols
            wild_pattern = self._convert_to_wild_pattern(pattern)
            if wild_pattern != pattern:
                wild_match = expression.match(wild_pattern)
                if wild_match is not None:
                    # Convert wild match back to original variable names
                    original_match = self._convert_wild_match_to_original(wild_match, pattern, wild_pattern)
                    if original_match:
                        new_expr = replacement.subs(original_match)
                        
                        # Validate transformation
                        if self._validate_transformation(expression, new_expr, rule):
                            return new_expr, True
            
            return expression, False
            
        except Exception:
            return expression, False
    
    def _convert_wild_match_to_original(self, wild_match: dict, original_pattern: sp.Expr, 
                                        wild_pattern: sp.Expr) -> dict:
        """
        Convert a wild match dictionary back to original variable names.
        
        Args:
            wild_match: Match dictionary using Wild symbols
            original_pattern: Original pattern with regular symbols
            wild_pattern: Wild pattern with Wild symbols
            
        Returns:
            Match dictionary with original variable names
        """
        try:
            # Create mapping from wild symbols back to original symbols
            original_symbols = sorted(original_pattern.free_symbols, key=str)
            
            if len(original_symbols) == 0:
                return {}
            
            # Map wild symbols back to original
            original_match = {}
            from sympy import Wild
            
            for i, orig_symbol in enumerate(original_symbols):
                wild_symbol = Wild(f'w{i}')
                
                # Find the wild symbol in the match
                if wild_symbol in wild_match:
                    original_match[orig_symbol] = wild_match[wild_symbol]
            
            return original_match
            
        except Exception as e:
            self.logger.debug(f"Error converting wild match: {e}")
            return {}
    
    def _apply_subexpression_substitution(self, expression: sp.Expr, pattern: sp.Expr,
                                          replacement: sp.Expr, rule: LogicalRule) -> Tuple[sp.Expr, bool, Dict[str, Any]]:
        """
        Apply pattern substitution to subexpressions.
        
        Args:
            expression: Expression to transform
            pattern: Pattern to match
            replacement: Replacement expression
            rule: Rule being applied
            
        Returns:
            Tuple of (transformed_expression, success_flag, application_info)
        """
        info = {'matches_found': 0, 'subexpression_matches': []}
        
        try:
            new_expr = expression
            transformation_applied = False
            
            # Try to substitute in all subexpressions
            def substitute_func(subexpr):
                nonlocal transformation_applied
                
                match_dict = subexpr.match(pattern)
                if match_dict is not None:
                    try:
                        substituted = replacement.subs(match_dict)
                        if self._validate_transformation(subexpr, substituted, rule):
                            info['matches_found'] += 1
                            info['subexpression_matches'].append({
                                'original': str(subexpr),
                                'transformed': str(substituted),
                                'match_dict': {str(k): str(v) for k, v in match_dict.items()}
                            })
                            transformation_applied = True
                            return substituted
                    except Exception:
                        pass
                
                return subexpr
            
            # Apply substitution recursively
            new_expr = new_expr.replace(lambda x: x.match(pattern) is not None, substitute_func)
            
            return new_expr, transformation_applied, info
            
        except Exception:
            return expression, False, info
    
    def _validate_transformation(self, original: sp.Expr, transformed: sp.Expr, rule: LogicalRule) -> bool:
        """
        Validate that a transformation is mathematically sound.
        
        Args:
            original: Original expression
            transformed: Transformed expression
            rule: Rule that was applied
            
        Returns:
            True if transformation is valid, False otherwise
        """
        try:
            # Basic sanity checks
            if original == transformed:
                return False  # No transformation occurred
            
            # Check that transformation preserves mathematical structure
            # (This is a simplified check - more sophisticated validation could be added)
            
            # For algebraic rules, check with numerical substitution
            if rule.rule_type == RuleType.ALGEBRAIC:
                return self._validate_numerical_equivalence(original, transformed)
            
            # For trigonometric identities, use SymPy's trigsimp
            elif rule.rule_type == RuleType.TRIGONOMETRIC:
                return self._validate_trigonometric_equivalence(original, transformed)
            
            # For other types, use general simplification
            else:
                return self._validate_general_equivalence(original, transformed)
            
        except Exception:
            return False
    
    def _validate_numerical_equivalence(self, expr1: sp.Expr, expr2: sp.Expr) -> bool:
        """Validate equivalence using numerical substitution."""
        try:
            variables = list(expr1.free_symbols | expr2.free_symbols)
            if not variables:
                # No variables, evaluate directly
                return abs(float(expr1.evalf()) - float(expr2.evalf())) < 1e-10
            
            # Test with random values
            import random
            random.seed(42)  # Reproducible
            
            for _ in range(5):  # Test 5 random points
                substitutions = {var: random.uniform(-10, 10) for var in variables}
                try:
                    val1 = float(expr1.subs(substitutions).evalf())
                    val2 = float(expr2.subs(substitutions).evalf())
                    if abs(val1 - val2) > 1e-8:
                        return False
                except Exception:
                    continue
            
            return True
            
        except Exception:
            return False
    
    def _validate_trigonometric_equivalence(self, expr1: sp.Expr, expr2: sp.Expr) -> bool:
        """Validate trigonometric equivalence."""
        try:
            from sympy import trigsimp
            simplified_diff = trigsimp(expr1 - expr2)
            return simplified_diff == 0 or simplified_diff.equals(sp.S.Zero)
        except Exception:
            return False
    
    def _validate_general_equivalence(self, expr1: sp.Expr, expr2: sp.Expr) -> bool:
        """Validate general mathematical equivalence."""
        try:
            simplified_diff = sp.simplify(expr1 - expr2)
            return simplified_diff == 0 or simplified_diff.equals(sp.S.Zero)
        except Exception:
            return False
    
    def apply_rule_sequence(self, expression: sp.Expr, max_steps: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Apply a sequence of rules to transform an expression as much as possible.
        
        Args:
            expression: Starting expression
            max_steps: Maximum transformation steps (defaults to engine's limit)
            
        Returns:
            List of transformation steps with details
        """
        if max_steps is None:
            max_steps = self.max_transformation_steps
        
        transformation_sequence = []
        current_expr = expression
        
        for step in range(max_steps):
            # Find applicable rules for current expression
            applicable_rules = self.find_applicable_rules(current_expr)
            
            if not applicable_rules:
                # No more rules can be applied
                break
            
            # Apply the highest priority rule
            best_rule = applicable_rules[0]
            new_expr, success, app_info = self.apply_rule(current_expr, best_rule)
            
            if not success or new_expr == current_expr:
                # Rule application failed or no change
                break
            
            # Record transformation step
            step_info = {
                'step_number': step + 1,
                'rule_applied': best_rule.id,
                'rule_name': best_rule.name,
                'from_expression': str(current_expr),
                'to_expression': str(new_expr),
                'justification': best_rule.justification,
                'application_info': app_info
            }
            transformation_sequence.append(step_info)
            
            current_expr = new_expr
        
        self.logger.info(f"Applied {len(transformation_sequence)} transformation steps")
        return transformation_sequence
    
    def find_transformation_path(self, start_expr: sp.Expr, target_expr: sp.Expr, 
                                 max_depth: int = 10) -> Optional[List[LogicalRule]]:
        """
        Find a sequence of rules to transform start expression to target expression.
        
        This method uses a breadth-first search approach to find the shortest
        sequence of rule applications that transforms the start expression
        into the target expression.
        
        Args:
            start_expr: Starting expression
            target_expr: Target expression to reach
            max_depth: Maximum search depth
            
        Returns:
            List of rules to apply in sequence, or None if no path found
        """
        from collections import deque
        
        # Check cache first
        cache_key = (str(start_expr), str(target_expr))
        if cache_key in self.transformation_cache:
            return self.transformation_cache[cache_key]
        
        # BFS to find transformation path
        queue = deque([(start_expr, [])])
        visited = {str(start_expr)}
        
        for depth in range(max_depth):
            if not queue:
                break
            
            current_level_size = len(queue)
            
            for _ in range(current_level_size):
                current_expr, path = queue.popleft()
                
                # Check if we've reached the target
                if sp.simplify(current_expr - target_expr) == 0:
                    self.logger.info(f"Found transformation path with {len(path)} steps")
                    self.transformation_cache[cache_key] = path
                    return path
                
                # Find applicable rules and explore next states
                applicable_rules = self.find_applicable_rules(current_expr)
                
                for rule in applicable_rules[:5]:  # Limit branching factor
                    new_expr, success, _ = self.apply_rule(current_expr, rule)
                    
                    if success and str(new_expr) not in visited:
                        visited.add(str(new_expr))
                        new_path = path + [rule]
                        queue.append((new_expr, new_path))
        
        # No path found
        self.logger.debug(f"No transformation path found from {start_expr} to {target_expr}")
        self.transformation_cache[cache_key] = None
        return None
    
    def validate_rule(self, rule: LogicalRule, test_cases: Optional[List[sp.Expr]] = None) -> Dict[str, Any]:
        """
        Validate that a rule is mathematically sound.
        
        Args:
            rule: Rule to validate
            test_cases: Optional list of test expressions
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            'rule_id': rule.id,
            'is_valid': True,
            'errors': [],
            'test_results': [],
            'performance_metrics': {}
        }
        
        try:
            # Test SymPy expression parsing
            pattern_expr = rule.get_sympy_pattern()
            replacement_expr = rule.get_sympy_replacement()
            
            # Generate test cases if not provided
            if test_cases is None:
                test_cases = self._generate_test_cases_for_rule(rule)
            
            # Test rule application on test cases
            for i, test_expr in enumerate(test_cases):
                try:
                    transformed_expr, success, app_info = self.apply_rule(test_expr, rule)
                    
                    test_result = {
                        'test_case': i + 1,
                        'original_expression': str(test_expr),
                        'transformation_success': success,
                        'transformed_expression': str(transformed_expr) if success else None,
                        'application_info': app_info
                    }
                    
                    validation_result['test_results'].append(test_result)
                    
                except Exception as e:
                    validation_result['errors'].append(f"Test case {i + 1} failed: {e}")
                    validation_result['is_valid'] = False
            
        except Exception as e:
            validation_result['errors'].append(f"Rule validation failed: {e}")
            validation_result['is_valid'] = False
        
        return validation_result
    
    def _generate_test_cases_for_rule(self, rule: LogicalRule) -> List[sp.Expr]:
        """Generate test cases for rule validation."""
        test_cases = []
        
        try:
            pattern_vars = rule.get_pattern_variables()
            
            if not pattern_vars:
                # Pattern has no variables, use it directly
                test_cases.append(rule.get_sympy_pattern())
            else:
                # Generate test cases by substituting values
                test_values = [0, 1, -1, 2, sp.pi, sp.Symbol('x'), sp.Symbol('y')]
                
                for i in range(min(5, len(test_values))):
                    substitutions = {}
                    for j, var in enumerate(pattern_vars):
                        val_idx = (i + j) % len(test_values)
                        substitutions[sp.Symbol(var)] = test_values[val_idx]
                    
                    try:
                        test_expr = rule.get_sympy_pattern().subs(substitutions)
                        test_cases.append(test_expr)
                    except Exception:
                        continue
        
        except Exception:
            pass
        
        return test_cases if test_cases else [sp.Symbol('x')]
    
    def get_engine_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the rule engine.
        
        Returns:
            Dictionary containing engine statistics
        """
        db_stats = self.rule_database.get_statistics()
        
        engine_stats = {
            'database_statistics': db_stats,
            'engine_configuration': {
                'max_transformation_steps': self.max_transformation_steps,
                'enable_bidirectional': self.enable_bidirectional,
                'rules_directory': str(self.rule_database.rules_dir)
            },
            'cache_statistics': {
                'pattern_cache_size': len(self.pattern_cache),
                'transformation_cache_size': len(self.transformation_cache)
            }
        }
        
        return engine_stats
    
    def clear_caches(self) -> None:
        """Clear all internal caches."""
        self.pattern_cache.clear()
        self.transformation_cache.clear()
        self.logger.debug("Cleared all engine caches")
    
    def reload_rules(self) -> int:
        """
        Reload rules from the database.
        
        Returns:
            Number of rules loaded
        """
        self.clear_caches()
        loaded_rules = self.rule_database.load_all_rules()
        total_rules = sum(len(rules) for rules in loaded_rules.values())
        self.logger.info(f"Reloaded {total_rules} rules")
        return total_rules 