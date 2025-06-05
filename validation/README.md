# Formula Validation Engine

The Formula Validation Engine is Phase 3 of the MathBot project, providing comprehensive validation of mathematical formulas through symbolic analysis and numerical testing.

## Overview

The validation engine verifies the correctness of mathematical formulas using multiple approaches:

- **Symbolic Validation**: Uses SymPy to verify formula structure and mathematical consistency
- **Numerical Testing**: Evaluates formulas with randomized inputs to detect computational errors
- **Edge Case Testing**: Tests formulas with special values (0, 1, π, e, etc.)
- **Round-trip Testing**: Verifies that symbolic transformations preserve mathematical meaning
- **Domain Analysis**: Identifies potential domain restrictions and undefined regions

## Quick Start

### Basic Usage

```bash
# Validate 10 random formulas from processed data
python main.py validate-formulas --count 10 --source processed

# Validate all formulas and save detailed report
python main.py validate-formulas --all --output-report validation_report.json

# Validate with custom settings
python main.py validate-formulas --count 5 --num-tests 50 --tolerance 1e-8 --seed 123
```

### Programmatic Usage

```python
from validation import FormulaValidator, ValidationConfig, TestRunner

# Configure validation
config = ValidationConfig(
    num_random_tests=100,
    random_seed=42,
    tolerance=1e-10,
    enable_symbolic=True,
    enable_numerical=True,
    enable_edge_cases=True
)

# Validate a single formula
validator = FormulaValidator(config)
result = validator.validate_formula("x**2 + 2*x + 1")

print(f"Status: {result.status}")
print(f"Confidence: {result.confidence_score:.3f}")
print(f"Pass Rate: {result.pass_rate:.2%}")

# Run batch validation
runner = TestRunner(config)
report = runner.validate_random_formulas(count=10, source="processed")
```

## Architecture

### Core Components

1. **FormulaValidator** (`formula_tester.py`)
   - Core validation logic
   - Handles symbolic parsing and analysis
   - Runs numerical tests and edge cases
   - Calculates confidence scores

2. **TestRunner** (`test_runner.py`)
   - Orchestrates validation workflows
   - Loads formulas from data sources
   - Generates comprehensive reports
   - Integrates with knowledge graphs

3. **ValidationResult** Classes
   - Standardized result format
   - Detailed test breakdowns
   - Metadata and timing information

### Data Flow

```
Formula Sources → TestRunner → FormulaValidator → ValidationResults → Reports/Graph Updates
     ↓               ↓              ↓                 ↓                    ↓
[processed/]    [orchestration] [validation]    [results]         [reports/updates]
[graph/]        [loading]       [tests]         [confidence]      [integration]
```

## Validation Types

### 1. Symbolic Validation
- **Well-formedness**: Verifies formula can be parsed correctly
- **Identity verification**: Compares against known equivalent forms
- **Simplification consistency**: Ensures symbolic operations preserve meaning

### 2. Numerical Testing
- **Random input testing**: Evaluates formula with random variable values
- **Finite result verification**: Checks for NaN/infinity in outputs
- **Consistency across evaluations**: Verifies reproducible results

### 3. Edge Case Testing
- **Special values**: Tests with 0, 1, -1, π, e, etc.
- **Boundary conditions**: Evaluates at domain boundaries
- **Error handling**: Verifies graceful handling of problematic inputs

### 4. Round-trip Testing
- **Simplification round-trip**: simplify(simplify(expr)) ≡ simplify(expr)
- **Expansion/factorization**: Tests algebraic transformation consistency
- **Numerical equivalence**: Verifies transformations preserve values

## Configuration

### ValidationConfig Options

```python
ValidationConfig(
    num_random_tests=100,      # Number of numerical tests per formula
    random_seed=42,            # Seed for reproducible results
    test_range=(-10.0, 10.0),  # Range for random test values
    tolerance=1e-10,           # Numerical tolerance for comparisons
    max_complexity=1000,       # Maximum formula complexity (characters)
    timeout_seconds=30,        # Timeout for individual validations
    
    # Enable/disable test types
    enable_symbolic=True,
    enable_numerical=True,
    enable_edge_cases=True,
    enable_round_trip=True,
    
    # Edge case values to test
    edge_case_values=[0, 1, -1, 0.5, -0.5, np.pi, np.e]
)
```

## Data Sources

### Processed Data (`data/processed/`)
Load formulas from ingestion pipeline output:
```json
{
  "formulas": ["x**2 + 1", "sin(x) + cos(x)"],
  "detailed_formulas": [
    {
      "expression": "a**2 + b**2",
      "known_identity": "c**2",
      "metadata": {"topic": "geometry"}
    }
  ],
  "cleaned_formulas": {
    "unicode_normalized": ["α + β"],
    "sympy_compatible": ["alpha + beta"]
  }
}
```

### Knowledge Graph (`data/graph/`)
Load formulas from graph nodes:
```json
{
  "nodes": [
    {
      "id": "node1",
      "attributes": {
        "formula": "E = m*c**2",
        "topic": "physics"
      }
    }
  ]
}
```

## Results and Reporting

### ValidationResult Structure
```python
ValidationResult(
    formula="x**2 + 2*x + 1",
    status=ValidationStatus.PASS,
    confidence_score=0.95,
    pass_rate=0.98,
    total_tests=105,
    passed_tests=103,
    failed_tests=2,
    error_tests=0,
    symbols_found={"x"},
    validation_time=0.25,
    test_results=[...],  # Detailed test breakdowns
    metadata={...}       # Additional information
)
```

### Status Levels
- **PASS**: ≥95% test pass rate, high confidence
- **PARTIAL**: 70-95% test pass rate, moderate confidence  
- **FAIL**: <70% test pass rate
- **ERROR**: Unable to parse or validate formula

### Confidence Scoring
Confidence scores (0-1) are calculated based on:
- Base score from test pass rate
- Penalty for errors and failures
- Bonus for comprehensive testing (multiple test types)
- Bonus for high-volume successful numerical tests

## Integration with Knowledge Graphs

The validation engine can update knowledge graph nodes with validation results:

```python
# Automatic graph updates
runner = TestRunner(config)
report = runner.validate_all_formulas(source="graph")
runner.update_graph_with_results(report)
```

Added node attributes:
- `validation_score`: Confidence score (0-1)
- `validation_status`: PASS/FAIL/PARTIAL/ERROR
- `validation_pass_rate`: Percentage of tests passed
- `tested_on`: Set of symbols/variables tested
- `validation_timestamp`: When validation was performed

## Testing

### Running Tests
```bash
# Run all validation tests
pytest tests/test_formula_tester.py -v

# Run specific test categories
pytest tests/test_formula_tester.py::TestKnownFormulas -v
pytest tests/test_test_runner.py::TestValidationExecution -v

# Run with coverage
pytest tests/ --cov=validation --cov-report=html
```

### Test Categories
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflow testing  
- **Known Formula Tests**: Validation with real mathematical formulas
- **Error Handling Tests**: Edge cases and error conditions
- **Reproducibility Tests**: Consistent results with same seeds

## Known Mathematical Formulas Tested

The test suite includes validation of real mathematical formulas:

1. **Algebra**
   - Quadratic formula: `(-b + sqrt(b**2 - 4*a*c)) / (2*a)`
   - Binomial expansion: `(a + b)**2 = a**2 + 2*a*b + b**2`

2. **Geometry**
   - Pythagorean theorem: `a**2 + b**2 = c**2`
   - Circle area: `pi * r**2`
   - Distance formula: `sqrt((x2-x1)**2 + (y2-y1)**2)`

3. **Trigonometry**
   - Fundamental identity: `sin(x)**2 + cos(x)**2 = 1`
   - Double angle: `sin(2*x) = 2*sin(x)*cos(x)`

4. **Calculus**
   - Power rule derivative: `d/dx[x**n] = n*x**(n-1)`
   - Gaussian function: `exp(-x**2/2)`

5. **Other**
   - Exponential-logarithm inverse: `exp(log(x)) = x`
   - Compound interest: `P*(1 + r/n)**(n*t)`

## CLI Reference

### Commands

#### validate-formulas
Main validation command with options:

- `--count N`: Number of random formulas to validate (default: 10)
- `--source {processed,graph}`: Data source (default: processed)
- `--all`: Validate all available formulas
- `--output-report PATH`: Save detailed JSON report
- `--update-graph`: Update knowledge graph with results
- `--seed N`: Random seed for reproducibility (default: 42)
- `--num-tests N`: Random tests per formula (default: 100)
- `--tolerance F`: Numerical tolerance (default: 1e-10)

### Examples

```bash
# Basic validation
python main.py validate-formulas

# Comprehensive validation with reporting
python main.py validate-formulas --all --source graph --output-report full_report.json --update-graph

# Custom testing parameters
python main.py validate-formulas --count 20 --num-tests 200 --tolerance 1e-12 --seed 999

# Debug mode with detailed logging
python main.py validate-formulas --log-level DEBUG --count 5
```

## Performance Considerations

- **Batch Size**: Large batches (>100 formulas) may take several minutes
- **Test Count**: More tests increase accuracy but slow execution
- **Complexity**: Very complex formulas may timeout or be rejected
- **Memory**: Large symbolic expressions can consume significant memory
- **Reproducibility**: Use consistent seeds for comparable results

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **No Formulas Found**: Check data directory structure
   ```bash
   ls data/processed/  # Should contain JSON files
   ls data/graph/      # Should contain graph files
   ```

3. **Parsing Failures**: Some LaTeX may need cleaning
   - Check formula syntax in source data
   - Review parsing logs for specific errors

4. **Slow Performance**: Reduce test parameters
   ```bash
   python main.py validate-formulas --num-tests 20 --count 5
   ```

### Debug Mode
Enable detailed logging to diagnose issues:
```bash
python main.py validate-formulas --log-level DEBUG
```

This will show:
- Formula parsing attempts
- Individual test results
- Symbolic transformation steps
- Error stack traces

## Extension Points

The validation engine is designed for extensibility:

1. **Custom Test Types**: Add new test categories to `TestType` enum
2. **Additional Validators**: Implement domain-specific validation logic
3. **New Data Sources**: Extend formula loading for other formats
4. **Custom Metrics**: Add specialized confidence scoring
5. **Parallel Execution**: Add multiprocessing for large batches

See the code documentation for implementation details. 