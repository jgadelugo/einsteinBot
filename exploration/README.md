# Phase 4 - Symbolic Pattern Discovery & Exploration

Phase 4 of MathBot provides comprehensive pattern discovery and hypothesis generation for mathematical formulas, building on the validated knowledge graph from Phase 3.

## Overview

The exploration engine discovers mathematical structures and generates new insights through:

- **Pattern Discovery**: Identifies clusters of structurally similar formulas
- **Gap Detection**: Finds missing connections and concepts in the knowledge graph
- **Hypothesis Generation**: Creates and validates new mathematical conjectures
- **Visualization**: Provides interactive charts and diagrams for analysis

## Quick Start

### Basic Usage

```bash
# Run complete exploration pipeline
python main.py explore-patterns --source processed

# With custom parameters and visualization
python main.py explore-patterns --source processed \
  --similarity-threshold 0.5 \
  --max-hypotheses 10 \
  --visualize \
  --output-dir results/exploration

# Using graph data source
python main.py explore-patterns --source graph \
  --embedding-method hybrid \
  --min-cluster-size 3
```

### Programmatic Usage

```python
from exploration import PatternFinder, GapDetector, HypothesisGenerator
from exploration.utils.embedding import FormulaEmbedder, ClusterVisualizer

# Pattern Discovery
pattern_finder = PatternFinder(similarity_threshold=0.7, min_cluster_size=2)
patterns = pattern_finder.find_patterns(formulas, metadata)

print(f"Found {len(patterns)} pattern clusters")
for cluster in patterns:
    print(f"Cluster {cluster.cluster_id}: {len(cluster.formulas)} formulas")
    print(f"Center: {cluster.cluster_center}")
    print(f"Common patterns: {cluster.common_patterns}")

# Gap Detection
gap_detector = GapDetector()
gaps = gap_detector.detect_gaps(formulas_data)

print(f"Detected {len(gaps)} potential gaps")
for gap in gaps[:5]:
    print(f"{gap.gap_type.value}: {gap.title}")
    print(f"Priority: {gap.priority:.3f}")

# Hypothesis Generation
hypothesis_generator = HypothesisGenerator(max_hypotheses_per_type=10)
hypotheses = hypothesis_generator.generate_hypotheses(formulas)

promising = hypothesis_generator.get_promising_hypotheses(hypotheses, min_confidence=0.7)
print(f"Generated {len(promising)} promising hypotheses")
```

## Architecture

### Core Components

1. **PatternFinder** (`pattern_finder.py`)
   - Structural fingerprinting of mathematical formulas
   - Similarity calculation using multiple metrics
   - Clustering algorithms (DBSCAN, K-means)
   - Pattern analysis and common structure extraction

2. **GapDetector** (`gap_detector.py`)
   - Knowledge graph construction from formula data
   - Sparse connection detection
   - Disconnected cluster identification
   - Missing variation and transformation suggestions

3. **HypothesisGenerator** (`hypothesis_generator.py`)
   - Algebraic identity discovery
   - Functional equation generation
   - Formula generalization and parametrization
   - Composition and transformation hypotheses
   - Integration with Phase 3 validation engine

4. **Embedding Utilities** (`utils/embedding.py`)
   - Multiple embedding methods (structural, TF-IDF, hybrid)
   - Dimension reduction (t-SNE, UMAP, PCA)
   - Clustering and visualization support

### Data Flow

```
Formula Data → Pattern Discovery → Gap Detection → Hypothesis Generation → Validation → Results
     ↓              ↓                ↓               ↓                      ↓          ↓
[processed/]   [fingerprints]   [knowledge      [conjectures]        [validated    [patterns.json]
[graph/]       [similarities]    graph]         [variations]          results]     [gaps.json]
               [clusters]        [gaps]         [compositions]                     [conjectures.json]
```

## Pattern Discovery

### Similarity Metrics

The pattern finder uses multiple similarity measures:

1. **Structural Similarity**: Hash-based comparison of AST structures
2. **Symbol Similarity**: Jaccard coefficient of symbol sets
3. **Operation Similarity**: Comparison of mathematical operations
4. **Function Similarity**: Shared function types (sin, exp, etc.)
5. **Complexity Similarity**: Normalized complexity differences

### Formula Fingerprinting

Each formula is converted to a fingerprint containing:

```python
FormulaFingerprint(
    formula="x**2 + 2*x + 1",
    symbol_count={"x": 3},
    operation_count={"Add": 2, "Pow": 1, "Mul": 1},
    structure_hash="a1b2c3d4...",
    depth=3,
    complexity=5,
    function_types=set(),
    symmetries=["even"],
    parsed_expr=x**2 + 2*x + 1
)
```

### Clustering Results

Pattern clusters include:

- **Formulas**: List of similar mathematical expressions
- **Cluster Center**: Most representative formula
- **Common Patterns**: Shared symbols, operations, functions
- **Topic Tags**: Extracted from source metadata
- **Confidence Score**: Measure of cluster coherence

## Gap Detection

### Gap Types

1. **Sparse Connection**: Concepts with few relationships
2. **Disconnected Cluster**: Isolated mathematical domains
3. **Missing Variation**: Formulas lacking common variations
4. **Incomplete Transformation**: Missing intermediate steps
5. **Domain Gap**: Underrepresented mathematical areas

### Knowledge Graph Construction

The gap detector builds a mathematical concept graph:

```python
ConceptNode(
    concept_id="x_polynomial_quadratic",
    formulas=["x**2 + 1", "x**2 + 2*x + 1"],
    topics={"algebra", "polynomial"},
    connections={"trigonometry_basic", "calculus_derivatives"},
    complexity_level=2,
    validation_scores=[0.95, 0.87]
)
```

### Gap Analysis

Detected gaps include:

- **Description**: Natural language explanation
- **Evidence**: Supporting data and statistics
- **Suggestions**: Potential formulas or connections
- **Priority Score**: Importance ranking
- **Related Concepts**: Connected mathematical areas

## Hypothesis Generation

### Hypothesis Types

1. **Algebraic Identity**: Relationships between expressions
   ```
   Example: (a + b)² + (a - b)² = 2(a² + b²)
   ```

2. **Functional Equation**: Function transformation patterns
   ```
   Example: f(2x) = 4f(x) - 2 for f(x) = x² + 1
   ```

3. **Generalization**: Parameterized versions of formulas
   ```
   Example: ax² + bx + c (generalization of x² + 2x + 1)
   ```

4. **Composition**: Combinations of existing formulas
   ```
   Example: g(f(x)) where f(x) = x², g(x) = sin(x)
   ```

5. **Transformation**: Calculus and algebraic transformations
   ```
   Example: d/dx[x² + 1] = 2x
   ```

6. **Limit Conjecture**: Limit behavior analysis
   ```
   Example: lim(x→0) sin(x)/x = 1
   ```

### Validation Integration

All hypotheses are automatically validated using the Phase 3 engine:

- **Symbolic Validation**: Structure and consistency checks
- **Numerical Testing**: Random value evaluation
- **Edge Case Testing**: Special values and boundary conditions
- **Confidence Scoring**: Combined validation metrics

### Hypothesis Status

- **VALIDATED**: Passed validation with high confidence
- **PROMISING**: Partial validation success
- **REJECTED**: Failed validation tests
- **ERROR**: Unable to validate due to parsing issues

## Embedding and Visualization

### Embedding Methods

1. **Structural Embeddings**: Based on mathematical properties
   - Symbol counts and types
   - Operation frequencies
   - Function usage patterns
   - Complexity measures
   - Mathematical properties (polynomial, rational, etc.)

2. **TF-IDF Embeddings**: Text-based similarity
   - Formula string preprocessing
   - N-gram extraction
   - Term frequency analysis

3. **Hybrid Embeddings**: Combined approach
   - Normalized structural features
   - TF-IDF text features
   - Weighted concatenation

### Visualization Components

1. **Cluster Plots**: 2D scatter plots of formula clusters
2. **Similarity Matrices**: Heatmaps of pairwise similarities
3. **Knowledge Graphs**: Network diagrams of concept relationships
4. **Pattern Summaries**: Statistical cluster analysis

### Dimension Reduction

- **t-SNE**: Non-linear manifold learning
- **UMAP**: Uniform manifold approximation
- **PCA**: Linear principal component analysis

## Results Interpretation

### Pattern Analysis

```json
{
  "cluster_id": 0,
  "formulas": ["x**2 + 1", "y**2 + 1", "a**2 + b**2"],
  "cluster_center": "x**2 + 1",
  "common_patterns": {
    "common_symbols": ["variable_letter"],
    "common_operations": ["Add", "Pow", "Integer"],
    "complexity_range": [2, 3]
  },
  "confidence_score": 0.85
}
```

**Interpretation**: This cluster contains quadratic expressions with similar structure, suggesting a pattern of "square plus constant" formulas.

### Gap Analysis

```json
{
  "gap_type": "sparse_connection",
  "title": "Isolated Trigonometric Identity",
  "description": "sin²(x) + cos²(x) = 1 has few connections",
  "confidence_score": 0.8,
  "suggested_formulas": ["tan²(x) + 1 = sec²(x)"]
}
```

**Interpretation**: The fundamental trigonometric identity is isolated, suggesting missing related identities could strengthen the knowledge graph.

### Hypothesis Evaluation

```json
{
  "hypothesis_type": "generalization",
  "formula": "a*x**2 + b*x + c",
  "description": "General quadratic form",
  "confidence_score": 0.95,
  "validation_summary": {
    "status": "PASS",
    "pass_rate": 1.0,
    "total_tests": 100
  }
}
```

**Interpretation**: High-confidence generalization that passed all validation tests, likely representing a true mathematical relationship.

## CLI Reference

### Commands

#### explore-patterns
Main exploration command with comprehensive options:

```bash
python main.py explore-patterns [OPTIONS]
```

**Options:**
- `--source {processed,graph}`: Data source (default: processed)
- `--output-dir PATH`: Output directory (default: results/exploration)
- `--similarity-threshold FLOAT`: Similarity threshold for clustering (default: 0.7)
- `--min-cluster-size INT`: Minimum formulas per cluster (default: 2)
- `--max-hypotheses INT`: Maximum hypotheses per type (default: 10)
- `--embedding-method {structural,tfidf,hybrid}`: Embedding method (default: structural)
- `--visualize`: Generate visualization plots

### Example Workflows

```bash
# Quick exploration with default settings
python main.py explore-patterns

# Comprehensive analysis with visualization
python main.py explore-patterns \
  --source processed \
  --similarity-threshold 0.6 \
  --max-hypotheses 15 \
  --embedding-method hybrid \
  --visualize \
  --output-dir results/comprehensive_analysis

# Focus on gap detection
python main.py explore-patterns \
  --similarity-threshold 0.8 \
  --min-cluster-size 3 \
  --output-dir results/gap_analysis

# Hypothesis generation emphasis
python main.py explore-patterns \
  --max-hypotheses 20 \
  --embedding-method structural \
  --output-dir results/hypothesis_generation
```

## Output Files

### patterns.json
```json
{
  "discovery_metadata": {
    "total_clusters": 5,
    "similarity_threshold": 0.7,
    "discovery_time": 1649123456.789
  },
  "clusters": [...]
}
```

### gaps.json
```json
{
  "detection_metadata": {
    "total_gaps": 12,
    "gap_types": {"sparse_connection": 8, "disconnected_cluster": 4}
  },
  "gaps": [...]
}
```

### conjectures.json
```json
{
  "generation_metadata": {
    "total_hypotheses": 25,
    "status_distribution": {"validated": 20, "promising": 5}
  },
  "hypotheses": [...]
}
```

### Visualization Files
- `cluster_plot.png`: 2D scatter plot of formula clusters
- `similarity_matrix.png`: Heatmap of formula similarities
- `visualization_data.json`: Raw data for custom visualization

## Performance Considerations

- **Formula Count**: Large collections (>100) may require sampling
- **Embedding Dimensions**: Higher dimensions increase memory usage
- **Validation Scope**: More hypothesis types increase runtime
- **Visualization Limits**: Large datasets may need aggregation
- **Memory Usage**: Complex formulas and large clusters require more RAM

## Extending Phase 4

### Custom Similarity Metrics

```python
class CustomPatternFinder(PatternFinder):
    def _calculate_similarity(self, fp1, fp2):
        # Custom similarity logic
        domain_similarity = self._calculate_domain_overlap(fp1, fp2)
        return 0.5 * super()._calculate_similarity(fp1, fp2) + 0.5 * domain_similarity
```

### Additional Gap Types

```python
class ExtendedGapDetector(GapDetector):
    def _detect_proof_gaps(self):
        # Detect missing proof steps
        return gaps
```

### New Hypothesis Types

```python
class AdvancedHypothesisGenerator(HypothesisGenerator):
    def _generate_differential_equations(self, parsed_formulas):
        # Generate DE hypotheses
        return hypotheses
```

## Integration with External Tools

### Formal Proof Systems
- Export hypotheses to Lean/Coq format
- Generate proof sketches
- Validate against theorem databases

### Computer Algebra Systems
- Integrate with Mathematica/Maple
- Advanced symbolic computation
- Specialized domain analysis

### Machine Learning
- Neural formula embeddings
- Deep learning pattern recognition
- Automated theorem proving

## Troubleshooting

### Common Issues

1. **No Patterns Found**: Lower similarity threshold or increase formula diversity
2. **Memory Issues**: Reduce formula count or embedding dimensions
3. **Slow Performance**: Enable sampling or reduce hypothesis count
4. **Visualization Errors**: Check matplotlib/seaborn installation

### Debug Mode

```bash
python main.py explore-patterns --log-level DEBUG
```

### Performance Profiling

```python
import cProfile
cProfile.run('pattern_finder.find_patterns(formulas)')
```

## Future Directions (Phase 5+)

### Planned Enhancements

1. **Theorem Generation**: Automated theorem discovery and proof
2. **Interactive Exploration**: Web-based formula analysis interface
3. **Scientific Publication**: Automatic paper generation from discoveries
4. **Formal Verification**: Integration with proof assistants
5. **Collaborative Platform**: Multi-user mathematical exploration

### Research Applications

- **Mathematical Discovery**: Novel theorem identification
- **Education**: Pattern-based learning systems
- **Scientific Computing**: Formula optimization and simplification
- **AI Research**: Symbolic reasoning advancement

### Example Phase 5 Integration

```python
# Future integration with formal proof systems
from phase5.theorem_prover import TheoremProver
from phase5.publication_generator import PaperGenerator

# Generate theorems from validated hypotheses
prover = TheoremProver()
theorems = prover.prove_hypotheses(promising_hypotheses)

# Auto-generate research paper
generator = PaperGenerator()
paper = generator.create_paper(theorems, gaps, patterns)
```

## Citation and Contributions

When using Phase 4 results in research:

```bibtex
@software{mathbot_phase4,
  title={MathBot Phase 4: Symbolic Pattern Discovery \& Exploration},
  author={MathBot Development Team},
  year={2024},
  url={https://github.com/mathbot/phase4}
}
```

## Contributing

Contributions welcome for:
- New similarity metrics
- Additional gap detection methods
- Novel hypothesis generation techniques
- Visualization improvements
- Performance optimizations
- Documentation enhancements

See `CONTRIBUTING.md` for development guidelines. 