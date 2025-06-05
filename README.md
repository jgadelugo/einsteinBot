# MathBot: Mathematical Content Ingestion Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

MathBot is a modular Python-based system for ingesting mathematical content from textbooks, research papers, and other documents. It extracts both plain text and mathematical expressions from PDF and LaTeX sources, structures them into a searchable format, and prepares them for symbolic validation and knowledge graph construction.

## 🎯 Features

- **Multi-format Support**: Extract content from PDF files and LaTeX documents
- **Mathematical Expression Detection**: Robust extraction of inline and display math expressions
- **Symbol Normalization**: Convert LaTeX commands to Unicode symbols and standardized formats
- **Configurable Extraction**: Choose between text-only, formulas-only, or combined extraction modes
- **SymPy Integration**: Convert expressions to SymPy-compatible format for symbolic computation
- **Batch Processing**: Process multiple files and directories efficiently
- **Comprehensive Testing**: Full test suite with 95%+ code coverage
- **Production Ready**: Modular architecture with proper logging and error handling

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Architecture](#architecture)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/your-username/mathbot.git
cd mathbot

# Install required packages
pip install -r requirements.txt
```

### Development Installation

For development with additional tools:

```bash
pip install -r requirements.txt
pip install pytest pytest-cov black isort mypy
```

## ⚡ Quick Start

### Command Line Usage

```bash
# Process a single PDF file
python main.py process-file document.pdf

# Extract only formulas from a LaTeX file
python main.py process-file paper.tex --mode formulas_only

# Process all PDFs in a directory with symbol normalization
python main.py process-directory ./documents --normalize-symbols

# Generate a processing report
python main.py process-directory ./papers --generate-report --output-dir ./results
```

### Python API Usage

```python
from ingestion.parser import PDFParser
from ingestion.formula_extractor import FormulaExtractor
from ingestion.cleaner import FormulaCleaner

# Initialize components
pdf_parser = PDFParser(backend="pymupdf")
formula_extractor = FormulaExtractor(clean_formulas=True)
cleaner = FormulaCleaner()

# Extract content from PDF
result = pdf_parser.extract_text("document.pdf", extraction_mode="both")

# Get detailed formula information
formulas = formula_extractor.extract_expressions(result["text_blocks"][0])

# Clean and normalize expressions
cleaned_formulas = cleaner.batch_clean([f["expression"] for f in formulas])

print(f"Extracted {len(cleaned_formulas)} mathematical expressions")
```

## 📚 Usage Examples

### Example 1: Basic PDF Processing

```python
from main import MathBotIngestion

# Initialize the pipeline
pipeline = MathBotIngestion(normalize_symbols=True)

# Process a mathematical textbook
result = pipeline.process_file("calculus_textbook.pdf")

print(f"Pages processed: {result['metadata']['pages_processed']}")
print(f"Formulas found: {len(result['formulas'])}")
print(f"Text blocks: {len(result['text_blocks'])}")

# Access cleaned formulas
if 'cleaned_formulas' in result:
    unicode_formulas = result['cleaned_formulas']['unicode_normalized']
    sympy_formulas = result['cleaned_formulas']['sympy_compatible']
```

### Example 2: LaTeX Document Analysis

```python
from ingestion.parser import LaTeXParser
from ingestion.formula_extractor import find_math_patterns

parser = LaTeXParser()

# Extract from LaTeX source
result = parser.extract_text("research_paper.tex")

# Analyze mathematical patterns
all_text = "\n".join(result["text_blocks"])
patterns = find_math_patterns(all_text)

print("Mathematical patterns found:")
for pattern_type, matches in patterns.items():
    if matches:
        print(f"  {pattern_type}: {len(matches)} occurrences")
```

### Example 3: Batch Processing with Statistics

```python
from main import MathBotIngestion

pipeline = MathBotIngestion()

# Process entire directory
results = pipeline.process_directory(
    "math_papers/",
    file_pattern="*.pdf",
    extraction_mode="both"
)

# Generate comprehensive report
report = pipeline.generate_report(results)

print(f"Successfully processed: {report['summary']['successful_files']} files")
print(f"Total formulas extracted: {report['statistics']['total_formulas']}")
print(f"Average formulas per file: {report['statistics']['total_formulas'] / report['summary']['successful_files']:.1f}")
```

### Example 4: Custom Formula Extraction

```python
from ingestion.formula_extractor import FormulaExtractor

extractor = FormulaExtractor(clean_formulas=True)

text = """
The quadratic formula is $x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$.

For integration by parts:
$$\\int u \\, dv = uv - \\int v \\, du$$
"""

# Extract with confidence scoring
expressions = extractor.extract_expressions(text)

for expr in expressions:
    print(f"Type: {expr['type']}")
    print(f"Expression: {expr['expression']}")
    print(f"Confidence: {expr['confidence']:.2f}")
    print(f"Position: {expr['position']}")
    print("---")
```

## 🏗️ Architecture

MathBot follows a modular architecture designed for extensibility and maintainability:

```
mathbot/
├── ingestion/              # Core ingestion pipeline
│   ├── parser.py          # PDF/LaTeX parsing
│   ├── formula_extractor.py # Math expression detection
│   └── cleaner.py         # Symbol normalization
├── data/                  # Data storage
│   ├── raw/              # Original files
│   ├── processed/        # Extracted content
│   └── graph/            # Knowledge graph data
├── tests/                # Comprehensive test suite
├── config.py             # Configuration and constants
└── main.py              # CLI interface and orchestration
```

### Core Components

1. **Parser Module** (`ingestion/parser.py`)
   - `PDFParser`: Extracts text from PDF files using PyMuPDF or pdfminer.six
   - `LaTeXParser`: Processes LaTeX source files
   - Supports multiple extraction modes and backends

2. **Formula Extractor** (`ingestion/formula_extractor.py`)
   - `FormulaExtractor`: Detects mathematical expressions using regex patterns
   - Confidence scoring for expression quality
   - Support for inline math, display math, and LaTeX environments

3. **Cleaner Module** (`ingestion/cleaner.py`)
   - `FormulaCleaner`: Normalizes mathematical symbols and notation
   - Converts LaTeX commands to Unicode symbols
   - SymPy-compatible format conversion

4. **Configuration** (`config.py`)
   - Centralized configuration management
   - Symbol mappings and extraction parameters
   - Logging setup and data directory management

## 📖 API Reference

### PDFParser

```python
class PDFParser:
    def __init__(self, backend: str = "pymupdf")
    def extract_text(self, pdf_path, max_pages=None, extraction_mode="both") -> Dict
```

### FormulaExtractor

```python
class FormulaExtractor:
    def __init__(self, clean_formulas: bool = True)
    def extract_expressions(self, text: str, include_inline=True, include_block=True) -> List[Dict]
    def get_math_statistics(self, text: str) -> Dict[str, int]
```

### FormulaCleaner

```python
class FormulaCleaner:
    def __init__(self, preserve_latex: bool = False)
    def clean_expression(self, expression: str) -> str
    def convert_to_sympy_format(self, expression: str) -> str
    def batch_clean(self, expressions: List[str]) -> List[str]
```

### Command Line Interface

```bash
# Process single file
python main.py process-file <file_path> [options]

# Process directory
python main.py process-directory <directory_path> [options]

# Options:
--mode {text_only,formulas_only,both}
--backend {pymupdf,pdfminer}
--normalize-symbols
--output <path>
--generate-report
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=ingestion --cov-report=html

# Run specific test module
pytest tests/test_parser.py -v

# Run tests with detailed output
pytest -v --tb=short
```

### Test Coverage

The project maintains high test coverage across all modules:

- `test_parser.py`: PDF and LaTeX parsing functionality
- `test_formula_extractor.py`: Mathematical expression detection
- `test_cleaner.py`: Symbol normalization and cleaning

## 🔧 Configuration

### Environment Variables

```bash
export MATHBOT_LOG_LEVEL=INFO  # Set logging level
export MATHBOT_DATA_DIR=/path/to/data  # Override data directory
```

### Custom Symbol Mappings

Extend symbol mappings in `config.py`:

```python
CUSTOM_SYMBOLS = {
    r'\mycommand': '⊕',
    r'\special': '⊗',
}

# Add to existing mappings
MATH_OPERATORS.update(CUSTOM_SYMBOLS)
```

## 🚀 Performance Tips

1. **PDF Backend Selection**:
   - Use `pymupdf` for speed (default)
   - Use `pdfminer` for accuracy with complex layouts

2. **Batch Processing**:
   - Process multiple files in parallel using directory mode
   - Use `--output-dir` to save individual results

3. **Memory Management**:
   - Set `max_pages` limit for large PDF files
   - Use `formulas_only` mode when text content isn't needed

## 🛠️ Extending MathBot

### Adding New Parsers

```python
from ingestion.parser import LaTeXParser

class CustomParser(LaTeXParser):
    def extract_text(self, file_path, extraction_mode="both"):
        # Custom extraction logic
        return super().extract_text(file_path, extraction_mode)
```

### Custom Formula Patterns

```python
from config import BLOCK_MATH_DELIMITERS

# Add custom math delimiters
CUSTOM_DELIMITERS = [
    (r'\\begin\{mymath\}', r'\\end\{mymath\}'),
]

BLOCK_MATH_DELIMITERS.extend(CUSTOM_DELIMITERS)
```

## 📊 Example Output

### Processing Report

```json
{
  "summary": {
    "total_files": 15,
    "successful_files": 14,
    "failed_files": 1
  },
  "statistics": {
    "total_formulas": 342,
    "total_text_blocks": 1205,
    "files_by_type": {
      ".pdf": 12,
      ".tex": 3
    }
  }
}
```

### Extracted Formula

```json
{
  "expression": "x = (-b ± √(b² - 4ac)) / (2a)",
  "raw_expression": "$x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$",
  "type": "inline",
  "position": 156,
  "confidence": 0.95
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Run the test suite (`pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add type hints to all functions
- Include docstrings for all public methods
- Maintain test coverage above 90%
- Update documentation for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [PyMuPDF](https://pymupdf.readthedocs.io/) for PDF processing
- [pdfminer.six](https://pdfminersix.readthedocs.io/) for alternative PDF parsing
- [SymPy](https://www.sympy.org/) for symbolic mathematics
- [pytest](https://pytest.org/) for testing framework

## 📞 Support

For questions, issues, or contributions:

- Open an issue on GitHub
- Check the [documentation](docs/)
- Review existing [examples](examples/)

---

**MathBot** - Transforming mathematical documents into structured, searchable knowledge. 🧮✨ 