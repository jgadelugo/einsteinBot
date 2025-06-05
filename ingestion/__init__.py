"""
MathBot Ingestion Pipeline

This package contains modules for ingesting mathematical content from various
sources including PDFs and LaTeX documents.

Modules:
    parser: Extract text and formulas from PDF/LaTeX sources
    formula_extractor: Detect and extract LaTeX-style mathematical expressions
    cleaner: Clean and normalize mathematical symbols and expressions
"""

from .parser import PDFParser, LaTeXParser, extract_text_blocks, extract_latex_blocks
from .formula_extractor import FormulaExtractor, extract_math_expressions
from .cleaner import FormulaCleaner, normalize_symbols

__all__ = [
    "PDFParser",
    "LaTeXParser", 
    "extract_text_blocks",
    "extract_latex_blocks",
    "FormulaExtractor",
    "extract_math_expressions",
    "FormulaCleaner",
    "normalize_symbols",
]

__version__ = "0.1.0" 