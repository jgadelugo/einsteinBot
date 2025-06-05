"""
Configuration module for MathBot ingestion pipeline.

This module contains all constants, configuration options, and environment
settings used throughout the ingestion pipeline.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Set

# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
GRAPH_DATA_DIR = DATA_DIR / "graph"

# Ensure data directories exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, GRAPH_DATA_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# LATEX MATH DELIMITERS
# ============================================================================

# Inline math delimiters
INLINE_MATH_DELIMITERS = [
    (r'\$', r'\$'),                    # Standard LaTeX inline: $...$
    (r'\\(', r'\\)'),                  # LaTeX inline: \(...\)
]

# Block math delimiters
BLOCK_MATH_DELIMITERS = [
    (r'\$\$', r'\$\$'),                # Display math: $$...$$
    (r'\\[', r'\\]'),                  # LaTeX display: \[...\]
    (r'\\begin\{equation\}', r'\\end\{equation\}'),
    (r'\\begin\{equation\*\}', r'\\end\{equation\*\}'),
    (r'\\begin\{align\}', r'\\end\{align\}'),
    (r'\\begin\{align\*\}', r'\\end\{align\*\}'),
    (r'\\begin\{gather\}', r'\\end\{gather\}'),
    (r'\\begin\{gather\*\}', r'\\end\{gather\*\}'),
    (r'\\begin\{multline\}', r'\\end\{multline\}'),
    (r'\\begin\{multline\*\}', r'\\end\{multline\*\}'),
]

# All math delimiters combined
ALL_MATH_DELIMITERS = INLINE_MATH_DELIMITERS + BLOCK_MATH_DELIMITERS

# ============================================================================
# SYMBOL NORMALIZATION MAPPINGS
# ============================================================================

# Greek letters mapping
GREEK_LETTERS: Dict[str, str] = {
    r'\alpha': 'α', r'\beta': 'β', r'\gamma': 'γ', r'\delta': 'δ',
    r'\epsilon': 'ε', r'\varepsilon': 'ε', r'\zeta': 'ζ', r'\eta': 'η',
    r'\theta': 'θ', r'\vartheta': 'θ', r'\iota': 'ι', r'\kappa': 'κ',
    r'\lambda': 'λ', r'\mu': 'μ', r'\nu': 'ν', r'\xi': 'ξ',
    r'\omicron': 'ο', r'\pi': 'π', r'\varpi': 'π', r'\rho': 'ρ',
    r'\varrho': 'ρ', r'\sigma': 'σ', r'\varsigma': 'ς', r'\tau': 'τ',
    r'\upsilon': 'υ', r'\phi': 'φ', r'\varphi': 'φ', r'\chi': 'χ',
    r'\psi': 'ψ', r'\omega': 'ω',
    # Capital Greek letters
    r'\Gamma': 'Γ', r'\Delta': 'Δ', r'\Theta': 'Θ', r'\Lambda': 'Λ',
    r'\Xi': 'Ξ', r'\Pi': 'Π', r'\Sigma': 'Σ', r'\Upsilon': 'Υ',
    r'\Phi': 'Φ', r'\Psi': 'Ψ', r'\Omega': 'Ω',
}

# Mathematical operators and symbols
MATH_OPERATORS: Dict[str, str] = {
    r'\cdot': '*',
    r'\times': '*',
    r'\div': '/',
    r'\pm': '±',
    r'\mp': '∓',
    r'\leq': '≤',
    r'\geq': '≥',
    r'\neq': '≠',
    r'\approx': '≈',
    r'\equiv': '≡',
    r'\sim': '∼',
    r'\simeq': '≃',
    r'\propto': '∝',
    r'\infty': '∞',
    r'\partial': '∂',
    r'\nabla': '∇',
    r'\int': '∫',
    r'\sum': '∑',
    r'\prod': '∏',
    r'\sqrt': 'sqrt',
    r'\lim': 'lim',
    r'\sin': 'sin',
    r'\cos': 'cos',
    r'\tan': 'tan',
    r'\log': 'log',
    r'\ln': 'ln',
    r'\exp': 'exp',
}

# Fraction patterns
FRACTION_PATTERNS: List[str] = [
    r'\\frac\{([^{}]+)\}\{([^{}]+)\}',  # Simple fractions
    r'\\dfrac\{([^{}]+)\}\{([^{}]+)\}', # Display fractions
    r'\\tfrac\{([^{}]+)\}\{([^{}]+)\}', # Text fractions
]

# ============================================================================
# EXTRACTION CONFIGURATION
# ============================================================================

class ExtractionMode:
    """Enumeration of extraction modes."""
    TEXT_ONLY = "text_only"
    FORMULAS_ONLY = "formulas_only"
    BOTH = "both"

# Default extraction settings
DEFAULT_EXTRACTION_MODE = ExtractionMode.BOTH
DEFAULT_CLEAN_FORMULAS = True
DEFAULT_DEDUPLICATE = True

# PDF processing settings
PDF_MAX_PAGES = 1000  # Maximum pages to process per PDF
PDF_ENCODING = "utf-8"
MIN_TEXT_LENGTH = 10  # Minimum text length to consider valid

# Formula extraction settings
MIN_FORMULA_LENGTH = 3
MAX_FORMULA_LENGTH = 1000
FORMULA_CONFIDENCE_THRESHOLD = 0.7

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Logging levels
LOG_LEVEL = os.getenv("MATHBOT_LOG_LEVEL", "INFO").upper()
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# File patterns to ignore during processing
IGNORE_PATTERNS: Set[str] = {
    "*.tmp", "*.temp", "*.bak", "*.swp", "*.log",
    ".*", "__pycache__", "*.pyc", "*.pyo",
}

# ============================================================================
# TOPIC TAXONOMY (Initial seed)
# ============================================================================

MATH_TOPICS = {
    "Calculus": [
        "Limits", "Derivatives", "Integrals", "Series", "Differential Equations"
    ],
    "Algebra": [
        "Polynomials", "Equations", "Matrices", "Linear Algebra", "Abstract Algebra"
    ],
    "Geometry": [
        "Euclidean", "Analytic", "Transformations", "Topology", "Differential Geometry"
    ],
    "Analysis": [
        "Real Analysis", "Complex Analysis", "Functional Analysis", "Measure Theory"
    ],
    "Probability": [
        "Probability Theory", "Statistics", "Stochastic Processes", "Bayesian Analysis"
    ],
    "Number Theory": [
        "Elementary", "Algebraic", "Analytic", "Modular Forms"
    ],
}

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(
    level: str = LOG_LEVEL,
    format_str: str = LOG_FORMAT,
    date_format: str = LOG_DATE_FORMAT
) -> logging.Logger:
    """
    Set up logging configuration for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_str: Format string for log messages
        date_format: Date format for timestamps
        
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=getattr(logging, level),
        format=format_str,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(PROJECT_ROOT / "mathbot.log")
        ]
    )
    
    logger = logging.getLogger("mathbot")
    logger.info(f"Logging initialized at level: {level}")
    return logger

# Default logger instance
logger = setup_logging() 