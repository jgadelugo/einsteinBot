"""
Parser module for extracting text and mathematical content from PDFs and LaTeX sources.

This module provides classes and functions to parse mathematical documents,
extracting both plain text and mathematical expressions for further processing.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import fitz  # PyMuPDF
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser as PDFMinerParser
from io import StringIO

from config import (
    ExtractionMode,
    DEFAULT_EXTRACTION_MODE,
    PDF_MAX_PAGES,
    PDF_ENCODING,
    MIN_TEXT_LENGTH,
    ALL_MATH_DELIMITERS,
    logger,
)

class PDFParser:
    """
    Parser for extracting text and mathematical content from PDF files.
    
    This class provides methods to extract plain text and mathematical expressions
    from PDF documents using both pdfminer.six and PyMuPDF backends.
    """
    
    def __init__(self, backend: str = "pymupdf"):
        """
        Initialize the PDF parser.
        
        Args:
            backend: Backend to use for PDF parsing ("pymupdf" or "pdfminer")
        """
        self.backend = backend.lower()
        self.logger = logging.getLogger(f"{__name__}.PDFParser")
        
        if self.backend not in ["pymupdf", "pdfminer"]:
            raise ValueError(f"Unsupported backend: {backend}")
        
        self.logger.info(f"Initialized PDF parser with backend: {self.backend}")
    
    def extract_text(
        self,
        pdf_path: Union[str, Path],
        max_pages: Optional[int] = None,
        extraction_mode: str = DEFAULT_EXTRACTION_MODE
    ) -> Dict[str, Union[List[str], str]]:
        """
        Extract text and mathematical content from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            max_pages: Maximum number of pages to process (None for all)
            extraction_mode: What to extract ("text_only", "formulas_only", "both")
            
        Returns:
            Dictionary containing extracted content with keys:
            - "text_blocks": List of text blocks (if applicable)
            - "formulas": List of mathematical expressions (if applicable)
            - "metadata": Dictionary with document metadata
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            ValueError: If extraction mode is invalid
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if extraction_mode not in [ExtractionMode.TEXT_ONLY, ExtractionMode.FORMULAS_ONLY, ExtractionMode.BOTH]:
            raise ValueError(f"Invalid extraction mode: {extraction_mode}")
        
        max_pages = max_pages or PDF_MAX_PAGES
        
        self.logger.info(f"Extracting from PDF: {pdf_path} (mode: {extraction_mode})")
        
        if self.backend == "pymupdf":
            return self._extract_with_pymupdf(pdf_path, max_pages, extraction_mode)
        else:
            return self._extract_with_pdfminer(pdf_path, max_pages, extraction_mode)
    
    def _extract_with_pymupdf(
        self,
        pdf_path: Path,
        max_pages: int,
        extraction_mode: str
    ) -> Dict[str, Union[List[str], str]]:
        """Extract content using PyMuPDF backend."""
        try:
            doc = fitz.open(str(pdf_path))
            
            text_blocks = []
            all_text = ""
            
            for page_num in range(min(len(doc), max_pages)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                
                if len(page_text.strip()) >= MIN_TEXT_LENGTH:
                    text_blocks.append(page_text)
                    all_text += page_text + "\n"
            
            doc.close()
            
            result = {
                "metadata": {
                    "source": str(pdf_path),
                    "pages_processed": len(text_blocks),
                    "backend": self.backend,
                    "extraction_mode": extraction_mode,
                }
            }
            
            if extraction_mode in [ExtractionMode.TEXT_ONLY, ExtractionMode.BOTH]:
                result["text_blocks"] = text_blocks
            
            if extraction_mode in [ExtractionMode.FORMULAS_ONLY, ExtractionMode.BOTH]:
                from .formula_extractor import extract_math_expressions
                result["formulas"] = extract_math_expressions(all_text)
            
            self.logger.info(f"Successfully extracted content: {len(text_blocks)} pages")
            return result
            
        except Exception as e:
            self.logger.error(f"Error extracting with PyMuPDF: {e}")
            raise
    
    def _extract_with_pdfminer(
        self,
        pdf_path: Path,
        max_pages: int,
        extraction_mode: str
    ) -> Dict[str, Union[List[str], str]]:
        """Extract content using pdfminer.six backend."""
        try:
            with open(pdf_path, 'rb') as file:
                parser = PDFMinerParser(file)
                doc = PDFDocument(parser)
                
                rsrcmgr = PDFResourceManager()
                laparams = LAParams()
                
                text_blocks = []
                all_text = ""
                
                for page_num, page in enumerate(PDFPage.create_pages(doc)):
                    if page_num >= max_pages:
                        break
                    
                    output_string = StringIO()
                    device = TextConverter(rsrcmgr, output_string, laparams=laparams)
                    interpreter = PDFPageInterpreter(rsrcmgr, device)
                    interpreter.process_page(page)
                    
                    page_text = output_string.getvalue()
                    
                    if len(page_text.strip()) >= MIN_TEXT_LENGTH:
                        text_blocks.append(page_text)
                        all_text += page_text + "\n"
                    
                    device.close()
                    output_string.close()
            
            result = {
                "metadata": {
                    "source": str(pdf_path),
                    "pages_processed": len(text_blocks),
                    "backend": self.backend,
                    "extraction_mode": extraction_mode,
                }
            }
            
            if extraction_mode in [ExtractionMode.TEXT_ONLY, ExtractionMode.BOTH]:
                result["text_blocks"] = text_blocks
            
            if extraction_mode in [ExtractionMode.FORMULAS_ONLY, ExtractionMode.BOTH]:
                from .formula_extractor import extract_math_expressions
                result["formulas"] = extract_math_expressions(all_text)
            
            self.logger.info(f"Successfully extracted content: {len(text_blocks)} pages")
            return result
            
        except Exception as e:
            self.logger.error(f"Error extracting with pdfminer: {e}")
            raise


class LaTeXParser:
    """
    Parser for extracting text and mathematical content from LaTeX documents.
    
    This class provides methods to parse LaTeX source files and extract
    both textual content and mathematical expressions.
    """
    
    def __init__(self):
        """Initialize the LaTeX parser."""
        self.logger = logging.getLogger(f"{__name__}.LaTeXParser")
        self.logger.info("Initialized LaTeX parser")
    
    def extract_text(
        self,
        latex_path: Union[str, Path],
        extraction_mode: str = DEFAULT_EXTRACTION_MODE
    ) -> Dict[str, Union[List[str], str]]:
        """
        Extract text and mathematical content from a LaTeX file.
        
        Args:
            latex_path: Path to the LaTeX file
            extraction_mode: What to extract ("text_only", "formulas_only", "both")
            
        Returns:
            Dictionary containing extracted content with keys:
            - "text_blocks": List of text blocks (if applicable)
            - "formulas": List of mathematical expressions (if applicable)
            - "metadata": Dictionary with document metadata
            
        Raises:
            FileNotFoundError: If LaTeX file doesn't exist
            ValueError: If extraction mode is invalid
        """
        latex_path = Path(latex_path)
        if not latex_path.exists():
            raise FileNotFoundError(f"LaTeX file not found: {latex_path}")
        
        if extraction_mode not in [ExtractionMode.TEXT_ONLY, ExtractionMode.FORMULAS_ONLY, ExtractionMode.BOTH]:
            raise ValueError(f"Invalid extraction mode: {extraction_mode}")
        
        self.logger.info(f"Extracting from LaTeX: {latex_path} (mode: {extraction_mode})")
        
        try:
            with open(latex_path, 'r', encoding=PDF_ENCODING) as file:
                content = file.read()
            
            # Remove LaTeX comments
            content = re.sub(r'%.*?\n', '\n', content)
            
            result = {
                "metadata": {
                    "source": str(latex_path),
                    "extraction_mode": extraction_mode,
                }
            }
            
            if extraction_mode in [ExtractionMode.TEXT_ONLY, ExtractionMode.BOTH]:
                text_blocks = self._extract_text_blocks(content)
                result["text_blocks"] = text_blocks
            
            if extraction_mode in [ExtractionMode.FORMULAS_ONLY, ExtractionMode.BOTH]:
                from .formula_extractor import extract_math_expressions
                result["formulas"] = extract_math_expressions(content)
            
            self.logger.info(f"Successfully extracted LaTeX content")
            return result
            
        except Exception as e:
            self.logger.error(f"Error extracting LaTeX content: {e}")
            raise
    
    def _extract_text_blocks(self, content: str) -> List[str]:
        """
        Extract text blocks from LaTeX content, excluding math environments.
        
        Args:
            content: Raw LaTeX content
            
        Returns:
            List of text blocks with LaTeX commands removed
        """
        # Remove math environments first
        from .formula_extractor import FormulaExtractor
        extractor = FormulaExtractor()
        
        # Get all math expressions to exclude them from text
        math_expressions = extractor.extract_expressions(content)
        
        # Create a copy of content with math expressions removed
        text_content = content
        for expr in math_expressions:
            text_content = text_content.replace(expr["raw_expression"], " ")
        
        # Remove common LaTeX commands
        text_content = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text_content)
        text_content = re.sub(r'\\[a-zA-Z]+', '', text_content)
        text_content = re.sub(r'\{|\}', '', text_content)
        
        # Split into paragraphs and filter
        paragraphs = [p.strip() for p in text_content.split('\n\n') if p.strip()]
        text_blocks = [p for p in paragraphs if len(p) >= MIN_TEXT_LENGTH]
        
        return text_blocks


# Convenience functions for backward compatibility and ease of use

def extract_text_blocks(
    source_path: Union[str, Path],
    source_type: str = "auto",
    max_pages: Optional[int] = None
) -> List[str]:
    """
    Extract text blocks from a document.
    
    Args:
        source_path: Path to the source document
        source_type: Type of source ("pdf", "latex", or "auto")
        max_pages: Maximum pages to process (PDF only)
        
    Returns:
        List of extracted text blocks
    """
    source_path = Path(source_path)
    
    if source_type == "auto":
        source_type = "pdf" if source_path.suffix.lower() == ".pdf" else "latex"
    
    if source_type == "pdf":
        parser = PDFParser()
        result = parser.extract_text(source_path, max_pages, ExtractionMode.TEXT_ONLY)
        return result.get("text_blocks", [])
    elif source_type == "latex":
        parser = LaTeXParser()
        result = parser.extract_text(source_path, ExtractionMode.TEXT_ONLY)
        return result.get("text_blocks", [])
    else:
        raise ValueError(f"Unsupported source type: {source_type}")


def extract_latex_blocks(
    source_path: Union[str, Path],
    source_type: str = "auto"
) -> List[str]:
    """
    Extract LaTeX mathematical expressions from a document.
    
    Args:
        source_path: Path to the source document
        source_type: Type of source ("pdf", "latex", or "auto")
        
    Returns:
        List of extracted mathematical expressions
    """
    source_path = Path(source_path)
    
    if source_type == "auto":
        source_type = "pdf" if source_path.suffix.lower() == ".pdf" else "latex"
    
    if source_type == "pdf":
        parser = PDFParser()
        result = parser.extract_text(source_path, extraction_mode=ExtractionMode.FORMULAS_ONLY)
        return result.get("formulas", [])
    elif source_type == "latex":
        parser = LaTeXParser()
        result = parser.extract_text(source_path, ExtractionMode.FORMULAS_ONLY)
        return result.get("formulas", [])
    else:
        raise ValueError(f"Unsupported source type: {source_type}")


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract content from PDF or LaTeX files")
    parser.add_argument("file_path", help="Path to the file to process")
    parser.add_argument("--type", choices=["pdf", "latex", "auto"], default="auto",
                        help="File type (auto-detected by default)")
    parser.add_argument("--mode", choices=["text_only", "formulas_only", "both"], 
                        default="both", help="Extraction mode")
    parser.add_argument("--max-pages", type=int, help="Maximum pages to process (PDF only)")
    
    args = parser.parse_args()
    
    try:
        if args.type in ["pdf", "auto"] and Path(args.file_path).suffix.lower() == ".pdf":
            parser_instance = PDFParser()
            result = parser_instance.extract_text(args.file_path, args.max_pages, args.mode)
        else:
            parser_instance = LaTeXParser()
            result = parser_instance.extract_text(args.file_path, args.mode)
        
        print(f"Extraction completed successfully!")
        print(f"Metadata: {result['metadata']}")
        
        if "text_blocks" in result:
            print(f"Text blocks found: {len(result['text_blocks'])}")
        
        if "formulas" in result:
            print(f"Formulas found: {len(result['formulas'])}")
            
    except Exception as e:
        print(f"Error: {e}")
        exit(1) 