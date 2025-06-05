"""
Tests for the parser module.

This module contains tests for PDF and LaTeX parsing functionality,
including text extraction, formula detection, and error handling.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from config import ExtractionMode
from ingestion.parser import PDFParser, LaTeXParser, extract_text_blocks, extract_latex_blocks


class TestPDFParser:
    """Test suite for PDFParser class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = PDFParser(backend="pymupdf")
        
    def test_init_valid_backend(self):
        """Test parser initialization with valid backends."""
        parser_pymupdf = PDFParser(backend="pymupdf")
        assert parser_pymupdf.backend == "pymupdf"
        
        parser_pdfminer = PDFParser(backend="pdfminer")
        assert parser_pdfminer.backend == "pdfminer"
    
    def test_init_invalid_backend(self):
        """Test parser initialization with invalid backend."""
        with pytest.raises(ValueError, match="Unsupported backend"):
            PDFParser(backend="invalid_backend")
    
    def test_extract_text_nonexistent_file(self):
        """Test extraction from non-existent file."""
        with pytest.raises(FileNotFoundError):
            self.parser.extract_text("nonexistent.pdf")
    
    def test_extract_text_invalid_mode(self):
        """Test extraction with invalid extraction mode."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            with pytest.raises(ValueError, match="Invalid extraction mode"):
                self.parser.extract_text(tmp_path, extraction_mode="invalid_mode")
        finally:
            tmp_path.unlink()
    
    @patch('fitz.open')
    def test_extract_with_pymupdf_success(self, mock_fitz_open):
        """Test successful extraction with PyMuPDF backend."""
        # Mock PyMuPDF document
        mock_doc = Mock()
        mock_page = Mock()
        mock_page.get_text.return_value = "Sample text with $x = y + z$ formula"
        mock_doc.load_page.return_value = mock_page
        mock_doc.__len__.return_value = 1
        mock_fitz_open.return_value = mock_doc
        
        # Create temporary PDF file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            result = self.parser.extract_text(tmp_path, extraction_mode=ExtractionMode.BOTH)
            
            assert "text_blocks" in result
            assert "formulas" in result
            assert "metadata" in result
            assert result["metadata"]["backend"] == "pymupdf"
            assert len(result["text_blocks"]) == 1
            
        finally:
            tmp_path.unlink()
    
    @patch('fitz.open')
    def test_extract_with_pymupdf_empty_pages(self, mock_fitz_open):
        """Test extraction with empty pages."""
        # Mock PyMuPDF document with empty pages
        mock_doc = Mock()
        mock_page = Mock()
        mock_page.get_text.return_value = ""  # Empty page
        mock_doc.load_page.return_value = mock_page
        mock_doc.__len__.return_value = 1
        mock_fitz_open.return_value = mock_doc
        
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            result = self.parser.extract_text(tmp_path, extraction_mode=ExtractionMode.TEXT_ONLY)
            
            assert "text_blocks" in result
            assert len(result["text_blocks"]) == 0  # No text blocks due to minimum length
            
        finally:
            tmp_path.unlink()
    
    def test_extract_text_only_mode(self):
        """Test extraction with text-only mode."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            with patch('fitz.open') as mock_fitz_open:
                mock_doc = Mock()
                mock_page = Mock()
                mock_page.get_text.return_value = "This is sample text content for testing."
                mock_doc.load_page.return_value = mock_page
                mock_doc.__len__.return_value = 1
                mock_fitz_open.return_value = mock_doc
                
                result = self.parser.extract_text(tmp_path, extraction_mode=ExtractionMode.TEXT_ONLY)
                
                assert "text_blocks" in result
                assert "formulas" not in result
                assert result["metadata"]["extraction_mode"] == ExtractionMode.TEXT_ONLY
                
        finally:
            tmp_path.unlink()
    
    def test_extract_formulas_only_mode(self):
        """Test extraction with formulas-only mode."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            with patch('fitz.open') as mock_fitz_open:
                mock_doc = Mock()
                mock_page = Mock()
                mock_page.get_text.return_value = "Text with $E = mc^2$ formula"
                mock_doc.load_page.return_value = mock_page
                mock_doc.__len__.return_value = 1
                mock_fitz_open.return_value = mock_doc
                
                result = self.parser.extract_text(tmp_path, extraction_mode=ExtractionMode.FORMULAS_ONLY)
                
                assert "formulas" in result
                assert "text_blocks" not in result
                assert result["metadata"]["extraction_mode"] == ExtractionMode.FORMULAS_ONLY
                
        finally:
            tmp_path.unlink()


class TestLaTeXParser:
    """Test suite for LaTeXParser class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = LaTeXParser()
    
    def test_init(self):
        """Test LaTeX parser initialization."""
        assert self.parser is not None
    
    def test_extract_text_nonexistent_file(self):
        """Test extraction from non-existent file."""
        with pytest.raises(FileNotFoundError):
            self.parser.extract_text("nonexistent.tex")
    
    def test_extract_text_invalid_mode(self):
        """Test extraction with invalid extraction mode."""
        with tempfile.NamedTemporaryFile(suffix=".tex", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(b"\\documentclass{article}")
        
        try:
            with pytest.raises(ValueError, match="Invalid extraction mode"):
                self.parser.extract_text(tmp_path, extraction_mode="invalid_mode")
        finally:
            tmp_path.unlink()
    
    def test_extract_simple_latex(self):
        """Test extraction from simple LaTeX content."""
        latex_content = r"""
        \documentclass{article}
        \begin{document}
        
        This is a simple document with mathematics.
        
        The quadratic formula is $x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$.
        
        We can also have display equations:
        $$\int_0^\infty e^{-x^2} dx = \frac{\sqrt{\pi}}{2}$$
        
        \end{document}
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix=".tex", delete=False, encoding='utf-8') as tmp:
            tmp.write(latex_content)
            tmp_path = Path(tmp.name)
        
        try:
            result = self.parser.extract_text(tmp_path, extraction_mode=ExtractionMode.BOTH)
            
            assert "text_blocks" in result
            assert "formulas" in result
            assert "metadata" in result
            assert len(result["text_blocks"]) > 0
            assert len(result["formulas"]) > 0
            
        finally:
            tmp_path.unlink()
    
    def test_extract_text_only_mode(self):
        """Test LaTeX extraction with text-only mode."""
        latex_content = r"""
        \section{Introduction}
        
        This section introduces the basic concepts of mathematics.
        Mathematics is the study of numbers, shapes, and patterns.
        
        \section{Methods}
        
        We will use various mathematical tools in this analysis.
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix=".tex", delete=False, encoding='utf-8') as tmp:
            tmp.write(latex_content)
            tmp_path = Path(tmp.name)
        
        try:
            result = self.parser.extract_text(tmp_path, extraction_mode=ExtractionMode.TEXT_ONLY)
            
            assert "text_blocks" in result
            assert "formulas" not in result
            assert len(result["text_blocks"]) > 0
            
            # Check that text blocks contain actual content
            text_content = " ".join(result["text_blocks"])
            assert "mathematics" in text_content.lower()
            assert "concepts" in text_content.lower()
            
        finally:
            tmp_path.unlink()
    
    def test_extract_formulas_only_mode(self):
        """Test LaTeX extraction with formulas-only mode."""
        latex_content = r"""
        The area of a circle is $A = \pi r^2$.
        
        For the integral:
        \begin{equation}
        \int_0^{2\pi} \sin(x) dx = 0
        \end{equation}
        
        And the famous equation:
        $$E = mc^2$$
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix=".tex", delete=False, encoding='utf-8') as tmp:
            tmp.write(latex_content)
            tmp_path = Path(tmp.name)
        
        try:
            result = self.parser.extract_text(tmp_path, extraction_mode=ExtractionMode.FORMULAS_ONLY)
            
            assert "formulas" in result
            assert "text_blocks" not in result
            assert len(result["formulas"]) > 0
            
        finally:
            tmp_path.unlink()
    
    def test_extract_with_comments(self):
        """Test LaTeX extraction with comments."""
        latex_content = r"""
        % This is a comment
        \section{Test}
        
        This is regular text. % Another comment
        
        The formula is $x = y + z$. % Formula comment
        
        % More comments
        Another paragraph here.
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix=".tex", delete=False, encoding='utf-8') as tmp:
            tmp.write(latex_content)
            tmp_path = Path(tmp.name)
        
        try:
            result = self.parser.extract_text(tmp_path, extraction_mode=ExtractionMode.BOTH)
            
            # Comments should be removed from processing
            text_content = " ".join(result["text_blocks"])
            assert "This is a comment" not in text_content
            assert "Another comment" not in text_content
            assert "regular text" in text_content
            
        finally:
            tmp_path.unlink()


class TestConvenienceFunctions:
    """Test suite for convenience functions."""
    
    def test_extract_text_blocks_pdf(self):
        """Test extract_text_blocks function with PDF."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            with patch('fitz.open') as mock_fitz_open:
                mock_doc = Mock()
                mock_page = Mock()
                mock_page.get_text.return_value = "Sample text for testing extraction"
                mock_doc.load_page.return_value = mock_page
                mock_doc.__len__.return_value = 1
                mock_fitz_open.return_value = mock_doc
                
                blocks = extract_text_blocks(tmp_path, source_type="pdf")
                
                assert isinstance(blocks, list)
                assert len(blocks) == 1
                assert "Sample text" in blocks[0]
                
        finally:
            tmp_path.unlink()
    
    def test_extract_text_blocks_latex(self):
        """Test extract_text_blocks function with LaTeX."""
        latex_content = "This is a test document with some content."
        
        with tempfile.NamedTemporaryFile(mode='w', suffix=".tex", delete=False, encoding='utf-8') as tmp:
            tmp.write(latex_content)
            tmp_path = Path(tmp.name)
        
        try:
            blocks = extract_text_blocks(tmp_path, source_type="latex")
            
            assert isinstance(blocks, list)
            assert len(blocks) >= 0  # May be 0 if content is too short
            
        finally:
            tmp_path.unlink()
    
    def test_extract_text_blocks_auto_detection(self):
        """Test extract_text_blocks with automatic type detection."""
        # Test PDF auto-detection
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            with patch('fitz.open') as mock_fitz_open:
                mock_doc = Mock()
                mock_page = Mock()
                mock_page.get_text.return_value = "Auto-detected PDF content"
                mock_doc.load_page.return_value = mock_page
                mock_doc.__len__.return_value = 1
                mock_fitz_open.return_value = mock_doc
                
                blocks = extract_text_blocks(tmp_path, source_type="auto")
                assert isinstance(blocks, list)
                
        finally:
            tmp_path.unlink()
    
    def test_extract_latex_blocks(self):
        """Test extract_latex_blocks function."""
        latex_content = r"""
        Text with math: $x = y + z$
        
        And display math:
        $$\int_0^1 x^2 dx = \frac{1}{3}$$
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix=".tex", delete=False, encoding='utf-8') as tmp:
            tmp.write(latex_content)
            tmp_path = Path(tmp.name)
        
        try:
            formulas = extract_latex_blocks(tmp_path, source_type="latex")
            
            assert isinstance(formulas, list)
            assert len(formulas) > 0
            
        finally:
            tmp_path.unlink()
    
    def test_unsupported_source_type(self):
        """Test with unsupported source type."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            with pytest.raises(ValueError, match="Unsupported source type"):
                extract_text_blocks(tmp_path, source_type="unsupported")
                
        finally:
            tmp_path.unlink()


class TestErrorHandling:
    """Test suite for error handling scenarios."""
    
    def test_pdf_parser_backend_error(self):
        """Test PDF parser with backend errors."""
        parser = PDFParser(backend="pymupdf")
        
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            with patch('fitz.open', side_effect=Exception("Mock PDF error")):
                with pytest.raises(Exception, match="Mock PDF error"):
                    parser.extract_text(tmp_path)
                    
        finally:
            tmp_path.unlink()
    
    def test_latex_parser_file_error(self):
        """Test LaTeX parser with file reading errors."""
        parser = LaTeXParser()
        
        with tempfile.NamedTemporaryFile(suffix=".tex", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            # Remove read permissions to simulate error
            tmp_path.chmod(0o000)
            
            with pytest.raises(PermissionError):
                parser.extract_text(tmp_path)
                
        finally:
            # Restore permissions before unlinking
            tmp_path.chmod(0o644)
            tmp_path.unlink()
    
    def test_max_pages_limit(self):
        """Test PDF parser with page limits."""
        parser = PDFParser(backend="pymupdf")
        
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            with patch('fitz.open') as mock_fitz_open:
                mock_doc = Mock()
                mock_page = Mock()
                mock_page.get_text.return_value = "Page content"
                mock_doc.load_page.return_value = mock_page
                mock_doc.__len__.return_value = 10  # 10 pages total
                mock_fitz_open.return_value = mock_doc
                
                # Limit to 3 pages
                result = parser.extract_text(tmp_path, max_pages=3)
                
                assert result["metadata"]["pages_processed"] <= 3
                
        finally:
            tmp_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__]) 