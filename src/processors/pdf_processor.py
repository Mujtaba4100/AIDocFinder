"""PDF processing utilities using PyMuPDF (fitz)."""

import fitz  # PyMuPDF
from typing import Dict, Any, List
import json
from pathlib import Path
from datetime import datetime


class PDFProcessor:
    def __init__(self):
        pass
    
    def extract_text(self, pdf_path: str) -> str:
        """Extract text from PDF"""
        text = ""
        try:
            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc):
                text += page.get_text()
            doc.close()
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
        return text
    
    def extract_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """Extract metadata from PDF"""
        metadata = {
            "title": "",
            "author": "",
            "subject": "",
            "keywords": "",
            "creator": "",
            "producer": "",
            "creation_date": "",
            "modification_date": "",
            "page_count": 0
        }
        
        try:
            doc = fitz.open(pdf_path)
            pdf_metadata = doc.metadata
            
            metadata.update({
                "title": pdf_metadata.get("title", ""),
                "author": pdf_metadata.get("author", ""),
                "subject": pdf_metadata.get("subject", ""),
                "keywords": pdf_metadata.get("keywords", ""),
                "creator": pdf_metadata.get("creator", ""),
                "producer": pdf_metadata.get("producer", ""),
                "creation_date": pdf_metadata.get("creationDate", ""),
                "modification_date": pdf_metadata.get("modDate", ""),
                "page_count": len(doc)
            })
            
            doc.close()
        except Exception as e:
            print(f"Error extracting metadata from {pdf_path}: {e}")
        
        return metadata
    
    def extract_tables(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract tables from PDF (simplified version)"""
        tables = []
        try:
            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc):
                # Simple table extraction - in production, use camelot or tabula
                text = page.get_text()
                # Look for table-like structures (lines, grids)
                # This is simplified - you'd want a proper table extraction library
                pass
            doc.close()
        except Exception as e:
            print(f"Error extracting tables from {pdf_path}: {e}")
        
        return tables
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            chunks.append({
                "text": chunk,
                "start_char": start,
                "end_char": min(end, len(text)),
                "chunk_id": len(chunks)
            })
            
            start = end - overlap
        
        return chunks
    
    def process(self, pdf_path: str) -> Dict[str, Any]:
        """Process PDF file and return structured data"""
        print(f"Processing PDF: {pdf_path}")
        
        # Extract text
        text = self.extract_text(pdf_path)
        
        # Extract metadata
        metadata = self.extract_metadata(pdf_path)
        
        # Create chunks
        chunks = self.chunk_text(text)
        
        # Basic text analysis
        word_count = len(text.split())
        char_count = len(text)
        
        return {
            "file_type": "pdf",
            "text": text,
            "chunks": chunks,
            "metadata": metadata,
            "summary": {
                "word_count": word_count,
                "char_count": char_count,
                "chunk_count": len(chunks),
                "has_tables": len(self.extract_tables(pdf_path)) > 0
            },
            "processing_time": datetime.now().isoformat()
        }

# Test the processor
if __name__ == "__main__":
    processor = PDFProcessor()
    
    # Create a test PDF if none exists
    test_pdf = Path("test.pdf")
    if not test_pdf.exists():
        import os
        os.system("echo 'This is a test PDF for DocuFind AI. It contains information about hostel rules and regulations. Students must follow all rules. Curfew is at 10 PM. No loud music after 9 PM.' > test.txt")
        os.system("pandoc test.txt -o test.pdf 2>/dev/null || echo 'Install pandoc to create test PDF'")
    
    if test_pdf.exists():
        result = processor.process(str(test_pdf))
        print(f"Processed PDF. Chunks: {len(result['chunks'])}")
        print(f"Text preview: {result['text'][:200]}...")
