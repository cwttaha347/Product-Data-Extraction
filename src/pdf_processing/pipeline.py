#!/usr/bin/env python

import os
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pdfplumber
import pytesseract
from pdf2image import convert_from_path, convert_from_bytes
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PDFProcessor:
    def __init__(self, ocr_lang: str = 'eng', dpi: int = 300):
        self.ocr_lang = ocr_lang
        self.dpi = dpi

        try:
            pytesseract.get_tesseract_version()
        except Exception as e:
            logger.warning(f"Tesseract not properly configured: {e}")
            logger.warning("OCR functionality will be limited.")

    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        logger.info(f"Processing PDF: {pdf_path}")

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        pdf_info = {
            'filename': os.path.basename(pdf_path),
            'path': pdf_path,
            'pages': [],
            'metadata': {},
            'requires_ocr': False
        }

        try:
            with pdfplumber.open(pdf_path) as pdf:
                pdf_info['metadata'] = pdf.metadata
                pdf_info['page_count'] = len(pdf.pages)

                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text() or ""

                    if len(page_text.strip()) < 50:
                        ocr_text = self._ocr_page(pdf_path, i)

                        if len(ocr_text.strip()) > len(page_text.strip()):
                            page_text = ocr_text
                            pdf_info['requires_ocr'] = True

                    pdf_info['pages'].append({
                        'number': i + 1,
                        'text': page_text,
                        'tables': self._extract_tables(page)
                    })

        except Exception as e:
            logger.error(f"Error extracting text with pdfplumber: {e}")
            pdf_info = self._full_ocr_fallback(pdf_path)

        return pdf_info

    def _extract_tables(self, page) -> List[Dict]:
        tables = []
        try:
            for table in page.extract_tables():
                if table:
                    headers = [
                        str(h).strip() if h else f"col_{i}" for i, h in enumerate(table[0])]
                    data = []

                    for row in table[1:]:
                        row_dict = {}
                        for i, cell in enumerate(row):
                            header = headers[i] if i < len(
                                headers) else f"col_{i}"
                            row_dict[header] = str(
                                cell).strip() if cell else ""
                        data.append(row_dict)

                    tables.append({
                        'headers': headers,
                        'data': data
                    })
        except Exception as e:
            logger.warning(f"Error extracting tables: {e}")

        return tables

    def _ocr_page(self, pdf_path: str, page_num: int) -> str:
        try:
            images = convert_from_path(
                pdf_path, dpi=self.dpi, first_page=page_num+1, last_page=page_num+1)

            if not images:
                logger.warning(f"Could not convert page {page_num+1} to image")
                return ""

            text = pytesseract.image_to_string(images[0], lang=self.ocr_lang)
            return text

        except Exception as e:
            logger.error(f"Error during OCR: {e}")
            return ""

    def _full_ocr_fallback(self, pdf_path: str) -> Dict[str, Any]:
        logger.info(f"Using full OCR fallback for: {pdf_path}")

        pdf_info = {
            'filename': os.path.basename(pdf_path),
            'path': pdf_path,
            'pages': [],
            'metadata': {},
            'requires_ocr': True
        }

        try:
            images = convert_from_path(pdf_path, dpi=self.dpi)
            pdf_info['page_count'] = len(images)

            for i, image in enumerate(images):
                text = pytesseract.image_to_string(image, lang=self.ocr_lang)

                pdf_info['pages'].append({
                    'number': i + 1,
                    'text': text,
                    'tables': []
                })

        except Exception as e:
            logger.error(f"Error during full OCR: {e}")
            pdf_info['error'] = str(e)

        return pdf_info


def process_pdf(pdf_path: str) -> Dict[str, Any]:
    processor = PDFProcessor()
    return processor.process_pdf(pdf_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <pdf_file>")
        sys.exit(1)

    result = process_pdf(sys.argv[1])
    print(f"Processed {result['filename']}")
    print(f"Pages: {result['page_count']}")
    print(f"Requires OCR: {result['requires_ocr']}")

    for page in result['pages']:
        preview = page['text'][:100] + \
            "..." if len(page['text']) > 100 else page['text']
        print(f"Page {page['number']}: {preview}")
