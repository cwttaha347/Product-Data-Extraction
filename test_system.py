#!/usr/bin/env python

import os
import sys
import json
import argparse
import logging
import importlib.util
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def is_module_available(module_name):
    """Check if a module can be imported"""
    return importlib.util.find_spec(module_name) is not None


def check_directories():
    """Verify all needed directories exist"""
    required_dirs = [
        "src",
        "src/pdf_processing",
        "src/nlp",
        "src/database",
        "src/api",
        "src/utils",
        "data",
        "data/raw",
        "data/processed"
    ]

    for directory in required_dirs:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created missing directory: {directory}")

    return True


def check_database(skip=False):
    """Test database connection and schema"""
    if skip:
        logger.info("Skipping database test as requested")
        return True

    if not is_module_available("sqlalchemy") or not is_module_available("psycopg2"):
        logger.warning(
            "Required modules for database connection are not available")
        logger.info("Install with: pip install sqlalchemy psycopg2-binary")
        return False

    try:
        from sqlalchemy import create_engine
        engine = create_engine(
            "postgresql://postgres:postgres@localhost:5432/product_data")

        with engine.connect() as conn:
            result = conn.execute("SELECT 1")
            row = result.fetchone()
            if row and row[0] == 1:
                logger.info("Database connection successful")
                return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        logger.info("Make sure PostgreSQL is running and the database exists.")
        return False


def check_pdf_processing(skip=False):
    """Test PDF processing functionality"""
    if skip:
        logger.info("Skipping PDF processing test as requested")
        return True

    logger.info("Testing PDF processing...")

    if not is_module_available("pdfplumber") or not is_module_available("pytesseract"):
        logger.warning("Required modules for PDF processing are not available")
        logger.info(
            "Install with: pip install pdfplumber pytesseract pdf2image")
        return False

    # Check if there are any PDFs to process in the data/raw directory
    pdf_files = list(Path("data/raw").glob("*.pdf"))
    if not pdf_files:
        logger.warning(
            "No PDF files found for testing. PDF processing test skipped.")
        logger.info("Please add a PDF file to the data/raw directory.")
        return False

    try:
        # Try importing the PDF processing module
        from src.pdf_processing.batch import BatchProcessor
        logger.info("PDF processing module imported successfully")
        return True
    except ImportError as e:
        logger.error(f"PDF processing module import failed: {e}")
        return False
    except Exception as e:
        logger.error(f"PDF processing test failed: {e}")
        return False


def check_nlp_pipeline(skip=False):
    """Test NLP pipeline functionality"""
    if skip:
        logger.info("Skipping NLP pipeline test as requested")
        return True

    # First check if PDF processing passed
    pdf_files = list(Path("data/raw").glob("*.pdf"))
    if not pdf_files:
        logger.info("Skipping NLP pipeline test - no PDF data available")
        return False

    if not is_module_available("spacy"):
        logger.warning("spaCy module is not available")
        logger.info("Install with: pip install spacy")
        logger.info(
            "And download model: python -m spacy download en_core_web_sm")
        return False

    try:
        # Try importing the NLP module
        from src.nlp.entity_extractor import EntityExtractor
        logger.info("NLP module imported successfully")
        return True
    except ImportError as e:
        logger.error(f"NLP module import failed: {e}")
        return False
    except Exception as e:
        logger.error(f"NLP pipeline test failed: {e}")
        return False


def check_api(skip=False):
    """Test API functionality"""
    if skip:
        logger.info("Skipping API test as requested")
        return True

    logger.info("Testing API connection...")

    if not is_module_available("requests"):
        logger.warning("requests module is not available")
        logger.info("Install with: pip install requests")
        return False

    import requests
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            logger.info("API connection successful")
            return True
        else:
            logger.error(
                f"API returned unexpected status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        logger.error("API connection failed - server not running")
        logger.info("Start the API with: python -m src.api.main")
        return False
    except Exception as e:
        logger.error(f"API test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test the system components")
    parser.add_argument("--skip-db", action="store_true",
                        help="Skip database tests")
    parser.add_argument("--skip-pdf", action="store_true",
                        help="Skip PDF processing tests")
    parser.add_argument("--skip-nlp", action="store_true",
                        help="Skip NLP pipeline tests")
    parser.add_argument("--skip-api", action="store_true",
                        help="Skip API tests")

    args = parser.parse_args()

    results = {}

    # Always check directories
    results["DIRECTORIES"] = check_directories()

    # Run tests as requested
    results["DATABASE"] = check_database(skip=args.skip_db)
    results["PDF_PROCESSING"] = check_pdf_processing(skip=args.skip_pdf)
    results["NLP"] = check_nlp_pipeline(skip=args.skip_nlp)
    results["API"] = check_api(skip=args.skip_api)

    # Print summary
    logger.info("=== TEST RESULTS ===")
    for test, result in results.items():
        logger.info(f"{test}: {'PASS' if result else 'FAIL'}")

    passed = sum(1 for result in results.values() if result)
    total = len(results)
    logger.info(f"SUMMARY: {passed}/{total} tests passed")

    if passed < total:
        logger.warning("SOME TESTS FAILED!")
        return 1
    else:
        logger.info("ALL TESTS PASSED!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
