#!/usr/bin/env python
"""
PDF Processing Pipeline Runner

This script runs the complete pipeline for processing PDF documents:
1. Extract text from PDF files
2. Identify entities using NLP
3. Process and normalize extracted data
4. Store structured data in PostgreSQL
"""

import os
import argparse
import logging
import json
import time
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.pdf_processing.pipeline import process_pdf
from src.pdf_processing.batch import process_pdf_batch
from src.nlp.entity_extractor import process_document as extract_entities
from src.utils.data_processor import process_extracted_data
from src.database.db_operations import store_processed_data

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def process_single_pdf(
    pdf_path: str,
    output_dir: str,
    model_path: Optional[str] = None,
    save_intermediates: bool = False,
    store_in_db: bool = True
) -> Dict[str, Any]:
    """
    Process a single PDF file through the entire pipeline.

    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save results
        model_path: Path to custom NER model (optional)
        save_intermediates: Whether to save intermediate results (default: False)
        store_in_db: Whether to store results in database (default: True)

    Returns:
        Processing results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get base filename without extension
    filename = os.path.basename(pdf_path)
    base_name = os.path.splitext(filename)[0]

    logger.info(f"Processing {filename}")

    # Step 1: Extract text from PDF
    start_time = time.time()
    pdf_info = process_pdf(pdf_path)
    pdf_time = time.time() - start_time
    logger.info(f"PDF extraction completed in {pdf_time:.2f} seconds")

    # Save intermediate results if requested
    if save_intermediates:
        pdf_text_path = os.path.join(output_dir, f"{base_name}_text.json")
        with open(pdf_text_path, 'w', encoding='utf-8') as f:
            json.dump(pdf_info, f, indent=2, ensure_ascii=False)
        logger.info(f"Extracted text saved to {pdf_text_path}")

    # Step 2: Extract entities
    start_time = time.time()
    entities_data = extract_entities(pdf_info, model_path)
    entity_time = time.time() - start_time
    logger.info(f"Entity extraction completed in {entity_time:.2f} seconds")

    # Save intermediate results if requested
    if save_intermediates:
        entities_path = os.path.join(output_dir, f"{base_name}_entities.json")
        with open(entities_path, 'w', encoding='utf-8') as f:
            json.dump(entities_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Extracted entities saved to {entities_path}")

    # Step 3: Process extracted data
    start_time = time.time()
    processed_data = process_extracted_data(entities_data)
    processing_time = time.time() - start_time
    logger.info(f"Data processing completed in {processing_time:.2f} seconds")

    # Save processed results
    processed_path = os.path.join(output_dir, f"{base_name}_processed.json")
    with open(processed_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Processed data saved to {processed_path}")

    # Step 4: Store in database if requested
    db_result = {}
    if store_in_db:
        start_time = time.time()
        db_result = store_processed_data(processed_data)
        db_time = time.time() - start_time

        # Save database result
        db_path = os.path.join(output_dir, f"{base_name}_db_result.json")
        with open(db_path, 'w', encoding='utf-8') as f:
            json.dump(db_result, f, indent=2)

        if db_result.get('success', False):
            logger.info(f"Data stored in database in {db_time:.2f} seconds")
            logger.info(f"Document ID: {db_result.get('document_id')}")
            logger.info(f"Product IDs: {db_result.get('product_ids', [])}")
        else:
            logger.error(
                f"Failed to store data in database: {db_result.get('errors', [])}")

    # Return combined results
    return {
        'pdf_info': {
            'filename': filename,
            'page_count': pdf_info.get('page_count', 0),
            'requires_ocr': pdf_info.get('requires_ocr', False),
            'processing_time': pdf_time
        },
        'entity_extraction': {
            'entity_types': list(entities_data.get('entities', {}).keys()),
            'entity_count': sum(len(entities) for entities in entities_data.get('entities', {}).values()),
            'processing_time': entity_time
        },
        'data_processing': {
            'product_count': len(processed_data.get('products', [])),
            'processing_time': processing_time
        },
        'database_storage': db_result if store_in_db else {'skipped': True}
    }


def process_directory(
    input_dir: str,
    output_dir: str,
    model_path: Optional[str] = None,
    file_pattern: str = "*.pdf",
    save_intermediates: bool = False,
    store_in_db: bool = True,
    max_workers: int = 4
) -> Dict[str, Any]:
    """
    Process all PDF files in a directory.

    Args:
        input_dir: Directory containing PDF files
        output_dir: Directory to save results
        model_path: Path to custom NER model (optional)
        file_pattern: File pattern to match PDF files (default: "*.pdf")
        save_intermediates: Whether to save intermediate results (default: False)
        store_in_db: Whether to store results in database (default: True)
        max_workers: Maximum number of parallel workers (default: 4)

    Returns:
        Processing results summary
    """
    logger.info(f"Processing directory: {input_dir}")
    logger.info(f"File pattern: {file_pattern}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Find PDF files
    pdf_files = list(glob.glob(os.path.join(input_dir, file_pattern)))

    if not pdf_files:
        logger.warning(
            f"No PDF files found in {input_dir} matching pattern {file_pattern}")
        return {'error': 'No PDF files found', 'file_count': 0}

    logger.info(f"Found {len(pdf_files)} PDF files")

    # Process batch in parallel
    batch_results = process_pdf_batch(
        input_dir=input_dir,
        output_dir=output_dir,
        max_workers=max_workers,
        file_pattern=file_pattern.lstrip("*"),
        save_results=True
    )

    # Process and store the extracted data
    successful = 0
    failed = 0
    products_found = 0

    for pdf_file in pdf_files:
        try:
            # Get base filename without extension
            filename = os.path.basename(pdf_file)
            base_name = os.path.splitext(filename)[0]

            # Path to extracted JSON file
            json_path = os.path.join(output_dir, f"{base_name}.json")

            if not os.path.exists(json_path):
                logger.warning(f"Extraction result not found for {filename}")
                failed += 1
                continue

            # Load extracted data
            with open(json_path, 'r', encoding='utf-8') as f:
                pdf_info = json.load(f)

            # Extract entities
            entities_data = extract_entities(pdf_info, model_path)

            # Save entities if requested
            if save_intermediates:
                entities_path = os.path.join(
                    output_dir, f"{base_name}_entities.json")
                with open(entities_path, 'w', encoding='utf-8') as f:
                    json.dump(entities_data, f, indent=2, ensure_ascii=False)

            # Process extracted data
            processed_data = process_extracted_data(entities_data)

            # Save processed data
            processed_path = os.path.join(
                output_dir, f"{base_name}_processed.json")
            with open(processed_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)

            # Store in database if requested
            if store_in_db:
                db_result = store_processed_data(processed_data)

                # Save database result
                db_path = os.path.join(
                    output_dir, f"{base_name}_db_result.json")
                with open(db_path, 'w', encoding='utf-8') as f:
                    json.dump(db_result, f, indent=2)

                if db_result.get('success', False):
                    logger.info(f"Data for {filename} stored in database")
                    products_found += len(processed_data.get('products', []))
                    successful += 1
                else:
                    logger.error(
                        f"Failed to store data for {filename} in database: {db_result.get('errors', [])}")
                    failed += 1
            else:
                products_found += len(processed_data.get('products', []))
                successful += 1

        except Exception as e:
            logger.error(f"Error processing {pdf_file}: {e}")
            failed += 1

    # Return summary
    return {
        'file_count': len(pdf_files),
        'successful': successful,
        'failed': failed,
        'products_found': products_found
    }


def main():
    """Run the pipeline based on command-line arguments."""
    parser = argparse.ArgumentParser(description="PDF Processing Pipeline")

    # General arguments
    parser.add_argument("--input", required=True,
                        help="Input PDF file or directory")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for results")
    parser.add_argument("--model", help="Path to custom NER model (optional)")

    # Processing options
    parser.add_argument("--save-intermediates",
                        action="store_true", help="Save intermediate results")
    parser.add_argument("--no-db", action="store_true",
                        help="Skip database storage")

    # Batch processing options
    parser.add_argument("--pattern", default="*.pdf",
                        help="File pattern for batch processing (default: *.pdf)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Maximum number of parallel workers (default: 4)")

    args = parser.parse_args()

    # Determine if input is a file or directory
    if os.path.isfile(args.input):
        # Process single file
        logger.info(f"Processing single file: {args.input}")
        result = process_single_pdf(
            pdf_path=args.input,
            output_dir=args.output_dir,
            model_path=args.model,
            save_intermediates=args.save_intermediates,
            store_in_db=not args.no_db
        )

        # Print summary
        logger.info("Processing completed")
        logger.info(f"PDF: {result['pdf_info']['filename']}")
        logger.info(f"Pages: {result['pdf_info']['page_count']}")
        logger.info(f"OCR required: {result['pdf_info']['requires_ocr']}")
        logger.info(
            f"Entities found: {result['entity_extraction']['entity_count']}")
        logger.info(
            f"Products found: {result['data_processing']['product_count']}")

        if not args.no_db and result['database_storage'].get('success', False):
            logger.info(
                f"Document ID: {result['database_storage'].get('document_id')}")
            logger.info(
                f"Product IDs: {result['database_storage'].get('product_ids', [])}")

    elif os.path.isdir(args.input):
        # Process directory
        logger.info(f"Processing directory: {args.input}")
        result = process_directory(
            input_dir=args.input,
            output_dir=args.output_dir,
            model_path=args.model,
            file_pattern=args.pattern,
            save_intermediates=args.save_intermediates,
            store_in_db=not args.no_db,
            max_workers=args.workers
        )

        # Print summary
        logger.info("Batch processing completed")
        logger.info(f"Files processed: {result['file_count']}")
        logger.info(f"Successful: {result['successful']}")
        logger.info(f"Failed: {result['failed']}")
        logger.info(f"Products found: {result['products_found']}")

    else:
        logger.error(f"Input not found: {args.input}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
