#!/usr/bin/env python
"""
Batch PDF Processing

Module for processing multiple PDF files in batch mode,
with support for parallel processing and error handling.
"""

import os
import glob
import json
import logging
import concurrent.futures
from typing import List, Dict, Any, Optional
from pathlib import Path

from src.pdf_processing.pipeline import process_pdf
from src.nlp.entity_extractor import process_document as extract_entities
from src.utils.data_processor import process_extracted_data
from src.database.db_operations import store_processed_data

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BatchProcessor:
    def __init__(self, input_dir: str, output_dir: str, model_path: Optional[str] = None,
                 max_workers: int = 4, file_pattern: str = "*.pdf", save_to_db: bool = True):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.model_path = model_path
        self.max_workers = max_workers
        self.file_pattern = file_pattern
        self.save_to_db = save_to_db

    def process_single_pdf(self, pdf_path: str, save_intermediates: bool = True) -> Dict[str, Any]:
        logger.info(f"Processing PDF: {pdf_path}")

        file_basename = os.path.basename(pdf_path)
        result_path = os.path.join(self.output_dir, f"{file_basename}.json")

        if os.path.exists(result_path):
            logger.info(f"Result already exists for {file_basename}, skipping")
            with open(result_path, 'r', encoding='utf-8') as f:
                return json.load(f)

        try:
            pdf_data = process_pdf(pdf_path)

            if save_intermediates:
                text_output_path = os.path.join(
                    self.output_dir, f"{file_basename}_text.json")
                with open(text_output_path, 'w', encoding='utf-8') as f:
                    json.dump(pdf_data, f, indent=2, ensure_ascii=False)

            entities_data = extract_entities(pdf_data, self.model_path)

            if save_intermediates:
                entities_output_path = os.path.join(
                    self.output_dir, f"{file_basename}_entities.json")
                with open(entities_output_path, 'w', encoding='utf-8') as f:
                    json.dump(entities_data, f, indent=2, ensure_ascii=False)

            processed_data = process_extracted_data(entities_data)
            processed_data["source_file"] = pdf_path

            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)

            return processed_data

        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            error_path = os.path.join(
                self.output_dir, f"{file_basename}_error.json")
            with open(error_path, 'w', encoding='utf-8') as f:
                json.dump({"error": str(e), "file": pdf_path}, f, indent=2)

            return {"error": str(e), "file": pdf_path}

    def process_batch(self) -> List[Dict[str, Any]]:
        os.makedirs(self.output_dir, exist_ok=True)

        pdf_files = glob.glob(os.path.join(
            self.input_dir, self.file_pattern), recursive=True)

        if not pdf_files:
            logger.warning(
                f"No PDF files found in {self.input_dir} matching pattern {self.file_pattern}")
            return []

        logger.info(f"Found {len(pdf_files)} PDF files to process")

        results = []
        failed = []

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {executor.submit(
                self.process_single_pdf, pdf): pdf for pdf in pdf_files}

            for future in concurrent.futures.as_completed(future_to_file):
                pdf_file = future_to_file[future]
                try:
                    data = future.result()

                    if "error" in data:
                        failed.append(pdf_file)
                        continue

                    results.append(data)

                    if self.save_to_db:
                        store_processed_data(data)

                except Exception as e:
                    logger.error(f"Error processing {pdf_file}: {e}")
                    failed.append(pdf_file)

        success_count = len(results)
        failed_count = len(failed)
        total = len(pdf_files)

        logger.info(
            f"Batch processing complete. {success_count}/{total} files processed successfully.")

        if failed:
            logger.warning(f"{failed_count} files failed: {failed}")

        summary_path = os.path.join(self.output_dir, "batch_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            summary = {
                "total_files": total,
                "successful": success_count,
                "failed": failed_count,
                "failed_files": failed
            }
            json.dump(summary, f, indent=2)

        return results


def process_single_pdf(pdf_path: str, output_dir: str, model_path: Optional[str] = None, save_intermediates: bool = True) -> Dict[str, Any]:
    processor = BatchProcessor(
        input_dir="", output_dir=output_dir, model_path=model_path)
    return processor.process_single_pdf(pdf_path, save_intermediates)


def process_pdf_batch(input_dir: str, output_dir: str, model_path: Optional[str] = None,
                      max_workers: int = 4, file_pattern: str = "*.pdf", save_to_db: bool = True) -> List[Dict[str, Any]]:
    processor = BatchProcessor(
        input_dir=input_dir,
        output_dir=output_dir,
        model_path=model_path,
        max_workers=max_workers,
        file_pattern=file_pattern,
        save_to_db=save_to_db
    )
    return processor.process_batch()


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch process PDF files")
    parser.add_argument("input_dir", help="Directory containing PDF files")
    parser.add_argument(
        "--output-dir", help="Directory to save results", default=None)
    parser.add_argument(
        "--max-workers", help="Maximum number of parallel workers", type=int, default=4)
    parser.add_argument(
        "--pattern", help="File pattern to match PDFs", default="*.pdf")

    args = parser.parse_args()

    process_pdf_batch(
        args.input_dir,
        args.output_dir,
        max_workers=args.max_workers,
        file_pattern=args.pattern
    )
