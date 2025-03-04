#!/usr/bin/env python
"""
NER Model Training

This module provides functionality for training custom spaCy NER models
using annotated data from Prodigy, with specific focus on product data extraction.
"""

import os
import logging
import json
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple

import spacy
from spacy.tokens import DocBin
from spacy.cli import train as spacy_train
import pandas as pd
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def prepare_training_data(
    input_dir: str,
    output_dir: str,
    file_pattern: str = "*.json",
    sample_limit: Optional[int] = None
) -> Tuple[int, List[str]]:
    """
    Prepare training data from processed PDF files for annotation with Prodigy.

    Args:
        input_dir: Directory containing processed PDF files (JSON format)
        output_dir: Directory to save training data
        file_pattern: File pattern to match JSON files (default: "*.json")
        sample_limit: Maximum number of pages to sample (default: None = all)

    Returns:
        Tuple of (number of examples prepared, list of entity types found)
    """
    logger.info(f"Preparing training data from: {input_dir}")

    # Ensure directories exist
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all JSON files
    json_files = list(input_path.glob(file_pattern))
    logger.info(f"Found {len(json_files)} JSON files")

    if not json_files:
        logger.warning(
            f"No JSON files found in {input_dir} matching pattern: {file_pattern}")
        return 0, []

    # Collect all text samples and entities
    all_samples = []
    all_entity_types = set()

    # Process each file
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            pdf_data = json.load(f)

        # Process each page
        for page in pdf_data.get('pages', []):
            page_text = page.get('text', '')

            # Skip empty pages
            if not page_text:
                continue

            # Add the page text as a sample
            all_samples.append({
                'text': page_text,
                'source': os.path.basename(json_file),
                'page': page.get('number', 0)
            })

            # If the file has extracted entities, add them to the set of entity types
            if 'entities' in pdf_data:
                for entity_type in pdf_data['entities'].keys():
                    all_entity_types.add(entity_type)

    # If sample limit is provided, take a random sample
    if sample_limit and len(all_samples) > sample_limit:
        import random
        all_samples = random.sample(all_samples, sample_limit)

    # Write samples to output file
    output_file = output_path / "annotation_data.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    logger.info(f"Prepared {len(all_samples)} text samples for annotation")
    logger.info(f"Entity types found: {', '.join(sorted(all_entity_types))}")

    # Write entity types to a configuration file
    entity_types = sorted(list(all_entity_types))
    with open(output_path / "entity_types.json", 'w', encoding='utf-8') as f:
        json.dump(entity_types, f, indent=2)

    return len(all_samples), entity_types


def run_prodigy_annotation(
    dataset_name: str,
    input_file: str,
    label: Union[str, List[str]],
    model: Optional[str] = None
) -> bool:
    """
    Run Prodigy for annotation (this will start the Prodigy web server).
    Note: This function requires Prodigy to be installed and licensed.

    Args:
        dataset_name: Name of the Prodigy dataset to create
        input_file: Path to the input JSONL file
        label: Entity types to annotate
        model: Path to a spaCy model for pre-annotation (optional)

    Returns:
        True if Prodigy command succeeded, False otherwise
    """
    if isinstance(label, list):
        label_str = ",".join(label)
    else:
        label_str = label

    # Prepare Prodigy command
    if model:
        cmd = ["prodigy", "ner.teach", dataset_name,
               model, input_file, "--label", label_str]
    else:
        cmd = ["prodigy", "ner.manual", dataset_name,
               input_file, "--label", label_str]

    logger.info(f"Running Prodigy annotation: {' '.join(cmd)}")

    try:
        # Run Prodigy as a subprocess
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info("Prodigy server started. Open your browser to annotate.")
        logger.info("Press Ctrl+C when done annotating.")

        # Wait for user to press Ctrl+C
        process.wait()

        return True
    except Exception as e:
        logger.error(f"Error running Prodigy: {e}")
        return False


def export_prodigy_data(
    dataset_name: str,
    output_dir: str
) -> Tuple[Optional[str], Optional[str]]:
    """
    Export annotated data from Prodigy for training a spaCy model.

    Args:
        dataset_name: Name of the Prodigy dataset
        output_dir: Directory to save exported data

    Returns:
        Tuple of (train_file_path, dev_file_path) or (None, None) if export failed
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_path = output_path / f"{dataset_name}_train.spacy"
    dev_path = output_path / f"{dataset_name}_dev.spacy"

    try:
        # Export data to spaCy format
        subprocess.run([
            "prodigy", "data-to-spacy", dataset_name,
            "--train-path", str(train_path),
            "--dev-path", str(dev_path),
            "--ner-label", "all"
        ], check=True, capture_output=True)

        logger.info(f"Exported Prodigy data to {train_path} and {dev_path}")
        return str(train_path), str(dev_path)

    except subprocess.CalledProcessError as e:
        logger.error(f"Error exporting Prodigy data: {e}")
        logger.error(f"stderr: {e.stderr.decode()}")
        return None, None


def train_ner_model(
    train_file: str,
    dev_file: str,
    output_dir: str,
    base_model: str = "en_core_web_sm",
    n_iter: int = 30,
    dropout: float = 0.2
) -> Optional[str]:
    """
    Train a custom spaCy NER model using exported Prodigy data.

    Args:
        train_file: Path to training data (spaCy binary format)
        dev_file: Path to development data (spaCy binary format)
        output_dir: Directory to save trained model
        base_model: Base spaCy model to start from (default: "en_core_web_sm")
        n_iter: Number of training iterations (default: 30)
        dropout: Dropout rate during training (default: 0.2)

    Returns:
        Path to the trained model if successful, None otherwise
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create temporary directory for config and training files
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create base config
        config = {
            "paths": {
                "train": train_file,
                "dev": dev_file
            },
            "system": {
                "gpu_allocator": "cuda" if spacy.prefer_gpu() else None
            },
            "nlp": {
                "lang": "en",
                "pipeline": ["tok2vec", "ner"],
                "batch_size": 1000
            },
            "components": {
                "ner": {
                    "factory": "ner",
                    "moves": None,
                    "update_with_oracle_cut_size": 100
                }
            },
            "training": {
                "dev_corpus": "corpora.dev",
                "train_corpus": "corpora.train",
                "seed": 42,
                "gpu_allocator": "cuda" if spacy.prefer_gpu() else None,
                "dropout": dropout,
                "accumulate_gradient": 1,
                "patience": 5,
                "max_epochs": n_iter,
                "max_steps": 0
            }
        }

        # Save config to file
        config_path = tmp_path / "config.cfg"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        try:
            # Train the model
            logger.info(f"Training spaCy NER model...")
            spacy_train.run_training(
                config_path=str(config_path),
                output_path=str(output_path),
                use_gpu=spacy.prefer_gpu(),
                overrides={
                    "paths.train": train_file,
                    "paths.dev": dev_file,
                    "training.max_epochs": n_iter
                }
            )

            # Find the best model
            model_path = output_path / "model-best"
            if model_path.exists():
                logger.info(f"Trained model saved to: {model_path}")
                return str(model_path)
            else:
                logger.error("Training completed, but model-best not found")
                return None

        except Exception as e:
            logger.error(f"Error training model: {e}")
            return None


def evaluate_model(
    model_path: str,
    test_file: Optional[str] = None,
    test_data: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Evaluate a trained NER model.

    Args:
        model_path: Path to the trained spaCy model
        test_file: Path to test data (spaCy binary format) or None
        test_data: List of annotated examples or None

    Returns:
        Dictionary with evaluation metrics
    """
    if not os.path.exists(model_path):
        raise ValueError(f"Model not found: {model_path}")

    # Load the model
    nlp = spacy.load(model_path)

    # Prepare test data
    examples = []

    if test_file:
        # Load from file
        doc_bin = DocBin().from_disk(test_file)
        docs = list(doc_bin.get_docs(nlp.vocab))
        examples = [{"text": doc.text, "entities": [(ent.start_char, ent.end_char, ent.label_)
                                                    for ent in doc.ents]}
                    for doc in docs]

    elif test_data:
        examples = test_data

    else:
        raise ValueError("Either test_file or test_data must be provided")

    # Evaluate model
    results = {
        "entity_types": {},
        "overall": {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        }
    }

    if not examples:
        logger.warning("No test examples available for evaluation")
        return results

    # Count entities by type in gold standard
    gold_counts = {}

    # Initialize with all entity types in the model
    for ent_type in nlp.pipe_labels['ner']:
        gold_counts[ent_type] = 0
        results["entity_types"][ent_type] = {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "gold_count": 0,
            "pred_count": 0,
            "correct": 0
        }

    # Count entities in gold standard
    for example in examples:
        for _, _, ent_type in example.get("entities", []):
            if ent_type in gold_counts:
                gold_counts[ent_type] += 1
            else:
                gold_counts[ent_type] = 1
                results["entity_types"][ent_type] = {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "gold_count": 1,
                    "pred_count": 0,
                    "correct": 0
                }

    for ent_type, count in gold_counts.items():
        if ent_type in results["entity_types"]:
            results["entity_types"][ent_type]["gold_count"] = count

    # Evaluate on each example
    total_correct = 0
    total_pred = 0
    total_gold = sum(gold_counts.values())

    for example in examples:
        text = example["text"]
        gold_entities = {(start, end, ent_type)
                         for start, end, ent_type in example.get("entities", [])}

        # Get predictions
        doc = nlp(text)
        pred_entities = {(ent.start_char, ent.end_char, ent.label_)
                         for ent in doc.ents}

        # Count correct predictions by type
        for ent_type in results["entity_types"]:
            # Count predictions for this type
            type_preds = sum(
                1 for _, _, label in pred_entities if label == ent_type)
            results["entity_types"][ent_type]["pred_count"] += type_preds

            # Count correct predictions for this type
            correct = sum(
                1 for ent in pred_entities if ent in gold_entities and ent[2] == ent_type)
            results["entity_types"][ent_type]["correct"] += correct
            total_correct += correct

        total_pred += len(pred_entities)

    # Calculate precision, recall, F1 for each entity type
    for ent_type, metrics in results["entity_types"].items():
        correct = metrics["correct"]
        pred_count = metrics["pred_count"]
        gold_count = metrics["gold_count"]

        # Calculate precision (correct / predicted)
        precision = correct / pred_count if pred_count > 0 else 0.0
        metrics["precision"] = precision

        # Calculate recall (correct / gold)
        recall = correct / gold_count if gold_count > 0 else 0.0
        metrics["recall"] = recall

        # Calculate F1 score
        f1 = 2 * (precision * recall) / (precision +
                                         recall) if (precision + recall) > 0 else 0.0
        metrics["f1"] = f1

    # Calculate overall metrics
    overall_precision = total_correct / total_pred if total_pred > 0 else 0.0
    overall_recall = total_correct / total_gold if total_gold > 0 else 0.0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision +
                                                             overall_recall) if (overall_precision + overall_recall) > 0 else 0.0

    results["overall"] = {
        "precision": overall_precision,
        "recall": overall_recall,
        "f1": overall_f1,
        "correct": total_correct,
        "predicted": total_pred,
        "gold": total_gold
    }

    # Print evaluation results
    logger.info("Model Evaluation Results:")
    logger.info(
        f"Overall - Precision: {overall_precision:.4f}, Recall: {overall_recall:.4f}, F1: {overall_f1:.4f}")

    for ent_type, metrics in sorted(results["entity_types"].items(), key=lambda x: x[1]["gold_count"], reverse=True):
        if metrics["gold_count"] > 0:
            logger.info(
                f"{ent_type} - Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f} (Gold: {metrics['gold_count']}, Pred: {metrics['pred_count']})")

    return results


def main(
    input_dir: str,
    output_dir: str,
    dataset_name: str,
    train_model: bool = True,
    base_model: str = "en_core_web_sm",
    entity_types: Optional[List[str]] = None
):
    """
    Full pipeline for preparing data, annotation, and training.

    Args:
        input_dir: Directory with processed PDF files
        output_dir: Directory to save trained model and data
        dataset_name: Name for the Prodigy dataset
        train_model: Whether to train a model after annotation
        base_model: Base model to use for training
        entity_types: List of entity types to annotate (or None to detect from data)
    """
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    annotation_dir = os.path.join(output_dir, "annotation")
    os.makedirs(annotation_dir, exist_ok=True)

    # Step 1: Prepare training data
    logger.info("Step 1: Preparing training data")
    n_samples, detected_entity_types = prepare_training_data(
        input_dir=input_dir,
        output_dir=annotation_dir
    )

    if n_samples == 0:
        logger.error("No training data prepared. Exiting.")
        return

    # Determine entity types to annotate
    if entity_types is None:
        entity_types = detected_entity_types

    # Step 2: Run annotation with Prodigy
    logger.info("Step 2: Running Prodigy for annotation")
    logger.info(f"Entity types to annotate: {', '.join(entity_types)}")

    input_file = os.path.join(annotation_dir, "annotation_data.jsonl")
    annotation_success = run_prodigy_annotation(
        dataset_name=dataset_name,
        input_file=input_file,
        label=entity_types,
        model=base_model
    )

    if not annotation_success:
        logger.error("Annotation process failed. Exiting.")
        return

    if train_model:
        # Step 3: Export annotated data
        logger.info("Step 3: Exporting annotated data")
        train_path, dev_path = export_prodigy_data(
            dataset_name=dataset_name,
            output_dir=annotation_dir
        )

        if train_path is None or dev_path is None:
            logger.error("Failed to export annotated data. Exiting.")
            return

        # Step 4: Train model
        logger.info("Step 4: Training spaCy NER model")
        model_path = train_ner_model(
            train_file=train_path,
            dev_file=dev_path,
            output_dir=output_dir,
            base_model=base_model
        )

        if model_path is None:
            logger.error("Model training failed.")
            return

        # Step 5: Evaluate model
        logger.info("Step 5: Evaluating trained model")
        evaluation = evaluate_model(
            model_path=model_path,
            test_file=dev_path
        )

        # Save evaluation results
        eval_file = os.path.join(output_dir, "evaluation_results.json")
        with open(eval_file, 'w') as f:
            json.dump(evaluation, f, indent=2)

        logger.info(f"Evaluation results saved to: {eval_file}")
        logger.info(f"Trained model saved to: {model_path}")

    logger.info("Process completed successfully.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare and train custom NER models for product data extraction")
    parser.add_argument("--input-dir", required=True,
                        help="Directory with processed PDF files")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to save model and data")
    parser.add_argument("--dataset", required=True,
                        help="Name for the Prodigy dataset")
    parser.add_argument(
        "--base-model", default="en_core_web_sm", help="Base spaCy model")
    parser.add_argument("--entity-types", nargs="+",
                        help="Entity types to annotate")
    parser.add_argument("--no-train", action="store_true",
                        help="Skip model training")

    args = parser.parse_args()

    main(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        dataset_name=args.dataset,
        train_model=not args.no_train,
        base_model=args.base_model,
        entity_types=args.entity_types
    )
