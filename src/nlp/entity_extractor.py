#!/usr/bin/env python
"""
Entity Extraction

This module handles the extraction of structured entities from processed text 
using spaCy NER models, with support for custom entity types relevant to product data.
"""

import os
import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import re

import spacy
from spacy.tokens import Doc, Span
from spacy.language import Language
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define product-specific entity types
PRODUCT_ENTITIES = [
    "PRODUCT",      # Product names
    "MANUFACTURER",  # Manufacturer names
    "BRAND",        # Brand names
    "MODEL",        # Model numbers/names
    "SPEC",         # Technical specifications
    "DIMENSION",    # Physical dimensions
    "MATERIAL",     # Materials used
    "CERTIFICATION",  # Certifications and standards
    "PRICE",        # Price information
    "SKU",          # SKU/product codes
    "DATE",         # Dates (manufacturing, expiry, etc.)
]

DEFAULT_MODEL = "en_core_web_md"


class EntityExtractor:
    """
    Extract product-related entities from text using spaCy models.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the entity extractor.

        Args:
            model_path: Path to a trained spaCy model (default: None, uses "en_core_web_md")
        """
        model_name = model_path if model_path else DEFAULT_MODEL
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading spaCy model '{model_name}': {e}")
            logger.info("Attempting to download the default model...")
            try:
                os.system(f"python -m spacy download {DEFAULT_MODEL}")
                self.nlp = spacy.load(DEFAULT_MODEL)
                logger.info(
                    f"Downloaded and loaded spaCy model: {DEFAULT_MODEL}")
            except Exception as download_error:
                logger.error(
                    f"Failed to download default model: {download_error}")
                raise RuntimeError(
                    f"Could not load or download spaCy model: {e}")

        self.add_product_pipeline_components()

    def add_product_pipeline_components(self):
        ruler = self.nlp.add_pipe("entity_ruler", before="ner")

        ruler.add_patterns([
            {"label": "PRODUCT", "pattern": [
                {"LOWER": {"REGEX": "(product|item)"}}]},
            {"label": "SKU", "pattern": [
                {"TEXT": {"REGEX": "^[A-Z0-9]{4,}-[A-Z0-9]{2,}$"}}
            ]},
            {"label": "DIMENSION", "pattern": [
                {"TEXT": {"REGEX": "^[0-9]+(\.[0-9]+)?$"}},
                {"LOWER": {"IN": ["mm", "cm", "m", "inch",
                                  "inches", "in", "ft", "foot", "feet"]}}
            ]},
            {"label": "WEIGHT", "pattern": [
                {"TEXT": {"REGEX": "^[0-9]+(\.[0-9]+)?$"}},
                {"LOWER": {"IN": ["kg", "g", "gram", "grams", "kilograms",
                                  "lb", "lbs", "pound", "pounds", "oz", "ounce", "ounces"]}}
            ]},
            {"label": "CERTIFICATION", "pattern": [
                {"LOWER": {
                    "REGEX": "(ce|rohs|iso|astm|en|iec|ul|csa|vde|tuv|fcc|reach|weee)"}}
            ]},
            {"label": "PRICE", "pattern": [
                {"TEXT": {"REGEX": "^(\$|€|£|¥|USD|EUR|GBP|JPY)$"}},
                {"TEXT": {"REGEX": "^[0-9]+(\.[0-9]+)?$"}}
            ]},
            {"label": "DATE", "pattern": [
                {"TEXT": {
                    "REGEX": "^(0?[1-9]|[12][0-9]|3[01])/(0?[1-9]|1[012])/((19|20)\\d\\d)$"}}
            ]},
        ])

    def extract_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        entities = {
            "PRODUCT": [],
            "ORGANIZATION": [],
            "SKU": [],
            "DIMENSION": [],
            "WEIGHT": [],
            "CERTIFICATION": [],
            "PRICE": [],
            "DATE": [],
            "MISC": []
        }

        doc = self.nlp(text)

        for ent in doc.ents:
            entity_data = {
                "text": ent.text,
                "start_char": ent.start_char,
                "end_char": ent.end_char,
                "context": text[max(0, ent.start_char - 40):min(len(text), ent.end_char + 40)]
            }

            if ent.label_ in entities:
                entities[ent.label_].append(entity_data)
            else:
                entities["MISC"].append(entity_data)

        return entities

    def extract_product_specifications(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        specs = []

        pattern = r"([\w\s]+):\s*([\w\s\d\.,-]+)(?:\s*\(([a-zA-Z%]+)\))?"
        matches = re.finditer(pattern, text)

        for match in matches:
            if match.group(1) and match.group(2):
                spec_data = {
                    "name": match.group(1).strip(),
                    "value": match.group(2).strip(),
                    "unit": match.group(3).strip() if match.group(3) else None,
                    "context": text[max(0, match.start() - 20):min(len(text), match.end() + 20)]
                }
                specs.append(spec_data)

        return {"specifications": specs}

    def extract_tabular_specifications(self, tables: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        specs = []

        for table in tables:
            for row in table.get("data", []):
                if len(row) >= 2:
                    spec_name = row[0].strip() if isinstance(
                        row[0], str) else ""
                    spec_value = row[1].strip() if isinstance(
                        row[1], str) else ""

                    if spec_name and spec_value and not spec_name.lower() in ["property", "specification", "parameter", "spec"]:
                        unit = None
                        value_parts = re.match(
                            r"([\d\.,-]+)\s*([a-zA-Z%]+)?", spec_value)

                        if value_parts:
                            if value_parts.group(2):
                                spec_value = value_parts.group(1).strip()
                                unit = value_parts.group(2).strip()

                        spec_data = {
                            "name": spec_name,
                            "value": spec_value,
                            "unit": unit,
                            "from_table": True,
                            "table_index": tables.index(table)
                        }
                        specs.append(spec_data)

        return {"specifications": specs}

    def process_document(self, pdf_data: Dict[str, Any]) -> Dict[str, Any]:
        results = {
            "metadata": pdf_data.get("metadata", {}),
            "entities": {},
            "specifications": []
        }

        text_content = pdf_data.get("text", "")

        entities = self.extract_entities(text_content)
        results["entities"] = entities

        specs_from_text = self.extract_product_specifications(text_content)
        results["specifications"].extend(
            specs_from_text.get("specifications", []))

        tables = pdf_data.get("tables", [])
        if tables:
            specs_from_tables = self.extract_tabular_specifications(tables)
            results["specifications"].extend(
                specs_from_tables.get("specifications", []))

        return results


def process_document(pdf_data: Dict[str, Any], model_path: Optional[str] = None) -> Dict[str, Any]:
    extractor = EntityExtractor(model_path)
    return extractor.process_document(pdf_data)


# Example usage
if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python entity_extractor.py <json_file> [model_path]")
        sys.exit(1)

    # Load PDF extraction results
    with open(sys.argv[1], 'r', encoding='utf-8') as f:
        pdf_info = json.load(f)

    # Get model path if provided
    model_path = sys.argv[2] if len(sys.argv) > 2 else None

    # Process the document
    result = process_document(pdf_info, model_path)

    # Print entities found
    print(f"Entities found in {pdf_info['filename']}:")
    for entity_type, entities in result['entities'].items():
        print(f"  {entity_type}: {len(entities)}")
        for entity in entities[:5]:  # Print first 5 of each type
            print(f"    - {entity['text']}")

        if len(entities) > 5:
            print(f"    ... and {len(entities) - 5} more")

    # Save results to file
    output_file = f"{os.path.splitext(sys.argv[1])[0]}_entities.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {output_file}")
