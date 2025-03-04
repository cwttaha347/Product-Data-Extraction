#!/usr/bin/env python
"""
Data Processing Utilities

This module handles the transformation of extracted entities into structured data,
including cleaning, normalization, validation, and preparation for database storage.
"""

import re
import logging
import json
import uuid
import os
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EntityProcessor:
    """Process extracted entities into structured product data."""

    def __init__(self):
        """Initialize the entity processor."""
        # Compile regex patterns for common data formats
        self.patterns = {
            'model_number': re.compile(r'^[A-Z0-9]+-[A-Z0-9]+$|^[A-Z0-9]{5,}$', re.IGNORECASE),
            'dimension': re.compile(r'(\d+(?:\.\d+)?)\s*(mm|cm|m|in|ft)', re.IGNORECASE),
            'price': re.compile(r'[$€£¥]\s*(\d+(?:,\d{3})*(?:\.\d{2})?)|\b(\d+(?:,\d{3})*(?:\.\d{2})?)\s*[$€£¥]'),
            'weight': re.compile(r'(\d+(?:\.\d+)?)\s*(kg|g|lb|oz)', re.IGNORECASE),
            'date': re.compile(r'\b(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4})\b|\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}\b', re.IGNORECASE)
        }

    def clean_entity_text(self, text: str) -> str:
        """
        Clean and normalize entity text.

        Args:
            text: Raw entity text

        Returns:
            Cleaned and normalized text
        """
        if not text:
            return ""

        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', text.strip())

        # Remove special characters that aren't meaningful
        cleaned = re.sub(r'[^\w\s.,;:()\-\'"/$%&@#*+]', '', cleaned)

        return cleaned

    def normalize_dimension(self, value: str) -> Optional[Dict[str, Union[float, str]]]:
        """
        Normalize dimension values to a standard format.

        Args:
            value: Dimension string (e.g., "10.5 mm", "3 ft")

        Returns:
            Dictionary with normalized value and unit, or None if parsing fails
        """
        match = self.patterns['dimension'].search(value)
        if not match:
            return None

        try:
            # Extract numeric value and unit
            numeric_value = float(match.group(1).replace(',', ''))
            unit = match.group(2).lower()

            # Convert to standard units (mm for small, m for large)
            if unit == 'cm':
                numeric_value = numeric_value * 10
                unit = 'mm'
            elif unit == 'in' or unit == 'inch':
                numeric_value = numeric_value * 25.4
                unit = 'mm'
            elif unit == 'ft' or unit == 'feet' or unit == 'foot':
                numeric_value = numeric_value * 0.3048
                unit = 'm'

            return {
                'value': numeric_value,
                'unit': unit,
                'original': value
            }

        except Exception as e:
            logger.warning(f"Error normalizing dimension '{value}': {e}")
            return None

    def normalize_price(self, value: str) -> Optional[Dict[str, Union[float, str]]]:
        """
        Normalize price values to a standard format.

        Args:
            value: Price string (e.g., "$10.99", "5,000 €")

        Returns:
            Dictionary with normalized value and currency, or None if parsing fails
        """
        match = self.patterns['price'].search(value)
        if not match:
            return None

        try:
            # Determine which group matched and extract numeric part
            if match.group(1):
                # Format is "$10.99"
                numeric_str = match.group(1)
                # Find currency symbol
                currency = re.search(r'[$€£¥]', value).group(0)
            else:
                # Format is "10.99 $"
                numeric_str = match.group(2)
                # Find currency symbol
                currency = re.search(r'[$€£¥]', value).group(0)

            # Convert to standard format
            numeric_value = float(numeric_str.replace(',', ''))

            # Map currency symbols to codes
            currency_map = {
                '$': 'USD',
                '€': 'EUR',
                '£': 'GBP',
                '¥': 'JPY'
            }

            return {
                'value': numeric_value,
                'currency': currency_map.get(currency, currency),
                'original': value
            }

        except Exception as e:
            logger.warning(f"Error normalizing price '{value}': {e}")
            return None

    def extract_product_data(self, entities: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Extract structured product data from entities.

        Args:
            entities: Dictionary of entities by type

        Returns:
            Structured product data
        """
        product_data = {
            'name': None,
            'manufacturer': None,
            'sku': None,
            'specifications': [],
            'certifications': [],
            'dimensions': [],
            'price': None,
            'description': None
        }

        # Process product names
        product_names = entities.get('PRODUCT', [])
        if product_names:
            # Use the longest product name as the primary one
            product_data['name'] = max(
                product_names, key=lambda x: len(x['text']))['text']

        # Process manufacturers
        manufacturers = entities.get(
            'MANUFACTURER', []) + entities.get('ORG', [])
        if manufacturers:
            # Use the most frequently mentioned manufacturer
            manufacturer_counts = {}
            for m in manufacturers:
                clean_name = self.clean_entity_text(m['text'])
                manufacturer_counts[clean_name] = manufacturer_counts.get(
                    clean_name, 0) + 1

            product_data['manufacturer'] = max(
                manufacturer_counts.items(), key=lambda x: x[1])[0]

        # Process SKUs or model numbers
        skus = []
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                clean_text = self.clean_entity_text(entity['text'])
                if self.patterns['model_number'].match(clean_text):
                    skus.append(clean_text)

        if skus:
            product_data['sku'] = skus[0]  # Take the first identified SKU

        # Process specifications
        if 'SPEC' in entities:
            for spec in entities['SPEC']:
                product_data['specifications'].append({
                    'name': spec.get('header', 'Specification'),
                    'value': self.clean_entity_text(spec['text']),
                    'confidence': spec.get('confidence', 1.0)
                })

        # Process certifications
        if 'CERTIFICATION' in entities:
            for cert in entities['CERTIFICATION']:
                product_data['certifications'].append(
                    self.clean_entity_text(cert['text'])
                )

        # Process dimensions
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                if 'DIMENSION' in entity_type or entity_type == 'QUANTITY':
                    dim = self.normalize_dimension(entity['text'])
                    if dim:
                        product_data['dimensions'].append(dim)

        # Process price information
        price_entities = entities.get('MONEY', []) + entities.get('PRICE', [])
        for entity in price_entities:
            price = self.normalize_price(entity['text'])
            if price:
                product_data['price'] = price
                break

        return product_data

    def process_document(self, document_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a document with extracted entities into structured product data.

        Args:
            document_data: Document data with extracted entities

        Returns:
            Dictionary with document info and structured product data
        """
        if 'entities' not in document_data:
            logger.warning("No entities found in document data")
            return {
                'document_info': {
                    'filename': document_data.get('filename', 'unknown'),
                    'page_count': document_data.get('page_count', 0)
                },
                'products': []
            }

        # Extract product data from entities
        product_data = self.extract_product_data(document_data['entities'])

        # Only include if we found a product name
        products = []
        if product_data['name']:
            products.append(product_data)

        # Add metadata
        processed_data = {
            'product_id': uuid.uuid4().hex,
            'product_name': product_data['name'],
            'manufacturer': product_data['manufacturer'],
            'specifications': product_data['specifications'],
            'source_document': document_data.get('source_document', ''),
            'processed_date': datetime.now().isoformat(),
            'confidence_score': 1.0
        }

        # Return document info and product data
        return {
            'document_info': {
                'filename': document_data.get('filename', 'unknown'),
                'path': document_data.get('path', ''),
                'page_count': document_data.get('page_count', 0),
                'processed_date': processed_data['processed_date'],
                'requires_ocr': document_data.get('requires_ocr', False)
            },
            'products': products,
            'processed_data': processed_data
        }


def prepare_for_database(processed_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare processed data for database insertion.

    Args:
        processed_data: Processed document data

    Returns:
        Dictionary with data formatted for database insertion
    """
    db_data = {
        'document': {
            'filename': processed_data['document_info']['filename'],
            'file_path': processed_data['document_info']['path'],
            'pages': processed_data['document_info']['page_count'],
            'processed': True
        },
        'products': []
    }

    # Process each product
    for product in processed_data['products']:
        db_product = {
            'name': product['name'],
            'sku': product['sku'],
            'manufacturer': {
                'name': product['manufacturer']
            } if product['manufacturer'] else None,
            'specifications': [],
            'certifications': []
        }

        # Add specifications
        for spec in product['specifications']:
            db_product['specifications'].append({
                'name': spec['name'],
                'value': spec['value'],
                'unit': None  # Unit would be determined during normalization
            })

        # Add dimensions as specifications
        for dim in product['dimensions']:
            db_product['specifications'].append({
                'name': 'Dimension',
                'value': str(dim['value']),
                'unit': dim['unit']
            })

        # Add certifications
        for cert in product['certifications']:
            db_product['certifications'].append({
                'name': cert
            })

        db_data['products'].append(db_product)

    return db_data


def process_extracted_data(entities_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to process extracted data.

    Args:
        entities_data: Document data with extracted entities

    Returns:
        Data ready for database insertion
    """
    processor = EntityProcessor()
    processed_data = processor.process_document(entities_data)
    return prepare_for_database(processed_data)


def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def extract_product_name(entities_data: Dict[str, Any]) -> Optional[str]:
    product_entities = entities_data.get("entities", {}).get("PRODUCT", [])
    if product_entities:
        return normalize_text(product_entities[0].get("text", ""))

    org_entities = entities_data.get("entities", {}).get("ORG", [])
    if org_entities:
        for org in org_entities:
            context = org.get("context", "")
            if "product" in context.lower() or "model" in context.lower():
                return normalize_text(org.get("text", ""))

    return None


def extract_manufacturer(entities_data: Dict[str, Any]) -> Optional[str]:
    org_entities = entities_data.get("entities", {}).get("ORGANIZATION", [])
    if org_entities:
        return normalize_text(org_entities[0].get("text", ""))

    org_entities = entities_data.get("entities", {}).get("ORG", [])
    if org_entities:
        return normalize_text(org_entities[0].get("text", ""))

    return None


def extract_sku(entities_data: Dict[str, Any]) -> Optional[str]:
    sku_entities = entities_data.get("entities", {}).get("SKU", [])
    if sku_entities:
        return normalize_text(sku_entities[0].get("text", ""))

    return None


def extract_specifications(entities_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    specs = []

    for spec in entities_data.get("specifications", []):
        if "name" not in spec or "value" not in spec:
            continue

        normalized_spec = {
            "name": normalize_text(spec["name"]),
            "value": normalize_text(spec["value"]),
            "unit": normalize_text(spec["unit"]) if spec.get("unit") else None
        }

        specs.append(normalized_spec)

    dimension_entities = entities_data.get("entities", {}).get("DIMENSION", [])
    for dim in dimension_entities:
        text = dim.get("text", "")
        context = dim.get("context", "")

        dim_match = re.match(r'(\d+\.?\d*)\s*([a-z]+)', text.lower())
        if dim_match:
            dim_value = dim_match.group(1)
            dim_unit = dim_match.group(2)

            dim_name = "Dimension"
            if "width" in context.lower():
                dim_name = "Width"
            elif "height" in context.lower():
                dim_name = "Height"
            elif "length" in context.lower():
                dim_name = "Length"
            elif "depth" in context.lower():
                dim_name = "Depth"

            specs.append({
                "name": dim_name,
                "value": dim_value,
                "unit": dim_unit
            })

    weight_entities = entities_data.get("entities", {}).get("WEIGHT", [])
    for weight in weight_entities:
        text = weight.get("text", "")

        weight_match = re.match(r'(\d+\.?\d*)\s*([a-z]+)', text.lower())
        if weight_match:
            weight_value = weight_match.group(1)
            weight_unit = weight_match.group(2)

            specs.append({
                "name": "Weight",
                "value": weight_value,
                "unit": weight_unit
            })

    return specs


def extract_certifications(entities_data: Dict[str, Any]) -> List[str]:
    cert_entities = entities_data.get("entities", {}).get("CERTIFICATION", [])

    certifications = set()
    for cert in cert_entities:
        text = normalize_text(cert.get("text", ""))
        if text:
            certifications.add(text.upper())

    return list(certifications)


def merge_product_data(existing_data: Dict[str, Any], new_data: Dict[str, Any]) -> Dict[str, Any]:
    if not existing_data.get("products") and new_data.get("products"):
        return new_data

    if not new_data.get("products"):
        return existing_data

    result = existing_data.copy()

    for new_product in new_data.get("products", []):
        existing_product = None

        for product in result.get("products", []):
            if product.get("sku") and new_product.get("sku") and product["sku"] == new_product["sku"]:
                existing_product = product
                break

            if product.get("name") and new_product.get("name") and product["name"] == new_product["name"]:
                if product.get("manufacturer") and new_product.get("manufacturer") and product["manufacturer"] == new_product["manufacturer"]:
                    existing_product = product
                    break

        if existing_product:
            if not existing_product.get("manufacturer") and new_product.get("manufacturer"):
                existing_product["manufacturer"] = new_product["manufacturer"]

            if not existing_product.get("sku") and new_product.get("sku"):
                existing_product["sku"] = new_product["sku"]

            if new_product.get("specifications"):
                existing_specs = {spec["name"].lower(
                ): spec for spec in existing_product.get("specifications", [])}

                for new_spec in new_product["specifications"]:
                    if new_spec["name"].lower() not in existing_specs:
                        if not existing_product.get("specifications"):
                            existing_product["specifications"] = []
                        existing_product["specifications"].append(new_spec)

            if new_product.get("certifications"):
                existing_certs = set(
                    existing_product.get("certifications", []))
                new_certs = set(new_product["certifications"])
                existing_product["certifications"] = list(
                    existing_certs.union(new_certs))
        else:
            if not result.get("products"):
                result["products"] = []
            result["products"].append(new_product)

    return result


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python data_processor.py <json_file>")
        sys.exit(1)

    # Load document data with entities
    with open(sys.argv[1], 'r', encoding='utf-8') as f:
        document_data = json.load(f)

    # Process the data
    result = process_extracted_data(document_data)

    # Print summary
    print(f"Document: {result['document']['filename']}")
    print(f"Products found: {len(result['products'])}")

    for i, product in enumerate(result['products']):
        print(f"\nProduct {i+1}:")
        print(f"  Name: {product['name']}")
        print(f"  SKU: {product['sku']}")
        print(
            f"  Manufacturer: {product['manufacturer']['name'] if product['manufacturer'] else 'Unknown'}")
        print(f"  Specifications: {len(product['specifications'])}")
        print(f"  Certifications: {len(product['certifications'])}")

    # Save to file
    output_file = f"{os.path.splitext(sys.argv[1])[0]}_processed.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nProcessed data saved to: {output_file}")
