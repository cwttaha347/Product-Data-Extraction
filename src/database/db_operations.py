#!/usr/bin/env python
"""
Database Operations

This module handles database operations for storing processed product data,
including efficient inserts, updates, and retrieval of information.
"""

import os
import logging
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime

from sqlalchemy import create_engine, update, select, delete, and_, or_, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

try:
    # When importing from src
    from database.init_db import (
        Base, Document, Product, Specification as ProductSpecification,
        Certification, product_certification as ProductCertification
    )
except ImportError:
    # Fallback to direct import with src prefix
    from src.database.init_db import (
        Base, Document, Product, Specification as ProductSpecification,
        Certification, product_certification as ProductCertification
    )

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatabaseManager:
    """Database operations manager for product data storage."""

    def __init__(self, db_url: Optional[str] = None):
        """
        Initialize the database manager.

        Args:
            db_url: Database connection URL (optional)
        """
        if not db_url:
            db_url = os.environ.get(
                "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/product_data")

        self.db_url = db_url
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)

    def get_session(self) -> Session:
        """
        Get a new session for database operations.

        Returns:
            SQLAlchemy session
        """
        return self.Session()

    def store_document(self, document_data: Dict[str, Any]) -> Optional[int]:
        """
        Store document information in the database.

        Args:
            document_data: Document data from processed PDF

        Returns:
            Document ID if successful, None if failed
        """
        session = self.get_session()

        try:
            document = Document(
                filename=document_data.get("filename", ""),
                file_path=document_data.get("file_path", ""),
                upload_date=datetime.now(),
                processed=document_data.get("processed", False),
                pages=document_data.get("pages", 0)
            )

            session.add(document)
            session.commit()

            return document.id

        except Exception as e:
            logger.error(f"Error storing document: {e}")
            session.rollback()
            return None

        finally:
            session.close()

    def store_extracted_text(self, document_id: int, extracted_text: List[Dict[str, Any]]) -> bool:
        """
        Store extracted text from document pages.

        Args:
            document_id: Document ID
            extracted_text: List of dictionaries with page numbers and text

        Returns:
            True if successful, False if failed
        """
        session = self.get_session()

        try:
            # Delete any existing extracted text for this document
            session.query(ExtractedText).filter_by(
                document_id=document_id).delete()

            # Insert new extracted text
            for item in extracted_text:
                text_entry = ExtractedText(
                    document_id=document_id,
                    page_number=item['page_number'],
                    text=item['text']
                )
                session.add(text_entry)

            session.commit()
            return True

        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error storing extracted text: {e}")
            return False

        finally:
            session.close()

    def get_or_create_manufacturer(self, session: Session, name: str) -> int:
        """
        Get an existing manufacturer or create a new one.

        Args:
            session: SQLAlchemy session
            name: Manufacturer name

        Returns:
            Manufacturer ID
        """
        # Check if manufacturer exists
        manufacturer = session.query(Manufacturer).filter_by(name=name).first()

        if manufacturer:
            return manufacturer.id

        # Create new manufacturer
        new_manufacturer = Manufacturer(name=name)
        session.add(new_manufacturer)
        session.flush()  # Get ID without committing

        return new_manufacturer.id

    def get_or_create_certification(self, session: Session, name: str) -> int:
        """
        Get an existing certification or create a new one.

        Args:
            session: SQLAlchemy session
            name: Certification name

        Returns:
            Certification ID
        """
        # Check if certification exists
        certification = session.query(
            Certification).filter_by(name=name).first()

        if certification:
            return certification.id

        # Create new certification
        new_certification = Certification(name=name)
        session.add(new_certification)
        session.flush()  # Get ID without committing

        return new_certification.id

    def store_product(self, product_data: Dict[str, Any], document_id: int) -> Optional[int]:
        """
        Store product information in the database.

        Args:
            product_data: Product data from processed document
            document_id: Document ID

        Returns:
            Product ID if successful, None if failed
        """
        session = self.get_session()

        try:
            # Get or create manufacturer if provided
            manufacturer_id = None
            if product_data.get('manufacturer'):
                manufacturer_id = self.get_or_create_manufacturer(
                    session,
                    product_data['manufacturer']['name']
                )

            # Check if product already exists (by name and SKU if available)
            query = session.query(Product).filter_by(name=product_data['name'])
            if product_data.get('sku'):
                query = query.filter_by(sku=product_data['sku'])

            existing_product = query.first()

            if existing_product:
                # Update existing product
                existing_product.description = product_data.get('description')
                existing_product.document_id = document_id

                if manufacturer_id:
                    existing_product.manufacturer_id = manufacturer_id

                # Update date extracted
                existing_product.date_extracted = datetime.now()

                # Delete existing specifications
                session.query(ProductSpecification).filter_by(
                    product_id=existing_product.id).delete()

                # Clear existing certifications
                existing_product.certifications = []

                product_id = existing_product.id

            else:
                # Create new product
                new_product = Product(
                    name=product_data['name'],
                    sku=product_data.get('sku'),
                    description=product_data.get('description'),
                    document_id=document_id,
                    manufacturer_id=manufacturer_id,
                    date_extracted=datetime.now(),
                    confidence_score=None  # Could be set if available
                )

                session.add(new_product)
                session.flush()  # Get ID without committing

                product_id = new_product.id

            # Add specifications
            for spec in product_data.get('specifications', []):
                specification = ProductSpecification(
                    product_id=product_id,
                    name=spec['name'],
                    value=spec['value'],
                    unit=spec.get('unit')
                )
                session.add(specification)

            # Add certifications
            product = session.query(Product).get(product_id)
            for cert_data in product_data.get('certifications', []):
                cert_id = self.get_or_create_certification(
                    session, cert_data['name'])
                certification = session.query(Certification).get(cert_id)
                product.certifications.append(certification)

            session.commit()
            return product_id

        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error storing product: {e}")
            return None

        finally:
            session.close()

    def store_processed_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store all processed data from a document.

        Args:
            data: Processed data containing document and product information

        Returns:
            Dictionary with results
        """
        session = self.get_session()
        result = {"success": False, "products": []}

        try:
            document = None

            if "source_file" in data:
                document_query = session.query(Document).filter_by(
                    file_path=data["source_file"])
                document = document_query.first()

                if document:
                    document.processed = True
                    document.process_date = datetime.now()
                    session.add(document)

            for product_data in data.get("products", []):
                product = Product(
                    name=product_data.get("name", "Unknown Product"),
                    sku=product_data.get("sku"),
                    description=None,
                    manufacturer=product_data.get("manufacturer"),
                    date_extracted=datetime.now()
                )

                if document:
                    product.document_id = document.id

                session.add(product)
                session.flush()

                for spec_data in product_data.get("specifications", []):
                    spec = ProductSpecification(
                        product_id=product.id,
                        name=spec_data.get("name", ""),
                        value=spec_data.get("value", ""),
                        unit=spec_data.get("unit")
                    )
                    session.add(spec)

                for cert_name in product_data.get("certifications", []):
                    cert = session.query(Certification).filter_by(
                        name=cert_name).first()

                    if not cert:
                        cert = Certification(name=cert_name)
                        session.add(cert)
                        session.flush()

                    # Add certification to product using the relationship
                    product.certifications.append(cert)

                result["products"].append({
                    "id": product.id,
                    "name": product.name,
                    "sku": product.sku
                })

            session.commit()
            result["success"] = True

            return result

        except Exception as e:
            logger.error(f"Error storing processed data: {e}")
            session.rollback()
            return {"success": False, "error": str(e)}

        finally:
            session.close()

    def search_products(self,
                        query: Optional[str] = None,
                        manufacturer: Optional[str] = None,
                        certification: Optional[str] = None,
                        limit: int = 20,
                        offset: int = 0) -> List[Dict[str, Any]]:
        """
        Search for products in the database.

        Args:
            query: Text search query
            manufacturer: Filter by manufacturer name
            certification: Filter by certification name
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of matching products with their details
        """
        session = self.get_session()

        try:
            base_query = session.query(Product)

            if query:
                base_query = base_query.filter(
                    (Product.name.ilike(f"%{query}%")) |
                    (Product.sku.ilike(f"%{query}%")) |
                    (Product.description.ilike(f"%{query}%"))
                )

            if manufacturer:
                base_query = base_query.filter(
                    Product.manufacturer.ilike(f"%{manufacturer}%"))

            if certification:
                base_query = base_query.join(
                    Product.certifications
                ).filter(
                    Certification.name.ilike(f"%{certification}%")
                )

            products = base_query.limit(limit).offset(offset).all()

            results = []
            for product in products:
                product_data = self.get_product_by_id(product.id)
                if product_data:
                    results.append(product_data)

            return results

        except Exception as e:
            logger.error(f"Error searching products: {e}")
            return []

        finally:
            session.close()

    def get_product_by_id(self, product_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a product by its ID with all details.

        Args:
            product_id: Product ID

        Returns:
            Product details or None if not found
        """
        session = self.get_session()

        try:
            product = session.query(Product).filter_by(id=product_id).first()

            if not product:
                return None

            specs = session.query(ProductSpecification).filter_by(
                product_id=product_id).all()

            result = {
                "id": product.id,
                "name": product.name,
                "sku": product.sku,
                "description": product.description,
                "manufacturer": product.manufacturer,
                "date_extracted": product.date_extracted.isoformat() if product.date_extracted else None,
                "specifications": [
                    {
                        "name": spec.name,
                        "value": spec.value,
                        "unit": spec.unit
                    } for spec in specs
                ],
                "certifications": [cert.name for cert in product.certifications]
            }

            return result

        except SQLAlchemyError as e:
            logger.error(f"Error getting product by ID: {e}")
            return None

        finally:
            session.close()

    def delete_product(self, product_id: int) -> bool:
        """
        Delete a product from the database.

        Args:
            product_id: ID of the product to delete

        Returns:
            True if successful, False if failed
        """
        session = self.get_session()

        try:
            product = session.query(Product).filter_by(id=product_id).first()

            if not product:
                return False

            session.query(ProductSpecification).filter_by(
                product_id=product_id).delete()
            product.certifications = []  # Remove all certifications
            session.delete(product)
            session.commit()

            return True

        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error deleting product: {e}")
            return False

        finally:
            session.close()

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dictionary with statistics
        """
        session = self.get_session()

        try:
            doc_count = session.query(Document).count()
            product_count = session.query(Product).count()

            manuf_count = session.query(
                Product.manufacturer).distinct().count()
            cert_count = session.query(Certification).count()

            recent_docs = session.query(Document).order_by(
                Document.upload_date.desc()).limit(5).all()

            recent_docs_data = []
            for doc in recent_docs:
                product_count = session.query(Product).filter_by(
                    document_id=doc.id).count()

                recent_docs_data.append({
                    "id": doc.id,
                    "filename": doc.filename,
                    "upload_date": doc.upload_date.isoformat() if doc.upload_date else None,
                    "pages": doc.pages,
                    "product_count": product_count
                })

            return {
                "total_documents": doc_count,
                "total_products": product_count,
                "total_manufacturers": manuf_count,
                "total_certifications": cert_count,
                "recent_documents": recent_docs_data
            }

        except SQLAlchemyError as e:
            logger.error(f"Error getting statistics: {e}")
            return {}

        finally:
            session.close()


def store_processed_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to store processed data.

    Args:
        processed_data: Processed document and product data

    Returns:
        Dictionary with results
    """
    db_manager = DatabaseManager()
    return db_manager.store_processed_data(data)


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python db_operations.py <processed_data_file>")
        sys.exit(1)

    # Load processed data
    with open(sys.argv[1], 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Store in database
    db_manager = DatabaseManager()
    result = db_manager.store_processed_data(data)

    # Print result
    if result['success']:
        print(f"Successfully stored document (ID: {result['document_id']})")
        print(f"Stored {len(result['product_ids'])} products")
        for product_id in result['product_ids']:
            print(f"  - Product ID: {product_id}")
    else:
        print("Failed to store data:")
        for error in result['errors']:
            print(f"  - {error}")
