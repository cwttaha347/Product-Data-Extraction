#!/usr/bin/env python

from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import FastAPI, File, UploadFile, Form, Query, HTTPException, BackgroundTasks
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import uuid
import tempfile
import json
import logging
import os
import sys
# Add the src directory to the path to make imports work properly
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')))

try:
    # When importing as module from src directory
    from database.db_operations import DatabaseManager
    from utils.data_processor import process_extracted_data
    from nlp.entity_extractor import process_document as extract_entities
    from pdf_processing.pipeline import process_pdf
except ImportError:
    # Fallback to direct import with src prefix
    from src.database.db_operations import DatabaseManager
    from src.utils.data_processor import process_extracted_data
    from src.nlp.entity_extractor import process_document as extract_entities
    from src.pdf_processing.pipeline import process_pdf


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Product Data Extraction API",
    description="API for extracting product data from PDF documents",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db_manager = DatabaseManager()


class ProductBase(BaseModel):
    name: str
    sku: Optional[str] = None
    description: Optional[str] = None
    manufacturer: Optional[str] = None


class ProductSpecification(BaseModel):
    name: str
    value: str
    unit: Optional[str] = None


class ProductDetail(ProductBase):
    id: int
    specifications: List[ProductSpecification] = []
    certifications: List[str] = []
    date_extracted: Optional[str] = None


class DocumentInfo(BaseModel):
    id: int
    filename: str
    upload_date: str
    pages: Optional[int] = None
    product_count: int


class Statistics(BaseModel):
    total_documents: int
    total_products: int
    total_manufacturers: int
    total_certifications: int
    recent_documents: List[DocumentInfo] = []


def process_pdf_background(
    file_path: str,
    output_dir: str,
    model_path: Optional[str] = None
):
    try:
        logger.info(f"Processing PDF: {file_path}")

        pdf_info = process_pdf(file_path)

        pdf_text_path = os.path.join(
            output_dir, f"{os.path.basename(file_path)}_text.json")
        with open(pdf_text_path, 'w', encoding='utf-8') as f:
            json.dump(pdf_info, f, indent=2, ensure_ascii=False)

        entities_data = extract_entities(pdf_info, model_path)

        entities_path = os.path.join(
            output_dir, f"{os.path.basename(file_path)}_entities.json")
        with open(entities_path, 'w', encoding='utf-8') as f:
            json.dump(entities_data, f, indent=2, ensure_ascii=False)

        processed_data = process_extracted_data(entities_data)

        processed_path = os.path.join(
            output_dir, f"{os.path.basename(file_path)}_processed.json")
        with open(processed_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)

        db_result = db_manager.store_processed_data(processed_data)

        result_path = os.path.join(
            output_dir, f"{os.path.basename(file_path)}_result.json")
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(db_result, f, indent=2, ensure_ascii=False)

        logger.info(f"Processing completed for: {file_path}")

    except Exception as e:
        logger.error(f"Error processing PDF: {e}")

        error_path = os.path.join(
            output_dir, f"{os.path.basename(file_path)}_error.json")
        with open(error_path, 'w', encoding='utf-8') as f:
            json.dump({"error": str(e)}, f, indent=2)


@app.get("/")
async def root():
    return {"message": "Product Data Extraction API v1.0.0"}


@app.post("/api/documents")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    process_now: bool = Form(False),
    custom_model: Optional[str] = Form(None)
):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    try:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), "data")
        processed_dir = os.path.join(data_dir, "processed")
        os.makedirs(processed_dir, exist_ok=True)

        unique_id = uuid.uuid4().hex
        file_name = f"{unique_id}_{file.filename}"
        file_path = os.path.join(processed_dir, file_name)

        with open(file_path, "wb") as f:
            f.write(await file.read())

        document_data = {
            "filename": file.filename,
            "file_path": file_path,
            "processed": False,
            "pages": 0
        }

        document_id = db_manager.store_document(document_data)

        if not document_id:
            raise HTTPException(
                status_code=500, detail="Failed to create document record")

        if process_now:
            background_tasks.add_task(
                process_pdf_background,
                file_path=file_path,
                output_dir=processed_dir,
                model_path=custom_model
            )
            processing_status = "processing"
        else:
            processing_status = "pending"

        return {
            "document_id": document_id,
            "filename": file.filename,
            "status": processing_status,
            "message": "Document uploaded successfully"
        }

    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error uploading document: {str(e)}")


@app.post("/api/documents/{document_id}/process")
async def process_document(
    document_id: int,
    background_tasks: BackgroundTasks,
    custom_model: Optional[str] = None
):
    session = db_manager.Session()

    try:
        from src.database.init_db import Document as DocumentModel

        document = session.query(DocumentModel).get(document_id)

        if not document:
            raise HTTPException(
                status_code=404, detail=f"Document {document_id} not found")

        if document.processed:
            return {"message": "Document already processed", "document_id": document_id}

        data_dir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), "data")
        processed_dir = os.path.join(data_dir, "processed")
        os.makedirs(processed_dir, exist_ok=True)

        background_tasks.add_task(
            process_pdf_background,
            file_path=document.file_path,
            output_dir=processed_dir,
            model_path=custom_model
        )

        return {
            "document_id": document_id,
            "filename": document.filename,
            "status": "processing",
            "message": "Document processing started"
        }

    except Exception as e:
        logger.error(f"Error starting document processing: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error starting document processing: {str(e)}")

    finally:
        session.close()


@app.get("/api/documents/{document_id}")
async def get_document(document_id: int):
    session = db_manager.Session()

    try:
        from src.database.init_db import Document as DocumentModel
        from src.database.init_db import Product as ProductModel

        document = session.query(DocumentModel).get(document_id)

        if not document:
            raise HTTPException(
                status_code=404, detail=f"Document {document_id} not found")

        products = session.query(ProductModel).filter_by(
            document_id=document_id).all()

        products_data = []
        for product in products:
            products_data.append({
                "id": product.id,
                "name": product.name,
                "sku": product.sku
            })

        return {
            "id": document.id,
            "filename": document.filename,
            "upload_date": document.upload_date.isoformat() if document.upload_date else None,
            "processed": document.processed,
            "pages": document.pages,
            "products": products_data
        }

    except Exception as e:
        logger.error(f"Error getting document: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error getting document: {str(e)}")

    finally:
        session.close()


@app.get("/api/products", response_model=List[ProductDetail])
async def list_products(
    query: Optional[str] = None,
    manufacturer: Optional[str] = None,
    certification: Optional[str] = None,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    try:
        products = db_manager.search_products(
            query=query,
            manufacturer=manufacturer,
            certification=certification,
            limit=limit,
            offset=offset
        )

        return products

    except Exception as e:
        logger.error(f"Error listing products: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error listing products: {str(e)}")


@app.get("/api/products/{product_id}", response_model=ProductDetail)
async def get_product(product_id: int):
    try:
        product = db_manager.get_product_by_id(product_id)

        if not product:
            raise HTTPException(
                status_code=404, detail=f"Product {product_id} not found")

        return product

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting product: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error getting product: {str(e)}")


@app.delete("/api/products/{product_id}")
async def delete_product(product_id: int):
    try:
        success = db_manager.delete_product(product_id)

        if not success:
            raise HTTPException(
                status_code=404, detail=f"Product {product_id} not found")

        return {"message": f"Product {product_id} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting product: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error deleting product: {str(e)}")


@app.get("/api/statistics", response_model=Statistics)
async def get_statistics():
    try:
        stats = db_manager.get_statistics()

        if not stats:
            raise HTTPException(
                status_code=500, detail="Failed to retrieve statistics")

        return stats

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error getting statistics: {str(e)}")


@app.get("/health")
def health_check():
    """Health check endpoint to verify API is running"""
    return {"status": "ok", "message": "API is running"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))

    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
