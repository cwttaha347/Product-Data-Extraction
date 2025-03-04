# Product Data Extraction Platform

A platform for extracting structured product data from PDF documents using NLP and storing it in a PostgreSQL database.

## Overview

This system extracts product specifications, SKUs, dimensions, and certifications from product documentation PDFs. It uses SpaCy for Named Entity Recognition (NER) and provides a RESTful API for managing the extraction process.

## Features

- PDF text extraction with OCR support
- Named Entity Recognition for product data
- PostgreSQL database for structured storage
- REST API for document upload and querying
- Batch processing for multiple documents

## Setup

### Prerequisites

- Python 3.8+
- PostgreSQL
- Tesseract OCR (optional, for image-based PDFs)

### Installation

1. Clone the repository:
```
git clone https://github.com/cwttaha347/Product-Data-Extraction.git
cd Product-Data-Extraction
```

2. Run the initialization script:
```
chmod +x init_db.sh
./init_db.sh
```

This script will:
- Verify PostgreSQL is running
- Create the database
- Initialize the schema
- Install required packages
- Download SpaCy models

## Usage

### Processing PDFs

Place PDF files in the `data/raw` directory, then run:

```
python run_pipeline.py --input data/raw --output-dir data/processed
```

### API

Start the API server:

```
python -m src.api.main
```

Access the API at `http://localhost:8000`:
- POST `/api/documents` - Upload PDFs
- GET `/api/products` - Query extracted products
- GET `/api/statistics` - System stats

The Swagger UI is available at `http://localhost:8000/docs`.

## Project Structure

```
.
├── data/
│   ├── raw/           # Raw PDF files
│   └── processed/     # Processing results
├── src/
│   ├── api/           # REST API
│   ├── database/      # Database operations
│   ├── pdf_processing/# PDF extraction
│   ├── nlp/           # Entity extraction
│   └── utils/         # Utilities
├── tests/             # Test suite
├── init_db.sh         # Database setup
├── requirements.txt   # Python dependencies
└── README.md
```

## License

[MIT License](LICENSE) 