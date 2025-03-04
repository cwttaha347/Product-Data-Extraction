#!/bin/bash
# Database initialization script

# Print colored output
function print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "\e[32m[SUCCESS]\e[0m $2"
    else
        echo -e "\e[31m[ERROR]\e[0m $2"
        exit 1
    fi
}

# Check if PostgreSQL is running
echo "Checking if PostgreSQL is running..."
pg_isready -q
if [ $? -ne 0 ]; then
    echo -e "\e[31m[ERROR]\e[0m PostgreSQL is not running. Please start PostgreSQL first."
    exit 1
fi
echo -e "\e[32m[SUCCESS]\e[0m PostgreSQL is running."

# Create database
echo "Creating database 'product_data'..."
createdb product_data 2>/dev/null || psql -c "DROP DATABASE IF EXISTS product_data; CREATE DATABASE product_data;" postgres
print_status $? "Database 'product_data' created."

# Initialize database schema
echo "Initializing database schema..."
python -m src.database.init_db
print_status $? "Database schema initialized."

# Install required Python packages
echo "Installing required Python packages..."
pip install -r requirements.txt
print_status $? "Python packages installed."

# Download spaCy models
echo "Downloading spaCy models..."
python -m spacy download en_core_web_sm
print_status $? "Downloaded spaCy model 'en_core_web_sm'."

# Check if we should download the larger model
read -p "Do you want to download the larger spaCy model (en_core_web_lg, ~800MB)? [y/N] " response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo "Downloading larger spaCy model..."
    python -m spacy download en_core_web_lg
    print_status $? "Downloaded spaCy model 'en_core_web_lg'."
fi

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p data/raw data/processed
print_status $? "Directories created."

echo -e "\e[32m[COMPLETE]\e[0m Database initialization complete."
echo "You can now run the application:"
echo "  - To start the API: python -m src.api.main"
echo "  - To process a PDF: python run_pipeline.py --input path/to/pdf --output-dir data/processed" 