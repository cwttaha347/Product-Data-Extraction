# PowerShell initialization script for Windows

# Print colored output
function Print-Status {
    param (
        [int]$Status,
        [string]$Message
    )
    
    if ($Status -eq 0) {
        Write-Host "[SUCCESS] $Message" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] $Message" -ForegroundColor Red
        exit 1
    }
}

# Check if PostgreSQL is installed
Write-Host "Checking if PostgreSQL is installed..."
try {
    $pgVersion = (Get-Command psql -ErrorAction Stop).Version
    Write-Host "[SUCCESS] PostgreSQL is installed (Version: $pgVersion)" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] PostgreSQL is not found. Please install PostgreSQL first." -ForegroundColor Red
    Write-Host "Download from: https://www.postgresql.org/download/windows/"
    exit 1
}

# Create database
Write-Host "Creating database 'product_data'..."
try {
    & psql -c "DROP DATABASE IF EXISTS product_data; CREATE DATABASE product_data;" postgres
    if ($LASTEXITCODE -eq 0) {
        Print-Status 0 "Database 'product_data' created."
    } else {
        Print-Status 1 "Failed to create database."
    }
} catch {
    Print-Status 1 "Failed to execute PostgreSQL commands: $_"
}

# Initialize database schema
Write-Host "Initializing database schema..."
python -m src.database.init_db
Print-Status $LASTEXITCODE "Database schema initialized."

# Ensure all required Python packages are installed
Write-Host "Installing required Python packages..."
pip install -r requirements.txt
Print-Status $LASTEXITCODE "Python packages installed."

# Download spaCy models
Write-Host "Downloading spaCy models..."
python -m spacy download en_core_web_sm
Print-Status $LASTEXITCODE "Downloaded spaCy model 'en_core_web_sm'."

# Check if we should download the larger model
$response = Read-Host "Do you want to download the larger spaCy model (en_core_web_lg, ~800MB)? [y/N]"
if ($response -match "^[yY]") {
    Write-Host "Downloading larger spaCy model..."
    python -m spacy download en_core_web_lg
    Print-Status $LASTEXITCODE "Downloaded spaCy model 'en_core_web_lg'."
}

# Create necessary directories
Write-Host "Creating necessary directories..."
$directories = @("data\raw", "data\processed")
foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}
Print-Status 0 "Directories created."

Write-Host "[COMPLETE] Database initialization complete." -ForegroundColor Green
Write-Host "You can now run the application:"
Write-Host "  - To start the API: python -m src.api.main"
Write-Host "  - To process a PDF: python run_pipeline.py --input path/to/pdf --output-dir data/processed" 