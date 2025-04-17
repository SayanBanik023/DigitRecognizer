<#
.SYNOPSIS
Sets up the Handwritten Digit Recognition project environment on Windows.
#>

# Check if Python is installed
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Python is not installed. Please install Python 3.6+ first." -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host "Creating virtual environment..." -ForegroundColor Cyan
python -m venv venv

# Activate environment
Write-Host "Activating virtual environment..." -ForegroundColor Cyan
.\venv\Scripts\activate

# Upgrade pip and install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Cyan
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
Write-Host "Verifying TensorFlow installation..." -ForegroundColor Cyan
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__} installed')"

Write-Host "`nSetup completed successfully!`n" -ForegroundColor Green
Write-Host "To activate the environment later, run:" -ForegroundColor Yellow
Write-Host "    .\venv\Scripts\activate" -ForegroundColor White