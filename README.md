# Handwritten Digit Recognition System 
- A CNN-based system for recognizing handwritten digits (0-9) using the MNIST dataset.

## Prerequisites
- Python 3.6 or higher
- pip package manager

## Setup

### Using Virtual Environment (Recommended)

### 1. Check Execution Policy
```bash
  Get-ExecutionPolicy
```
### 2. If restricted, allow script execution (admin not required)
```bash
  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```
### 3. Create Environment
```bash
  python -m venv venv
```
### 4. Activate Environment
```bash
  .venv\Scripts\activate
```
### 5. Upgrade PIP and Install Dependencies
```bash
  python.exe -m pip install --upgrade pip
  pip install -r requirements.txt
```
### 6. Deactivate Environment
```bash
  deactivate
```
## Alternative for Windows: Use the setup script
```bash
  ./setup.ps1  # Automates all above steps
```