@echo off
echo 🚀 Starting DocuFind AI...

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if requirements are installed
if not exist requirements_installed.flag (
    echo Installing dependencies...
    pip install -r requirements.txt
    type nul > requirements_installed.flag
)

REM Create necessary directories
mkdir data\documents 2>nul
mkdir data\images 2>nul
mkdir vector_store 2>nul
mkdir logs 2>nul
mkdir static 2>nul

REM Add sample data if empty
if not exist "data\documents\hostel_rules.txt" (
    echo Creating sample data...
    echo Hostel Rules and Regulations > data\documents\hostel_rules.txt
    echo 1. Curfew at 10 PM >> data\documents\hostel_rules.txt
    echo 2. No loud music after 9 PM >> data\documents\hostel_rules.txt
    echo 3. Keep rooms clean >> data\documents\hostel_rules.txt
)

REM Run the application
python run.py
