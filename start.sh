#!/bin/bash

echo "🚀 Starting DocuFind AI..."

# Activate virtual environment
source venv/bin/activate

# Check if requirements are installed
if [ ! -f "requirements_installed.flag" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    touch requirements_installed.flag
fi

# Create necessary directories
mkdir -p data/documents data/images vector_store logs static

# Add sample data if empty
if [ ! -f "data/documents/hostel_rules.txt" ]; then
    echo "Creating sample data..."
    echo "Hostel Rules and Regulations" > data/documents/hostel_rules.txt
    echo "1. Curfew at 10 PM" >> data/documents/hostel_rules.txt
    echo "2. No loud music after 9 PM" >> data/documents/hostel_rules.txt
    echo "3. Keep rooms clean" >> data/documents/hostel_rules.txt
fi

# Run the application
python run.py
