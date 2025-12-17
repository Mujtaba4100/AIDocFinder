# 🔍 DocuFind AI

A unified document and image search system that allows natural language search across PDFs, documents, and images.

## ✨ Features

- **Multi-format support**: PDF, DOC, TXT, JPG, PNG, and more
- **Natural language search**: "Find PDFs about hostel rules with blue buildings"
- **Hybrid search**: Combines text and image search
- **Smart processing**: OCR, metadata extraction, auto-tagging
- **Multiple interfaces**: Web UI, CLI, and API

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/DocuFindAI.git
cd DocuFindAI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 2. Add Sample Data

```bash
# Create some test files
echo "Hostel Rules and Regulations" > data/documents/rules.txt
echo "1. Curfew at 10 PM" >> data/documents/rules.txt
echo "2. No loud music after 9 PM" >> data/documents/rules.txt
```

## 3. Run the Application

```bash
# Run the complete system
python run.py

# Or run components separately:

# FastAPI Backend (http://localhost:8000)
python app.py

# Streamlit UI (http://localhost:8501)
streamlit run src/ui/streamlit_app.py

# CLI Interface
python src/ui/cli.py
```

## 📁 Project Structure

DocuFindAI/
├── app.py                    # FastAPI backend
├── run.py                    # Main runner
├── requirements.txt          # Dependencies
├── data/                     # Uploaded files
│   ├── documents/           # PDF, DOC, TXT files
│   └── images/              # JPG, PNG files
├── src/
│   ├── processors/          # File processors
│   ├── database/            # Vector store
│   ├── search/              # Search algorithms
│   ├── ui/                  # Interfaces
│   └── utils/               # Utilities
├── vector_store/            # ChromaDB storage
└── logs/                    # Application logs

## 🔧 Usage Examples

### Web Interface
Open http://localhost:8501

Upload files using the sidebar

Search using natural language:

"hostel rules"

"images with blue cars"

"PDFs about policies"

### CLI Interface

```bash
# Search
python src/ui/cli.py search "hostel rules"

# Index a folder
python src/ui/cli.py index --folder ./data

# Show statistics
python src/ui/cli.py stats
```

### API Usage

```bash
# Search via API
curl -X POST "http://localhost:8000/search/" \
  -H "Content-Type: application/json" \
  -d '{"query": "hostel rules"}'

# Upload file
curl -X POST "http://localhost:8000/upload/" \
  -F "file=@./data/documents/rules.pdf"
```

## 🛠️ Technology Stack

Backend: FastAPI, Uvicorn

Frontend: Streamlit, HTML/CSS

Vector Database: ChromaDB

AI/ML: Sentence Transformers, CLIP, PyTorch

Document Processing: PyMuPDF, python-docx, EasyOCR

Utilities: Pydantic, NumPy, Pandas

## 🤝 Contributing

Fork the repository

Create a feature branch

Commit your changes

Push to the branch

Open a Pull Request

## 📄 License
MIT License - see LICENSE file for details
