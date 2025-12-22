# DocuFindAI – Desktop Local Search (Windows)

**DocuFindAI** is a fully **offline desktop client** for indexing and searching documents locally.  
No files are sent over the internet — everything stays on your PC.

---

## Features

- 100% local search and indexing
- Raw files never leave your PC
- Local SQLite database stores extracted text, embeddings, and metadata
- Semantic search using cosine similarity over cached embeddings
- Local OCR with **RapidOCR (ONNXRuntime)** — no system Tesseract required

---

## Run (Development Mode)

```powershell
cd E:\Python\DocFinder\DocuFindAI\DocuFindLocal
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python .\main.py
Usage
Click Browse Folder and select the folder to index.

Click Sync Folder to index new or changed files. Unchanged files are skipped automatically.

Enter a query and click Search.

Click a result row to open the corresponding file.

Local storage paths:

local_index.sqlite3 — your search index and metadata

fastembed_cache/ — downloaded embedding models

These are stored in your Kivy user data directory by default.

Build a Single-File EXE (PyInstaller)
powershell
Copy code
cd E:\Python\DocFinder\DocuFindAI\DocuFindLocal
pyinstaller --clean --noconsole --onefile launcher.spec
Output: dist/DocuFindLocal.exe

Optional: Pre-Bundle Embedding Model Cache
To avoid downloading models on first run:

Run the app once (or use a small script) to download the model cache.

Copy the resulting cache directory into assets/fastembed_cache/.

Build the EXE using the spec file:

powershell
Copy code
cd E:\Python\DocFinder\DocuFindAI\DocuFindLocal
pyinstaller --clean --noconsole --onefile launcher.spec
At runtime, the EXE will copy fastembed_cache/ into the user cache directory if it’s empty.

Notes / Tips
Large folders may take longer to sync; indexing runs in a background thread.

EXE distribution is fully offline — no internet connection required.

Recommended: use ZIP folder build for faster startup if distributing via Downloads.

Downloads (GitHub Release / Google Drive)
DocuFindLocal.exe → Single-file executable

DocuFindLocal.zip → Folder-based build (recommended)

pgsql
Copy code

This is **fully polished**, clickable, and ready for users to follow and download.  

If you want, I can also **make an even shorter “Quick Start” version** for GitHub front page so users can get running in 1–2 minutes. Do you want me to do that?