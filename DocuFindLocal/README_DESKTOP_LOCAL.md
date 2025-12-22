# DocuFindLocal — Desktop Local Search (Windows)

DocuFindLocal is a privacy-first, offline desktop client for indexing and searching
documents on your PC. All text extraction, OCR and embedding work locally — no data
is sent to external services.

Key benefits
- 100% local: files never leave your machine
- Local SQLite index stores extracted text, embeddings and metadata
- Semantic search (embeddings + cosine similarity)
- Optional local OCR using RapidOCR (ONNXRuntime) — no system Tesseract required

Quick Start (development)
1. Open PowerShell and change into the project folder:

```powershell
cd E:\Python\DocFinder\DocuFindAI\DocuFindLocal
```

2. Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. Install runtime dependencies:

```powershell
pip install -r requirements.txt
```

4. Launch the local desktop app:

```powershell
python .\main.py
```

Usage (brief)
- Click **Browse Folder** and choose a folder to index.
- Click **Sync Folder** to extract text/ocr and build embeddings for new or changed files.
- Enter a query and click **Search** to run a local semantic search.
- Click a result row to open the file with the OS default app.

Where data is stored
- Local SQLite DB: `local_index.sqlite3` (located in your Kivy user data directory)
- Model cache: `fastembed_cache/` (downloaded models are stored in the user cache)
- Logs: stored under your Kivy user data directory (Kivy manages this path)

Building a single-file EXE (PyInstaller)
You can produce a one-file offline executable that bundles the app and models.

1. (Optional) Prefetch the fastembed model cache by running the app once.
	 Copy the resulting cache into `assets/fastembed_cache/` if you want to pre-bundle it.

2. Build the EXE from the `DocuFindLocal` folder:

```powershell
cd E:\Python\DocFinder\DocuFindAI\DocuFindLocal
pyinstaller --clean --noconsole --onefile launcher.spec
```

The produced executable will be written to `dist/` (e.g. `dist/DocuFindLocal.exe`). When
the EXE runs for the first time it will copy the bundled `fastembed_cache` into the
user cache directory if that directory is empty.

Tips & troubleshooting
- Indexing large folders can take time; the UI performs indexing in a background
	thread and updates status messages.
- If embedding models fail to load, ensure `fastembed` and `onnxruntime` are installed
	and that `FASTEMBED_CACHE_PATH` (if set) points to a writable directory.
- On Windows, if the app fails to start, check graphics drivers and Kivy dependencies
	(`kivy_deps.sdl2`, `kivy_deps.glew`, `kivy_deps.angle`).

Advanced: run as a module (recommended for development)

```powershell
python -m docufind_local.ui.kivy_local_app
```

Contributing & support
- The desktop app is intentionally minimal and privacy-preserving. If you want a
	short Quick Start for a GitHub release page, I can produce a one-paragraph
	summary and a pre-built ZIP containing the `dist/` executable.

License
- This project is provided as-is. See the repository root for licensing details.

If you want a shorter Quick Start or a README tailored to GitHub releases, tell me
the target audience and I will prepare a second, shorter variant.
