# DocuFindAI – Desktop Local Search (Windows)

This desktop client performs **100% local** indexing + search.

- Raw files stay on your PC.
- A local SQLite DB stores only extracted text + embeddings + metadata.
- Searching is local (cosine similarity over cached embeddings).

## Run (dev)

```powershell
cd E:\Python\DocFinder\DocuFindAI
python -m venv venv
./venv/Scripts/Activate.ps1

pip install -r requirements-desktop.txt

python .\src\ui\kivy_local_app.py
```

## Usage

1. Click **Browse Folder** and choose a folder.
2. Click **Sync Folder** (indexes new/changed files; unchanged files are skipped).
3. Enter a query and click **Search**.
4. Click a result row to open the file (desktop).

The local DB is stored in your Kivy user data directory:
- `local_index.sqlite3`
- `model_cache/`

## Build a single-file EXE (PyInstaller)

From the UI folder:

```powershell
cd E:\Python\DocFinder\DocuFindAI\src\ui
pyinstaller --clean --onefile --noconsole --additional-hooks-dir=. kivy_local_app.py
```

The exe will be created under `dist/DocuFindAI_Local.exe` (or similar).

### Optional: bundle embedding model cache

Fastembed models may download on first run. If you want to **pre-bundle** a model cache:

1) Run the app once (or write a small script that embeds a sample string) so the model downloads into the cache.
2) Copy the resulting cache directory into `assets/fastembed_cache/`.
3) Build using the provided spec:

```powershell
cd E:\Python\DocFinder\DocuFindAI\src\ui
pyinstaller --clean --noconsole --onefile kivy_local_app.spec
```

At runtime, the exe will copy `fastembed_cache/` into the user cache dir if empty.

## Notes

- OCR uses **RapidOCR (ONNXRuntime)** by default. This avoids requiring a system Tesseract install.
- If the folder is large, Sync can take time; it runs in a background thread.
