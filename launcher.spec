# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for DocuFindLocal (onedir build).

Build with:
    python -m PyInstaller --clean --noconfirm launcher.spec

Output:
    dist/DocuFindLocal/DocuFindLocal.exe
"""
from __future__ import annotations

import os
from pathlib import Path


block_cipher = None

# This spec lives in DocuFindLocal/, so project_root is that folder.
# When PyInstaller executes a spec via runpy, __file__ may be undefined; fall back to CWD.
if '__file__' in globals():
    project_root = Path(__file__).resolve().parent
else:
    project_root = Path.cwd()

# ---------------------------------------------------------------------------
# Data files to bundle
# ---------------------------------------------------------------------------
datas = []

# 1. Pre-fetched fastembed model cache (if exists)
#    This allows the app to work offline immediately without downloading models.
assets_cache = project_root / "assets" / "fastembed_cache"
if assets_cache.exists() and assets_cache.is_dir():
    # Bundle entire fastembed_cache directory
    datas.append((str(assets_cache), "fastembed_cache"))

# 2. Any other assets folder (icons, images, etc.)
assets_dir = project_root / "assets"
if assets_dir.exists():
    # Add individual asset files (not subdirs already handled)
    for item in assets_dir.iterdir():
        if item.is_file():
            datas.append((str(item), "assets"))

# ---------------------------------------------------------------------------
# Hidden imports for docufind_local package
# ---------------------------------------------------------------------------
hiddenimports = [
    # Core local_search modules
    'docufind_local.local_search.indexer',
    'docufind_local.local_search.searcher',
    'docufind_local.local_search.embedder',
    'docufind_local.local_search.text_extractors',
    'docufind_local.local_search.ocr',
    'docufind_local.local_search.db',
    'docufind_local.local_search.utils',
    'docufind_local.local_search.constants',
    
    # UI module
    'docufind_local.ui.kivy_local_app',
    
    # Common dependencies that may not be auto-detected
    'PIL',
    'PIL.Image',
    'numpy',
    'sqlite3',
    'tkinter',
    'tkinter.filedialog',
    
    # fastembed and onnxruntime
    'fastembed',
    'onnxruntime',
    
    # Kivy/KivyMD components
    'kivy',
    'kivymd',
    'kivy.uix.screenmanager',
]

# Custom hooks directory
hookspath = [str(project_root / "packaging" / "pyinstaller" / "hooks")]

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
a = Analysis(
    ['launcher.py'],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=hookspath,
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary modules to reduce size
        'matplotlib',
        'scipy',
        'pandas',
        'IPython',
        'notebook',
        'jupyter',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ---------------------------------------------------------------------------
# EXE (onedir mode - scripts only, no binaries/datas embedded)
# ---------------------------------------------------------------------------
exe = EXE(
    pyz,
    a.scripts,
    [],  # Empty - binaries/datas go in COLLECT for onedir
    exclude_binaries=True,  # Required for onedir
    name='DocuFindLocal',
    debug=False,
    strip=False,
    upx=True,
    console=True,  # TEMPORARY: Enable console for diagnostics
    icon=str(project_root / "assets" / "icon.ico") if (project_root / "assets" / "icon.ico").exists() else None,
)

# ---------------------------------------------------------------------------
# COLLECT (onedir output folder)
# ---------------------------------------------------------------------------
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='DocuFindLocal',  # Output folder name: dist/DocuFindLocal/
)


