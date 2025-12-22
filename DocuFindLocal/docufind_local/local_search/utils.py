from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def safe_relpath(path: str | Path, start: str | Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(start).resolve()))
    except Exception:
        # Fallback: best-effort relpath
        try:
            return str(Path(path))
        except Exception:
            return str(path)
