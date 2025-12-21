import hashlib
import json
import pickle
from pathlib import Path
from typing import Optional, Any, Dict


class FolderCache:
    """Simple disk cache for folder embeddings keyed by folder fingerprint."""

    def __init__(self, base_dir: str = ".cache/folder_embeddings"):
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)

    def _fingerprint(self, folder: Path) -> str:
        # fingerprint based on file paths, sizes and mtimes
        items = []
        for f in sorted(folder.iterdir()):
            if not f.is_file():
                continue
            st = f.stat()
            items.append(f"{f.name}:{st.st_mtime_ns}:{st.st_size}")
        raw = "|".join(items)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _cache_file(self, folder: Path) -> Path:
        key = hashlib.sha256(str(folder.resolve()).encode("utf-8")).hexdigest()
        return self.base / f"{key}.pkl"

    def load(self, folder: str) -> Optional[Dict[str, Any]]:
        folder_p = Path(folder)
        if not folder_p.exists():
            return None
        cache_file = self._cache_file(folder_p)
        if not cache_file.exists():
            return None
        try:
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            # validate fingerprint
            current_fp = self._fingerprint(folder_p)
            if data.get("fingerprint") != current_fp:
                return None
            return data
        except Exception:
            return None

    def save(self, folder: str, items: Any) -> None:
        folder_p = Path(folder)
        cache_file = self._cache_file(folder_p)
        try:
            payload = {
                "fingerprint": self._fingerprint(folder_p),
                "items": items,
            }
            with open(cache_file, "wb") as f:
                pickle.dump(payload, f)
        except Exception:
            pass

    def invalidate(self, folder: str) -> None:
        folder_p = Path(folder)
        cache_file = self._cache_file(folder_p)
        if cache_file.exists():
            try:
                cache_file.unlink()
            except Exception:
                pass
