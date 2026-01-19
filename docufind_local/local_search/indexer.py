from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np

from .constants import SUPPORTED_EXTS
from .db import now_ts, open_db
from .embedder import ClipEmbedder, TextEmbedder
from .ocr import OcrEngine
from .text_extractors import TextExtractor
from .utils import safe_relpath


ProgressCb = Callable[[str], None]


@dataclass
class IndexStats:
    scanned: int = 0
    indexed: int = 0
    skipped: int = 0
    failed: int = 0


class LocalIndexer:
    def __init__(self, db_path: str | Path, cache_dir: str | Path) -> None:
        self.db_path = Path(db_path)
        self.cache_dir = Path(cache_dir)
        self.ocr = OcrEngine()
        self.extractor = TextExtractor(self.ocr)
        self.embedder = TextEmbedder(cache_dir=self.cache_dir)
        self.clip = ClipEmbedder(cache_dir=self.cache_dir)

    def iter_files(self, folder: str | Path, recursive: bool = True, file_types: Optional[set[str]] = None) -> Iterable[Path]:
        root = Path(folder)
        pattern = "**/*" if recursive else "*"
        for p in root.glob(pattern):
            if not p.is_file():
                continue
            ext = p.suffix.lower()
            if file_types and ext not in file_types:
                continue
            if ext in SUPPORTED_EXTS:
                yield p

    def index_folder(self, folder: str | Path, progress: Optional[ProgressCb] = None, recursive: bool = True, file_types: Optional[set[str]] = None) -> IndexStats:
        folder = str(Path(folder).resolve())
        stats = IndexStats()

        if progress:
            progress("Opening local index...")

        con = open_db(self.db_path)
        try:
            pending_images: list[tuple[Path, str, str, int, int]] = []

            for file_path in self.iter_files(folder, recursive=recursive, file_types=file_types):
                stats.scanned += 1
                try:
                    rel = safe_relpath(file_path, folder)
                    st = file_path.stat()
                    mtime = int(st.st_mtime)
                    size = int(st.st_size)
                    ext = file_path.suffix.lower()

                    row = con.execute(
                        "SELECT mtime, size FROM files WHERE path = ?",
                        (str(file_path),),
                    ).fetchone()

                    if row and int(row[0]) == mtime and int(row[1]) == size:
                        stats.skipped += 1
                        continue

                    if ext in {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".tif"}:
                        pending_images.append((file_path, rel, ext, mtime, size))
                        continue

                    if progress:
                        progress(f"Processing: {rel}")

                    text = (self.extractor.extract(file_path) or "").strip()
                    if not text:
                        con.execute(
                            """
                            INSERT INTO files(folder, path, rel_path, ext, mtime, size, extracted_text, embedding, embedding_dim, image_embedding, image_embedding_dim, updated_at)
                            VALUES(?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, ?)
                            ON CONFLICT(path) DO UPDATE SET
                              folder=excluded.folder,
                              rel_path=excluded.rel_path,
                              ext=excluded.ext,
                              mtime=excluded.mtime,
                              size=excluded.size,
                              extracted_text=excluded.extracted_text,
                              embedding=NULL,
                              embedding_dim=NULL,
                              image_embedding=NULL,
                              image_embedding_dim=NULL,
                              updated_at=excluded.updated_at
                            """,
                            (folder, str(file_path), rel, ext, mtime, size, "", now_ts()),
                        )
                        con.commit()
                        stats.indexed += 1
                        continue

                    if not self.embedder.available:
                        raise RuntimeError(f"Embedding model unavailable: {self.embedder.init_error}")

                    vec = np.asarray(self.embedder.embed_text(text), dtype=np.float32)
                    con.execute(
                        """
                        INSERT INTO files(folder, path, rel_path, ext, mtime, size, extracted_text, embedding, embedding_dim, image_embedding, image_embedding_dim, updated_at)
                        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?)
                        ON CONFLICT(path) DO UPDATE SET
                          folder=excluded.folder,
                          rel_path=excluded.rel_path,
                          ext=excluded.ext,
                          mtime=excluded.mtime,
                          size=excluded.size,
                          extracted_text=excluded.extracted_text,
                          embedding=excluded.embedding,
                          embedding_dim=excluded.embedding_dim,
                          image_embedding=NULL,
                          image_embedding_dim=NULL,
                          updated_at=excluded.updated_at
                        """,
                        (folder, str(file_path), rel, ext, mtime, size, text, vec.tobytes(), int(vec.shape[0]), now_ts()),
                    )
                    con.commit()
                    stats.indexed += 1
                except Exception:
                    stats.failed += 1
                    continue

            if pending_images:
                if progress:
                    progress(f"Embedding {len(pending_images)} images...")

                clip_vecs: list[Optional[np.ndarray]] = []
                if self.clip.available:
                    try:
                        paths = [p for (p, _, _, _, _) in pending_images]
                        clip_vecs = self.clip.embed_images(paths, batch_size=16)
                    except Exception:
                        clip_vecs = []

                for idx, (img_path, rel, ext, mtime, size) in enumerate(pending_images):
                    try:
                        if progress:
                            progress(f"Processing image: {rel}")

                        text = (self.extractor.extract(img_path) or "").strip()

                        text_blob = None
                        text_dim = None
                        if text and self.embedder.available:
                            tv = np.asarray(self.embedder.embed_text(text), dtype=np.float32)
                            text_blob = tv.tobytes()
                            text_dim = int(tv.shape[0])

                        img_blob = None
                        img_dim = None
                        if clip_vecs and idx < len(clip_vecs):
                            iv = np.asarray(clip_vecs[idx], dtype=np.float32)
                            if iv.ndim == 1 and iv.shape[0] > 1:
                                img_blob = iv.tobytes()
                                img_dim = int(iv.shape[0])

                        con.execute(
                            """
                            INSERT INTO files(folder, path, rel_path, ext, mtime, size, extracted_text, embedding, embedding_dim, image_embedding, image_embedding_dim, updated_at)
                            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT(path) DO UPDATE SET
                              folder=excluded.folder,
                              rel_path=excluded.rel_path,
                              ext=excluded.ext,
                              mtime=excluded.mtime,
                              size=excluded.size,
                              extracted_text=excluded.extracted_text,
                              embedding=excluded.embedding,
                              embedding_dim=excluded.embedding_dim,
                              image_embedding=excluded.image_embedding,
                              image_embedding_dim=excluded.image_embedding_dim,
                              updated_at=excluded.updated_at
                            """,
                            (folder, str(img_path), rel, ext, mtime, size, text, text_blob, text_dim, img_blob, img_dim, now_ts()),
                        )
                        con.commit()
                        stats.indexed += 1
                    except Exception:
                        stats.failed += 1
                        continue
        finally:
            con.close()

        if progress:
            progress(f"Done. Indexed: {stats.indexed}, skipped: {stats.skipped}, failed: {stats.failed}")
        return stats
