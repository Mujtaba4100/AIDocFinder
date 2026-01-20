from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Callable, Iterable, List, Optional, Tuple

import numpy as np

from .constants import SUPPORTED_EXTS
from .db import now_ts, open_db
from .embedder import ClipEmbedder, TextEmbedder
from .ocr import OcrEngine
from .text_extractors import TextExtractor
from .utils import safe_relpath


ProgressCb = Callable[[str], None]

# Batch sizes for efficient processing
TEXT_BATCH_SIZE = 64
IMAGE_BATCH_SIZE = 16
MAX_WORKERS = 4  # Thread pool size for parallel text extraction


@dataclass
class IndexStats:
    scanned: int = 0
    indexed: int = 0
    skipped: int = 0
    failed: int = 0
    _lock: Lock = field(default_factory=Lock, repr=False)
    
    def inc_scanned(self) -> None:
        with self._lock:
            self.scanned += 1
    
    def inc_indexed(self) -> None:
        with self._lock:
            self.indexed += 1
    
    def inc_skipped(self) -> None:
        with self._lock:
            self.skipped += 1
    
    def inc_failed(self) -> None:
        with self._lock:
            self.failed += 1


@dataclass
class _PendingDoc:
    """Holds document data pending batch embedding."""
    file_path: Path
    rel: str
    ext: str
    mtime: int
    size: int
    text: str


@dataclass
class _PendingImage:
    """Holds image data pending batch processing."""
    file_path: Path
    rel: str
    ext: str
    mtime: int
    size: int


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

    def _extract_text_for_file(self, file_path: Path) -> Tuple[Path, str]:
        """Extract text from a single file (used for parallel processing)."""
        try:
            text = (self.extractor.extract(file_path) or "").strip()
            return (file_path, text)
        except Exception:
            return (file_path, "")

    def _extract_ocr_for_image(self, file_path: Path) -> Tuple[Path, str]:
        """Extract OCR text from a single image (used for parallel processing)."""
        try:
            text = (self.extractor.extract(file_path) or "").strip()
            return (file_path, text)
        except Exception:
            return (file_path, "")

    def index_folder(self, folder: str | Path, progress: Optional[ProgressCb] = None, recursive: bool = True, file_types: Optional[set[str]] = None) -> IndexStats:
        folder = str(Path(folder).resolve())
        stats = IndexStats()

        if progress:
            progress("Opening local index...")

        con = open_db(self.db_path)
        try:
            # BEGIN TRANSACTION - single transaction for entire indexing
            con.execute("BEGIN")
            
            # Collect files to process
            files_to_process: List[Tuple[Path, str, str, int, int]] = []  # path, rel, ext, mtime, size
            pending_images: List[_PendingImage] = []
            
            image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".tif"}

            # Phase 1: Scan files and check which need processing
            if progress:
                progress("Scanning files...")
                
            for file_path in self.iter_files(folder, recursive=recursive, file_types=file_types):
                stats.scanned += 1
                try:
                    rel = safe_relpath(file_path, folder)
                    st = file_path.stat()
                    mtime = int(st.st_mtime)
                    size = int(st.st_size)
                    ext = file_path.suffix.lower()

                    # Check if already indexed with same mtime/size
                    row = con.execute(
                        "SELECT mtime, size FROM files WHERE path = ?",
                        (str(file_path),),
                    ).fetchone()

                    if row and int(row[0]) == mtime and int(row[1]) == size:
                        stats.skipped += 1
                        continue

                    # Separate images from documents
                    if ext in image_exts:
                        pending_images.append(_PendingImage(file_path, rel, ext, mtime, size))
                    else:
                        files_to_process.append((file_path, rel, ext, mtime, size))

                except Exception:
                    stats.failed += 1
                    continue

            # Phase 2: Parallel text extraction for documents
            pending_docs: List[_PendingDoc] = []
            docs_without_text: List[_PendingDoc] = []
            
            if files_to_process:
                if progress:
                    progress(f"Extracting text from {len(files_to_process)} documents (parallel)...")
                
                # Use thread pool for parallel text extraction
                with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    futures = {
                        executor.submit(self._extract_text_for_file, fp): (fp, rel, ext, mtime, size)
                        for fp, rel, ext, mtime, size in files_to_process
                    }
                    
                    for future in concurrent.futures.as_completed(futures):
                        fp, rel, ext, mtime, size = futures[future]
                        try:
                            _, text = future.result()
                            doc = _PendingDoc(fp, rel, ext, mtime, size, text)
                            if text:
                                pending_docs.append(doc)
                            else:
                                docs_without_text.append(doc)
                        except Exception:
                            stats.failed += 1

            # Phase 3: Batch embed all document texts
            if pending_docs and self.embedder.available:
                if progress:
                    progress(f"Batch embedding {len(pending_docs)} documents...")
                
                texts = [doc.text for doc in pending_docs]
                embeddings = self.embedder.embed_texts_batch(texts)
                
                for doc, vec in zip(pending_docs, embeddings):
                    try:
                        vec_arr = np.asarray(vec, dtype=np.float32)
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
                            (folder, str(doc.file_path), doc.rel, doc.ext, doc.mtime, doc.size, doc.text, vec_arr.tobytes(), int(vec_arr.shape[0]), now_ts()),
                        )
                        stats.indexed += 1
                    except Exception:
                        stats.failed += 1

            # Phase 3: Insert documents without text (no embedding needed)
            for doc in docs_without_text:
                try:
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
                        (folder, str(doc.file_path), doc.rel, doc.ext, doc.mtime, doc.size, "", now_ts()),
                    )
                    stats.indexed += 1
                except Exception:
                    stats.failed += 1

            # Phase 4: Process images - batch CLIP embedding + parallel OCR
            if pending_images:
                if progress:
                    progress(f"Processing {len(pending_images)} images...")

                # Batch CLIP embeddings for all images (already efficient)
                clip_vecs: List[Optional[np.ndarray]] = []
                if self.clip.available:
                    try:
                        if progress:
                            progress(f"CLIP embedding {len(pending_images)} images...")
                        paths = [img.file_path for img in pending_images]
                        clip_vecs = self.clip.embed_images(paths, batch_size=IMAGE_BATCH_SIZE)
                    except Exception:
                        clip_vecs = []

                # Parallel OCR extraction for images
                image_texts: List[str] = [""] * len(pending_images)
                if progress:
                    progress(f"Parallel OCR on {len(pending_images)} images...")
                    
                with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    futures = {
                        executor.submit(self._extract_ocr_for_image, img.file_path): idx
                        for idx, img in enumerate(pending_images)
                    }
                    for future in concurrent.futures.as_completed(futures):
                        idx = futures[future]
                        try:
                            _, text = future.result()
                            image_texts[idx] = text
                        except Exception:
                            pass  # Keep empty string on failure

                # Batch embed image OCR texts (only non-empty ones)
                text_embeddings: List[Optional[np.ndarray]] = [None] * len(pending_images)
                if self.embedder.available:
                    non_empty_indices = [i for i, t in enumerate(image_texts) if t]
                    if non_empty_indices:
                        if progress:
                            progress(f"Embedding {len(non_empty_indices)} image OCR texts...")
                        non_empty_texts = [image_texts[i] for i in non_empty_indices]
                        batch_vecs = self.embedder.embed_texts_batch(non_empty_texts)
                        for idx, vec in zip(non_empty_indices, batch_vecs):
                            text_embeddings[idx] = vec

                # Insert all images into DB
                if progress:
                    progress(f"Saving {len(pending_images)} images to index...")
                    
                for idx, img in enumerate(pending_images):
                    try:
                        text = image_texts[idx]
                        
                        text_blob = None
                        text_dim = None
                        if text_embeddings[idx] is not None:
                            tv = np.asarray(text_embeddings[idx], dtype=np.float32)
                            if tv.ndim == 1 and tv.shape[0] > 1:
                                text_blob = tv.tobytes()
                                text_dim = int(tv.shape[0])

                        img_blob = None
                        img_dim = None
                        if clip_vecs and idx < len(clip_vecs) and clip_vecs[idx] is not None:
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
                            (folder, str(img.file_path), img.rel, img.ext, img.mtime, img.size, text, text_blob, text_dim, img_blob, img_dim, now_ts()),
                        )
                        stats.indexed += 1
                    except Exception:
                        stats.failed += 1
                        continue

            # COMMIT TRANSACTION
            con.commit()

        except Exception as e:
            # ROLLBACK on any error
            try:
                con.rollback()
            except Exception:
                pass
            raise e
        finally:
            con.close()

        if progress:
            progress(f"Done. Indexed: {stats.indexed}, skipped: {stats.skipped}, failed: {stats.failed}")
        return stats

