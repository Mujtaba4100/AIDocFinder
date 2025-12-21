from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .db import open_db
from .embedder import ClipEmbedder, TextEmbedder


@dataclass
class SearchResult:
    path: str
    rel_path: str
    score: float
    source: str  # 'text' | 'image'


class LocalSearcher:
    def __init__(self, db_path: str | Path, cache_dir: str | Path) -> None:
        self.db_path = Path(db_path)
        self.cache_dir = Path(cache_dir)
        self.embedder = TextEmbedder(cache_dir=self.cache_dir)
        self.clip = ClipEmbedder(cache_dir=self.cache_dir)

    def search(
        self,
        folder: str,
        query: str,
        limit: int = 10,
        min_score: Optional[float] = 0.15,
    ) -> List[SearchResult]:
        """Search locally across BOTH text embeddings and image embeddings.

        - Text modality: query embedded with BGE (TextEmbedder) and compared to `embedding`.
        - Image modality: query embedded with CLIP text encoder and compared to `image_embedding`.

        Scores are normalized per-modality before merging.
        """

        folder_norm = str(Path(folder).resolve())

        q_text: Optional[np.ndarray] = None
        if self.embedder.available:
            q_text = np.asarray(self.embedder.embed_text(query), dtype=np.float32)
            if q_text.ndim != 1 or q_text.shape[0] < 2:
                q_text = None

        q_img: Optional[np.ndarray] = None
        if self.clip.available:
            q_img = np.asarray(self.clip.embed_query(query), dtype=np.float32)
            if q_img.ndim != 1 or q_img.shape[0] < 2:
                q_img = None

        if q_text is None and q_img is None:
            raise RuntimeError(
                "No embedding backend available for search. "
                "Install fastembed/onnxruntime for image search and fastembed or sentence-transformers for text search."
            )

        con = open_db(self.db_path)
        try:
            rows = con.execute(
                """
                SELECT path, rel_path,
                       embedding, embedding_dim,
                       image_embedding, image_embedding_dim
                FROM files
                WHERE folder = ?
                """,
                (folder_norm,),
            ).fetchall()
        finally:
            con.close()

        if not rows:
            return []

        # Collect vectors by modality
        text_items: List[Tuple[str, str, np.ndarray]] = []
        img_items: List[Tuple[str, str, np.ndarray]] = []

        for p, rel, tblob, tdim, iblob, idim in rows:
            if tblob is not None and q_text is not None:
                td = int(tdim) if tdim is not None else 0
                if td > 0:
                    tv = np.frombuffer(tblob, dtype=np.float32, count=td)
                    if tv.shape[0] == q_text.shape[0]:
                        text_items.append((str(p), str(rel), tv))

            if iblob is not None and q_img is not None:
                idv = int(idim) if idim is not None else 0
                if idv > 0:
                    iv = np.frombuffer(iblob, dtype=np.float32, count=idv)
                    if iv.shape[0] == q_img.shape[0]:
                        img_items.append((str(p), str(rel), iv))

        scores_by_path: Dict[str, SearchResult] = {}

        # Text scores
        if text_items and q_text is not None:
            mat = np.vstack([v for (_, _, v) in text_items])
            scores = mat @ q_text
            max_s = float(scores.max()) if scores.size else 1.0
            if max_s <= 0:
                max_s = 1.0
            for (p, rel, _), s in zip(text_items, scores):
                norm_s = float(s) / max_s
                if min_score is not None and norm_s < float(min_score):
                    continue
                prev = scores_by_path.get(p)
                if prev is None or norm_s > prev.score:
                    scores_by_path[p] = SearchResult(path=p, rel_path=rel, score=norm_s, source="text")

        # Image scores
        if img_items and q_img is not None:
            mat = np.vstack([v for (_, _, v) in img_items])
            scores = mat @ q_img
            max_s = float(scores.max()) if scores.size else 1.0
            if max_s <= 0:
                max_s = 1.0
            for (p, rel, _), s in zip(img_items, scores):
                norm_s = float(s) / max_s
                if min_score is not None and norm_s < float(min_score):
                    continue
                prev = scores_by_path.get(p)
                if prev is None or norm_s > prev.score:
                    scores_by_path[p] = SearchResult(path=p, rel_path=rel, score=norm_s, source="image")

        merged = sorted(scores_by_path.values(), key=lambda r: r.score, reverse=True)
        return merged[: max(1, int(limit))]
