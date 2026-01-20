from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .db import open_db
from .embedder import ClipEmbedder, TextEmbedder

# Import detector for object matching
try:
    from .detector import normalize_query_objects, objects_match_query
    DETECTOR_AVAILABLE = True
except ImportError:
    DETECTOR_AVAILABLE = False
    def normalize_query_objects(query: str) -> List[str]:
        return []
    def objects_match_query(objects_str: str, query_objects: List[str]) -> float:
        return 0.0


# ---------------------------------------------------------------------------
# Query Preprocessing: Minimal stop word removal for semantic search
# ---------------------------------------------------------------------------
# Keep preprocessing minimal - let semantic embeddings handle understanding
STOP_WORDS: Set[str] = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare', 'ought',
    'used', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
    'image', 'picture', 'photo', 'file', 'document', 'find', 'show', 'get',
    'me', 'my', 'any',
}


def preprocess_query(query: str) -> str:
    """
    Minimal query preprocessing: remove stop words, keep semantic content.
    
    Unlike keyword-based search, semantic embeddings understand meaning,
    so we keep meaningful adjectives, verbs, and context words.
    
    Example: "an image of a boy playing with a dog" -> "boy playing dog"
    Example: "rooftop selfie smiling" -> "rooftop selfie smiling"
    """
    query_lower = query.lower().strip()
    
    # Extract words
    words = re.findall(r'\b\w+\b', query_lower)
    
    # Remove only true stop words, keep descriptive terms
    meaningful_words = [w for w in words if w not in STOP_WORDS and len(w) >= 2]
    
    # Return cleaned query (for embedding) or original if nothing left
    return ' '.join(meaningful_words) if meaningful_words else query_lower


def _extract_keywords(query: str) -> Set[str]:
    """Extract keywords from query for document text validation."""
    query_lower = query.lower().strip()
    words = re.findall(r'\b\w{3,}\b', query_lower)
    return {w for w in words if w not in STOP_WORDS}


def _keyword_matches_text(keywords: Set[str], filename: str, extracted_text: str) -> bool:
    """Check if any query keyword appears in filename or extracted text."""
    if not keywords:
        return True
    
    combined = (filename + " " + (extracted_text or "")).lower()
    return any(kw in combined for kw in keywords)


# ---------------------------------------------------------------------------
# Scoring Weights for Hybrid Search (Updated for object detection)
# ---------------------------------------------------------------------------
TEXT_WEIGHT = 0.4       # Weight for document text embedding matches
CAPTION_WEIGHT = 0.4    # Weight for BLIP caption embedding matches
OBJECTS_WEIGHT = 0.2    # Weight for YOLOv8 detected objects
# Note: For images, we combine CLIP visual + caption + objects


@dataclass
class SearchResult:
    path: str
    rel_path: str
    score: float
    source: str  # 'text' | 'image' | 'caption' | 'objects' | 'hybrid'
    caption: str = ""  # BLIP-generated caption for images
    objects: str = ""  # YOLOv8 detected objects


class LocalSearcher:
    """
    Intelligent hybrid search combining:
    - Text embeddings for document semantic search
    - CLIP embeddings for visual-semantic image understanding
    - BLIP caption embeddings for descriptive image search
    - YOLOv8 object detection embeddings for object-based search
    - Weighted scoring: text=0.4, caption=0.4, objects=0.2
    
    Handles natural language queries like:
    - "rooftop selfie smiling"
    - "boy playing with a dog"
    - "invoice from 2024"
    - "car in parking lot"
    """
    
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
        file_types: Optional[set[str]] = None,
    ) -> List[SearchResult]:
        """
        Perform hybrid semantic search across documents and images.
        
        Args:
            folder: Root folder to search within.
            query: Natural language query (e.g., "boy playing with dog").
            limit: Maximum number of results.
            min_score: Minimum similarity threshold.
            file_types: Optional set of extensions to filter by.
            
        Returns:
            List of SearchResult sorted by relevance score.
        """
        folder_norm = str(Path(folder).resolve())
        
        # Preprocess query for semantic embedding
        cleaned_query = preprocess_query(query)
        embed_query = cleaned_query if cleaned_query else query
        
        # Extract keywords for document text validation
        query_keywords = _extract_keywords(query)
        
        # Extract objects from query for direct matching boost
        query_objects = normalize_query_objects(query) if DETECTOR_AVAILABLE else []
        
        # Thresholds
        text_min_score = float(min_score) if min_score else 0.15
        image_min_score = 0.15  # Lower threshold for visual matches

        # Compute text embedding for document and caption search
        q_text: Optional[np.ndarray] = None
        if self.embedder.available:
            q_text = np.asarray(self.embedder.embed_text(embed_query), dtype=np.float32)
            if q_text.ndim != 1 or q_text.shape[0] < 2:
                q_text = None

        # Compute CLIP embedding for visual image search
        q_img: Optional[np.ndarray] = None
        if self.clip.available:
            q_img = np.asarray(self.clip.embed_query(embed_query), dtype=np.float32)
            if q_img.ndim != 1 or q_img.shape[0] < 2:
                q_img = None

        if q_text is None and q_img is None:
            raise RuntimeError(
                "No embedding backend available. Install fastembed/onnxruntime."
            )

        # Fetch indexed files from DB (including objects columns)
        con = open_db(self.db_path)
        try:
            where_clause = "WHERE folder = ?"
            params = [folder_norm]
            if file_types:
                placeholders = ','.join('?' * len(file_types))
                where_clause += f" AND ext IN ({placeholders})"
                params.extend(file_types)
            rows = con.execute(
                f"""
                SELECT path, rel_path, extracted_text, caption, objects, ext,
                       embedding, embedding_dim,
                       image_embedding, image_embedding_dim,
                       caption_embedding, caption_embedding_dim,
                       objects_embedding, objects_embedding_dim
                FROM files
                {where_clause}
                """,
                params,
            ).fetchall()
        finally:
            con.close()

        if not rows:
            return []

        # Categorize items by type
        doc_items: List[Tuple[str, str, str, np.ndarray]] = []  # path, rel, text, vec
        img_items: List[Tuple[str, str, str, str, str, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]] = []
        # img_items: path, rel, ocr_text, caption, objects, clip_vec, ocr_vec, caption_vec, objects_vec

        image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".tif"}
        doc_exts = {".pdf", ".docx", ".doc", ".txt", ".md", ".rtf"}

        for row in rows:
            p, rel, extracted_text, caption, objects, ext, tblob, tdim, iblob, idim, cblob, cdim, oblob, odim = row
            extracted_text = extracted_text or ""
            caption = caption or ""
            objects = objects or ""
            ext = (ext or "").lower()
            
            # Documents: use text embedding
            if ext in doc_exts and tblob is not None and q_text is not None:
                td = int(tdim) if tdim is not None else 0
                if td > 0:
                    tv = np.frombuffer(tblob, dtype=np.float32, count=td)
                    if tv.shape[0] == q_text.shape[0]:
                        doc_items.append((str(p), str(rel), extracted_text, tv))
            
            # Images: use CLIP + caption + objects + OCR embeddings
            if ext in image_exts and iblob is not None and q_img is not None:
                idv = int(idim) if idim is not None else 0
                if idv > 0:
                    clip_vec = np.frombuffer(iblob, dtype=np.float32, count=idv)
                    if clip_vec.shape[0] != q_img.shape[0]:
                        continue
                    
                    # OCR text embedding
                    ocr_vec = None
                    if tblob is not None and q_text is not None:
                        td = int(tdim) if tdim is not None else 0
                        if td > 0:
                            tv = np.frombuffer(tblob, dtype=np.float32, count=td)
                            if tv.shape[0] == q_text.shape[0]:
                                ocr_vec = tv
                    
                    # Caption embedding
                    caption_vec = None
                    if cblob is not None and q_text is not None:
                        cd = int(cdim) if cdim is not None else 0
                        if cd > 0:
                            cv = np.frombuffer(cblob, dtype=np.float32, count=cd)
                            if cv.shape[0] == q_text.shape[0]:
                                caption_vec = cv
                    
                    # Objects embedding
                    objects_vec = None
                    if oblob is not None and q_text is not None:
                        od = int(odim) if odim is not None else 0
                        if od > 0:
                            ov = np.frombuffer(oblob, dtype=np.float32, count=od)
                            if ov.shape[0] == q_text.shape[0]:
                                objects_vec = ov
                    
                    img_items.append((str(p), str(rel), extracted_text, caption, objects,
                                     clip_vec, ocr_vec, caption_vec, objects_vec))

        scores_by_path: Dict[str, SearchResult] = {}

        # Score documents using semantic text embedding
        if doc_items and q_text is not None:
            mat = np.vstack([v for (_, _, _, v) in doc_items])
            scores = mat @ q_text
            
            for (p, rel, extracted_text, _), s in zip(doc_items, scores):
                raw_score = float(s)
                raw_score = max(0.0, min(1.0, raw_score))
                
                if raw_score < text_min_score:
                    continue
                
                # Keyword validation for documents (additional filter)
                filename = Path(rel).name
                if not _keyword_matches_text(query_keywords, filename, extracted_text):
                    continue
                
                weighted_score = raw_score * TEXT_WEIGHT
                
                prev = scores_by_path.get(p)
                if prev is None or weighted_score > prev.score:
                    scores_by_path[p] = SearchResult(
                        path=p, rel_path=rel, score=weighted_score, source="text"
                    )

        # Score images using CLIP + caption + objects (hybrid semantic search)
        # Weights: caption=0.4, objects=0.2, CLIP provides base matching
        if img_items and q_img is not None:
            clip_mat = np.vstack([v for (_, _, _, _, _, v, _, _, _) in img_items])
            clip_scores = clip_mat @ q_img
            
            for idx, (p, rel, ocr_text, caption, objects, _, ocr_vec, caption_vec, objects_vec) in enumerate(img_items):
                clip_score = float(clip_scores[idx])
                clip_score = max(0.0, min(1.0, clip_score))
                
                # Caption embedding score (BLIP-generated captions)
                caption_score = 0.0
                if caption_vec is not None and q_text is not None:
                    caption_score = float(np.dot(caption_vec, q_text))
                    caption_score = max(0.0, min(1.0, caption_score))
                
                # Objects embedding score (YOLOv8 detected objects)
                objects_score = 0.0
                if objects_vec is not None and q_text is not None:
                    objects_score = float(np.dot(objects_vec, q_text))
                    objects_score = max(0.0, min(1.0, objects_score))
                
                # Direct object matching boost (if query contains object names)
                object_match_boost = 0.0
                if DETECTOR_AVAILABLE and objects and query_objects:
                    object_match_boost = objects_match_query(objects, query_objects)
                
                # OCR text score (text found in images)
                ocr_score = 0.0
                if ocr_vec is not None and q_text is not None:
                    ocr_score = float(np.dot(ocr_vec, q_text))
                    ocr_score = max(0.0, min(1.0, ocr_score))
                
                # Combined weighted score:
                # Caption = 0.4, Objects = 0.2, CLIP visual provides base
                # Use CLIP as tie-breaker/base score
                weighted_score = (
                    caption_score * CAPTION_WEIGHT +
                    objects_score * OBJECTS_WEIGHT +
                    clip_score * 0.3 +  # CLIP as base visual match
                    ocr_score * 0.05 +  # Small boost for OCR matches
                    object_match_boost * 0.1  # Bonus for exact object matches
                )
                
                # Minimum threshold: either good CLIP match or good caption/objects
                if clip_score < image_min_score and caption_score < 0.2 and objects_score < 0.2:
                    continue
                
                # Determine source type based on dominant score
                if object_match_boost > 0.3 or objects_score > max(caption_score, clip_score * 0.8):
                    source = "objects"
                elif caption_score > max(objects_score, clip_score * 0.8):
                    source = "caption"
                elif ocr_score > 0.3:
                    source = "hybrid"
                else:
                    source = "image"
                
                prev = scores_by_path.get(p)
                if prev is None or weighted_score > prev.score:
                    scores_by_path[p] = SearchResult(
                        path=p, rel_path=rel, score=weighted_score, 
                        source=source, caption=caption, objects=objects
                    )

        if not scores_by_path:
            return []

        # Sort by score and return top results
        merged = sorted(scores_by_path.values(), key=lambda r: r.score, reverse=True)
        return merged[: max(1, int(limit))]



