from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .db import open_db
from .embedder import ClipEmbedder, TextEmbedder


# ---------------------------------------------------------------------------
# Query Preprocessing: Stop Words and Synonyms
# ---------------------------------------------------------------------------
STOP_WORDS: Set[str] = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare', 'ought',
    'used', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
    'we', 'they', 'what', 'which', 'who', 'whom', 'where', 'when', 'why', 'how',
    'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some',
    'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
    'very', 'just', 'also', 'now', 'here', 'there', 'then', 'once', 'if',
    'image', 'picture', 'photo', 'file', 'document', 'find', 'show', 'get',
    'me', 'my', 'any', 'about', 'into', 'through', 'during', 'before', 'after',
}

# Synonym expansion dictionary for better recall
SYNONYMS: Dict[str, List[str]] = {
    # People
    'boy': ['child', 'kid', 'son', 'male', 'young'],
    'girl': ['child', 'kid', 'daughter', 'female', 'young'],
    'child': ['kid', 'boy', 'girl', 'young', 'children'],
    'kid': ['child', 'boy', 'girl', 'young'],
    'man': ['male', 'person', 'guy', 'adult'],
    'woman': ['female', 'person', 'lady', 'adult'],
    'person': ['human', 'people', 'individual'],
    'people': ['persons', 'humans', 'crowd', 'group'],
    
    # Animals
    'dog': ['puppy', 'canine', 'pet', 'hound'],
    'cat': ['kitten', 'feline', 'pet'],
    'bird': ['avian', 'fowl'],
    'car': ['vehicle', 'automobile', 'auto'],
    
    # Food
    'food': ['meal', 'dish', 'cuisine', 'eating'],
    'burger': ['hamburger', 'sandwich', 'food'],
    'pizza': ['food', 'meal'],
    
    # Nature
    'tree': ['plant', 'forest', 'nature'],
    'flower': ['plant', 'bloom', 'floral'],
    'water': ['ocean', 'sea', 'river', 'lake'],
    'sky': ['clouds', 'blue', 'heaven'],
    'mountain': ['hill', 'peak', 'nature'],
    'beach': ['shore', 'coast', 'sand', 'ocean'],
    
    # Objects
    'house': ['home', 'building', 'residence'],
    'building': ['structure', 'architecture', 'house'],
    'phone': ['mobile', 'cellphone', 'smartphone'],
    'computer': ['laptop', 'pc', 'desktop'],
    'book': ['document', 'text', 'reading'],
    
    # Actions/States
    'happy': ['joyful', 'smiling', 'cheerful'],
    'sad': ['unhappy', 'crying', 'depressed'],
    'beautiful': ['pretty', 'gorgeous', 'attractive'],
    'big': ['large', 'huge', 'giant'],
    'small': ['little', 'tiny', 'mini'],
    
    # Documents
    'invoice': ['bill', 'receipt', 'statement'],
    'report': ['document', 'summary', 'analysis'],
    'contract': ['agreement', 'document', 'legal'],
    'menu': ['list', 'options', 'food'],
    'receipt': ['invoice', 'bill', 'proof'],
}


def preprocess_query(query: str) -> Tuple[str, Set[str]]:
    """
    Preprocess query: remove stop words and expand with synonyms.
    Returns (cleaned_query, expanded_keywords).
    """
    query_lower = query.lower().strip()
    
    # Extract words
    words = re.findall(r'\b\w+\b', query_lower)
    
    # Remove stop words
    meaningful_words = [w for w in words if w not in STOP_WORDS and len(w) >= 2]
    
    # Expand with synonyms
    expanded_keywords: Set[str] = set(meaningful_words)
    for word in meaningful_words:
        if word in SYNONYMS:
            expanded_keywords.update(SYNONYMS[word])
    
    # Create cleaned query (original words only, for embedding)
    cleaned_query = ' '.join(meaningful_words) if meaningful_words else query_lower
    
    return cleaned_query, expanded_keywords


def _keyword_matches_text(keywords: Set[str], filename: str, extracted_text: str) -> bool:
    """Check if any query keyword appears in filename or extracted text (for documents only)."""
    if not keywords:
        return True
    
    filename_lower = filename.lower()
    text_lower = (extracted_text or "").lower()
    combined = filename_lower + " " + text_lower
    
    for kw in keywords:
        if kw in combined:
            return True
    
    return False


# ---------------------------------------------------------------------------
# Scoring Weights
# ---------------------------------------------------------------------------
TEXT_WEIGHT = 0.4      # Weight for text document matches
IMAGE_WEIGHT = 0.5     # Weight for CLIP image similarity
CAPTION_WEIGHT = 0.1   # Weight for image OCR/caption text matches


@dataclass
class SearchResult:
    path: str
    rel_path: str
    score: float
    source: str  # 'text' | 'image' | 'hybrid'


class LocalSearcher:
    """
    Hybrid search combining:
    - Text embeddings for documents
    - CLIP embeddings for visual-semantic image search
    - Weighted scoring for intelligent ranking
    """
    
    def __init__(self, db_path: str | Path, cache_dir: str | Path) -> None:
        self.db_path = Path(db_path)
        self.cache_dir = Path(cache_dir)
        self.embedder = TextEmbedder(cache_dir=self.cache_dir)
        self.clip = ClipEmbedder(cache_dir=self.cache_dir)
        
        # In-memory cache for repeated searches
        self._embedding_cache: Dict[str, np.ndarray] = {}

    def search(
        self,
        folder: str,
        query: str,
        limit: int = 10,
        min_score: Optional[float] = 0.15,
        file_types: Optional[set[str]] = None,
    ) -> List[SearchResult]:
        folder_norm = str(Path(folder).resolve())
        
        # Preprocess query: remove stop words, expand synonyms
        cleaned_query, expanded_keywords = preprocess_query(query)
        
        # Use cleaned query for embeddings (better semantic matching)
        embed_query = cleaned_query if cleaned_query else query
        
        # Thresholds
        text_min_score = float(min_score) if min_score else 0.15
        image_min_score = 0.20  # Slightly lower for images to allow visual matches

        # Compute text embedding
        q_text: Optional[np.ndarray] = None
        if self.embedder.available:
            q_text = np.asarray(self.embedder.embed_text(embed_query), dtype=np.float32)
            if q_text.ndim != 1 or q_text.shape[0] < 2:
                q_text = None

        # Compute CLIP embedding for image search
        q_img: Optional[np.ndarray] = None
        if self.clip.available:
            q_img = np.asarray(self.clip.embed_query(embed_query), dtype=np.float32)
            if q_img.ndim != 1 or q_img.shape[0] < 2:
                q_img = None

        if q_text is None and q_img is None:
            raise RuntimeError(
                "No embedding backend available for search. Install fastembed/onnxruntime."
            )

        # Fetch indexed files from DB
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
                SELECT path, rel_path, extracted_text, ext,
                       embedding, embedding_dim,
                       image_embedding, image_embedding_dim
                FROM files
                {where_clause}
                """,
                params,
            ).fetchall()
        finally:
            con.close()

        if not rows:
            return []

        # Categorize items
        doc_items: List[Tuple[str, str, str, np.ndarray]] = []  # path, rel, text, vec
        img_items: List[Tuple[str, str, str, np.ndarray, Optional[np.ndarray]]] = []  # path, rel, text, clip_vec, text_vec

        image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".tif"}
        doc_exts = {".pdf", ".docx", ".doc", ".txt", ".md"}

        for p, rel, extracted_text, ext, tblob, tdim, iblob, idim in rows:
            extracted_text = extracted_text or ""
            ext = (ext or "").lower()
            
            # Documents: use text embedding
            if ext in doc_exts and tblob is not None and q_text is not None:
                td = int(tdim) if tdim is not None else 0
                if td > 0:
                    tv = np.frombuffer(tblob, dtype=np.float32, count=td)
                    if tv.shape[0] == q_text.shape[0]:
                        doc_items.append((str(p), str(rel), extracted_text, tv))
            
            # Images: use CLIP embedding + optional text embedding for OCR
            if ext in image_exts and iblob is not None and q_img is not None:
                idv = int(idim) if idim is not None else 0
                if idv > 0:
                    iv = np.frombuffer(iblob, dtype=np.float32, count=idv)
                    if iv.shape[0] == q_img.shape[0]:
                        # Also get text embedding if OCR text exists
                        text_vec = None
                        if tblob is not None and q_text is not None and extracted_text:
                            td = int(tdim) if tdim is not None else 0
                            if td > 0:
                                tv = np.frombuffer(tblob, dtype=np.float32, count=td)
                                if tv.shape[0] == q_text.shape[0]:
                                    text_vec = tv
                        img_items.append((str(p), str(rel), extracted_text, iv, text_vec))

        scores_by_path: Dict[str, SearchResult] = {}

        # Score documents (text embedding + keyword validation)
        if doc_items and q_text is not None:
            mat = np.vstack([v for (_, _, _, v) in doc_items])
            scores = mat @ q_text
            
            for (p, rel, extracted_text, _), s in zip(doc_items, scores):
                raw_score = float(s)
                raw_score = max(0.0, min(1.0, raw_score))
                
                if raw_score < text_min_score:
                    continue
                
                # Keyword validation for documents (use expanded keywords)
                filename = Path(rel).name
                if not _keyword_matches_text(expanded_keywords, filename, extracted_text):
                    continue
                
                # Apply text weight
                weighted_score = raw_score * TEXT_WEIGHT
                
                prev = scores_by_path.get(p)
                if prev is None or weighted_score > prev.score:
                    scores_by_path[p] = SearchResult(
                        path=p, rel_path=rel, score=weighted_score, source="text"
                    )

        # Score images (CLIP visual + optional OCR text, NO keyword filtering)
        if img_items and q_img is not None:
            clip_mat = np.vstack([v for (_, _, _, v, _) in img_items])
            clip_scores = clip_mat @ q_img
            
            for idx, (p, rel, extracted_text, _, text_vec) in enumerate(img_items):
                clip_score = float(clip_scores[idx])
                clip_score = max(0.0, min(1.0, clip_score))
                
                # Calculate caption/OCR score if available
                caption_score = 0.0
                if text_vec is not None and q_text is not None:
                    caption_score = float(np.dot(text_vec, q_text))
                    caption_score = max(0.0, min(1.0, caption_score))
                
                # Combined weighted score for images
                # IMAGE_WEIGHT for visual, CAPTION_WEIGHT for OCR text
                weighted_score = (clip_score * IMAGE_WEIGHT) + (caption_score * CAPTION_WEIGHT)
                
                # Filter by minimum threshold (applied to visual score)
                if clip_score < image_min_score:
                    continue
                
                prev = scores_by_path.get(p)
                if prev is None or weighted_score > prev.score:
                    source = "hybrid" if caption_score > 0.1 else "image"
                    scores_by_path[p] = SearchResult(
                        path=p, rel_path=rel, score=weighted_score, source=source
                    )

        if not scores_by_path:
            return []

        # Sort by score and return top results
        merged = sorted(scores_by_path.values(), key=lambda r: r.score, reverse=True)
        return merged[: max(1, int(limit))]



