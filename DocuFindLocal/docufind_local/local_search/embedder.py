from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np


class TextEmbedder:
    """Embedding wrapper.

    Uses `fastembed` (ONNX) when available for better packaging vs torch.
    """

    def __init__(self, cache_dir: str | Path, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name

        self._fastembed = None
        self._init_error: Optional[str] = None

        os.environ.setdefault("FASTEMBED_CACHE_PATH", str(self.cache_dir))

        try:
            from fastembed import TextEmbedding  # type: ignore

            self._fastembed = TextEmbedding(model_name=model_name)
        except Exception as e:
            self._fastembed = None
            self._init_error = str(e)

    @property
    def available(self) -> bool:
        return self._fastembed is not None

    @property
    def init_error(self) -> Optional[str]:
        return self._init_error

    def embed_text(self, text: str) -> np.ndarray:
        text = (text or "").strip()
        if not text:
            return np.zeros((1,), dtype=np.float32)

        if self._fastembed is None:
            raise RuntimeError(f"No embedding backend available: {self._init_error}")

        vec = next(self._fastembed.embed([text]))
        arr = np.asarray(vec, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        if norm > 0:
            arr = arr / norm
        return arr.astype(np.float32)


class ClipEmbedder:
    """CLIP-like embedding wrapper for image search (fastembed ONNX models)."""

    def __init__(
        self,
        cache_dir: str | Path,
        vision_model: str = "Qdrant/clip-ViT-B-32-vision",
        text_model: str = "Qdrant/clip-ViT-B-32-text",
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("FASTEMBED_CACHE_PATH", str(self.cache_dir))

        self._image = None
        self._text = None
        self._init_error: Optional[str] = None

        try:
            from fastembed import ImageEmbedding, TextEmbedding  # type: ignore

            self._image = ImageEmbedding(model_name=vision_model)
            self._text = TextEmbedding(model_name=text_model)
        except Exception as e:
            self._image = None
            self._text = None
            self._init_error = str(e)

    @property
    def available(self) -> bool:
        return self._image is not None and self._text is not None

    @property
    def init_error(self) -> Optional[str]:
        return self._init_error

    def embed_query(self, text: str) -> np.ndarray:
        if not self.available:
            raise RuntimeError(f"CLIP embedder unavailable: {self._init_error}")
        text = (text or "").strip()
        if not text:
            return np.zeros((1,), dtype=np.float32)
        vec = next(self._text.embed([text]))
        arr = np.asarray(vec, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        if norm > 0:
            arr = arr / norm
        return arr.astype(np.float32)

    def embed_images(self, image_paths: list[str | Path], batch_size: int = 16) -> list[np.ndarray]:
        if not self.available:
            raise RuntimeError(f"CLIP embedder unavailable: {self._init_error}")
        out: list[np.ndarray] = []
        for vec in self._image.embed(image_paths, batch_size=batch_size):
            arr = np.asarray(vec, dtype=np.float32)
            norm = float(np.linalg.norm(arr))
            if norm > 0:
                arr = arr / norm
            out.append(arr.astype(np.float32))
        return out
