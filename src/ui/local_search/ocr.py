from __future__ import annotations

from pathlib import Path
from typing import Optional


class OcrEngine:
    """OCR wrapper.

    We prefer ONNX-based OCR engines for easier PyInstaller bundling.
    If OCR dependencies are missing, we gracefully return an empty string.
    """

    def __init__(self) -> None:
        self._engine = None
        self._init_error: Optional[str] = None

        try:
            # RapidOCR (ONNX Runtime) is generally easier to bundle than Tesseract.
            from rapidocr_onnxruntime import RapidOCR  # type: ignore

            self._engine = RapidOCR()
        except Exception as e:
            self._init_error = str(e)
            self._engine = None

    @property
    def available(self) -> bool:
        return self._engine is not None

    @property
    def init_error(self) -> Optional[str]:
        return self._init_error

    def image_to_text(self, image_path: str | Path) -> str:
        if not self._engine:
            return ""
        try:
            # RapidOCR returns (result, elapsed)
            result, _ = self._engine(str(image_path))
            if not result:
                return ""
            # result: list of [box, text, score]
            lines = []
            for item in result:
                if not item or len(item) < 2:
                    continue
                text = item[1]
                if text:
                    lines.append(str(text))
            return "\n".join(lines).strip()
        except Exception:
            return ""
