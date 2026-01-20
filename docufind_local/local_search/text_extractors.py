from __future__ import annotations

from pathlib import Path

from .constants import SUPPORTED_DOC_EXTS, SUPPORTED_IMAGE_EXTS, SUPPORTED_TEXT_EXTS
from .ocr import OcrEngine


class TextExtractor:
    def __init__(self, ocr: OcrEngine) -> None:
        self.ocr = ocr

    def extract(self, file_path: str | Path) -> str:
        p = Path(file_path)
        ext = p.suffix.lower()

        if ext in SUPPORTED_TEXT_EXTS:
            try:
                return p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                return ""

        if ext == ".docx":
            return self._extract_docx(p)

        if ext == ".pdf":
            return self._extract_pdf(p)

        if ext in SUPPORTED_IMAGE_EXTS:
            return self.ocr.image_to_text(p)

        return ""

    def _extract_docx(self, path: Path) -> str:
        try:
            import docx  # python-docx

            document = docx.Document(str(path))
            parts = [p.text for p in document.paragraphs if p.text]
            return "\n".join(parts).strip()
        except Exception:
            return ""

    def _extract_pdf(self, path: Path) -> str:
        text = self._extract_pdf_text(path)
        # If we have meaningful text (>100 chars), skip OCR entirely
        if text and len(text.strip()) > 100:
            return text

        try:
            import fitz  # PyMuPDF

            if not self.ocr.available:
                return text

            doc = fitz.open(str(path))
            lines: list[str] = []
            # Limit OCR to max 2 pages for speed
            max_pages = min(2, doc.page_count)
            for i in range(max_pages):
                page = doc.load_page(i)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_bytes = pix.tobytes("png")
                try:
                    from PIL import Image
                    import io
                    import tempfile

                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tf:
                        img.save(tf.name)
                        lines.append(self.ocr.image_to_text(tf.name))
                except Exception:
                    continue
            ocr_text = "\n".join([x for x in lines if x]).strip()
            return ocr_text or text
        except Exception:
            return text

    def _extract_pdf_text(self, path: Path) -> str:
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(str(path))
            parts = []
            for page in doc:
                parts.append(page.get_text("text"))
            return "\n".join(parts).strip()
        except Exception:
            pass

        try:
            from pypdf import PdfReader  # type: ignore

            reader = PdfReader(str(path))
            parts = []
            for page in reader.pages:
                parts.append(page.extract_text() or "")
            return "\n".join(parts).strip()
        except Exception:
            return ""
