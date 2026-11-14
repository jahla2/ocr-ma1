from __future__ import annotations

import time
from pathlib import Path

import cv2
import pymupdf
import numpy as np

from app.config import Settings
from app.models import PageResult
from app.ocr_engine import OCREngine
from app.table_engine import TableExtractor
from app.validation import validate_page


class DocumentPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.ocr = OCREngine(settings)
        self.tables = TableExtractor(settings)

    def warmup(self) -> None:
        blank = np.full((256, 768, 3), 255, dtype=np.uint8)
        cv2.putText(blank, "OCR warmup 123.45", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        self.ocr.run(blank)

    def _render_pdf(self, data: bytes):
        document = pymupdf.open(stream=data, filetype="pdf")
        scale = self.settings.pdf_dpi / 72.0
        matrix = pymupdf.Matrix(scale, scale)
        try:
            for index, page in enumerate(document):
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                if pix.n == 4:
                    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2RGB)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                yield index + 1, bgr
        finally:
            document.close()

    @staticmethod
    def _decode_image(data: bytes):
        arr = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Unsupported or corrupt image.")
        yield 1, image

    def extract_bytes(self, data: bytes, filename: str) -> dict:
        if len(data) > self.settings.max_file_mb * 1024 * 1024:
            raise ValueError(f"File exceeds MAX_FILE_MB={self.settings.max_file_mb}.")

        ext = Path(filename).suffix.lower()
        pages = self._render_pdf(data) if ext == ".pdf" else self._decode_image(data)

        started = time.perf_counter()
        page_results: list[PageResult] = []

        for page_number, image in pages:
            page_started = time.perf_counter()

            lines, ocr_seconds, _ = self.ocr.run(image)
            lines = self.ocr.improve_uncertain_lines(image, lines)

            tables = self.tables.extract(image, lines)
            validation = validate_page(lines, tables, self.settings)

            page_results.append(
                PageResult(
                    page=page_number,
                    width=image.shape[1],
                    height=image.shape[0],
                    duration_seconds=time.perf_counter() - page_started,
                    ocr_seconds=ocr_seconds,
                    lines=lines,
                    tables=tables,
                    validation=validation,
                )
            )

        total = time.perf_counter() - started
        return {
            "engine": {
                "ocr": f"RapidOCR PP-OCRv6 {self.settings.ocr_det_model} det / {self.settings.ocr_rec_model} rec",
                "device": self.settings.ocr_device,
                "table": "RapidTable SLANet-plus" if self.settings.enable_tables else None,
                "pdf_dpi": self.settings.pdf_dpi,
            },
            "file": filename,
            "total_seconds": round(total, 4),
            "pages": [page.to_dict() for page in page_results],
        }
