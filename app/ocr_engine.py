from __future__ import annotations

import threading
from typing import Iterable

import cv2
import numpy as np
from rapidocr import RapidOCR

from app.config import Settings
from app.models import OCRLine


class OCREngine:
    """Long-lived RapidOCR engines.

    Primary path:
      PP-OCRv6 small detector + configurable recognizer (medium by default).

    Retry path:
      PP-OCRv6 medium detector + medium recognizer with a larger input side.
      It is created lazily and only used for uncertain crops.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._primary = self._build_engine(
            det_model=settings.ocr_det_model,
            rec_model=settings.ocr_rec_model,
            limit_side_len=settings.ocr_limit_side_len,
        )
        self._retry: RapidOCR | None = None
        self._lock = threading.Lock()

    def _build_engine(self, det_model: str, rec_model: str, limit_side_len: int) -> RapidOCR:
        use_cuda = self.settings.ocr_device == "cuda"
        return RapidOCR(
            params={
                "Det.engine_type": "onnxruntime",
                "Rec.engine_type": "onnxruntime",
                "Cls.engine_type": "onnxruntime",
                "Det.model_type": det_model,
                "Rec.model_type": rec_model,
                "Det.ocr_version": "PP-OCRv6",
                "Rec.ocr_version": "PP-OCRv6",
                "Det.limit_side_len": limit_side_len,
                "Global.text_score": self.settings.ocr_text_score,
                "Global.return_word_box": True,
                "EngineConfig.onnxruntime.use_cuda": use_cuda,
                "EngineConfig.onnxruntime.cuda_ep_cfg.device_id": 0,
            }
        )

    def _get_retry_engine(self) -> RapidOCR:
        if self._retry is None:
            with self._lock:
                if self._retry is None:
                    self._retry = self._build_engine(
                        det_model="medium",
                        rec_model="medium",
                        limit_side_len=max(self.settings.ocr_limit_side_len, 1280),
                    )
        return self._retry

    @staticmethod
    def _to_lines(result) -> list[OCRLine]:
        if result is None or result.boxes is None:
            return []
        boxes = result.boxes.tolist() if hasattr(result.boxes, "tolist") else result.boxes
        txts = list(result.txts or [])
        scores = list(result.scores or [])
        return [
            OCRLine(text=str(text), confidence=float(score), box=[[float(x), float(y)] for x, y in box])
            for box, text, score in zip(boxes, txts, scores)
        ]

    def run(self, image: np.ndarray) -> tuple[list[OCRLine], float, object]:
        result = self._primary(image, return_word_box=True)
        return self._to_lines(result), float(result.elapse or 0.0), result

    def retry_crop(self, crop: np.ndarray) -> list[OCRLine]:
        if crop.size == 0:
            return []

        # Upscale only uncertain regions; this preserves normal-path speed.
        h, w = crop.shape[:2]
        if min(h, w) < 900:
            crop = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

        result = self._get_retry_engine()(crop, return_word_box=True)
        return self._to_lines(result)

    @staticmethod
    def crop_for_line(image: np.ndarray, line: OCRLine, padding: int = 12) -> tuple[np.ndarray, tuple[int, int]]:
        pts = np.asarray(line.box, dtype=np.float32)
        x1 = max(int(np.floor(pts[:, 0].min())) - padding, 0)
        y1 = max(int(np.floor(pts[:, 1].min())) - padding, 0)
        x2 = min(int(np.ceil(pts[:, 0].max())) + padding, image.shape[1])
        y2 = min(int(np.ceil(pts[:, 1].max())) + padding, image.shape[0])
        return image[y1:y2, x1:x2], (x1, y1)

    def improve_uncertain_lines(self, image: np.ndarray, lines: Iterable[OCRLine]) -> list[OCRLine]:
        if not self.settings.enable_targeted_retry:
            return list(lines)

        improved: list[OCRLine] = []
        for line in lines:
            if line.confidence >= self.settings.ocr_retry_score:
                improved.append(line)
                continue

            crop, (x0, y0) = self.crop_for_line(image, line)
            candidates = self.retry_crop(crop)
            if not candidates:
                improved.append(line)
                continue

            best = max(candidates, key=lambda item: item.confidence)
            if best.confidence <= line.confidence:
                improved.append(line)
                continue

            # Keep the original page-level geometry; retry is for text recognition quality.
            improved.append(OCRLine(text=best.text, confidence=best.confidence, box=line.box))

        return improved
