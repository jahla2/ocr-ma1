from __future__ import annotations

import re
from typing import Iterable

import numpy as np
from bs4 import BeautifulSoup
from rapid_table import ModelType, RapidTable, RapidTableInput

from app.config import Settings
from app.models import OCRLine, TableResult


class TableExtractor:
    """Fast table extraction with optional table-region detection.

    RapidTable uses the page OCR result whenever possible so a table crop does not
    require a second OCR pass.
    """

    TABLE_TERMS = {
        "item", "description", "qty", "quantity", "uom", "unit",
        "price", "amount", "total", "code", "rate", "tax", "gst",
    }

    def __init__(self, settings: Settings):
        self.settings = settings
        use_cuda = settings.ocr_device == "cuda"
        model_type = getattr(ModelType, "SLANETPLUS", ModelType.PPSTRUCTURE_EN)
        self._table = RapidTable(
            RapidTableInput(
                model_type=model_type,
                engine_cfg={
                    "use_cuda": use_cuda,
                    "cuda_ep_cfg.gpu_id": 0,
                },
            )
        )
        self._detector = self._build_optional_detector(use_cuda)

    def _build_optional_detector(self, use_cuda: bool):
        if not self.settings.enable_table_detector:
            return None
        try:
            from rapid_table_det.inference import TableDetector

            try:
                return TableDetector(
                    use_cuda=use_cuda,
                    use_edge_det=False,
                    use_cls_det=False,
                )
            except TypeError:
                return TableDetector()
        except Exception:
            return None

    @classmethod
    def _looks_tabular(cls, lines: Iterable[OCRLine]) -> bool:
        texts = [line.text.lower() for line in lines]
        term_hits = sum(any(term in text for term in cls.TABLE_TERMS) for text in texts)
        numeric_hits = sum(bool(re.search(r"\b\d+(?:[.,]\d{1,4})?\b", text)) for text in texts)
        return term_hits >= 3 or (term_hits >= 1 and numeric_hits >= 6)

    @staticmethod
    def _html_rows(html: str) -> list[list[str]]:
        if not html:
            return []
        soup = BeautifulSoup(html, "html.parser")
        rows: list[list[str]] = []
        for tr in soup.find_all("tr"):
            cells = [
                cell.get_text(" ", strip=True)
                for cell in tr.find_all(["th", "td"])
            ]
            if cells:
                rows.append(cells)
        return rows

    @staticmethod
    def _bbox_from_detection(item: dict, image: np.ndarray) -> tuple[int, int, int, int]:
        points = []
        for key in ("lt", "rt", "rb", "lb"):
            point = item.get(key)
            if point is not None and len(point) >= 2:
                points.append((float(point[0]), float(point[1])))

        if not points:
            box = item.get("box")
            if box is not None and len(box) >= 4:
                arr = np.asarray(box, dtype=float).reshape(-1, 2)
                points = [(float(x), float(y)) for x, y in arr]

        if not points:
            return 0, 0, image.shape[1], image.shape[0]

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        pad = 8
        x1 = max(int(min(xs)) - pad, 0)
        y1 = max(int(min(ys)) - pad, 0)
        x2 = min(int(max(xs)) + pad, image.shape[1])
        y2 = min(int(max(ys)) + pad, image.shape[0])
        return x1, y1, x2, y2

    @staticmethod
    def _ocr_for_crop(
        lines: list[OCRLine],
        bbox: tuple[int, int, int, int],
    ) -> tuple[np.ndarray, list[str], np.ndarray]:
        x1, y1, x2, y2 = bbox
        boxes: list[list[list[float]]] = []
        texts: list[str] = []
        scores: list[float] = []

        for line in lines:
            pts = np.asarray(line.box, dtype=np.float32)
            cx = float(pts[:, 0].mean())
            cy = float(pts[:, 1].mean())
            if not (x1 <= cx <= x2 and y1 <= cy <= y2):
                continue

            local = pts.copy()
            local[:, 0] -= x1
            local[:, 1] -= y1
            boxes.append(local.tolist())
            texts.append(line.text)
            scores.append(line.confidence)

        return (
            np.asarray(boxes, dtype=np.float32),
            texts,
            np.asarray(scores, dtype=np.float32),
        )

    def _recognize_crop(
        self,
        crop: np.ndarray,
        lines: list[OCRLine],
        bbox: tuple[int, int, int, int],
    ) -> TableResult | None:
        boxes, texts, scores = self._ocr_for_crop(lines, bbox)
        if len(texts) < 2:
            return None

        result = self._table(
            crop,
            ocr_results=[(boxes, texts, scores)],
        )
        htmls = getattr(result, "pred_htmls", None) or []
        html = htmls[0] if htmls else ""
        if not html:
            return None

        mean_conf = float(scores.mean()) if len(scores) else None
        return TableResult(
            bbox=list(bbox),
            html=html,
            rows=self._html_rows(html),
            confidence=mean_conf,
        )

    def extract(self, image: np.ndarray, lines: list[OCRLine]) -> list[TableResult]:
        if not self.settings.enable_tables or not self._looks_tabular(lines):
            return []

        regions: list[tuple[int, int, int, int]] = []
        if self._detector is not None:
            try:
                detections, _ = self._detector(image)
                regions = [
                    self._bbox_from_detection(item, image)
                    for item in (detections or [])
                ]
            except Exception:
                regions = []

        # Safe fallback: run structure recognition on the page only when OCR
        # geometry/text strongly suggests a table is present.
        if not regions:
            regions = [(0, 0, image.shape[1], image.shape[0])]

        tables: list[TableResult] = []
        for bbox in regions:
            x1, y1, x2, y2 = bbox
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image[y1:y2, x1:x2]
            table = self._recognize_crop(crop, lines, bbox)
            if table is not None:
                tables.append(table)
        return tables
