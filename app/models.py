from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class OCRLine:
    text: str
    confidence: float
    box: list[list[float]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TableResult:
    bbox: list[int]
    html: str
    rows: list[list[str]]
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ValidationResult:
    score: float
    status: str
    warnings: list[str] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PageResult:
    page: int
    width: int
    height: int
    duration_seconds: float
    ocr_seconds: float
    lines: list[OCRLine]
    tables: list[TableResult]
    validation: ValidationResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "page": self.page,
            "width": self.width,
            "height": self.height,
            "duration_seconds": round(self.duration_seconds, 4),
            "ocr_seconds": round(self.ocr_seconds, 4),
            "lines": [line.to_dict() for line in self.lines],
            "tables": [table.to_dict() for table in self.tables],
            "validation": self.validation.to_dict(),
        }
