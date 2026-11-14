from __future__ import annotations

import re
from statistics import mean

from app.config import Settings
from app.models import OCRLine, TableResult, ValidationResult


_AMOUNT_RE = re.compile(r"(?<!\d)(?:[$€£₱]\s*)?(-?\d{1,3}(?:[ ,]\d{3})*(?:[.,]\d{2})|-?\d+[.,]\d{2})(?!\d)")


def _money(value: str) -> float | None:
    match = _AMOUNT_RE.search(value.replace("*", ""))
    if not match:
        return None
    raw = match.group(1).replace(" ", "").replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return None


def _line_center(line: OCRLine) -> tuple[float, float]:
    xs = [p[0] for p in line.box]
    ys = [p[1] for p in line.box]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _labeled_amount(
    lines: list[OCRLine],
    labels: tuple[str, ...],
    *,
    exclude: tuple[str, ...] = (),
) -> float | None:
    for label_line in lines:
        lowered = label_line.text.lower()
        if any(blocked in lowered for blocked in exclude):
            continue
        if not any(label in lowered for label in labels):
            continue

        direct = _money(label_line.text)
        if direct is not None:
            return direct

        lx, ly = _line_center(label_line)
        height = max(p[1] for p in label_line.box) - min(p[1] for p in label_line.box)
        tolerance = max(12.0, height * 1.2)
        candidates: list[tuple[float, float]] = []

        for line in lines:
            if line is label_line:
                continue
            x, y = _line_center(line)
            amount = _money(line.text)
            if amount is None:
                continue
            if x > lx and abs(y - ly) <= tolerance:
                candidates.append((x, amount))

        if candidates:
            return max(candidates, key=lambda pair: pair[0])[1]
    return None


def _table_last_column_sum(tables: list[TableResult]) -> float | None:
    values: list[float] = []
    for table in tables:
        for row in table.rows:
            if len(row) < 3:
                continue
            amount = _money(row[-1])
            if amount is None:
                continue
            joined = " ".join(row).lower()
            if any(word in joined for word in ("total", "subtotal", "gst", "tax", "vat")):
                continue
            values.append(amount)
    return round(sum(values), 2) if values else None


def validate_page(lines: list[OCRLine], tables: list[TableResult], settings: Settings) -> ValidationResult:
    warnings: list[str] = []
    checks: dict[str, bool] = {}

    confidences = [line.confidence for line in lines]
    avg_conf = mean(confidences) if confidences else 0.0
    hard_low = [line for line in lines if line.confidence < settings.ocr_hard_fail_score]
    hard_low_numeric = [line for line in hard_low if re.search(r"\d", line.text)]

    checks["ocr_present"] = bool(lines)
    checks["mean_ocr_confidence"] = avg_conf >= 0.98
    checks["critical_numeric_confidence"] = not hard_low_numeric
    checks["hard_low_confidence_ratio"] = (len(hard_low) / max(len(lines), 1)) <= 0.05

    if hard_low_numeric:
        warnings.append(
            f"{len(hard_low_numeric)} numeric OCR line(s) remain below "
            f"{settings.ocr_hard_fail_score:.2f} confidence."
        )
    elif len(hard_low) / max(len(lines), 1) > 0.05:
        warnings.append("Too much low-confidence text remains on the page.")

    subtotal = _labeled_amount(
        lines,
        ("total exclusive", "subtotal", "sub total", "total excl", "net total"),
    )
    tax = _labeled_amount(
        lines,
        ("gst$", "gst ", "tax", "vat"),
        exclude=("total", "incl", "grand"),
    )
    total = _labeled_amount(
        lines,
        ("total incl", "grand total", "amount due", "total due", "invoice total"),
    )

    if subtotal is not None and tax is not None and total is not None:
        checks["subtotal_plus_tax_equals_total"] = abs((subtotal + tax) - total) <= 0.02
        if not checks["subtotal_plus_tax_equals_total"]:
            warnings.append(
                f"Financial check failed: subtotal {subtotal:.2f} + tax {tax:.2f} != total {total:.2f}."
            )

    table_sum = _table_last_column_sum(tables)
    if subtotal is not None and table_sum is not None:
        checks["table_rows_equal_subtotal"] = abs(table_sum - subtotal) <= 0.02
        if not checks["table_rows_equal_subtotal"]:
            warnings.append(
                f"Table arithmetic needs review: line-item sum {table_sum:.2f} != subtotal {subtotal:.2f}."
            )

    if tables:
        checks["table_structure_present"] = any(bool(table.rows) for table in tables)

    score = 0.0 if not checks else 100.0 * sum(1 for ok in checks.values() if ok) / len(checks)
    status = "accept" if score >= 98.0 and not warnings else "review"
    return ValidationResult(score=round(score, 2), status=status, warnings=warnings, checks=checks)
