from app.config import Settings
from app.models import OCRLine, TableResult
from app.validation import validate_page


def line(text: str, x: int, y: int, confidence: float = 0.999) -> OCRLine:
    return OCRLine(
        text=text,
        confidence=confidence,
        box=[[x, y], [x + 100, y], [x + 100, y + 12], [x, y + 12]],
    )


def test_financial_validation_accepts_consistent_invoice():
    lines = [
        line("Invoice Number:", 10, 10),
        line("46426897", 250, 10),
        line("Total Exclusive of GST$", 10, 100),
        line("37.57", 300, 100),
        line("GST$", 10, 120),
        line("5.63", 300, 120),
        line("Total Incl. GST$", 10, 140),
        line("43.20", 300, 140),
    ]
    # Typical invoices contain many clean lines; one bad tiny footer line should
    # not fail the page when critical numeric data and arithmetic are correct.
    lines.extend(line(f"clean text {i}", 10, 220 + i * 15) for i in range(20))
    lines.append(line("tiny legal footer", 10, 600, confidence=0.70))

    table = TableResult(
        bbox=[0, 20, 400, 90],
        html="<table></table>",
        rows=[
            ["517072", "VALVE GATE FEM 15MM BRASS", "EA", "1.00", "27.63"],
            ["509236", "NIPPLE HEX CROX BRASS BHN 15MM", "EA", "2.00", "6.42"],
            ["509158", "NUT CROX BRASS BCN 15MM", "EA", "2.00", "3.52"],
        ],
        confidence=0.999,
    )

    result = validate_page(lines, [table], Settings(ocr_device="cpu"))

    assert result.checks["subtotal_plus_tax_equals_total"] is True
    assert result.checks["table_rows_equal_subtotal"] is True
    assert result.status == "accept"


def test_low_confidence_numeric_value_requires_review():
    lines = [
        line("Total Incl. GST$", 10, 100),
        line("43.20", 300, 100, confidence=0.70),
    ]

    result = validate_page(lines, [], Settings(ocr_device="cpu"))

    assert result.checks["critical_numeric_confidence"] is False
    assert result.status == "review"
