# ocr-ma1

Fast local document extraction for invoices, receipts, PDFs, screenshots, and table-heavy documents.

## Pipeline

```text
PDF / image
   |
   v
PyMuPDF render (300 DPI by default)
   |
   v
RapidOCR PP-OCRv6
  - small detector
  - medium recognizer
  - ONNX Runtime CUDA
   |
   +--> targeted retry only for weak/critical numeric text
   |      - medium detector
   |      - medium recognizer
   |      - larger input side
   |
   v
Optional table region detector
   |
   v
RapidTable SLANet-plus
  - reuses page OCR boxes/text when a crop is available
   |
   v
Validation
  - OCR confidence
  - critical numeric confidence
  - subtotal + tax = total
  - table row sum = subtotal when detectable
   |
   v
Structured JSON
```

The design keeps the normal path fast and spends extra compute only on uncertain regions.

## Requirements

Recommended local setup:

- Windows 11
- Python 3.11
- NVIDIA RTX GPU
- Recent NVIDIA driver
- 8 GB VRAM is enough for this pipeline
- 16 GB system RAM or more

## Windows + NVIDIA setup

Clone the repository and use the implementation branch:

```powershell
git clone https://github.com/jahla2/ocr-ma1.git
cd ocr-ma1
git checkout feature/fast-document-pipeline
```

Create a Python 3.11 virtual environment:

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the project, including the optional table-region detector:

```powershell
pip install -e ".[table-det,dev]"
```

Some OCR/table packages may install the CPU ONNX Runtime build. Replace it with the NVIDIA GPU build:

```powershell
pip uninstall -y onnxruntime
pip install --upgrade "onnxruntime-gpu[cuda,cudnn]"
```

Create your local configuration:

```powershell
Copy-Item .env.example .env
```

Verify CUDA is visible to ONNX Runtime:

```powershell
python scripts/check_gpu.py
```

Expected output includes:

```text
CUDAExecutionProvider
```

If CUDA is not available, you can still test on CPU:

```powershell
pip uninstall -y onnxruntime-gpu
pip install onnxruntime
```

Then set this in `.env`:

```env
OCR_DEVICE=cpu
```

## Run the API

Start the service:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The first startup/request is slower because RapidOCR loads models lazily. Later requests are the useful performance measurement.

Health check:

```powershell
curl.exe http://127.0.0.1:8000/health
```

Extract a PDF from PowerShell:

```powershell
$file = "C:\path\to\INV-MICO (2).pdf"
curl.exe -X POST -F "file=@$file" http://127.0.0.1:8000/extract > result.json
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

## Benchmark your invoice

Run multiple warm measurements:

```powershell
python scripts/benchmark.py "C:\path\to\INV-MICO (2).pdf" --runs 3
```

Include the extracted JSON:

```powershell
python scripts/benchmark.py "C:\path\to\INV-MICO (2).pdf" --runs 3 --json
```

The benchmark reports:

- total document time
- OCR inference time per page
- validation score
- accept/review state
- detected table count

Do not compare only the first run because model initialization and model downloads can dominate it.

## Accuracy-focused defaults

The included `.env.example` starts with:

```env
OCR_DEVICE=cuda
OCR_DET_MODEL=small
OCR_REC_MODEL=medium
OCR_LIMIT_SIDE_LEN=960
OCR_TEXT_SCORE=0.50
OCR_RETRY_SCORE=0.95
OCR_HARD_FAIL_SCORE=0.85
PDF_DPI=300
ENABLE_TABLES=true
ENABLE_TABLE_DETECTOR=true
TABLE_MODEL=slanet_plus
ENABLE_TARGETED_RETRY=true
```

Why these defaults:

- **Small detection** keeps the expensive full-page pass light.
- **Medium recognition** improves text/number recognition without making detection heavy.
- **960 detector side** gives more small-text detail than the 736 default while staying practical.
- **300 DPI** is a good invoice/PDF baseline.
- **Targeted retry** runs the heavier medium/medium path only for very weak lines or uncertain text containing digits.
- **Low OCR threshold (0.50)** preserves uncertain detections so application-level validation can decide what needs review instead of silently dropping text.

## Speed tuning

For maximum speed:

```env
OCR_REC_MODEL=small
OCR_LIMIT_SIDE_LEN=736
PDF_DPI=250
ENABLE_TARGETED_RETRY=false
```

For higher small-text accuracy:

```env
OCR_REC_MODEL=medium
OCR_LIMIT_SIDE_LEN=1280
PDF_DPI=300
ENABLE_TARGETED_RETRY=true
```

Do not permanently increase DPI or input size until you benchmark your real document set. A targeted retry is usually cheaper than making every page expensive.

## Table extraction

The recommended install includes:

- `rapid_table` using SLANet-plus for table structure
- optional `rapid-table-det` for locating table regions first

When a table region is detected, the pipeline maps the already-recognized page OCR boxes into the table crop and sends those results into RapidTable. This avoids a full duplicate OCR pass.

If `rapid-table-det` is unavailable or fails to initialize, the pipeline falls back to whole-page RapidTable only when OCR content looks tabular.

## Validation strategy

A displayed OCR confidence of 99% is not the same as 99% real extraction accuracy.

The pipeline therefore checks:

1. overall OCR confidence
2. whether any critical numeric text remains at very low confidence
3. subtotal + tax = total when those fields can be located
4. table last-column sum = subtotal when a usable table is extracted

Very small low-confidence footer/legal text does not automatically fail the document if important numeric fields and financial checks pass.

The response includes:

```json
{
  "validation": {
    "score": 100.0,
    "status": "accept",
    "warnings": [],
    "checks": {
      "ocr_present": true,
      "mean_ocr_confidence": true,
      "critical_numeric_confidence": true,
      "subtotal_plus_tax_equals_total": true
    }
  }
}
```

## Run tests

```powershell
pytest
```

## Recommended evaluation before production

Create a benchmark set of at least 50-100 representative documents and measure:

- exact invoice-number accuracy
- exact date accuracy
- exact quantity accuracy
- exact unit-price accuracy
- exact total accuracy
- table cell accuracy
- average latency/page
- P95 latency/page
- GPU VRAM usage

For finance documents, exact numeric-field accuracy matters more than average OCR confidence.

## Upstream projects

- RapidOCR: https://github.com/RapidAI/RapidOCR
- RapidOCR docs: https://rapidai.github.io/RapidOCRDocs/
- RapidTable: https://github.com/RapidAI/RapidTable
- RapidTableDetection: https://github.com/RapidAI/RapidTableDetection
