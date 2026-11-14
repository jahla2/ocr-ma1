from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.config import get_settings
from app.pipeline import DocumentPipeline


settings = get_settings()
pipeline = DocumentPipeline(settings)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Warm models once at service startup. RapidOCR >=3.10 loads lazily.
    pipeline.warmup()
    yield


app = FastAPI(
    title="ocr-ma1",
    version="0.1.0",
    description="Fast local OCR + table extraction pipeline",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "ocr_device": settings.ocr_device,
        "ocr_det_model": settings.ocr_det_model,
        "ocr_rec_model": settings.ocr_rec_model,
        "tables": settings.enable_tables,
    }


@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    filename = file.filename or "document.bin"
    try:
        data = await file.read()
        return pipeline.extract_bytes(data, filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {exc}") from exc
