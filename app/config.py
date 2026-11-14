from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    ocr_device: Literal["cpu", "cuda"] = "cuda"
    ocr_det_model: Literal["tiny", "small", "medium"] = "small"
    ocr_rec_model: Literal["tiny", "small", "medium"] = "medium"
    ocr_limit_side_len: int = 960
    ocr_text_score: float = 0.50
    ocr_retry_score: float = 0.95
    ocr_hard_fail_score: float = 0.85

    pdf_dpi: int = 300
    enable_tables: bool = True
    enable_table_detector: bool = True
    table_model: str = "slanet_plus"
    table_det_confidence: float = 0.70

    enable_targeted_retry: bool = True
    max_file_mb: int = 30


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
