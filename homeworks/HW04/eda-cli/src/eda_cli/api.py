# src/eda_cli/api.py
from __future__ import annotations

import io
import time
import uuid
from typing import Any, Dict

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# импортируем ядро из вашего eda-cli (HW03)
from .core import compute_quality_flags, missing_table, summarize_dataset

app = FastAPI(title="eda-cli quality API", version="0.1")

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class QualityRequest(BaseModel):
    """
    Небольшая облегчённая модель запроса для /quality.
    (Семинарный шаблон использует похожую модель — это упрощённая версия.)
    """
    n_rows: int
    n_cols: int
    max_missing_share: float = 0.0
    min_missing_share: float = 0.1


@app.get("/health", response_model=HealthResponse)
def health() -> Dict[str, str]:
    """
    Простая проверка работоспособности сервиса.
    """
    return {"status": "ok", "service": "eda-cli-api", "version": "0.1"}


@app.post("/quality")
def quality(req: QualityRequest) -> Dict[str, Any]:
    """
    Упрощённый вычислитель качества по набору простых метрик (JSON).
    Этот endpoint полезен для быстрой проверки (на практике вы
    чаще будете использовать /quality-from-csv или /quality-flags-from-csv).
    """
    start = time.perf_counter()
    flags = {
        "too_few_rows": req.n_rows < 100,
        "too_many_columns": req.n_cols > 100,
        "max_missing_share": req.max_missing_share,
        "too_many_missing": req.max_missing_share > 0.5,
        # остальные флаги — по умолчанию False (т.к. нет full-summary)
        "has_constant_columns": False,
        "has_high_cardinality_categoricals": False,
        "has_suspicious_id_duplicates": False,
        "has_many_zero_values": False,
    }
    # простая эвристика ok_for_model
    quality_score = 1.0
    quality_score -= flags["max_missing_share"]
    if flags["too_few_rows"]:
        quality_score -= 0.15
    if flags["too_many_columns"]:
        quality_score -= 0.05
    quality_score = max(0.0, min(1.0, quality_score))
    latency_ms = (time.perf_counter() - start) * 1000.0
    ok_for_model = quality_score >= 0.5
    return {
        "ok_for_model": ok_for_model,
        "quality_score": round(quality_score, 4),
        "latency_ms": latency_ms,
        "flags": flags,
    }


@app.post("/quality-from-csv")
async def quality_from_csv(
    file: UploadFile = File(...),
    min_missing_share: float = Query(0.1, description="Порог доли пропусков для пометки проблемной колонки"),
):
    """
    Аналог семинарного /quality-from-csv: принимает CSV (multipart/form-data),
    читает его в DataFrame, вызывает summarize_dataset -> missing_table -> compute_quality_flags
    и возвращает качество + флаги + служебную информацию.
    """
    start = time.perf_counter()
    request_id = str(uuid.uuid4())
    # читаем CSV
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать CSV: {exc}")

    if df is None:
        raise HTTPException(status_code=400, detail="CSV не содержит данных")

    summary = summarize_dataset(df)
    missing = missing_table(df)
    flags = compute_quality_flags(summary, missing, min_missing_share, df=df)
    latency_ms = (time.perf_counter() - start) * 1000.0
    ok_for_model = flags.get("quality_score", 0.0) >= 0.5

    resp = {
        "request_id": request_id,
        "n_rows": summary.n_rows,
        "n_cols": summary.n_cols,
        "ok_for_model": ok_for_model,
        "quality_score": flags.get("quality_score"),
        "flags": flags,
        "latency_ms": latency_ms,
    }
    return JSONResponse(resp)


@app.post("/quality-flags-from-csv")
async def quality_flags_from_csv(
    file: UploadFile = File(...),
    min_missing_share: float = Query(0.1, description="Порог доли пропусков для пометки проблемной колонки"),
):
    """
    НОВЫЙ ЭНДПОИНТ (HW04, вариант A).
    Возвращает подробный набор флагов качества (включая эвристики из HW03).
    Формат ответа:
    {
      "flags": { ... },
      "quality_score": 0.72,
      "ok_for_model": true,
      "n_rows": 100,
      "n_cols": 12,
      "latency_ms": 12.3
    }
    """
    start = time.perf_counter()
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать CSV: {exc}")

    if df is None or df.shape[0] == 0:
        raise HTTPException(status_code=400, detail="CSV пуст или не содержит строк")

    summary = summarize_dataset(df)
    missing = missing_table(df)
    flags = compute_quality_flags(summary, missing, min_missing_share, df=df)
    latency_ms = (time.perf_counter() - start) * 1000.0
    ok_for_model = flags.get("quality_score", 0.0) >= 0.5

    return JSONResponse(
        {
            "flags": flags,
            "quality_score": flags.get("quality_score"),
            "ok_for_model": ok_for_model,
            "n_rows": summary.n_rows,
            "n_cols": summary.n_cols,
            "latency_ms": latency_ms,
        }
    )
