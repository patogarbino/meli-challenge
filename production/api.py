"""FastAPI API for the MeLi new/used classifier.

Endpoint POST /api/predict accepts a JSONL file upload
and returns predictions using the hexagonal architecture.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from production.adapters.model_repository import LocalModelRepository

# ── Config ─────────────────────────────────────────────────────────────────────

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models/"))

app = FastAPI(
    title="MeLi New/Used Classifier",
    description="Predice si un artículo de MercadoLibre es nuevo o usado",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy model loading ─────────────────────────────────────────────────────────

_model = None
_preprocessor = None


def _load_artifacts():
    global _model, _preprocessor
    if _model is None:
        repo = LocalModelRepository()
        _model, _preprocessor = repo.load(MODEL_DIR)


# ── Endpoints ──────────────────────────────────────────────────────────────────


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """Recibe un archivo .jsonlines y devuelve predicciones."""
    if not file.filename.endswith((".jsonlines", ".jsonl", ".json")):
        raise HTTPException(
            status_code=400,
            detail="El archivo debe ser .jsonlines, .jsonl o .json",
        )

    try:
        content = await file.read()
        lines = content.decode("utf-8").strip().split("\n")
        items = [json.loads(line) for line in lines if line.strip()]
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error leyendo el archivo: {e}"
        )

    if len(items) == 0:
        raise HTTPException(status_code=400, detail="El archivo está vacío")

    # Load model lazily
    _load_artifacts()

    try:
        X = _preprocessor.transform(items)
        predictions = _model.predict(X)
        probabilities = _model.predict_proba(X)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error en predicción: {e}"
        )

    results = []
    for i, item in enumerate(items):
        results.append(
            {
                "index": i,
                "prediction": str(predictions[i]),
                "prob_new": round(float(probabilities[i][1]), 4),
                "prob_used": round(float(probabilities[i][0]), 4),
                "title": item.get("title", ""),
            }
        )

    summary = pd.Series(predictions).value_counts().to_dict()

    return JSONResponse(
        {
            "total_items": len(results),
            "summary": {str(k): int(v) for k, v in summary.items()},
            "predictions": results,
        }
    )
