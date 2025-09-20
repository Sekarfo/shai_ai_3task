#!/usr/bin/env python3
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import json, joblib

ART = Path(r"C:\Users\trudk\OneDrive\Desktop\shai_ai_3task\artifacts")

# загрузка одного файла с моделью LGBM
_bundle = joblib.load(ART / "model_lgbm.pkl")
IMPUTER = _bundle["imputer"]
MODEL   = _bundle["model"]
NUM_COLS = _bundle["features"]  # список признаков из тренировки

# опционально читаем отчёт (не обязателен для работы API)
REPORT_PATH = ART / "report_lgbm.json"
REPORT = json.loads(REPORT_PATH.read_text(encoding="utf-8")) if REPORT_PATH.exists() else {}

app = FastAPI(title="SupplyRisk Regressor (0..100)", version="0.2.0")

class ScoreRequest(BaseModel):
    routes: List[Dict[str, Any]]

class ScoreItem(BaseModel):
    route_id: str = ""
    risk_score: float   # 0..100
    risk_level: str     

def _prep_df(payload: List[Dict[str, Any]]) -> pd.DataFrame:
    X = pd.DataFrame(payload)
    
    for c in NUM_COLS:
        if c not in X.columns:
            X[c] = np.nan
    # порядок столбцов и импьютация
    X = X[NUM_COLS]
    X_imp = IMPUTER.transform(X)
    return pd.DataFrame(X_imp, columns=NUM_COLS)

def _score_to_level(s: float) -> str:
    # временная логика (замени в своём бэке, если нужно другое)
    if s >= 60: return "red"
    if s >= 30: return "yellow"
    return "green"

@app.get("/health")
def health():
    return {
        "status": "ok",
        "features": NUM_COLS,
        "report_available": REPORT_PATH.exists(),
        "time": datetime.utcnow().isoformat()+"Z"
    }

@app.post("/score", response_model=List[ScoreItem])
def score(req: ScoreRequest):
    X = _prep_df(req.routes)
    preds = MODEL.predict(X)                # регрессия
    preds = np.clip(preds, 0.0, 100.0)      # приводим к шкале 0..100
    out: List[ScoreItem] = []
    for row, s in zip(req.routes, preds):
        lvl = _score_to_level(float(s))
        out.append(ScoreItem(
            route_id=str(row.get("route_id","")),
            risk_score=float(np.round(s, 3)),
            risk_level=lvl
        ))
    return out

class WhatIfRequest(BaseModel):
    features: Dict[str, Any]

@app.post("/whatif")
def whatif(req: WhatIfRequest):
    X = _prep_df([req.features])
    s = float(np.clip(MODEL.predict(X)[0], 0.0, 100.0))
    return {"risk_score": round(s,3), "risk_level": _score_to_level(s)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
