import os, json, re
from datetime import date
from typing import Optional, Dict, Any, List

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

load_dotenv()

WEATHER_API_BASE = os.getenv("WEATHER_API_BASE", "https://api.open-meteo.com/v1/forecast")
ML_SCORE_URL     = os.getenv("ML_SCORE_URL", "http://localhost:8000/score")
LLM_MODE         = os.getenv("LLM_MODE", "simple").lower()   # simple | openai
LLM_URL          = os.getenv("LLM_URL", "").rstrip("/")
LLM_API_KEY      = os.getenv("LLM_API_KEY", "")
LLM_MODEL        = os.getenv("LLM_MODEL", "gpt-4o-mini")
HTTP_TIMEOUT     = float(os.getenv("HTTP_TIMEOUT", "15"))

app = FastAPI(title="SupplyChain Risk Orchestrator", version="0.2.0")

# ---------- 1) СХЕМЫ ----------
class PredictIn(BaseModel):
    lat: float
    lon: float
    eta_date: date
    route_id: Optional[str] = None
    fx_volatility_7d: Optional[float] = None
    road_load_index: Optional[float] = None
    extra: Optional[Dict[str, Any]] = None

    @field_validator("lat")
    @classmethod
    def check_lat(cls, v): 
        if not (-90 <= v <= 90): raise ValueError("lat out of range")
        return v

    @field_validator("lon")
    @classmethod
    def check_lon(cls, v):
        if not (-180 <= v <= 180): raise ValueError("lon out of range")
        return v

class MlRouteScoreOut(BaseModel):
    route_id: str = ""
    risk_score: float

class OrchestratorOut(BaseModel):
    route_id: str
    eta_date: date
    features: Dict[str, Any]
    ml_output: MlRouteScoreOut
    ml_raw: Any  
    llm_explanation: Dict[str, Any]

# ---------- 2) Погода ----------
def openmeteo_url(lat: float, lon: float, day: date) -> str:
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join([
            "precipitation_sum",
            "temperature_2m_max",
            "temperature_2m_min",
            "wind_speed_10m_max",
            "wind_gusts_10m_max",
        ]),
        "start_date": day.isoformat(),
        "end_date": day.isoformat(),
        "timezone": "auto",
    }
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    return f"{WEATHER_API_BASE}?{qs}"

async def fetch_weather(lat: float, lon: float, day: date) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        r = await client.get(openmeteo_url(lat, lon, day))
        if r.status_code != 200:
            raise HTTPException(502, f"Weather API error: {r.status_code} {r.text[:200]}")
        return r.json()

def build_features(wx: Dict[str, Any], eta_date: date,
                   fx_volatility_7d: Optional[float],
                   road_load_index: Optional[float]) -> Dict[str, Any]:
    daily = wx.get("daily") or {}
    def first(name: str, default=None):
        lst = daily.get(name) or []
        return lst[0] if lst else default

    precipitation_sum_mm      = first("precipitation_sum", 0.0)
    temperature_2m_max_c      = first("temperature_2m_max", None)
    temperature_2m_min_c      = first("temperature_2m_min", None)
    wind_speed_10m_max_kmh    = first("wind_speed_10m_max", None)
    wind_gusts_10m_max_kmh    = first("wind_gusts_10m_max", None)

    precip_heavy_flag   = int((precipitation_sum_mm or 0) >= 10.0)
    wind_strong_flag    = int((wind_speed_10m_max_kmh or 0) >= 50.0)
    gusts_extreme_flag  = int((wind_gusts_10m_max_kmh or 0) >= 75.0)
    freeze_flag         = int((temperature_2m_min_c or 99) <= 0.0)
    heat_flag           = int((temperature_2m_max_c or -99) >= 32.0)

    return {
        "eta_date": eta_date.isoformat(),
        "wx_date": eta_date.isoformat(),
        "precipitation_sum_mm": precipitation_sum_mm,
        "wind_speed_10m_max_kmh": wind_speed_10m_max_kmh,
        "wind_gusts_10m_max_kmh": wind_gusts_10m_max_kmh,
        "temperature_2m_max_c": temperature_2m_max_c,
        "temperature_2m_min_c": temperature_2m_min_c,
        "fx_volatility_7d": fx_volatility_7d or 0.0,
        "road_load_index": road_load_index or 0.0,
        "precip_heavy_flag": precip_heavy_flag,
        "wind_strong_flag": wind_strong_flag,
        "gusts_extreme_flag": gusts_extreme_flag,
        "freeze_flag": freeze_flag,
        "heat_flag": heat_flag,
    }

# ---------- 3) Скорер ----------
async def score_ml(route_id: str, features: Dict[str, Any]) -> (MlRouteScoreOut, Any):
    payload = {"routes": [{**features, "route_id": route_id}]}
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        r = await client.post(ML_SCORE_URL, json=payload)
        if r.status_code != 200:
            raise HTTPException(502, f"ML score error: {r.status_code} {r.text[:200]}")
        data = r.json()

    try:
        item = _pick_first_route(data)
        rs = float(item.get("risk_score"))
        rid = item.get("route_id", route_id)
        # вернём и нормализованную структуру, и сырой ответ
        return MlRouteScoreOut(route_id=rid, risk_score=rs), data
    except Exception as e:
        raise HTTPException(502, f"Unexpected ML response: {data}") from e

def _pick_first_route(obj: Any) -> Dict[str, Any]:
    """
    Поддерживаем варианты:
      A) {"routes":[{...}]}
      B) [{...}]
      C) {"route_id": "...", ...}
    Возвращаем первый словарь с route данными.
    """
    if isinstance(obj, dict) and "routes" in obj and isinstance(obj["routes"], list) and obj["routes"]:
        return obj["routes"][0]
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        return obj[0]
    if isinstance(obj, dict) and "route_id" in obj:
        return obj
    raise ValueError(f"Cannot find route item in ML response: {obj}")



# ---------- 4) LLM ----------
import json, httpx
from typing import Any, Dict

# Ожидается, что эти переменные уже загружены выше из .env:
# LLM_URL, LLM_MODE, LLM_MODEL, LLM_API_KEY, HTTP_TIMEOUT

# -------- JSON extractor без regex --------
def extract_json(text: str) -> Dict[str, Any]:
    """
    1) Попробовать распарсить весь текст как JSON.
    2) Найти первый сбалансированный JSON-объект { ... } стеком
       (учитывая кавычки и экранирование) и распарсить его.
    3) Вернуть summary, если ничего не вышло.
    """
    if not isinstance(text, str):
        try:
            return json.loads(json.dumps(text, ensure_ascii=False))
        except Exception:
            return {"summary": str(text)[:800]}

    # Попытка №1: весь текст — JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # Попытка №2: извлечь первый балансный объект { ... }
    start = text.find("{")
    while start != -1:
        i = start
        depth = 0
        in_str = False
        esc = False
        for j in range(start, len(text)):
            ch = text[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : j + 1]
                        try:
                            return json.loads(candidate)
                        except Exception:
                            break  # ищем следующий старт
        # не получилось — ищем следующий '{'
        start = text.find("{", start + 1)

    return {"summary": text.strip()[:800]}

# -------- утилиты URL/headers --------
def _base_url() -> str:
    url = (LLM_URL or "").rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    return url

def _safe_headers(h: dict) -> dict:
    out = {}
    for k, v in (h or {}).items():
        ks, vs = str(k), "" if v is None else str(v)
        if ks.isascii() and vs.isascii():
            out[ks] = vs
    return out

def _build_prompt(route_id: str, features: dict, ml_out, ml_raw) -> str:
    return (
        "Ты — логистический ассистент. На основе входных чисел и ответа ML дай краткое объяснение риска.\n"
        "Ответ строго JSON:\n"
        '{ "risk_level":"низкий|средний|высокий",'
        '  "summary":"1–3 предложения",'
        '  "drivers":["фактор1","фактор2"],'
        '  "recommendations":["шаг1","шаг2","шаг3"],'
        '  "confidence":0..1 }\n'
        "Не выдумывай факты вне переданных значений. Если ML уже вернул risk_level — учти.\n\n"
        f"route_id: {route_id}\n"
        f"features: {json.dumps(features, ensure_ascii=False)}\n"
        f"ml_output_normalized: {ml_out.model_dump_json()}\n"
        f"ml_output_raw: {json.dumps(ml_raw, ensure_ascii=False)}\n"
    )

# -------- вызов /v1/chat/completions --------
async def _call_openai_chat(prompt: str) -> Dict[str, Any]:
    url = f"{_base_url()}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"
    headers = _safe_headers(headers)

    body = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "Отвечай строго валидным JSON-ом по заданной схеме."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 800,
        # если сервер не поддерживает response_format — он просто проигнорирует
        "response_format": {"type": "json_object"},
    }

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        r = await client.post(url, headers=headers, json=body)
        if r.status_code == 404:
            raise FileNotFoundError("chat.completions not found")
        if r.status_code != 200:
            raise httpx.HTTPStatusError(f"{r.status_code} {r.text[:200]}", request=None, response=r)
        try:
            data = r.json()
            text = data["choices"][0]["message"]["content"]
            return extract_json(text)
        except Exception:
            return extract_json(r.text)

# -------- вызов /v1/responses --------
def _parse_responses_api_payload(d: Dict[str, Any]) -> str:
    # попытка достать текст из разных форматов ответа
    try:
        # vLLM-style: output -> [ { content: [ {text: "..."} ] } ]
        out = d.get("output")
        if isinstance(out, list) and out:
            content = out[0].get("content", [])
            for part in content:
                if isinstance(part, dict) and ("text" in part or "output_text" in part):
                    return part.get("text") or part.get("output_text") or ""
    except Exception:
        pass
    try:
        # OpenAI-like
        return d["choices"][0]["message"]["content"]
    except Exception:
        pass
    return json.dumps(d, ensure_ascii=False)

async def _call_openai_responses(prompt: str) -> Dict[str, Any]:
    url = f"{_base_url()}/v1/responses"
    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"
    headers = _safe_headers(headers)

    body = {
        "model": LLM_MODEL,
        "input": [
            {"role": "system", "content": "Отвечай строго валидным JSON-ом по заданной схеме."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_output_tokens": 800,
        "response_format": {"type": "json_object"},
    }

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        r = await client.post(url, headers=headers, json=body)
        if r.status_code != 200:
            raise httpx.HTTPStatusError(f"{r.status_code} {r.text[:200]}", request=None, response=r)
        try:
            data = r.json()
            text = _parse_responses_api_payload(data)
            return extract_json(text)
        except Exception:
            return extract_json(r.text)

# -------- публичные функции --------
async def call_llm_openai(prompt: str) -> Dict[str, Any]:
    """
    1) пробуем /v1/chat/completions,
    2) при 404 — /v1/responses,
    3) при любой другой ошибке тоже пытаемся /v1/responses.
    """
    if not LLM_URL:
        return {"summary": "LLM_URL не задан.", "risk_level": "средний", "confidence": 0.5}

    try:
        return await _call_openai_chat(prompt)
    except FileNotFoundError:
        return await _call_openai_responses(prompt)
    except httpx.HTTPStatusError:
        try:
            return await _call_openai_responses(prompt)
        except Exception as e:
            raise HTTPException(502, f"LLM error: {e}")

async def call_llm_simple(prompt: str) -> Dict[str, Any]:
    """
    Резерв на случай собственного /generate (не используется при LLM_MODE=openai).
    """
    if not LLM_URL:
        return {"summary": "LLM_URL не задан.", "risk_level": "средний", "confidence": 0.5}
    url = f"{_base_url()}/generate"
    headers = _safe_headers({"Content-Type": "application/json",
                             **({"Authorization": f"Bearer {LLM_API_KEY}"} if LLM_API_KEY else {})})
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        r = await client.post(url, headers=headers, json={"prompt": prompt})
        if r.status_code != 200:
            raise HTTPException(502, f"LLM error: {r.status_code} {r.text[:200]}")
        try:
            data = r.json()
            if isinstance(data, dict) and isinstance(data.get("text"), str):
                return extract_json(data["text"])
            return data if isinstance(data, dict) else {"summary": str(data)[:800]}
        except Exception:
            return extract_json(r.text)

async def explain_with_llm(route_id: str, features: Dict[str, Any], ml_out, ml_raw: Any) -> Dict[str, Any]:
    prompt = _build_prompt(route_id, features, ml_out, ml_raw)
    mode = (LLM_MODE or "openai").lower()
    if mode == "openai":
        return await call_llm_openai(prompt)
    else:
        return await call_llm_simple(prompt)

# ---------- 5) Оркестратор ----------
@app.post("/predict", response_model=OrchestratorOut)
async def predict(payload: PredictIn):
    route_id = payload.route_id or f"{payload.lat:.4f},{payload.lon:.4f}@{payload.eta_date.isoformat()}"

    wx    = await fetch_weather(payload.lat, payload.lon, payload.eta_date)
    feats = build_features(wx, payload.eta_date, payload.fx_volatility_7d, payload.road_load_index)
    ml_out, ml_raw = await score_ml(route_id, feats)       # <— теперь получаем два значения
    llm    = await explain_with_llm(route_id, feats, ml_out, ml_raw)

    return OrchestratorOut(
        route_id=route_id,
        eta_date=payload.eta_date,
        features=feats,
        ml_output=ml_out,
        ml_raw=ml_raw,                                     # <— отдадим клиенту для отладки
        llm_explanation=llm,
    )


@app.get("/health")
def health(): return {"ok": True}
