import os
import json
import httpx
import streamlit as st
from datetime import date
from dateutil.parser import parse as parse_dt

# URL вашего FastAPI-оркестратора
API_URL = os.getenv("ORCH_URL", "http://localhost:9001/predict")

st.set_page_config(page_title="Supply Risk Demo", layout="centered")
st.title("Supply Risk")

st.caption("Вводите маршрут, ETA и FX-волну — остальное подтянем автоматически.")

# === Простая карта назначения (дополняйте под себя) ===
DEST_LOOKUP = {
    "KZ-ALMATY":   {"name": "Almaty",    "lat": 43.2567, "lon": 76.9286},
    "KZ-PAVLODAR": {"name": "Pavlodar",  "lat": 52.2873, "lon": 76.9719},
    "CN-SHANGHAI": {"name": "Shanghai",  "lat": 31.2304, "lon": 121.4737},
    "KZ-ASTANA":   {"name": "Astana",    "lat": 51.1694, "lon": 71.4491},

    
    "KZ-KARAGANDA": {"name": "Karaganda", "lat": 49.8028, "lon": 73.0875},
    "KZ-SHYMKENT":  {"name": "Shymkent",  "lat": 42.3417, "lon": 69.5901},
    "KZ-AKTAY":     {"name": "Aktau",     "lat": 43.6481, "lon": 51.1722},

    "RU-OMSK":   {"name": "Omsk",   "lat": 54.9885, "lon": 73.3242},
    "RU-NOVOSIB": {"name": "Novosibirsk", "lat": 55.0084, "lon": 82.9357},
    "RU-CHELYAB": {"name": "Chelyabinsk", "lat": 55.1644, "lon": 61.4368},

    
    "CN-URUMQI":   {"name": "Urumqi",   "lat": 43.8256, "lon": 87.6168},
    "CN-KASHGAR":  {"name": "Kashgar",  "lat": 39.4704, "lon": 75.9898},
    "CN-LANZHOU":  {"name": "Lanzhou",  "lat": 36.0611, "lon": 103.8343},
}

def infer_destination(route_id: str):
    """
    Из 'CN-SHANGHAI -> KZ-ALMATY' достаём правую часть как ID назначения.
    """
    if not route_id:
        return None, None, None
    parts = [p.strip() for p in route_id.split("->")]
    dest_key = parts[-1].upper() if parts else None
    meta = DEST_LOOKUP.get(dest_key)
    if meta:
        return dest_key, meta["lat"], meta["lon"]
    return dest_key, None, None

with st.form("risk_form"):
    route_id = st.text_input("route_id", value="CN-SHANGHAI -> KZ-ALMATY")
    eta = st.date_input("eta_date", value=date.today())
    fx_vol = st.number_input("fx_volatility_7d", min_value=0.0, step=0.001, value=0.041, format="%.3f")

    # Авто-вывод координат по route_id
    dest_key, auto_lat, auto_lon = infer_destination(route_id)
    st.write(f"Назначение: **{dest_key or 'не распознано'}**")
    st.write(f"lat/lon из словаря: **{auto_lat if auto_lat is not None else '—'} / {auto_lon if auto_lon is not None else '—'}**")

    with st.expander("Если координаты не распознаны — укажите вручную"):
        lat = st.number_input("lat (manual)", value=float(auto_lat) if auto_lat is not None else 43.2567, format="%.6f")
        lon = st.number_input("lon (manual)", value=float(auto_lon) if auto_lon is not None else 76.928600, format="%.6f")
    submitted = st.form_submit_button("Сосчитать риск")

if submitted:
    # Готовим тело запроса к оркестратору
    payload = {
        "lat": float(lat if auto_lat is None else auto_lat),
        "lon": float(lon if auto_lon is None else auto_lon),
        "eta_date": eta.isoformat(),
        "route_id": route_id,
        "fx_volatility_7d": float(fx_vol),
        # road_load_index опционален — можно добавить поле в UI при необходимости
    }

    with st.spinner("Собираем погоду → фичи → скоринг → объяснение..."):
        try:
            resp = httpx.post(API_URL, json=payload, timeout=30.0)
            if resp.status_code != 200:
                st.error(f"Backend error {resp.status_code}: {resp.text[:300]}")
            else:
                data = resp.json()

                # Блок результатов
                st.success("Готово!")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Risk score", f"{data['ml_output']['risk_score']:.2f}")
                    st.write("**ETA**:", data["eta_date"])
                    st.write("**Route**:", data["route_id"])
                with col2:
                    ex = data.get("llm_explanation") or {}
                    st.metric("Risk level", ex.get("risk_level", "—"))
                    st.write("Confidence:", ex.get("confidence", "—"))

                with st.expander("Объяснение LLM (JSON)"):
                    st.code(json.dumps(data.get("llm_explanation", {}), ensure_ascii=False, indent=2), language="json")

                with st.expander("Фичи, переданные в /score (JSON)"):
                    st.code(json.dumps(data.get("features", {}), ensure_ascii=False, indent=2), language="json")

        except Exception as e:
            st.exception(e)
