#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import lightgbm as lgb

BASE_NUM_COLS = [
    "precipitation_sum_mm","wind_speed_10m_max_kmh","wind_gusts_10m_max_kmh",
    "temperature_2m_max_c","temperature_2m_min_c","fx_volatility_7d","road_load_index",
]
OPT_FLAG_COLS = ["precip_heavy_flag","wind_strong_flag","gusts_extreme_flag","freeze_flag","heat_flag"]
TARGET_COL = "risk_score"
DATE_HINTS = ["eta_date","date","day","time"]

def find_date_col(df):
    for c in df.columns:
        if any(h in c.lower() for h in DATE_HINTS):
            dt = pd.to_datetime(df[c], errors="coerce")
            if dt.notna().any():
                df[c] = dt
                return c
    return None

def prepare_df(df: pd.DataFrame):
    df = df.copy()
    date_col = find_date_col(df)

    if TARGET_COL not in df.columns:
        raise SystemExit(f"Нужен столбец '{TARGET_COL}' (0..100).")
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.dropna(subset=[TARGET_COL])

    num_cols = [c for c in BASE_NUM_COLS if c in df.columns] + [c for c in OPT_FLAG_COLS if c in df.columns]
    for c in BASE_NUM_COLS:
        if c not in df.columns:
            df[c] = np.nan
            num_cols.append(c)

    for c in num_cols + [TARGET_COL]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    keep = list(dict.fromkeys(num_cols + [TARGET_COL] + ([date_col] if date_col else [])))
    return df[keep], num_cols, TARGET_COL, date_col

def time_aware_split(df, ycol, dcol):
    if dcol:
        df = df.sort_values(dcol)
        cut = int(0.8 * len(df))
        tr, te = df.iloc[:cut], df.iloc[cut:]
        if len(tr)==0 or len(te)==0:
            tr, te = train_test_split(df, test_size=0.2, random_state=42)
    else:
        tr, te = train_test_split(df, test_size=0.2, random_state=42)
    return tr, te

def fit_and_eval(train_df, test_df, num_cols, ycol, outdir: Path):
    # работаем с DataFrame (сохраняем имена признаков)
    X_train_df = train_df[num_cols].copy()
    y_train = train_df[ycol].astype(float).copy()
    X_test_df  = test_df[num_cols].copy()
    y_test  = test_df[ycol].astype(float).copy()

    # вал-сплит для ранней остановки
    X_tr_df, X_val_df, y_tr, y_val = train_test_split(
        X_train_df, y_train, test_size=0.2, random_state=42
    )

    # Импьютация и возврат в DataFrame c теми же колонками
    imp = SimpleImputer(strategy="median")
    X_tr  = pd.DataFrame(imp.fit_transform(X_tr_df), columns=num_cols, index=X_tr_df.index)
    X_val = pd.DataFrame(imp.transform(X_val_df), columns=num_cols, index=X_val_df.index)
    X_all = pd.DataFrame(imp.fit_transform(X_train_df), columns=num_cols, index=X_train_df.index)
    X_tst = pd.DataFrame(imp.transform(X_test_df),  columns=num_cols, index=X_test_df.index)

    # LightGBM + ранняя остановка (устойчивые параметры)
    reg = lgb.LGBMRegressor(
        n_estimators=5000,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=-1,
        subsample=0.85,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        min_child_samples=20,
        random_state=42
    )
    reg.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)]
    )

    best_iter = getattr(reg, "best_iteration_", None)
    if not best_iter or best_iter <= 0:
        best_iter = 300  # защита от 0 итераций

    # финальное дообучение на всём train
    reg_final = lgb.LGBMRegressor(
        n_estimators=int(best_iter),
        learning_rate=0.03,
        num_leaves=63,
        max_depth=-1,
        subsample=0.85,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        min_child_samples=20,
        random_state=42
    )
    reg_final.fit(X_all, y_train)

    pred = reg_final.predict(X_tst)
    mae  = float(mean_absolute_error(y_test, pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))  # без параметра squared
    r2   = float(r2_score(y_test, pred))

    outdir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"imputer": imp, "model": reg_final, "features": num_cols}, outdir / "model_lgbm.pkl")

    fi = dict(zip(num_cols, reg_final.feature_importances_.tolist()))
    report = {
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "num_cols": num_cols,
        "model": "lightgbm",
        "best_iter": int(best_iter),
        "metrics": {"MAE": mae, "RMSE": rmse, "R2": r2},
        "feature_importance": fi
    }
    (outdir / "report_lgbm.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))

def main():
    # Указываем путь к CSV и папке артефактов вручную (ОСТАВЛЕНО БЕЗ ИЗМЕНЕНИЙ)
    csv_path = r"C:\Users\trudk\OneDrive\Desktop\shai_ai_3task\data\logistics_risk_dataset_v3.csv"
    outdir = Path(r"C:\Users\trudk\OneDrive\Desktop\shai_ai_3task\artifacts")

    df = pd.read_csv(csv_path)
    df, num_cols, target_col, date_col = prepare_df(df)
    train_df, test_df = time_aware_split(df, target_col, date_col)
    fit_and_eval(train_df, test_df, num_cols, target_col, outdir)

if __name__ == "__main__":
    main()
