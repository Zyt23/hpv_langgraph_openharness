from __future__ import annotations

import random
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import AppConfig
from .row_context import build_row_context, load_flight
from .schema import WindowRecord


def sample_flights(index_df: pd.DataFrame, aircraft_ids: List[str], per_class: int, seed: int) -> List[dict]:
    rng = random.Random(seed)
    rows = []
    for label in [0, 1]:
        sub = index_df[(index_df["aircraft_id"].isin(aircraft_ids)) & (index_df["folder_label"] == label)].to_dict(orient="records")
        rng.shuffle(sub)
        rows.extend(sub[:per_class])
    return rows


def profile_observation_schema(index_df: pd.DataFrame, aircraft_ids: List[str], cfg: AppConfig, round_no: int) -> dict:
    sampled = sample_flights(index_df, aircraft_ids, cfg.loop.observation_flights_per_class, cfg.split.seed + round_no)
    out = {"sampled_flights": [], "feature_profiles": {0: {}, 1: {}}}
    per_label_rows = {0: [], 1: []}
    for row in sampled:
        df = load_flight(row["flight_path"])
        ctx = build_row_context(df, row["faulty_side"])
        numeric_cols = [c for c in ctx.columns if pd.api.types.is_numeric_dtype(ctx[c])]
        prof = {"aircraft_id": row["aircraft_id"], "folder_label": int(row["folder_label"]), "flight_path": row["flight_path"], "n_rows": len(ctx), "columns": numeric_cols}
        out["sampled_flights"].append(prof)
        ctx = ctx[numeric_cols].copy()
        ctx["folder_label"] = int(row["folder_label"])
        per_label_rows[int(row["folder_label"])] += ctx.to_dict(orient="records")
    for label in [0, 1]:
        df = pd.DataFrame(per_label_rows[label])
        if df.empty:
            continue
        for col in [c for c in df.columns if c != "folder_label"]:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if s.empty:
                continue
            out["feature_profiles"][label][col] = {
                "min": float(s.min()),
                "p05": float(s.quantile(0.05)),
                "p50": float(s.quantile(0.50)),
                "p95": float(s.quantile(0.95)),
                "max": float(s.max()),
            }
    return out


def summarize_window_features(feature_df: pd.DataFrame, top_k: int = 12) -> dict:
    if feature_df.empty:
        return {"n": 0, "top_features": []}
    numeric_cols = [c for c in feature_df.columns if c not in {"aircraft_id", "folder_label", "flight_path", "faulty_side", "window_id", "start_idx", "end_idx"} and pd.api.types.is_numeric_dtype(feature_df[c])]
    normal = feature_df[feature_df["folder_label"] == 0]
    abnormal = feature_df[feature_df["folder_label"] == 1]
    rows = []
    for col in numeric_cols:
        a = pd.to_numeric(normal[col], errors="coerce").dropna().to_numpy()
        b = pd.to_numeric(abnormal[col], errors="coerce").dropna().to_numpy()
        if len(a) < 2 or len(b) < 2:
            continue
        mean0 = float(np.mean(a))
        mean1 = float(np.mean(b))
        pooled = float(np.sqrt((np.var(a) + np.var(b)) / 2.0)) + 1e-8
        effect = abs(mean1 - mean0) / pooled
        rows.append({"feature": col, "mean_normal": mean0, "mean_abnormal": mean1, "delta": mean1 - mean0, "effect_size": effect})
    rows = sorted(rows, key=lambda x: x["effect_size"], reverse=True)
    return {"n": int(len(feature_df)), "top_features": rows[:top_k]}
