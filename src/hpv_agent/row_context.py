from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .expressions import safe_eval
from .schema import ConditionSpec, WindowRecord

EPS = 1e-8


def load_flight(path: str | Path) -> pd.DataFrame:
    p = Path(path).resolve()
    if str(p).startswith("\\\\?\\"):
        long_path = str(p)
    elif p.drive:
        long_path = "\\\\?\\" + str(p)
    else:
        long_path = str(p)
    try:
        return pd.read_parquet(long_path)
    except Exception:
        return pd.read_parquet(str(p))


def _first_existing(df: pd.DataFrame, candidates: List[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def resolve_column_mapping(df: pd.DataFrame, faulty_side: str) -> Dict[str, str | None]:
    side = 1 if faulty_side == "left" else 2
    other = 2 if side == 1 else 1
    mapping = {
        "affected_n2": f"N2{side}",
        "healthy_n2": f"N2{other}",
        "affected_pressure": _first_existing(df, [f"BMPS{side}", f"PUD{side}"]),
        "healthy_pressure": _first_existing(df, [f"BMPS{other}", f"PUD{other}"]),
        "affected_precool": f"PRECOOL_PRESS{side}",
        "healthy_precool": f"PRECOOL_PRESS{other}",
        "affected_hpv": f"HPV_ENG{side}_R",
        "healthy_hpv": f"HPV_ENG{other}_R",
        "phase": "FLIGHT_PHASE",
        "lat": "LATP",
        "lon": "LONP",
        "altitude": "ALT_STD",
        "time_group": "time_group",
        "tail_num": "TAIL_NUM",
    }
    for k, v in list(mapping.items()):
        if v is not None and v not in df.columns:
            mapping[k] = None
    return mapping


def build_row_context(df: pd.DataFrame, faulty_side: str) -> pd.DataFrame:
    mapping = resolve_column_mapping(df, faulty_side)
    out = pd.DataFrame(index=df.index)
    for alias, col in mapping.items():
        if col and col in df.columns:
            if alias == "tail_num":
                out[alias] = df[col].astype(str)
            else:
                out[alias] = pd.to_numeric(df[col], errors="coerce")
        else:
            out[alias] = np.nan
    out["pressure_diff"] = out["affected_pressure"] - out["healthy_pressure"]
    out["pressure_abs_diff"] = out["pressure_diff"].abs()
    out["precool_diff"] = out["affected_precool"] - out["healthy_precool"]
    out["precool_abs_diff"] = out["precool_diff"].abs()
    out["hpv_diff"] = out["affected_hpv"] - out["healthy_hpv"]
    out["hpv_abs_diff"] = out["hpv_diff"].abs()
    return out


def condition_mask(row_ctx: pd.DataFrame, spec: ConditionSpec) -> pd.Series:
    if not spec.expression.strip():
        return pd.Series(True, index=row_ctx.index)
    env = {col: pd.to_numeric(row_ctx[col], errors="coerce") if col != "tail_num" else row_ctx[col] for col in row_ctx.columns}
    raw = safe_eval(spec.expression, env)
    if isinstance(raw, pd.Series):
        mask = raw.fillna(False).astype(bool)
    else:
        mask = pd.Series(bool(raw), index=row_ctx.index)
    return mask


def extract_segments(row_ctx: pd.DataFrame, spec: ConditionSpec, default_min_len: int) -> List[tuple[int, int]]:
    mask = condition_mask(row_ctx, spec)
    arr = mask.to_numpy(dtype=bool)
    min_len = spec.min_segment_len or default_min_len
    segments: List[tuple[int, int]] = []
    start = None
    for i, flag in enumerate(arr):
        if flag and start is None:
            start = i
        elif (not flag) and start is not None:
            if i - start >= min_len:
                segments.append((start, i))
            start = None
    if start is not None and len(arr) - start >= min_len:
        segments.append((start, len(arr)))
    return segments


def _stats(prefix: str, arr: np.ndarray) -> Dict[str, float]:
    if len(arr) == 0:
        return {}

    def slope(values: np.ndarray) -> float:
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values), dtype=float)
        y = values.astype(float)
        x = x - x.mean()
        y = y - y.mean()
        denom = float((x**2).sum()) + EPS
        return float((x * y).sum() / denom)

    return {
        f"{prefix}__mean": float(np.mean(arr)),
        f"{prefix}__std": float(np.std(arr)),
        f"{prefix}__p05": float(np.quantile(arr, 0.05)),
        f"{prefix}__p50": float(np.quantile(arr, 0.50)),
        f"{prefix}__p95": float(np.quantile(arr, 0.95)),
        f"{prefix}__min": float(np.min(arr)),
        f"{prefix}__max": float(np.max(arr)),
        f"{prefix}__slope": slope(arr),
    }


def window_features(seg_ctx: pd.DataFrame) -> Dict[str, float]:
    feats: Dict[str, float] = {"window_rows": float(len(seg_ctx))}
    cols = [
        "affected_n2",
        "healthy_n2",
        "affected_pressure",
        "healthy_pressure",
        "affected_precool",
        "healthy_precool",
        "pressure_diff",
        "pressure_abs_diff",
        "precool_diff",
        "precool_abs_diff",
        "affected_hpv",
        "healthy_hpv",
        "hpv_diff",
        "hpv_abs_diff",
        "altitude",
        "phase",
        "time_group",
    ]
    for col in cols:
        arr = pd.to_numeric(seg_ctx[col], errors="coerce").dropna().to_numpy()
        feats.update(_stats(col, arr))
    s1 = pd.to_numeric(seg_ctx["affected_hpv"], errors="coerce").dropna()
    if not s1.empty:
        feats["affected_hpv_open_ratio"] = float((s1 > 0.5).mean())
    s2 = pd.to_numeric(seg_ctx["healthy_hpv"], errors="coerce").dropna()
    if not s2.empty:
        feats["healthy_hpv_open_ratio"] = float((s2 > 0.5).mean())
    return feats


def extract_windows_for_flight(
    flight_path: str,
    folder_label: int,
    aircraft_id: str,
    faulty_side: str,
    min_rows_per_window: int,
    default_min_segment_len: int,
    spec: ConditionSpec,
) -> List[WindowRecord]:
    df = load_flight(flight_path)
    row_ctx = build_row_context(df, faulty_side)
    segments = extract_segments(row_ctx, spec, default_min_segment_len)
    out: List[WindowRecord] = []
    for idx, (start, end) in enumerate(segments):
        seg = row_ctx.iloc[start:end].copy()
        if len(seg) < min_rows_per_window:
            continue
        feats = window_features(seg)
        out.append(
            WindowRecord(
                aircraft_id=aircraft_id,
                folder_label=folder_label,
                flight_path=flight_path,
                faulty_side=faulty_side,  # type: ignore[arg-type]
                window_id=f"{Path(flight_path).stem}__{idx}",
                start_idx=start,
                end_idx=end,
                n_rows=len(seg),
                features=feats,
            )
        )
    return out


def windows_to_frame(windows: List[WindowRecord]) -> pd.DataFrame:
    rows = []
    for w in windows:
        row = {
            "aircraft_id": w.aircraft_id,
            "folder_label": w.folder_label,
            "flight_path": w.flight_path,
            "faulty_side": w.faulty_side,
            "window_id": w.window_id,
            "start_idx": w.start_idx,
            "end_idx": w.end_idx,
            "n_rows": w.n_rows,
        }
        row.update(w.features)
        rows.append(row)
    return pd.DataFrame(rows)
