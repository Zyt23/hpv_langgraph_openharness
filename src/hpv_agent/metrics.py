from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from .config import AppConfig
from .expressions import safe_eval
from .row_context import extract_windows_for_flight, windows_to_frame
from .schema import FolderPrediction, RuleBundle


def apply_rule_row(row: pd.Series, rule: RuleBundle) -> int:
    expr = (rule.score_expression or "").strip()
    if not expr:
        return 0
    env = {k: v for k, v in row.items()}
    try:
        score = safe_eval(expr, env)
        if isinstance(score, bool):
            return int(score)
        return int(float(score) >= float(rule.decision_threshold))
    except Exception:
        return 0


def score_windows(feature_df: pd.DataFrame, rule: RuleBundle) -> pd.Series:
    return feature_df.apply(lambda r: apply_rule_row(r, rule), axis=1) if not feature_df.empty else pd.Series(dtype=int)


def evaluate_windows(feature_df: pd.DataFrame, rule: RuleBundle) -> dict:
    if feature_df.empty:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n": 0}
    y_true = feature_df["folder_label"].astype(int).tolist()
    y_pred = score_windows(feature_df, rule).astype(int).tolist()
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    return {"precision": float(p), "recall": float(r), "f1": float(f), "n": len(feature_df)}


def _downsample_rows(df: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
    if max_rows is None or max_rows <= 0 or len(df) <= max_rows:
        return df
    idx = np.linspace(0, len(df) - 1, num=max_rows, dtype=int)
    return df.iloc[idx].copy()


def classify_folder_units(
    index_df: pd.DataFrame,
    aircraft_ids: List[str],
    rule: RuleBundle,
    cfg: AppConfig,
    max_flights_per_folder: int | None = None,
    verbose: bool = False,
    tag: str = "",
) -> List[FolderPrediction]:
    preds: List[FolderPrediction] = []
    total_aircraft = len(aircraft_ids)
    for i, aircraft_id in enumerate(aircraft_ids, start=1):
        if verbose:
            prefix = f"[{tag}] " if tag else ""
            print(f"{prefix}classify aircraft {i}/{total_aircraft}: {aircraft_id}", flush=True)
        sub_aircraft = index_df[index_df["aircraft_id"] == aircraft_id]
        faulty_side = sub_aircraft["faulty_side"].iloc[0]
        for true_label in [0, 1]:
            sub = sub_aircraft[sub_aircraft["folder_label"] == true_label]
            sub = _downsample_rows(sub, max_flights_per_folder)
            flight_scores = []
            for _, row in sub.iterrows():
                windows = extract_windows_for_flight(
                    row["flight_path"],
                    int(row["folder_label"]),
                    row["aircraft_id"],
                    faulty_side,
                    cfg.features.min_rows_per_window,
                    cfg.features.min_segment_len,
                    rule.condition_spec,
                )
                df = windows_to_frame(windows)
                if df.empty:
                    flight_scores.append(0.0)
                else:
                    flight_scores.append(float(score_windows(df, rule).mean()))
            mean_score = sum(flight_scores) / len(flight_scores) if flight_scores else 0.0
            pred = int(mean_score >= rule.folder_vote_threshold)
            preds.append(FolderPrediction(aircraft_id=aircraft_id, true_label=true_label, predicted_label=pred, mean_flight_score=mean_score, n_flights=len(flight_scores)))
    return preds


def summarize_folder_metrics(preds: List[FolderPrediction]) -> dict:
    if not preds:
        return {"Prec0": 0.0, "Rec0": 0.0, "F1_0": 0.0, "Prec1": 0.0, "Rec1": 0.0, "F1_1": 0.0, "n": 0}
    y_true = [p.true_label for p in preds]
    y_pred = [p.predicted_label for p in preds]
    p0, r0, f0, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0], average=None, zero_division=0)
    p1, r1, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[1], average=None, zero_division=0)
    return {"Prec0": float(p0[0]), "Rec0": float(r0[0]), "F1_0": float(f0[0]), "Prec1": float(p1[0]), "Rec1": float(r1[0]), "F1_1": float(f1[0]), "n": len(preds)}


def format_metrics_table(rows: dict) -> str:
    header = f"{'文件夹':<10} {'Prec0':>7} {'Rec0':>7} {'F1_0':>7} {'Prec1':>7} {'Rec1':>7} {'F1_1':>7}"
    sep = "-" * len(header)
    body = [f"{name:<10} {m['Prec0']:>7.4f} {m['Rec0']:>7.4f} {m['F1_0']:>7.4f} {m['Prec1']:>7.4f} {m['Rec1']:>7.4f} {m['F1_1']:>7.4f}" for name, m in rows.items()]
    return "\n".join([header, sep] + body)
