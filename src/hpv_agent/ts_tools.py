from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GroupKFold

from .config import AppConfig
from .metrics import evaluate_windows, score_windows
from .row_context import build_row_context, extract_windows_for_flight, load_flight, windows_to_frame
from .schema import ConditionSpec, RuleBundle


META_COLUMNS = {
    "aircraft_id",
    "folder_label",
    "flight_path",
    "faulty_side",
    "window_id",
    "start_idx",
    "end_idx",
    "n_rows",
}
LEAKY_FEATURE_PREFIXES = ("time_group__", "tail_num__", "lat__", "lon__")
GATING_FEATURE_PREFIXES = ("phase__", "altitude__")
RAW_N2_FEATURE_PREFIXES = ("affected_n2__", "healthy_n2__")
RULE_EXCLUDED_FEATURES = {"window_rows"}
EPS = 1e-8


@dataclass
class WindowToolData:
    frame: pd.DataFrame
    sampled_flights: list[dict]
    errors: list[dict]
    guard: dict


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return _finite_float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def _json_records(frame: pd.DataFrame) -> list[dict]:
    return [{str(k): _json_value(v) for k, v in row.items()} for row in frame.to_dict(orient="records")]


def _fmt_float(value: float) -> str:
    if not math.isfinite(float(value)):
        return "0.0"
    return f"{float(value):.8g}"


def _split_ids(split_manifest: dict, split: str) -> list[str]:
    if split == "train":
        return list(split_manifest.get("train_aircraft", []))
    if split in {"validation", "val"}:
        return list(split_manifest.get("validation_aircraft", []))
    if split == "holdout":
        return list(split_manifest.get("holdout_aircraft", []))
    raise ValueError(f"Unsupported split: {split}")


def _guard_train_aircraft(index_df: pd.DataFrame, split_manifest: dict, aircraft_ids: Sequence[str] | None) -> tuple[list[str], dict]:
    train_ids = set(_split_ids(split_manifest, "train"))
    requested = list(aircraft_ids) if aircraft_ids else sorted(index_df[index_df["aircraft_id"].isin(train_ids)]["aircraft_id"].unique().tolist())
    selected = [str(aid) for aid in requested if str(aid) in train_ids]
    blocked = sorted({str(aid) for aid in requested if str(aid) not in train_ids})
    return selected, {
        "allowed_split": "train",
        "holdout_used": False,
        "requested_aircraft_ids": [str(a) for a in requested],
        "selected_train_aircraft_ids": selected,
        "blocked_non_train_aircraft_ids": blocked,
        "n_blocked_non_train_aircraft": len(blocked),
    }


def _sample_training_rows(
    index_df: pd.DataFrame,
    split_manifest: dict,
    aircraft_ids: Sequence[str] | None,
    max_flights_per_label_per_aircraft: int,
) -> tuple[pd.DataFrame, dict]:
    selected, guard = _guard_train_aircraft(index_df, split_manifest, aircraft_ids)
    if not selected:
        return index_df.iloc[0:0].copy(), guard
    sub = index_df[index_df["aircraft_id"].isin(selected)].copy()
    sub = sub.sort_values(["folder_label", "aircraft_id", "flight_path"])
    max_n = max(1, int(max_flights_per_label_per_aircraft))
    sampled = sub.groupby(["folder_label", "aircraft_id"], group_keys=False).head(max_n).copy()
    return sampled, guard


def _collect_condition_windows(
    index_df: pd.DataFrame,
    split_manifest: dict,
    cfg: AppConfig,
    condition_spec: ConditionSpec,
    aircraft_ids: Sequence[str] | None = None,
    max_flights_per_label_per_aircraft: int = 3,
) -> WindowToolData:
    sampled_rows, guard = _sample_training_rows(
        index_df,
        split_manifest,
        aircraft_ids,
        max_flights_per_label_per_aircraft,
    )
    windows = []
    sampled_flights: list[dict] = []
    errors: list[dict] = []
    for row in sampled_rows.to_dict(orient="records"):
        sampled_flights.append(
            {
                "aircraft_id": str(row["aircraft_id"]),
                "folder_label": int(row["folder_label"]),
                "faulty_side": str(row["faulty_side"]),
                "flight_path": str(row["flight_path"]),
            }
        )
        try:
            windows.extend(
                extract_windows_for_flight(
                    row["flight_path"],
                    int(row["folder_label"]),
                    str(row["aircraft_id"]),
                    str(row["faulty_side"]),
                    cfg.features.min_rows_per_window,
                    cfg.features.min_segment_len,
                    condition_spec,
                )
            )
        except Exception as exc:
            errors.append(
                {
                    "aircraft_id": str(row.get("aircraft_id", "")),
                    "folder_label": int(row.get("folder_label", -1)),
                    "flight_path": str(row.get("flight_path", "")),
                    "error": str(exc),
                }
            )
    frame = windows_to_frame(windows)
    return WindowToolData(frame=frame, sampled_flights=sampled_flights, errors=errors, guard=guard)


def _feature_columns(frame: pd.DataFrame, include_gating: bool = False, include_raw_n2: bool = True) -> list[str]:
    if frame.empty:
        return []
    cols: list[str] = []
    for col in frame.columns:
        if col in META_COLUMNS or col in RULE_EXCLUDED_FEATURES:
            continue
        if col.startswith(LEAKY_FEATURE_PREFIXES):
            continue
        if not include_gating and col.startswith(GATING_FEATURE_PREFIXES):
            continue
        if not include_raw_n2 and col.startswith(RAW_N2_FEATURE_PREFIXES):
            continue
        if not pd.api.types.is_numeric_dtype(frame[col]):
            continue
        s = pd.to_numeric(frame[col], errors="coerce")
        if s.notna().sum() < 2:
            continue
        cols.append(col)
    return cols


def _series_stats(values: pd.Series | np.ndarray) -> dict:
    s = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return {"n": 0}
    return {
        "n": int(len(s)),
        "mean": _finite_float(s.mean()),
        "std": _finite_float(s.std(ddof=0)),
        "p05": _finite_float(s.quantile(0.05)),
        "p25": _finite_float(s.quantile(0.25)),
        "p50": _finite_float(s.quantile(0.50)),
        "p75": _finite_float(s.quantile(0.75)),
        "p95": _finite_float(s.quantile(0.95)),
        "min": _finite_float(s.min()),
        "max": _finite_float(s.max()),
    }


def _ks_test(a: np.ndarray, b: np.ndarray) -> dict:
    try:
        from scipy.stats import ks_2samp

        stat, pvalue = ks_2samp(a, b, alternative="two-sided", method="auto")
        return {"ks_stat": _finite_float(stat), "ks_pvalue": _finite_float(pvalue), "ks_available": True}
    except Exception:
        return {"ks_stat": None, "ks_pvalue": None, "ks_available": False}


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if len(y_true) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "balanced_accuracy": 0.0, "n": 0}
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    pos = y_true == 1
    neg = y_true == 0
    rec1 = float((y_pred[pos] == 1).mean()) if pos.any() else 0.0
    rec0 = float((y_pred[neg] == 0).mean()) if neg.any() else 0.0
    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f),
        "balanced_accuracy": float((rec0 + rec1) / 2.0),
        "n": int(len(y_true)),
    }


def _best_threshold(values: pd.Series, labels: pd.Series, max_thresholds: int = 25) -> dict | None:
    df = pd.DataFrame({"x": pd.to_numeric(values, errors="coerce"), "y": pd.to_numeric(labels, errors="coerce")})
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if df.empty or df["y"].nunique() < 2 or df["x"].nunique() < 2:
        return None
    quantiles = np.linspace(0.05, 0.95, num=max(3, max_thresholds))
    thresholds = sorted({float(x) for x in df["x"].quantile(quantiles).tolist() if math.isfinite(float(x))})
    if not thresholds:
        return None
    y = df["y"].astype(int).to_numpy()
    best: dict | None = None
    for threshold in thresholds:
        for operator in [">", "<="]:
            pred = (df["x"].to_numpy() > threshold).astype(int) if operator == ">" else (df["x"].to_numpy() <= threshold).astype(int)
            metrics = _binary_metrics(y, pred)
            row = {
                "operator": operator,
                "threshold": _finite_float(threshold),
                **metrics,
            }
            key = (row["f1"], row["balanced_accuracy"], row["precision"], row["recall"])
            if best is None or key > (best["f1"], best["balanced_accuracy"], best["precision"], best["recall"]):
                best = row
    return best


def _profile_feature_comparisons(frame: pd.DataFrame, top_k: int = 20, include_raw_n2: bool = True) -> list[dict]:
    if frame.empty or "folder_label" not in frame.columns:
        return []
    normal = frame[frame["folder_label"].astype(int) == 0]
    abnormal = frame[frame["folder_label"].astype(int) == 1]
    rows: list[dict] = []
    for col in _feature_columns(frame, include_gating=False, include_raw_n2=include_raw_n2):
        a = pd.to_numeric(normal[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        b = pd.to_numeric(abnormal[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        if len(a) < 2 or len(b) < 2:
            continue
        mean0 = float(np.mean(a))
        mean1 = float(np.mean(b))
        pooled = float(np.sqrt((np.var(a) + np.var(b)) / 2.0)) + EPS
        effect = (mean1 - mean0) / pooled
        best = _best_threshold(frame[col], frame["folder_label"])
        rows.append(
            {
                "feature": col,
                "normal": _series_stats(a),
                "abnormal": _series_stats(b),
                "delta_mean": _finite_float(mean1 - mean0),
                "effect_size_signed": _finite_float(effect),
                "effect_size_abs": _finite_float(abs(effect)),
                "best_threshold": best,
                **_ks_test(a, b),
            }
        )
    rows.sort(
        key=lambda r: (
            float(r.get("effect_size_abs") or 0.0),
            float((r.get("best_threshold") or {}).get("f1") or 0.0),
        ),
        reverse=True,
    )
    return rows[: max(1, int(top_k))]


def _window_counts(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"n_windows": 0, "windows_per_label": {}, "n_aircraft_with_windows": 0}
    counts = {int(k): int(v) for k, v in frame["folder_label"].value_counts().to_dict().items()}
    return {
        "n_windows": int(len(frame)),
        "windows_per_label": counts,
        "n_aircraft_with_windows": int(frame["aircraft_id"].astype(str).nunique()),
        "faulty_side_window_coverage": {str(k): int(v) for k, v in frame["faulty_side"].value_counts().to_dict().items()},
    }


def profile_condition_windows_tool(
    index_df: pd.DataFrame,
    split_manifest: dict,
    cfg: AppConfig,
    condition_spec: ConditionSpec,
    aircraft_ids: list[str] | None = None,
    max_flights_per_label_per_aircraft: int = 3,
    top_k: int = 20,
) -> dict:
    """Profile normal-vs-abnormal training windows for one condition."""
    data = _collect_condition_windows(
        index_df,
        split_manifest,
        cfg,
        condition_spec,
        aircraft_ids=aircraft_ids,
        max_flights_per_label_per_aircraft=max_flights_per_label_per_aircraft,
    )
    frame = data.frame
    return {
        "tool": "profile_condition_windows",
        "split_used": "train",
        "condition_spec": condition_spec.model_dump(),
        **_window_counts(frame),
        "n_sampled_flights": len(data.sampled_flights),
        "sampled_aircraft_ids": sorted({x["aircraft_id"] for x in data.sampled_flights}),
        "sampled_flights": data.sampled_flights[:80],
        "feature_comparisons": _profile_feature_comparisons(frame, top_k=top_k),
        "excluded_feature_prefixes": list(LEAKY_FEATURE_PREFIXES + GATING_FEATURE_PREFIXES),
        "guard": data.guard,
        "errors": data.errors[:20],
    }


def _prepare_feature_matrix(frame: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, dict]:
    x = frame[feature_cols].replace([np.inf, -np.inf], np.nan).copy()
    medians = x.median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x = x.fillna(medians)
    return x.astype(float), {str(k): _finite_float(v) for k, v in medians.to_dict().items()}


def _rf_group_cv_metrics(x: pd.DataFrame, y: pd.Series, groups: pd.Series, seed: int) -> dict:
    unique_groups = sorted(pd.Series(groups).astype(str).unique().tolist())
    if len(unique_groups) < 3 or y.nunique() < 2:
        return {"available": False, "reason": "need_at_least_3_aircraft_groups_and_both_labels"}
    n_splits = min(5, len(unique_groups))
    preds = np.zeros(len(y), dtype=int)
    tested = np.zeros(len(y), dtype=bool)
    splitter = GroupKFold(n_splits=n_splits)
    for train_idx, test_idx in splitter.split(x, y.astype(int), groups.astype(str)):
        if y.iloc[train_idx].nunique() < 2:
            continue
        model = RandomForestClassifier(
            n_estimators=120,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(x.iloc[train_idx], y.iloc[train_idx].astype(int))
        preds[test_idx] = model.predict(x.iloc[test_idx])
        tested[test_idx] = True
    if not tested.any():
        return {"available": False, "reason": "no_valid_group_fold_with_both_labels"}
    metrics = _binary_metrics(y.iloc[tested].astype(int).to_numpy(), preds[tested])
    metrics["available"] = True
    metrics["n_splits"] = int(n_splits)
    metrics["note"] = "Internal train-only grouped CV; not validation or holdout."
    return metrics


def _rule_expr(feature: str, threshold_info: dict) -> str:
    op = threshold_info.get("operator", ">")
    threshold = _fmt_float(float(threshold_info.get("threshold", 0.0)))
    return f"({feature} {op} {threshold})"


def _sampled_folder_metrics(frame: pd.DataFrame, rule: RuleBundle) -> dict:
    if frame.empty:
        return {"Prec0": 0.0, "Rec0": 0.0, "F1_0": 0.0, "Prec1": 0.0, "Rec1": 0.0, "F1_1": 0.0, "n": 0}
    scored = frame[["aircraft_id", "folder_label", "flight_path"]].copy()
    scored["window_pred"] = score_windows(frame, rule).astype(int).to_numpy()
    flight_scores = (
        scored.groupby(["aircraft_id", "folder_label", "flight_path"], as_index=False)["window_pred"]
        .mean()
        .rename(columns={"window_pred": "flight_score"})
    )
    folder_scores = flight_scores.groupby(["aircraft_id", "folder_label"], as_index=False)["flight_score"].mean()
    y_true = folder_scores["folder_label"].astype(int).to_numpy()
    y_pred = (folder_scores["flight_score"].to_numpy(dtype=float) >= float(rule.folder_vote_threshold)).astype(int)
    if len(y_true) == 0:
        return {"Prec0": 0.0, "Rec0": 0.0, "F1_0": 0.0, "Prec1": 0.0, "Rec1": 0.0, "F1_1": 0.0, "n": 0}
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], average=None, zero_division=0)
    return {
        "Prec0": float(p[0]),
        "Rec0": float(r[0]),
        "F1_0": float(f[0]),
        "Prec1": float(p[1]),
        "Rec1": float(r[1]),
        "F1_1": float(f[1]),
        "n": int(len(folder_scores)),
    }


def mine_candidate_rules_tool(
    index_df: pd.DataFrame,
    split_manifest: dict,
    cfg: AppConfig,
    condition_spec: ConditionSpec,
    aircraft_ids: list[str] | None = None,
    max_flights_per_label_per_aircraft: int = 3,
    top_k: int = 8,
) -> dict:
    """Mine simple RuleBundle candidates from training windows only."""
    data = _collect_condition_windows(
        index_df,
        split_manifest,
        cfg,
        condition_spec,
        aircraft_ids=aircraft_ids,
        max_flights_per_label_per_aircraft=max_flights_per_label_per_aircraft,
    )
    frame = data.frame
    base = {
        "tool": "mine_candidate_rules",
        "split_used": "train",
        "condition_spec": condition_spec.model_dump(),
        **_window_counts(frame),
        "guard": data.guard,
        "errors": data.errors[:20],
    }
    if frame.empty or frame.get("folder_label", pd.Series(dtype=int)).nunique() < 2:
        return {**base, "candidate_rules": [], "error": "need_both_labels_with_windows"}
    feature_cols = _feature_columns(frame, include_gating=False, include_raw_n2=False)
    if not feature_cols:
        return {**base, "candidate_rules": [], "error": "no_safe_numeric_features"}
    x, medians = _prepare_feature_matrix(frame, feature_cols)
    y = frame["folder_label"].astype(int)
    rf = RandomForestClassifier(
        n_estimators=240,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=cfg.split.seed,
        n_jobs=-1,
    )
    rf.fit(x, y)
    importances = [
        {"feature": feature, "importance": _finite_float(score)}
        for feature, score in sorted(zip(feature_cols, rf.feature_importances_), key=lambda t: float(t[1]), reverse=True)
    ]
    cv_metrics = _rf_group_cv_metrics(x, y, frame["aircraft_id"].astype(str), cfg.split.seed)

    threshold_rows: list[dict] = []
    for item in importances[: max(10, int(top_k) * 2)]:
        feature = item["feature"]
        best = _best_threshold(frame[feature], y)
        if not best:
            continue
        threshold_rows.append({"feature": feature, "importance": item["importance"], "threshold": best})
    threshold_rows.sort(
        key=lambda r: (
            float((r.get("threshold") or {}).get("f1") or 0.0),
            float((r.get("threshold") or {}).get("balanced_accuracy") or 0.0),
            float(r.get("importance") or 0.0),
        ),
        reverse=True,
    )

    candidates: list[dict] = []
    for rank, row in enumerate(threshold_rows[: max(1, int(top_k))], start=1):
        expr = _rule_expr(row["feature"], row["threshold"])
        rule = RuleBundle(
            condition_spec=condition_spec,
            rule_name=f"mined_threshold_{rank}_{row['feature']}",
            score_expression=expr,
            decision_threshold=0.5,
            folder_vote_threshold=cfg.loop.folder_vote_threshold,
            rationale=(
                "Train-only threshold candidate from random-forest feature ranking "
                "and window-level threshold search."
            ),
        )
        candidates.append(
            {
                "rule_bundle": rule.model_dump(),
                "source": "single_feature_threshold",
                "feature": row["feature"],
                "importance": row["importance"],
                "threshold": row["threshold"],
                "training_window_metrics": evaluate_windows(frame, rule),
                "sampled_train_folder_metrics": _sampled_folder_metrics(frame, rule),
            }
        )
    if len(threshold_rows) >= 2:
        first, second = threshold_rows[0], threshold_rows[1]
        expr = f"{_rule_expr(first['feature'], first['threshold'])} | {_rule_expr(second['feature'], second['threshold'])}"
        rule = RuleBundle(
            condition_spec=condition_spec,
            rule_name="mined_or_top2_thresholds",
            score_expression=expr,
            decision_threshold=0.5,
            folder_vote_threshold=cfg.loop.folder_vote_threshold,
            rationale="Train-only OR rule combining the two strongest safe threshold predicates.",
        )
        candidates.append(
            {
                "rule_bundle": rule.model_dump(),
                "source": "two_feature_or_threshold",
                "features": [first["feature"], second["feature"]],
                "thresholds": [first["threshold"], second["threshold"]],
                "training_window_metrics": evaluate_windows(frame, rule),
                "sampled_train_folder_metrics": _sampled_folder_metrics(frame, rule),
            }
        )
    candidates.sort(
        key=lambda r: (
            min(
                float((r.get("sampled_train_folder_metrics") or {}).get("F1_0") or 0.0),
                float((r.get("sampled_train_folder_metrics") or {}).get("F1_1") or 0.0),
            ),
            (
                float((r.get("sampled_train_folder_metrics") or {}).get("F1_0") or 0.0)
                + float((r.get("sampled_train_folder_metrics") or {}).get("F1_1") or 0.0)
            )
            / 2.0,
            float((r.get("sampled_train_folder_metrics") or {}).get("F1_1") or 0.0),
            float((r.get("training_window_metrics") or {}).get("f1") or 0.0),
        ),
        reverse=True,
    )
    return {
        **base,
        "feature_medians_used_for_imputation": medians,
        "rf_feature_importances": importances[:30],
        "rf_group_cv_window_metrics": cv_metrics,
        "threshold_candidates": threshold_rows[:30],
        "candidate_rules": candidates[: max(1, int(top_k))],
        "excluded_feature_prefixes": list(LEAKY_FEATURE_PREFIXES + GATING_FEATURE_PREFIXES + RAW_N2_FEATURE_PREFIXES),
    }


def robust_normal_baseline_tool(
    index_df: pd.DataFrame,
    split_manifest: dict,
    cfg: AppConfig,
    condition_spec: ConditionSpec,
    aircraft_ids: list[str] | None = None,
    max_flights_per_label_per_aircraft: int = 3,
    max_features: int = 5,
) -> dict:
    """Build a robust healthy-baseline score from train label=0 windows only."""
    data = _collect_condition_windows(
        index_df,
        split_manifest,
        cfg,
        condition_spec,
        aircraft_ids=aircraft_ids,
        max_flights_per_label_per_aircraft=max_flights_per_label_per_aircraft,
    )
    frame = data.frame
    base = {
        "tool": "robust_normal_baseline",
        "split_used": "train",
        "condition_spec": condition_spec.model_dump(),
        **_window_counts(frame),
        "guard": data.guard,
        "errors": data.errors[:20],
    }
    if frame.empty or frame.get("folder_label", pd.Series(dtype=int)).nunique() < 2:
        return {**base, "candidate_rules": [], "error": "need_both_labels_with_windows"}
    normal = frame[frame["folder_label"].astype(int) == 0]
    if len(normal) < 3:
        return {**base, "candidate_rules": [], "error": "need_at_least_3_normal_windows"}
    comparisons = _profile_feature_comparisons(frame, top_k=max(12, max_features * 3), include_raw_n2=False)
    selected: list[dict] = []
    for row in comparisons:
        feature = row["feature"]
        normal_values = pd.to_numeric(normal[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(normal_values) < 3:
            continue
        median = float(normal_values.median())
        mad = float((normal_values - median).abs().median()) * 1.4826
        q25 = float(normal_values.quantile(0.25))
        q75 = float(normal_values.quantile(0.75))
        iqr_scale = (q75 - q25) / 1.349 if q75 > q25 else 0.0
        all_values = pd.to_numeric(frame[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        global_scale = float(all_values.std(ddof=0)) * 0.05 if len(all_values) >= 2 else 0.0
        scale = max(mad, iqr_scale, float(normal_values.std(ddof=0)) * 0.1, global_scale, 1e-3)
        if not math.isfinite(scale) or scale <= EPS:
            continue
        selected.append(
            {
                "feature": feature,
                "normal_median": _finite_float(median),
                "robust_scale": _finite_float(scale),
                "effect_size_abs": row.get("effect_size_abs"),
            }
        )
        if len(selected) >= max(1, int(max_features)):
            break
    if not selected:
        return {**base, "candidate_rules": [], "error": "no_stable_normal_baseline_features"}

    scores = np.zeros(len(frame), dtype=float)
    terms: list[str] = []
    for item in selected:
        feature = str(item["feature"])
        median = float(item["normal_median"])
        scale = float(item["robust_scale"])
        values = pd.to_numeric(frame[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(median).to_numpy(dtype=float)
        scores += np.abs((values - median) / scale)
        terms.append(f"abs(({feature} - {_fmt_float(median)}) / {_fmt_float(scale)})")
    scores = scores / float(len(selected))
    best = _best_threshold(pd.Series(scores), frame["folder_label"].astype(int))
    if not best:
        return {**base, "candidate_rules": [], "selected_features": selected, "error": "could_not_find_score_threshold"}
    rule = RuleBundle(
        condition_spec=condition_spec,
        rule_name="robust_normal_baseline_zscore",
        score_expression=f"({' + '.join(terms)}) / {_fmt_float(float(len(terms)))}",
        decision_threshold=float(best["threshold"]),
        folder_vote_threshold=cfg.loop.folder_vote_threshold,
        rationale="Train-only healthy baseline: average robust z-score against label=0 windows.",
    )
    score_frame = frame.copy()
    score_frame["_robust_baseline_score"] = scores
    score_summary = {
        "normal": _series_stats(score_frame[score_frame["folder_label"].astype(int) == 0]["_robust_baseline_score"]),
        "abnormal": _series_stats(score_frame[score_frame["folder_label"].astype(int) == 1]["_robust_baseline_score"]),
    }
    return {
        **base,
        "selected_features": selected,
        "score_summary": score_summary,
        "threshold": best,
        "candidate_rules": [
            {
                "rule_bundle": rule.model_dump(),
                "source": "robust_normal_baseline",
                "training_window_metrics": evaluate_windows(frame, rule),
                "sampled_train_folder_metrics": _sampled_folder_metrics(frame, rule),
            }
        ],
        "excluded_feature_prefixes": list(LEAKY_FEATURE_PREFIXES + GATING_FEATURE_PREFIXES + RAW_N2_FEATURE_PREFIXES),
    }


def isolation_forest_window_tool(
    index_df: pd.DataFrame,
    split_manifest: dict,
    cfg: AppConfig,
    condition_spec: ConditionSpec,
    aircraft_ids: list[str] | None = None,
    max_flights_per_label_per_aircraft: int = 3,
    contamination: float = 0.20,
) -> dict:
    """Explore train-window anomalies by fitting IsolationForest on label=0 windows."""
    data = _collect_condition_windows(
        index_df,
        split_manifest,
        cfg,
        condition_spec,
        aircraft_ids=aircraft_ids,
        max_flights_per_label_per_aircraft=max_flights_per_label_per_aircraft,
    )
    frame = data.frame
    base = {
        "tool": "isolation_forest_windows",
        "split_used": "train",
        "condition_spec": condition_spec.model_dump(),
        **_window_counts(frame),
        "guard": data.guard,
        "errors": data.errors[:20],
    }
    if frame.empty or frame.get("folder_label", pd.Series(dtype=int)).nunique() < 2:
        return {**base, "error": "need_both_labels_with_windows"}
    feature_cols = _feature_columns(frame, include_gating=False, include_raw_n2=False)
    if not feature_cols:
        return {**base, "error": "no_safe_numeric_features"}
    x, medians = _prepare_feature_matrix(frame, feature_cols)
    normal_mask = frame["folder_label"].astype(int) == 0
    if normal_mask.sum() < 5:
        return {**base, "error": "need_at_least_5_normal_windows"}
    model = IsolationForest(
        n_estimators=240,
        contamination=max(0.01, min(0.49, float(contamination))),
        random_state=cfg.split.seed,
        n_jobs=-1,
    )
    model.fit(x[normal_mask])
    anomaly_score = -model.decision_function(x)
    best = _best_threshold(pd.Series(anomaly_score), frame["folder_label"].astype(int))
    meta_cols = [c for c in ["aircraft_id", "folder_label", "flight_path", "faulty_side", "window_id", "start_idx", "end_idx"] if c in frame.columns]
    scored = frame[meta_cols].copy()
    scored["anomaly_score"] = anomaly_score
    top_windows = _json_records(scored.sort_values("anomaly_score", ascending=False).head(20))
    return {
        **base,
        "feature_columns": feature_cols,
        "feature_medians_used_for_imputation": medians,
        "score_summary": {
            "normal": _series_stats(anomaly_score[normal_mask.to_numpy()]),
            "abnormal": _series_stats(anomaly_score[(~normal_mask).to_numpy()]),
        },
        "best_training_score_threshold": best,
        "top_anomalous_windows": top_windows,
        "note": "Exploratory train-only model score; not directly serializable as a final RuleBundle.",
    }


def _default_signal_columns(faulty_side: Literal["left", "right"]) -> list[str]:
    return [
        "affected_n2",
        "healthy_n2",
        "affected_pressure",
        "healthy_pressure",
        "affected_precool",
        "healthy_precool",
        "affected_hpv",
        "healthy_hpv",
        "pressure_diff",
        "pressure_abs_diff",
        "precool_diff",
        "hpv_diff",
    ]


def _numeric_context_for_flight(flight_path: str, faulty_side: Literal["left", "right"]) -> pd.DataFrame:
    df = load_flight(flight_path)
    ctx = build_row_context(df, faulty_side)
    return ctx.apply(pd.to_numeric, errors="coerce")


def _corr_at_lag(x: np.ndarray, y: np.ndarray, lag: int) -> float | None:
    if lag > 0:
        xx, yy = x[:-lag], y[lag:]
    elif lag < 0:
        xx, yy = x[-lag:], y[:lag]
    else:
        xx, yy = x, y
    if len(xx) < 10:
        return None
    mask = np.isfinite(xx) & np.isfinite(yy)
    if mask.sum() < 10:
        return None
    xx = xx[mask]
    yy = yy[mask]
    if float(np.std(xx)) <= EPS or float(np.std(yy)) <= EPS:
        return None
    corr = float(np.corrcoef(xx, yy)[0, 1])
    return corr if math.isfinite(corr) else None


def lag_correlation_flight_tool(
    flight_path: str,
    faulty_side: Literal["left", "right"],
    max_lag: int = 300,
) -> dict:
    """Compute lagged correlations among physical signals for one already-selected flight."""
    ctx = _numeric_context_for_flight(flight_path, faulty_side)
    cols = [c for c in _default_signal_columns(faulty_side) if c in ctx.columns and ctx[c].notna().sum() >= 20]
    max_lag = max(1, min(int(max_lag), max(1, len(ctx) // 3)))
    step = max(1, max_lag // 60)
    lag_grid = list(range(-max_lag, max_lag + 1, step))
    rows: list[dict] = []
    for i, left in enumerate(cols):
        for right in cols[i + 1 :]:
            x = ctx[left].to_numpy(dtype=float)
            y = ctx[right].to_numpy(dtype=float)
            best: dict | None = None
            for lag in lag_grid:
                corr = _corr_at_lag(x, y, lag)
                if corr is None:
                    continue
                row = {"left": left, "right": right, "lag": int(lag), "corr": _finite_float(corr), "abs_corr": _finite_float(abs(corr))}
                if best is None or float(row["abs_corr"] or 0.0) > float(best["abs_corr"] or 0.0):
                    best = row
            if best:
                rows.append(best)
    rows.sort(key=lambda r: float(r.get("abs_corr") or 0.0), reverse=True)
    return {
        "tool": "lag_correlation_flight",
        "flight_path": str(Path(flight_path)),
        "faulty_side": faulty_side,
        "n_rows": int(len(ctx)),
        "max_lag": int(max_lag),
        "lag_step": int(step),
        "top_lagged_correlations": rows[:30],
        "note": "Caller is responsible for passing train-selected flight paths only.",
    }


def _standardized_signal(ctx: pd.DataFrame, columns: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    selected = [c for c in columns if c in ctx.columns and ctx[c].notna().sum() >= 20]
    if not selected:
        return np.empty((0, 0)), []
    x = ctx[selected].replace([np.inf, -np.inf], np.nan).copy()
    x = x.interpolate(limit_direction="both").ffill().bfill()
    x = x.fillna(x.median(numeric_only=True)).fillna(0.0)
    arr = x.to_numpy(dtype=float)
    arr = (arr - np.nanmean(arr, axis=0)) / (np.nanstd(arr, axis=0) + EPS)
    return arr, selected


def _fallback_change_points(signal: np.ndarray, top_k: int = 8) -> list[dict]:
    if signal.shape[0] < 20:
        return []
    score = np.abs(np.diff(signal, axis=0)).mean(axis=1)
    order = np.argsort(score)[::-1]
    min_distance = max(10, signal.shape[0] // 50)
    chosen: list[dict] = []
    for idx in order:
        point = int(idx) + 1
        if any(abs(point - int(row["index"])) < min_distance for row in chosen):
            continue
        chosen.append({"index": point, "score": _finite_float(score[idx])})
        if len(chosen) >= top_k:
            break
    return sorted(chosen, key=lambda r: int(r["index"]))


def change_point_flight_tool(
    flight_path: str,
    faulty_side: Literal["left", "right"],
    columns: list[str] | None = None,
    penalty: float = 8.0,
) -> dict:
    """Detect regime changes in selected time-series signals for one already-selected flight."""
    ctx = _numeric_context_for_flight(flight_path, faulty_side)
    wanted = columns or ["affected_pressure", "healthy_pressure", "pressure_diff", "affected_hpv", "affected_n2", "altitude"]
    signal, selected = _standardized_signal(ctx, wanted)
    if signal.size == 0:
        return {"tool": "change_point_flight", "flight_path": str(Path(flight_path)), "faulty_side": faulty_side, "error": "no_numeric_columns"}
    try:
        import ruptures as rpt

        algo = rpt.Pelt(model="rbf").fit(signal)
        points = [int(p) for p in algo.predict(pen=float(penalty)) if int(p) < signal.shape[0]]
        method = "ruptures_pelt_rbf"
        change_points = [{"index": p, "score": None} for p in points]
    except Exception as exc:
        method = "fallback_diff_score"
        change_points = _fallback_change_points(signal)
        fallback_error = str(exc)
    out = {
        "tool": "change_point_flight",
        "flight_path": str(Path(flight_path)),
        "faulty_side": faulty_side,
        "n_rows": int(len(ctx)),
        "columns": selected,
        "method": method,
        "penalty": _finite_float(penalty),
        "change_points": change_points[:40],
        "note": "Caller is responsible for passing train-selected flight paths only.",
    }
    if method == "fallback_diff_score":
        out["optional_dependency_error"] = fallback_error
    return out


def _top_non_overlapping(scores: np.ndarray, top_k: int, min_distance: int) -> list[dict]:
    finite = np.where(np.isfinite(scores), scores, -np.inf)
    order = np.argsort(finite)[::-1]
    chosen: list[dict] = []
    for idx in order:
        if not math.isfinite(float(finite[idx])):
            continue
        if any(abs(int(idx) - int(row["start_idx"])) < min_distance for row in chosen):
            continue
        chosen.append({"start_idx": int(idx), "score": _finite_float(finite[idx])})
        if len(chosen) >= top_k:
            break
    return chosen


def matrix_profile_discords_tool(
    flight_path: str,
    faulty_side: Literal["left", "right"],
    column: str = "pressure_diff",
    subseq_len: int = 120,
    top_k: int = 5,
) -> dict:
    """Find anomalous subsequences for one selected flight and signal."""
    ctx = _numeric_context_for_flight(flight_path, faulty_side)
    if column not in ctx.columns:
        return {"tool": "matrix_profile_discords", "flight_path": str(Path(flight_path)), "faulty_side": faulty_side, "column": column, "error": "column_not_found"}
    s = pd.to_numeric(ctx[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    s = s.interpolate(limit_direction="both").ffill().bfill().dropna()
    if len(s) < max(20, int(subseq_len) + 5):
        return {"tool": "matrix_profile_discords", "flight_path": str(Path(flight_path)), "faulty_side": faulty_side, "column": column, "error": "series_too_short"}
    m = max(8, min(int(subseq_len), len(s) // 2))
    values = s.to_numpy(dtype=float)
    values = (values - np.mean(values)) / (np.std(values) + EPS)
    try:
        import stumpy

        mp = stumpy.stump(values, m=m)
        scores = np.asarray(mp[:, 0], dtype=float)
        method = "stumpy_matrix_profile"
    except Exception as exc:
        rolling = pd.Series(values).rolling(window=m, min_periods=m)
        scores = rolling.apply(lambda x: float(np.mean(np.abs(x))), raw=True).to_numpy()
        scores[: m - 1] = np.nan
        method = "fallback_rolling_abs_z"
        fallback_error = str(exc)
    discords = _top_non_overlapping(scores, max(1, int(top_k)), min_distance=m)
    for row in discords:
        row["end_idx"] = int(row["start_idx"] + m)
    out = {
        "tool": "matrix_profile_discords",
        "flight_path": str(Path(flight_path)),
        "faulty_side": faulty_side,
        "column": column,
        "n_rows": int(len(s)),
        "subseq_len": int(m),
        "method": method,
        "discords": discords,
        "note": "Caller is responsible for passing train-selected flight paths only.",
    }
    if method == "fallback_rolling_abs_z":
        out["optional_dependency_error"] = fallback_error
    return out
