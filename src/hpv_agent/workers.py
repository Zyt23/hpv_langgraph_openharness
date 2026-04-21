from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import time

import pandas as pd

from .config import AppConfig
from .knowledge_base import fetch_knowledge_context
from .metrics import classify_folder_units, summarize_folder_metrics
from .observation import summarize_window_features
from .openharness_adapter import run_openharness_structured
from .row_context import build_row_context, extract_windows_for_flight, load_flight, windows_to_frame
from .schema import ConditionSpec, HypothesizeAction, KnowledgeRequest, ObserveAction, ReflectionDecision, RuleBundle


MECHANISM_PROMPT = """
HPV precursor classification plan for A320 bleed system (must follow).

Data organization
- Folder naming: B-aircraft_id-HPV-fault_seq-fault_side, where fault_side: 1=left, 2=right.
- Each aircraft folder has two labels: 0 and 1.
  - 0: healthy / far from fault.
  - 1: degraded / close to fault.

Column semantics
- Left side (1):
  - N21: engine rotational speed.
  - PUD1 or BMPS1 (exactly one exists): transmission duct pressure.
  - PRECOOL_PRESS1: main manifold pressure.
  - HPV_ENG1_R: bleed valve open state (0=closed, 1=open; angle unknown).
- Right side (2):
  - N22: engine rotational speed.
  - PUD2 or BMPS2 (exactly one exists): transmission duct pressure.
  - PRECOOL_PRESS2: main manifold pressure.
  - HPV_ENG2_R: bleed valve open state (0=closed, 1=open; angle unknown).
- Common:
  - FLIGHT_PHASE: flight phase code.
  - LATP: latitude.
  - LONP: longitude.
  - ALT_STD: standard-pressure altitude.
  - time_group: segment index within a flight.
  - TAIL_NUM: aircraft registration identifier.
- Data format:
  - All columns are float32 in parquet storage.
  - One parquet file equals one flight, typically about 4,000 to 15,000 rows.

Hard constraints
- On each side, only one of PUD or BMPS may exist. The program must auto-detect it.
- Read strategy: read full columns from selected flights only. Do not preload all data.
- Condition expressions are Python-style boolean expressions executed by the program.

Mechanism preference
- Focus on fault-vs-healthy asymmetry: pressure_diff, hpv_diff, precool_diff and abs variants.
- Prefer coupled anomalies (HPV state with pressure/speed), not isolated single-variable thresholds.
- Use FLIGHT_PHASE, ALT_STD, time_group to constrain operating windows and reduce confounding.

Convergence policy
- When evidence is sufficient, finalize_condition promptly. Avoid repeated list_aircraft loops.
- If at least one inspect_flight and one test_condition are done, and both labels are covered in windows,
  prefer finalize_condition.
"""


def plan_knowledge(cfg: AppConfig, task_name: str, task_context: dict, knowledge_index: dict, workspace: str) -> list[dict]:
    try:
        req = run_openharness_structured(
            prompt=(
                f"You are preparing knowledge retrieval for task={task_name}. "
                "Choose a few search queries and/or direct doc_ids from the available docs.\n\n"
                f"available_docs={[d['doc_id'] for d in knowledge_index['docs']]}\n"
                f"task_context={json.dumps(task_context, ensure_ascii=False)[:8000]}"
            ),
            output_model=KnowledgeRequest,
            cfg=cfg,
            workspace=workspace,
        )
        return fetch_knowledge_context(knowledge_index, req.queries, req.doc_ids)
    except Exception:
        return []


def _compact_trace_for_prompt(trace_items: list[dict]) -> list[dict]:
    compact = []
    for item in trace_items:
        d = item.get("decision", {})
        r = item.get("result", {})
        row = {"action": d.get("action", "")}
        for k in ["split", "aircraft_id", "folder_label", "note"]:
            if k in d and d.get(k) not in ("", None, []):
                row[k] = d.get(k)
        if isinstance(r, dict):
            rr = {}
            for k in ["n_aircraft", "n_flights", "n_rows", "n_windows", "n_folder_units", "windows_per_label", "metrics", "error"]:
                if k in r:
                    rr[k] = r.get(k)
            row["result_summary"] = rr
        compact.append(row)
    return compact


def _trace_action_list(trace_items: list[dict]) -> list[str]:
    actions = []
    for item in trace_items:
        action = str(item.get("decision", {}).get("action", "")).strip()
        if not action:
            continue
        if action.startswith("bootstrap_probe_") or action.startswith("model_call_"):
            continue
        actions.append(action)
    return actions


def _consecutive_tail_count(actions: list[str], target: str) -> int:
    n = 0
    for a in reversed(actions):
        if a != target:
            break
        n += 1
    return n


def _last_trace_item(trace_items: list[dict], action: str) -> dict | None:
    for item in reversed(trace_items):
        if item.get("decision", {}).get("action") == action:
            return item
    return None


def _progress_summary_for_observe(trace_items: list[dict]) -> dict:
    actions = _trace_action_list(trace_items)
    cnt = Counter(actions)
    last_test = _last_trace_item(trace_items, "test_condition")
    last_windows = None
    last_windows_per_label = {}
    last_test_spec = None
    if last_test:
        r = last_test.get("result", {}) if isinstance(last_test.get("result", {}), dict) else {}
        last_windows = r.get("n_windows")
        last_windows_per_label = r.get("windows_per_label", {}) if isinstance(r.get("windows_per_label", {}), dict) else {}
        if isinstance(r.get("condition_spec"), dict):
            last_test_spec = r.get("condition_spec")
    return {
        "counts": dict(cnt),
        "consecutive_list_aircraft": _consecutive_tail_count(actions, "list_aircraft"),
        "has_list_flights": cnt.get("list_flights", 0) > 0,
        "has_inspect_flight": cnt.get("inspect_flight", 0) > 0,
        "has_test_condition": cnt.get("test_condition", 0) > 0,
        "last_test_n_windows": last_windows,
        "last_test_windows_per_label": last_windows_per_label,
        "last_test_condition_spec": last_test_spec,
    }


def _progress_summary_for_hypothesize(trace_items: list[dict]) -> dict:
    actions = _trace_action_list(trace_items)
    cnt = Counter(actions)
    last_test = _last_trace_item(trace_items, "test_rule")
    last_metrics = {}
    if last_test:
        r = last_test.get("result", {}) if isinstance(last_test.get("result", {}), dict) else {}
        if isinstance(r.get("metrics"), dict):
            last_metrics = r.get("metrics")
    return {
        "counts": dict(cnt),
        "consecutive_list_aircraft": _consecutive_tail_count(actions, "list_aircraft"),
        "has_list_flights": cnt.get("list_flights", 0) > 0,
        "has_inspect_flight": cnt.get("inspect_flight", 0) > 0,
        "has_test_rule": cnt.get("test_rule", 0) > 0,
        "last_test_metrics": last_metrics,
    }


def _split_aircraft(split_manifest: dict, split: str) -> list[str]:
    if split == "train":
        return split_manifest["train_aircraft"]
    if split == "val":
        return split_manifest["holdout_aircraft"]
    return sorted(set(split_manifest["train_aircraft"] + split_manifest["holdout_aircraft"]))


def _tool_list_aircraft(index_df: pd.DataFrame, split_manifest: dict, split: str) -> dict:
    allowed = set(_split_aircraft(split_manifest, split))
    ids = sorted(index_df[index_df["aircraft_id"].isin(allowed)]["aircraft_id"].unique().tolist())
    return {"split": split, "aircraft_ids": ids, "n_aircraft": len(ids)}


def _tool_list_flights(index_df: pd.DataFrame, split_manifest: dict, split: str, aircraft_id: str, folder_label: int, max_items: int) -> dict:
    allowed = set(_split_aircraft(split_manifest, split))
    sub = index_df[
        (index_df["aircraft_id"] == aircraft_id)
        & (index_df["folder_label"] == int(folder_label))
        & (index_df["aircraft_id"].isin(allowed))
    ]
    flights = sub["flight_path"].tolist()[: max(1, int(max_items))]
    return {"split": split, "aircraft_id": aircraft_id, "folder_label": int(folder_label), "n_flights": len(flights), "flight_paths": flights}


def _pick_flight(index_df: pd.DataFrame, split_manifest: dict, split: str, aircraft_id: str, folder_label: int, flight_path: str) -> tuple[str, str]:
    allowed = set(_split_aircraft(split_manifest, split))
    sub = index_df[
        (index_df["aircraft_id"] == aircraft_id)
        & (index_df["folder_label"] == int(folder_label))
        & (index_df["aircraft_id"].isin(allowed))
    ]
    if sub.empty:
        raise ValueError(f"No flights found for aircraft={aircraft_id}, label={folder_label}, split={split}")
    if flight_path:
        hit = sub[sub["flight_path"] == flight_path]
        if hit.empty:
            hit = sub[sub["flight_path"].str.endswith(Path(flight_path).name)]
        if not hit.empty:
            row = hit.iloc[0]
            return str(row["flight_path"]), str(row["faulty_side"])
    row = sub.iloc[0]
    return str(row["flight_path"]), str(row["faulty_side"])


def _tool_inspect_flight(
    index_df: pd.DataFrame,
    split_manifest: dict,
    split: str,
    aircraft_id: str,
    folder_label: int,
    flight_path: str,
    max_rows: int,
) -> dict:
    chosen_path, faulty_side = _pick_flight(index_df, split_manifest, split, aircraft_id, folder_label, flight_path)
    df = load_flight(chosen_path)
    row_ctx = build_row_context(df, faulty_side)
    if max_rows > 0 and len(row_ctx) > max_rows:
        row_ctx = row_ctx.iloc[:max_rows].copy()
    stats = {}
    for col in ["affected_n2", "healthy_n2", "affected_pressure", "healthy_pressure", "affected_hpv", "healthy_hpv", "pressure_abs_diff", "precool_abs_diff", "altitude", "phase"]:
        s = pd.to_numeric(row_ctx[col], errors="coerce").dropna()
        if s.empty:
            continue
        stats[col] = {
            "mean": float(s.mean()),
            "p05": float(s.quantile(0.05)),
            "p50": float(s.quantile(0.5)),
            "p95": float(s.quantile(0.95)),
        }
    return {
        "aircraft_id": aircraft_id,
        "folder_label": int(folder_label),
        "flight_path": chosen_path,
        "faulty_side": faulty_side,
        "n_rows": int(len(df)),
        "columns": list(df.columns),
        "row_ctx_columns": list(row_ctx.columns),
        "stats": stats,
    }


def _test_condition_on_aircraft(index_df: pd.DataFrame, aircraft_ids: list[str], cfg: AppConfig, spec: ConditionSpec) -> dict:
    sub = index_df[index_df["aircraft_id"].isin(aircraft_ids)]
    windows = []
    sampled_flights = []
    for label in [0, 1]:
        rows = sub[sub["folder_label"] == label].head(cfg.agent_tools.test_flights_per_label).to_dict(orient="records")
        for row in rows:
            sampled_flights.append({"aircraft_id": row["aircraft_id"], "folder_label": int(row["folder_label"]), "flight_path": row["flight_path"]})
            windows.extend(
                extract_windows_for_flight(
                    row["flight_path"],
                    int(row["folder_label"]),
                    row["aircraft_id"],
                    row["faulty_side"],
                    cfg.features.min_rows_per_window,
                    cfg.features.min_segment_len,
                    spec,
                )
            )
    df = windows_to_frame(windows)
    top = summarize_window_features(df, top_k=12)
    label_counts = {}
    if not df.empty:
        cnt = df["folder_label"].value_counts().to_dict()
        label_counts = {int(k): int(v) for k, v in cnt.items()}
    return {
        "condition_spec": spec.model_dump(),
        "n_windows": int(len(df)),
        "windows_per_label": label_counts,
        "sampled_flights": sampled_flights,
        "top_features": top.get("top_features", []),
    }


def _test_rule_on_aircraft(index_df: pd.DataFrame, aircraft_ids: list[str], cfg: AppConfig, rule: RuleBundle) -> dict:
    preds = classify_folder_units(
        index_df,
        aircraft_ids,
        rule,
        cfg,
        max_flights_per_folder=cfg.loop.tool_eval_max_flights_per_folder,
        verbose=False,
        tag="tool_eval",
    )
    return {
        "n_folder_units": len(preds),
        "metrics": summarize_folder_metrics(preds),
    }


def choose_condition(
    cfg: AppConfig,
    index_df: pd.DataFrame,
    split_manifest: dict,
    knowledge_index: dict,
    workspace: str,
    round_no: int,
) -> tuple[ConditionSpec, dict]:
    kn = plan_knowledge(cfg, "observe_condition", {"round_no": round_no}, knowledge_index, workspace)
    trace: list[dict] = []
    # Deterministic bootstrap probe: guarantees at least one real parquet inspection attempt.
    try:
        probe_aircraft = _tool_list_aircraft(index_df, split_manifest, "train")
        trace.append({"decision": {"action": "bootstrap_probe_list_aircraft", "split": "train"}, "result": probe_aircraft})
        ids = probe_aircraft.get("aircraft_ids", [])
        if ids:
            aid = ids[0]
            probe_flights = _tool_list_flights(index_df, split_manifest, "train", aid, 0, 1)
            trace.append(
                {
                    "decision": {"action": "bootstrap_probe_list_flights", "split": "train", "aircraft_id": aid, "folder_label": 0},
                    "result": probe_flights,
                }
            )
            fps = probe_flights.get("flight_paths", [])
            if fps:
                probe_inspect = _tool_inspect_flight(index_df, split_manifest, "train", aid, 0, fps[0], cfg.agent_tools.inspect_max_rows)
                trace.append(
                    {
                        "decision": {
                            "action": "bootstrap_probe_inspect_flight",
                            "split": "train",
                            "aircraft_id": aid,
                            "folder_label": 0,
                            "flight_path": fps[0],
                        },
                        "result": probe_inspect,
                    }
                )
    except Exception as e:
        trace.append({"decision": {"action": "bootstrap_probe"}, "result": {"error": str(e)}})
    default_spec = ConditionSpec(
        name="baseline_operating_window",
        description="Fallback condition when tool loop does not finalize.",
        expression="(affected_n2 > 55) & (healthy_n2 > 55) & (phase >= 2) & (phase <= 9)",
        min_segment_len=cfg.features.min_segment_len,
    )
    for _ in range(cfg.agent_tools.observe_max_steps):
        compact_trace = _compact_trace_for_prompt(trace[-6:])
        progress = _progress_summary_for_observe(trace)
        print(f"[observe-worker] round={round_no} step_start trace_steps={len(trace)}", flush=True)
        print(
            f"[observe-worker] progress counts={progress.get('counts', {})} "
            f"consecutive_list_aircraft={progress.get('consecutive_list_aircraft', 0)} "
            f"has_inspect={progress.get('has_inspect_flight')} "
            f"has_test={progress.get('has_test_condition')} "
            f"last_test_n_windows={progress.get('last_test_n_windows')}",
            flush=True,
        )
        prompt = (
            "You are in OBSERVE stage. Decide one tool action now.\n"
            "Return ObserveAction JSON only.\n\n"
            f"{MECHANISM_PROMPT}\n"
            "Allowed actions: list_aircraft, list_flights, inspect_flight, test_condition, finalize_condition.\n"
            "All tool calls must stay on training split only.\n"
            "For inspect_flight/test_condition, you must first choose aircraft and flights via tools.\n"
            "Action policy:\n"
            "- Do not repeat list_aircraft if you already listed aircraft once.\n"
            "- After list_flights, prefer inspect_flight.\n"
            "- After inspect_flight and test_condition with both labels covered, prefer finalize_condition.\n"
            "Condition expression must be a Python-style expression over row-context aliases, such as:\n"
            "(affected_n2 > 60) & (healthy_n2 > 60) & (phase >= 2) & (phase <= 8) & (altitude > 1000)\n\n"
            f"knowledge_context={json.dumps(kn, ensure_ascii=False)}\n"
            f"progress_summary={json.dumps(progress, ensure_ascii=False)}\n"
            f"last_trace={json.dumps(compact_trace, ensure_ascii=False)}\n"
        )
        try:
            t_call = time.perf_counter()
            print(f"[observe-worker] model_call prompt_chars={len(prompt)}", flush=True)
            decision = run_openharness_structured(prompt, ObserveAction, cfg, workspace)
            decision_info = {
                "action": decision.action,
                "aircraft_id": decision.aircraft_id,
                "folder_label": decision.folder_label,
                "max_items": decision.max_items,
                "flight_path": decision.flight_path,
                "n_aircraft_ids": len(decision.aircraft_ids or []),
                "has_condition_spec": decision.condition_spec is not None,
            }
            print(
                f"[observe-worker] model_decision={json.dumps(decision_info, ensure_ascii=False)} "
                f"in {time.perf_counter()-t_call:.1f}s",
                flush=True,
            )
        except Exception as e:
            trace.append({"decision": {"action": "model_call_observe"}, "result": {"error": str(e)}})
            break
        # Anti-loop correction: avoid repeated list_aircraft when evidence already exists.
        if decision.action == "list_aircraft" and int(progress.get("consecutive_list_aircraft", 0)) >= 2:
            if progress.get("has_inspect_flight") and progress.get("has_test_condition"):
                tested = progress.get("last_test_condition_spec")
                decision.action = "finalize_condition"
                decision.condition_spec = ConditionSpec.model_validate(tested) if isinstance(tested, dict) else default_spec
                decision.note = "auto_finalize_after_repeated_list_aircraft"
                print("[observe-worker] auto_override list_aircraft -> finalize_condition", flush=True)
            else:
                decision.action = "inspect_flight"
                decision.aircraft_id = decision.aircraft_id or _split_aircraft(split_manifest, "train")[0]
                decision.folder_label = int(decision.folder_label) if int(decision.folder_label) in (0, 1) else 1
                decision.max_items = 1
                decision.note = "auto_inspect_after_repeated_list_aircraft"
                print("[observe-worker] auto_override list_aircraft -> inspect_flight", flush=True)
        try:
            if decision.action == "list_aircraft":
                result = _tool_list_aircraft(index_df, split_manifest, "train")
            elif decision.action == "list_flights":
                result = _tool_list_flights(
                    index_df,
                    split_manifest,
                    "train",
                    decision.aircraft_id,
                    decision.folder_label,
                    decision.max_items,
                )
            elif decision.action == "inspect_flight":
                result = _tool_inspect_flight(
                    index_df,
                    split_manifest,
                    "train",
                    decision.aircraft_id,
                    decision.folder_label,
                    decision.flight_path,
                    cfg.agent_tools.inspect_max_rows,
                )
            elif decision.action == "test_condition":
                aircraft_ids = decision.aircraft_ids or _split_aircraft(split_manifest, "train")[:3]
                spec = decision.condition_spec or default_spec
                result = _test_condition_on_aircraft(index_df, aircraft_ids, cfg, spec)
            elif decision.action == "finalize_condition":
                spec = decision.condition_spec or default_spec
                return spec, {"tool_trace": trace, "final_note": decision.note}
            else:
                result = {"error": f"Unsupported action {decision.action}"}
            trace.append({"decision": decision.model_dump(), "result": result})
            print(f"[observe-worker] tool_done={decision.action}", flush=True)
        except Exception as e:
            trace.append({"decision": decision.model_dump(), "result": {"error": str(e)}})
    return default_spec, {"tool_trace": trace, "final_note": "fallback_default_condition"}


def propose_rule(
    cfg: AppConfig,
    index_df: pd.DataFrame,
    split_manifest: dict,
    condition_spec: ConditionSpec,
    history: list[dict],
    knowledge_index: dict,
    workspace: str,
    round_no: int,
) -> tuple[RuleBundle, dict]:
    kn = plan_knowledge(
        cfg,
        "propose_rule",
        {"round_no": round_no, "condition_spec": condition_spec.model_dump(), "history": history[-3:]},
        knowledge_index,
        workspace,
    )
    trace: list[dict] = []
    default_rule = RuleBundle(
        condition_spec=condition_spec,
        rule_name="fallback_rule",
        score_expression="pressure_abs_diff__p95 * 0.5 + (1.0 - affected_hpv_open_ratio)",
        decision_threshold=0.8,
        folder_vote_threshold=cfg.loop.folder_vote_threshold,
        rationale="Fallback rule from pressure asymmetry and HPV opening behavior.",
    )
    for _ in range(cfg.agent_tools.hypothesize_max_steps):
        compact_trace = _compact_trace_for_prompt(trace[-6:])
        progress = _progress_summary_for_hypothesize(trace)
        print(f"[hypothesize-worker] round={round_no} step_start trace_steps={len(trace)}", flush=True)
        print(
            f"[hypothesize-worker] progress counts={progress.get('counts', {})} "
            f"consecutive_list_aircraft={progress.get('consecutive_list_aircraft', 0)} "
            f"has_inspect={progress.get('has_inspect_flight')} "
            f"has_test={progress.get('has_test_rule')}",
            flush=True,
        )
        prompt = (
            "You are in HYPOTHESIZE stage. Decide one tool action now.\n"
            "Return HypothesizeAction JSON only.\n\n"
            f"{MECHANISM_PROMPT}\n"
            "Allowed actions: list_aircraft, list_flights, inspect_flight, test_rule, finalize_rule.\n"
            "All tool calls must stay on training split only.\n"
            "Action policy:\n"
            "- Do not repeat list_aircraft if you already listed aircraft once.\n"
            "- After inspect_flight and test_rule, prefer finalize_rule with a concrete rule_bundle.\n"
            "Rule uses window feature columns. score_expression can be numeric or boolean.\n"
            "If numeric, window predicted as class1 when score_expression >= decision_threshold.\n"
            "Examples:\n"
            "0.6*pressure_abs_diff__p95 + 0.4*(1-affected_hpv_open_ratio)\n"
            "(pressure_abs_diff__p95 > 12) & (affected_hpv_open_ratio < 0.7)\n\n"
            f"fixed_condition_spec={json.dumps(condition_spec.model_dump(), ensure_ascii=False)}\n"
            f"knowledge_context={json.dumps(kn, ensure_ascii=False)}\n"
            f"history={json.dumps(history[-5:], ensure_ascii=False)}\n"
            f"progress_summary={json.dumps(progress, ensure_ascii=False)}\n"
            f"last_trace={json.dumps(compact_trace, ensure_ascii=False)}\n"
        )
        try:
            t_call = time.perf_counter()
            print(f"[hypothesize-worker] model_call prompt_chars={len(prompt)}", flush=True)
            decision = run_openharness_structured(prompt, HypothesizeAction, cfg, workspace)
            decision_info = {
                "action": decision.action,
                "aircraft_id": decision.aircraft_id,
                "folder_label": decision.folder_label,
                "max_items": decision.max_items,
                "flight_path": decision.flight_path,
                "n_aircraft_ids": len(decision.aircraft_ids or []),
                "has_rule_bundle": decision.rule_bundle is not None,
            }
            print(
                f"[hypothesize-worker] model_decision={json.dumps(decision_info, ensure_ascii=False)} "
                f"in {time.perf_counter()-t_call:.1f}s",
                flush=True,
            )
        except Exception as e:
            trace.append({"decision": {"action": "model_call_hypothesize"}, "result": {"error": str(e)}})
            break
        if decision.action == "list_aircraft" and int(progress.get("consecutive_list_aircraft", 0)) >= 2:
            if progress.get("has_test_rule"):
                decision.action = "finalize_rule"
                decision.rule_bundle = decision.rule_bundle or default_rule
                decision.note = "auto_finalize_after_repeated_list_aircraft"
                print("[hypothesize-worker] auto_override list_aircraft -> finalize_rule", flush=True)
            else:
                decision.action = "test_rule"
                decision.aircraft_ids = decision.aircraft_ids or _split_aircraft(split_manifest, "train")[:4]
                decision.rule_bundle = decision.rule_bundle or default_rule
                decision.note = "auto_test_rule_after_repeated_list_aircraft"
                print("[hypothesize-worker] auto_override list_aircraft -> test_rule", flush=True)
        try:
            if decision.action == "list_aircraft":
                result = _tool_list_aircraft(index_df, split_manifest, "train")
            elif decision.action == "list_flights":
                result = _tool_list_flights(
                    index_df,
                    split_manifest,
                    "train",
                    decision.aircraft_id,
                    decision.folder_label,
                    decision.max_items,
                )
            elif decision.action == "inspect_flight":
                result = _tool_inspect_flight(
                    index_df,
                    split_manifest,
                    "train",
                    decision.aircraft_id,
                    decision.folder_label,
                    decision.flight_path,
                    cfg.agent_tools.inspect_max_rows,
                )
            elif decision.action == "test_rule":
                rule = decision.rule_bundle or default_rule
                rule.condition_spec = condition_spec
                aircraft_ids = decision.aircraft_ids or _split_aircraft(split_manifest, "train")[:4]
                result = _test_rule_on_aircraft(index_df, aircraft_ids, cfg, rule)
            elif decision.action == "finalize_rule":
                rule = decision.rule_bundle or default_rule
                rule.condition_spec = condition_spec
                if not rule.folder_vote_threshold:
                    rule.folder_vote_threshold = cfg.loop.folder_vote_threshold
                return rule, {"tool_trace": trace, "final_note": decision.note}
            else:
                result = {"error": f"Unsupported action {decision.action}"}
            trace.append({"decision": decision.model_dump(), "result": result})
            print(f"[hypothesize-worker] tool_done={decision.action}", flush=True)
        except Exception as e:
            trace.append({"decision": decision.model_dump(), "result": {"error": str(e)}})
    return default_rule, {"tool_trace": trace, "final_note": "fallback_default_rule"}


def reflect(
    cfg: AppConfig,
    rule: RuleBundle,
    validation: dict,
    history: list[dict],
    knowledge_index: dict,
    workspace: str,
) -> ReflectionDecision:
    kn = plan_knowledge(
        cfg,
        "reflect",
        {"rule": rule.model_dump(), "validation": validation, "history": history[-3:]},
        knowledge_index,
        workspace,
    )
    prompt = (
        "Decide whether to restart from observe, restart from hypothesize, or stop.\n"
        "Use stop only when val metrics are good enough and stable.\n\n"
        f"{MECHANISM_PROMPT}\n"
        f"rule={json.dumps(rule.model_dump(), ensure_ascii=False)}\n"
        f"validation={json.dumps(validation, ensure_ascii=False)}\n"
        f"history={json.dumps(history[-5:], ensure_ascii=False)}\n"
        f"knowledge_context={json.dumps(kn, ensure_ascii=False)}\n"
    )
    try:
        return run_openharness_structured(prompt, ReflectionDecision, cfg, workspace)
    except Exception:
        return ReflectionDecision(restart_from="observe", reason="Fallback: re-observe when reflection call fails.")
