from __future__ import annotations

import hashlib
from pathlib import Path
import time
from typing import Literal

import pandas as pd
from langgraph.graph import END, START, StateGraph

from .artifacts import ensure_dir, load_json, save_df, save_json
from .config import AppConfig
from .indexing import build_aircraft_index, build_split_manifest
from .knowledge_base import build_knowledge_index
from .metrics import classify_folder_units, format_metrics_table, summarize_folder_metrics
from .preflight import check_openharness_connectivity
from .schema import ConditionSpec, RuleBundle
from .state import WorkflowState
from .workers import choose_condition, propose_rule, reflect


def _extract_observe_used_aircraft_ids(observe_trace: dict) -> list[str]:
    used: set[str] = set()
    trace = observe_trace.get("tool_trace", []) if isinstance(observe_trace, dict) else []
    for item in trace:
        if not isinstance(item, dict):
            continue
        d = item.get("decision", {}) if isinstance(item.get("decision", {}), dict) else {}
        r = item.get("result", {}) if isinstance(item.get("result", {}), dict) else {}
        aid = d.get("aircraft_id")
        if aid:
            used.add(str(aid))
        for x in d.get("aircraft_ids", []) if isinstance(d.get("aircraft_ids", []), list) else []:
            if x:
                used.add(str(x))
        ra = r.get("aircraft_id")
        if ra:
            used.add(str(ra))
        for sf in r.get("sampled_flights", []) if isinstance(r.get("sampled_flights", []), list) else []:
            if isinstance(sf, dict) and sf.get("aircraft_id"):
                used.add(str(sf["aircraft_id"]))
        for x in r.get("sampled_aircraft_ids", []) if isinstance(r.get("sampled_aircraft_ids", []), list) else []:
            if x:
                used.add(str(x))
    return sorted(used)


def _extract_condition_eval_summary(observe_trace: dict) -> dict:
    trace = observe_trace.get("tool_trace", []) if isinstance(observe_trace, dict) else []
    for item in reversed(trace):
        if not isinstance(item, dict):
            continue
        d = item.get("decision", {}) if isinstance(item.get("decision", {}), dict) else {}
        r = item.get("result", {}) if isinstance(item.get("result", {}), dict) else {}
        if d.get("action") != "test_condition" or not isinstance(r, dict):
            continue
        return {
            "n_windows": r.get("n_windows", 0),
            "windows_per_label": r.get("windows_per_label", {}),
            "n_aircraft_sampled": r.get("n_aircraft_sampled", 0),
            "n_aircraft_with_windows": r.get("n_aircraft_with_windows", 0),
            "faulty_side_window_coverage": r.get("faulty_side_window_coverage", {}),
            "sampled_aircraft_ids": r.get("sampled_aircraft_ids", []),
            "excluded_feature_prefixes": r.get("excluded_feature_prefixes", []),
            "top_features": (r.get("top_features", []) or [])[:12],
        }
    return {}


def _inner_validation_aircraft(split_manifest: dict) -> tuple[list[str], list[str]]:
    train = list(split_manifest["train_aircraft"])
    if len(train) <= 3:
        return train, train
    n_val = max(2, min(3, len(train) // 3))
    inner_val = train[-n_val:]
    inner_train = train[:-n_val]
    return inner_train, inner_val


def bootstrap_node(state: WorkflowState) -> WorkflowState:
    t0 = time.perf_counter()
    cfg = AppConfig.from_yaml(state["config_path"])
    run_dir = ensure_dir(Path(state["run_dir_override"])) if state.get("run_dir_override") else ensure_dir(cfg.run_dir)
    preflight_workspace = ensure_dir(run_dir / cfg.runtime.workspace_subdir / "preflight")
    try:
        preflight = check_openharness_connectivity(cfg, preflight_workspace)
        save_json(run_dir / "preflight_openharness.json", preflight)
        print(
            f"[preflight] ok={preflight.get('ok')} elapsed={preflight.get('elapsed_sec', 0)}s "
            f"model={preflight.get('model', '')}",
            flush=True,
        )
    except Exception as e:
        err = {"ok": False, "error": str(e)}
        save_json(run_dir / "preflight_openharness.json", err)
        raise RuntimeError(
            "OpenHarness preflight failed. "
            "Please check API key/network/proxy before running full workflow. "
            f"error={e}"
        )
    shared_cache_dir = ensure_dir(cfg.paths.artifacts_root / "_shared_cache")
    data_key = hashlib.md5(str(cfg.paths.data_root.resolve()).encode("utf-8")).hexdigest()[:12]
    split_key = hashlib.md5(f"{data_key}|{cfg.split.seed}|{cfg.split.train_aircraft_count}".encode("utf-8")).hexdigest()[:12]
    know_key = hashlib.md5(("|".join(sorted(cfg.paths.knowledge_globs))).encode("utf-8")).hexdigest()[:12]

    aircraft_index_path = shared_cache_dir / f"aircraft_index_{data_key}.parquet"
    split_manifest_path = shared_cache_dir / f"split_manifest_{split_key}.json"
    knowledge_index_path = shared_cache_dir / f"knowledge_index_{know_key}.json"
    latest_state_path = run_dir / "state_latest.json"

    if not aircraft_index_path.exists():
        print("[bootstrap] building aircraft_index cache ...", flush=True)
        idx = build_aircraft_index(cfg.paths.data_root)
        save_df(aircraft_index_path, idx)
        print(f"[bootstrap] aircraft_index cache created: {aircraft_index_path}", flush=True)
    else:
        print(f"[bootstrap] reuse aircraft_index cache: {aircraft_index_path}", flush=True)
        idx = pd.read_parquet(aircraft_index_path)
    if not split_manifest_path.exists():
        print("[bootstrap] building split_manifest cache ...", flush=True)
        save_json(split_manifest_path, build_split_manifest(idx, cfg))
        print(f"[bootstrap] split_manifest cache created: {split_manifest_path}", flush=True)
    else:
        print(f"[bootstrap] reuse split_manifest cache: {split_manifest_path}", flush=True)
    if not knowledge_index_path.exists():
        print("[bootstrap] building knowledge_index cache ...", flush=True)
        save_json(knowledge_index_path, build_knowledge_index(cfg))
        print(f"[bootstrap] knowledge_index cache created: {knowledge_index_path}", flush=True)
    else:
        print(f"[bootstrap] reuse knowledge_index cache: {knowledge_index_path}", flush=True)
    if not latest_state_path.exists():
        save_json(
            latest_state_path,
            {
                "current_round": 1,
                "best_rule": None,
                "best_f1_class1": 0.0,
                "plateau_rounds": 0,
                "stop": False,
                "stop_reason": "",
                "history": [],
            },
        )
    latest = load_json(latest_state_path)
    out = {
        "run_dir": str(run_dir),
        "aircraft_index_path": str(aircraft_index_path),
        "split_manifest_path": str(split_manifest_path),
        "knowledge_index_path": str(knowledge_index_path),
        "current_round": int(latest.get("current_round", 1)),
        "best_rule": latest.get("best_rule"),
        "best_f1_class1": float(latest.get("best_f1_class1", 0.0)),
        "plateau_rounds": int(latest.get("plateau_rounds", 0)),
        "stop": bool(latest.get("stop", False)),
        "stop_reason": str(latest.get("stop_reason", "")),
        "history": latest.get("history", []),
    }
    print(f"[bootstrap] run_dir={run_dir} loaded in {time.perf_counter()-t0:.1f}s", flush=True)
    return out


def observe_node(state: WorkflowState) -> WorkflowState:
    t0 = time.perf_counter()
    cfg = AppConfig.from_yaml(state["config_path"])
    run_dir = Path(state["run_dir"])
    round_dir = ensure_dir(run_dir / f"round_{int(state['current_round']):03d}")
    workspace = ensure_dir(run_dir / cfg.runtime.workspace_subdir / f"round_{int(state['current_round']):03d}")
    index_df = pd.read_parquet(state["aircraft_index_path"])
    split_manifest = load_json(state["split_manifest_path"])
    knowledge_index = load_json(state["knowledge_index_path"])
    observe_trace_live_path = round_dir / "observe_trace_live.json"

    condition_spec, observe_trace = choose_condition(
        cfg,
        index_df,
        split_manifest,
        knowledge_index,
        str(workspace),
        int(state["current_round"]),
        live_trace_path=str(observe_trace_live_path),
    )
    selected_condition_path = round_dir / "selected_condition.json"
    observe_trace_path = round_dir / "observe_trace.json"
    save_json(selected_condition_path, condition_spec.model_dump())
    save_json(observe_trace_path, observe_trace)
    save_json(observe_trace_live_path, observe_trace)
    out = {
        "selected_condition_path": str(selected_condition_path),
        "observed_schema_path": str(observe_trace_path),
        "observed_schema_live_path": str(observe_trace_live_path),
    }
    print(f"[observe] round={int(state['current_round'])} trace_steps={len(observe_trace.get('tool_trace', []))} done in {time.perf_counter()-t0:.1f}s")
    return out


def hypothesize_node(state: WorkflowState) -> WorkflowState:
    t0 = time.perf_counter()
    cfg = AppConfig.from_yaml(state["config_path"])
    run_dir = Path(state["run_dir"])
    round_dir = ensure_dir(run_dir / f"round_{int(state['current_round']):03d}")
    workspace = ensure_dir(run_dir / cfg.runtime.workspace_subdir / f"round_{int(state['current_round']):03d}")
    knowledge_index = load_json(state["knowledge_index_path"])
    index_df = pd.read_parquet(state["aircraft_index_path"])
    split_manifest = load_json(state["split_manifest_path"])
    condition_spec = ConditionSpec.model_validate(load_json(state["selected_condition_path"]))
    hypothesize_trace_live_path = round_dir / "hypothesize_trace_live.json"

    rule, trace = propose_rule(
        cfg,
        index_df,
        split_manifest,
        condition_spec,
        state.get("history", []),
        knowledge_index,
        str(workspace),
        int(state["current_round"]),
        live_trace_path=str(hypothesize_trace_live_path),
    )
    candidate_rule_path = round_dir / "candidate_rule.json"
    hypothesize_trace_path = round_dir / "hypothesize_trace.json"
    save_json(candidate_rule_path, rule.model_dump())
    save_json(hypothesize_trace_path, trace)
    save_json(hypothesize_trace_live_path, trace)
    out = {
        "candidate_rule_path": str(candidate_rule_path),
        "observation_summary_path": str(hypothesize_trace_path),
        "hypothesize_trace_live_path": str(hypothesize_trace_live_path),
    }
    print(f"[hypothesize] round={int(state['current_round'])} trace_steps={len(trace.get('tool_trace', []))} done in {time.perf_counter()-t0:.1f}s")
    return out


def validate_node(state: WorkflowState) -> WorkflowState:
    t0 = time.perf_counter()
    cfg = AppConfig.from_yaml(state["config_path"])
    run_dir = Path(state["run_dir"])
    round_dir = ensure_dir(run_dir / f"round_{int(state['current_round']):03d}")
    index_df = pd.read_parquet(state["aircraft_index_path"])
    split_manifest = load_json(state["split_manifest_path"])
    rule = RuleBundle.model_validate(load_json(state["candidate_rule_path"]))
    observe_trace = {}
    observed_schema_path = state.get("observed_schema_path", "")
    if observed_schema_path and Path(observed_schema_path).exists():
        observe_trace = load_json(observed_schema_path)
    used_in_observe = _extract_observe_used_aircraft_ids(observe_trace)
    train_all = list(split_manifest["train_aircraft"])
    train_eval = [aid for aid in train_all if aid not in set(used_in_observe)]
    if not train_eval:
        train_eval = train_all

    print(
        f"[validate] round={int(state['current_round'])} "
        f"max_flights_per_folder={cfg.loop.validation_max_flights_per_folder} "
        f"train_eval_aircraft={len(train_eval)}/{len(train_all)}"
    )
    train_preds = classify_folder_units(
        index_df,
        train_eval,
        rule,
        cfg,
        max_flights_per_folder=cfg.loop.validation_max_flights_per_folder,
        verbose=True,
        tag="validate_train",
    )
    val_preds = classify_folder_units(
        index_df,
        split_manifest["holdout_aircraft"],
        rule,
        cfg,
        max_flights_per_folder=cfg.loop.validation_max_flights_per_folder,
        verbose=True,
        tag="validate_val",
    )
    report = {
        "training": summarize_folder_metrics(train_preds),
        "val": summarize_folder_metrics(val_preds),
        "n_training_units": len(train_preds),
        "n_val_units": len(val_preds),
        "observe_used_train_aircraft": used_in_observe,
        "validate_train_aircraft": train_eval,
        "n_observe_used_train_aircraft": len(used_in_observe),
    }
    validation_report_path = round_dir / "validation_report.json"
    save_json(validation_report_path, report)
    print(f"[validate] round={int(state['current_round'])} done in {time.perf_counter()-t0:.1f}s")
    return {"validation_report_path": str(validation_report_path)}


def reflect_node(state: WorkflowState) -> WorkflowState:
    t0 = time.perf_counter()
    cfg = AppConfig.from_yaml(state["config_path"])
    run_dir = Path(state["run_dir"])
    round_dir = ensure_dir(run_dir / f"round_{int(state['current_round']):03d}")
    workspace = ensure_dir(run_dir / cfg.runtime.workspace_subdir / f"round_{int(state['current_round']):03d}")
    latest_state_path = run_dir / "state_latest.json"
    knowledge_index = load_json(state["knowledge_index_path"])
    rule = RuleBundle.model_validate(load_json(state["candidate_rule_path"]))
    validation = load_json(state["validation_report_path"])
    observe_trace = {}
    observed_schema_path = state.get("observed_schema_path", "")
    if observed_schema_path and Path(observed_schema_path).exists():
        observe_trace = load_json(observed_schema_path)
    condition_eval_summary = _extract_condition_eval_summary(observe_trace)
    current_f1 = float(validation.get("val", {}).get("F1_1", 0.0))
    best_f1 = float(state.get("best_f1_class1", 0.0))

    reflection = reflect(cfg, rule, validation, state.get("history", []), knowledge_index, str(workspace))
    reflection_path = round_dir / "reflection.json"
    save_json(reflection_path, reflection.model_dump())

    improved = (state.get("best_rule") is None) or (current_f1 > best_f1 + cfg.loop.min_improvement)
    best_rule = rule.model_dump() if improved else state.get("best_rule")
    best_f1 = current_f1 if improved else best_f1
    plateau = 0 if improved else int(state.get("plateau_rounds", 0)) + 1

    hard_stop = False
    hard_reason = ""
    max_rounds = int(state.get("max_rounds_override", 0)) or cfg.loop.max_rounds
    if best_f1 >= cfg.loop.target_f1_class1:
        hard_stop = True
        hard_reason = f"class1_f1_reached_target_{cfg.loop.target_f1_class1}"
    elif int(state["current_round"]) >= max_rounds:
        hard_stop = True
        hard_reason = "max_rounds_reached"
    elif plateau >= cfg.loop.patience_rounds:
        hard_stop = True
        hard_reason = f"no_material_improvement_for_{cfg.loop.patience_rounds}_rounds"

    stop = hard_stop or reflection.restart_from == "stop"
    stop_reason = hard_reason if hard_stop else (reflection.reason if reflection.restart_from == "stop" else "")
    history = list(state.get("history", [])) + [
        {
            "round": int(state["current_round"]),
            "condition_spec": rule.condition_spec.model_dump(),
            "rule": rule.model_dump(),
            "validation": validation,
            "condition_eval_summary": condition_eval_summary,
            "reflection": reflection.model_dump(),
        }
    ]
    latest = {
        "current_round": int(state["current_round"]) + (0 if stop else 1),
        "best_rule": best_rule,
        "best_f1_class1": best_f1,
        "plateau_rounds": plateau,
        "stop": stop,
        "stop_reason": stop_reason,
        "history": history,
    }
    save_json(latest_state_path, latest)
    if best_rule is not None:
        save_json(run_dir / "best_rule.json", best_rule)
    out = {
        "reflection_path": str(reflection_path),
        "best_rule": best_rule,
        "best_f1_class1": best_f1,
        "plateau_rounds": plateau,
        "stop": stop,
        "stop_reason": stop_reason,
        "history": history,
        "current_round": latest["current_round"],
    }
    print(
        f"[reflect] round={int(state['current_round'])} "
        f"val_F1_1={current_f1:.4f} best_F1_1={best_f1:.4f} stop={stop} "
        f"done in {time.perf_counter()-t0:.1f}s"
    )
    return out


def final_test_node(state: WorkflowState) -> WorkflowState:
    t0 = time.perf_counter()
    cfg = AppConfig.from_yaml(state["config_path"])
    run_dir = Path(state["run_dir"])
    split_manifest = load_json(state["split_manifest_path"])
    index_df = pd.read_parquet(state["aircraft_index_path"])
    best_rule_raw = state.get("best_rule")
    if not best_rule_raw:
        return {}
    rule = RuleBundle.model_validate(best_rule_raw)
    print(
        f"[final_test] max_flights_per_folder={cfg.loop.final_eval_max_flights_per_folder}"
    )
    training_preds = classify_folder_units(
        index_df,
        split_manifest["train_aircraft"],
        rule,
        cfg,
        max_flights_per_folder=cfg.loop.final_eval_max_flights_per_folder,
        verbose=True,
        tag="final_train",
    )
    holdout_preds = classify_folder_units(
        index_df,
        split_manifest["holdout_aircraft"],
        rule,
        cfg,
        max_flights_per_folder=cfg.loop.final_eval_max_flights_per_folder,
        verbose=True,
        tag="final_holdout",
    )
    training_metrics = summarize_folder_metrics(training_preds)
    holdout_metrics = summarize_folder_metrics(holdout_preds)
    save_json(run_dir / "training_eval.json", training_metrics)
    save_json(run_dir / "holdout_eval.json", holdout_metrics)
    metrics_txt = format_metrics_table({"training": training_metrics, "holdout": holdout_metrics})
    metrics_table_path = run_dir / "metrics_table.txt"
    metrics_table_path.write_text(metrics_txt, encoding="utf-8")
    print(f"[final_test] done in {time.perf_counter()-t0:.1f}s")
    return {"metrics_table_path": str(metrics_table_path)}


def route_after_reflect(state: WorkflowState) -> Literal["observe", "hypothesize", "final_test"]:
    if state.get("stop", False):
        return "final_test"
    reflection = load_json(state["reflection_path"])
    return reflection.get("restart_from", "hypothesize")


def build_graph():
    graph = StateGraph(WorkflowState)
    graph.add_node("bootstrap", bootstrap_node)
    graph.add_node("observe", observe_node)
    graph.add_node("hypothesize", hypothesize_node)
    graph.add_node("validate", validate_node)
    graph.add_node("reflect", reflect_node)
    graph.add_node("final_test", final_test_node)
    graph.add_edge(START, "bootstrap")
    graph.add_edge("bootstrap", "observe")
    graph.add_edge("observe", "hypothesize")
    graph.add_edge("hypothesize", "validate")
    graph.add_edge("validate", "reflect")
    graph.add_conditional_edges(
        "reflect",
        route_after_reflect,
        {"observe": "observe", "hypothesize": "hypothesize", "final_test": "final_test"},
    )
    graph.add_edge("final_test", END)
    return graph.compile()
