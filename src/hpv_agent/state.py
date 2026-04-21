from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class WorkflowState(TypedDict, total=False):
    config_path: str
    run_dir_override: str
    max_rounds_override: int
    run_dir: str
    aircraft_index_path: str
    split_manifest_path: str
    knowledge_index_path: str
    current_round: int
    best_rule: Dict[str, Any]
    best_f1_class1: float
    plateau_rounds: int
    stop: bool
    stop_reason: str
    history: List[Dict[str, Any]]
    observed_schema_path: str
    selected_condition_path: str
    observation_summary_path: str
    candidate_rule_path: str
    validation_report_path: str
    reflection_path: str
    metrics_table_path: str
