from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field


Operator = Literal[">", ">=", "<", "<="]


class Predicate(BaseModel):
    feature: str
    operator: Operator
    threshold: float


class ConditionSpec(BaseModel):
    name: str = "candidate_condition"
    description: str = ""
    expression: str = "True"
    min_segment_len: int = 60


class RuleBundle(BaseModel):
    condition_spec: ConditionSpec
    rule_name: str = "candidate_rule"
    score_expression: str = "0.0"
    decision_threshold: float = 0.5
    folder_vote_threshold: float = 0.5
    rationale: str = ""


class KnowledgeRequest(BaseModel):
    queries: List[str] = Field(default_factory=list)
    doc_ids: List[str] = Field(default_factory=list)


class ReflectionDecision(BaseModel):
    restart_from: Literal["observe", "hypothesize", "stop"] = "hypothesize"
    reason: str = ""
    next_queries: List[str] = Field(default_factory=list)


class ObserveAction(BaseModel):
    action: Literal["list_aircraft", "list_flights", "inspect_flight", "test_condition", "profile_condition_windows", "finalize_condition"] = "finalize_condition"
    split: Literal["train"] = "train"
    aircraft_id: str = ""
    folder_label: int = 0
    max_items: int = 20
    flight_path: str = ""
    aircraft_ids: List[str] = Field(default_factory=list)
    condition_spec: ConditionSpec | None = None
    note: str = ""


class HypothesizeAction(BaseModel):
    action: Literal[
        "list_aircraft",
        "list_flights",
        "inspect_flight",
        "profile_condition_windows",
        "mine_candidate_rules",
        "robust_normal_baseline",
        "isolation_forest_windows",
        "lag_correlation_flight",
        "change_point_flight",
        "matrix_profile_discords",
        "test_rule",
        "finalize_rule",
    ] = "finalize_rule"
    split: Literal["train"] = "train"
    aircraft_id: str = ""
    folder_label: int = 0
    max_items: int = 20
    flight_path: str = ""
    aircraft_ids: List[str] = Field(default_factory=list)
    rule_bundle: RuleBundle | None = None
    contamination: float = 0.20
    max_lag: int = 300
    columns: List[str] = Field(default_factory=list)
    penalty: float = 8.0
    column: str = "pressure_diff"
    subseq_len: int = 120
    top_k: int = 5
    note: str = ""


class WindowRecord(BaseModel):
    aircraft_id: str
    folder_label: int
    flight_path: str
    faulty_side: Literal["left", "right"]
    window_id: str
    start_idx: int
    end_idx: int
    n_rows: int
    features: Dict[str, float]


class FolderPrediction(BaseModel):
    aircraft_id: str
    true_label: int
    predicted_label: int
    mean_flight_score: float
    n_flights: int
