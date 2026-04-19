from __future__ import annotations

import random
from typing import List

import pandas as pd

from .config import AppConfig
from .row_context import extract_windows_for_flight
from .schema import ConditionSpec, WindowRecord


def sample_validation_windows(index_df: pd.DataFrame, aircraft_ids: List[str], spec: ConditionSpec, per_class: int, cfg: AppConfig, round_no: int) -> List[WindowRecord]:
    rng = random.Random(cfg.split.seed + 1000 + round_no)
    out: List[WindowRecord] = []
    for label in [0, 1]:
        rows = index_df[(index_df["aircraft_id"].isin(aircraft_ids)) & (index_df["folder_label"] == label)].to_dict(orient="records")
        rng.shuffle(rows)
        label_windows: List[WindowRecord] = []
        for row in rows:
            ws = extract_windows_for_flight(
                row["flight_path"],
                int(row["folder_label"]),
                row["aircraft_id"],
                row["faulty_side"],
                cfg.features.min_rows_per_window,
                cfg.features.min_segment_len,
                spec,
            )
            rng.shuffle(ws)
            label_windows.extend(ws)
            if len(label_windows) >= per_class:
                break
        out.extend(label_windows[:per_class])
    return out
