from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from .config import AppConfig


def parse_faulty_side(aircraft_dir_name: str) -> str:
    tail = aircraft_dir_name.strip().split("-")[-1]
    if tail == "1":
        return "left"
    if tail == "2":
        return "right"
    raise ValueError(f"Cannot parse faulty side from aircraft folder name: {aircraft_dir_name}")


def build_aircraft_index(data_root: str | Path) -> pd.DataFrame:
    data_root = Path(data_root)
    rows = []
    for aircraft_dir in sorted([p for p in data_root.iterdir() if p.is_dir()]):
        try:
            faulty_side = parse_faulty_side(aircraft_dir.name)
        except ValueError:
            continue
        for label_name in ["0", "1"]:
            label_dir = aircraft_dir / label_name
            if not label_dir.exists():
                continue
            for pq in sorted(label_dir.glob("*.parquet")):
                rows.append({
                    "aircraft_id": aircraft_dir.name,
                    "faulty_side": faulty_side,
                    "folder_label": int(label_name),
                    "flight_path": str(pq.resolve()),
                })
    if not rows:
        raise FileNotFoundError(f"No parquet flights found under {data_root}")
    return pd.DataFrame(rows)


def build_split_manifest(index_df: pd.DataFrame, cfg: AppConfig) -> dict:
    rng = random.Random(cfg.split.seed)
    aircraft_ids = sorted(index_df["aircraft_id"].unique().tolist())
    rng.shuffle(aircraft_ids)
    train_count = int(cfg.split.train_aircraft_count)
    validation_count = max(0, int(cfg.split.validation_aircraft_count))
    holdout_count = max(0, int(cfg.split.holdout_aircraft_count))
    if train_count >= len(aircraft_ids):
        raise ValueError("train_aircraft_count must be smaller than total aircraft count")
    train_aircraft = sorted(aircraft_ids[:train_count])
    remaining = aircraft_ids[train_count:]
    if not remaining:
        raise ValueError("Need at least one non-training aircraft for validation/holdout")

    if validation_count <= 0:
        validation_count = max(1, len(remaining) // 2)
    validation_count = min(validation_count, len(remaining))
    validation_aircraft = remaining[:validation_count]
    rest = remaining[validation_count:]

    if holdout_count > 0:
        holdout_aircraft = rest[:holdout_count]
    else:
        holdout_aircraft = rest

    if not holdout_aircraft and validation_aircraft:
        holdout_aircraft = [validation_aircraft.pop()]
    if not validation_aircraft and len(train_aircraft) > 1:
        validation_aircraft = [train_aircraft.pop()]
    if not validation_aircraft:
        raise ValueError("Could not create validation split; reduce train_aircraft_count")
    if not holdout_aircraft:
        raise ValueError("Could not create holdout split; reduce train_aircraft_count or validation_aircraft_count")

    return {
        "train_aircraft": sorted(train_aircraft),
        "validation_aircraft": sorted(validation_aircraft),
        "holdout_aircraft": sorted(holdout_aircraft),
    }
