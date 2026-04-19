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
    if cfg.split.train_aircraft_count >= len(aircraft_ids):
        raise ValueError("train_aircraft_count must be smaller than total aircraft count")
    train_aircraft = sorted(aircraft_ids[: cfg.split.train_aircraft_count])
    holdout_aircraft = sorted(aircraft_ids[cfg.split.train_aircraft_count :])
    return {"train_aircraft": train_aircraft, "holdout_aircraft": holdout_aircraft}
