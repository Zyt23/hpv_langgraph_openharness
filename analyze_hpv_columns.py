from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq


LEFT_CORE = {"N21", "PRECOOL_PRESS1", "HPV_ENG1_R"}
RIGHT_CORE = {"N22", "PRECOOL_PRESS2", "HPV_ENG2_R"}
LEFT_PRESSURE = {"PUD1", "BMPS1"}
RIGHT_PRESSURE = {"PUD2", "BMPS2"}
COMMON = {"FLIGHT_PHASE", "LATP", "LONP", "ALT_STD", "time_group", "TAIL_NUM"}


def _long_path(path: Path) -> str:
    p = path.resolve()
    s = str(p)
    if s.startswith("\\\\?\\"):
        return s
    if p.drive:
        return "\\\\?\\" + s
    return s


def _iter_parquet(root: Path) -> Iterable[Path]:
    for p in root.rglob("*.parquet"):
        yield p


def _has_side(cols: set[str], core: set[str], pressure_candidates: set[str]) -> bool:
    return core.issubset(cols) and bool(cols.intersection(pressure_candidates))


def main() -> None:
    root = Path("hpv").resolve()
    out_dir = Path("_outputs").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hpv_column_audit.json"

    files = list(_iter_parquet(root))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {root}")

    column_counter: Counter[str] = Counter()
    folder_mode_counter: Counter[str] = Counter()
    side_presence_counter: Counter[str] = Counter()
    missing_common_counter: Counter[str] = Counter()
    pressure_mode_counter: Counter[str] = Counter()
    sample_by_mode: dict[str, str] = {}

    read_ok = 0
    read_fail = 0
    sample_read_fail: list[str] = []

    for i, f in enumerate(files, start=1):
        try:
            cols = set(pq.ParquetFile(_long_path(f)).schema.names)
            read_ok += 1
        except Exception:
            read_fail += 1
            if len(sample_read_fail) < 10:
                sample_read_fail.append(str(f))
            continue
        for c in cols:
            column_counter[c] += 1

        has_left = _has_side(cols, LEFT_CORE, LEFT_PRESSURE)
        has_right = _has_side(cols, RIGHT_CORE, RIGHT_PRESSURE)

        if has_left and has_right:
            mode = "both_sides"
        elif has_left:
            mode = "left_only"
        elif has_right:
            mode = "right_only"
        else:
            mode = "neither_side_core_complete"
        folder_mode_counter[mode] += 1
        sample_by_mode.setdefault(mode, str(f))

        if cols.intersection(LEFT_PRESSURE):
            pressure_mode_counter["left_has_pressure_sensor"] += 1
            if "PUD1" in cols:
                pressure_mode_counter["left_has_PUD1"] += 1
            if "BMPS1" in cols:
                pressure_mode_counter["left_has_BMPS1"] += 1
        if cols.intersection(RIGHT_PRESSURE):
            pressure_mode_counter["right_has_pressure_sensor"] += 1
            if "PUD2" in cols:
                pressure_mode_counter["right_has_PUD2"] += 1
            if "BMPS2" in cols:
                pressure_mode_counter["right_has_BMPS2"] += 1

        for common in COMMON:
            if common in cols:
                side_presence_counter[f"has_{common}"] += 1
            else:
                missing_common_counter[f"missing_{common}"] += 1

        if i % 1000 == 0:
            print(f"[audit] scanned {i}/{len(files)} parquet files")

    total = len(files)
    report = {
        "root": str(root),
        "total_parquet_files": total,
        "parquet_schema_read_ok": read_ok,
        "parquet_schema_read_fail": read_fail,
        "sample_read_fail": sample_read_fail,
        "side_mode_counts": dict(folder_mode_counter),
        "side_mode_ratios": {k: round(v / total, 4) for k, v in folder_mode_counter.items()},
        "pressure_presence_counts": dict(pressure_mode_counter),
        "common_presence_counts": dict(side_presence_counter),
        "common_missing_counts": dict(missing_common_counter),
        "top_columns_by_presence": column_counter.most_common(30),
        "sample_file_by_mode": sample_by_mode,
    }
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[audit] done, report written to: {out_path}")


if __name__ == "__main__":
    main()
