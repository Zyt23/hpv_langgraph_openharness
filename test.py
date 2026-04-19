from __future__ import annotations

import argparse
from pathlib import Path

from hpv_agent.config import AppConfig
from hpv_agent.indexing import build_aircraft_index
from hpv_agent.metrics import classify_folder_units, format_metrics_table, summarize_folder_metrics
from hpv_agent.schema import RuleBundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default="./test")
    parser.add_argument("--rule", default="")
    parser.add_argument("--max-flights-per-folder", type=int, default=20)
    args = parser.parse_args()

    cfg = AppConfig.from_yaml(args.config)
    cfg.paths.data_root = Path(args.data_root).resolve()
    index_df = build_aircraft_index(cfg.paths.data_root)
    rule_path = Path(args.rule) if args.rule else (cfg.run_dir / "best_rule.json")
    if not rule_path.exists():
        raise FileNotFoundError(f"Rule file not found: {rule_path}")
    rule = RuleBundle.model_validate_json(rule_path.read_text(encoding="utf-8"))
    aircraft_ids = sorted(index_df["aircraft_id"].unique().tolist())
    preds = classify_folder_units(
        index_df,
        aircraft_ids,
        rule,
        cfg,
        max_flights_per_folder=args.max_flights_per_folder,
        verbose=True,
        tag="test",
    )
    metrics = summarize_folder_metrics(preds)
    print(format_metrics_table({"test": metrics}))


if __name__ == "__main__":
    main()
