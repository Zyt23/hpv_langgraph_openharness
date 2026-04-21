from __future__ import annotations

import argparse
from pathlib import Path

from .config import AppConfig
from .graph import build_graph


def _next_run_dir(cfg: AppConfig) -> Path:
    root = cfg.paths.artifacts_root
    root.mkdir(parents=True, exist_ok=True)
    base = cfg.paths.run_name
    i = 1
    while True:
        candidate = root / f"{base}_{i:03d}"
        if not candidate.exists():
            return candidate
        i += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume-run", default="", help="Resume a previous run by directory name under artifacts_root")
    parser.add_argument("--run-suffix", default="", help="Optional suffix for a new run directory name")
    parser.add_argument("--max-rounds", type=int, default=0, help="Override max rounds for this run (0 means use config)")
    args = parser.parse_args()

    cfg = AppConfig.from_yaml(args.config)
    run_dir_override: Path
    if args.resume_run:
        run_dir_override = (cfg.paths.artifacts_root / args.resume_run).resolve()
        if not run_dir_override.exists():
            raise FileNotFoundError(f"resume run not found: {run_dir_override}")
        mode = "resume"
    else:
        if args.run_suffix:
            run_dir_override = (cfg.paths.artifacts_root / f"{cfg.paths.run_name}_{args.run_suffix}").resolve()
        else:
            run_dir_override = _next_run_dir(cfg).resolve()
        mode = "new"
    run_dir_override.mkdir(parents=True, exist_ok=True)
    print(f"[run_mode] {mode}")
    print(f"[run_dir] {run_dir_override}")

    graph = build_graph()
    result = graph.invoke(
        {
            "config_path": str(Path(args.config).resolve()),
            "run_dir_override": str(run_dir_override),
            "max_rounds_override": int(args.max_rounds) if args.max_rounds and args.max_rounds > 0 else 0,
        }
    )
    if result.get("metrics_table_path"):
        print(Path(result["metrics_table_path"]).read_text(encoding="utf-8"))
    else:
        print(result)


if __name__ == "__main__":
    main()
