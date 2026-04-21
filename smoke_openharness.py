from __future__ import annotations

import argparse
from pathlib import Path

from hpv_agent.config import AppConfig
from hpv_agent.preflight import check_openharness_connectivity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--workspace", default="./_outputs/_preflight_workspace")
    args = parser.parse_args()

    cfg = AppConfig.from_yaml(args.config)
    result = check_openharness_connectivity(cfg, Path(args.workspace).resolve())
    print(result)


if __name__ == "__main__":
    main()
