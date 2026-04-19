#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

if ! command -v python >/dev/null 2>&1; then
  echo "python not found in PATH"
  exit 1
fi

if ! command -v openharness >/dev/null 2>&1; then
  echo "OpenHarness CLI not found in PATH"
  exit 1
fi

python -m hpv_agent.run_agent --config configs/hpv_openharness.yaml
