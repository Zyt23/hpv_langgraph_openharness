from __future__ import annotations

import time
from pathlib import Path

from pydantic import BaseModel

from .config import AppConfig
from .openharness_adapter import run_openharness_structured


class PingResponse(BaseModel):
    ok: bool


def check_openharness_connectivity(cfg: AppConfig, workspace: str | Path) -> dict:
    if not cfg.openharness.enabled:
        return {"enabled": False, "ok": True, "reason": "openharness_disabled"}

    probe_cfg = cfg.model_copy(deep=True)
    probe_cfg.openharness.timeout_sec = min(
        int(cfg.openharness.preflight_timeout_sec),
        int(cfg.openharness.timeout_sec),
    )
    probe_cfg.openharness.max_retries = 0

    prompt = (
        "Connectivity preflight. "
        "Return ONLY JSON object: {\"ok\": true}"
    )
    t0 = time.perf_counter()
    resp = run_openharness_structured(prompt, PingResponse, probe_cfg, workspace)
    elapsed = time.perf_counter() - t0
    return {
        "enabled": True,
        "ok": bool(resp.ok),
        "elapsed_sec": round(elapsed, 3),
        "model": cfg.openharness.model,
        "command": cfg.openharness.command,
        "timeout_sec": probe_cfg.openharness.timeout_sec,
    }
