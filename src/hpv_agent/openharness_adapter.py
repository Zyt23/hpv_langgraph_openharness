from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
import time
from typing import Type

from pydantic import BaseModel

from .config import AppConfig

JSON_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
RAW_JSON_RE = re.compile(r"(\{.*\})", re.DOTALL)


class OpenHarnessError(RuntimeError):
    pass


def extract_json(text: str) -> str:
    m = JSON_BLOCK_RE.findall(text)
    if m:
        return m[-1]
    m2 = RAW_JSON_RE.findall(text)
    if m2:
        return m2[-1]
    raise OpenHarnessError(f"No JSON object found in OpenHarness output:\n{text[:4000]}")


def run_openharness_structured(prompt: str, output_model: Type[BaseModel], cfg: AppConfig, workspace: str | Path) -> BaseModel:
    if not cfg.openharness.enabled:
        raise OpenHarnessError("OpenHarness disabled")
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    full_prompt = f"Output ONLY one JSON object matching this schema exactly:\n{output_model.model_json_schema()}\n\nTask:\n{prompt}"
    env = os.environ.copy()
    or_key = env.get("OPENROUTER_API_KEY", "")
    if or_key and not env.get("OPENAI_API_KEY"):
        env["OPENAI_API_KEY"] = or_key
    for k in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
        env.pop(k, None)

    cmd = [
        cfg.openharness.command,
        "-m",
        cfg.openharness.model,
        "-p",
        full_prompt,
        "--output-format",
        "text",
    ]
    if or_key:
        cmd.extend(["--api-format", "openai", "--base-url", "https://openrouter.ai/api/v1"])
    last_error = None
    cli_available = shutil.which(cfg.openharness.command) is not None
    if not cli_available:
        raise OpenHarnessError(
            f"OpenHarness command not found: {cfg.openharness.command}. "
            "Please install OpenHarness CLI and ensure it is in PATH."
        )
    for _ in range(cfg.openharness.max_retries + 1):
        try:
            t0 = time.perf_counter()
            print(f"[llm] calling {cfg.openharness.model} (timeout={cfg.openharness.timeout_sec}s)", flush=True)
            cp = subprocess.run(
                cmd,
                cwd=str(workspace),
                text=True,
                capture_output=True,
                timeout=cfg.openharness.timeout_sec,
                check=False,
                env=env,
            )
            text = (cp.stdout or "") + "\n" + (cp.stderr or "")
            if cp.returncode != 0:
                raise OpenHarnessError(text[:4000])
            payload = extract_json(text)
            print(f"[llm] response parsed in {time.perf_counter()-t0:.1f}s", flush=True)
            return output_model.model_validate_json(payload)
        except Exception as e:
            last_error = e
    raise OpenHarnessError(str(last_error))
