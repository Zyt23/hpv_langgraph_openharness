from __future__ import annotations

from pathlib import Path
from typing import List

import yaml
from pydantic import BaseModel, Field


class PathsConfig(BaseModel):
    data_root: Path
    artifacts_root: Path = Path("./artifacts")
    run_name: str = "hpv_openharness_benchmark"
    knowledge_globs: List[str] = Field(default_factory=lambda: ["./*.pdf", "./*.docx", "./*.md", "./*.txt"])


class OpenHarnessConfig(BaseModel):
    enabled: bool = True
    command: str = "openharness"
    model: str = "openai/gpt-4.1-mini"
    timeout_sec: int = 180
    max_retries: int = 2


class AgentToolLoopConfig(BaseModel):
    observe_max_steps: int = 8
    hypothesize_max_steps: int = 8
    inspect_max_rows: int = 4000
    test_flights_per_label: int = 20


class SplitConfig(BaseModel):
    seed: int = 42
    train_aircraft_count: int = 10


class LoopConfig(BaseModel):
    max_rounds: int = 12
    observation_flights_per_class: int = 10
    validation_windows_per_class: int = 10
    folder_vote_threshold: float = 0.5
    target_f1_class1: float = 0.80
    patience_rounds: int = 3
    min_improvement: float = 0.02
    tool_eval_max_flights_per_folder: int = 20
    validation_max_flights_per_folder: int = 20
    final_eval_max_flights_per_folder: int = 0


class FeatureConfig(BaseModel):
    min_segment_len: int = 60
    min_rows_per_window: int = 60
    profile_quantiles: list[float] = Field(default_factory=lambda: [0.05, 0.5, 0.95])


class RuntimeConfig(BaseModel):
    workspace_subdir: str = "workspace"
    cache_subdir: str = "cache"


class AppConfig(BaseModel):
    paths: PathsConfig
    openharness: OpenHarnessConfig = Field(default_factory=OpenHarnessConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    agent_tools: AgentToolLoopConfig = Field(default_factory=AgentToolLoopConfig)
    loop: LoopConfig = Field(default_factory=LoopConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        cfg_path = Path(path)
        with cfg_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        cfg = cls.model_validate(raw)
        if not cfg.paths.data_root.is_absolute():
            cfg.paths.data_root = (cfg_path.parent / cfg.paths.data_root).resolve()
        if not cfg.paths.artifacts_root.is_absolute():
            cfg.paths.artifacts_root = (cfg_path.parent / cfg.paths.artifacts_root).resolve()
        cfg.paths.knowledge_globs = [str((cfg_path.parent / p).resolve()) if not Path(p).is_absolute() else p for p in cfg.paths.knowledge_globs]
        return cfg

    @property
    def run_dir(self) -> Path:
        return self.paths.artifacts_root / self.paths.run_name
