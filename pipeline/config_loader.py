# pipeline/config_loader.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import yaml
import tomli  # optional: support pyproject.toml config later

@dataclass
class CVConfig:
    random_seeds: List[int] = field(default_factory=lambda: [42])

@dataclass
class PipelineConfig:
    algorithm: str
    feature_selection: str
    use_outlier_detection: bool
    output_folder: Path

@dataclass
class DataConfig:
    paths: List[Path]
    response: str

@dataclass
class Config:
    data: DataConfig
    pipeline: PipelineConfig
    cv: CVConfig

def load_config(path: Path | str = "config.yaml") -> Config:
    with open(path, "r", encoding="utf8") as fp:
        cfg_raw = yaml.safe_load(fp)
    return Config(
        data=DataConfig(**cfg_raw["data"]),
        pipeline=PipelineConfig(**cfg_raw["pipeline"]),
        cv=CVConfig(**cfg_raw.get("cv", {})),
    )
