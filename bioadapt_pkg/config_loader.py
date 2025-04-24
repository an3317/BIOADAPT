# pipeline/config_loader.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import yaml
from typing import Optional

@dataclass
class CVConfig:
    random_seeds: List[int] = field(default_factory=lambda: [42])

@dataclass
class PipelineConfig:
    algorithm: str
    feature_selection: str
    explain: str = "none"                # ← add this!
    use_outlier_detection: bool = False
    outlier_method: str | None = None
    iqr_threshold: float = 0.05
    zscore_threshold: float = 0.05
    zscore_limit: float = 3.0
    iso_forest_threshold: float = 0.05
    pca_reconstruction_error_threshold: float = 0.1
    n_components_pca: int | None = None
    output_folder: Path = Path("results")

@dataclass
class DataConfig:
    paths: List[Path]
    response: str
    independent_test_path: Optional[Path] = None

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
