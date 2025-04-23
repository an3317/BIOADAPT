from pathlib import Path
from bioadapt_pkg.config_loader import load_config

def test_load_config_outlier_fields(tmp_path):
    cfg_path = tmp_path / "mini.yaml"
    cfg_path.write_text("""
    data:
      paths: []
      response: response
    pipeline:
      algorithm: svm
      feature_selection: anova
      use_outlier_detection: true
      outlier_method: zscore
    """)
    cfg = load_config(cfg_path)
    assert cfg.pipeline.use_outlier_detection is True
    assert cfg.pipeline.outlier_method == "zscore"
