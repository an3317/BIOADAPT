# bioadapt_pkg/explain.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from pathlib import Path

def shap_explain(
    model,
    X: pd.DataFrame,
    selected_features: list[str],
    outdir: Path,
    max_samples: int = 100
) -> tuple[Path, Path]:
    """
    Generate SHAP explanation plots for ONLY the selected features,
    and produce a minimal HTML that embeds the summary plot.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Restrict to selected features
    X_sel = X[selected_features]

    # 2) Sample for background
    background = shap.sample(X_sel, min(len(X_sel), max_samples))

    # 3) Choose explainer and compute SHAP values on the sampled background
    if hasattr(model.named_steps['est'], 'get_booster') or \
       hasattr(model.named_steps['est'], 'estimators_'):
        explainer = shap.TreeExplainer(model.named_steps['est'])
        shap_vals = explainer.shap_values(background)
    else:
        # kernel explainer needs a DataFrame wrapper
        def wrapped_proba(arr: np.ndarray):
            df = pd.DataFrame(arr, columns=selected_features)
            return model.predict_proba(df)
        explainer = shap.KernelExplainer(wrapped_proba, background)
        shap_vals = explainer.shap_values(background)

    # 4) For binary classification, pick the positive‐class SHAP array
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    # 5) Create & save the beeswarm summary plot
    summary_png = outdir / "shap_summary.png"
    shap.summary_plot(shap_vals, background, show=False)
    plt.tight_layout()
    plt.savefig(summary_png)
    plt.close()

    # 6) Write a tiny HTML page embedding the PNG
    html_path = outdir / "shap_report.html"
    html_content = f"""<!DOCTYPE html>
<html lang="en">
  <head><meta charset="utf-8"><title>SHAP Summary</title></head>
  <body>
    <h1>SHAP Summary (top {len(selected_features)} features)</h1>
    <img src="{summary_png.name}" alt="SHAP summary plot">
  </body>
</html>"""
    html_path.write_text(html_content, encoding="utf-8")

    return summary_png, html_path
