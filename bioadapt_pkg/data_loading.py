"""
pipeline/data_loading.py
------------------------

Utilities to load CSV(s), validate them against Pandera schemas,
and hand back cleaned DataFrames to the pipeline.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Union

import pandas as pd
import pandera as pa

from data.schema import training_schema


def _validate(df: pd.DataFrame, schema: pa.DataFrameSchema) -> pd.DataFrame:
    """Validate `df` against `schema` and return the coerced frame."""
    try:
        return schema.validate(df, lazy=True)  # lazy=True → gather all problems
    except pa.errors.SchemaErrors as exc:
        # Pretty‑print all failures then re‑raise to stop the run
        print("\n❌ Schema validation failed:\n")
        print(exc.failure_cases.head(20))  # show first few rows
        raise


def load_csvs(paths: List[Union[str, Path]]) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    Load one or more CSV files and validate each with the training schema.
    """
    frames = []
    for p in paths:
        df_raw = pd.read_csv(p)
        df_ok = _validate(df_raw, training_schema)
        frames.append(df_ok)

    return frames[0] if len(frames) == 1 else frames
