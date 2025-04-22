"""
data/schema.py
==============

Pandera schema definitions describing *exactly* what the pipeline expects
from any raw CSV before modelling begins.

This schema is used to validate the data at various stages of the pipeline, including during training and testing.
"""

import pandera as pa
from pandera import Column, DataFrameSchema, Check


# --- Example primary training‑data schema ----------------------------- #
training_schema = DataFrameSchema(
    {
        # --- meta or ID columns --------------------------------------- #
        "patient_id": Column(pa.String, nullable=True, coerce=True, required=False),

        # --- demographic features ------------------------------------- #
        "age": Column(pa.Float, Check.in_range(0, 120)),
        "sex": Column(pa.String, Check.isin(["M", "F"])),

        # --- numeric biomarkers --------------------------------------- #
        "biomarker_1": Column(pa.Float, nullable=True),
        "biomarker_2": Column(pa.Float, nullable=True),

        # --- batch column (optional) ---------------------------------- #
        "batch": Column(pa.String, nullable=True, required=False),

        # --- response -------------------------------------------------- #
        "response": Column(pa.Int, Check.isin([0, 1])),
    },
    strict=False,          # allow extra columns, we’ll drop or keep later
    coerce=True            # auto‑convert dtypes where possible
)
