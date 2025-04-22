# tests/test_schema.py
import pandas as pd
import pytest
from pandera.errors import SchemaErrors
from data.schema import training_schema


def test_schema_rejects_invalid_rows():
    bad = pd.DataFrame({
        "age": [-50],               # invalid
        "sex": ["X"],               # invalid
        "biomarker_1": [0.1],
        "biomarker_2": [0.2],
        "response": [3],            # invalid
    })
    with pytest.raises(SchemaErrors):
        training_schema.validate(bad, lazy=True)


def test_schema_allows_valid_rows():
    good = pd.DataFrame({
        "age": [5],
        "sex": ['M'],
        "biomarker_1": [0.1],
        "biomarker_2": [0.2],
        "response": [1],
    })
    # Will raise if validation unexpectedly fails
    training_schema.validate(good, lazy=True)
