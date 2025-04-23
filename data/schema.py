import pandera as pa
from pandera import Column, DataFrameSchema, Check

# explicit checks
is_age      = Check.in_range(0, 120)
is_sex      = Check.isin(["M", "F"])
is_response = Check.isin([0, 1])

# regex that matches every column *except* response|age|sex
other_cols_pattern = r"^(?!response$|age$|sex$).*"

training_schema = DataFrameSchema(
    {

        "patient_id": Column(
            None,             # no dtype enforcement
            nullable=True,
            required=False,
            coerce=True
        ),
        "patient": Column(
            None,
            nullable=True,
            required=False,
            coerce=True
        ),

        "response": Column(pa.Int, is_response, coerce=True),
        "age":      Column(pa.Float, is_age,   nullable=True, required=False, coerce=True),
        "sex":      Column(pa.String, is_sex,  nullable=True, required=False, coerce=True),
        other_cols_pattern: Column(
            pa.Float,
            nullable=True,
            required=False,
            coerce=True,
            regex=True,   # wildcard rule applies only to “other” cols
        ),
    },
    strict=False,
    coerce=True,
)
