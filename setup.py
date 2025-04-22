# setup.py
from setuptools import setup, find_packages

setup(
    name="bioadapt",
    version="0.1.0",
    packages=find_packages(include=["data", "pipeline"]),     
    install_requires=[
        "annotated-types", "colorama", "contourpy", "cycler", "defusedxml", "fonttools", "fpdf2", "imbalanced-learn", "joblib", "kiwisolver", "matplotlib", "multimethod", "mypy_extensions", "numpy", "packaging", "pandas", "pandera", "pillow", "psutil", "pydantic", "pydantic_core", "pyparsing", "python-dateutil", "pytz", "scikit-learn", "scipy", "seaborn", "six", "sklearn-compat", "threadpoolctl", "tqdm", "typeguard", "typing-inspect", "typing-inspection", "typing_extensions", "tzdata", "wrapt", "xgboost"
    ],
)
