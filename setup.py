from setuptools import setup, find_packages

setup(
    name="bioadapt",
    version="0.1.0",
    packages=find_packages(include=["bioadapt_pkg", "bioadapt_pkg.*"]),
    entry_points={
        "console_scripts": [
            "bioadapt = bioadapt_pkg.cli:cli",
        ],
    },
)
