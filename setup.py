from setuptools import setup, find_packages

setup(
    name="bioadapt",
    version="0.1.0",
    packages=find_packages(),       # ← this must see both "pipeline" and "data"
    include_package_data=True,
    install_requires=[
        # your requirements…
    ],
    entry_points={
        "console_scripts": [
            "bioadapt = pipeline.cli:cli",
        ],
    },
)
