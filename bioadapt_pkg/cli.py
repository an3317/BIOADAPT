# pipeline/cli.py
import click
from pathlib import Path

from bioadapt_pkg.config_loader import load_config
from bioadapt_pkg.core import machine_learning_pipeline

@click.group()
def cli():
    """BIOADAPT ML pipeline command‑line interface."""
    pass

@cli.command()
@click.option("--config", "-c", type=click.Path(exists=True), default="config.yaml",
              help="Path to YAML config file.")
def train(config):
    """Train model(s) using the pipeline."""
    cfg = load_config(Path(config))
    machine_learning_pipeline(cfg)

if __name__ == "__main__":
    cli()
