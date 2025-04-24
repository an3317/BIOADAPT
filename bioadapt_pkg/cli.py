# bioadapt_pkg/cli.py

import logging
import click
from pathlib import Path

import joblib
import pandas as pd

from bioadapt_pkg.config_loader import load_config
from bioadapt_pkg.core import machine_learning_pipeline
from bioadapt_pkg.evaluate import evaluate_on_independent_dataset

# ——————————————————————————————————————————————————————————————
# configure root logger
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

@click.group()
def cli():
    """BIOADAPT ML pipeline command-line interface."""
    pass

@cli.command()
@click.option(
    "--config", "-c",
    type=click.Path(exists=True, dir_okay=False),
    default="config.yaml",
    show_default=True,
    help="Path to YAML config file."
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Enable debug logging (very detailed)."
)
def train(config, verbose):
    """Train model(s) using the pipeline."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("Verbose mode ON — debug logging enabled")

    cfg = load_config(Path(config))
    logging.debug("Loaded configuration: %s", cfg)
    machine_learning_pipeline(cfg)

@cli.command()
@click.option(
    "--config", "-c",
    type=click.Path(exists=True, dir_okay=False),
    default="config.yaml",
    show_default=True,
    help="Path to YAML config file."
)
def evaluate(config):
    """Run trained model on independent test set."""
    cfg = load_config(Path(config))

    # where the model was saved
    model_path = Path(cfg.pipeline.output_folder) / "seed_42" / "model.pkl"
    logging.info("Loading model from %s", model_path)
    model = joblib.load(model_path)

    # read the held-out test CSV
    test_path = cfg.data.independent_test_path
    logging.info("Reading independent test set from %s", test_path)
    df_test = pd.read_csv(test_path)

    # evaluation output folder
    out_folder = Path(cfg.pipeline.output_folder) / "independent"
    metrics = evaluate_on_independent_dataset(
        model=model,
        df=df_test,
        response_col=cfg.data.response,
        output_folder=out_folder,
    )

    logging.info("Independent-test metrics: %s", metrics)

if __name__ == "__main__":
    cli()
