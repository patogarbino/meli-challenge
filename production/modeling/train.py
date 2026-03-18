"""CLI entry point for training the new/used classifier.

Usage:
    PYTHONPATH=. python -m production.modeling.train \
        --data-path data/raw/MLA_100k_checked_v3.jsonlines \
        --output-dir models/ \
        --warranty-samples warranty_samples.csv \
        --optimize
"""

from pathlib import Path

import typer

app = typer.Typer()


@app.command()
def main(
    data_path: Path = typer.Option(
        "data/raw/MLA_100k_checked_v3.jsonlines",
        help="Path to the raw JSONL data file.",
    ),
    output_dir: Path = typer.Option(
        "models/",
        help="Directory to save model artifacts.",
    ),
    warranty_samples: Path = typer.Option(
        "warranty_samples.csv",
        help="Path to labeled warranty samples CSV.",
    ),
    optimize: bool = typer.Option(
        False,
        help="Run Optuna hyperparameter optimization.",
    ),
) -> None:
    """Train a CatBoost model to classify MeLi items as new or used."""
    from production.adapters.catboost_model import CatBoostModel
    from production.adapters.data_loader import JsonlinesDataLoader
    from production.adapters.model_repository import LocalModelRepository
    from production.adapters.preprocessor import MeliPreprocessor
    from production.application.train_use_case import TrainUseCase

    # Wire up adapters (dependency injection)
    data_loader = JsonlinesDataLoader()
    preprocessor = MeliPreprocessor(warranty_samples_path=warranty_samples)
    model = CatBoostModel()
    repository = LocalModelRepository()

    # Execute
    use_case = TrainUseCase(data_loader, preprocessor, model, repository)
    result = use_case.execute(
        data_path=data_path,
        output_dir=output_dir,
        optimize=optimize,
    )

    raise typer.Exit(code=0 if result.accuracy >= 0.86 else 1)


if __name__ == "__main__":
    app()
