"""CLI entry point for prediction using a trained model.

Usage:
    PYTHONPATH=. python -m production.modeling.predict \
        --data-path data/raw/MLA_100k_checked_v3.jsonlines \
        --model-dir models/ \
        --output-path data/processed/predictions.csv
"""

from pathlib import Path

import typer

app = typer.Typer()


@app.command()
def main(
    data_path: Path = typer.Option(
        ...,
        help="Path to the raw JSONL data file to predict.",
    ),
    model_dir: Path = typer.Option(
        "models/",
        help="Directory containing saved model artifacts.",
    ),
    output_path: Path = typer.Option(
        "data/processed/predictions.csv",
        help="Path to save prediction results.",
    ),
) -> None:
    """Predict new/used for MeLi items using a trained model."""
    from production.adapters.model_repository import LocalModelRepository
    from production.application.predict_use_case import PredictUseCase

    repository = LocalModelRepository()
    use_case = PredictUseCase(repository)
    use_case.execute(
        data_path=data_path,
        model_dir=model_dir,
        output_path=output_path,
    )


if __name__ == "__main__":
    app()
