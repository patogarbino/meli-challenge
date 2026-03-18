"""Use case: orchestrates the prediction flow."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from production.domain.entities import RawItem
from production.domain.ports import ModelRepositoryPort


class PredictUseCase:
    """Load model → preprocess new data → predict."""

    def __init__(self, repository: ModelRepositoryPort) -> None:
        self.repository = repository

    def execute(
        self,
        data_path: Path,
        model_dir: Path,
        output_path: Path | None = None,
    ) -> pd.DataFrame:
        # 1. Load trained artifacts
        print("=" * 60)
        print("STEP 1: Loading model and preprocessor")
        print("=" * 60)
        model, preprocessor = self.repository.load(model_dir)

        # 2. Load new data
        print("\n" + "=" * 60)
        print("STEP 2: Loading input data")
        print("=" * 60)
        items: list[RawItem] = [json.loads(line) for line in open(data_path)]
        print(f"  Items to predict: {len(items):,}")

        # 3. Preprocess
        print("\n" + "=" * 60)
        print("STEP 3: Preprocessing")
        print("=" * 60)
        X = preprocessor.transform(items)
        print(f"  Feature matrix: {X.shape}")

        # 4. Predict
        print("\n" + "=" * 60)
        print("STEP 4: Generating predictions")
        print("=" * 60)
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)

        results = pd.DataFrame(
            {
                "prediction": predictions,
                "prob_new": probabilities[:, 1],
                "prob_used": probabilities[:, 0],
            }
        )

        print(f"  Predictions: {len(results):,}")
        print(
            f"  Distribution: {pd.Series(predictions).value_counts().to_dict()}"
        )

        # 5. Save if output path given
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            results.to_csv(output_path, index=False)
            print(f"\n  Results saved to {output_path}")

        return results
