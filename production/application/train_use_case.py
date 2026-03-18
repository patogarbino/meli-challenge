"""Use case: orchestrates the full training flow."""

from __future__ import annotations

from pathlib import Path

from production.domain.entities import TrainResult
from production.domain.ports import (
    DataLoaderPort,
    ModelPort,
    ModelRepositoryPort,
    PreprocessorPort,
)


class TrainUseCase:
    """Load raw data → preprocess → train → evaluate → save."""

    def __init__(
        self,
        data_loader: DataLoaderPort,
        preprocessor: PreprocessorPort,
        model: ModelPort,
        repository: ModelRepositoryPort,
    ) -> None:
        self.data_loader = data_loader
        self.preprocessor = preprocessor
        self.model = model
        self.repository = repository

    def execute(
        self,
        data_path: Path,
        output_dir: Path,
        optimize: bool = False,
    ) -> TrainResult:
        # 1. Load raw data
        print("=" * 60)
        print("STEP 1: Loading raw data")
        print("=" * 60)
        X_train_raw, y_train, X_test_raw, y_test = self.data_loader.load(data_path)
        print(f"  Train: {len(X_train_raw):,} items")
        print(f"  Test:  {len(X_test_raw):,} items")

        # 2. Preprocess (fit on train, transform test)
        print("\n" + "=" * 60)
        print("STEP 2: Preprocessing (feature engineering)")
        print("=" * 60)
        X_train = self.preprocessor.fit_transform(X_train_raw, y_train)
        X_test = self.preprocessor.transform(X_test_raw)
        print(f"  Train features: {X_train.shape}")
        print(f"  Test features:  {X_test.shape}")
        print(f"  Columns: {list(X_train.columns)}")

        # 3. Train model
        print("\n" + "=" * 60)
        print("STEP 3: Training model")
        if optimize:
            print("  (with Optuna hyperparameter optimization)")
        print("=" * 60)

        import pandas as pd
        y_train_s = pd.Series(y_train)
        y_test_s = pd.Series(y_test)

        result = self.model.train(
            X_train, y_train_s,
            X_val=X_test, y_val=y_test_s,
            optimize=optimize,
        )

        # 4. Save artifacts
        print("\n" + "=" * 60)
        print("STEP 4: Saving model artifacts")
        print("=" * 60)
        self.repository.save(self.model, self.preprocessor, output_dir)

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print(f"  Accuracy: {result.accuracy:.4f}")
        print(f"  AUC-ROC:  {result.auc_roc:.4f}")
        print("=" * 60)

        return result
