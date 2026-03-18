"""Adapter: CatBoost classifier with optional Optuna tuning."""

from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)

from production.domain.entities import TrainResult
from production.domain.ports import ModelPort

CAT_FEATURES = [
    "category_id", "city", "warranty", "listing_type_id",
    "shipping_mode", "buying_mode", "state",
]

DEFAULT_PARAMS = {
    "iterations": 1000,
    "random_seed": 42,
    "eval_metric": "AUC",
    "verbose": 100,
}


class CatBoostModel(ModelPort):
    """Wraps CatBoostClassifier for the new/used classification task."""

    def __init__(self, params: dict | None = None) -> None:
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.model: CatBoostClassifier | None = None
        self.cat_features = CAT_FEATURES

    # ── Training ───────────────────────────────────────────────────────────

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        optimize: bool = False,
    ) -> TrainResult:
        # Encode target: new → 1, used → 0
        y_train_enc = (y_train == "new").astype(int)
        y_val_enc = (y_val == "new").astype(int) if y_val is not None else None

        # Resolve cat features that actually exist in data
        cat_feats = [c for c in self.cat_features if c in X_train.columns]

        if optimize:
            best_params = self._optuna_search(
                X_train, y_train_enc, X_val, y_val_enc, cat_feats
            )
            self.params.update(best_params)

        self.model = CatBoostClassifier(**self.params)

        eval_set = (X_val, y_val_enc) if X_val is not None else None
        self.model.fit(
            X_train,
            y_train_enc,
            cat_features=cat_feats,
            eval_set=eval_set,
            early_stopping_rounds=50 if eval_set else None,
        )

        # Evaluate
        X_eval = X_val if X_val is not None else X_train
        y_eval = y_val_enc if y_val_enc is not None else y_train_enc

        y_pred = self.model.predict(X_eval)
        y_proba = self.model.predict_proba(X_eval)[:, 1]

        acc = accuracy_score(y_eval, y_pred)
        auc = roc_auc_score(y_eval, y_proba)
        report = classification_report(
            y_eval, y_pred, target_names=["used", "new"]
        )

        print(report)
        print(f"Accuracy: {acc:.4f}")
        print(f"AUC-ROC:  {auc:.4f}")

        return TrainResult(
            accuracy=acc,
            auc_roc=auc,
            classification_report=report,
            best_params=self.params,
        )

    # ── Prediction ─────────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        preds = self.model.predict(X).flatten()
        return np.where(preds == 1, "new", "used")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        return self.model.predict_proba(X)

    # ── Optuna ─────────────────────────────────────────────────────────────

    def _optuna_search(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None,
        y_val: pd.Series | None,
        cat_feats: list[str],
        n_trials: int = 50,
    ) -> dict:
        import optuna
        from sklearn.model_selection import train_test_split

        # Use provided val or create one
        if X_val is None or y_val is None:
            X_tr, X_vl, y_tr, y_vl = train_test_split(
                X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
            )
        else:
            X_tr, X_vl, y_tr, y_vl = X_train, X_val, y_train, y_val

        def objective(trial: optuna.Trial) -> float:
            params = {
                "iterations": trial.suggest_int("iterations", 500, 2000),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "depth": trial.suggest_int("depth", 4, 10),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
                "random_strength": trial.suggest_float("random_strength", 0, 1),
                "random_seed": 42,
                "verbose": 0,
            }

            model = CatBoostClassifier(**params)
            model.fit(
                X_tr, y_tr,
                cat_features=cat_feats,
                eval_set=(X_vl, y_vl),
                early_stopping_rounds=50,
            )

            from sklearn.metrics import precision_score
            y_pred = model.predict(X_vl)
            return precision_score(y_vl, y_pred, pos_label=0)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        print(f"Best params: {study.best_params}")
        print(f"Best precision (used): {study.best_value:.4f}")
        return study.best_params
