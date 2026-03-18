"""Port interfaces (hexagonal architecture boundaries)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

from production.domain.entities import RawItem, TrainResult


# ── Data Loading ───────────────────────────────────────────────────────────────


class DataLoaderPort(ABC):
    """Loads raw data and splits it into train/test."""

    @abstractmethod
    def load(
        self, path: Path
    ) -> tuple[list[RawItem], list[str], list[RawItem], list[str]]:
        """Return (X_train, y_train, X_test, y_test) from a raw file."""
        ...


# ── Preprocessing ──────────────────────────────────────────────────────────────


class PreprocessorPort(ABC):
    """Transforms raw item dicts into a model-ready DataFrame."""

    @abstractmethod
    def fit_transform(
        self, X_raw: list[RawItem], y: list[str] | None = None
    ) -> pd.DataFrame:
        """Fit internal state (e.g. warranty classifier) and transform."""
        ...

    @abstractmethod
    def transform(self, X_raw: list[RawItem]) -> pd.DataFrame:
        """Transform using previously fitted state."""
        ...


# ── Model ──────────────────────────────────────────────────────────────────────


class ModelPort(ABC):
    """Machine-learning model contract."""

    @abstractmethod
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        optimize: bool = False,
    ) -> TrainResult:
        """Train the model. If *optimize* is True, run hyperparameter search."""
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return class predictions."""
        ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability estimates."""
        ...


# ── Model Repository ──────────────────────────────────────────────────────────


class ModelRepositoryPort(ABC):
    """Persists and retrieves trained artifacts."""

    @abstractmethod
    def save(
        self, model: ModelPort, preprocessor: PreprocessorPort, path: Path
    ) -> None: ...

    @abstractmethod
    def load(self, path: Path) -> tuple[ModelPort, PreprocessorPort]: ...
