"""Domain entities for the MeLi new/used classification pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Raw item from the JSON lines file
RawItem = dict[str, Any]


@dataclass
class TrainResult:
    """Holds the results of a training run."""

    accuracy: float
    auc_roc: float
    classification_report: str
    best_params: dict[str, Any] = field(default_factory=dict)
