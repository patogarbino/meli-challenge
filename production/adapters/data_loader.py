"""Adapter: loads raw JSONL data and splits into train/test."""

from __future__ import annotations

import json
from pathlib import Path

from production.domain.entities import RawItem
from production.domain.ports import DataLoaderPort


class JsonlinesDataLoader(DataLoaderPort):
    """Reads a `.jsonlines` file and reproduces the canonical train/test split.

    The split mirrors `build_dataset()`: last *test_size* rows → test,
    the rest → train.  The ``condition`` field is extracted as the target
    and removed from the test dicts (same behaviour as the original fn).
    """

    def __init__(self, test_size: int = 10_000) -> None:
        self.test_size = test_size

    def load(
        self, path: Path
    ) -> tuple[list[RawItem], list[str], list[RawItem], list[str]]:
        data = [json.loads(line) for line in open(path)]

        target = lambda x: x.get("condition")
        N = -self.test_size

        X_train = data[:N]
        X_test = data[N:]

        y_train = [target(x) for x in X_train]
        y_test = [target(x) for x in X_test]

        # Remove target from test set (same as original build_dataset)
        for x in X_test:
            del x["condition"]

        return X_train, y_train, X_test, y_test
