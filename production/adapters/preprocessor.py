"""Adapter: preprocessor that consolidates all feature engineering from EDA.

Transforms raw item dicts → model-ready DataFrame with 22 features.
Fits and serialises the warranty classifier during `fit_transform`.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from production.domain.entities import RawItem
from production.domain.ports import PreprocessorPort

# ── Column sets ────────────────────────────────────────────────────────────────

COLS_TO_DROP = [
    "sub_status", "listing_source", "coverage_areas",
    "international_delivery_mode", "differential_pricing",
    "original_price", "catalog_product_id", "subtitle", "site_id",
    "parent_item_id", "descriptions", "last_updated", "id",
    "currency_id", "thumbnail", "date_created", "secure_thumbnail",
    "stop_time", "status", "start_time", "permalink", "seller_id",
    "title",
]

COLS_TO_BOOL = ["deal_ids", "official_store_id", "video_id"]

LIST_COLS_TO_COUNT = [
    "non_mercado_pago_payment_methods",
    "variations",
    "tags",
    "attributes",
]

CAT_FEATURES = [
    "category_id", "city", "warranty", "listing_type_id",
    "shipping_mode", "buying_mode", "state",
]

# ── Helpers ────────────────────────────────────────────────────────────────────


def _clean_dataset_objects(df: pd.DataFrame) -> pd.DataFrame:
    """Replace hidden nulls (empty lists, 'none' strings, etc.) with pd.NA."""
    cols_objeto = df.select_dtypes(include=["object"]).columns
    for col in cols_objeto:
        df[col] = df[col].map(
            lambda x: pd.NA
            if (
                (isinstance(x, list) and len(x) == 0)
                or (isinstance(x, str) and x.strip().lower() in ["none", ""])
                or (x is None)
            )
            else x
        )
    return df


def _process_address(df: pd.DataFrame, col: str = "seller_address") -> pd.DataFrame:
    """Extract state and city from the nested seller_address dict."""
    df = df.reset_index(drop=True)
    df_temp = pd.json_normalize(df[col])
    cols_map = {"state.name": "state", "city.name": "city"}
    existing = [c for c in cols_map if c in df_temp.columns]
    df_names = df_temp[existing].rename(columns=cols_map)
    df = pd.concat([df, df_names], axis=1).drop(columns=[col])
    return df


def _parse_shipping(df: pd.DataFrame, col: str = "shipping") -> pd.DataFrame:
    """Extract shipping_local_pick_up, shipping_free, shipping_mode."""
    parsed = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df = df.copy()
    df["shipping_local_pick_up"] = parsed.apply(lambda x: x.get("local_pick_up", False)).astype(int)
    df["shipping_free"] = parsed.apply(lambda x: x.get("free_shipping", False)).astype(int)
    df["shipping_mode"] = parsed.apply(lambda x: x.get("mode", "not_specified"))
    return df.drop(columns=[col])


def _count_list_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Count elements in a list-type column, handling strings and nulls."""

    def count(val: Any) -> int:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return 0
        if isinstance(val, list):
            return len(val)
        if isinstance(val, str):
            if val.strip() == "":
                return 0
            try:
                parsed = ast.literal_eval(val)
                return len(parsed) if isinstance(parsed, list) else 0
            except Exception:
                return 0
        return 0

    df = df.copy()
    df[f"{col}_count"] = df[col].apply(count)
    return df.drop(columns=[col])


def _parse_pictures(df: pd.DataFrame, col: str = "pictures") -> pd.DataFrame:
    """Extract pictures_count and pictures_max_area."""

    def process(val: Any) -> tuple[int, int]:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return 0, 0
        if isinstance(val, str):
            if val.strip() == "":
                return 0, 0
            try:
                val = ast.literal_eval(val)
            except Exception:
                return 0, 0
        if isinstance(val, list):
            count = len(val)
            max_area = 0
            for pic in val:
                try:
                    w, h = pic.get("max_size", "0x0").split("x")
                    max_area = max(max_area, int(w) * int(h))
                except Exception:
                    continue
            return count, max_area
        return 0, 0

    results = df[col].apply(process)
    df = df.copy()
    df["pictures_count"] = results.apply(lambda x: x[0])
    df["pictures_max_area"] = results.apply(lambda x: x[1])
    return df.drop(columns=[col])


def _convert_to_bool(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Convert columns with mostly nulls into binary flags (0/1)."""
    for col in cols:
        df[col] = np.where(df[col].isnull(), 0, 1)
    return df


# ── Main Preprocessor ─────────────────────────────────────────────────────────


class MeliPreprocessor(PreprocessorPort):
    """Full feature engineering pipeline: raw dicts → 22-column DataFrame.

    During ``fit_transform`` the warranty classifier is trained and cached.
    Subsequent ``transform`` calls reuse the fitted classifier.
    """

    def __init__(self, warranty_samples_path: Path | None = None) -> None:
        self.warranty_samples_path = warranty_samples_path
        self._warranty_model = None  # sentence-transformers model
        self._warranty_clf = None    # SGDClassifier
        self._warranty_le = None     # LabelEncoder
        self._is_fitted = False

    # ── Public API ─────────────────────────────────────────────────────────

    def fit_transform(
        self, X_raw: list[RawItem], y: list[str] | None = None
    ) -> pd.DataFrame:
        df = pd.DataFrame(X_raw)

        # Drop condition if present (it's the target, not a feature)
        if "condition" in df.columns:
            df = df.drop(columns=["condition"])

        df = self._apply_pipeline(df, fit_warranty=True)
        self._is_fitted = True
        return df

    def transform(self, X_raw: list[RawItem]) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError(
                "Preprocessor is not fitted. Call fit_transform first."
            )
        df = pd.DataFrame(X_raw)

        if "condition" in df.columns:
            df = df.drop(columns=["condition"])

        df = self._apply_pipeline(df, fit_warranty=False)
        return df

    # ── Pipeline ───────────────────────────────────────────────────────────

    def _apply_pipeline(self, df: pd.DataFrame, *, fit_warranty: bool) -> pd.DataFrame:
        # 1. Clean hidden nulls
        df = _clean_dataset_objects(df)

        # 2. Drop irrelevant columns
        existing_drop = [c for c in COLS_TO_DROP if c in df.columns]
        df = df.drop(columns=existing_drop)

        # 3. Extract address → state, city
        if "seller_address" in df.columns:
            df = _process_address(df)

        # 4. Parse shipping
        if "shipping" in df.columns:
            df = _parse_shipping(df)

        # 5. Count list columns
        for col in LIST_COLS_TO_COUNT:
            if col in df.columns:
                df = _count_list_col(df, col)

        # 6. Parse pictures
        if "pictures" in df.columns:
            df = _parse_pictures(df)

        # 7. Warranty classification
        if "warranty" in df.columns:
            df = self._process_warranty(df, fit=fit_warranty)

        # 8. Convert sparse columns to bool
        existing_bool = [c for c in COLS_TO_BOOL if c in df.columns]
        df = _convert_to_bool(df, existing_bool)

        return df

    # ── Warranty ───────────────────────────────────────────────────────────

    def _process_warranty(self, df: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        if fit:
            self._fit_warranty_classifier()

        if self._warranty_model is None:
            # Fallback: just flag presence
            df["warranty"] = np.where(df["warranty"].isna(), "Sin dato", "Con garantía")
            return df

        mask = df["warranty"].notna() & (df["warranty"].astype(str).str.strip() != "")
        df["warranty_label"] = "Sin dato"

        if mask.sum() > 0:
            X_emb = self._warranty_model.encode(
                df.loc[mask, "warranty"].astype(str).tolist(),
                show_progress_bar=False,
                batch_size=64,
            )
            preds = self._warranty_le.inverse_transform(self._warranty_clf.predict(X_emb))
            df.loc[mask, "warranty_label"] = preds

        df = df.drop(columns=["warranty"]).rename(columns={"warranty_label": "warranty"})
        return df

    def _fit_warranty_classifier(self) -> None:
        if self.warranty_samples_path is None or not self.warranty_samples_path.exists():
            print("[warranty] No samples CSV found – falling back to binary flag.")
            return

        from sentence_transformers import SentenceTransformer
        from sklearn.linear_model import SGDClassifier
        from sklearn.preprocessing import LabelEncoder

        MERGE_MAP = {"12+ meses": "7–12 meses", "3–5 años": "5+ años"}

        df_labeled = pd.read_csv(self.warranty_samples_path)
        df_labeled = df_labeled.dropna(subset=["warranty_label"]).copy()
        df_labeled["warranty_label"] = df_labeled["warranty_label"].replace(MERGE_MAP)

        print(f"[warranty] Training on {len(df_labeled)} samples, "
              f"{df_labeled['warranty_label'].nunique()} classes")

        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        X = model.encode(
            df_labeled["warranty"].fillna("").astype(str).tolist(),
            show_progress_bar=True,
            batch_size=64,
        )

        le = LabelEncoder()
        y = le.fit_transform(df_labeled["warranty_label"])

        clf = SGDClassifier(
            loss="modified_huber",
            alpha=0.001,
            max_iter=1000,
            random_state=42,
            class_weight="balanced",
        )
        clf.fit(X, y)

        self._warranty_model = model
        self._warranty_clf = clf
        self._warranty_le = le
