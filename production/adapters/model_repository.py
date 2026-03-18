"""Adapter: persists and loads trained model + preprocessor to disk."""

from __future__ import annotations

import pickle
from pathlib import Path

from production.domain.ports import ModelPort, ModelRepositoryPort, PreprocessorPort

SENTENCE_TRANSFORMER_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


class LocalModelRepository(ModelRepositoryPort):
    """Saves/loads CatBoost model (.cbm) and preprocessor state (.pkl)."""

    MODEL_FILENAME = "catboost_model.cbm"
    PREPROCESSOR_FILENAME = "preprocessor.pkl"

    def save(
        self, model: ModelPort, preprocessor: PreprocessorPort, path: Path
    ) -> None:
        path.mkdir(parents=True, exist_ok=True)

        # Save CatBoost native format
        model_path = path / self.MODEL_FILENAME
        model.model.save_model(str(model_path))
        print(f"[repository] Model saved to {model_path}")

        # Save preprocessor state — WITHOUT the heavy SentenceTransformer
        # (it will be reloaded by name at load time)
        preprocessor_path = path / self.PREPROCESSOR_FILENAME
        state = {
            "warranty_clf": preprocessor._warranty_clf,
            "warranty_le": preprocessor._warranty_le,
            "warranty_samples_path": preprocessor.warranty_samples_path,
            "has_warranty_model": preprocessor._warranty_model is not None,
            "is_fitted": preprocessor._is_fitted,
        }
        with open(preprocessor_path, "wb") as f:
            pickle.dump(state, f)
        print(f"[repository] Preprocessor saved to {preprocessor_path}")

    def load(self, path: Path) -> tuple[ModelPort, PreprocessorPort]:
        from catboost import CatBoostClassifier

        from production.adapters.catboost_model import CatBoostModel
        from production.adapters.preprocessor import MeliPreprocessor

        # Load CatBoost model
        model_path = path / self.MODEL_FILENAME
        cb_model = CatBoostClassifier()
        cb_model.load_model(str(model_path))

        model_adapter = CatBoostModel()
        model_adapter.model = cb_model
        print(f"[repository] Model loaded from {model_path}")

        # Load preprocessor state
        preprocessor_path = path / self.PREPROCESSOR_FILENAME
        with open(preprocessor_path, "rb") as f:
            state = pickle.load(f)

        preprocessor = MeliPreprocessor(
            warranty_samples_path=state.get("warranty_samples_path")
        )
        preprocessor._warranty_clf = state["warranty_clf"]
        preprocessor._warranty_le = state["warranty_le"]
        preprocessor._is_fitted = state["is_fitted"]

        # Reload the SentenceTransformer model by name (not from pkl)
        if state.get("has_warranty_model", False):
            from sentence_transformers import SentenceTransformer
            print("[repository] Loading SentenceTransformer model...")
            preprocessor._warranty_model = SentenceTransformer(
                SENTENCE_TRANSFORMER_NAME
            )
            print("[repository] SentenceTransformer loaded.")

        print(f"[repository] Preprocessor loaded from {preprocessor_path}")

        return model_adapter, preprocessor
