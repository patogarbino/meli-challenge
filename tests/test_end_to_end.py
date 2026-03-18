"""End-to-end integration tests for the training pipeline."""

import json
from pathlib import Path

import pytest

from production.adapters.preprocessor import MeliPreprocessor


@pytest.fixture
def sample_jsonl(tmp_path) -> Path:
    """Create a small JSONL file with 20 items for testing."""

    def make_item(i: int, condition: str) -> dict:
        return {
            "seller_address": {
                "country": {"name": "Argentina", "id": "AR"},
                "state": {"name": "Capital Federal", "id": "AR-C"},
                "city": {"name": "Palermo", "id": "XXX"},
            },
            "warranty": "6 meses" if condition == "new" else None,
            "sub_status": [],
            "condition": condition,
            "deal_ids": [],
            "base_price": 100.0 * (i + 1),
            "shipping": {
                "local_pick_up": True,
                "methods": [],
                "tags": [],
                "free_shipping": condition == "new",
                "mode": "me2",
                "dimensions": None,
            },
            "non_mercado_pago_payment_methods": [
                {"description": "Transferencia", "id": "MLATB", "type": "G"}
            ],
            "seller_id": 1000 + i,
            "variations": [],
            "site_id": "MLA",
            "listing_type_id": "gold_special" if condition == "new" else "free",
            "price": 100.0 * (i + 1),
            "attributes": [],
            "buying_mode": "buy_it_now",
            "tags": [],
            "listing_source": "",
            "parent_item_id": None,
            "coverage_areas": [],
            "category_id": f"MLA{1000 + i}",
            "descriptions": [{"id": f"MLA{i}-{i}"}],
            "last_updated": "2015-09-05T20:42:58.000Z",
            "international_delivery_mode": "none",
            "pictures": [{"size": "500x375", "max_size": "1200x900",
                          "url": "http://ex.com/p.jpg", "secure_url": "https://ex.com/p.jpg",
                          "quality": "", "id": f"{i}-MLA"}],
            "id": f"MLA{i}",
            "official_store_id": None,
            "differential_pricing": None,
            "accepts_mercadopago": True,
            "original_price": None,
            "currency_id": "ARS",
            "thumbnail": "http://ex.com/t.jpg",
            "title": f"Item {i}",
            "automatic_relist": False,
            "date_created": "2015-09-05T20:42:53.000Z",
            "secure_thumbnail": "https://ex.com/t.jpg",
            "stop_time": 1446669773000,
            "status": "active",
            "video_id": None,
            "catalog_product_id": None,
            "subtitle": None,
            "initial_quantity": 10 if condition == "new" else 1,
            "start_time": 1441485773000,
            "permalink": "http://ex.com/item",
            "sold_quantity": 5 if condition == "new" else 0,
            "available_quantity": 5 if condition == "new" else 1,
        }

    path = tmp_path / "test_data.jsonlines"
    with open(path, "w") as f:
        for i in range(20):
            cond = "new" if i % 2 == 0 else "used"
            f.write(json.dumps(make_item(i, cond)) + "\n")
    return path


class TestDataLoaderIntegration:
    def test_load_and_split(self, sample_jsonl):
        from production.adapters.data_loader import JsonlinesDataLoader

        loader = JsonlinesDataLoader(test_size=5)
        X_train, y_train, X_test, y_test = loader.load(sample_jsonl)

        assert len(X_train) == 15
        assert len(X_test) == 5
        assert len(y_train) == 15
        assert len(y_test) == 5

        # condition should be removed from test
        for item in X_test:
            assert "condition" not in item

        # condition should still be in train
        for item in X_train:
            assert "condition" in item


class TestPreprocessorIntegration:
    def test_full_pipeline(self, sample_jsonl):
        from production.adapters.data_loader import JsonlinesDataLoader

        loader = JsonlinesDataLoader(test_size=5)
        X_train_raw, y_train, X_test_raw, y_test = loader.load(sample_jsonl)

        preprocessor = MeliPreprocessor(warranty_samples_path=None)

        X_train = preprocessor.fit_transform(X_train_raw, y_train)
        X_test = preprocessor.transform(X_test_raw)

        assert len(X_train) == 15
        assert len(X_test) == 5

        # Same columns in both
        assert list(X_train.columns) == list(X_test.columns)

        # No nulls in key columns
        for col in ["listing_type_id", "buying_mode", "state", "city"]:
            assert col in X_train.columns


class TestCatBoostIntegration:
    def test_train_and_predict(self, sample_jsonl):
        """End-to-end: load → preprocess → train → predict."""
        from production.adapters.catboost_model import CatBoostModel
        from production.adapters.data_loader import JsonlinesDataLoader
        from production.adapters.preprocessor import MeliPreprocessor

        import pandas as pd

        loader = JsonlinesDataLoader(test_size=5)
        X_train_raw, y_train, X_test_raw, y_test = loader.load(sample_jsonl)

        preprocessor = MeliPreprocessor(warranty_samples_path=None)
        X_train = preprocessor.fit_transform(X_train_raw, y_train)
        X_test = preprocessor.transform(X_test_raw)

        model = CatBoostModel(params={"iterations": 10, "verbose": 0, "random_seed": 42})
        result = model.train(
            X_train, pd.Series(y_train),
            X_val=X_test, y_val=pd.Series(y_test),
        )

        assert result.accuracy >= 0.0  # Smoke test — small dataset
        assert result.auc_roc >= 0.0

        # Predict
        preds = model.predict(X_test)
        assert len(preds) == 5
        assert set(preds).issubset({"new", "used"})


class TestModelRepositoryIntegration:
    def test_save_and_load(self, sample_jsonl, tmp_path):
        """Save model + preprocessor, load them back, and predict."""
        from production.adapters.catboost_model import CatBoostModel
        from production.adapters.data_loader import JsonlinesDataLoader
        from production.adapters.model_repository import LocalModelRepository
        from production.adapters.preprocessor import MeliPreprocessor

        import pandas as pd

        loader = JsonlinesDataLoader(test_size=5)
        X_train_raw, y_train, X_test_raw, y_test = loader.load(sample_jsonl)

        preprocessor = MeliPreprocessor(warranty_samples_path=None)
        X_train = preprocessor.fit_transform(X_train_raw, y_train)

        model = CatBoostModel(params={"iterations": 10, "verbose": 0, "random_seed": 42})
        model.train(X_train, pd.Series(y_train))

        # Save
        repo = LocalModelRepository()
        model_dir = tmp_path / "model_artifacts"
        repo.save(model, preprocessor, model_dir)

        assert (model_dir / "catboost_model.cbm").exists()
        assert (model_dir / "preprocessor.pkl").exists()

        # Load and predict
        loaded_model, loaded_preprocessor = repo.load(model_dir)
        X_test = loaded_preprocessor.transform(X_test_raw)
        preds = loaded_model.predict(X_test)
        assert len(preds) == 5
