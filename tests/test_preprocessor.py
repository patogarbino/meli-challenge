"""Tests for the MeliPreprocessor adapter."""

import numpy as np
import pandas as pd
import pytest

from production.adapters.preprocessor import MeliPreprocessor

# ── Fixtures ───────────────────────────────────────────────────────────────────


def _make_raw_item(**overrides) -> dict:
    """Create a minimal raw item dict for testing."""
    base = {
        "seller_address": {
            "country": {"name": "Argentina", "id": "AR"},
            "state": {"name": "Capital Federal", "id": "AR-C"},
            "city": {"name": "Palermo", "id": "XXX"},
        },
        "warranty": None,
        "sub_status": [],
        "condition": "new",
        "deal_ids": [],
        "base_price": 100.0,
        "shipping": {
            "local_pick_up": True,
            "methods": [],
            "tags": [],
            "free_shipping": False,
            "mode": "me2",
            "dimensions": None,
        },
        "non_mercado_pago_payment_methods": [
            {"description": "Transferencia bancaria", "id": "MLATB", "type": "G"}
        ],
        "seller_id": 123456,
        "variations": [],
        "site_id": "MLA",
        "listing_type_id": "bronze",
        "price": 100.0,
        "attributes": [],
        "buying_mode": "buy_it_now",
        "tags": ["dragged_bids_and_visits"],
        "listing_source": "",
        "parent_item_id": None,
        "coverage_areas": [],
        "category_id": "MLA1234",
        "descriptions": [{"id": "MLA123-456"}],
        "last_updated": "2015-09-05T20:42:58.000Z",
        "international_delivery_mode": "none",
        "pictures": [
            {
                "size": "500x375",
                "max_size": "1200x900",
                "url": "http://example.com/pic.jpg",
                "secure_url": "https://example.com/pic.jpg",
                "quality": "",
                "id": "123-MLA",
            }
        ],
        "id": "MLA123",
        "official_store_id": None,
        "differential_pricing": None,
        "accepts_mercadopago": True,
        "original_price": None,
        "currency_id": "ARS",
        "thumbnail": "http://example.com/thumb.jpg",
        "title": "Test Item Title",
        "automatic_relist": False,
        "date_created": "2015-09-05T20:42:53.000Z",
        "secure_thumbnail": "https://example.com/thumb.jpg",
        "stop_time": 1446669773000,
        "status": "active",
        "video_id": None,
        "catalog_product_id": None,
        "subtitle": None,
        "initial_quantity": 5,
        "start_time": 1441485773000,
        "permalink": "http://example.com/item",
        "sold_quantity": 2,
        "available_quantity": 3,
    }
    base.update(overrides)
    return base


@pytest.fixture
def raw_items():
    """Two sample items: one new, one used."""
    return [
        _make_raw_item(condition="new", price=500.0, initial_quantity=10),
        _make_raw_item(
            condition="used",
            price=50.0,
            listing_type_id="free",
            initial_quantity=1,
            warranty="6 meses de garantia",
        ),
    ]


@pytest.fixture
def preprocessor():
    """Preprocessor without warranty classifier (no samples CSV)."""
    return MeliPreprocessor(warranty_samples_path=None)


# ── Tests ──────────────────────────────────────────────────────────────────────


class TestMeliPreprocessor:
    def test_fit_transform_returns_dataframe(self, preprocessor, raw_items):
        result = preprocessor.fit_transform(raw_items, y=["new", "used"])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_condition_column_dropped(self, preprocessor, raw_items):
        result = preprocessor.fit_transform(raw_items)
        assert "condition" not in result.columns

    def test_irrelevant_columns_dropped(self, preprocessor, raw_items):
        result = preprocessor.fit_transform(raw_items)
        for col in ["site_id", "id", "permalink", "thumbnail", "subtitle"]:
            assert col not in result.columns, f"{col} should have been dropped"

    def test_address_extracted(self, preprocessor, raw_items):
        result = preprocessor.fit_transform(raw_items)
        assert "state" in result.columns
        assert "city" in result.columns
        assert "seller_address" not in result.columns
        assert result["state"].iloc[0] == "Capital Federal"

    def test_shipping_parsed(self, preprocessor, raw_items):
        result = preprocessor.fit_transform(raw_items)
        assert "shipping_local_pick_up" in result.columns
        assert "shipping_free" in result.columns
        assert "shipping_mode" in result.columns
        assert "shipping" not in result.columns

    def test_list_columns_counted(self, preprocessor, raw_items):
        result = preprocessor.fit_transform(raw_items)
        assert "non_mercado_pago_payment_methods_count" in result.columns
        assert "variations_count" in result.columns
        assert "tags_count" in result.columns
        assert "attributes_count" in result.columns

    def test_pictures_parsed(self, preprocessor, raw_items):
        result = preprocessor.fit_transform(raw_items)
        assert "pictures_count" in result.columns
        assert "pictures_max_area" in result.columns
        assert result["pictures_count"].iloc[0] == 1
        assert result["pictures_max_area"].iloc[0] == 1200 * 900

    def test_bool_columns_created(self, preprocessor, raw_items):
        result = preprocessor.fit_transform(raw_items)
        for col in ["deal_ids", "official_store_id", "video_id"]:
            assert col in result.columns
            assert set(result[col].unique()).issubset({0, 1})

    def test_warranty_fallback_without_samples(self, preprocessor, raw_items):
        result = preprocessor.fit_transform(raw_items)
        assert "warranty" in result.columns
        # Without classifier, falls back to "Sin dato" / "Con garantía"
        assert result["warranty"].iloc[0] == "Sin dato"  # None warranty
        assert result["warranty"].iloc[1] == "Con garantía"  # has warranty text

    def test_transform_after_fit(self, preprocessor, raw_items):
        preprocessor.fit_transform(raw_items)
        result = preprocessor.transform(raw_items)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_transform_without_fit_raises(self, preprocessor, raw_items):
        with pytest.raises(RuntimeError, match="not fitted"):
            preprocessor.transform(raw_items)

    def test_output_column_count(self, preprocessor, raw_items):
        result = preprocessor.fit_transform(raw_items)
        # Expected: ~22 columns matching the CatBoost training notebook
        expected_cols = {
            "base_price", "price", "initial_quantity", "sold_quantity",
            "available_quantity", "listing_type_id", "buying_mode",
            "category_id", "shipping_local_pick_up", "shipping_free",
            "shipping_mode", "state", "city", "warranty",
            "deal_ids", "official_store_id", "video_id",
            "non_mercado_pago_payment_methods_count", "variations_count",
            "tags_count", "attributes_count", "pictures_count",
            "pictures_max_area", "accepts_mercadopago", "automatic_relist",
        }
        for col in expected_cols:
            assert col in result.columns, f"Missing expected column: {col}"
