"""Layer 0: raw data loading, standardization, cleaning, and product master creation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config
from src.schemas import (
    EVENT_ALIASES,
    EVENTS_OPTIONAL_SAFE_DEFAULTS,
    EVENTS_REQUIRED,
    ORDER_ALIASES,
    ORDER_ITEMS_ALIASES,
    ORDER_ITEMS_OPTIONAL_SAFE_DEFAULTS,
    ORDER_ITEMS_REQUIRED,
    ORDERS_OPTIONAL_SAFE_DEFAULTS,
    ORDERS_REQUIRED,
    PRODUCT_ALIASES,
    PRODUCTS_OPTIONAL_SAFE_DEFAULTS,
    PRODUCTS_REQUIRED,
    REVIEW_ALIASES,
    REVIEWS_OPTIONAL_SAFE_DEFAULTS,
    REVIEWS_REQUIRED,
    USERS_ALIASES,
    USERS_OPTIONAL_SAFE_DEFAULTS,
)
from src.utils import (
    add_missing_columns,
    clean_numeric,
    coalesce_columns,
    dataframe_summary,
    ensure_dir,
    find_column,
    normalize_column_names,
    normalize_text_columns,
    parse_datetime_safe,
    resolve_input_csv,
    safe_read_csv,
    safe_write_csv,
    save_json,
    setup_logging,
    validate_required_columns,
)

LOGGER = logging.getLogger("spark.layer0")

ARABIC_STATUS_MAP = {
    "مكتمل": "completed",
    "قيد الشحن": "shipped",
    "قيد المعالجة": "processing",
    "ملغي": "cancelled",
    "ملغى": "cancelled",
    "completed": "completed",
    "delivered": "delivered",
    "shipped": "shipped",
    "paid": "paid",
    "processing": "processing",
    "cancelled": "cancelled",
    "canceled": "cancelled",
}


def _rename_with_aliases(df: pd.DataFrame, alias_map: dict[str, list[str]]) -> pd.DataFrame:
    out = normalize_column_names(df)
    rename_map: dict[str, str] = {}
    for canonical, candidates in alias_map.items():
        found = find_column(out, [c.lower() for c in candidates])
        if found and found != canonical:
            rename_map[found] = canonical
    return out.rename(columns=rename_map).copy()


def load_raw_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    LOGGER.info("Loading raw input files from %s", data_dir)

    def _read(filename: str, required: bool) -> pd.DataFrame:
        resolved = resolve_input_csv(data_dir, filename, required=required)
        if resolved is not None:
            LOGGER.info("Resolved %s -> %s", filename, resolved.name)
        return safe_read_csv(resolved, required=required)

    return {
        "products": _read(config.PRODUCTS_FILE, True),
        "events": _read(config.EVENTS_FILE, True),
        "orders": _read(config.ORDERS_FILE, True),
        "order_items": _read(config.ORDER_ITEMS_FILE, True),
        "reviews": _read(config.REVIEWS_FILE, False),
        "users": _read(config.USERS_FILE, False),
    }


def standardize_products(products: pd.DataFrame) -> pd.DataFrame:
    df = _rename_with_aliases(products, PRODUCT_ALIASES)
    validate_required_columns(df, PRODUCTS_REQUIRED, "products")
    df = add_missing_columns(df, PRODUCTS_OPTIONAL_SAFE_DEFAULTS)

    df = normalize_text_columns(df, ["name", "category", "subcategory", "brand", "description"])
    df["display_name"] = df["name"].fillna("unknown_product").astype(str)
    df["price"] = clean_numeric(df["price"]).fillna(0.0)
    df["image_count"] = clean_numeric(df["image_count"]).fillna(0)
    df["rating"] = clean_numeric(df["rating"]).fillna(0.0)
    df["review_count"] = clean_numeric(df["review_count"]).fillna(0)
    df["created_at"] = parse_datetime_safe(df["created_at"])
    df["image_url"] = df["image_url"].fillna("").astype(str).str.strip()
    df["has_image"] = df["image_url"].ne("")
    df["product_id"] = df["product_id"].astype(str).str.strip()
    df = df.dropna(subset=["product_id"]).drop_duplicates(subset=["product_id"], keep="first")
    return df.reset_index(drop=True)


def standardize_events(events: pd.DataFrame) -> pd.DataFrame:
    df = _rename_with_aliases(events, EVENT_ALIASES)
    validate_required_columns(df, EVENTS_REQUIRED, "events")
    df = add_missing_columns(df, EVENTS_OPTIONAL_SAFE_DEFAULTS)

    for col in ["event_id", "user_id", "session_id", "product_id"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    df["event_type"] = (
        df["event_type"].astype(str).str.strip().str.lower().replace(
            {
                "مشاهدة": "view",
                "مشاهده": "view",
                "view_item": "view",
                "product_view": "view",
                "page_view": "view",
                "اضافة للسلة": "cart",
                "اضافه للسله": "cart",
                "إضافة للسلة": "cart",
                "add_to_cart": "cart",
                "basket": "cart",
                "شراء": "purchase",
                "order": "purchase",
                "checkout": "purchase",
                "remove_from_basket": "remove_from_cart",
                "حذف من السلة": "remove_from_cart",
            }
        )
    )
    if "event_time" in df.columns:
        df["event_time"] = parse_datetime_safe(df["event_time"])
    if "session_id" in df.columns and "user_session" not in df.columns:
        df["user_session"] = df["session_id"]

    if "event_id" in df.columns and df["event_id"].notna().any():
        df = df.drop_duplicates(subset=["event_id"], keep="first")
    else:
        df = df.drop_duplicates()

    return df.reset_index(drop=True)


def standardize_orders(orders: pd.DataFrame) -> pd.DataFrame:
    df = _rename_with_aliases(orders, ORDER_ALIASES)
    validate_required_columns(df, ORDERS_REQUIRED, "orders")
    df = add_missing_columns(df, ORDERS_OPTIONAL_SAFE_DEFAULTS)

    for col in ["order_id", "user_id"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    df["status"] = df["status"].astype(str).str.strip().str.lower().replace(ARABIC_STATUS_MAP)
    df["total_amount"] = clean_numeric(df["total_amount"]).fillna(0.0)
    df["order_date"] = parse_datetime_safe(df["order_date"])
    df = df.dropna(subset=["order_id"]).drop_duplicates(subset=["order_id"], keep="first")
    return df.reset_index(drop=True)


def standardize_order_items(order_items: pd.DataFrame) -> pd.DataFrame:
    df = _rename_with_aliases(order_items, ORDER_ITEMS_ALIASES)
    validate_required_columns(df, ORDER_ITEMS_REQUIRED, "order_items")
    df = add_missing_columns(df, ORDER_ITEMS_OPTIONAL_SAFE_DEFAULTS)

    for col in ["order_id", "product_id", "user_id"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    if "item_price" in df.columns:
        df = coalesce_columns(df, "price", ["item_price", "price"], default=0.0)
        df = coalesce_columns(df, "unit_price", ["item_price", "unit_price", "price"], default=0.0)

    df["quantity"] = clean_numeric(df["quantity"]).fillna(1).clip(lower=1)
    df["price"] = clean_numeric(df["price"]).fillna(0.0)
    df["unit_price"] = clean_numeric(df["unit_price"]).fillna(df["price"])
    df = df.dropna(subset=["order_id", "product_id"]).drop_duplicates()
    return df.reset_index(drop=True)


def standardize_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    if reviews.empty:
        return pd.DataFrame(columns=list(REVIEWS_OPTIONAL_SAFE_DEFAULTS.keys()))

    df = _rename_with_aliases(reviews, REVIEW_ALIASES)
    validate_required_columns(df, REVIEWS_REQUIRED, "reviews")
    df = add_missing_columns(df, REVIEWS_OPTIONAL_SAFE_DEFAULTS)

    for col in ["product_id", "user_id"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    df["rating"] = clean_numeric(df["rating"]).fillna(0.0)
    df["review_text"] = df["review_text"].fillna("").astype(str)
    df["review_date"] = parse_datetime_safe(df["review_date"])
    df = df.dropna(subset=["product_id"]).drop_duplicates()
    return df.reset_index(drop=True)


def standardize_users(users: pd.DataFrame) -> pd.DataFrame:
    if users.empty:
        return pd.DataFrame(columns=list(USERS_OPTIONAL_SAFE_DEFAULTS.keys()))

    df = _rename_with_aliases(users, USERS_ALIASES)
    df = add_missing_columns(df, USERS_OPTIONAL_SAFE_DEFAULTS)
    if "user_id" in df.columns:
        df["user_id"] = df["user_id"].astype(str).str.strip()
    if "age" in df.columns:
        df["age"] = clean_numeric(df["age"])
    return df.drop_duplicates().reset_index(drop=True)


def build_product_master(products: pd.DataFrame, reviews: pd.DataFrame | None = None) -> pd.DataFrame:
    df = products.copy()
    if reviews is not None and not reviews.empty:
        review_agg = reviews.groupby("product_id", as_index=False).agg(
            reviews_rating_avg=("rating", "mean"),
            reviews_count=("product_id", "size"),
        )
        df = df.merge(review_agg, on="product_id", how="left")
        df["rating"] = df["rating"].fillna(df["reviews_rating_avg"])
        df["review_count"] = df["review_count"].fillna(df["reviews_count"])

    today = pd.Timestamp.now().normalize()
    df["description"] = df["description"].fillna("").astype(str)
    df["description_length"] = df["description"].str.len().fillna(0)
    df["description_length_chars"] = df["description_length"]
    df["description_length_words"] = df["description"].str.split().map(len).fillna(0)
    df["number_of_images"] = df["image_count"].fillna(0)
    if "category_code" not in df.columns:
        df["category_code"] = df["category"]
    df["product_age_days"] = np.where(
        df["created_at"].notna(),
        (today - pd.to_datetime(df["created_at"]).dt.normalize()).dt.days.clip(lower=0),
        np.nan,
    )

    keep_cols = [
        "product_id",
        "name",
        "display_name",
        "category",
        "subcategory",
        "brand",
        "price",
        "description_length",
        "description_length_words",
        "description_length_chars",
        "image_count",
        "number_of_images",
        "image_url",
        "has_image",
        "category_code",
        "rating",
        "review_count",
        "product_age_days",
        "created_at",
    ]
    return df[keep_cols].drop_duplicates(subset=["product_id"]).reset_index(drop=True)


def generate_quality_report(data_map: dict[str, pd.DataFrame]) -> dict[str, Any]:
    return {
        "datasets": {name: dataframe_summary(df) for name, df in data_map.items()},
        "optional_inputs_present": {
            "reviews": not data_map.get("reviews", pd.DataFrame()).empty,
            "users": not data_map.get("users", pd.DataFrame()).empty,
        },
    }


def generate_data_coverage(cleaned_events: pd.DataFrame, cleaned_orders: pd.DataFrame, cleaned_reviews: pd.DataFrame) -> dict[str, Any]:
    coverage: dict[str, Any] = {}
    if not cleaned_events.empty and "event_time" in cleaned_events.columns:
        coverage["events_date_range"] = {
            "min": str(pd.to_datetime(cleaned_events["event_time"], errors="coerce").min()),
            "max": str(pd.to_datetime(cleaned_events["event_time"], errors="coerce").max()),
        }
    if not cleaned_orders.empty and "order_date" in cleaned_orders.columns:
        coverage["orders_date_range"] = {
            "min": str(pd.to_datetime(cleaned_orders["order_date"], errors="coerce").min()),
            "max": str(pd.to_datetime(cleaned_orders["order_date"], errors="coerce").max()),
        }
    if not cleaned_reviews.empty and "review_date" in cleaned_reviews.columns:
        coverage["reviews_date_range"] = {
            "min": str(pd.to_datetime(cleaned_reviews["review_date"], errors="coerce").min()),
            "max": str(pd.to_datetime(cleaned_reviews["review_date"], errors="coerce").max()),
        }
    return coverage


def run_layer0(data_dir: Path, output_dir: Path) -> dict[str, pd.DataFrame]:
    setup_logging()
    ensure_dir(output_dir)
    LOGGER.info("Starting Layer 0")

    raw = load_raw_data(data_dir)
    cleaned_products = standardize_products(raw["products"])
    cleaned_events = standardize_events(raw["events"])
    cleaned_orders = standardize_orders(raw["orders"])
    cleaned_order_items = standardize_order_items(raw["order_items"])
    cleaned_reviews = standardize_reviews(raw["reviews"])
    cleaned_users = standardize_users(raw["users"])
    product_master = build_product_master(cleaned_products, cleaned_reviews if not cleaned_reviews.empty else None)

    safe_write_csv(cleaned_products, output_dir / config.LAYER0_PRODUCTS_FILE)
    safe_write_csv(cleaned_events, output_dir / config.LAYER0_EVENTS_FILE)
    safe_write_csv(cleaned_orders, output_dir / config.LAYER0_ORDERS_FILE)
    safe_write_csv(cleaned_order_items, output_dir / config.LAYER0_ORDER_ITEMS_FILE)
    # Notebook-compatible alias files
    safe_write_csv(cleaned_events, output_dir / "events_clean.csv")
    safe_write_csv(cleaned_order_items, output_dir / "order_items_clean.csv")
    safe_write_csv(cleaned_orders, output_dir / "orders_clean.csv")
    safe_write_csv(cleaned_products, output_dir / "products_clean.csv")
    if not cleaned_reviews.empty:
        safe_write_csv(cleaned_reviews, output_dir / config.LAYER0_REVIEWS_FILE)
        safe_write_csv(cleaned_reviews, output_dir / "reviews_clean.csv")
    if not cleaned_users.empty:
        safe_write_csv(cleaned_users, output_dir / config.LAYER0_USERS_FILE)
        safe_write_csv(cleaned_users, output_dir / "users_clean.csv")
    safe_write_csv(product_master, output_dir / config.LAYER0_PRODUCT_MASTER_FILE)
    safe_write_csv(product_master, output_dir / "product_master_layer0.csv")

    save_json(
        generate_quality_report(
            {
                "products": cleaned_products,
                "events": cleaned_events,
                "orders": cleaned_orders,
                "order_items": cleaned_order_items,
                "reviews": cleaned_reviews,
                "users": cleaned_users,
                "product_master": product_master,
            }
        ),
        output_dir / config.LAYER0_QUALITY_FILE,
    )
    save_json(
        generate_data_coverage(cleaned_events, cleaned_orders, cleaned_reviews),
        output_dir / config.LAYER0_COVERAGE_FILE,
    )

    LOGGER.info("Layer 0 finished successfully")
    return {
        "cleaned_products": cleaned_products,
        "cleaned_events": cleaned_events,
        "cleaned_orders": cleaned_orders,
        "cleaned_order_items": cleaned_order_items,
        "cleaned_reviews": cleaned_reviews,
        "cleaned_users": cleaned_users,
        "product_master": product_master,
    }
