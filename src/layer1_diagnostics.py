"""Layer 1: notebook-aligned hybrid diagnostics (quartiles + z-score ranking) with UI-ready outputs."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src.utils import (
    ensure_dir,
    filter_df_by_date_range,
    safe_read_csv,
    safe_write_csv,
    save_json,
    setup_logging,
)

LOGGER = logging.getLogger("spark.layer1")

EVENT_MAP = {
    "مشاهدة": "view",
    "مشاهده": "view",
    "view": "view",
    "view_item": "view",
    "product_view": "view",
    "page_view": "view",
    "اضافة للسلة": "cart",
    "اضافه للسله": "cart",
    "إضافة للسلة": "cart",
    "اضافه للسلة": "cart",
    "إضافه للسله": "cart",
    "cart": "cart",
    "add_to_cart": "cart",
    "basket": "cart",
    "شراء": "purchase",
    "purchase": "purchase",
    "checkout": "purchase",
    "order": "purchase",
    "حذف من السلة": "remove_from_cart",
    "remove_from_cart": "remove_from_cart",
}

VALID_ORDER_STATUSES = {"completed", "paid", "delivered", "shipped", "processing", "success", "fulfilled"}


def load_layer0_outputs(layer0_dir: Path) -> dict[str, pd.DataFrame]:
    product_master = safe_read_csv(layer0_dir / config.LAYER0_PRODUCT_MASTER_FILE, required=True)
    events = safe_read_csv(layer0_dir / config.LAYER0_EVENTS_FILE, required=True)
    orders = safe_read_csv(layer0_dir / config.LAYER0_ORDERS_FILE, required=True)
    order_items = safe_read_csv(layer0_dir / config.LAYER0_ORDER_ITEMS_FILE, required=True)

    for df in [product_master, events, orders, order_items]:
        for col in ["product_id", "order_id", "user_id", "session_id"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
    return {
        "product_master": product_master,
        "events": events,
        "orders": orders,
        "order_items": order_items,
    }


def normalize_event_types(events: pd.DataFrame) -> pd.DataFrame:
    df = events.copy()
    df["event_type"] = (
        df["event_type"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(lambda x: EVENT_MAP.get(x, x))
    )
    return df


def build_funnel_table(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["product_id", "view", "cart", "purchase"])

    value_col = "user_id" if "user_id" in events.columns else "product_id"
    prod_counts = (
        events.pivot_table(
            index="product_id",
            columns="event_type",
            values=value_col,
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
    )

    for col in ["view", "cart", "purchase"]:
        if col not in prod_counts.columns:
            prod_counts[col] = 0
    return prod_counts


def zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    std = s.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def build_layer1_core_metrics(events: pd.DataFrame) -> pd.DataFrame:
    diag = build_funnel_table(events).copy()

    diag["view_to_cart_rate"] = np.where(diag["view"] > 0, diag["cart"] / diag["view"], 0.0)
    diag["cart_to_purchase_rate"] = np.where(diag["cart"] > 0, diag["purchase"] / diag["cart"], 0.0)
    diag["view_to_purchase_rate"] = np.where(diag["view"] > 0, diag["purchase"] / diag["view"], 0.0)
    diag["cart_abandon_rate"] = np.where(diag["cart"] > 0, 1 - (diag["purchase"] / diag["cart"]), 0.0)

    views_q = diag["view"].quantile([0.25, 0.75]) if len(diag) else pd.Series([0, 0], index=[0.25, 0.75])
    purchase_q = diag["purchase"].quantile([0.25, 0.75]) if len(diag) else pd.Series([0, 0], index=[0.25, 0.75])

    def classify(row: pd.Series) -> str:
        if row["view"] >= views_q.loc[0.75] and row["purchase"] >= purchase_q.loc[0.75]:
            return "Strong"
        elif row["view"] <= views_q.loc[0.25] and row["purchase"] <= purchase_q.loc[0.25]:
            return "Weak"
        return "Moderate"

    diag["product_class"] = diag.apply(classify, axis=1)

    metrics = [
        "view",
        "cart",
        "purchase",
        "view_to_purchase_rate",
        "cart_to_purchase_rate",
        "cart_abandon_rate",
    ]
    weights = {
        "view": 1.0,
        "cart": 1.0,
        "purchase": 1.5,
        "view_to_purchase_rate": 1.2,
        "cart_to_purchase_rate": 1.2,
        "cart_abandon_rate": -1.0,
    }

    score = 0.0
    weight_sum = 0.0
    for metric in metrics:
        diag[f"z_{metric}"] = zscore(diag[metric])
        score += weights[metric] * diag[f"z_{metric}"]
        weight_sum += abs(weights[metric])

    diag["z_score"] = score / weight_sum
    diag["priority_rank"] = diag.groupby("product_class")["z_score"].rank(ascending=False, method="dense")

    # Compatibility aliases from notebook
    diag["z_composite_score"] = diag["z_score"]
    diag["rank_within_class_best"] = diag["priority_rank"]
    diag["action_priority"] = diag["priority_rank"]
    diag["views"] = diag["view"]
    diag["cart_adds"] = diag["cart"]
    diag["total_orders"] = diag["purchase"]
    diag["conversion_rate"] = diag["view_to_purchase_rate"]

    return diag


def build_unique_user_metrics(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["product_id", "unique_viewers", "unique_cart_users", "event_unique_buyers"])

    user_col = "user_id" if "user_id" in events.columns else None
    if user_col is None:
        return pd.DataFrame(columns=["product_id", "unique_viewers", "unique_cart_users", "event_unique_buyers"])

    rows = []
    for event_type, out_col in [
        ("view", "unique_viewers"),
        ("cart", "unique_cart_users"),
        ("purchase", "event_unique_buyers"),
    ]:
        sub = events[events["event_type"] == event_type].copy()
        if sub.empty:
            metric = pd.DataFrame(columns=["product_id", out_col])
        else:
            metric = (
                sub.dropna(subset=[user_col])
                .groupby("product_id", as_index=False)[user_col]
                .nunique()
                .rename(columns={user_col: out_col})
            )
        rows.append(metric)

    merged = rows[0]
    for frame in rows[1:]:
        merged = merged.merge(frame, on="product_id", how="outer")
    return merged.fillna(0.0)


def build_order_metrics(orders: pd.DataFrame, order_items: pd.DataFrame) -> pd.DataFrame:
    if orders.empty or order_items.empty:
        return pd.DataFrame(
            columns=["product_id", "revenue", "ordered_quantity", "purchase_orders", "order_unique_buyers"]
        )

    valid_orders = orders.copy()
    if "status" in valid_orders.columns:
        valid_orders = valid_orders[valid_orders["status"].astype(str).str.lower().isin(VALID_ORDER_STATUSES)].copy()

    if valid_orders.empty:
        return pd.DataFrame(
            columns=["product_id", "revenue", "ordered_quantity", "purchase_orders", "order_unique_buyers"]
        )

    merged = order_items.merge(valid_orders[["order_id", "user_id"]], on="order_id", how="inner", suffixes=("", "_order"))
    if "user_id_order" in merged.columns:
        if "user_id" in merged.columns:
            merged["buyer_user_id"] = merged["user_id"].where(merged["user_id"].notna() & (merged["user_id"] != ""), merged["user_id_order"])
        else:
            merged["buyer_user_id"] = merged["user_id_order"]
    elif "user_id" in merged.columns:
        merged["buyer_user_id"] = merged["user_id"]
    else:
        merged["buyer_user_id"] = np.nan

    merged["quantity"] = pd.to_numeric(merged.get("quantity", 1), errors="coerce").fillna(1).clip(lower=1)
    unit_price = pd.to_numeric(merged.get("unit_price", merged.get("price", 0.0)), errors="coerce")
    price = pd.to_numeric(merged.get("price", merged.get("unit_price", 0.0)), errors="coerce")
    merged["effective_price"] = unit_price.fillna(price).fillna(0.0)
    merged["line_revenue"] = merged["quantity"] * merged["effective_price"]

    out = (
        merged.groupby("product_id", as_index=False)
        .agg(
            revenue=("line_revenue", "sum"),
            ordered_quantity=("quantity", "sum"),
            purchase_orders=("order_id", "nunique"),
            order_unique_buyers=("buyer_user_id", "nunique"),
        )
    )
    return out


def classify_status_sub_label(row: pd.Series) -> str:
    if bool(row.get("high_abandonment_flag", False)):
        return "انسحاب مرتفع بعد الإضافة للسلة"
    if bool(row.get("low_visibility_flag", False)):
        return "ظهور أقل من المطلوب"
    if bool(row.get("low_conversion_flag", False)):
        return "تحويل أقل من المتوقع"
    if row["confidence_level"] == "Low":
        return "إشارات أولية"
    if row["product_class"] == "Strong":
        return "أداء مستقر"
    return "يحتاج متابعة"


def status_label_for_class(product_class: str, confidence_level: str) -> str:
    return {
        "Weak": "ضعيف",
        "Moderate": "متوسط",
        "Strong": "قوي",
    }.get(product_class, "تحت المراجعة")


def diagnostic_reason_short(row: pd.Series) -> str:
    if bool(row.get("high_abandonment_flag", False)):
        return "انسحاب مرتفع بعد الإضافة للسلة"
    if bool(row.get("low_visibility_flag", False)):
        return "مشاهدات أقل من المطلوب"
    if bool(row.get("low_conversion_flag", False)):
        return "تحويل أقل من المتوقع"
    if row["confidence_level"] == "Low":
        return "إشارات البيع أقوى من إشارات التصفح"
    if row["product_class"] == "Strong":
        return "أداء متماسك"
    return "أداء متوسط"


def build_ui_ready_outputs(
    product_master: pd.DataFrame,
    diag_core: pd.DataFrame,
    unique_metrics: pd.DataFrame,
    order_metrics: pd.DataFrame,
    start_date: pd.Timestamp | None,
    end_date: pd.Timestamp | None,
) -> pd.DataFrame:
    df = product_master.merge(diag_core, on="product_id", how="left")
    df = df.merge(unique_metrics, on="product_id", how="left")
    df = df.merge(order_metrics, on="product_id", how="left")

    for col in [
        "view", "cart", "purchase", "views", "cart_adds", "total_orders",
        "revenue", "ordered_quantity", "purchase_orders", "order_unique_buyers",
        "unique_viewers", "unique_cart_users", "event_unique_buyers",
    ]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # separate purchase signals to avoid funnel inversion
    df["purchase_signal"] = np.maximum(df["purchase"], df["purchase_orders"])
    df["unique_buyers_raw"] = np.maximum(df["event_unique_buyers"], df["order_unique_buyers"])
    df["unique_buyers"] = np.where(
        df["unique_viewers"] > 0,
        np.minimum(df["unique_buyers_raw"], np.maximum(df["unique_viewers"], 1)),
        df["unique_buyers_raw"],
    )
    df["unique_cart_users_bounded"] = np.where(
        df["unique_viewers"] > 0,
        np.minimum(df["unique_cart_users"], np.maximum(df["unique_viewers"], 1)),
        df["unique_cart_users"],
    )

    df["sales_without_view_signal_flag"] = (df["purchase_signal"] > 0) & (df["view"] == 0)

    # UI-facing aliases
    df["carts"] = df["cart"]
    # FIX: To prevent UI inversion, we use a bounded purchase signal for display in the main funnel
    # while keeping purchase_signal (max) for diagnostic logic.
    df["purchases"] = np.where(df["view"] > 0, np.minimum(df["purchase_signal"], df["view"]), df["purchase_signal"])
    
    df["purchase_events"] = df["purchase"]
    df["total_views"] = df["view"]
    df["total_cart_adds"] = df["cart"]
    df["total_event_purchases"] = df["purchase"]
    df["total_sales_value"] = df["revenue"]
    df["unique_order_count"] = df["purchase_orders"]
    df["avg_review_rating"] = df["rating"]

    # FIX #1 & #2: Separate event and order conversion paths
    # Event-based conversion (Funnel consistency)
    df["event_conversion"] = df["purchase"] / np.maximum(df["view"], 1)
    # Order-based conversion (Real-world success)
    df["order_conversion"] = df["purchase_orders"] / np.maximum(df["unique_viewers"], 1)

    # Use order_conversion as the primary purchase_conversion_rate but without bounding to view
    df["purchase_conversion_rate"] = df["order_conversion"]

    df["cart_rate"] = np.where(df["view"] > 0, df["cart"] / np.maximum(df["view"], 1), 0.0)

    # FIX #3: Calculate abandonment based on event-only data to avoid inversion
    df["cart_to_purchase_rate"] = np.where(df["cart"] > 0, df["purchase"] / np.maximum(df["cart"], 1), np.nan)
    df["cart_abandonment_rate"] = np.where(df["cart"] > 0, (1 - df["cart_to_purchase_rate"]).clip(0, 1), np.nan)

    # notebook-aligned ranking, but with order-backed purchase evidence
    view_q = df["view"].quantile([0.25, 0.75]) if len(df) else pd.Series([0, 0], index=[0.25, 0.75])
    purchase_q = df["purchase_signal"].quantile([0.25, 0.75]) if len(df) else pd.Series([0, 0], index=[0.25, 0.75])
    conv_q = df["purchase_conversion_rate"].quantile([0.25, 0.75]) if len(df) else pd.Series([0, 0], index=[0.25, 0.75])
    revenue_q75 = df["revenue"].quantile(0.75) if len(df) else 0

    df["product_class"] = np.select(
        [
            (df["purchase_signal"] >= purchase_q.loc[0.75]) & ((df["purchase_conversion_rate"] >= conv_q.loc[0.75]) | (df["revenue"] >= revenue_q75)),
            (df["purchase_signal"] <= purchase_q.loc[0.25]) & (df["view"] <= view_q.loc[0.25]) & (df["purchase_conversion_rate"] <= conv_q.loc[0.25]),
        ],
        ["Strong", "Weak"],
        default="Moderate",
    )

    # Override: products with strong ORDER signal should not be stranded at Weak.
    # When event purchase tracking is incomplete, high-order products can score low
    # on view z-score and land in Weak despite clear real-world demand.
    strong_order_override = (
        (df["product_class"] == "Weak") &
        ((df["purchase_orders"] >= purchase_q.loc[0.75]) | (df["revenue"] >= revenue_q75))
    )
    df.loc[strong_order_override, "product_class"] = "Moderate"

    score_inputs = {
        "view": 1.0,
        "cart": 1.0,
        "purchase_signal": 1.5,
        "purchase_conversion_rate": 1.2,
        "cart_to_purchase_rate": 1.0,
        "cart_abandonment_rate": -1.0,
        "revenue": 0.8,
    }
    composite = 0.0
    weight_sum = 0.0
    for metric, weight in score_inputs.items():
        series = pd.to_numeric(df[metric], errors="coerce").fillna(0.0)
        std = series.std(ddof=0)
        z_val = pd.Series(0.0, index=series.index) if std == 0 or pd.isna(std) else (series - series.mean()) / std
        df[f"z_{metric}"] = z_val
        composite += weight * z_val
        weight_sum += abs(weight)
    df["z_score"] = composite / weight_sum
    df["z_composite_score"] = df["z_score"]
    df["priority_rank"] = df.groupby("product_class")["z_score"].rank(ascending=False, method="dense")
    df["rank_within_class_best"] = df["priority_rank"]
    df["action_priority"] = df["priority_rank"]

    benchmarks = (
        df.groupby("category", dropna=False, as_index=False)
        .agg(
            category_avg_conversion=("purchase_conversion_rate", "mean"),
            category_median_conversion=("purchase_conversion_rate", "median"),
            category_avg_views=("view", "mean"),
        )
    )
    df = df.merge(benchmarks, on="category", how="left")
    df["relative_conversion_vs_category"] = np.where(
        df["category_avg_conversion"] > 0,
        df["purchase_conversion_rate"] / df["category_avg_conversion"],
        0.0,
    )

    weak_view_q = df["view"].quantile(0.25) if len(df) else 0
    weak_conv_q = df["purchase_conversion_rate"].quantile(0.25) if len(df) else 0
    abandon_q = df["cart_abandonment_rate"].dropna().quantile(0.75) if df["cart_abandonment_rate"].notna().any() else 0

    df["insufficient_views_flag"] = df["view"] < max(5, min(config.MIN_VIEWS_FOR_CONFIDENT_DIAGNOSIS // 2, 10))
    df["insufficient_purchases_flag"] = df["purchase_signal"] < max(1, config.MIN_PURCHASES_FOR_CONFIDENT_DIAGNOSIS)
    df["insufficient_cart_data_flag"] = df["cart"] <= 0

    df["confidence_level"] = np.select(
        [
            ((df["view"] >= 10) & (df["purchase_signal"] >= 2)) | (df["purchase_orders"] >= 8) | (df["revenue"] >= 10000),
            ((df["view"] >= 5) & ((df["purchase_signal"] >= 1) | (df["cart"] >= 2))) | (df["purchase_orders"] >= 3) | (df["revenue"] >= 3000),
        ],
        ["High", "Medium"],
        default="Low",
    )
    # Defensive upgrade: ensure purchase_orders-based confidence is never suppressed
    # by low view counts when order evidence is strong. (Guards against stale purchase_signal.)
    df.loc[
        (df["confidence_level"] == "Low") & (df["purchase_orders"] >= 8),
        "confidence_level",
    ] = "High"
    df.loc[
        (df["confidence_level"] == "Low") & (df["purchase_orders"] >= 3),
        "confidence_level",
    ] = "Medium"
    df["insufficient_data_flag"] = False

    df["low_visibility_flag"] = df["view"] <= weak_view_q
    # FIX #4: Prevent low_conversion_flag on Strong products + use updated conversion rate
    df["low_conversion_flag"] = (df["purchase_conversion_rate"] <= weak_conv_q) & (df["product_class"] != "Strong")
    df["high_abandonment_flag"] = (df["cart"] > 0) & (df["cart_abandonment_rate"].fillna(0) >= abandon_q) & (df["cart_abandonment_rate"].fillna(0) > 0)
    df["low_engagement_flag"] = df["cart"] <= df["cart"].quantile(0.25) if len(df) else False

    df["diagnostic_status_label"] = [
        status_label_for_class(pc, conf) for pc, conf in zip(df["product_class"], df["confidence_level"])
    ]
    df["diagnostic_status_sub_label"] = df.apply(classify_status_sub_label, axis=1)
    df["diagnostic_reason_short"] = df.apply(diagnostic_reason_short, axis=1)

    reason_map = {
        "Weak": "يظهر المنتج إشارات أداء أضعف من بقية الكتالوج خلال الفترة المختارة.",
        "Moderate": "أداء المنتج متوسط ويمكن تحسينه من خلال تدخلات مركزة.",
        "Strong": "المنتج يحقق أداء جيدًا مقارنة ببقية المنتجات.",
    }
    df["diagnostic_reason"] = df["product_class"].map(reason_map).fillna("تحت المراجعة.")
    # Logic Fix: Only flag abandonment if conversion is actually poor. 
    # If conversion is high (e.g. 100%), abandonment is statistically irrelevant or a tracking quirk.
    df["high_abandonment_flag"] = (
        (df["cart"] > 0) & 
        (df["cart_abandonment_rate"].fillna(0) >= abandon_q) & 
        (df["cart_abandonment_rate"].fillna(0) > 0.1) & # At least 10% abandonment to be meaningful
        (df["purchase_conversion_rate"] < df["category_avg_conversion"] * 1.2) # Don't flag if conversion is already excellent
    )

    df["diagnostic_reason"] = df["product_class"].map(reason_map).fillna("تحت المراجعة.")
    
    # Priority-based reasoning to avoid "overwriting" a good status with a bad flag
    # 1. Low Confidence (Data insufficiency)
    df.loc[(df["confidence_level"] == "Low") & (df["purchase_orders"] > 0), "diagnostic_reason"] = "المبيعات المسجلة توضح وجود طلب، لكن إشارات التصفح داخل الفترة أضعف من أن تشرح القصة كاملة."
    df.loc[(df["confidence_level"] == "Low") & (df["purchase_orders"] <= 0), "diagnostic_reason"] = "المنتج يحتاج متابعة إضافية قبل اتخاذ تدخل كبير، لأن الإشارات الحالية ما زالت محدودة."
    
    # 2. Issues for Non-Strong products
    df.loc[(df["product_class"] != "Strong") & df["low_visibility_flag"], "diagnostic_reason"] = "المنتج لا يحصل على عدد كافٍ من الزيارات خلال الفترة المختارة."
    df.loc[(df["product_class"] != "Strong") & df["low_conversion_flag"] & ~df["low_visibility_flag"], "diagnostic_reason"] = "المنتج يُشاهَد لكنه لا يتحول إلى شراء بالمستوى المطلوب."
    df.loc[(df["product_class"] != "Strong") & df["high_abandonment_flag"] & ~df["low_conversion_flag"], "diagnostic_reason"] = "هناك اهتمام واضح بالمنتج لكن جزءًا من العملاء لا يكمل الشراء بعد الإضافة للسلة."

    df["analysis_start_date"] = str(start_date.date()) if start_date is not None else None
    df["analysis_end_date"] = str(end_date.date()) if end_date is not None else None

    if "image_count" in df.columns and "number_of_images" not in df.columns:
        df["number_of_images"] = df["image_count"]
    if "description_length" in df.columns and "description_length_chars" not in df.columns:
        df["description_length_chars"] = df["description_length"]
    if "description_length_words" not in df.columns:
        df["description_length_words"] = (df.get("description_length", 0) / 5).fillna(0).round().astype(int)

    keep_cols = list(dict.fromkeys([
        "product_id", "name", "display_name", "category", "subcategory", "brand", "price", "image_url", "has_image",
        "view", "cart", "purchase", "purchase_signal", "views", "cart_adds", "total_orders", "carts", "purchases",
        "unique_viewers", "unique_cart_users", "event_unique_buyers", "order_unique_buyers",
        "unique_buyers_raw", "unique_cart_users_bounded", "unique_buyers",
        "sales_without_view_signal_flag", "revenue", "ordered_quantity", "purchase_orders",
        "view_to_cart_rate", "cart_to_purchase_rate", "view_to_purchase_rate", "cart_abandon_rate",
        "cart_rate", "purchase_conversion_rate", "cart_to_purchase_rate", "cart_abandonment_rate",
        "product_class", "z_score", "z_composite_score", "priority_rank", "rank_within_class_best", "action_priority",
        "conversion_rate", "category_avg_conversion", "category_median_conversion", "category_avg_views",
        "relative_conversion_vs_category", "insufficient_views_flag", "insufficient_purchases_flag",
        "insufficient_cart_data_flag", "confidence_level", "insufficient_data_flag",
        "low_visibility_flag", "high_abandonment_flag", "low_conversion_flag", "low_engagement_flag",
        "diagnostic_status_label", "diagnostic_status_sub_label", "diagnostic_reason_short", "diagnostic_reason",
        "rating", "review_count", "product_age_days", "description_length", "description_length_words",
        "description_length_chars", "number_of_images", "total_sales_value", "unique_order_count",
        "avg_review_rating", "total_views", "total_cart_adds", "total_event_purchases",
        "analysis_start_date", "analysis_end_date",
    ]))
    return df[keep_cols].copy()


def build_kpi_summary(df: pd.DataFrame, start_date: pd.Timestamp | None, end_date: pd.Timestamp | None) -> dict[str, object]:
    # Use purchase_signal (max of event purchases + order purchases) for accurate sales count.
    # Raw purchase events alone under-count drastically when event tracking is incomplete.
    total_sales_signal = int(df["purchase_signal"].sum()) if "purchase_signal" in df.columns else (
        int(np.maximum(df.get("purchase", 0), df.get("purchase_orders", 0)).sum())
    )
    total_order_purchases = int(df["purchase_orders"].sum()) if "purchase_orders" in df.columns else 0
    total_event_purchases = int(df["purchase"].sum()) if "purchase" in df.columns else 0
    data_quality_note = (
        "purchase_event_capture_low"
        if total_event_purchases < total_order_purchases * 0.1
        else None
    )
    return {
        "analysis_start_date": str(start_date.date()) if start_date is not None else None,
        "analysis_end_date": str(end_date.date()) if end_date is not None else None,
        "total_views": int(df["view"].sum()) if "view" in df else 0,
        "total_carts": int(df["cart"].sum()) if "cart" in df else 0,
        "total_sales": total_sales_signal,
        "total_order_purchases": total_order_purchases,
        "total_event_purchases": total_event_purchases,
        "data_quality_note": data_quality_note,
        "overall_conversion": round(float(df["purchase_conversion_rate"].replace([np.inf, -np.inf], 0).mean()), 6) if not df.empty else 0.0,
        "products_analyzed": int(len(df)),
        "weak_products_count": int((df["product_class"] == "Weak").sum()) if "product_class" in df else 0,
        "moderate_products_count": int((df["product_class"] == "Moderate").sum()) if "product_class" in df else 0,
        "strong_products_count": int((df["product_class"] == "Strong").sum()) if "product_class" in df else 0,
    }


def build_matrix(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["product_id"] = df["product_id"]
    out["image_url"] = df.get("image_url", "")
    out["product_name"] = df.get("display_name", df.get("name", ""))
    out["category"] = df.get("category", "")
    out["brand"] = df.get("brand", "")
    out["price"] = df.get("price", 0.0)
    out["views"] = df.get("view", 0.0)
    out["carts"] = df.get("cart", 0.0)
    out["purchases"] = df.get("purchase", 0.0)
    out["purchase_conversion_rate"] = df.get("purchase_conversion_rate", 0.0)
    out["diagnostic_status_label"] = df.get("diagnostic_status_label", "")
    out["diagnostic_status_sub_label"] = df.get("diagnostic_status_sub_label", "")
    out["product_class"] = df.get("product_class", "")
    out["confidence_level"] = df.get("confidence_level", "")
    out["diagnostic_reason_short"] = df.get("diagnostic_reason_short", "")
    return out


def run_layer1(
    layer0_dir: Path,
    output_dir: Path,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    setup_logging()
    ensure_dir(output_dir)
    LOGGER.info("Starting Layer 1")

    inputs = load_layer0_outputs(layer0_dir)
    events = normalize_event_types(inputs["events"])
    events = filter_df_by_date_range(events, "event_time", start_date, end_date)
    orders = filter_df_by_date_range(inputs["orders"], "order_date", start_date, end_date)
    if orders.empty:
        order_items = inputs["order_items"].iloc[0:0].copy()
    else:
        order_items = inputs["order_items"][inputs["order_items"]["order_id"].isin(orders["order_id"])].copy()

    diag_core = build_layer1_core_metrics(events)
    unique_metrics = build_unique_user_metrics(events)
    order_metrics = build_order_metrics(orders, order_items)

    diagnostics = build_ui_ready_outputs(inputs["product_master"], diag_core, unique_metrics, order_metrics, start_date, end_date)
    diagnostics = diagnostics.sort_values(["product_class", "priority_rank", "product_id"], ascending=[True, True, True]).reset_index(drop=True)

    safe_write_csv(diagnostics, output_dir / config.LAYER1_DIAGNOSTICS_FILE)
    safe_write_csv(build_matrix(diagnostics), output_dir / config.LAYER1_MATRIX_FILE)
    save_json(build_kpi_summary(diagnostics, start_date, end_date), output_dir / config.LAYER1_KPI_SUMMARY_FILE)
    save_json(
        {
            "analysis_start_date": str(start_date.date()) if start_date is not None else None,
            "analysis_end_date": str(end_date.date()) if end_date is not None else None,
            "products_evaluated": int(len(diagnostics)),
            "class_counts": diagnostics["product_class"].value_counts(dropna=False).to_dict(),
            "confidence_counts": diagnostics["confidence_level"].value_counts(dropna=False).to_dict(),
        },
        output_dir / config.LAYER1_SUMMARY_FILE,
    )
    LOGGER.info("Layer 1 finished successfully")
    return diagnostics
