"""Layer 4: deterministic prescriptive decision engine."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src.utils import ensure_dir, safe_divide, safe_read_csv, safe_write_csv, save_json, setup_logging

LOGGER = logging.getLogger("spark.layer4")


def load_layer_inputs(layer1_dir: Path, layer2_dir: Path, layer3_dir: Path) -> dict[str, pd.DataFrame]:
    diagnostics = safe_read_csv(layer1_dir / config.LAYER1_DIAGNOSTICS_FILE, required=True)
    peers = safe_read_csv(layer2_dir / config.LAYER2_PEERS_FILE, required=True)
    opportunities = safe_read_csv(layer3_dir / config.LAYER3_OPPORTUNITIES_FILE, required=False)
    scatter = safe_read_csv(layer2_dir / config.LAYER2_SCATTER_FILE, required=False)
    for df in [diagnostics, peers, opportunities, scatter]:
        for col in ["product_id", "bundle_partner_product_id", "selected_product_id", "plot_product_id"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
    return {
        "diagnostics": diagnostics,
        "peers": peers,
        "opportunities": opportunities,
        "scatter": scatter,
    }


def merge_layer_outputs(diagnostics: pd.DataFrame, peers: pd.DataFrame, opportunities: pd.DataFrame) -> pd.DataFrame:
    peer_cols = [
        "product_id",
        "peer_product_ids",
        "peer_names",
        "peer_avg_price",
        "peer_avg_rating",
        "peer_avg_review_count",
        "peer_avg_image_count",
        "peer_avg_description_length",
        "peer_avg_conversion",
        "top_similarity_score",
        "top_peer_product_id",
        "top_peer_name",
        "top_peer_image_url",
        "price_gap_flag",
        "rating_gap_flag",
        "review_gap_flag",
        "image_gap_flag",
        "content_gap_flag",
        "new_product_flag",
        "category_context_flag",
        "context_primary_gap",
        "context_secondary_gap",
        "contextual_risk_score",
        "contextual_weakness_reason",
        "similarity_cluster",
    ]
    merged = diagnostics.merge(
        peers[[col for col in peer_cols if col in peers.columns]].drop_duplicates(subset=["product_id"]),
        on="product_id",
        how="left",
    )

    if opportunities.empty:
        merged["has_bundle_opportunity"] = False
        merged["bundle_partner_product_id"] = np.nan
        merged["bundle_partner_name"] = np.nan
        merged["bundle_partner_image_url"] = np.nan
        merged["bundle_strategy"] = np.nan
        merged["bundle_score"] = 0.0
        merged["bundle_reason"] = ""
        return merged

    best_bundle = opportunities.sort_values(
        by=["bundle_score", "lift", "confidence"], ascending=[False, False, False]
    ).drop_duplicates(subset=["product_id"])
    best_bundle = best_bundle.rename(
        columns={
            "bundle_type": "bundle_strategy",
        }
    )
    keep = [
        "product_id",
        "bundle_partner_product_id",
        "bundle_partner_name",
        "bundle_partner_image_url",
        "bundle_strategy",
        "bundle_score",
        "bundle_reason",
        "bundle_business_explanation",
    ]
    merged = merged.merge(best_bundle[keep], on="product_id", how="left")
    merged["has_bundle_opportunity"] = merged["bundle_partner_product_id"].notna() & (merged["bundle_partner_product_id"] != "")
    merged["bundle_score"] = pd.to_numeric(merged["bundle_score"], errors="coerce").fillna(0.0)
    return merged


def classify_root_cause(row: pd.Series) -> tuple[str, str]:
    """
    Multi-signal root-cause detection — يفحص كل المشاكل ولا يتوقف عند أول شرط.
    يُرجع السبب الرئيسي (الأعلى أولوية) + سرد نصي يذكر كل المشاكل المكتشفة.
    """
    # ── جمع كل المشاكل الفعلية بترتيب الأولوية ───────────────────────────
    issues: list[tuple[str, str]] = []   # (code, english_detail)

    if bool(row.get("high_abandonment_flag", False)):
        issues.append(("conversion_friction",
                        "Cart activity exists, but too few cart users convert to buyers."))
    if bool(row.get("price_gap_flag", False)):
        issues.append(("pricing",
                        "The product appears overpriced relative to its closest peer group."))
    if (bool(row.get("rating_gap_flag", False)) or bool(row.get("review_gap_flag", False))) and row.get("confidence_level") != "High":
        issues.append(("trust",
                        "Ratings or review count trail comparable products."))
    if bool(row.get("image_gap_flag", False)) or bool(row.get("content_gap_flag", False)):
        issues.append(("content",
                        "Product content richness or image coverage lags similar products."))
    if bool(row.get("low_visibility_flag", False)):
        issues.append(("visibility",
                        "The product receives limited qualified visibility in the selected period."))

    # حالة: بيانات غير كافية (لا مشاكل محددة + ثقة منخفضة)
    if not issues:
        if row.get("confidence_level") == "Low":
            return ("mixed_context",
                    "Early mixed signals — apply recommendations carefully and revisit after more activity.")
        return ("mixed_context",
                str(row.get("contextual_weakness_reason",
                            row.get("diagnostic_reason", "Mixed performance signals detected."))))

    # السبب الرئيسي = أول مشكلة بالقائمة المرتبة
    primary_code, primary_detail = issues[0]

    # ── بناء السرد الكامل لكل المشاكل المكتشفة ──────────────────────────
    if len(issues) == 1:
        full_detail = primary_detail
    else:
        all_codes = " | ".join(c for c, _ in issues)
        all_details = " Also: ".join(d for _, d in issues[1:])
        full_detail = f"{primary_detail} Additional issues detected ({all_codes}): {all_details}"

    return primary_code, full_detail


def classify_performance_substatus(row: pd.Series) -> str:
    """
    يُظهر كل المشاكل الفعلية مفصولة بـ ' + ' بدل تجاهل ما بعد الأولى.
    مثال: 'High Price vs Peers + Low Trust vs Peers'
    """
    if row.get("confidence_level") == "Low":
        return "إشارات أولية"
    labels = []
    if bool(row.get("high_abandonment_flag", False)):
        labels.append("High Abandonment")
    if bool(row.get("price_gap_flag", False)):
        labels.append("High Price vs Peers")
    if (bool(row.get("rating_gap_flag", False)) or bool(row.get("review_gap_flag", False))) and row.get("confidence_level") != "High":
        labels.append("Low Trust vs Peers")
    if bool(row.get("image_gap_flag", False)) or bool(row.get("content_gap_flag", False)):
        labels.append("Weak Content")
    if bool(row.get("low_visibility_flag", False)):
        labels.append("Low Visibility")
    if not labels:
        return row.get("diagnostic_status_sub_label", "Balanced Signals")
    return " + ".join(labels)


def generate_recommended_actions(row: pd.Series) -> dict[str, str]:
    root_cause = row["root_cause"]
    primary_map = {
        "pricing": (
            "Test a sharper price point or targeted promotion against peer price levels.",
            "اختبر سعرا اكثر تنافسية او عرضا ترويجيا موجها مقارنة بالبدائل القريبة.",
            "pricing",
        ),
        "trust": (
            "Strengthen trust signals with more reviews, ratings, and proof points.",
            "عزز الثقة عبر زيادة المراجعات والتقييمات وعناصر الاثبات.",
            "trust",
        ),
        "content": (
            "Improve product content with stronger visuals, richer copy, and clearer value proposition.",
            "حسن المحتوى عبر صور اقوى ووصف اغنى وتوضيح ادق للقيمة.",
            "content",
        ),
        "conversion_friction": (
            "Investigate checkout or value-friction and reduce drop-off after add-to-cart.",
            "افحص تعثر ما بعد الاضافة للسلة وخفف اسباب الانسحاب قبل الشراء.",
            "conversion",
        ),
        "visibility": (
            "Increase visibility through merchandising, search placement, and demand generation.",
            "ارفع الظهور عبر الترويج والتموضع في البحث وتنشيط الطلب.",
            "visibility",
        ),
        "insufficient_data": (
            "Collect more in-period evidence before applying a strong intervention.",
            "اجمع ادلة اكثر ضمن الفترة المحددة قبل تنفيذ تدخل قوي.",
            "data_collection",
        ),
        "mixed_context": (
            "Apply a focused optimization plan across pricing, content, and conversion levers.",
            "طبق خطة تحسين مركزة تشمل السعر والمحتوى ومحفزات التحويل.",
            "mixed",
        ),
    }
    primary, primary_ar, action_type = primary_map[root_cause]
    secondary = ""
    secondary_ar = ""
    if bool(row.get("has_bundle_opportunity", False)):
        secondary = f"Bundle with {row.get('bundle_partner_name', row.get('bundle_partner_product_id'))} where relevant."
        secondary_ar = f"فعّل باقة او ربطا مع المنتج {row.get('bundle_partner_name', row.get('bundle_partner_product_id'))} عندما يكون ذلك مناسبا."
        action_type = f"{action_type}+bundle"
    elif root_cause == "pricing" and bool(row.get("content_gap_flag", False)):
        secondary = "Improve product images and description to support price perception."
        secondary_ar = "ادعم السعر عبر تحسين الصور والوصف لرفع ادراك القيمة."
    return {
        "recommended_action_primary": primary,
        "recommended_action_primary_ar": primary_ar,
        "recommended_action_secondary": secondary,
        "recommended_action_secondary_ar": secondary_ar,
        "action_type": action_type,
    }


def build_evidence_lines(row: pd.Series) -> tuple[str, str, str]:
    evidence = []
    if pd.notna(row.get("purchase_conversion_rate")) and pd.notna(row.get("category_avg_conversion")):
        evidence.append(
            f"Conversion {row['purchase_conversion_rate']:.2%} vs category avg {row['category_avg_conversion']:.2%}."
        )
    if bool(row.get("price_gap_flag", False)) and pd.notna(row.get("peer_avg_price")):
        evidence.append(f"Price is above peer average ({row['price']:.2f} vs {row['peer_avg_price']:.2f}).")
    if bool(row.get("rating_gap_flag", False)) and pd.notna(row.get("peer_avg_rating")):
        evidence.append(f"Rating trails peers ({row.get('rating', 0):.2f} vs {row['peer_avg_rating']:.2f}).")
    if bool(row.get("image_gap_flag", False)) and pd.notna(row.get("peer_avg_image_count")):
        evidence.append(f"Image count trails peers ({row.get('image_count', 0):.0f} vs {row['peer_avg_image_count']:.0f}).")
    if bool(row.get("high_abandonment_flag", False)) and pd.notna(row.get("cart_abandonment_rate")):
        evidence.append(f"Cart abandonment is high at {row['cart_abandonment_rate']:.2%}.")
    if not evidence:
        evidence.append(str(row.get("diagnostic_reason", "Performance requires closer review.")))
    evidence = (evidence + ["", "", ""])[:3]
    return evidence[0], evidence[1], evidence[2]


def compute_priority_score(df: pd.DataFrame) -> pd.DataFrame:
    scored = df.copy()
    class_score = scored["product_class"].map({"Weak": 1.0, "Moderate": 0.5, "Strong": 0.1}).fillna(0.5)
    underperformance = (
        (1 - scored["relative_conversion_vs_category"].replace([np.inf, -np.inf], np.nan).fillna(1.0))
        .clip(lower=0, upper=2) / 2
    )
    urgency = np.where(
        (scored["product_class"] == "Weak") & (scored["confidence_level"] != "Low"),
        1.0,
        np.where(scored["product_class"] == "Moderate", 0.5, 0.2),
    )
    context = scored["contextual_risk_score"].fillna(0).clip(lower=0, upper=5) / 5
    bundle_signal = np.where(scored["has_bundle_opportunity"], np.minimum(scored["bundle_score"].fillna(0) / 100, 1.0), 0.0)

    scored["priority_score"] = (
        100
        * (
            config.PRODUCT_CLASS_WEIGHT * class_score
            + config.UNDERPERFORMANCE_WEIGHT * underperformance
            + config.URGENCY_WEIGHT * urgency
            + config.CONTEXT_PRIORITY_WEIGHT * context
            + config.BUNDLE_SIGNAL_WEIGHT * bundle_signal
        )
    ).round(2)
    return scored


def assign_priority_tier(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["priority_tier"] = np.select([out["priority_score"] >= 55, out["priority_score"] >= 38], ["High", "Medium"], default="Low")
    out["urgency_level"] = np.select([out["priority_score"] >= 60, out["priority_score"] >= 45], ["High", "Medium"], default="Low")
    return out


def prepare_genai_input(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "product_id",
        "display_name",
        "name",
        "category",
        "subcategory",
        "brand",
        "price",
        "image_url",
        "product_class",
        "diagnostic_status_label",
        "performance_substatus",
        "confidence_level",
        "priority_score",
        "priority_tier",
        "urgency_level",
        "root_cause",
        "root_cause_details",
        "all_issues",
        "issues_count",
        "recommended_action",
        "recommended_action_ar",
        "recommended_action_primary",
        "recommended_action_secondary",
        "recommended_action_primary_ar",
        "recommended_action_secondary_ar",
        "diagnostic_reason",
        "contextual_weakness_reason",
        "evidence_line_1",
        "evidence_line_2",
        "evidence_line_3",
        "unique_viewers",
        "unique_cart_users",
        "unique_buyers",
        "purchase_conversion_rate",
        "category_avg_conversion",
        "relative_conversion_vs_category",
        "revenue",
        "has_bundle_opportunity",
        "bundle_partner_product_id",
        "bundle_partner_name",
        "bundle_strategy",
        "bundle_reason",
    ]
    out = df[[col for col in cols if col in df.columns]].copy()
    out = out.rename(columns={"display_name": "name"})
    return out


def build_overview_payload(recommendations: pd.DataFrame) -> dict[str, object]:
    return {
        "kpis": {
            "total_views": int(recommendations["views"].sum()) if "views" in recommendations else 0,
            "total_carts": int(recommendations["carts"].sum()) if "carts" in recommendations else 0,
            "total_sales": int(recommendations["purchases"].sum()) if "purchases" in recommendations else 0,
            "overall_conversion": round(
                float(safe_divide(recommendations["unique_buyers"].sum(), recommendations["unique_viewers"].sum())),
                6,
            )
            if not recommendations.empty
            else None,
        },
        "priority_tier_counts": recommendations["priority_tier"].value_counts(dropna=False).to_dict(),
        "root_cause_counts": recommendations["root_cause"].value_counts(dropna=False).to_dict(),
    }


def simulate_price_scenario(product_row: pd.Series | dict[str, object], new_price: float) -> dict[str, object]:
    row = product_row if isinstance(product_row, dict) else product_row.to_dict()
    current_price = float(row.get("price", 0) or 0)
    peer_avg_price = float(row.get("peer_avg_price", 0) or 0)
    price_gap_before = safe_divide(current_price - peer_avg_price, peer_avg_price)
    price_gap_after = safe_divide(new_price - peer_avg_price, peer_avg_price)

    if pd.isna(price_gap_after):
        direction = "unknown"
        note = "Peer price context is unavailable."
    elif price_gap_after < price_gap_before:
        direction = "improved"
        note = "The new price narrows the gap versus peers and may reduce pricing friction."
    elif price_gap_after > price_gap_before:
        direction = "worse"
        note = "The new price widens the gap versus peers and may increase pricing friction."
    else:
        direction = "neutral"
        note = "The new price keeps the same relative position versus peers."

    return {
        "product_id": row.get("product_id"),
        "current_price": current_price,
        "new_price": new_price,
        "peer_avg_price": peer_avg_price or None,
        "price_gap_before": price_gap_before if pd.notna(price_gap_before) else None,
        "price_gap_after": price_gap_after if pd.notna(price_gap_after) else None,
        "direction": direction,
        "scenario_note": note,
    }


def run_layer4(layer1_dir: Path, layer2_dir: Path, layer3_dir: Path, output_dir: Path) -> dict[str, pd.DataFrame]:
    setup_logging()
    ensure_dir(output_dir)
    LOGGER.info("Starting Layer 4")

    inputs = load_layer_inputs(layer1_dir, layer2_dir, layer3_dir)
    merged = merge_layer_outputs(inputs["diagnostics"], inputs["peers"], inputs["opportunities"])

    root_causes = merged.apply(classify_root_cause, axis=1)
    merged["root_cause"] = [item[0] for item in root_causes]
    merged["root_cause_details"] = [item[1] for item in root_causes]
    merged["performance_status"] = merged["product_class"].map(lambda x: f"{x} Performance")
    merged["performance_substatus"] = merged.apply(classify_performance_substatus, axis=1)

    # ── all_issues: جمع كل المشاكل الفعلية في عمود واحد نظيف ──────────────
    _flag_label_map = {
        "high_abandonment_flag": "conversion_friction",
        "price_gap_flag":         "pricing",
        "rating_gap_flag":        "trust",
        "review_gap_flag":        "trust",
        "image_gap_flag":         "content",
        "content_gap_flag":       "content",
        "low_visibility_flag":    "visibility",
    }

    def _collect_all_issues(row: pd.Series) -> str:
        seen, labels = set(), []
        for flag, label in _flag_label_map.items():
            if bool(row.get(flag, False)) and label not in seen:
                labels.append(label)
                seen.add(label)
        return "|".join(labels) if labels else "none"

    merged["all_issues"]   = merged.apply(_collect_all_issues, axis=1)
    merged["issues_count"] = merged["all_issues"].apply(
        lambda x: 0 if x == "none" else len(x.split("|"))
    )
    # ─────────────────────────────────────────────────────────────────────────

    actions = merged.apply(generate_recommended_actions, axis=1, result_type="expand")
    merged = pd.concat([merged, actions], axis=1)
    merged["recommended_action"] = merged["recommended_action_primary"].fillna("")
    merged["recommended_action_ar"] = merged["recommended_action_primary_ar"].fillna("")
    merged["recommended_action_secondary"] = merged["recommended_action_secondary"].fillna("")
    merged["recommended_action_secondary_ar"] = merged["recommended_action_secondary_ar"].fillna("")

    evidence = merged.apply(build_evidence_lines, axis=1, result_type="expand")
    evidence.columns = ["evidence_line_1", "evidence_line_2", "evidence_line_3"]
    merged = pd.concat([merged, evidence], axis=1)

    merged = assign_priority_tier(compute_priority_score(merged))
    merged["analysis_start_date"] = merged.get("analysis_start_date", None)
    merged["analysis_end_date"] = merged.get("analysis_end_date", None)

    recommendations = merged.sort_values(
        by=["priority_score", "priority_tier", "confidence_level"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    high_priority = recommendations[recommendations["priority_tier"] == "High"].copy().reset_index(drop=True)
    detail_view = recommendations.copy()
    genai_input = prepare_genai_input(recommendations)

    safe_write_csv(recommendations, output_dir / config.LAYER4_RECOMMENDATIONS_FILE)
    safe_write_csv(high_priority, output_dir / config.LAYER4_HIGH_PRIORITY_FILE)
    safe_write_csv(genai_input, output_dir / config.LAYER4_GENAI_INPUT_FILE)
    safe_write_csv(detail_view, output_dir / config.LAYER4_DETAIL_VIEW_FILE)
    save_json(build_overview_payload(recommendations), output_dir / config.LAYER4_OVERVIEW_FILE)
    save_json(
        {
            "products_scored": int(len(recommendations)),
            "priority_tier_counts": recommendations["priority_tier"].value_counts(dropna=False).to_dict(),
            "root_cause_counts": recommendations["root_cause"].value_counts(dropna=False).to_dict(),
            "bundle_opportunity_products": int(recommendations["has_bundle_opportunity"].fillna(False).sum()),
        },
        output_dir / config.LAYER4_SUMMARY_FILE,
    )
    LOGGER.info("Layer 4 finished successfully")
    return {
        "recommendations": recommendations,
        "high_priority": high_priority,
        "products_for_genai": genai_input,
        "detail_view": detail_view,
    }
