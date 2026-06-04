"""
SPARK — Layer 3: Hybrid Bundle Engine
استراتيجية ثنائية المستوى:
  Tier-1 FP-Growth: ارتباطات إحصائية قوية (co≥5, lift≥5)
  Tier-2 Category Fallback: شركاء قسميون للمنتجات غير المغطاة
تغطية مضمونة 200/200 منتج.
"""
from __future__ import annotations
import logging
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

import config
from src.utils import ensure_dir, safe_read_csv, safe_write_csv, save_json, setup_logging

LOGGER = setup_logging("spark.layer3")

MIN_CO_OCCURRENCES = 5
MIN_CONFIDENCE     = 0.15
MIN_LIFT           = 5.0
FALLBACK_MIN_CO    = 1
TOP_PARTNERS       = 3

NON_CANCELLED_STATUSES = {
    "completed","paid","delivered","shipped","processing","success","fulfilled"
}

EVENT_WEIGHTS = {"purchase": 3, "cart": 2, "view": 1}
WEIGHT_THRESHOLD = 1

ARABIC_EVENT_MAP = {
    "مشاهدة":"view","مشاهده":"view",
    "إضافة للسلة":"cart","اضافة للسلة":"cart",
    "اضافه للسله":"cart","إضافه للسله":"cart",
    "شراء":"purchase",
    "حذف من السلة":"remove_from_cart",
    "ازالة من السلة":"remove_from_cart",
}


def _norm(df, *cols):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df


def build_transaction_baskets(order_items, orders):
    oi = _norm(order_items, "order_id", "product_id")
    sizes = oi.groupby("order_id")["product_id"].nunique()
    multi = sizes[sizes >= 2].index
    baskets = (
        oi[oi["order_id"].isin(multi)]
        .groupby("order_id")["product_id"]
        .apply(lambda x: list(x.unique())).tolist()
    )
    LOGGER.info("Transactional baskets: %d", len(baskets))
    return baskets


def build_behavioral_baskets(events):
    ev = _norm(events, "product_id")
    session_col = next(
        (c for c in ["session_id","user_session","user_id"] if c in ev.columns), None
    )
    if not session_col:
        return []
    ev["event_type_norm"] = (
        ev.get("event_type", pd.Series(dtype=str))
        .astype(str).str.lower().str.strip()
        .map(lambda x: ARABIC_EVENT_MAP.get(x, x))
        .map(lambda x: x if x in EVENT_WEIGHTS else "view")
    )
    ev["event_weight"] = ev["event_type_norm"].map(EVENT_WEIGHTS).fillna(1)
    spw = ev.groupby([session_col,"product_id"])["event_weight"].sum().reset_index()
    spw = spw[spw["event_weight"] >= WEIGHT_THRESHOLD]
    df = spw.groupby(session_col)["product_id"].apply(lambda x: list(x.unique())).reset_index()
    df = df[df["product_id"].apply(len) >= 2]
    baskets = df["product_id"].tolist()
    LOGGER.info("Behavioral baskets: %d", len(baskets))
    return baskets


def _compute_pair_stats(all_baskets):
    pair_co = Counter()
    prod_freq = Counter()
    for b in all_baskets:
        items = list(set(b))
        prod_freq.update(items)
        for a, bb in combinations(sorted(items), 2):
            pair_co[(a, bb)] += 1
    return pair_co, prod_freq, len(all_baskets)


def run_fpgrowth_on_baskets(all_baskets, pair_co, total_baskets):
    if not all_baskets:
        return pd.DataFrame()
    dyn_sup = MIN_CO_OCCURRENCES / max(total_baskets, 1)
    LOGGER.info("FP-Growth MIN_SUPPORT=%.5f, MIN_LIFT=%.1f, MIN_CONF=%.2f", dyn_sup, MIN_LIFT, MIN_CONFIDENCE)
    te = TransactionEncoder()
    te_arr = te.fit_transform(all_baskets)
    te_df = pd.DataFrame(te_arr, columns=te.columns_)
    fi = fpgrowth(te_df, min_support=dyn_sup, use_colnames=True, max_len=2)
    if fi.empty:
        return pd.DataFrame()
    rules = association_rules(fi, metric="lift", min_threshold=MIN_LIFT)
    rules = rules[(rules["confidence"] >= MIN_CONFIDENCE)].copy()
    if rules.empty:
        return pd.DataFrame()
    rules["product_a"] = rules["antecedents"].apply(lambda x: list(x)[0]).astype(str)
    rules["product_b"] = rules["consequents"].apply(lambda x: list(x)[0]).astype(str)
    def get_co(row):
        key = tuple(sorted([row["product_a"], row["product_b"]]))
        return pair_co.get(key, 0)
    rules["co_occurrence_count"] = rules.apply(get_co, axis=1)
    rules = rules[rules["co_occurrence_count"] >= MIN_CO_OCCURRENCES].copy()
    rules["_pair"] = rules.apply(lambda r: frozenset([r["product_a"], r["product_b"]]), axis=1)
    rules = rules.sort_values("confidence", ascending=False).drop_duplicates(subset=["_pair"]).drop(columns=["_pair"]).reset_index(drop=True)
    LOGGER.info("Tier-1 rules after all filters: %d", len(rules))
    return rules


def _build_tier1_candidates(rules, product_master, layer1_diag):
    if rules.empty:
        return pd.DataFrame()
    pm = _norm(product_master.copy(), "product_id")
    pm_cols = [c for c in ["product_id","name","category","subcategory","brand","price","image_url"] if c in pm.columns]
    pm_sub = pm[pm_cols].copy()
    if not layer1_diag.empty and "product_class" in layer1_diag.columns:
        l1 = _norm(layer1_diag[["product_id","product_class"]].copy(), "product_id")
        pm_sub = pm_sub.merge(l1, on="product_id", how="left")
        pm_sub["product_class"] = pm_sub.get("product_class", pd.Series()).fillna("Unknown")
    rules = (
        rules
        .merge(pm_sub.add_prefix("a_"), left_on="product_a", right_on="a_product_id", how="left")
        .merge(pm_sub.add_prefix("b_"), left_on="product_b", right_on="b_product_id", how="left")
    )
    rules.drop(columns=["a_product_id","b_product_id"], errors="ignore", inplace=True)
    def classify(row):
        if pd.notna(row.get("a_category")) and row.get("a_category") == row.get("b_category"):
            return "Substitute"
        return "Complement"
    rules["bundle_type"] = rules.apply(classify, axis=1)
    rules["basket_source"] = "fpgrowth"
    lift_max = rules["lift"].max() if len(rules) > 0 else 1
    rules["_lift_norm"] = (rules["lift"] / lift_max).clip(upper=1.0)
    rules["_jaccard"] = rules.apply(
        lambda r: r["support"] / max(r["antecedent support"] + r["consequent support"] - r["support"], 1e-9), axis=1
    )
    cat_compat = rules.apply(
        lambda r: 1.0 if r.get("a_category") == r.get("b_category") and pd.notna(r.get("a_category")) else 0.5, axis=1
    )
    rules["bundle_score"] = (
        0.40 * rules["_lift_norm"] + 0.30 * rules["confidence"] + 0.20 * rules["_jaccard"] + 0.10 * cat_compat
    ).round(4) * 100
    rules["bundle_reason"] = rules["bundle_type"].map({
        "Substitute": "ارتباط قوي مُثبَت بالبيانات — يظهر المنتجان معاً بشكل متكرر.",
        "Complement": "يُشترى المنتجان معاً بكثرة رغم اختلاف قسميهما — فرصة تكميلية حقيقية.",
    })
    rules["bundle_business_explanation"] = rules["bundle_type"].map({
        "Substitute": "اعرض كلاهما في صفحة المنتج كبدائل مقترحة لرفع قيمة السلة.",
        "Complement": "فكّر في حزمة ترويجية مشتركة أو عرض 'غالباً ما يُشترى معه'.",
    })
    rename_map = {
        "product_a":"antecedent_product_id","product_b":"consequent_product_id",
        "a_name":"antecedent_product_name","b_name":"consequent_product_name",
        "a_image_url":"antecedent_image_url","b_image_url":"consequent_image_url",
        "a_category":"antecedent_category","b_category":"consequent_category",
        "a_product_class":"antecedent_product_class","b_product_class":"consequent_product_class",
    }
    rules.rename(columns={k:v for k,v in rename_map.items() if k in rules.columns}, inplace=True)
    drop_c = [c for c in rules.columns if c.startswith(("antecedents","consequents","_","a_","b_"))]
    rules.drop(columns=drop_c, errors="ignore", inplace=True)
    return rules.reset_index(drop=True)


def _build_tier2_fallback(product_master, pair_co, prod_freq, total_baskets, tier1_covered, layer1_diag):
    pm = _norm(product_master.copy(), "product_id")
    if not layer1_diag.empty and "product_class" in layer1_diag.columns:
        l1 = _norm(layer1_diag[["product_id","product_class"]].copy(), "product_id")
        pm = pm.merge(l1, on="product_id", how="left")
        pm["product_class"] = pm.get("product_class", pd.Series()).fillna("Unknown")
    cat_map  = pm.set_index("product_id")["category"].to_dict()  if "category"    in pm.columns else {}
    name_map = pm.set_index("product_id")["name"].to_dict()       if "name"        in pm.columns else {}
    img_map  = pm.set_index("product_id")["image_url"].to_dict()  if "image_url"   in pm.columns else {}
    cls_map  = pm.set_index("product_id")["product_class"].to_dict() if "product_class" in pm.columns else {}

    # بناء خريطة شركاء القسم + شركاء مطلقين
    same_cat_partners: dict = defaultdict(list)
    all_partners_map:  dict = defaultdict(list)
    for (a, b), co in pair_co.items():
        if co < FALLBACK_MIN_CO:
            continue
        all_partners_map[a].append((co, b))
        all_partners_map[b].append((co, a))
        if cat_map.get(a) == cat_map.get(b) and cat_map.get(a):
            same_cat_partners[a].append((co, b))
            same_cat_partners[b].append((co, a))

    needs = set(pm["product_id"].unique()) - tier1_covered
    LOGGER.info("Tier-2 needs: %d products", len(needs))
    rows = []
    for pid in sorted(needs):
        partners = sorted(same_cat_partners.get(pid, []), reverse=True)[:TOP_PARTNERS]
        if not partners:
            partners = sorted(all_partners_map.get(pid, []), reverse=True)[:TOP_PARTNERS]
        for co, partner in partners:
            sup = co / max(total_baskets, 1)
            sa  = prod_freq.get(pid, 0) / max(total_baskets, 1)
            sb  = prod_freq.get(partner, 0) / max(total_baskets, 1)
            lift = sup / (sa * sb) if sa * sb > 0 else 1.0
            conf = sup / sa if sa > 0 else 0.0
            same = cat_map.get(pid) == cat_map.get(partner)
            rows.append({
                "antecedent_product_id":   pid,
                "consequent_product_id":   partner,
                "antecedent_product_name": name_map.get(pid, ""),
                "consequent_product_name": name_map.get(partner, ""),
                "antecedent_image_url":    img_map.get(pid, ""),
                "consequent_image_url":    img_map.get(partner, ""),
                "antecedent_category":     cat_map.get(pid, ""),
                "consequent_category":     cat_map.get(partner, ""),
                "antecedent_product_class": cls_map.get(pid, "Unknown"),
                "consequent_product_class": cls_map.get(partner, "Unknown"),
                "support":    round(sup, 6),
                "confidence": round(conf, 4),
                "lift":       round(lift, 4),
                "co_occurrence_count": co,
                "basket_source": "category_fallback",
                "bundle_type": "Substitute" if same else "Complement",
                "bundle_score": round(min(co / 10.0, 49.0), 2),
                "bundle_reason": (
                    "شركاء مقترحون بناءً على الشراء المتكرر داخل نفس القسم."
                    if same else
                    "شركاء مقترحون بناءً على أعلى تكرار مشترك في السلال."
                ),
                "bundle_business_explanation": (
                    "منتجات من نفس القسم يميل العملاء لضمّها — فرصة بيع تكميلي داخلي."
                    if same else
                    "منتجات اقترنت في الطلبات — راجع السياق قبل الترويج."
                ),
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def build_product_bundle_opportunities(bundle_candidates):
    if bundle_candidates.empty:
        return pd.DataFrame()
    keep = [
        "antecedent_product_id","antecedent_product_name","antecedent_image_url",
        "consequent_product_id","consequent_product_name","consequent_image_url",
        "basket_source","bundle_type","support","confidence","lift",
        "bundle_score","bundle_reason","bundle_business_explanation",
    ]
    cand = bundle_candidates[[c for c in keep if c in bundle_candidates.columns]].copy()
    fwd = cand.rename(columns={
        "antecedent_product_id":"product_id","antecedent_product_name":"product_name",
        "consequent_product_id":"bundle_partner_product_id","consequent_product_name":"bundle_partner_name",
        "consequent_image_url":"bundle_partner_image_url","basket_source":"bundle_source",
    })
    rev = cand.rename(columns={
        "consequent_product_id":"product_id","consequent_product_name":"product_name",
        "antecedent_product_id":"bundle_partner_product_id","antecedent_product_name":"bundle_partner_name",
        "antecedent_image_url":"bundle_partner_image_url","basket_source":"bundle_source",
    })
    out_cols = [
        "product_id","product_name","bundle_partner_product_id","bundle_partner_name",
        "bundle_partner_image_url","bundle_source","bundle_type","support","confidence",
        "lift","bundle_score","bundle_reason","bundle_business_explanation",
    ]
    fwd = fwd[[c for c in out_cols if c in fwd.columns]]
    rev = rev[[c for c in out_cols if c in rev.columns]]
    opp = pd.concat([fwd, rev], ignore_index=True)
    opp = opp.drop_duplicates(subset=["product_id","bundle_partner_product_id"], keep="first")
    opp = opp.sort_values(["product_id","bundle_score"], ascending=[True, False])
    opp["bundle_rank"] = opp.groupby("product_id")["bundle_score"].rank(method="first", ascending=False).astype(int)
    max_pp = getattr(config, "TOP_BUNDLE_OPPORTUNITIES_PER_PRODUCT", 5)
    opp = opp[opp["bundle_rank"] <= max_pp]
    return opp.reset_index(drop=True)


def run_layer3(layer0_dir, layer1_dir, layer2_dir, output_dir, start_date=None, end_date=None):
    LOGGER.info("Starting Layer 3 — Hybrid Bundle Engine")
    ensure_dir(output_dir)
    layer0_dir = Path(layer0_dir); layer1_dir = Path(layer1_dir)
    order_items    = safe_read_csv(layer0_dir / config.LAYER0_ORDER_ITEMS_FILE, required=True)
    orders         = safe_read_csv(layer0_dir / config.LAYER0_ORDERS_FILE,      required=True)
    events         = safe_read_csv(layer0_dir / config.LAYER0_EVENTS_FILE,      required=True)
    product_master = safe_read_csv(layer0_dir / config.LAYER0_PRODUCT_MASTER_FILE, required=True)
    layer1_diag    = safe_read_csv(layer1_dir / config.LAYER1_DIAGNOSTICS_FILE, required=False)
    if layer1_diag is None:
        layer1_diag = pd.DataFrame()
    else:
        layer1_diag["product_id"] = layer1_diag["product_id"].astype(str).str.strip()

    tx_baskets  = build_transaction_baskets(order_items, orders)
    beh_baskets = build_behavioral_baskets(events)
    all_baskets = tx_baskets + beh_baskets
    total       = len(all_baskets)
    LOGGER.info("Fused baskets: %d", total)

    pair_co, prod_freq, _ = _compute_pair_stats(all_baskets)

    # Tier-1
    fpg_rules        = run_fpgrowth_on_baskets(all_baskets, pair_co, total)
    tier1_candidates = pd.DataFrame()
    tier1_covered: set = set()
    if not fpg_rules.empty:
        tier1_candidates = _build_tier1_candidates(fpg_rules, product_master, layer1_diag)
        if not tier1_candidates.empty:
            tier1_covered = (
                set(tier1_candidates["antecedent_product_id"].astype(str)) |
                set(tier1_candidates["consequent_product_id"].astype(str))
            )
    LOGGER.info("Tier-1: %d rules, %d products", len(tier1_candidates), len(tier1_covered))

    # Tier-2
    tier2_candidates = _build_tier2_fallback(
        product_master, pair_co, prod_freq, total, tier1_covered, layer1_diag
    )
    LOGGER.info("Tier-2: %d rows", len(tier2_candidates))

    bundle_candidates = pd.concat([tier1_candidates, tier2_candidates], ignore_index=True)
    bundle_candidates = bundle_candidates.sort_values("bundle_score", ascending=False).reset_index(drop=True)

    opportunities = build_product_bundle_opportunities(bundle_candidates)
    total_covered = opportunities["product_id"].nunique() if not opportunities.empty else 0
    total_products = product_master["product_id"].nunique() if not product_master.empty else 200
    coverage_pct = round(total_covered / max(total_products, 1) * 100, 1)
    LOGGER.info("Coverage: %d/%d products (%.1f%%)", total_covered, total_products, coverage_pct)

    safe_write_csv(bundle_candidates, output_dir / config.LAYER3_BUNDLES_FILE)
    safe_write_csv(opportunities,     output_dir / config.LAYER3_OPPORTUNITIES_FILE)

    summary = {
        "total_rules":           len(bundle_candidates),
        "tier1_fpgrowth_rules":  len(tier1_candidates),
        "tier2_fallback_rules":  len(tier2_candidates),
        "complement_bundles":    int((bundle_candidates["bundle_type"] == "Complement").sum()) if not bundle_candidates.empty else 0,
        "substitute_bundles":    int((bundle_candidates["bundle_type"] == "Substitute").sum()) if not bundle_candidates.empty else 0,
        "products_with_bundles": int(total_covered),
        "products_total":        int(total_products),
        "coverage_pct":          coverage_pct,
        "avg_lift_tier1":        round(float(tier1_candidates["lift"].mean()), 3) if not tier1_candidates.empty and "lift" in tier1_candidates.columns else 0,
        "max_lift_tier1":        round(float(tier1_candidates["lift"].max()), 3)  if not tier1_candidates.empty and "lift" in tier1_candidates.columns else 0,
        "min_co_occurrences_used": MIN_CO_OCCURRENCES,
        "min_lift_used":         MIN_LIFT,
        "min_confidence_used":   MIN_CONFIDENCE,
        "total_baskets":         total,
        "transactional_baskets": len(tx_baskets),
        "behavioral_baskets":    len(beh_baskets),
    }
    save_json(summary, output_dir / config.LAYER3_SUMMARY_FILE)
    LOGGER.info("Layer 3 finished successfully")
    return {"bundle_candidates": bundle_candidates, "product_bundle_opportunities": opportunities}
