"""Layer 2: Advanced Similarity & Clustering with Contextual Awareness."""

from __future__ import annotations

import logging
import re
import warnings
from typing import List
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.cluster import AgglomerativeClustering

import config
from src.utils import ensure_dir, safe_read_csv, safe_write_csv, save_json, setup_logging

warnings.filterwarnings("ignore")
LOGGER = logging.getLogger("spark.layer2")

# ============================================================
# CONFIG & WEIGHTS (From layer2_FINAL.py)
# ============================================================
PRICE_GAP_TH = 0.15
BRAND_GAP_TH = 0.20
RATING_GAP_TH = 0.35
DESC_GAP_RATIO = 0.70
IMG_GAP_RATIO = 0.70
REVIEW_GAP_RATIO = 0.50
NEWNESS_RATIO = 0.50
TOP_K_PEERS = 5

WEIGHT_NUM_CORE = 1.20
WEIGHT_NUM_SUPPORT = 0.70
WEIGHT_TEXT_STRUCT = 0.45
WEIGHT_CAT = 2.50

MIN_K = 2
MAX_K = 8
MIN_CLUSTER_SIZE = 8

# Keywords from user's layer2_FINAL.py
TECH_TERMS = ["بطارية", "شاشة", "رام", "ذاكرة", "معالج", "كاميرا", "دقة", "ميجابكسل", "بلوتوث", "لاسلكي", "واي فاي", "wifi", "usb", "type c", "hd", "fhd", "amoled", "oled", "led", "سريع", "شحن", "نانو", "جيجابايت", "تيرابايت", "هرتز", "بصمة", "مقاوم", "ماء", "صوت", "ستيريو", "اتصال", "حساس", "سماعة"]
FASHION_TERMS = ["قماش", "قطن", "حرير", "بوليستر", "جينز", "كاجوال", "رسمي", "مقاس", "سليم فيت", "أوفر سايز", "جاكيت", "تيشيرت", "بنطلون", "فستان", "أكمام", "ياقة", "حذاء", "نعل", "جلد", "تنورة", "أنيق", "موضة", "ستايل", "تصميم", "خياطة", "مريح", "خفيف", "شتوي", "صيفي"]
SPORTS_TERMS = ["لياقة", "تمارين", "عضلات", "كارديو", "قوة", "تحمل", "حرق الدهون", "وزن", "جيم", "تدريب", "مقاومة", "دامبل", "باربل", "يوغا", "سترتش", "جري", "سرعة", "نشاط", "صحة", "بروتين", "سعرات", "نط الحبل", "دراجة", "توازن", "مرونة", "أداء", "رياضة"]
HOME_TERMS = ["أثاث", "ديكور", "منزل", "غرفة", "معيشة", "نوم", "سجاد", "ستائر", "منظم", "تخزين", "رفوف", "خزانة", "وسادة", "مخدة", "بطانية", "إضاءة", "مصباح", "لمبة", "راحة", "تنظيم", "ترتيب", "مساحة", "تصميم داخلي", "متين", "عملي", "عصري", "بسيط", "فاخر"]
KITCHEN_TERMS = ["طبخ", "طهي", "مقلاة", "قدر", "فرن", "مايكروويف", "خلاط", "محضر طعام", "سكين", "لوح تقطيع", "أدوات مطبخ", "أكواب", "صحون", "أوعية", "تخزين طعام", "عازل حراري", "ستانلس ستيل", "مقاوم للصدأ", "سهل التنظيف", "سعة", "لتر", "حرارة", "استخدام يومي", "آمن غذائي", "يد مقاومة للحرارة", "غطاء"]
BOOKS_EDUCATION_TERMS = ["كتاب", "مؤلف", "طبعة", "رواية", "قصة", "تعليمي", "ملخص", "فصل", "محتوى", "منهج", "دراسة", "تعلم", "مهارات", "تدريب", "اختبار", "شرح", "تمارين", "لغة", "إنجليزي", "عربي", "رياضيات", "علوم", "تاريخ", "تنمية بشرية", "ثقافة", "معرفة", "قراءة"]
BEAUTY_CARE_TERMS = ["بشرة", "ترطيب", "تغذية", "كريم", "سيروم", "تونر", "تنظيف", "تقشير", "زيوت طبيعية", "فيتامين", "كولاجين", "مكياج", "أحمر شفاه", "أساس", "فاونديشن", "مسكارا", "مضاد تجاعيد", "تفتيح", "حماية", "واقي شمس", "شعر", "شامبو", "بلسم", "لمعان", "تساقط", "عناية يومية", "نعومة", "جمال", "رائحة"]

CATEGORY_KEYWORDS = {
    "الالكترونيات": TECH_TERMS,
    "الرياضة واللياقة": SPORTS_TERMS,
    "الازياء والملابس": FASHION_TERMS,
    "المنزل والمطبخ": HOME_TERMS,
    "الكتب والتعليم": BOOKS_EDUCATION_TERMS,
    "الجمال والعناية": BEAUTY_CARE_TERMS
}

# ============================================================
# HELPERS
# ============================================================
def normalize_arabic_text(text):
    if pd.isna(text): return ""
    text = str(text).strip()
    text = re.sub(r"ـ+", "", text)
    text = re.sub(r"[أإآٱ]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def evaluate_description(row):
    # Use description_content if available, else description
    description = str(row.get("description_content", row.get("description", ""))).lower()
    category = str(row.get("category", "")).strip()
    keywords = CATEGORY_KEYWORDS.get(category, [])
    if not keywords: return 0.0
    matches = sum(1 for word in keywords if word in description)
    return matches / len(keywords)

def make_arabic_reason_text(reason_codes: List[str]) -> str:
    mapping = {
        "price_high": "السعر أعلى من المستوى المعتاد داخل المجموعة المنافسة",
        "weak_brand": "قوة العلامة التجارية أقل من منافسيه داخل المجموعة",
        "category_mismatch": "المنتج لا ينسجم جيدًا مع الفئة الغالبة داخل المجموعة",
        "low_rating": "تقييم المنتج أقل من متوسط المنتجات المشابهة",
        "low_visual_support": "الدعم البصري أضعف من المنافسين من حيث عدد الصور",
        "short_description": "الوصف أقصر من المعتاد ولا يقدم معلومات كافية",
        "low_review_trust": "عدد المراجعات أقل من المنتجات المشابهة مما يضعف الثقة",
        "too_new_product": "المنتج أحدث من معظم منافسيه وقد يكون ما زال في مرحلة مبكرة",
        "no_clear_issue": "لا توجد مشكلة واضحة مقارنةً بالمجموعة"
    }
    return " | ".join(mapping.get(r, r) for r in reason_codes)

def safe_mode(series: pd.Series, default="Unknown"):
    s = series.dropna().astype(str)
    if len(s) == 0: return default
    return s.mode().iloc[0]

# ============================================================
# CORE LOGIC
# ============================================================
def compute_brand_strength(df: pd.DataFrame) -> pd.DataFrame:
    """Advanced brand strength logic from layer2_FINAL.py"""
    brand_stats = df.groupby("brand").agg(
        brand_total_sales=("purchase_signal", "sum"),
        brand_avg_efficiency=("purchase_conversion_rate", "mean"),
        brand_avg_rating=("rating", "mean"),
        brand_product_count=("product_id", "count")
    ).reset_index()
    
    # Normalize components
    for col in ["brand_total_sales", "brand_avg_efficiency", "brand_avg_rating"]:
        s = brand_stats[col]
        brand_stats[f"norm_{col}"] = (s - s.min()) / (s.max() - s.min() + 1e-6)
    
    brand_stats["brand_strength"] = (
        brand_stats["norm_brand_total_sales"] * 0.45 +
        brand_stats["norm_brand_avg_efficiency"] * 0.35 +
        brand_stats["norm_brand_avg_rating"] * 0.20
    )
    
    return df.merge(brand_stats[["brand", "brand_strength"]], on="brand", how="left")

def build_feature_matrix(df: pd.DataFrame):
    # Numerical Core
    num_core = ["price", "purchase_conversion_rate", "brand_strength"]
    scaler_core = StandardScaler()
    X_num_core = scaler_core.fit_transform(df[num_core]) * WEIGHT_NUM_CORE
    
    # Numerical Support
    num_support = ["rating", "review_count", "number_of_images", "product_age_days"]
    scaler_support = StandardScaler()
    X_num_support = scaler_support.fit_transform(df[num_support]) * WEIGHT_NUM_SUPPORT
    
    # Textual Structural
    text_struct = ["description_length_words", "desc_quality_score"]
    scaler_text = StandardScaler()
    X_text_struct = scaler_text.fit_transform(df[text_struct]) * WEIGHT_TEXT_STRUCT
    
    # Categorical
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    X_cat = ohe.fit_transform(df[["category"]]) * WEIGHT_CAT
    
    return hstack([csr_matrix(X_num_core), csr_matrix(X_num_support), csr_matrix(X_text_struct), X_cat]).tocsr()

def run_layer2(layer0_dir: Path, layer1_dir: Path, output_dir: Path) -> dict[str, pd.DataFrame]:
    setup_logging()
    ensure_dir(output_dir)
    LOGGER.info("Starting Layer 2 (Improved Version)")
    
    # Load inputs
    master = safe_read_csv(layer0_dir / config.LAYER0_PRODUCT_MASTER_FILE, required=True)
    diag = safe_read_csv(layer1_dir / config.LAYER1_DIAGNOSTICS_FILE, required=True)
    
    # Prepare Data
    master["product_id"] = master["product_id"].astype(str).str.strip()
    diag["product_id"] = diag["product_id"].astype(str).str.strip()
    
    df = master.merge(diag, on="product_id", how="inner", suffixes=("", "_diag"))
    
    # Text processing - Handle missing description column gracefully
    if "description" not in df.columns:
        # Fallback to display_name or empty if both missing
        df["description_content"] = df.get("display_name", df.get("name", ""))
    else:
        df["description_content"] = df["description"]
        
    if "description_length_words" not in df.columns:
        df["description_length_words"] = df["description_content"].fillna("").str.split().str.len()
        
    df["desc_quality_score"] = df.apply(evaluate_description, axis=1)
    
    # Ensure numeric columns
    for col in ["price", "rating", "review_count", "number_of_images", "product_age_days", "purchase_signal", "purchase_conversion_rate"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    
    # Brand Strength
    df = compute_brand_strength(df)
    
    # Feature Matrix & Similarity
    X = build_feature_matrix(df)
    cosine_distance = pairwise_distances(X, metric="cosine")
    cosine_similarity = 1 - cosine_distance
    
    # Clustering (Auto-K)
    best_k = MIN_K
    best_score = -1
    best_labels = None
    
    n_samples = len(df)
    for k in range(MIN_K, min(MAX_K, n_samples - 1) + 1):
        clusterer = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
        labels = clusterer.fit_predict(cosine_distance)
        if len(np.unique(labels)) > 1:
            score = silhouette_score(cosine_distance, labels, metric="precomputed")
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels
    
    df["similarity_cluster"] = best_labels if best_labels is not None else 0
    
    # Cluster Profiles
    profiles = df.groupby("similarity_cluster").agg(
        cluster_size=("product_id", "count"),
        cluster_avg_price=("price", "mean"),
        cluster_median_price=("price", "median"),
        cluster_avg_rating=("rating", "mean"),
        cluster_avg_images=("number_of_images", "mean"),
        cluster_median_images=("number_of_images", "median"),
        cluster_avg_desc_words=("description_length_words", "mean"),
        cluster_median_desc_words=("description_length_words", "median"),
        cluster_avg_review_count=("review_count", "mean"),
        cluster_median_review_count=("review_count", "median"),
        cluster_avg_product_age=("product_age_days", "mean"),
        cluster_median_product_age=("product_age_days", "median"),
        cluster_avg_brand_strength=("brand_strength", "mean"),
        cluster_dominant_category=("category", lambda s: safe_mode(s))
    ).reset_index()
    
    df = df.merge(profiles, on="similarity_cluster", how="left")
    
    # Extract Top Peers
    peer_rows = []
    peer_summary_data = []
    
    product_ids = df["product_id"].tolist()
    names = df["display_name"].tolist()
    clusters = df["similarity_cluster"].tolist()
    
    for i in range(len(df)):
        # Peers must be in same cluster
        same_cluster = [j for j in range(len(df)) if clusters[j] == clusters[i] and j != i]
        if not same_cluster: continue
        
        sims = sorted([(j, cosine_similarity[i, j]) for j in same_cluster], key=lambda x: x[1], reverse=True)[:TOP_K_PEERS]
        
        peer_ids = []
        peer_names = []
        peer_prices = []
        peer_convs = []
        peer_ratings = []
        
        for rank, (j, score) in enumerate(sims, start=1):
            peer_rows.append({
                "product_id": product_ids[i],
                "peer_rank": rank,
                "peer_product_id": product_ids[j],
                "peer_name": names[j],
                "peer_similarity_score": float(score)
            })
            peer_ids.append(product_ids[j])
            peer_names.append(names[j])
            peer_prices.append(df.iloc[j]["price"])
            peer_convs.append(df.iloc[j]["purchase_conversion_rate"])
            peer_ratings.append(df.iloc[j]["rating"])
            
        # Top peer for UI
        top_j, top_score = sims[0]
        peer_summary_data.append({
            "product_id": product_ids[i],
            "peer_product_ids": "|".join(peer_ids),
            "peer_names": "|".join(peer_names),
            "peer_avg_price": np.mean(peer_prices),
            "peer_avg_rating": np.mean(peer_ratings),
            "peer_avg_conversion": np.mean(peer_convs),
            "top_similarity_score": float(top_score),
            "top_peer_product_id": product_ids[top_j],
            "top_peer_name": names[top_j]
        })
    
    peer_summary = pd.DataFrame(peer_summary_data)
    
    # Weakness Analysis (Flags)
    df["price_high_flag"] = (df["price"] > (df["cluster_median_price"] * (1 + PRICE_GAP_TH))).astype(int)
    df["weak_brand_flag"] = (df["brand_strength"] < (df["cluster_avg_brand_strength"] * (1 - BRAND_GAP_TH))).astype(int)
    df["category_mismatch_flag"] = (df["category"] != df["cluster_dominant_category"]).astype(int)
    df["low_rating_flag"] = (df["rating"] < (df["cluster_avg_rating"] - RATING_GAP_TH)).astype(int)
    df["low_visual_support_flag"] = (df["number_of_images"] < (df["cluster_median_images"] * IMG_GAP_RATIO)).astype(int)
    df["short_description_flag"] = (df["description_length_words"] < (df["cluster_median_desc_words"] * DESC_GAP_RATIO)).astype(int)
    df["low_review_trust_flag"] = (df["review_count"] < (df["cluster_median_review_count"].replace(0,1) * REVIEW_GAP_RATIO)).astype(int)
    df["too_new_product_flag"] = (df["product_age_days"] < (df["cluster_median_product_age"].replace(0,1) * NEWNESS_RATIO)).astype(int)
    
    # Generate Arabic Reason Text
    def collect_reasons(row):
        codes = []
        if row.price_high_flag: codes.append("price_high")
        if row.weak_brand_flag: codes.append("weak_brand")
        if row.category_mismatch_flag: codes.append("category_mismatch")
        if row.low_rating_flag: codes.append("low_rating")
        if row.low_visual_support_flag: codes.append("low_visual_support")
        if row.short_description_flag: codes.append("short_description")
        if row.low_review_trust_flag: codes.append("low_review_trust")
        if row.too_new_product_flag: codes.append("too_new_product")
        return codes if codes else ["no_clear_issue"]
        
    df["weakness_reason_codes"] = df.apply(collect_reasons, axis=1)
    df["weakness_reason_text_ar"] = df["weakness_reason_codes"].apply(make_arabic_reason_text)
    
    # Priority Score for Layer 4
    class_penalty = {"Weak": 2.0, "Moderate": 1.0, "Strong": 0.3}
    df["context_priority_score"] = (
        df["product_class"].map(class_penalty).fillna(1.0) * 2.0 +
        df["price_high_flag"] * 1.2 +
        df["weak_brand_flag"] * 1.0 +
        df["category_mismatch_flag"] * 1.3 +
        df["low_rating_flag"] * 1.5 +
        df["low_visual_support_flag"] * 1.1 +
        df["short_description_flag"] * 1.1 +
        df["low_review_trust_flag"] * 0.8 -
        df["too_new_product_flag"] * 0.5
    )
    
    # Final Output Prep
    out = df.merge(peer_summary, on="product_id", how="left")
    
    # UI Compat Fields
    out["price_gap_flag"] = out["price_high_flag"].astype(bool)
    out["rating_gap_flag"] = out["low_rating_flag"].astype(bool)
    out["review_gap_flag"] = out["low_review_trust_flag"].astype(bool)
    out["image_gap_flag"] = out["low_visual_support_flag"].astype(bool)
    out["content_gap_flag"] = out["short_description_flag"].astype(bool)
    out["new_product_flag"] = out["too_new_product_flag"].astype(bool)
    out["category_context_flag"] = (~out["category_mismatch_flag"].astype(bool))
    out["contextual_risk_score"] = out["context_priority_score"].round(2)
    out["contextual_weakness_reason"] = out["weakness_reason_text_ar"]
    
    # Map primary/secondary gaps for UI
    def map_gaps(row):
        gaps = []
        if row.price_high_flag: gaps.append("pricing")
        if row.low_rating_flag or row.low_review_trust_flag: gaps.append("trust")
        if row.low_visual_support_flag or row.short_description_flag: gaps.append("content")
        if row.too_new_product_flag: gaps.append("newness")
        return (gaps[0] if gaps else "none", gaps[1] if len(gaps)>1 else "none")
        
    gaps_mapped = out.apply(map_gaps, axis=1, result_type="expand")
    out["context_primary_gap"] = gaps_mapped[0]
    out["context_secondary_gap"] = gaps_mapped[1]
    
    # Scatter Data
    scatter_rows = []
    for _, prod in out.iterrows():
        # Self
        scatter_rows.append({
            "selected_product_id": prod["product_id"],
            "plot_product_id": prod["product_id"],
            "plot_product_name": prod["display_name"],
            "is_selected_product": True,
            "price": prod["price"],
            "purchase_conversion_rate": prod["purchase_conversion_rate"],
            "rating": prod["rating"],
            "category": prod["category"],
            "peer_group": "selected"
        })
        # Peers
        p_ids = str(prod.get("peer_product_ids", "")).split("|")
        for p_id in p_ids:
            if not p_id: continue
            p_data = out[out["product_id"] == p_id]
            if p_data.empty: continue
            p = p_data.iloc[0]
            scatter_rows.append({
                "selected_product_id": prod["product_id"],
                "plot_product_id": p_id,
                "plot_product_name": p["display_name"],
                "is_selected_product": False,
                "price": p["price"],
                "purchase_conversion_rate": p["purchase_conversion_rate"],
                "rating": p["rating"],
                "category": p["category"],
                "peer_group": "peer"
            })
    scatter = pd.DataFrame(scatter_rows)
    
    # Write files
    safe_write_csv(out, output_dir / config.LAYER2_PEERS_FILE)
    safe_write_csv(pd.DataFrame(peer_rows), output_dir / config.LAYER2_SIMILARITY_FILE)
    safe_write_csv(profiles, output_dir / config.LAYER2_CLUSTERS_FILE)
    safe_write_csv(scatter, output_dir / config.LAYER2_SCATTER_FILE)
    save_json({"products_evaluated": len(out), "cluster_count": best_k}, output_dir / config.LAYER2_SUMMARY_FILE)
    
    LOGGER.info("Layer 2 finished successfully")
    return {"product_peers": out, "similarity": pd.DataFrame(peer_rows), "cluster_profiles": profiles, "scatter": scatter}
