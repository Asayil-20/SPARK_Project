"""Central configuration for the SPARK backend (UI-ready backend v2)."""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

LAYER0_DIR = OUTPUT_DIR / "layer0"
LAYER1_DIR = OUTPUT_DIR / "layer1"
LAYER2_DIR = OUTPUT_DIR / "layer2"
LAYER3_DIR = OUTPUT_DIR / "layer3"
LAYER4_DIR = OUTPUT_DIR / "layer4"
LAYER5_DIR = OUTPUT_DIR / "layer5"

RUN_CONTEXT_FILE = "run_context.json"

PRODUCTS_FILE = "products.csv"
EVENTS_FILE = "events.csv"
ORDERS_FILE = "orders.csv"
ORDER_ITEMS_FILE = "order_items.csv"
REVIEWS_FILE = "reviews.csv"
USERS_FILE = "users.csv"

LAYER0_PRODUCTS_FILE = "cleaned_products.csv"
LAYER0_EVENTS_FILE = "cleaned_events.csv"
LAYER0_ORDERS_FILE = "cleaned_orders.csv"
LAYER0_ORDER_ITEMS_FILE = "cleaned_order_items.csv"
LAYER0_REVIEWS_FILE = "cleaned_reviews.csv"
LAYER0_USERS_FILE = "cleaned_users.csv"
LAYER0_PRODUCT_MASTER_FILE = "product_master.csv"
LAYER0_QUALITY_FILE = "layer0_quality_report.json"
LAYER0_COVERAGE_FILE = "layer0_data_coverage.json"

LAYER1_DIAGNOSTICS_FILE = "layer1_product_diagnostics.csv"
LAYER1_MATRIX_FILE = "layer1_diagnostic_matrix.csv"
LAYER1_KPI_SUMMARY_FILE = "layer1_kpi_summary.json"
LAYER1_SUMMARY_FILE = "layer1_summary.json"

LAYER2_SIMILARITY_FILE = "layer2_contextual_similarity.csv"
LAYER2_PEERS_FILE = "layer2_product_peers.csv"
LAYER2_CLUSTERS_FILE = "layer2_cluster_profiles.csv"
LAYER2_SCATTER_FILE = "layer2_scatter_plot_data.csv"
LAYER2_SUMMARY_FILE = "layer2_summary.json"

LAYER3_BUNDLES_FILE = "layer3_bundle_candidates.csv"
LAYER3_OPPORTUNITIES_FILE = "layer3_product_bundle_opportunities.csv"
LAYER3_NODES_FILE = "layer3_bundle_network_nodes.csv"
LAYER3_EDGES_FILE = "layer3_bundle_network_edges.csv"
LAYER3_SUMMARY_FILE = "layer3_bundle_summary.json"

LAYER4_RECOMMENDATIONS_FILE = "layer4_prescriptive_recommendations.csv"
LAYER4_HIGH_PRIORITY_FILE = "layer4_high_priority.csv"
LAYER4_GENAI_INPUT_FILE = "layer4_products_for_genai.csv"
LAYER4_DETAIL_VIEW_FILE = "layer4_product_detail_view.csv"
LAYER4_OVERVIEW_FILE = "layer4_overview_payload.json"
LAYER4_SUMMARY_FILE = "layer4_summary.json"

LAYER5_RECOMMENDATIONS_FILE = "layer5_genai_recommendations.csv"
LAYER5_CACHE_FILE = "layer5_cache.json"
LAYER5_SUMMARY_FILE = "layer5_summary.json"

# ── Data boundary constants ───────────────────────────────────────────────────
# These are derived from the actual dataset and used to clamp all date pickers
# and preset calculations so the UI never suggests a date outside the data range.
# Update these if the underlying dataset is refreshed.
from datetime import date as _date
DATA_START_DATE: _date = _date(2023, 1, 1)   # earliest record in events + orders
DATA_END_DATE:   _date = _date(2024, 12, 30)  # latest  record in events + orders
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_ANALYSIS_DAYS = 90
MIN_VIEWS_FOR_CONFIDENT_DIAGNOSIS = 30
MIN_PURCHASES_FOR_CONFIDENT_DIAGNOSIS = 3
MIN_CART_USERS_FOR_ABANDONMENT = 5

WEAK_CONVERSION_QUANTILE = 0.35
STRONG_CONVERSION_QUANTILE = 0.70

TOP_K_PEERS = 5
SIMILARITY_MIN_SCORE = 0.15
PRICE_GAP_THRESHOLD = 0.15
RATING_GAP_THRESHOLD = 0.35
REVIEW_GAP_THRESHOLD = 0.50
IMAGE_GAP_THRESHOLD = 0.30
CONTENT_GAP_THRESHOLD = 0.30
NEW_PRODUCT_DAYS = 30

MIN_SUPPORT = 0.01
MIN_CONFIDENCE = 0.10
MIN_LIFT = 1.10
MAX_BASKET_SIZE = 25
MIN_CO_OCCURRENCE = 2
TOP_BUNDLE_OPPORTUNITIES_PER_PRODUCT = 5

PRODUCT_CLASS_WEIGHT = 0.30
CONTEXT_PRIORITY_WEIGHT = 0.20
UNDERPERFORMANCE_WEIGHT = 0.25
URGENCY_WEIGHT = 0.15
BUNDLE_SIGNAL_WEIGHT = 0.10

GENAI_MODEL = "llama-3.3-70b-versatile"
GENAI_TEMPERATURE = 0.15
GENAI_MAX_TOKENS = 900
GENAI_API_KEY_ENV_NAME = "GROQ_API_KEY"
ENABLE_GENAI_BATCH_DEFAULT = False

DEFAULT_LOG_LEVEL = os.getenv("SPARK_LOG_LEVEL", "INFO")
RANDOM_SEED = 42
CSV_ENCODING = "utf-8-sig"


def all_output_dirs() -> list[Path]:
    return [OUTPUT_DIR, LAYER0_DIR, LAYER1_DIR, LAYER2_DIR, LAYER3_DIR, LAYER4_DIR, LAYER5_DIR]
