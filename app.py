from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

import config
from src.layer0_data_processing import run_layer0
from src.layer1_diagnostics import run_layer1
from src.layer2_similarity import run_layer2
from src.layer3_bundles import run_layer3
from src.layer4_decision_engine import run_layer4, simulate_price_scenario
from src.layer5_genai_explainer import generate_one_explanation
from src.utils import ensure_dir, load_json, save_json, setup_logging

# -----------------------------------------------------------------------------
# App configuration
# -----------------------------------------------------------------------------
load_dotenv()
setup_logging()

st.set_page_config(
    page_title="SPARK",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800&display=swap');

        html, body, [class*="css"], [data-testid="stAppViewContainer"] {
            font-family: 'Tajawal', sans-serif !important;
            direction: rtl;
        }

        :root {
            --primary: #0f766e;
            --accent: #5ec8af;
            --bg-body: #F9FAFB;
            --text-main: #111827;
            --text-muted: #6B7280;
            --border-color: #E5E7EB;
            --card-bg: #FFFFFF;
            --success: #10B981;
            --warning: #F59E0B;
            --danger: #EF4444;
        }

        .stApp { background: var(--bg-body); color: var(--text-main); }
        [data-testid="stHeader"] { background: transparent; }
        .block-container { max-width: 1200px; padding: 2rem 1rem !important; }
        #MainMenu, footer, header, .stDeployButton { visibility: hidden; }

        /* Modern Shell/Card System */
        .shell {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 1.5rem;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .shell:hover { transform: translateY(-2px); box-shadow: 0 10px 15px -3px rgba(0,0,0,0.05); }

        .brand-wrap { display:flex; align-items:center; gap:14px; }
        .brand-icon {
            width: 48px; height: 48px; border-radius: 16px;
            background: linear-gradient(135deg, #14b8a6, #23d4bf);
            display:flex; align-items:center; justify-content:center;
            color:white; font-size:24px; font-weight:800;
            box-shadow: 0 12px 28px rgba(23,185,166,.25);
        }
        .brand-title { color: var(--primary); font-size: 22px; font-weight: 800; margin-bottom: 2px; }
        .brand-sub { color: var(--text-muted); font-size: 12px; }
        .top-pill {
            display:inline-block; background: linear-gradient(135deg, #0f766e, #14b8a6);
            color:white; padding:6px 12px; border-radius:999px; font-size:12px; font-weight:700;
        }

        .hero {
            background: linear-gradient(135deg, #0f766e 0%, #0f766e 100%);
            border-radius: 24px;
            padding: 60px 40px;
            color: white;
            text-align: center;
            margin-bottom: 3rem;
            box-shadow: 0 20px 25px -5px rgba(15, 118, 110, 0.1);
        }
        .hero-tag {
            background: rgba(94, 200, 175, 0.2);
            color: #5ec8af;
            padding: 8px 16px;
            border-radius: 100px;
            font-size: 14px;
            font-weight: 600;
            display: inline-block;
            margin-bottom: 20px;
        }
        .hero h1 { font-size: 48px; font-weight: 800; margin-bottom: 20px; color: white; letter-spacing: -0.02em; }
        .hero p { font-size: 18px; color: #d1d5db; max-width: 700px; margin: 0 auto; line-height: 1.6; }
        .hero-foot { margin-top: 30px; font-weight: 600; color: #5ec8af; }

        .card, .metric-card, .product-row, .soft-panel, .action-box {
            background: white;
            border: 1px solid var(--border-color);
            border-radius: 16px;
            box-shadow: 0 2px 8px rgba(15, 118, 110, 0.04);
        }
        .card { padding: 24px; }
        .soft-panel { padding: 20px; }
        .section-title { color: var(--primary); font-size: 22px; font-weight: 800; margin-bottom: .3rem; }
        .section-sub { color: var(--text-muted); font-size: 13px; margin-bottom: 1rem; line-height: 1.6; }
        .metric-card { padding: 16px; min-height: 100px; }
        .metric-label { color: var(--text-muted); font-size: 12px; font-weight: 600; margin-bottom: 8px; }
        .metric-value { color: var(--primary); font-size: 28px; font-weight: 800; line-height: 1.1; }
        .metric-note { color: var(--accent); font-size: 11px; font-weight: 600; margin-top: 8px; }
        .journey-card {
            padding: 24px;
            height: 100%;
            border-radius: 16px;
            background: white;
            border: 1px solid var(--border-color);
            transition: all 0.3s ease;
        }
        .journey-card:hover { border-color: var(--accent); transform: translateY(-4px); }
        .journey-icon {
            font-size: 32px;
            margin-bottom: 16px;
            display: block;
        }
        .journey-title { font-size: 18px; font-weight: 700; color: var(--primary); margin-bottom: 8px; }
        .journey-text { font-size: 14px; color: var(--text-muted); line-height: 1.5; }
        .journey-link { color: #10b8a4; font-size: 14px; font-weight: 700; margin-top: 18px; }

        .summary-chip, .filter-chip, .state-pill {
            display:inline-flex; align-items:center; gap:6px; padding: 6px 12px; border-radius: 999px;
            font-size: 12px; font-weight: 700; border: 1px solid transparent;
        }
        .summary-chip { background: #edf9f6; color: #0f766e; border-color: var(--border); }
        .state-urgent { background: var(--danger-bg); color: var(--danger-tx); border-color: #fecaca; }
        .state-improve { background: var(--warn-bg); color: var(--warn-tx); border-color: #fed7aa; }
        .state-good { background: var(--ok-bg); color: var(--ok-tx); border-color: #bbf7d0; }
        .state-info { background: var(--neutral-bg); color: var(--neutral-tx); border-color: #e4e7ec; }

        .product-row { padding: 16px; margin-bottom: 8px; }
        .prod-name { color: var(--primary); font-size: 17px; font-weight: 800; margin-bottom: 6px; }
        .prod-meta { color: var(--text-muted); font-size: 12px; }
        .prod-tagline { color: var(--accent); font-size: 12px; font-weight: 600; margin-top: 4px; }

        .detail-header {
            background: white;
            border: 1px solid var(--border-color); border-radius: 20px;
            padding: 32px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            margin-bottom: 2rem;
        }
        .detail-title { color: var(--primary); font-size: 32px; font-weight: 800; margin-bottom: 12px; }
        .detail-meta { color: var(--text-muted); font-size: 14px; font-weight: 500; }
        .detail-why { color: var(--accent); font-size: 14px; font-weight: 600; margin-top: 12px; }
        
        .evidence-box {
            background: #F3F4F6; border-radius: 12px; padding: 20px; border: none;
        }
        .evidence-label { color: var(--text-muted); font-size: 12px; font-weight: 600; text-transform: uppercase; margin-bottom: 8px; }
        .evidence-text { color: var(--primary); font-size: 16px; font-weight: 700; }
        
        .decision-box {
            background: #E0F2F1; border-left: 4px solid var(--primary); border-radius: 8px; padding: 24px; margin-top: 20px;
        }
        .decision-title { color: var(--primary); font-size: 20px; font-weight: 800; margin-bottom: 12px; }
        .decision-text { color: var(--text-main); font-size: 15px; line-height: 1.8; }

        .chart-shell {
            background: white; border: 1px solid var(--border-color); border-radius: 16px; padding: 16px;
            box-shadow: 0 2px 8px rgba(15, 118, 110, 0.04);
        }
        .chart-title { color: var(--primary); font-size: 16px; font-weight: 800; margin-bottom: 8px; }
        .chart-sub { color: var(--text-muted); font-size: 12px; margin-bottom: 6px; }

        .stButton > button, .stDownloadButton > button {
            border-radius: 16px !important; min-height: 46px !important; font-weight: 700 !important;
            border: 1px solid var(--border) !important;
        }
        .stButton > button[kind="primary"] {
            background: var(--primary) !important; color: white !important; border: none !important;
            box-shadow: 0 4px 10px rgba(15, 118, 110, 0.15);
        }
        .stButton > button:hover {
            border-color: var(--accent) !important; color: var(--accent) !important;
        }
        .stButton > button[kind="primary"]:hover {
            background: var(--accent) !important; color: white !important;
        }
        div[data-testid="metric-container"] {
            background: white !important; border: 1px solid var(--border-color) !important; border-radius: 12px !important;
            padding: 12px !important; box-shadow: 0 2px 8px rgba(15, 118, 110, 0.04) !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px; background: white; border-radius: 12px; padding: 4px; border: 1px solid var(--border-color);
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px !important; height: 40px; padding: 0 14px; font-weight: 700; color: var(--text-muted);
        }
        .stTabs [aria-selected="true"] {
            background: var(--bg-light) !important; color: var(--primary) !important; box-shadow: none; border: 1px solid var(--accent) !important;
        }
        .stDateInput label, .stSelectbox label, .stTextInput label, .stNumberInput label {
            font-weight: 700 !important; color: var(--primary) !important;
        }
        .small-note { color: var(--muted); font-size: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------
def init_session_state() -> None:
    # ── استخدم حدود البيانات الفعلية بدل date.today() ──────────────────────────
    # date.today() يُوهم المستخدم بإمكانية تحليل تواريخ خارج نطاق البيانات.
    # الحد الأقصى الآمن دائماً هو آخر سجل في مجموعة البيانات.
    data_end   = config.DATA_END_DATE    # 2024-12-30
    data_start = config.DATA_START_DATE  # 2023-01-01
    default_end   = data_end
    default_start = max(data_start, data_end - timedelta(days=89))  # آخر 90 يوم من البيانات

    st.session_state.setdefault("page", "home")
    st.session_state.setdefault("selected_product_id", None)
    st.session_state.setdefault("preset_label", "آخر 90 يوم")
    st.session_state.setdefault("analysis_start", default_start)
    st.session_state.setdefault("analysis_end", default_end)
    st.session_state.setdefault("date_start_input", default_start)
    st.session_state.setdefault("date_end_input", default_end)
    st.session_state.setdefault("data_refresh_key", datetime.now().isoformat())


init_session_state()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def safe_text(value: Any, fallback: str = "—") -> str:
    if value is None:
        return fallback
    if isinstance(value, float) and pd.isna(value):
        return fallback
    text = str(value).strip()
    return text if text else fallback


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return default
        return float(value)
    except Exception:
        return default


def to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return default
        return int(round(float(value)))
    except Exception:
        return default


def fmt_number(value: Any) -> str:
    return f"{to_int(value):,}"


def fmt_currency(value: Any) -> str:
    return f"{to_float(value):,.0f} ر.س"


def fmt_pct(value: Any) -> str:
    v = to_float(value)
    if v <= 1.5:
        v *= 100
    return f"{v:.1f}%"


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def apply_preset(days: int, label: str) -> None:
    # نهاية الفترة = آخر يوم بيانات متاح (DATA_END_DATE) لا تاريخ اليوم الحقيقي.
    # استخدام date.today() يُوهم المستخدم بوجود بيانات حديثة غير موجودة فعلاً.
    data_end   = config.DATA_END_DATE
    data_start = config.DATA_START_DATE
    end   = data_end
    start = max(data_start, data_end - timedelta(days=days - 1))
    st.session_state.analysis_start = start
    st.session_state.analysis_end = end
    st.session_state.date_start_input = start
    st.session_state.date_end_input = end
    st.session_state.preset_label = label
    # Flag that user has staged a new window (shown in navbar as a warning badge)
    st.session_state.setdefault("staged_preset_pending", True)
    st.session_state["staged_preset_pending"] = True


def apply_preset_and_run(days: int, label: str) -> None:
    """Stage the preset (محدودة بنطاق البيانات الفعلي) AND immediately re-run the pipeline."""
    apply_preset(days, label)
    with st.spinner(f"جاري تحليل {label}..."):
        ok, message = run_pipeline_from_ui(
            st.session_state.analysis_start,
            st.session_state.analysis_end,
            st.session_state.preset_label,
        )
    st.session_state["staged_preset_pending"] = False
    if ok:
        st.session_state.page = "dashboard"
        st.rerun()
    else:
        st.error(message)


def sync_dates_from_inputs() -> None:
    start = st.session_state.get("date_start_input")
    end = st.session_state.get("date_end_input")
    if start and end:
        # clamp ضمن نطاق البيانات الفعلي لمنع إرسال تواريخ خارج النطاق للـ pipeline
        _dmin, _dmax = config.DATA_START_DATE, config.DATA_END_DATE
        start = max(start, _dmin) if hasattr(start, 'year') else start
        end   = min(end,   _dmax) if hasattr(end,   'year') else end
        st.session_state.analysis_start = start
        st.session_state.analysis_end = end
        st.session_state.preset_label = "فترة مخصصة"


def status_bucket(row: pd.Series) -> tuple[str, str]:
    # Use product_class (Strong, Moderate, Weak) which is the statistical source of truth
    raw_class = safe_text(row.get("product_class"), "Moderate")
    
    # Map to Arabic labels for UI consistency (Red, Yellow, Green)
    if raw_class == "Strong":
        return "قوي", "state-good"      # Green
    elif raw_class == "Moderate":
        return "متوسط", "state-improve"  # Yellow
    elif raw_class == "Weak":
        return "ضعيف", "state-urgent"   # Red
    
    return "تحت المراجعة", "state-info"


def root_cause_label(value: Any) -> str:
    mapping = {
        "pricing": "السعر",
        "content": "جودة المحتوى",
        "trust": "الثقة",
        "visibility": "ضعف الظهور",
        "conversion_friction": "عدم إكمال الشراء",
        "insufficient_data": "مؤشرات أولية",
        "mixed_context": "عوامل متعددة",
    }
    return mapping.get(str(value), safe_text(value))


def human_reason_short(row: pd.Series) -> str:
    sub = safe_text(row.get("performance_substatus"))
    root = root_cause_label(row.get("root_cause"))
    if sub != "—":
        return sub
    return root


def pill_html(text: str, css_class: str) -> str:
    return f"<span class='state-pill {css_class}'>{text}</span>"


def find_product_row(df: pd.DataFrame, product_id: int) -> pd.Series | None:
    if df.empty or "product_id" not in df.columns:
        return None
    rows = df[df["product_id"] == product_id]
    if rows.empty:
        return None
    return rows.iloc[0]


@st.cache_data(show_spinner=False)
def _sync_data_boundaries() -> None:
    """
    اقرأ الحدود الزمنية الفعلية من ملفات layer0 وحدّث config ديناميكياً.
    يُضمن أن DATA_START_DATE / DATA_END_DATE تعكس دائماً البيانات الموجودة
    حتى لو تم تحديث ملفات data/ بمجموعة بيانات مختلفة.
    """
    try:
        ev_path  = config.LAYER0_DIR / config.LAYER0_EVENTS_FILE
        ord_path = config.LAYER0_DIR / config.LAYER0_ORDERS_FILE
        if ev_path.exists() and ord_path.exists():
            ev  = pd.read_csv(ev_path,  usecols=["event_time"])
            ord_ = pd.read_csv(ord_path, usecols=["order_date"])
            ev["event_time"]  = pd.to_datetime(ev["event_time"],  errors="coerce")
            ord_["order_date"] = pd.to_datetime(ord_["order_date"], errors="coerce")
            all_min = min(ev["event_time"].min(), ord_["order_date"].min())
            all_max = max(ev["event_time"].max(), ord_["order_date"].max())
            if pd.notna(all_min) and pd.notna(all_max):
                config.DATA_START_DATE = all_min.date()
                config.DATA_END_DATE   = all_max.date()
    except Exception:
        pass  # non-fatal — fallback to hard-coded values in config.py


_sync_data_boundaries()   # استدعاء عند بدء تشغيل التطبيق


def load_app_data(refresh_key: str) -> dict[str, Any]:
    base = config.OUTPUT_DIR
    layer1 = config.LAYER1_DIR
    layer2 = config.LAYER2_DIR
    layer3 = config.LAYER3_DIR
    layer4 = config.LAYER4_DIR
    layer5 = config.LAYER5_DIR

    payload = {
        "run_context": load_json(base / config.RUN_CONTEXT_FILE),
        "kpis": load_json(layer1 / config.LAYER1_KPI_SUMMARY_FILE),
        "overview": load_json(layer4 / config.LAYER4_OVERVIEW_FILE),
        "diagnostics": read_csv_if_exists(layer1 / config.LAYER1_DIAGNOSTICS_FILE),
        "matrix": read_csv_if_exists(layer1 / config.LAYER1_MATRIX_FILE),
        "peers": read_csv_if_exists(layer2 / config.LAYER2_PEERS_FILE),
        "scatter": read_csv_if_exists(layer2 / config.LAYER2_SCATTER_FILE),
        "bundle_opportunities": read_csv_if_exists(layer3 / config.LAYER3_OPPORTUNITIES_FILE),
        "recommendations": read_csv_if_exists(layer4 / config.LAYER4_RECOMMENDATIONS_FILE),
        "detail_view": read_csv_if_exists(layer4 / config.LAYER4_DETAIL_VIEW_FILE),
        "high_priority": read_csv_if_exists(layer4 / config.LAYER4_HIGH_PRIORITY_FILE),
        "genai_output": read_csv_if_exists(layer5 / config.LAYER5_RECOMMENDATIONS_FILE),
    }
    return payload


def run_pipeline_from_ui(start_date: date, end_date: date, preset_label: str | None = None) -> tuple[bool, str]:
    if start_date is None or end_date is None:
        return False, "يرجى تحديد تاريخ البداية والنهاية."
    if start_date > end_date:
        return False, "تاريخ البداية يجب أن يكون قبل تاريخ النهاية."
    # التحقق أن التواريخ ضمن نطاق البيانات الفعلي
    _dmin, _dmax = config.DATA_START_DATE, config.DATA_END_DATE
    if end_date < _dmin or start_date > _dmax:
        return False, (
            f"الفترة المختارة ({start_date} → {end_date}) خارج نطاق البيانات المتاحة "
            f"({_dmin} → {_dmax}). لن تظهر أي نتائج."
        )
    # تحذير إن كانت الفترة تتجاوز نهاية البيانات (لكنها ليست خطأ فادحاً)
    _overlap_warning = None
    if isinstance(end_date, date) and end_date > _dmax:
        end_date = _dmax
        _overlap_warning = f"تم تعديل تاريخ النهاية إلى {_dmax} (آخر بيانات متاحة)."

    try:
        ensure_dir(config.OUTPUT_DIR)
        layer0_dir = ensure_dir(config.LAYER0_DIR)
        layer1_dir = ensure_dir(config.LAYER1_DIR)
        layer2_dir = ensure_dir(config.LAYER2_DIR)
        layer3_dir = ensure_dir(config.LAYER3_DIR)
        layer4_dir = ensure_dir(config.LAYER4_DIR)
        ensure_dir(config.LAYER5_DIR)

        run_layer0(config.DATA_DIR, layer0_dir)
        save_json(
            {
                "analysis_start_date": str(start_date),
                "analysis_end_date": str(end_date),
                "preset_label": preset_label,
                "analysis_days": max((end_date - start_date).days, 0),
            },
            config.OUTPUT_DIR / config.RUN_CONTEXT_FILE,
        )
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        run_layer1(layer0_dir, layer1_dir, start_date=start_ts, end_date=end_ts)
        run_layer2(layer0_dir, layer1_dir, layer2_dir)
        run_layer3(layer0_dir, layer1_dir, layer2_dir, layer3_dir)
        run_layer4(layer1_dir, layer2_dir, layer3_dir, layer4_dir)

        # ── Post-run consistency check ─────────────────────────────────────
        try:
            _l1 = pd.read_csv(layer1_dir / config.LAYER1_DIAGNOSTICS_FILE)
            _l4 = pd.read_csv(layer4_dir / config.LAYER4_RECOMMENDATIONS_FILE)
            _l1_rev = _l1["revenue"].sum() if "revenue" in _l1.columns else 0
            _l4_rev = _l4["revenue"].sum() if "revenue" in _l4.columns else 0
            _ratio = _l4_rev / _l1_rev if _l1_rev > 0 else 0.0
            if _ratio > 3.0 or (_ratio < 0.33 and _l1_rev > 0):
                import logging as _log
                _log.getLogger("spark.app").warning(
                    "Layer consistency check: L1 revenue=%.0f, L4 revenue=%.0f, ratio=%.2f. "
                    "Possible partial run or data mismatch.", _l1_rev, _l4_rev, _ratio
                )
        except Exception:
            pass  # non-fatal — don't block UI on validation errors
        # ──────────────────────────────────────────────────────────────────

        st.cache_data.clear()
        st.session_state.data_refresh_key = datetime.now().isoformat()
        st.session_state["staged_preset_pending"] = False
        return True, "تم تحديث النتائج بنجاح."
    except Exception as exc:
        return False, f"تعذر تشغيل التحليل: {exc}"


# -----------------------------------------------------------------------------
# Charts
# -----------------------------------------------------------------------------
def build_priority_bar(df: pd.DataFrame) -> go.Figure:
    plot_df = df["priority_tier"].fillna("غير محدد").value_counts().reset_index()
    plot_df.columns = ["priority_tier", "count"]
    fig = px.bar(plot_df, x="priority_tier", y="count", text="count")
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="",
        yaxis_title="",
        font=dict(family="Tajawal, sans-serif"),
    )
    return fig


def build_reason_donut(df: pd.DataFrame) -> go.Figure:
    plot_df = df["root_cause"].fillna("other").map(root_cause_label).value_counts().reset_index()
    plot_df.columns = ["root_cause", "count"]
    fig = px.pie(plot_df, names="root_cause", values="count", hole=0.62)
    fig.update_traces(textposition="inside", textinfo="percent")
    fig.update_layout(
        height=330,
        margin=dict(l=10, r=150, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Tajawal, sans-serif"),
        legend_title="",
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
            font=dict(size=12),
            traceorder="normal",
        ),
    )
    return fig


def build_funnel_chart(product: pd.Series) -> go.Figure:
    views = max(to_float(product.get("views"), 1), 1)
    carts = max(to_float(product.get("carts"), 0), 0)
    purchases = max(to_float(product.get("purchases"), 0), 0)
    fig = go.Figure(go.Funnel(y=["المشاهدات", "إضافات السلة", "المشتريات"], x=[views, carts, purchases], textinfo="value+percent initial"))
    fig.update_layout(
        height=360,
        margin=dict(l=130, r=130, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Tajawal, sans-serif"),
        yaxis=dict(automargin=True, tickfont=dict(size=13)),
    )
    return fig


def build_context_scatter(scatter_df: pd.DataFrame, product_id: int) -> go.Figure | None:
    subset = scatter_df[scatter_df["selected_product_id"] == product_id].copy() if not scatter_df.empty else pd.DataFrame()
    if subset.empty:
        return None
    subset["highlight"] = subset["is_selected_product"].map({True: "المنتج المحدد", False: "منتجات مشابهة"})
    fig = px.scatter(
        subset,
        x="price",
        y="purchase_conversion_rate",
        size="rating" if "rating" in subset.columns else None,
        color="highlight",
        hover_name="plot_product_name",
        hover_data={"category": True, "price": ':.2f', "purchase_conversion_rate": ':.2f'},
    )
    fig.update_layout(
        height=340,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="السعر",
        yaxis_title="معدل التحويل",
        font=dict(family="Tajawal, sans-serif"),
        legend_title="",
    )
    return fig


def build_bundle_network_chart(opportunities: pd.DataFrame, product_id: int, product_name: str) -> go.Figure | None:
    subset = opportunities[opportunities["product_id"] == product_id].head(5).copy() if not opportunities.empty else pd.DataFrame()
    if subset.empty:
        return None

    fig = go.Figure()
    origin_y = 0.0
    fig.add_trace(go.Scatter(x=[0], y=[origin_y], mode="markers+text", marker=dict(size=28), text=[product_name], textposition="bottom center", showlegend=False))

    for idx, (_, row) in enumerate(subset.iterrows(), start=1):
        y = idx * 1.05
        fig.add_trace(go.Scatter(x=[0, 1], y=[origin_y, y], mode="lines", showlegend=False, hoverinfo="skip"))
        fig.add_trace(
            go.Scatter(
                x=[1], y=[y], mode="markers+text", marker=dict(size=22),
                text=[safe_text(row.get("bundle_partner_name"))], textposition="bottom center", showlegend=False,
            )
        )
        fig.add_annotation(
            x=0.5, y=(origin_y + y) / 2,
            text=f"تكرار قوي • lift {to_float(row.get('lift')):.2f}", showarrow=False, font=dict(size=11),
        )

    fig.update_layout(
        height=350,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        font=dict(family="Tajawal, sans-serif"),
    )
    return fig


# -----------------------------------------------------------------------------
# Render blocks
# -----------------------------------------------------------------------------
def render_navbar(run_context: dict[str, Any]) -> None:
    # Prefer session_state dates so that preset/date changes are reflected immediately in navbar,
    # even before the pipeline is re-run. Fall back to run_context (last executed run).
    staged_start = st.session_state.get("analysis_start")
    staged_end = st.session_state.get("analysis_end")
    run_start = safe_text(run_context.get("analysis_start_date"))
    run_end = safe_text(run_context.get("analysis_end_date"))
    start = str(staged_start) if staged_start is not None else run_start
    end = str(staged_end) if staged_end is not None else run_end
    staged_preset = st.session_state.get("preset_label")
    run_preset = safe_text(run_context.get("preset_label"), fallback="فترة مخصصة")
    preset = staged_preset if staged_preset is not None else run_preset
    # Visual indicator: is the staged window different from the last executed run?
    dates_are_staged = (str(staged_start) != run_start or str(staged_end) != run_end) if staged_start else False

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(
            """
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:1rem;">
                <div style="background:var(--primary); color:white; width:40px; height:40px; border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:20px; font-weight:bold;">S</div>
                <div>
                    <div style="font-size:18px; font-weight:800; color:var(--primary); line-height:1;">SPARK</div>
                    <div style="font-size:12px; color:var(--text-muted);">Business Intelligence</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div style="text-align:left; color:var(--text-muted); font-size:14px; margin-top:5px;">
                <span style="font-weight:700; color:var(--primary);">{preset}</span> • {start} - {end}
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    st.markdown('<div style="margin-bottom:2rem;"></div>', unsafe_allow_html=True)
    
    m1, m2, m3, m4 = st.columns([1, 1, 1, 2])
    with m1:
        if st.button("🏠 الرئيسية", use_container_width=True, type="secondary" if st.session_state.page != "home" else "primary"):
            st.session_state.page = "home"
            st.rerun()
    with m2:
        if st.button("📊 لوحة البيانات", use_container_width=True, type="secondary" if st.session_state.page != "dashboard" else "primary"):
            st.session_state.page = "dashboard"
            st.rerun()
    with m3:
        disabled = st.session_state.selected_product_id is None
        if st.button("🔍 التفاصيل", use_container_width=True, disabled=disabled, type="secondary" if st.session_state.page != "details" else "primary"):
            st.session_state.page = "details"
            st.rerun()
    with m4:
        if dates_are_staged:
            st.warning("⚠️ الفترة مُحدَّدة — اضغط تحديث النتائج لتطبيقها", icon="⚠️")


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-tag">نظام SPARK • يساعدك على تحسين مبيعات المنتجات</div>
            <h1>حوّل بيانات متجرك إلى قرارات ترفع النمو وتوضح فرص التحسين</h1>
            <p>
                افهم أين يتسرب الاهتمام قبل الشراء، وكيف يظهر منتجك أمام البدائل القريبة،
                وما الذي يمكن تنفيذه الآن لرفع المبيعات أو زيادة متوسط قيمة الطلب.
            </p>
            <div class="hero-foot">التحليل موجه للتاجر وصاحب القرار.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_journey_cards() -> None:
    st.markdown('<div class="section-title" style="text-align:center;margin-top:1.6rem">رحلة نمو مبيعاتك</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub" style="text-align:center">خطوات واضحة تبدأ من قراءة الأداء وتنتهي بإجراء يمكن تنفيذه مباشرة.</div>', unsafe_allow_html=True)

    cols = st.columns(4)
    cards = [
        ("📊", "فحص الأداء", "تحليل شامل لجميع المنتجات لتحديد نقاط القوة والضعف."),
        ("🎯", "الموقع التنافسي", "مقارنة دقيقة مع المنافسين المباشرين في السوق."),
        ("🔗", "رفع قيمة السلة", "اكتشاف فرص البيع المتقاطع لزيادة الربحية."),
        ("✳️", "قرار مقترح", "توصيات ذكية مدعومة بالذكاء الاصطناعي للتنفيذ الفوري."),
    ]
    for col, (icon, title, text) in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div class="journey-card">
                    <span class="journey-icon">{icon}</span>
                    <div class="journey-title">{title}</div>
                    <div class="journey-text">{text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_kpi_cards(kpis: dict[str, Any], recommendations: pd.DataFrame) -> None:
    total_revenue = to_float(recommendations.get("revenue", pd.Series(dtype=float)).sum()) if not recommendations.empty else 0.0
    # Use total_sales (purchase_signal = max(events, orders)) for the primary sales KPI.
    # Show a sub-note if data quality warning exists (e.g. event purchase capture is low).
    sales_note = "عدد الطلبات المؤكدة خلال الفترة"
    dq = kpis.get("data_quality_note")
    if dq == "purchase_event_capture_low":
        sales_note = "مبني على الطلبات الفعلية • تتبع أحداث الشراء أقل من المتوقع"
    values = [
        ("إجمالي المشاهدات", fmt_number(kpis.get("total_views", 0)), "حركة التصفح خلال الفترة"),
        ("إضافات السلة", fmt_number(kpis.get("total_carts", 0)), "منتجات جذبت اهتمامًا قبل الشراء"),
        ("إجمالي المبيعات", fmt_number(kpis.get("total_sales", 0)), sales_note),
        ("معدل التحويل", fmt_pct(kpis.get("overall_conversion", 0)), "من المشاهدة إلى الشراء"),
        ("الإيرادات", fmt_currency(total_revenue), "إجمالي القيمة البيعية ضمن النتائج"),
    ]
    cols = st.columns(len(values))
    for col, (label, value, note) in zip(cols, values):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-note">{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def build_dashboard_listing_df(recommendations: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Use Layer 4 recommendations as the main dashboard list, then append missing Weak products from Layer 1 diagnostics."""
    if recommendations.empty:
        base = pd.DataFrame()
    else:
        base = recommendations.copy()

    if diagnostics.empty or "product_class" not in diagnostics.columns or "product_id" not in diagnostics.columns:
        return base

    weak_rows = diagnostics[diagnostics["product_class"].astype(str).str.strip().str.lower() == "weak"].copy()
    if weak_rows.empty:
        return base

    existing_ids = set()
    if not base.empty and "product_id" in base.columns:
        existing_ids = set(base["product_id"].astype(str))

    weak_rows = weak_rows[~weak_rows["product_id"].astype(str).isin(existing_ids)].copy()
    if weak_rows.empty:
        return base

    # Ensure rows coming only from Layer 1 still fit the existing product-row renderer.
    if "display_name" not in weak_rows.columns:
        weak_rows["display_name"] = weak_rows["name"] if "name" in weak_rows.columns else weak_rows["product_id"].astype(str)
    if "priority_tier" not in weak_rows.columns:
        weak_rows["priority_tier"] = "غير محدد"
    else:
        weak_rows["priority_tier"] = weak_rows["priority_tier"].fillna("غير محدد")
    if "root_cause" not in weak_rows.columns:
        weak_rows["root_cause"] = "insufficient_data"
    if "performance_substatus" not in weak_rows.columns:
        weak_rows["performance_substatus"] = "—"

    if base.empty:
        return weak_rows

    for col in base.columns:
        if col not in weak_rows.columns:
            weak_rows[col] = pd.NA
    for col in weak_rows.columns:
        if col not in base.columns:
            base[col] = pd.NA

    return pd.concat([base, weak_rows[base.columns]], ignore_index=True)


def render_product_rows(df: pd.DataFrame) -> None:
    if df.empty:
        st.warning("لا توجد منتجات مطابقة للفلاتر الحالية.")
        return

    # Use a cleaner table-like layout instead of heavy individual card rows for performance
    for _, row in df.iterrows():
        ui_status, ui_class = status_bucket(row)
        with st.container():
            c1, c2, c3, c4 = st.columns([5, 2, 2, 1.3])
            with c1:
                st.markdown(
                    f"""
                    <div style="padding:10px 0;">
                        <div style="font-weight:700; font-size:16px; color:var(--primary);">{safe_text(row.get('display_name') or row.get('name'))}</div>
                        <div style="font-size:12px; color:var(--text-muted);">{safe_text(row.get('category'))} • {safe_text(row.get('brand'))}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with c2:
                st.markdown(
                    f"""
                    <div style="padding:10px 0; text-align:center;">
                        <div style="font-weight:700; font-size:18px;">{fmt_currency(row.get('price'))}</div>
                        <div style="font-size:11px; color:var(--accent);">تحويل {fmt_pct(row.get('purchase_conversion_rate'))}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with c3:
                st.markdown(
                    f"""
                    <div style="padding:10px 0; text-align:center; display:flex; gap:5px; justify-content:center; align-items:center; height:100%;">
                        {pill_html(ui_status, ui_class)}
                        {pill_html(safe_text(row.get('priority_tier')), 'state-info')}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with c4:
                st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
                if st.button("تفاصيل", key=f"details_{row['product_id']}", use_container_width=True):
                    st.session_state.selected_product_id = int(row["product_id"])
                    st.session_state.page = "details"
                    st.rerun()
            st.markdown('<hr style="margin: 0.5rem 0; border:0; border-top:1px solid #eee;">', unsafe_allow_html=True)


def render_info_box(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="evidence-box">
            <div class="evidence-label">{label}</div>
            <div class="evidence-text">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Pages
# -----------------------------------------------------------------------------
def page_home(data: dict[str, Any]) -> None:
    render_navbar(data.get("run_context", {}))
    
    # 1) Analysis Period Selection (Centralized & Prominent)
    st.markdown('<div class="section-title" style="text-align:center;margin-top:2rem">ابدأ تحليلك الذكي</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub" style="text-align:center;margin-bottom:2rem">اختر الفترة الزمنية التي ترغب في تحليل أدائها بعمق</div>', unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 5, 1])
    with mid:
        st.markdown('<div class="soft-panel" style="padding:32px">', unsafe_allow_html=True)
        # Presets
        _preset_options = [
            (7,   "آخر 7 أيام"),
            (30,  "آخر 30 يوم"),
            (90,  "آخر 90 يوم"),
            (365, "السنة كاملة"),
        ]
        p1, p2, p3, p4 = st.columns(4)
        for _col, (_days, _label) in zip([p1, p2, p3, p4], _preset_options):
            _preset_start = max(config.DATA_START_DATE, config.DATA_END_DATE - timedelta(days=_days - 1))
            _has_data = _preset_start <= config.DATA_END_DATE
            with _col:
                if st.button(_label, use_container_width=True, disabled=not _has_data):
                    apply_preset_and_run(_days, _label)

        st.markdown('<div style="margin:24px 0; border-top:1px solid var(--border-color)"></div>', unsafe_allow_html=True)
        
        # Custom range
        d1, d2, d3 = st.columns([2, 2, 1.5])
        _data_min = config.DATA_START_DATE
        _data_max = config.DATA_END_DATE
        with d1:
            st.date_input("تاريخ البداية", key="date_start_input", min_value=_data_min, max_value=_data_max, on_change=sync_dates_from_inputs)
        with d2:
            st.date_input("تاريخ النهاية", key="date_end_input", min_value=_data_min, max_value=_data_max, on_change=sync_dates_from_inputs)
        with d3:
            st.markdown('<div style="margin-top:28px"></div>', unsafe_allow_html=True)
            if st.button("تحديث البيانات", type="primary", use_container_width=True):
                sync_dates_from_inputs()
                with st.spinner("جاري تحليل البيانات..."):
                    ok, message = run_pipeline_from_ui(
                        st.session_state.analysis_start,
                        st.session_state.analysis_end,
                        st.session_state.preset_label,
                    )
                if ok:
                    st.session_state.page = "dashboard"
                    st.rerun()
                else:
                    st.error(message)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="margin-top:3rem"></div>', unsafe_allow_html=True)
    
    # 2) Hero Section
    render_hero()
    
    # 3) Journey Cards (Secondary)
    render_journey_cards()


def page_dashboard(data: dict[str, Any]) -> None:
    recommendations = data.get("recommendations", pd.DataFrame())
    diagnostics = data.get("diagnostics", pd.DataFrame())
    dashboard_products = build_dashboard_listing_df(recommendations, diagnostics)
    run_context = data.get("run_context", {})
    kpis = data.get("kpis", {})

    render_navbar(run_context)
    # 1) KPI Section (Clean & Flat)
    render_kpi_cards(kpis, recommendations)
    st.markdown("<br>", unsafe_allow_html=True)

    # 2) Key Charts (Simplified)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown('<div class="chart-shell">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">توزيع أولويات التدخل</div>', unsafe_allow_html=True)
        st.plotly_chart(build_priority_bar(recommendations), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="chart-shell">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">العوامل المؤثرة على الأداء</div>', unsafe_allow_html=True)
        st.plotly_chart(build_reason_donut(recommendations), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">قائمة المنتجات</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">استخدمي الفلاتر لتضييق القائمة حسب الحالة أو الفئة أو العلامة التجارية.</div>', unsafe_allow_html=True)

    df = dashboard_products.copy()
    temp = df.apply(status_bucket, axis=1, result_type="expand")
    df["ui_status_label"] = temp[0]
    df["ui_status_class"] = temp[1]

    f1, f2, f3, f4 = st.columns([1, 1, 1, 1.5])
    with f1:
        status_filter = st.selectbox("تصفية حسب الحالة", ["الكل", "قوي", "متوسط", "ضعيف"])
    with f2:
        category_filter = st.selectbox("الفئة", ["الكل"] + sorted(df["category"].dropna().unique().tolist()))
    with f3:
        brand_filter = st.selectbox("العلامة التجارية", ["الكل"] + sorted(df["brand"].dropna().unique().tolist()))
    with f4:
        search_term = st.text_input("بحث سريع", placeholder="اسم المنتج...")

    filtered = df.copy()
    if status_filter != "الكل":
        filtered = filtered[filtered["ui_status_label"] == status_filter]
    if category_filter != "الكل":
        filtered = filtered[filtered["category"] == category_filter]
    if brand_filter != "الكل":
        filtered = filtered[filtered["brand"] == brand_filter]
    if search_term:
        pattern = str(search_term).strip()
        filtered = filtered[filtered["display_name"].fillna("").str.contains(pattern, case=False)]

    st.caption(f"عدد المنتجات المطابقة: {len(filtered):,}")

    # ── Pagination — عرض كامل المنتجات بدون حد ────────────────────────────
    PAGE_SIZE = 20
    total = len(filtered)
    total_pages = max(1, -(-total // PAGE_SIZE))  # ceiling division

    if total_pages > 1:
        page_key = "dashboard_page"
        if page_key not in st.session_state or st.session_state.get("_last_filter_hash") != hash(str(status_filter)+str(category_filter)+str(brand_filter)+str(search_term)):
            st.session_state[page_key] = 1
        st.session_state["_last_filter_hash"] = hash(str(status_filter)+str(category_filter)+str(brand_filter)+str(search_term))

        # شريط التنقل بين الصفحات
        nav_cols = st.columns([1, 3, 1])
        with nav_cols[0]:
            if st.button("→ السابق", disabled=(st.session_state[page_key] <= 1), use_container_width=True):
                st.session_state[page_key] -= 1
                st.rerun()
        with nav_cols[1]:
            st.markdown(
                f"<div style='text-align:center;padding:8px 0;color:#374151;font-size:14px;font-weight:600'>"
                f"الصفحة {st.session_state[page_key]} من {total_pages} — ({total} منتج)</div>",
                unsafe_allow_html=True,
            )
        with nav_cols[2]:
            if st.button("التالي ←", disabled=(st.session_state[page_key] >= total_pages), use_container_width=True):
                st.session_state[page_key] += 1
                st.rerun()

        current_page = st.session_state.get(page_key, 1)
        start_idx = (current_page - 1) * PAGE_SIZE
        page_slice = filtered.iloc[start_idx : start_idx + PAGE_SIZE]
    else:
        page_slice = filtered

    render_product_rows(page_slice)

    # قفز مباشر لصفحة محددة (يظهر فقط إذا أكثر من صفحتين)
    if total_pages > 2:
        jump_col = st.columns([2, 1, 2])[1]
        with jump_col:
            jump_to = st.number_input(
                "انتقل إلى صفحة", min_value=1, max_value=total_pages,
                value=st.session_state.get("dashboard_page", 1),
                step=1, key="page_jump_input",
                label_visibility="collapsed",
            )
            if st.button("انتقل", use_container_width=True):
                st.session_state["dashboard_page"] = int(jump_to)
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


def page_details(data: dict[str, Any]) -> None:
    recommendations = data.get("recommendations", pd.DataFrame())
    detail_view = data.get("detail_view", pd.DataFrame())
    peers = data.get("peers", pd.DataFrame())
    scatter = data.get("scatter", pd.DataFrame())
    opportunities = data.get("bundle_opportunities", pd.DataFrame())

    product_id = st.session_state.selected_product_id
    render_navbar(data.get("run_context", {}))

    if product_id is None:
        st.warning("اختاري منتجًا أولًا من لوحة المنتجات.")
        return

    product = find_product_row(detail_view, int(product_id))
    if product is None:
        product = find_product_row(recommendations, int(product_id))
    if product is None:
        st.error("تعذر العثور على بيانات هذا المنتج.")
        return

    ui_status, ui_class = status_bucket(product)
    product_name = safe_text(product.get("display_name") or product.get("name"))
    st.markdown(
        f"""
        <div class="detail-header">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:16px;flex-wrap:wrap">
                <div>
                    <div class="top-pill">تفاصيل المنتج</div>
                    <div class="detail-title">{product_name}</div>
                    <div class="detail-meta">{safe_text(product.get('category'))} • {safe_text(product.get('brand'))} • السعر {fmt_currency(product.get('price'))}</div>
                    <div class="detail-why">&nbsp;</div>
                </div>
                <div style="text-align:left">
                    <div style="margin-bottom:10px">{pill_html(ui_status, ui_class)}</div>
                    <div style="margin-bottom:10px">{pill_html('أولوية ' + safe_text(product.get('priority_tier')), 'state-info')}</div>
                    <div>{pill_html('ثقة ' + {'High':'مرتفعة','Medium':'متوسطة','Low':'أولية'}.get(safe_text(product.get('confidence_level')), safe_text(product.get('confidence_level'))) , 'state-info')}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(6)
    metric_payload = [
        ("المشاهدات", fmt_number(product.get("views"))),
        ("إضافات السلة", fmt_number(product.get("carts"))),
        ("المشتريات", fmt_number(product.get("purchases"))),
        ("الإيراد", fmt_currency(product.get("revenue"))),
        ("معدل التحويل", fmt_pct(product.get("purchase_conversion_rate"))),
        ("شرح المؤشر", "معدل التحويل يرمز إلى نسبة تحول حالة المنتج من مشاهدة إلى إضافة للسلة إلى الشراء."),
    ]
    for col, (label, value) in zip(metric_cols, metric_payload):
        with col:
            if label == "شرح المؤشر":
                st.markdown(
                    f"""
                    <div style="padding-top:2px;text-align:center">
                        <div class="metric-label">{label}</div>
                        <div style="color:#344e49;font-size:11px;line-height:1.65;font-weight:600;max-width:190px;margin:0 auto;white-space:normal">
                            معدل التحويل يرمز إلى نسبة تحول حالة المنتج<br>
                            من مشاهدة إلى إضافة للسلة إلى الشراء.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.metric(label, value)

    tabs = st.tabs(["ملخص الأداء", "مقارنة مع المنتجات", "فرص رفع قيمة السلة", "القرار المقترح"])

    with tabs[0]:
        l1, l2 = st.columns([1.2, 1])
        with l1:
            st.markdown('<div class="chart-shell">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">رحلة المنتج من المشاهدة إلى الشراء</div>', unsafe_allow_html=True)
            st.markdown('<div class="chart-sub">يوضح لك أين يتراجع الزخم قبل اكتمال الشراء.</div>', unsafe_allow_html=True)
            st.plotly_chart(build_funnel_chart(product), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with l2:
            st.markdown('<div class="soft-panel">', unsafe_allow_html=True)
            st.markdown('<div class="section-title" style="font-size:20px">ما الذي يحدث؟</div>', unsafe_allow_html=True)
            render_info_box("الخلاصة", safe_text(product.get("diagnostic_reason")))
            st.markdown('</div>', unsafe_allow_html=True)

    with tabs[1]:
        l1, l2 = st.columns([1.2, 1])
        with l1:
            st.markdown('<div class="chart-shell">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">موقع المنتج بين البدائل القريبة</div>', unsafe_allow_html=True)
            st.markdown('<div class="chart-sub">مقارنة سريعة بين السعر والتحويل للمنتج والبدائل المشابهة.</div>', unsafe_allow_html=True)
            fig = build_context_scatter(scatter, int(product_id))
            if fig is None:
                st.info("لا توجد بيانات كافية للمقارنة المرئية لهذا المنتج.")
            else:
                st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with l2:
            st.markdown('<div class="soft-panel">', unsafe_allow_html=True)
            st.markdown('<div class="section-title" style="font-size:20px">ما الذي يفسر هذا الوضع؟</div>', unsafe_allow_html=True)
            render_info_box("العامل الأبرز", root_cause_label(product.get("context_primary_gap")))
            render_info_box("عامل إضافي", root_cause_label(product.get("context_secondary_gap")))
            render_info_box("قراءة سريعة", safe_text(product.get("contextual_weakness_reason")))
            render_info_box("متوسط سعر البدائل", fmt_currency(product.get("peer_avg_price")))
            st.markdown('</div>', unsafe_allow_html=True)


    with tabs[2]:
        l1, l2 = st.columns([1.05, 0.95])
        with l1:
            st.markdown('<div class="chart-shell">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">فرص ربط يمكن أن ترفع قيمة السلة</div>', unsafe_allow_html=True)
            st.markdown('<div class="chart-sub">شبكة مبسطة توضح أكثر المنتجات التي يمكن ربطها مع هذا المنتج.</div>', unsafe_allow_html=True)
            fig = build_bundle_network_chart(opportunities, int(product_id), product_name)
            if fig is None:
                st.info("لا توجد فرصة ربط واضحة لهذا المنتج ضمن الفترة المحددة.")
            else:
                st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with l2:
            st.markdown('<div class="soft-panel">', unsafe_allow_html=True)
            st.markdown('<div class="section-title" style="font-size:20px">أفضل فرصة حالية</div>', unsafe_allow_html=True)
            render_info_box("هل توجد فرصة؟", "نعم" if bool(product.get("has_bundle_opportunity")) else "لا")
            render_info_box("الشريك المقترح", safe_text(product.get("bundle_partner_name")))
            render_info_box("طريقة الاستفادة", safe_text(product.get("bundle_strategy")))
            render_info_box("لماذا هذه الفرصة؟", safe_text(product.get("bundle_business_explanation")))
            st.markdown('</div>', unsafe_allow_html=True)

    with tabs[3]:
        l1, l2 = st.columns([1.05, 0.95])
        with l1:
            st.markdown('<div class="decision-box">', unsafe_allow_html=True)
            st.markdown('<div class="decision-title">الإجراء المقترح</div>', unsafe_allow_html=True)
            st.markdown(f"<div class='decision-text'><b>الإجراء الرئيسي:</b> {safe_text(product.get('recommended_action_ar'))}</div>", unsafe_allow_html=True)
            if safe_text(product.get("recommended_action_secondary_ar")) != "—":
                st.markdown(f"<div class='decision-text'><b>خطوة إضافية:</b> {safe_text(product.get('recommended_action_secondary_ar'))}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='decision-text'><b>درجة الأولوية:</b> {safe_text(product.get('priority_tier'))} • <b>مستوى الحاجة للتدخل:</b> {safe_text(product.get('urgency_level'))}</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="soft-panel">', unsafe_allow_html=True)
            st.markdown('<div class="section-title" style="font-size:20px">تجربة سعر مختلفة</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-sub">استخدميها لتقدير ما إذا كان السعر الجديد يقرّب المنتج من البدائل القريبة أو يبعده عنها.</div>', unsafe_allow_html=True)
            new_price = st.number_input("السعر المقترح", min_value=0.0, value=max(to_float(product.get("price")), 0.0), step=1.0)
            if st.button("اعرض أثر السعر المقترح", use_container_width=True):
                scenario = simulate_price_scenario(product, new_price=new_price)
                st.success(safe_text(scenario.get("scenario_note")))
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("السعر الحالي", fmt_currency(scenario.get("current_price")))
                with c2:
                    st.metric("السعر المقترح", fmt_currency(scenario.get("new_price")))
                with c3:
                    st.metric("متوسط البدائل", fmt_currency(scenario.get("peer_avg_price")))
            st.markdown('</div>', unsafe_allow_html=True)

        with l2:
            st.markdown('<div class="soft-panel">', unsafe_allow_html=True)
            st.markdown('<div class="section-title" style="font-size:20px">ملخص القرار</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-sub">يمكن تجهيز ملخص أوضح لهذا القرار عند الحاجة، مع الالتزام الكامل بنفس التوصية دون تغيير معناها.</div>', unsafe_allow_html=True)

            existing_genai = data.get("genai_output", pd.DataFrame())
            genai_row = find_product_row(existing_genai, int(product_id)) if not existing_genai.empty else None

            if st.button("حضّر ملخصًا أوضح", type="primary", use_container_width=True):
                with st.spinner("جاري تجهيز الملخص..."):
                    result = generate_one_explanation(product, use_cache=True, output_dir=config.LAYER5_DIR)
                st.success("تم تجهيز الملخص بنجاح.")
                st.session_state.data_refresh_key = datetime.now().isoformat()
                st.cache_data.clear()
                genai_row = pd.Series(result)

            if genai_row is None:
                st.info("لا يوجد ملخص محفوظ لهذا المنتج بعد.")
            else:
                row = genai_row if isinstance(genai_row, pd.Series) else genai_row.iloc[0]
                render_info_box("خلاصة مختصرة", safe_text(row.get("executive_summary_ar_short") or row.get("executive_summary_ar")))
                render_info_box("سبب التوصية", safe_text(row.get("reasoning_ar")))
                render_info_box("الأثر المتوقع", safe_text(row.get("expected_impact_ar")))
                render_info_box("تنبيه مهم", safe_text(row.get("risk_note_ar")))
            st.markdown('</div>', unsafe_allow_html=True)


def main() -> None:
    data = load_app_data(st.session_state.data_refresh_key)
    if st.session_state.page == "home":
        page_home(data)
    elif st.session_state.page == "dashboard":
        page_dashboard(data)
    elif st.session_state.page == "details":
        page_details(data)
    else:
        st.session_state.page = "home"
        page_home(data)


if __name__ == "__main__":
    main()
