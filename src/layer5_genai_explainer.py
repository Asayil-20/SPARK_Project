"""Layer 5: on-demand GenAI Arabic explanation engine."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

import config
from src.utils import ensure_dir, load_json, safe_read_csv, safe_write_csv, save_json, setup_logging

LOGGER = logging.getLogger("spark.layer5")

REQUIRED_RESPONSE_FIELDS = [
    "executive_summary_ar_short",
    "executive_summary_ar",
    "reasoning_ar",
    "recommended_action_ar",
    "expected_impact_ar",
    "risk_note_ar",
    "confidence_level",
]


def get_genai_client():
    load_dotenv()
    api_key = os.getenv(config.GENAI_API_KEY_ENV_NAME)
    if not api_key:
        raise EnvironmentError(
            f"Missing GenAI API key. Set environment variable {config.GENAI_API_KEY_ENV_NAME}."
        )
    try:
        from groq import Groq
    except ImportError as exc:
        raise ImportError("groq package is not installed. Add it to requirements and install dependencies.") from exc
    return Groq(api_key=api_key)


def _product_row_to_dict(product_row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    if isinstance(product_row, pd.Series):
        return {k: (None if pd.isna(v) else v) for k, v in product_row.to_dict().items()}
    return {k: (None if pd.isna(v) else v) for k, v in dict(product_row).items()}


def _recommendation_hash(product_data: dict[str, Any]) -> str:
    payload = {
        "product_id": product_data.get("product_id"),
        "root_cause": product_data.get("root_cause"),
        "priority_tier": product_data.get("priority_tier"),
        "recommended_action": product_data.get("recommended_action"),
        "recommended_action_secondary": product_data.get("recommended_action_secondary"),
        "priority_score": product_data.get("priority_score"),
        "confidence_level": product_data.get("confidence_level"),
        "bundle_partner_product_id": product_data.get("bundle_partner_product_id"),
        "bundle_strategy": product_data.get("bundle_strategy"),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def build_genai_prompt(product_row: pd.Series | dict[str, Any]) -> str:
    product = _product_row_to_dict(product_row)
    facts_json = json.dumps(product, ensure_ascii=False, indent=2, default=str)

    return f"""
You are SPARK Layer 5, an Arabic executive explanation engine.

Critical rule:
You are only explaining the existing recommendation.
Do not create a new business decision.
Do not invent metrics, percentages, numbers, causes, risks, or actions.
Do not change root_cause, priority_tier, confidence_level, or recommended_action.
Keep the answer grounded only in the supplied facts.

Return valid JSON only with these exact fields:
{{
  "executive_summary_ar_short": "...",
  "executive_summary_ar": "...",
  "reasoning_ar": "...",
  "recommended_action_ar": "...",
  "expected_impact_ar": "...",
  "risk_note_ar": "...",
  "confidence_level": "High or Medium or Low"
}}

Facts to preserve:
{facts_json}

Instructions:
- Write in professional business Arabic.
- executive_summary_ar_short should be concise and suitable for a small UI card.
- Mention only facts present in the input.
- recommended_action_ar must stay semantically aligned with the provided recommended_action_ar if available.
- If data is limited, explicitly say that confidence is limited.
- Do not wrap the JSON in markdown fences.
""".strip()


def extract_json_from_response(response_text: str) -> dict[str, Any]:
    if not response_text:
        raise ValueError("Empty GenAI response.")
    cleaned = response_text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def validate_genai_response(parsed_json: dict[str, Any], product_row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    if not isinstance(parsed_json, dict):
        raise ValueError("GenAI response must be a JSON object.")
    for field in REQUIRED_RESPONSE_FIELDS:
        if field not in parsed_json or not str(parsed_json[field]).strip():
            raise ValueError(f"GenAI response missing required field: {field}")

    product = _product_row_to_dict(product_row)
    if parsed_json["confidence_level"] not in {"High", "Medium", "Low"}:
        parsed_json["confidence_level"] = str(product.get("confidence_level", "Medium"))

    if product.get("recommended_action_ar") and len(str(parsed_json["recommended_action_ar"]).strip()) < 10:
        parsed_json["recommended_action_ar"] = str(product["recommended_action_ar"]).strip()

    for field in REQUIRED_RESPONSE_FIELDS:
        parsed_json[field] = str(parsed_json[field]).strip()
    return parsed_json


def fallback_arabic_explanation(product_row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    product = _product_row_to_dict(product_row)
    name = product.get("name") or product.get("display_name") or product.get("product_id") or "المنتج"
    root_cause = product.get("root_cause", "mixed_context")
    priority = product.get("priority_tier", "Medium")
    confidence = str(product.get("confidence_level", "Medium"))
    reason = product.get("root_cause_details") or product.get("diagnostic_reason") or "توجد مؤشرات اداء تحتاج إلى تحسين."
    action_ar = product.get("recommended_action_ar") or "نفذ توصية التحسين المحددة في الطبقة التحليلية."
    return {
        "executive_summary_ar_short": f"{name}: اولوية {priority} بسبب {root_cause}.",
        "executive_summary_ar": f"المنتج {name} مصنف باولوية {priority} ويرتبط سبب التحسين الرئيسي بمحور {root_cause}.",
        "reasoning_ar": f"يعتمد هذا التفسير على مخرجات الطبقة الرابعة فقط. السبب التفصيلي الحالي هو: {reason}.",
        "recommended_action_ar": str(action_ar),
        "expected_impact_ar": "من المتوقع أن يساعد تنفيذ هذه الخطوة على تحسين أداء المنتج ضمن نفس السياق التجاري دون تغيير قرار النظام الأصلي.",
        "risk_note_ar": "يجب قراءة هذا التفسير مع مستوى الثقة الحالي وعدم افتراض أرقام غير موجودة في البيانات.",
        "confidence_level": confidence if confidence in {"High", "Medium", "Low"} else "Medium",
    }


def load_cache(output_dir: Path | None = None) -> dict[str, Any]:
    cache_path = (output_dir or config.LAYER5_DIR) / config.LAYER5_CACHE_FILE
    return load_json(cache_path)


def save_cache(cache: dict[str, Any], output_dir: Path | None = None) -> Path:
    cache_path = (output_dir or config.LAYER5_DIR) / config.LAYER5_CACHE_FILE
    return save_json(cache, cache_path)


def _upsert_recommendation_row(result: dict[str, Any], output_dir: Path) -> None:
    csv_path = output_dir / config.LAYER5_RECOMMENDATIONS_FILE
    existing = safe_read_csv(csv_path, required=False)
    new_row = pd.DataFrame([result])

    if existing.empty:
        safe_write_csv(new_row, csv_path)
        return

    if "cache_key" in existing.columns and result["cache_key"] in existing["cache_key"].astype(str).tolist():
        existing = existing[existing["cache_key"].astype(str) != result["cache_key"]].copy()

    combined = pd.concat([existing, new_row], ignore_index=True)
    safe_write_csv(combined, csv_path)


def generate_one_explanation(
    product_row: pd.Series | dict[str, Any],
    use_cache: bool = True,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    setup_logging()
    output_dir = ensure_dir(output_dir or config.LAYER5_DIR)

    product = _product_row_to_dict(product_row)
    cache_key = f"{product.get('product_id', 'unknown')}::{_recommendation_hash(product)}"
    cache = load_cache(output_dir)

    if use_cache and cache_key in cache:
        return cache[cache_key]

    prompt = build_genai_prompt(product)
    used_fallback = False
    try:
        client = get_genai_client()
        response = client.chat.completions.create(
            model=config.GENAI_MODEL,
            temperature=config.GENAI_TEMPERATURE,
            max_tokens=config.GENAI_MAX_TOKENS,
            messages=[
                {"role": "system", "content": "You explain existing recommendations only."},
                {"role": "user", "content": prompt},
            ],
        )
        response_text = response.choices[0].message.content if response and response.choices else ""
        parsed = extract_json_from_response(response_text)
        validated = validate_genai_response(parsed, product)
    except Exception as exc:
        LOGGER.warning("GenAI generation failed for product %s: %s", product.get("product_id"), exc)
        validated = fallback_arabic_explanation(product)
        used_fallback = True

    # ── Layer 5 output: حقول جديدة فقط — لا تكرار لأعمدة layer4 ──────────────
    # نحتفظ بـ product_id كمفتاح ربط، وكل ما عدا ذلك يجب أن يكون مُضافاً جديداً.
    # الأعمدة مثل recommended_action وroot_cause موجودة بالفعل في layer4
    # ولا يجب تكرارها هنا لمنع Data Redundancy في الـ DataFrame النهائي.
    result = {
        "product_id":             product.get("product_id"),
        "cache_key":              cache_key,
        "recommendation_hash":    _recommendation_hash(product),
        "used_fallback":          used_fallback,
        # الحقول المُولَّدة حصراً بواسطة GenAI (لا تُوجد في layer4)
        "executive_summary_ar_short": validated.get("executive_summary_ar_short", ""),
        "executive_summary_ar":       validated.get("executive_summary_ar", ""),
        "reasoning_ar":               validated.get("reasoning_ar", ""),
        "recommended_action_ar_genai": validated.get("recommended_action_ar", ""),  # renamed لتمييزه
        "expected_impact_ar":         validated.get("expected_impact_ar", ""),
        "risk_note_ar":               validated.get("risk_note_ar", ""),
        "genai_confidence_level":     validated.get("confidence_level", ""),
    }

    cache[cache_key] = result
    save_cache(cache, output_dir)
    _upsert_recommendation_row(result, output_dir)
    return result


def run_layer5_batch(layer4_dir: Path, output_dir: Path, max_products: int = 10) -> pd.DataFrame:
    setup_logging()
    ensure_dir(output_dir)
    LOGGER.info("Starting optional Layer 5 batch mode for up to %s products", max_products)

    candidates = safe_read_csv(layer4_dir / config.LAYER4_GENAI_INPUT_FILE, required=False)
    if candidates.empty:
        candidates = safe_read_csv(layer4_dir / config.LAYER4_HIGH_PRIORITY_FILE, required=False)
    if candidates.empty:
        candidates = safe_read_csv(layer4_dir / config.LAYER4_RECOMMENDATIONS_FILE, required=True)

    candidates = candidates.head(max_products).copy()
    results = [generate_one_explanation(row, use_cache=True, output_dir=output_dir) for _, row in candidates.iterrows()]
    result_df = pd.DataFrame(results)
    safe_write_csv(result_df, output_dir / config.LAYER5_RECOMMENDATIONS_FILE)

    save_json(
        {
            "requested_products": int(max_products),
            "processed_products": int(len(result_df)),
            "fallback_count": int(result_df["used_fallback"].fillna(False).sum()) if not result_df.empty else 0,
        },
        output_dir / config.LAYER5_SUMMARY_FILE,
    )
    LOGGER.info("Layer 5 batch mode finished")
    return result_df
