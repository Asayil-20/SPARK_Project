"""CLI runner for the SPARK backend pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

import config
from src.layer0_data_processing import run_layer0
from src.layer1_diagnostics import run_layer1
from src.layer2_similarity import run_layer2
from src.layer3_bundles import run_layer3
from src.layer4_decision_engine import run_layer4
from src.layer5_genai_explainer import run_layer5_batch
from src.utils import ensure_dir, infer_max_timestamp, parse_date_arg, safe_read_csv, save_json, setup_logging

LOGGER = logging.getLogger("spark.pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SPARK backend pipeline.")
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR, help="Directory containing input CSV files.")
    parser.add_argument("--output-dir", type=Path, default=config.OUTPUT_DIR, help="Base output directory.")
    parser.add_argument("--analysis-days", type=int, default=config.DEFAULT_ANALYSIS_DAYS, help="Fallback analysis window in days.")
    parser.add_argument("--start-date", type=str, default=None, help="Analysis start date in YYYY-MM-DD.")
    parser.add_argument("--end-date", type=str, default=None, help="Analysis end date in YYYY-MM-DD.")
    parser.add_argument("--preset-label", type=str, default=None, help="Optional UI preset label such as Last 7 Days.")
    parser.add_argument("--run-genai-batch", action="store_true", default=config.ENABLE_GENAI_BATCH_DEFAULT, help="Run optional Layer 5 batch generation after Layers 0-4.")
    parser.add_argument("--max-genai-products", type=int, default=10, help="Maximum products to process in optional Layer 5 batch mode.")
    return parser.parse_args()


def resolve_analysis_window(layer0_dir: Path, start_date_raw: str | None, end_date_raw: str | None, analysis_days: int) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    start_date = parse_date_arg(start_date_raw)
    end_date = parse_date_arg(end_date_raw)

    if start_date is not None and end_date is not None:
        if start_date > end_date:
            raise ValueError("start_date cannot be after end_date.")
        return start_date, end_date

    events = safe_read_csv(layer0_dir / config.LAYER0_EVENTS_FILE, required=False)
    orders = safe_read_csv(layer0_dir / config.LAYER0_ORDERS_FILE, required=False)
    reviews = safe_read_csv(layer0_dir / config.LAYER0_REVIEWS_FILE, required=False)
    inferred_end = infer_max_timestamp((events, "event_time"), (orders, "order_date"), (reviews, "review_date"))
    if inferred_end is None:
        return start_date, end_date

    if end_date is None:
        end_date = inferred_end
    if start_date is None and analysis_days:
        start_date = end_date - pd.Timedelta(days=int(analysis_days))
    return start_date, end_date


def main() -> int:
    load_dotenv()
    setup_logging()

    args = parse_args()

    base_output = ensure_dir(args.output_dir)
    layer0_dir = ensure_dir(base_output / "layer0")
    layer1_dir = ensure_dir(base_output / "layer1")
    layer2_dir = ensure_dir(base_output / "layer2")
    layer3_dir = ensure_dir(base_output / "layer3")
    layer4_dir = ensure_dir(base_output / "layer4")
    layer5_dir = ensure_dir(base_output / "layer5")

    try:
        LOGGER.info("Running Layer 0")
        run_layer0(args.data_dir, layer0_dir)

        start_date, end_date = resolve_analysis_window(layer0_dir, args.start_date, args.end_date, args.analysis_days)
        save_json(
            {
                "analysis_start_date": str(start_date.date()) if start_date is not None else None,
                "analysis_end_date": str(end_date.date()) if end_date is not None else None,
                "preset_label": args.preset_label,
                "analysis_days": args.analysis_days,
            },
            base_output / config.RUN_CONTEXT_FILE,
        )

        LOGGER.info("Running Layer 1")
        run_layer1(layer0_dir, layer1_dir, start_date=start_date, end_date=end_date)

        LOGGER.info("Running Layer 2")
        run_layer2(layer0_dir, layer1_dir, layer2_dir)

        LOGGER.info("Running Layer 3")
        run_layer3(layer0_dir, layer1_dir, layer2_dir, layer3_dir, start_date=start_date, end_date=end_date)

        LOGGER.info("Running Layer 4")
        run_layer4(layer1_dir, layer2_dir, layer3_dir, layer4_dir)

        if args.run_genai_batch:
            LOGGER.info("Running optional Layer 5 batch mode")
            run_layer5_batch(layer4_dir, layer5_dir, max_products=args.max_genai_products)
        else:
            LOGGER.info("Skipping Layer 5 batch mode by default")

    except FileNotFoundError as exc:
        LOGGER.error("Pipeline stopped because a critical required file is missing: %s", exc)
        return 1
    except ValueError as exc:
        LOGGER.error("Pipeline stopped because input schema validation failed: %s", exc)
        return 1
    except Exception as exc:
        LOGGER.exception("Pipeline failed unexpectedly: %s", exc)
        return 1

    LOGGER.info("SPARK pipeline completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
