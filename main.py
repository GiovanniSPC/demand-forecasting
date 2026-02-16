"""
Main entry point — Full pipeline:
    1. Clean raw data
    2. Validate
    3. Run experiments (Naive models at D, W, M frequencies)
    4. Compare results (always evaluated monthly)
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.data_cleaner import DataCleaner
from src.data.data_validator import DataValidator
from src.models.naive_model import NaiveModel
from src.experiment.experiment_runner import ExperimentRunner


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
RAW_FILE = "./datasets/Transaction_Details.xlsx"
CUTOFF_DATE = "2025-01-01"
RESULTS_DIR = "results"


def main():
    # ── Step 1: Clean ──
    print("\n" + "=" * 60)
    print("  STEP 1: DATA CLEANING")
    print("=" * 60)
    cleaner = DataCleaner(RAW_FILE)
    df_clean = cleaner.clean()

    # ── Step 2: Validate ──
    print("\n" + "=" * 60)
    print("  STEP 2: DATA VALIDATION")
    print("=" * 60)
    df_raw = cleaner.df_raw.copy()
    df_raw.columns = df_raw.columns.str.strip().str.lower().str.replace(' ', '_')
    validator = DataValidator(df_raw, df_clean)
    if not validator.validate():
        print("❌ Validation failed. Stopping pipeline.")
        return

    # ── Step 3: Run experiments ──
    print("\n" + "=" * 60)
    print("  STEP 3: EXPERIMENTS")
    print("=" * 60)

    experiments = [
        # Naive Last — at all frequencies
        {'model': NaiveModel(strategy='last'),            'freq': 'D'},
        {'model': NaiveModel(strategy='last'),            'freq': 'W'},
        {'model': NaiveModel(strategy='last'),            'freq': 'M'},

        # Naive Rolling Mean — at all frequencies
        {'model': NaiveModel(strategy='rolling_mean', window=7),  'freq': 'D'},
        {'model': NaiveModel(strategy='rolling_mean', window=4),  'freq': 'W'},
        {'model': NaiveModel(strategy='rolling_mean', window=3),  'freq': 'M'},

        # Naive Same Period — daily only
        {'model': NaiveModel(strategy='same_period'),     'freq': 'D'},
    ]

    runner = ExperimentRunner(
        df=df_clean,
        cutoff_date=CUTOFF_DATE,
        experiments=experiments,
        results_dir=RESULTS_DIR,
    )
    results = runner.run_all()

    # ── Done ──
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nBest model by WAPE:")
    best = results.loc[results['WAPE'].idxmin()]
    print(f"  {best['model']} @ {best['freq']} — WAPE: {best['WAPE']}%")


if __name__ == "__main__":
    main()