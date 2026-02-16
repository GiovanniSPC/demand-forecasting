# Drinks Demand Forecasting

A modular demand forecasting pipeline for beverage sales in Mauritius. The project processes raw transaction data (2018–2026), engineers features, and supports experimentation across multiple forecasting models and time granularities.

## About

This project aims to forecast daily, weekly, and monthly sales quantities per product. It provides a structured framework to clean raw data, validate integrity, generate predictive features, and compare forecasting models — all evaluated at a consistent monthly level for fair comparison.

## Project Structure

```
project/
├── main.py                          # Pipeline entry point
├── README.md
├── environment.yml                  # Conda environment
├── src/
│   ├── data/
│   │   ├── data_cleaner.py          # Raw data → clean daily dataset
│   │   ├── data_validator.py        # Integrity checks (raw vs clean)
│   │   └── feature_engineer.py      # Temporal, lag, rolling & categorical features
│   ├── resampler/
│   │   └── resampler.py             # Frequency conversion (daily → weekly/monthly)
│   ├── models/
│   │   ├── base_model.py            # Abstract model interface
│   │   └── naive_model.py           # Baseline naive models
│   ├── evaluation/
│   │   └── evaluator.py             # Monthly evaluation metrics
│   └── experiment/
│       └── experiment_runner.py      # Experiment orchestrator
├── data/
│   ├── raw/
│   ├── cleaned/
│   └── features/
└── results/
```

## How It Works

The pipeline runs in four steps:

1. **Data Cleaning** — Loads raw transactions, standardizes columns, aggregates to daily level per product, and fills date gaps with zero quantities to ensure a continuous daily frequency.

2. **Data Validation** — Compares the cleaned dataset against the raw data to verify that item counts, category counts, total quantities, and date continuity are preserved.

3. **Feature Engineering** — Enriches the daily dataset with calendar features, holiday indicators, category hierarchy, lag values, rolling statistics, and activity flags.

4. **Experimentation** — Resamples data to the desired frequency, splits into train/test, fits a model, generates predictions, and evaluates results at the monthly level. All runs are logged for comparison.

## Adding a New Model

Create a new file in `src/models/` that inherits from `BaseModel` and implements three methods: `fit`, `predict`, and `get_name`. Then add it to the experiments list in `main.py`. The framework handles resampling, splitting, and evaluation automatically.

## Evaluation

All models are evaluated at the monthly level regardless of their training granularity. This ensures a fair and consistent comparison. Metrics include MAE, RMSE, MAPE, WAPE, and BIAS.

## Installation

Requires Conda. Create and activate the environment:

```
conda env create -f environment.yml
conda activate drinks-forecast
```

Then run the pipeline:

```
python main.py
```

## Tech Stack

Python 3.11, pandas, NumPy, openpyxl, scikit-learn, matplotlib, seaborn.