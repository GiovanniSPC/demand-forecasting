import pandas as pd
import os
from datetime import datetime
from typing import List, Dict

from src.resampler.resampler import Resampler
from src.models.base_model import BaseModel
from src.evaluation.evaluator import Evaluator


class ExperimentRunner:
    """
    Orchestrates experiments across multiple models and granularities.

    Workflow per experiment:
        1. Split data into train/test by cutoff date.
        2. Resample train and test to target frequency.
        3. Fit the model on train.
        4. Predict on test.
        5. Evaluate at the monthly level.
        6. Log results.

    Usage:
        runner = ExperimentRunner(
            df=df_enriched,
            cutoff_date='2025-07-01',
            experiments=[
                {'model': NaiveModel('last'),          'freq': 'D'},
                {'model': NaiveModel('rolling_mean'),  'freq': 'W'},
            ]
        )
        results = runner.run_all()
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cutoff_date: str,
        experiments: List[Dict],
        results_dir: str = 'results'
    ):
        """
        Args:
            df:           Enriched daily DataFrame.
            cutoff_date:  Train/test split date (ISO format). Test starts from this date.
            experiments:  List of dicts with keys 'model' (BaseModel) and 'freq' (str).
            results_dir:  Directory to save experiment logs.
        """
        self.df = df.copy()
        self.df['date'] = pd.to_datetime(self.df['date'], utc=True)
        self.cutoff_date = pd.Timestamp(cutoff_date, tz='UTC')
        self.experiments = experiments
        self.results_dir = results_dir
        self.results = []

    def _split(self, df: pd.DataFrame):
        """Split into train and test based on cutoff date."""
        train = df[df['date'] < self.cutoff_date].copy()
        test = df[df['date'] >= self.cutoff_date].copy()
        return train, test

    def run_single(self, model: BaseModel, freq: str) -> Dict:
        """
        Run a single experiment.

        Args:
            model: A BaseModel instance.
            freq:  Frequency ('D', 'W', 'M').

        Returns:
            Dict with model name, freq, and metrics.
        """
        print(f"\n{'─' * 50}")
        print(f"  Running: {model.get_name()} @ {freq}")
        print(f"{'─' * 50}")

        # Resample
        resampler = Resampler(freq=freq)
        df_resampled = resampler.resample(self.df)

        # Split
        train, test = self._split(df_resampled)
        print(f"  Train: {len(train):,} rows | {train['date'].min().date()} → {train['date'].max().date()}")
        print(f"  Test:  {len(test):,} rows | {test['date'].min().date()} → {test['date'].max().date()}")

        # Fit
        model.fit(train)

        # Predict
        predictions = model.predict(test)

        # Evaluate (always monthly)
        evaluator = Evaluator()

        # Prepare actuals from test set
        actuals = test[['date', 'item_number', 'quantity']].copy()

        overall = evaluator.evaluate(actuals, predictions, level='overall')
        evaluator.summary()

        # Build result entry
        result = {
            'model': model.get_name(),
            'freq': freq,
            'train_rows': len(train),
            'test_rows': len(test),
            'train_start': str(train['date'].min().date()),
            'train_end': str(train['date'].max().date()),
            'test_start': str(test['date'].min().date()),
            'test_end': str(test['date'].max().date()),
        }
        result.update(overall.iloc[0].to_dict())

        print(f"\n  Results: MAE={result['MAE']}, RMSE={result['RMSE']}, "
              f"MAPE={result['MAPE']}%, WAPE={result['WAPE']}%")

        return result

    def run_all(self) -> pd.DataFrame:
        """
        Run all experiments and return a comparison table.

        Returns:
            DataFrame with one row per experiment and metrics columns.
        """
        print("=" * 60)
        print(f"  EXPERIMENT RUNNER — {len(self.experiments)} experiment(s)")
        print(f"  Cutoff date: {self.cutoff_date.date()}")
        print("=" * 60)

        self.results = []
        for exp in self.experiments:
            result = self.run_single(exp['model'], exp['freq'])
            self.results.append(result)

        results_df = pd.DataFrame(self.results)

        # Print comparison table
        print("\n" + "=" * 60)
        print("  COMPARISON TABLE")
        print("=" * 60)
        display_cols = ['model', 'freq', 'MAE', 'RMSE', 'MAPE', 'WAPE', 'BIAS']
        print(results_df[display_cols].to_string(index=False))

        # Save to CSV
        self._save_results(results_df)

        return results_df

    def _save_results(self, results_df: pd.DataFrame):
        """Append results to the experiment log CSV."""
        os.makedirs(self.results_dir, exist_ok=True)
        log_path = os.path.join(self.results_dir, 'experiments_log.csv')

        results_df['run_timestamp'] = datetime.now().isoformat()

        if os.path.exists(log_path):
            existing = pd.read_csv(log_path)
            combined = pd.concat([existing, results_df], ignore_index=True)
        else:
            combined = results_df

        combined.to_csv(log_path, index=False)
        print(f"\n📁 Results saved to {log_path}")
