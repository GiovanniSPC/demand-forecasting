import pandas as pd
import numpy as np
from typing import Dict, Optional


class Evaluator:
    """
    Evaluates forecasting predictions at the MONTHLY level,
    regardless of the input granularity (D, W, M).

    Predictions and actuals are aggregated to monthly totals per item
    before computing metrics.

    Metrics:
        - MAE:  Mean Absolute Error
        - RMSE: Root Mean Squared Error
        - MAPE: Mean Absolute Percentage Error (excluding zero actuals)
        - WAPE: Weighted Absolute Percentage Error (total |error| / total actual)
        - BIAS: Mean bias (positive = over-forecast, negative = under-forecast)

    Usage:
        evaluator = Evaluator()
        metrics = evaluator.evaluate(actuals_df, predictions_df)
    """

    def __init__(self):
        self.monthly_comparison = None

    def _aggregate_monthly(self, df: pd.DataFrame, value_col: str) -> pd.DataFrame:
        """
        Aggregate a DataFrame to monthly totals per item.

        Args:
            df:         DataFrame with date, item_number, and a value column.
            value_col:  Name of the column to sum.

        Returns:
            DataFrame with columns: year_month, item_number, {value_col}
        """
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'], utc=True)
        df['year_month'] = df['date'].dt.to_period('M')

        monthly = (
            df.groupby(['year_month', 'item_number'])[value_col]
            .sum()
            .reset_index()
        )
        return monthly

    def _compute_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute all metrics from the monthly comparison DataFrame.

        Args:
            df: DataFrame with columns: actual, predicted

        Returns:
            Dictionary of metric_name -> value
        """
        actual = df['actual'].values
        predicted = df['predicted'].values
        errors = predicted - actual

        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors ** 2))
        bias = np.mean(errors)

        # MAPE: only where actual > 0
        mask = actual > 0
        if mask.sum() > 0:
            mape = np.mean(np.abs(errors[mask]) / actual[mask]) * 100
        else:
            mape = np.nan

        # WAPE: total absolute error / total actual
        total_actual = np.sum(np.abs(actual))
        if total_actual > 0:
            wape = np.sum(np.abs(errors)) / total_actual * 100
        else:
            wape = np.nan

        return {
            'MAE': round(mae, 2),
            'RMSE': round(rmse, 2),
            'MAPE': round(mape, 2),
            'WAPE': round(wape, 2),
            'BIAS': round(bias, 2),
        }

    def evaluate(
        self,
        actuals_df: pd.DataFrame,
        predictions_df: pd.DataFrame,
        level: str = 'overall'
    ) -> pd.DataFrame:
        """
        Evaluate predictions against actuals at the monthly level.

        Args:
            actuals_df:     DataFrame with date, item_number, quantity.
            predictions_df: DataFrame with date, item_number, quantity_pred.
            level:          'overall' for global metrics,
                            'per_item' for per-item metrics,
                            'per_month' for per-month metrics.

        Returns:
            DataFrame with metrics.
        """
        # Aggregate both to monthly
        actual_monthly = self._aggregate_monthly(actuals_df, 'quantity')
        pred_monthly = self._aggregate_monthly(predictions_df, 'quantity_pred')

        # Merge
        merged = actual_monthly.merge(
            pred_monthly,
            on=['year_month', 'item_number'],
            how='left'
        )
        merged['quantity_pred'] = merged['quantity_pred'].fillna(0)
        merged = merged.rename(columns={'quantity': 'actual', 'quantity_pred': 'predicted'})

        self.monthly_comparison = merged

        # Compute metrics at requested level
        if level == 'overall':
            metrics = self._compute_metrics(merged)
            return pd.DataFrame([metrics])

        elif level == 'per_item':
            results = []
            for item, group in merged.groupby('item_number'):
                m = self._compute_metrics(group)
                m['item_number'] = item
                results.append(m)
            return pd.DataFrame(results).set_index('item_number')

        elif level == 'per_month':
            results = []
            for month, group in merged.groupby('year_month'):
                m = self._compute_metrics(group)
                m['year_month'] = str(month)
                results.append(m)
            return pd.DataFrame(results).set_index('year_month')

        else:
            raise ValueError(f"level must be 'overall', 'per_item', or 'per_month', got '{level}'")

    def summary(self) -> None:
        """Print a summary of the monthly comparison."""
        if self.monthly_comparison is None:
            print("No evaluation run yet. Call evaluate() first.")
            return

        df = self.monthly_comparison
        print(f"Monthly comparison: {len(df)} rows")
        print(f"  Months:  {df['year_month'].nunique()}")
        print(f"  Items:   {df['item_number'].nunique()}")
        print(f"  Total actual:    {df['actual'].sum():,.0f}")
        print(f"  Total predicted: {df['predicted'].sum():,.0f}")
