import pandas as pd
import numpy as np


class Resampler:
    """
    Resamples the daily enriched dataset to a target frequency (D, W, M).

    - Quantity is summed over the period.
    - Calendar features are recalculated for the period start date.
    - Lag and rolling features are recomputed at the new frequency.

    Usage:
        resampler = Resampler(freq='W')
        df_weekly = resampler.resample(df_daily)
    """

    VALID_FREQS = ['D', 'W', 'M']

    # Pandas >= 2.2 uses 'ME' instead of 'M' for month-end
    FREQ_MAP = {'D': 'D', 'W': 'W', 'M': 'ME'}

    # Lag and rolling windows (in periods, not days)
    LAG_PERIODS = [1, 2, 4]
    ROLLING_WINDOWS = [2, 4]

    def __init__(self, freq: str = 'D'):
        """
        Args:
            freq: Target frequency. 'D' (daily), 'W' (weekly), 'M' (monthly).
        """
        if freq not in self.VALID_FREQS:
            raise ValueError(f"freq must be one of {self.VALID_FREQS}, got '{freq}'")
        self.freq = freq

    def resample(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample the dataset to the target frequency.

        Args:
            df: Daily enriched DataFrame with at least:
                date, item_number, item_description, quantity, category, brand, sub_category

        Returns:
            Resampled DataFrame with recalculated features.
        """
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'], utc=True)

        # If daily, just return as-is (features already computed)
        if self.freq == 'D':
            return df

        # ── Aggregate to target frequency ──
        df_agg = self._aggregate(df)

        # ── Recalculate calendar features ──
        df_agg = self._add_calendar_features(df_agg)

        # ── Recalculate lag features ──
        df_agg = self._add_lag_features(df_agg)

        # ── Recalculate rolling features ──
        df_agg = self._add_rolling_features(df_agg)

        # ── Sort and reset index ──
        df_agg = df_agg.sort_values(['item_number', 'date']).reset_index(drop=True)

        return df_agg

    def _aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate quantity to target frequency per item."""
        # Use period start as the date label
        grouper = pd.Grouper(key='date', freq=self.FREQ_MAP[self.freq])

        df_agg = (
            df.groupby([grouper, 'item_number'])
            .agg(
                quantity=('quantity', 'sum'),
                item_description=('item_description', 'first'),
                category=('category', 'first'),
                brand=('brand', 'first'),
                sub_category=('sub_category', 'first'),
                is_holiday=('is_holiday', 'max'),           # 1 if any day in period is holiday
                is_pre_holiday=('is_pre_holiday', 'max'),
                is_post_holiday=('is_post_holiday', 'max'),
            )
            .reset_index()
        )

        return df_agg

    def _add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Recalculate calendar features for the period start date."""
        d = df['date']

        df['year'] = d.dt.year
        df['month'] = d.dt.month
        df['quarter'] = d.dt.quarter

        if self.freq == 'W':
            df['week_of_year'] = d.dt.isocalendar().week.astype(int)

        # Cyclical encoding
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        return df

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lag features at the resampled frequency."""
        for lag in self.LAG_PERIODS:
            df[f'lag_{lag}p'] = (
                df.groupby('item_number')['quantity']
                .shift(lag)
            )
        return df

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling mean and std at the resampled frequency."""
        for window in self.ROLLING_WINDOWS:
            grp = df.groupby('item_number')['quantity']

            df[f'roll_mean_{window}p'] = grp.transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
            df[f'roll_std_{window}p'] = grp.transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).std()
            )
        return df
