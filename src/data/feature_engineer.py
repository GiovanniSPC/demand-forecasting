import pandas as pd
import numpy as np
import os
from typing import Optional


class FeatureEngineer:
    """
    Enriches the cleaned daily transaction dataset with features
    useful for forecasting and analysis.

    Features added:
        - Calendar features (year, month, day, weekday, weekend, etc.)
        - Holiday flags (from holidays file)
        - Category hierarchy (brand, sub_category)
        - Lag features (1, 7, 14, 28 days)
        - Rolling statistics (7, 14, 28 day mean/std/min/max)
        - Expanding mean (cumulative average)
        - Days since last sale
        - Quantity flags (is_zero, is_above_mean)

    Usage:
        fe = FeatureEngineer(df_clean, holidays_path="holidays_2018_2025.csv")
        df_enriched = fe.transform()
    """

    # Lag windows in days
    LAG_DAYS = [1, 7, 14, 28]

    # Rolling windows in days
    ROLLING_WINDOWS = [7, 14, 28]

    def __init__(self, df: pd.DataFrame, holidays_path: Optional[str] = None):
        """
        Args:
            df:             Cleaned DataFrame with daily frequency.
            holidays_path:  Path to the holidays CSV file (optional).
        """
        self.df = df.copy()
        self.holidays_path = holidays_path
        self.holidays = None

    def _prepare(self):
        """Parse date column and sort data."""
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values(['item_number', 'date']).reset_index(drop=True)

    def _add_calendar_features(self):
        """Extract calendar features from date."""
        d = self.df['date']

        self.df['year'] = d.dt.year
        self.df['month'] = d.dt.month
        self.df['day'] = d.dt.day
        self.df['day_of_week'] = d.dt.dayofweek          # 0=Mon, 6=Sun
        self.df['day_of_year'] = d.dt.dayofyear
        self.df['week_of_year'] = d.dt.isocalendar().week.astype(int)
        self.df['quarter'] = d.dt.quarter
        self.df['is_weekend'] = d.dt.dayofweek.isin([5, 6]).astype(int)
        self.df['is_month_start'] = d.dt.is_month_start.astype(int)
        self.df['is_month_end'] = d.dt.is_month_end.astype(int)

        # Cyclical encoding for month and day_of_week
        self.df['month_sin'] = np.sin(2 * np.pi * self.df['month'] / 12)
        self.df['month_cos'] = np.cos(2 * np.pi * self.df['month'] / 12)
        self.df['dow_sin'] = np.sin(2 * np.pi * self.df['day_of_week'] / 7)
        self.df['dow_cos'] = np.cos(2 * np.pi * self.df['day_of_week'] / 7)

    def _add_holiday_features(self):
        """Add holiday flag and holiday name from holidays file."""
        if self.holidays_path is None:
            self.df['is_holiday'] = 0
            self.df['holiday_name'] = None
            return

        holidays = pd.read_csv(self.holidays_path)
        holidays['Date'] = pd.to_datetime(holidays['Date'])
        self.holidays = holidays

        # Create a lookup set and dict
        holiday_dates = set(holidays['Date'].dt.normalize())
        holiday_names = holidays.set_index(holidays['Date'].dt.normalize())['Holiday Name'].to_dict()

        self.df['is_holiday'] = self.df['date'].dt.normalize().isin(holiday_dates).astype(int)
        self.df['holiday_name'] = self.df['date'].dt.normalize().map(holiday_names)

        # Days before/after nearest holiday
        holiday_list = sorted(holiday_dates)
        date_array = self.df['date'].dt.normalize().values

        # Pre/post holiday flags (1 day before and after)
        holiday_set = set(holidays['Date'].dt.normalize())
        self.df['is_pre_holiday'] = self.df['date'].apply(
            lambda x: int((x + pd.Timedelta(days=1)).normalize() in holiday_set)
        )
        self.df['is_post_holiday'] = self.df['date'].apply(
            lambda x: int((x - pd.Timedelta(days=1)).normalize() in holiday_set)
        )

    def _add_category_features(self):
        """Split hierarchical category into brand and sub_category."""
        cat_split = self.df['category'].str.split('.', expand=True)

        self.df['brand'] = cat_split[1]                  # e.g. SUNQUICK, SUNTOP
        self.df['sub_category'] = cat_split[2]           # e.g. JUICE CONCENTRATE

    def _add_lag_features(self):
        """Add lag features per item (shifted quantities)."""
        for lag in self.LAG_DAYS:
            self.df[f'lag_{lag}d'] = (
                self.df.groupby('item_number')['quantity']
                .shift(lag)
            )

    def _add_rolling_features(self):
        """Add rolling window statistics per item."""
        for window in self.ROLLING_WINDOWS:
            grp = self.df.groupby('item_number')['quantity']

            # shift(1) to avoid data leakage
            rolling = grp.transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1)
            )

            self.df[f'roll_mean_{window}d'] = grp.transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
            self.df[f'roll_std_{window}d'] = grp.transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).std()
            )
            self.df[f'roll_min_{window}d'] = grp.transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).min()
            )
            self.df[f'roll_max_{window}d'] = grp.transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).max()
            )

    def _add_expanding_features(self):
        """Add expanding (cumulative) mean per item."""
        self.df['expanding_mean'] = (
            self.df.groupby('item_number')['quantity']
            .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
        )

    def _add_days_since_last_sale(self):
        """Count days since the last non-zero quantity per item."""
        def _days_since(series):
            result = pd.Series(np.nan, index=series.index)
            last_sale = np.nan
            for i, (idx, val) in enumerate(series.items()):
                if val > 0:
                    if np.isnan(last_sale):
                        result.iloc[i] = 0
                    else:
                        result.iloc[i] = i - last_sale
                    last_sale = i
                else:
                    if not np.isnan(last_sale):
                        result.iloc[i] = i - last_sale
            return result

        self.df['days_since_last_sale'] = (
            self.df.groupby('item_number')['quantity']
            .transform(_days_since)
        )

    def _add_quantity_flags(self):
        """Add binary flags based on quantity."""
        self.df['is_zero'] = (self.df['quantity'] == 0).astype(int)

        # is_above_mean: 1 if quantity > item's historical mean (excluding zeros)
        item_mean = (
            self.df[self.df['quantity'] > 0]
            .groupby('item_number')['quantity']
            .mean()
            .rename('item_mean_qty')
        )
        self.df = self.df.merge(item_mean, on='item_number', how='left')
        self.df['is_above_mean'] = (self.df['quantity'] > self.df['item_mean_qty']).astype(int)
        self.df.drop(columns=['item_mean_qty'], inplace=True)

    def transform(self) -> pd.DataFrame:
        """
        Run the full feature engineering pipeline.

        Returns:
            Enriched DataFrame with all features.
        """
        print("Step 1/9 - Preparing data...")
        self._prepare()

        print("Step 2/9 - Adding calendar features...")
        self._add_calendar_features()

        print("Step 3/9 - Adding holiday features...")
        self._add_holiday_features()

        print("Step 4/9 - Adding category features...")
        self._add_category_features()

        print("Step 5/9 - Adding lag features...")
        self._add_lag_features()

        print("Step 6/9 - Adding rolling features...")
        self._add_rolling_features()

        print("Step 7/9 - Adding expanding mean...")
        self._add_expanding_features()

        print("Step 8/9 - Adding days since last sale...")
        self._add_days_since_last_sale()

        print("Step 9/9 - Adding quantity flags...")
        self._add_quantity_flags()

        print(f"\n✅ Done! Shape: {self.df.shape}")
        print(f"   Features added: {self.df.shape[1]} columns total")

        return self.df

    def save(self, output_path: str):
        """Save the enriched DataFrame to CSV."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.df.to_csv(output_path, index=False)
        print(f"📁 Saved to {output_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    from data_cleaner import DataCleaner
    from data_validator import DataValidator

    # Clean
    cleaner = DataCleaner("/mnt/user-data/uploads/Transaction_Details.xlsx")
    df_clean = cleaner.clean()

    # Validate
    df_raw = cleaner.df_raw.copy()
    df_raw.columns = df_raw.columns.str.strip().str.lower().str.replace(' ', '_')
    validator = DataValidator(df_raw, df_clean)
    validator.validate()

    # Feature engineering
    fe = FeatureEngineer(df_clean, holidays_path="/mnt/project/holidays_2018_2025.csv")
    df_enriched = fe.transform()

    # Preview
    print("\nColumns:")
    print(df_enriched.columns.tolist())
    print("\nSample:")
    print(df_enriched.head())
    print("\nDtypes:")
    print(df_enriched.dtypes)

    # Save
    fe.save("/mnt/user-data/outputs/transaction_details_enriched.csv")
