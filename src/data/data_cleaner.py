import pandas as pd
import numpy as np
from typing import Optional


class DataCleaner:
    """
    Cleans the raw Transaction_Details dataset and returns a daily-frequency
    DataFrame with missing dates filled (quantity = 0).

    Output columns:
        transaction_number, date, item_number, item_description, quantity, category

    Usage:
        cleaner = DataCleaner("Transaction_Details.xlsx")
        df_clean = cleaner.clean()
    """

    def __init__(self, filepath: str):
        """
        Load and prepare the raw dataset.

        Args:
            filepath: Path to the raw Excel file.
        """
        self.filepath = filepath
        self.df_raw = None
        self.df_clean = None

    def _load(self) -> pd.DataFrame:
        """Load the raw Excel file and standardize column names."""
        df = pd.read_excel(self.filepath)
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        return df

    def _parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert transaction_date to datetime and extract date only."""
        df['date'] = pd.to_datetime(df['transaction_date']).dt.normalize()
        df = df.drop(columns=['transaction_date'])
        return df

    def _aggregate_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate transactions at the daily level per item.

        Multiple transactions can occur on the same day for the same item.
        We sum the quantities and keep the most frequent description.
        Transaction numbers are concatenated (comma-separated).
        """
        # Build a mapping: item_number -> most frequent item_description
        desc_map = (
            df.groupby('item_number')['item_description']
            .agg(lambda x: x.value_counts().index[0])
        )

        # Build a mapping: item_number -> category (1:1 relationship)
        cat_map = df.groupby('item_number')['category'].first()

        # Aggregate daily: sum quantity, concat transaction numbers
        df_agg = (
            df.groupby(['date', 'item_number'])
            .agg(
                quantity=('quantity', 'sum'),
                transaction_number=('transaction_number', lambda x: ','.join(x.astype(str).unique()))
            )
            .reset_index()
        )

        # Map back description and category
        df_agg['item_description'] = df_agg['item_number'].map(desc_map)
        df_agg['category'] = df_agg['item_number'].map(cat_map)

        return df_agg

    def _fill_missing_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill gaps in the daily frequency.

        For each item_number, create a row for every day in the full date range.
        Missing dates get quantity = 0 and transaction_number = NaN.
        """
        # Full date range
        date_range = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='D')

        # All unique items with their description and category
        items = df[['item_number', 'item_description', 'category']].drop_duplicates('item_number')

        # Create full skeleton: every date × every item
        skeleton = pd.MultiIndex.from_product(
            [date_range, items['item_number']],
            names=['date', 'item_number']
        ).to_frame(index=False)

        # Merge with actual data
        df_full = skeleton.merge(
            df[['date', 'item_number', 'quantity', 'transaction_number']],
            on=['date', 'item_number'],
            how='left'
        )

        # Fill missing quantities with 0
        df_full['quantity'] = df_full['quantity'].fillna(0).astype(int)

        # Map back item_description and category from items lookup
        item_lookup = items.set_index('item_number')
        df_full['item_description'] = df_full['item_number'].map(item_lookup['item_description'])
        df_full['category'] = df_full['item_number'].map(item_lookup['category'])

        return df_full

    def _reorder_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reorder and select final columns."""
        return df[['transaction_number', 'date', 'item_number', 'item_description', 'quantity', 'category']]

    def clean(self) -> pd.DataFrame:
        """
        Run the full cleaning pipeline:
            1. Load raw data
            2. Parse dates
            3. Aggregate daily per item
            4. Fill missing dates with quantity = 0
            5. Reorder columns

        Returns:
            A clean DataFrame with daily frequency and no date gaps.
        """
        print("Step 1/5 - Loading raw data...")
        df = self._load()
        self.df_raw = df.copy()

        print("Step 2/5 - Parsing dates...")
        df = self._parse_dates(df)

        print("Step 3/5 - Aggregating daily per item...")
        df = self._aggregate_daily(df)

        print("Step 4/5 - Filling missing dates (quantity = 0)...")
        df = self._fill_missing_dates(df)

        print("Step 5/5 - Reordering columns...")
        df = self._reorder_columns(df)

        # Sort by date and item
        df = df.sort_values(['date', 'item_number']).reset_index(drop=True)

        self.df_clean = df
        print(f"\n✅ Done! Shape: {df.shape}")
        print(f"   Date range: {df['date'].min().date()} → {df['date'].max().date()}")
        print(f"   Items: {df['item_number'].nunique()}")
        print(f"   Days: {df['date'].nunique()}")

        return df


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    cleaner = DataCleaner("/mnt/user-data/uploads/Transaction_Details.xlsx")
    df_clean = cleaner.clean()

    print("\nSample output:")
    print(df_clean.head(10))
    print("\nInfo:")
    print(df_clean.dtypes)
    print(f"\nRows with quantity = 0: {(df_clean['quantity'] == 0).sum():,}")
    print(f"Rows with quantity > 0: {(df_clean['quantity'] > 0).sum():,}")

    # Save to csv
    output_path = "/mnt/user-data/outputs/transaction_details_cleaned.csv"
    df_clean.to_csv(output_path, index=False)
    print(f"\n📁 Saved to {output_path}")
