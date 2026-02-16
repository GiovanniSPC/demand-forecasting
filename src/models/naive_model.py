import pandas as pd
from src.models.base_model import BaseModel


class NaiveModel(BaseModel):
    """
    Naive forecasting model.

    Strategies:
        - 'last':          Use the last known quantity as prediction.
        - 'same_period':   Use the same period from last year.
        - 'rolling_mean':  Use the rolling mean of the last N periods.

    Usage:
        model = NaiveModel(strategy='last')
        model.fit(train_df)
        preds = model.predict(test_df)
    """

    VALID_STRATEGIES = ['last', 'same_period', 'rolling_mean']

    def __init__(self, strategy: str = 'last', window: int = 7):
        """
        Args:
            strategy: Naive strategy to use.
            window:   Rolling window size (only used for 'rolling_mean').
        """
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(f"strategy must be one of {self.VALID_STRATEGIES}, got '{strategy}'")

        self.strategy = strategy
        self.window = window
        self.last_values = None
        self.rolling_values = None
        self.historical = None

    def fit(self, train_df: pd.DataFrame):
        """
        Learn from training data.

        For 'last': store the last quantity per item.
        For 'rolling_mean': store the rolling mean per item.
        For 'same_period': store full history for date matching.
        """
        df = train_df.copy()
        df['date'] = pd.to_datetime(df['date'], utc=True)
        df = df.sort_values(['item_number', 'date'])

        if self.strategy == 'last':
            # Last known quantity per item
            self.last_values = (
                df.groupby('item_number')['quantity']
                .last()
                .to_dict()
            )

        elif self.strategy == 'rolling_mean':
            # Mean of last N periods per item
            self.rolling_values = (
                df.groupby('item_number')['quantity']
                .apply(lambda x: x.tail(self.window).mean())
                .to_dict()
            )

        elif self.strategy == 'same_period':
            # Store full history with month/week info for matching
            self.historical = df[['date', 'item_number', 'quantity']].copy()
            self.historical['month'] = self.historical['date'].dt.month
            self.historical['day_of_year'] = self.historical['date'].dt.dayofyear
            self.historical['week_of_year'] = self.historical['date'].dt.isocalendar().week.astype(int)

        return self

    def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate naive predictions.

        Returns:
            DataFrame with columns: date, item_number, quantity_pred
        """
        df = test_df[['date', 'item_number']].copy()
        df['date'] = pd.to_datetime(df['date'], utc=True)

        if self.strategy == 'last':
            df['quantity_pred'] = df['item_number'].map(self.last_values).fillna(0)

        elif self.strategy == 'rolling_mean':
            df['quantity_pred'] = df['item_number'].map(self.rolling_values).fillna(0)

        elif self.strategy == 'same_period':
            df = self._predict_same_period(df)

        # Ensure non-negative predictions
        df['quantity_pred'] = df['quantity_pred'].clip(lower=0).round(0).astype(int)

        return df[['date', 'item_number', 'quantity_pred']]

    def _predict_same_period(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict using same period from last year."""
        df['month'] = df['date'].dt.month
        df['day_of_year'] = df['date'].dt.dayofyear

        # Get last year's average for same day_of_year per item
        hist = self.historical.copy()
        hist_avg = (
            hist.groupby(['item_number', 'day_of_year'])['quantity']
            .mean()
            .rename('quantity_pred')
            .reset_index()
        )

        df = df.merge(hist_avg, on=['item_number', 'day_of_year'], how='left')
        df['quantity_pred'] = df['quantity_pred'].fillna(0)
        df = df.drop(columns=['month', 'day_of_year'])

        return df

    def get_name(self) -> str:
        """Return model display name."""
        if self.strategy == 'rolling_mean':
            return f"Naive(rolling_mean_{self.window})"
        return f"Naive({self.strategy})"
