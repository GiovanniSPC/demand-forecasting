from abc import ABC, abstractmethod
import pandas as pd


class BaseModel(ABC):
    """
    Abstract base class for all forecasting models.

    Every model must implement:
        - fit(train_df)      : Train on historical data.
        - predict(test_df)   : Return predictions for the test period.
        - get_name()         : Return a human-readable model name.

    Convention:
        - train_df and test_df have at least: date, item_number, quantity
        - predict() returns a DataFrame with: date, item_number, quantity_pred
    """

    @abstractmethod
    def fit(self, train_df: pd.DataFrame):
        """
        Train the model.

        Args:
            train_df: Training DataFrame with columns date, item_number, quantity (+ features).
        """
        pass

    @abstractmethod
    def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions.

        Args:
            test_df: Test DataFrame with same structure as train_df.

        Returns:
            DataFrame with columns: date, item_number, quantity_pred
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return the model's display name."""
        pass
