import numpy as np
import pandas as pd

from src.config import (
    PROCESSED_DATA_DIR,
    TRAIN_START,
    TRAIN_END,
    TEST_START,
    TEST_END,
)


class DataPreprocessor:
    """
    Clean raw market data and prepare it for downstream tasks.
    """

    def __init__(self):

        self.processed_file = PROCESSED_DATA_DIR / "dow30_processed.csv"
        self.train_file = PROCESSED_DATA_DIR / "train.csv"
        self.test_file = PROCESSED_DATA_DIR / "test.csv"

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw market data.
        """

        df = data.copy()

        df.columns.name = None

        df = df.drop_duplicates()

        df = df.sort_values(["Ticker", "Date"])

        df = df.reset_index(drop=True)

        df = df.dropna()

        return df

    def calculate_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate daily simple returns and logarithmic returns.
        """

        df = data.copy()

        df["Daily Return"] = (
            df.groupby("Ticker")["Close"]
            .pct_change()
        )

        df["Log Return"] = (
            df.groupby("Ticker")["Close"]
            .transform(lambda x: np.log(x / x.shift(1)))
        )

        return df

    def split_data(
        self,
        data: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into train and test periods.
        """

        train = data[
            (data["Date"] >= TRAIN_START)
            &
            (data["Date"] <= TRAIN_END)
        ].copy()

        test = data[
            (data["Date"] >= TEST_START)
            &
            (data["Date"] <= TEST_END)
        ].copy()

        return train, test

    def save_processed_data(
        self,
        processed: pd.DataFrame,
        train: pd.DataFrame,
        test: pd.DataFrame
    ) -> None:
        """
        Save processed datasets.
        """

        processed.to_csv(self.processed_file, index=False)

        train.to_csv(self.train_file, index=False)

        test.to_csv(self.test_file, index=False)

    def preprocess(
        self,
        data: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Execute the complete preprocessing pipeline.
        """

        processed = self.clean_data(data)

        processed = self.calculate_returns(processed)

        processed = processed.dropna(
            subset=["Daily Return", "Log Return"]
        ).reset_index(drop=True)

        train, test = self.split_data(processed)

        self.save_processed_data(
            processed,
            train,
            test
        )

        return processed, train, test

    def load_processed_data(self):
        """
        Load processed datasets from disk.
        """

        processed = pd.read_csv(
            self.processed_file,
            parse_dates=["Date"]
        )

        train = pd.read_csv(
            self.train_file,
            parse_dates=["Date"]
        )

        test = pd.read_csv(
            self.test_file,
            parse_dates=["Date"]
        )

        return processed, train, test