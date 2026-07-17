import pandas as pd
import yfinance as yf

from pathlib import Path
from src.config import (
    DJIA_TICKERS,
    START_DATE,
    END_DATE,
    RAW_DATA_DIR,
)


class DataLoader:
    """
    Download, save and load historical market data.
    """

    def __init__(self):

        self.file_path = RAW_DATA_DIR / "dow30_raw.csv"

    def download_data(self) -> pd.DataFrame:
        """
        Download historical OHLCV data from Yahoo Finance.
        """

        data = yf.download(
            tickers=DJIA_TICKERS,
            start=START_DATE,
            end=END_DATE,
            auto_adjust=False,
            progress=True,
            group_by="ticker",
        )

        frames = []

        for ticker in DJIA_TICKERS:

            df = data[ticker].copy()

            df.reset_index(inplace=True)

            df["Ticker"] = ticker

            frames.append(df)

        df = pd.concat(frames, ignore_index=True)

        df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

        df.columns.name = None

        return df

    def save_data(self, data: pd.DataFrame) -> None:
        """
        Save downloaded market data.
        """

        data.to_csv(self.file_path, index=False)

    def load_data(self) -> pd.DataFrame:
        """
        Load raw market data from disk.
        """

        return pd.read_csv(self.file_path, parse_dates=["Date"])

    def get_data(self, force_download: bool = False) -> pd.DataFrame:
        """
        Return raw market data.

        If the dataset already exists it is loaded from disk,
        otherwise it is downloaded automatically.
        """

        if self.file_path.exists() and not force_download:

            print("Loading raw data...")

            return self.load_data()

        print("Downloading raw data...")

        df = self.download_data()

        self.save_data(df)

        return df