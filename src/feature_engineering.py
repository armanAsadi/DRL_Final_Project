from pathlib import Path

import numpy as np
import pandas as pd

from datetime import datetime, timedelta

from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator

from src.config import PROCESSED_DATA_DIR



class FeatureEngineer:
    """
    Build the state representation required by the reinforcement learning environment.
    """

    def __init__(self, lookback: int = 252):

        self.lookback = lookback

        self.output_file = (
            PROCESSED_DATA_DIR / "environment_data.pkl"
        )

    def add_technical_indicators(
    self,
    data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute technical indicators for every stock.
        """

        data = data.copy()

        frames = []

        for ticker in sorted(data["Ticker"].unique()):

            df = (
                data[data["Ticker"] == ticker]
                .copy()
                .sort_values("Date")
            )

            close = df["Close"]

            # MACD
            macd = MACD(close)

            df["MACD"] = macd.macd()

            # RSI
            df["RSI"] = RSIIndicator(close).rsi()

            # Bollinger Bands (Middle Band)
            bb = BollingerBands(close)

            df["BB"] = bb.bollinger_mavg()

            # ATR
            atr = AverageTrueRange(
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
            )

            df["ATR"] = atr.average_true_range()

            # OBV
            obv = OnBalanceVolumeIndicator(
                close=df["Close"],
                volume=df["Volume"],
            )

            df["OBV"] = obv.on_balance_volume()
            
            # Remove rows where at least one indicator is unavailable
            df = df.dropna(
                subset=[
                    "MACD",
                    "RSI",
                    "BB",
                    "ATR",
                    "OBV",
                ]
            )

            frames.append(df)

        data = (
            pd.concat(frames)
            .sort_values(["Date", "Ticker"])
            .reset_index(drop=True)
        )

        return data
    
    def compute_covariance(
        self,
        data: pd.DataFrame,
        tickers: list[str],
    ) -> np.ndarray:
        """
        Compute covariance matrix using daily log returns.
        Returns covariance matrix in the same order as tickers list.
        """
        # Filter and pivot
        filtered_data = data[data["Ticker"].isin(tickers)]
        stock_data = filtered_data.pivot(
            index="Date", 
            columns="Ticker", 
            values="Log Return"
        ).dropna()
        
        # Ensure columns are in the specified order
        stock_data = stock_data[tickers]  # Reorder columns
        
        # Get covariance matrix as numpy array
        covariance_matrix = stock_data.cov().values
        
        return covariance_matrix
    
    def build_state_dataset(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build the environment dataset.
        """

        rows = []

        data = (
            data
            .copy()
            .sort_values(
                [
                    "Date",
                    "Risk Level",
                    "Ticker",
                ]
            )
        )

        portfolios = (
            "Conservative",
            "Moderate",
            "Aggressive",
        )

        dates = (
            data["Date"]
            .drop_duplicates()
            .sort_values()
        )

        for date in dates:

            start_date =  date - timedelta(days=self.lookback)
            window = data[
                (data["Date"] <= date)
                &
                (data["Date"] >= start_date)
            ].copy().sort_values("Ticker")

            for portfolio in portfolios:

                daily = (
                    data[
                        (data["Date"] == date)
                        &
                        (data["Risk Level"] == portfolio)
                    ]
                    .copy()
                    .sort_values("Ticker")
                )

                if len(daily) != 10:
                    continue

                tickers = sorted(daily["Ticker"])

                covariance = self.compute_covariance(
                    window,
                    tickers
                )
                
                # if np.any(np.isnan(covariance)):
                #     continue
                
                covariance = np.nan_to_num(covariance, nan=0.0)

                print(f'State created for date: {date}')
                
                rows.append(
                    {
                        "Date": date,

                        "Portfolio": portfolio,

                        "Close Prices":
                            daily["Close"]
                            .to_numpy(np.float32),

                        "Covariance Matrix":
                            covariance,

                        "MACD":
                            daily["MACD"]
                            .to_numpy(np.float32),

                        "RSI":
                            daily["RSI"]
                            .to_numpy(np.float32),

                        "BB":
                            daily["BB"]
                            .to_numpy(np.float32),

                        "ATR":
                            daily["ATR"]
                            .to_numpy(np.float32),

                        "OBV":
                            daily["OBV"]
                            .to_numpy(np.float32),
                    }
                )

        dataset = (
            pd.DataFrame(rows)
            .sort_values(
                [
                    "Portfolio",
                    "Date",
                ]
            )
            .reset_index(drop=True)
        )

        return dataset
    
    def save_dataset(
    self,
    dataset: pd.DataFrame,
    ):

        dataset.to_pickle(
            self.output_file
        )

    def load_dataset(self):

        return pd.read_pickle(
            self.output_file
        )