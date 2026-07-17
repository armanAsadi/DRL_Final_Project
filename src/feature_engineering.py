from pathlib import Path

import numpy as np
import pandas as pd

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
    ):
        """
        Compute rolling covariance matrices.
        """

        df = (
            data[data["Ticker"].isin(tickers)]
            .copy()
            .sort_values(["Date", "Ticker"])
        )

        price = df.pivot(
            index="Date",
            columns="Ticker",
            values="Close",
        )

        returns = price.pct_change()

        covariance = {}

        for i in range(
            self.lookback,
            len(returns),
        ):

            cov = (
                returns.iloc[
                    i - self.lookback:i
                ]
                .dropna()
                .cov()
                .values
            )

            covariance[
                returns.index[i]
            ] = cov

        return covariance
    
    def build_state_dataset(
    self,
    data: pd.DataFrame,
    portfolio_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build the environment dataset.
        """

        rows = []

        for portfolio in [
            "Conservative",
            "Moderate",
            "Aggressive",
        ]:

            tickers = portfolio_df.loc[
                portfolio_df["Risk Level"] == portfolio,
                "Ticker",
            ].tolist()

            covariance = self.compute_covariance(
                data,
                tickers,
            )

            portfolio_data = (
                data[data["Ticker"].isin(tickers)]
                .copy()
            )

            for date, cov in covariance.items():

                daily = (
                    portfolio_data[
                        portfolio_data["Date"] == date
                    ]
                    .sort_values("Ticker")
                )

                rows.append(
                    {
                        "Date": date,
                        "Portfolio": portfolio,
                        "Close Prices": daily[
                            "Close"
                        ].to_numpy(),

                        "Covariance Matrix": cov,

                        "MACD": daily["MACD"].to_numpy(),

                        "RSI": daily["RSI"].to_numpy(),

                        "BB": daily["BB"].to_numpy(),

                        "ATR": daily["ATR"].to_numpy(),

                        "OBV": daily["OBV"].to_numpy(),
                    }
                )

        dataset = pd.DataFrame(rows)

        dataset = dataset.sort_values(
            [
                "Portfolio",
                "Date",
            ]
        ).reset_index(drop=True)

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