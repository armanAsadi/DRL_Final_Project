from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate.base import ARCHModelResult

from src.config import PROCESSED_DATA_DIR


class GARCHModel:
    """
    Rolling GARCH(1,1) volatility forecasting.
    """

    def __init__(
        self,
        window: int = 252,
        refit_every: int = 5,
        n_jobs: int = -1,
    ):

        self.window = window
        self.refit_every = refit_every
        self.n_jobs = n_jobs

        self.output_file = (
            PROCESSED_DATA_DIR /
            "volatility.csv"
        )

    def _fit(
        self,
        returns: pd.Series,
    ) -> ARCHModelResult:
        """
        Fit a zero-mean GARCH(1,1) model.
        """

        model = arch_model(
            returns.dropna(),
            mean="Zero",
            vol="GARCH",
            p=1,
            q=1,
            dist="normal",
            rescale=True,
        )

        result = model.fit(
            disp="off",
            update_freq=0,
            show_warning=False
        )

        return result

    def _forecast(
        self,
        result: ARCHModelResult,
    ) -> float:
        """
        Forecast one-step-ahead volatility.
        """

        forecast = result.forecast(
            horizon=1,
            reindex=False,
        )

        variance = (
            forecast
            .variance
            .iloc[-1, 0]
        )

        volatility = np.sqrt(
            variance /
            (result.scale ** 2)
        )

        return float(volatility)

    def _estimate_single_ticker(
        self,
        ticker_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Rolling volatility estimation for one ticker.
        """

        df = (
            ticker_data
            .sort_values("Date")
            .reset_index(drop=True)
            .copy()
        )

        returns = df["Log Return"]

        forecast_volatility = np.full(
            len(df),
            np.nan,
            dtype=np.float32,
        )

        result = None

        for current in range(len(df)):

            start = max(
                0,
                current - self.window,
            )

            train = (
                returns
                .iloc[start:current]
                .dropna()
            )

            if len(train) < 30:
                continue

            if (
                result is None or
                (current - 1) % self.refit_every == 0
            ):
                result = self._fit(train)

            forecast_volatility[current] = (
                self._forecast(result)
            )

        df["Forecast Volatility"] = (
            forecast_volatility
        )
        
        ticker = ticker_data["Ticker"].unique()[0]
        print(f'Forecast daily volatility completed for ticker: {ticker}')
        
        return (
            df
            .dropna(
                subset=[
                    "Forecast Volatility"
                ]
            )
            .reset_index(drop=True)
        )

    def estimate_all(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Estimate rolling volatility for every ticker.
        """

        groups = [
            group.copy()
            for _,
            group
            in data.groupby(
                "Ticker",
                sort=True,
            )
        ]

        if self.n_jobs == 1:

            results = [
                self._estimate_single_ticker(
                    group
                )
                for group in groups
            ]

        else:

            workers = (
                None
                if self.n_jobs == -1
                else self.n_jobs
            )

            with ProcessPoolExecutor(
                max_workers=workers
            ) as executor:

                results = list(
                    executor.map(
                        self._estimate_single_ticker,
                        groups,
                    )
                )

        return (
            pd.concat(
                results,
                ignore_index=True,
            )
            .sort_values(
                [
                    "Date",
                    "Ticker",
                ]
            )
            .reset_index(drop=True)
        )

    def save_results(
        self,
        results: pd.DataFrame,
    ) -> None:
        """
        Save volatility forecasts.
        """

        results.to_csv(
            self.output_file,
            index=False,
        )

    def load_results(
        self,
    ) -> pd.DataFrame:
        """
        Load volatility forecasts.
        """

        return pd.read_csv(
            self.output_file
        )