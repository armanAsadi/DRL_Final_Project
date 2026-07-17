import numpy as np
import pandas as pd

from arch import arch_model

from src.config import PROCESSED_DATA_DIR


class GARCHModel:
    """
    Estimate GARCH(1,1) volatility for one or multiple assets.
    """

    def __init__(self):

        self.model = None
        self.result = None

        self.output_file = PROCESSED_DATA_DIR / "volatility.csv"

    def fit(self, returns: pd.Series):
        """
        Fit a zero-mean GARCH(1,1) model.
        """

        returns = returns.dropna()

        self.model = arch_model(
            returns,
            mean="Zero",
            vol="GARCH",
            p=1,
            q=1,
            dist="normal",
            rescale=True,
        )

        self.result = self.model.fit(
            disp="off",
            update_freq=0,
        )

        return self.result

    def forecast(self) -> float:
        """
        Forecast one-step-ahead conditional volatility.
        """

        if self.result is None:
            raise ValueError("Model has not been fitted.")

        forecast = self.result.forecast(horizon=1)

        variance = forecast.variance.iloc[-1, 0]

        volatility = np.sqrt(
            variance / (self.result.scale ** 2)
        )

        return float(volatility)

    def get_parameters(self) -> dict:
        """
        Return fitted GARCH parameters.
        """

        if self.result is None:
            raise ValueError("Model has not been fitted.")

        params = self.result.params

        omega = float(params["omega"])
        alpha = float(params["alpha[1]"])
        beta = float(params["beta[1]"])

        return {
            "Omega": omega,
            "Alpha": alpha,
            "Beta": beta,
            "Persistence": alpha + beta,
        }

    def estimate_all(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Estimate GARCH volatility for every stock.
        """

        results = []

        for ticker in sorted(data["Ticker"].unique()):

            returns = data.loc[
                data["Ticker"] == ticker,
                "Log Return",
            ]

            self.fit(returns)

            params = self.get_parameters()

            results.append(
                {
                    "Ticker": ticker,
                    **params,
                    "Forecast Volatility": self.forecast(),
                }
            )

        results = pd.DataFrame(results)

        return results

    def save_results(
        self,
        results: pd.DataFrame,
    ) -> None:
        """
        Save estimated volatilities.
        """

        results.to_csv(
            self.output_file,
            index=False,
        )

    def load_results(self) -> pd.DataFrame:
        """
        Load estimated volatilities.
        """

        return pd.read_csv(self.output_file)