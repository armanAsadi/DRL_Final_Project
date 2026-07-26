import pandas as pd

from src.config import PROCESSED_DATA_DIR


class PortfolioBuilder:
    """
    Build rolling daily portfolios based on forecast volatility.
    """

    def __init__(self):

        self.output_file = PROCESSED_DATA_DIR / "portfolios.csv"

    def classify_stocks(
        self,
        volatility: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Classify stocks separately for every trading day.
        """

        portfolios = []

        for _, group in volatility.groupby("Date"):

            group = (
                group.sort_values(
                    "Forecast Volatility",
                    ascending=True,
                )
                .reset_index(drop=True)
            )

            labels = (
                ["Conservative"] * 10
                + ["Moderate"] * 10
                + ["Aggressive"] * 10
            )

            group["Risk Level"] = labels

            portfolios.append(group)

        return (
            pd.concat(
                portfolios,
                ignore_index=True,
            )
            .sort_values(
                ["Date", "Ticker"]
            )
            .reset_index(drop=True)
        )

    def save_portfolios(
        self,
        portfolios: pd.DataFrame,
    ) -> None:

        portfolios.to_csv(
            self.output_file,
            index=False,
        )

    def load_portfolios(
        self,
    ) -> pd.DataFrame:

        return pd.read_csv(
            self.output_file,
            parse_dates=["Date"],
        )

    def build_portfolios(
        self,
        market_data: pd.DataFrame,
        volatility: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Complete rolling portfolio construction pipeline.
        """

        portfolios = self.classify_stocks(
            volatility
        )

        self.save_portfolios(
            portfolios
        )

        return portfolios