import pandas as pd

from src.config import PROCESSED_DATA_DIR


class PortfolioBuilder:
    """
    Build risk-based stock portfolios using forecasted volatility.
    """

    def __init__(self):

        self.output_file = PROCESSED_DATA_DIR / "portfolios.csv"

    def classify_stocks(
        self,
        volatility_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Classify stocks into three risk groups based on forecast volatility.
        """

        df = volatility_data.copy()

        df = df.sort_values(
            "Forecast Volatility",
            ascending=True,
        ).reset_index(drop=True)

        labels = (
            ["Conservative"] * 10 +
            ["Moderate"] * 10 +
            ["Aggressive"] * 10
        )

        df["Risk Level"] = labels

        return df

    def save_portfolios(
        self,
        portfolios: pd.DataFrame,
    ) -> None:
        """
        Save portfolio classifications.
        """

        portfolios.to_csv(
            self.output_file,
            index=False,
        )

    def load_portfolios(self):
        """
        Load portfolio classifications.
        """

        return pd.read_csv(self.output_file)

    def build_portfolios(
        self,
        volatility_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Complete portfolio construction pipeline.
        """

        portfolios = self.classify_stocks(
            volatility_data
        )

        self.save_portfolios(
            portfolios
        )

        return portfolios