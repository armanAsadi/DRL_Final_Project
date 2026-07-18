import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


class FeatureScaler:
    """
    Scale engineered features before training the reinforcement learning agent.
    """

    FEATURE_COLUMNS = {
        "close": "Close Prices",
        "macd": "MACD",
        "atr": "ATR",
        "bb": "BB",
        "obv": "OBV",
    }

    def __init__(self):
        """
        Initialize all feature scalers.
        """

        self.scalers = {
            feature: StandardScaler()
            for feature in self.FEATURE_COLUMNS
        }

        self.scalers["covariance"] = StandardScaler()

    def fit(
        self,
        dataset: pd.DataFrame,
    ):
        """
        Fit all feature scalers using the training dataset.
        """

        for feature, column in self.FEATURE_COLUMNS.items():

            values = self._stack_feature(
                dataset,
                column,
            )

            self.scalers[feature].fit(values)

        covariance = self._stack_covariance(dataset)

        self.scalers["covariance"].fit(
            covariance
        )

    def transform(
        self,
        dataset: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Transform all engineered features.
        """

        dataset = dataset.copy()

        for feature, column in self.FEATURE_COLUMNS.items():

            values = self._stack_feature(
                dataset,
                column,
            )

            values = self.scalers[
                feature
            ].transform(values)

            dataset[column] = list(values)

        covariance = self._stack_covariance(
            dataset
        )

        covariance = self.scalers[
            "covariance"
        ].transform(covariance)

        n_assets = dataset["Covariance Matrix"].iloc[0].shape[0]

        dataset["Covariance Matrix"] = (
            self._restore_covariance(
                covariance,
                n_assets
            )
        )

        dataset["RSI"] = dataset["RSI"].apply(
            lambda x: x / 100.0
        )

        return dataset

    def fit_transform(
        self,
        dataset: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Fit and transform the dataset.
        """

        self.fit(dataset)

        return self.transform(dataset)

    def save(
        self,
        path: str,
    ):
        """
        Save fitted scalers.
        """

        joblib.dump(
            self.scalers,
            path,
        )

    def load(
        self,
        path: str,
    ):
        """
        Load fitted scalers.
        """

        self.scalers = joblib.load(path)

    def _stack_feature(
        self,
        dataset: pd.DataFrame,
        column: str,
    ) -> np.ndarray:
        """
        Stack feature arrays into a matrix.
        """

        return np.vstack(
            dataset[column].to_numpy()
        )

    def _stack_covariance(
        self,
        dataset: pd.DataFrame,
    ) -> np.ndarray:
        """
        Stack covariance matrices into a 2D array.
        """

        covariance = np.stack(
            dataset[
                "Covariance Matrix"
            ].to_numpy()
        )

        return covariance.reshape(
            len(covariance),
            -1,
        )

    def _restore_covariance(
        self,
        covariance: np.ndarray,
        n_assets: int
    ):
        """
        Restore covariance matrices to their original shape.
        """

        return list(
            covariance.reshape(
                -1,
                n_assets,
                n_assets,
            )
        )