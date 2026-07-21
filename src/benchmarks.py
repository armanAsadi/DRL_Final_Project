"""Independent benchmark strategies with the evaluator's common result schema."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import INITIAL_CAPITAL, TRANSACTION_COST


RESULT_COLUMNS = ["Date", "Portfolio Value", "Daily Return", "Weights"]


class BenchmarkStrategy(ABC):
    """Common interface implemented by all non-RL benchmark strategies."""

    @abstractmethod
    def run(self) -> pd.DataFrame:
        """Return the standard portfolio-history DataFrame."""


class _PortfolioBenchmark(BenchmarkStrategy):
    def __init__(
        self,
        price_data: pd.DataFrame,
        portfolio: str,
        *,
        initial_capital: float = INITIAL_CAPITAL,
        transaction_cost: float = TRANSACTION_COST,
    ) -> None:
        self.data = (
            price_data.loc[price_data["Portfolio"] == portfolio]
            .copy()
            .sort_values("Date")
            .reset_index(drop=True)
        )
        if len(self.data) < 2:
            raise ValueError("A benchmark needs at least two price observations.")
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.n_assets = len(self.data.iloc[0]["Close Prices"])

    @abstractmethod
    def weights_for_step(self, step: int, previous_weights: np.ndarray) -> np.ndarray:
        """Produce long-only weights before the return from ``step`` to ``step + 1``."""

    def run(self) -> pd.DataFrame:
        value = float(self.initial_capital)
        previous_weights = np.zeros(self.n_assets, dtype=float)
        records: list[dict[str, object]] = []
        for step in range(len(self.data) - 1):
            weights = np.asarray(self.weights_for_step(step, previous_weights), dtype=float)
            if weights.shape != (self.n_assets,) or np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
                raise ValueError("A benchmark must return non-negative weights summing to one.")
            current = np.asarray(self.data.iloc[step]["Close Prices"], dtype=float)
            next_price = np.asarray(self.data.iloc[step + 1]["Close Prices"], dtype=float)
            gross_return = float(np.dot(weights, (next_price - current) / current))
            cost = float(np.abs(weights - previous_weights).sum() * self.transaction_cost)
            net_return = gross_return - cost
            value *= 1.0 + net_return
            records.append(
                {
                    "Date": pd.Timestamp(self.data.iloc[step + 1]["Date"]),
                    "Portfolio Value": value,
                    "Daily Return": net_return,
                    "Weights": weights.copy(),
                }
            )
            previous_weights = weights
        return pd.DataFrame(records, columns=RESULT_COLUMNS)


class EqualWeightStrategy(_PortfolioBenchmark):
    """Rebalanced equal-weight benchmark for one risk portfolio."""

    def weights_for_step(self, step: int, previous_weights: np.ndarray) -> np.ndarray:
        return np.full(self.n_assets, 1.0 / self.n_assets)


class MVOStrategy(_PortfolioBenchmark):
    """Long-only minimum-variance Markowitz portfolio using each state's covariance."""

    def weights_for_step(self, step: int, previous_weights: np.ndarray) -> np.ndarray:
        covariance = np.asarray(self.data.iloc[step]["Covariance Matrix"], dtype=float)
        covariance = (covariance + covariance.T) / 2.0
        covariance += np.eye(self.n_assets) * 1e-8
        initial = previous_weights if np.isclose(previous_weights.sum(), 1.0) else np.full(self.n_assets, 1.0 / self.n_assets)
        result = minimize(
            fun=lambda weights: float(weights @ covariance @ weights),
            x0=initial,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * self.n_assets,
            constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        )
        return result.x if result.success else np.full(self.n_assets, 1.0 / self.n_assets)


class DJIIndexStrategy(BenchmarkStrategy):
    """Price-weighted DJIA proxy built from all 30 raw component close prices.

    The supplied market data must contain ``Date``, ``Ticker``, and ``Close``.
    A price-weighted component index has the same daily returns as the DJIA
    before divisor adjustments, which do not affect percentage returns.
    """

    def __init__(self, market_data: pd.DataFrame, *, initial_capital: float = INITIAL_CAPITAL) -> None:
        required_columns = {"Date", "Ticker", "Close"}
        if not required_columns.issubset(market_data.columns):
            raise ValueError(f"market_data must contain {sorted(required_columns)}.")
        prices = market_data.pivot(index="Date", columns="Ticker", values="Close").sort_index().dropna()
        if len(prices) < 2:
            raise ValueError("DJI benchmark needs at least two complete market dates.")
        self.prices = prices
        self.initial_capital = initial_capital

    def run(self) -> pd.DataFrame:
        index_level = self.prices.sum(axis=1)
        returns = index_level.pct_change().dropna()
        value = self.initial_capital * (1.0 + returns).cumprod()
        weights = self.prices.div(index_level, axis=0).loc[returns.index]
        return pd.DataFrame(
            {
                "Date": returns.index,
                "Portfolio Value": value.to_numpy(),
                "Daily Return": returns.to_numpy(),
                "Weights": list(weights.to_numpy()),
            },
            columns=RESULT_COLUMNS,
        ).reset_index(drop=True)
