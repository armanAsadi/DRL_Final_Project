"""Evaluation of a trained policy into a standard portfolio-history table."""

from __future__ import annotations

import numpy as np
import pandas as pd
from stable_baselines3.common.base_class import BaseAlgorithm

from .environment import PortfolioEnv


class Evaluator:
    """Run one trained PPO policy and return its realized portfolio history."""

    def __init__(self, model: BaseAlgorithm, environment: PortfolioEnv) -> None:
        self.model = model
        self.environment = environment

    def evaluate(self, deterministic: bool = True) -> pd.DataFrame:
        """Return Date, Portfolio Value, Daily Return, and Weights for each trade."""
        observation, _ = self.environment.reset()
        records: list[dict[str, object]] = []
        terminated = truncated = False

        while not (terminated or truncated):
            action, _ = self.model.predict(observation, deterministic=deterministic)
            action = np.asarray(action, dtype=np.float32)
            observation, _, terminated, truncated, info = self.environment.step(action)
            records.append(
                {
                    "Date": pd.Timestamp(info["date"]),
                    "Portfolio Value": float(info["portfolio_value"]),
                    "Daily Return": float(info["portfolio_return"]),
                    "Weights": action.copy(),
                }
            )

        return pd.DataFrame(records, columns=["Date", "Portfolio Value", "Daily Return", "Weights"])
