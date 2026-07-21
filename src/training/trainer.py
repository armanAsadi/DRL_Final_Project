"""A single-responsibility PPO trainer."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

from stable_baselines3 import PPO

from ..custom_policy import PortfolioPolicy
from ..environment import PortfolioEnv
from ..feature_extractor import PortfolioFeatureExtractor
from .callbacks import TrainingHistoryCallback


LOGGER = logging.getLogger(__name__)


class PPOTrainer:
    """Train one PPO model; this class has no optimizer or benchmark knowledge."""

    def __init__(
        self,
        environment: PortfolioEnv,
        hyperparameters: dict[str, Any] | None = None,
        *,
        device: str = "auto",
        seed: int | None = 42,
        verbose: int = 0,
    ) -> None:
        self.environment = environment
        self.hyperparameters = dict(hyperparameters or {})
        self.device = device
        self.seed = seed
        self.verbose = verbose
        self.history = TrainingHistoryCallback().to_frame()
        self.model: PPO | None = None

    def learn(self, total_timesteps: int) -> PPO:
        """Train and return a PPO model, exposing optimizer losses in ``history``."""
        if total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive.")

        LOGGER.info(
            "Starting PPO training | timesteps=%s | device=%s | hyperparameters=%s",
            total_timesteps,
            self.device,
            self.hyperparameters,
        )

        hyperparameters = dict(self.hyperparameters)
        policy_kwargs = {
            "features_extractor_class": PortfolioFeatureExtractor,
            "features_extractor_kwargs": {"features_dim": 128},
            "net_arch": {"pi": [64, 64], "vf": [64, 64]},
        }
        policy_kwargs.update(hyperparameters.pop("policy_kwargs", {}))
        ppo_parameters = {**hyperparameters, "policy_kwargs": policy_kwargs}
        callback = TrainingHistoryCallback()
        self.model = PPO(
            PortfolioPolicy,
            self.environment,
            device=self.device,
            seed=self.seed,
            verbose=self.verbose,
            **ppo_parameters,
        )
        started_at = perf_counter()
        self.model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)
        self.history = callback.to_frame()
        LOGGER.info(
            "PPO training completed | elapsed=%.1fs | updates=%s | final_timesteps=%s",
            perf_counter() - started_at,
            len(self.history),
            self.model.num_timesteps,
        )
        return self.model

    def train(self, total_timesteps: int) -> PPO:
        """Backward-compatible alias for :meth:`learn`."""
        return self.learn(total_timesteps)
