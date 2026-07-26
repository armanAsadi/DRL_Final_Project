"""Bayesian optimization for PPO hyperparameters."""

from __future__ import annotations

from collections.abc import Callable
import logging
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

from ..evaluator import Evaluator
from ..metrics import sharpe_ratio
from .trainer import PPOTrainer


LOGGER = logging.getLogger(__name__)


class PPOHyperparameterTuner:
    """Optimize PPO settings while keeping training and evaluation decoupled."""

    def __init__(
        self,
        train_env_factory: Callable[[], Any],
        evaluation_env_factory: Callable[[], Any],
        *,
        total_timesteps: int,
        n_trials: int = 20,
        device: str = "auto",
        seed: int = 42,
        fixed_hyperparameters: dict[str, Any] | None = None,
    ) -> None:
        if total_timesteps <= 0 or n_trials <= 0:
            raise ValueError("total_timesteps and n_trials must be positive.")
        self.train_env_factory = train_env_factory
        self.evaluation_env_factory = evaluation_env_factory
        self.total_timesteps = total_timesteps
        self.n_trials = n_trials
        self.device = device
        self.seed = seed
        self.fixed_hyperparameters = fixed_hyperparameters or {}
        self.best_model = None
        self.best_params: dict[str, Any] | None = None
        self.optimization_history = pd.DataFrame()
        self.best_training_history = pd.DataFrame()

    def optimize(self):
        """Run Gaussian-process Bayesian optimization and return the best PPO model."""
        records: list[dict[str, Any]] = []
        best_score = -np.inf
        random = np.random.default_rng(self.seed)
        LOGGER.info(
            "Starting Bayesian optimization | trials=%s | training_timesteps_per_trial=%s",
            self.n_trials,
            self.total_timesteps,
        )

        def sample_params() -> dict[str, Any]:
            n_steps = int(random.choice([64, 128, 256, 512]))
            batch_size = int(random.choice([size for size in [16, 32, 64, 128] if size <= n_steps]))
            return {
                "learning_rate": float(10 ** random.uniform(np.log10(1e-5), np.log10(3e-4))),
                "n_steps": n_steps,
                "batch_size": batch_size,
                "n_epochs": int(random.integers(3, 11)),
                "clip_range": float(random.uniform(0.1, 0.3)),
            }

        def encode(params: dict[str, Any]) -> list[float]:
            return [
                np.log10(params["learning_rate"]),
                params["n_steps"] / 512.0,
                params["batch_size"] / 128.0,
                params["n_epochs"] / 10.0,
                params["clip_range"],
            ]

        def suggest_params() -> dict[str, Any]:
            initial_trials = min(5, self.n_trials)
            if len(records) < initial_trials:
                return sample_params()
            valid_records = [record for record in records if np.isfinite(record["Sharpe"])]
            if len(valid_records) < 2:
                return sample_params()
            x_train = np.asarray([encode(record) for record in valid_records])
            y_train = np.asarray([record["Sharpe"] for record in valid_records])
            gaussian_process = GaussianProcessRegressor(
                kernel=Matern(nu=2.5) + WhiteKernel(noise_level=1e-5),
                normalize_y=True,
                random_state=self.seed,
            ).fit(x_train, y_train)
            candidates = [sample_params() for _ in range(256)]
            x_candidates = np.asarray([encode(candidate) for candidate in candidates])
            mean, standard_deviation = gaussian_process.predict(x_candidates, return_std=True)
            improvement = mean - y_train.max() - 0.01
            z_score = np.divide(improvement, standard_deviation, out=np.zeros_like(improvement), where=standard_deviation > 0)
            expected_improvement = improvement * norm.cdf(z_score) + standard_deviation * norm.pdf(z_score)
            return candidates[int(np.argmax(expected_improvement))]

        for trial_number in range(1, self.n_trials + 1):
            params = {**suggest_params(), **self.fixed_hyperparameters}
            LOGGER.info("Trial %s/%s started | parameters=%s", trial_number, self.n_trials, params)
            started_at = perf_counter()

            trainer = PPOTrainer(
                self.train_env_factory(),
                params,
                device=self.device,
                seed=self.seed + trial_number,
            )
            model = trainer.train(self.total_timesteps)
            results = Evaluator(model, self.evaluation_env_factory()).evaluate()
            score = sharpe_ratio(results)
            score = -np.inf if not np.isfinite(score) else score
            records.append({"Trial": trial_number, **params, "Sharpe": score})
            LOGGER.info(
                "Trial %s/%s completed | Sharpe=%.4f | elapsed=%.1fs",
                trial_number,
                self.n_trials,
                score,
                perf_counter() - started_at,
            )

            if score > best_score:
                best_score = score
                self.best_model = model
                self.best_params = dict(params)
                self.best_training_history = trainer.history.copy()
                LOGGER.info("New best trial: %s | Sharpe=%.4f", trial_number, score)

        self.optimization_history = pd.DataFrame(records)
        if self.best_model is None:
            raise RuntimeError("No valid hyperparameter trial was completed.")
        LOGGER.info("Bayesian optimization completed | best Sharpe=%.4f | parameters=%s", best_score, self.best_params)
        return self.best_model
