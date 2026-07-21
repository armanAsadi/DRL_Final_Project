"""Callbacks used during PPO training."""

from __future__ import annotations

import logging
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback


LOGGER = logging.getLogger(__name__)


class TrainingHistoryCallback(BaseCallback):
    """Collect PPO's optimizer losses once per policy update.

    PPO writes these values to SB3's logger after an update.  The callback
    captures them at the next rollout boundary and once more at training end.
    This is intentionally an optimizer-loss history, not an episode-reward
    history.
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self._records: list[dict[str, float | int]] = []

    def _on_step(self) -> bool:
        return True

    def _capture_losses(self) -> None:
        values = self.logger.name_to_value
        required_keys = ("train/policy_gradient_loss", "train/value_loss", "train/loss")
        if not all(key in values for key in required_keys):
            return
        self._records.append(
            {
                "Update": len(self._records) + 1,
                "Timesteps": self.num_timesteps,
                "Policy Loss": float(values["train/policy_gradient_loss"]),
                "Value Loss": float(values["train/value_loss"]),
                "Total Loss": float(values["train/loss"]),
            }
        )
        LOGGER.info(
            "PPO update %s | timesteps=%s/%s | policy_loss=%.6f | value_loss=%.6f | total_loss=%.6f",
            len(self._records),
            self.num_timesteps,
            self.model._total_timesteps,
            values["train/policy_gradient_loss"],
            values["train/value_loss"],
            values["train/loss"],
        )

    def _on_rollout_end(self) -> None:
        progress = self.num_timesteps / self.model._total_timesteps * 100.0
        LOGGER.info(
            "PPO rollout collected | timesteps=%s/%s (%.1f%%)",
            self.num_timesteps,
            self.model._total_timesteps,
            progress,
        )
        self._capture_losses()

    def _on_training_end(self) -> None:
        self._capture_losses()

    def to_frame(self) -> pd.DataFrame:
        """Return the learning curve with one row per PPO update."""
        return pd.DataFrame(
            self._records,
            columns=["Update", "Timesteps", "Policy Loss", "Value Loss", "Total Loss"],
        )
