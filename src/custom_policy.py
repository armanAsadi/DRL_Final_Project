"""Stable-Baselines3 policy using a Dirichlet portfolio distribution."""

from __future__ import annotations

from functools import partial

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy

from .distribution import PortfolioDistribution


class PortfolioPolicy(ActorCriticPolicy):
    """Actor-critic policy for long-only, fully-invested portfolios.

    This policy is compatible with PPO.  It replaces SB3's default diagonal
    Gaussian action distribution with a Dirichlet distribution, guaranteeing
    that sampled actions lie on the simplex.
    """

    def __init__(self, *args, min_concentration: float = 1e-3, **kwargs) -> None:
        if kwargs.get("use_sde", False):
            raise ValueError("PortfolioPolicy does not support generalized SDE.")
        self.min_concentration = min_concentration
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule) -> None:
        """Build SB3's actor/critic networks with a Dirichlet actor head."""
        if not isinstance(self.action_space, spaces.Box) or len(self.action_space.shape) != 1:
            raise ValueError("PortfolioPolicy requires a one-dimensional Box action space.")
        if not (np.allclose(self.action_space.low, 0.0) and np.allclose(self.action_space.high, 1.0)):
            raise ValueError("PortfolioPolicy requires action bounds [0, 1].")

        self._build_mlp_extractor()
        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        self.action_dist = PortfolioDistribution(
            action_dim=self.action_space.shape[0],
            min_concentration=self.min_concentration,
        )
        self.action_net = self.action_dist.proba_distribution_net(latent_dim_pi)
        self.value_net = th.nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        """Return the Dirichlet distribution parameterized by actor outputs."""
        alpha_logits = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(alpha_logits)
