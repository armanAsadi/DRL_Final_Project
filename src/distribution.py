"""Dirichlet action distribution for long-only portfolio allocations."""

from __future__ import annotations

import torch as th
from torch import nn
from torch.distributions import Dirichlet
from torch.nn import functional as F

from stable_baselines3.common.distributions import Distribution


class PortfolioDistribution(Distribution):
    """A Dirichlet distribution over portfolio weights.

    Each action is a point on the probability simplex: every weight is positive
    and all weights sum to one.  The policy network emits unconstrained values;
    they are transformed into the Dirichlet concentration parameters ``alpha``
    with ``softplus``.
    """

    def __init__(self, action_dim: int, min_concentration: float = 1e-3) -> None:
        super().__init__()
        if action_dim < 2:
            raise ValueError("A Dirichlet portfolio needs at least two assets.")
        if min_concentration <= 0:
            raise ValueError("min_concentration must be strictly positive.")

        self.action_dim = action_dim
        self.min_concentration = min_concentration
        self.distribution: Dirichlet | None = None

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """Create the actor head that emits raw concentration parameters."""
        return nn.Linear(latent_dim, self.action_dim)

    def proba_distribution(self, alpha_logits: th.Tensor) -> "PortfolioDistribution":
        """Construct the distribution from raw actor-network outputs."""
        if alpha_logits.shape[-1] != self.action_dim:
            raise ValueError(
                f"Expected {self.action_dim} concentration values, "
                f"got {alpha_logits.shape[-1]}."
            )

        alpha = F.softplus(alpha_logits) + self.min_concentration
        self.distribution = Dirichlet(alpha)
        return self

    def _require_distribution(self) -> Dirichlet:
        if self.distribution is None:
            raise RuntimeError("Call proba_distribution() before using the distribution.")
        return self.distribution

    def sample(self) -> th.Tensor:
        """Sample differentiably from the Dirichlet distribution."""
        return self._require_distribution().rsample()

    def mode(self) -> th.Tensor:
        """Return the mode, falling back to the mean when no interior mode exists.

        A Dirichlet distribution has an interior mode only when every
        concentration is greater than one.  Otherwise its mode is on the
        boundary or non-unique; the mean is then a stable valid action for
        deterministic inference.
        """
        distribution = self._require_distribution()
        alpha = distribution.concentration
        mean = distribution.mean
        has_interior_mode = (alpha > 1.0).all(dim=-1, keepdim=True)
        denominator = alpha.sum(dim=-1, keepdim=True) - self.action_dim
        safe_denominator = th.where(has_interior_mode, denominator, th.ones_like(denominator))
        mode = (alpha - 1.0) / safe_denominator
        return th.where(has_interior_mode, mode, mean)

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        return self._require_distribution().log_prob(actions)

    def entropy(self) -> th.Tensor:
        return self._require_distribution().entropy()

    def actions_from_params(
        self, alpha_logits: th.Tensor, deterministic: bool = False
    ) -> th.Tensor:
        self.proba_distribution(alpha_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, alpha_logits: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(alpha_logits)
        return actions, self.log_prob(actions)
