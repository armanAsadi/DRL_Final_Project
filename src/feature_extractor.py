import numpy as np
import torch
import torch.nn as nn

from gymnasium import spaces

from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
)


class PortfolioFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for the portfolio management environment.

    The observation is a 2D matrix composed of:

        - Covariance matrix
        - Close prices
        - Technical indicators

    Each component is encoded independently before being fused into a
    single latent feature vector.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 128,
        close_dim: int = 32,
        indicator_dim: int = 64,
        covariance_dim: int = 128,
    ):
        super().__init__(
            observation_space,
            features_dim,
        )

        state_rows, n_assets = observation_space.shape

        self.n_assets = n_assets
        self.n_indicators = state_rows - n_assets - 1

        covariance_input = n_assets * n_assets
        indicator_input = self.n_indicators * n_assets

        self.close_encoder = nn.Sequential(
            nn.Linear(n_assets, close_dim),
            nn.ReLU(),
        )

        self.indicator_encoder = nn.Sequential(
            nn.Linear(indicator_input, indicator_dim),
            nn.ReLU(),
        )

        self.covariance_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                covariance_input,
                covariance_dim,
            ),
            nn.ReLU(),
        )

        fusion_dim = (
            close_dim
            + indicator_dim
            + covariance_dim
        )

        self.fusion = nn.Sequential(
            nn.Linear(
                fusion_dim,
                features_dim,
            ),
            nn.ReLU(),
        )

    def forward(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract latent features from the observation.
        """

        covariance = observations[
            :,
            : self.n_assets,
            :
        ]

        close = observations[
            :,
            self.n_assets,
            :
        ]

        indicators = observations[
            :,
            self.n_assets + 1 :,
            :
        ]

        indicators = indicators.flatten(
            start_dim=1
        )

        covariance_features = (
            self.covariance_encoder(
                covariance
            )
        )

        close_features = (
            self.close_encoder(
                close
            )
        )

        indicator_features = (
            self.indicator_encoder(
                indicators
            )
        )

        features = torch.cat(
            (
                covariance_features,
                close_features,
                indicator_features,
            ),
            dim=1,
        )

        return self.fusion(features)