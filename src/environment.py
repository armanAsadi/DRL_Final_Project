import gymnasium as gym
import numpy as np
import pandas as pd

from gymnasium import spaces


class PortfolioEnv(gym.Env):
    """
    Reinforcement learning environment for portfolio allocation.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        dataset: pd.DataFrame,
        portfolio: str,
    ):
        """
        Initialize the portfolio environment.
        """

        super().__init__()

        self.portfolio = portfolio

        self.portfolio_value = 1.0

        self.portfolio_history = []

        self.data = (
            dataset[
                dataset["Portfolio"] == portfolio
            ]
            .copy()
            .reset_index(drop=True)
        )

        self.current_step = 0

        self.n_assets = len(
            self.data.iloc[0]["Close Prices"]
        )

        self.transaction_cost_rate = 0.0005

        self.previous_weights = np.zeros(
            self.n_assets,
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(16, self.n_assets),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_assets,),
            dtype=np.float32,
        )
    
    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        """
        Reset the environment.
        """

        super().reset(seed=seed)

        self.current_step = 0

        self.portfolio_value = 1.0

        self.portfolio_history = [1.0]

        self.previous_weights = np.zeros(
            self.n_assets,
            dtype=np.float32,
        )

        observation = self._get_state()

        info = {
            "portfolio": self.portfolio,
            "date": self.data.iloc[0]["Date"],
        }

        return observation, info
    
    def step(
        self,
        action,
    ):
        """
        Execute one environment step.
        """

        assert np.all(action >= 0)

        assert np.isclose(
            action.sum(),
            1.0,
            atol=1e-5,
        )

        portfolio_return = (
            self._calculate_portfolio_return(
                action
            )
        )

        reward, transaction_cost = (
            self._calculate_reward(
                portfolio_return,
                action,
            )
        )

        self._update_portfolio_value(
            reward
        )

        self.previous_weights = action.copy()

        self.current_step += 1

        terminated = (
            self.current_step
            >= len(self.data) - 1
        )

        truncated = False

        observation = self._get_state()

        info = {
            "portfolio_return": portfolio_return,
            "portfolio_value": self.portfolio_value,
            "date": self._get_current_row()["Date"],
            "transaction_cost": transaction_cost,
        }

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )
    
    def render(
        self,
    ):
        """
        Render the current environment state.
        """

        row = self._get_current_row()

        print("=" * 50)

        print(f"Portfolio : {self.portfolio}")

        print(f"Step      : {self.current_step}")

        print(f"Date      : {row['Date']}")

        print("=" * 50)

    def close(
        self,
    ):
        """
        Close the environment.
        """

        pass

    def _get_state(
        self,
    ) -> np.ndarray:
        """
        Construct the current observation.
        """

        row = self._get_current_row()

        state = np.vstack(
            [
                row["Covariance Matrix"],

                row["Close Prices"].reshape(1, -1),

                row["MACD"].reshape(1, -1),

                row["RSI"].reshape(1, -1),

                row["BB"].reshape(1, -1),

                row["ATR"].reshape(1, -1),

                row["OBV"].reshape(1, -1),
            ]
        )

        return state.astype(np.float32)
    
    def _calculate_portfolio_return(
        self,
        action,
    ):
        """
        Calculate portfolio return.
        """

        asset_returns = (
            self._calculate_asset_returns()
        )

        portfolio_return = np.dot(
            action,
            asset_returns,
        )

        return float(portfolio_return)
    
    def _calculate_reward(
        self,
        portfolio_return,
        action,
    ):
        """
        Calculate reward.
        """

        transaction_cost = (
            self._calculate_transaction_cost(
                action
            )
        )

        reward = (
            portfolio_return
            - transaction_cost
        )

        return reward, transaction_cost
    
    def _calculate_asset_returns(self):
        """
        Calculate daily returns for all assets.
        """

        current = self._get_current_row()

        next_row = self.data.iloc[
            self.current_step + 1
        ]

        current_price = current["Close Prices"]

        next_price = next_row["Close Prices"]

        asset_returns = (
            next_price - current_price
        ) / current_price

        return asset_returns.astype(np.float32)
    
    def _update_portfolio_value(
        self,
        reward,
    ):
        """
        Update portfolio value.
        """

        self.portfolio_value *= (
            1 + reward
        )

        self.portfolio_history.append(
            self.portfolio_value
        )
    
    def _calculate_transaction_cost(
        self,
        action,
    ):
        """
        Calculate portfolio transaction cost.
        """

        turnover = np.sum(
            np.abs(
                action - self.previous_weights
            )
        )

        cost = (
            turnover
            * self.transaction_cost_rate
        )

        return float(cost)

    def _get_current_row(self):
        return self.data.iloc[self.current_step]