
import numpy as np                    # Numerical computations and arrays
import random                        # Random number generation
import matplotlib.pyplot as plt      # Plotting and visualization
import scipy.stats as stats          # Statistical distributions and functions
from scipy import optimize          # Optimization algorithms
from collections import Counter     # For counting frequency distributions


# =============================================================================
# HEDGE ALGORITHM: EXPERT-BASED ONLINE LEARNING
# =============================================================================

class HedgeAgent1D:
    """
    Hedge algorithm agent for 1-dimensional expert problem.

    This agent implements the multiplicative weights strategy where:
    1. K arms are available, each with an initially equal probability
    2. Action selection is probabilistic, weighted by the current arm weights
    3. Rewards are clipped to [0, 1]
    4. Losses are computed as (1 - reward) for weight updates
    5. Weights are updated multiplicatively: w_i ← w_i * exp(-η * loss_i)
    6. The learning rate (η) controls how quickly the agent adapts to losses
    """

    def __init__(self, K, learning_rate):
        """
        Initialize the HedgeAgent1D.

        Args:
            K (int): Number of available arms, i.e. number of possible prices (assumed equal for each type of product)
            learning_rate (float): Learning rate η for the multiplicative weights update
        """
        self.K = K
        self.lr = learning_rate
        self.weights = np.ones(K, dtype=float)    # Initial equal weights (= 1) for all arms
        self.prob = np.ones(K, dtype=float) / K   # Initial uniform probabilities
        self.last_arm = 0                         # Index of the last pulled arm

    def pull_arm(self):
        """
        Select an arm to pull based on the current weight distribution.

        Returns:
            int: Index of the chosen arm
        """
        self.prob = self.weights / self.weights.sum()      # Normalize weights to probabilities
        self.last_arm = np.random.choice(np.arange(self.K), p=self.prob)
        
        return self.last_arm
    
    def update(self, reward_vector_01):
        """
        Update the arm weights based on observed rewards (full-feedback).

        Args:
            reward_vector_01 (array-like): Vector of observed rewards for all arms
                                           each in the range [0, 1]
                                           1 = full reward, 0 = no reward
        """
        loss = 1.0 - np.clip(reward_vector_01, 0.0, 1.0)   # Compute loss as 1 - reward (clipped to [0, 1])
        self.weights *= np.exp(-self.lr * loss)            # Multiplicative weights update rule
