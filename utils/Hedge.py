
import numpy as np                    # Numerical computations and arrays
import random                        # Random number generation
import matplotlib.pyplot as plt      # Plotting and visualization
import scipy.stats as stats          # Statistical distributions and functions
from scipy import optimize          # Optimization algorithms
from collections import Counter     # For counting frequency distributions


# =============================================================================
# HEDGE ALGORITHM: EXPERT-BASED ONLINE LEARNING
# =============================================================================

class HedgeAgent:
    """
    Implementation of the Hedge algorithm for online learning with expert advice.
    
    The Hedge algorithm maintains weights over a set of experts (actions) and
    updates these weights based on observed losses. It provides strong theoretical
    guarantees for regret minimization in adversarial settings.
    
    Key Properties:
    - Exponentially weighted average of experts
    - Regret bound: O(√T log K) where K is number of experts, T is time horizon
    - Robust to adversarial losses
    """
    
    def __init__(self, K, learning_rate):
        """
        Initialize the Hedge algorithm.
        
        Args:
            K: Number of experts (actions/arms)
            learning_rate: Step size for weight updates (typically O(√(log K / T)))
        """
        
        self.K = K                                   # Number of experts/actions
        self.learning_rate = learning_rate           # Learning rate parameter
        self.weights = np.ones(K, dtype=float)       # Expert weights (start uniform)
        self.x_t = np.ones(K, dtype=float) / K       # Probability distribution over experts
        self.a_t = None                              # Last selected action
        self.t = 0                                   # Current time step

    def pull_arm(self):
        """
        Select an expert (action) according to current probability distribution.
        
        Uses the exponentially weighted average strategy where experts with
        lower cumulative losses receive higher selection probabilities.
        
        Returns:
            int: Index of selected expert/action
        """
        # Compute probability distribution from current weights
        self.x_t = self.weights / np.sum(self.weights)
        
        # Sample action according to probability distribution
        self.a_t = np.random.choice(np.arange(self.K), p=self.x_t)
        
        return self.a_t
    
    def update(self, l_t):
        """
        Update expert weights based on observed losses.
        
        Uses exponential update rule: w_{i,t+1} = w_{i,t} * exp(-η * l_{i,t})
        where η is the learning rate and l_{i,t} is the loss of expert i at time t.
        
        Args:
            l_t: Array of losses for each expert at current time step
        """
        # Exponential weight update: experts with higher losses get lower weights
        self.weights *= np.exp(-self.learning_rate * l_t)
        
        # Increment time counter
        self.t += 1