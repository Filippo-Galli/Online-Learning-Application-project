# =============================================================================
# EXP3 ALGORITHM: EXPERT-BASED ONLINE LEARNING
# =============================================================================
import numpy as np                    # Numerical computations and arrays
import random                        # Random number generation
import matplotlib.pyplot as plt      # Plotting and visualization
import scipy.stats as stats          # Statistical distributions and functions
from scipy import optimize          # Optimization algorithms
from collections import Counter     # For counting frequency distributions

class Exp3Agent:
    """
    Implementation of the EXP3 algorithm for online learning with expert advice.
    
    The EXP3 algorithm is designed for adversarial multi-armed bandit problems,
    where the agent only observes the loss of the chosen action. It combines
    exploration and exploitation using a probability distribution over actions.
    
    Key Properties:
    - Exponentially weighted average of experts with exploration
    - Regret bound: O(√(T K log K)) where K is number of experts, T is time horizon
    - Suitable for adversarial settings with partial feedback
    """
    
    def __init__(self, K, learning_rate, exploration_rate = 0.1):
        """
        Initialize the EXP3 algorithm.
        
        Args:
            K: Number of experts (actions/arms)
            learning_rate: Step size for weight updates (typically O(√(log K / (K T))))
            exploration_rate: Probability of exploring random actions (typically O(√(K log K / T)))
        """
        
        self.K = K                                   # Number of experts/actions
        self.learning_rate = learning_rate           # Learning rate parameter
        self.exploration_rate = exploration_rate     # Exploration parameter
        self.weights = np.ones(K, dtype=float)       # Expert weights (start uniform)
        self.x_t = np.ones(K, dtype=float) / K       # Probability distribution over experts
        self.a_t = None                              # Last selected action
        self.t = 0                                   # Current time step
        self.prob_distribution = None                # Current probability distribution

    def pull_arm(self):
        """
        Select an expert (action) according to current probability distribution.
        
        Combines exploitation (based on weights) and exploration (uniform random).
        
        Returns:
            int: Index of selected expert/action
        """
        # Compute probability distribution with exploration
        self.prob_distribution = ((1 - self.exploration_rate) * 
                             (self.weights / np.sum(self.weights)) + 
                             (self.exploration_rate / self.K))
        
        # Sample action according to probability distribution
        self.a_t = np.random.choice(np.arange(self.K), p=self.prob_distribution)
        
        return self.a_t
    
    def update(self, losses):
        """
        Update expert weights based on observed loss from chosen action.
        
        Uses importance-weighted loss to estimate losses for all experts.
        
        Args:
            losses: Losses received from the chosen action
        """
        # Estimate loss for chosen action using importance weighting
        estimated_loss = np.zeros(self.K)
        
        # Importance-weighted loss estimate for the chosen action
        estimated_loss[self.a_t] = losses[self.a_t] / self.prob_distribution[self.a_t]
        
        # Exponential weight update: experts with higher estimated losses get lower weights
        self.weights *= np.exp(-self.learning_rate * estimated_loss)
        
        # Increment time counter
        self.t += 1