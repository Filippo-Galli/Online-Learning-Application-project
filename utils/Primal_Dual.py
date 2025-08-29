# =============================================================================
# PRIMAL-DUAL AGENT: ONLINE LEARNING WITH CONSTRAINTS
# =============================================================================
import numpy as np                    # Numerical computations and arrays
import random                        # Random number generation
import matplotlib.pyplot as plt      # Plotting and visualization
import scipy.stats as stats          # Statistical distributions and functions
from scipy import optimize          # Optimization algorithms
from collections import Counter     # For counting frequency distributions
import sys
sys.path.append('../utils')

from Hedge import HedgeAgent
from EXP3_P import Exp3Agent

import os


class PrimalDualAgent:
    """
    Primal-Dual agent for dynamic pricing with inventory constraints.
    
    This agent combines the Hedge algorithm with Lagrangian multipliers to handle
    constrained online optimization problems. It's particularly suited for:
    - Non-stationary environments
    - Resource/inventory constraints
    - Revenue maximization under uncertainty
    
    Algorithm Overview:
    1. Use Hedge for exploration over price options
    2. Maintain Lagrangian multiplier for inventory constraint
    3. Construct Lagrangian: L = revenue - λ * (sales_rate - target_rate)
    4. Update both Hedge weights and multiplier based on feedback
    
    Theoretical Guarantees:
    - Regret bound: O(√T log K) for stationary environments
    - Constraint violation: O(√T) on average
    """
    
    def __init__(self, prices, valuation, P, T, eta, algorithm = 'exp3'):
        """
        Initialize the Primal-Dual agent.
        
        Args:
            prices: Array of available price options
            valuation: Customer valuation distributions (for compatibility)
            P: Total inventory (budget constraint)
            T: Time horizon (total number of rounds)
            eta: Learning rate for Lagrangian multiplier updates
        """
                
       
        # Basic parameters
        self.prices = prices                         # Available price options
        self.K = len(prices)                        # Number of price options
        self.valuation = valuation                  # Valuation distributions (compatibility)
        self.P = P                                  # Total inventory constraint
        self.T = T                                  # Time horizon
        self.eta = eta                              # Learning rate for multiplier
        
        # Constraint management
        self.rho = self.P / self.T                 # Target selling rate (ρ = P/T)
        self.lmbd = 1                              # Lagrangian multiplier (λ)
        
        # Online learning components
        hedge_lr = np.sqrt(np.log(self.K) / T)     # Optimal Hedge learning rate
        exp3_lr = np.sqrt( np.log(self.K) / (self.K * T))  # Optimal EXP3 learning rate
        self.hedge = HedgeAgent(self.K, hedge_lr)  # Hedge algorithm for exploration
        self.exp3 = Exp3Agent(self.K, exp3_lr)  # EXP3 algorithm for exploration
        
        # State tracking
        self.t = 0                                 # Current time step
        self.remaining_inventory = P               # Inventory remaining
        self.bid_index = 0                         # Last selected price index
        
        # Performance tracking
        self.N_pulls = np.zeros(len(prices))       # Number of times each price used
        self.reward = np.zeros(self.K)             # Cumulative rewards per price

        # Choosen algorithm
        self.algorithm = algorithm                  # Algorithm to use: 'hedge' or 'exp3
        
        # History for analysis
        self.history = {
            'prices': [],        # Selected prices over time
            'rewards': [],       # Rewards received
            'purchases': [],     # Purchase indicators
            'inventory': []      # Remaining inventory levels
        }

    def bid(self):
        """
        Select a price using the Hedge algorithm with inventory awareness.
        
        If inventory is exhausted, automatically selects lowest price (index 0)
        to minimize regret while avoiding further sales.
        
        Returns:
            float: Selected price value
        """
        # If no inventory remaining, select lowest price to avoid sales
        if self.remaining_inventory < 1:
            self.bid_index = 0
            return self.prices[0]
        
        # Use Hedge algorithm to select price index
        if(self.algorithm == 'hedge'):
            self.bid_index = self.hedge.pull_arm()
        else:
            self.bid_index = self.exp3.pull_arm()
            
        return self.prices[self.bid_index]
    
    def update(self, f_t, c_t, f_t_full, c_t_full):
        """
        Update the agent based on observed outcomes using primal-dual approach.
        
        This method implements the core primal-dual update:
        1. Construct Lagrangian for all actions: L_i = f_i - λ * (c_i - ρ)
        2. Update Hedge algorithm with normalized Lagrangian
        3. Update Lagrangian multiplier based on constraint violation
        4. Update inventory and statistics
        
        Args:
            f_t: Observed revenue for selected action
            c_t: Observed purchase indicator for selected action  
            f_t_full: Expected revenues for all actions (counterfactual)
            c_t_full: Expected purchase probabilities for all actions
        """
        # Step 1: Construct Lagrangian for all price options
        # L_i = revenue_i - λ * (sales_rate_i - target_rate)
        # This balances revenue maximization with constraint satisfaction
        L_full = f_t_full - self.lmbd * (c_t_full - self.rho)
        
        # Step 2: Normalize Lagrangian to [0,1] for Hedge algorithm
        # Hedge expects losses in [0,1], so we need proper normalization
        max_possible_revenue = np.max(self.prices)
        min_lagrangian = -self.lmbd  # When f=0, c=1 (worst case)
        max_lagrangian = max_possible_revenue + self.lmbd * self.rho  # When f=max_price, c=0 (best case)

        # Avoid division by zero in edge cases
        if max_lagrangian > min_lagrangian:
            normalized_L = (L_full - min_lagrangian) / (max_lagrangian - min_lagrangian)
        else:
            normalized_L = np.ones_like(L_full) * 0.5
            
        # Step 3: Update Hedge with losses (1 - normalized_L since we maximize)
        losses = 1 - normalized_L
        if(self.algorithm == 'hedge'):
            self.hedge.update(losses)
        else:   
            self.exp3.update(losses)

        # Step 4: Update Lagrangian multiplier using projected gradient ascent
        # λ_{t+1} = Proj_{[0, 1/ρ]}(λ_t - η * (ρ - c_t))
        # The projection ensures λ stays in feasible range
        constraint_violation = self.rho - c_t  # Positive if under-selling
        self.lmbd = np.clip(
            self.lmbd - self.eta * constraint_violation, 
            a_min=0,                    # λ ≥ 0 (dual feasibility)
            a_max=1/self.rho           # λ ≤ 1/ρ (reasonable upper bound)
        )
        
        # Step 5: Update inventory and performance tracking
        self.remaining_inventory -= c_t
        self.N_pulls[self.bid_index] += 1
        self.reward += f_t_full  # Accumulate counterfactual rewards

        # Step 6: Record history for analysis
        self.history['prices'].append(self.prices[self.bid_index])
        self.history['rewards'].append(f_t)
        self.history['purchases'].append(c_t)
        self.history['inventory'].append(self.remaining_inventory)

        # Increment time counter
        self.t += 1
    
    def get_reward(self):
        """Get cumulative rewards for all actions."""
        return self.reward
    
    def get_argmax_reward(self):
        """Get the index of the best-performing price."""
        return np.argmax(self.reward)
    
    def get_max_reward(self):
        """Get the maximum cumulative reward achieved."""
        best_idx = np.argmax(self.reward)
        return self.reward[best_idx]