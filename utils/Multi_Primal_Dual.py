import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import multivariate_normal
import itertools
plt.rcParams['figure.figsize'] = (16, 9)
np.random.seed(42)
import sys

sys.path.append('../utils')
from Hedge import HedgeAgent
from EXP3_P import Exp3Agent


class MultiProductPrimalDualAgent:
    """
    Multi-product pricing agent using a primal-dual approach with per-product learners.

    This agent is designed for an online pricing problem with multiple products and an overall
    inventory constraint. It combines:
    1. A primal-dual update for the dual variable λ controlling the inventory constraint
    2. A separate learner for each product to choose prices (arms) from a discrete grid
    3. Proper reward computation based on Lagrangian with inventory penalty
    4. Inventory-aware pricing that adapts to remaining inventory
    5. History tracking for analysis (prices, purchases, revenue, inventory, λ)

    Attributes:
        prices_grid (ndarray): Matrix (n_products x K) of discrete price options for each product
        n_products (int): Number of products
        K (int): Number of price options (arms) per product
        T (int): Time horizon
        P (int): Total inventory across all products
        eta (float): Learning rate for λ updates (dual variable)
        learners (list): One learner instance per product
        rho (float): Average inventory allowed per round (P/T)
        lmbd (float): Dual variable for inventory constraint
        remaining_inventory (int): Inventory remaining at current step
        history (dict): Records time series of key variables
        cum_reward_per_arm (ndarray): Cumulative unnormalized reward per arm per product
        max_price (float): Maximum price in the price grid
        algorithm (str): Type of learning algorithm ('Hedge' or 'Exp3')
        current_round (int): Current round number
    """

    def __init__(self, prices_grid, T, P, eta, lambda_init=None, algorithm='Exp3'):
        """
        Initialize the primal-dual multi-product pricing agent.

        Args:
            prices_grid (ndarray): Array of shape (n_products, K) containing price options for each product
            T (int): Time horizon (number of rounds)
            P (int): Total inventory available across all products
            eta (float): Learning rate for updating the dual variable λ
            lambda_init (float, optional): Initial λ value; defaults to 1.0 if None
            algorithm (str): Learning algorithm - 'Hedge' for full feedback or 'Exp3' for bandit feedback
        """

        # Store parameters
        self.prices_grid = prices_grid
        self.n_products, self.K = prices_grid.shape
        self.T = T
        self.P = P
        self.eta = float(eta)
        self.algorithm = algorithm
        self.current_round = 0

        # Initialize learners based on algorithm choice
        if algorithm == 'Hedge':
            # Hedge requires full feedback - we'll compute counterfactual rewards
            lr = np.sqrt(max(1e-12, np.log(self.K)) / max(1, T))
            self.learners = [HedgeAgent(self.K, lr) for _ in range(self.n_products)]
        else:  # Exp3
            # EXP3 uses bandit feedback - only selected arm reward
            lr = np.sqrt(max(1e-12, np.log(self.K)) / max(1, T * self.K))
            exploration_rate = min(1.0, np.sqrt(self.K * np.log(self.K) / max(1, T)))
            self.learners = [Exp3Agent(self.K, lr, exploration_rate) for _ in range(self.n_products)]

        # Inventory constraint parameters
        self.rho = self.P / float(self.T)
        self.lmbd = 1.0 if (lambda_init is None) else float(lambda_init)
        self.remaining_inventory = int(self.P)

        # Tracking
        self.history = {"prices": [], "purchases": [], "revenue": [], "inventory": [], "lambda": []}
        self.cum_reward_per_arm = np.zeros((self.n_products, self.K), dtype=float)
        self.max_price = float(self.prices_grid.max())
        
        # Store last action for updates
        self.last_arms = None
        self.last_prices = None

    def bid(self):
        """
        Select prices for all products based on the current learners.
        
        Incorporates inventory-aware pricing: when inventory is low relative to 
        remaining time, prices are adjusted upward to be more conservative.

        Returns:
            ndarray: Array of length n_products with the chosen prices for each product
        """
        # Get base arms from learners
        arms = np.array([learner.pull_arm() for learner in self.learners], dtype=int)
        
        # Inventory-aware adjustment: increase prices when inventory is scarce
        remaining_time = max(1, self.T - self.current_round)
        inventory_ratio = self.remaining_inventory / max(1, remaining_time)
        expected_ratio = self.rho
        
        # If inventory is running low, bias toward higher prices
        if inventory_ratio < 0.5 * expected_ratio and self.remaining_inventory > 0:
            # Probability of choosing higher price arms increases when inventory is low
            for i in range(self.n_products):
                if arms[i] < self.K - 1 and np.random.random() < 0.3:
                    arms[i] = min(self.K - 1, arms[i] + 1)
        
        # Extract actual prices
        prices = self.prices_grid[np.arange(self.n_products), arms]
        
        # Store for update
        self.last_arms = arms
        self.last_prices = prices
        
        return prices
    
    def update(self, purchases, per_product_revenue, valuations=None):
        """
        Update learners and dual variable λ after observing the results of the last bid.

        Args:
            purchases (ndarray): Number of units sold per product in the last round
            per_product_revenue (ndarray): Revenue per product in the last round
            valuations (ndarray, optional): Actual customer valuations for better counterfactual estimation
        """
        self.current_round += 1
        
        # Compute Lagrangian rewards: revenue - λ * inventory_consumption
        total_consumption = float(purchases.sum())
        
        # The Lagrangian for each product: L_i = revenue_i - λ * consumption_i
        lagrangian_rewards = per_product_revenue - self.lmbd * purchases
        
        # Normalize rewards to [0, 1] for the learning algorithms
        # Bounds: min = -λ (when no revenue, full consumption), max = max_price (when full revenue, no consumption)
        reward_min = -self.lmbd
        reward_max = self.max_price
        reward_range = max(1e-9, reward_max - reward_min)
        
        normalized_rewards = np.clip((lagrangian_rewards - reward_min) / reward_range, 0.0, 1.0)
        
        # Update learners based on algorithm type
        if self.algorithm == 'Hedge':
            # Hedge needs full feedback: compute counterfactual rewards for all arms
            for i in range(self.n_products):
                # Compute what reward would have been for each possible price
                counterfactual_rewards = np.zeros(self.K)
                
                # Use actual valuations if available, otherwise estimate
                if valuations is not None:
                    customer_valuation = valuations[i]
                else:
                    # Fallback estimation if valuations not provided
                    if purchases[i] > 0:
                        customer_valuation = self.last_prices[i] + 0.1
                    else:
                        customer_valuation = max(0.0, self.last_prices[i] - 0.1)
                
                for k in range(self.K):
                    price_k = self.prices_grid[i, k]
                    # Would customer have bought at this price?
                    would_buy = (customer_valuation >= price_k)
                    
                    if would_buy:
                        revenue_k = price_k
                        consumption_k = 1.0
                    else:
                        revenue_k = 0.0
                        consumption_k = 0.0
                    
                    lagrangian_k = revenue_k - self.lmbd * consumption_k
                    counterfactual_rewards[k] = np.clip((lagrangian_k - reward_min) / reward_range, 0.0, 1.0)
                
                self.learners[i].update(counterfactual_rewards)
                
        else:  # Exp3
            # EXP3 needs only the reward for the selected arm
            for i in range(self.n_products):
                reward_vector = np.zeros(self.K)
                reward_vector[self.last_arms[i]] = normalized_rewards[i]
                self.learners[i].update(reward_vector)
        
        # Update dual variable λ (CORRECTED DIRECTION)
        # λ should increase when constraint is violated (total_consumption > rho)
        constraint_violation = total_consumption - self.rho
        self.lmbd = max(0.0, self.lmbd + self.eta * constraint_violation)
        
        # Cap λ to prevent numerical issues
        max_lambda = 10.0 / max(self.rho, 1e-9)
        self.lmbd = min(self.lmbd, max_lambda)
        
        # Update remaining inventory and tracking
        self.remaining_inventory = max(0, self.remaining_inventory - int(total_consumption))
        
        # Store history
        self.history["prices"].append(self.last_prices.copy())
        self.history["purchases"].append(purchases.copy())
        self.history["revenue"].append(float(per_product_revenue.sum()))
        self.history["inventory"].append(int(self.remaining_inventory))
        self.history["lambda"].append(float(self.lmbd))

