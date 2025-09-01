import numpy as np

class ThompsonLikeAgent:
    """
    Revenue-Aware Thompson Sampling-based pricing agent for pricing problems with optional inventory constraints.
    
    Treats pricing as a multi-armed bandit problem where each price is an arm.
    Uses Beta-Bernoulli conjugate prior for binary rewards (sale/no sale).
    
    Key insights:
    1. Instead of sampling success probabilities and choosing the highest one,
       this agent samples expected revenues (price × probability) and chooses the price 
       with highest sampled revenue. This is the correct approach for pricing problems
       where the objective is revenue maximization, not success rate maximization.
    2. When inventory constraints are active (P and T provided), the agent considers
       remaining inventory and time to avoid overselling while maintaining revenue optimization.
    
    The inventory constraint is handled by:
    - Tracking remaining inventory and required selling rate
    - Penalizing price choices that would lead to demand exceeding capacity
    - Returning NaN when inventory is exhausted
    """
    def __init__(self, prices, P=None, T=None, alpha_prior=1.0, beta_prior=1.0):
        """
        Args:
            prices: List of possible prices to choose from
            P: Total inventory (number of products available). If None, no inventory constraint
            T: Time horizon (number of rounds). Required if P is provided
            alpha_prior: Prior alpha parameter for Beta distribution (successes + 1)
            beta_prior: Prior beta parameter for Beta distribution (failures + 1)
        """
        self.prices = prices
        self.K = len(prices)
        
        # Initialize Beta distribution parameters for each arm
        self.alpha = np.full(self.K, alpha_prior)  # Success count + prior
        self.beta = np.full(self.K, beta_prior)   # Failure count + prior
        
        # Statistics for tracking performance
        self.N_pulls = np.zeros(self.K)  # Number of times each price was used
        self.total_revenue = np.zeros(self.K)  # Total revenue for each price
        self.average_revenue = np.zeros(self.K)  # Average revenue for each price
        self.t = 0  # Current time step
        
        # Inventory management (optional)
        self.has_inventory_constraint = P is not None
        if self.has_inventory_constraint:
            if T is None:
                raise ValueError("Time horizon T must be provided when inventory P is specified")
            self.inventory = P               # Initial inventory
            self.remaining_inventory = P     # Current remaining inventory
            self.T = T                      # Time horizon
            self.rho = P / T               # Target selling rate
        else:
            self.inventory = None
            self.remaining_inventory = None
            self.T = None
            self.rho = None
            
        # History tracking
        self.history = {
            'prices': [],     # Selected prices over time
            'rewards': [],    # Observed revenues over time
            'purchases': [],  # Purchase indicators over time
            'inventory': []   # Inventory levels over time
        }
        
    def select_price(self):
        """
        Select price using Revenue-Aware Thompson Sampling algorithm with optional inventory constraints.
        
        Key difference from standard Thompson Sampling:
        1. Sample success probability from Beta distribution for each price
        2. Multiply by price to get sampled expected revenue 
        3. Select price with HIGHEST sampled revenue (not probability)
        4. If inventory constraints are active, consider remaining inventory and time
        """
        self.t += 1
        
        # Check inventory constraint if applicable
        if self.has_inventory_constraint and self.remaining_inventory < 1:
            # No inventory left - return highest price (arbitrary choice)
            self.current_price_idx = np.argmax(self.prices)
            return np.nan
        
        # Sample expected revenues for each price
        revenue_samples = np.zeros(self.K)
        demand_samples = np.zeros(self.K)
        
        for i in range(self.K):
            # Sample success probability from Beta distribution
            prob_sample = np.random.beta(self.alpha[i], self.beta[i])
            demand_samples[i] = prob_sample
            # Convert to revenue sample by multiplying with price
            revenue_samples[i] = self.prices[i] * prob_sample
        
        # Apply inventory constraint if active
        if self.has_inventory_constraint:
            # Calculate required selling rate for remaining time
            remaining_rounds = max(1, self.T - self.t + 1)
            current_rho = self.remaining_inventory / remaining_rounds
            
            # Filter out prices that would lead to demand exceeding our capacity
            # We use a conservative approach: if expected demand > current_rho, penalize heavily
            for i in range(self.K):
                if demand_samples[i] > current_rho * 1.2:  # 20% buffer
                    revenue_samples[i] *= 0.1  # Heavy penalty
        
        # Select the price with highest sampled revenue
        price_idx = np.argmax(revenue_samples)
        self.current_price_idx = price_idx
        return self.prices[price_idx]
    
    def update(self, sale_made, revenue):
        """
        Update agent's statistics after observing the outcome.
        Updates Beta distribution parameters based on success/failure.
        
        Args:
            sale_made: Boolean indicating if a sale was made
            revenue: Revenue obtained (used for tracking, not for Beta update)
        """
        idx = self.current_price_idx
        
        # Update Beta distribution parameters
        if sale_made:
            self.alpha[idx] += 1  # Increment success count
        else:
            self.beta[idx] += 1   # Increment failure count
        
        # Update statistics for tracking
        self.N_pulls[idx] += 1
        self.total_revenue[idx] += revenue
        self.average_revenue[idx] = self.total_revenue[idx] / self.N_pulls[idx]
        
        # Update inventory if constraint is active and purchase was made
        if self.has_inventory_constraint:
            if sale_made and self.remaining_inventory > 0:
                self.remaining_inventory -= 1
            elif sale_made and self.remaining_inventory <= 0:
                # This shouldn't happen with proper price selection, but handle gracefully
                revenue = 0
                sale_made = False
            
            # Record history
            self.history['prices'].append(self.prices[idx])
            self.history['rewards'].append(revenue)
            self.history['purchases'].append(sale_made)
            self.history['inventory'].append(self.remaining_inventory)
    
    def get_best_price(self):
        """
        Return the price with highest average revenue so far.
        """
        best_idx = np.argmax(self.average_revenue)
        return self.prices[best_idx], self.average_revenue[best_idx]
    
    def get_success_probabilities(self):
        """
        Return the current estimated success probability for each price.
        Uses the mean of the Beta distribution: alpha / (alpha + beta)
        
        Note: This is an alias for get_demand_probabilities() for backward compatibility.
        """
        return self.get_demand_probabilities()
    
    def get_expected_revenues(self):
        """
        Return the current estimated expected revenue for each price.
        This is what the algorithm actually optimizes for.
        """
        success_probs = self.get_success_probabilities()
        return np.array(self.prices) * success_probs
    
    def get_beta_parameters(self):
        """
        Return the current Beta distribution parameters for each arm.
        Useful for analysis and debugging.
        """
        return self.alpha.copy(), self.beta.copy()
    
    def get_confidence_intervals(self, confidence=0.95):
        """
        Return confidence intervals for success probabilities.
        
        Args:
            confidence: Confidence level (default 0.95 for 95% CI)
        """
        from scipy.stats import beta
        alpha_level = (1 - confidence) / 2
        
        lower_bounds = []
        upper_bounds = []
        
        for i in range(self.K):
            lower = beta.ppf(alpha_level, self.alpha[i], self.beta[i])
            upper = beta.ppf(1 - alpha_level, self.alpha[i], self.beta[i])
            lower_bounds.append(lower)
            upper_bounds.append(upper)
        
        return np.array(lower_bounds), np.array(upper_bounds)
    
    def get_inventory_status(self):
        """
        Return current inventory status.
        
        Returns:
            dict: Inventory information including remaining, used, and target rate
        """
        if not self.has_inventory_constraint:
            return {"status": "No inventory constraint"}
        
        remaining_rounds = max(1, self.T - self.t)
        current_rho = self.remaining_inventory / remaining_rounds if remaining_rounds > 0 else 0
        
        return {
            "initial_inventory": self.inventory,
            "remaining_inventory": self.remaining_inventory,
            "used_inventory": self.inventory - self.remaining_inventory,
            "target_rho": self.rho,
            "current_required_rho": current_rho,
            "remaining_rounds": remaining_rounds,
            "inventory_utilization": (self.inventory - self.remaining_inventory) / self.inventory if self.inventory > 0 else 0
        }
    
    def get_demand_probabilities(self):
        """
        Return the current estimated demand probability for each price.
        Uses the mean of the Beta distribution: alpha / (alpha + beta)
        """
        return self.alpha / (self.alpha + self.beta)