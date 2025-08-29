import numpy as np
from scipy import optimize 


class UCBLikeAgent:
    """
    UCB1-based agent for dynamic pricing with inventory constraints.
    
    This agent implements a variant of the UCB1 algorithm adapted for:
    1. Dynamic pricing (instead of traditional MAB rewards)
    2. Inventory constraints (limited number of products to sell)
    3. Dual optimization: maximize revenue while respecting inventory constraint
    
    The algorithm maintains upper confidence bounds on revenue (f_UCB) and 
    lower confidence bounds on demand probability (c_LCB), then solves a 
    linear program to find the optimal price distribution.
    """
    
    def __init__(self, P, T, prices, confidence_bound=1, rho_penalty=1.0, use_pen_rho=True):
        """
        Initialize the UCB agent for constrained dynamic pricing.
        
        Args:
            P: Total inventory (number of products available)
            T: Time horizon (number of rounds)
            prices: List of available prices to choose from
            confidence_bound: Confidence parameter for UCB/LCB bounds
            rho_penalty: Penalty factor for inventory constraint (>1 = more conservative)
        """
        k = len(prices)

        # Environment parameters
        self.use_pen_rho=use_pen_rho
        self.prices = prices  # Available price options
        self.K = k           # Number of price arms
        self.T = T           # Total number of rounds
        self.confidence_bound = confidence_bound  # UCB confidence parameter
        
        # Current state
        self.current_price_idx = None  # Index of currently selected price
        self.t = 0                     # Current round number
        
        # Statistics for each price arm
        self.avg_revenue = np.zeros(k)  # Average revenue per price
        self.avg_demand = np.zeros(k)   # Average demand probability per price
        self.N_pulls = np.zeros(k)      # Number of times each price was selected
        
        # Inventory management
        self.inventory = P               # Initial inventory
        self.remaining_inventory = P     # Current remaining inventory
        self.rho = P / T                # Target selling rate
        self.rho_penalty = rho_penalty  # Penalty factor for inventory constraint
        
        # History tracking
        self.history = {
            'prices': [],     # Selected prices over time
            'rewards': [],    # Observed revenues over time
            'purchases': [],  # Purchase indicators over time
            'inventory': []   # Inventory levels over time
        }
    
    def select_price(self):
        """
        Select the next price using UCB1 with inventory constraints.
        
        Strategy:
        1. If inventory is empty, return NaN (no meaningful price)
        2. First K rounds: explore each price once (initialization)
        3. Subsequent rounds: solve LP with UCB revenue bounds and LCB demand bounds
        
        Returns:
            Selected price, or np.nan if no inventory remaining
        """
        # No inventory left - cannot make meaningful pricing decisions
        if self.remaining_inventory < 1:
            self.current_price_idx = np.argmax(self.prices)  # Arbitrary selection
            return np.nan
            
        # Exploration phase: try each price at least once
        if self.t < self.K:
            self.current_price_idx = self.t
        # Exploitation-Exploration phase: use UCB with inventory constraints
        
        else:            
            # Compute upper confidence bounds on revenue for each price
            confidence_radius = self.confidence_bound * np.sqrt(2 * np.log(self.t) / np.maximum(self.N_pulls, 1))
            f_ucbs = self.avg_revenue + confidence_radius
            
            demand_confidence_radius = self.confidence_bound * np.sqrt(2 * np.log(self.t) / np.maximum(self.N_pulls, 1))
            c_lcbs = np.maximum(0,self.avg_demand - demand_confidence_radius)
            
            # Solve linear program to get optimal price distribution
            gamma_t = self.compute_opt(f_ucbs, c_lcbs)
            
            # Sample price according to computed distribution
            self.current_price_idx = np.random.choice(self.K, p=gamma_t)

        return self.prices[self.current_price_idx]

    def compute_opt(self, f_ucbs, c_lcbs): 
        """
        Solve constrained optimization problem to find optimal price distribution.
        
        Formulation:
        maximize: sum_i gamma_i * f_ucb_i (expected revenue)
        subject to: sum_i gamma_i * c_lcb_i <= rho_penalty * current_rho (inventory constraint)
                   sum_i gamma_i = 1 (probability constraint)
                   gamma_i >= 0 (non-negativity)
        
        where current_rho is the required selling rate to use remaining inventory.
        
        Args:
            f_ucbs: Upper confidence bounds on revenue for each price
            c_lcbs: Lower confidence bounds on demand probability for each price
            
        Returns:
            gamma: Probability distribution over prices
        """
        # Handle edge case: if no positive demand expected, choose highest revenue price
        if np.all(c_lcbs <= 1e-10):
            gamma = np.zeros(len(f_ucbs))
            gamma[np.argmax(f_ucbs)] = 1.0
            return gamma
        
        # Convert to minimization problem (negate revenues)
        c = -f_ucbs
        
        # Compute current required selling rate
        remaining_rounds = max(1, self.T - self.t)
        current_rho = max(self.remaining_inventory / remaining_rounds, 0)
        
        # Linear program constraints
        A_ub = [c_lcbs]      # Inventory constraint coefficients
        
        if self.use_pen_rho:
            
            # Apply penalty to make constraint more conservative, but not too tight
            # If inventory is high relative to time remaining, relax the constraint
            inventory_ratio = self.remaining_inventory / self.inventory
            time_ratio = (self.T - self.t) / self.T
            
            if inventory_ratio > 0.5 and time_ratio > 0.5:
                # Early stages with plenty of inventory: relax constraint
                penalty_factor = self.rho_penalty * 1.5
            elif inventory_ratio < 0.1:
                # Low inventory: tighten constraint
                penalty_factor = self.rho_penalty * 0.5
            else:
                penalty_factor = self.rho_penalty
                
            penalized_rho = current_rho * penalty_factor
            
            # If the constraint is too tight (lower than minimum demand), relax it
            min_demand = np.min(c_lcbs[c_lcbs > 0]) if np.any(c_lcbs > 0) else 0
            
            if penalized_rho < min_demand * 0.5:
                penalized_rho = min_demand * 0.8
            
            b_ub = [penalized_rho]       # Inventory constraint bound
        
        
        else:

            b_ub = [current_rho]       # Inventory constraint bound

        A_eq = [np.ones(self.K)]     # Probability constraint coefficients  
        b_eq = [1]                   # Probability constraint bound
        
        # Solve linear program
        try:
            res = optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, 1))
            if res.success:
                gamma = res.x
                # Ensure valid probability distribution
                gamma = np.maximum(gamma, 0)
                gamma = gamma / np.sum(gamma) if np.sum(gamma) > 0 else np.ones(self.K) / self.K
                return gamma
            else:
                # Fallback: uniform distribution
                return np.ones(self.K) / self.K
        except:
            # Fallback: uniform distribution
            return np.ones(self.K) / self.K
    
    def update(self, reward, purchased):
        """
        Update agent's statistics based on observed outcome.
        
        Args:
            reward: Revenue obtained (price if purchased, 0 otherwise)
            purchased: Boolean indicating if purchase was made
        """
        idx = self.current_price_idx
        
        # Update pull count
        self.N_pulls[idx] += 1
        
        # Update average revenue using incremental mean formula
        self.avg_revenue[idx] += (reward - self.avg_revenue[idx]) / self.N_pulls[idx]
        
        # Update average demand probability
        purchased_indicator = 1.0 if purchased else 0.0
        self.avg_demand[idx] += (purchased_indicator - self.avg_demand[idx]) / self.N_pulls[idx]
        
        # Update inventory only if purchase was actually made and inventory available
        if purchased and self.remaining_inventory > 0:
            self.remaining_inventory -= 1
        elif purchased and self.remaining_inventory <= 0:
            # This shouldn't happen with proper price selection, but handle gracefully
            reward = 0
            purchased = False
        
        # Record history
        self.history['prices'].append(self.prices[idx])
        self.history['rewards'].append(reward)
        self.history['purchases'].append(purchased)
        self.history['inventory'].append(self.remaining_inventory)
        
        # Increment time
        self.t += 1

    def get_best_price(self):
        """
        Return the price with highest average revenue observed so far.
        
        Returns:
            tuple: (best_price, best_average_revenue)
        """
        best_idx = np.argmax(self.avg_revenue)
        return self.prices[best_idx], self.avg_revenue[best_idx]