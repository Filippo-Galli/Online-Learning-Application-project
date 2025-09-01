import numpy as np
from scipy import optimize

class ThompsonLikeAgent:
    """Fixed Thompson agent with correct interface and missing attributes"""
    
    def __init__(self, P, T, prices, alpha_prior=1.0, beta_prior=1.0):
        self.prices = prices
        self.K = len(prices)
        self.P = P
        self.T = T
        
        # Initialize Beta distribution parameters
        self.alpha = np.full(self.K, alpha_prior)
        self.beta = np.full(self.K, beta_prior)
        
        # Statistics tracking (compatible with UCB interface)
        self.avg_revenue = np.zeros(self.K)
        self.avg_demand = np.zeros(self.K)
        self.N_pulls = np.zeros(self.K)
        
        # Inventory management
        self.remaining_inventory = P
        self.inventory = P
        self.rho = P / T
        
        # Fix missing attributes
        self.use_pen_rho = False  # This was missing!
        self.rho_penalty = 1.0
        
        self.t = 0
        self.current_price_idx = None
        
        self.history = {
            'prices': [],
            'rewards': [],
            'purchases': [],
            'inventory': []
        }
    
    def select_price(self):
        if self.remaining_inventory < 1:
            self.current_price_idx = np.argmax(self.prices)
            return np.nan
            
        if self.t < self.K:
            self.current_price_idx = self.t
        else:
            # Sample from Beta distributions
            sampled_probs = np.random.beta(self.alpha, self.beta)
            sampled_revenues = self.prices * sampled_probs
            
            # Use the same LP optimization as UCB but with samples
            gamma_t = self.compute_opt(sampled_revenues, sampled_probs)
            self.current_price_idx = np.random.choice(self.K, p=gamma_t)
        
        return self.prices[self.current_price_idx]
    
    def compute_opt(self, f_ucbs, c_lcbs):
        """Same as UCB optimization but with sampled values"""
        if np.all(c_lcbs <= 1e-10):
            gamma = np.zeros(len(f_ucbs))
            gamma[np.argmax(f_ucbs)] = 1.0
            return gamma
        
        c = -f_ucbs
        remaining_rounds = max(1, self.T - self.t)
        current_rho = max(self.remaining_inventory / remaining_rounds, 0)
        
        A_ub = [c_lcbs]
        b_ub = [current_rho]
        A_eq = [np.ones(self.K)]
        b_eq = [1]
        
        try:
            res = optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, 1))
            if res.success:
                gamma = res.x
                gamma = np.maximum(gamma, 0)
                gamma = gamma / np.sum(gamma) if np.sum(gamma) > 0 else np.ones(self.K) / self.K
                return gamma
            else:
                return np.ones(self.K) / self.K
        except:
            return np.ones(self.K) / self.K
    
    def update(self, revenue, sale_made):  # CORRECT signature!
        """Fixed update method with correct parameter order"""
        idx = self.current_price_idx
        
        # Update Beta distribution
        if sale_made:
            self.alpha[idx] += 1
        else:
            self.beta[idx] += 1
        
        # Update statistics (same as UCB)
        self.N_pulls[idx] += 1
        self.avg_revenue[idx] += (revenue - self.avg_revenue[idx]) / self.N_pulls[idx]
        
        sale_indicator = 1.0 if sale_made else 0.0
        self.avg_demand[idx] += (sale_indicator - self.avg_demand[idx]) / self.N_pulls[idx]
        
        # Update inventory
        if sale_made and self.remaining_inventory > 0:
            self.remaining_inventory -= 1
        elif sale_made and self.remaining_inventory <= 0:
            revenue = 0
            sale_made = False
        
        # Record history
        self.history['prices'].append(self.prices[idx])
        self.history['rewards'].append(revenue)
        self.history['purchases'].append(sale_made)
        self.history['inventory'].append(self.remaining_inventory)
        
        self.t += 1
    
    def get_best_price(self):
        best_idx = np.argmax(self.avg_revenue)
        return self.prices[best_idx], self.avg_revenue[best_idx]
