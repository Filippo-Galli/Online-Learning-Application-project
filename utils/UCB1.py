import numpy as np

class UCB1PricingAgent:
    """
    UCB1-based pricing agent that ignores inventory constraints.
    Treats pricing as a multi-armed bandit problem where each price is an arm.
    """
    def __init__(self, prices, confidence_bound=2.0):
        """
        Args:
            prices: List of possible prices to choose from
            confidence_bound: UCB confidence parameter (higher = more exploration)
        """
        self.prices = prices
        self.K = len(prices)
        self.confidence_bound = confidence_bound
        
        # Initialize statistics
        self.N_pulls = np.zeros(self.K)  # Number of times each price was used
        self.total_revenue = np.zeros(self.K)  # Total revenue for each price
        self.average_revenue = np.zeros(self.K)  # Average revenue for each price
        self.t = 0  # Current time step
        
    def select_price(self):
        """
        Select price using UCB1 algorithm.
        """
        self.t += 1
        
        # Calculate UCB values for each price
        ucb_values = np.zeros(self.K)
        for i in range(self.K):
            if self.N_pulls[i] == 0:
                ucb_values[i] = float('inf')  # Unplayed arms get infinite UCB
            else:
                # UCB formula: average_reward + confidence_bound * sqrt(ln(t) / n_i)
                confidence_radius = self.confidence_bound * np.sqrt(2*np.log(self.t) / self.N_pulls[i])
                ucb_values[i] = self.average_revenue[i] + confidence_radius
            
            # Select the price with highest UCB value
            price_idx = np.argmax(ucb_values)
        
        self.current_price_idx = price_idx
        return self.prices[price_idx]
    
    def update(self, sale_made, revenue):
        """
        Update agent's statistics after observing the outcome.
        """
        idx = self.current_price_idx
        self.N_pulls[idx] += 1
        self.total_revenue[idx] += revenue
        self.average_revenue[idx] = self.total_revenue[idx] / self.N_pulls[idx]
    
    def get_best_price(self):
        """
        Return the price with highest average revenue so far.
        """
        best_idx = np.argmax(self.average_revenue)
        return self.prices[best_idx], self.average_revenue[best_idx]