import numpy as np

class ThompsonSamplingPricingAgent:
    """
    Revenue-Aware Thompson Sampling-based pricing agent for pricing problems.
    
    Treats pricing as a multi-armed bandit problem where each price is an arm.
    Uses Beta-Bernoulli conjugate prior for binary rewards (sale/no sale).
    
    Key insight: Instead of sampling success probabilities and choosing the highest one,
    this agent samples expected revenues (price × probability) and chooses the price 
    with highest sampled revenue. This is the correct approach for pricing problems
    where the objective is revenue maximization, not success rate maximization.
    """
    def __init__(self, prices, alpha_prior=1.0, beta_prior=1.0):
        """
        Args:
            prices: List of possible prices to choose from
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
        
    def select_price(self):
        """
        Select price using Revenue-Aware Thompson Sampling algorithm.
        
        Key difference from standard Thompson Sampling:
        1. Sample success probability from Beta distribution for each price
        2. Multiply by price to get sampled expected revenue 
        3. Select price with HIGHEST sampled revenue (not probability)
        """
        self.t += 1
        
        # Sample expected revenues for each price
        revenue_samples = np.zeros(self.K)
        for i in range(self.K):
            # Sample success probability from Beta distribution
            prob_sample = np.random.beta(self.alpha[i], self.beta[i])
            # Convert to revenue sample by multiplying by price
            revenue_samples[i] = self.prices[i] * prob_sample
        
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
        """
        return self.alpha / (self.alpha + self.beta)
    
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