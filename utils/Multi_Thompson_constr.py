import numpy as np


import matplotlib.pyplot as plt

from scipy.stats import norm, multivariate_normal

from scipy.optimize import linear_sum_assignment, linprog

from scipy import optimize


class MultiThompsonSamplingPricingAgent:
    
    
    
    """
    Multi-product Thompson Sampling pricing agent with shared inventory constraint.
    - For each (product, price) uses Beta(alpha, beta) on purchase prob.
    - Samples prob per arm, sets:
        C[j,i] = sampled_prob (expected consumption)
        W[j,i] = price * sampled_prob (expected revenue)
    - Solves LP to get per-product price distribution Gamma under shared inventory constraint.
    - Samples one price per product from Gamm
    a and offers all products while inventory > 0.
    
    """
    
    
    def __init__(self, price_options, alpha_prior=1.0, beta_prior=1.0,
                 n_products=1, T=10000, inventory=100,
                 rho_penalty=1.0, use_pen_rho=False, confidence_bound=1.0):

        self.price_options = np.array(price_options, dtype=float)
        self.n_price_options = len(self.price_options)

        self.K = self.n_price_options
        self.n_products = n_products
        self.T = T

        # Shared inventory
        self.initial_inventory = int(inventory)
        self.remaining_inventory = int(inventory)
        self.rho = self.initial_inventory / max(1, self.T)

        # Optional params (kept for API parity)
        self.confidence_bound = confidence_bound
        self.rho_penalty = rho_penalty
        self.use_pen_rho = use_pen_rho
        
       # Beta priors per (product, price)
        self.alpha = np.full((self.n_products, self.K), alpha_prior, dtype=float)
        self.beta = np.full((self.n_products, self.K), beta_prior, dtype=float)

        # Stats
        self.N_pulls = np.zeros((self.n_products, self.K), dtype=float)
        self.total_revenue = np.zeros((self.n_products, self.K), dtype=float)
        self.average_revenue = np.zeros((self.n_products, self.K), dtype=float)

        self.t = 0  # round

        # Current state
        self.current_action_indices_map = {}  # {product_idx: price_idx}

        # History
        self.history = {
            'actions': [],            # (product_subset, price_indices)
            'revenues': [],
            'purchases': [],
            'inventory_levels': [],
            'price_distributions': [] # Gamma snapshot
        }

    def select_action(self):
        """
        Select price for each product using TS + LP (Gamma). Returns:
        - product_subset: list(range(n_products)) if inventory > 0, else []
        - prices: list of selected prices per product
        """
        if self.remaining_inventory < 1:
            self.current_action_indices_map = {}
            return [], []

        # Sample prob and build W, C
        W = np.zeros((self.n_products, self.K), dtype=float)
        C = np.zeros((self.n_products, self.K), dtype=float)
        for j in range(self.n_products):
            # vectorized Beta sampling per product
            prob_samples = np.random.beta(self.alpha[j], self.beta[j])
            C[j] = prob_samples
            W[j] = self.price_options * prob_samples

        # Solve LP for Gamma (per-product distributions)
        Gamma = self.compute_opt(W, C)
        self.history['price_distributions'].append(Gamma.copy())

        # Sample one price per product from Gamma
        selected_price_indices = []
        for j in range(self.n_products):
            prob_dist = Gamma[j].clip(min=0)
            s = prob_dist.sum()
            prob_dist = prob_dist / s if s > 0 else np.ones(self.K) / self.K
            price_idx = np.random.choice(self.K, p=prob_dist)
            selected_price_indices.append(price_idx)

        # Build action
        product_subset = list(range(self.n_products))
        price_options = [float(self.price_options[idx]) for idx in selected_price_indices]
        self.current_action_indices_map = {prod: idx for prod, idx in enumerate(selected_price_indices)}

        return product_subset, price_options

    def compute_opt(self, W, C):
        """
        LP:
          maximize sum_{j,i} gamma_{j,i} * W_{j,i}
          s.t.   sum_i gamma_{j,i} = 1   for each product j
                 sum_{j,i} gamma_{j,i} * C_{j,i} <= remaining_inventory / remaining_rounds
                 gamma_{j,i} >= 0
        """
        num_vars = self.n_products * self.K
        c = -W.flatten()

        remaining_rounds = max(1, self.T - self.t)
        current_rho_total = max(self.remaining_inventory / remaining_rounds, 0.0)

        A_ub = [C.flatten()]
        b_ub = [current_rho_total]

        A_eq = np.zeros((self.n_products, num_vars))
        b_eq = np.ones(self.n_products)
        for j in range(self.n_products):
            start = j * self.K
            A_eq[j, start:start + self.K] = 1.0

        bounds = [(0.0, 1.0) for _ in range(num_vars)]

        try:
            res = optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                                   bounds=bounds, method='highs')
            if res.success:
                gamma = res.x.reshape((self.n_products, self.K))
                # normalize row-wise to guard numerical drift
                for j in range(self.n_products):
                    row = np.maximum(gamma[j], 0.0)
                    s = row.sum()
                    gamma[j] = row / s if s > 0 else np.ones(self.K) / self.K
                return gamma
        except Exception as e:
            print(f"LP Error: {e}")

        # Fallback
        return np.ones((self.n_products, self.K)) / self.K

    def update(self, products_purchased, total_revenue):
        """
        Multi-product update after a round.
        products_purchased: list of product indices that actually purchased
        total_revenue: float (for logging)
        """
        if not self.current_action_indices_map:
            return

        purchased_set = set(products_purchased or [])
        for product_idx, price_idx in self.current_action_indices_map.items():
            sale = (product_idx in purchased_set)

            # Thompson update on purchase probability
            if sale:
                self.alpha[product_idx, price_idx] += 1.0
            else:
                self.beta[product_idx, price_idx] += 1.0

            # Stats
            self.N_pulls[product_idx, price_idx] += 1.0
            rev = self.price_options[price_idx] if sale else 0.0
            self.total_revenue[product_idx, price_idx] += rev
            self.average_revenue[product_idx, price_idx] = (
                self.total_revenue[product_idx, price_idx] / max(1.0, self.N_pulls[product_idx, price_idx])
            )

        # Update inventory
        units_sold = len(purchased_set)
        if self.remaining_inventory > 0:
            self.remaining_inventory = max(0, self.remaining_inventory - units_sold)

        # Log history
        product_subset = list(self.current_action_indices_map.keys())
        price_indices = list(self.current_action_indices_map.values())
        self.history['actions'].append((product_subset.copy(), price_indices.copy()))
        self.history['revenues'].append(float(total_revenue))
        self.history['purchases'].append(list(purchased_set))
        self.history['inventory_levels'].append(int(self.remaining_inventory))

        self.t += 1

    def get_success_probabilities(self):
        return self.alpha / (self.alpha + self.beta)

    def get_expected_revenues(self):
        probs = self.get_success_probabilities()
        return probs * self.price_options  # broadcast over price_options

    def get_best_strategy(self):
        """
        Best price per product by average revenue.
        """
        best = {}
        for j in range(self.n_products):
            idx = int(np.argmax(self.average_revenue[j]))
            best[j] = {
                'best_price': float(self.price_options[idx]),
                'avg_revenue': float(self.average_revenue[j, idx]),
                'purchase_prob': float(self.alpha[j, idx] / (self.alpha[j, idx] + self.beta[j, idx])),
                'pulls': int(self.N_pulls[j, idx]),
            }
        return best