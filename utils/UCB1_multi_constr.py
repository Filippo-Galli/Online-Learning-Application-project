# We assume that every agent can complete every task (all edges exist)

import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import norm, multivariate_normal

from scipy.optimize import linear_sum_assignment, linprog

from scipy import optimize



class UCBMatchingAgent:


    def __init__(self, n_products, price_options, inventory, T,
                 confidence_bound=1.0, rho_penalty=1.0, use_pen_rho=True, selection_method='sampling'):

        self.price_options=price_options
        self.n_products = n_products
        self.n_price_options=len(price_options)
        self.W_avg = np.zeros((n_products,self.n_price_options))
        self.N_pulls = np.zeros((n_products,self.n_price_options))

        self.T = T # not strictly necessary, you can use the anytime version of UCB
        self.t = 0

        # Inventory is now a single value representing total inventory
        self.initial_inventory = inventory
        self.remaining_inventory = inventory

        self.rho = self.initial_inventory / self.T

        self.confidence_bound = confidence_bound
        self.rho_penalty = rho_penalty
        self.use_pen_rho = use_pen_rho

        self.A_t = None
        self.rows_t = None
        self.cols_t = None

        self.selection_method = selection_method

        self.avg_revenue = np.zeros((n_products, self.n_price_options))
        self.avg_purchase_prob = np.zeros((n_products, self.n_price_options))

        # Current state
        self.current_action_indices_map = {}  # Maps product_idx to price_idx for offered products

        # History tracking
        self.history = {
            'actions': [], # Stores (product_subset, price_indices)
            'revenues': [],
            'purchases': [],
            'inventory_levels': [], # This will now store a single value per round
            'price_distributions': []  # Store computed price distributions
        }



    def select_action(self):
        """
        Select actions for each product using LP-based UCB with shared inventory constraints.

        1. Compute UCB bounds on revenue and LCB bounds on purchase probability.
        2. Solve LP to get optimal price distribution for all products considering shared inventory.
        3. Sample prices for each product according to the distribution.

        Returns:
            tuple: (product_subset, prices) where:
                - product_subset: List of product indices with inventory
                - prices: List of selected prices for each product
        """

        # If no inventory remaining, return empty action
        if self.remaining_inventory < 1:
            self.current_action_indices_map = {}
            return [], []

        # if an arm is unexplored, then the UCB is a large value
        W = np.zeros(self.W_avg.shape, dtype=float)
        C = np.zeros(self.avg_purchase_prob.shape, dtype=float)

        large_value = 1e7
        small_value = 1e-7

        W[self.N_pulls==0] = large_value
        C[self.N_pulls==0] = small_value

        mask = self.N_pulls>0

        sigma = self.confidence_bound * np.sqrt(2*np.log(max(1, self.t))/self.N_pulls[mask])

        W[mask] = self.W_avg[mask] + sigma
        C[mask] = np.maximum(self.avg_purchase_prob[mask] - sigma,0)

        Gamma = self.compute_opt(W, C)
        product_subset = []
        prices = []
        self.current_action_indices_map = {} # Reset for this round

        # NEW: Use Linear Sum Assignment to select prices
        #selected_price_indices = self._select_prices_lsa(Gamma, W)
        
        # Option 2: Use original sampling
        # selected_price_indices = self._select_prices_sampling(Gamma)
        
        # Option 3: Use hybrid approach
        # selected_price_indices = self._select_prices_hybrid(Gamma, W, explore_prob=0.1)
        
        # Option 4: Use unified method switcher
        selected_price_indices = self._select_prices(Gamma, W, method=self.selection_method)

        # Build the action using LSA-selected price indices
        for product_idx in range(self.n_products):
            price_idx = selected_price_indices[product_idx]
            
            # Add to action (all products are offered as long as total inventory > 0)
            product_subset.append(product_idx)
            prices.append(self.price_options[price_idx])
            self.current_action_indices_map[product_idx] = price_idx

        return product_subset, prices

    def _select_prices_lsa(self, Gamma, W):
        """
        Use Linear Sum Assignment to select optimal price for each product
        based on gamma-weighted revenues.
        
        Args:
            Gamma: Optimal probability distribution matrix (n_products x n_price_options)
            W: Upper confidence bounds on revenue (n_products x n_price_options)
        
        Returns:
            selected_price_indices: List of price indices for each product
        """
        # Create cost matrix: gamma-weighted revenues
        # Negative because LSA minimizes but we want to maximize revenue
        cost_matrix = -(Gamma * W)  # Element-wise multiplication
        
        # Solve assignment problem
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        return col_indices.tolist()

    # Alternative implementation with exploration probability
    def _select_prices_hybrid(self, Gamma, W, explore_prob=0.1):
        """
        Hybrid approach: use LSA with probability (1-explore_prob), 
        sampling with probability explore_prob.
        
        Args:
            Gamma: Optimal probability distribution matrix (n_products x n_price_options)
            W: Upper confidence bounds on revenue (n_products x n_price_options)
            explore_prob: Probability of using sampling instead of LSA
        
        Returns:
            selected_price_indices: List of price indices for each product
        """
        if np.random.random() < explore_prob:
            # Exploration: use original sampling approach
            selected_price_indices = []
            for product_idx in range(self.n_products):
                prob_dist = Gamma[product_idx, :]
                sum_prob = np.sum(prob_dist)
                if sum_prob > 0:
                    prob_dist /= sum_prob
                else:
                    prob_dist = np.ones(self.n_price_options) / self.n_price_options
                
                price_idx = np.random.choice(self.n_price_options, p=prob_dist)
                selected_price_indices.append(price_idx)
            
            return selected_price_indices
        else:
            # Exploitation: use LSA
            return self._select_prices_lsa(Gamma, W)
    
        
    def _select_prices_sampling(self, Gamma):
        """
        Original sampling approach: sample price for each product according to gamma distribution.
        
        Args:
            Gamma: Optimal probability distribution matrix (n_products x n_price_options)
        
        Returns:
            selected_price_indices: List of price indices for each product
        """
        selected_price_indices = []
        
        for product_idx in range(self.n_products):
            # Ensure the distribution sums to 1 for sampling
            prob_dist = Gamma[product_idx, :]
            sum_prob = np.sum(prob_dist)
            if sum_prob > 0:
                prob_dist /= sum_prob
            else:
                prob_dist = np.ones(self.n_price_options) / self.n_price_options  # Fallback uniform
            
            price_idx = np.random.choice(self.n_price_options, p=prob_dist)
            selected_price_indices.append(price_idx)
        
        return selected_price_indices
    
    # Method switcher for easy comparison
    def _select_prices(self, Gamma, W, method='lsa'):
        """
        Unified method to select prices using different strategies.
        
        Args:
            Gamma: Optimal probability distribution matrix (n_products x n_price_options)
            W: Upper confidence bounds on revenue (n_products x n_price_options)
            method: Selection method ('lsa', 'sampling', 'hybrid')
        
        Returns:
            selected_price_indices: List of price indices for each product
        """
        if method == 'lsa':
            return self._select_prices_lsa(Gamma, W)
        elif method == 'sampling':
            return self._select_prices_sampling(Gamma)
        elif method == 'hybrid':
            return self._select_prices_hybrid(Gamma, W, explore_prob=0.1)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'lsa', 'sampling', or 'hybrid'")

    def compute_opt(self, W, C):
        """
        Solve LP to find optimal price distribution for multiple products with shared inventory constraints.

        Formulation:
        maximize: sum_(k,i) gamma_(k,i) * W_(k,i) (total expected revenue)
        subject to: sum_i gamma_(k,i) = 1 for each product k (probability constraint per product)
                   sum_(k,i) gamma_(k,i) * C_(k,i) <= current_rho_total (shared inventory constraint)
                   gamma_(k,i) >= 0 (non-negativity)

        Args:
            W: Upper confidence bounds on revenue for each (product, price) combination (n_products x n_price_options)
            C: Lower confidence bounds on purchase probability for each (product, price) combination (n_products x n_price_options)

        Returns:
            gamma: Probability distribution over prices for each product (n_products x n_price_options)
        """

        # Number of variables is n_products * n_price_options
        num_vars = self.n_products * self.n_price_options

        # Flatten W and C for LP
        c = -W.flatten()

        # Compute current required selling rate for the shared inventory
        remaining_rounds = max(1, self.T - self.t)
        current_rho_total = max(self.remaining_inventory / remaining_rounds, 0)

        # Inequality constraints (shared inventory)
        A_ub = [C.flatten()]
        b_ub = [current_rho_total]

        # Equality constraints (probability distribution for each product)
        A_eq = np.zeros((self.n_products, num_vars))
        b_eq = np.ones(self.n_products)

        for product_idx in range(self.n_products):
            start_idx = product_idx * self.n_price_options
            end_idx = start_idx + self.n_price_options
            A_eq[product_idx, start_idx:end_idx] = 1

        # Bounds for each variable (gamma_ki between 0 and 1)
        bounds = [(0, 1) for _ in range(num_vars)]

        # Solve linear program
        try:
            res = optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                                 bounds=bounds, method='highs')

            if res.success:
                gamma_flat = res.x
                # Reshape gamma back to (n_products, n_price_options)
                gamma = gamma_flat.reshape((self.n_products, self.n_price_options))

                # Ensure valid probability distribution for each product
                for product_idx in range(self.n_products):
                    gamma[product_idx, :] = np.maximum(gamma[product_idx, :], 0)
                    sum_gamma = np.sum(gamma[product_idx, :])
                    if sum_gamma > 0:
                         gamma[product_idx, :] /= sum_gamma
                    else:
                         # Fallback: uniform distribution for this product
                         gamma[product_idx, :] = np.ones(self.n_price_options) / self.n_price_options

                return gamma
            else:
                # Fallback: uniform distribution for all products
                return np.ones((self.n_products, self.n_price_options)) / self.n_price_options
        except Exception as e:
            print(f"LP Error: {e}")
            # Fallback: uniform distribution for all products
            return np.ones((self.n_products, self.n_price_options)) / self.n_price_options


    def update(self, products_purchased, total_revenue):
        """
        Update agent statistics based on observed outcome.

        Args:
            products_purchased: List of products actually purchased in this round
            total_revenue: Total revenue obtained in this round
        """
        if not self.current_action_indices_map:
            return

        # Update statistics for each product that was offered (all products are offered if total inventory > 0)
        # Iterate through the products that were *offered* in this round
        for product_idx, price_idx in self.current_action_indices_map.items():

            # Update pull count for this product-price arm
            self.N_pulls[product_idx, price_idx] += 1

            # Determine if this specific product was purchased in this round
            was_purchased = product_idx in products_purchased
            price = self.price_options[price_idx] # Assuming price_options is globally accessible or passed

            # Update average revenue for this product-price arm
            # Revenue contribution is the price if purchased, 0 otherwise
            revenue_contribution = price if was_purchased else 0.0
            self.W_avg[product_idx, price_idx] += (
                revenue_contribution - self.W_avg[product_idx, price_idx]
            ) / self.N_pulls[product_idx, price_idx]

            # Update average purchase probability for this product-price arm
            purchase_indicator = 1.0 if was_purchased else 0.0
            self.avg_purchase_prob[product_idx, price_idx] += (
                purchase_indicator - self.avg_purchase_prob[product_idx, price_idx]
            ) / self.N_pulls[product_idx, price_idx]

        # Update total remaining inventory based on the number of products purchased in this round
        num_purchased_this_round = len(products_purchased)
        if self.remaining_inventory > 0:
            self.remaining_inventory -= num_purchased_this_round
            self.remaining_inventory = max(0, self.remaining_inventory) # Ensure inventory doesn't go below zero


        # Record history
        # Store the sampled action (product_subset and price indices)
        product_subset = list(self.current_action_indices_map.keys())
        price_indices = list(self.current_action_indices_map.values())
        self.history['actions'].append((product_subset.copy(), price_indices.copy()))

        self.history['revenues'].append(total_revenue)
        self.history['purchases'].append(products_purchased.copy())
        self.history['inventory_levels'].append(self.remaining_inventory) # Store the single inventory value

        # Increment time
        self.t += 1

    def get_best_strategy(self):
        """
        Return the best strategy learned so far for each product based on average revenue.

        Returns:
            Dictionary with best price and performance for each product
        """
        best_strategies = {}

        for product_idx in range(self.n_products):
            # Find the price index with the highest average revenue for this product
            best_price_idx = np.argmax(self.W_avg[product_idx, :])
            best_price = price_options[best_price_idx] # Assuming price_options is globally accessible
            best_revenue = self.W_avg[product_idx, best_price_idx]
            best_prob = self.avg_purchase_prob[product_idx, best_price_idx]

            best_strategies[product_idx] = {
                'best_price': best_price,
                'avg_revenue': best_revenue,
                'purchase_prob': best_prob,
                'pulls': self.N_pulls[product_idx, best_price_idx]
            }

        return best_strategies