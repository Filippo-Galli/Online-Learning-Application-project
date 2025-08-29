import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
from scipy.optimize import linear_sum_assignment, linprog
from scipy import optimize





class UCBSWAgent:

    def __init__(self, n_products, price_options, inventory, T, W,
                 confidence_bound=1.0, rho_penalty=1.0, use_pen_rho=True, selection_method='sampling', rng = None):

        self.rng = rng if rng else np.random.default_rng()

        #Number of rounds
        self.T = T

        #Window size
        self.W = W

        #Current round
        self.t = 0
        #Number of products
        self.n = n_products
        #Number of possible prices
        self.n_price = len(price_options)

        #Total initial inventory
        self.B = inventory

        self.remaining_inventory = inventory

        #Map product -> last price chosen for the product
        self.current_action = np.zeros(self.n, dtype=int)

        #Sliding window for the rewards of each price for each product and price
        self.cache_f = np.full((n_products, W, self.n_price), np.nan)    #3D Matrix [product, t in the window, price]

        #Sliding window for the purcase probabilities for each product and price
        self.cache_c = np.full((n_products, W, self.n_price), np.nan)    #3D Matrix [product, t in the window, price]

        self.rho = self.B / self.T

         # equality constraints: sum of arm probabilities = 1 for each product
        self.A_eq = np.zeros((self.n, self.n*self.n_price))
        for dim in range(self.n):
            self.A_eq[dim,dim*self.n_price:(dim+1)*self.n_price] = 1
        self.b_eq = np.ones(self.n)

        #Value that tells me if we have fulled the sliding window for the first time
        self.if_full = False

        #Current position in the sliding window
        self.current_window_index = 0

        #print(self.cache_f)
        #print(self.cache_c)
    

    #For each product select a price. Return the list of prices for each product
    def pull_arm(self):



        #No inventory left, return an empty action
        if self.remaining_inventory < self.n:
            self.current_action = {}
            return self.current_action

        #First n_price round, pull each product with the t price
        if self.t < self.n_price:
            #print("We are in the initial phase, exploring each arm uniformly")
            self.current_action = np.full(self.n, self.t, dtype=int)
            #print("Subset: ", price_subset)
            return self.current_action

        #large_value = 1e7
        small_value = 1e-7

        #Number of times price p for product x was picked in the last W rounds
        n_pulls_last_w = np.sum(~np.isnan(self.cache_f), axis=1)

        #Print cache_f for first product
        """for i in range(self.n_price):
            for j in range(self.n_price):
                print(f"Window {i}, Price option {j}, Rewards in the window: {self.cache_f[0,i,j]}")"""

        #Exploration term of the CB, we add a small value to avoid division by zero
        sigma = np.sqrt(2*np.log(self.W)/(n_pulls_last_w+small_value))
        f_ucbs = np.nanmean(self.cache_f, axis=1) + sigma
        c_lcbs = np.nanmean(self.cache_c, axis=1) - sigma

        """print("Ucb on the reward for every arm")
        for i in range(self.n):
            for j in range(self.n_price):
                print(f"Product {i}, Price option {j}, Expected reward: {f_ucbs[i,j]}")"""

        gamma = self.compute_opt(f_ucbs, c_lcbs)

        #Sample prices for each product based on the computed distribution gamma
        for product in range(self.n):
            #Index of the price
            self.current_action[product] = self.rng.choice(self.n_price, p=gamma[product])

        return self.current_action


    def update(self, f_t, c_t):
        
        #print("Current window index: ", self.current_window_index)
        
        for product in range(self.n):
            if self.if_full:
                #Set all the revenue to null
                self.cache_f[product, self.current_window_index] = np.nan
                self.cache_c[product, self.current_window_index] = np.nan

            #print("Price index: ", self.current_action[product])            
            self.cache_f[product, self.current_window_index, self.current_action[product]] = f_t[product]
            self.cache_c[product, self.current_window_index, self.current_action[product]] = c_t[product]
            self.B -= c_t[product]
        self.rho = self.B/(self.T-self.t)
        self.t += 1

        #We fulled the window for the first time
        if (self.current_window_index +1) == self.W and self.if_full == False:
            self.if_full = True
            
        self.current_window_index = (self.current_window_index + 1) % self.W
            

    def compute_opt(self, f_ucbs, c_lcbs):
    

        # Flatten W and C for LP
        c = -f_ucbs.flatten()

        # Inequality constraints (shared inventory)
        A_ub = [c_lcbs.flatten()]
        b_ub = [self.rho]

        #Coefficients for equality constraints
        A_eq = self.A_eq

        # Bounds for each variable (gamma_ki between 0 and 1)
        bounds = [(0, 1) for _ in range(self.n*self.n_price)]

        # Solve linear program
        try:
            res = optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=self.b_eq,
                                 bounds=bounds, method='highs')

            if res.success:
                gamma_flat = res.x
                # Reshape gamma back to (n_products, n_price_options)
                gamma = gamma_flat.reshape((self.n, self.n_price))

                # Ensure valid probability distribution for each product
                for product_idx in range(self.n):
                    gamma[product_idx, :] = np.maximum(gamma[product_idx, :], 0)
                    sum_gamma = np.sum(gamma[product_idx, :])
                    if sum_gamma > 0:
                         gamma[product_idx, :] /= sum_gamma
                    else:
                         # Fallback: uniform distribution for this product
                         gamma[product_idx, :] = np.ones(self.n_price) / self.n_price

                return gamma
            else:
                # Fallback: uniform distribution for all products
                return np.ones((self.n, self.n_price)) / self.n_price
        except Exception as e:
            print(f"LP Error: {e}")
            # Fallback: uniform distribution for all products
            return np.ones((self.n, self.n_price)) / self.n_price