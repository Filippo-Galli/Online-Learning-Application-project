# Online Learning Application Project

This project implements and compares various multi-armed bandit algorithms for dynamic pricing problems with inventory constraints and non-stationary environments. The work explores different scenarios from simple single-product stochastic environments to complex multi-product highly non-stationary settings.

## Project Overview

The project investigates online learning algorithms for dynamic pricing in e-commerce and revenue management scenarios. It implements and compares several state-of-the-art algorithms including UCB variants, Thompson Sampling, and Primal-Dual approaches across different problem complexities.

### Key Features

- **Multi-Armed Bandit Algorithms**: UCB1, UCB-SW, Thompson Sampling, EXP3, Hedge, Primal-Dual
- **Inventory Constraints**: Shared and individual inventory management
- **Non-Stationary Environments**: Sliding window approaches and adaptive algorithms
- **Multi-Product Pricing**: Correlated and independent product valuations
- **Comprehensive Evaluation**: Regret analysis, performance comparisons, and visualization

### Problem Scenarios

1. **Single Product Stochastic** (`1_single_prod_stoc/`): Basic bandit problem without inventory constraints
2. **Single Product with Budget** (`1_single_prod_stoc_B_const/`): Adding inventory constraints to single product pricing
3. **Multi-Product Stochastic** (`2_multi_prod_stoc_B_constr/`): Multiple products with correlated/independent valuations
4. **Single Product Non-Stationary** (`3_single_prod_bobw/`): Handling non-stationary customer preferences
5. **Multi-Product Non-Stationary** (`4_multi_prod_bobw/`): Complex multi-product highly non-stationary environment
6. **Slightly Non-Stationary Multi-Product** (`5_slightly_non_stat_multi_prod/`): Mild non-stationarity with sliding window approaches

## Algorithms Implemented

### Core Algorithms (`utils/`)

- **UCB1** (`UCB1.py`): Upper Confidence Bound algorithm
- **UCB-SW** (`UCB_SW.py`): UCB with sliding window for non-stationary environments
- **Thompson Sampling** (`Thompson.py`): Bayesian approach with Beta-Bernoulli conjugate priors
- **Multi-Product Thompson** (`Multi_Thompson_constr.py`, `Multi_Thompson_constr_SW.py`): Thompson Sampling with inventory constraints
- **UCB Multi-Constraint** (`UCB1_multi_constr.py`): UCB with Linear Sum Assignment for multi-product selection
- **Primal-Dual** (`Primal_Dual.py`, `Multi_Primal_Dual.py`): Lagrangian approach for constrained optimization
- **EXP3** (`EXP3_P.py`): Exponential weights for adversarial bandits
- **Hedge** (`Hedge.py`): Expert-based online learning

### Key Innovations

- **Linear Sum Assignment (LSA)**: Optimal product-price matching for multi-product scenarios
- **Sliding Window Techniques**: Adaptive algorithms for non-stationary environments
- **Inventory-Aware Pricing**: Algorithms that adapt to remaining inventory levels
- **Lagrangian Formulations**: Primal-dual approaches for constrained optimization

## Setup Instructions

### Prerequisites
- Python 3.8+
- Jupyter Notebook/Lab
- Required packages (see `requirements.txt`)

### Installation

#### Non-Nix users:
```bash
# Clone the repository
git clone <repo-url>
cd Online-Learning-Application-project

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Register Jupyter kernel
python -m ipykernel install --user --name "$(basename $PWD)-venv"
```

#### Nix users:
The project includes `flake.nix` for reproducible development environments.

### Running Experiments

Each subfolder contains a `main.ipynb` notebook with complete experiments:

```bash
# Navigate to any scenario folder
cd 1_single_prod_stoc/

# Launch Jupyter
jupyter notebook main.ipynb
```

## Project Structure

```
├── utils/                          # Algorithm implementations
│   ├── UCB1.py                     # UCB1 algorithm
│   ├── Thompson.py                 # Thompson Sampling
│   ├── Multi_Thompson_constr.py    # Multi-product Thompson Sampling
│   ├── Primal_Dual.py             # Primal-Dual algorithm
│   └── ...                        # Other algorithms
├── 1_single_prod_stoc/             # Single product, stochastic
├── 1_single_prod_stoc_B_const/     # Single product with budget
├── 2_multi_prod_stoc_B_constr/     # Multi-product stochastic
├── 3_single_prod_bobw/             # Single product non-stationary
├── 4_multi_prod_bobw/              # Multi-product highly non-stationary
├── 5_slightly_non_stat_multi_prod/ # Slightly non-stationary multi-product
├── lab/                            # Reference implementations
├── images/                         # Generated plots and results
└── requirements.txt                # Python dependencies
```

## Key Results

The project demonstrates:

- **Thompson Sampling** generally outperforms UCB in stochastic environments
- **Sliding Window techniques** are crucial for non-stationary environments
- **Linear Sum Assignment** provides better coordination in multi-product settings
- **Primal-Dual approaches** handle inventory constraints effectively
- **Performance trade-offs** between exploration, exploitation, and inventory management

## Adding New Dependencies

1. Install the package: `pip install <package-name>`
2. Update requirements: `pip freeze > requirements.txt`
3. For Nix users: run `direnv reload`
4. For non-Nix users: run `pip install -r requirements.txt`

### Troubleshooting

#### Kernel not appearing:
```bash
# Re-register the kernel
python -m ipykernel install --user --name "$(basename $PWD)-venv" --force
```

#### Module import errors:
Ensure you're in the correct directory and the utils folder is accessible:
```python
import sys
sys.path.append('../utils')  # Adjust path as needed
```

## Contributing

When adding new algorithms or experiments:
1. Follow the existing code structure in `utils/`
2. Include comprehensive documentation and comments
3. Add corresponding experiments in appropriate scenario folders
4. Update relevant README files

## References

This project implements algorithms from online learning and multi-armed bandit literature, with applications to dynamic pricing and revenue management. See individual notebook files for specific algorithmic references and theoretical analysis.