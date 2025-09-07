# Task 1B: Single Product Stochastic with Budget Constraints

This folder extends the basic single-product pricing problem by introducing **inventory constraints**, making it a more realistic revenue management scenario.

## Problem Description

- **Environment**: Single product with stochastic customer valuations
- **Customer Model**: Valuations follow N(μ=0.5, σ=0.15), clipped to [0,1]
- **Decision**: Select optimal price considering inventory limitations
- **Objective**: Maximize cumulative revenue subject to inventory constraint
- **Constraint**: Limited inventory P (typically 80% of time horizon T)

## Key Challenge

The inventory constraint fundamentally changes the problem structure:
- **Resource Allocation**: Must balance immediate revenue vs. future opportunities
- **Exploration vs. Exploitation**: Limited samples due to inventory depletion
- **Constraint Satisfaction**: Avoid premature inventory exhaustion

## Algorithms Compared

### UCB1 Constrained
- **Implementation**: `../utils/UCB1_constrained.py`
- **Approach**: UCB with inventory-aware optimization
- **Key Feature**: Solves linear program to get optimal price distribution
- **Innovation**: Incorporates inventory constraint into confidence bounds

### Thompson Sampling Constrained
- **Implementation**: `../utils/Thompson_constrained.py`
- **Approach**: Bayesian approach with constraint handling
- **Key Feature**: Samples from posteriors while respecting inventory
- **Advantage**: Natural uncertainty quantification for constrained problems

## Algorithmic Innovations

### Linear Programming Formulation
Both algorithms solve:
```
maximize:   Σ γᵢ × revenue_estimate_i
subject to: Σ γᵢ × demand_estimate_i ≤ P/T  (inventory constraint)
           Σ γᵢ = 1                          (probability constraint)
           γᵢ ≥ 0                            (non-negativity)
```

### Confidence-Based Estimates
- **UCB**: Uses upper confidence bounds for revenue, lower bounds for demand
- **Thompson**: Uses sampled estimates from posterior distributions

## Key Experiments

### 1. Clairvoyant Baseline
- Computes optimal policy knowing true distribution parameters
- Provides performance upper bound for algorithm comparison
- Uses linear programming to handle inventory constraint optimally

### 2. Multi-Simulation Analysis
- Runs multiple independent simulations for statistical robustness
- Compares both algorithms across different random seeds
- Analyzes convergence properties and variance

### 3. Inventory Utilization Analysis
- Tracks inventory consumption patterns
- Studies the trade-off between revenue maximization and constraint satisfaction
- Analyzes endgame behavior when inventory runs low

## Performance Metrics

- **Cumulative Revenue**: Total revenue achieved
- **Regret**: Gap from optimal constrained policy
- **Inventory Efficiency**: How well algorithms utilize available inventory
- **Constraint Violation**: Whether algorithms respect inventory limits
- **Price Discovery**: Convergence to optimal pricing policy

## Key Results

### Performance Comparison
- Both algorithms successfully handle inventory constraints
- **Thompson Sampling** typically achieves lower regret
- **UCB** provides more consistent performance across simulations
- Both achieve ~85-90% of clairvoyant performance

### Inventory Management
- Algorithms learn to pace inventory usage appropriately
- Early rounds: more exploration, later rounds: more conservative pricing
- Successful avoidance of premature inventory depletion

### Theoretical vs. Practical
- Algorithms achieve performance close to theoretical optimum
- Constraint handling adds complexity but remains tractable
- Linear programming formulation proves effective

## Files

- `main.ipynb`: Complete experiment notebook with:
  - Constrained environment setup
  - Clairvoyant baseline computation
  - Algorithm comparison with multiple simulations
  - Inventory utilization analysis
  - Statistical significance testing

## Running the Experiment

```bash
cd 1_single_prod_stoc_B_const/
jupyter notebook main.ipynb
```

## Key Insights

1. **Constraint Integration**: Successfully incorporating inventory constraints into bandit algorithms
2. **LP Formulation**: Linear programming provides effective solution for constrained optimization
3. **Exploration-Constraint Trade-off**: Balancing learning and resource conservation
4. **Practical Applicability**: Results demonstrate real-world revenue management viability

## Technical Contributions

- **Constrained UCB**: Extension of UCB1 to handle resource constraints
- **Constrained Thompson**: Bayesian approach for constrained bandits
- **Inventory-Aware Policies**: Algorithms that adapt to remaining resources
- **Performance Benchmarking**: Rigorous comparison methodology

## Applications

This framework applies to:
- **E-commerce**: Limited inventory dynamic pricing
- **Revenue Management**: Hotel/airline pricing with capacity constraints
- **Ad Auctions**: Budget-constrained bidding strategies
- **Resource Allocation**: Any scarce resource optimization problem

## Next Steps

The constrained single-product framework enables:
- Multi-product extensions with shared inventory
- Non-stationary environments with changing constraints
- More complex constraint structures (multiple resources)
- Real-time applications with streaming data
