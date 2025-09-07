# Task 1: Single Product Stochastic Environment

This folder contains experiments for the fundamental multi-armed bandit problem applied to dynamic pricing of a single product in a stochastic environment **without inventory constraints**.

## Problem Description

- **Environment**: Single product with stochastic customer valuations
- **Customer Model**: Valuations follow a normal distribution N(μ=0.5, σ=0.15), clipped to [0,1]
- **Decision**: Select optimal price from discrete price grid
- **Objective**: Maximize cumulative revenue over T rounds
- **Constraints**: None (unlimited inventory)

## Algorithms Compared

### UCB1 (Upper Confidence Bound)
- **Implementation**: `../utils/UCB1.py`
- **Approach**: Confidence-based exploration using optimistic estimates
- **Key Feature**: Balances exploration and exploitation through confidence intervals
- **Theoretical Guarantee**: O(√T log T) regret bound

### Thompson Sampling
- **Implementation**: `../utils/Thompson.py`
- **Approach**: Bayesian approach with Beta-Bernoulli conjugate priors
- **Key Feature**: Samples from posterior distributions for decision making
- **Advantage**: Often performs better in practice than UCB algorithms

## Key Experiments

### 1. Theoretical Analysis
- Computes optimal price and expected revenue for known valuation distribution
- Establishes baseline performance for algorithm comparison

### 2. Comparative Simulation
- Runs both UCB1 and Thompson Sampling for 10,000 rounds
- Tracks price selection frequency, cumulative revenue, and regret
- Generates comprehensive performance plots

### 3. Performance Metrics
- **Cumulative Revenue**: Total revenue accumulated over time
- **Regret**: Difference between optimal and achieved performance
- **Price Discovery**: Convergence to optimal pricing strategy
- **Exploration vs Exploitation**: Balance analysis through selection frequency

## Key Results

- **Thompson Sampling** significantly outperforms UCB1 in this stochastic setting
- **Final Regret**: Thompson (~51) vs UCB1 (~514) after 10,000 rounds
- **Convergence**: Both algorithms identify the optimal price (0.4) but Thompson converges faster
- **Price Selection**: Thompson shows more focused selection around optimal price

## Files

- `main.ipynb`: Complete experiment notebook with:
  - Environment setup and theoretical analysis
  - Algorithm implementation and comparison
  - Performance visualization and statistical analysis
  - Detailed results interpretation

## Running the Experiment

```bash
cd 1_single_prod_stoc/
jupyter notebook main.ipynb
```

## Key Insights

1. **Algorithm Choice Matters**: In stochastic environments, Thompson Sampling's Bayesian approach provides superior performance
2. **Exploration Strategy**: Proper balance between exploration and exploitation is crucial for good performance
3. **Convergence Behavior**: Thompson Sampling converges faster to optimal strategies
4. **Practical Application**: Results demonstrate the effectiveness of bandit algorithms for dynamic pricing

## Next Steps

This foundational experiment sets the stage for more complex scenarios:
- Adding inventory constraints (Task 1B)
- Multi-product extensions (Task 2)
- Non-stationary environments (Tasks 3-5)

The insights from this simple setting inform algorithm choices and parameter tuning for more complex scenarios.
