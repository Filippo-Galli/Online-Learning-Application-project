# Task 4: Multi-Product Highly Non-Stationary (Best-of-Both-Worlds)

This folder tackles the most challenging scenario: **multi-product dynamic pricing in highly non-stationary environments** with correlated customer valuations and shared inventory constraints.

## Problem Description

- **Environment**: Multiple products with correlated, time-varying customer valuations
- **Non-Stationarity**: Multivariate valuation distributions change every round
- **Decision**: Select optimal prices for multiple products simultaneously
- **Objective**: Maximize total revenue despite continuous market evolution
- **Constraint**: Shared inventory budget with non-stationary demand patterns

## Key Challenges

### Multi-Dimensional Non-Stationarity
- **Correlated Drift**: Product valuations change together in complex patterns
- **High-Dimensional Space**: Exponential action space with time-varying rewards
- **Coordination Under Uncertainty**: Must coordinate across products with changing preferences
- **Resource Allocation**: Dynamic inventory allocation with shifting product priorities

### Highly Non-Stationary Environment
- **Maximum Drift**: Distribution parameters change every single round
- **Correlated Changes**: Product valuations follow multivariate patterns
- **No Stationary Periods**: Continuous adaptation required
- **Realistic Complexity**: Models real-world market dynamics

## Algorithm: Multi-Product Primal-Dual

### Core Innovation
- **Implementation**: `../utils/Multi_Primal_Dual.py`
- **Per-Product Learners**: Separate Hedge/EXP3 agent for each product
- **Shared Dual Variable**: Single λ coordinates inventory across products
- **Lagrangian Decomposition**: Distributed optimization with global coordination

### Mathematical Framework

#### Multi-Product Lagrangian
```
L_t = Σᵢ revenue_t,i - λ_t × Σᵢ consumption_t,i
```

#### Distributed Learning
Each product i maintains:
- **Price Selection**: Independent learner (Hedge or EXP3)
- **Reward Adjustment**: `reward_i = revenue_i - λ × consumption_i`
- **Local Optimization**: Maximizes adjusted reward per product

#### Global Coordination
- **Dual Update**: `λ_{t+1} = max(0, λ_t + η × (total_consumption_t - ρ))`
- **Inventory Awareness**: Prices adjust based on remaining inventory
- **Conservative Bias**: Higher prices when inventory is scarce

## Environment Design

### Correlated Valuation Evolution
```python
# Time-varying correlation matrix
Σ_t = correlation_matrix(ρ_t)  

# Multivariate evolution
μ_t = μ_{t-1} + drift_vector_t
Σ_t = Σ_{t-1} + covariance_drift_t

# Customer valuations
v_t ~ MVN(μ_t, Σ_t)
```

### Characteristics
- **Strong Correlation**: Products exhibit related preference patterns
- **Synchronized Drift**: Market-wide trends affect all products
- **Product-Specific Noise**: Individual product variations
- **Realistic Patterns**: Seasonal effects, market cycles, external shocks

## Key Algorithmic Features

### 1. Inventory-Aware Pricing
```python
# Adjust prices when inventory is low
if inventory_ratio < 0.5 * expected_ratio:
    # Bias toward higher prices
    arms[i] = min(K-1, arms[i] + 1) with probability 0.3
```

### 2. Algorithm Variants
- **Hedge**: Full information feedback for each product
- **EXP3**: Bandit feedback with exploration bonus
- **Adaptive λ**: Dynamic dual variable for constraint handling

### 3. Counterfactual Reward Estimation
For Hedge (full feedback):
```python
# Estimate rewards for all prices given customer valuation
for k in range(K):
    would_buy = (customer_valuation >= price_k)
    revenue_k = price_k if would_buy else 0
    lagrangian_k = revenue_k - λ * would_buy
```

## Key Experiments

### 1. Multi-Product Coordination Analysis
- **Product Interaction**: How algorithms coordinate across correlated products
- **Resource Competition**: Allocation strategies under scarcity
- **Adaptation Patterns**: Response to correlated vs. independent changes

### 2. Non-Stationarity Impact Assessment
- **Tracking Performance**: Following time-varying multi-product optima
- **Correlation Benefits**: How correlation helps or hurts adaptation
- **Drift Sensitivity**: Performance under different change rates

### 3. Clairvoyant Baseline Construction
- **Hindsight Optimal**: Best fixed price combination in retrospect
- **Chronological Allocation**: Optimal inventory usage pattern
- **Performance Upper Bound**: Theoretical maximum achievable

### 4. Algorithm Variant Comparison
- **Hedge vs. EXP3**: Full vs. bandit feedback performance
- **Learning Rate Sensitivity**: Optimal parameter selection
- **Inventory Strategies**: Different pacing approaches

## Performance Metrics

- **Total Revenue**: Sum across all products and time
- **Per-Product Performance**: Individual product revenue tracking
- **Dynamic Regret**: Performance vs. time-varying optimum
- **Inventory Efficiency**: Resource utilization quality
- **Coordination Quality**: Cross-product decision coherence
- **Adaptation Speed**: Response time to market changes

## Key Results

### Multi-Product Coordination
- **Successful coordination** across correlated products
- **Effective resource sharing** through dual variable λ
- **Adaptive product prioritization** based on market conditions
- **Robust performance** despite high-dimensional complexity

### Non-Stationarity Handling
- **Excellent adaptation** to continuous changes
- **Correlation exploitation**: Benefits from product relationships
- **Stable inventory management** throughout time horizon
- **Superior to static algorithms** by large margins

### Algorithm Comparison
- **EXP3 variant preferred** for practical applications
- **Hedge requires more information** but slightly better performance
- **λ parameter crucial** for inventory constraint satisfaction
- **Inventory-aware pricing** significantly improves endgame performance

## Files

- `main.ipynb`: Complete experiment notebook with:
  - Highly non-stationary multi-product environment
  - Multi-product primal-dual algorithm implementation
  - Comprehensive performance analysis across scenarios
  - Clairvoyant baseline and regret computation
  - Detailed visualization and statistical analysis

## Running the Experiment

```bash
cd 4_multi_prod_bobw/
jupyter notebook main.ipynb
```

## Key Insights

1. **Scalable Coordination**: Primal-dual approach scales to multiple products elegantly
2. **Correlation Benefits**: Correlated products enable better learning and adaptation
3. **Distributed Learning**: Per-product learners with global coordination work effectively
4. **Practical Viability**: Framework handles realistic market complexity

## Technical Contributions

- **Multi-Product BOBW**: Extension of best-of-both-worlds to multi-dimensional settings
- **Distributed Primal-Dual**: Novel decomposition for multi-product constrained bandits
- **Correlated Non-Stationarity**: Framework for handling complex market dynamics
- **Inventory-Aware Coordination**: Resource allocation under uncertainty

## Applications

This advanced framework enables:
- **Dynamic Marketplace Pricing**: Amazon, eBay-style multi-product optimization
- **Portfolio Management**: Multi-asset allocation under changing markets
- **Resource Allocation**: Cloud computing, energy distribution
- **Campaign Management**: Multi-campaign advertising optimization

## Theoretical Properties

- **Distributed Regret Bound**: O(√T log K) per product with global coordination
- **Constraint Satisfaction**: Shared inventory managed effectively
- **Correlation Exploitation**: Benefits from product relationships
- **Adaptation Guarantees**: Handles arbitrary non-stationarity

## Next Steps

This comprehensive framework enables:
- **Real-world deployment** with streaming market data
- **Integration with forecasting** systems for better adaptation
- **Meta-learning extensions** for faster market change detection
- **Causal inference** for understanding market dynamics

## Complexity Analysis

- **State Space**: O(K^n) where n is number of products, K is prices per product
- **Computational Complexity**: O(n × K) per round (linear in problem size)
- **Memory Requirements**: O(n × K × W) for windowed approaches
- **Scalability**: Proven effective up to 10+ products in experiments
