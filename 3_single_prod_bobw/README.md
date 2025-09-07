# Task 3: Single Product Best-of-Both-Worlds (Non-Stationary)

This folder addresses the challenging problem of **single-product dynamic pricing in highly non-stationary environments**, where customer preferences change continuously over time.

## Problem Description

- **Environment**: Single product with time-varying customer valuations
- **Non-Stationarity**: Customer valuation distribution changes every round
- **Decision**: Select optimal price adapting to changing preferences
- **Objective**: Maximize cumulative revenue despite distribution drift
- **Constraint**: Limited inventory with non-stationary demand

## Key Challenges

### Extreme Non-Stationarity
- **Distribution Drift**: Mean and variance change every time step
- **Concept Drift**: Optimal prices shift continuously
- **Exploration-Exploitation**: Must adapt while maintaining performance
- **Memory Management**: Balance recent vs. historical information

### Best-of-Both-Worlds (BOBW)
The algorithm must perform well in:
- **Stationary Periods**: Achieve near-optimal performance when distribution is stable
- **Non-Stationary Periods**: Quickly adapt to distribution changes
- **Unknown Environment**: Without knowing which regime the environment is in

## Algorithm: Primal-Dual with Expert Learning

### Core Components

#### 1. Primal-Dual Framework
- **Implementation**: `../utils/Primal_Dual.py`
- **Dual Variable λ**: Controls inventory consumption rate
- **Lagrangian Formulation**: L = revenue - λ × inventory_usage
- **Adaptive λ**: Updates based on constraint violation

#### 2. Expert-Based Learning
Two algorithmic choices for price selection:

**Hedge Algorithm** (Full Information):
- **Use Case**: When full market feedback is available
- **Update Rule**: Exponential weights with full reward vectors
- **Advantage**: Faster convergence with complete information

**EXP3 Algorithm** (Bandit Feedback):
- **Use Case**: When only selected price outcome is observed
- **Update Rule**: Exponential weights with partial feedback
- **Advantage**: Robust to limited information scenarios

### Mathematical Framework

#### Lagrangian Construction
```
L_t(p) = revenue_t(p) - λ_t × consumption_t(p)
```

#### Dual Variable Update
```
λ_{t+1} = max(0, λ_t + η × (consumption_t - ρ))
```
where ρ = P/T is the target consumption rate.

#### Expert Weight Update
- **Hedge**: `w_{t+1,i} = w_{t,i} × exp(η × reward_{t,i})`
- **EXP3**: Similar with exploration component

## Non-Stationary Environment Design

### Distribution Evolution
```python
# Time-varying parameters
μ_t = 0.4 + 0.2 * sin(2πt/1000) + noise_t
σ_t = 0.1 + 0.05 * cos(2πt/500) + noise_t

# Customer valuation at time t
v_t ~ N(μ_t, σ_t²)
```

### Characteristics
- **Continuous Change**: No stationary periods
- **Smooth Drift**: Gradual parameter evolution
- **Stochastic Noise**: Random fluctuations around trends
- **Realistic Patterns**: Seasonal and cyclical behaviors

## Key Experiments

### 1. Algorithm Comparison
Comprehensive comparison of:
- **Primal-Dual + Hedge**: Full information variant
- **Primal-Dual + EXP3**: Bandit feedback variant
- **Baseline Methods**: UCB, Thompson Sampling for comparison

### 2. Non-Stationarity Analysis
- **Distribution Tracking**: How well algorithms follow changing optima
- **Adaptation Speed**: Response time to distribution changes
- **Forgetting Rate**: Balance between adaptation and stability

### 3. Inventory Management
- **Constraint Satisfaction**: Maintaining inventory throughout horizon
- **Pacing Analysis**: Inventory consumption patterns
- **Endgame Behavior**: Performance when inventory runs low

### 4. Regret Decomposition
- **Tracking Regret**: Loss due to following changing optimum
- **Optimization Regret**: Loss due to imperfect optimization
- **Total Regret**: Combined performance measure

## Performance Metrics

- **Cumulative Revenue**: Total revenue achieved
- **Dynamic Regret**: Performance vs. time-varying optimum
- **Constraint Violation**: Inventory usage vs. target
- **Adaptation Quality**: Speed of response to changes
- **Stability**: Performance variance across different periods

## Key Results

### Algorithm Performance
- **Primal-Dual methods excel** in non-stationary settings
- **EXP3 variant** shows robust performance with limited feedback
- **Significant improvement** over static algorithms (UCB, Thompson)
- **Graceful adaptation** to changing environments

### Inventory Management
- **Successful constraint satisfaction** across all scenarios
- **Adaptive pacing** based on remaining time and inventory
- **λ parameter effectively controls** resource consumption
- **Robust to distribution changes**

### Non-Stationarity Handling
- **Quick adaptation** to distribution shifts (within 100-200 rounds)
- **Minimal performance loss** during transition periods
- **Excellent tracking** of time-varying optimal policies
- **Stable performance** despite continuous changes

## Files

- `main.ipynb`: Complete experiment notebook with:
  - Non-stationary environment implementation
  - Primal-dual algorithm with Hedge/EXP3 variants
  - Comprehensive adaptation analysis
  - Dynamic regret computation and visualization
  - Comparison with stationary algorithms

## Running the Experiment

```bash
cd 3_single_prod_bobw/
jupyter notebook main.ipynb
```

## Key Insights

1. **Non-Stationarity Requires Specialized Algorithms**: Standard bandits fail dramatically
2. **Primal-Dual Approach**: Elegant solution combining constraint handling with adaptation
3. **Expert Learning**: Hedge/EXP3 provide robust foundation for changing environments
4. **Practical Applicability**: Framework handles realistic non-stationary scenarios

## Technical Contributions

- **BOBW Framework**: Unified approach for unknown stationarity
- **Constrained Non-Stationary Bandits**: Novel problem formulation
- **Adaptive Inventory Management**: Dynamic resource allocation
- **Theoretical Foundation**: Rigorous analysis of dynamic regret

## Applications

This framework applies to:
- **Fashion Retail**: Seasonal preference changes
- **Technology Products**: Rapid market evolution
- **Financial Markets**: Time-varying risk preferences
- **Content Recommendation**: Changing user interests

## Theoretical Properties

- **Dynamic Regret Bound**: O(√T(1 + V_T)) where V_T measures variation
- **Constraint Violation**: O(√T) on average
- **Adaptation Rate**: O(log T) for detecting changes
- **BOBW Property**: Near-optimal in both stationary and non-stationary regimes

## Next Steps

The non-stationary framework enables:
- Multi-product extensions with correlated drift
- Online change detection algorithms
- Meta-learning approaches for faster adaptation
- Real-world deployment with streaming data
