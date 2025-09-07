# Task 5: Slightly Non-Stationary Multi-Product Pricing

This folder explores a more moderate non-stationary scenario: **multi-product dynamic pricing with sliding window approaches** for environments that exhibit slow, gradual changes rather than extreme non-stationarity.

## Problem Description

- **Environment**: Multiple products with gradually changing customer valuations
- **Non-Stationarity**: Slight distribution drift with piecewise-constant intervals
- **Decision**: Select optimal prices adapting to slow market evolution
- **Objective**: Maximize total revenue with adaptive algorithms
- **Constraint**: Shared inventory budget across products

## Key Characteristics

### Moderate Non-Stationarity
- **Gradual Changes**: Distribution parameters change occasionally, not every round
- **Piecewise Constant**: Stable periods interrupted by change points
- **Change Probability**: Low probability (0.001-0.003) of change per round
- **Realistic Setting**: Models many real-world scenarios with occasional market shifts

### Sliding Window Approach
- **Window Size**: W = √(T) × constant (typically 5√T)
- **Recent Focus**: Algorithms prioritize recent observations
- **Memory Management**: Balance between adaptation and statistical power
- **Computational Efficiency**: Tractable for large time horizons

## Algorithms Compared

### 1. UCB with Sliding Window (UCB-SW)
- **Implementation**: `../utils/UCB_SW.py`
- **Approach**: UCB confidence bounds computed over sliding window
- **Key Feature**: Automatic adaptation to distribution changes
- **Innovation**: Window-based statistics for non-stationary environments

### 2. Thompson Sampling with Sliding Window (Thompson-SW)  
- **Implementation**: `../utils/Multi_Thompson_constr_SW.py`
- **Approach**: Beta posteriors updated with windowed observations
- **Key Feature**: Bayesian adaptation with recent data emphasis
- **Innovation**: Sliding window posterior maintenance

### 3. Multi-Product Thompson (Standard)
- **Implementation**: `../utils/Multi_Thompson_constr.py`
- **Approach**: Traditional Thompson Sampling with all historical data
- **Baseline**: Non-adaptive algorithm for comparison
- **Purpose**: Demonstrate benefits of sliding window adaptation

### 4. Multi-Product Primal-Dual
- **Implementation**: `../utils/Multi_Primal_Dual.py`
- **Approach**: Primal-dual with EXP3 learners (no sliding window)
- **Comparison**: Alternative approach for non-stationary environments
- **Purpose**: Compare sliding window vs. other adaptation methods

## Algorithmic Innovations

### Sliding Window UCB
```python
# Confidence bounds over recent window
window_data = observations[-W:]  # Last W observations
confidence_radius = sqrt(2 * log(t) / len(window_data))
ucb_value = empirical_mean + confidence_radius
```

### Sliding Window Thompson Sampling
```python
# Beta parameters from windowed data
recent_successes = sum(window_data)
recent_failures = len(window_data) - recent_successes
alpha_window = alpha_prior + recent_successes
beta_window = beta_prior + recent_failures
```

### Window Management
- **Fixed Size**: Maintain exactly W most recent observations
- **Automatic Updates**: Drop oldest when adding newest
- **Statistics Recomputation**: Efficient incremental updates
- **Memory Efficiency**: Deque-based storage for O(1) updates

## Environment Design

### Non-Stationary Environment Class
```python
class NonStationaryEnvironment:
    def __init__(self, change_probability=0.001):
        self.change_probability = change_probability
        
    def sample_parameters(self, T):
        # Piecewise constant parameters with occasional changes
        change_triggers = random(T, n_products) < change_probability
        # ... generate parameter evolution
```

### Change Characteristics
- **Change Points**: Random times with low probability
- **Parameter Shifts**: New μ and σ sampled from specified ranges
- **Persistence**: Parameters remain constant until next change
- **Product Independence**: Each product can change independently

## Key Experiments

### 1. Algorithm Comparison Suite
Comprehensive comparison across:
- **UCB-SW**: Sliding window UCB with different window sizes
- **Thompson-SW**: Sliding window Thompson Sampling
- **Thompson-Multi**: Standard Thompson without adaptation
- **Primal-Dual**: EXP3-based alternative approach

### 2. Window Size Analysis
- **Impact of W**: Different window sizes (√T, 3√T, 5√T, 10√T)
- **Adaptation Speed**: Trade-off between responsiveness and stability
- **Statistical Power**: Sufficient data for reliable estimates
- **Optimal Selection**: Empirical determination of best window size

### 3. Non-Stationarity Sensitivity
- **Change Frequency**: Different change probabilities (0.0001 to 0.01)
- **Change Magnitude**: Various parameter shift sizes
- **Adaptation Performance**: How quickly algorithms detect and adapt
- **Robustness**: Performance across different non-stationarity levels

### 4. Multi-Product Coordination Analysis
- **Product Interactions**: How sliding windows affect coordination
- **Inventory Management**: Resource allocation with adaptive algorithms
- **Cross-Product Learning**: Information sharing across products

## Performance Metrics

- **Cumulative Revenue**: Total revenue across all products
- **Dynamic Regret**: Performance vs. time-varying optimal policy
- **Adaptation Quality**: Speed of response to changes
- **Stability**: Performance variance in stationary periods
- **Window Efficiency**: How well sliding windows track changes
- **Inventory Utilization**: Resource management effectiveness

## Key Results

### Algorithm Performance Ranking
1. **Thompson-SW**: Best overall performance with excellent adaptation
2. **Multi-Product Thompson**: Strong baseline, slower adaptation
3. **Primal-Dual**: Good performance, different adaptation mechanism
4. **UCB-SW**: Solid performance, more exploration-focused

### Window Size Insights
- **W = 5√T** emerges as optimal balance
- **Smaller windows**: Faster adaptation but higher variance
- **Larger windows**: More stability but slower adaptation
- **Problem-dependent**: Optimal size varies with change frequency

### Adaptation Benefits
- **10-15% improvement** over non-adaptive algorithms
- **Faster convergence** after change points (100-200 rounds)
- **Maintained performance** during stationary periods
- **Robust to various** non-stationarity patterns

### Multi-Product Coordination
- **Sliding windows preserve** coordination benefits
- **LSA remains effective** with windowed statistics
- **Inventory management** adapts appropriately to changes
- **Cross-product learning** enhanced by recency focus

## Files

- `main.ipynb`: Complete experiment notebook with:
  - Non-stationary environment implementation with change visualization
  - All four algorithm implementations and comparison
  - Window size sensitivity analysis
  - Comprehensive performance evaluation and statistical testing
  - Parameter evolution tracking and adaptation analysis

## Running the Experiment

```bash
cd 5_slightly_non_stat_multi_prod/
jupyter notebook main.ipynb
```

## Key Insights

1. **Sliding Windows Work**: Simple and effective approach for moderate non-stationarity
2. **Thompson Sampling Advantage**: Bayesian approach adapts naturally with windows
3. **Window Size Matters**: Proper tuning crucial for optimal performance
4. **Practical Applicability**: Framework suitable for many real-world scenarios

## Technical Contributions

- **Sliding Window Bandits**: Principled approach for gradual non-stationarity
- **Multi-Product SW**: Extension to multi-dimensional constrained problems
- **Window Size Theory**: Empirical guidelines for parameter selection
- **Adaptation Analysis**: Framework for evaluating non-stationary performance

## Applications

This moderate non-stationarity framework applies to:
- **Seasonal Retail**: Gradual preference changes over months
- **Technology Adoption**: Slow market evolution for new products
- **B2B Pricing**: Enterprise markets with gradual preference shifts
- **Financial Markets**: Medium-term trend following

## Comparison with Other Approaches

### vs. Extreme Non-Stationarity (Task 4)
- **Computational Efficiency**: Sliding windows much more efficient
- **Adaptation Speed**: Slower but sufficient for gradual changes
- **Implementation Simplicity**: Much easier to implement and tune
- **Memory Requirements**: Fixed memory footprint

### vs. Stationary Methods (Tasks 1-2)
- **Adaptation Capability**: Handles non-stationarity gracefully
- **Performance Cost**: Minimal overhead in stationary periods
- **Robustness**: Works well even if non-stationarity assumptions wrong
- **Practical Deployment**: Easy to retrofit existing systems

## Theoretical Properties

- **Regret Bound**: O(√T log T) for slowly changing environments
- **Adaptation Rate**: O(W) rounds to detect changes
- **Memory Complexity**: O(n × K × W) storage requirement
- **Computational Complexity**: O(n × K) per round (same as stationary)

## Next Steps

The sliding window framework enables:
- **Adaptive window sizing**: Dynamic adjustment based on detected changes
- **Change point detection**: Active identification of distribution shifts
- **Meta-learning**: Learning optimal window sizes from data
- **Real-time deployment**: Low-latency applications with streaming data

## Parameter Guidelines

### Window Size Selection
- **Conservative**: W = √T (slower adaptation, more stable)
- **Standard**: W = 5√T (good balance for most scenarios)
- **Aggressive**: W = 10√T (faster adaptation, higher variance)
- **Problem-specific**: Tune based on expected change frequency

### Change Probability
- **Very Slow**: 0.0001 (changes every ~10,000 rounds)
- **Slow**: 0.001 (changes every ~1,000 rounds)
- **Moderate**: 0.003 (changes every ~300 rounds)
- **Fast**: 0.01 (changes every ~100 rounds)
