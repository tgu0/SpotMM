# SpotMM: Cryptocurrency Market Making Strategy

This repository implements a regression-based trading strategy for cryptocurrency markets using high-frequency order book and trade data. The pipeline consists of four main stages: data loading, feature engineering, feature selection + model diagnostics, and strategy backtesting.

## Pipeline Overview

```
Raw Data → DataLoader.py → FeaturesEngineering.py → FeatureSelection.py → Model Diagnostics  → Toy Trading Model → ROI Analysis
```

---

## 1. DataLoader.py

### Purpose
Loads and preprocesses raw cryptocurrency market data from a centralized exchange, merging order book snapshots with executed trades.

### Input Data
- **Order Book Data**: JSON file containing order book snapshots
  - Format:  bids/asks arrays
  - Location: `data/{coin}/book_update_{coin}.json`

- **Trade Data**: Parquet file containing executed trades
  - Required columns: `content_p` (price), `content_q` (quantity), `content_t` (timestamp), `content_b` (is buyer)
  - Location: `data/{coin}/trades_{coin}.parquet`

### Main Function Workflow
1. **Stream and filter order book data**: Removes dust levels (orders below `min_size` threshold), keeps top-of-book regardless of size
2. **Aggregate trades**: Groups trades at millisecond granularity, computing:
   - Notional volume traded
   - Volume-weighted average price (VWAP)
   - Buyer strength (proportion of volume initiated by buyers)
3. **Merge order book with trades**: Uses `merge_asof` to match each trade timestamp with the most recent order book snapshot
4. **Compute forward-looking metrics**: For each observation, calculates:
   - `time_to_target`: Time (in seconds) until a target notional volume is traded
   - `forward_vwap`: The VWAP over that forward period
5. **Stationarity analysis**: Runs Augmented Dickey-Fuller test on forward VWAP and returns

### Output Data
- **File**: `data/{coin}/{coin}_data_all.parquet`
- **Key Columns**:
  - `timestamp_ms` (index): Millisecond-precision timestamp
  - `vwap`: Current volume-weighted average price
  - `notional_traded`: Dollar volume traded
  - `content_q`: Total quantity traded
  - `buyer_strength`: Proportion of volume from buy orders [0, 1]
  - `bids`: List of [price, size] bid levels
  - `asks`: List of [price, size] ask levels
  - `time_to_target`: Seconds until target notional is reached
  - `forward_vwap`: Future VWAP over that period

- **Analysis File**: `{coin}_time_analysis.csv` containing statistics on time-to-notional for various volume thresholds

---

## 2. FeaturesEngineering.py

### Purpose
Engineers a comprehensive set of features from raw market data, capturing order book microstructure, trade flow dynamics, and price momentum.

### Input Data
- **File**: `data/{coin}/{coin}_data_all.parquet` (output from DataLoader.py)
- **Required Columns**: `bids`, `asks`, `vwap`, `notional_traded`, `buyer_strength`

### Main Function Workflow
1. **Order Book Features** (lines 184-220):
   - Top-of-book bid/ask prices and sizes
   - Maximum order sizes across all levels
   - Z-score anomaly detection for large orders (5-minute rolling window)
   - Consecutive streak counting for persistent large orders
   - Price-weighted order book imbalance (liquidity weighted by distance from midprice)

2. **Trade Flow Features** (`compute_trade_flow_features`):
   - Multi-horizon volume aggregation (10s, 30s, 60s, 5min)
   - Time-weighted and volume-weighted buyer ratios
   - Flow acceleration metrics (rate of change in directional pressure)

3. **Momentum Features** (`compute_momentum_features`):
   - Multi-horizon returns (10s, 30s, 60s, 300s) using asynchronous timestamp matching
   - Weighted momentum composite emphasizing recent price changes
   - Price position within 5-minute rolling range
   - Multi-scale realized volatility (30s, 1m, 5m)

### Output Data
- **File**: `data/{coin}/{coin}_with_features.parquet`
- **Feature Categories**:
  - **Microstructure** (15 features): bid/ask prices and sizes, large order indicators, z-scores, streaks, price imbalance
  - **Trade Flow** (16 features): volumes, buyer ratios (simple and volume-weighted), flow acceleration
  - **Momentum** (9 features): returns at multiple horizons, weighted momentum, price position, volatilities

**Total Features**: ~40 engineered features + original columns

---

## 3. FeatureSelection.py

### Purpose
Trains a Ridge regression model to predict short-term returns, selects the most important features, generates trading signals, and backtests a simplistic bare bones execution strategy.

### Input Data
- **File**: `data/{coin}/{coin}_with_features.parquet` (output from FeaturesEngineering.py)
- **Required Columns**: All engineered features plus `forward_vwap`, `time_to_target`, `bid_price_1`, `ask_price_1`

### Main Function Workflow
1. **Train-Test Split**: Temporal split at a fixed date (default: 2025-11-07)
2. **Target Variable**: Computes future returns: `(forward_vwap - vwap) / vwap`
3. **Model Training**:
   - Ridge regression with L2 regularization (alpha=0.1)
   - Features are standardized (zero mean, unit variance)
4. **Feature Selection**:
   - Computes scaled coefficients: `coefficient / feature_std`
   - Selects top 15 features by absolute magnitude
5. **Performance Metrics**:
   - Out-of-sample R², RMSE
   - Directional accuracy (sign prediction)
6. **Trading Strategy**:
   - **Signal Generation**: Quantile thresholding (95th percentile for longs, 5th for shorts)
   - **Execution Modeling**:
     - Long positions enter at ask price
     - Short positions enter at bid price
     - Optional midprice execution for comparison
   - **Position Management**: Enforces single position at a time with time-to-target holding periods
   - **Return Calculation**: Accounts for bid-ask spread costs in realized returns

### Core Function: `enforce_single_position` (lines 13-105)
Implements realistic trade execution logic:
- Prevents overlapping positions
- Computes per-trade returns with proper entry/exit pricing
- Handles missing data and edge cases
- Optionally simulates fills at midprice vs. bid/ask

### Output Data
- **Performance Report**: `{coin}_marketmaking.html`
  - Generated using QuantStats library
  - Contains: equity curve, drawdown analysis, Sharpe ratio, win rate, trade statistics
- **Console Output**:
  - Model R² and RMSE (train and test)
  - Feature importance rankings
  - Directional accuracy

---

## Data Flow Summary

| Stage | Input | Output | Key Transformation |
|-------|-------|--------|-------------------|
| **DataLoader** | Raw JSON order books + trade parquets | `{coin}_data_all.parquet` | Merge order book with trades, compute forward VWAP |
| **FeaturesEngineering** | `{coin}_data_all.parquet` | `{coin}_with_features.parquet` | Engineer 40+ microstructure, flow, and momentum features |
| **FeatureSelection** | `{coin}_with_features.parquet` | `{coin}_marketmaking.html` | Train model, backtest strategy with realistic execution |

---

## Usage

### Step 1: Load and Preprocess Data
```bash
# Set coin variable in DataLoader.py (line 191)
python DataLoader.py
```
**Output**: `data/{coin}/{coin}_data_all.parquet`

### Step 2: Engineer Features
```bash
# Set coin variable in FeaturesEngineering.py (line 174)
python FeaturesEngineering.py
```
**Output**: `data/{coin}/{coin}_with_features.parquet`

### Step 3: Train Model and Backtest
```bash
# Set coin variable in FeatureSelection.py (line 109)
python FeatureSelection.py
```
**Output**: `{coin}_marketmaking.html` (performance report)

---

## Configuration Parameters

### DataLoader.py
- `target_notional` (line 248): Target volume threshold for forward VWAP calculation (default: 250,000)
- `min_size` (line 7): Minimum order size to keep in order book filtering (default: 0.01)

### FeaturesEngineering.py
- `window` (line 192): Rolling window for z-score computation (default: '5min')
- Horizons for momentum features (line 99): `[10, 30, 60, 300]` seconds
- Horizons for trade flow features (line 155): `["10s", "30s", "60s", "5min"]`

### FeatureSelection.py
- `split_date` (line 113): Date separating train/test sets (default: 2025-11-07)
- `alpha` (line 137): Ridge regression regularization parameter (default: 0.1)
- `buy_thresh` / `sell_thresh` (lines 165-166): Quantile thresholds for signals (default: 0.95 / 0.05)
- `use_mid` (line 172): Whether to simulate execution at midprice vs. bid/ask (default: True)

---

## Dependencies

```
pandas
numpy
sklearn
quantstats
ijson
statsmodels
```

---

## Notes

- All timestamps are in UTC and floored to millisecond precision
- Order book data is filtered to remove dust levels while always preserving top-of-book
- Forward VWAP computation uses a sliding window approach to find when target notional is reached
- Strategy returns explicitly account for bid-ask spread costs
- The pipeline is designed for cryptocurrency data but can be adapted to other high-frequency markets

---

## Research Applications

This pipeline is suitable for:
- Short-term return prediction research
- Market microstructure analysis
- Order book dynamics studies
- High-frequency trading strategy development
- Feature importance analysis in limit order book markets
