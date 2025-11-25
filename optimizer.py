# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 16:59:32 2025

@author: alpus
"""

# -*- coding: utf-8 -*-
"""
Navimod Portfolio Optimizer - Massive Scale (v3.1)
320 Combinations | 20 Assets | Secure Backtest
"""

import cvxpy as cp
import numpy as np
import pandas as pd
import yfinance as yf
import json
import sys
from datetime import datetime

# --- STRATEGIC ASSET UNIVERSE (20 ETFs) ---
ASSETS = [
    # 1. CORE EQUITIES
    "SPY", "QQQ", "IWM", "VGK", "EEM",
    # 2. SECTOR & THEMATIC
    "SMH", "XLE",
    # 3. FIXED INCOME
    "TLT", "LQD", "HYG",
    # 4. CASH / DEFENSIVE
    "SHV",
    # 5. COMMODITIES
    "GLD", "SLV", "USO", "COPX", "URA", "DBA",
    # 6. ALTERNATIVES
    "VNQ", "IBIT", "ETHA"
]

# 5 Risk Levels (Gamma Values)
RISK_PROFILES = {
    1: 100.0,   # Conservative
    2: 10.0,    # Moderate
    3: 2.0,     # Balanced
    4: 0.5,     # Growth
    5: 0.01     # Aggressive
}

# 8 Lookback Periods (days) - EXPANDED
LOOKBACK_PERIODS = {
    '1W': 5,
    '2W': 10,
    '1M': 21,
    '2M': 42,
    '3M': 63,
    '6M': 126,
    '9M': 189,
    '12M': 252
}

# 8 Rebalance Periods (days) - EXPANDED
REBALANCE_PERIODS = {
    '1W': 5,
    '2W': 10,
    '1M': 21,
    '2M': 42,
    '3M': 63,
    '6M': 126,
    '9M': 189,
    '12M': 252
}

def calculate_metrics(price_curve):
    """Calculate financial metrics safely"""
    try:
        prices = np.array(price_curve)
        if len(prices) < 2:
            return None
        
        returns = np.diff(prices) / prices[:-1]
        
        # Total Return
        total_ret = (prices[-1] - prices[0]) / prices[0]
        
        # CAGR
        days = len(prices)
        years = days / 252
        if years > 0 and prices[0] > 0 and prices[-1] > 0:
            cagr = (prices[-1] / prices[0]) ** (1 / years) - 1
        else:
            cagr = 0
        
        # Monthly Average Return
        if len(returns) > 0:
            monthly_ret = np.mean(returns) * 21
        else:
            monthly_ret = 0
        
        # Volatility (Annualized)
        if len(returns) > 1:
            vol = np.std(returns) * np.sqrt(252)
        else:
            vol = 0
        
        # Sharpe Ratio (Risk Free ~3%)
        rf = 0.03
        if vol > 1e-6:
            sharpe = (cagr - rf) / vol
        else:
            sharpe = 0
        
        # Max Drawdown
        peak = prices[0]
        max_dd = 0
        for p in prices:
            if p > peak:
                peak = p
            if peak > 0:
                dd = (peak - p) / peak
                if dd > max_dd:
                    max_dd = dd
            
        return {
            "total": f"{total_ret*100:.2f}",
            "annual": f"{cagr*100:.2f}%",
            "monthly": f"{monthly_ret*100:.2f}%",
            "sharpe": f"{sharpe:.2f}",
            "dd": f"{max_dd*100:.2f}%"
        }
    except Exception as e:
        return None

def optimize_portfolio(returns_window, gamma_val):
    """
    Solves Mean-Variance Optimization.
    Handles dynamic universe & matrix stability constraints.
    """
    try:
        # --- SECURITY CHECK 1: DYNAMIC ASSET SELECTION ---
        valid_cols = returns_window.columns[returns_window.notna().all()].tolist()
        
        # Need at least 2 assets to optimize, or if lookback is too short vs assets
        if len(valid_cols) < 2:
            return np.zeros(len(ASSETS))

        # Create subset
        subset_returns = returns_window[valid_cols]
        mu = subset_returns.mean().values
        Sigma = subset_returns.cov().values
        
        # --- SECURITY CHECK 2: MATRIX STABILITY ---
        # Short lookbacks (e.g. 1W) with many assets cause singular matrices
        if np.any(np.isnan(Sigma)) or np.any(np.isinf(Sigma)):
            return np.zeros(len(ASSETS))

        # Optimization
        n = len(valid_cols)
        w = cp.Variable(n)
        gamma = cp.Parameter(nonneg=True, value=gamma_val)
        ret = mu.T @ w
        risk = cp.quad_form(w, Sigma)
        
        prob = cp.Problem(cp.Maximize(ret - gamma * risk), [cp.sum(w) == 1, w >= 0])
        
        try:
            # Try standard solver first
            prob.solve(solver=cp.OSQP, verbose=False)
        except:
            try:
                # Try robust solver if first fails
                prob.solve(solver=cp.SCS, verbose=False)
            except:
                return np.zeros(len(ASSETS))
            
        if w.value is None:
            return np.zeros(len(ASSETS))
        
        # Map weights back
        subset_weights = np.array(w.value).flatten()
        subset_weights[subset_weights < 0.001] = 0 # Clean dust
        
        if np.sum(subset_weights) > 0:
            subset_weights = subset_weights / np.sum(subset_weights)
        else:
            return np.zeros(len(ASSETS))

        final_weights = np.zeros(len(ASSETS))
        for i, asset_name in enumerate(valid_cols):
            original_idx = ASSETS.index(asset_name)
            final_weights[original_idx] = subset_weights[i]
            
        return final_weights
        
    except:
        return np.zeros(len(ASSETS))

def run_rolling_backtest(full_returns, lookback_days, rebalance_days, gamma_val):
    """
    Executes backtest ensuring NO look-ahead bias.
    """
    backtest_window = 756  # ~3 Years
    if len(full_returns) < (backtest_window + lookback_days):
        return None

    start_idx = len(full_returns) - backtest_window
    
    curve = [100.0]
    current_weights = np.zeros(len(ASSETS)) 
    weights_history = []
    
    rebalance_counter = 0
    
    for i in range(start_idx, len(full_returns)):
        # --- REBALANCING LOGIC ---
        if rebalance_counter % rebalance_days == 0:
            # Past window only [i-lookback : i]
            window = full_returns.iloc[i-lookback_days : i]
            
            # Check if we have enough data points for the requested lookback
            if len(window) >= lookback_days:
                new_w = optimize_portfolio(window, gamma_val)
                if np.sum(new_w) > 0.9: 
                    current_weights = new_w
        
        # --- EXECUTION ---
        # Apply PAST weights to CURRENT day returns
        todays_return_vector = full_returns.iloc[i].fillna(0).values
        daily_ret = np.dot(current_weights, todays_return_vector)
        
        # Store non-zero weights
        daily_w_fmt = [{"t": t, "w": round(w*100, 1)} for t, w in zip(ASSETS, current_weights) if w > 0.001]
        daily_w_fmt.sort(key=lambda x: x['w'], reverse=True)
        weights_history.append(daily_w_fmt)
        
        curve.append(curve[-1] * (1 + daily_ret))
        rebalance_counter += 1
        
    stats = calculate_metrics(curve)
    
    return {
        "curve": curve,
        "stats": stats,
        "weights_history": weights_history
    }

def calculate_current_frontier(data_slice):
    """Calculate Efficient Frontier for Scatter Plot"""
    try:
        clean_slice = data_slice.dropna(axis=1) 
        if clean_slice.empty or clean_slice.shape[1] < 2:
            return None
            
        valid_assets = clean_slice.columns.tolist()
        mu = clean_slice.mean().values
        Sigma = clean_slice.cov().values
        
        if np.any(np.isnan(Sigma)) or np.any(np.isinf(Sigma)):
            return None

        w = cp.Variable(len(valid_assets))
        gamma = cp.Parameter(nonneg=True)
        ret = mu.T @ w
        risk = cp.quad_form(w, Sigma)
        
        prob = cp.Problem(cp.Maximize(ret - gamma * risk), [cp.sum(w) == 1, w >= 0])
        gammas = np.logspace(-1, 3, 50)
        
        frontier = []
        for g in gammas:
            gamma.value = g
            try:
                prob.solve(solver=cp.OSQP, verbose=False)
                if w.value is None: continue
                
                ann_ret = ret.value * 252
                ann_vol = np.sqrt(risk.value) * np.sqrt(252)
                
                weights = np.array(w.value).flatten()
                weights[weights < 0.001] = 0
                if np.sum(weights) < 1e-6: continue
                weights = weights / np.sum(weights)
                
                comp = [{"t": valid_assets[i], "w": round(weights[i]*100, 1)} for i in range(len(valid_assets))]
                comp.sort(key=lambda x: x['w'], reverse=True)
                
                frontier.append({
                    "vol": round(ann_vol*100, 2),
                    "ret": round(ann_ret*100, 2),
                    "sharpe": round(ann_ret/ann_vol, 2) if ann_vol > 0 else 0,
                    "composition": comp
                })
            except:
                continue
                
        if not frontier: return None
        
        frontier.sort(key=lambda x: x['vol'])
        best_idx = max(range(len(frontier)), key=lambda i: frontier[i]['sharpe'])
        return {"points": frontier, "best_idx": best_idx}
    except:
        return None

def main():
    print("=" * 60)
    print("Navimod Portfolio Optimizer - v3.1 (Massive)")
    print("320 Combinations | 20 Assets | Anti-Leakage")
    print("=" * 60)
    
    try:
        # 1. Fetch Data
        print("\n[1/3] Fetching market data...")
        raw_data = yf.download(ASSETS, period="10y", interval="1d", progress=True)['Close']
        
        if raw_data.empty:
            raise ValueError("No data received")
        
        if isinstance(raw_data.columns, pd.MultiIndex):
            if 'Ticker' in raw_data.columns.names:
                raw_data.columns = raw_data.columns.get_level_values('Ticker')
            else:
                raw_data.columns = raw_data.columns.get_level_values(-1)
            
        # SECURITY: Forward Fill ONLY
        raw_data = raw_data.ffill()
        all_returns = raw_data.pct_change()
        
        print(f"   Total trading days: {len(all_returns)}")
        print(f"   Assets tracked: {len(ASSETS)}")

        # 2. Benchmark (SPY)
        print("\n[2/3] Calculating benchmark...")
        spy_slice = all_returns['SPY'].tail(756).fillna(0)
        benchmark_curve = (100 * (1 + spy_slice).cumprod()).tolist()
        benchmark_stats = calculate_metrics(benchmark_curve)
        
        # 3. Calculate all combinations
        print("\n[3/3] Running optimizations (320 combinations)...")
        
        final_output = {
            "datasets": {},
            "benchmark": {
                "dates": spy_slice.index.strftime('%Y-%m-%d').tolist(),
                "curve": benchmark_curve,
                "stats": benchmark_stats
            },
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        }

        total_combinations = len(LOOKBACK_PERIODS) * len(REBALANCE_PERIODS) * len(RISK_PROFILES)
        current = 0
        
        for lookback_name, lookback_days in LOOKBACK_PERIODS.items():
            final_output["datasets"][lookback_name] = {}
            
            # Frontier uses current valid assets
            frontier_data = calculate_current_frontier(all_returns.tail(lookback_days))
            final_output["datasets"][lookback_name]["frontier"] = frontier_data
            final_output["datasets"][lookback_name]["rebalance"] = {}
            
            for rebalance_name, rebalance_days in REBALANCE_PERIODS.items():
                final_output["datasets"][lookback_name]["rebalance"][rebalance_name] = {}
                
                for risk_level, gamma_val in RISK_PROFILES.items():
                    current += 1
                    # Simple progress indicator
                    if current % 10 == 0:
                        progress = (current / total_combinations) * 100
                        print(f"   [{current:3d}/{total_combinations}] Processing... ({progress:.1f}%)")
                    
                    # Run backtest
                    backtest_res = run_rolling_backtest(
                        all_returns,
                        lookback_days,
                        rebalance_days,
                        gamma_val
                    )
                    
                    # Current allocation
                    last_window = all_returns.tail(lookback_days)
                    curr_w = optimize_portfolio(last_window, gamma_val)
                    
                    curr_comp = [{"t": ASSETS[i], "w": round(curr_w[i]*100, 1)} for i in range(len(ASSETS)) if curr_w[i] > 0.001]
                    curr_comp.sort(key=lambda x: x['w'], reverse=True)
                    
                    if backtest_res:
                        final_output["datasets"][lookback_name]["rebalance"][rebalance_name][str(risk_level)] = {
                            "curve": backtest_res["curve"],
                            "stats": backtest_res["stats"],
                            "composition": curr_comp,
                            "history_weights": backtest_res["weights_history"]
                        }

        # Save output
        print("\n" + "=" * 60)
        with open("frontier_data.json", "w") as f:
            json.dump(final_output, f)
        
        file_size = len(json.dumps(final_output)) / (1024 * 1024)
        print(f"SUCCESS! Output saved to frontier_data.json ({file_size:.1f} MB)")
        print("=" * 60)

    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()