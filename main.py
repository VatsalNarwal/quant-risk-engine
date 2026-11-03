import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stochastic_simulator import StochasticSimulator
from feature_extractor import StatisticalFeatureExtractor
from backtester import MeanReversionBacktester

def main():
    print("=======================================================================")
    print("      Quantitative Market Simulator & Stochastic Risk Engine (Python)  ")
    print("=======================================================================\n")

    # 1. Multi-Asset Simulation Setup
    assets = ["Asset_A (Equities)", "Asset_B (Tech/Growth)", "Asset_C (Defensive)"]
    S0 = [100.0, 150.0, 80.0]
    mu = [0.10, 0.14, 0.06]
    sigma = [0.18, 0.28, 0.12]
    corr = [
        [1.00, 0.72, 0.25],
        [0.72, 1.00, 0.15],
        [0.25, 0.15, 1.00]
    ]
    weights = [0.40, 0.40, 0.20]

    print("[+] Initializing Multi-Asset Stochastic Simulator (GBM with Cholesky)...")
    sim = StochasticSimulator(S0, mu, sigma, corr)
    paths = sim.simulate_paths(T=1.0, dt=1/252, n_simulations=10000, seed=42)
    print(f"    - Simulated 10,000 correlated price paths across {len(assets)} assets over 252 trading days.")

    # 2. Risk Metrics: VaR & CVaR
    risk_95 = sim.calculate_var_cvar(weights, paths, confidence_level=0.95)
    risk_99 = sim.calculate_var_cvar(weights, paths, confidence_level=0.99)

    print("\n------------------- Portfolio Risk Analytics -------------------")
    print(f"  Mean Portfolio 1-Yr Return   : {risk_95['mean_return']:.2%}")
    print(f"  Portfolio Volatility (Std)   : {risk_95['std_return']:.2%}")
    print(f"  Value at Risk (95% VaR)      : {risk_95['VaR']:.2%} loss")
    print(f"  Expected Shortfall (95% CVaR): {risk_95['CVaR']:.2%} loss")
    print(f"  Value at Risk (99% VaR)      : {risk_99['VaR']:.2%} loss")
    print(f"  Expected Shortfall (99% CVaR): {risk_99['CVaR']:.2%} loss")

    call_price, call_err = sim.price_european_option(asset_idx=0, paths=paths, strike=105.0, r=0.04, T=1.0, option_type="call")
    print(f"  1-Yr European Call Price ($105 Strike on Asset A): ${call_price:.2f} (±${call_err:.3f})")

    # 3. Statistical Signal Feature Extraction
    sample_price_path = paths[:, 0, 0]
    feature_ext = StatisticalFeatureExtractor(sample_price_path)
    autocorr = feature_ext.compute_autocorrelation(lags=[1, 5, 20])
    vol_clust = feature_ext.compute_volatility_clustering(window=20)
    z_scores = feature_ext.compute_z_score_spread(window=25)
    drawdowns = feature_ext.compute_drawdown_profile()

    print("\n----------------- Statistical Signal Features ------------------")
    for lag, val in autocorr.items():
        print(f"  {lag:<18}: {val:+.4f}")
    print(f"  Volatility Persistence (Lag-1 Sq Return): {vol_clust['vol_persistence_lag1']:+.4f}")
    print(f"  Underlying Asset Max Drawdown           : {drawdowns['max_drawdown']:.2%}")

    # 4. Strategy Backtesting with Slippage & Fees
    print("\n[+] Executing Mean-Reversion Strategy Backtest with 7 bps cost/slippage...")
    backtester = MeanReversionBacktester(sample_price_path, z_scores, entry_threshold=1.4, exit_threshold=0.1, fee_bps=5.0, slippage_bps=2.0)
    equity_df = backtester.run_backtest(initial_capital=100000.0)
    perf = backtester.evaluate_performance(equity_df, risk_free_rate=0.04)

    print("\n----------------- Backtest Performance Report ------------------")
    for k, v in perf.items():
        print(f"  {k:<24}: {v}")
    print("=======================================================================")

    # 5. Generate 4-Panel Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor="#0f172a")
    for ax in axes.flat:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#94a3b8")
        ax.grid(True, color="#334155", linestyle="--", alpha=0.5)
        for spine in ax.spines.values():
            spine.set_color("#475569")

    axes[0, 0].plot(paths[:100, :100, 0], color="#38bdf8", alpha=0.15, linewidth=0.8)
    axes[0, 0].plot(paths[:100, 0, 0], color="#f43f5e", linewidth=2.0, label="Sample Trajectory")
    axes[0, 0].set_title("Monte Carlo Price Paths (Asset A)", color="#f8fafc", fontsize=12, fontweight="bold")
    axes[0, 0].set_ylabel("Price ($)", color="#94a3b8")
    axes[0, 0].legend(loc="upper left", facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc")

    returns_pct = risk_95["returns_distribution"] * 100
    axes[0, 1].hist(returns_pct, bins=60, color="#6366f1", alpha=0.7, edgecolor="#4338ca", density=True)
    axes[0, 1].axvline(-risk_95["VaR"] * 100, color="#f59e0b", linestyle="--", linewidth=2.0, label=f"95% VaR: -{risk_95['VaR']:.1%}")
    axes[0, 1].axvline(-risk_95["CVaR"] * 100, color="#ef4444", linestyle="--", linewidth=2.0, label=f"95% CVaR: -{risk_95['CVaR']:.1%}")
    axes[0, 1].set_title("Terminal Return Distribution & Downside Tail Risk", color="#f8fafc", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Return (%)", color="#94a3b8")
    axes[0, 1].legend(loc="upper right", facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc")

    axes[1, 0].plot(equity_df.index, equity_df["Z_Score"], color="#22d3ee", label="Z-Score Spread", linewidth=1.2)
    axes[1, 0].axhline(1.4, color="#ef4444", linestyle=":", label="Short Threshold (+1.4σ)")
    axes[1, 0].axhline(-1.4, color="#10b981", linestyle=":", label="Long Threshold (-1.4σ)")
    axes[1, 0].set_title("Statistical Mean-Reversion Signals (Z-Score)", color="#f8fafc", fontsize=12, fontweight="bold")
    axes[1, 0].set_ylabel("Standard Deviations", color="#94a3b8")
    axes[1, 0].legend(loc="lower left", facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc")

    asset_benchmark = (equity_df["Price"] / equity_df["Price"].iloc[0]) * 100000.0
    axes[1, 1].plot(equity_df.index, equity_df["Equity"], color="#10b981", linewidth=2.0, label="Strategy Equity ($)")
    axes[1, 1].plot(equity_df.index, asset_benchmark, color="#94a3b8", linestyle="--", label="Buy & Hold Benchmark ($)")
    axes[1, 1].set_title(f"Cumulative PnL (Sharpe: {perf['Sharpe Ratio']}, MDD: {perf['Max Drawdown']})", color="#f8fafc", fontsize=12, fontweight="bold")
    axes[1, 1].set_ylabel("Portfolio Value ($)", color="#94a3b8")
    axes[1, 1].legend(loc="upper left", facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc")

    plt.suptitle("Quantitative Market Simulator & Stochastic Risk Engine Analytics", color="#f8fafc", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    plot_path = "quant_performance_report.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    print(f"\n[+] Generated performance analytics visualization: {plot_path}")

if __name__ == "__main__":
    main()
