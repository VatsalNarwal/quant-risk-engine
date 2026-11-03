# Quantitative Market Simulator & Stochastic Risk Engine

A modular quantitative finance framework in Python for multi-asset stochastic simulation, parametric/historical tail-risk estimation (Value at Risk & Expected Shortfall), statistical signal extraction, and algorithmic strategy backtesting with realistic execution slippage.

---

## 1. Mathematical Formulation

### Geometric Brownian Motion (GBM) with Cholesky Factorization
Multi-asset price trajectories are governed by coupled stochastic differential equations:
$$dS_i(t) = \mu_i S_i(t) dt + \sigma_i S_i(t) dW_i(t)$$

Where $dW_i(t)$ are correlated Brownian motions with correlation matrix $\Sigma$. Using **Cholesky decomposition** ($\Sigma = L L^T$), we transform independent standard normal variates $Z \sim \mathcal{N}(0, I)$ into correlated Gaussian increments:
$$Z_{\text{corr}} = Z L^T$$

The discrete log-price update over step $\Delta t$ is:
$$S_i(t + \Delta t) = S_i(t) \exp\left( (\mu_i - \frac{1}{2}\sigma_i^2)\Delta t + \sigma_i \sqrt{\Delta t} Z_{\text{corr}, i} \right)$$

### Coherent Risk Metrics: VaR vs. CVaR
* **Value at Risk (VaR):** The maximum expected loss at confidence level $\alpha$ (e.g. 95%):
  $$\text{VaR}_\alpha(X) = -\inf \{ x \in \mathbb{R} : P(X \le x) > 1 - \alpha \}$$
* **Conditional Value at Risk (CVaR / Expected Shortfall):** A coherent risk measure representing the expected loss conditional on exceeding the VaR threshold:
  $$\text{CVaR}_\alpha(X) = -\mathbb{E}[X \mid X \le -\text{VaR}_\alpha(X)]$$

### Performance Metrics
* **Sharpe Ratio:** $S = \frac{R_p - R_f}{\sigma_p}$
* **Sortino Ratio:** $S_{\text{down}} = \frac{R_p - R_f}{\sigma_{\text{downside}}}$
* **Maximum Drawdown (MDD):** $\text{MDD} = \min_{t} \left( \frac{V(t) - \max_{s \le t} V(s)}{\max_{s \le t} V(s)} \right)$

---

## 2. Directory Structure

```
quant-risk-engine/
├── stochastic_simulator.py      # Correlated multi-asset GBM & VaR/CVaR calculations
├── feature_extractor.py         # Autocorrelation, volatility clustering, and Z-scores
├── backtester.py                # Mean-reversion backtester with transaction costs
├── main.py                      # Orchestration pipeline and visualization generator
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation & interview defense
```

---

## 3. Quickstart

```bash
# Clone the repository
git clone https://github.com/VatsalNarwal/quant-risk-engine.git
cd quant-risk-engine

# Install dependencies
pip install -r requirements.txt

# Run the simulation, risk analytics, and strategy backtest
python3 main.py
```

---
