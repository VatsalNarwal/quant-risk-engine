import numpy as np

class StochasticSimulator:
    """
    Monte Carlo Multi-Asset Stochastic Simulator using Geometric Brownian Motion (GBM)
    and Cholesky decomposition for correlated asset returns.
    """
    def __init__(self, initial_prices, expected_returns, volatilities, correlation_matrix):
        self.S0 = np.array(initial_prices, dtype=np.float64)
        self.mu = np.array(expected_returns, dtype=np.float64)
        self.sigma = np.array(volatilities, dtype=np.float64)
        self.corr = np.array(correlation_matrix, dtype=np.float64)
        self.n_assets = len(self.S0)
        
        # Cholesky decomposition: L * L^T = Correlation Matrix
        # Allows generating correlated standard normal variables from independent ones
        self.L = np.linalg.cholesky(self.corr)

    def simulate_paths(self, T=1.0, dt=1/252, n_simulations=10000, seed=42):
        """
        Simulates geometric Brownian motion paths:
        S(t + dt) = S(t) * exp((mu - 0.5 * sigma^2)*dt + sigma * sqrt(dt) * Z_corr)
        """
        np.random.seed(seed)
        n_steps = int(T / dt)
        
        # Shape: [n_steps + 1, n_simulations, n_assets]
        paths = np.zeros((n_steps + 1, n_simulations, self.n_assets))
        paths[0] = self.S0

        drift = (self.mu - 0.5 * self.sigma**2) * dt
        vol = self.sigma * np.sqrt(dt)

        for t in range(1, n_steps + 1):
            # Uncorrelated standard normal variates [n_simulations, n_assets]
            z_uncorr = np.random.standard_normal((n_simulations, self.n_assets))
            # Correlated random variates: Z_corr = Z_uncorr * L^T
            z_corr = z_uncorr @ self.L.T
            paths[t] = paths[t - 1] * np.exp(drift + vol * z_corr)

        return paths

    def calculate_var_cvar(self, portfolio_weights, paths, confidence_level=0.95):
        """
        Calculate Value at Risk (VaR) and Conditional Value at Risk (CVaR / Expected Shortfall)
        for a weighted multi-asset portfolio at the end of the simulation horizon.
        """
        weights = np.array(portfolio_weights)
        v0 = np.sum(self.S0 * weights)
        v_terminal = paths[-1] @ weights
        returns = (v_terminal - v0) / v0

        alpha = 1.0 - confidence_level
        # VaR cutoff
        var_cutoff = np.percentile(returns, alpha * 100)
        var = -var_cutoff

        # CVaR is the expectation of losses exceeding VaR
        tail_losses = returns[returns <= var_cutoff]
        cvar = -np.mean(tail_losses) if len(tail_losses) > 0 else var

        return {
            "confidence_level": confidence_level,
            "VaR": var,
            "CVaR": cvar,
            "mean_return": np.mean(returns),
            "std_return": np.std(returns),
            "returns_distribution": returns
        }

    def price_european_option(self, asset_idx, paths, strike, r, T, option_type="call"):
        """
        Prices European options via risk-neutral Monte Carlo discounting.
        """
        terminal_prices = paths[-1, :, asset_idx]
        if option_type.lower() == "call":
            payoffs = np.maximum(terminal_prices - strike, 0.0)
        else:
            payoffs = np.maximum(strike - terminal_prices, 0.0)
        
        price = np.exp(-r * T) * np.mean(payoffs)
        std_err = np.exp(-r * T) * (np.std(payoffs) / np.sqrt(len(payoffs)))
        return price, std_err
