import numpy as np
import pandas as pd

class StatisticalFeatureExtractor:
    """
    Extracts quantitative signal features from asset price series:
    autocorrelation, volatility clustering, rolling Z-score, and drawdowns.
    """
    def __init__(self, price_series):
        if isinstance(price_series, pd.Series):
            self.prices = price_series
        else:
            self.prices = pd.Series(price_series)
        self.returns = self.prices.pct_change().dropna()

    def compute_autocorrelation(self, lags=(1, 5, 10, 20)):
        """
        Calculates return autocorrelation across multiple lag horizons.
        Negative autocorrelation indicates mean-reversion tendencies.
        """
        autocorr_results = {}
        for lag in lags:
            autocorr_results[f"autocorr_lag_{lag}"] = self.returns.autocorr(lag=lag)
        return autocorr_results

    def compute_volatility_clustering(self, window=20):
        """
        Quantifies volatility clustering via rolling standard deviation and squared return persistence.
        """
        rolling_vol = self.returns.rolling(window=window).std() * np.sqrt(252)
        squared_returns = self.returns ** 2
        sq_autocorr_1 = squared_returns.autocorr(lag=1)
        sq_autocorr_5 = squared_returns.autocorr(lag=5)
        
        return {
            "rolling_volatility_annualized": rolling_vol,
            "vol_persistence_lag1": sq_autocorr_1,
            "vol_persistence_lag5": sq_autocorr_5
        }

    def compute_z_score_spread(self, window=30):
        """
        Computes rolling Z-score of price relative to moving average for mean-reversion triggers.
        Z = (P_t - MA_t) / STD_t
        """
        rolling_mean = self.prices.rolling(window=window).mean()
        rolling_std = self.prices.rolling(window=window).std()
        z_score = (self.prices - rolling_mean) / rolling_std
        return z_score

    def compute_drawdown_profile(self):
        """
        Computes rolling peak, drawdown series, and maximum drawdown (MDD).
        """
        cum_max = self.prices.cummax()
        drawdowns = (self.prices - cum_max) / cum_max
        max_drawdown = drawdowns.min()
        return {
            "drawdown_series": drawdowns,
            "max_drawdown": max_drawdown
        }
