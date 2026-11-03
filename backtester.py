import numpy as np
import pandas as pd

class MeanReversionBacktester:
    """
    Simulates statistical mean-reversion trading strategy with realistic
    transaction costs (brokerage/fees) and slippage models.
    """
    def __init__(self, price_series, z_scores, entry_threshold=1.5, exit_threshold=0.2, fee_bps=5.0, slippage_bps=2.0):
        self.prices = pd.Series(price_series)
        self.z_scores = pd.Series(z_scores)
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.cost_per_trade = (fee_bps + slippage_bps) / 10000.0  # 7 bps = 0.0007

    def run_backtest(self, initial_capital=100000.0):
        n = len(self.prices)
        positions = np.zeros(n)
        cash = np.zeros(n)
        holdings = np.zeros(n)
        total_equity = np.zeros(n)
        trade_signals = np.zeros(n)

        cash[0] = initial_capital
        total_equity[0] = initial_capital
        current_pos = 0.0

        for t in range(1, n):
            z = self.z_scores.iloc[t]
            p = self.prices.iloc[t]
            prev_cash = cash[t - 1]
            prev_pos = current_pos

            signal = 0
            if np.isnan(z):
                pass
            elif current_pos == 0:
                if z < -self.entry_threshold:
                    signal = 1   # Oversold -> Long
                elif z > self.entry_threshold:
                    signal = -1  # Overbought -> Short
            elif current_pos > 0:
                if z >= -self.exit_threshold:
                    signal = -1  # Close Long
            elif current_pos < 0:
                if z <= self.exit_threshold:
                    signal = 1   # Close Short

            if signal != 0:
                trade_signals[t] = signal
                if signal == 1 and current_pos <= 0:
                    target_pos = (prev_cash * 0.95) / p if current_pos == 0 else 0.0
                    delta_shares = target_pos - current_pos
                    cost = abs(delta_shares * p) * self.cost_per_trade
                    current_pos = target_pos
                    cash[t] = prev_cash - (delta_shares * p) - cost
                elif signal == -1 and current_pos >= 0:
                    target_pos = -(prev_cash * 0.95) / p if current_pos == 0 else 0.0
                    delta_shares = target_pos - current_pos
                    cost = abs(delta_shares * p) * self.cost_per_trade
                    current_pos = target_pos
                    cash[t] = prev_cash - (delta_shares * p) - cost
            else:
                cash[t] = prev_cash

            positions[t] = current_pos
            holdings[t] = current_pos * p
            total_equity[t] = cash[t] + holdings[t]

        equity_df = pd.DataFrame({
            "Price": self.prices,
            "Z_Score": self.z_scores,
            "Position": positions,
            "Signal": trade_signals,
            "Equity": total_equity,
            "Return": pd.Series(total_equity).pct_change().fillna(0.0)
        })

        return equity_df

    @staticmethod
    def evaluate_performance(equity_df, risk_free_rate=0.04):
        equity = equity_df["Equity"]
        daily_returns = equity_df["Return"]
        total_days = len(equity_df)

        cum_return = (equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0]
        annualized_return = (1.0 + cum_return) ** (252.0 / total_days) - 1.0
        annualized_vol = daily_returns.std() * np.sqrt(252.0)
        
        excess_return = annualized_return - risk_free_rate
        sharpe_ratio = excess_return / annualized_vol if annualized_vol > 0 else 0.0

        downside_returns = daily_returns[daily_returns < 0.0]
        downside_std = downside_returns.std() * np.sqrt(252.0) if len(downside_returns) > 0 else 1e-6
        sortino_ratio = excess_return / downside_std

        cum_max = equity.cummax()
        drawdown = (equity - cum_max) / cum_max
        max_drawdown = drawdown.min()

        trades_executed = np.count_nonzero(equity_df["Signal"])

        return {
            "Cumulative Return": f"{cum_return:.2%}",
            "Annualized Return": f"{annualized_return:.2%}",
            "Annualized Volatility": f"{annualized_vol:.2%}",
            "Sharpe Ratio": f"{sharpe_ratio:.2f}",
            "Sortino Ratio": f"{sortino_ratio:.2f}",
            "Max Drawdown": f"{max_drawdown:.2%}",
            "Total Trades Executed": trades_executed,
            "Initial Capital": f"${equity.iloc[0]:,.2f}",
            "Ending Equity": f"${equity.iloc[-1]:,.2f}"
        }
