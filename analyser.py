import pandas as pd
import numpy as np
from ta import volatility, momentum
import matplotlib.pyplot as plt
from datetime import time, timedelta

class HighProbabilityBacktester:
    def __init__(self, file_path):
        self.df = self._prepare_data(file_path)
        self.initial_balance = 10000
        self.trade_log = []
        self.analytics = {}
        self._setup_parameters()

    def _setup_parameters(self):
        self.risk_per_trade = 0.01  # 1% per trade
        self.rr_ratio = 0.82  # Optimized reward ratio
        self.atr_multiplier = 1.48  # Fine-tuned SL multiplier
        self.rsi_bands = (24.8, 75.2)  # Optimized thresholds

    def _prepare_data(self, file_path):
        df = pd.read_csv(file_path, parse_dates=['time'])
        df.set_index('time', inplace=True)
        
        if df.index.tz is None:
            df = df.tz_localize('UTC')
        
        df['EMA_34'] = df['close'].ewm(span=34, adjust=False).mean()
        df['RSI_9'] = momentum.rsi(df['close'], window=9)
        df['ATR_14'] = volatility.average_true_range(
            df['high'], df['low'], df['close'], 14)
        df['volume_ma'] = df['tick_volume'].rolling(20).mean()
        
        return df.dropna()

    def _is_avoided_session(self, dt):
        gmt_time = dt.tz_convert('UTC').time()
        return time(2,0) <= gmt_time < time(5,0)

    def _generate_signal(self, row):
        if self._is_avoided_session(row.name):
            return None
            
        bullish = (row['close'] > row['EMA_34'] and 
                 row['RSI_9'] < self.rsi_bands[0] and
                 row['tick_volume'] > row['volume_ma'] and
                 row['ATR_14'] > 1.18)
                  
        bearish = (row['close'] < row['EMA_34'] and 
                 row['RSI_9'] > self.rsi_bands[1] and
                 row['tick_volume'] > row['volume_ma'] and
                 row['ATR_14'] > 1.18)
        
        return 'long' if bullish else 'short' if bearish else None

    def _calculate_position_size(self, balance, entry, sl):
        risk_amount = balance * self.risk_per_trade
        price_diff = abs(entry - sl)
        if price_diff == 0:
            return 0
        return risk_amount / price_diff

    def run_backtest(self):
        balance = self.initial_balance
        equity = [balance]
        in_position = None
        
        for idx, row in self.df.iterrows():
            # Close position logic
            if in_position:
                open_time, direction, entry, sl, tp, units = in_position
                current_low = row['low']
                current_high = row['high']
                pnl = 0
                status = None
                
                if direction == 'long':
                    if current_low <= sl:
                        pnl = (sl - entry) * units
                        status = 'SL'
                    elif current_high >= tp:
                        pnl = (tp - entry) * units
                        status = 'TP'
                else:
                    if current_high >= sl:
                        pnl = (entry - sl) * units
                        status = 'SL'
                    elif current_low <= tp:
                        pnl = (entry - tp) * units
                        status = 'TP'
                
                if status:
                    balance += pnl
                    equity.append(balance)
                    
                    self.trade_log.append({
                        'time': idx,
                        'direction': direction,
                        'entry': entry,
                        'exit': sl if status == 'SL' else tp,
                        'pnl': pnl,
                        'status': status,
                        'duration': idx - open_time,
                        'balance': balance
                    })
                    
                    in_position = None
                    if len(self.trade_log) >= 300:
                        break
            
            # Open new position
            if not in_position and len(self.trade_log) < 300:
                signal = self._generate_signal(row)
                if signal:
                    entry_price = row['close']
                    atr = row['ATR_14']
                    
                    if signal == 'long':
                        sl_price = entry_price - self.atr_multiplier * atr
                        tp_price = entry_price + self.rr_ratio * (entry_price - sl_price)
                    else:
                        sl_price = entry_price + self.atr_multiplier * atr
                        tp_price = entry_price - self.rr_ratio * (sl_price - entry_price)
                        
                    units = self._calculate_position_size(balance, entry_price, sl_price)
                    if units > 0:
                        in_position = (idx, signal, entry_price, sl_price, tp_price, units)
        
        # Close any remaining position at end
        if in_position:
            open_time, direction, entry, sl, tp, units = in_position
            exit_price = self.df.iloc[-1]['close']
            pnl = (exit_price - entry) * units if direction == 'long' else (entry - exit_price) * units
            status = 'FORCE CLOSE'
            
            balance += pnl
            equity.append(balance)
            
            self.trade_log.append({
                'time': self.df.index[-1],
                'direction': direction,
                'entry': entry,
                'exit': exit_price,
                'pnl': pnl,
                'status': status,
                'duration': self.df.index[-1] - open_time,
                'balance': balance
            })
        
        self._calculate_analytics(equity)
        return self.analytics

    def _calculate_analytics(self, equity):
        df_trades = pd.DataFrame(self.trade_log)
        
        self.analytics = {
            'total_trades': 0,
            'win_rate': 0,
            'total_return': (equity[-1]/self.initial_balance - 1) * 100,
            'max_drawdown': 0,
            'sharpe': 0,
            'profit_factor': 0,
            'session_analysis': None,
            'equity_curve': equity
        }
        
        if df_trades.empty:
            return

        wins = df_trades[df_trades['pnl'] > 0]
        losses = df_trades[df_trades['pnl'] < 0]
        total_trades = len(df_trades)

        # Win rate calculation
        win_rate = len(wins)/total_trades if total_trades > 0 else 0
        
        # Profit factor calculation
        profit_factor = abs(wins['pnl'].sum()/losses['pnl'].sum()) if len(losses) > 0 else np.inf
        
        # Drawdown calculation
        equity_array = np.array(equity)
        peak = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - peak)/peak
        max_drawdown = np.min(drawdown) * 100
        
        # Sharpe ratio calculation
        returns = pd.Series(equity).pct_change().dropna()
        sharpe = np.sqrt(252) * returns.mean()/returns.std() if not returns.empty else 0

        self.analytics.update({
            'total_trades': total_trades,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'sharpe': sharpe,
            'profit_factor': profit_factor,
            'session_analysis': self._session_analysis(df_trades)
        })

    def _session_analysis(self, df):
        df['session'] = df['time'].apply(lambda x: 
            'Asian' if self._is_avoided_session(x) else 
            'London' if time(7,0) <= x.time() < time(16,0) else
            'New York')
        return df.groupby('session')['status'].value_counts(normalize=True)

    def monte_carlo_analysis(self, n=1000):
        if not self.trade_log:
            return pd.DataFrame()

        pnls = [t['pnl'] for t in self.trade_log]
        results = []
        
        for _ in range(n):
            shuffled = [p * -1 if np.random.rand() < 0.15 else p for p in pnls]
            wr = sum(1 for p in shuffled if p > 0)/len(shuffled)
            dd = self._calculate_drawdown(shuffled)
            results.append((wr, dd))
            
        return pd.DataFrame(results, columns=['win_rate', 'drawdown'])

    def _calculate_drawdown(self, pnls):
        equity = np.cumsum(pnls) + self.initial_balance
        peak = np.maximum.accumulate(equity)
        return np.min((equity - peak)/peak) * 100

    def plot_results(self):
        if not self.trade_log:
            print("No trades to plot")
            return

        plt.figure(figsize=(12,6))
        plt.plot(self.analytics['equity_curve'])
        plt.title(f"Equity Curve ({self.analytics['total_trades']} Trades)")
        plt.ylabel("Balance")
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10,6))
        plt.hist([t['pnl'] for t in self.trade_log if t['pnl'] > 0], 
                 bins=20, alpha=0.7, label='Profitable Trades')
        plt.hist([t['pnl'] for t in self.trade_log if t['pnl'] < 0], 
                 bins=20, alpha=0.7, label='Losing Trades')
        plt.legend()
        plt.title("P&L Distribution")
        plt.show()

if __name__ == "__main__":
    backtester = HighProbabilityBacktester(
        r"C:\Users\veren\OneDrive\Рабочий стол\Trader\DATA\XAUUSD\1M\history_1M.csv"
    )
    results = backtester.run_backtest()
    
    print(f"Backtest Results ({results['total_trades']} Trades)")
    print("========================================")
    print(f"Win Rate: {results['win_rate']:.2%}")
    print(f"Total Return: {results['total_return']:.1f}%")
    print(f"Max Drawdown: {results['max_drawdown']:.1f}%")
    print(f"Sharpe Ratio: {results['sharpe']:.2f}")
    print(f"Profit Factor: {results['profit_factor']:.1f}:1")
    
    if results['session_analysis'] is not None:
        print("\nSession Performance:")
        print(results['session_analysis'])
    
    mc_results = backtester.monte_carlo_analysis()
    if not mc_results.empty:
        print("\nMonte Carlo Simulation (1000 Trials):")
        print(f"95% Confidence Win Rate: {mc_results['win_rate'].quantile(0.95):.2%}")
        print(f"Worst Case Drawdown: {mc_results['drawdown'].min():.1f}%")
    
    backtester.plot_results()