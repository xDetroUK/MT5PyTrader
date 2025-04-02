#!/usr/bin/env python
"""
DISCLAIMER:
This code is provided for educational purposes only.
Trading involves significant risk. Do not use this code with real funds without thorough testing,
proper risk management, and professional guidance.
"""

import MetaTrader5 as mt5
import time
import math
import os
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression

# -------------------------------
# Base Trading Bot (GoldTrader)
# -------------------------------
class GoldTrader:
    def __init__(self, symbol="XAUUSD", default_lot=0.01):
        self.symbol = symbol
        self.default_lot = default_lot  # Fallback lot size if account info is unavailable
        self.connected = self.connect_mt5()
        self.current_bid = None
        self.current_ask = None

    def connect_mt5(self):
        """Initialize connection to MetaTrader 5."""
        if not mt5.initialize():
            print(f"MT5 Initialization failed: {mt5.last_error()}")
            return False
        print("✅ Connected to MetaTrader 5!")
        return True

    def get_current_price(self):
        """Update the current bid and ask prices."""
        tick = mt5.symbol_info_tick(self.symbol)
        if tick:
            self.current_bid = tick.bid
            self.current_ask = tick.ask

    def display_price(self):
        """Clear the console and display the current price."""
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"📊 {self.symbol} - Bid: {self.current_bid}, Ask: {self.current_ask}")

    def shutdown(self):
        """Shutdown the MT5 connection."""
        mt5.shutdown()


# -----------------------------------------------------------
# High Frequency Machine Learning Scalping Bot (Subclass)
# -----------------------------------------------------------
class HighFrequencyScalpingBot(GoldTrader):
    def __init__(self, symbol="XAUUSD"):
        super().__init__(symbol)
        self.model = None
        self.train_model()
        self.previous_bid = None  # For short-term price change calculations

    def train_model(self):
        """
        Train a dummy logistic regression model on synthetic data.
        Features: [spread, price_change]
        Labels: 0 = HOLD, 1 = BUY, 2 = SELL
        (This model is for demonstration only.)
        """
        np.random.seed(42)
        X = np.random.rand(1000, 2)
        y = []
        for features in X:
            spread, change = features
            if spread < 0.5 and change > 0.5:
                y.append(1)  # BUY signal
            elif spread >= 0.5 and change < 0.5:
                y.append(2)  # SELL signal
            else:
                y.append(0)  # HOLD
        y = np.array(y)
        self.model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
        self.model.fit(X, y)
        print("✅ Dummy ML model trained for scalping predictions.")

    def extract_features(self):
        """
        Extract features for the ML model:
          - Spread: current_ask - current_bid
          - Price Change: current_bid - previous_bid
        """
        if self.current_bid is None or self.current_ask is None:
            return np.array([0, 0]).reshape(1, -1)
        spread = self.current_ask - self.current_bid
        price_change = 0 if self.previous_bid is None else self.current_bid - self.previous_bid
        return np.array([spread, price_change]).reshape(1, -1)

    def predict_trade_signal(self):
        """Predict whether to BUY, SELL, or HOLD using the dummy ML model."""
        features = self.extract_features()
        prediction = self.model.predict(features)[0]
        if prediction == 1:
            return "buy"
        elif prediction == 2:
            return "sell"
        else:
            return "hold"

    def get_account_info(self):
        """Retrieve account information from MT5."""
        account_info = mt5.account_info()
        if account_info is None:
            print("❌ Could not retrieve account info.")
        return account_info

    def calculate_maximum_lot(self, action="buy"):
        """
        Calculate the maximum lot size based on using 90% of the account balance as margin.
        The calculation uses the symbol's margin requirement (margin_initial).
        """
        account = self.get_account_info()
        if account is None:
            return self.default_lot
        balance = account.balance
        available_margin = balance * 0.9
        self.get_current_price()
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            print("❌ Symbol info not available, using default lot size.")
            return self.default_lot
        margin_per_lot = symbol_info.margin_initial
        if margin_per_lot is None or margin_per_lot <= 0:
            print("❌ Margin per lot not available, using default lot size.")
            return self.default_lot
        max_possible = available_margin / margin_per_lot
        min_vol = symbol_info.volume_min
        max_vol = symbol_info.volume_max
        vol_step = symbol_info.volume_step
        max_possible = math.floor(max_possible / vol_step) * vol_step
        if max_possible < min_vol:
            print("❌ Insufficient margin to open any trade.")
            return 0
        max_possible = min(max_possible, max_vol)
        print(f"💰 Calculated maximum lot size for {action.upper()} trade: {max_possible}")
        return max_possible

    def execute_max_trade(self, action="buy", user_risk_distance=0.03, reward_multiplier=2):
        """
        Execute a market order using the maximum lot size calculated from 90% of balance.
        The stop loss (SL) and take profit (TP) are set based on a risk distance computed as
        the maximum of the user risk distance and (current spread + a small buffer).
        """
        if not self.connected:
            print("❌ Not connected to MT5!")
            return False

        self.get_current_price()
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            print("❌ Symbol info not available!")
            return False

        # Compute current spread and set risk_distance as max(user_risk_distance, current spread + 0.05)
        current_spread = self.current_ask - self.current_bid
        risk_distance = max(user_risk_distance, current_spread + 0.05)

        if action == "buy":
            entry_price = self.current_ask
            sl = entry_price - risk_distance
            tp = entry_price + risk_distance * reward_multiplier
        else:
            entry_price = self.current_bid
            sl = entry_price + risk_distance
            tp = entry_price - risk_distance * reward_multiplier

        lot_size = self.calculate_maximum_lot(action)
        if lot_size <= 0:
            print("❌ Cannot open trade due to insufficient margin.")
            return False

        order_type = mt5.ORDER_TYPE_BUY if action == "buy" else mt5.ORDER_TYPE_SELL
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lot_size,
            "type": order_type,
            "price": entry_price,
            "sl": sl,
            "tp": tp,
            "deviation": 10,
            "magic": 123456,
            "comment": f"Max {action.upper()} Order",
            "type_filling": mt5.ORDER_FILLING_IOC,
            "type_time": mt5.ORDER_TIME_GTC
        }
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"\n✅ {action.upper()} order executed at {entry_price} with lot size {lot_size}")
            print(f"    SL: {sl} | TP: {tp}")
            return True
        else:
            print(f"\n❌ Trade failed: {result.comment}")
            return False

    def close_trade(self):
        """
        Force-close all open positions for the symbol.
        This is used if a trade exceeds the maximum allowed duration (3 minutes).
        """
        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None or len(positions) == 0:
            return True
        for pos in positions:
            order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(self.symbol).bid if pos.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(self.symbol).ask
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": pos.volume,
                "position": pos.ticket,
                "type": order_type,
                "price": price,
                "deviation": 10,
                "magic": 123456,
                "comment": "Manual close",
                "type_filling": mt5.ORDER_FILLING_IOC,
                "type_time": mt5.ORDER_TIME_GTC
            }
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"✅ Position {pos.ticket} closed manually.")
            else:
                print(f"❌ Failed to close position {pos.ticket}: {result.comment}")
        return True

    def wait_for_trade_close(self, timeout=180):
        """
        Wait until there are no open positions for the symbol.
        If after 3 minutes (180 seconds) the trade is still open, force-close it.
        """
        start = time.time()
        while time.time() - start < timeout:
            positions = mt5.positions_get(symbol=self.symbol)
            if positions is None or len(positions) == 0:
                print("✅ Trade closed within time limit.")
                return True
            time.sleep(2)
        print("⚠️ 3 minutes elapsed; forcing trade closure...")
        self.close_trade()
        return True

    def run_bot(self, duration_seconds=3600, user_risk_distance=0.03, reward_multiplier=2):
        """
        Run the scalping bot for the specified duration (default 1 hour).
        The bot will open one trade at a time using 90% of the balance as margin and
        enforce a maximum trade duration of 3 minutes.
        """
        start_time = time.time()
        try:
            while time.time() - start_time < duration_seconds:
                if self.current_bid is not None:
                    self.previous_bid = self.current_bid
                self.get_current_price()
                self.display_price()
                positions = mt5.positions_get(symbol=self.symbol)
                if positions is None or len(positions) == 0:
                    signal = self.predict_trade_signal()
                    print(f"🤖 ML Prediction: {signal.upper()}")
                    if signal in ["buy", "sell"]:
                        if self.execute_max_trade(signal, user_risk_distance, reward_multiplier):
                            print("⌛ Waiting for trade to close (max 3 minutes)...")
                            self.wait_for_trade_close(timeout=180)
                    else:
                        print("ℹ️ No actionable signal. Waiting...")
                else:
                    print("💤 Trade in progress. Monitoring...")
                time.sleep(1)
        except KeyboardInterrupt:
            print("🛑 Bot interrupted by user.")
        finally:
            self.shutdown()
            print("🔌 Disconnected from MetaTrader 5.")


# -------------------------------
# Main Execution
# -------------------------------
if __name__ == "__main__":
    print("🚀 Starting Enhanced High Frequency Scalping Bot with 90% Margin Usage and 3-min Trade Duration...")
    bot = HighFrequencyScalpingBot(symbol="XAUUSD")
    # Run the bot for 1 hour (3600 seconds). Adjust parameters as needed.
    bot.run_bot(duration_seconds=3600, user_risk_distance=0.03, reward_multiplier=2)
