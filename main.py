#!/usr/bin/env python
"""
DISCLAIMER:
This code is provided for educational and research purposes only.
Trading involves significant risk and you should not use this code with real funds
without extensive testing, proper risk management, and professional advice.
Past performance is not indicative of future results.
"""

import MetaTrader5 as mt5
import time
import pandas as pd
from datetime import datetime, timedelta
import os
import numpy as np
from sklearn.linear_model import LogisticRegression

# -------------------------------
# Base Trading Bot (GoldTrader)
# -------------------------------
class GoldTrader:
    def __init__(self, symbol="XAUUSD", lot_size=0.01):
        self.symbol = symbol
        self.lot_size = lot_size
        self.connected = self.connect_mt5()
        self.current_bid = None
        self.current_ask = None
        self.breakout_executed = False  # For potential breakout strategy

    def connect_mt5(self):
        """Connect to MetaTrader 5."""
        if not mt5.initialize():
            print(f"MT5 Initialization failed: {mt5.last_error()}")
            return False
        print("✅ Connected to MetaTrader 5!")
        return True

    def get_current_price(self):
        """Fetch the latest bid/ask prices."""
        tick = mt5.symbol_info_tick(self.symbol)
        if tick:
            self.current_bid = tick.bid
            self.current_ask = tick.ask

    def display_price(self):
        """Display the current price on the console."""
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"📊 {self.symbol} - Bid: {self.current_bid}, Ask: {self.current_ask}")

    def execute_trade(self, action="buy"):
        """Execute a market trade (buy or sell)."""
        if not self.connected:
            print("❌ Not connected to MT5!")
            return False

        order_type = mt5.ORDER_TYPE_BUY if action == "buy" else mt5.ORDER_TYPE_SELL
        price = self.current_ask if action == "buy" else self.current_bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": self.lot_size,
            "type": order_type,
            "price": price,
            "deviation": 10,
            "magic": 123456,
            "comment": f"{action.upper()} Order",
            "type_filling": mt5.ORDER_FILLING_IOC,
            "type_time": mt5.ORDER_TIME_GTC
        }

        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"\n✅ {action.upper()} order executed successfully at {price}")
            time.sleep(1)
            return True
        else:
            print(f"\n❌ Trade failed: {result.comment}")
            return False

    def shutdown(self):
        """Shutdown the connection to MetaTrader 5."""
        mt5.shutdown()


# -----------------------------------------------------------
# High Frequency Machine Learning Scalping Bot (Subclass)
# -----------------------------------------------------------
class HighFrequencyScalpingBot(GoldTrader):
    def __init__(self, symbol="XAUUSD", lot_size=0.01):
        super().__init__(symbol, lot_size)
        # Train a dummy ML model for trade signal predictions.
        self.model = None
        self.train_model()
        # For feature extraction: track previous bid to calculate short-term price change.
        self.previous_bid = None

    def train_model(self):
        """
        Train a dummy logistic regression model on synthetic data.
        Features: [spread, price_change]
        Target classes: 0 = HOLD, 1 = BUY, 2 = SELL
        """
        np.random.seed(42)
        X = np.random.rand(1000, 2)  # 1000 samples, 2 features
        # Generate synthetic labels with an arbitrary rule:
        # if spread < 0.5 and price_change > 0.5 then BUY; if spread >= 0.5 and price_change < 0.5 then SELL; else HOLD.
        y = []
        for features in X:
            spread, change = features
            if spread < 0.5 and change > 0.5:
                y.append(1)  # BUY
            elif spread >= 0.5 and change < 0.5:
                y.append(2)  # SELL
            else:
                y.append(0)  # HOLD
        y = np.array(y)
        self.model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
        self.model.fit(X, y)
        print("✅ Dummy ML model trained for scalping predictions.")

    def extract_features(self):
        """
        Extract features for the ML model.
        Feature 1: Spread = current_ask - current_bid
        Feature 2: Price change = current_bid - previous_bid (if available, else 0)
        """
        if self.current_bid is None or self.current_ask is None:
            return np.array([0, 0])
        spread = self.current_ask - self.current_bid
        price_change = 0 if self.previous_bid is None else self.current_bid - self.previous_bid
        features = np.array([spread, price_change])
        return features.reshape(1, -1)

    def predict_trade_signal(self):
        """
        Predict trade signal using the ML model.
        Returns: "buy", "sell", or "hold"
        """
        features = self.extract_features()
        prediction = self.model.predict(features)[0]
        if prediction == 1:
            return "buy"
        elif prediction == 2:
            return "sell"
        else:
            return "hold"

    def execute_scalping_trade(self):
        """
        Check ML prediction and execute the trade if signal is BUY or SELL.
        """
        signal = self.predict_trade_signal()
        print(f"🤖 ML Prediction: {signal.upper()}")
        if signal in ["buy", "sell"]:
            if self.execute_trade(signal):
                print(f"✅ Scalping {signal.upper()} trade executed.")
            else:
                print(f"❌ Failed to execute {signal.upper()} trade.")
        else:
            print("ℹ️ No trade signal detected. Holding position.")

    def run_bot(self, duration_seconds=3600):
        """
        Run the scalping bot for a given duration (in seconds).
        Default is 1 hour (3600 seconds).
        """
        start_time = time.time()
        try:
            while time.time() - start_time < duration_seconds:
                # Save previous bid before updating
                if self.current_bid is not None:
                    self.previous_bid = self.current_bid

                self.get_current_price()
                self.display_price()
                self.execute_scalping_trade()

                # High frequency: adjust the sleep time as needed.
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
    print("🚀 Starting High Frequency Machine Learning Scalping Bot...")
    bot = HighFrequencyScalpingBot(symbol="XAUUSD", lot_size=0.01)
    # Run the bot for 1 hour (3600 seconds); adjust as necessary.
    bot.run_bot(duration_seconds=3600)
