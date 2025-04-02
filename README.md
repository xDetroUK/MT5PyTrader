High-Frequency Machine Learning Scalping Bot for MetaTrader 5
Overview
This Python script implements a sophisticated high-frequency trading (HFT) bot that specializes in scalping the XAUUSD (Gold/US Dollar) market using MetaTrader 5. The bot combines machine learning predictions with aggressive risk management to execute rapid trades.

Key Features
Core Functionality
MetaTrader 5 Integration: Connects directly to MT5 terminal for real-time price data and trade execution

Machine Learning Model: Uses logistic regression to generate BUY/SELL/HOLD signals

High-Frequency Trading: Designed for rapid trade execution with short holding periods

Risk Management
Aggressive Position Sizing: Uses 90% of account balance as margin for maximum position size

Automatic Trade Closure: Forces closure of any trade exceeding 3 minutes

Dynamic Stop-Loss: Calculates stop-loss based on current spread plus buffer

Technical Implementation
Class Structure
GoldTrader (Base Class)

Handles basic MT5 connection and price monitoring

Provides core functionality for all trading bots

HighFrequencyScalpingBot (Subclass)

Implements machine learning predictions

Manages aggressive trading strategy

Enforces strict risk controls

Key Methods
Machine Learning Integration

train_model(): Creates synthetic training data and trains logistic regression model

predict_trade_signal(): Generates real-time trading signals

Trade Execution

execute_max_trade(): Opens positions using 90% of account balance

close_trade(): Force-closes all open positions

wait_for_trade_close(): Enforces maximum trade duration

Risk Management

calculate_maximum_lot(): Computes position size based on available margin

Dynamic stop-loss calculation based on current market spread

Trading Strategy
Signal Generation

Uses spread and price change as model features

Predicts BUY/SELL/HOLD signals every second

Trade Execution

Enters trades immediately when signal is generated

Uses IOC (Immediate or Cancel) order filling

Sets stop-loss at max(user distance, spread + buffer)

Take-profit at 2x risk distance (configurable)

Position Management

Monitors open positions continuously

Automatically closes trades after 3 minutes

Prevents overlapping positions

Usage
python
Copy
# Initialize bot for XAUUSD
bot = HighFrequencyScalpingBot(symbol="XAUUSD")

# Run for 1 hour with:
# - 0.03 default risk distance
# - 2x reward multiplier
bot.run_bot(duration_seconds=3600, user_risk_distance=0.03, reward_multiplier=2)
Risk Warning
⚠️ This is an extremely aggressive trading strategy ⚠️

Uses 90% of account balance per trade

Designed for demonstration/educational purposes only

Requires thorough testing before live use

Includes automatic disclaimer in code

Dependencies
MetaTrader 5 Python package (MetaTrader5)

NumPy

scikit-learn

Standard Python libraries
