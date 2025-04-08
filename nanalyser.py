import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
import pandas_ta as ta
import xgboost as xgb
import joblib
import os
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class XAUUSD_TradingSystem:
    def __init__(self):
        """Initialize trading system with multi-timeframe analysis including 1M"""
        self.timeframes = ['1M', '5M', '15M', '1H', '4H', '1D']  # Added 1M
        self.data = {}
        self.scaler = RobustScaler()
        self.model = None
        self.model_features = []
        self.last_retrained = datetime.min
        self.min_data_length = 500
        self.model_version = "3.1-multi-timeframe-with-1M"  # Updated version

        # Directory setup
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)

    def load_and_preprocess_data(self, base_path="DATA/XAUUSD"):
        """Load and align multi-timeframe data with full OHLCV"""
        logger.info("Loading and preprocessing data...")
        try:
            for tf in self.timeframes:
                try:
                    df = pd.read_csv(
                        os.path.join(base_path, f"{tf}/{tf}.csv"),
                        parse_dates=['time'],
                        usecols=['time', 'open', 'high', 'low', 'close', 'tick_volume'],
                        dtype={
                            'open': 'float32',
                            'high': 'float32',
                            'low': 'float32',
                            'close': 'float32',
                            'tick_volume': 'float32'
                        }
                    ).rename(columns={'tick_volume': 'volume'}).set_index('time')

                    if df.empty:
                        raise ValueError(f"Empty {tf} data file")

                    freq_map = {'1M': '1min', '5M': '5min', '15M': '15min', '1H': '1h', '4H': '4h', '1D': '1d'}
                    if pd.infer_freq(df.index) != freq_map[tf]:
                        df = df.resample(freq_map[tf]).agg({
                            'open': 'first',
                            'high': 'max',
                            'low': 'min',
                            'close': 'last',
                            'volume': 'sum'
                        }).dropna()

                    cutoff_time = df.index[-1] - pd.Timedelta(days=30)
                    self.data[tf] = df[df.index >= cutoff_time].copy()

                except Exception as e:
                    logger.warning(f"Failed to load full OHLCV for {tf}: {str(e)}. Trying close-only.")
                    try:
                        df = pd.read_csv(
                            os.path.join(base_path, f"{tf}/{tf}.csv"),
                            parse_dates=['time'],
                            usecols=['time', 'close'],
                            dtype={'close': 'float32'}
                        ).set_index('time')
                        self.data[tf] = df[df.index >= cutoff_time].copy()
                    except Exception as e2:
                        logger.error(f"Skipping {tf}: {str(e2)}")
                        continue

            # Align all timeframes to 5M index (keeping 5M as base for consistency)
            base_index = self.data['5M'].index
            for tf in self.timeframes:
                if tf != '5M' and tf in self.data:
                    self.data[tf] = self.data[tf].reindex(base_index, method='ffill')

            return True

        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}", exc_info=True)
            return False

    def calculate_features(self):
        """Feature engineering across all timeframes with optimized concatenation"""
        logger.info("Calculating features across all timeframes...")
        feature_dfs = []
        base_index = self.data['5M'].index

        # Calculate features for each timeframe
        for tf in self.timeframes:
            if tf not in self.data:
                logger.warning(f"No data available for {tf}")
                continue

            df = self.data[tf].copy()
            if df.empty:
                logger.warning(f"Empty DataFrame for {tf}")
                continue

            # Rename original OHLCV columns
            if 'open' in df.columns:
                df[f'{tf}_open'] = df['open']
                df[f'{tf}_high'] = df['high']
                df[f'{tf}_low'] = df['low']
                df[f'{tf}_close'] = df['close']
                if 'volume' in df.columns:
                    df[f'{tf}_volume'] = df['volume']

            # Base price features
            df[f'{tf}_returns'] = df[f'{tf}_close'].pct_change(fill_method=None)
            df[f'{tf}_volatility'] = df[f'{tf}_returns'].rolling(20).std()

            # Technical indicators
            if all(col in df.columns for col in [f'{tf}_open', f'{tf}_high', f'{tf}_low', f'{tf}_close']):
                df[f'{tf}_EMA_50'] = ta.ema(df[f'{tf}_close'], length=50)
                df[f'{tf}_EMA_200'] = ta.ema(df[f'{tf}_close'], length=200)
                df[f'{tf}_RSI'] = ta.rsi(df[f'{tf}_close'], length=14)
                df[f'{tf}_MACD'] = ta.macd(df[f'{tf}_close'])['MACD_12_26_9']
                df[f'{tf}_ATR'] = ta.atr(df[f'{tf}_high'], df[f'{tf}_low'], df[f'{tf}_close'], length=14)
                if f'{tf}_volume' in df.columns:
                    df[f'{tf}_VWAP'] = ta.vwap(df[f'{tf}_high'], df[f'{tf}_low'], df[f'{tf}_close'], df[f'{tf}_volume'])

            # Lag features
            for lag in [1, 2, 3, 5, 10]:
                df[f'{tf}_return_lag_{lag}'] = df[f'{tf}_returns'].shift(lag)
                if f'{tf}_volume' in df.columns:
                    df[f'{tf}_volume_lag_{lag}'] = df[f'{tf}_volume'].shift(lag).pct_change(fill_method=None)

            # Reindex to 5M base and drop unprefixed columns
            df = df.reindex(base_index, method='ffill')
            df = df.drop(columns=['open', 'high', 'low', 'close', 'volume'], errors='ignore')
            feature_dfs.append(df)

        # Combine all timeframe features efficiently
        combined_df = pd.concat(feature_dfs, axis=1)

        # Interaction terms (updated to include 1M)
        if '5M_EMA_200' in combined_df.columns and '1H_EMA_200' in combined_df.columns:
            combined_df['5M_1H_EMA_diff'] = (combined_df['5M_EMA_50'] - combined_df['1H_EMA_200']) / combined_df['1H_EMA_200']
        if '1M_EMA_200' in combined_df.columns and '5M_EMA_200' in combined_df.columns:
            combined_df['1M_5M_EMA_diff'] = (combined_df['1M_EMA_50'] - combined_df['5M_EMA_200']) / combined_df['5M_EMA_200']
        if '5M_VWAP' in combined_df.columns:
            combined_df['5M_VWAP_spread'] = (combined_df['5M_close'] - combined_df['5M_VWAP']) / combined_df['5M_VWAP']

        return combined_df.replace([np.inf, -np.inf], np.nan).ffill().dropna()

    def create_target(self, df):
        """Binary target: 1=LONG, 0=SHORT based on 5M future returns"""
        logger.info("Creating binary target variable...")
        future_returns = df['5M_close'].pct_change(6).shift(-6)
        df['target'] = np.where(future_returns > 0, 1, 0)
        return df.dropna()

    def train_model(self):
        """Binary classification training with multi-timeframe features"""
        logger.info("Training binary classifier...")
        try:
            df = self.calculate_features().pipe(self.create_target)
            
            if len(df) < self.min_data_length:
                raise ValueError(f"Insufficient data ({len(df)} < {self.min_data_length})")

            self.model_features = [col for col in df.columns 
                                 if col not in ['target', '5M_open', '5M_high', '5M_low', '5M_close', '5M_volume']]
            X = df[self.model_features]
            y = df['target']

            tscv = TimeSeriesSplit(n_splits=5)
            best_score = 0
            best_model = None

            for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                model = xgb.XGBClassifier(
                    n_estimators=1000,
                    max_depth=7,
                    learning_rate=0.01,
                    subsample=0.8,
                    colsample_bytree=0.7,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    objective='binary:logistic',
                    scale_pos_weight=np.sum(y_train == 0) / np.sum(y_train == 1),
                    tree_method='hist',
                    random_state=42
                )

                pipeline = make_pipeline(RobustScaler(), model)
                pipeline.fit(X_train, y_train)

                preds = pipeline.predict(X_test)
                report = classification_report(y_test, preds, output_dict=True)
                precision = report['weighted avg']['precision']
                
                logger.info(f"Fold {fold+1} Precision: {precision:.2%}")
                logger.info(f"Confusion Matrix:\n{confusion_matrix(y_test, preds)}")

                if precision > best_score:
                    best_score = precision
                    best_model = pipeline
                    logger.info(f"New best model (Precision: {precision:.2%})")

            self.model = best_model
            self.last_retrained = datetime.now()
            
            joblib.dump(self.model, os.path.join(self.model_dir, 'model.joblib'))
            pd.Series(self.model_features).to_csv(
                os.path.join(self.model_dir, 'features.csv'), header=False)
            
            logger.info(f"Training complete. Best precision: {best_score:.2%}")
            return True

        except Exception as e:
            logger.error(f"Training failed: {str(e)}", exc_info=True)
            return False

    def generate_signal(self):
        """Generate LONG/SHORT signal using multi-timeframe features"""
        try:
            if (self.model is None or 
                (datetime.now() - self.last_retrained) > timedelta(hours=1)):
                if not self.train_model():
                    raise RuntimeError("Model training failed")

            latest_data = self.calculate_features().iloc[[-1]]
            features = latest_data[self.model_features]

            prob_long = self.model.predict_proba(features)[0][1]
            signal = 1 if prob_long >= 0.5 else 0

            atr = latest_data['5M_ATR'].values[0] if '5M_ATR' in latest_data.columns else 1.0
            price = latest_data['5M_close'].values[0]

            return {
                'timestamp': latest_data.index[0].strftime('%Y-%m-%d %H:%M:%S'),
                'signal': 'LONG' if signal == 1 else 'SHORT',
                'confidence': float(prob_long if signal == 1 else 1 - prob_long),
                'entry': float(price),
                'take_profit': float(price + (3 * atr)) if signal == 1 else float(price - (3 * atr)),
                'stop_loss': float(price - (1.5 * atr)) if signal == 1 else float(price + (1.5 * atr)),
                'atr': float(atr),
                'model_version': self.model_version,
                'status': 'SUCCESS'
            }

        except Exception as e:
            logger.error(f"Signal generation failed: {str(e)}", exc_info=True)
            return {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'signal': 'ERROR',
                'error': str(e),
                'status': 'FAILED'
            }

def main():
    """Deployment entry point"""
    print("=== XAUUSD Multi-Timeframe Trading System ===")
    print(f"Initialized at {datetime.now()}")
    
    trader = XAUUSD_TradingSystem()
    
    if not trader.load_and_preprocess_data():
        print("❌ Failed to load data - check logs")
        return
    
    signal = trader.generate_signal()
    
    if signal['status'] == 'SUCCESS':
        print("\n=== TRADING SIGNAL ===")
        print(f"Time:    {signal['timestamp']}")
        print(f"Signal:  {signal['signal']} (Confidence: {signal['confidence']:.1%})")
        print(f"Price:   {signal['entry']:.2f}")
        print(f"ATR:     {signal['atr']:.2f}")
        print(f"TP:      {signal['take_profit']:.2f}")
        print(f"SL:      {signal['stop_loss']:.2f}")
        print(f"Model:   {signal['model_version']}")
    else:
        print("\n=== ERROR ===")
        print(f"Failed: {signal.get('error', 'Unknown error')}")
    
    print("\nSystem ready for next execution")

if __name__ == "__main__":
    main()