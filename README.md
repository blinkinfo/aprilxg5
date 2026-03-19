# BTC 5-Minute Signal Bot

An ML-powered Telegram bot that predicts Bitcoin 5-minute candle direction using XGBoost with multi-timeframe feature engineering. Uses MEXC exchange data.

## Features

- **ML Predictions**: XGBoost classifier with 50+ engineered features
- **Multi-Timeframe Analysis**: Combines 5m, 15m, and 1h data
- **Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, ADX, MFI, OBV, Stochastic RSI
- **Signal Tracking**: Win/Loss, Win Rate, PnL, Streaks
- **Auto-Retrain**: Model retrains periodically on fresh data
- **Telegram Commands**: /stats, /recent, /status, /retrain

## Quick Deploy to Railway

1. Fork/clone this repo
2. Create a new Railway project
3. Connect your GitHub repo
4. Set environment variables:
   - `TELEGRAM_BOT_TOKEN` - Get from [@BotFather](https://t.me/BotFather)
   - `TELEGRAM_CHAT_ID` - Get by sending /start to your bot
5. Deploy!

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `TELEGRAM_BOT_TOKEN` | Yes | - | Telegram bot token |
| `TELEGRAM_CHAT_ID` | Yes | - | Chat ID for signals |
| `TRADING_SYMBOL` | No | BTCUSDT | Trading pair |
| `PREDICTION_THRESHOLD` | No | 0.52 | Min confidence for signal |
| `RETRAIN_INTERVAL_HOURS` | No | 6 | Hours between retrains |
| `LOOKBACK_CANDLES` | No | 100 | Candles for prediction |
| `LOG_LEVEL` | No | INFO | Logging level |

## How It Works

1. **Data Collection**: Fetches OHLCV candles from MEXC API (5m, 15m, 1h)
2. **Feature Engineering**: Computes 50+ technical indicators and price action features
3. **Prediction**: XGBoost model predicts next candle direction (UP/DOWN)
4. **Signal Delivery**: Sends signal to Telegram 15 seconds before candle close
5. **Tracking**: Records entry price, resolves at next candle close, tracks P&L
6. **Auto-Retrain**: Retrains on fresh data every 6 hours

## Telegram Commands

- `/start` - Welcome message and chat ID
- `/stats` - Full performance stats (W/L, PnL, streaks)
- `/recent` - Last 10 signals with results
- `/status` - Bot and model status
- `/retrain` - Force model retraining
- `/help` - Help message

## Local Development

```bash
# Clone
git clone https://github.com/blinkinfo/aprilxg.git
cd aprilxg

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your values

# Run
python main.py
```

## Architecture

```
src/
  config.py         - Configuration (env vars, model params)
  data_fetcher.py   - MEXC API client for candle data
  features.py       - Feature engineering (50+ indicators)
  model.py          - XGBoost training, evaluation, prediction
  signal_tracker.py - Signal tracking with PnL & stats
  telegram_bot.py   - Telegram bot commands & messaging
  bot.py            - Main orchestrator
main.py             - Entry point
```

## Model Details

- **Algorithm**: XGBoost (Gradient Boosted Trees)
- **Features**: 50+ including RSI, MACD, BBands, ATR, ADX, MFI, OBV, multi-TF trends
- **Training**: Time-series cross-validation (5 folds)
- **Evaluation**: Chronological 80/20 split (no data leakage)
- **Retraining**: Every 6 hours on latest 5000 candles
