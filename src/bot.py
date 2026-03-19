"""Main bot orchestrator - ties everything together.

Preserves the 15-second pre-signal timing by design.
Integrates: retraining gate messaging, signal strength labels, Optuna status.
"""
import asyncio
import logging
import os
import signal
from datetime import datetime, timezone

from .config import BotConfig
from .data_fetcher import MEXCFetcher
from .features import FeatureEngineer
from .model import PredictionModel
from .signal_tracker import SignalTracker
from .telegram_bot import TelegramBot

logger = logging.getLogger(__name__)


class SignalBot:
    """Main signal bot orchestrator."""

    def __init__(self, config: BotConfig):
        self.config = config
        self.fetcher = MEXCFetcher(config.mexc)
        self.model = PredictionModel(config.model)
        self.tracker = SignalTracker(config.data_dir)
        self.telegram = TelegramBot(config.telegram)
        self._running = False
        self._last_signal_candle_ts = None  # Prevent duplicate signals per candle

    async def start(self):
        """Start the bot."""
        logger.info("="*50)
        logger.info("BTC 5m Signal Bot starting (aprilxg v2)...")
        logger.info("="*50)

        # Create directories
        os.makedirs(self.config.data_dir, exist_ok=True)
        os.makedirs(self.config.model_dir, exist_ok=True)

        # Initialize Telegram bot
        await self.telegram.initialize()
        self.telegram.set_callbacks(
            stats_cb=self._get_stats_text,
            recent_cb=self._get_recent_text,
            status_cb=self._get_status_text,
            retrain_cb=self._retrain_model,
        )
        await self.telegram.start_polling()

        # Try loading existing model
        loaded = self.model.load(self.config.model_dir)
        if not loaded:
            logger.info("No saved model found, training initial model...")
            await self._train_model()
        else:
            logger.info(f"Model loaded (val_acc={self.model.val_accuracy:.4f})")

        # Send startup message
        optuna_status = "ON" if self.config.model.enable_optuna_tuning else "OFF"
        await self.telegram.send_message(
            "BTC 5m Signal Bot ONLINE (aprilxg v2)\n\n"
            f"Model accuracy: {self.model.val_accuracy:.1%}\n"
            f"Confidence threshold: {self.config.model.confidence_min:.0%}\n"
            f"Training data: {self.config.model.train_candles} candles (~{self.config.model.train_candles * 5 // 1440} days)\n"
            f"Optuna tuning: {optuna_status}\n"
            f"Retraining gate: {self.config.model.retrain_min_improvement:.3f} min improvement\n"
            f"Tracked signals: {len(self.tracker.signals)}\n"
            f"Symbol: {self.config.mexc.symbol}\n\n"
            "Signals will be posted automatically.\n"
            "Use /stats /recent /status for info."
        )

        # Main loop
        self._running = True
        await self._main_loop()

    async def stop(self):
        """Stop the bot gracefully."""
        logger.info("Stopping bot...")
        self._running = False
        await self.telegram.send_message("Bot shutting down...")
        await self.telegram.stop()
        await self.fetcher.close()
        self.model.save(self.config.model_dir)
        logger.info("Bot stopped")

    async def _main_loop(self):
        """Main prediction loop.

        CRITICAL: The 15-second pre-signal timing is preserved here.
        prediction_lead_seconds controls when the signal fires before candle close.
        """
        logger.info("Entering main prediction loop")

        while self._running:
            try:
                now = datetime.now(timezone.utc)

                # Check if it's time to predict (15 seconds before candle close)
                seconds_in_candle = (now.minute % 5) * 60 + now.second
                candle_duration = 5 * 60  # 300 seconds
                seconds_until_close = candle_duration - seconds_in_candle

                if seconds_until_close <= self.config.prediction_lead_seconds and seconds_until_close > 0:
                    # Calculate current candle's open timestamp for dedup
                    candle_open_minute = now.minute - (now.minute % 5)
                    candle_ts = now.replace(minute=candle_open_minute, second=0, microsecond=0)

                    if self._last_signal_candle_ts != candle_ts:
                        await self._run_prediction_cycle()
                        self._last_signal_candle_ts = candle_ts

                # Resolve pending signals after candle close (within first 15 seconds)
                if seconds_in_candle < 15 and seconds_in_candle >= 5:
                    await self._resolve_pending_signals()

                # Check if model needs retraining
                if self.model.needs_retrain():
                    logger.info("Model retrain interval reached")
                    await self._train_model()

                await asyncio.sleep(self.config.main_loop_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}", exc_info=True)
                await asyncio.sleep(10)

    async def _run_prediction_cycle(self):
        """Fetch data and make a prediction."""
        try:
            # Fetch multi-timeframe data
            data = await self.fetcher.fetch_multi_timeframe(
                intervals=["5m", "15m", "1h"],
                limit=self.config.model.lookback_candles,
            )

            df_5m = data.get("5m")
            if df_5m is None or df_5m.empty:
                logger.warning("No 5m data available")
                return

            higher_tf = {k: v for k, v in data.items() if k != "5m" and not v.empty}

            # Make prediction
            prediction = self.model.predict(df_5m, higher_tf)

            if prediction["signal"] in ("UP", "DOWN"):
                # Record signal
                sig = self.tracker.add_signal(
                    direction=prediction["signal"],
                    confidence=prediction["confidence"],
                    entry_price=prediction["current_price"],
                )

                # Send to Telegram with strength indicator
                msg = self.tracker.format_signal_message(sig, prediction)
                await self.telegram.send_message(msg)
                logger.info(
                    f"Signal sent: {prediction['signal']} [{prediction.get('strength', 'NORMAL')}] "
                    f"@ ${prediction['current_price']:,.2f}"
                )
            else:
                logger.info(
                    f"No signal: confidence below {self.config.model.confidence_min} "
                    f"({prediction['confidence']:.4f})"
                )

        except Exception as e:
            logger.error(f"Prediction cycle error: {e}", exc_info=True)

    async def _resolve_pending_signals(self):
        """Resolve pending signals with the latest candle close price."""
        pending = self.tracker.get_pending_signals()
        if not pending:
            return

        try:
            # Fetch latest candle to get close price
            df = await self.fetcher.fetch_klines(interval="5m", limit=2)
            if df.empty or len(df) < 2:
                return

            # The second-to-last candle is the one that just closed
            close_price = float(df["close"].iloc[-2])

            for sig in pending:
                resolved = self.tracker.resolve_signal(sig.signal_id, close_price)
                if resolved:
                    msg = self.tracker.format_resolution_message(resolved)
                    await self.telegram.send_message(msg)

        except Exception as e:
            logger.error(f"Signal resolution error: {e}", exc_info=True)

    async def _train_model(self):
        """Train or retrain the model."""
        try:
            logger.info("Fetching training data...")
            # Fetch historical 5m data (now 43,200 candles = ~150 days)
            df_5m = await self.fetcher.fetch_historical_klines(
                interval="5m",
                total_candles=self.config.model.train_candles,
            )

            if df_5m.empty or len(df_5m) < 500:
                logger.error(f"Insufficient training data: {len(df_5m)} candles")
                return

            # Fetch higher timeframe data
            higher_tf = await self.fetcher.fetch_multi_timeframe(
                intervals=["15m", "1h"],
                limit=500,
            )
            higher_tf = {k: v for k, v in higher_tf.items() if not v.empty}

            # Train (model internally handles the retraining gate)
            metrics = self.model.train(df_5m, higher_tf)

            # Save model
            self.model.save(self.config.model_dir)

            # Notify with retraining gate status
            swapped_str = "NEW MODEL ACTIVE" if metrics["model_swapped"] else "KEPT PREVIOUS MODEL (new one wasn't better)"
            optuna_str = "Optuna-tuned" if metrics.get("optuna_tuned") else "Default params"

            msg = (
                "Model Training Complete\n\n"
                f"Result: {swapped_str}\n\n"
                f"Training samples: {metrics['total_samples']}\n"
                f"Features: {metrics['n_features']}\n"
                f"New model val accuracy: {metrics['val_accuracy']:.1%}\n"
                f"Active model val accuracy: {metrics['active_val_accuracy']:.1%}\n"
                f"CV accuracy: {metrics['cv_accuracy']:.1%}\n"
                f"Val log loss: {metrics['val_logloss']:.4f}\n"
                f"Params: {optuna_str}"
            )
            await self.telegram.send_message(msg)
            logger.info("Model training complete")

        except Exception as e:
            logger.error(f"Training error: {e}", exc_info=True)
            await self.telegram.send_message(f"Model training failed: {str(e)[:200]}")

    # --- Callback methods for Telegram commands ---

    def _get_stats_text(self) -> str:
        return self.tracker.format_stats_message()

    def _get_recent_text(self) -> str:
        recent = self.tracker.get_recent_signals(10)
        if not recent:
            return "No signals recorded yet."

        lines = ["---------- RECENT SIGNALS ----------"]
        for s in recent:
            result_str = s.result or "PENDING"
            pnl_str = f"{s.pnl_pct:+.4f}%" if s.pnl_pct is not None else "---"
            lines.append(
                f"#{s.signal_id} | {s.direction} | ${s.entry_price:,.2f} | "
                f"{result_str} | {pnl_str} | conf={s.confidence:.1%}"
            )
        lines.append("------------------------------------")
        return "\n".join(lines)

    async def _get_status_text(self) -> str:
        stats = self.tracker.get_stats()
        retrain_in = "N/A"
        if self.model.last_train_time:
            elapsed = (datetime.now(timezone.utc) - self.model.last_train_time).total_seconds()
            remaining = max(0, self.config.model.retrain_interval_hours * 3600 - elapsed)
            hours = int(remaining // 3600)
            minutes = int((remaining % 3600) // 60)
            retrain_in = f"{hours}h {minutes}m"

        optuna_status = "ON" if self.config.model.enable_optuna_tuning else "OFF"
        tuned_str = "Yes" if self.model.best_xgb_params else "No (using defaults)"

        return (
            "---------- BOT STATUS (v2) ----------\n"
            f"Status: {'RUNNING' if self._running else 'STOPPED'}\n"
            f"Symbol: {self.config.mexc.symbol}\n"
            f"Model accuracy: {self.model.val_accuracy:.1%}\n"
            f"Training samples: {self.model.train_samples}\n"
            f"Last trained: {self.model.last_train_time.strftime('%Y-%m-%d %H:%M') if self.model.last_train_time else 'Never'}\n"
            f"Next retrain in: {retrain_in}\n"
            f"Confidence threshold: {self.config.model.confidence_min:.0%}\n"
            f"Retraining gate: {self.config.model.retrain_min_improvement:.3f}\n"
            f"Optuna: {optuna_status} | Tuned: {tuned_str}\n"
            f"Total signals: {stats.total_signals}\n"
            f"Pending: {stats.pending}\n"
            "--------------------------------------"
        )

    async def _retrain_model(self) -> str:
        try:
            await self._train_model()
            return f"Retrain complete! Active model val accuracy: {self.model.val_accuracy:.1%}"
        except Exception as e:
            return f"Retrain failed: {str(e)[:200]}"


async def run_bot():
    """Entry point to run the bot."""
    config = BotConfig.from_env()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Reduce noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.WARNING)

    bot = SignalBot(config)

    # Handle graceful shutdown
    loop = asyncio.get_event_loop()
    for sig_name in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig_name, lambda: asyncio.create_task(bot.stop()))

    try:
        await bot.start()
    except KeyboardInterrupt:
        await bot.stop()
    except Exception as e:
        logger.error(f"Bot crashed: {e}", exc_info=True)
        await bot.stop()
        raise
