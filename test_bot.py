"""Comprehensive test suite for BTC Signal Bot.
Tests: MEXC API, feature engineering, model training, signal tracking, prediction.
"""
import asyncio
import sys
import os
import json
import traceback
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import BotConfig, MEXCConfig, ModelConfig
from src.data_fetcher import MEXCFetcher
from src.features import FeatureEngineer
from src.model import PredictionModel
from src.signal_tracker import SignalTracker, Signal


class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def ok(self, name):
        self.passed += 1
        print(f"  [PASS] {name}")

    def fail(self, name, error):
        self.failed += 1
        self.errors.append((name, str(error)))
        print(f"  [FAIL] {name}: {error}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*50}")
        print(f"TEST RESULTS: {self.passed}/{total} passed")
        if self.errors:
            print(f"\nFailed tests:")
            for name, err in self.errors:
                print(f"  - {name}: {err}")
        print(f"{'='*50}")
        return self.failed == 0


async def test_mexc_api(results: TestResults):
    """Test MEXC API connectivity and data fetching."""
    print("\n--- Testing MEXC API ---")
    config = MEXCConfig()
    fetcher = MEXCFetcher(config)

    try:
        # Test 1: Fetch 5m klines
        df_5m = await fetcher.fetch_klines(interval="5m", limit=100)
        assert not df_5m.empty, "5m klines returned empty"
        assert len(df_5m) >= 50, f"Only got {len(df_5m)} candles, expected >= 50"
        assert all(col in df_5m.columns for col in ["timestamp", "open", "high", "low", "close", "volume"]), "Missing columns"
        assert df_5m["close"].dtype == float, "Close should be float"
        assert df_5m["volume"].dtype == float, "Volume should be float"
        assert (df_5m["high"] >= df_5m["low"]).all(), "High should be >= Low"
        assert (df_5m["high"] >= df_5m["close"]).all(), "High should be >= Close"
        assert (df_5m["low"] <= df_5m["close"]).all(), "Low should be <= Close"
        results.ok(f"5m klines: {len(df_5m)} candles, price=${df_5m['close'].iloc[-1]:,.2f}")

        # Test 2: Fetch 15m klines
        df_15m = await fetcher.fetch_klines(interval="15m", limit=100)
        assert not df_15m.empty, "15m klines returned empty"
        results.ok(f"15m klines: {len(df_15m)} candles")

        # Test 3: Fetch 1h klines
        df_1h = await fetcher.fetch_klines(interval="1h", limit=100)
        assert not df_1h.empty, "1h klines returned empty"
        results.ok(f"1h klines: {len(df_1h)} candles")

        # Test 4: Multi-timeframe fetch
        multi = await fetcher.fetch_multi_timeframe(intervals=["5m", "15m", "1h"], limit=100)
        assert len(multi) == 3, f"Expected 3 timeframes, got {len(multi)}"
        assert all(not v.empty for v in multi.values()), "Some timeframes returned empty"
        results.ok(f"Multi-TF fetch: {list(multi.keys())}")

        # Test 5: Historical pagination (forward startTime-based)
        df_hist = await fetcher.fetch_historical_klines(interval="5m", total_candles=1000)
        assert len(df_hist) >= 900, f"Historical fetch only got {len(df_hist)}, expected >= 900"
        # Check no duplicates
        assert df_hist["timestamp"].is_unique, "Historical data has duplicate timestamps"
        # Check chronological order
        assert df_hist["timestamp"].is_monotonic_increasing, "Historical data not sorted"
        results.ok(f"Historical klines: {len(df_hist)} candles (paginated)")

        return df_5m, df_15m, df_1h, df_hist

    except Exception as e:
        results.fail("MEXC API", f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return None, None, None, None
    finally:
        await fetcher.close()


def test_features(results: TestResults, df_5m, df_15m, df_1h):
    """Test feature engineering."""
    print("\n--- Testing Feature Engineering ---")
    config = ModelConfig()
    fe = FeatureEngineer(config)

    try:
        # Test 1: Basic feature computation
        features = fe.compute_features(df_5m)
        assert not features.empty, "Features returned empty"
        assert len(features) > 0, "No feature rows"
        n_features = len(features.columns)
        assert n_features >= 30, f"Only {n_features} features, expected >= 30"
        results.ok(f"Feature computation: {n_features} features, {len(features)} rows")

        # Test 2: Check no infinite values
        import numpy as np
        inf_count = np.isinf(features.select_dtypes(include=[np.number])).sum().sum()
        assert inf_count == 0, f"Found {inf_count} infinite values"
        results.ok("No infinite values in features")

        # Test 3: Check key features exist
        expected_features = [
            "rsi", "macd_histogram", "bb_pctb", "atr_norm", "adx",
            "volume_ratio", "mfi", "ema_crossover", "returns_1", "candle_body"
        ]
        for feat_name in expected_features:
            assert feat_name in features.columns, f"Missing feature: {feat_name}"
        results.ok(f"All {len(expected_features)} key features present")

        # Test 4: Multi-timeframe features
        higher_tf = {"15m": df_15m, "1h": df_1h}
        features_mtf = fe.compute_features(df_5m, higher_tf)
        mtf_cols = [c for c in features_mtf.columns if "15min" in c or "1hr" in c or "60min" in c]
        assert len(mtf_cols) > 0, "No multi-timeframe features found"
        results.ok(f"Multi-TF features: {len(mtf_cols)} MTF columns added")

        # Test 5: Labels
        labels = fe.create_labels(df_5m)
        assert len(labels) == len(df_5m), "Label count mismatch"
        assert set(labels.dropna().unique()).issubset({0, 1}), "Labels should be 0 or 1"
        up_pct = labels.mean() * 100
        results.ok(f"Labels: {up_pct:.1f}% UP, {100-up_pct:.1f}% DOWN")

        return features_mtf

    except Exception as e:
        results.fail("Feature Engineering", f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return None


def test_model(results: TestResults, df_hist, df_15m, df_1h):
    """Test model training and prediction."""
    print("\n--- Testing Model Training ---")
    config = ModelConfig()
    model = PredictionModel(config)

    try:
        # Test 1: Train model
        higher_tf = {"15m": df_15m, "1h": df_1h}
        metrics = model.train(df_hist, higher_tf)

        assert "train_accuracy" in metrics, "Missing train_accuracy"
        assert "val_accuracy" in metrics, "Missing val_accuracy"
        assert "cv_accuracy" in metrics, "Missing cv_accuracy"
        assert metrics["val_accuracy"] > 0.45, f"Val accuracy too low: {metrics['val_accuracy']}"
        results.ok(
            f"Model trained: val_acc={metrics['val_accuracy']:.4f}, "
            f"cv_acc={metrics['cv_accuracy']:.4f}, "
            f"{metrics['n_features']} features, {metrics['total_samples']} samples"
        )

        # Test 2: Check CV scores are reasonable
        cv_scores = metrics["cv_scores"]
        assert len(cv_scores) == 5, f"Expected 5 CV folds, got {len(cv_scores)}"
        assert all(0.4 < s < 0.7 for s in cv_scores), f"CV scores out of range: {cv_scores}"
        results.ok(f"CV scores: {[f'{s:.4f}' for s in cv_scores]}")

        # Test 3: Prediction
        prediction = model.predict(df_hist.tail(120), higher_tf)
        assert "signal" in prediction, "Missing signal"
        assert "confidence" in prediction, "Missing confidence"
        assert "prob_up" in prediction, "Missing prob_up"
        assert "prob_down" in prediction, "Missing prob_down"
        assert "current_price" in prediction, "Missing current_price"
        assert prediction["signal"] in ("UP", "DOWN", "NEUTRAL"), f"Invalid signal: {prediction['signal']}"
        assert 0 <= prediction["confidence"] <= 1, f"Confidence out of range: {prediction['confidence']}"
        assert abs(prediction["prob_up"] + prediction["prob_down"] - 1.0) < 0.01, "Probabilities don't sum to 1"
        results.ok(
            f"Prediction: {prediction['signal']} (conf={prediction['confidence']:.4f}, "
            f"P(up)={prediction['prob_up']:.4f}, P(down)={prediction['prob_down']:.4f})"
        )

        # Test 4: Save and load model
        test_dir = "/tmp/test_model"
        model.save(test_dir)
        assert os.path.exists(os.path.join(test_dir, "xgb_model.pkl")), "Model file not saved"

        model2 = PredictionModel(config)
        loaded = model2.load(test_dir)
        assert loaded, "Model failed to load"
        assert model2.val_accuracy == model.val_accuracy, "Loaded accuracy mismatch"

        # Verify loaded model predicts the same
        pred2 = model2.predict(df_hist.tail(120), higher_tf)
        assert pred2["signal"] == prediction["signal"], "Loaded model gives different signal"
        assert abs(pred2["confidence"] - prediction["confidence"]) < 0.001, "Loaded model confidence differs"
        results.ok("Model save/load: verified identical predictions")

        # Test 5: Retrain check
        assert not model.needs_retrain(), "Model should not need retrain immediately after training"
        results.ok("Retrain logic: correctly reports no retrain needed")

        return model, metrics

    except Exception as e:
        results.fail("Model", f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return None, None


def test_signal_tracker(results: TestResults):
    """Test signal tracking with win/loss/PnL calculations."""
    print("\n--- Testing Signal Tracker ---")
    test_dir = "/tmp/test_signals"
    os.makedirs(test_dir, exist_ok=True)

    try:
        tracker = SignalTracker(test_dir)

        # Test 1: Add signals
        s1 = tracker.add_signal("UP", 0.58, 84000.00)
        assert s1.signal_id == 1, f"Expected ID 1, got {s1.signal_id}"
        assert s1.direction == "UP"
        assert s1.result is None, "New signal should have no result"

        s2 = tracker.add_signal("DOWN", 0.62, 84100.00)
        s3 = tracker.add_signal("UP", 0.55, 83900.00)
        s4 = tracker.add_signal("DOWN", 0.60, 84200.00)
        s5 = tracker.add_signal("UP", 0.57, 83800.00)
        results.ok(f"Added 5 signals (IDs: 1-5)")

        # Test 2: Resolve signals with known prices
        # Signal 1: UP from 84000 -> 84100 = WIN
        r1 = tracker.resolve_signal(1, 84100.00)
        assert r1.result == "WIN", f"Expected WIN, got {r1.result}"
        expected_pnl1 = ((84100 - 84000) / 84000) * 100
        assert abs(r1.pnl_pct - expected_pnl1) < 0.001, f"PnL mismatch: {r1.pnl_pct} vs {expected_pnl1}"
        results.ok(f"Signal #1 UP 84000->84100: WIN {r1.pnl_pct:+.4f}% (correct)")

        # Signal 2: DOWN from 84100 -> 84000 = WIN (price went down)
        r2 = tracker.resolve_signal(2, 84000.00)
        assert r2.result == "WIN", f"Expected WIN, got {r2.result}"
        expected_pnl2 = ((84100 - 84000) / 84100) * 100
        assert abs(r2.pnl_pct - expected_pnl2) < 0.001, f"PnL mismatch: {r2.pnl_pct} vs {expected_pnl2}"
        results.ok(f"Signal #2 DOWN 84100->84000: WIN {r2.pnl_pct:+.4f}% (correct)")

        # Signal 3: UP from 83900 -> 83800 = LOSS
        r3 = tracker.resolve_signal(3, 83800.00)
        assert r3.result == "LOSS", f"Expected LOSS, got {r3.result}"
        expected_pnl3 = ((83800 - 83900) / 83900) * 100
        assert abs(r3.pnl_pct - expected_pnl3) < 0.001, f"PnL mismatch: {r3.pnl_pct} vs {expected_pnl3}"
        results.ok(f"Signal #3 UP 83900->83800: LOSS {r3.pnl_pct:+.4f}% (correct)")

        # Signal 4: DOWN from 84200 -> 84300 = LOSS (price went up, wrong)
        r4 = tracker.resolve_signal(4, 84300.00)
        assert r4.result == "LOSS", f"Expected LOSS, got {r4.result}"
        expected_pnl4 = ((84200 - 84300) / 84200) * 100
        assert abs(r4.pnl_pct - expected_pnl4) < 0.001, f"PnL mismatch: {r4.pnl_pct} vs {expected_pnl4}"
        results.ok(f"Signal #4 DOWN 84200->84300: LOSS {r4.pnl_pct:+.4f}% (correct)")

        # Signal 5: UP from 83800 -> 84000 = WIN
        r5 = tracker.resolve_signal(5, 84000.00)
        assert r5.result == "WIN", f"Expected WIN, got {r5.result}"
        results.ok(f"Signal #5 UP 83800->84000: WIN {r5.pnl_pct:+.4f}% (correct)")

        # Test 3: Stats accuracy
        stats = tracker.get_stats()
        assert stats.total_signals == 5, f"Expected 5 total, got {stats.total_signals}"
        assert stats.wins == 3, f"Expected 3 wins, got {stats.wins}"
        assert stats.losses == 2, f"Expected 2 losses, got {stats.losses}"
        assert stats.pending == 0, f"Expected 0 pending, got {stats.pending}"
        expected_wr = 3 / 5 * 100  # 60%
        assert abs(stats.win_rate - expected_wr) < 0.1, f"Win rate: {stats.win_rate} vs {expected_wr}"
        results.ok(f"Stats: W={stats.wins} L={stats.losses} WR={stats.win_rate:.1f}% (correct)")

        # Test 4: PnL accuracy
        total_pnl = sum(s.pnl_pct for s in [r1, r2, r3, r4, r5])
        assert abs(stats.total_pnl_pct - total_pnl) < 0.001, f"Total PnL mismatch: {stats.total_pnl_pct} vs {total_pnl}"
        results.ok(f"Total PnL: {stats.total_pnl_pct:+.4f}% (matches sum of individual trades)")

        # Test 5: Streak calculation
        # Order: WIN, WIN, LOSS, LOSS, WIN -> current streak = 1 WIN
        assert stats.current_streak == 1, f"Current streak: {stats.current_streak}"
        assert stats.current_streak_type == "WIN", f"Streak type: {stats.current_streak_type}"
        assert stats.longest_win_streak == 2, f"Longest win streak: {stats.longest_win_streak}"
        assert stats.longest_loss_streak == 2, f"Longest loss streak: {stats.longest_loss_streak}"
        results.ok(f"Streaks: current={stats.current_streak} {stats.current_streak_type}, max_win={stats.longest_win_streak}, max_loss={stats.longest_loss_streak}")

        # Test 6: Persistence
        tracker2 = SignalTracker(test_dir)
        assert len(tracker2.signals) == 5, f"Loaded {len(tracker2.signals)} signals, expected 5"
        stats2 = tracker2.get_stats()
        assert stats2.wins == stats.wins, "Persisted stats mismatch"
        assert abs(stats2.total_pnl_pct - stats.total_pnl_pct) < 0.001, "Persisted PnL mismatch"
        results.ok("Persistence: signals correctly saved and reloaded")

        # Test 7: Message formatting
        stats_msg = tracker.format_stats_message()
        assert "Win Rate" in stats_msg, "Stats message missing Win Rate"
        assert "Total PnL" in stats_msg, "Stats message missing Total PnL"
        assert "Streaks" in stats_msg, "Stats message missing Streaks"
        results.ok("Message formatting: stats message contains all sections")

        # Cleanup
        import shutil
        shutil.rmtree(test_dir, ignore_errors=True)

    except Exception as e:
        results.fail("Signal Tracker", f"{type(e).__name__}: {e}")
        traceback.print_exc()


def test_config(results: TestResults):
    """Test configuration loading."""
    print("\n--- Testing Configuration ---")
    try:
        # Test default config
        config = BotConfig()
        assert config.mexc.symbol == "BTCUSDT"
        assert config.mexc.base_url == "https://api.mexc.com"
        assert config.model.lookback_candles == 100
        assert config.model.prediction_threshold == 0.52
        results.ok("Default config loaded correctly")

        # Test env override
        os.environ["TRADING_SYMBOL"] = "ETHUSDT"
        os.environ["PREDICTION_THRESHOLD"] = "0.55"
        config2 = BotConfig.from_env()
        assert config2.mexc.symbol == "ETHUSDT"
        assert config2.model.prediction_threshold == 0.55
        results.ok("Environment variable overrides work")

        # Cleanup
        del os.environ["TRADING_SYMBOL"]
        del os.environ["PREDICTION_THRESHOLD"]

    except Exception as e:
        results.fail("Config", f"{type(e).__name__}: {e}")
        traceback.print_exc()


async def run_all_tests():
    """Run all tests."""
    print("="*50)
    print("BTC 5m SIGNAL BOT - COMPREHENSIVE TEST SUITE")
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print("="*50)

    results = TestResults()

    # Test 1: Config
    test_config(results)

    # Test 2: MEXC API
    df_5m, df_15m, df_1h, df_hist = await test_mexc_api(results)

    if df_5m is not None and df_hist is not None:
        # Test 3: Features
        test_features(results, df_5m, df_15m, df_1h)

        # Test 4: Model
        model, metrics = test_model(results, df_hist, df_15m, df_1h)

        if metrics:
            print(f"\n--- Model Metrics Summary ---")
            print(f"  Train Accuracy: {metrics['train_accuracy']:.4f}")
            print(f"  Val Accuracy:   {metrics['val_accuracy']:.4f}")
            print(f"  CV Accuracy:    {metrics['cv_accuracy']:.4f}")
            print(f"  Val Log Loss:   {metrics['val_logloss']:.4f}")
            print(f"  Samples:        {metrics['total_samples']}")
            print(f"  Features:       {metrics['n_features']}")
    else:
        results.fail("Features (skipped)", "No data from MEXC API")
        results.fail("Model (skipped)", "No data from MEXC API")

    # Test 5: Signal Tracker (independent of API)
    test_signal_tracker(results)

    # Final summary
    all_passed = results.summary()

    if all_passed:
        print("\nAll tests PASSED! Bot is ready for deployment.")
    else:
        print(f"\n{results.failed} test(s) FAILED. Review errors above.")

    return all_passed


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
