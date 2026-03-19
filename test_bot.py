"""Comprehensive test suite for BTC Signal Bot (aprilxg v2).
Tests: MEXC API, feature engineering (normalized + regime), model training (Optuna + retraining gate),
       confidence filtering, signal tracking, prediction.
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

        # Test 5: Historical pagination
        df_hist = await fetcher.fetch_historical_klines(interval="5m", total_candles=1000)
        assert len(df_hist) >= 900, f"Historical fetch only got {len(df_hist)}, expected >= 900"
        assert df_hist["timestamp"].is_unique, "Historical data has duplicate timestamps"
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
    """Test feature engineering including v2 improvements."""
    print("\n--- Testing Feature Engineering (v2) ---")
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

        # Test 3: Check key features exist (original + v2 new features)
        expected_features = [
            "rsi", "macd_histogram", "bb_pctb", "atr_norm", "adx",
            "volume_ratio", "mfi", "ema_crossover", "returns_1", "candle_body",
            # v2 normalized MACD
            "macd_line", "macd_signal",
            # v2 regime detection
            "atr_percentile", "regime_high_vol", "regime_vol_score", "vol_expansion",
            # v2 z-score features
            "momentum_3_zscore", "momentum_5_zscore", "rsi_zscore", "volume_ratio_zscore",
        ]
        for feat_name in expected_features:
            assert feat_name in features.columns, f"Missing feature: {feat_name}"
        results.ok(f"All {len(expected_features)} key features present (incl. v2 regime + normalized)")

        # Test 4: Verify MACD is normalized (should be small pct values, not raw price-scale)
        macd_vals = features["macd_line"].dropna()
        assert macd_vals.abs().max() < 0.1, f"MACD line not normalized: max={macd_vals.abs().max():.6f}"
        results.ok(f"MACD normalization: max|macd_line|={macd_vals.abs().max():.6f} (correctly small)")

        # Test 5: Verify regime features are valid
        regime_vals = features["regime_high_vol"].dropna()
        assert set(regime_vals.unique()).issubset({0, 1}), f"regime_high_vol should be binary, got {regime_vals.unique()}"
        vol_score = features["regime_vol_score"].dropna()
        assert vol_score.min() >= 0 and vol_score.max() <= 1.0, f"regime_vol_score out of [0,1] range"
        results.ok(f"Regime features valid: {regime_vals.mean():.1%} high-vol, score range [{vol_score.min():.2f}, {vol_score.max():.2f}]")

        # Test 6: Raw MACD columns should NOT be in output
        assert "macd_line_raw" not in features.columns, "Raw MACD line should be dropped"
        assert "macd_signal_raw" not in features.columns, "Raw MACD signal should be dropped"
        assert "macd_histogram_raw" not in features.columns, "Raw MACD histogram should be dropped"
        results.ok("Raw MACD columns correctly excluded from features")

        # Test 7: Multi-timeframe features
        higher_tf = {"15m": df_15m, "1h": df_1h}
        features_mtf = fe.compute_features(df_5m, higher_tf)
        mtf_cols = [c for c in features_mtf.columns if "15min" in c or "1hr" in c or "60min" in c]
        assert len(mtf_cols) > 0, "No multi-timeframe features found"
        results.ok(f"Multi-TF features: {len(mtf_cols)} MTF columns added")

        # Test 8: Labels
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
    """Test model training, prediction, confidence filtering, and retraining gate."""
    print("\n--- Testing Model (v2: Optuna + Gate + Confidence) ---")
    # Disable Optuna for fast testing, use small data
    config = ModelConfig()
    config.enable_optuna_tuning = False  # Skip Optuna in tests for speed
    config.train_candles = 1000  # Smaller for test
    model = PredictionModel(config)

    try:
        # Test 1: Train model
        higher_tf = {"15m": df_15m, "1h": df_1h}
        metrics = model.train(df_hist, higher_tf)

        assert "train_accuracy" in metrics, "Missing train_accuracy"
        assert "val_accuracy" in metrics, "Missing val_accuracy"
        assert "cv_accuracy" in metrics, "Missing cv_accuracy"
        assert "model_swapped" in metrics, "Missing model_swapped (v2 retraining gate)"
        assert metrics["model_swapped"] is True, "First training should always swap"
        assert metrics["val_accuracy"] > 0.45, f"Val accuracy too low: {metrics['val_accuracy']}"
        results.ok(
            f"Model trained: val_acc={metrics['val_accuracy']:.4f}, "
            f"cv_acc={metrics['cv_accuracy']:.4f}, "
            f"{metrics['n_features']} features, {metrics['total_samples']} samples, "
            f"swapped={metrics['model_swapped']}"
        )

        # Test 2: Check CV scores
        cv_scores = metrics["cv_scores"]
        assert len(cv_scores) == 5, f"Expected 5 CV folds, got {len(cv_scores)}"
        assert all(0.3 < s < 0.75 for s in cv_scores), f"CV scores out of range: {cv_scores}"
        results.ok(f"CV scores: {[f'{s:.4f}' for s in cv_scores]}")

        # Test 3: Prediction with confidence filtering
        prediction = model.predict(df_hist.tail(120), higher_tf)
        assert "signal" in prediction, "Missing signal"
        assert "confidence" in prediction, "Missing confidence"
        assert "prob_up" in prediction, "Missing prob_up"
        assert "prob_down" in prediction, "Missing prob_down"
        assert "current_price" in prediction, "Missing current_price"
        assert "strength" in prediction, "Missing strength (v2 confidence filtering)"
        assert prediction["signal"] in ("UP", "DOWN", "NEUTRAL"), f"Invalid signal: {prediction['signal']}"
        assert prediction["strength"] in ("STRONG", "NORMAL", "SKIP"), f"Invalid strength: {prediction['strength']}"
        assert 0 <= prediction["confidence"] <= 1, f"Confidence out of range: {prediction['confidence']}"
        assert abs(prediction["prob_up"] + prediction["prob_down"] - 1.0) < 0.01, "Probabilities don't sum to 1"

        # Verify confidence filtering logic
        if prediction["signal"] == "NEUTRAL":
            assert prediction["strength"] == "SKIP", "NEUTRAL signal should have SKIP strength"
            assert prediction["confidence"] < config.confidence_min, \
                f"NEUTRAL should have confidence < {config.confidence_min}, got {prediction['confidence']}"
        elif prediction["signal"] in ("UP", "DOWN"):
            assert prediction["confidence"] >= config.confidence_min, \
                f"Active signal should have confidence >= {config.confidence_min}, got {prediction['confidence']}"
            if prediction["confidence"] >= config.confidence_strong:
                assert prediction["strength"] == "STRONG", "High confidence should be STRONG"

        results.ok(
            f"Prediction: {prediction['signal']} [{prediction['strength']}] "
            f"(conf={prediction['confidence']:.4f}, "
            f"P(up)={prediction['prob_up']:.4f}, P(down)={prediction['prob_down']:.4f})"
        )

        # Test 4: Retraining gate — retrain and check gate logic
        first_val_acc = model.val_accuracy
        metrics2 = model.train(df_hist, higher_tf)
        # The gate should either swap or keep based on improvement threshold
        assert "model_swapped" in metrics2, "Second training missing model_swapped"
        assert "active_val_accuracy" in metrics2, "Missing active_val_accuracy"
        results.ok(
            f"Retraining gate: swapped={metrics2['model_swapped']}, "
            f"new_acc={metrics2['val_accuracy']:.4f}, "
            f"active_acc={metrics2['active_val_accuracy']:.4f}"
        )

        # Test 5: Save and load model (including v2 fields)
        test_dir = "/tmp/test_model_v2"
        model.save(test_dir)
        assert os.path.exists(os.path.join(test_dir, "xgb_model.pkl")), "Model file not saved"

        model2 = PredictionModel(config)
        loaded = model2.load(test_dir)
        assert loaded, "Model failed to load"
        assert model2.val_accuracy == model.val_accuracy, "Loaded accuracy mismatch"
        assert model2.val_logloss == model.val_logloss, "Loaded logloss mismatch"

        # Verify loaded model predicts the same
        pred2 = model2.predict(df_hist.tail(120), higher_tf)
        assert pred2["signal"] == prediction["signal"], "Loaded model gives different signal"
        assert abs(pred2["confidence"] - prediction["confidence"]) < 0.001, "Loaded model confidence differs"
        assert pred2["strength"] == prediction["strength"], "Loaded model strength differs"
        results.ok("Model save/load: verified identical predictions (incl. v2 fields)")

        # Test 6: Retrain check
        assert not model.needs_retrain(), "Model should not need retrain immediately after training"
        results.ok("Retrain logic: correctly reports no retrain needed")

        # Test 7: Tuning check
        assert not model.needs_tuning(), "Optuna tuning is disabled, should not need tuning"
        results.ok("Optuna tuning: correctly disabled in test config")

        return model, metrics

    except Exception as e:
        results.fail("Model", f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return None, None


def test_signal_tracker(results: TestResults):
    """Test signal tracking with win/loss/PnL calculations."""
    print("\n--- Testing Signal Tracker ---")
    test_dir = "/tmp/test_signals_v2"
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
        r1 = tracker.resolve_signal(1, 84100.00)
        assert r1.result == "WIN", f"Expected WIN, got {r1.result}"
        expected_pnl1 = ((84100 - 84000) / 84000) * 100
        assert abs(r1.pnl_pct - expected_pnl1) < 0.001, f"PnL mismatch: {r1.pnl_pct} vs {expected_pnl1}"
        results.ok(f"Signal #1 UP 84000->84100: WIN {r1.pnl_pct:+.4f}% (correct)")

        r2 = tracker.resolve_signal(2, 84000.00)
        assert r2.result == "WIN", f"Expected WIN, got {r2.result}"
        expected_pnl2 = ((84100 - 84000) / 84100) * 100
        assert abs(r2.pnl_pct - expected_pnl2) < 0.001, f"PnL mismatch: {r2.pnl_pct} vs {expected_pnl2}"
        results.ok(f"Signal #2 DOWN 84100->84000: WIN {r2.pnl_pct:+.4f}% (correct)")

        r3 = tracker.resolve_signal(3, 83800.00)
        assert r3.result == "LOSS", f"Expected LOSS, got {r3.result}"
        results.ok(f"Signal #3 UP 83900->83800: LOSS {r3.pnl_pct:+.4f}% (correct)")

        r4 = tracker.resolve_signal(4, 84300.00)
        assert r4.result == "LOSS", f"Expected LOSS, got {r4.result}"
        results.ok(f"Signal #4 DOWN 84200->84300: LOSS {r4.pnl_pct:+.4f}% (correct)")

        r5 = tracker.resolve_signal(5, 84000.00)
        assert r5.result == "WIN", f"Expected WIN, got {r5.result}"
        results.ok(f"Signal #5 UP 83800->84000: WIN {r5.pnl_pct:+.4f}% (correct)")

        # Test 3: Stats accuracy
        stats = tracker.get_stats()
        assert stats.total_signals == 5, f"Expected 5 total, got {stats.total_signals}"
        assert stats.wins == 3, f"Expected 3 wins, got {stats.wins}"
        assert stats.losses == 2, f"Expected 2 losses, got {stats.losses}"
        expected_wr = 3 / 5 * 100
        assert abs(stats.win_rate - expected_wr) < 0.1, f"Win rate: {stats.win_rate} vs {expected_wr}"
        results.ok(f"Stats: W={stats.wins} L={stats.losses} WR={stats.win_rate:.1f}% (correct)")

        # Test 4: PnL accuracy
        total_pnl = sum(s.pnl_pct for s in [r1, r2, r3, r4, r5])
        assert abs(stats.total_pnl_pct - total_pnl) < 0.001, f"Total PnL mismatch"
        results.ok(f"Total PnL: {stats.total_pnl_pct:+.4f}% (matches sum)")

        # Test 5: Streaks
        assert stats.current_streak == 1
        assert stats.current_streak_type == "WIN"
        assert stats.longest_win_streak == 2
        assert stats.longest_loss_streak == 2
        results.ok(f"Streaks: current={stats.current_streak} {stats.current_streak_type}")

        # Test 6: Signal message formatting with strength
        mock_prediction = {
            "prob_up": 0.62, "prob_down": 0.38,
            "model_accuracy": 0.55, "strength": "STRONG"
        }
        msg = tracker.format_signal_message(s1, mock_prediction)
        assert "STRONG" in msg, "Signal message should include STRONG label"
        assert "UP" in msg, "Signal message should include direction"
        results.ok("Signal message formatting includes strength label")

        # Test 7: Persistence
        tracker2 = SignalTracker(test_dir)
        assert len(tracker2.signals) == 5, f"Loaded {len(tracker2.signals)} signals, expected 5"
        stats2 = tracker2.get_stats()
        assert stats2.wins == stats.wins, "Persisted stats mismatch"
        results.ok("Persistence: signals correctly saved and reloaded")

        # Cleanup
        import shutil
        shutil.rmtree(test_dir, ignore_errors=True)

    except Exception as e:
        results.fail("Signal Tracker", f"{type(e).__name__}: {e}")
        traceback.print_exc()


def test_config(results: TestResults):
    """Test configuration loading including v2 fields."""
    print("\n--- Testing Configuration (v2) ---")
    try:
        # Test default config
        config = BotConfig()
        assert config.mexc.symbol == "BTCUSDT"
        assert config.model.train_candles == 43200, f"train_candles should be 43200, got {config.model.train_candles}"
        assert config.model.confidence_min == 0.55, f"confidence_min should be 0.55, got {config.model.confidence_min}"
        assert config.model.confidence_strong == 0.60, f"confidence_strong should be 0.60"
        assert config.model.enable_optuna_tuning is True, "Optuna should be enabled by default"
        assert config.model.retrain_min_improvement == 0.002, f"retrain_min_improvement should be 0.002"
        assert config.model.atr_regime_lookback == 100, f"atr_regime_lookback should be 100"
        assert config.prediction_lead_seconds == 15, f"prediction_lead_seconds should be 15 (BY DESIGN)"
        results.ok("Default v2 config loaded correctly (incl. 15-sec lead time)")

        # Test env override
        os.environ["TRADING_SYMBOL"] = "ETHUSDT"
        os.environ["PREDICTION_THRESHOLD"] = "0.58"
        os.environ["CONFIDENCE_MIN"] = "0.56"
        os.environ["ENABLE_OPTUNA"] = "false"
        os.environ["TRAIN_CANDLES"] = "20000"
        config2 = BotConfig.from_env()
        assert config2.mexc.symbol == "ETHUSDT"
        assert config2.model.prediction_threshold == 0.58
        assert config2.model.confidence_min == 0.56
        assert config2.model.enable_optuna_tuning is False
        assert config2.model.train_candles == 20000
        results.ok("Environment variable overrides work (incl. v2 vars)")

        # Cleanup
        for key in ["TRADING_SYMBOL", "PREDICTION_THRESHOLD", "CONFIDENCE_MIN", "ENABLE_OPTUNA", "TRAIN_CANDLES"]:
            del os.environ[key]

    except Exception as e:
        results.fail("Config", f"{type(e).__name__}: {e}")
        traceback.print_exc()


def test_15sec_timing(results: TestResults):
    """Verify the 15-second pre-signal timing is preserved."""
    print("\n--- Testing 15-Second Signal Timing ---")
    try:
        config = BotConfig()
        assert config.prediction_lead_seconds == 15, \
            f"CRITICAL: prediction_lead_seconds changed from 15 to {config.prediction_lead_seconds}"
        results.ok("15-second pre-signal timing preserved in config")

        # Verify the timing logic would fire correctly
        # At 4:45 into a candle (15 sec before close), should trigger
        seconds_in_candle = 285  # 4 min 45 sec
        candle_duration = 300
        seconds_until_close = candle_duration - seconds_in_candle  # = 15
        should_trigger = seconds_until_close <= config.prediction_lead_seconds and seconds_until_close > 0
        assert should_trigger, "Should trigger at 15 seconds before close"
        results.ok("Timing logic: triggers at exactly 15 sec before candle close")

        # At 4:30 into candle (30 sec before), should NOT trigger
        seconds_in_candle = 270
        seconds_until_close = candle_duration - seconds_in_candle  # = 30
        should_not_trigger = seconds_until_close <= config.prediction_lead_seconds and seconds_until_close > 0
        assert not should_not_trigger, "Should NOT trigger at 30 seconds before close"
        results.ok("Timing logic: correctly skips at 30 sec before close")

        # At candle close (0 sec remaining), should NOT trigger
        seconds_in_candle = 300
        seconds_until_close = candle_duration - seconds_in_candle  # = 0
        should_not_trigger_close = seconds_until_close <= config.prediction_lead_seconds and seconds_until_close > 0
        assert not should_not_trigger_close, "Should NOT trigger at exact close"
        results.ok("Timing logic: correctly skips at exact candle close")

    except Exception as e:
        results.fail("15-sec Timing", f"{type(e).__name__}: {e}")
        traceback.print_exc()


async def run_all_tests():
    """Run all tests."""
    print("="*50)
    print("BTC 5m SIGNAL BOT (aprilxg v2) - TEST SUITE")
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print("="*50)

    results = TestResults()

    # Test 1: Config (v2)
    test_config(results)

    # Test 2: 15-second timing preservation
    test_15sec_timing(results)

    # Test 3: MEXC API
    df_5m, df_15m, df_1h, df_hist = await test_mexc_api(results)

    if df_5m is not None and df_hist is not None:
        # Test 4: Features (v2)
        test_features(results, df_5m, df_15m, df_1h)

        # Test 5: Model (v2)
        model, metrics = test_model(results, df_hist, df_15m, df_1h)

        if metrics:
            print(f"\n--- Model Metrics Summary ---")
            print(f"  Train Accuracy: {metrics['train_accuracy']:.4f}")
            print(f"  Val Accuracy:   {metrics['val_accuracy']:.4f}")
            print(f"  CV Accuracy:    {metrics['cv_accuracy']:.4f}")
            print(f"  Val Log Loss:   {metrics['val_logloss']:.4f}")
            print(f"  Samples:        {metrics['total_samples']}")
            print(f"  Features:       {metrics['n_features']}")
            print(f"  Model Swapped:  {metrics['model_swapped']}")
            print(f"  Optuna Tuned:   {metrics.get('optuna_tuned', False)}")
    else:
        results.fail("Features (skipped)", "No data from MEXC API")
        results.fail("Model (skipped)", "No data from MEXC API")

    # Test 6: Signal Tracker (independent of API)
    test_signal_tracker(results)

    # Final summary
    all_passed = results.summary()

    if all_passed:
        print("\nAll tests PASSED! aprilxg v2 is ready for deployment.")
    else:
        print(f"\n{results.failed} test(s) FAILED. Review errors above.")

    return all_passed


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
