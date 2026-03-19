"""Feature engineering for BTC price prediction.

Improvements over original:
- Improvement 3: All raw price-scale features normalized to percentages/z-scores
- Improvement 6: Volatility regime detection (ATR percentile)
"""
import logging

import numpy as np
import pandas as pd

from .config import ModelConfig

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Computes technical indicators and features from OHLCV data."""

    def __init__(self, config: ModelConfig):
        self.config = config

    def compute_features(self, df: pd.DataFrame, higher_tf_data: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
        """Compute all features from OHLCV data.

        Args:
            df: DataFrame with open, high, low, close, volume columns
            higher_tf_data: Optional dict of higher timeframe DataFrames for multi-TF features

        Returns:
            DataFrame with all computed features (NaN rows dropped)
        """
        if df.empty or len(df) < 50:
            logger.warning("Insufficient data for feature computation")
            return pd.DataFrame()

        feat = df.copy()

        # --- Price Action Features (all normalized as pct of price) ---
        feat["returns_1"] = feat["close"].pct_change(1)
        feat["returns_3"] = feat["close"].pct_change(3)
        feat["returns_5"] = feat["close"].pct_change(5)
        feat["returns_10"] = feat["close"].pct_change(10)

        feat["candle_body"] = (feat["close"] - feat["open"]) / feat["open"]
        feat["upper_wick"] = (feat["high"] - feat[["open", "close"]].max(axis=1)) / feat["open"]
        feat["lower_wick"] = (feat[["open", "close"]].min(axis=1) - feat["low"]) / feat["open"]
        feat["candle_range"] = (feat["high"] - feat["low"]) / feat["open"]

        # High/Low relative to close
        feat["high_close_ratio"] = (feat["high"] - feat["close"]) / feat["close"]
        feat["low_close_ratio"] = (feat["close"] - feat["low"]) / feat["close"]

        # --- Moving Averages ---
        feat["ema_fast"] = feat["close"].ewm(span=self.config.ema_fast, adjust=False).mean()
        feat["ema_slow"] = feat["close"].ewm(span=self.config.ema_slow, adjust=False).mean()
        feat["ema_crossover"] = (feat["ema_fast"] - feat["ema_slow"]) / feat["close"]
        feat["ema_fast_slope"] = feat["ema_fast"].pct_change(3)
        feat["ema_slow_slope"] = feat["ema_slow"].pct_change(3)

        feat["sma_50"] = feat["close"].rolling(50).mean()
        feat["price_sma50_ratio"] = (feat["close"] - feat["sma_50"]) / feat["sma_50"]

        # --- RSI ---
        feat["rsi"] = self._compute_rsi(feat["close"], self.config.rsi_period)

        # --- Stochastic RSI ---
        rsi = feat["rsi"]
        rsi_min = rsi.rolling(self.config.stoch_period).min()
        rsi_max = rsi.rolling(self.config.stoch_period).max()
        rsi_range = rsi_max - rsi_min
        feat["stoch_rsi"] = np.where(rsi_range != 0, (rsi - rsi_min) / rsi_range, 0.5)

        # --- MACD (Improvement 3: normalized to percentage of price) ---
        ema_fast_macd = feat["close"].ewm(span=self.config.macd_fast, adjust=False).mean()
        ema_slow_macd = feat["close"].ewm(span=self.config.macd_slow, adjust=False).mean()
        feat["macd_line_raw"] = ema_fast_macd - ema_slow_macd
        feat["macd_signal_raw"] = feat["macd_line_raw"].ewm(span=self.config.macd_signal, adjust=False).mean()
        feat["macd_histogram_raw"] = feat["macd_line_raw"] - feat["macd_signal_raw"]

        # Normalized MACD features (percentage of close price)
        feat["macd_line"] = feat["macd_line_raw"] / feat["close"]
        feat["macd_signal"] = feat["macd_signal_raw"] / feat["close"]
        feat["macd_histogram"] = feat["macd_histogram_raw"] / feat["close"]
        feat["macd_hist_norm"] = feat["macd_histogram"]  # alias for lag features

        # --- Bollinger Bands ---
        bb_sma = feat["close"].rolling(self.config.bb_period).mean()
        bb_std = feat["close"].rolling(self.config.bb_period).std()
        feat["bb_upper"] = bb_sma + self.config.bb_std * bb_std
        feat["bb_lower"] = bb_sma - self.config.bb_std * bb_std
        bb_range = feat["bb_upper"] - feat["bb_lower"]
        feat["bb_pctb"] = np.where(bb_range != 0, (feat["close"] - feat["bb_lower"]) / bb_range, 0.5)
        feat["bb_width"] = bb_range / bb_sma

        # --- ATR ---
        high_low = feat["high"] - feat["low"]
        high_close = (feat["high"] - feat["close"].shift(1)).abs()
        low_close = (feat["low"] - feat["close"].shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        feat["atr"] = true_range.rolling(self.config.atr_period).mean()
        feat["atr_norm"] = feat["atr"] / feat["close"]

        # --- ADX ---
        feat["adx"] = self._compute_adx(feat, self.config.adx_period)

        # --- Volume Features ---
        feat["volume_sma"] = feat["volume"].rolling(20).mean()
        feat["volume_ratio"] = np.where(
            feat["volume_sma"] != 0,
            feat["volume"] / feat["volume_sma"],
            1.0
        )

        # OBV (On Balance Volume)
        obv = [0.0]
        closes = feat["close"].values
        volumes = feat["volume"].values
        for i in range(1, len(closes)):
            if closes[i] > closes[i - 1]:
                obv.append(obv[-1] + volumes[i])
            elif closes[i] < closes[i - 1]:
                obv.append(obv[-1] - volumes[i])
            else:
                obv.append(obv[-1])
        feat["obv"] = obv
        feat["obv_sma"] = pd.Series(obv).rolling(20).mean().values
        feat["obv_ratio"] = np.where(
            feat["obv_sma"] != 0,
            feat["obv"] / feat["obv_sma"],
            1.0
        )

        # MFI (Money Flow Index)
        feat["mfi"] = self._compute_mfi(feat, self.config.mfi_period)

        # --- Volatility Features ---
        feat["volatility_5"] = feat["returns_1"].rolling(5).std()
        feat["volatility_20"] = feat["returns_1"].rolling(20).std()
        feat["vol_ratio"] = np.where(
            feat["volatility_20"] != 0,
            feat["volatility_5"] / feat["volatility_20"],
            1.0
        )

        # --- Improvement 6: Volatility Regime Detection ---
        atr_lookback = self.config.atr_regime_lookback
        feat["atr_percentile"] = feat["atr_norm"].rolling(atr_lookback, min_periods=1).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        # Binary regime: 1 = high-vol (top 30%), 0 = low-vol
        feat["regime_high_vol"] = (feat["atr_percentile"] > 0.70).astype(int)
        # Continuous regime score (0-1, more granular than binary)
        feat["regime_vol_score"] = feat["atr_percentile"].clip(0, 1)
        # Volatility expansion/contraction: is current vol expanding vs recent?
        feat["vol_expansion"] = feat["atr_norm"].pct_change(5)

        # --- Improvement 3: Normalized momentum features ---
        # Replace raw momentum_3 with pct-based (returns_3 already exists but
        # add explicit momentum z-scores for the model)
        feat["momentum_3_zscore"] = self._rolling_zscore(feat["returns_3"], 50)
        feat["momentum_5_zscore"] = self._rolling_zscore(feat["returns_5"], 50)
        feat["rsi_zscore"] = self._rolling_zscore(feat["rsi"], 50)
        feat["volume_ratio_zscore"] = self._rolling_zscore(feat["volume_ratio"], 50)

        # --- Pattern Features ---
        feat["higher_high"] = (feat["high"] > feat["high"].shift(1)).astype(int)
        feat["lower_low"] = (feat["low"] < feat["low"].shift(1)).astype(int)
        feat["green_candle"] = (feat["close"] > feat["open"]).astype(int)
        feat["consecutive_green"] = self._consecutive_count(feat["green_candle"])
        feat["consecutive_red"] = self._consecutive_count(1 - feat["green_candle"])

        # --- Multi-Timeframe Features ---
        if higher_tf_data:
            feat = self._add_multi_tf_features(feat, higher_tf_data)

        # --- Lag Features ---
        for lag in [1, 2, 3, 5]:
            feat[f"rsi_lag{lag}"] = feat["rsi"].shift(lag)
            feat[f"macd_hist_lag{lag}"] = feat["macd_hist_norm"].shift(lag)
            feat[f"volume_ratio_lag{lag}"] = feat["volume_ratio"].shift(lag)

        # Drop non-feature columns and NaN rows
        drop_cols = ["timestamp", "open", "high", "low", "close", "volume",
                     "close_time", "quote_volume", "ema_fast", "ema_slow",
                     "sma_50", "bb_upper", "bb_lower", "obv", "obv_sma",
                     "volume_sma", "atr",
                     # Drop raw MACD columns (kept normalized versions)
                     "macd_line_raw", "macd_signal_raw", "macd_histogram_raw"]
        feature_cols = [c for c in feat.columns if c not in drop_cols]
        result = feat[feature_cols].copy()
        return result

    def get_feature_names(self, df: pd.DataFrame) -> list[str]:
        """Get list of feature column names (excludes target)."""
        return [c for c in df.columns if c != "target"]

    @staticmethod
    def create_labels(df: pd.DataFrame) -> pd.Series:
        """Create binary labels: 1 if next candle closes higher, 0 otherwise.

        Args:
            df: DataFrame with 'close' column

        Returns:
            Series of 0/1 labels
        """
        return (df["close"].shift(-1) > df["close"]).astype(int)

    @staticmethod
    def _compute_rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    @staticmethod
    def _compute_adx(df: pd.DataFrame, period: int) -> pd.Series:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        idx = df.index

        plus_dm = high.diff().values
        minus_dm = (-low.diff()).values
        plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
        minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)

        high_low = high - low
        high_close = (high - close.shift(1)).abs()
        low_close = (low - close.shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        atr = tr.rolling(period).mean()
        plus_di = 100 * pd.Series(plus_dm, index=idx).rolling(period).mean() / atr.replace(0, np.nan)
        minus_di = 100 * pd.Series(minus_dm, index=idx).rolling(period).mean() / atr.replace(0, np.nan)

        di_sum = plus_di + minus_di
        dx = 100 * (plus_di - minus_di).abs() / di_sum.replace(0, np.nan)
        adx = dx.rolling(period).mean()
        return adx.fillna(25)

    @staticmethod
    def _compute_mfi(df: pd.DataFrame, period: int) -> pd.Series:
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        money_flow = typical_price * df["volume"]
        positive_flow = np.where(typical_price > typical_price.shift(1), money_flow, 0)
        negative_flow = np.where(typical_price < typical_price.shift(1), money_flow, 0)

        pos_sum = pd.Series(positive_flow, index=df.index).rolling(period).sum()
        neg_sum = pd.Series(negative_flow, index=df.index).rolling(period).sum()

        mfi_ratio = pos_sum / neg_sum.replace(0, np.nan)
        mfi = 100 - (100 / (1 + mfi_ratio))
        return mfi.fillna(50)

    @staticmethod
    def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
        """Compute rolling z-score for normalization."""
        rolling_mean = series.rolling(window).mean()
        rolling_std = series.rolling(window).std()
        zscore = np.where(rolling_std != 0, (series - rolling_mean) / rolling_std, 0.0)
        return pd.Series(zscore, index=series.index)

    @staticmethod
    def _consecutive_count(binary_series: pd.Series) -> pd.Series:
        groups = binary_series.ne(binary_series.shift()).cumsum()
        return binary_series.groupby(groups).cumsum()

    def _add_multi_tf_features(self, df: pd.DataFrame, higher_tf_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Add features from higher timeframes by mapping to nearest timestamp."""
        for tf_name, tf_df in higher_tf_data.items():
            if tf_df.empty or len(tf_df) < 30:
                continue

            suffix = f"_{tf_name.replace('m', 'min').replace('h', 'hr')}"

            # Compute trend direction on higher TF (already normalized as pct)
            tf_ema_fast = tf_df["close"].ewm(span=9, adjust=False).mean()
            tf_ema_slow = tf_df["close"].ewm(span=21, adjust=False).mean()
            tf_trend = ((tf_ema_fast - tf_ema_slow) / tf_df["close"]).values

            # Compute RSI on higher TF (0-100 scale, already normalized)
            tf_rsi = self._compute_rsi(tf_df["close"], 14).values

            # Compute momentum as pct change (already normalized)
            tf_momentum = tf_df["close"].pct_change(5).values

            # Map to 5m candles using timestamp alignment
            if "timestamp" in df.columns and "timestamp" in tf_df.columns:
                tf_timestamps = tf_df["timestamp"].values
                main_timestamps = df["timestamp"].values

                trend_mapped = []
                rsi_mapped = []
                momentum_mapped = []

                for ts in main_timestamps:
                    idx = np.searchsorted(tf_timestamps, ts, side="right") - 1
                    idx = max(0, min(idx, len(tf_trend) - 1))
                    trend_mapped.append(tf_trend[idx])
                    rsi_mapped.append(tf_rsi[idx])
                    momentum_mapped.append(tf_momentum[idx])

                df[f"trend{suffix}"] = trend_mapped
                df[f"rsi{suffix}"] = rsi_mapped
                df[f"momentum{suffix}"] = momentum_mapped

        return df
