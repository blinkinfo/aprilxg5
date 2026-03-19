"""XGBoost model for BTC 5-minute candle direction prediction."""
import logging
import os
import pickle
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, log_loss
from xgboost import XGBClassifier

from .config import ModelConfig
from .features import FeatureEngineer

logger = logging.getLogger(__name__)


class PredictionModel:
    """XGBoost-based prediction model with training, evaluation, and inference."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.feature_engineer = FeatureEngineer(config)
        self.model: Optional[XGBClassifier] = None
        self.feature_names: list[str] = []
        self.last_train_time: Optional[datetime] = None
        self.train_accuracy: float = 0.0
        self.val_accuracy: float = 0.0
        self.train_samples: int = 0

    def train(
        self,
        df_5m: pd.DataFrame,
        higher_tf_data: Optional[dict[str, pd.DataFrame]] = None,
    ) -> dict:
        """Train the model on historical data.

        Args:
            df_5m: 5-minute OHLCV DataFrame
            higher_tf_data: Optional higher timeframe data

        Returns:
            Dict with training metrics
        """
        logger.info(f"Training model on {len(df_5m)} candles...")

        # Create labels before computing features (need close column)
        labels = self.feature_engineer.create_labels(df_5m)

        # Compute features
        features_df = self.feature_engineer.compute_features(df_5m, higher_tf_data)

        if features_df.empty:
            raise ValueError("Feature computation returned empty DataFrame")

        # Align labels with features (drop last row since label is NaN)
        min_len = min(len(features_df), len(labels))
        features_df = features_df.iloc[:min_len]
        labels = labels.iloc[:min_len]

        # Align indices
        valid_idx = features_df.dropna().index.intersection(labels.dropna().index)
        X = features_df.loc[valid_idx]
        y = labels.loc[valid_idx]

        if len(X) < 200:
            raise ValueError(f"Insufficient training data: {len(X)} samples (need >= 200)")

        self.feature_names = list(X.columns)
        logger.info(f"Training with {len(X)} samples, {len(self.feature_names)} features")

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            fold_model = XGBClassifier(**self.config.xgb_params)
            fold_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            val_pred = fold_model.predict(X_val)
            fold_acc = accuracy_score(y_val, val_pred)
            cv_scores.append(fold_acc)
            logger.info(f"  Fold {fold + 1}/5: val_accuracy={fold_acc:.4f}")

        avg_cv = np.mean(cv_scores)
        logger.info(f"Average CV accuracy: {avg_cv:.4f}")

        # Train final model on all data
        # Use 80/20 chronological split for final metrics
        split_idx = int(len(X) * 0.8)
        X_train_final = X.iloc[:split_idx]
        y_train_final = y.iloc[:split_idx]
        X_val_final = X.iloc[split_idx:]
        y_val_final = y.iloc[split_idx:]

        self.model = XGBClassifier(**self.config.xgb_params)
        self.model.fit(
            X_train_final, y_train_final,
            eval_set=[(X_val_final, y_val_final)],
            verbose=False,
        )

        # Evaluate
        train_pred = self.model.predict(X_train_final)
        val_pred = self.model.predict(X_val_final)
        val_proba = self.model.predict_proba(X_val_final)[:, 1]

        self.train_accuracy = accuracy_score(y_train_final, train_pred)
        self.val_accuracy = accuracy_score(y_val_final, val_pred)
        self.train_samples = len(X)
        self.last_train_time = datetime.now(timezone.utc)

        val_logloss = log_loss(y_val_final, val_proba)

        # Class distribution in validation
        val_class_dist = y_val_final.value_counts().to_dict()

        metrics = {
            "train_accuracy": self.train_accuracy,
            "val_accuracy": self.val_accuracy,
            "cv_accuracy": avg_cv,
            "cv_scores": cv_scores,
            "val_logloss": val_logloss,
            "train_samples": len(X_train_final),
            "val_samples": len(X_val_final),
            "total_samples": len(X),
            "n_features": len(self.feature_names),
            "val_class_distribution": val_class_dist,
            "train_time": self.last_train_time.isoformat(),
        }

        logger.info(f"Final model - Train acc: {self.train_accuracy:.4f}, Val acc: {self.val_accuracy:.4f}")
        logger.info(f"Val log loss: {val_logloss:.4f}")

        # Feature importance
        importances = self.model.feature_importances_
        top_features = sorted(zip(self.feature_names, importances), key=lambda x: x[1], reverse=True)[:15]
        logger.info("Top 15 features:")
        for fname, imp in top_features:
            logger.info(f"  {fname}: {imp:.4f}")
        metrics["top_features"] = {f: float(i) for f, i in top_features}

        return metrics

    def predict(self, df_5m: pd.DataFrame, higher_tf_data: Optional[dict[str, pd.DataFrame]] = None) -> dict:
        """Make a prediction for the next candle.

        Args:
            df_5m: Recent 5-minute OHLCV data
            higher_tf_data: Optional higher timeframe data

        Returns:
            Dict with prediction, confidence, and signal info
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet")

        features_df = self.feature_engineer.compute_features(df_5m, higher_tf_data)

        if features_df.empty:
            return {"signal": "SKIP", "confidence": 0.0, "reason": "Insufficient data"}

        # Use the last row (most recent candle)
        latest = features_df.iloc[[-1]]

        # Ensure feature alignment
        missing_feats = set(self.feature_names) - set(latest.columns)
        for feat in missing_feats:
            latest[feat] = 0.0
        latest = latest[self.feature_names]

        # Handle any NaN in the latest row
        if latest.isna().any().any():
            latest = latest.fillna(0)

        # Predict
        proba = self.model.predict_proba(latest)[0]
        prob_up = float(proba[1])
        prob_down = float(proba[0])

        # Determine signal
        confidence = max(prob_up, prob_down)
        threshold = self.config.prediction_threshold

        if prob_up >= threshold:
            signal = "UP"
            direction_confidence = prob_up
        elif prob_down >= threshold:
            signal = "DOWN"
            direction_confidence = prob_down
        else:
            signal = "NEUTRAL"
            direction_confidence = confidence

        current_price = float(df_5m["close"].iloc[-1])

        result = {
            "signal": signal,
            "confidence": round(direction_confidence, 4),
            "prob_up": round(prob_up, 4),
            "prob_down": round(prob_down, 4),
            "current_price": current_price,
            "model_accuracy": self.val_accuracy,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(f"Prediction: {signal} (confidence={direction_confidence:.4f}, price={current_price})")
        return result

    def needs_retrain(self) -> bool:
        """Check if model needs retraining based on time interval."""
        if self.model is None or self.last_train_time is None:
            return True
        elapsed = (datetime.now(timezone.utc) - self.last_train_time).total_seconds()
        return elapsed > self.config.retrain_interval_hours * 3600

    def save(self, model_dir: str) -> str:
        """Save model to disk."""
        os.makedirs(model_dir, exist_ok=True)
        path = os.path.join(model_dir, "xgb_model.pkl")
        state = {
            "model": self.model,
            "feature_names": self.feature_names,
            "last_train_time": self.last_train_time,
            "train_accuracy": self.train_accuracy,
            "val_accuracy": self.val_accuracy,
            "train_samples": self.train_samples,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info(f"Model saved to {path}")
        return path

    def load(self, model_dir: str) -> bool:
        """Load model from disk."""
        path = os.path.join(model_dir, "xgb_model.pkl")
        if not os.path.exists(path):
            logger.info("No saved model found")
            return False
        try:
            with open(path, "rb") as f:
                state = pickle.load(f)
            self.model = state["model"]
            self.feature_names = state["feature_names"]
            self.last_train_time = state["last_train_time"]
            self.train_accuracy = state["train_accuracy"]
            self.val_accuracy = state["val_accuracy"]
            self.train_samples = state["train_samples"]
            logger.info(f"Model loaded from {path} (val_acc={self.val_accuracy:.4f})")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
