"""Signal tracking with win/loss, PnL, streaks, and statistics."""
import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """A single signal record."""
    signal_id: int
    direction: str  # UP or DOWN
    confidence: float
    entry_price: float
    timestamp: str
    # Filled after candle closes
    exit_price: Optional[float] = None
    result: Optional[str] = None  # WIN, LOSS, or NEUTRAL
    pnl_pct: Optional[float] = None
    resolved_at: Optional[str] = None


@dataclass
class TrackerStats:
    """Aggregated statistics."""
    total_signals: int = 0
    wins: int = 0
    losses: int = 0
    neutral: int = 0
    pending: int = 0
    win_rate: float = 0.0
    total_pnl_pct: float = 0.0
    avg_pnl_pct: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    best_trade_pct: float = 0.0
    worst_trade_pct: float = 0.0
    current_streak: int = 0
    current_streak_type: str = ""  # WIN or LOSS
    longest_win_streak: int = 0
    longest_loss_streak: int = 0
    avg_confidence: float = 0.0
    # Session info
    session_start: str = ""
    last_signal_time: str = ""


class SignalTracker:
    """Tracks signals and computes performance statistics."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.signals: list[Signal] = []
        self._next_id = 1
        self._session_start = datetime.now(timezone.utc).isoformat()
        self._load()

    def add_signal(self, direction: str, confidence: float, entry_price: float) -> Signal:
        """Add a new signal.

        Args:
            direction: UP or DOWN
            confidence: Model confidence (0-1)
            entry_price: Price at signal time

        Returns:
            The created Signal object
        """
        signal = Signal(
            signal_id=self._next_id,
            direction=direction,
            confidence=confidence,
            entry_price=entry_price,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self.signals.append(signal)
        self._next_id += 1
        self._save()
        logger.info(f"Signal #{signal.signal_id}: {direction} @ ${entry_price:,.2f} (conf={confidence:.4f})")
        return signal

    def resolve_signal(self, signal_id: int, exit_price: float) -> Optional[Signal]:
        """Resolve a pending signal with the exit price.

        Args:
            signal_id: Signal ID to resolve
            exit_price: Price at candle close

        Returns:
            The resolved Signal or None
        """
        signal = self._find_signal(signal_id)
        if signal is None or signal.result is not None:
            return None

        # Calculate PnL
        if signal.direction == "UP":
            pnl_pct = ((exit_price - signal.entry_price) / signal.entry_price) * 100
        else:  # DOWN
            pnl_pct = ((signal.entry_price - exit_price) / signal.entry_price) * 100

        # Determine result
        if pnl_pct > 0:
            signal.result = "WIN"
        elif pnl_pct < 0:
            signal.result = "LOSS"
        else:
            signal.result = "NEUTRAL"

        signal.exit_price = exit_price
        signal.pnl_pct = round(pnl_pct, 4)
        signal.resolved_at = datetime.now(timezone.utc).isoformat()

        self._save()
        logger.info(f"Signal #{signal_id} resolved: {signal.result} ({pnl_pct:+.4f}%)")
        return signal

    def get_pending_signals(self) -> list[Signal]:
        """Get all unresolved signals."""
        return [s for s in self.signals if s.result is None]

    def get_stats(self) -> TrackerStats:
        """Compute comprehensive statistics."""
        stats = TrackerStats()
        stats.session_start = self._session_start
        stats.total_signals = len(self.signals)

        resolved = [s for s in self.signals if s.result is not None]
        stats.pending = len(self.signals) - len(resolved)

        if not resolved:
            return stats

        stats.wins = sum(1 for s in resolved if s.result == "WIN")
        stats.losses = sum(1 for s in resolved if s.result == "LOSS")
        stats.neutral = sum(1 for s in resolved if s.result == "NEUTRAL")

        decided = stats.wins + stats.losses
        stats.win_rate = (stats.wins / decided * 100) if decided > 0 else 0.0

        pnls = [s.pnl_pct for s in resolved if s.pnl_pct is not None]
        if pnls:
            stats.total_pnl_pct = round(sum(pnls), 4)
            stats.avg_pnl_pct = round(stats.total_pnl_pct / len(pnls), 4)
            stats.best_trade_pct = round(max(pnls), 4)
            stats.worst_trade_pct = round(min(pnls), 4)

            wins_pnl = [p for p in pnls if p > 0]
            losses_pnl = [p for p in pnls if p < 0]
            stats.avg_win_pct = round(sum(wins_pnl) / len(wins_pnl), 4) if wins_pnl else 0.0
            stats.avg_loss_pct = round(sum(losses_pnl) / len(losses_pnl), 4) if losses_pnl else 0.0

        # Streaks
        streak_count = 0
        streak_type = ""
        longest_win = 0
        longest_loss = 0

        for s in resolved:
            if s.result == "NEUTRAL":
                continue
            if s.result == streak_type:
                streak_count += 1
            else:
                streak_type = s.result
                streak_count = 1

            if streak_type == "WIN":
                longest_win = max(longest_win, streak_count)
            elif streak_type == "LOSS":
                longest_loss = max(longest_loss, streak_count)

        stats.current_streak = streak_count
        stats.current_streak_type = streak_type
        stats.longest_win_streak = longest_win
        stats.longest_loss_streak = longest_loss

        # Average confidence
        confidences = [s.confidence for s in self.signals]
        stats.avg_confidence = round(sum(confidences) / len(confidences), 4) if confidences else 0.0

        # Last signal time
        stats.last_signal_time = self.signals[-1].timestamp if self.signals else ""

        return stats

    def get_recent_signals(self, n: int = 10) -> list[Signal]:
        """Get the N most recent signals."""
        return self.signals[-n:]

    def format_stats_message(self) -> str:
        """Format stats as a Telegram-friendly message."""
        s = self.get_stats()

        if s.total_signals == 0:
            return "No signals recorded yet."

        streak_emoji = ""
        if s.current_streak_type == "WIN" and s.current_streak >= 2:
            streak_emoji = " [HOT]"
        elif s.current_streak_type == "LOSS" and s.current_streak >= 3:
            streak_emoji = " [COLD]"

        lines = [
            "========== SIGNAL TRACKER ==========",
            "",
            f"Total Signals: {s.total_signals}",
            f"Resolved: {s.wins + s.losses + s.neutral} | Pending: {s.pending}",
            "",
            "---------- Performance ----------",
            f"Wins: {s.wins} | Losses: {s.losses} | Neutral: {s.neutral}",
            f"Win Rate: {s.win_rate:.1f}%",
            "",
            "---------- PnL ----------",
            f"Total PnL: {s.total_pnl_pct:+.4f}%",
            f"Avg PnL/Trade: {s.avg_pnl_pct:+.4f}%",
            f"Avg Win: {s.avg_win_pct:+.4f}% | Avg Loss: {s.avg_loss_pct:+.4f}%",
            f"Best: {s.best_trade_pct:+.4f}% | Worst: {s.worst_trade_pct:+.4f}%",
            "",
            "---------- Streaks ----------",
            f"Current: {s.current_streak} {s.current_streak_type}{streak_emoji}",
            f"Longest Win Streak: {s.longest_win_streak}",
            f"Longest Loss Streak: {s.longest_loss_streak}",
            "",
            "---------- Meta ----------",
            f"Avg Confidence: {s.avg_confidence:.4f}",
            f"Session Start: {s.session_start[:19]}Z",
            f"Last Signal: {s.last_signal_time[:19]}Z" if s.last_signal_time else "",
            "====================================",
        ]
        return "\n".join(lines)

    def format_signal_message(self, signal: Signal, prediction: dict) -> str:
        """Format a new signal as a Telegram message.

        Includes signal strength from confidence filtering.
        """
        direction_arrow = ">> UP" if signal.direction == "UP" else ">> DOWN"
        strength = prediction.get("strength", "NORMAL")
        strength_label = f" [{strength}]" if strength == "STRONG" else ""

        lines = [
            "========== BTC 5m SIGNAL ==========",
            "",
            f"Direction: {direction_arrow}{strength_label}",
            f"Confidence: {signal.confidence:.1%}",
            "",
            f"Entry Price: ${signal.entry_price:,.2f}",
            f"P(Up): {prediction.get('prob_up', 0):.1%} | P(Down): {prediction.get('prob_down', 0):.1%}",
            "",
            f"Model Accuracy: {prediction.get('model_accuracy', 0):.1%}",
            f"Signal #{signal.signal_id}",
            f"Time: {signal.timestamp[:19]}Z",
            "====================================",
        ]
        return "\n".join(lines)

    def format_resolution_message(self, signal: Signal) -> str:
        """Format a signal resolution as a Telegram message."""
        result_label = signal.result
        if signal.result == "WIN":
            result_label = "[WIN]"
        elif signal.result == "LOSS":
            result_label = "[LOSS]"

        stats = self.get_stats()

        lines = [
            "---------- SIGNAL RESOLVED ----------",
            "",
            f"Signal #{signal.signal_id}: {signal.direction}",
            f"Result: {result_label}",
            "",
            f"Entry: ${signal.entry_price:,.2f}",
            f"Exit:  ${signal.exit_price:,.2f}",
            f"PnL: {signal.pnl_pct:+.4f}%",
            "",
            f"Running W/L: {stats.wins}/{stats.losses} ({stats.win_rate:.1f}%)",
            f"Total PnL: {stats.total_pnl_pct:+.4f}%",
            "--------------------------------------",
        ]
        return "\n".join(lines)

    def _find_signal(self, signal_id: int) -> Optional[Signal]:
        for s in self.signals:
            if s.signal_id == signal_id:
                return s
        return None

    def _save(self):
        """Persist signals to disk."""
        os.makedirs(self.data_dir, exist_ok=True)
        path = os.path.join(self.data_dir, "signals.json")
        data = {
            "next_id": self._next_id,
            "session_start": self._session_start,
            "signals": [asdict(s) for s in self.signals],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self):
        """Load signals from disk."""
        path = os.path.join(self.data_dir, "signals.json")
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                data = json.load(f)
            self._next_id = data.get("next_id", 1)
            self._session_start = data.get("session_start", self._session_start)
            self.signals = [
                Signal(**s) for s in data.get("signals", [])
            ]
            logger.info(f"Loaded {len(self.signals)} signals from disk")
        except Exception as e:
            logger.error(f"Failed to load signals: {e}")
