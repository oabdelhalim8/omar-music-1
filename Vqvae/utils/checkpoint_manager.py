"""Metric-aware checkpoint retention with three interdependent strategies.

Strategies
----------
Keep-Top-K Mode
    Retains only the N best checkpoints by validation metric; older/worse
    files are automatically deleted.

Rolling-Window Mode
    Keeps the last K checkpoints as a FIFO safety net regardless of metric
    value, preventing total loss of recent states.

Weights-Only Mode
    Saves ``model.state_dict()`` only (~50 % smaller); no optimizer state.

CV-Checkpoint Coupling (Adaptive Threshold Mode)
--------------------------------------------------
Divergence risk is assessed from two signals:

* **Train–val gap** – absolute difference between training and validation
  loss, tracked over a sliding window.
* **Fold variance** – variance of per-fold validation losses from a
  lightweight k-fold cross-validation pass triggered when the gap widens.

Risk levels drive both *save frequency* and *retention count*:

=========  ==================  ==========  ===========
Risk       Condition           Save every  Keep top-K
=========  ==================  ==========  ===========
LOW        gap stable, folds   5 epochs    top-5
           agree
MEDIUM     gap widening or     2 epochs    top-3
           some fold disagree
HIGH       gap diverging or    every epoch top-1
           high fold variance
=========  ==================  ==========  ===========

Checkpoint filenames encode ``(epoch, fold_agreement_score)`` for
reproducibility and risk-aware selection.
"""

import heapq
import os
from collections import deque
from enum import Enum
from typing import Optional

import numpy as np
import torch


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# Maps risk level -> (save_every_n_epochs, save_top_k)
RISK_POLICY = {
    RiskLevel.LOW:    {"save_every": 5, "save_top_k": 5},
    RiskLevel.MEDIUM: {"save_every": 2, "save_top_k": 3},
    RiskLevel.HIGH:   {"save_every": 1, "save_top_k": 1},
}


class DivergenceRiskTracker:
    """Monitors training divergence risk from train-val gap and CV fold variance.

    Parameters
    ----------
    gap_threshold:
        Absolute gap value (e.g. 0.05) above which risk starts to rise.
    high_variance_threshold:
        Fold-loss variance above which risk is set to HIGH.
    history_window:
        Number of past epochs to include in the rolling gap average.
    """

    def __init__(
        self,
        gap_threshold: float = 0.05,
        high_variance_threshold: float = 0.02,
        history_window: int = 5,
    ) -> None:
        self.gap_threshold = gap_threshold
        self.high_variance_threshold = high_variance_threshold
        self._gap_history: deque = deque(maxlen=history_window)
        self._fold_variance: float = 0.0
        self._fold_agreement: float = 1.0  # 1.0 = perfect agreement

    # ------------------------------------------------------------------
    # Public update API
    # ------------------------------------------------------------------

    def update_gap(self, train_loss: float, val_loss: float) -> None:
        """Record the train-val gap for the current epoch."""
        self._gap_history.append(abs(val_loss - train_loss))

    def update_fold_stats(self, fold_losses: list) -> None:
        """Update fold agreement score from k-fold CV results.

        Parameters
        ----------
        fold_losses:
            List of per-fold validation losses from the most recent CV pass.
        """
        if len(fold_losses) < 2:
            return
        self._fold_variance = float(np.var(fold_losses))
        mean_loss = float(np.mean(fold_losses))
        # Normalised fold agreement: 1.0 when all folds agree, 0.0 when diverging
        self._fold_agreement = 1.0 / (1.0 + self._fold_variance / (mean_loss + 1e-8))

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def fold_agreement(self) -> float:
        return self._fold_agreement

    @property
    def fold_variance(self) -> float:
        return self._fold_variance

    @property
    def current_gap(self) -> float:
        return float(np.mean(self._gap_history)) if self._gap_history else 0.0

    @property
    def gap_trend(self) -> float:
        """Positive value means the gap is widening."""
        if len(self._gap_history) < 2:
            return 0.0
        gaps = list(self._gap_history)
        return gaps[-1] - gaps[0]

    # ------------------------------------------------------------------
    # Risk assessment
    # ------------------------------------------------------------------

    def compute_risk(self) -> RiskLevel:
        """Return the current :class:`RiskLevel` based on gap and fold variance."""
        gap = self.current_gap
        trend = self.gap_trend
        var = self._fold_variance

        if var > self.high_variance_threshold or (gap > self.gap_threshold and trend > 0):
            return RiskLevel.HIGH
        if gap > self.gap_threshold * 0.5 or var > self.high_variance_threshold * 0.5:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW


class CheckpointManager:
    """Manages checkpoint files with Keep-Top-K, Rolling-Window, and Weights-Only modes.

    Checkpoints are indexed by ``(epoch, fold_agreement_score)`` in their
    filenames for reproducibility and risk-aware downstream selection.

    Parameters
    ----------
    save_dir:
        Directory where checkpoint files are written.
    model_name:
        Prefix used in checkpoint filenames.
    save_top_k:
        Maximum number of best-metric checkpoints to retain (Keep-Top-K).
    keep_last_k:
        Maximum number of most-recent checkpoints to retain (Rolling-Window).
        Files in the rolling window are never deleted even if they fall
        outside the top-k set.
    weights_only:
        When *True*, save only ``model.state_dict()`` (Weights-Only mode).
        When *False*, also persist ``optimizer.state_dict()`` and metadata.
    mode:
        ``"min"`` to treat lower metric as better (e.g. loss);
        ``"max"`` to treat higher metric as better (e.g. accuracy).
    """

    def __init__(
        self,
        save_dir: str,
        model_name: str,
        save_top_k: int = 5,
        keep_last_k: int = 3,
        weights_only: bool = True,
        mode: str = "min",
    ) -> None:
        self.save_dir = save_dir
        self.model_name = model_name
        self.save_top_k = save_top_k
        self.keep_last_k = keep_last_k
        self.weights_only = weights_only
        self.mode = mode

        # Min-heap of (sort_key, path); for "max" mode we negate the value.
        self._top_k_heap: list = []
        self._rolling_window: deque = deque(maxlen=keep_last_k)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        model,
        optimizer,
        epoch: int,
        metric_value: float,
        fold_agreement: float = 1.0,
        risk_level: RiskLevel = RiskLevel.LOW,
    ) -> str:
        """Save a checkpoint and apply retention policies.

        Parameters
        ----------
        model:
            PyTorch model whose state will be saved.
        optimizer:
            Optimizer whose state is saved when *weights_only* is *False*.
        epoch:
            Current training epoch (0-based).
        metric_value:
            Validation metric used for top-k ranking.
        fold_agreement:
            Fold agreement score (0–1) from the most recent CV pass.
        risk_level:
            Current divergence risk level; encoded in the filename.

        Returns
        -------
        str
            Path of the checkpoint file that was written.
        """
        filename = "{model}_epoch{epoch:05d}_fa{fa:.4f}_{risk}.pt".format(
            model=self.model_name,
            epoch=epoch,
            fa=fold_agreement,
            risk=risk_level.value,
        )
        path = os.path.join(self.save_dir, filename)

        if self.weights_only:
            torch.save(model.state_dict(), path)
        else:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metric_value": metric_value,
                    "fold_agreement": fold_agreement,
                    "risk_level": risk_level.value,
                },
                path,
            )

        self._update_top_k(metric_value, path)
        self._update_rolling_window(path)
        return path

    def set_top_k(self, k: int) -> None:
        """Tighten (or relax) top-k retention and evict excess checkpoints."""
        self.save_top_k = k
        while len(self._top_k_heap) > self.save_top_k:
            _, evicted = heapq.heappop(self._top_k_heap)
            if evicted not in self._rolling_window:
                self._safe_remove(evicted)

    def best_checkpoint(self) -> Optional[str]:
        """Return the path of the checkpoint with the best metric value.

        Heap keys are negated for ``mode="min"`` so that the Python min-heap
        can evict the worst (largest) metric value on overflow.  As a result,
        the *best* checkpoint always has the **largest** key in the heap,
        regardless of mode:

        * ``mode="min"``, metrics [0.3, 0.4] → keys [-0.3, -0.4].
          ``max(keys) = -0.3`` → metric 0.3 (best, i.e. lowest).
        * ``mode="max"``, metrics [0.7, 0.8] → keys [0.7, 0.8].
          ``max(keys) = 0.8`` → metric 0.8 (best, i.e. highest).
        """
        if not self._top_k_heap:
            return None
        return max(self._top_k_heap, key=lambda x: x[0])[1]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _metric_key(self, value: float) -> float:
        # For "min" mode (lower is better) we negate so that heappop removes
        # the most-negative key, which corresponds to the LARGEST (worst) value.
        # For "max" mode (higher is better) we keep the value; heappop then
        # removes the smallest (worst) value directly.
        return -value if self.mode == "min" else value

    def _update_top_k(self, metric_value: float, path: str) -> None:
        key = self._metric_key(metric_value)
        heapq.heappush(self._top_k_heap, (key, path))
        while len(self._top_k_heap) > self.save_top_k:
            _, evicted = heapq.heappop(self._top_k_heap)
            if evicted not in self._rolling_window:
                self._safe_remove(evicted)

    def _update_rolling_window(self, path: str) -> None:
        if len(self._rolling_window) == self._rolling_window.maxlen:
            oldest = self._rolling_window[0]
            # Protect file if it still lives in the top-k heap
            top_k_paths = {p for _, p in self._top_k_heap}
            if oldest not in top_k_paths:
                self._safe_remove(oldest)
        self._rolling_window.append(path)

    @staticmethod
    def _safe_remove(path: str) -> None:
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass
