from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight


class ImbalanceHandler:
    def __init__(self, strategy: str = "focal_loss"):
        self.strategy = strategy

    def focal_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2.0,
        from_logits: bool = True,
    ) -> torch.Tensor:
        """Focal loss for imbalanced binary or multi-label classification."""
        if from_logits:
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
            probs = torch.sigmoid(inputs)
        else:
            bce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
            probs = inputs

        pt = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        focal = alpha_t * (1 - pt).pow(gamma) * bce_loss
        return focal.mean()

    def apply_smote(self, X, y, use_tomek: bool = True):
        """
        Resample tabular features.
        Note: SMOTE/SMOTETomek expects a 1D target; not suitable for true multi-label y directly.
        """
        y_arr = np.asarray(y)
        if y_arr.ndim != 1:
            raise ValueError("SMOTE expects a 1D target. For multi-label tasks, resample per label or use alternatives.")

        sampler = SMOTETomek(random_state=42) if use_tomek else SMOTE(random_state=42)
        X_resampled, y_resampled = sampler.fit_resample(X, y_arr)
        return X_resampled, y_resampled

    def compute_class_weights(self, y) -> torch.Tensor:
        """
        Compute class weights:
        - 1D y: balanced class weights from sklearn (for CE-style losses).
        - 2D y: per-label pos_weight = negatives/positives (for BCEWithLogitsLoss).
        """
        y_arr = np.asarray(y)

        if y_arr.ndim == 1:
            classes = np.unique(y_arr)
            weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_arr)
            return torch.tensor(weights, dtype=torch.float32)

        if y_arr.ndim == 2:
            positives = y_arr.sum(axis=0).astype(np.float64)
            negatives = y_arr.shape[0] - positives
            positives = np.where(positives == 0, 1.0, positives)
            pos_weight = negatives / positives
            return torch.tensor(pos_weight, dtype=torch.float32)

        raise ValueError("y must be 1D or 2D.")
