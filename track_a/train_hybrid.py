from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
import wandb
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import Batch

from imbalance_handler import ImbalanceHandler


@dataclass
class HybridBatch:
    graph_data: Batch
    tabular_features: torch.Tensor
    y: torch.Tensor


def hybrid_collate_fn(samples: Iterable[tuple]) -> HybridBatch:
    """
    Collate helper for hybrid graph+tabular datasets.
    Each sample should be: (graph_data, tabular_features, y)
    """
    graphs, tabs, targets = zip(*samples)
    graph_batch = Batch.from_data_list(list(graphs))
    tab_batch = torch.stack([torch.as_tensor(t, dtype=torch.float32) for t in tabs], dim=0)
    y_batch = torch.stack([torch.as_tensor(y, dtype=torch.float32) for y in targets], dim=0)
    return HybridBatch(graph_data=graph_batch, tabular_features=tab_batch, y=y_batch)


def _extract_batch(batch: Any, device: torch.device) -> tuple[Batch, torch.Tensor, torch.Tensor]:
    """Support dataclass/object/dict/tuple batch formats."""
    if hasattr(batch, "graph_data") and hasattr(batch, "tabular_features") and hasattr(batch, "y"):
        graph_data = batch.graph_data
        tabular_features = batch.tabular_features
        y = batch.y
    elif isinstance(batch, dict):
        graph_data = batch["graph_data"]
        tabular_features = batch["tabular_features"]
        y = batch["y"]
    elif isinstance(batch, (list, tuple)) and len(batch) == 3:
        graph_data, tabular_features, y = batch
    else:
        raise TypeError(
            "Unsupported batch format. Expected object/dict/tuple with graph_data, tabular_features, and y."
        )

    graph_data = graph_data.to(device)
    tabular_features = torch.as_tensor(tabular_features, dtype=torch.float32, device=device)
    y = torch.as_tensor(y, dtype=torch.float32, device=device)
    return graph_data, tabular_features, y


def _macro_multilabel_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    scores: list[float] = []
    for idx in range(y_true.shape[1]):
        col = y_true[:, idx]
        if np.unique(col).size < 2:
            continue
        scores.append(roc_auc_score(col, y_prob[:, idx]))
    return float(np.mean(scores)) if scores else float("nan")


def _macro_multilabel_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    scores: list[float] = []
    for idx in range(y_true.shape[1]):
        col = y_true[:, idx]
        if col.sum() == 0:
            continue
        scores.append(average_precision_score(col, y_prob[:, idx]))
    return float(np.mean(scores)) if scores else float("nan")


def train_model(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    epochs: int = 100,
    project: str = "codecure-toxicity",
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    device: str | None = None,
) -> torch.nn.Module:
    """
    Train the hybrid model with focal loss and multi-label validation metrics.
    Expects batches containing graph_data, tabular_features, and y.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)
    model = model.to(torch_device)

    wandb.init(
        project=project,
        config={
            "architecture": "GNN+Tabular Hybrid",
            "dataset": "Tox21",
            "epochs": epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
        },
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    imbalance = ImbalanceHandler(strategy="focal_loss")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_train_batches = 0

        for batch in train_loader:
            graph_data, tabular_features, y = _extract_batch(batch, device=torch_device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(graph_data, tabular_features)
            loss = imbalance.focal_loss(logits, y, from_logits=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            n_train_batches += 1

        model.eval()
        val_preds: list[np.ndarray] = []
        val_labels: list[np.ndarray] = []
        with torch.no_grad():
            for batch in val_loader:
                graph_data, tabular_features, y = _extract_batch(batch, device=torch_device)
                logits = model(graph_data, tabular_features)
                probs = torch.sigmoid(logits)
                val_preds.append(probs.cpu().numpy())
                val_labels.append(y.cpu().numpy())

        if val_preds:
            y_prob = np.vstack(val_preds)
            y_true = np.vstack(val_labels)
            roc_auc = _macro_multilabel_roc_auc(y_true, y_prob)
            pr_auc = _macro_multilabel_pr_auc(y_true, y_prob)
        else:
            roc_auc = float("nan")
            pr_auc = float("nan")

        avg_loss = total_loss / max(n_train_batches, 1)
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": avg_loss,
                "val_roc_auc": roc_auc,
                "val_pr_auc": pr_auc,
                "learning_rate": scheduler.get_last_lr()[0],
            }
        )
        scheduler.step()

        print(f"Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}, roc_auc={roc_auc:.4f}, pr_auc={pr_auc:.4f}")

    wandb.finish()
    return model
