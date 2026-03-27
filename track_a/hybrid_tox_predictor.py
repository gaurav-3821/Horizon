from __future__ import annotations

import torch
from torch_geometric.data import Batch


class HybridToxPredictor(torch.nn.Module):
    """
    Late-fusion model combining GNN graph logits with tabular descriptor embeddings.
    """

    def __init__(
        self,
        gnn_model: torch.nn.Module,
        tabular_dim: int = 1613,
        gnn_output_dim: int = 12,
        num_classes: int = 12,
    ):
        super().__init__()
        self.gnn = gnn_model

        self.tabular_encoder = torch.nn.Sequential(
            torch.nn.Linear(tabular_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
        )

        self.fusion_layer = torch.nn.Linear(gnn_output_dim + 128, num_classes)

    def forward(self, graph_data: Batch, tabular_features: torch.Tensor) -> torch.Tensor:
        # GNN pathway (returns logits from molecular_gnn.MolecularGNN)
        gnn_out = self.gnn(graph_data.x, graph_data.edge_index, graph_data.batch)

        # Tabular pathway
        tab_out = self.tabular_encoder(tabular_features)

        if gnn_out.size(0) != tab_out.size(0):
            raise ValueError(
                f"Batch size mismatch: gnn_out has {gnn_out.size(0)} rows but tabular_features has {tab_out.size(0)}."
            )

        # Late fusion
        combined = torch.cat([gnn_out, tab_out], dim=1)
        logits = self.fusion_layer(combined)
        return logits

    @torch.no_grad()
    def predict_proba(self, graph_data: Batch, tabular_features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(graph_data, tabular_features))
