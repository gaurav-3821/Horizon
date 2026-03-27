from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool


def atom_to_feature_vector(atom: Chem.Atom) -> list[float]:
    """Compact numeric atom feature vector."""
    return [
        float(atom.GetAtomicNum()),
        float(atom.GetDegree()),
        float(atom.GetFormalCharge()),
        float(int(atom.GetHybridization())),
        float(atom.GetIsAromatic()),
        float(atom.GetTotalNumHs()),
    ]


class MolecularGNN(torch.nn.Module):
    def __init__(self, num_node_features: int, hidden_channels: int = 128, num_classes: int = 12):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, num_classes)
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        logits = self.lin(x)
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x, edge_index, batch))


def smiles_to_graph(smiles: str, target_labels: Sequence[float]) -> Data:
    """Convert a SMILES string to a PyTorch Geometric Data graph."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    node_features = [atom_to_feature_vector(atom) for atom in mol.GetAtoms()]

    edges: list[list[int]] = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges.append([i, j])
        edges.append([j, i])

    x = torch.tensor(node_features, dtype=torch.float32)

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    y = torch.tensor(target_labels, dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, y=y)


def build_graph_dataset(smiles_list: Iterable[str], labels: Iterable[Sequence[float]]) -> list[Data]:
    """Build a list of Data objects and skip invalid SMILES."""
    dataset: list[Data] = []
    for smiles, target in zip(smiles_list, labels):
        try:
            dataset.append(smiles_to_graph(smiles, target))
        except ValueError:
            continue
    return dataset


def create_dataloader(dataset: list[Data], batch_size: int = 32, shuffle: bool = True) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
