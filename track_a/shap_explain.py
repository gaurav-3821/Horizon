from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import shap
import torch
from torch_geometric.data import Batch, Data


def _to_float_tensor(x) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32)


def _as_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _select_shap_output(shap_values, output_index: Optional[int]):
    """
    Normalize SHAP outputs for plotting:
    - list[num_outputs] -> pick one output if requested.
    - ndarray with output dimension -> optionally slice one output.
    """
    if isinstance(shap_values, list):
        if output_index is None:
            return shap_values[0]
        return shap_values[output_index]

    arr = np.asarray(shap_values)
    if output_index is not None and arr.ndim == 3:
        return arr[:, :, output_index]
    return arr


def generate_explanations(
    model: torch.nn.Module,
    background_data,
    test_sample,
    feature_names: Optional[Sequence[str]] = None,
    output_index: Optional[int] = None,
    plot_type: str = "dot",
    show_plot: bool = False,
):
    """
    Generate SHAP values for torch models with tensor-like inputs.
    Returns raw SHAP output from explainer.
    """
    model.eval()

    background_tensor = _to_float_tensor(background_data)
    test_tensor = _to_float_tensor(test_sample)

    try:
        explainer = shap.DeepExplainer(model, background_tensor)
    except Exception:
        # Fallback for architectures unsupported by DeepExplainer internals.
        explainer = shap.GradientExplainer(model, background_tensor)

    shap_values = explainer.shap_values(test_tensor)
    values_for_plot = _select_shap_output(shap_values, output_index=output_index)

    test_np = _as_numpy(test_tensor)
    n_features = test_np.shape[1]
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    shap.summary_plot(values_for_plot, test_np, feature_names=feature_names, plot_type=plot_type, show=show_plot)
    return shap_values


class FixedGraphHybridWrapper(torch.nn.Module):
    """
    Adapter for SHAP on HybridToxPredictor:
    explains tabular features while keeping a fixed graph context.
    """

    def __init__(self, hybrid_model: torch.nn.Module, graph_template: Data | Batch, output_index: Optional[int] = None):
        super().__init__()
        self.hybrid_model = hybrid_model
        self.graph_template = graph_template
        self.output_index = output_index

    def _build_graph_batch(self, batch_size: int, device: torch.device) -> Batch:
        if isinstance(self.graph_template, Batch):
            graph_batch = self.graph_template.to(device)
            if graph_batch.num_graphs != batch_size:
                raise ValueError(
                    f"graph_template batch has {graph_batch.num_graphs} graphs, but tabular batch has {batch_size} rows."
                )
            return graph_batch

        if isinstance(self.graph_template, Data):
            graphs = [self.graph_template.clone() for _ in range(batch_size)]
            return Batch.from_data_list(graphs).to(device)

        raise TypeError("graph_template must be a torch_geometric.data.Data or Batch.")

    def forward(self, tabular_features: torch.Tensor) -> torch.Tensor:
        tabular_features = tabular_features.float()
        device = tabular_features.device
        graph_batch = self._build_graph_batch(tabular_features.size(0), device=device)

        logits = self.hybrid_model(graph_batch, tabular_features)
        probs = torch.sigmoid(logits)
        if self.output_index is None:
            return probs
        return probs[:, self.output_index : self.output_index + 1]


def generate_hybrid_explanations(
    hybrid_model: torch.nn.Module,
    graph_template: Data | Batch,
    background_tabular,
    test_tabular,
    feature_names: Optional[Sequence[str]] = None,
    output_index: int = 0,
    plot_type: str = "dot",
    show_plot: bool = False,
):
    """
    SHAP explanations for HybridToxPredictor tabular inputs under fixed graph context.
    output_index selects which toxicity task to explain.
    """
    wrapped = FixedGraphHybridWrapper(hybrid_model, graph_template=graph_template, output_index=output_index)
    return generate_explanations(
        model=wrapped,
        background_data=background_tabular,
        test_sample=test_tabular,
        feature_names=feature_names,
        output_index=None,
        plot_type=plot_type,
        show_plot=show_plot,
    )
