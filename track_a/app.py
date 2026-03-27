# RUN COMMAND: streamlit run track_a/app.py
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
import torch
import torch.nn.functional as F
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Draw, MACCSkeys
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv, global_mean_pool


warnings.filterwarnings("ignore")
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

TARGET_COLS = [
    "NR-AhR",
    "NR-AR",
    "NR-AR-LBD",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
]

EXAMPLES = {
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Ethanol": "CCO",
    "Paracetamol": "CC(=O)Nc1ccc(O)cc1",
}

RDKIT_FEATURE_COLS = [
    "MolWt",
    "LogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "NumAromaticRings",
    "RingCount",
    "NumSaturatedRings",
    "NumAliphaticRings",
]
FINGERPRINT_BITS = 166
TABULAR_DIM = len(RDKIT_FEATURE_COLS) + FINGERPRINT_BITS


class MolecularGNN(torch.nn.Module):
    def __init__(self, num_node_features=5, hidden_channels=128, num_targets=12):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.dropout = torch.nn.Dropout(0.3)
        self.batch_norm1 = torch.nn.BatchNorm1d(hidden_channels)
        self.batch_norm2 = torch.nn.BatchNorm1d(hidden_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        return global_mean_pool(x, batch)


class HybridModel(torch.nn.Module):
    def __init__(self, num_node_features=5, tabular_dim=TABULAR_DIM, hidden=128, num_targets=12):
        super().__init__()
        self.gnn = MolecularGNN(num_node_features, hidden, num_targets)
        self.tabular_encoder = torch.nn.Sequential(
            torch.nn.Linear(tabular_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 128),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, num_targets),
        )

    def forward(self, x, edge_index, batch, tabular):
        gnn_out = self.gnn(x, edge_index, batch)
        tab_out = self.tabular_encoder(tabular)
        combined = torch.cat([gnn_out, tab_out], dim=1)
        return torch.sigmoid(self.classifier(combined))


class FeatureEngine:
    def __init__(self):
        self.maccs = MACCSkeys

    def validate_smiles(self, smiles_string):
        try:
            return Chem.MolFromSmiles(str(smiles_string))
        except Exception:
            return None

    def get_rdkit_features(self, mol):
        return {
            "MolWt": Descriptors.MolWt(mol),
            "LogP": Descriptors.MolLogP(mol),
            "TPSA": Descriptors.TPSA(mol),
            "NumHDonors": Descriptors.NumHDonors(mol),
            "NumHAcceptors": Descriptors.NumHAcceptors(mol),
            "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
            "NumAromaticRings": Descriptors.NumAromaticRings(mol),
            "RingCount": Descriptors.RingCount(mol),
            "NumSaturatedRings": Descriptors.NumSaturatedRings(mol),
            "NumAliphaticRings": Descriptors.NumAliphaticRings(mol),
        }

    def get_maccs_fp(self, mol):
        fp = self.maccs.GenMACCSKeys(mol)
        arr = np.zeros((fp.GetNumBits(),), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        if arr.shape[0] == FINGERPRINT_BITS + 1:
            return arr[1:]
        return arr[:FINGERPRINT_BITS]

    def mol_to_graph(self, smiles_string):
        mol = self.validate_smiles(smiles_string)
        if mol is None:
            return None
        node_features = []
        for atom in mol.GetAtoms():
            node_features.append(
                [
                    float(atom.GetAtomicNum()),
                    float(atom.GetDegree()),
                    float(atom.GetFormalCharge()),
                    float(int(atom.GetHybridization())),
                    float(atom.GetIsAromatic()),
                ]
            )
        edges = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edges.append([i, j])
            edges.append([j, i])
        x = torch.tensor(node_features, dtype=torch.float32)
        edge_index = (
            torch.tensor(edges, dtype=torch.long).t().contiguous()
            if edges
            else torch.empty((2, 0), dtype=torch.long)
        )
        return {"x": x, "edge_index": edge_index}


@st.cache_resource
def load_model():
    model_path = ARTIFACTS_DIR / "best_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found.")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    ckpt_tab_dim = checkpoint.get("tabular_dim", TABULAR_DIM) if isinstance(checkpoint, dict) else TABULAR_DIM
    model = HybridModel(num_node_features=5, tabular_dim=ckpt_tab_dim, num_targets=12)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        raise TypeError("best_model.pth must contain a state_dict checkpoint.")
    model.eval()
    return model


@st.cache_data
def load_background_tabular():
    pkl_path = ARTIFACTS_DIR / "tox21_processed.pkl"
    if not pkl_path.exists():
        return None
    df = pd.read_pickle(pkl_path)
    rdkit = df[RDKIT_FEATURE_COLS].to_numpy(dtype=np.float32)
    fp = np.vstack(df["maccs_fp"].values).astype(np.float32)
    return np.concatenate([rdkit, fp], axis=1)[:40]


def predict_smiles(model, smiles_string):
    fe = FeatureEngine()
    mol = fe.validate_smiles(smiles_string)
    if mol is None:
        raise ValueError("Invalid SMILES string.")
    rdkit_feat = fe.get_rdkit_features(mol)
    fp = fe.get_maccs_fp(mol)
    tab = np.concatenate(
        [np.array([rdkit_feat[c] for c in RDKIT_FEATURE_COLS], dtype=np.float32), fp]
    ).reshape(1, -1)
    graph = fe.mol_to_graph(smiles_string)
    data = Data(x=graph["x"], edge_index=graph["edge_index"])
    batch = Batch.from_data_list([data])
    with torch.no_grad():
        pred = model(batch.x, batch.edge_index, batch.batch, torch.tensor(tab, dtype=torch.float32)).cpu().numpy()[0]
    return mol, rdkit_feat, pred, tab, data


def get_risk_level(probability):
    if probability > 0.5:
        return "High"
    if probability >= 0.3:
        return "Moderate"
    return "Low"


def risk_color(probability):
    if probability > 0.5:
        return "#d62728"
    if probability >= 0.3:
        return "#ffbf00"
    return "#2ca02c"


def compute_single_shap(model, graph_data, tab_input, background):
    if background is None:
        return None

    def wrapper_predict(tab_np):
        tab_np = np.asarray(tab_np, dtype=np.float32)
        graph_batch = Batch.from_data_list([graph_data.clone() for _ in range(tab_np.shape[0])])
        with torch.no_grad():
            out = model(
                graph_batch.x,
                graph_batch.edge_index,
                graph_batch.batch,
                torch.tensor(tab_np, dtype=torch.float32),
            )
        return out.numpy()

    explainer = shap.KernelExplainer(wrapper_predict, background)
    shap_values = explainer.shap_values(tab_input, nsamples=100)
    if isinstance(shap_values, list):
        values = np.asarray(shap_values[0][0])
        base = explainer.expected_value[0]
    else:
        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            values = arr[0, :, 0]
            base = explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
        else:
            values = arr[0]
            base = explainer.expected_value
    return values, base


st.set_page_config(page_title="ToxPredict AI", layout="wide", page_icon="🧬")

st.sidebar.title("About ToxPredict")
st.sidebar.info(
    "ToxPredict estimates 12 Tox21 toxicity endpoints from molecular structure "
    "using a hybrid GNN plus tabular feature model."
)
st.sidebar.markdown("### Model Metrics")
st.sidebar.metric("Mean ROC-AUC", "0.8416")
st.sidebar.metric("Sample Count", "7,831")

st.title("🧬 Molecular Toxicity Prediction")
st.caption("CodeCure AI Hackathon | IIT BHU Varanasi")

left, right = st.columns([1, 1])

with left:
    example_name = st.selectbox("Choose example molecule", list(EXAMPLES.keys()), index=0)
    custom_smiles = st.text_input("Or enter custom SMILES", value=EXAMPLES[example_name])
    run_pred = st.button("Predict", type="primary")

with right:
    fe = FeatureEngine()
    mol_preview = fe.validate_smiles(custom_smiles)
    if mol_preview is not None:
        st.image(Draw.MolToImage(mol_preview, size=(420, 280)), caption="2D Molecule")
        props = fe.get_rdkit_features(mol_preview)
        c1, c2, c3 = st.columns(3)
        c1.metric("MolWt", f"{props['MolWt']:.2f}")
        c2.metric("LogP", f"{props['LogP']:.2f}")
        c3.metric("TPSA", f"{props['TPSA']:.2f}")
    else:
        st.warning("Invalid SMILES")

if run_pred:
    try:
        model = load_model()
        mol, rdkit_feat, pred, tab, graph_data = predict_smiles(model, custom_smiles)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

    st.subheader("Toxicity Probabilities")
    colors = [risk_color(float(p)) for p in pred]
    fig = go.Figure(data=[go.Bar(x=TARGET_COLS, y=pred, marker_color=colors)])
    fig.update_layout(yaxis_title="Probability", yaxis_range=[0, 1], xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

    risk_levels = [get_risk_level(float(p)) for p in pred]
    high = sum(level == "High" for level in risk_levels)
    moderate = sum(level == "Moderate" for level in risk_levels)
    low = sum(level == "Low" for level in risk_levels)

    m1, m2, m3 = st.columns(3)
    m1.metric("High Risk", str(high))
    m2.metric("Moderate", str(moderate))
    m3.metric("Low", str(low))

    with st.expander("Detailed Target Table", expanded=False):
        detail_df = pd.DataFrame(
            {
                "Target": TARGET_COLS,
                "Probability": np.round(pred, 4),
                "Risk Level": risk_levels,
            }
        )
        st.dataframe(detail_df, use_container_width=True, hide_index=True)

    st.subheader("SHAP Waterfall (Target: NR-AhR)")
    background = load_background_tabular()
    shap_result = compute_single_shap(model, graph_data, tab, background)
    if shap_result is None:
        st.info("SHAP unavailable: track_a/artifacts/tox21_processed.pkl not found.")
    else:
        shap_values, base_value = shap_result
        feature_names = RDKIT_FEATURE_COLS + [f"maccs_{i+1}" for i in range(FINGERPRINT_BITS)]
        explanation = shap.Explanation(
            values=shap_values,
            base_values=base_value,
            data=tab[0],
            feature_names=feature_names,
        )
        shap.plots.waterfall(explanation, max_display=10, show=False)
        st.pyplot()
