from __future__ import annotations

import importlib
import random
import subprocess
import sys
import traceback
import warnings
from pathlib import Path
from typing import Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent if SCRIPT_DIR.parent.name == "code" else SCRIPT_DIR
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def ensure_runtime_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def resolve_input_path(path_value: str, fallback_names: List[str] | None = None) -> Path:
    p = Path(path_value)
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend([SCRIPT_DIR / p, PROJECT_ROOT / p, DATA_DIR / p])

    if fallback_names:
        for name in fallback_names:
            candidates.extend([PROJECT_ROOT / name, DATA_DIR / name])

    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Could not resolve input path '{path_value}' in project or data directories.")


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


def run_section(name: str, fn):
    print(f"\n{'=' * 20} {name} {'=' * 20}")
    try:
        return fn()
    except Exception:
        print("ERROR:")
        print(traceback.format_exc())
        raise


def section_1_install_and_import():
    required = [
        ("torch", "torch"),
        ("torch-geometric", "torch_geometric"),
        ("rdkit", "rdkit"),
        ("xgboost", "xgboost"),
        ("streamlit", "streamlit"),
        ("shap", "shap"),
        ("plotly", "plotly"),
        ("scikit-learn", "sklearn"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
        ("imbalanced-learn", "imblearn"),
    ]

    missing = []
    for pip_name, import_name in required:
        if importlib.util.find_spec(import_name) is None:
            missing.append(pip_name)

    if missing:
        print(f"Installing missing packages: {missing}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
    else:
        print("All required packages already installed.")

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    import xgboost  # noqa: F401
    from rdkit import Chem, RDLogger  # noqa: F401
    from rdkit.Chem import Descriptors, Draw, MACCSkeys  # noqa: F401
    from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: F401
    from sklearn.model_selection import train_test_split  # noqa: F401
    from torch_geometric.data import Data  # noqa: F401
    from torch_geometric.loader import DataLoader  # noqa: F401
    from torch_geometric.nn import GCNConv, global_mean_pool  # noqa: F401

    warnings.filterwarnings("ignore")
    RDLogger.DisableLog("rdApp.*")
    print("Setup complete")
    print("SECTION 1 COMPLETE")


def section_2_load_and_explore(tox_path: Path, zinc_path: Path) -> Dict:
    import pandas as pd
    from rdkit import Chem

    df_tox = pd.read_csv(tox_path)
    df_zinc = pd.read_csv(zinc_path, compression="infer")

    print(f"df_tox shape: {df_tox.shape}")
    print(f"df_zinc shape: {df_zinc.shape}")
    print(f"df_tox columns: {df_tox.columns.tolist()}")
    print(f"df_zinc columns: {df_zinc.columns.tolist()}")
    print(f"target_cols: {TARGET_COLS}")

    print("\nTox21 NaN percentages per target:")
    nan_pct = (df_tox[TARGET_COLS].isna().mean() * 100).round(2)
    print(nan_pct.to_string())

    print("\nTox21 positive-label percentages per target (on non-NaN labels):")
    pos_pct = {}
    for col in TARGET_COLS:
        valid = df_tox[col].notna()
        if valid.sum() == 0:
            pos_pct[col] = float("nan")
        else:
            pos_pct[col] = round((df_tox.loc[valid, col] == 1).mean() * 100, 2)
    print(pos_pct)

    valid_smiles_count = df_tox["smiles"].apply(lambda s: Chem.MolFromSmiles(str(s)) is not None).sum()
    print(f"\nValid SMILES count in tox21: {int(valid_smiles_count)}")

    print("\nZINC stats (logP, qed, SAS):")
    print(df_zinc[["logP", "qed", "SAS"]].describe().to_string())

    print("SECTION 2 COMPLETE")
    return {"df_tox": df_tox, "df_zinc": df_zinc}


class FeatureEngine:
    def __init__(self):
        from rdkit import Chem
        from rdkit.Chem import Descriptors, MACCSkeys

        self.Chem = Chem
        self.Desc = Descriptors
        self.MACCSkeys = MACCSkeys

    def validate_smiles(self, smiles_string):
        try:
            return self.Chem.MolFromSmiles(str(smiles_string))
        except Exception:
            return None

    def get_rdkit_features(self, mol):
        return {
            "MolWt": self.Desc.MolWt(mol),
            "LogP": self.Desc.MolLogP(mol),
            "TPSA": self.Desc.TPSA(mol),
            "NumHDonors": self.Desc.NumHDonors(mol),
            "NumHAcceptors": self.Desc.NumHAcceptors(mol),
            "NumRotatableBonds": self.Desc.NumRotatableBonds(mol),
            "NumAromaticRings": self.Desc.NumAromaticRings(mol),
            "RingCount": self.Desc.RingCount(mol),
            "NumSaturatedRings": self.Desc.NumSaturatedRings(mol),
            "NumAliphaticRings": self.Desc.NumAliphaticRings(mol),
        }

    def get_maccs_fp(self, mol):
        import numpy as np
        from rdkit import DataStructs

        fp = self.MACCSkeys.GenMACCSKeys(mol)
        arr = np.zeros((fp.GetNumBits(),), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        # RDKit MACCS is 167 bits with index 0 unused; keep the stable 166-key vector.
        if arr.shape[0] == FINGERPRINT_BITS + 1:
            arr = arr[1:]
        elif arr.shape[0] > FINGERPRINT_BITS:
            arr = arr[:FINGERPRINT_BITS]
        elif arr.shape[0] < FINGERPRINT_BITS:
            arr = np.pad(arr, (0, FINGERPRINT_BITS - arr.shape[0]), mode="constant")
        return arr

    def mol_to_graph(self, smiles_string):
        import torch

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
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        return {"x": x, "edge_index": edge_index}


def section_3_feature_engineering(df_tox):
    import numpy as np
    import pandas as pd

    fe = FeatureEngine()
    df = df_tox.copy()
    df["mol"] = df["smiles"].apply(fe.validate_smiles)
    df = df[df["mol"].notna()].reset_index(drop=True)

    rdkit_df = df["mol"].apply(fe.get_rdkit_features).apply(pd.Series)
    df = pd.concat([df, rdkit_df], axis=1)
    df["maccs_fp"] = df["mol"].apply(fe.get_maccs_fp)
    df["graph"] = df["smiles"].apply(fe.mol_to_graph)

    mask_matrix = df[TARGET_COLS].notna().astype(np.float32).to_numpy()
    label_matrix = df[TARGET_COLS].fillna(0).astype(np.float32).to_numpy()

    df["mask_vector"] = list(mask_matrix)
    df["label_vector"] = list(label_matrix)

    print(f"Final valid molecule count: {len(df)}")
    print(f"mask_matrix shape: {mask_matrix.shape}")
    print(f"label_matrix shape: {label_matrix.shape}")

    ensure_runtime_dirs()
    out_path = ARTIFACTS_DIR / "tox21_processed.pkl"
    df.to_pickle(out_path)
    print(f"Saved processed dataframe to {out_path.resolve()}")
    print("SECTION 3 COMPLETE")

    return {"processed_df": df, "feature_engine": fe, "mask_matrix": mask_matrix, "label_matrix": label_matrix}


def section_4_build_dataset(processed_df):
    import numpy as np
    import torch
    from sklearn.model_selection import train_test_split
    from torch.utils.data import Dataset
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader as GeoDataLoader

    class ToxDataset(Dataset):
        def __init__(self, dataframe):
            self.graphs = dataframe["graph"].tolist()
            self.y = torch.tensor(np.vstack(dataframe["label_vector"].values), dtype=torch.float32)
            self.mask = torch.tensor(np.vstack(dataframe["mask_vector"].values), dtype=torch.float32)
            self.y_raw = torch.tensor(dataframe[TARGET_COLS].to_numpy(dtype=np.float32), dtype=torch.float32)
            rdkit_matrix = dataframe[RDKIT_FEATURE_COLS].to_numpy(dtype=np.float32)
            fp_matrix = np.vstack(dataframe["maccs_fp"].values).astype(np.float32)
            self.tabular = torch.tensor(np.concatenate([rdkit_matrix, fp_matrix], axis=1), dtype=torch.float32)

        def __len__(self):
            return len(self.graphs)

        def __getitem__(self, idx):
            graph = self.graphs[idx]
            data = Data(x=graph["x"], edge_index=graph["edge_index"])
            data.y = self.y[idx].unsqueeze(0)
            data.mask = self.mask[idx].unsqueeze(0)
            data.y_raw = self.y_raw[idx].unsqueeze(0)
            data.tabular = self.tabular[idx].unsqueeze(0)
            return data

    train_df, val_df = train_test_split(processed_df, test_size=0.2, random_state=42, shuffle=True)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_dataset = ToxDataset(train_df)
    val_dataset = ToxDataset(val_df)

    train_loader = GeoDataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = GeoDataLoader(val_dataset, batch_size=64, shuffle=False)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    batch = next(iter(train_loader))
    print(f"Batch x shape: {tuple(batch.x.shape)}")
    print(f"Batch edge_index shape: {tuple(batch.edge_index.shape)}")
    print(f"Batch y shape: {tuple(batch.y.shape)}")
    print(f"Batch mask shape: {tuple(batch.mask.shape)}")
    print(f"Batch tabular shape: {tuple(batch.tabular.shape)}")
    print("SECTION 4 COMPLETE")

    return {"train_loader": train_loader, "val_loader": val_loader, "train_dataset": train_dataset, "val_dataset": val_dataset}


def section_5_model_architecture():
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, global_mean_pool

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
            x = global_mean_pool(x, batch)
            return x

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
            output = self.classifier(combined)
            return torch.sigmoid(output)

    model = HybridModel()
    n_params = sum(p.numel() for p in model.parameters())
    print(model)
    print(f"Total parameters: {n_params:,}")
    print("SECTION 5 COMPLETE")
    return {"model": model, "HybridModel": HybridModel}


def section_6_train(
    model,
    train_loader,
    val_loader,
    df_zinc=None,
    epochs=80,
    pretrain_epochs=5,
    freeze_gnn_epochs=5,
    zinc_batch_size=256,
    zinc_max_samples=50000,
):
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F
    from rdkit import Chem
    from sklearn.metrics import roc_auc_score
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader as GeoDataLoader

    class MaskedFocalLoss(torch.nn.Module):
        def __init__(self, alpha=0.75, gamma=2):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma

        def forward(self, predictions, targets, mask):
            bce = F.binary_cross_entropy(predictions, targets, reduction="none")
            pt = torch.exp(-bce)
            focal = self.alpha * (1 - pt) ** self.gamma * bce
            masked_loss = focal * mask
            return masked_loss.sum() / torch.clamp(mask.sum(), min=1.0)

    def evaluate_auc(y_true, y_pred, y_mask):
        per_target = {}
        aucs = []
        for i, col in enumerate(TARGET_COLS):
            valid = y_mask[:, i] == 1
            if valid.sum() < 2:
                per_target[col] = float("nan")
                continue
            yt = y_true[valid, i]
            yp = y_pred[valid, i]
            if len(np.unique(yt)) < 2:
                per_target[col] = float("nan")
                continue
            auc = roc_auc_score(yt, yp)
            per_target[col] = float(auc)
            aucs.append(float(auc))
        mean_auc = float(np.mean(aucs)) if aucs else float("nan")
        return mean_auc, per_target

    def smiles_to_graph_data(smiles):
        mol = Chem.MolFromSmiles(str(smiles))
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
        return Data(x=x, edge_index=edge_index)

    def build_zinc_pretrain_dataset(zinc_df):
        if zinc_df is None:
            return []
        required_cols = {"smiles", "logP", "qed"}
        if not required_cols.issubset(set(zinc_df.columns)):
            return []
        sub = zinc_df.dropna(subset=["smiles", "logP", "qed"]).copy()
        if zinc_max_samples and len(sub) > zinc_max_samples:
            sub = sub.sample(n=zinc_max_samples, random_state=42)

        data_list = []
        for row in sub.itertuples(index=False):
            graph = smiles_to_graph_data(getattr(row, "smiles", None))
            if graph is None:
                continue
            graph.z = torch.tensor([[float(getattr(row, "logP")), float(getattr(row, "qed"))]], dtype=torch.float32)
            data_list.append(graph)
        return data_list

    class ZINCPretrainHead(torch.nn.Module):
        def __init__(self, gnn_encoder):
            super().__init__()
            self.gnn_encoder = gnn_encoder
            self.reg_head = torch.nn.Sequential(
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(64, 2),
            )

        def forward(self, x, edge_index, batch):
            emb = self.gnn_encoder(x, edge_index, batch)
            return self.reg_head(emb)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Stage 1: Pre-train GNN encoder on ZINC (logP, qed).
    zinc_graphs = build_zinc_pretrain_dataset(df_zinc)
    if len(zinc_graphs) > 0 and pretrain_epochs > 0:
        zinc_loader = GeoDataLoader(zinc_graphs, batch_size=zinc_batch_size, shuffle=True)
        pretrain_model = ZINCPretrainHead(model.gnn).to(device)
        pretrain_optimizer = torch.optim.AdamW(pretrain_model.parameters(), lr=0.002, weight_decay=1e-6)
        pretrain_criterion = torch.nn.MSELoss()

        print(f"Starting Stage 1 ZINC pre-training on {len(zinc_graphs)} molecules...")
        for p_epoch in range(pretrain_epochs):
            pretrain_model.train()
            running = 0.0
            n_batches = 0
            for batch in zinc_loader:
                batch = batch.to(device)
                pretrain_optimizer.zero_grad(set_to_none=True)
                pred = pretrain_model(batch.x, batch.edge_index, batch.batch)
                target = batch.z.view(-1, 2)
                loss = pretrain_criterion(pred, target)
                loss.backward()
                pretrain_optimizer.step()
                running += float(loss.item())
                n_batches += 1
            print(f"Pre-train epoch {p_epoch + 1}/{pretrain_epochs} | mse={running / max(n_batches, 1):.4f}")
    else:
        print("Skipping Stage 1 pre-training (ZINC unavailable or empty after filtering).")

    # Stage 2: Transfer + fine-tune on Tox21.
    freeze_epochs = min(max(int(freeze_gnn_epochs), 0), int(epochs))
    gnn_is_frozen = freeze_epochs > 0
    if gnn_is_frozen:
        for p in model.gnn.parameters():
            p.requires_grad = False
        print(f"Stage 2 fine-tuning: GNN frozen for first {freeze_epochs} epoch(s).")

    def build_finetune_optimizer(frozen):
        if frozen:
            params = [p for p in model.parameters() if p.requires_grad]
            return torch.optim.AdamW(params, lr=0.001, weight_decay=1e-5)
        gnn_params = [p for p in model.gnn.parameters() if p.requires_grad]
        head_params = [p for n, p in model.named_parameters() if not n.startswith("gnn.")]
        return torch.optim.AdamW(
            [
                {"params": gnn_params, "lr": 1e-4},
                {"params": head_params, "lr": 1e-3},
            ],
            weight_decay=1e-5,
        )

    optimizer = build_finetune_optimizer(gnn_is_frozen)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    criterion = MaskedFocalLoss(alpha=0.75, gamma=2)

    class EarlyStopping:
        def __init__(self, patience=10, min_delta=1e-4):
            self.patience = patience
            self.min_delta = min_delta
            self.best_score = None
            self.counter = 0
            self.best_state = None

        def step(self, score, model):
            if self.best_score is None or score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
                self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                self.counter += 1
            return self.counter >= self.patience

        def restore(self, model):
            model.load_state_dict(self.best_state)

    train_losses: List[float] = []
    val_aucs: List[float] = []
    best_auc = -1.0
    best_per_target = {}
    best_state_dict = None
    early_stopping = EarlyStopping(patience=10, min_delta=1e-4)

    for epoch in range(epochs):
        if gnn_is_frozen and epoch == freeze_epochs:
            for p in model.gnn.parameters():
                p.requires_grad = True
            gnn_is_frozen = False
            optimizer = build_finetune_optimizer(False)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
            print("Unfroze pre-trained GNN encoder with low LR for transfer fine-tuning.")

        model.train()
        running_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            y_raw = batch.y_raw.view(-1, len(TARGET_COLS))
            m = batch.mask.view(-1, len(TARGET_COLS))
            tab = batch.tabular.view(-1, TABULAR_DIM)

            optimizer.zero_grad(set_to_none=True)
            out = model(batch.x, batch.edge_index, batch.batch, tab)

            # Critical debug guard: ensure mask matches non-NaN ground truth exactly.
            non_nan_mask = torch.isfinite(y_raw).float()
            mask_sum = int(m.sum().item())
            non_nan_sum = int(non_nan_mask.sum().item())
            assert mask_sum == non_nan_sum, (
                f"Mask mismatch: mask.sum()={mask_sum}, non_nan_count={non_nan_sum}"
            )
            assert torch.equal(m, non_nan_mask), "Mask tensor does not match non-NaN label positions."

            y = torch.nan_to_num(y_raw, nan=0.0)
            loss = criterion(out, y, m)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            n_batches += 1

        train_loss = running_loss / max(n_batches, 1)
        train_losses.append(train_loss)

        model.eval()
        val_preds = []
        val_labels = []
        val_masks = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                y_raw = batch.y_raw.view(-1, len(TARGET_COLS))
                m = batch.mask.view(-1, len(TARGET_COLS))
                tab = batch.tabular.view(-1, TABULAR_DIM)
                out = model(batch.x, batch.edge_index, batch.batch, tab)
                val_preds.append(out.cpu().numpy())
                val_labels.append(torch.nan_to_num(y_raw, nan=0.0).cpu().numpy())
                val_masks.append(m.cpu().numpy())

        y_pred = np.vstack(val_preds)
        y_true = np.vstack(val_labels)
        y_mask = np.vstack(val_masks)
        mean_auc, per_target_auc = evaluate_auc(y_true, y_pred, y_mask)
        val_auc = mean_auc
        val_aucs.append(val_auc)

        if np.isfinite(mean_auc) and mean_auc > best_auc:
            best_auc = mean_auc
            best_per_target = per_target_auc
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        scheduler_metric = val_auc if np.isfinite(val_auc) else -1.0
        scheduler.step(scheduler_metric)

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch + 1}/{epochs} | train_loss={train_loss:.4f} | mean_val_roc_auc={mean_auc:.4f}")

        early_metric = val_auc if np.isfinite(val_auc) else -1.0
        if early_stopping.step(early_metric, model):
            print(f"Early stopping at epoch {epoch+1}. Best val AUC: {early_stopping.best_score:.4f}")
            break

    early_stopping.restore(model)
    print(f"Restored best model with val AUC: {early_stopping.best_score:.4f}")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_aucs, label="Val Mean ROC-AUC")
    plt.xlabel("Epoch")
    plt.ylabel("ROC-AUC")
    plt.title("Validation ROC-AUC")
    plt.grid(alpha=0.3)
    plt.legend()

    ensure_runtime_dirs()
    training_plot_path = ARTIFACTS_DIR / "training_results.png"
    plt.tight_layout()
    plt.savefig(training_plot_path, dpi=150)
    plt.close()

    if best_state_dict is None:
        best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state_dict)
    checkpoint = {
        "state_dict": best_state_dict,
        "num_node_features": 5,
        "tabular_dim": TABULAR_DIM,
        "num_targets": len(TARGET_COLS),
    }
    model_path = ARTIFACTS_DIR / "best_model.pth"
    torch.save(checkpoint, model_path)

    print("Per-target ROC-AUC (best epoch):")
    for k in TARGET_COLS:
        print(f"  {k}: {best_per_target.get(k, float('nan')):.4f}")

    print(f"Saved: {training_plot_path}")
    print(f"Saved: {model_path}")
    student_results = train_student_model(model, train_loader, val_loader)
    print("SECTION 6 COMPLETE")

    return {
        "train_losses": train_losses,
        "val_aucs": val_aucs,
        "best_per_target_auc": best_per_target,
        "model": model,
        "student_results": student_results,
    }


def train_student_model(teacher_model, train_loader, val_loader, epochs=30, temperature=2.0, lr=1e-3):
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score

    class StudentMLP(torch.nn.Module):
        def __init__(self, input_dim=TABULAR_DIM, num_targets=len(TARGET_COLS)):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.15),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, num_targets),
            )

        def forward(self, tabular):
            return torch.sigmoid(self.net(tabular))

    def softened_probs(probs, temp):
        eps = 1e-5
        probs = probs.clamp(eps, 1 - eps)
        logits = torch.log(probs / (1 - probs))
        return torch.sigmoid(logits / max(float(temp), 1e-3))

    def masked_macro_auc(y_true, y_pred, y_mask):
        aucs = []
        for i in range(len(TARGET_COLS)):
            valid = y_mask[:, i] == 1
            if valid.sum() < 2:
                continue
            yt = y_true[valid, i]
            yp = y_pred[valid, i]
            if len(np.unique(yt)) < 2:
                continue
            aucs.append(float(roc_auc_score(yt, yp)))
        return float(np.mean(aucs)) if aucs else float("nan")

    device = next(teacher_model.parameters()).device
    teacher_model.eval()
    student = StudentMLP().to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-5)
    criterion = torch.nn.BCELoss()

    best_auc = -1.0
    best_state = None

    print("Starting teacher-student distillation on tabular features...")
    for epoch in range(epochs):
        student.train()
        running_loss = 0.0
        batches = 0

        for batch in train_loader:
            batch = batch.to(device)
            tab = batch.tabular.view(-1, TABULAR_DIM)
            with torch.no_grad():
                teacher_probs = teacher_model(batch.x, batch.edge_index, batch.batch, tab)
                soft_targets = softened_probs(teacher_probs, temperature)

            optimizer.zero_grad(set_to_none=True)
            student_probs = student(tab)
            loss = criterion(student_probs, soft_targets)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            batches += 1

        student.eval()
        val_preds = []
        val_labels = []
        val_masks = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                tab = batch.tabular.view(-1, TABULAR_DIM)
                preds = student(tab)
                val_preds.append(preds.cpu().numpy())
                val_labels.append(torch.nan_to_num(batch.y_raw.view(-1, len(TARGET_COLS)), nan=0.0).cpu().numpy())
                val_masks.append(batch.mask.view(-1, len(TARGET_COLS)).cpu().numpy())

        y_pred = np.vstack(val_preds)
        y_true = np.vstack(val_labels)
        y_mask = np.vstack(val_masks)
        val_auc = masked_macro_auc(y_true, y_pred, y_mask)

        if np.isfinite(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
            mean_loss = running_loss / max(batches, 1)
            print(f"Student epoch {epoch + 1}/{epochs} | distill_loss={mean_loss:.4f} | val_auc={val_auc:.4f}")

    if best_state is not None:
        student.load_state_dict(best_state)

    ensure_runtime_dirs()
    student_path = ARTIFACTS_DIR / "student_mlp_distilled.pth"
    torch.save(
        {
            "state_dict": student.state_dict(),
            "tabular_dim": TABULAR_DIM,
            "num_targets": len(TARGET_COLS),
            "temperature": float(temperature),
        },
        student_path,
    )
    print(f"Saved: {student_path}")
    print(f"Student best validation ROC-AUC: {best_auc:.4f}")
    return {"student_model_path": str(student_path), "best_val_auc": best_auc}


def section_7_shap(model, val_dataset):
    import matplotlib.pyplot as plt
    import numpy as np
    import shap
    import torch
    from torch_geometric.data import Batch, Data

    model.eval()
    device = next(model.parameters()).device

    n_samples = min(100, len(val_dataset))
    idx = np.random.choice(len(val_dataset), size=n_samples, replace=False)
    sample_graphs = [val_dataset[int(i)] for i in idx]
    sample_tab = np.vstack([d.tabular.squeeze(0).numpy() for d in sample_graphs]).astype(np.float32)

    background_size = min(30, n_samples)
    background = sample_tab[:background_size]

    ref = sample_graphs[0]
    ref_graph = Data(x=ref.x, edge_index=ref.edge_index)

    def wrapper_predict(tab_np):
        tab_np = np.asarray(tab_np, dtype=np.float32)
        graph_batch = Batch.from_data_list([ref_graph.clone() for _ in range(tab_np.shape[0])]).to(device)
        tab_t = torch.tensor(tab_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            out = model(graph_batch.x, graph_batch.edge_index, graph_batch.batch, tab_t)
        return out.detach().cpu().numpy()

    explain_tab = sample_tab[: min(20, len(sample_tab))]

    try:
        explainer = shap.KernelExplainer(wrapper_predict, background)
        shap_values = explainer.shap_values(explain_tab, nsamples=100)
        expected_value = explainer.expected_value
    except Exception:
        # Fallback to model-agnostic explainer if KernelExplainer fails.
        explainer = shap.Explainer(wrapper_predict, background)
        shap_exp = explainer(explain_tab)
        shap_values = shap_exp.values
        expected_value = shap_exp.base_values.mean(axis=0) if shap_exp.base_values.ndim > 1 else shap_exp.base_values

    feature_names = RDKIT_FEATURE_COLS + [f"maccs_{i+1}" for i in range(FINGERPRINT_BITS)]
    target_idx = 0

    if isinstance(shap_values, list):
        sv = np.asarray(shap_values[target_idx])
        expected = expected_value[target_idx] if isinstance(expected_value, (list, np.ndarray)) else expected_value
    else:
        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            sv = arr[:, :, target_idx]
            expected = expected_value[target_idx] if isinstance(expected_value, (list, np.ndarray)) else expected_value
        else:
            sv = arr
            expected = expected_value

    ensure_runtime_dirs()
    shap_summary_path = ARTIFACTS_DIR / "shap_summary.png"
    shap_waterfall_path = ARTIFACTS_DIR / "shap_waterfall.png"

    plt.figure(figsize=(10, 6))
    shap.summary_plot(sv, explain_tab, feature_names=feature_names, max_display=10, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(shap_summary_path, dpi=150)
    plt.close()

    exp = shap.Explanation(values=sv[0], base_values=expected, data=sample_tab[0], feature_names=feature_names)
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(exp, max_display=10, show=False)
    plt.tight_layout()
    plt.savefig(shap_waterfall_path, dpi=150)
    plt.close()

    print(f"Saved: {shap_summary_path}")
    print(f"Saved: {shap_waterfall_path}")
    print("SHAP analysis complete")
    print("SECTION 7 COMPLETE")


def section_8_streamlit_app():
    app_code = """# RUN COMMAND: streamlit run app.py --server.port 8501
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
PROJECT_ROOT = SCRIPT_DIR.parent.parent if SCRIPT_DIR.parent.name == "code" else SCRIPT_DIR
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

TARGET_COLS = [
    "NR-AhR", "NR-AR", "NR-AR-LBD", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
    "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
]

EXAMPLES = {
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Ethanol": "CCO",
    "Paracetamol": "CC(=O)Nc1ccc(O)cc1",
}

RDKIT_FEATURE_COLS = [
    "MolWt", "LogP", "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "NumAromaticRings", "RingCount",
    "NumSaturatedRings", "NumAliphaticRings",
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
        out = self.classifier(combined)
        return torch.sigmoid(out)


class FeatureEngine:
    def __init__(self):
        self.MACCSkeys = MACCSkeys

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
        fp = self.MACCSkeys.GenMACCSKeys(mol)
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
            node_features.append([
                float(atom.GetAtomicNum()),
                float(atom.GetDegree()),
                float(atom.GetFormalCharge()),
                float(int(atom.GetHybridization())),
                float(atom.GetIsAromatic()),
            ])
        edges = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edges.append([i, j])
            edges.append([j, i])
        x = torch.tensor(node_features, dtype=torch.float32)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
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
    tab = np.concatenate([rdkit, fp], axis=1)
    return tab[:40]


def predict_smiles(model, smiles_string):
    fe = FeatureEngine()
    mol = fe.validate_smiles(smiles_string)
    if mol is None:
        raise ValueError("Invalid SMILES string.")
    rdkit_feat = fe.get_rdkit_features(mol)
    fp = fe.get_maccs_fp(mol)
    tab = np.concatenate([np.array([rdkit_feat[c] for c in RDKIT_FEATURE_COLS], dtype=np.float32), fp]).reshape(1, -1)
    graph = fe.mol_to_graph(smiles_string)
    data = Data(x=graph["x"], edge_index=graph["edge_index"])
    batch = Batch.from_data_list([data])
    with torch.no_grad():
        pred = model(batch.x, batch.edge_index, batch.batch, torch.tensor(tab, dtype=torch.float32)).cpu().numpy()[0]
    return mol, rdkit_feat, pred, tab, data


def get_risk_level(p):
    if p > 0.5:
        return "High"
    if p >= 0.3:
        return "Moderate"
    return "Low"


def risk_color(p):
    if p > 0.5:
        return "#d62728"
    if p >= 0.3:
        return "#ffbf00"
    return "#2ca02c"


def compute_single_shap(model, graph_data, tab_input, background):
    if background is None:
        return None

    def wrapper_predict(tab_np):
        tab_np = np.asarray(tab_np, dtype=np.float32)
        graph_batch = Batch.from_data_list([graph_data.clone() for _ in range(tab_np.shape[0])])
        with torch.no_grad():
            out = model(graph_batch.x, graph_batch.edge_index, graph_batch.batch, torch.tensor(tab_np, dtype=torch.float32))
        return out.numpy()

    explainer = shap.KernelExplainer(wrapper_predict, background)
    shap_values = explainer.shap_values(tab_input, nsamples=100)
    if isinstance(shap_values, list):
        sv = np.asarray(shap_values[0][0])
        base = explainer.expected_value[0]
    else:
        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            sv = arr[0, :, 0]
            base = explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
        else:
            sv = arr[0]
            base = explainer.expected_value
    return sv, base


st.set_page_config(page_title="ToxPredict AI", layout="wide", page_icon="🧬")

st.sidebar.title("About ToxPredict")
st.sidebar.info(
    "ToxPredict estimates 12 Tox21 toxicity endpoints from molecular structure "
    "using a hybrid GNN + tabular feature model."
)
st.sidebar.markdown("### Model Metrics")
st.sidebar.metric("Mean ROC-AUC", "See training output")
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
    fig = go.Figure(
        data=[go.Bar(x=TARGET_COLS, y=pred, marker_color=colors)]
    )
    fig.update_layout(yaxis_title="Probability", yaxis_range=[0, 1], xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

    risk_levels = [get_risk_level(float(p)) for p in pred]
    high = sum(r == "High" for r in risk_levels)
    moderate = sum(r == "Moderate" for r in risk_levels)
    low = sum(r == "Low" for r in risk_levels)

    m1, m2, m3 = st.columns(3)
    m1.metric("High Risk", str(high))
    m2.metric("Moderate", str(moderate))
    m3.metric("Low", str(low))

    with st.expander("Detailed Target Table", expanded=False):
        detail_df = pd.DataFrame({
            "Target": TARGET_COLS,
            "Probability": np.round(pred, 4),
            "Risk Level": risk_levels,
        })
        st.dataframe(detail_df, use_container_width=True, hide_index=True)

    st.subheader("SHAP Waterfall (Target: NR-AhR)")
    background = load_background_tabular()
    shap_result = compute_single_shap(model, graph_data, tab, background)
    if shap_result is None:
        st.info("SHAP unavailable: run training pipeline first to generate artifacts/tox21_processed.pkl.")
    else:
        shap_values, base_value = shap_result
        feature_names = RDKIT_FEATURE_COLS + [f"maccs_{i+1}" for i in range(FINGERPRINT_BITS)]
        exp = shap.Explanation(values=shap_values, base_values=base_value, data=tab[0], feature_names=feature_names)
        shap.plots.waterfall(exp, max_display=10, show=False)
        st.pyplot()

st.markdown("---")
st.caption("Built for CodeCure AI Hackathon 2025")
"""

    app_path = PROJECT_ROOT / "app.py"
    app_path.write_text(app_code, encoding="utf-8")
    print(f"Created {app_path} with full Streamlit dashboard.")
    print("Run command: streamlit run app.py --server.port 8501")
    print("SECTION 8 COMPLETE")


def main():
    import argparse
    import numpy as np
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--tox-path", type=str, default="tox21.csv")
    parser.add_argument("--zinc-path", type=str, default="250k_rndm_zinc_drugs_clean_3.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_runtime_dirs()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tox_path = resolve_input_path(
        args.tox_path,
        fallback_names=[
            "tox21.csv",
            "Tox21 DATASET/tox21.csv",
        ],
    )
    zinc_path = resolve_input_path(
        args.zinc_path,
        fallback_names=[
            "250k_rndm_zinc_drugs_clean_3.csv",
            "ZINC-250k  DATASET.zip",
        ],
    )

    run_section("SECTION 1: INSTALL AND IMPORT", section_1_install_and_import)
    loaded = run_section(
        "SECTION 2: LOAD AND EXPLORE DATA",
        lambda: section_2_load_and_explore(tox_path, zinc_path),
    )
    engineered = run_section("SECTION 3: FEATURE ENGINEERING", lambda: section_3_feature_engineering(loaded["df_tox"]))
    data_objs = run_section("SECTION 4: PYTORCH GEOMETRIC DATASET", lambda: section_4_build_dataset(engineered["processed_df"]))
    model_objs = run_section("SECTION 5: MODEL ARCHITECTURE", section_5_model_architecture)
    trained = run_section(
        "SECTION 6: MASKED FOCAL LOSS AND TRAINING",
        lambda: section_6_train(
            model_objs["model"],
            data_objs["train_loader"],
            data_objs["val_loader"],
            df_zinc=loaded["df_zinc"],
            epochs=args.epochs,
        ),
    )
    run_section("SECTION 7: SHAP EXPLAINABILITY", lambda: section_7_shap(trained["model"], data_objs["val_dataset"]))
    run_section("SECTION 8: STREAMLIT DASHBOARD", section_8_streamlit_app)

    print("\nPipeline execution finished.")
    print(f"Final mean validation ROC-AUC (last epoch): {trained['val_aucs'][-1]:.4f}")
    print("Run dashboard with: streamlit run app.py --server.port 8501")


if __name__ == "__main__":
    main()
