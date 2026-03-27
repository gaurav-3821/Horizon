from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from mordred import Calculator, descriptors
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors


class ToxDataLoader:
    """Data loader and feature generator for toxicity modeling."""

    def __init__(self, tox21_path: str, zinc_path: Optional[str] = None):
        """
        tox21_path: Path to Tox21 dataset (NR-AhR, SR-p53, etc.)
        zinc_path: Optional ZINC250k for transfer learning
        """
        self.tox21 = self._read_csv(tox21_path)
        self.zinc = self._read_csv(zinc_path) if zinc_path else None
        self._rdkit_desc = Descriptors._descList
        self._mordred_calc = Calculator(descriptors, ignore_3D=True)

    @staticmethod
    def _read_csv(path: str) -> pd.DataFrame:
        csv_path = Path(path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset not found: {csv_path}")
        return pd.read_csv(csv_path)

    def validate_smiles(self, df: pd.DataFrame, smiles_col: str = "smiles") -> pd.DataFrame:
        """Remove invalid SMILES strings and keep RDKit molecule objects."""
        if smiles_col not in df.columns:
            raise KeyError(f"SMILES column '{smiles_col}' not present in dataframe.")

        out = df.copy()
        out["mol"] = out[smiles_col].astype(str).apply(Chem.MolFromSmiles)
        return out.loc[out["mol"].notna()].copy()

    def compute_rdkit_features(self, mol: Chem.Mol, max_descriptors: int = 200) -> dict:
        """
        Generate RDKit molecular descriptors.
        max_descriptors controls truncation to a fixed feature budget.
        """
        if mol is None:
            raise ValueError("mol cannot be None.")

        result = {}
        for name, func in self._rdkit_desc[:max_descriptors]:
            try:
                value = func(mol)
            except Exception:
                value = np.nan
            result[name] = value
        return result

    def compute_mordred_descriptors(self, mols: pd.Series) -> pd.DataFrame:
        """Generate Mordred descriptors and coerce non-numeric values to NaN."""
        mordred_df = self._mordred_calc.pandas(mols)
        mordred_df = mordred_df.apply(pd.to_numeric, errors="coerce")
        mordred_df = mordred_df.replace([np.inf, -np.inf], np.nan)
        mordred_df = mordred_df.dropna(axis=1, how="all")
        return mordred_df

    @staticmethod
    def generate_fingerprints(
        mol: Chem.Mol, radius: int = 2, n_bits: int = 2048, as_array: bool = True
    ):
        """Generate Morgan fingerprints from an RDKit Mol."""
        if mol is None:
            raise ValueError("mol cannot be None.")

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        if not as_array:
            return fp

        arr = np.zeros((n_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    def featurize_dataframe(
        self,
        df: pd.DataFrame,
        smiles_col: str = "smiles",
        include_mordred: bool = True,
        rdkit_max_descriptors: int = 200,
        fp_radius: int = 2,
        fp_n_bits: int = 2048,
    ) -> pd.DataFrame:
        """
        Build a training-ready feature dataframe from a raw dataframe.
        Keeps original columns and appends RDKit, Mordred (optional), and fingerprint features.
        """
        clean = self.validate_smiles(df, smiles_col=smiles_col).reset_index(drop=True)

        rdkit_rows = [self.compute_rdkit_features(m, max_descriptors=rdkit_max_descriptors) for m in clean["mol"]]
        rdkit_df = pd.DataFrame(rdkit_rows)

        fp_rows = [
            self.generate_fingerprints(m, radius=fp_radius, n_bits=fp_n_bits, as_array=True) for m in clean["mol"]
        ]
        fp_df = pd.DataFrame(fp_rows, columns=[f"fp_{i}" for i in range(fp_n_bits)], dtype=np.int8)

        pieces = [clean, rdkit_df, fp_df]
        if include_mordred:
            pieces.append(self.compute_mordred_descriptors(clean["mol"]))

        features = pd.concat(pieces, axis=1)
        return features
