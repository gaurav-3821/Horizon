import pandas as pd
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent if SCRIPT_DIR.parent.name == "code" else SCRIPT_DIR
DATA_DIR = PROJECT_ROOT / "data"

# -- CLSI breakpoints for Dataset 1 (MIC disk diffusion mm) --
# Values BELOW these thresholds = Resistant
BREAKPOINTS = {
    "IMIPENEM": {"R": 13, "S": 16},
    "CEFTAZIDIME": {"R": 14, "S": 18},
    "GENTAMICIN": {"R": 12, "S": 15},
    "AUGMENTIN": {"R": 13, "S": 18},
    "CIPROFLOXACIN": {"R": 15, "S": 21},
}


def mic_to_label(value, breakpoints):
    """Convert numeric MIC to R/I/S label using CLSI breakpoints."""
    if pd.isna(value):
        return np.nan
    if value <= breakpoints["R"]:
        return "R"
    elif value <= breakpoints["S"]:
        return "I"
    else:
        return "S"


def resolve_input_path(path_value: str) -> Path:
    p = Path(path_value)
    if p.is_absolute():
        return p
    candidates = [SCRIPT_DIR / p, PROJECT_ROOT / p, DATA_DIR / p]
    for c in candidates:
        if c.exists():
            return c
    return p


def load_dataset1(path="Dataset_track_b.xlsx"):
    df = pd.read_excel(resolve_input_path(path))
    # Convert MIC numbers to R/I/S
    antibiotic_cols = ["IMIPENEM", "CEFTAZIDIME", "GENTAMICIN", "AUGMENTIN", "CIPROFLOXACIN"]
    for col in antibiotic_cols:
        df[col] = df[col].apply(lambda x: mic_to_label(x, BREAKPOINTS[col]))
    # Parse location into site and sample_type
    df["site"] = df["Location"].str.split("-").str[0]
    df["sample_type"] = df["Location"].str.split("-").str[1]
    df["source"] = "dataset1"
    df.drop(columns=["Location"], inplace=True)
    return df


# Column name mapping: Dataset2 columns -> unified names matching Dataset1
COL_MAP = {
    "IPM": "IMIPENEM",
    "CZ": "CEFTAZIDIME",
    "GEN": "GENTAMICIN",
    "AMC": "AUGMENTIN",
    "CIP": "CIPROFLOXACIN",
}


def normalize_label(val):
    """Normalize messy R/S/I labels to uppercase standard."""
    if pd.isna(val) or str(val).strip() in ["?", "missing", ""]:
        return np.nan
    val = str(val).strip().upper()
    if val == "INTERMEDIATE":
        return "I"
    if val in ["R", "S", "I"]:
        return val
    return np.nan


def load_dataset2(path="Bacteria_dataset_Multiresictance.csv"):
    df = pd.read_csv(resolve_input_path(path))
    # Drop PII columns
    df.drop(columns=["ID", "Name", "Email", "Address", "Notes", "Collection_Date"], inplace=True, errors="ignore")
    # Rename antibiotic cols to unified names
    df.rename(columns=COL_MAP, inplace=True)
    # Normalize all antibiotic label columns
    antibiotic_cols = ["IMIPENEM", "CEFTAZIDIME", "GENTAMICIN", "AUGMENTIN", "CIPROFLOXACIN"]
    for col in antibiotic_cols:
        if col in df.columns:
            df[col] = df[col].apply(normalize_label)
    # Parse age and gender from 'age/gender' column
    df["age"] = df["age/gender"].str.extract(r"(\d+)").astype(float)
    df["gender"] = df["age/gender"].str.extract(r"/([MF])", expand=False)
    df.drop(columns=["age/gender"], inplace=True)
    # Normalize Souches (bacteria species) -- strip ID prefix
    df["Souches"] = df["Souches"].str.replace(r"^S\d+\s+", "", regex=True).str.strip()
    df["source"] = "dataset2"
    return df


def load_combined():
    """Load, clean, and merge both datasets into one unified dataframe."""
    df1 = load_dataset1()
    df2 = load_dataset2()
    combined = pd.concat([df1, df2], axis=0, ignore_index=True)
    return combined


if __name__ == "__main__":
    df = load_combined()
    print("Combined shape:", df.shape)
    print("Columns:", list(df.columns))
    print(df.head())
