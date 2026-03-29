import json
import re
import tarfile
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
EXTERNAL_ROOT = REPO_ROOT.parent
DATA_DIR = SCRIPT_DIR / "data"
ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"

ANTIBIOTIC_TARGETS = ["IMIPENEM", "CEFTAZIDIME", "GENTAMICIN", "AUGMENTIN", "CIPROFLOXACIN"]
ANTIBIOTIC_CLASS_MAP = {
    "IMIPENEM": "carbapenem",
    "CEFTAZIDIME": "beta-lactam",
    "GENTAMICIN": "aminoglycoside",
    "AUGMENTIN": "beta-lactam",
    "CIPROFLOXACIN": "fluoroquinolone",
}
ANTIBIOTIC_JOIN_TERMS = {
    "IMIPENEM": ["imipenem", "carbapenem", "beta-lactam"],
    "CEFTAZIDIME": ["ceftazidime", "cephalosporin", "beta-lactam"],
    "GENTAMICIN": ["gentamicin", "aminoglycoside"],
    "AUGMENTIN": ["augmentin", "amoxicillin", "clavulanate", "beta-lactam"],
    "CIPROFLOXACIN": ["ciprofloxacin", "fluoroquinolone"],
}
CARD_CLASS_KEYWORDS = {
    "vancomycin": ["vancomycin", "glycopeptide"],
    "fluoroquinolone": ["fluoroquinolone", "ciprofloxacin", "nalidixic", "quinolone"],
    "beta-lactam": ["beta-lactam", "penicillin", "cephalosporin", "cephamycin", "augmentin", "amoxicillin"],
    "tetracycline": ["tetracycline", "tigecycline", "doxycycline", "minocycline"],
    "aminoglycoside": ["aminoglycoside", "gentamicin", "amikacin", "tobramycin", "kanamycin"],
    "carbapenem": ["carbapenem", "imipenem", "meropenem", "doripenem", "ertapenem"],
    "macrolide": ["macrolide", "erythromycin", "azithromycin", "clarithromycin"],
    "colistin": ["colistin", "polymyxin"],
}
CLASS_PRIORITY = [
    "carbapenem",
    "fluoroquinolone",
    "aminoglycoside",
    "tetracycline",
    "macrolide",
    "colistin",
    "vancomycin",
    "beta-lactam",
]
BREAKPOINTS = {
    "IMIPENEM": {"R": 13, "S": 16},
    "CEFTAZIDIME": {"R": 14, "S": 18},
    "GENTAMICIN": {"R": 12, "S": 15},
    "AUGMENTIN": {"R": 13, "S": 18},
    "CIPROFLOXACIN": {"R": 15, "S": 21},
}
COL_MAP = {
    "IPM": "IMIPENEM",
    "CZ": "CEFTAZIDIME",
    "GEN": "GENTAMICIN",
    "AMC": "AUGMENTIN",
    "CIP": "CIPROFLOXACIN",
}
KMER_FEATURE_LIMIT = 256
LEAKAGE_EXPORT_DROP_COLUMNS = {
    "source_row_id",
    "aro_gene_candidates",
    "aro_gene_name",
    "card_gene_name",
    "card_aro_accession",
    "card_gene_candidates",
    "card_associated_antibiotics",
    "card_resistance_mechanism",
    "card_amr_gene_family",
    "card_match_count",
    "antibiotic_name_norm",
    "resolved_gene_name",
    "resolved_gene_name_norm",
}


def ensure_runtime_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_gene_name(value) -> str:
    return normalize_text(value).replace(" ", "")


def unique_join(values, sep="|", limit=12):
    clean = []
    seen = set()
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        clean.append(text)
    return sep.join(clean[:limit]) if clean else np.nan


def keyword_match_classes(text: str):
    norm = normalize_text(text)
    matches = []
    for class_name in CLASS_PRIORITY:
        if any(keyword in norm for keyword in CARD_CLASS_KEYWORDS[class_name]):
            matches.append(class_name)
    return matches


def mic_to_label(value, breakpoints):
    if pd.isna(value):
        return np.nan
    if value <= breakpoints["R"]:
        return "R"
    if value <= breakpoints["S"]:
        return "I"
    return "S"


def normalize_label(val):
    if pd.isna(val) or str(val).strip() in ["?", "missing", ""]:
        return np.nan
    val = str(val).strip().upper()
    if val == "INTERMEDIATE":
        return "I"
    if val in ["R", "S", "I"]:
        return val
    return np.nan


def resolve_input_path(path_value: str) -> Path:
    p = Path(path_value)
    if p.is_absolute() and p.exists():
        return p
    candidates = [
        DATA_DIR / p,
        SCRIPT_DIR / p,
        REPO_ROOT / p,
        EXTERNAL_ROOT / p,
        EXTERNAL_ROOT / p.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return p


def card_archive_paths():
    return {
        "data_tar": resolve_input_path(
            "card-data.tar track B dataset/card-data.tar track B dataset"
        ),
        "ontology_tar": resolve_input_path(
            "card-ontology.tar dataset fot track B/card-ontology.tar dataset fot track B"
        ),
    }


def ensure_card_asset(filename: str) -> Path:
    ensure_runtime_dirs()
    target = DATA_DIR / filename
    if target.exists():
        return target

    archives = card_archive_paths()
    if filename == "aro.tsv":
        archive_path = archives["ontology_tar"]
    else:
        archive_path = archives["data_tar"]

    if not archive_path.exists():
        raise FileNotFoundError(f"Required CARD archive not found for {filename}: {archive_path}")

    with tarfile.open(archive_path, "r") as tar:
        members = [m for m in tar.getmembers() if Path(m.name).name == filename]
        if not members:
            raise FileNotFoundError(f"{filename} not found inside archive: {archive_path}")
        extracted = tar.extractfile(members[0])
        if extracted is None:
            raise FileNotFoundError(f"Could not extract {filename} from archive: {archive_path}")
        target.write_bytes(extracted.read())
    return target


def load_dataset1(path="Dataset track b.xlsx"):
    df = pd.read_excel(resolve_input_path(path))
    for col in ANTIBIOTIC_TARGETS:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: mic_to_label(x, BREAKPOINTS[col]))

    df["site"] = df["Location"].astype(str).str.split("-").str[0]
    df["sample_type"] = df["Location"].astype(str).str.split("-").str[1]
    df["species"] = np.nan
    df["age"] = np.nan
    df["gender"] = np.nan
    df["Diabetes"] = np.nan
    df["Hypertension"] = np.nan
    df["Hospital_before"] = np.nan
    df["Infection_Freq"] = np.nan
    df["source_dataset"] = "mendeley"
    df["source_row_id"] = [f"mendeley_{i}" for i in range(len(df))]
    return df


def load_dataset2(path="dataset track2.zip"):
    zip_path = resolve_input_path(path)
    with zipfile.ZipFile(zip_path) as zip_file:
        csv_names = [name for name in zip_file.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            raise FileNotFoundError(f"No CSV found inside zip archive: {zip_path}")
        with zip_file.open(csv_names[0]) as handle:
            df = pd.read_csv(handle)

    df = df.rename(columns=COL_MAP)
    df.drop(columns=["ID", "Name", "Email", "Address", "Notes", "Collection_Date"], inplace=True, errors="ignore")
    for col in ANTIBIOTIC_TARGETS:
        if col in df.columns:
            df[col] = df[col].apply(normalize_label)

    age_gender = df.get("age/gender", pd.Series(index=df.index, dtype=object)).astype(str)
    df["age"] = pd.to_numeric(age_gender.str.extract(r"(\d+)")[0], errors="coerce")
    df["gender"] = age_gender.str.extract(r"/([MF])", expand=False)
    df["species"] = (
        df.get("Souches", pd.Series(index=df.index, dtype=object))
        .astype(str)
        .str.replace(r"^S\d+\s+", "", regex=True)
        .str.strip()
        .replace({"nan": np.nan})
    )
    df["site"] = np.nan
    df["sample_type"] = np.nan
    df["source_dataset"] = "kaggle"
    df["source_row_id"] = [f"kaggle_{i}" for i in range(len(df))]
    return df


def to_long_format(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "source_row_id",
        "source_dataset",
        "site",
        "sample_type",
        "species",
        "age",
        "gender",
        "Diabetes",
        "Hypertension",
        "Hospital_before",
        "Infection_Freq",
    ]
    existing_keep = [col for col in keep_cols if col in df.columns]
    long_df = df.melt(
        id_vars=existing_keep,
        value_vars=[col for col in ANTIBIOTIC_TARGETS if col in df.columns],
        var_name="antibiotic_name",
        value_name="resistance_label",
    )
    long_df["resistance_label"] = long_df["resistance_label"].apply(normalize_label)
    long_df = long_df[long_df["resistance_label"].notna()].reset_index(drop=True)
    long_df["antibiotic_class"] = long_df["antibiotic_name"].map(ANTIBIOTIC_CLASS_MAP)
    long_df["antibiotic_name_norm"] = long_df["antibiotic_name"].map(normalize_text)
    return long_df


def load_aro_dataframe() -> pd.DataFrame:
    aro_path = ensure_card_asset("aro.tsv")
    aro_df = pd.read_csv(aro_path, sep="\t")
    aro_df["aro_gene_name"] = aro_df["CARD Short Name"].fillna(aro_df["Name"]).astype(str).str.strip()
    aro_df["aro_antibiotic_class"] = aro_df["Description"].fillna("").apply(
        lambda text: unique_join(keyword_match_classes(text))
    )
    aro_df = aro_df[aro_df["aro_antibiotic_class"].notna()].copy()
    aro_df["primary_class"] = aro_df["aro_antibiotic_class"].astype(str).str.split("|").str[0]
    return aro_df


def aggregate_aro_metadata(aro_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for antibiotic_name, class_name in ANTIBIOTIC_CLASS_MAP.items():
        subset = aro_df[aro_df["primary_class"] == class_name]
        gene_candidates = sorted(subset["aro_gene_name"].dropna().astype(str).unique().tolist())
        records.append(
            {
                "antibiotic_name": antibiotic_name,
                "aro_antibiotic_class": class_name,
                "aro_gene_name": gene_candidates[0] if gene_candidates else np.nan,
                "aro_gene_candidates": unique_join(gene_candidates, limit=20),
                "aro_match_count": int(len(subset)),
            }
        )
    return pd.DataFrame(records)


def parse_card_json() -> pd.DataFrame:
    card_json_path = ensure_card_asset("card.json")
    with open(card_json_path, encoding="utf-8") as handle:
        raw = json.load(handle)

    records = []
    for value in raw.values():
        if not isinstance(value, dict):
            continue

        gene_name = str(value.get("CARD_short_name") or value.get("ARO_name") or "").strip()
        aro_accession = str(value.get("ARO_accession") or "").strip()
        aro_description = str(value.get("ARO_description") or "").strip()
        categories = value.get("ARO_category") or {}

        antibiotics = []
        drug_classes = []
        mechanisms = []
        families = []

        for category in categories.values():
            if not isinstance(category, dict):
                continue
            cat_name = str(category.get("category_aro_name") or "").strip()
            cat_class = str(category.get("category_aro_class_name") or "").strip()
            if cat_class == "Antibiotic":
                antibiotics.append(cat_name)
            elif cat_class == "Drug Class":
                drug_classes.append(cat_name)
            elif cat_class == "Resistance Mechanism":
                mechanisms.append(cat_name)
            elif cat_class == "AMR Gene Family":
                families.append(cat_name)

        matched_classes = []
        for text in antibiotics + drug_classes + [aro_description]:
            matched_classes.extend(keyword_match_classes(text))
        matched_classes = [item for item in CLASS_PRIORITY if item in set(matched_classes)]

        records.append(
            {
                "card_aro_accession": aro_accession or np.nan,
                "card_gene_name": gene_name or np.nan,
                "card_associated_antibiotics": unique_join(antibiotics + drug_classes, limit=30),
                "card_resistance_mechanism": unique_join(mechanisms, limit=20),
                "card_amr_gene_family": unique_join(families, limit=20),
                "matched_classes": matched_classes,
            }
        )

    rows = []
    for record in records:
        for class_name in record["matched_classes"]:
            out = record.copy()
            out["card_antibiotic_class"] = class_name
            rows.append(out)
    return pd.DataFrame(rows)


def aggregate_card_metadata(card_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for antibiotic_name, class_name in ANTIBIOTIC_CLASS_MAP.items():
        subset = card_df[card_df["card_antibiotic_class"] == class_name]
        gene_candidates = sorted(subset["card_gene_name"].dropna().astype(str).unique().tolist())
        records.append(
            {
                "antibiotic_name": antibiotic_name,
                "card_aro_accession": unique_join(subset["card_aro_accession"].tolist(), limit=20),
                "card_gene_name": gene_candidates[0] if gene_candidates else np.nan,
                "card_gene_candidates": unique_join(gene_candidates, limit=20),
                "card_associated_antibiotics": unique_join(subset["card_associated_antibiotics"].tolist(), limit=20),
                "card_resistance_mechanism": unique_join(subset["card_resistance_mechanism"].tolist(), limit=20),
                "card_amr_gene_family": unique_join(subset["card_amr_gene_family"].tolist(), limit=20),
                "card_match_count": int(len(subset)),
            }
        )
    return pd.DataFrame(records)


def extract_gene_name_from_fasta_header(header: str) -> str:
    text = header.lstrip(">").strip()
    parts = text.split("|")
    if len(parts) >= 4:
        return parts[3].split("[")[0].strip()
    return text.split("[")[0].strip()


def parse_fasta_sequences() -> dict:
    fasta_path = ensure_card_asset("protein_fasta_protein_homolog_model.fasta")
    sequences = {}
    current_gene = None
    seq_parts = []

    with open(fasta_path, encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_gene and seq_parts and current_gene not in sequences:
                    sequences[current_gene] = "".join(seq_parts)
                current_gene = extract_gene_name_from_fasta_header(line)
                seq_parts = []
            else:
                seq_parts.append(re.sub(r"[^A-Za-z]", "", line).upper())

    if current_gene and seq_parts and current_gene not in sequences:
        sequences[current_gene] = "".join(seq_parts)
    return sequences


def kmer_frequency(sequence: str, k: int = 4) -> Counter:
    seq = re.sub(r"[^A-Z]", "", sequence.upper())
    if len(seq) < k:
        return Counter()
    return Counter(seq[i : i + k] for i in range(len(seq) - k + 1))


def build_fasta_feature_table(gene_names) -> pd.DataFrame:
    sequences = parse_fasta_sequences()
    normalized_map = {normalize_gene_name(gene): seq for gene, seq in sequences.items()}

    counts_by_gene = {}
    corpus_counts = Counter()
    for gene in gene_names:
        gene_norm = normalize_gene_name(gene)
        sequence = normalized_map.get(gene_norm)
        if not sequence:
            continue
        kmer_counts = kmer_frequency(sequence, k=4)
        if not kmer_counts:
            continue
        counts_by_gene[gene_norm] = kmer_counts
        corpus_counts.update(kmer_counts)

    top_kmers = [kmer for kmer, _ in corpus_counts.most_common(KMER_FEATURE_LIMIT)]
    rows = []
    for gene_norm, counts in counts_by_gene.items():
        total = float(sum(counts.values()))
        row = {"resolved_gene_name_norm": gene_norm}
        row["fasta_sequence_found"] = 1
        for kmer in top_kmers:
            row[f"kmer4_{kmer}"] = counts.get(kmer, 0.0) / total if total else 0.0
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["resolved_gene_name_norm", "fasta_sequence_found"])
    return pd.DataFrame(rows)


def merge_all_sources() -> tuple[pd.DataFrame, dict]:
    ensure_runtime_dirs()

    df1_long = to_long_format(load_dataset1())
    df2_long = to_long_format(load_dataset2())
    base_df = pd.concat([df1_long, df2_long], ignore_index=True)

    aro_df = load_aro_dataframe()
    aro_meta = aggregate_aro_metadata(aro_df)
    card_df = parse_card_json()
    card_meta = aggregate_card_metadata(card_df)

    unified = base_df.merge(aro_meta, on="antibiotic_name", how="left")
    unified = unified.merge(card_meta, on="antibiotic_name", how="left")

    unified["resolved_gene_name"] = unified["aro_gene_name"].fillna(unified["card_gene_name"])
    unified["resolved_gene_name_norm"] = unified["resolved_gene_name"].apply(normalize_gene_name)

    matched_genes = unified["resolved_gene_name"].dropna().astype(str).unique().tolist()
    fasta_features = build_fasta_feature_table(matched_genes)
    unified = unified.merge(fasta_features, on="resolved_gene_name_norm", how="left")

    if "fasta_sequence_found" not in unified.columns:
        unified["fasta_sequence_found"] = 0
    unified["fasta_sequence_found"] = unified["fasta_sequence_found"].fillna(0).astype(int)

    report = {
        "total_rows": int(len(unified)),
        "total_features": int(unified.shape[1]),
        "source_rows": {
            "mendeley": int(len(df1_long)),
            "kaggle": int(len(df2_long)),
        },
        "match_rates": {
            "aro_metadata": round(float(unified["aro_gene_name"].notna().mean() * 100), 2),
            "card_metadata": round(float(unified["card_aro_accession"].notna().mean() * 100), 2),
            "fasta_kmer": round(float((unified["fasta_sequence_found"] == 1).mean() * 100), 2),
        },
        "class_distribution": unified["resistance_label"].value_counts(dropna=False).to_dict(),
        "drop_before_training": sorted(
            [
                "aro_gene_candidates",
                "aro_gene_name",
                "card_gene_name",
                "card_aro_accession",
                "card_gene_candidates",
                "card_associated_antibiotics",
                "card_resistance_mechanism",
                "card_amr_gene_family",
                "card_match_count",
                "antibiotic_name_norm",
                "resolved_gene_name",
                "resolved_gene_name_norm",
                "source_row_id",
                "source_dataset",
            ]
        ),
    }
    return unified, report


def save_unified_dataset(df: pd.DataFrame) -> Path:
    ensure_runtime_dirs()
    output_path = ARTIFACTS_DIR / "unified_dataset_final.csv"
    export_df = df.drop(columns=sorted(LEAKAGE_EXPORT_DROP_COLUMNS), errors="ignore")
    export_df.to_csv(output_path, index=False)
    return output_path


def load_combined() -> pd.DataFrame:
    unified, _ = merge_all_sources()
    return unified


if __name__ == "__main__":
    unified_df, summary = merge_all_sources()
    output_file = save_unified_dataset(unified_df)

    print(f"Total rows: {summary['total_rows']}")
    print(f"Total features: {summary['total_features']}")
    print("Source row counts:", summary["source_rows"])
    print("Match rates (%):", summary["match_rates"])
    print("Class distribution:", summary["class_distribution"])
    print(f"Saved unified dataset to: {output_file}")
