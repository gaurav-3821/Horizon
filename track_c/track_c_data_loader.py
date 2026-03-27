import pandas as pd
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent if SCRIPT_DIR.parent.name == "code" else SCRIPT_DIR
JHU_PATH = str(PROJECT_ROOT / "data" / "time_series_covid19_confirmed_global.csv")
OWID_PATH = str(PROJECT_ROOT / "data" / "owid-covid-data.csv")


def resolve_input_path(path_value: str | Path) -> Path:
    p = Path(path_value)
    if p.is_absolute() and p.exists():
        return p
    candidates = [
        SCRIPT_DIR / p,
        PROJECT_ROOT / p,
        PROJECT_ROOT / "data" / p.name,
        PROJECT_ROOT / p.name,
    ]
    for c in candidates:
        if c.exists():
            return c
    return p


def load_jhu(path=JHU_PATH):
    df = pd.read_csv(resolve_input_path(path))
    date_cols = [c for c in df.columns if "/" in c]
    df_long = df.melt(
        id_vars=["Province/State", "Country/Region", "Lat", "Long"],
        value_vars=date_cols,
        var_name="date",
        value_name="confirmed",
    )
    df_long["date"] = pd.to_datetime(df_long["date"])
    df_long = df_long.groupby(["Country/Region", "date"])["confirmed"].sum().reset_index()
    df_long = df_long.sort_values(["Country/Region", "date"]).reset_index(drop=True)

    df_long["daily_new"] = df_long.groupby("Country/Region")["confirmed"].diff().fillna(0).clip(lower=0)
    df_long["rolling_7d"] = df_long.groupby("Country/Region")["daily_new"].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )

    df_long["growth_rate"] = (
        df_long.groupby("Country/Region")["confirmed"].pct_change(periods=14).replace([np.inf, -np.inf], np.nan)
    )
    df_long["doubling_time"] = np.log(2) / df_long["growth_rate"].replace(0, np.nan)

    first_case = df_long[df_long["confirmed"] > 0].groupby("Country/Region")["date"].min()
    df_long["days_since_first"] = df_long.apply(
        lambda r: (r["date"] - first_case.get(r["Country/Region"], r["date"])).days, axis=1
    )

    for lag in [1, 3, 7, 14]:
        df_long[f"lag_{lag}d"] = df_long.groupby("Country/Region")["daily_new"].shift(lag)

    p33 = df_long["rolling_7d"].quantile(0.33)
    p66 = df_long["rolling_7d"].quantile(0.66)
    df_long["risk_label"] = pd.cut(
        df_long["rolling_7d"],
        bins=[-np.inf, p33, p66, np.inf],
        labels=["Low", "Medium", "High"],
    )
    return df_long


def load_owid(path=OWID_PATH):
    cols = [
        "iso_code",
        "location",
        "date",
        "total_vaccinations_per_hundred",
        "people_fully_vaccinated_per_hundred",
        "total_tests_per_thousand",
        "hospital_patients_per_million",
        "population_density",
        "median_age",
        "gdp_per_capita",
        "human_development_index",
        "stringency_index",
    ]
    resolved = resolve_input_path(path)
    header_cols = pd.read_csv(resolved, nrows=0).columns
    df = pd.read_csv(resolved, usecols=[c for c in cols if c in header_cols])
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"location": "Country/Region"})
    return df


def add_global_rolling_infection_rate(df):
    global_wave = (
        df.groupby("date", as_index=False)["daily_new"]
        .sum()
        .sort_values("date")
        .rename(columns={"daily_new": "global_daily_new"})
    )
    global_wave["Global_Rolling_Infection_Rate"] = global_wave["global_daily_new"].rolling(7, min_periods=1).mean()
    return df.merge(global_wave[["date", "Global_Rolling_Infection_Rate"]], on="date", how="left")


def load_combined(jhu_path=JHU_PATH, owid_path=OWID_PATH):
    jhu = load_jhu(jhu_path)
    owid_available = resolve_input_path(owid_path).exists()

    if owid_available:
        owid = load_owid(owid_path)
        combined = jhu.merge(owid, on=["Country/Region", "date"], how="left")

        if "stringency_index" in combined.columns:
            combined = combined.sort_values(["Country/Region", "date"]).reset_index(drop=True)
            combined["stringency_index"] = combined.groupby("Country/Region")["stringency_index"].ffill().bfill()

        print("Merged with OWID. Shape:", combined.shape)
    else:
        combined = jhu.copy()
        print("OWID not found - using JHU only. Shape:", combined.shape)

    combined = add_global_rolling_infection_rate(combined)
    combined = combined.dropna(subset=["daily_new", "lag_7d"])

    print("Final shape after dropping NaN lags:", combined.shape)
    print("Columns:", list(combined.columns))
    return combined


if __name__ == "__main__":
    df = load_combined()
    print(df.head())
