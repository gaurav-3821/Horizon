import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
import json

import joblib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"
DATA_DIR = SCRIPT_DIR / "data"

BACKGROUND = "#f5f5f0"
CARD_BG = "#ffffff"
BORDER = "#000000"
ACCENT = "#0066cc"
TEXT = "#0a0a0f"

st.set_page_config(page_title="Horizon | Track C", layout="wide", page_icon="🦠")

CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    *, *::before, *::after {{
        border-radius: 0px !important;
    }}
    html, body, [class*="css"] {{
        font-family: 'Inter', system-ui, sans-serif;
    }}
    .stApp {{
        background: {BACKGROUND};
        color: {TEXT};
    }}
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    [data-testid="stHeader"] {{display:none;}}
    [data-testid="stSidebar"] {{
        background: #eeeeea;
        border-right: 2px solid {BORDER};
    }}
    [data-testid="stSidebar"] * {{
        color: {TEXT};
    }}
    .block-container {{
        padding-top: 1.25rem;
        padding-bottom: 1.25rem;
        max-width: 1800px;
    }}
    .metric-card {{
        background: {CARD_BG};
        border: 2px solid {BORDER};
        padding: 18px 20px;
        min-height: 122px;
        box-shadow: 4px 4px 0px {BORDER};
    }}
    .metric-icon {{
        font-size: 1.25rem;
        margin-bottom: 10px;
        color: {TEXT};
    }}
    .metric-label {{
        color: {TEXT};
        font-size: 0.86rem;
        margin-bottom: 8px;
    }}
    .metric-value {{
        color: {TEXT};
        font-size: 1.8rem;
        font-weight: 800;
        line-height: 1.1;
    }}
    .metric-sub {{
        color: {TEXT};
        font-size: 0.82rem;
        margin-top: 6px;
    }}
    .hero-title {{
        font-size: 2.3rem;
        font-weight: 800;
        color: {TEXT};
        margin-bottom: 0.2rem;
    }}
    .hero-subtitle {{
        color: {TEXT};
        font-size: 1.02rem;
        margin-bottom: 0.75rem;
    }}
    .divider {{
        height: 2px;
        background: linear-gradient(90deg, {ACCENT}, transparent);
        margin-bottom: 0.9rem;
    }}
    .warning-banner {{
        background: #fff3cd;
        border: 2px solid {BORDER};
        color: #856404;
        padding: 12px 16px;
        margin-bottom: 1rem;
        font-weight: 600;
        box-shadow: 4px 4px 0px {BORDER};
    }}
    .card {{
        background: {CARD_BG};
        border: 2px solid {ACCENT};
        padding: 20px;
        box-shadow: 4px 4px 0px {ACCENT};
        height: 100%;
        margin-top: 0.4rem;
    }}
    .section-title {{
        font-size: 1.02rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: {TEXT};
        margin-bottom: 0.9rem;
    }}
    .small-note {{
        color: {TEXT};
        font-size: 0.9rem;
        line-height: 1.6;
    }}
    div[data-baseweb="select"] > div {{
        background: #ffffff !important;
        border: 2px solid {ACCENT} !important;
        color: {TEXT} !important;
    }}
    div[role="listbox"] {{
        background: #ffffff !important;
        color: {TEXT} !important;
        border: 2px solid {ACCENT} !important;
    }}
    div[role="option"] {{
        background: #ffffff !important;
        color: {TEXT} !important;
    }}
    div[role="option"]:hover {{
        background: #e8f1ff !important;
        color: {TEXT} !important;
    }}
    label, .stSelectbox label {{
        color: {TEXT} !important;
        font-weight: 600 !important;
    }}
    [data-testid="stMetric"] {{
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
    }}
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] * {{
        color: {TEXT} !important;
        opacity: 1 !important;
    }}
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] * {{
        color: {TEXT} !important;
        opacity: 1 !important;
    }}
    [data-testid="stMetricDelta"],
    [data-testid="stMetricDelta"] * {{
        color: {TEXT} !important;
        opacity: 1 !important;
    }}
    [data-baseweb="tab-list"] {{
        gap: 8px;
        margin-top: 0.4rem;
        margin-bottom: 0.8rem;
    }}
    button[role="tab"] {{
        background: #ffffff !important;
        color: {TEXT} !important;
        border: 2px solid {BORDER} !important;
        box-shadow: 4px 4px 0px {BORDER};
        font-weight: 700 !important;
        padding: 0.45rem 0.9rem !important;
    }}
    button[role="tab"][aria-selected="true"] {{
        background: #e8f1ff !important;
        color: {TEXT} !important;
        border: 2px solid {ACCENT} !important;
        box-shadow: 4px 4px 0px {ACCENT};
    }}
    button[role="tab"] p {{
        color: {TEXT} !important;
    }}
    [data-testid="stTabPanel"] {{
        background: {CARD_BG};
        border: 2px solid {ACCENT};
        box-shadow: 4px 4px 0px {ACCENT};
        padding: 20px;
        margin-top: 0.35rem;
    }}
</style>
"""


def resolve_artifact(name: str) -> Path:
    candidates = [ARTIFACTS_DIR / name, SCRIPT_DIR / name]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


@st.cache_resource
def load_models():
    forecaster = joblib.load(resolve_artifact("track_c_forecaster.pkl"))
    classifier = joblib.load(resolve_artifact("track_c_classifier.pkl"))
    return forecaster, classifier


@st.cache_data
def load_results():
    with open(resolve_artifact("track_c_results.json"), encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data
def load_data():
    from track_c_data_loader import load_combined

    jhu_path = DATA_DIR / "time_series_covid19_confirmed_global.csv"
    owid_path = DATA_DIR / "owid-covid-data.csv"
    if jhu_path.exists():
        return load_combined(str(jhu_path), str(owid_path))
    return load_combined()


def render_metric_card(icon: str, label: str, value: str, subtext: str = ""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon">{icon}</div>
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def bounded_score(value: float, baseline: float) -> float:
    if baseline <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 / (1.0 + (float(value) / float(baseline)))))


def make_radar_chart(title: str, labels: list[str], values: list[float], color: str, note: str):
    theta = labels + [labels[0]]
    r_values = values + [values[0]]
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=r_values,
            theta=theta,
            fill="toself",
            fillcolor="rgba(0, 102, 204, 0.16)" if color == ACCENT else "rgba(204, 0, 0, 0.12)",
            line=dict(color=color, width=3),
            marker=dict(color=color, size=8),
            hovertemplate="%{theta}: %{r:.3f}<extra></extra>",
            name=title,
        )
    )
    fig.update_layout(
        margin=dict(l=30, r=30, t=60, b=70),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        showlegend=False,
        title=dict(text=title, font=dict(color=TEXT, size=18)),
        polar=dict(
            bgcolor="#ffffff",
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(color=TEXT),
                gridcolor="#d9d9d9",
                linecolor="#000000",
                tickcolor="#000000",
            ),
            angularaxis=dict(
                tickfont=dict(color=TEXT, size=12),
                gridcolor="#d9d9d9",
                linecolor="#000000",
                tickcolor="#000000",
            ),
        ),
        annotations=[
            dict(
                text=note,
                x=0.5,
                y=-0.14,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(color=TEXT, size=11),
                align="center",
            )
        ],
    )
    return fig


def main():
    st.markdown(CSS, unsafe_allow_html=True)
    st.sidebar.title("Horizon")
    st.sidebar.caption("Track navigation")
    st.sidebar.radio(
        "Tracks",
        [
            "🧬 Track B — Antibiotic Resistance",
            "🧪 Track A — Drug Toxicity",
            "🦠 Track C — Epidemic Spread",
        ],
        index=2,
        label_visibility="collapsed",
    )

    forecaster_model, classifier_model = load_models()
    _ = forecaster_model, classifier_model
    results = load_results()
    df = load_data()

    st.markdown('<div class="hero-title">Epidemic Spread Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Track C — COVID-19 Spread Forecasting</div>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="warning-banner">&#9888; Model trained on historical epidemic time-series data. For research use only.</div>',
        unsafe_allow_html=True,
    )

    forecaster_metrics = results.get("forecaster", {})
    classifier_metrics = results.get("classifier", {})

    m1, m2, m3, m4 = st.columns(4, gap="small")
    with m1:
        render_metric_card("&#128200;", "Forecaster R²", f"{forecaster_metrics.get('r2', 'N/A')}", "Temporal holdout evaluation")
    with m2:
        render_metric_card("&#128202;", "MAE", f"{forecaster_metrics.get('mae', 'N/A')} cases", "Average absolute error")
    with m3:
        render_metric_card("&#9889;", "RMSE", f"{forecaster_metrics.get('rmse', 'N/A')} cases", "Root mean squared error")
    with m4:
        render_metric_card("&#129514;", "Classifier Accuracy", f"{classifier_metrics.get('accuracy', 'N/A')}", "Risk label performance")

    tab1, tab2, tab3 = st.tabs(["Case Forecast", "Risk Map", "Model Performance"])

    with tab1:
        st.markdown('<div class="section-title">Country-level Case Forecast</div>', unsafe_allow_html=True)
        if df is not None and not df.empty:
            countries = sorted(df["Country/Region"].dropna().unique().tolist())
            country = st.selectbox(
                "Select country",
                countries,
                index=countries.index("India") if "India" in countries else 0,
            )
            cdf = df[df["Country/Region"] == country].sort_values("date")

            bubble_df = cdf.copy()
            bubble_df["risk_level"] = bubble_df["risk_label"].astype("object").where(
                bubble_df["risk_label"].notna(), "Unknown"
            )
            bubble_df["bubble_size"] = bubble_df["rolling_7d"].fillna(0).clip(lower=0)
            color_map = {"Low": "#006600", "Medium": "#b36b00", "High": "#cc0000", "Unknown": "#666666"}

            fig = px.scatter(
                bubble_df,
                x="date",
                y="daily_new",
                size="bubble_size",
                color="risk_level",
                color_discrete_map=color_map,
                hover_data={
                    "Country/Region": True,
                    "date": True,
                    "daily_new": ":,.0f",
                    "rolling_7d": ":,.2f",
                    "risk_level": True,
                    "bubble_size": False,
                },
                labels={
                    "date": "Date",
                    "daily_new": "Daily new cases",
                    "rolling_7d": "7-day average cases",
                    "risk_level": "Risk level",
                    "Country/Region": "Country",
                },
                template="plotly_white",
            )
            fig.update_traces(
                marker=dict(line=dict(color="#000000", width=0.8), opacity=0.75),
                hovertemplate=(
                    "Country: %{customdata[0]}<br>"
                    "Date: %{x|%d %b %Y}<br>"
                    "Daily new cases: %{y:,.0f}<br>"
                    "7-day average: %{customdata[2]:,.2f}<br>"
                    "Risk level: %{customdata[3]}<extra></extra>"
                ),
            )
            fig.update_layout(
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                font=dict(color=TEXT),
                xaxis=dict(title="Date", color=TEXT, gridcolor="#d9d9d9"),
                yaxis=dict(title="Daily new cases", color=TEXT, gridcolor="#d9d9d9"),
                legend=dict(
                    title="Risk level",
                    font=dict(color=TEXT),
                    bgcolor="#ffffff",
                    bordercolor="#000000",
                    borderwidth=1,
                ),
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

            st.markdown('<div class="section-title">Risk level over time</div>', unsafe_allow_html=True)
            risk_counts = cdf["risk_label"].value_counts()
            c1, c2, c3 = st.columns(3)
            for col, label in zip([c1, c2, c3], ["Low", "Medium", "High"]):
                count = risk_counts.get(label, 0)
                pct = round(100 * count / len(cdf), 1)
                col.metric(label, f"{pct}% of days")
        else:
            st.info("No Track C data available.")

    with tab2:
        st.markdown('<div class="section-title">Global Risk Snapshot</div>', unsafe_allow_html=True)
        if df is not None and not df.empty:
            latest = df.groupby("Country/Region").last().reset_index()
            risk_df = latest[["Country/Region", "rolling_7d", "risk_label", "confirmed"]].copy()
            risk_df = risk_df.sort_values("rolling_7d", ascending=False).head(40)
            risk_df["log_confirmed"] = risk_df["confirmed"].apply(lambda x: max(x, 1))
            risk_df["bubble_size"] = risk_df["rolling_7d"].clip(lower=0.1)

            color_map = {"Low": "#006600", "Medium": "#b36b00", "High": "#cc0000"}
            risk_df["color"] = risk_df["risk_label"].map(lambda r: color_map.get(str(r), "#666666"))

            fig2 = go.Figure()
            for risk_level, color in color_map.items():
                subset = risk_df[risk_df["risk_label"] == risk_level]
                if subset.empty:
                    continue
                fig2.add_trace(
                    go.Scatter(
                        x=subset["log_confirmed"],
                        y=subset["rolling_7d"],
                        mode="markers+text",
                        name=risk_level,
                        marker=dict(
                            size=subset["bubble_size"],
                            sizemode="area",
                            sizeref=2.0 * risk_df["bubble_size"].max() / (40.0 ** 2),
                            sizemin=6,
                            color=color,
                            opacity=0.8,
                            line=dict(color="#000000", width=1.5),
                        ),
                        text=subset["Country/Region"],
                        textposition="top center",
                        textfont=dict(size=9, color=TEXT),
                        customdata=subset[["Country/Region", "confirmed", "rolling_7d", "risk_label"]].values,
                        hovertemplate=(
                            "<b>%{customdata[0]}</b><br>"
                            "Total Cases: %{customdata[1]:,.0f}<br>"
                            "7-day Avg: %{customdata[2]:,.0f}<br>"
                            "Risk: %{customdata[3]}"
                            "<extra></extra>"
                        ),
                    )
                )

            fig2.update_layout(
                xaxis=dict(
                    title="Total Confirmed Cases (log scale)",
                    type="log",
                    color=TEXT,
                    gridcolor="#d9d9d9",
                    showgrid=True,
                    linecolor="#000000",
                    tickcolor="#000000",
                    tickfont=dict(color=TEXT),
                ),
                yaxis=dict(
                    title="7-day Avg Daily New Cases",
                    color=TEXT,
                    gridcolor="#d9d9d9",
                    showgrid=True,
                    linecolor="#000000",
                    tickcolor="#000000",
                    tickfont=dict(color=TEXT),
                ),
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                legend=dict(
                    title="Risk Level",
                    font=dict(color=TEXT),
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#000000",
                    borderwidth=1,
                ),
                margin=dict(l=60, r=40, t=60, b=60),
                height=550,
                title=dict(
                    text="Top 40 Countries — Confirmed Cases vs Current Spread",
                    font=dict(color=TEXT, size=16),
                ),
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No Track C data available.")

    with tab3:
        st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)
        forecaster_radar = make_radar_chart(
            "Forecaster Radar",
            ["R²", "MAE Quality", "RMSE Quality"],
            [
                float(forecaster_metrics.get("r2", 0.0)),
                bounded_score(forecaster_metrics.get("mae", 0.0), 500.0),
                bounded_score(forecaster_metrics.get("rmse", 0.0), 2000.0),
            ],
            ACCENT,
            "Lower error is better. MAE/RMSE are shown as inverse quality scores for visual comparison.",
        )
        classifier_radar = make_radar_chart(
            "Classifier Radar",
            ["Accuracy", "Macro F1", "Class Coverage"],
            [
                float(classifier_metrics.get("accuracy", 0.0)),
                float(classifier_metrics.get("macro_f1", 0.0)),
                min(len(classifier_metrics.get("classes", [])) / 3.0, 1.0),
            ],
            "#cc0000",
            "Class coverage equals observed risk classes divided by the expected three-label set.",
        )
        r1, r2 = st.columns(2, gap="large")
        with r1:
            st.plotly_chart(forecaster_radar, use_container_width=True)
        with r2:
            st.plotly_chart(classifier_radar, use_container_width=True)

        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown('<div class="section-title">Forecaster</div>', unsafe_allow_html=True)
            st.metric("MAE", f"{forecaster_metrics.get('mae', 'N/A')} cases")
            st.metric("RMSE", f"{forecaster_metrics.get('rmse', 'N/A')} cases")
            st.metric("R²", forecaster_metrics.get("r2", "N/A"))
            st.markdown(
                f'<div class="small-note">Train rows: {forecaster_metrics.get("n_train", "N/A")} | Test rows: {forecaster_metrics.get("n_test", "N/A")}</div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown('<div class="section-title">Risk Classifier</div>', unsafe_allow_html=True)
            st.metric("Accuracy", classifier_metrics.get("accuracy", "N/A"))
            st.metric("Macro F1", classifier_metrics.get("macro_f1", "N/A"))
            st.metric("Classes", ", ".join(classifier_metrics.get("classes", [])))
            st.markdown(
                f'<div class="small-note">Train rows: {classifier_metrics.get("n_train", "N/A")} | Test rows: {classifier_metrics.get("n_test", "N/A")}</div>',
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
