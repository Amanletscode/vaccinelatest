"""
visualizations.py — Plotly chart generators.

Each function accepts a pandas DataFrame of trials and returns a plotly
Figure (or None if plotly is not installed / data is empty).
"""

import re
import pandas as pd


# ══════════════════════════════════════════════════════════════
#  CORE CHARTS
# ══════════════════════════════════════════════════════════════

def create_phase_chart(df: pd.DataFrame):
    """Bar chart of phase distribution."""
    try:
        import plotly.express as px
        phase_counts = {}
        for phases in df["Phase"].dropna():
            for phase in str(phases).split(","):
                phase = phase.strip()
                phase_counts[phase] = phase_counts.get(phase, 0) + 1
        if not phase_counts:
            return None
        phase_df = pd.DataFrame(list(phase_counts.items()), columns=["Phase", "Count"])
        fig = px.bar(phase_df, x="Phase", y="Count", title="Phase Distribution",
                     color="Count", color_continuous_scale="Blues")
        fig.update_layout(showlegend=False, height=300)
        return fig
    except ImportError:
        return None


def create_status_chart(df: pd.DataFrame):
    """Pie chart of trial status distribution."""
    try:
        import plotly.express as px
        status_counts = df["Status"].value_counts()
        if status_counts.empty:
            return None
        fig = px.pie(values=status_counts.values, names=status_counts.index,
                     title="Trial Status Distribution")
        fig.update_layout(height=300)
        return fig
    except ImportError:
        return None


def create_sponsor_chart(df: pd.DataFrame, top_n=10):
    """Horizontal bar chart of top sponsors by trial count."""
    try:
        import plotly.express as px
        sponsor_counts = df["Sponsor"].value_counts().head(top_n)
        if sponsor_counts.empty:
            return None
        fig = px.bar(
            x=sponsor_counts.values, y=sponsor_counts.index,
            orientation="h", title=f"Top {top_n} Sponsors",
            labels={"x": "Number of Trials", "y": "Sponsor"},
        )
        fig.update_layout(height=400)
        fig.update_xaxes(dtick=1, rangemode="tozero")
        return fig
    except ImportError:
        return None


def create_country_heatmap(df: pd.DataFrame):
    """World choropleth map of trial site density."""
    try:
        import plotly.express as px
        if "Locations" not in df.columns:
            return None
        country_counts = {}
        for locs in df["Locations"].dropna():
            if not isinstance(locs, list):
                continue
            for loc in locs:
                if isinstance(loc, dict) and loc.get("country"):
                    c = loc["country"]
                    country_counts[c] = country_counts.get(c, 0) + 1
        if not country_counts:
            return None
        map_df = pd.DataFrame(list(country_counts.items()), columns=["Country", "Sites"])
        fig = px.choropleth(
            map_df, locations="Country", locationmode="country names",
            color="Sites", hover_name="Country",
            color_continuous_scale="Reds", title="Global Trial Footprint",
        )
        fig.update_layout(height=400, margin={"r": 0, "t": 40, "l": 0, "b": 0})
        return fig
    except ImportError:
        return None


# ══════════════════════════════════════════════════════════════
#  TREND ANALYSIS CHARTS
# ══════════════════════════════════════════════════════════════

def _parse_year(date_str):
    """Extract 4-digit year from a date string like '2024-03-15' or 'March 2023'."""
    if not date_str or not isinstance(date_str, str):
        return None
    m = re.search(r"(20\d{2}|19\d{2})", date_str)
    return int(m.group(1)) if m else None


def create_trial_timeline(df: pd.DataFrame):
    """Grouped bar chart: trials started per year, coloured by phase."""
    try:
        import plotly.express as px
        if "Start Date" not in df.columns:
            return None
        rows = []
        for _, row in df.iterrows():
            year = _parse_year(str(row.get("Start Date", "")))
            if year:
                for phase in str(row.get("Phase", "Not reported")).split(","):
                    rows.append({"Year": year, "Phase": phase.strip()})
        if not rows:
            return None
        timeline_df = pd.DataFrame(rows)
        agg = timeline_df.groupby(["Year", "Phase"]).size().reset_index(name="Trials")
        fig = px.bar(
            agg, x="Year", y="Trials", color="Phase",
            title="Trial Starts by Year & Phase",
            barmode="group",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(height=350, xaxis_dtick=1)
        fig.update_xaxes(type="category")
        return fig
    except ImportError:
        return None


def create_status_trend(df: pd.DataFrame):
    """Stacked area chart: trials started per year, coloured by current status."""
    try:
        import plotly.express as px
        if "Start Date" not in df.columns:
            return None
        rows = []
        for _, row in df.iterrows():
            year = _parse_year(str(row.get("Start Date", "")))
            if year:
                rows.append({"Year": year, "Status": str(row.get("Status", "Unknown"))})
        if not rows:
            return None
        trend_df = pd.DataFrame(rows)
        agg = trend_df.groupby(["Year", "Status"]).size().reset_index(name="Trials")
        fig = px.area(
            agg, x="Year", y="Trials", color="Status",
            title="Trial Activity Over Time (by Status)",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig.update_layout(height=350, xaxis_dtick=1)
        fig.update_xaxes(type="category")
        return fig
    except ImportError:
        return None


def create_completion_timeline(df: pd.DataFrame):
    """Bar chart of trial completions per year."""
    try:
        import plotly.express as px
        if "Completion Date" not in df.columns:
            return None
        years = []
        for val in df["Completion Date"].dropna():
            y = _parse_year(str(val))
            if y:
                years.append(y)
        if not years:
            return None
        yr_df = pd.DataFrame({"Year": years})
        agg = yr_df["Year"].value_counts().sort_index().reset_index()
        agg.columns = ["Year", "Completed Trials"]
        fig = px.bar(
            agg, x="Year", y="Completed Trials",
            title="Trial Completions by Year",
            color_discrete_sequence=["#2ca02c"],
        )
        fig.update_layout(height=300, xaxis_dtick=1)
        fig.update_xaxes(type="category")
        return fig
    except ImportError:
        return None


# ══════════════════════════════════════════════════════════════
#  HEAD-TO-HEAD COMPARISON (simplified, no heuristic scores)
# ══════════════════════════════════════════════════════════════

def compute_comparison_metrics(trials: list, label: str, known_manufacturer: str = None):
    """Compute key metrics for a set of trials belonging to one vaccine.
    
    ``known_manufacturer`` (optional) is used to count how many trials are
    sponsored by the originator vs external parties.
    """
    total = len(trials)
    if total == 0:
        return {
            "label": label, "total_trials": 0, "phase3_count": 0,
            "completed_count": 0, "recruiting_count": 0,
            "country_count": 0, "top_sponsor": "N/A",
            "phase3_pct": 0, "completed_pct": 0, "recruiting_pct": 0,
            "manufacturer": known_manufacturer or "Unknown",
        }

    phase3 = sum(1 for t in trials if "3" in str(t.get("Phase", "")))
    completed = sum(1 for t in trials if "completed" in str(t.get("Status", "")).lower())
    recruiting = sum(1 for t in trials if "recruiting" in str(t.get("Status", "")).lower())

    countries = set()
    for t in trials:
        locs = t.get("Locations", [])
        if isinstance(locs, list):
            for loc in locs:
                if isinstance(loc, dict) and loc.get("country"):
                    countries.add(loc["country"])

    sponsor_counts = {}
    for t in trials:
        sp = t.get("Sponsor", "Unknown")
        sponsor_counts[sp] = sponsor_counts.get(sp, 0) + 1
    top_sponsor = max(sponsor_counts, key=sponsor_counts.get) if sponsor_counts else "N/A"

    return {
        "label": label,
        "total_trials": total,
        "phase3_count": phase3,
        "completed_count": completed,
        "recruiting_count": recruiting,
        "country_count": len(countries),
        "top_sponsor": top_sponsor,
        "phase3_pct": round(100 * phase3 / total, 1) if total else 0,
        "completed_pct": round(100 * completed / total, 1) if total else 0,
        "recruiting_pct": round(100 * recruiting / total, 1) if total else 0,
        "manufacturer": known_manufacturer or "Unknown",
    }


def create_comparison_table(metrics_a: dict, metrics_b: dict):
    """Side-by-side comparison DataFrame for display in Streamlit."""
    rows = [
        ("Manufacturer", metrics_a.get("manufacturer", "Unknown"), metrics_b.get("manufacturer", "Unknown")),
        ("Total Trials", metrics_a["total_trials"], metrics_b["total_trials"]),
        ("Phase 3 Trials", metrics_a["phase3_count"], metrics_b["phase3_count"]),
        ("Phase 3 %", f"{metrics_a['phase3_pct']}%", f"{metrics_b['phase3_pct']}%"),
        ("Completed Trials", metrics_a["completed_count"], metrics_b["completed_count"]),
        ("Recruiting Trials", metrics_a["recruiting_count"], metrics_b["recruiting_count"]),
        ("Countries w/ Sites", metrics_a["country_count"], metrics_b["country_count"]),
        ("Most Frequent Sponsor", metrics_a["top_sponsor"], metrics_b["top_sponsor"]),
    ]
    return pd.DataFrame(rows, columns=["Metric", metrics_a["label"], metrics_b["label"]])


def create_comparison_bar(metrics_a: dict, metrics_b: dict):
    """Simple grouped bar chart comparing two vaccines on key counts."""
    try:
        import plotly.graph_objects as go

        categories = ["Total Trials", "Phase 3", "Completed", "Recruiting", "Countries"]
        vals_a = [
            metrics_a["total_trials"], metrics_a["phase3_count"],
            metrics_a["completed_count"], metrics_a["recruiting_count"],
            metrics_a["country_count"],
        ]
        vals_b = [
            metrics_b["total_trials"], metrics_b["phase3_count"],
            metrics_b["completed_count"], metrics_b["recruiting_count"],
            metrics_b["country_count"],
        ]

        fig = go.Figure(data=[
            go.Bar(name=metrics_a["label"], x=categories, y=vals_a, marker_color="#636EFA"),
            go.Bar(name=metrics_b["label"], x=categories, y=vals_b, marker_color="#EF553B"),
        ])
        fig.update_layout(
            barmode="group", height=350,
            title="Side-by-Side Comparison",
            yaxis_title="Count",
        )
        return fig
    except ImportError:
        return None
