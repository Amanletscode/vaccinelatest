"""
analyst_workspace.py — Interactive analyst tools.

Contains the threat-score heuristic, analyst notes persistence,
and the Streamlit ``data_editor``-based interactive table wrapper.
"""

import os
import json
import pandas as pd
import streamlit as st
from modules.config import ANALYST_NOTES_FILE
from modules.utils import _supports_width_string


# ════════════════════════════════════════════════════════════════
#  THREAT SCORE
# ════════════════════════════════════════════════════════════════

def calculate_threat_score(t: dict) -> int:
    """
    Heuristic competitive threat score (0-100).

    Components:
    - Phase weight (Phase 3 = 40, Phase 2 = 20, Phase 1 = 10)
    - Status weight (Completed = 20, Recruiting/Active = 15)
    - Sponsor recognition (+25 if major-pharma name detected)
    - Combination vaccine boost (+10 if multi-antigen product)
    """
    score = 0

    phase = str(t.get("Phase", "")).lower()
    if "3" in phase:
        score += 40
    elif "2" in phase:
        score += 20
    elif "1" in phase:
        score += 10

    status = str(t.get("Status", "")).lower()
    if "recruiting" in status or "active" in status:
        score += 15
    elif "completed" in status:
        score += 20

    sponsor = str(t.get("Sponsor", "")).lower()
    major_sponsors = [
        "pfizer", "gsk", "sanofi", "merck", "moderna",
        "astrazeneca", "jnj", "johnson", "biontech", "novartis",
    ]
    if any(m in sponsor for m in major_sponsors):
        score += 25

    vaccines = str(t.get("Vaccines", ""))
    if "+" in vaccines or "/" in vaccines or len([v for v in vaccines.split(",") if v.strip()]) > 1:
        score += 10

    return min(score, 100)


# ════════════════════════════════════════════════════════════════
#  ANALYST NOTES PERSISTENCE
# ════════════════════════════════════════════════════════════════

def _load_analyst_notes():
    if os.path.exists(ANALYST_NOTES_FILE):
        try:
            with open(ANALYST_NOTES_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_analyst_notes(notes):
    try:
        with open(ANALYST_NOTES_FILE, "w") as f:
            json.dump(notes, f, indent=2)
    except Exception:
        pass


# ════════════════════════════════════════════════════════════════
#  INTERACTIVE TABLE
# ════════════════════════════════════════════════════════════════

def show_interactive_df(df: pd.DataFrame, key: str, height: int = 420):
    """Render an interactive workspace table with persistable Favorite/Note columns."""
    notes = _load_analyst_notes()
    safe_df = df.copy()

    if "Locations" in safe_df.columns:
        safe_df = safe_df.drop(columns=["Locations"])

    safe_df.insert(0, "Favorite", safe_df["NCT ID"].apply(
        lambda x: notes.get(str(x), {}).get("Favorite", False)))
    safe_df.insert(1, "Analyst_Note", safe_df["NCT ID"].apply(
        lambda x: notes.get(str(x), {}).get("Note", "")))

    for col in safe_df.columns:
        if col not in ["Favorite", "Analyst_Note", "Threat Score"]:
            safe_df[col] = safe_df[col].apply(lambda x: "" if pd.isna(x) else str(x))

    disabled_cols = [c for c in safe_df.columns if c not in ["Favorite", "Analyst_Note"]]

    try:
        if _supports_width_string():
            edited_df = st.data_editor(safe_df, width="stretch", height=height,
                                       key=f"editor_{key}", disabled=disabled_cols)
        else:
            edited_df = st.data_editor(safe_df, use_container_width=True, height=height,
                                       key=f"editor_{key}", disabled=disabled_cols)

        changed = False
        for _, row in edited_df.iterrows():
            nid = str(row["NCT ID"])
            fav = bool(row.get("Favorite", False))
            note = str(row.get("Analyst_Note", "")).strip()
            existing = notes.get(nid, {})
            if existing.get("Favorite", False) != fav or existing.get("Note", "") != note:
                notes[nid] = {"Favorite": fav, "Note": note}
                changed = True
        if changed:
            _save_analyst_notes(notes)

    except Exception as e:
        st.warning(f"Interactive table failed: {e}. Showing static table instead.")
        st.table(safe_df.head(50))