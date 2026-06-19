"""
utils.py — Shared low-level utilities.

Functions here are dependency-free (no Streamlit, no domain logic).
They are consumed by almost every other module.
"""

import re
import time
import requests
from packaging.version import Version
import streamlit as st


def _norm_txt(s: str) -> str:
    """Normalize text: lowercase, keep only alphanumeric chars, collapse whitespace."""
    if not s:
        return ""
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _request_json(url, params=None, session=None, retries=3, timeout=30, headers=None):
    """GET JSON with basic retry + HTTP 429 (rate-limit) back-off."""
    s = session or requests.Session()
    for attempt in range(retries):
        try:
            resp = s.get(url, params=params, headers=headers, timeout=timeout)
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", "1"))
                time.sleep(wait + 0.5)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            if attempt < retries - 1:
                time.sleep(1.2 * (2 ** attempt))
                continue
            raise


def _supports_width_string() -> bool:
    """True if st.dataframe supports width='stretch' (Streamlit >= 1.39)."""
    try:
        return Version(st.__version__) >= Version("1.39.0")
    except Exception:
        return False


from datetime import datetime


def calculate_trial_risk_index(status: str, expected_completion: str) -> dict:
    """
    Enterprise-grade delay calculator handling messy CT.gov date formats.
    Returns a risk dictionary with a severity flag and mathematical delay.
    """
    if not isinstance(status, str):
        status = "Unknown"
    if not isinstance(expected_completion, str):
        expected_completion = ""

    # If it's already done or terminated, no active delay risk
    if status.lower() in ['completed', 'terminated', 'withdrawn', 'suspended', 'unknown status']:
        return {"risk_level": "Low", "flag": "⚪ Inactive/Resolved", "months_delayed": 0}

    if not expected_completion or expected_completion in ["Not reported", "N/A"]:
        return {"risk_level": "Unknown", "flag": "⚪ No Date Set", "months_delayed": 0}

    try:
        expected_clean = expected_completion.strip()
        # Handle "Month YYYY" e.g., "May 2026"
        if re.match(r'^[A-Za-z]+\s\d{4}$', expected_clean):
            target_date = datetime.strptime(expected_clean, "%B %Y")
        # Handle "YYYY-MM-DD" e.g., "2026-05-15"
        elif re.match(r'^\d{4}-\d{2}-\d{2}$', expected_clean):
            target_date = datetime.strptime(expected_clean, "%Y-%m-%d")
        # Handle "YYYY-MM" e.g., "2026-05"
        elif re.match(r'^\d{4}-\d{2}$', expected_clean):
            target_date = datetime.strptime(expected_clean, "%Y-%m")
        else:
            return {"risk_level": "Unknown", "flag": "⚪ Parse Error", "months_delayed": 0}

        current_date = datetime.now()
        delta_days = (current_date - target_date).days
        months_delayed = delta_days / 30.44

        # Enterprise Threshold Logic
        active_statuses = ['recruiting', 'active, not recruiting', 'not yet recruiting', 'enrolling by invitation']
        if months_delayed > 6 and status.lower() in active_statuses:
            return {"risk_level": "Critical", "flag": "🔴 Severe Delay (>6m)", "months_delayed": round(months_delayed, 1)}
        elif months_delayed > 0 and status.lower() in active_statuses:
            return {"risk_level": "Moderate", "flag": "🟠 Slipping Timeline", "months_delayed": round(months_delayed, 1)}
        else:
            return {"risk_level": "Low", "flag": "🟢 On Track", "months_delayed": 0}

    except Exception:
        return {"risk_level": "Unknown", "flag": "⚪ Parse Error", "months_delayed": 0}


def calculate_us_site_saturation(trials: list) -> dict:
    """
    Analyzes US clinical trial footprints down to the City and Hospital level.
    Uses dynamic data-driven logic to find proven but low-competition cities.
    """
    if not trials:
        return {"error": "No trials provided for geographic analysis."}

    city_data = {}
    hospital_data = {}

    active_statuses = ["recruiting", "not yet recruiting", "active, not recruiting", "enrolling by invitation"]

    for trial in trials:
        status = str(trial.get("Status", "")).lower()
        nct_id = str(trial.get("NCT ID", "Unknown"))
        if status not in active_statuses:
            continue

        locations = trial.get("Locations", [])
        if isinstance(locations, list):
            for loc in locations:
                country = str(loc.get("country", "")).strip().lower()
                if country in ["united states", "united states of america", "us", "usa"]:

                    # Track Cities dynamically
                    city = str(loc.get("city", "")).strip().title()
                    if city:
                        if city not in city_data:
                            city_data[city] = 0
                        city_data[city] += 1

                    # Track Hospitals + NCT IDs
                    facility = str(loc.get("name", "")).strip()
                    if facility and "investigational site" not in facility.lower():
                        facility_clean = facility.replace(",", "").replace(".", "").title()
                        if facility_clean not in hospital_data:
                            hospital_data[facility_clean] = {"count": 0, "competing_trials": set()}
                        hospital_data[facility_clean]["count"] += 1
                        hospital_data[facility_clean]["competing_trials"].add(nct_id)

    if not city_data:
        return {"error": "No active US location data available in these trials."}

    # DYNAMIC WHITESPACE LOGIC:
    # Cities that appear in the data (proven infrastructure) but only have exactly 1 active trial (low competition).
    low_competition_cities = [city for city, count in city_data.items() if count == 1]

    # Get top 10 congested hospitals and format for the LLM
    top_hospitals_raw = sorted(hospital_data.items(), key=lambda x: x[1]["count"], reverse=True)[:10]
    top_hospitals_formatted = {
        name: f"{data['count']} active trials (Includes {', '.join(list(data['competing_trials']))})"
        for name, data in top_hospitals_raw if data['count'] > 1
    }

    return {
        "Disease_Target": "Supplied by user",
        "Highly_Saturated_US_Cities": dict(sorted(city_data.items(), key=lambda x: x[1], reverse=True)[:10]),
        "Most_Congested_Hospitals": top_hospitals_formatted,
        "Low_Competition_Emerging_Hubs": low_competition_cities[:15],
    }