"""
alerts.py — Watchlist persistence and poll-on-load alert detection.

Architecture:
- ``watchlist.json`` stores user-defined vaccines/diseases to monitor.
- ``last_seen.json`` stores a snapshot of trial counts and publication IDs
  from the last session.
- On each app load, ``check_watchlist_updates()`` compares current data
  to the snapshot and surfaces "new" items as alerts.
- No backend required — fully local and free.
"""

import os
import json
from datetime import datetime
from modules.config import WATCHLIST_FILE, LAST_SEEN_FILE


# ══════════════════════════════════════════════════════════════
#  WATCHLIST CRUD
# ══════════════════════════════════════════════════════════════

def _load_watchlist() -> list:
    if os.path.exists(WATCHLIST_FILE):
        try:
            with open(WATCHLIST_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def _save_watchlist(watchlist: list):
    try:
        with open(WATCHLIST_FILE, "w") as f:
            json.dump(watchlist, f, indent=2)
    except Exception:
        pass


def add_to_watchlist(name: str, item_type: str = "vaccine"):
    """Add an item to the watchlist. ``item_type`` is 'vaccine' or 'disease'."""
    wl = _load_watchlist()
    # Prevent duplicates
    for item in wl:
        if item.get("name", "").lower() == name.lower():
            return wl
    wl.append({
        "name": name,
        "type": item_type,
        "added": datetime.now().strftime("%Y-%m-%d %H:%M"),
    })
    _save_watchlist(wl)
    return wl


def remove_from_watchlist(name: str):
    """Remove an item by name."""
    wl = _load_watchlist()
    wl = [item for item in wl if item.get("name", "").lower() != name.lower()]
    _save_watchlist(wl)
    return wl


# ══════════════════════════════════════════════════════════════
#  LAST-SEEN SNAPSHOT
# ══════════════════════════════════════════════════════════════

def _load_last_seen() -> dict:
    if os.path.exists(LAST_SEEN_FILE):
        try:
            with open(LAST_SEEN_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_last_seen(snapshot: dict):
    try:
        with open(LAST_SEEN_FILE, "w") as f:
            json.dump(snapshot, f, indent=2)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════
#  POLL & ALERT DETECTION
# ══════════════════════════════════════════════════════════════

def check_watchlist_updates(watchlist: list) -> list:
    """
    For each watchlist item, quickly fetch current trial counts and recent
    publications, then compare to ``last_seen.json``.

    Returns a list of alert dicts:
    ``{name, type, new_trials, new_pubs, pub_titles, timestamp}``
    """
    # Import here to avoid circular imports
    from modules.trial_fetchers import fetch_all_vaccine_trials, fetch_pipeline_publications

    if not watchlist:
        return []

    last_seen = _load_last_seen()
    alerts = []
    new_snapshot = {}

    for item in watchlist:
        name = item.get("name", "")
        itype = item.get("type", "vaccine")
        if not name:
            continue

        key = name.lower()
        prev = last_seen.get(key, {})
        prev_trial_count = prev.get("trial_count", 0)
        prev_pub_ids = set(prev.get("pub_ids", []))

        # Fetch current data (only page 1 for speed)
        try:
            trials = fetch_all_vaccine_trials(name, max_pages=1) if itype == "disease" else []
            if itype == "vaccine":
                # For vaccines we do a lighter check — same call path but limited
                trials = fetch_all_vaccine_trials(name, max_pages=1)
        except Exception:
            trials = []

        try:
            pubs = fetch_pipeline_publications(name, max_items=5)
        except Exception:
            pubs = []

        current_trial_count = len(trials)
        current_pub_ids = set()
        current_pub_titles = []
        for p in pubs:
            pid = p.get("link", "")
            current_pub_ids.add(pid)
            if pid not in prev_pub_ids:
                current_pub_titles.append(p.get("title", ""))

        new_trials = max(0, current_trial_count - prev_trial_count) if prev_trial_count > 0 else 0
        new_pubs = len(current_pub_ids - prev_pub_ids) if prev_pub_ids else 0

        # Save snapshot
        new_snapshot[key] = {
            "trial_count": current_trial_count,
            "pub_ids": list(current_pub_ids),
            "checked": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }

        # Only alert if there's something new (and we have a previous baseline)
        if (new_trials > 0 or new_pubs > 0) and prev:
            alerts.append({
                "name": name,
                "type": itype,
                "new_trials": new_trials,
                "new_pubs": new_pubs,
                "pub_titles": current_pub_titles[:3],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            })

    _save_last_seen(new_snapshot)
    return alerts
