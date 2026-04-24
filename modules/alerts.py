"""
alerts.py — Watchlist persistence and poll-on-load alert detection.

Architecture:
- ``watchlist.json`` stores user-defined vaccines/diseases to monitor.
- ``last_seen.json`` stores a dictionary of specific NCT IDs and their statuses
  from the last session to accurately detect phase progression.
- On each app load or manual check, ``check_watchlist_updates()`` compares current data
  to the snapshot and surfaces specific milestone alerts.
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
#  POLL & ALERT DETECTION (UPGRADED CAPABILITY)
# ══════════════════════════════════════════════════════════════

def check_watchlist_updates(watchlist: list) -> list:
    """
    For each watchlist item, fetch current trials and publications.
    Compares specific NCT IDs and Statuses to detect actual clinical milestones.
    """
    from modules.trial_fetchers import (
        fetch_all_vaccine_trials,
        fetch_pipeline_publications,
        fetch_vaccine_trials_by_aliases,
    )
    from modules.vaccine_data import _get_vaccine_search_terms

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
        
        # Identify if this is the old basic counter format
        is_legacy_snapshot = "trial_count" in prev and "trials" not in prev
        
        prev_trials_dict = prev.get("trials", {})
        prev_pub_ids = set(prev.get("pub_ids", []))

        # Fetch current data
        try:
            if itype == "disease":
                trials = fetch_all_vaccine_trials(name, max_pages=1)
            else:
                search_terms = _get_vaccine_search_terms(name)
                if search_terms:
                    result = fetch_vaccine_trials_by_aliases(
                        tuple(search_terms), max_pages=1,
                    )
                    trials = result.get("trials", []) if isinstance(result, dict) else []
                else:
                    trials = []
        except Exception:
            trials = []

        try:
            pubs = fetch_pipeline_publications(name, max_items=5)
        except Exception:
            pubs = []

        # Process Current Trials
        current_trials_dict = {}
        new_trial_alerts = []
        status_change_alerts = []
        
        for t in trials:
            nct_id = t.get("NCT ID")
            if not nct_id:
                continue
                
            current_status = t.get("Status", "Unknown")
            current_trials_dict[nct_id] = current_status
            
            # Only generate alerts if we have a valid recent snapshot (skip legacy transition)
            if prev and not is_legacy_snapshot:
                if nct_id not in prev_trials_dict:
                    new_trial_alerts.append({
                        "id": nct_id, 
                        "title": t.get("Title", "Untitled"), 
                        "status": current_status
                    })
                else:
                    old_status = prev_trials_dict[nct_id]
                    if old_status != current_status:
                        status_change_alerts.append({
                            "id": nct_id, 
                            "old": old_status, 
                            "new": current_status
                        })

        # Process Current Publications
        current_pub_ids = set()
        current_pub_titles = []
        for p in pubs:
            pid = p.get("link", "")
            current_pub_ids.add(pid)
            if pid not in prev_pub_ids:
                current_pub_titles.append(p.get("title", ""))

        new_pubs = len(current_pub_ids - prev_pub_ids) if prev_pub_ids else 0

        # Save the new enriched snapshot
        new_snapshot[key] = {
            "trials": current_trials_dict,
            "pub_ids": list(current_pub_ids),
            "checked": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }

        # Only append to master alerts if there is actual news
        if (new_trial_alerts or status_change_alerts or new_pubs > 0) and prev and not is_legacy_snapshot:
            alerts.append({
                "name": name,
                "type": itype,
                "new_trials": new_trial_alerts,
                "status_changes": status_change_alerts,
                "new_pubs": new_pubs,
                "pub_titles": current_pub_titles[:3],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            })

    _save_last_seen(new_snapshot)
    return alerts