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
