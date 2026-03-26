"""
vaccine_data.py — Curated vaccine synonym tables, manufacturer indices, and matching logic.

This module owns the domain knowledge that maps free-text intervention names
from ClinicalTrials.gov to canonical brand names, code names, INNs, and
originating manufacturers.  It also provides:

- ``_is_vaccine_study()`` — heuristic classifier that decides whether a study
  record is actually a vaccine trial.
- ``_extract_vaccine_names()`` — pull all intervention / arm-group names that
  look like vaccine products.
- ``_matches_vaccine_name()`` — check if a target vaccine appears in a list of
  intervention names.
"""

import os
import json
import streamlit as st
from modules.utils import _norm_txt, _request_json
from modules.config import LEARNED_MFR_FILE
from modules.data_extraction import (
    _mesh_terms_intervention,
    _primary_outcomes_from_protocol,
)


# ════════════════════════════════════════════════════════════════
#  CURATED SYNONYM GROUPS
# ════════════════════════════════════════════════════════════════
# Each inner list contains names that all refer to the SAME vaccine product.
# Order does not matter.  Add new groups as the platform covers more products.

_VACCINE_SYNONYM_GROUPS = [
    # Pfizer / BioNTech – COVID-19
    ["Comirnaty", "BNT162b2", "Tozinameran",
     "Pfizer-BioNTech COVID-19 vaccine", "Pfizer-BioNTech mRNA COVID-19 vaccine"],
    # Moderna – COVID-19
    ["Spikevax", "mRNA-1273", "mRNA 1273", "Elasomeran", "Moderna COVID-19 vaccine"],
    # Pfizer – RSV
    ["Abrysvo", "RSVpreF", "RSVpreF3", "bivalent RSVpreF3", "Pfizer RSVpreF vaccine"],
    # GSK – RSV
    ["Arexvy", "respiratory syncytial virus vaccine recombinant adjuvanted", "GSK RSV vaccine"],
    # Pfizer – pneumococcal
    ["Prevnar 13", "Prevenar 13", "13-valent pneumococcal conjugate vaccine", "PCV13"],
    ["Prevnar 20", "20-valent pneumococcal conjugate vaccine", "PCV20"],
]

_VACCINE_MANUFACTURER_DATA = {
    "Pfizer": [
        "Comirnaty", "BNT162b2", "Tozinameran",
        "Abrysvo", "RSVpreF", "RSVpreF3",
        "Prevnar 13", "Prevenar 13", "13-valent pneumococcal conjugate vaccine", "PCV13",
        "Prevnar 20", "20-valent pneumococcal conjugate vaccine", "PCV20",
    ],
    "BioNTech": ["Comirnaty", "BNT162b2", "Tozinameran"],
    "Moderna": ["Spikevax", "mRNA-1273", "mRNA 1273", "Elasomeran", "Moderna COVID-19 vaccine"],
    "GSK": [
        "Arexvy", "Bexsero", "Menveo",
        "respiratory syncytial virus vaccine recombinant adjuvanted", "GSK RSV vaccine",
    ],
}


# ════════════════════════════════════════════════════════════════
#  INDEX BUILDERS  (run once at import time)
# ════════════════════════════════════════════════════════════════

def _build_vaccine_synonym_index():
    idx = {}
    for group in _VACCINE_SYNONYM_GROUPS:
        for name in group:
            norm = _norm_txt(name)
            if not norm:
                continue
            if norm not in idx:
                idx[norm] = set()
            for other in group:
                idx[norm].add(other)
    return {k: sorted(list(v)) for k, v in idx.items()}


def _build_vaccine_manufacturer_index():
    idx = {}
    for mfr, names in _VACCINE_MANUFACTURER_DATA.items():
        for name in names:
            norm = _norm_txt(name)
            if not norm:
                continue
            idx.setdefault(norm, mfr)
    for norm, mfr in _load_learned_manufacturers().items():
        idx.setdefault(norm, mfr)
    return idx


# ── Learned-manufacturer persistence ──

def _load_learned_manufacturers():
    if os.path.exists(LEARNED_MFR_FILE):
        try:
            with open(LEARNED_MFR_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_learned_manufacturer(vaccine_name, mfr):
    learned = _load_learned_manufacturers()
    norm = _norm_txt(vaccine_name)
    if norm:
        learned[norm] = mfr
        try:
            with open(LEARNED_MFR_FILE, "w") as f:
                json.dump(learned, f, indent=2)
        except Exception:
            pass


# Build indices at module load time
_VACCINE_SYNONYM_INDEX = _build_vaccine_synonym_index()
_VACCINE_MANUFACTURER_INDEX = _build_vaccine_manufacturer_index()


# ════════════════════════════════════════════════════════════════
#  PUBLIC HELPERS
# ════════════════════════════════════════════════════════════════

def _get_vaccine_search_terms(raw_name: str) -> list:
    """Return all search terms for a vaccine including curated synonyms."""
    base = (raw_name or "").strip()
    if not base:
        return []
    norm = _norm_txt(base)
    extra = _VACCINE_SYNONYM_INDEX.get(norm, [])
    seen = set()
    terms = []
    for t in [base] + extra:
        t = t.strip()
        if not t:
            continue
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        terms.append(t)
    return terms


def _get_vaccine_manufacturer(raw_name: str, use_llm_fallback: bool = True):
    """Return the primary manufacturer, using hardcoded data → JSON cache → LLM fallback."""
    if not raw_name:
        return None
    norm = _norm_txt(raw_name)

    # 1. Hardcoded index
    if norm in _VACCINE_MANUFACTURER_INDEX:
        return _VACCINE_MANUFACTURER_INDEX[norm]
    # 2. Check synonyms
    syns = _VACCINE_SYNONYM_INDEX.get(norm, [])
    for s in syns:
        s_norm = _norm_txt(s)
        if s_norm in _VACCINE_MANUFACTURER_INDEX:
            return _VACCINE_MANUFACTURER_INDEX[s_norm]
    # 3. Local JSON cache
    learned = _load_learned_manufacturers()
    if norm in learned:
        _VACCINE_MANUFACTURER_INDEX[norm] = learned[norm]
        return learned[norm]
    # 4. LLM fallback (imports llm_helpers lazily to avoid circular imports)
    if use_llm_fallback:
        from modules.llm_helpers import _infer_manufacturer_llm
        st.toast(f"🤖 AI is researching originator for '{raw_name}'...", icon="🔍")
        inferred = _infer_manufacturer_llm(raw_name)
        if inferred and inferred.strip().lower() != "unknown":
            st.toast(f"✅ AI found originator: {inferred}", icon="🧠")
            _save_learned_manufacturer(raw_name, inferred)
            _VACCINE_MANUFACTURER_INDEX[norm] = inferred
            return inferred
        else:
            st.toast("⚠️ AI could not determine the originator.", icon="🤷")
    return None


# ════════════════════════════════════════════════════════════════
#  VACCINE NAME EXTRACTION & MATCHING
# ════════════════════════════════════════════════════════════════

def _extract_vaccine_names(study) -> list:
    """Collect intervention names + otherNames + armGroup interventionNames."""
    proto = (study or {}).get("protocolSection", {}) or {}
    arms = proto.get("armsInterventionsModule", {}) or {}
    names = set()
    for intr in arms.get("interventions", []) or []:
        nm = intr.get("name")
        if nm:
            names.add(nm.strip())
        other = intr.get("otherNames")
        if isinstance(other, list):
            for on in other:
                if on:
                    names.add(on.strip())
    for grp in arms.get("armGroups", []) or []:
        for nm in grp.get("interventionNames", []) or []:
            if nm:
                names.add(nm.strip())
    return sorted(names)


def _matches_vaccine_name(names, target_norms) -> bool:
    """True if the normalised target name matches any intervention name."""
    if isinstance(target_norms, str):
        target_norms = [target_norms]
    target_norms = [t for t in (target_norms or []) if t]
    if not target_norms:
        return False
    for raw in names or []:
        norm = _norm_txt(raw)
        if not norm:
            continue
        tokens = set(norm.split())
        for t in target_norms:
            if norm == t or t in tokens:
                return True
    return False


def _is_vaccine_study(study) -> bool:
    """
    Heuristic classifier — returns True if a study is a vaccine trial.

    Strategy (in order):
    1. Intervention MeSH terms contain 'Vaccine'
    2. Biological intervention with name mentioning 'vaccine'
    3. Primary outcome text contains immunogenicity keywords
    """
    mesh = [m.lower() for m in _mesh_terms_intervention(study)]
    if any("vaccine" in m for m in mesh):
        return True

    proto = (study or {}).get("protocolSection", {}) or {}
    arms = proto.get("armsInterventionsModule", {}) or {}
    intrs = arms.get("interventions", []) or []
    has_bio = any((i.get("type") or "").lower() == "biological" for i in intrs)
    has_vaccine_word = False
    for i in intrs:
        other = i.get("otherNames", [])
        if not isinstance(other, list):
            other = []
        txt = " ".join([_norm_txt(i.get("name", "")), _norm_txt(" ".join(other))])
        if "vaccine" in txt or " vax " in f" {txt} ":
            has_vaccine_word = True
            break
    if has_bio and has_vaccine_word:
        return True

    po = _primary_outcomes_from_protocol(study)
    titles = _norm_txt(" ".join([(x.get("Title") or "") for x in po]))
    if any(k in titles for k in ["immunogenicity", "antibody", "neutralizing", "seroconversion"]):
        return True

    return False
