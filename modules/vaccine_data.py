"""
vaccine_data.py — Curated vaccine synonym tables, manufacturer indices, and matching logic.

This module owns the domain knowledge that maps free-text intervention names
from ClinicalTrials.gov to canonical brand names, code names, INNs, and
originating manufacturers.

**Ontology layer (v2):**
- ``vaccine_ontology.json`` stores human-verified AND LLM-discovered aliases.
- ``_get_ontology_aliases()`` checks local cache → fuzzy match → LLM fallback.
- Results are cached additively (only new aliases are appended, never replaced).
- All three tabs benefit automatically because they all call
  ``_get_vaccine_search_terms()`` which chains the ontology.

It also provides:
- ``_is_vaccine_study()`` — heuristic classifier for vaccine trials.
- ``_extract_vaccine_names()`` — pull intervention names from a study.
- ``_matches_vaccine_name()`` — check if a target vaccine appears in names.
"""

import os
import json
import difflib
import streamlit as st
from modules.utils import _norm_txt, _request_json
from modules.config import LEARNED_MFR_FILE, VACCINE_ONTOLOGY_FILE
from modules.data_extraction import (
    _mesh_terms_intervention,
    _primary_outcomes_from_protocol,
)


# ════════════════════════════════════════════════════════════════
#  CURATED SYNONYM GROUPS  (hardcoded baseline)
# ════════════════════════════════════════════════════════════════

_VACCINE_SYNONYM_GROUPS = [
    # Pfizer / BioNTech – COVID-19
    ["Comirnaty", "BNT162b2", "Tozinameran",
     "Pfizer-BioNTech COVID-19 vaccine", "Pfizer-BioNTech mRNA COVID-19 vaccine"],
    # Moderna – COVID-19
    ["Spikevax", "mRNA-1273", "mRNA 1273", "Elasomeran", "Moderna COVID-19 vaccine"],
    # Pfizer – RSV
    ["Abrysvo", "RSVpreF", "bivalent RSVpreF", "PF-06928316", "Pfizer RSVpreF vaccine"],
    # GSK – RSV
    ["Arexvy", "RSVPreF3", "GSK3844766A",
     "respiratory syncytial virus vaccine recombinant adjuvanted", "GSK RSV vaccine"],
    # Pfizer – pneumococcal
    ["Prevnar 13", "Prevenar 13", "13-valent pneumococcal conjugate vaccine", "PCV13"],
    ["Prevnar 20", "20-valent pneumococcal conjugate vaccine", "PCV20"],
]

_VACCINE_MANUFACTURER_DATA = {
    "Pfizer": [
        "Comirnaty", "BNT162b2", "Tozinameran",
        "Abrysvo", "RSVpreF", "PF-06928316",
        "Prevnar 13", "Prevenar 13", "13-valent pneumococcal conjugate vaccine", "PCV13",
        "Prevnar 20", "20-valent pneumococcal conjugate vaccine", "PCV20",
    ],
    "BioNTech": ["Comirnaty", "BNT162b2", "Tozinameran"],
    "Moderna": ["Spikevax", "mRNA-1273", "mRNA 1273", "Elasomeran", "Moderna COVID-19 vaccine"],
    "GSK": [
        "Arexvy", "Bexsero", "Menveo", "Shingrix", "RSVPreF3", "GSK3844766A",
        "respiratory syncytial virus vaccine recombinant adjuvanted", "GSK RSV vaccine",
    ],
}


# ════════════════════════════════════════════════════════════════
#  ONTOLOGY  (persistent JSON cache)
# ════════════════════════════════════════════════════════════════

def _load_ontology() -> dict:
    """Load the vaccine ontology JSON file."""
    if os.path.exists(VACCINE_ONTOLOGY_FILE):
        try:
            with open(VACCINE_ONTOLOGY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_ontology_entry(key: str, new_aliases: list):
    """Additively merge aliases into the ontology JSON.

    Only ADDS new names — never replaces or removes existing ones.
    """
    ontology = _load_ontology()
    existing = set(ontology.get(key, []))
    for alias in new_aliases:
        if isinstance(alias, str) and alias.strip():
            existing.add(alias.strip())
    ontology[key] = sorted(existing, key=str.lower)
    try:
        with open(VACCINE_ONTOLOGY_FILE, "w") as f:
            json.dump(ontology, f, indent=2)
    except Exception:
        pass


# ════════════════════════════════════════════════════════════════
#  INDEX BUILDERS
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


def _merge_ontology_into_synonym_index(synonym_index: dict):
    """Merge vaccine_ontology.json entries into the in-memory synonym index.

    Called once at module init so that all ontology aliases are
    instantly available to ``_get_vaccine_search_terms`` without an
    LLM call.
    """
    ontology = _load_ontology()
    for _key, aliases in ontology.items():
        # Build the full merged group from ontology + existing index
        all_names = set(aliases)
        for alias in aliases:
            norm = _norm_txt(alias)
            if norm in synonym_index:
                all_names.update(synonym_index[norm])
        sorted_names = sorted(all_names, key=str.lower)
        # Every alias in the group points to the full merged group
        for alias in sorted_names:
            norm = _norm_txt(alias)
            if norm:
                synonym_index[norm] = sorted_names


def _merge_single_entry_into_synonym_index(key: str, aliases: list, synonym_index: dict):
    """Runtime merge for a single newly-discovered ontology entry."""
    all_names = set(aliases)
    for alias in aliases:
        norm = _norm_txt(alias)
        if norm in synonym_index:
            all_names.update(synonym_index[norm])
    sorted_names = sorted(all_names, key=str.lower)
    for alias in sorted_names:
        norm = _norm_txt(alias)
        if norm:
            synonym_index[norm] = sorted_names


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


# ── Build indices at module load time ──
_VACCINE_SYNONYM_INDEX = _build_vaccine_synonym_index()
_merge_ontology_into_synonym_index(_VACCINE_SYNONYM_INDEX)   # ← NEW: merge ontology
_VACCINE_MANUFACTURER_INDEX = _build_vaccine_manufacturer_index()


# ════════════════════════════════════════════════════════════════
#  ONTOLOGY ALIAS RESOLVER  (cache → fuzzy → LLM)
# ════════════════════════════════════════════════════════════════

def _get_ontology_aliases(raw_name: str) -> list:
    """Resolve a vaccine name to all known aliases.

    Resolution order:
    1. Exact key match in ``vaccine_ontology.json``
    2. Reverse lookup (input is a VALUE in an existing entry)
    3. Fuzzy key match (catches typos — cutoff 0.75)
    4. LLM fallback via Gemini Flash (result cached for future calls)
    """
    norm = _norm_txt(raw_name)
    if not norm:
        return []

    ontology = _load_ontology()

    # 1. Exact key match
    if norm in ontology:
        return list(ontology[norm])

    # 2. Reverse lookup — input appears as a value alias in some entry
    for _key, aliases in ontology.items():
        alias_norms = [_norm_txt(a) for a in aliases]
        if norm in alias_norms:
            return list(aliases)

    # 3. Fuzzy match (catches typos like "abryvo" → "abrysvo")
    all_keys = list(ontology.keys())
    if all_keys:
        close = difflib.get_close_matches(norm, all_keys, n=1, cutoff=0.75)
        if close:
            return list(ontology[close[0]])

    # 4. LLM fallback  (lazy import to avoid circular dependency)
    try:
        from modules.llm_helpers import _infer_vaccine_aliases_llm
        st.toast(f"🤖 AI is researching aliases for '{raw_name}'…", icon="🔍")
        llm_aliases = _infer_vaccine_aliases_llm(raw_name)
        if llm_aliases and len(llm_aliases) > 1:
            _save_ontology_entry(norm, llm_aliases)
            _merge_single_entry_into_synonym_index(norm, llm_aliases, _VACCINE_SYNONYM_INDEX)
            st.toast(f"✅ Found {len(llm_aliases)} aliases for '{raw_name}'", icon="🧠")
            return llm_aliases
    except Exception:
        pass

    return []


# ════════════════════════════════════════════════════════════════
#  PUBLIC HELPERS
# ════════════════════════════════════════════════════════════════

def _get_vaccine_search_terms(raw_name: str) -> list:
    """Return all search terms for a vaccine, including:

    1. Curated synonym index (hardcoded groups)
    2. Ontology aliases (pre-populated JSON + fuzzy + LLM fallback)

    Deduplicates by lower-cased name.  Used by all three tabs.
    """
    base = (raw_name or "").strip()
    if not base:
        return []
    norm = _norm_txt(base)

    # 1. Curated synonyms
    synonyms = list(_VACCINE_SYNONYM_INDEX.get(norm, []))

    # 2. Ontology aliases (includes fuzzy + LLM fallback)
    ontology_aliases = _get_ontology_aliases(base)

    # 3. Deduplicate
    seen = set()
    terms = []
    for t in [base] + synonyms + ontology_aliases:
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
    # 2. Check synonyms (including ontology-merged entries)
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
    # 4. LLM fallback
    if use_llm_fallback:
        from modules.llm_helpers import _infer_manufacturer_llm
        st.toast(f"🤖 AI is researching originator for '{raw_name}'…", icon="🔍")
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
#  ONTOLOGY ENRICHMENT FROM TRIAL DATA
# ════════════════════════════════════════════════════════════════

_NOISE_INTERVENTION_TERMS = {
    "placebo", "saline", "normal saline", "control", "comparator",
    "adjuvant", "diluent", "buffer", "sham", "observation",
    "standard of care", "no intervention", "not reported",
}


def _enrich_ontology_from_trials(search_key: str, trials: list):
    """Scan fetched trial results and additively merge any new vaccine names
    discovered in intervention fields back into the ontology.

    This is the 'learning from real data' step that builds the dictionary
    over time.  Only ADDS new names — never removes existing ones.
    Common noise terms (placebo, saline, etc.) are filtered out.
    """
    norm_key = _norm_txt(search_key)
    if not norm_key:
        return

    ontology = _load_ontology()
    existing = list(ontology.get(norm_key, []))
    existing_norms = {_norm_txt(a) for a in existing if a}

    new_names = []
    for t in trials:
        vaccines_str = t.get("Vaccines", "")
        if isinstance(vaccines_str, str):
            names = [n.strip() for n in vaccines_str.split(",") if n.strip()]
        elif isinstance(vaccines_str, list):
            names = list(vaccines_str)
        else:
            continue
        for name in names:
            if not name or name == "Not reported":
                continue
            norm = _norm_txt(name)
            if norm and norm not in existing_norms and norm not in _NOISE_INTERVENTION_TERMS:
                new_names.append(name)
                existing_norms.add(norm)  # prevent duplicates within batch

    if new_names:
        _save_ontology_entry(norm_key, existing + new_names)
        _merge_single_entry_into_synonym_index(
            norm_key, existing + new_names, _VACCINE_SYNONYM_INDEX
        )


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
    """True if the normalised target name matches any intervention name.

    Matching strategy (in priority order):
    1. Exact normalised string match
    2. Token match  (target is a standalone word in the name)
    3. Substring match for terms ≥ 5 chars (catches compound codes
       embedded in longer intervention descriptions)

    ``target_norms`` includes ontology-expanded aliases, so matching
    is automatically broader (catches compound codes, INNs, etc.).
    """
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
            # 1. Exact match
            if norm == t:
                return True
            # 2. Token match
            if t in tokens:
                return True
            # 3. Substring match (≥5 chars avoids false positives from short terms)
            if len(t) >= 5 and t in norm:
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
