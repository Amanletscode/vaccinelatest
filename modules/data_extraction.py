"""
data_extraction.py — Parse raw ClinicalTrials.gov API JSON into structured Python dicts.

Each function receives a raw *study* dict (one element from the ``studies`` array
returned by ``/api/v2/studies``) and extracts a specific facet of the trial:
enrollment, dates, locations, design, eligibility, collaborators, results, and
linked PubMed publications.

These helpers are consumed by :func:`trial_fetchers.fetch_trial_details_with_vaccines`
and the bulk fetcher :func:`trial_fetchers.fetch_all_vaccine_trials`.
"""

from modules.utils import _request_json


# ──────────────────── Extraction helpers ────────────────────

def _extract_enrollment(study):
    """Extract enrollment count/type from the designModule."""
    proto = (study or {}).get("protocolSection", {}) or {}
    design = proto.get("designModule", {}) or {}
    enrollment = design.get("enrollmentInfo", {}) or {}
    return {
        "enrollment_count": enrollment.get("count"),
        "enrollment_type": enrollment.get("type"),
        "actual_enrollment": enrollment.get("actualEnrollment"),
    }


def _extract_dates(study):
    """Extract start, completion, first-posted, and last-update dates."""
    proto = (study or {}).get("protocolSection", {}) or {}
    status = proto.get("statusModule", {}) or {}
    return {
        "start_date": status.get("startDateStruct", {}).get("date") or status.get("startDate"),
        "completion_date": status.get("completionDateStruct", {}).get("date") or status.get("completionDate"),
        "first_posted": status.get("firstPostedDateStruct", {}).get("date") or status.get("firstPostedDate"),
        "last_update": status.get("lastUpdatePostedDateStruct", {}).get("date") or status.get("lastUpdatePostedDate"),
    }


def _extract_locations(study):
    """Return a list of ``{name, city, state, country}`` dicts for each study site."""
    proto = (study or {}).get("protocolSection", {}) or {}
    contacts = proto.get("contactsLocationsModule", {}) or {}
    locations = contacts.get("locations", []) or []
    return [
        {
            "name": loc.get("facility"),
            "city": loc.get("city"),
            "state": loc.get("state"),
            "country": loc.get("country"),
        }
        for loc in locations
        if loc.get("facility")
    ]


def _extract_design_details(study):
    """Extract study type, allocation, masking, and other design metadata."""
    proto = (study or {}).get("protocolSection", {}) or {}
    design = proto.get("designModule", {}) or {}
    return {
        "study_type": design.get("studyType"),
        "allocation": design.get("allocation"),
        "intervention_model": design.get("interventionModel"),
        "masking": design.get("maskingInfo", {}).get("masking") if design.get("maskingInfo") else None,
        "primary_purpose": design.get("primaryPurpose"),
        "number_of_arms": len(design.get("armsInterventionsModule", {}).get("armGroups", []) or []),
    }


def _extract_eligibility(study):
    """Extract eligibility criteria, age range, gender, healthy-volunteer flag."""
    proto = (study or {}).get("protocolSection", {}) or {}
    eligibility = proto.get("eligibilityModule", {}) or {}
    return {
        "criteria": eligibility.get("eligibilityCriteria"),
        "gender": eligibility.get("gender"),
        "minimum_age": eligibility.get("minimumAge"),
        "maximum_age": eligibility.get("maximumAge"),
        "healthy_volunteers": eligibility.get("healthyVolunteers"),
    }


def _extract_collaborators(study):
    """Return a flat list of collaborator names."""
    proto = (study or {}).get("protocolSection", {}) or {}
    sponsor = proto.get("sponsorCollaboratorsModule", {}) or {}
    collaborators = sponsor.get("collaborators", []) or []
    return [col.get("name", "") for col in collaborators if col.get("name")]


def _extract_results_summary(study):
    """Extract a lightweight results summary (baseline, outcomes, adverse events)."""
    results = (study or {}).get("resultsSection", {}) or {}
    if not results:
        return None
    baseline = results.get("baselineCharacteristicsModule", {}) or {}
    outcome = results.get("outcomeMeasuresModule", {}) or {}
    adverse = results.get("adverseEventsModule", {}) or {}
    return {
        "has_results": True,
        "participant_flow": results.get("participantFlowModule", {}).get("recruitmentDetails"),
        "baseline_characteristics": baseline.get("baselineMeasurements", []),
        "outcome_measures": outcome.get("outcomeMeasures", []),
        "adverse_events": adverse.get("events", []),
    }


# ──────────────────── PubMed link helpers ────────────────────

def _check_for_publications(study) -> str:
    """Return '📄 Yes' if the trial record contains linked PubMed IDs."""
    proto = (study or {}).get("protocolSection", {}) or {}
    refs_mod = proto.get("referencesModule", {}) or {}
    has_pmid = any(ref.get("pmid") for ref in refs_mod.get("references", []))
    return "📄 Yes" if has_pmid else "➖ No"


def _fetch_pubmed_articles_for_trial(nct_id: str, trial_data: dict) -> list:
    """Fetch PubMed publications linked to a trial (explicit PMIDs first, then search fallback)."""
    try:
        pmids = []
        proto = trial_data.get("protocolSection", {}) or {}
        refs_mod = proto.get("referencesModule", {}) or {}
        for ref in refs_mod.get("references", []):
            if ref.get("pmid"):
                pmids.append(str(ref["pmid"]))
        pmids = list(dict.fromkeys(pmids))  # dedupe preserving order

        if not pmids:
            search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={nct_id}&retmode=json"
            search_data = _request_json(search_url, timeout=10)
            pmids = search_data.get("esearchresult", {}).get("idlist", [])

        if not pmids:
            return []

        pmids = pmids[:5]
        id_str = ",".join(pmids)
        summary_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={id_str}&retmode=json"
        summary_data = _request_json(summary_url, timeout=10)

        articles = []
        result = summary_data.get("result", {})
        for pmid in pmids:
            if pmid in result:
                meta = result[pmid]
                title = meta.get("title", "No title")
                if title:
                    title = title.replace("&lt;", "<").replace("&gt;", ">").replace("<i>", "").replace("</i>", "")
                articles.append({
                    "pmid": pmid,
                    "title": title,
                    "source": meta.get("source", ""),
                    "pubdate": meta.get("pubdate", ""),
                    "authors": [a.get("name", "") for a in meta.get("authors", [])][:3],
                })
        return articles
    except Exception:
        return []


# ──────────────────── Outcome helpers ────────────────────

def _primary_outcomes_from_protocol(study) -> list:
    """Primary outcome measures from the protocol section."""
    proto = (study or {}).get("protocolSection", {}) or {}
    out_mod = proto.get("outcomesModule", {}) or {}
    out = []
    for o in out_mod.get("primaryOutcomes", []) or []:
        out.append({
            "Title": o.get("measure") or o.get("title") or "",
            "Description": o.get("description", "") or o.get("timeFrame", "") or "",
        })
    return out


def _results_outcomes(study) -> list:
    """Outcome measures from the posted results section (if available)."""
    results = (study or {}).get("resultsSection", {}) or {}
    res_mod = results.get("outcomeMeasuresModule", {}) or {}
    out = []
    for o in res_mod.get("outcomeMeasures", []) or []:
        out.append({
            "Title": o.get("title") or "",
            "Description": o.get("description", "") or "",
        })
    return out


# ──────────────────── MeSH helpers ────────────────────

def _mesh_terms_intervention(study) -> list:
    """Return MeSH terms for interventions — used to detect vaccine studies."""
    derived = (study or {}).get("derivedSection", {}) or {}
    iv_browse = derived.get("interventionBrowseModule", {}) or {}
    leaves = iv_browse.get("browseLeaves", []) or []
    return [x.get("meshTerm", "") for x in leaves if x.get("meshTerm")]


def _mesh_terms_condition(study) -> list:
    """Return MeSH terms for conditions — aids disease normalisation."""
    derived = (study or {}).get("derivedSection", {}) or {}
    cond_browse = derived.get("conditionBrowseModule", {}) or {}
    leaves = cond_browse.get("browseLeaves", []) or []
    return [x.get("meshTerm", "") for x in leaves if x.get("meshTerm")]
