"""
trial_fetchers.py — API data fetchers (cached with Streamlit).

Covers:
- ClinicalTrials.gov bulk trial fetch by disease
- ClinicalTrials.gov single trial detail fetch
- PubMed E-utilities (publications)
- FDA Press Releases RSS
- openFDA NDC (regulatory data)
- openFDA FAERS (adverse event report counts)
"""

import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import requests
import streamlit as st

from modules.utils import _request_json, _norm_txt
from modules.data_extraction import (
    _extract_enrollment,
    _extract_dates,
    _extract_locations,
    _extract_design_details,
    _extract_eligibility,
    _extract_collaborators,
    _extract_results_summary,
    _check_for_publications,
    _fetch_pubmed_articles_for_trial,
    _mesh_terms_condition,
    _results_outcomes,
    _primary_outcomes_from_protocol,
)
from modules.vaccine_data import (
    _extract_vaccine_names,
    _is_vaccine_study,
    _VACCINE_SYNONYM_INDEX,
    _VACCINE_SYNONYM_GROUPS,
    _VACCINE_MANUFACTURER_DATA,
)
# NOTE: calculate_threat_score removed per user feedback (heuristic unreliable)


# ════════════════════════════════════════════════════════════════
#  CLINICALTRIALS.GOV
# ════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all_vaccine_trials(disease: str, max_pages: int = 10):
    """
    Fetch vaccine trials by disease from ClinicalTrials.gov v2 API.

    Strategy: search by disease, page through results, classify vaccines
    client-side using ``_is_vaccine_study``.
    Fallback: if zero vaccine trials found, repeat with ``query.term=vaccin*``.
    """
    url = "https://clinicaltrials.gov/api/v2/studies"
    all_results = []
    page_token = None
    page_count = 0
    seen = set()
    session = requests.Session()
    headers = {"User-Agent": "VaccinePipeline/1.0"}

    try:
        while page_count < max_pages:
            params = {"query.cond": disease, "pageSize": 100}
            if page_token:
                params["pageToken"] = page_token
            data = _request_json(url, params=params, session=session, headers=headers)
            studies = data.get("studies", []) or []
            if not studies:
                break
            for s in studies:
                if not _is_vaccine_study(s):
                    continue
                proto = s.get("protocolSection", {}) or {}
                ident = proto.get("identificationModule", {}) or {}
                design = proto.get("designModule", {}) or {}
                status_mod = proto.get("statusModule", {}) or {}
                sponsor_mod = proto.get("sponsorCollaboratorsModule", {}) or {}
                nct_id = ident.get("nctId")
                if not nct_id or nct_id in seen:
                    continue
                seen.add(nct_id)
                vaccines = _extract_vaccine_names(s)
                locations = _extract_locations(s)
                trial_obj = {
                    "NCT ID": str(nct_id),
                    "Title": str(ident.get("briefTitle") or ident.get("officialTitle") or "No title"),
                    "Phase": ", ".join(design.get("phases") or ["Not reported"]),
                    "Status": str(status_mod.get("overallStatus") or "Unknown"),
                    "Sponsor": str(sponsor_mod.get("leadSponsor", {}).get("name", "Unknown")),
                    "Vaccines": ", ".join(vaccines) if vaccines else "Not reported",
                    "Locations": locations,
                    "Publications": _check_for_publications(s),
                    "Start Date": str(status_mod.get("startDateStruct", {}).get("date", "") or ""),
                    "Completion Date": str(status_mod.get("completionDateStruct", {}).get("date", "") or ""),
                }
                all_results.append(trial_obj)
            page_token = data.get("nextPageToken")
            page_count += 1
            if not page_token:
                break
            time.sleep(0.25)

        # Fallback: broaden text search
        if not all_results:
            page_token = None
            page_count = 0
            while page_count < max_pages:
                params = {"query.cond": disease, "query.term": "vaccin*", "pageSize": 100}
                if page_token:
                    params["pageToken"] = page_token
                data = _request_json(url, params=params, session=session, headers=headers)
                studies = data.get("studies", []) or []
                if not studies:
                    break
                for s in studies:
                    if not _is_vaccine_study(s):
                        continue
                    proto = s.get("protocolSection", {}) or {}
                    ident = proto.get("identificationModule", {}) or {}
                    design = proto.get("designModule", {}) or {}
                    status_mod = proto.get("statusModule", {}) or {}
                    sponsor_mod = proto.get("sponsorCollaboratorsModule", {}) or {}
                    nct_id = ident.get("nctId")
                    if not nct_id or nct_id in seen:
                        continue
                    seen.add(nct_id)
                    vaccines = _extract_vaccine_names(s)
                    locations = _extract_locations(s)
                    trial_obj = {
                        "NCT ID": str(nct_id),
                        "Title": str(ident.get("briefTitle") or ident.get("officialTitle") or "No title"),
                        "Phase": ", ".join(design.get("phases") or ["Not reported"]),
                        "Status": str(status_mod.get("overallStatus") or "Unknown"),
                        "Sponsor": str(sponsor_mod.get("leadSponsor", {}).get("name", "Unknown")),
                        "Vaccines": ", ".join(vaccines) if vaccines else "Not reported",
                        "Locations": locations,
                        "Start Date": str(status_mod.get("startDateStruct", {}).get("date", "") or ""),
                        "Completion Date": str(status_mod.get("completionDateStruct", {}).get("date", "") or ""),
                    }
                    all_results.append(trial_obj)
                page_token = data.get("nextPageToken")
                page_count += 1
                if not page_token:
                    break
                time.sleep(0.25)

        return all_results
    except requests.RequestException as e:
        st.error(f"Error fetching data: {e}")
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_trial_details_with_vaccines(nct_id: str):
    """Fetch comprehensive detail for a single trial."""
    url = f"https://clinicaltrials.gov/api/v2/studies/{nct_id}"
    try:
        data = _request_json(url, params=None)
        proto = data.get("protocolSection", {}) or {}
        design = proto.get("designModule", {}) or {}
        status_mod = proto.get("statusModule", {}) or {}
        sponsor_mod = proto.get("sponsorCollaboratorsModule", {}) or {}
        conditions = proto.get("conditionsModule", {}) or {}
        vaccines = _extract_vaccine_names(data)

        disease_list = []
        disease_list.extend(conditions.get("conditions", []) or [])
        disease_list.extend(_mesh_terms_condition(data))
        seen_d = set()
        disease_list = [d for d in disease_list if d and not (d in seen_d or seen_d.add(d))]

        outcomes = _results_outcomes(data)
        if not outcomes:
            outcomes = _primary_outcomes_from_protocol(data)

        ident = proto.get("identificationModule", {}) or {}
        return {
            "NCT ID": nct_id,
            "Title": str(ident.get("briefTitle") or ident.get("officialTitle") or "No title"),
            "Phase": ", ".join(design.get("phases") or ["Not reported"]),
            "Status": str(status_mod.get("overallStatus", "Unknown")),
            "Sponsor": str(sponsor_mod.get("leadSponsor", {}).get("name", "Unknown")),
            "Vaccines": sorted(vaccines) if vaccines else ["Not reported"],
            "Diseases": disease_list,
            "Outcomes": outcomes,
            "Enrollment": _extract_enrollment(data),
            "Dates": _extract_dates(data),
            "Locations": _extract_locations(data),
            "Design": _extract_design_details(data),
            "Eligibility": _extract_eligibility(data),
            "Collaborators": _extract_collaborators(data),
            "Results": _extract_results_summary(data),
            "PubMed_Articles": _fetch_pubmed_articles_for_trial(nct_id, data),
        }
    except requests.RequestException:
        return None


# ════════════════════════════════════════════════════════════════
#  PUBMED + FDA RSS
# ════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_pipeline_publications(query: str, max_items: int = 5):
    """Fetch recent English-language PubMed articles + FDA press releases."""
    items = []

    # PubMed: English-only, clinical-trial-biased, recent
    try:
        search_q = urllib.parse.quote(f"{query} vaccine clinical trial eng[la]")
        search_url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            f"?db=pubmed&term={search_q}&retmode=json&retmax={max_items}"
            f"&sort=pub_date&datetype=pdat&mindate=2024/01/01"
        )
        search_data = _request_json(search_url, timeout=10)
        pmids = search_data.get("esearchresult", {}).get("idlist", [])
        if pmids:
            id_str = ",".join(pmids)
            summary_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={id_str}&retmode=json"
            summary_data = _request_json(summary_url, timeout=10)
            result = summary_data.get("result", {})
            for pmid in pmids:
                if pmid in result:
                    meta = result[pmid]
                    title = meta.get("title", "No title")
                    if title:
                        title = title.replace("&lt;", "<").replace("&gt;", ">").replace("<i>", "").replace("</i>", "")
                    source = meta.get("source", "")
                    pubdate = meta.get("pubdate", "")
                    authors = meta.get("authors", [])
                    author_str = ""
                    if authors:
                        first_authors = [a.get("name", "") for a in authors[:2]]
                        author_str = ", ".join(first_authors)
                        if len(authors) > 2:
                            author_str += " et al."
                    items.append({
                        "title": title,
                        "link": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        "pubdate": pubdate,
                        "source": f"📄 {source}",
                        "authors": author_str,
                        "type": "pubmed",
                    })
    except Exception:
        pass

    # FDA Press Releases RSS (up to 3, vaccine-filtered)
    try:
        fda_url = "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/press-releases/rss.xml"
        req = urllib.request.Request(fda_url, headers={"User-Agent": "Mozilla/5.0 VaccinePipeline/1.0"})
        with urllib.request.urlopen(req, timeout=8) as response:
            xml_data = response.read()
        root = ET.fromstring(xml_data)
        query_lower = query.lower()
        fda_count = 0
        entries = root.findall(".//{http://www.w3.org/2005/Atom}entry")
        if not entries:
            entries = root.findall(".//item")
        for item_el in entries:
            if fda_count >= 3:
                break
            title_el = item_el.find("{http://www.w3.org/2005/Atom}title")
            link_el = item_el.find("{http://www.w3.org/2005/Atom}link")
            updated_el = item_el.find("{http://www.w3.org/2005/Atom}updated")
            if title_el is None:
                title_el = item_el.find("title")
            if link_el is None:
                link_el = item_el.find("link")
            if updated_el is None:
                updated_el = item_el.find("pubDate")
            if title_el is not None and title_el.text:
                title_text = title_el.text.strip()
                title_lower = title_text.lower()
                if any(kw in title_lower for kw in ["vaccine", "biologics", "bla", query_lower, "immunization", "cber"]):
                    href = ""
                    if link_el is not None:
                        href = link_el.get("href", "") or (link_el.text or "")
                    pub_text = ""
                    if updated_el is not None and updated_el.text:
                        pub_text = updated_el.text[:10] if len(updated_el.text) >= 10 else updated_el.text
                    items.append({
                        "title": title_text,
                        "link": href,
                        "pubdate": pub_text,
                        "source": "🏛️ FDA Press Release",
                        "authors": "",
                        "type": "fda",
                    })
                    fda_count += 1
    except Exception:
        pass

    return items


# ════════════════════════════════════════════════════════════════
#  OPENFDA  (NDC + FAERS)
# ════════════════════════════════════════════════════════════════

@st.cache_data(ttl=86400, show_spinner=False)
def _fetch_faers_count(brand_name: str) -> int:
    """Total FAERS adverse-event report count for a vaccine."""
    try:
        q = urllib.parse.quote(brand_name.upper())
        url = f'https://api.fda.gov/drug/event.json?search=patient.drug.medicinalproduct:%22{q}%22&limit=1'
        data = _request_json(url, timeout=10)
        return data.get("meta", {}).get("results", {}).get("total", 0)
    except Exception:
        return 0


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_openfda_data(vaccine_name: str):
    """Fetch regulatory info from openFDA NDC + FAERS adverse-event data."""

    def _try_ndc_search(search_term: str):
        try:
            q = urllib.parse.quote(search_term)
            url = f"https://api.fda.gov/drug/ndc.json?search=brand_name:{q}&limit=3"
            data = _request_json(url, timeout=10)
            results = data.get("results", [])
            if results:
                return results
        except Exception:
            pass
        return None

    # Strategy 1: Direct brand name search
    results = _try_ndc_search(vaccine_name)

    # Strategy 2: Known synonyms
    if not results:
        norm = _norm_txt(vaccine_name)
        synonyms = _VACCINE_SYNONYM_INDEX.get(norm, [])
        for syn in synonyms:
            results = _try_ndc_search(syn)
            if results:
                break

    # Strategy 3: Manufacturer index → synonym groups
    if not results:
        for mfr, names in _VACCINE_MANUFACTURER_DATA.items():
            for name in names:
                if _norm_txt(name) == _norm_txt(vaccine_name):
                    for syn_group in _VACCINE_SYNONYM_GROUPS:
                        if name in syn_group:
                            for syn in syn_group:
                                results = _try_ndc_search(syn)
                                if results:
                                    break
                        if results:
                            break
                if results:
                    break
            if results:
                break

    if not results:
        return None

    best = results[0]
    for r in results:
        if r.get("product_type", "").upper() == "VACCINE":
            best = r
            break

    ingredients = best.get("active_ingredients", [])
    ingredient_str = "; ".join(
        [f"{i.get('name', '')} ({i.get('strength', '')})" for i in ingredients]
    ) if ingredients else "Not listed"

    mkt_date = best.get("marketing_start_date", "")
    if mkt_date and len(mkt_date) == 8:
        mkt_date = f"{mkt_date[:4]}-{mkt_date[4:6]}-{mkt_date[6:8]}"

    brand = best.get("brand_name", "")
    app_num = best.get("application_number", "")
    faers_total = _fetch_faers_count(brand) if brand else 0

    dailymed_link = ""
    if brand:
        dailymed_link = f"https://dailymed.nlm.nih.gov/dailymed/search.cfm?labeltype=all&query={urllib.parse.quote(brand)}"
    drugsfda_link = ""
    if app_num:
        drugsfda_link = f"https://www.accessdata.fda.gov/scripts/cder/daf/index.cfm?event=overview.process&ApplNo={app_num.replace('BLA', '')}"

    return {
        "brand_name": brand,
        "generic_name": best.get("generic_name", ""),
        "manufacturer": best.get("labeler_name", ""),
        "product_type": best.get("product_type", ""),
        "dosage_form": best.get("dosage_form", ""),
        "route": ", ".join(best.get("route", [])) if isinstance(best.get("route"), list) else best.get("route", ""),
        "active_ingredients": ingredient_str,
        "marketing_category": best.get("marketing_category", ""),
        "application_number": app_num,
        "marketing_start_date": mkt_date,
        "ndc": best.get("product_ndc", ""),
        "faers_total": faers_total,
        "dailymed_link": dailymed_link,
        "drugsfda_link": drugsfda_link,
    }
