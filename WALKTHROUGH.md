# 📘 Vaccine Pipeline Platform — Developer Walkthrough

This document is a **comprehensive technical guide** for developers and analysts who want to understand, modify, or extend the platform. Read the [README](README.md) first for setup instructions.

---

## Table of Contents

1. [How the App Starts](#1-how-the-app-starts)
2. [Module-by-Module Guide](#2-module-by-module-guide)
3. [Data Flow: Search → Display → Summarize](#3-data-flow)
4. [API Endpoint Reference](#4-api-endpoint-reference)
5. [Heuristic & Scoring Logic](#5-heuristic--scoring-logic)
6. [Caching Strategy](#6-caching-strategy)
7. [Error Handling Philosophy](#7-error-handling-philosophy)
8. [Extension Guide](#8-extension-guide)
9. [Known Limitations](#9-known-limitations)

---

## 1. How the App Starts

When you run `streamlit run app1.py`:

1. **Imports resolve** — Python loads all `modules/` files. At import time:
   - `config.py` sets constants (model names, file paths).
   - `vaccine_data.py` builds synonym and manufacturer indices (fast, in-memory dicts).
   - `utils.py`, `data_extraction.py`, etc. register their functions.
2. **`st.set_page_config()`** — sets page title, icon, layout=wide.
3. **Sidebar renders** — Gemini model selector + admin cache-clear button.
4. **Session state initialises** — keys like `studies`, `vaccine_trials`, `competitor_trials`.
5. **Two tabs render** — "Search by Disease" and "Search by Vaccine Product".
6. **User interacts** — clicking buttons triggers data fetching, AI calls, PDF exports.

> **Key insight:** `app.py` (monolith) and `app1.py` (modular) produce the *same* application. `app.py` is preserved as-is for reading; `app1.py` is the recommended entry point going forward.

---

## 2. Module-by-Module Guide

### `modules/config.py`
Every tuneable constant lives here. When adding a new API or changing a model default, this is the first file to touch.

| Constant | Purpose |
|----------|---------|
| `DEFAULT_LLM_MODEL` | Which Gemini model to use when no override is set |
| `MAX_TRIALS_FOR_SUMMARY` | Cap on how many trials are sent to Gemini (keeps tokens low) |
| `GEMINI_BASE_URL` | REST endpoint prefix for Google Generative AI |
| `GEMINI_MODEL_OPTIONS` | Dict that populates the sidebar model dropdown |
| `ANALYST_NOTES_FILE` | Path to the JSON file storing analyst favorites & notes |
| `LEARNED_MFR_FILE` | Path to the JSON cache of LLM-inferred manufacturers |

---

### `modules/utils.py`
Three pure-utility functions used everywhere:

- **`_norm_txt(s)`** — Lowercase, strip non-alphanumeric, collapse whitespace. This is the backbone of all fuzzy matching in the platform. Example: `"Pfizer-BioNTech COVID-19"` → `"pfizer biontech covid 19"`.
- **`_request_json(url, ...)`** — GET with retry, 429 backoff, and raise-on-error. Every external API call flows through this function.
- **`_supports_width_string()`** — Feature-detect Streamlit ≥1.39 for `width="stretch"` support.

---

### `modules/data_extraction.py`
Parses the nested JSON returned by ClinicalTrials.gov `/api/v2/studies` into flat, predictable Python dicts. Each function takes a raw `study` dict and returns one facet:

| Function | Returns |
|----------|---------|
| `_extract_enrollment` | `{enrollment_count, enrollment_type, actual_enrollment}` |
| `_extract_dates` | `{start_date, completion_date, first_posted, last_update}` |
| `_extract_locations` | List of `{name, city, state, country}` dicts |
| `_extract_design_details` | `{study_type, allocation, masking, primary_purpose, ...}` |
| `_extract_eligibility` | `{criteria, gender, minimum_age, maximum_age, healthy_volunteers}` |
| `_extract_collaborators` | Flat list of collaborator names |
| `_extract_results_summary` | Baseline, outcome, adverse-event summaries (or `None`) |
| `_check_for_publications` | `"📄 Yes"` / `"➖ No"` based on linked PMIDs |
| `_fetch_pubmed_articles_for_trial` | List of PubMed article summaries (title, authors, PMID) |
| `_primary_outcomes_from_protocol` | Protocol-level outcome measures |
| `_results_outcomes` | Posted-results outcome measures |
| `_mesh_terms_intervention` | MeSH terms for interventions |
| `_mesh_terms_condition` | MeSH terms for conditions |

---

### `modules/vaccine_data.py`
This is the **domain knowledge** module. It contains:

**Curated Data:**
- `_VACCINE_SYNONYM_GROUPS` — Lists of names that all refer to the same product (e.g., `["Comirnaty", "BNT162b2", "Tozinameran"]`).
- `_VACCINE_MANUFACTURER_DATA` — Maps manufacturer names to their vaccine portfolio.

**Indices (built at import time):**
- `_VACCINE_SYNONYM_INDEX` — `{normalised_name → [all synonyms]}`. Used by `_get_vaccine_search_terms()` to expand a user query.
- `_VACCINE_MANUFACTURER_INDEX` — `{normalised_name → manufacturer}`. Used by `_get_vaccine_manufacturer()`.

**Key Functions:**
- `_is_vaccine_study(study)` — 3-layer heuristic: MeSH terms → biological intervention name → immunogenicity outcome keywords. This classifier is **critical** — it decides which studies from a disease search actually make it into the trial list.
- `_extract_vaccine_names(study)` — Pulls all intervention names + other names + arm-group names.
- `_matches_vaccine_name(names, target_norms)` — Token-level matching, not substring. Prevents `"RSV"` matching `"RSVP"`.
- `_get_vaccine_manufacturer(raw_name)` — 4-step cascade: hardcoded → synonym lookup → JSON cache → Gemini LLM fallback.

**Adding a new vaccine:**
```python
# In _VACCINE_SYNONYM_GROUPS, add:
["NewVaxBrand", "NVX-1234", "Genericname", "Company NewVax Vaccine"],

# In _VACCINE_MANUFACTURER_DATA, add:
"CompanyName": ["NewVaxBrand", "NVX-1234", ...],
```

---

### `modules/llm_helpers.py`
All Gemini API communication and prompt engineering:

- **`_call_gemini(messages, ...)`** — Core API call. Converts OpenAI-style `{role, content}` messages to Gemini format. Handles `systemInstruction`, temperature, `maxOutputTokens`.
- **`_summarize_trials_with_llm(records, context)`** — Multi-trial executive summary with structured sections: Executive Summary → Key Insights → Competitive Intelligence → Risk Assessment → Strategic Recommendations.
- **`_summarize_single_trial(nct_id, details)`** — Single-trial due-diligence brief with 12 sections.
- **`_vaccine_intel_summary(...)`** — Combines product trials + competitor trials + Wikipedia context into a unified intelligence brief.
- **`_fetch_wikipedia_summary(term)`** — Quick encyclopedic context. Only used as optional input to Gemini prompts.
- **`_infer_manufacturer_llm(vaccine_name)`** — Uses Gemini Flash to guess the originator of an unknown vaccine. Results are cached to `learned_manufacturers.json`.

---

### `modules/trial_fetchers.py`
All external API calls (Streamlit-cached):

- **`fetch_all_vaccine_trials(disease, max_pages)`** — Paginated ClinicalTrials.gov search with vaccine classification. Has fallback broadening if zero results.
- **`fetch_trial_details_with_vaccines(nct_id)`** — Full detail fetch for one trial.
- **`fetch_pipeline_publications(query, max_items)`** — PubMed + FDA RSS. PubMed uses `eng[la]`, `mindate=2024/01/01`, `sort=pub_date`. FDA RSS handles both Atom and RSS 2.0 formats.
- **`fetch_openfda_data(vaccine_name)`** — 3-strategy NDC lookup (brand name → synonyms → manufacturer data) + FAERS count + DailyMed/Drugs@FDA links.
- **`_fetch_faers_count(brand_name)`** — FAERS adverse-event report count via `drug/event.json`.

---

### `modules/visualizations.py`
Four Plotly chart generators — all gracefully return `None` if Plotly is not installed:

| Function | Chart Type |
|----------|------------|
| `create_phase_chart` | Bar chart of phase distribution |
| `create_status_chart` | Pie chart of trial statuses |
| `create_sponsor_chart` | Horizontal bar of top N sponsors |
| `create_country_heatmap` | World choropleth of trial site density |

---

### `modules/analyst_workspace.py`
- **`calculate_threat_score(t)`** — Heuristic 0–100 score.
- **`_load_analyst_notes()` / `_save_analyst_notes()`** — JSON persistence for Favorites + Notes.
- **`show_interactive_df(df, key, height)`** — Renders `st.data_editor` with Favorite (checkbox) and Note (text) columns. Auto-saves changes.

---

### `modules/pdf_export.py`
- **`generate_pdf_summary(text, title)`** — Parses markdown-like text (headings, bold, bullets) into ReportLab Paragraph objects. Falls back to fpdf2 if ReportLab is not installed.

---

## 3. Data Flow

### Disease Search Flow
```
User types "RSV" → clicks "Fetch All Trials"
  │
  ├─ fetch_all_vaccine_trials("RSV")
  │    ├─ ClinicalTrials.gov API: query.cond=RSV, page through results
  │    ├─ For each study: _is_vaccine_study() → True/False
  │    ├─ Extract: NCT ID, Title, Phase, Status, Sponsor, Vaccines, Locations
  │    └─ Calculate threat score for each trial
  │
  ├─ Display: sidebar filters → df_filtered → charts + interactive table
  │
  ├─ [Optional] "Summarize" → _summarize_trials_with_llm() → Gemini API
  │
  └─ [Optional] "View Detail" → fetch_trial_details_with_vaccines(nct_id)
       └─ _render_trial_details() → full detail view with PubMed articles
```

### Vaccine Product Search Flow
```
User types "Comirnaty" → clicks "Search Vaccine & Competitors"
  │
  ├─ Step 1: _get_vaccine_search_terms("Comirnaty")
  │    → ["Comirnaty", "BNT162b2", "Tozinameran", ...]
  │    → Search ClinicalTrials.gov for each term
  │    → _matches_vaccine_name() to verify each result
  │    → Collect all target diseases
  │
  ├─ Step 2: For each top disease:
  │    → fetch_all_vaccine_trials(disease)
  │    → Filter OUT our vaccine → remaining = competitors
  │
  ├─ Display: vaccine trials table + competitor trials table
  │    ├─ openFDA regulatory data (NDC + FAERS)
  │    ├─ PubMed + FDA news
  │    └─ Sponsor Type classification (Originator vs External)
  │
  └─ [Optional] "Unified Intelligence" → _vaccine_intel_summary()
       → Combines product + competitor + Wikipedia context → Gemini
```

---

## 4. API Endpoint Reference

### ClinicalTrials.gov v2
| Endpoint | Parameters | Used In |
|----------|-----------|---------|
| `GET /api/v2/studies` | `query.cond`, `query.intr`, `query.term`, `pageSize`, `pageToken` | `fetch_all_vaccine_trials`, vaccine search |
| `GET /api/v2/studies/{nctId}` | — | `fetch_trial_details_with_vaccines` |

### PubMed E-utilities
| Endpoint | Parameters | Used In |
|----------|-----------|---------|
| `esearch.fcgi` | `db=pubmed`, `term`, `retmax`, `sort`, `mindate` | `fetch_pipeline_publications`, `_fetch_pubmed_articles_for_trial` |
| `esummary.fcgi` | `db=pubmed`, `id` | Same |

### openFDA
| Endpoint | Parameters | Used In |
|----------|-----------|---------|
| `drug/ndc.json` | `search=brand_name:{name}` | `fetch_openfda_data` |
| `drug/event.json` | `search=patient.drug.medicinalproduct:{name}` | `_fetch_faers_count` |

### Google Gemini
| Endpoint | Payload | Used In |
|----------|---------|---------|
| `/{model}:generateContent` | `contents`, `generationConfig`, `systemInstruction` | `_call_gemini` |

---

## 5. Heuristic & Scoring Logic

### Threat Score (0–100)

```
Phase 3           → +40
Phase 2           → +20
Phase 1           → +10
Recruiting/Active → +15
Completed         → +20
Major Sponsor     → +25  (Pfizer, GSK, Sanofi, Merck, Moderna, etc.)
Combo Vaccine     → +10  (contains + or / or multiple comma-separated names)
Cap at 100
```

### Vaccine Study Classifier (`_is_vaccine_study`)

```
Layer 1: MeSH intervention terms contain "vaccine"? → YES
Layer 2: Has BIOLOGICAL intervention with name containing "vaccine"? → YES
Layer 3: Primary outcomes mention immunogenicity/antibody/seroconversion? → YES
→ Otherwise: NOT a vaccine study
```

### Sponsor Type Classification

The platform classifies trial sponsors relative to the vaccine's originator:
- **Originator / Manufacturer** — sponsor name contains the manufacturer's name (or an alias like "Wyeth" for Pfizer)
- **External / Other** — academic, government, or competitor-run trials using this vaccine

---

## 6. Caching Strategy

| Function | TTL | Rationale |
|----------|-----|-----------|
| `fetch_all_vaccine_trials` | 1 hour | Trial data changes rarely; avoids API hammering |
| `fetch_trial_details_with_vaccines` | 1 hour | Same |
| `fetch_pipeline_publications` | 1 hour | PubMed data updates daily, 1h is sufficient |
| `fetch_openfda_data` | 24 hours | NDC data changes very rarely |
| `_fetch_faers_count` | 24 hours | FAERS data is quarterly |
| `_infer_manufacturer_llm` | 24 hours | Gemini results are stable; also file-cached |

All caching uses `@st.cache_data` which survives page reruns but not app restarts.

---

## 7. Error Handling Philosophy

- **API failures** are caught and return empty/None — the UI shows informative warnings rather than crashing.
- **Missing data** is handled with `.get()` chains and `or {}` defaults everywhere.
- **Import errors** (e.g., Plotly not installed) are caught per-function and return `None`.
- **LLM failures** return `(None, error_message)` tuples — the UI displays the error as `st.warning()`.

---

## 8. Extension Guide

### Adding a New Registry (e.g., EU CTR)

1. Create a new fetcher in `modules/trial_fetchers.py`:
   ```python
   @st.cache_data(ttl=3600, show_spinner=False)
   def fetch_euctr_trials(disease: str) -> list:
       ...
   ```
2. Import and call it in `app1.py` alongside the existing disease search.
3. Add any new constants to `modules/config.py`.

### Adding a New Visualization

1. Add the function to `modules/visualizations.py`:
   ```python
   def create_enrollment_timeline(df: pd.DataFrame):
       ...
   ```
2. Call it in `app1.py` inside the relevant tab's visualization section.

### Adding a New LLM Prompt

1. Add the prompt function to `modules/llm_helpers.py`:
   ```python
   def _custom_summary(data):
       system_prompt = "..."
       user_prompt = "..."
       return _call_gemini([...])
   ```
2. Wire it up with a Streamlit button in `app1.py`.

---

## 9. Known Limitations

| Limitation | Impact | Workaround |
|-----------|--------|------------|
| openFDA only covers approved products | Pipeline assets won't have NDC/FAERS data | UI warns user explicitly |
| PubMed E-utilities has rate limits | Heavy concurrent use may trigger 429s | `_request_json` auto-retries with backoff |
| Gemini free tier has token limits | Very large trial sets may time out | `MAX_TRIALS_FOR_SUMMARY = 10` caps input |
| Analyst notes are local JSON | Not multi-user; lost if file deleted | Future: database backend |
| Threat score is heuristic only | Not a rigorous competitive assessment | Users can add analyst notes to supplement |

---

*Last updated: March 2026*
