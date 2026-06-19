"""
llm_helpers.py — Gemini API integration and LLM-powered summarisation.

Contains all functions that communicate with Google's Generative AI (Gemini)
REST API.  Prompt templates for executive summaries, single-trial briefs,
and competitive intelligence are defined here as well.
"""

import os
import json
import requests
import streamlit as st
from modules.config import (
    DEFAULT_LLM_MODEL, GEMINI_BASE_URL,
    GEMINI_MODEL_TOKEN_LIMITS, DEFAULT_TOKEN_LIMIT,
    PROMPT_TOKEN_BUDGETS, ANTI_TRUNCATION_SUFFIX,
)


# ════════════════════════════════════════════════════════════════
#  API KEY & MODEL SELECTION
# ════════════════════════════════════════════════════════════════

def _get_secret(name: str):
    try:
        return st.secrets.get(name)
    except Exception:
        return None


def _get_gemini_key():
    key = _get_secret("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        return None, "Set GEMINI_API_KEY in Streamlit secrets or environment."
    return key, None


# ════════════════════════════════════════════════════════════════
#  ADAPTIVE TOKEN BUDGET
# ════════════════════════════════════════════════════════════════

def _get_token_budget(prompt_type: str = "default") -> int:
    """Return the max output tokens for the currently selected model + prompt type."""
    model = None
    try:
        model = st.session_state.get("gemini_model")
    except Exception:
        pass
    model = model or DEFAULT_LLM_MODEL
    limit = GEMINI_MODEL_TOKEN_LIMITS.get(model, DEFAULT_TOKEN_LIMIT)
    fraction = PROMPT_TOKEN_BUDGETS.get(prompt_type, PROMPT_TOKEN_BUDGETS["default"])
    return int(limit * fraction)


# ════════════════════════════════════════════════════════════════
#  CORE GEMINI CALL
# ════════════════════════════════════════════════════════════════

def _call_gemini(messages, model=None, max_tokens=900, temperature=0.25, response_mime_type="text/plain"):
    """Send a chat-style request to the Gemini REST API and return (text, error)."""
    api_key, err = _get_gemini_key()
    if not api_key:
        return None, err

    ui_model = None
    try:
        ui_model = st.session_state.get("gemini_model")
    except Exception:
        ui_model = None
    model_name = (
        model
        or ui_model
        or _get_secret("GEMINI_MODEL")
        or os.getenv("GEMINI_MODEL")
        or DEFAULT_LLM_MODEL
    )

    # Convert OpenAI-style messages → Gemini format
    contents = []
    system_instruction = None
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role == "system":
            system_instruction = content
        elif role == "user":
            contents.append({"role": "user", "parts": [{"text": content}]})
        elif role == "assistant":
            contents.append({"role": "model", "parts": [{"text": content}]})

    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "responseMimeType": response_mime_type
        },
    }
    if system_instruction:
        payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

    url = f"{GEMINI_BASE_URL}/{model_name}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return None, "No response from Gemini"
        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        if not parts:
            return None, "Empty response from Gemini"
        text = parts[0].get("text", "")
        return text, None
    except requests.Timeout:
        return None, "Gemini request timed out. Try again, use a smaller filter set, or switch to a faster model (e.g., Flash)."
    except requests.HTTPError as e:
        try:
            err_json = resp.json()
            detail = err_json.get("error", {}).get("message") or str(err_json)
        except Exception:
            detail = str(e)
        return None, f"Gemini request failed: {detail}"
    except Exception as e:
        return None, f"Gemini request failed: {e}"


# ════════════════════════════════════════════════════════════════
#  SUMMARISATION PROMPTS
# ════════════════════════════════════════════════════════════════

def _summarize_trials_with_llm(records: list, context_instructions: str):
    """Build an executive summary of multiple trials using Gemini."""
    from modules.config import MAX_TRIALS_FOR_SUMMARY

    trials = records[:MAX_TRIALS_FOR_SUMMARY]
    if not trials:
        return None, "Nothing to summarize."

    trial_details = []
    for idx, t in enumerate(trials, start=1):
        detail = f"""
Trial {idx}:
- NCT ID: {t.get('NCT ID', 'N/A')}
- Title: {t.get('Title', 'N/A')}
- Phase: {t.get('Phase', 'Unknown')}
- Status: {t.get('Status', 'Unknown')}
- Sponsor: {t.get('Sponsor', 'Unknown')}
- Vaccines: {t.get('Vaccines', 'Not reported')}
"""
        if t.get('Enrollment'):
            enroll = t.get('Enrollment', {})
            if enroll.get('actual_enrollment') or enroll.get('enrollment_count'):
                detail += f"- Enrollment: {enroll.get('actual_enrollment') or enroll.get('enrollment_count', 'N/A')}\n"
        if t.get('Dates'):
            dates = t.get('Dates', {})
            if dates.get('start_date'):
                detail += f"- Start Date: {dates.get('start_date')}\n"
            if dates.get('completion_date'):
                detail += f"- Completion Date: {dates.get('completion_date')}\n"
        if t.get('Design'):
            design = t.get('Design', {})
            if design.get('study_type'):
                detail += f"- Study Type: {design.get('study_type')}\n"
            if design.get('allocation'):
                detail += f"- Allocation: {design.get('allocation')}\n"
        trial_details.append(detail.strip())

    system_prompt = """You are an expert regulatory intelligence analyst specializing in vaccine clinical development. 
Your role is to synthesize complex clinical trial data into actionable executive briefings for pharmaceutical executives, 
BD teams, and medical affairs professionals.

Your summaries must be:
- Factual and evidence-based (always ground claims in specific NCT IDs and data points)
- Strategic (highlight competitive positioning, regulatory implications, market timing)
- Actionable (provide clear next steps for business development)
- Risk-aware (identify potential regulatory, safety, or competitive risks)
- Concise but comprehensive (executive-level detail without overwhelming)

When you make a concrete statement (e.g., 'most RSV trials are Phase 3' or 'enrollment is slowing'),
back it up by citing 1–3 example NCT IDs in parentheses, like (e.g., NCT01234567, NCT08976543).

Focus on: phase progression signals, enrollment trends, sponsor competitive landscape, regulatory timeline implications, 
and strategic opportunities or threats."""

    user_prompt = f"""Context: {context_instructions}

Detailed Clinical Trial Data:
{chr(10).join(trial_details)}

Please provide a comprehensive executive summary structured as follows:

## EXECUTIVE SUMMARY
[2-3 sentence high-level overview of the vaccine trial landscape]

## KEY INSIGHTS
[3-5 bullet points covering:
- Phase distribution and progression signals
- Sponsor competitive landscape and market positioning
- Enrollment trends and study maturity
- Notable design features or regulatory implications]

## COMPETITIVE INTELLIGENCE
[Analysis of:
- Leading sponsors and their pipeline depth
- Phase advancement velocity
- Geographic or indication coverage gaps/opportunities]

## RISK ASSESSMENT
[Identify:
- Regulatory or safety concerns
- Competitive threats
- Data gaps or study limitations]

## STRATEGIC RECOMMENDATIONS
[Actionable next steps for:
- Business development opportunities
- Medical affairs engagement priorities
- Competitive monitoring focus areas]

Format the response in clear, professional language suitable for C-suite presentations.""" + ANTI_TRUNCATION_SUFFIX

    return _call_gemini(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=_get_token_budget("executive_summary"),
        temperature=0.3,
    )


def _summarize_single_trial(nct_id: str, details: dict):
    """Create a comprehensive due-diligence brief for one trial."""
    payload = {
        "nct_id": nct_id,
        "title": details.get("Title"),
        "phase": details.get("Phase"),
        "status": details.get("Status"),
        "sponsor": details.get("Sponsor"),
        "vaccines": details.get("Vaccines"),
        "diseases": details.get("Diseases"),
        "outcomes": details.get("Outcomes"),
        "enrollment": details.get("Enrollment"),
        "dates": details.get("Dates"),
        "locations": details.get("Locations"),
        "design": details.get("Design"),
        "eligibility": details.get("Eligibility"),
        "collaborators": details.get("Collaborators"),
        "has_results": details.get("Results") is not None,
        "pubmed_articles": details.get("PubMed_Articles", []),
    }

    system_prompt = """You are a senior clinical development analyst preparing a comprehensive due-diligence brief 
for a vaccine clinical trial. Your audience includes regulatory affairs, business development, and medical affairs teams.

Your brief must be:
- Comprehensive: Cover all critical aspects of the trial
- Regulatory-focused: Highlight FDA/EMA submission implications
- Risk-aware: Identify safety, efficacy, or regulatory concerns
- Actionable: Provide clear assessment for decision-making
- Professional: Suitable for executive review and client presentations

Always cite the NCT ID prominently and structure information logically."""

    user_prompt = f"""Create a comprehensive due-diligence brief for clinical trial {nct_id}.

Trial Data:
{json.dumps(payload, indent=2, default=str)}

Structure your brief as follows:

## TRIAL OVERVIEW
[Opening paragraph: NCT ID, title, phase, status, sponsor, target disease(s), and key vaccine product(s)]

## STUDY DESIGN & METHODOLOGY
[Cover: study type, allocation, masking, primary purpose, number of arms, intervention model]

## ENROLLMENT & TIMELINE
[Detail: enrollment numbers (target vs actual), start/completion dates, study duration, recruitment status]

## POPULATION & ELIGIBILITY
[Describe: age range, gender, inclusion/exclusion criteria, healthy volunteers status]

## PRIMARY & SECONDARY OUTCOMES
[List and describe all outcome measures, timeframes, and assessment methods]

## GEOGRAPHIC SCOPE
[Detail: study locations, countries, sites (if available)]

## COLLABORATORS & PARTNERSHIPS
[List: lead sponsor, collaborators, any notable partnerships]

## RESULTS STATUS
[Note: whether results are posted, preliminary data availability, expected completion timeline]

## REGULATORY & COMPETITIVE CONTEXT
[Assess: regulatory pathway implications, competitive positioning, market timing]

## PUBLISHED LITERATURE
[Summarize any findings from the provided PubMed articles. Focus on conclusions relevant to efficacy/safety if available.]

## RISK ASSESSMENT
[Identify: potential safety concerns, study limitations, data gaps, regulatory risks]

## STRATEGIC IMPLICATIONS
[Provide: BD opportunities, competitive threats, partnership potential, market implications]

Format professionally with clear sections and bullet points where appropriate.""" + ANTI_TRUNCATION_SUFFIX

    return _call_gemini(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=_get_token_budget("single_trial"),
        temperature=0.3,
    )


def _fetch_wikipedia_summary(term: str, lang: str = "en"):
    """Fetch a short encyclopedic summary from Wikipedia (optional context for Gemini)."""
    if not term:
        return None
    try:
        import urllib.parse
        title = urllib.parse.quote(term.strip().replace(" ", "_"))
        url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title}"
        resp = requests.get(url, headers={"User-Agent": "VaccinePipeline/1.0"}, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
        return data.get("extract")
    except Exception:
        return None


def _vaccine_intel_summary(
    vaccine_name: str,
    manufacturer,
    diseases: list,
    vaccine_trials: list,
    competitor_trials: list,
    external_context,
):
    """High-level Gemini intelligence brief combining product + trial + competitor data."""
    if not vaccine_trials and not competitor_trials:
        return None, "No trials to analyze for this vaccine yet."

    self_trials = vaccine_trials[:20]
    comp_trials = competitor_trials[:20]

    payload = {
        "vaccine_name": vaccine_name,
        "manufacturer": manufacturer,
        "target_diseases": diseases,
        "vaccine_trials": self_trials,
        "competitor_trials": comp_trials,
        "external_context": external_context,
    }

    system_prompt = """
You are a senior vaccine market and clinical development intelligence analyst.
Your audience is vaccine strategy, BD, and medical affairs leaders.
You combine structured clinical trial data (including NCT IDs) with high‑level background context
to produce actionable, up‑to‑date insights about a specific vaccine product
and its competitive landscape.

Your responses must be:
- Fact‑focused and analytical (no hype)
- Clear about what comes from clinicaltrials.gov data vs general background
- Explicit about uncertainties or missing data

When you assert a directional or quantitative insight (e.g., "GSK appears as a top sponsor
for Abrysvo despite Pfizer being the originator"), explicitly reference 1–3 supporting
NCT IDs in parentheses taken from the provided trial lists.
"""

    user_prompt = f"""
Prepare an integrated intelligence brief for the vaccine product "{vaccine_name}".
Data payload (JSON, partially abbreviated for length; all NCT IDs are real trials from clinicaltrials.gov):
{json.dumps(payload, indent=2, default=str)}

Please structure your answer as:

## 1. PRODUCT OVERVIEW
- Brief description of the vaccine (type, indication[s]) using any available context.
- Mention manufacturer / originator if known.

## 2. CLINICAL DEVELOPMENT LANDSCAPE
- Summarize the current trial footprint for this product.
- Highlight key sponsors running trials.
- Cite example NCT IDs in parentheses.

## 3. COMPETITOR VACCINES
- Identify notable competitor vaccines from the competitor_trials list.
- Compare development stages.

## 4. MARKET / REGULATORY CONTEXT (HIGH-LEVEL)
- Approval or launch status if clearly indicated.
- Say so if information is ambiguous or missing.

## 5. STRATEGIC TAKEAWAYS
- 3–5 concise bullets for actionable next steps.

## 6. KEY TRIAL ANNEX (NCT IDs)
- Bullet list of 8–15 informative trials.

Be explicit about what is inferred from clinicaltrials.gov vs general background.
Do NOT fabricate specific approval dates, exact sales numbers, or unpublished outcomes.
""" + ANTI_TRUNCATION_SUFFIX

    return _call_gemini(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=_get_token_budget("vaccine_intel"),
        temperature=0.35,
    )


def _infer_manufacturer_llm(vaccine_name: str) -> str:
    """Use Gemini Flash to dynamically infer a vaccine's originator company."""
    system_prompt = (
        "You are a pharmaceutical intelligence API. Given a vaccine pipeline name, "
        "return ONLY the name of the primary pharmaceutical company/originator that "
        "developed it. If you don't know, return exactly 'Unknown'. "
        "Do not add any conversational text or punctuation."
    )
    user_prompt = f"Vaccine name: {vaccine_name}"
    mfr, err = _call_gemini(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        model="models/gemini-1.5-flash-latest",
        temperature=0.0,
    )
    if mfr and mfr.strip().lower() != "unknown":
        return mfr.strip()
    return None


def _infer_vaccine_aliases_llm(vaccine_name: str) -> list:
    """Use Gemini Flash to infer all known aliases for a vaccine/drug.

    Returns a list of alias strings (brand name, INN, compound codes).
    Falls back to ``[vaccine_name]`` on any failure.
    """
    system_prompt = (
        "You are a pharmaceutical data extractor. Given a vaccine or drug name, "
        "return a JSON list of strings containing ALL known names for this product: "
        "the commercial brand name, generic/INN name, and all historical "
        "investigational compound codes. "
        "Output ONLY a valid JSON list."
    )
    user_prompt = f"Vaccine/drug name: {vaccine_name}"
    response, err = _call_gemini(
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": user_prompt}],
        model="models/gemini-1.5-flash-latest",
        temperature=0.0,
        response_mime_type="application/json"
    )
    if response:
        try:
            aliases = json.loads(response.strip())
            if isinstance(aliases, list) and aliases:
                result = [vaccine_name]
                for a in aliases:
                    if isinstance(a, str) and a.strip() and a.strip().lower() != vaccine_name.strip().lower():
                        result.append(a.strip())
                return result
        except (json.JSONDecodeError, ValueError):
            pass
    return [vaccine_name]


# ════════════════════════════════════════════════════════════════
#  HEAD-TO-HEAD VACCINE COMPARISON
# ════════════════════════════════════════════════════════════════

def _compare_vaccines_llm(vaccine_a_name: str, vaccine_a_data: dict,
                          vaccine_b_name: str, vaccine_b_data: dict):
    """AI-powered head-to-head comparison using pre-computed metrics + regulatory data.

    ``vaccine_a_data`` / ``vaccine_b_data`` should contain:
      - metrics: dict from ``compute_comparison_metrics``
      - fda: dict from ``fetch_openfda_data`` (or None)
      - manufacturer: str
      - sample_nct_ids: list[str] — up to 10 representative NCT IDs
    """

    system_prompt = """You are a senior vaccine competitive intelligence analyst presenting to the Head of Analytics.
Produce a concise, factual comparison of two vaccine products using ONLY the pre-computed metrics and regulatory data provided. 
Do NOT invent numbers. You must explicitly cite NCT IDs from the provided lists when making specific claims.

Your analysis must ruthlessly highlight the strategic delta between the two programs:
- Who has the clinical maturity advantage (completed Phase 3 volume)?
- Who has the active expansion advantage (highly recruiting trials)?
- What is the time-to-market differential based on FDA data?

Keep your response concise and complete — do NOT leave sections unfinished."""

    user_prompt = f"""Compare {vaccine_a_name} vs {vaccine_b_name}.

=== {vaccine_a_name} ===
{json.dumps(vaccine_a_data, indent=2, default=str)}

=== {vaccine_b_name} ===
{json.dumps(vaccine_b_data, indent=2, default=str)}

Structure your response as:

## EXECUTIVE COMPARISON
[3-4 sentences comparing clinical maturity, active pipeline expansion, and regulatory standing]

## KEY DIFFERENTIATORS
### {vaccine_a_name} Strengths
[3-4 concise bullets focusing on trial phase distribution, recruitment velocity, or geographic reach. Cite NCT IDs.]
### {vaccine_b_name} Strengths
[3-4 concise bullets focusing on trial phase distribution, recruitment velocity, or geographic reach. Cite NCT IDs.]

## REGULATORY STATUS
[Compare approval dates and FAERS data if available. Note who holds the first-mover advantage.]

## STRATEGIC IMPLICATIONS
[4-5 actionable bullets regarding clinical catch-up strategies, post-marketing safety monitoring, or label expansion opportunities.]""" + ANTI_TRUNCATION_SUFFIX

    return _call_gemini(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=_get_token_budget("comparison"),
        temperature=0.3,
    )


# ════════════════════════════════════════════════════════════════
#  PATIENT BURDEN SCORING  (NEW — was missing)
# ════════════════════════════════════════════════════════════════

def _score_patient_burden(nct_id: str, eligibility_criteria: str):
    """Use Gemini to score how burdensome a trial's eligibility criteria are
    for patient recruitment on a 1-10 scale, broken into three dimensions.

    Returns ``(score_dict, error_str)`` where ``score_dict`` has keys:

    - ``Overall_Burden_Score``  (int 1-10)
    - ``Primary_Bottleneck``    (str)
    - ``Dimensional_Analysis``  (dict with Demographic_Constraints,
      Comorbidity_Exclusions, Procedural_Complexity — each a dict
      with ``Level`` and ``Reason``)
    - ``Evidence_Quote``        (str — a verbatim snippet from the criteria)
    """
    if not eligibility_criteria or not eligibility_criteria.strip():
        return None, "No eligibility criteria text provided."

    # Truncate extremely long criteria to stay within token budgets
    criteria_snippet = eligibility_criteria[:6000]

    system_prompt = """You are a clinical operations feasibility analyst specializing in patient recruitment modeling.

Your task is to read a clinical trial's eligibility criteria and produce a structured JSON
assessment of how difficult it will be to recruit patients.  You score on three independent
dimensions, then produce an overall burden score.

Scoring guidelines:
- **Demographic Constraints**: How narrow is the age range? Are gender restrictions unusual?
  Does it exclude pregnant/breastfeeding women without justification?
- **Comorbidity Exclusions**: How many exclusion criteria rule out common comorbidities
  (diabetes, hypertension, obesity, autoimmune disease, prior cancers, etc.)?
- **Procedural Complexity**: How many visits, invasive procedures, washout periods,
  biopsies, lumbar punctures, prolonged inpatient stays are required?

Output ONLY a valid JSON object with exactly this structure (no markdown fences, no commentary):
{
  "Overall_Burden_Score": <integer 1-10>,
  "Primary_Bottleneck": "<short label>",
  "Dimensional_Analysis": {
    "Demographic_Constraints": { "Level": "Low|Medium|High|Severe", "Reason": "<1 sentence>" },
    "Comorbidity_Exclusions": { "Level": "Low|Medium|High|Severe", "Reason": "<1 sentence>" },
    "Procedural_Complexity":   { "Level": "Low|Medium|High|Severe", "Reason": "<1 sentence>" }
  },
  "Evidence_Quote": "<verbatim snippet from the criteria, max 200 chars>"
}"""

    user_prompt = f"""Trial NCT ID: {nct_id}

Eligibility Criteria:
---
{criteria_snippet}
---

Return ONLY the JSON object described in the system instructions.  No markdown fences."""

    raw, err = _call_gemini(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=_get_token_budget("default"),
        temperature=0.15,           # low temperature for structured extraction
        response_mime_type="application/json",
    )

    if err:
        return None, err
    if not raw:
        return None, "Empty response from Gemini for patient burden scoring."

    # ── Robust JSON parsing ──
    try:
        # Try direct parse first
        score_data = json.loads(raw.strip())
    except json.JSONDecodeError:
        # Sometimes Gemini wraps JSON in markdown fences despite instructions
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
        try:
            score_data = json.loads(cleaned)
        except json.JSONDecodeError:
            return None, f"Failed to parse Gemini response as JSON: {raw[:200]}..."

    # ── Validate required keys ──
    required_top = {"Overall_Burden_Score", "Primary_Bottleneck", "Dimensional_Analysis", "Evidence_Quote"}
    missing = required_top - set(score_data.keys())
    if missing:
        return None, f"Gemini response missing required keys: {missing}"

    dims = score_data.get("Dimensional_Analysis", {})
    required_dims = {"Demographic_Constraints", "Comorbidity_Exclusions", "Procedural_Complexity"}
    missing_dims = required_dims - set(dims.keys())
    if missing_dims:
        return None, f"Gemini response missing dimensional keys: {missing_dims}"

    # Ensure Overall_Burden_Score is an int
    try:
        score_data["Overall_Burden_Score"] = int(score_data["Overall_Burden_Score"])
    except (ValueError, TypeError):
        return None, "Overall_Burden_Score is not a valid integer."

    return score_data, None


# ════════════════════════════════════════════════════════════════
#  US SITE FEASIBILITY / WHITESPACE ANALYSIS  (NEW — was missing)
# ════════════════════════════════════════════════════════════════

def _analyze_us_site_feasibility(disease_name: str, feasibility_data: dict):
    """Use Gemini to interpret US site-saturation data and recommend low-competition
    whitespace cities + high-risk congested hospitals.

    ``feasibility_data`` is the dict returned by ``calculate_us_site_saturation``.

    Returns ``(analysis_dict, error_str)`` where ``analysis_dict`` has keys:

    - ``Executive_Verdict``              (str)
    - ``High_Risk_Hospitals``            (list[str])
    - ``Strategic_City_Whitespaces``     (list[str])
    - ``Operational_Recommendation``     (str)
    """
    if not feasibility_data or "error" in feasibility_data:
        return None, feasibility_data.get("error", "No feasibility data provided.")

    system_prompt = """You are a clinical trial site-feasibility strategist at a top-5 CRO.
You receive structured JSON describing US city-level trial saturation and congested hospitals
for a specific disease area.

Your job is to produce a concise, actionable JSON report that:
1. States an executive verdict on US site feasibility (one of: "Highly Favorable", "Moderate Competition", or "Severe Congestion").
2. Lists up to 5 high-risk hospitals where site fatigue is likely (from the Most_Congested_Hospitals map).
3. Lists up to 5 strategic whitespace cities where proven infrastructure exists but competition is low
   (from Low_Competition_Emerging_Hubs).  If there are fewer than 5, list all of them.
4. Provides a 2-3 sentence operational recommendation for a sponsor deciding where to place sites.

Output ONLY a valid JSON object with exactly this structure (no markdown fences, no commentary):
{
  "Executive_Verdict": "<Highly Favorable|Moderate Competition|Severe Congestion>",
  "High_Risk_Hospitals": ["Hospital A", "Hospital B"],
  "Strategic_City_Whitespaces": ["City X", "City Y"],
  "Operational_Recommendation": "<2-3 sentence recommendation>"
}"""

    user_prompt = f"""Disease/Indication: {disease_name}

US Site Saturation Data:
{json.dumps(feasibility_data, indent=2, default=str)}

Return ONLY the JSON object described in the system instructions.  No markdown fences."""

    raw, err = _call_gemini(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=_get_token_budget("default"),
        temperature=0.2,
        response_mime_type="application/json",
    )

    if err:
        return None, err
    if not raw:
        return None, "Empty response from Gemini for site feasibility analysis."

    # ── Robust JSON parsing ──
    try:
        analysis = json.loads(raw.strip())
    except json.JSONDecodeError:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
        try:
            analysis = json.loads(cleaned)
        except json.JSONDecodeError:
            return None, f"Failed to parse Gemini response as JSON: {raw[:200]}..."

    # ── Validate required keys ──
    required = {"Executive_Verdict", "High_Risk_Hospitals", "Strategic_City_Whitespaces", "Operational_Recommendation"}
    missing = required - set(analysis.keys())
    if missing:
        return None, f"Gemini response missing required keys: {missing}"

    # Ensure list fields are actually lists
    for list_key in ("High_Risk_Hospitals", "Strategic_City_Whitespaces"):
        if not isinstance(analysis.get(list_key), list):
            analysis[list_key] = [str(analysis[list_key])] if analysis.get(list_key) else []

    return analysis, None