import os
import json
import streamlit as st
import requests
import pandas as pd
import numpy as np
import re
import time
from collections import Counter
from packaging.version import Version
from dotenv import load_dotenv
from datetime import datetime
from io import BytesIO

load_dotenv() 

# ---------------- App config + options ----------------
st.set_page_config(page_title="Vaccine Pipeline Platform", page_icon="💉", layout="wide")

DEFAULT_LLM_MODEL = "models/gemini-robotics-er-1.5-preview"  # or use your robotics model
MAX_TRIALS_FOR_SUMMARY = 10  # how many trials to summarize per call (keep moderate for latency)
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

# Available Gemini models (user-selectable via sidebar)
GEMINI_MODEL_OPTIONS = {
    "Robotics 1.5 Preview (default)": DEFAULT_LLM_MODEL,
    "Gemini 1.5 Pro (general analysis)": "models/gemini-1.5-pro-latest",
    "Gemini 1.5 Flash (fast, cheaper)": "models/gemini-1.5-flash-latest",
}

# Enhanced data extraction helpers
def _extract_enrollment(study):
    """Extract enrollment information."""
    proto = (study or {}).get("protocolSection", {}) or {}
    design = proto.get("designModule", {}) or {}
    enrollment = design.get("enrollmentInfo", {}) or {}
    return {
        "enrollment_count": enrollment.get("count"),
        "enrollment_type": enrollment.get("type"),
        "actual_enrollment": enrollment.get("actualEnrollment")
    }

def _extract_dates(study):
    """Extract study dates."""
    proto = (study or {}).get("protocolSection", {}) or {}
    status = proto.get("statusModule", {}) or {}
    dates = {
        "start_date": status.get("startDateStruct", {}).get("date") or status.get("startDate"),
        "completion_date": status.get("completionDateStruct", {}).get("date") or status.get("completionDate"),
        "first_posted": status.get("firstPostedDateStruct", {}).get("date") or status.get("firstPostedDate"),
        "last_update": status.get("lastUpdatePostedDateStruct", {}).get("date") or status.get("lastUpdatePostedDate"),
    }
    return dates

def _extract_locations(study):
    """Extract study locations."""
    proto = (study or {}).get("protocolSection", {}) or {}
    contacts = proto.get("contactsLocationsModule", {}) or {}
    locations = contacts.get("locations", []) or []
    return [{
        "name": loc.get("facility"),
        "city": loc.get("city"),
        "state": loc.get("state"),
        "country": loc.get("country")
    } for loc in locations if loc.get("facility")]

def _extract_design_details(study):
    """Extract study design information."""
    proto = (study or {}).get("protocolSection", {}) or {}
    design = proto.get("designModule", {}) or {}
    return {
        "study_type": design.get("studyType"),
        "allocation": design.get("allocation"),
        "intervention_model": design.get("interventionModel"),
        "masking": design.get("maskingInfo", {}).get("masking") if design.get("maskingInfo") else None,
        "primary_purpose": design.get("primaryPurpose"),
        "number_of_arms": len(design.get("armsInterventionsModule", {}).get("armGroups", []) or [])
    }

def _extract_eligibility(study):
    """Extract eligibility criteria."""
    proto = (study or {}).get("protocolSection", {}) or {}
    eligibility = proto.get("eligibilityModule", {}) or {}
    return {
        "criteria": eligibility.get("eligibilityCriteria"),
        "gender": eligibility.get("gender"),
        "minimum_age": eligibility.get("minimumAge"),
        "maximum_age": eligibility.get("maximumAge"),
        "healthy_volunteers": eligibility.get("healthyVolunteers")
    }

def _extract_collaborators(study):
    """Extract collaborators."""
    proto = (study or {}).get("protocolSection", {}) or {}
    sponsor = proto.get("sponsorCollaboratorsModule", {}) or {}
    collaborators = sponsor.get("collaborators", []) or []
    return [col.get("name", "") for col in collaborators if col.get("name")]

def _extract_results_summary(study):
    """Extract results summary if available."""
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
        "adverse_events": adverse.get("events", [])
    }

def _fetch_pubmed_articles_for_trial(nct_id: str, trial_data: dict) -> list:
    """Fetch PubMed publications using explicit PMIDs from the trial record, with fallback to search."""
    try:
        pmids = []
        # First try to get explicit PMIDs from the clinical trial record's reference modules
        proto = trial_data.get("protocolSection", {}) or {}
        refs_mod = proto.get("referencesModule", {}) or {}
        for ref in refs_mod.get("references", []):
            if ref.get("pmid"):
                pmids.append(str(ref["pmid"]))
                
        # Remove duplicates while preserving order
        pmids = list(dict.fromkeys(pmids))
        
        # If no explicit PMIDs found, fallback to searching PubMed for the NCT ID
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
                    # Clean up random html-like encoding sometimes from PubMed
                    title = title.replace("&lt;", "<").replace("&gt;", ">").replace("<i>", "").replace("</i>", "")
                articles.append({
                    "pmid": pmid,
                    "title": title,
                    "source": meta.get("source", ""),
                    "pubdate": meta.get("pubdate", ""),
                    "authors": [a.get("name", "") for a in meta.get("authors", [])][:3]
                })
        return articles
    except Exception:
        return []

# ---------------- Export Functions ----------------
def generate_pdf_summary(summary_text: str, title: str = "Vaccine Pipeline Summary"):
    """Generate professionally formatted PDF from summary text."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
        from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
        from reportlab.lib.colors import HexColor
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer, 
            pagesize=letter, 
            topMargin=0.75*inch, 
            bottomMargin=0.75*inch,
            leftMargin=0.75*inch,
            rightMargin=0.75*inch
        )
        
        styles = getSampleStyleSheet()
        
        # Professional title style
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=22,
            textColor=HexColor('#1f4788'),
            spaceAfter=20,
            spaceBefore=0,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        # Section heading style (bold, larger, colored)
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=HexColor('#2c5aa0'),
            spaceAfter=10,
            spaceBefore=20,
            fontName='Helvetica-Bold',
            leftIndent=0
        )
        
        # Subsection style
        subheading_style = ParagraphStyle(
            'SubHeading',
            parent=styles['Heading3'],
            fontSize=13,
            textColor=HexColor('#3d6bb3'),
            spaceAfter=8,
            spaceBefore=12,
            fontName='Helvetica-Bold',
            leftIndent=0
        )
        
        # Normal text style
        normal_style = ParagraphStyle(
            'NormalText',
            parent=styles['Normal'],
            fontSize=11,
            leading=14,
            alignment=TA_JUSTIFY,
            spaceAfter=6
        )
        
        # Bullet point style
        bullet_style = ParagraphStyle(
            'BulletText',
            parent=normal_style,
            leftIndent=20,
            bulletIndent=10,
            spaceAfter=4
        )
        
        story = []
        
        # Title
        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph(f"<i>Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</i>", 
                              ParagraphStyle('DateStyle', parent=normal_style, alignment=TA_CENTER, fontSize=9)))
        story.append(Spacer(1, 0.3*inch))
        
        # Parse and format summary text
        import re
        
        # Helper function to safely convert markdown to HTML
        def markdown_to_html(text):
            """Safely convert markdown bold to HTML bold tags."""
            if not text:
                return text
            # Handle **bold** (non-greedy, multiple occurrences)
            text = re.sub(r'\*\*([^*]+?)\*\*', r'<b>\1</b>', text)
            # Handle __bold__ (non-greedy, multiple occurrences)
            text = re.sub(r'__([^_]+?)__', r'<b>\1</b>', text)
            # Remove any remaining markdown artifacts
            text = text.replace('**', '').replace('__', '')
            # Validate HTML tags - ensure all <b> have matching </b>
            open_tags = text.count('<b>')
            close_tags = text.count('</b>')
            if open_tags != close_tags:
                # Fix unclosed tags by removing all bold tags if mismatched
                text = re.sub(r'<b>|</b>', '', text)
            return text
        
        lines = summary_text.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if not line:
                story.append(Spacer(1, 0.05*inch))
                i += 1
                continue
            
            # Main heading (##)
            if line.startswith('##'):
                heading_text = line.replace('##', '').strip()
                # Remove all markdown formatting from headings
                heading_text = re.sub(r'\*\*|__', '', heading_text)
                story.append(Paragraph(heading_text, heading_style))
                i += 1
            
            # Subheading (#)
            elif line.startswith('#'):
                heading_text = line.replace('#', '').strip()
                # Remove all markdown formatting from headings
                heading_text = re.sub(r'\*\*|__', '', heading_text)
                story.append(Paragraph(heading_text, subheading_style))
                i += 1
            
            # Bullet points
            elif line.startswith('-') or line.startswith('•') or line.startswith('*'):
                bullet_text = line.lstrip('-•*').strip()
                # Convert markdown bold to HTML bold with proper regex
                import re
                # Handle **bold** (non-greedy, multiple occurrences)
                bullet_text = re.sub(r'\*\*([^*]+?)\*\*', r'<b>\1</b>', bullet_text)
                # Handle __bold__ (non-greedy, multiple occurrences)
                bullet_text = re.sub(r'__([^_]+?)__', r'<b>\1</b>', bullet_text)
                # Remove any remaining markdown artifacts
                bullet_text = bullet_text.replace('**', '').replace('__', '')
                # Escape any problematic HTML characters
                bullet_text = bullet_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                # Re-apply bold tags (now safe)
                bullet_text = re.sub(r'&lt;b&gt;([^&]+?)&lt;/b&gt;', r'<b>\1</b>', bullet_text)
                story.append(Paragraph(f"• {bullet_text}", bullet_style))
                i += 1
            
            # Regular text
            else:
                # Convert markdown bold to HTML bold tags
                text = line
                import re
                # Handle **bold** (non-greedy, multiple occurrences)
                text = re.sub(r'\*\*([^*]+?)\*\*', r'<b>\1</b>', text)
                # Handle __bold__ (non-greedy, multiple occurrences)
                text = re.sub(r'__([^_]+?)__', r'<b>\1</b>', text)
                # Remove any remaining markdown artifacts
                text = text.replace('**', '').replace('__', '')
                # Escape HTML characters but preserve our bold tags
                text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                # Re-apply bold tags (now safe)
                text = re.sub(r'&lt;b&gt;([^&]+?)&lt;/b&gt;', r'<b>\1</b>', text)
                story.append(Paragraph(text, normal_style))
                i += 1
        
        doc.build(story)
        buffer.seek(0)
        return buffer
    except ImportError:
        # Fallback to simple text-based PDF
        try:
            from fpdf import FPDF
            
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.set_fill_color(31, 71, 136)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(0, 12, title, ln=1, align='C', fill=True)
            pdf.set_text_color(0, 0, 0)
            pdf.ln(10)
            pdf.set_font("Arial", size=10)
            
            for line in summary_text.split('\n'):
                line = line.strip()
                if line:
                    if line.startswith('##'):
                        pdf.ln(5)
                        pdf.set_font("Arial", 'B', 14)
                        clean_line = line.replace('##', '').replace('**', '').replace('__', '').strip()
                        pdf.cell(0, 8, clean_line, ln=1)
                        pdf.set_font("Arial", size=10)
                    elif line.startswith('#'):
                        pdf.ln(3)
                        pdf.set_font("Arial", 'B', 12)
                        clean_line = line.replace('#', '').replace('**', '').replace('__', '').strip()
                        pdf.cell(0, 7, clean_line, ln=1)
                        pdf.set_font("Arial", size=10)
                    else:
                        clean_line = line.replace('**', '').replace('__', '')
                        pdf.multi_cell(0, 5, clean_line)
            
            buffer = BytesIO()
            buffer.write(pdf.output(dest='S').encode('latin-1'))
            buffer.seek(0)
            return buffer
        except ImportError:
            return None


# Try to force the legacy dataframe serializer (ignored if not supported in your version)
try:
    st.set_option("global.dataFrameSerialization", "legacy")
except Exception:
    pass

def _supports_width_string() -> bool:
    """True if st.dataframe supports width='stretch'/'content' (Streamlit >= 1.39)."""
    try:
        return Version(st.__version__) >= Version("1.39.0")
    except Exception:
        return False

# ---------------- Utilities ----------------
def _norm_txt(s: str) -> str:
    """Normalize a text string (lowercase, alnum, single spaces)."""
    if not s:
        return ""
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

def _request_json(url, params=None, session=None, retries=3, timeout=30, headers=None):
    """GET JSON with basic retry + 429 handling."""
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

# ---------------- LLM helpers (OpenRouter) ----------------
def _get_secret(name: str):
    try:
        return st.secrets.get(name)  # type: ignore[attr-defined]
    except Exception:
        return None


def _get_gemini_key():
    key = _get_secret("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        return None, "Set GEMINI_API_KEY in Streamlit secrets or environment."
    return key, None


def _call_gemini(messages, model=None, max_tokens=900, temperature=0.25):
    api_key, err = _get_gemini_key()
    if not api_key:
        return None, err

    # Get model name (priority: explicit arg -> sidebar selection -> secrets/env -> default)
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
    
    # Convert OpenAI-style messages to Gemini format
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
    
    # Build payload
    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }
    }
    
    if system_instruction:
        payload["systemInstruction"] = {
            "parts": [{"text": system_instruction}]
        }
    
    # Build URL
    url = f"{GEMINI_BASE_URL}/{model_name}:generateContent?key={api_key}"
    
    headers = {
        "Content-Type": "application/json",
    }

    try:
        # Slightly lower read timeout so we fail fast on slow free-tier responses
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        # Extract response from Gemini format
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

def _summarize_trials_with_llm(records: list, context_instructions: str):
    """Summarize a list of trials using OpenRouter-backed LLM with enhanced prompts."""
    trials = records[:MAX_TRIALS_FOR_SUMMARY]
    if not trials:
        return None, "Nothing to summarize."

    # Build rich trial summaries with all available data
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
        # Add enrollment if available
        if t.get('Enrollment'):
            enroll = t.get('Enrollment', {})
            if enroll.get('actual_enrollment') or enroll.get('enrollment_count'):
                detail += f"- Enrollment: {enroll.get('actual_enrollment') or enroll.get('enrollment_count', 'N/A')}\n"
        
        # Add dates if available
        if t.get('Dates'):
            dates = t.get('Dates', {})
            if dates.get('start_date'):
                detail += f"- Start Date: {dates.get('start_date')}\n"
            if dates.get('completion_date'):
                detail += f"- Completion Date: {dates.get('completion_date')}\n"
        
        # Add design if available
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

Format the response in clear, professional language suitable for C-suite presentations."""

    return _call_gemini(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=3500,
        temperature=0.3
    )


def _summarize_single_trial(nct_id: str, details: dict):
    """Summarize one trial with OpenRouter-backed LLM using comprehensive data."""
    
    # Build comprehensive payload
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
        "pubmed_articles": details.get("PubMed_Articles", [])
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

Format professionally with clear sections and bullet points where appropriate."""

    return _call_gemini(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=2500,
        temperature=0.3
    )

def _fetch_wikipedia_summary(term: str, lang: str = "en") -> str | None:
    """
    Fetch a short encyclopedic summary for a vaccine or disease from Wikipedia.
    This is used only as high-level context for Gemini and is optional.
    """
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
        extract = data.get("extract")
        return extract
    except Exception:
        return None

def _vaccine_intel_summary(
    vaccine_name: str,
    manufacturer: str | None,
    diseases: list[str],
    vaccine_trials: list[dict],
    competitor_trials: list[dict],
    external_context: str | None,
):
    """
    High-level Gemini intelligence brief combining:
    - Product metadata (name, manufacturer, diseases)
    - ClinicalTrials.gov trials for this product
    - Competitor vaccine trials in same diseases
    - Optional external encyclopedic context (e.g., Wikipedia)
    """
    if not vaccine_trials and not competitor_trials:
        return None, "No trials to analyze for this vaccine yet."

    # Keep context reasonably sized for free-tier latency
    max_self = 20
    max_comp = 20
    self_trials = vaccine_trials[:max_self]
    comp_trials = competitor_trials[:max_comp]

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
- Summarize the current trial footprint for this product (phases, status mix, geographies if visible).
- Highlight key sponsors running trials (note: sponsors may differ from manufacturer).
- Explicitly discuss any divergence between originator/manufacturer and top sponsors, and explain scenarios
  like head‑to‑head or real‑world studies where competitors (e.g., GSK) or academic centers run trials that
  still use this product.
- In this section, whenever you describe a pattern (e.g., "most trials are Phase 3 and completed"), cite
  example NCT IDs in parentheses, such as (e.g., NCT01234567, NCT08976543).

## 3. COMPETITOR VACCINES
- Identify notable competitor vaccines from the competitor_trials list (by product name and sponsor).
- Compare approximate development stage (phases and status) of this product vs key competitors.

## 4. MARKET / REGULATORY CONTEXT (HIGH-LEVEL)
- Using external_context only as high‑level background, describe approval or launch status
  (e.g., approved, under review, still developmental) if that is clearly indicated.
- If information is ambiguous or missing, say so instead of guessing.

## 5. STRATEGIC TAKEAWAYS
- 3–5 concise bullets for how a user of a vaccine intelligence platform could use this information
  (e.g., prioritizing indications, watching certain competitors, data gaps to track).

## 6. KEY TRIAL ANNEX (NCT IDs)
- Provide a bullet list of 8–15 of the most informative trials (mix of this product and key competitors).
- For each trial, include at least: NCT ID, main vaccine(s), phase, status, and lead sponsor.

Be explicit about what is inferred primarily from clinicaltrials.gov trial metadata vs general background.
Do NOT fabricate specific approval dates, exact sales numbers, or unpublished outcomes.
"""

    return _call_gemini(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=3500,
        temperature=0.35,
    )

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
    """
    True if the normalized target name matches any intervention name
    exactly or as a standalone token (not just arbitrary substring).
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
            if norm == t or t in tokens:
                return True
    return False

# Minimal curated synonym + manufacturer sets for major vaccines.
_VACCINE_SYNONYM_GROUPS = [
    # Pfizer / BioNTech – COVID-19
    [
        "Comirnaty",
        "BNT162b2",
        "Tozinameran",
        "Pfizer-BioNTech COVID-19 vaccine",
        "Pfizer-BioNTech mRNA COVID-19 vaccine",
    ],
    # Moderna – COVID-19
    [
        "Spikevax",
        "mRNA-1273",
        "mRNA 1273",
        "Elasomeran",
        "Moderna COVID-19 vaccine",
    ],
    # Pfizer – RSV
    [
        "Abrysvo",
        "RSVpreF",
        "RSVpreF3",
        "bivalent RSVpreF3",
        "Pfizer RSVpreF vaccine",
    ],
    # GSK – RSV
    [
        "Arexvy",
        "respiratory syncytial virus vaccine recombinant adjuvanted",
        "GSK RSV vaccine",
    ],
    # Pfizer – pneumococcal
    [
        "Prevnar 13",
        "Prevenar 13",
        "13-valent pneumococcal conjugate vaccine",
        "PCV13",
    ],
    [
        "Prevnar 20",
        "20-valent pneumococcal conjugate vaccine",
        "PCV20",
    ],
]

_VACCINE_MANUFACTURER_DATA = {
    "Pfizer": [
        "Comirnaty",
        "BNT162b2",
        "Tozinameran",
        "Abrysvo",
        "RSVpreF",
        "RSVpreF3",
        "Prevnar 13",
        "Prevenar 13",
        "13-valent pneumococcal conjugate vaccine",
        "PCV13",
        "Prevnar 20",
        "20-valent pneumococcal conjugate vaccine",
        "PCV20",
    ],
    "BioNTech": [
        "Comirnaty",
        "BNT162b2",
        "Tozinameran",
    ],
    "Moderna": [
        "Spikevax",
        "mRNA-1273",
        "mRNA 1273",
        "Elasomeran",
        "Moderna COVID-19 vaccine",
    ],
    "GSK": [
        "Arexvy",
        "respiratory syncytial virus vaccine recombinant adjuvanted",
        "GSK RSV vaccine",
    ],
}

LEARNED_MFR_FILE = "learned_manufacturers.json"

def _load_learned_manufacturers():
    if os.path.exists(LEARNED_MFR_FILE):
        try:
            with open(LEARNED_MFR_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}

LEARNED_MFRS = _load_learned_manufacturers()

def _save_learned_manufacturer(vaccine_name, mfr):
    learned = _load_learned_manufacturers()
    norm = _norm_txt(vaccine_name)
    if norm:
        learned[norm] = mfr
        try:
            with open(LEARNED_MFR_FILE, 'w') as f:
                json.dump(learned, f, indent=2)
        except Exception:
            pass

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
            # Prefer first mapping if collisions ever occur
            idx.setdefault(norm, mfr)
    
    # Add dynamically learned mappings from LLM fallback
    for norm, mfr in LEARNED_MFRS.items():
        idx.setdefault(norm, mfr)
        
    return idx

_VACCINE_SYNONYM_INDEX = _build_vaccine_synonym_index()
_VACCINE_MANUFACTURER_INDEX = _build_vaccine_manufacturer_index()

def _get_vaccine_search_terms(raw_name: str) -> list:
    """
    Return a list of intervention search terms for a vaccine, including
    curated synonyms (brand name, code, INN) where available.
    """
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

@st.cache_data(ttl=86400, show_spinner=False)
def _infer_manufacturer_llm(vaccine_name: str) -> str:
    """Use Gemini to dynamically infer the lead sponsor/originator of an unknown vaccine."""
    system_prompt = "You are a pharmaceutical intelligence API. Given a vaccine pipeline name, return ONLY the name of the primary pharmaceutical company/originator that developed it. If you don't know, return exactly 'Unknown'. Do not add any conversational text or punctuation."
    user_prompt = f"Vaccine name: {vaccine_name}"
    mfr, err = _call_gemini([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ], model="models/gemini-1.5-flash-latest", temperature=0.0)
    
    if mfr and mfr.strip().lower() != "unknown":
        return mfr.strip()
    return None

def _get_vaccine_manufacturer(raw_name: str, use_llm_fallback: bool = True):
    """Return a single primary manufacturer name for a vaccine, falling back to LLM inference if unknown."""
    if not raw_name:
        return None
    # Try direct norm lookup
    norm = _norm_txt(raw_name)
    if norm in _VACCINE_MANUFACTURER_INDEX:
        return _VACCINE_MANUFACTURER_INDEX[norm]
    # Try through synonyms
    syns = _VACCINE_SYNONYM_INDEX.get(norm, [])
    for s in syns:
        s_norm = _norm_txt(s)
        mfr = _VACCINE_MANUFACTURER_INDEX.get(s_norm)
        if mfr:
            return mfr
            
    # LLM Dynamic Fallback
    if use_llm_fallback:
        inferred = _infer_manufacturer_llm(raw_name)
        if inferred:
            _save_learned_manufacturer(raw_name, inferred)
            _VACCINE_MANUFACTURER_INDEX[_norm_txt(raw_name)] = inferred
            return inferred
        
    return None

def _mesh_terms_intervention(study) -> list:
    """Return MeSH terms for interventions (used to detect vaccines)."""
    derived = (study or {}).get("derivedSection", {}) or {}
    iv_browse = derived.get("interventionBrowseModule", {}) or {}
    leaves = iv_browse.get("browseLeaves", []) or []
    return [x.get("meshTerm", "") for x in leaves if x.get("meshTerm")]

def _mesh_terms_condition(study) -> list:
    """Return MeSH terms for conditions (helps disease normalization)."""
    derived = (study or {}).get("derivedSection", {}) or {}
    cond_browse = derived.get("conditionBrowseModule", {}) or {}
    leaves = cond_browse.get("browseLeaves", []) or []
    return [x.get("meshTerm", "") for x in leaves if x.get("meshTerm")]

def _primary_outcomes_from_protocol(study) -> list:
    """Primary outcomes from protocol (exists even when no results posted)."""
    proto = (study or {}).get("protocolSection", {}) or {}
    out_mod = proto.get("outcomesModule", {}) or {}
    out = []
    for o in out_mod.get("primaryOutcomes", []) or []:
        out.append({
            "Title": o.get("measure") or o.get("title") or "",
            "Description": o.get("description", "") or o.get("timeFrame", "") or ""
        })
    return out

def _results_outcomes(study) -> list:
    """Outcomes from posted results (if available)."""
    results = (study or {}).get("resultsSection", {}) or {}
    res_mod = results.get("outcomeMeasuresModule", {}) or {}
    out = []
    for o in res_mod.get("outcomeMeasures", []) or []:
        out.append({
            "Title": o.get("title") or "",
            "Description": o.get("description", "") or ""
        })
    return out

def _is_vaccine_study(study) -> bool:
    """
    Classify study as vaccine:
    1) Intervention MeSH terms include 'Vaccine' (most reliable)
    2) Biological intervention with name/otherNames mentioning 'vaccine'
    3) Outcomes text contains immunogenicity/antibody keywords
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
        txt = " ".join([
            _norm_txt(i.get("name", "")),
            _norm_txt(" ".join(other))
        ])
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

def show_df(df: pd.DataFrame, height: int = 420):
    """Render an interactive table robustly across Streamlit/Arrow versions."""
    # Convert all columns to pure strings to completely bypass PyArrow schema inference errors
    safe_df = df.copy()
    for col in safe_df.columns:
        safe_df[col] = safe_df[col].apply(lambda x: "" if pd.isna(x) else str(x))

    try:
        if _supports_width_string():
            st.dataframe(safe_df, width="stretch", height=height)
        else:
            st.dataframe(safe_df, use_container_width=True, height=height)
    except Exception as e:
        st.warning(f"Interactive table failed: {e}. Showing static table instead.")
        st.table(safe_df.head(50))

# ---------------- Data fetchers (cached) ----------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all_vaccine_trials(disease: str, max_pages: int = 10):
    """
    Fetch vaccine trials by disease.
    Strategy: search by disease, page through, classify vaccines client-side.
    Fallback: if none found, add query.term=vaccin*.
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
                status = proto.get("statusModule", {}) or {}
                sponsor = proto.get("sponsorCollaboratorsModule", {}) or {}

                nct_id = ident.get("nctId")
                if not nct_id or nct_id in seen:
                    continue
                seen.add(nct_id)

                title = ident.get("briefTitle") or ident.get("officialTitle") or "No title"
                phases = design.get("phases") or ["Not reported"]
                overall_status = status.get("overallStatus") or "Unknown"
                sponsor_name = sponsor.get("leadSponsor", {}).get("name", "Unknown")
                vaccines = _extract_vaccine_names(s)

                all_results.append({
                    "NCT ID": str(nct_id),
                    "Title": str(title),
                    "Phase": ", ".join(phases),
                    "Status": str(overall_status),
                    "Sponsor": str(sponsor_name),
                    "Vaccines": ", ".join(vaccines) if vaccines else "Not reported"
                })

            page_token = data.get("nextPageToken")
            page_count += 1
            if not page_token:
                break
            time.sleep(0.25)

        # Fallback: no vaccine trials detected -> broaden text search
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
                    status = proto.get("statusModule", {}) or {}
                    sponsor = proto.get("sponsorCollaboratorsModule", {}) or {}
                    nct_id = ident.get("nctId")
                    if not nct_id or nct_id in seen:
                        continue
                    seen.add(nct_id)
                    vaccines = _extract_vaccine_names(s)
                    all_results.append({
                        "NCT ID": str(nct_id),
                        "Title": str(ident.get("briefTitle") or ident.get("officialTitle") or "No title"),
                        "Phase": ", ".join(design.get("phases") or ["Not reported"]),
                        "Status": str(status.get("overallStatus") or "Unknown"),
                        "Sponsor": str(sponsor.get("leadSponsor", {}).get("name", "Unknown")),
                        "Vaccines": ", ".join(vaccines) if vaccines else "Not reported"
                    })

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
    """Fetch detailed trial info with comprehensive data extraction."""
    url = f"https://clinicaltrials.gov/api/v2/studies/{nct_id}"
    try:
        data = _request_json(url, params=None)
        proto = data.get("protocolSection", {}) or {}
        design = proto.get("designModule", {}) or {}
        status = proto.get("statusModule", {}) or {}
        sponsor = proto.get("sponsorCollaboratorsModule", {}) or {}
        conditions = proto.get("conditionsModule", {}) or {}

        vaccines = _extract_vaccine_names(data)

        # Conditions: protocol + MeSH
        disease_list = []
        disease_list.extend(conditions.get("conditions", []) or [])
        disease_list.extend(_mesh_terms_condition(data))
        seen_d = set()
        disease_list = [d for d in disease_list if d and not (d in seen_d or seen_d.add(d))]

        # Outcomes
        outcomes = _results_outcomes(data)
        if not outcomes:
            outcomes = _primary_outcomes_from_protocol(data)

        ident = proto.get("identificationModule", {}) or {}
        title = ident.get("briefTitle") or ident.get("officialTitle") or "No title"
        phases = design.get("phases") or ["Not reported"]
        overall_status = status.get("overallStatus", "Unknown")
        sponsor_name = sponsor.get("leadSponsor", {}).get("name", "Unknown")

        # Enhanced data extraction
        enrollment = _extract_enrollment(data)
        dates = _extract_dates(data)
        locations = _extract_locations(data)
        design_details = _extract_design_details(data)
        eligibility = _extract_eligibility(data)
        collaborators = _extract_collaborators(data)
        results_summary = _extract_results_summary(data)
        pubmed_articles = _fetch_pubmed_articles_for_trial(nct_id, data)

        return {
            "NCT ID": nct_id,
            "Title": str(title),
            "Phase": ", ".join(phases),
            "Status": str(overall_status),
            "Sponsor": str(sponsor_name),
            "Vaccines": sorted(vaccines) if vaccines else ["Not reported"],
            "Diseases": disease_list,
            "Outcomes": outcomes,
            "Enrollment": enrollment,
            "Dates": dates,
            "Locations": locations,
            "Design": design_details,
            "Eligibility": eligibility,
            "Collaborators": collaborators,
            "Results": results_summary,
            "PubMed_Articles": pubmed_articles
        }
    except requests.RequestException:
        return None

# ---------------- Visualization Functions ----------------
def create_phase_chart(df: pd.DataFrame):
    """Create phase distribution chart."""
    try:
        import plotly.express as px
        phase_counts = {}
        for phases in df["Phase"].dropna():
            for phase in str(phases).split(","):
                phase = phase.strip()
                phase_counts[phase] = phase_counts.get(phase, 0) + 1
        
        if not phase_counts:
            return None
        
        phase_df = pd.DataFrame(list(phase_counts.items()), columns=["Phase", "Count"])
        fig = px.bar(phase_df, x="Phase", y="Count", title="Phase Distribution", 
                    color="Count", color_continuous_scale="Blues")
        fig.update_layout(showlegend=False, height=300)
        return fig
    except ImportError:
        return None

def create_status_chart(df: pd.DataFrame):
    """Create status distribution chart."""
    try:
        import plotly.express as px
        status_counts = df["Status"].value_counts()
        if status_counts.empty:
            return None
        
        fig = px.pie(values=status_counts.values, names=status_counts.index, 
                    title="Trial Status Distribution")
        fig.update_layout(height=300)
        return fig
    except ImportError:
        return None

def create_sponsor_chart(df: pd.DataFrame, top_n=10):
    """Create top sponsors chart."""
    try:
        import plotly.express as px
        sponsor_counts = df["Sponsor"].value_counts().head(top_n)
        if sponsor_counts.empty:
            return None
        
        fig = px.bar(x=sponsor_counts.values, y=sponsor_counts.index, 
                    orientation='h', title=f"Top {top_n} Sponsors",
                    labels={'x': 'Number of Trials', 'y': 'Sponsor'})
        # Force integer tick marks so counts never appear as fractional
        fig.update_layout(height=400)
        fig.update_xaxes(dtick=1, rangemode="tozero")
        return fig
    except ImportError:
        return None

# ---------------- Main UI ----------------
st.title("💉 Vaccine Pipeline Platform")
st.markdown("Explore complete vaccine trial data from ClinicalTrials.gov. Search by disease condition or vaccine product name with competitor analysis.")

# Sidebar: Gemini model selection (single control for all intelligence features)
st.sidebar.markdown("### 🧠 Gemini model")
model_label = st.sidebar.selectbox(
    "Model for all AI summaries",
    options=list(GEMINI_MODEL_OPTIONS.keys()),
    index=0,
    key="gemini_model_label",
)
st.session_state["gemini_model"] = GEMINI_MODEL_OPTIONS.get(model_label, DEFAULT_LLM_MODEL)

# Ensure state keys exist
for k in ["studies", "vaccine_trials", "competitor_trials", "target_vaccine", "target_diseases"]:
    st.session_state.setdefault(k, [] if "trials" in k or "studies" in k or "diseases" in k else "")

tab1, tab2 = st.tabs(["🔍 Search by Disease", "💊 Search by Vaccine Product"])

# ---------------- TAB 1: Search by Disease ----------------
with tab1:
    st.subheader("Search Vaccine Trials by Disease")
    st.caption("Fetches trials by disease and classifies vaccines using MeSH and heuristics.")

    disease = st.text_input("Enter Disease Name", value="RSV", key="disease_input")

    if st.button("🔍 Fetch All Trials", key="fetch_disease"):
        with st.spinner("Fetching all vaccine trials (this may take a moment)..."):
            studies = fetch_all_vaccine_trials(disease, max_pages=10)
            if not studies:
                st.warning(f"No vaccine studies found for '{disease}'. Try another disease or broader term.")
                st.session_state["studies"] = []
            else:
                st.session_state["studies"] = studies
                st.success(f"✅ Found {len(studies)} vaccine trials for {disease}.")

    studies = st.session_state.get("studies", [])

    if studies:
        df = pd.DataFrame(studies)

        # Sidebar filters (Disease Search) - single filter set here
        st.sidebar.header("🎛️ Filters (Disease Search)")
        phase_options = sorted({p.strip() for val in df["Phase"].dropna() for p in str(val).split(",")})
        status_options = sorted([s for s in df["Status"].dropna().unique()])

        selected_phases = st.sidebar.multiselect("Phase", options=phase_options, default=phase_options, key="phase_filter_disease")
        selected_status = st.sidebar.multiselect("Status", options=status_options, default=status_options, key="status_filter_disease")

        def _row_has_phase(ph_str: str, selected: list) -> bool:
            row_phases = [p.strip() for p in str(ph_str).split(",")]
            return any(p in row_phases or p in ph_str for p in selected) if selected else True

        df_filtered = df[df["Phase"].apply(lambda x: _row_has_phase(x, selected_phases))]
        if selected_status:
            df_filtered = df_filtered[df_filtered["Status"].isin(selected_status)]

        st.info(f"📊 Showing {len(df_filtered)} of {len(studies)} trials")
        
        # Visualizations
        col1, col2 = st.columns(2)
        with col1:
            phase_chart = create_phase_chart(df_filtered)
            if phase_chart:
                st.plotly_chart(phase_chart, use_container_width=True)
        with col2:
            status_chart = create_status_chart(df_filtered)
            if status_chart:
                st.plotly_chart(status_chart, use_container_width=True)
        
        show_df(df_filtered, height=420)

        if st.button("🧠 Summarize Displayed Trials", key="summarize_disease_trials"):
            with st.spinner("Generating AI executive summary..."):
                summary_txt, summary_err = _summarize_trials_with_llm(
                    df_filtered.to_dict("records"),
                    context_instructions=f"Disease search term: {disease}. Showing {len(df_filtered)} of {len(df)} vaccine trials."
                )
            if summary_txt:
                st.markdown("#### 🤖 AI Executive Summary")
                st.write(summary_txt)
                
                # Export button
                pdf_buffer = generate_pdf_summary(summary_txt, f"Vaccine Pipeline Summary - {disease}")
                if pdf_buffer:
                    st.download_button(
                        label="📄 Download PDF Report",
                        data=pdf_buffer,
                        file_name=f"vaccine_summary_{disease}_{datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf",
                        key=f"pdf_disease_{disease}",
                        use_container_width=True
                    )
            elif summary_err:
                st.warning(summary_err)

        selected_id = st.selectbox(
            "🔬 View Detailed Info",
            options=["Select a study..."] + [str(x) for x in df_filtered["NCT ID"].tolist()],
            key="select_disease_detail"
        )

        if selected_id != "Select a study...":
            with st.spinner("Loading details..."):
                details = fetch_trial_details_with_vaccines(selected_id)

            if details:
                st.markdown("---")
                st.subheader(f"📋 Study Details: {selected_id}")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Title:** {details['Title']}")
                    st.markdown(f"**Phase:** {details['Phase']}")
                    if details.get("Enrollment"):
                        enroll = details["Enrollment"]
                        if enroll.get("actual_enrollment"):
                            st.markdown(f"**Enrollment:** {enroll.get('actual_enrollment')}")
                        elif enroll.get("enrollment_count"):
                            st.markdown(f"**Target Enrollment:** {enroll.get('enrollment_count')}")
                with col2:
                    st.markdown(f"**Status:** {details['Status']}")
                    st.markdown(f"**Sponsor:** {details['Sponsor']}")
                    if details.get("Dates"):
                        dates = details["Dates"]
                        if dates.get("start_date"):
                            st.markdown(f"**Start Date:** {dates.get('start_date')}")
                        if dates.get("completion_date"):
                            st.markdown(f"**Completion Date:** {dates.get('completion_date')}")

                # Enhanced details
                if details.get("Design"):
                    design = details["Design"]
                    with st.expander("📐 Study Design Details"):
                        col_d1, col_d2 = st.columns(2)
                        with col_d1:
                            if design.get("study_type"):
                                st.markdown(f"**Study Type:** {design.get('study_type')}")
                            if design.get("allocation"):
                                st.markdown(f"**Allocation:** {design.get('allocation')}")
                            if design.get("intervention_model"):
                                st.markdown(f"**Intervention Model:** {design.get('intervention_model')}")
                        with col_d2:
                            if design.get("masking"):
                                st.markdown(f"**Masking:** {design.get('masking')}")
                            if design.get("primary_purpose"):
                                st.markdown(f"**Primary Purpose:** {design.get('primary_purpose')}")
                            if design.get("number_of_arms"):
                                st.markdown(f"**Number of Arms:** {design.get('number_of_arms')}")

                if details.get("Diseases"):
                    st.markdown("**🦠 Diseases/Conditions:**")
                    st.write(", ".join(details["Diseases"]))

                st.markdown("**💉 Vaccine Products:**")
                for v in details["Vaccines"]:
                    st.markdown(f"- {v}")

                if details.get("Locations"):
                    with st.expander("🌍 Study Locations"):
                        for loc in details["Locations"][:10]:  # Show first 10
                            loc_str = f"{loc.get('name', '')}"
                            if loc.get("city"):
                                loc_str += f", {loc.get('city')}"
                            if loc.get("state"):
                                loc_str += f", {loc.get('state')}"
                            if loc.get("country"):
                                loc_str += f", {loc.get('country')}"
                            st.write(f"• {loc_str}")

                if details.get("Eligibility"):
                    with st.expander("👥 Eligibility Criteria"):
                        elig = details["Eligibility"]
                        if elig.get("gender"):
                            st.markdown(f"**Gender:** {elig.get('gender')}")
                        if elig.get("minimum_age") or elig.get("maximum_age"):
                            age_range = f"{elig.get('minimum_age', 'N/A')} - {elig.get('maximum_age', 'N/A')}"
                            st.markdown(f"**Age Range:** {age_range}")
                        if elig.get("healthy_volunteers"):
                            st.markdown(f"**Healthy Volunteers:** {elig.get('healthy_volunteers')}")
                        if elig.get("criteria"):
                            st.markdown("**Criteria:**")
                            st.text(elig.get("criteria")[:500] + "..." if len(elig.get("criteria", "")) > 500 else elig.get("criteria"))

                if details.get("Collaborators"):
                    st.markdown("**🤝 Collaborators:**")
                    st.write(", ".join(details["Collaborators"]))

                if details["Outcomes"]:
                    st.markdown("**📊 Primary Outcome Measures:**")
                    for o in details["Outcomes"]:
                        st.write(f"• {o['Title']}")
                        if o["Description"]:
                            st.caption(o["Description"])
                else:
                    st.info("No outcomes reported yet.")

                if details.get("Results"):
                    st.success("✅ Results data available for this study")

                if details.get("PubMed_Articles"):
                    with st.expander(f"📚 PubMed Articles ({len(details['PubMed_Articles'])})"):
                        for pm in details["PubMed_Articles"]:
                            authors = ", ".join(pm["authors"]) + (" et al." if len(pm["authors"]) == 3 else "")
                            st.markdown(f"**[{pm['title']}](https://pubmed.ncbi.nlm.nih.gov/{pm['pmid']})**")
                            st.caption(f"{pm['source']} | {pm['pubdate']} | {authors}")

                if st.button("🧠 Summarize This Study", key=f"summarize_detail_{selected_id}"):
                    with st.spinner("Creating AI summary..."):
                        trial_summary, detail_err = _summarize_single_trial(selected_id, details)
                    if trial_summary:
                        st.markdown("#### 🤖 AI Trial Brief")
                        st.write(trial_summary)
                        
                        # Export button for single trial
                        pdf_buffer = generate_pdf_summary(trial_summary, f"Trial Brief - {selected_id}")
                        if pdf_buffer:
                            st.download_button(
                                label="📄 Download PDF Report",
                                data=pdf_buffer,
                                file_name=f"trial_brief_{selected_id}_{datetime.now().strftime('%Y%m%d')}.pdf",
                                mime="application/pdf",
                                key=f"pdf_trial_{selected_id}",
                                use_container_width=True
                            )
                    elif detail_err:
                        st.warning(detail_err)
    else:
        st.info("👆 Enter a disease name and click Fetch All Trials to begin.")

# ---------------- TAB 2: Search by Vaccine Product + Competitors ----------------
with tab2:
    st.subheader("Search Trials by Vaccine Product")
    st.caption("Find your vaccine’s trials + competitor vaccines targeting the same disease(s).")

    vaccine_name = st.text_input("Enter Vaccine Product Name", value="", key="vaccine_input")

    if st.button("💊 Search Vaccine & Competitors", key="fetch_vaccine"):
        if not vaccine_name.strip():
            st.warning("Please enter a vaccine name.")
        else:
            search_url = "https://clinicaltrials.gov/api/v2/studies"

            try:
                with st.spinner(f"Step 1/2: Finding trials for '{vaccine_name}'..."):
                    search_terms = _get_vaccine_search_terms(vaccine_name)
                    if not search_terms:
                        search_terms = [vaccine_name]
                    target_norms = [_norm_txt(t) for t in search_terms if _norm_txt(t)]

                    vaccine_results = []
                    all_diseases = []
                    seen_nct = set()

                    for term in search_terms:
                        params = {"query.intr": term, "pageSize": 100}
                        data = _request_json(search_url, params=params)
                        studies = data.get("studies", []) or []

                        for s in studies:
                            names = _extract_vaccine_names(s)
                            if not _matches_vaccine_name(names, target_norms):
                                continue
                            if not _is_vaccine_study(s):
                                continue
                            proto = s.get("protocolSection", {}) or {}
                            ident = proto.get("identificationModule", {}) or {}
                            design = proto.get("designModule", {}) or {}
                            status = proto.get("statusModule", {}) or {}
                            sponsor = proto.get("sponsorCollaboratorsModule", {}) or {}
                            nct_id = ident.get("nctId")
                            if not nct_id or nct_id in seen_nct:
                                continue
                            seen_nct.add(nct_id)
                            vaccine_results.append({
                                "NCT ID": str(nct_id),
                                "Title": str(ident.get("briefTitle") or ident.get("officialTitle") or "No title"),
                                "Phase": ", ".join(design.get("phases") or ["Not reported"]),
                                "Status": str(status.get("overallStatus") or "Unknown"),
                                "Sponsor": str(sponsor.get("leadSponsor", {}).get("name", "Unknown")),
                                "Vaccines": ", ".join(names) if names else "Not reported"
                            })
                            ds = []
                            ds.extend(proto.get("conditionsModule", {}).get("conditions", []) or [])
                            ds.extend(_mesh_terms_condition(s))
                            all_diseases.extend(ds)

                    st.session_state["target_vaccine"] = vaccine_name
                    st.session_state["vaccine_trials"] = vaccine_results

                    disease_counts = Counter([d for d in all_diseases if d])
                    top_diseases = [d for d, _ in disease_counts.most_common(2)]
                    st.session_state["target_diseases"] = top_diseases

                # Step 2: competitors
                competitor_trials = []
                top_diseases = st.session_state.get("target_diseases", [])
                if top_diseases:
                    with st.spinner(f"Step 2/2: Finding competitor vaccines for {', '.join(top_diseases)}..."):
                        seen = set()
                        for d in top_diseases:
                            trials = fetch_all_vaccine_trials(d, max_pages=5)
                            for t in trials:
                                vacc_norm = _norm_txt(t.get("Vaccines", ""))
                                if any(tn and (tn in vacc_norm) for tn in target_norms):
                                    continue
                                nct = t.get("NCT ID")
                                if nct and nct not in seen:
                                    seen.add(nct)
                                    competitor_trials.append(t)

                    st.session_state["competitor_trials"] = competitor_trials
                    st.success(f"✅ Found {len(vaccine_results)} trials for '{vaccine_name}' and {len(competitor_trials)} competitor trials!")
                else:
                    st.session_state["competitor_trials"] = []
                    st.success(f"✅ Found {len(st.session_state['vaccine_trials'])} trials for '{vaccine_name}' (no clear disease context detected)")
            except requests.RequestException as e:
                st.error(f"Search failed: {e}")
                st.session_state["vaccine_trials"] = []
                st.session_state["competitor_trials"] = []

    # Show vaccine trials (WITH its own filters)
    vaccine_trials = st.session_state.get("vaccine_trials", [])
    competitor_trials = st.session_state.get("competitor_trials", [])
    target_vaccine = st.session_state.get("target_vaccine", "")
    target_diseases = st.session_state.get("target_diseases", [])

    if vaccine_trials:
        st.markdown("---")
        st.subheader(f"🎯 Your Vaccine: {target_vaccine}")
        meta_bits = []
        if target_diseases:
            meta_bits.append(f"Primary Disease(s): {', '.join(target_diseases)}")
        mfr = _get_vaccine_manufacturer(target_vaccine)
        if mfr:
            meta_bits.append(f"Originator/Manufacturer: {mfr}")
        if meta_bits:
            st.caption(" | ".join(meta_bits))

        df_vaccine = pd.DataFrame(vaccine_trials)

        # Classify sponsor type relative to originator (if known)
        mfr_norm = _norm_txt(mfr) if (isinstance(mfr, str) and mfr) else None
        if mfr_norm:
            def _sponsor_type(name: str) -> str:
                n = _norm_txt(name or "")
                return "Originator / Manufacturer" if mfr_norm and mfr_norm in n else "External / Other"
            df_vaccine["Sponsor Type"] = df_vaccine["Sponsor"].apply(_sponsor_type)

        # Sidebar filters for YOUR vaccine trials (Tab 2)
        st.sidebar.header("🎛️ Vaccine Filters")
        phase_options_v = sorted({p.strip() for val in df_vaccine["Phase"].dropna() for p in str(val).split(",")})
        status_options_v = sorted([s for s in df_vaccine["Status"].dropna().unique()])
        sponsor_scope_options = ["All sponsors"]
        if "Sponsor Type" in df_vaccine.columns:
            sponsor_scope_options.append("Originator-sponsored only")

        selected_phases_v = st.sidebar.multiselect(
            "Phase (Your Vaccine)", options=phase_options_v, default=phase_options_v, key="phase_filter_vaccine"
        )
        selected_status_v = st.sidebar.multiselect(
            "Status (Your Vaccine)", options=status_options_v, default=status_options_v, key="status_filter_vaccine"
        )
        sponsor_scope = st.sidebar.selectbox(
            "Sponsor scope (Your Vaccine)",
            options=sponsor_scope_options,
            index=0,
            key="sponsor_scope_vaccine",
        )

        def _row_has_phase_v(ph_str: str, selected: list) -> bool:
            row_phases = [p.strip() for p in str(ph_str).split(",")]
            return any(p in row_phases or p in ph_str for p in selected) if selected else True

        df_vaccine_filtered = df_vaccine[df_vaccine["Phase"].apply(lambda x: _row_has_phase_v(x, selected_phases_v))]
        if selected_status_v:
            df_vaccine_filtered = df_vaccine_filtered[df_vaccine_filtered["Status"].isin(selected_status_v)]
        if sponsor_scope == "Originator-sponsored only" and "Sponsor Type" in df_vaccine_filtered.columns:
            df_vaccine_filtered = df_vaccine_filtered[df_vaccine_filtered["Sponsor Type"] == "Originator / Manufacturer"]

        st.info(f"📊 Showing {len(df_vaccine_filtered)} of {len(df_vaccine)} trials")
        
        # Visualizations
        sponsor_chart = create_sponsor_chart(df_vaccine_filtered, top_n=5)
        if sponsor_chart:
            st.plotly_chart(sponsor_chart, use_container_width=True)
        
        show_df(df_vaccine_filtered, height=320)

        # Gemini-powered unified vaccine intelligence (single combined entry point)
        if st.button("🧠 Unified Vaccine Intelligence (Gemini)", key="vaccine_intel"):
            with st.spinner("Generating unified Gemini intelligence brief..."):
                mfr_for_llm = _get_vaccine_manufacturer(target_vaccine)
                wiki_ctx = _fetch_wikipedia_summary(target_vaccine)
                # Use the filtered set for tighter, faster context
                intel_text, intel_err = _vaccine_intel_summary(
                    vaccine_name=target_vaccine,
                    manufacturer=mfr_for_llm,
                    diseases=target_diseases or [],
                    vaccine_trials=df_vaccine_filtered.to_dict("records"),
                    competitor_trials=competitor_trials,
                    external_context=wiki_ctx,
                )
            if intel_text:
                st.markdown("#### 🤖 Unified Vaccine Intelligence (Gemini)")
                st.write(intel_text)

                pdf_buffer = generate_pdf_summary(intel_text, f"Vaccine Intelligence - {target_vaccine}")
                if pdf_buffer:
                    st.download_button(
                        label="📄 Download Gemini Intelligence PDF",
                        data=pdf_buffer,
                        file_name=f"vaccine_intel_{target_vaccine.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf",
                        key=f"pdf_vaccine_intel_{target_vaccine}",
                        use_container_width=True,
                    )
            elif intel_err:
                st.warning(intel_err)

        selected_vaccine_id = st.selectbox(
            "🔬 View Detailed Info",
            options=["Select a study..."] + [str(x) for x in df_vaccine_filtered["NCT ID"].tolist()],
            key="select_vaccine_detail"
        )

        if selected_vaccine_id != "Select a study...":
            with st.spinner("Loading details..."):
                details_v = fetch_trial_details_with_vaccines(selected_vaccine_id)

            if details_v:
                st.markdown("---")
                st.subheader(f"📋 Study Details: {selected_vaccine_id}")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Title:** {details_v['Title']}")
                    st.markdown(f"**Phase:** {details_v['Phase']}")
                    if details_v.get("Enrollment"):
                        enroll = details_v["Enrollment"]
                        if enroll.get("actual_enrollment"):
                            st.markdown(f"**Enrollment:** {enroll.get('actual_enrollment')}")
                        elif enroll.get("enrollment_count"):
                            st.markdown(f"**Target Enrollment:** {enroll.get('enrollment_count')}")
                with col2:
                    st.markdown(f"**Status:** {details_v['Status']}")
                    st.markdown(f"**Sponsor:** {details_v['Sponsor']}")
                    if details_v.get("Dates"):
                        dates = details_v["Dates"]
                        if dates.get("start_date"):
                            st.markdown(f"**Start Date:** {dates.get('start_date')}")
                        if dates.get("completion_date"):
                            st.markdown(f"**Completion Date:** {dates.get('completion_date')}")

                # Enhanced details (same as tab 1)
                if details_v.get("Design"):
                    design = details_v["Design"]
                    with st.expander("📐 Study Design Details"):
                        col_d1, col_d2 = st.columns(2)
                        with col_d1:
                            if design.get("study_type"):
                                st.markdown(f"**Study Type:** {design.get('study_type')}")
                            if design.get("allocation"):
                                st.markdown(f"**Allocation:** {design.get('allocation')}")
                            if design.get("intervention_model"):
                                st.markdown(f"**Intervention Model:** {design.get('intervention_model')}")
                        with col_d2:
                            if design.get("masking"):
                                st.markdown(f"**Masking:** {design.get('masking')}")
                            if design.get("primary_purpose"):
                                st.markdown(f"**Primary Purpose:** {design.get('primary_purpose')}")
                            if design.get("number_of_arms"):
                                st.markdown(f"**Number of Arms:** {design.get('number_of_arms')}")

                if details_v.get("Diseases"):
                    st.markdown("**🦠 Diseases/Conditions:**")
                    st.write(", ".join(details_v["Diseases"]))

                st.markdown("**💉 Vaccine Products:**")
                for v in details_v["Vaccines"]:
                    st.markdown(f"- {v}")

                if details_v.get("Locations"):
                    with st.expander("🌍 Study Locations"):
                        for loc in details_v["Locations"][:10]:
                            loc_str = f"{loc.get('name', '')}"
                            if loc.get("city"):
                                loc_str += f", {loc.get('city')}"
                            if loc.get("state"):
                                loc_str += f", {loc.get('state')}"
                            if loc.get("country"):
                                loc_str += f", {loc.get('country')}"
                            st.write(f"• {loc_str}")

                if details_v.get("Eligibility"):
                    with st.expander("👥 Eligibility Criteria"):
                        elig = details_v["Eligibility"]
                        if elig.get("gender"):
                            st.markdown(f"**Gender:** {elig.get('gender')}")
                        if elig.get("minimum_age") or elig.get("maximum_age"):
                            age_range = f"{elig.get('minimum_age', 'N/A')} - {elig.get('maximum_age', 'N/A')}"
                            st.markdown(f"**Age Range:** {age_range}")
                        if elig.get("healthy_volunteers"):
                            st.markdown(f"**Healthy Volunteers:** {elig.get('healthy_volunteers')}")
                        if elig.get("criteria"):
                            st.markdown("**Criteria:**")
                            st.text(elig.get("criteria")[:500] + "..." if len(elig.get("criteria", "")) > 500 else elig.get("criteria"))

                if details_v.get("Collaborators"):
                    st.markdown("**🤝 Collaborators:**")
                    st.write(", ".join(details_v["Collaborators"]))

                if details_v["Outcomes"]:
                    st.markdown("**📊 Primary Outcome Measures:**")
                    for o in details_v["Outcomes"]:
                        st.write(f"• {o['Title']}")
                        if o["Description"]:
                            st.caption(o["Description"])
                else:
                    st.info("No outcomes reported yet.")

                if details_v.get("Results"):
                    st.success("✅ Results data available for this study")

                if details_v.get("PubMed_Articles"):
                    with st.expander(f"📚 PubMed Articles ({len(details_v['PubMed_Articles'])})"):
                        for pm in details_v["PubMed_Articles"]:
                            authors = ", ".join(pm["authors"]) + (" et al." if len(pm["authors"]) == 3 else "")
                            st.markdown(f"**[{pm['title']}](https://pubmed.ncbi.nlm.nih.gov/{pm['pmid']})**")
                            st.caption(f"{pm['source']} | {pm['pubdate']} | {authors}")

                if st.button("🧠 Summarize This Study", key=f"summarize_vaccine_detail_{selected_vaccine_id}"):
                    with st.spinner("Creating AI trial brief..."):
                        trial_summary_v, detail_err_v = _summarize_single_trial(selected_vaccine_id, details_v)
                    if trial_summary_v:
                        st.markdown("#### 🤖 AI Trial Brief")
                        st.write(trial_summary_v)
                        
                        # Export button
                        pdf_buffer = generate_pdf_summary(trial_summary_v, f"Trial Brief - {selected_vaccine_id}")
                        if pdf_buffer:
                            st.download_button(
                                label="📄 Download PDF Report",
                                data=pdf_buffer,
                                file_name=f"trial_brief_{selected_vaccine_id}_{datetime.now().strftime('%Y%m%d')}.pdf",
                                mime="application/pdf",
                                key=f"pdf_vaccine_trial_{selected_vaccine_id}",
                                use_container_width=True
                            )
                    elif detail_err_v:
                        st.warning(detail_err_v)

    # Show competitor trials (existing filters kept as-is)
    if competitor_trials:
        st.markdown("---")
        st.subheader(f"🔄 Competitor Vaccines for {', '.join(target_diseases) if target_diseases else 'Same Disease'}")
        st.caption("All other vaccines targeting the same disease(s)")

        df_competitor = pd.DataFrame(competitor_trials)

        st.sidebar.header("🎛️ Competitor Filters")
        phase_options_c = sorted({p.strip() for val in df_competitor["Phase"].dropna() for p in str(val).split(",")})
        status_options_c = sorted([s for s in df_competitor["Status"].dropna().unique()])

        selected_phases_c = st.sidebar.multiselect(
            "Phase (Competitors)", options=phase_options_c, default=phase_options_c, key="phase_filter_comp"
        )
        selected_status_c = st.sidebar.multiselect(
            "Status (Competitors)", options=status_options_c, default=status_options_c, key="status_filter_comp"
        )

        def _row_has_phase_c(ph_str: str, selected: list) -> bool:
            row_phases = [p.strip() for p in str(ph_str).split(",")]
            return any(p in row_phases or p in ph_str for p in selected) if selected else True

        df_competitor_filtered = df_competitor[df_competitor["Phase"].apply(lambda x: _row_has_phase_c(x, selected_phases_c))]
        if selected_status_c:
            df_competitor_filtered = df_competitor_filtered[df_competitor_filtered["Status"].isin(selected_status_c)]

        st.info(f"📊 Showing {len(df_competitor_filtered)} of {len(competitor_trials)} competitor trials")
        
        # Competitor visualizations
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            comp_phase_chart = create_phase_chart(df_competitor_filtered)
            if comp_phase_chart:
                st.plotly_chart(comp_phase_chart, use_container_width=True)
        with col_c2:
            comp_sponsor_chart = create_sponsor_chart(df_competitor_filtered, top_n=5)
            if comp_sponsor_chart:
                st.plotly_chart(comp_sponsor_chart, use_container_width=True)
        
        show_df(df_competitor_filtered, height=420)

        if st.button("🧠 Summarize Competitor Trials", key="summarize_comp_trials"):
            with st.spinner("Creating AI competitor synopsis..."):
                comp_summary, comp_err = _summarize_trials_with_llm(
                    df_competitor_filtered.to_dict("records"),
                    context_instructions=f"Competitor vaccines targeting diseases: {', '.join(target_diseases) if target_diseases else 'Unknown'}."
                )
            if comp_summary:
                st.markdown("#### 🤖 AI Summary — Competitors")
                st.write(comp_summary)
                
                # Export button
                pdf_buffer = generate_pdf_summary(comp_summary, f"Competitor Analysis - {', '.join(target_diseases) if target_diseases else 'Competitors'}")
                if pdf_buffer:
                    st.download_button(
                        label="📄 Download PDF Report",
                        data=pdf_buffer,
                        file_name=f"competitor_analysis_{datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf",
                        key="pdf_competitor",
                        use_container_width=True
                    )
            elif comp_err:
                st.warning(comp_err)

        selected_comp_id = st.selectbox(
            "🔬 View Competitor Trial Details",
            options=["Select a study..."] + [str(x) for x in df_competitor_filtered["NCT ID"].tolist()],
            key="select_comp_detail"
        )

        if selected_comp_id != "Select a study...":
            with st.spinner("Loading details..."):
                details_c = fetch_trial_details_with_vaccines(selected_comp_id)

            if details_c:
                st.markdown("---")
                st.subheader(f"📋 Competitor Study: {selected_comp_id}")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Title:** {details_c['Title']}")
                    st.markdown(f"**Phase:** {details_c['Phase']}")
                with col2:
                    st.markdown(f"**Status:** {details_c['Status']}")
                    st.markdown(f"**Sponsor:** {details_c['Sponsor']}")

                if details_c.get("Diseases"):
                    st.markdown("**🦠 Diseases/Conditions:**")
                    st.write(", ".join(details_c["Diseases"]))

                st.markdown("**💉 Vaccine Products:**")
                for v in details_c["Vaccines"]:
                    st.markdown(f"- {v}")
                if details_c["Outcomes"]:
                    st.markdown("**📊 Primary Outcome Measures:**")
                    for o in details_c["Outcomes"]:
                        st.write(f"• {o['Title']}")
                        if o["Description"]:
                            st.caption(o["Description"])

                if details_c.get("PubMed_Articles"):
                    with st.expander(f"📚 PubMed Articles ({len(details_c['PubMed_Articles'])})"):
                        for pm in details_c["PubMed_Articles"]:
                            authors = ", ".join(pm["authors"]) + (" et al." if len(pm["authors"]) == 3 else "")
                            st.markdown(f"**[{pm['title']}](https://pubmed.ncbi.nlm.nih.gov/{pm['pmid']})**")
                            st.caption(f"{pm['source']} | {pm['pubdate']} | {authors}")

                if st.button("🧠 Summarize This Study", key=f"summarize_comp_detail_{selected_comp_id}"):
                    with st.spinner("Creating AI trial brief..."):
                        trial_summary_c, detail_err_c = _summarize_single_trial(selected_comp_id, details_c)
                    if trial_summary_c:
                        st.markdown("#### 🤖 AI Trial Brief")
                        st.write(trial_summary_c)
                    elif detail_err_c:
                        st.warning(detail_err_c)

if not st.session_state.get("vaccine_trials") and not st.session_state.get("competitor_trials"):
    st.info("👆 Enter a vaccine product name and click Search Vaccine & Competitors to begin.")



# ---------------- Footer ----------------
st.markdown("---")
st.caption("💡 Vaccine Pipeline Platform | Data from ClinicalTrials.gov | Developed by Aman & Smriti")