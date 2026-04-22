"""
app1.py — Modular entry point for the Vaccine Pipeline Platform.
================================================================

This file contains ONLY the Streamlit UI / layout code.
All business logic, data fetching, LLM calls, visualizations, and
helper functions are imported from the ``modules/`` package.

Run with:
    streamlit run app1.py
"""

import json
import pandas as pd
import requests
import streamlit as st
from collections import Counter
from datetime import datetime
from dotenv import load_dotenv

# ── Module imports ──
from modules.config import (
    DEFAULT_LLM_MODEL,
    GEMINI_MODEL_OPTIONS,
    LEARNED_MFR_FILE,
)
from modules.utils import _norm_txt, _request_json
from modules.data_extraction import (
    _extract_locations,
    _check_for_publications,
    _mesh_terms_condition,
)
from modules.vaccine_data import (
    _extract_vaccine_names,
    _matches_vaccine_name,
    _is_vaccine_study,
    _get_vaccine_search_terms,
    _get_vaccine_manufacturer,
)
from modules.llm_helpers import (
    _call_gemini,
    _summarize_trials_with_llm,
    _summarize_single_trial,
    _fetch_wikipedia_summary,
    _vaccine_intel_summary,
    _infer_manufacturer_llm,
    _compare_vaccines_llm,
)
from modules.trial_fetchers import (
    fetch_all_vaccine_trials,
    fetch_trial_details_with_vaccines,
    fetch_pipeline_publications,
    fetch_openfda_data,
)
from modules.visualizations import (
    create_phase_chart,
    create_status_chart,
    create_sponsor_chart,
    create_country_heatmap,
    create_trial_timeline,
    create_status_trend,
    create_completion_timeline,
    compute_comparison_metrics,
    create_comparison_table,
    create_comparison_bar,
)
from modules.analyst_workspace import (
    show_interactive_df,
)
from modules.pdf_export import generate_pdf_summary
from modules.alerts import (
    _load_watchlist,
    add_to_watchlist,
    remove_from_watchlist,
    check_watchlist_updates,
)

load_dotenv()

# ══════════════════════════════════════════════════════════════
#  APP CONFIG
# ══════════════════════════════════════════════════════════════

st.set_page_config(page_title="Vaccine Pipeline Platform", page_icon="💉", layout="wide")

# Legacy dataframe serializer (optional, ignored if not supported)
try:
    st.set_option("global.dataFrameSerialization", "legacy")
except Exception:
    pass

# ══════════════════════════════════════════════════════════════
#  HEADER + SIDEBAR
# ══════════════════════════════════════════════════════════════

st.title("💉 Vaccine Pipeline Platform")
st.markdown(
    "Explore complete vaccine trial data from ClinicalTrials.gov and connecting "
    "trials to pubmed articles. Search by disease condition or vaccine product "
    "name with competitor analysis."
)

# Sidebar: Gemini model selector
st.sidebar.markdown("### 🧠 Gemini model")
model_label = st.sidebar.selectbox(
    "Model for all AI summaries",
    options=list(GEMINI_MODEL_OPTIONS.keys()),
    index=0,
    key="gemini_model_label",
)
st.session_state["gemini_model"] = GEMINI_MODEL_OPTIONS.get(model_label, DEFAULT_LLM_MODEL)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Admin Tools")
if st.sidebar.button("🗑️ Clear AI Manufacturer Cache"):
    _infer_manufacturer_llm.clear()
    try:
        with open(LEARNED_MFR_FILE, "w") as f:
            json.dump({}, f)
        st.sidebar.success("Cache cleared!")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

# ── Watchlist Sidebar ──
st.sidebar.markdown("---")
st.sidebar.markdown("### 📢 Watchlist")
wl_name = st.sidebar.text_input("Add vaccine or disease", key="wl_add_input")
wl_type = st.sidebar.selectbox("Type", ["vaccine", "disease"], key="wl_type_select")
if st.sidebar.button("➕ Add to Watchlist", key="wl_add_btn"):
    if wl_name.strip():
        add_to_watchlist(wl_name.strip(), wl_type)
        st.sidebar.success(f"Added '{wl_name.strip()}'!")
    else:
        st.sidebar.warning("Enter a name first.")

watchlist = _load_watchlist()
if watchlist:
    st.sidebar.markdown("**Current Watchlist:**")
    for idx, wl_item in enumerate(watchlist):
        col_wl1, col_wl2 = st.sidebar.columns([3, 1])
        with col_wl1:
            emoji = "💊" if wl_item.get("type") == "vaccine" else "🦠"
            if st.button(f"{emoji} {wl_item.get('name', '')}", key=f"wl_search_{idx}", help="Click to search"):
                if wl_item.get("type") == "vaccine":
                    st.session_state["vaccine_input"] = wl_item.get("name")
                    st.session_state["trigger_vaccine_search"] = True
                elif wl_item.get("type") == "disease":
                    st.session_state["disease_input"] = wl_item.get("name")
                    st.session_state["trigger_disease_search"] = True
                st.rerun()
        with col_wl2:
            if st.button("❌", key=f"wl_rm_{idx}"):
                remove_from_watchlist(wl_item.get("name", ""))
                st.rerun()

# Session state defaults
for k in ["studies", "vaccine_trials", "competitor_trials", "target_vaccine", "target_diseases"]:
    st.session_state.setdefault(k, [] if "trials" in k or "studies" in k or "diseases" in k else "")


# ══════════════════════════════════════════════════════════════
#  SHARED UI HELPERS
# ══════════════════════════════════════════════════════════════

def _render_trial_details(details, prefix_key: str):
    """Render the expanded detail view for a single trial (shared by both tabs)."""
    st.markdown("---")
    st.subheader(f"📋 Study Details: {details['NCT ID']}")

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
            for loc in details["Locations"][:10]:
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
                criteria_text = elig.get("criteria", "")
                st.text(criteria_text[:500] + "..." if len(criteria_text) > 500 else criteria_text)

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

    nct_id = details["NCT ID"]
    if st.button("🧠 Summarize This Study", key=f"summarize_{prefix_key}_{nct_id}"):
        with st.spinner("Creating AI summary..."):
            trial_summary, detail_err = _summarize_single_trial(nct_id, details)
        if trial_summary:
            st.markdown("#### 🤖 AI Trial Brief")
            st.write(trial_summary)
            pdf_buffer = generate_pdf_summary(trial_summary, f"Trial Brief - {nct_id}")
            if pdf_buffer:
                st.download_button(
                    label="📄 Download PDF Report",
                    data=pdf_buffer,
                    file_name=f"trial_brief_{nct_id}_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    key=f"pdf_{prefix_key}_{nct_id}",
                    use_container_width=True,
                )
        elif detail_err:
            st.warning(detail_err)


def _render_publications(query: str):
    """Render the publications / FDA news expandable panel."""
    with st.expander(f"📰 Recent Publications & Regulatory News for '{query}'", icon="📡"):
        news_items = fetch_pipeline_publications(query, max_items=5)
        if news_items:
            for item in news_items:
                st.markdown(f"{item.get('source', '')} **[{item['title']}]({item['link']})**")
                caption_parts = []
                if item.get("authors"):
                    caption_parts.append(item["authors"])
                if item.get("pubdate"):
                    caption_parts.append(item["pubdate"])
                if caption_parts:
                    st.caption(" · ".join(caption_parts))
        else:
            st.info("No recent publications found.")


def _row_has_phase(ph_str: str, selected: list) -> bool:
    row_phases = [p.strip() for p in str(ph_str).split(",")]
    return any(p in row_phases or p in ph_str for p in selected) if selected else True


# ══════════════════════════════════════════════════════════════
#  WATCHLIST ALERTS (poll-on-load)
# ══════════════════════════════════════════════════════════════

watchlist = _load_watchlist()
if watchlist:
    with st.expander("🔔 Watchlist Alerts", expanded=False):
        if st.button("🔄 Check for Updates", key="wl_check"):
            with st.spinner("Checking watchlist for new data..."):
                alerts = check_watchlist_updates(watchlist)
            if alerts:
                for alert in alerts:
                    emoji = "💊" if alert["type"] == "vaccine" else "🦠"
                    parts = []
                    if alert["new_trials"] > 0:
                        parts.append(f"🆕 {alert['new_trials']} new trial(s)")
                    if alert["new_pubs"] > 0:
                        parts.append(f"📰 {alert['new_pubs']} new publication(s)")
                        
                    alert_col1, alert_col2 = st.columns([3, 1])
                    with alert_col1:
                        st.markdown(f"{emoji} **{alert['name']}**: {' · '.join(parts)}")
                    with alert_col2:
                        if st.button("🔍 View", key=f"alert_btn_{alert['name']}"):
                            if alert["type"] == "vaccine":
                                st.session_state["vaccine_input"] = alert["name"]
                                st.session_state["trigger_vaccine_search"] = True
                            else:
                                st.session_state["disease_input"] = alert["name"]
                                st.session_state["trigger_disease_search"] = True
                            st.rerun()

                    if alert.get("pub_titles"):
                        for title in alert["pub_titles"]:
                            st.caption(f"   → {title}")
            else:
                st.info("No new updates since last check. Your watchlist items are up to date.")
        else:
            st.caption("Click 'Check for Updates' to poll ClinicalTrials.gov and PubMed for new data on your watchlist items.")

# ══════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs(["🔍 Search by Disease", "💊 Search by Vaccine Product", "⚖️ Head-to-Head Comparison"])

# ────────────────────────────────────────────────────────────
#  TAB 1: Search by Disease
# ────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Search Vaccine Trials by Disease")
    st.caption("Fetches trials by disease and classifies vaccines using MeSH and heuristics.")

    if "disease_input" not in st.session_state:
        st.session_state["disease_input"] = "RSV"
    disease = st.text_input("Enter Disease Name", key="disease_input")
    trigger_d = st.session_state.pop("trigger_disease_search", False)

    if st.button("🔍 Fetch All Trials", key="fetch_disease") or trigger_d:
        if trigger_d:
            st.info(f"✅ Auto-searching '{disease}' from Watchlist. Please ensure you are viewing the 'Search by Disease' tab to see results.")
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

        # Sidebar filters
        st.sidebar.header("🎛️ Filters (Disease Search)")
        phase_options = sorted({p.strip() for val in df["Phase"].dropna() for p in str(val).split(",")})
        status_options = sorted([s for s in df["Status"].dropna().unique()])

        selected_phases = st.sidebar.multiselect("Phase", options=phase_options, default=phase_options, key="phase_filter_disease")
        selected_status = st.sidebar.multiselect("Status", options=status_options, default=status_options, key="status_filter_disease")

        df_filtered = df[df["Phase"].apply(lambda x: _row_has_phase(x, selected_phases))]
        if selected_status:
            df_filtered = df_filtered[df_filtered["Status"].isin(selected_status)]

        st.info(f"📊 Showing {len(df_filtered)} of {len(studies)} trials")

        # Visualizations
        col1, col2, col3 = st.columns(3)
        with col1:
            phase_chart = create_phase_chart(df_filtered)
            if phase_chart:
                st.plotly_chart(phase_chart, use_container_width=True)
        with col2:
            status_chart = create_status_chart(df_filtered)
            if status_chart:
                st.plotly_chart(status_chart, use_container_width=True)
        with col3:
            heatmap = create_country_heatmap(df_filtered)
            if heatmap:
                st.plotly_chart(heatmap, use_container_width=True)

        # ── Trend Analysis ──
        with st.expander("📈 Trend Analysis", expanded=False):
            t_col1, t_col2 = st.columns(2)
            with t_col1:
                timeline = create_trial_timeline(df_filtered)
                if timeline:
                    st.plotly_chart(timeline, use_container_width=True)
                else:
                    st.caption("No start date data available for timeline.")
            with t_col2:
                status_trend = create_status_trend(df_filtered)
                if status_trend:
                    st.plotly_chart(status_trend, use_container_width=True)
                else:
                    st.caption("No date data available for status trend.")
            comp_timeline = create_completion_timeline(df_filtered)
            if comp_timeline:
                st.plotly_chart(comp_timeline, use_container_width=True)

        _render_publications(disease)
        show_interactive_df(df_filtered, key="disease_tab", height=420)

        if st.button("🧠 Summarize Displayed Trials", key="summarize_disease_trials"):
            with st.spinner("Generating AI executive summary..."):
                summary_txt, summary_err = _summarize_trials_with_llm(
                    df_filtered.to_dict("records"),
                    context_instructions=f"Disease search term: {disease}. Showing {len(df_filtered)} of {len(df)} vaccine trials.",
                )
            if summary_txt:
                st.markdown("#### 🤖 AI Executive Summary")
                st.write(summary_txt)
                pdf_buffer = generate_pdf_summary(summary_txt, f"Vaccine Pipeline Summary - {disease}")
                if pdf_buffer:
                    st.download_button(
                        label="📄 Download PDF Report", data=pdf_buffer,
                        file_name=f"vaccine_summary_{disease}_{datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf", key=f"pdf_disease_{disease}", use_container_width=True,
                    )
            elif summary_err:
                st.warning(summary_err)

        selected_id = st.selectbox(
            "🔬 View Detailed Info",
            options=["Select a study..."] + [str(x) for x in df_filtered["NCT ID"].tolist()],
            key="select_disease_detail",
        )
        if selected_id != "Select a study...":
            with st.spinner("Loading details..."):
                details = fetch_trial_details_with_vaccines(selected_id)
            if details:
                _render_trial_details(details, prefix_key="disease_detail")
    else:
        st.info("👆 Enter a disease name and click Fetch All Trials to begin.")


# ────────────────────────────────────────────────────────────
#  TAB 2: Search by Vaccine Product + Competitors
# ────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Search Trials by Vaccine Product")
    st.caption("Find your vaccine's trials + competitor vaccines targeting the same disease(s).")

    if "vaccine_input" not in st.session_state:
        st.session_state["vaccine_input"] = ""
    vaccine_name = st.text_input("Enter Vaccine Product Name", key="vaccine_input")
    trigger_v = st.session_state.pop("trigger_vaccine_search", False)

    if st.button("💊 Search Vaccine & Competitors", key="fetch_vaccine") or trigger_v:
        if not vaccine_name.strip():
            st.warning("Please enter a vaccine name.")
        else:
            if trigger_v:
                st.info(f"✅ Auto-searching '{vaccine_name}' from Watchlist. Please ensure you are viewing the 'Search by Vaccine Product' tab to see results.")
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
                        studies_v = data.get("studies", []) or []

                        for s in studies_v:
                            names = _extract_vaccine_names(s)
                            if not _matches_vaccine_name(names, target_norms):
                                continue
                            if not _is_vaccine_study(s):
                                continue
                            proto = s.get("protocolSection", {}) or {}
                            ident = proto.get("identificationModule", {}) or {}
                            design = proto.get("designModule", {}) or {}
                            status_mod = proto.get("statusModule", {}) or {}
                            sponsor_mod = proto.get("sponsorCollaboratorsModule", {}) or {}
                            nct_id = ident.get("nctId")
                            if not nct_id or nct_id in seen_nct:
                                continue
                            seen_nct.add(nct_id)
                            locations = _extract_locations(s)
                            trial_obj_v = {
                                "NCT ID": str(nct_id),
                                "Title": str(ident.get("briefTitle") or ident.get("officialTitle") or "No title"),
                                "Phase": ", ".join(design.get("phases") or ["Not reported"]),
                                "Status": str(status_mod.get("overallStatus") or "Unknown"),
                                "Sponsor": str(sponsor_mod.get("leadSponsor", {}).get("name", "Unknown")),
                                "Vaccines": ", ".join(names) if names else "Not reported",
                                "Locations": locations,
                                "Publications": _check_for_publications(s),
                            }
                            vaccine_results.append(trial_obj_v)
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

    # ── Display vaccine trials ──
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

        # Classify sponsor type relative to originator
        mfr = _get_vaccine_manufacturer(target_vaccine)
        mfr_norm = _norm_txt(mfr) if mfr else None

        def _sponsor_type(name: str) -> str:
            if not mfr_norm:
                return "Unknown Originator"
            sponsor_name_norm = _norm_txt(name or "")
            if mfr_norm in sponsor_name_norm or sponsor_name_norm in mfr_norm:
                return "Originator / Manufacturer"
            aliases_groups = [
                ["pfizer", "biontech", "wyeth", "hospira"],
                ["gsk", "glaxosmithkline"],
                ["astrazeneca", "medimmune"],
                ["sanofi", "pasteur", "aventis"],
                ["merck", "msd"],
                ["jnj", "johnson", "janssen"],
            ]
            for group in aliases_groups:
                if any(alias in mfr_norm for alias in group):
                    if any(alias in sponsor_name_norm for alias in group):
                        return "Originator / Manufacturer"
            return "External / Other"

        df_vaccine["Sponsor Type"] = df_vaccine["Sponsor"].apply(_sponsor_type)

        # Sidebar filters
        st.sidebar.header("🎛️ Vaccine Filters")
        phase_options_v = sorted({p.strip() for val in df_vaccine["Phase"].dropna() for p in str(val).split(",")})
        status_options_v = sorted([s for s in df_vaccine["Status"].dropna().unique()])
        sponsor_scope_options = ["All sponsors"]
        if "Sponsor Type" in df_vaccine.columns:
            sponsor_scope_options.append("Originator-sponsored only")

        selected_phases_v = st.sidebar.multiselect("Phase (Your Vaccine)", options=phase_options_v, default=phase_options_v, key="phase_filter_vaccine")
        selected_status_v = st.sidebar.multiselect("Status (Your Vaccine)", options=status_options_v, default=status_options_v, key="status_filter_vaccine")
        sponsor_scope = st.sidebar.selectbox("Sponsor scope (Your Vaccine)", options=sponsor_scope_options, index=0, key="sponsor_scope_vaccine")

        df_vaccine_filtered = df_vaccine[df_vaccine["Phase"].apply(lambda x: _row_has_phase(x, selected_phases_v))]
        if selected_status_v:
            df_vaccine_filtered = df_vaccine_filtered[df_vaccine_filtered["Status"].isin(selected_status_v)]
        if sponsor_scope == "Originator-sponsored only" and "Sponsor Type" in df_vaccine_filtered.columns:
            df_vaccine_filtered = df_vaccine_filtered[df_vaccine_filtered["Sponsor Type"] == "Originator / Manufacturer"]

        st.info(f"📊 Showing {len(df_vaccine_filtered)} of {len(df_vaccine)} trials")

        # Visualizations
        v_col1, v_col2 = st.columns(2)
        with v_col1:
            sponsor_chart = create_sponsor_chart(df_vaccine_filtered, top_n=5)
            if sponsor_chart:
                st.plotly_chart(sponsor_chart, use_container_width=True)
        with v_col2:
            heatmap_v = create_country_heatmap(df_vaccine_filtered)
            if heatmap_v:
                st.plotly_chart(heatmap_v, use_container_width=True)

        # ── Trend Analysis (Vaccine) ──
        with st.expander("📈 Trend Analysis", expanded=False):
            tv_col1, tv_col2 = st.columns(2)
            with tv_col1:
                timeline_v = create_trial_timeline(df_vaccine_filtered)
                if timeline_v:
                    st.plotly_chart(timeline_v, use_container_width=True)
                else:
                    st.caption("No start date data available.")
            with tv_col2:
                status_trend_v = create_status_trend(df_vaccine_filtered)
                if status_trend_v:
                    st.plotly_chart(status_trend_v, use_container_width=True)
                else:
                    st.caption("No date data available.")

        # Regulatory & News
        col_reg, col_news = st.columns(2)
        with col_reg:
            with st.expander("🏛️ openFDA Regulatory Intel", icon="🇺🇸"):
                fda_data = fetch_openfda_data(target_vaccine)
                if isinstance(fda_data, dict) and "error" in fda_data:
                    st.error(fda_data["error"])
                elif fda_data:
                    col_fda1, col_fda2 = st.columns(2)
                    with col_fda1:
                        if fda_data.get("brand_name"):
                            st.markdown(f"**Brand Name:** {fda_data['brand_name']}")
                        if fda_data.get("generic_name"):
                            st.markdown(f"**Generic Name:** {fda_data['generic_name']}")
                        if fda_data.get("manufacturer"):
                            st.markdown(f"**Labeler:** {fda_data['manufacturer']}")
                        if fda_data.get("product_type"):
                            st.markdown(f"**Product Type:** {fda_data['product_type']}")
                    with col_fda2:
                        if fda_data.get("marketing_category"):
                            st.markdown(f"**Category:** {fda_data['marketing_category']}")
                        if fda_data.get("application_number"):
                            st.markdown(f"**Application #:** {fda_data['application_number']}")
                        if fda_data.get("marketing_start_date"):
                            st.markdown(f"**Marketing Start:** {fda_data['marketing_start_date']}")
                        if fda_data.get("dosage_form"):
                            st.markdown(f"**Dosage Form:** {fda_data['dosage_form']}")
                    if fda_data.get("route"):
                        st.markdown(f"**Route:** {fda_data['route']}")
                    if fda_data.get("active_ingredients") and fda_data["active_ingredients"] != "Not listed":
                        st.markdown(f"**Active Ingredients:** {fda_data['active_ingredients']}")
                    if fda_data.get("ndc"):
                        st.markdown(f"**NDC:** {fda_data['ndc']}")
                    st.markdown("---")
                    faers = fda_data.get("faers_total", 0)
                    if faers > 0:
                        st.markdown(f"⚠️ **FAERS Adverse Event Reports:** {faers:,} total reports")
                    else:
                        st.markdown("⚠️ **FAERS:** No adverse event reports found")
                    link_parts = []
                    if fda_data.get("dailymed_link"):
                        link_parts.append(f"[📋 DailyMed Label]({fda_data['dailymed_link']})")
                    if fda_data.get("drugsfda_link"):
                        link_parts.append(f"[🏛️ Drugs@FDA]({fda_data['drugsfda_link']})")
                    if link_parts:
                        st.markdown(" | ".join(link_parts))
                else:
                    st.info(f"No openFDA NDC record found for '{target_vaccine}'. The FDA database covers approved/marketed products — early-stage pipeline assets won't appear here.")

        with col_news:
            with st.expander(f"📰 Recent Publications for '{target_vaccine}'", icon="📡"):
                news_items_v = fetch_pipeline_publications(target_vaccine, max_items=5)
                if news_items_v:
                    for item in news_items_v:
                        st.markdown(f"{item.get('source', '')} **[{item['title']}]({item['link']})**")
                        caption_parts = []
                        if item.get("authors"):
                            caption_parts.append(item["authors"])
                        if item.get("pubdate"):
                            caption_parts.append(item["pubdate"])
                        if caption_parts:
                            st.caption(" · ".join(caption_parts))
                else:
                    st.info("No recent publications found.")

        show_interactive_df(df_vaccine_filtered, key="vaccine_tab", height=320)

        # Unified Gemini intelligence
        if st.button("🧠 Unified Vaccine Intelligence (Gemini)", key="vaccine_intel"):
            with st.spinner("Generating unified Gemini intelligence brief..."):
                mfr_for_llm = _get_vaccine_manufacturer(target_vaccine)
                wiki_ctx = _fetch_wikipedia_summary(target_vaccine)
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
            key="select_vaccine_detail",
        )
        if selected_vaccine_id != "Select a study...":
            with st.spinner("Loading details..."):
                details_v = fetch_trial_details_with_vaccines(selected_vaccine_id)
            if details_v:
                _render_trial_details(details_v, prefix_key="vaccine_detail")

    # ── Competitor Trials ──
    if competitor_trials:
        st.markdown("---")
        st.subheader(f"🔄 Competitor Vaccines for {', '.join(target_diseases) if target_diseases else 'Same Disease'}")
        st.caption("All other vaccines targeting the same disease(s)")

        df_competitor = pd.DataFrame(competitor_trials)

        st.sidebar.header("🎛️ Competitor Filters")
        phase_options_c = sorted({p.strip() for val in df_competitor["Phase"].dropna() for p in str(val).split(",")})
        status_options_c = sorted([s for s in df_competitor["Status"].dropna().unique()])

        selected_phases_c = st.sidebar.multiselect("Phase (Competitors)", options=phase_options_c, default=phase_options_c, key="phase_filter_comp")
        selected_status_c = st.sidebar.multiselect("Status (Competitors)", options=status_options_c, default=status_options_c, key="status_filter_comp")

        df_competitor_filtered = df_competitor[df_competitor["Phase"].apply(lambda x: _row_has_phase(x, selected_phases_c))]
        if selected_status_c:
            df_competitor_filtered = df_competitor_filtered[df_competitor_filtered["Status"].isin(selected_status_c)]

        st.info(f"📊 Showing {len(df_competitor_filtered)} of {len(competitor_trials)} competitor trials")

        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            comp_phase = create_phase_chart(df_competitor_filtered)
            if comp_phase:
                st.plotly_chart(comp_phase, use_container_width=True)
        with col_c2:
            comp_sponsor = create_sponsor_chart(df_competitor_filtered, top_n=5)
            if comp_sponsor:
                st.plotly_chart(comp_sponsor, use_container_width=True)
        with col_c3:
            comp_heatmap = create_country_heatmap(df_competitor_filtered)
            if comp_heatmap:
                st.plotly_chart(comp_heatmap, use_container_width=True)

        show_interactive_df(df_competitor_filtered, key="competitor_tab", height=420)

        if st.button("🧠 Summarize Competitor Trials", key="summarize_comp_trials"):
            with st.spinner("Creating AI competitor synopsis..."):
                comp_summary, comp_err = _summarize_trials_with_llm(
                    df_competitor_filtered.to_dict("records"),
                    context_instructions=f"Competitor vaccines targeting diseases: {', '.join(target_diseases) if target_diseases else 'Unknown'}.",
                )
            if comp_summary:
                st.markdown("#### 🤖 AI Summary — Competitors")
                st.write(comp_summary)
                pdf_buffer = generate_pdf_summary(
                    comp_summary,
                    f"Competitor Analysis - {', '.join(target_diseases) if target_diseases else 'Competitors'}",
                )
                if pdf_buffer:
                    st.download_button(
                        label="📄 Download PDF Report", data=pdf_buffer,
                        file_name=f"competitor_analysis_{datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf", key="pdf_competitor", use_container_width=True,
                    )
            elif comp_err:
                st.warning(comp_err)

        selected_comp_id = st.selectbox(
            "🔬 View Competitor Trial Details",
            options=["Select a study..."] + [str(x) for x in df_competitor_filtered["NCT ID"].tolist()],
            key="select_comp_detail",
        )
        if selected_comp_id != "Select a study...":
            with st.spinner("Loading details..."):
                details_c = fetch_trial_details_with_vaccines(selected_comp_id)
            if details_c:
                _render_trial_details(details_c, prefix_key="comp_detail")

if not st.session_state.get("vaccine_trials") and not st.session_state.get("competitor_trials"):
    st.info("👆 Enter a vaccine product name and click Search Vaccine & Competitors to begin.")


# ────────────────────────────────────────────────────────────
#  TAB 3: Head-to-Head Vaccine Comparison
# ────────────────────────────────────────────────────────────
with tab3:
    st.subheader("⚖️ Head-to-Head Vaccine Comparison")
    st.caption(
        "Enter two vaccine product names to compare their clinical trial profiles, "
        "regulatory data, and competitive positioning side by side."
    )

    h2h_col1, h2h_col2 = st.columns(2)
    with h2h_col1:
        vaccine_a = st.text_input("Vaccine A", value="", key="h2h_vaccine_a", placeholder="e.g. Comirnaty")
    with h2h_col2:
        vaccine_b = st.text_input("Vaccine B", value="", key="h2h_vaccine_b", placeholder="e.g. Arexvy")

    if st.button("⚖️ Compare Vaccines", key="h2h_compare"):
        if not vaccine_a.strip() or not vaccine_b.strip():
            st.warning("Please enter both vaccine names.")
        elif vaccine_a.strip().lower() == vaccine_b.strip().lower():
            st.warning("Please enter two different vaccines to compare.")
        else:
            search_url = "https://clinicaltrials.gov/api/v2/studies"

            def _fetch_vaccine_trials_for_compare(vax_name: str):
                """Fetch trials for a single vaccine (for comparison)."""
                search_terms = _get_vaccine_search_terms(vax_name)
                if not search_terms:
                    search_terms = [vax_name]
                target_norms = [_norm_txt(t) for t in search_terms if _norm_txt(t)]
                results = []
                seen = set()
                for term in search_terms:
                    try:
                        params = {"query.intr": term, "pageSize": 100}
                        data = _request_json(search_url, params=params)
                        for s in data.get("studies", []) or []:
                            names = _extract_vaccine_names(s)
                            if not _matches_vaccine_name(names, target_norms):
                                continue
                            if not _is_vaccine_study(s):
                                continue
                            proto = s.get("protocolSection", {}) or {}
                            ident = proto.get("identificationModule", {}) or {}
                            design_mod = proto.get("designModule", {}) or {}
                            status_mod = proto.get("statusModule", {}) or {}
                            sponsor_mod = proto.get("sponsorCollaboratorsModule", {}) or {}
                            nct_id = ident.get("nctId")
                            if not nct_id or nct_id in seen:
                                continue
                            seen.add(nct_id)
                            locations = _extract_locations(s)
                            trial_obj = {
                                "NCT ID": str(nct_id),
                                "Title": str(ident.get("briefTitle") or ident.get("officialTitle") or "No title"),
                                "Phase": ", ".join(design_mod.get("phases") or ["Not reported"]),
                                "Status": str(status_mod.get("overallStatus") or "Unknown"),
                                "Sponsor": str(sponsor_mod.get("leadSponsor", {}).get("name", "Unknown")),
                                "Vaccines": ", ".join(names) if names else "Not reported",
                                "Locations": locations,
                                "Start Date": str(status_mod.get("startDateStruct", {}).get("date", "") or ""),
                                "Completion Date": str(status_mod.get("completionDateStruct", {}).get("date", "") or ""),
                            }
                            results.append(trial_obj)
                    except Exception:
                        pass
                return results

            with st.spinner(f"Fetching trials for '{vaccine_a}' and '{vaccine_b}'..."):
                trials_a = _fetch_vaccine_trials_for_compare(vaccine_a.strip())
                trials_b = _fetch_vaccine_trials_for_compare(vaccine_b.strip())

            st.session_state["h2h_trials_a"] = trials_a
            st.session_state["h2h_trials_b"] = trials_b
            st.session_state["h2h_name_a"] = vaccine_a.strip()
            st.session_state["h2h_name_b"] = vaccine_b.strip()

            st.success(f"✅ Found {len(trials_a)} trials for '{vaccine_a}' and {len(trials_b)} trials for '{vaccine_b}'.")

    # ── Display comparison results ──
    h2h_a = st.session_state.get("h2h_trials_a", [])
    h2h_b = st.session_state.get("h2h_trials_b", [])
    h2h_name_a = st.session_state.get("h2h_name_a", "")
    h2h_name_b = st.session_state.get("h2h_name_b", "")

    if h2h_a or h2h_b:
        st.markdown("---")

        # Get manufacturers
        mfr_a = _get_vaccine_manufacturer(h2h_name_a)
        mfr_b = _get_vaccine_manufacturer(h2h_name_b)

        # Compute metrics (with known manufacturer for context)
        metrics_a = compute_comparison_metrics(h2h_a, h2h_name_a, known_manufacturer=mfr_a)
        metrics_b = compute_comparison_metrics(h2h_b, h2h_name_b, known_manufacturer=mfr_b)

        # KPI row (simple trial counts)
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        with kpi_col1:
            st.metric(f"{h2h_name_a} Trials", metrics_a["total_trials"])
        with kpi_col2:
            st.metric(f"{h2h_name_b} Trials", metrics_b["total_trials"])
        with kpi_col3:
            st.metric(f"{h2h_name_a} Phase 3", metrics_a["phase3_count"])
        with kpi_col4:
            st.metric(f"{h2h_name_b} Phase 3", metrics_b["phase3_count"])

        # Side-by-side: table + grouped bar chart
        cmp_col1, cmp_col2 = st.columns(2)
        with cmp_col1:
            st.markdown("#### 📊 Metrics Comparison")
            cmp_table = create_comparison_table(metrics_a, metrics_b)
            st.dataframe(cmp_table, use_container_width=True, hide_index=True)
        with cmp_col2:
            bar_fig = create_comparison_bar(metrics_a, metrics_b)
            if bar_fig:
                st.plotly_chart(bar_fig, use_container_width=True)

        # openFDA comparison
        with st.expander("🏛️ Regulatory Comparison (openFDA)", expanded=False):
            reg_col1, reg_col2 = st.columns(2)
            with reg_col1:
                st.markdown(f"**{h2h_name_a}**")
                fda_a = fetch_openfda_data(h2h_name_a)
                if fda_a:
                    if fda_a.get("brand_name"): st.markdown(f"Brand: {fda_a['brand_name']}")
                    if fda_a.get("manufacturer"): st.markdown(f"Labeler: {fda_a['manufacturer']}")
                    if fda_a.get("marketing_category"): st.markdown(f"Category: {fda_a['marketing_category']}")
                    if fda_a.get("application_number"): st.markdown(f"App #: {fda_a['application_number']}")
                    if fda_a.get("marketing_start_date"): st.markdown(f"Marketing Start: {fda_a['marketing_start_date']}")
                    faers_a = fda_a.get("faers_total", 0)
                    st.markdown(f"⚠️ FAERS Reports: **{faers_a:,}**" if faers_a else "⚠️ FAERS: None found")
                else:
                    st.info("No openFDA data found.")
            with reg_col2:
                st.markdown(f"**{h2h_name_b}**")
                fda_b = fetch_openfda_data(h2h_name_b)
                if fda_b:
                    if fda_b.get("brand_name"): st.markdown(f"Brand: {fda_b['brand_name']}")
                    if fda_b.get("manufacturer"): st.markdown(f"Labeler: {fda_b['manufacturer']}")
                    if fda_b.get("marketing_category"): st.markdown(f"Category: {fda_b['marketing_category']}")
                    if fda_b.get("application_number"): st.markdown(f"App #: {fda_b['application_number']}")
                    if fda_b.get("marketing_start_date"): st.markdown(f"Marketing Start: {fda_b['marketing_start_date']}")
                    faers_b = fda_b.get("faers_total", 0)
                    st.markdown(f"⚠️ FAERS Reports: **{faers_b:,}**" if faers_b else "⚠️ FAERS: None found")
                else:
                    st.info("No openFDA data found.")

        # Publications comparison
        with st.expander("📰 Publications Comparison", expanded=False):
            pub_col1, pub_col2 = st.columns(2)
            with pub_col1:
                st.markdown(f"**{h2h_name_a}**")
                pubs_a = fetch_pipeline_publications(h2h_name_a, max_items=3)
                if pubs_a:
                    for p in pubs_a:
                        st.markdown(f"{p.get('source','')} **[{p['title']}]({p['link']})**")
                        if p.get('pubdate'): st.caption(p['pubdate'])
                else:
                    st.caption("No recent publications found.")
            with pub_col2:
                st.markdown(f"**{h2h_name_b}**")
                pubs_b = fetch_pipeline_publications(h2h_name_b, max_items=3)
                if pubs_b:
                    for p in pubs_b:
                        st.markdown(f"{p.get('source','')} **[{p['title']}]({p['link']})**")
                        if p.get('pubdate'): st.caption(p['pubdate'])
                else:
                    st.caption("No recent publications found.")

        # Trend comparison
        with st.expander("📈 Trial Timeline Comparison", expanded=False):
            trend_col1, trend_col2 = st.columns(2)
            if h2h_a:
                df_a = pd.DataFrame(h2h_a)
                with trend_col1:
                    st.markdown(f"**{h2h_name_a}**")
                    tl_a = create_trial_timeline(df_a)
                    if tl_a:
                        st.plotly_chart(tl_a, use_container_width=True)
                    else:
                        st.caption("No date data available.")
            if h2h_b:
                df_b = pd.DataFrame(h2h_b)
                with trend_col2:
                    st.markdown(f"**{h2h_name_b}**")
                    tl_b = create_trial_timeline(df_b)
                    if tl_b:
                        st.plotly_chart(tl_b, use_container_width=True)
                    else:
                        st.caption("No date data available.")

        # AI comparison — send computed metrics + sample NCT IDs (not raw trials)
        if st.button("🧠 AI Head-to-Head Comparison (Gemini)", key="h2h_ai"):
            with st.spinner("Generating AI head-to-head comparison..."):
                vax_a_data = {
                    "metrics": {k: v for k, v in metrics_a.items() if k != "label"},
                    "fda": fetch_openfda_data(h2h_name_a),
                    "manufacturer": mfr_a or "Unknown",
                    "sample_nct_ids": [t.get("NCT ID") for t in h2h_a[:10]],
                }
                vax_b_data = {
                    "metrics": {k: v for k, v in metrics_b.items() if k != "label"},
                    "fda": fetch_openfda_data(h2h_name_b),
                    "manufacturer": mfr_b or "Unknown",
                    "sample_nct_ids": [t.get("NCT ID") for t in h2h_b[:10]],
                }
                cmp_text, cmp_err = _compare_vaccines_llm(
                    h2h_name_a, vax_a_data, h2h_name_b, vax_b_data
                )
            if cmp_text:
                st.markdown("#### 🤖 AI Head-to-Head Analysis")
                st.write(cmp_text)
                pdf_buffer = generate_pdf_summary(
                    cmp_text,
                    f"Head-to-Head: {h2h_name_a} vs {h2h_name_b}",
                )
                if pdf_buffer:
                    st.download_button(
                        label="📄 Download Comparison PDF", data=pdf_buffer,
                        file_name=f"h2h_{h2h_name_a}_{h2h_name_b}_{datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf", key="pdf_h2h", use_container_width=True,
                    )
            elif cmp_err:
                st.warning(cmp_err)

    else:
        st.info("👆 Enter two vaccine names above and click Compare to begin.")


# ══════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════

st.markdown("---")
st.caption("💡 Vaccine Pipeline Platform | Data from ClinicalTrials.gov | Developed by Aman & Smriti")
