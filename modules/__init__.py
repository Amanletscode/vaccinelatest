"""
modules — Vaccine Pipeline Platform supporting package.

Module map
----------
config.py           Constants and tuneable configuration.
utils.py            Shared helper functions.
data_extraction.py  ClinicalTrials.gov JSON → structured dicts.
vaccine_data.py     Curated vaccine synonyms, mfr. data, matching logic.
llm_helpers.py      Gemini API integration, all summarisation prompts.
trial_fetchers.py   API data fetchers (ClinicalTrials.gov, PubMed, openFDA, FDA RSS).
visualizations.py   Plotly chart generators (phases, sponsors, heatmap, timelines, radar).
analyst_workspace.py Interactive analyst tools (threat score, notes, data editor).
pdf_export.py       PDF generation (ReportLab primary, fpdf2 fallback).
alerts.py           Watchlist persistence and poll-on-load alert detection.
"""
