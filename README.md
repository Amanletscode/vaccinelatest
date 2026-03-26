# 💉 Vaccine Pipeline Platform

A real-time competitive intelligence platform for vaccine clinical trials, regulatory data, and AI-powered analysis.

<p align="center">
  <strong>Search · Analyze · Summarize · Export</strong>
</p>

---

## 🚀 What It Does

| Feature | Description |
|---------|-------------|
| **Disease Search** | Fetch all vaccine trials for any disease from ClinicalTrials.gov |
| **Vaccine Product Search** | Find trials for a specific vaccine + auto-detect competitors |
| **openFDA Integration** | Regulatory intel — NDC data, FAERS adverse events, DailyMed/Drugs@FDA links |
| **PubMed + FDA News** | Recent English-language publications + FDA press releases |
| **AI Summaries** | Gemini-powered executive briefs, trial due-diligence, competitor analysis |
| **Interactive Workspace** | Favorite trials, add analyst notes, filter by phase/status/sponsor |
| **Plotly Dashboards** | Phase distribution, status pie, sponsor bar, global heatmap |
| **PDF Export** | Download AI summaries as professionally formatted PDFs |

## 📁 Project Structure

```
vaccinelatest/
├── app.py                  # Original monolith (preserved, still runnable)
├── app1.py                 # ✅ Modular entry point (recommended)
├── APP_MONOLITH.md         # Full app.py code in markdown for reference reading
├── modules/
│   ├── __init__.py         # Package init with module map
│   ├── config.py           # Constants: model names, file paths, API URLs
│   ├── utils.py            # _norm_txt, _request_json, version checks
│   ├── data_extraction.py  # Parse ClinicalTrials.gov JSON → structured dicts
│   ├── vaccine_data.py     # Synonym tables, manufacturer index, matching logic
│   ├── llm_helpers.py      # Gemini API, all summarisation prompts
│   ├── trial_fetchers.py   # ClinicalTrials.gov, openFDA, PubMed, FDA RSS
│   ├── visualizations.py   # Plotly charts (phase, status, sponsor, heatmap)
│   ├── analyst_workspace.py# Threat score, analyst notes, interactive table
│   └── pdf_export.py       # ReportLab PDF export with fpdf2 fallback
├── requirements.txt
├── .env                    # Your GEMINI_API_KEY (not committed)
├── .gitignore
├── README.md               # ← You are here
└── WALKTHROUGH.md          # Deep technical walkthrough
```

## ⚙️ Setup & Installation

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/vaccinelatest.git
cd vaccinelatest
```

### 2. Create a virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set your Gemini API key
Create a `.env` file in the project root:
```
GEMINI_API_KEY=your_google_gemini_api_key_here
```
> Get a free key at [Google AI Studio](https://aistudio.google.com/apikey)

### 5. Run the app
```bash
# Modular version (recommended)
streamlit run app1.py

# Original monolith (also works)
streamlit run app.py
```

## 🔑 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | ✅ | Google Gemini API key for AI summaries |
| `GEMINI_MODEL` | ❌ | Override default model (e.g., `models/gemini-1.5-flash-latest`) |

## 📡 Data Sources (All Free)

| Source | Endpoint | Used For |
|--------|----------|----------|
| **ClinicalTrials.gov** | `/api/v2/studies` | Trial metadata, phases, sponsors, vaccines |
| **PubMed E-utilities** | `esearch`, `esummary` | Linked publications & recent research |
| **openFDA** | `drug/ndc.json`, `drug/event.json` | NDC regulatory data, FAERS adverse events |
| **FDA RSS** | Press releases feed | Regulatory news |
| **Wikipedia** | REST API `/page/summary` | Background context for Gemini prompts |
| **Google Gemini** | Generative Language API | AI-powered intelligence summaries |

## 🧠 AI Models Available

Select in the sidebar:
- **Robotics 1.5 Preview** (default) — balanced speed/quality
- **Gemini 1.5 Pro** — deeper analysis
- **Gemini 1.5 Flash** — fastest, cheapest

## 🏗️ Architecture Overview

```
User Input (Disease / Vaccine Name)
         │
         ▼
   ┌─────────────┐
   │ app1.py     │  ← UI layer (Streamlit widgets, layout, tabs)
   │ (entry pt)  │
   └──────┬──────┘
          │ imports
          ▼
   ┌──────────────────────────────────────────────┐
   │ modules/                                      │
   │  ├─ config.py         constants               │
   │  ├─ utils.py          shared helpers           │
   │  ├─ data_extraction   parse trial JSON         │
   │  ├─ vaccine_data      synonyms + matching      │
   │  ├─ llm_helpers       Gemini API + prompts     │
   │  ├─ trial_fetchers    API calls (cached)       │
   │  ├─ visualizations    Plotly charts             │
   │  ├─ analyst_workspace threat score + notes      │
   │  └─ pdf_export        PDF generation            │
   └───────────┬──────────────────────────────────┘
               │ HTTP calls
               ▼
   ┌───────────────────────────────────────┐
   │  External APIs                         │
   │  ClinicalTrials.gov · PubMed · openFDA│
   │  FDA RSS · Wikipedia · Gemini         │
   └───────────────────────────────────────┘
```

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Make changes in the appropriate `modules/` file
4. Ensure `python -m py_compile app1.py` passes
5. Test with `streamlit run app1.py`
6. Open a Pull Request

### Where to add new features:
- **New API source** → `modules/trial_fetchers.py`
- **New visualization** → `modules/visualizations.py`
- **New LLM prompt** → `modules/llm_helpers.py`
- **New vaccine synonyms** → `modules/vaccine_data.py`
- **New constants** → `modules/config.py`
- **New UI section** → `app1.py`

## 🗺️ Scaling Roadmap

| Phase | Feature | Difficulty |
|-------|---------|------------|
| 🟢 Easy | Add more vaccine synonym groups | Low |
| 🟢 Easy | EU Clinical Trials Register (EUCTR) scraper | Low–Med |
| 🟡 Medium | WHO ICTRP registry integration | Medium |
| 🟡 Medium | Streamlit Cloud deployment | Medium |
| 🟡 Medium | Multi-user analyst notes (database-backed) | Medium |
| 🔴 Hard | Patent landscape integration (Google Patents) | High |
| 🔴 Hard | Market size estimation module | High |

## 📜 License

This project is open-source. Please add a LICENSE file when publishing.

---

<p align="center">
  <sub>Built by <strong>Aman & Smriti</strong> · Data from ClinicalTrials.gov</sub>
</p>
