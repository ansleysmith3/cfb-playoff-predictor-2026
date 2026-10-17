# CollegeFootball2026

A data pipeline that predicts the 2026 College Football Playoff top 12 (with seeding),
built to show what data can do - not for betting.

## Pipeline

1. **Ingest** - pull games and rankings from the CollegeFootballData API (`src/ingest.py`)
2. **Clean** - standardize team names, validate (`src/clean.py`, coming next)
3. **Features** - ratings, strength of schedule, quality wins (`src/features.py`)
4. **Model** - predict the top 12 and compare to the real committee (`src/model.py`)
5. **Automate** - GitHub Actions refreshes everything weekly

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env   # then paste your CFBD API key into .env
python -m src.ingest
```

Get a free API key at https://collegefootballdata.com.
