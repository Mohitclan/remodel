# Canonicalisation-Control Audit for Phishing URL Benchmarks

Code and working model for the paper *Domain-Disjoint Splitting Is Not Enough:
Canonicalisation Control as a Diagnostic for Phishing URL Benchmarks*
(IEEE ICCSAI 2026, Paper ID 3112).

## What it does

1. Rewrites every URL in three ways, identically for both classes:
   - **R1 raw** – the URL as shipped
   - **R2 canonical** – scheme, leading `www.` and trailing separators removed
   - **R3 host-only** – the registrable domain (eTLD+1)
2. Recomputes 24 lexical features from the string (shipped columns are discarded).
3. Evaluates LR, RF and XGBoost under a stratified random split and a
   registrable-domain-disjoint split (70/30, seed 42), plus a 300k-feature
   character n-gram model.
4. Reports accuracy, F1, ROC-AUC, bootstrap CIs, McNemar's test, the template
   regex baseline and base-rate-corrected precision, on four public corpora.

## Layout

```
audit/canon.py      canonicalisation ladder, registrable domain, template regex
audit/features.py   24 lexical features
audit/evaluate.py   splits, models, bootstrap CI, McNemar, base-rate precision
audit/data.py       loaders for PhiUSIIL, LegitPhish, URL-Phish, Mowar and Jain
run_audit.py        runs every experiment -> results/results.json, models/*.joblib
make_figures.py     draws all figures -> figures/*.png
app.py              Streamlit working model (URL checker)
data/raw/           downloaded corpora (not redistributed)
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

On macOS XGBoost needs the OpenMP runtime (`brew install libomp`).

## Run

```bash
.venv/bin/python run_audit.py        # about 15-25 minutes on a laptop
.venv/bin/python make_figures.py
.venv/bin/streamlit run app.py       # opens the working model in a browser
```

## Data sources

| Corpus | Source | Rows |
|---|---|---|
| PhiUSIIL | UCI ML Repository, dataset 967 (CC BY 4.0) | 235,795 |
| LegitPhish | Mendeley Data, doi:10.17632/hx4m73v2sf.2 | 101,218 |
| URL-Phish | Mendeley Data, doi:10.17632/65z9twcx3r.1 | 116,600 |
| Mowar and Jain | Zenodo, doi:10.5281/zenodo.5807622 | 38,800 |

**Note:** `data/raw/urlphish/Dataset.csv` is not included in this repository, because a few phishing URLs in it contain
strings that GitHub's secret scanning flags as AWS keys. To rerun the cross-corpus experiment, download it from
Mendeley Data (doi:10.17632/65z9twcx3r.1) and save it as `data/raw/urlphish/Dataset.csv`. Everything else,
including the working model, runs without it.

## Quick start after cloning

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
./start_demo.sh            # opens the working model at http://localhost:8501
.venv/bin/python cli.py     # terminal demo: type URLs, see each model's input, features and decision
```
