"""Loaders for the four public corpora. Every loader returns a DataFrame with
columns ``url`` and ``y`` where y = 1 means phishing (the positive class)."""
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

CORPORA = ["PhiUSIIL", "LegitPhish", "URL-Phish", "Mowar and Jain"]
CITATION = {"PhiUSIIL": "[3]", "LegitPhish": "[12]", "URL-Phish": "[13]",
            "Mowar and Jain": "[14]"}


def load_phiusiil(with_shipped=False):
    df = pd.read_csv(RAW / "phiusiil" / "PhiUSIIL_Phishing_URL_Dataset.csv",
                     encoding="utf-8-sig")
    # In the released CSV label 1 = legitimate, 0 = phishing; recode so that
    # the positive class is phishing.
    out = pd.DataFrame({"url": df["URL"].astype(str), "y": 1 - df["label"].astype(int)})
    if with_shipped:
        return out, df
    return out


def load_legitphish():
    df = pd.read_csv(RAW / "legitphish" / "url_features_extracted1.csv")
    df = df.dropna(subset=["ClassLabel"])   # one row in the release has no label
    # ClassLabel: 0 = phishing, 1 = legitimate
    return pd.DataFrame({"url": df["URL"].astype(str), "y": 1 - df["ClassLabel"].astype(int)})


def load_urlphish():
    df = pd.read_csv(RAW / "urlphish" / "Dataset.csv")
    # label: 1 = phishing, 0 = legitimate
    return pd.DataFrame({"url": df["url"].astype(str), "y": df["label"].astype(int)})


def load_mowar():
    df = pd.read_csv(RAW / "mowar" / "phishing_and_benign_websites.csv",
                     encoding="utf-8-sig")
    lab = df["Label"].astype(str).str.strip().str.lower()
    return pd.DataFrame({"url": df["URLs"].astype(str), "y": (lab != "benign").astype(int)})


LOADERS = {"PhiUSIIL": load_phiusiil, "LegitPhish": load_legitphish,
           "URL-Phish": load_urlphish, "Mowar and Jain": load_mowar}


def load(name):
    df = LOADERS[name]()
    return df.dropna(subset=["url"]).reset_index(drop=True)
