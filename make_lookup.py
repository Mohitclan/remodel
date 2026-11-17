"""Writes results/url_lookup.csv.gz: every PhiUSIIL URL with its true label and
whether it fell in the training or test part of the domain-disjoint split.
Used by the terminal demo (cli.py) to say whether a typed URL was seen in training."""
from pathlib import Path

import numpy as np
import pandas as pd

from audit import data
from audit.canon import host_of, registrable_domain
from audit.evaluate import domain_split

ROOT = Path(__file__).resolve().parent
df = data.load_phiusiil()
groups = np.array([registrable_domain(host_of(u)) for u in df.url])
tr, te = domain_split(groups)
part = np.empty(len(df), dtype=object)
part[tr], part[te] = "train", "test"
pd.DataFrame({"url": df.url, "label": np.where(df.y == 1, "Phishing", "Legitimate"),
              "split": part, "domain": groups}).to_csv(ROOT / "results" / "url_lookup.csv.gz", index=False)
print("lookup:", len(df), "URLs;", (part == "train").sum(), "train,", (part == "test").sum(), "test")
