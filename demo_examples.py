"""Score example URL pairs that differ only in surface form with the saved
working-model classifiers; writes results/demo.json (report Table 5.10)."""
import json
from pathlib import Path

import joblib

from audit.canon import RUNGS, canonicalise, matches_template
from audit.features import FEATURE_NAMES, featurise

ROOT = Path(__file__).resolve().parent
EXAMPLES = [
    "https://www.wikipedia.org", "wikipedia.org",
    "https://www.sbi.co.in", "http://sbi.co.in/",
    "https://www.paypal-account-verify.com", "paypal-account-verify.com",
    "https://www.secure-login-update.net", "http://secure-login-update.net/",
]

models = {r: joblib.load(ROOT / "models" / f"xgb_{r}.joblib") for r in RUNGS}
rows = []
for url in EXAMPLES:
    row = {"url": url, "template": matches_template(url)}
    for r in RUNGS:
        row[r] = float(models[r].predict_proba(featurise([canonicalise(url, r)]))[0, 1])
    rows.append(row)
(ROOT / "results" / "demo.json").write_text(json.dumps(rows, indent=2))

# share of total split gain per feature, used in report Section 5.8
importance = {}
for r in RUNGS:
    gain = models[r].get_booster().get_score(importance_type="gain")
    vals = [gain.get(f"f{i}", gain.get(n, 0.0)) for i, n in enumerate(FEATURE_NAMES)]
    total = sum(vals)
    importance[r] = dict(sorted(((n, v / total) for n, v in zip(FEATURE_NAMES, vals)),
                                key=lambda kv: -kv[1]))
(ROOT / "results" / "importance.json").write_text(json.dumps(importance, indent=2))
for row in rows:
    print(f"{row['url']:42s} tpl={row['template']!s:5s} " + " ".join(f"{r}={row[r]:.4f}" for r in RUNGS))


# Real dataset URLs for the app's example buttons and dataset preview: drawn
# from the domain-disjoint TEST fold, so no model saw them during training.
import numpy as np
import pandas as pd

from audit import data
from audit.canon import host_of, registrable_domain
from audit.evaluate import domain_split

df = data.load_phiusiil()
groups = np.array([registrable_domain(host_of(u)) for u in df.url])
_, te = domain_split(groups)
test = df.iloc[te].reset_index(drop=True)
rng = np.random.default_rng(42)
pick = np.concatenate([rng.choice(np.where(test.y == 0)[0], 40, replace=False),
                       rng.choice(np.where(test.y == 1)[0], 40, replace=False)])
sample = test.iloc[pick].reset_index(drop=True)
out = pd.DataFrame({"url": sample.url,
                    "label": np.where(sample.y == 1, "Phishing", "Legitimate")})
for r in RUNGS:
    out[f"p_{r}"] = models[r].predict_proba(
        featurise([canonicalise(u, r) for u in sample.url]))[:, 1].round(4)
out.to_csv(ROOT / "results" / "demo_samples.csv", index=False)
print("demo samples:", out.label.value_counts().to_dict())
