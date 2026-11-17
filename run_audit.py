"""Run the full canonicalisation-control audit and write results/results.json.

Usage:  .venv/bin/python run_audit.py            (everything)
        .venv/bin/python run_audit.py --quick    (skip the n-gram model)
"""
import argparse
import json
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from audit import data
from audit.canon import (RUNGS, canonicalise, host_of, matches_template,
                         registrable_domain)
from audit.evaluate import (SEED, bootstrap_ci, domain_split, make_model,
                            make_ngram_model, mcnemar, precision_at_recall,
                            random_split, scores)
from audit.features import featurise

OUT = Path(__file__).resolve().parent / "results"
MODELS = Path(__file__).resolve().parent / "models"
PREVALENCES = [0.5, 0.1, 0.05, 0.0005]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def baseline(y):
    p = float(np.mean(y))
    return max(p, 1 - p)


# ------------------------------------------------------------------ data audit
def formatting_table(df):
    """Class-conditional URL formatting statistics (paper Table II)."""
    rows = {
        "Matches ^https://www.[^/?#]+$": df.url.map(matches_template),
        "Scheme is https": df.url.str.lower().str.startswith("https://"),
        "Has www. prefix": df.url.map(lambda u: bool(re.match(r"^(https?://)?www\.", u, re.I))),
        "Contains path, query or fragment": df.url.map(
            lambda u: bool(re.search(r"[/?#]", re.sub(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", "", u)))),
        "Ends with /": df.url.str.endswith("/"),
    }
    table = {}
    for name, mask in rows.items():
        table[name] = {"legitimate": float(mask[df.y == 0].mean() * 100),
                       "phishing": float(mask[df.y == 1].mean() * 100)}
    return table


def template_rule(df):
    """Single-regex classifier: template match -> legitimate. The accuracy is
    reported in the rule's better orientation, so a corpus in which the
    template carries no class information scores at most its base rate."""
    pred = (~df.url.map(matches_template)).astype(int)   # matches -> legitimate
    acc = float((pred == df.y).mean())
    return {"accuracy": max(acc, 1 - acc), "accuracy_as_stated": acc,
            "false_positives": int(((pred == 1) & (df.y == 0)).sum()),
            "false_negatives": int(((pred == 0) & (df.y == 1)).sum()),
            "phishing_matching_template": int(((pred == 0) & (df.y == 1)).sum())}


def duplication(df, groups, tr_r, te_r, tr_d, te_d):
    norm = df.url.str.lower().str.replace(r"[/?#&.]+$", "", regex=True)
    exact = int(df.url.duplicated().sum())
    folded = int(norm.duplicated().sum())
    conflict = int(df.groupby(norm).y.nunique().gt(1).sum())
    shared_urls_rand = len(set(df.url.iloc[tr_r]) & set(df.url.iloc[te_r]))
    shared_dom_rand = len(set(groups[tr_r]) & set(groups[te_r]))
    shared_urls_dom = len(set(df.url.iloc[tr_d]) & set(df.url.iloc[te_d]))
    shared_dom_dom = len(set(groups[tr_d]) & set(groups[te_d]))
    return {"exact_duplicate_urls": exact, "duplicates_after_casefold_trailing": folded,
            "conflicting_label_groups": conflict,
            "random_split_shared_urls": shared_urls_rand,
            "random_split_shared_domains": shared_dom_rand,
            "domain_split_shared_urls": shared_urls_dom,
            "domain_split_shared_domains": shared_dom_dom,
            "duplicate_url_percent": float(df.url.duplicated().mean() * 100)}


def shipped_feature_audit(raw_df, y):
    """Depth-1 stumps on each numeric shipped feature, random split, plus the
    URLSimilarityIndex ablation (paper Section VI-B)."""
    drop = {"FILENAME", "URL", "Domain", "TLD", "Title", "label"}
    cols = [c for c in raw_df.columns if c not in drop and pd.api.types.is_numeric_dtype(raw_df[c])]
    X = raw_df[cols].astype(float)
    tr, te = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=SEED)
    stumps = {}
    for c in cols:
        m = DecisionTreeClassifier(max_depth=1, random_state=SEED).fit(X.iloc[tr][[c]], y[tr])
        stumps[c] = float((m.predict(X.iloc[te][[c]]) == y[te]).mean())
    stumps = dict(sorted(stumps.items(), key=lambda kv: -kv[1]))
    full = make_model("XGB").fit(X.iloc[tr], y[tr])
    acc_full = float((full.predict(X.iloc[te]) == y[te]).mean())
    cols_wo = [c for c in cols if c != "URLSimilarityIndex"]
    wo = make_model("XGB").fit(X.iloc[tr][cols_wo], y[tr])
    acc_wo = float((wo.predict(X.iloc[te][cols_wo]) == y[te]).mean())
    usi = raw_df["URLSimilarityIndex"]
    legit = raw_df["label"] == 1
    url_len_off_by_one = (raw_df["URLLength"] == raw_df["URL"].str.len() - 1)
    return {
        "n_numeric_features": len(cols),
        "stump_accuracy": stumps,
        "n_stumps_above_0.90": int(sum(v > 0.90 for v in stumps.values())),
        "random_split_baseline": baseline(y),
        "xgb_all_shipped_accuracy": acc_full,
        "xgb_without_URLSimilarityIndex_accuracy": acc_wo,
        "ablation_delta": acc_full - acc_wo,
        "usi_100_legit": int(((usi == 100) & legit).sum()),
        "usi_100_phish": int(((usi == 100) & ~legit).sum()),
        "usi_std_legit": float(usi[legit].std()),
        "urllength_off_by_one_legit_pct": float(url_len_off_by_one[legit].mean() * 100),
        "urllength_off_by_one_phish_pct": float(url_len_off_by_one[~legit].mean() * 100),
    }


# ------------------------------------------------------------------ main audit
def representation_audit(df, groups, splits, models=("LR", "RF", "XGB"), keep=None):
    y = df.y.to_numpy()
    res, preds, probs = {}, {}, {}
    for rung in RUNGS:
        log(f"  featurising {rung}")
        X = featurise(df.url.map(lambda u: canonicalise(u, rung)))
        for split_name, (tr, te) in splits.items():
            for m in models:
                model = make_model(m).fit(X.iloc[tr], y[tr])
                p = model.predict_proba(X.iloc[te])[:, 1]
                yhat = (p >= 0.5).astype(int)
                r = scores(y[te], yhat, p)
                r["ci95"] = bootstrap_ci(y[te], yhat)
                res[f"{rung}|{split_name}|{m}"] = r
                preds[(rung, split_name, m)] = yhat
                probs[(rung, split_name, m)] = p
                if keep is not None and split_name == "domain" and m == "XGB":
                    keep[rung] = model
                log(f"    {rung} {split_name:6s} {m:3s} acc={r['accuracy']:.4f}")
    return res, preds, probs


def main(quick=False):
    OUT.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    results = {}

    log("PhiUSIIL: loading")
    df, raw = data.load_phiusiil(with_shipped=True)
    y = df.y.to_numpy()
    groups = np.array([registrable_domain(host_of(u)) for u in df.url])
    tr_r, te_r = random_split(y)
    tr_d, te_d = domain_split(groups)
    splits = {"random": (tr_r, te_r), "domain": (tr_d, te_d)}

    results["phiusiil"] = {
        "n": int(len(df)), "n_phishing": int(y.sum()), "n_legitimate": int((1 - y).sum()),
        "baseline_full": baseline(y), "baseline_domain_test": baseline(y[te_d]),
        "baseline_random_test": baseline(y[te_r]),
        "domain_split_train": int(len(tr_d)), "domain_split_test": int(len(te_d)),
        "distinct_domains": int(len(set(groups))),
        "formatting": formatting_table(df),
        "template_rule": template_rule(df),
        "duplication": duplication(df, groups, tr_r, te_r, tr_d, te_d),
    }
    r2 = df.url.map(lambda u: canonicalise(u, "R2"))
    results["phiusiil"]["residual_path_after_R2"] = {
        "phishing_pct": float(r2[df.y == 1].str.contains(r"[/?#]").mean() * 100),
        "legitimate_pct": float(r2[df.y == 0].str.contains(r"[/?#]").mean() * 100)}

    log("PhiUSIIL: shipped-feature audit")
    results["shipped_features"] = shipped_feature_audit(raw, y)

    log("PhiUSIIL: representation x split x model")
    keep = {}
    res, preds, probs = representation_audit(df, groups, splits, keep=keep)
    results["main"] = res
    for rung, model in keep.items():
        joblib.dump(model, MODELS / f"xgb_{rung}.joblib")

    yd = y[te_d]
    results["mcnemar"] = {
        "R1_vs_R2": mcnemar(yd, preds[("R1", "domain", "XGB")], preds[("R2", "domain", "XGB")]),
        "R2_vs_R3": mcnemar(yd, preds[("R2", "domain", "XGB")], preds[("R3", "domain", "XGB")]),
        "R1_vs_R3": mcnemar(yd, preds[("R1", "domain", "XGB")], preds[("R3", "domain", "XGB")]),
    }
    results["base_rate"] = {rung: precision_at_recall(yd, probs[(rung, "domain", "XGB")], PREVALENCES)
                            for rung in RUNGS}
    # test-fold predictions for the demo / figures
    np.savez_compressed(OUT / "phiusiil_domain_test.npz", y=yd,
                        **{f"{r}_XGB": probs[(r, "domain", "XGB")] for r in RUNGS})

    if not quick:
        log("PhiUSIIL: character n-gram model (domain split)")
        ng = {}
        for rung in RUNGS:
            s = df.url.map(lambda u: canonicalise(u, rung)).to_numpy()
            m = make_ngram_model().fit(s[tr_d], y[tr_d])
            p = m.predict_proba(s[te_d])[:, 1]
            yhat = (p >= 0.5).astype(int)
            ng[rung] = scores(yd, yhat, p)
            ng[rung]["ci95"] = bootstrap_ci(yd, yhat)
            ng[rung]["n_features"] = int(len(m.named_steps["tfidfvectorizer"].vocabulary_))
            log(f"    n-gram {rung} acc={ng[rung]['accuracy']:.4f}")
        results["ngram"] = ng

    log("Cross-corpus audit (XGB, domain split)")
    cc = {}
    for name in data.CORPORA:
        d = data.load(name)
        yy = d.y.to_numpy()
        g = np.array([registrable_domain(host_of(u)) for u in d.url])
        tr, te = domain_split(g)
        accs = {}
        for rung in RUNGS:
            X = featurise(d.url.map(lambda u: canonicalise(u, rung)))
            mdl = make_model("XGB").fit(X.iloc[tr], yy[tr])
            accs[rung] = float((mdl.predict(X.iloc[te]) == yy[te]).mean())
        rule = template_rule(d)
        legit_tpl = d.url[d.y == 0].map(matches_template).mean() * 100
        phish_tpl = d.url[d.y == 1].map(matches_template).mean() * 100
        cc[name] = {"n": int(len(d)), **accs,
                    "delta_cos": accs["R1"] - accs["R2"], "delta_path": accs["R2"] - accs["R3"],
                    "rule": rule["accuracy"], "base": baseline(yy[te]),
                    "legit_template_pct": float(legit_tpl), "phish_template_pct": float(phish_tpl),
                    "duplicate_url_percent": float(d.url.duplicated().mean() * 100)}
        log(f"  {name}: R1={accs['R1']:.4f} R2={accs['R2']:.4f} R3={accs['R3']:.4f} rule={rule['accuracy']:.4f}")
    results["cross_corpus"] = cc

    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    log(f"wrote {OUT / 'results.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(**vars(ap.parse_args()))
