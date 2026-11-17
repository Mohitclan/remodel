"""Generate every report figure from results/results.json and the saved models."""
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.metrics import roc_curve

from audit.canon import RUNG_NAMES, RUNGS
from audit.features import FEATURE_NAMES

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
R = json.loads((ROOT / "results" / "results.json").read_text())

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#444", "axes.labelcolor": "black",
                     "xtick.color": "black", "ytick.color": "black", "savefig.dpi": 200,
                     "savefig.bbox": "tight"})
NAVY, BLUE, LIGHT, RED, GREY = "#1F3864", "#2E75B6", "#9DC3E6", "#C0392B", "#7F7F7F"
MODEL_COLOURS = {"LR": LIGHT, "RF": BLUE, "XGB": NAVY}
RUNG_COLOURS = {"R1": NAVY, "R2": RED, "R3": GREY}


def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)
    print("wrote", name)


def box(ax, x, y, w, h, title, body="", fc="#EEF3F9", ec=NAVY, tc=NAVY):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                                fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h * (0.66 if body else 0.5), title, ha="center", va="center",
            fontsize=11, fontweight="bold", color=tc)
    if body:
        ax.text(x + w / 2, y + h * 0.3, body, ha="center", va="center", fontsize=9, color="black")


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                 color=GREY, lw=1.3))


# ----------------------------------------------------------- 1 ladder example
def fig_ladder():
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 7)
    rows = [("Legitimate URL (PhiUSIIL)", ["https://www.taps.org", "taps.org", "taps.org"]),
            ("Phishing URL (PhiUSIIL)", ["http://www.brisai[.]ga/ato/", "brisai[.]ga/ato", "brisai[.]ga"])]
    y = 6.4
    for title, forms in rows:
        ax.text(0.1, y, title, fontsize=11, fontweight="bold")
        y -= 0.85
        for rung, form in zip(RUNGS, forms):
            ax.text(0.3, y, f"{rung}  {RUNG_NAMES[rung]}", fontsize=10, va="center")
            ax.add_patch(FancyBboxPatch((3.0, y - 0.3), 6.6, 0.6, boxstyle="round,pad=0.02",
                                        fc="white", ec=GREY))
            ax.text(3.15, y, form, family="monospace", fontsize=10, va="center")
            y -= 0.75
        y -= 0.2
    save(fig, "fig_ladder.png")


# ----------------------------------------------------------- 2 pipeline
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    box(ax, 1, 8.6, 8, 1.1, "Public phishing URL corpora",
        "PhiUSIIL  |  LegitPhish  |  URL-Phish  |  Mowar and Jain")
    ax.add_patch(FancyBboxPatch((1, 5.6), 8, 2.4, boxstyle="round,pad=0.02", fc="#FBE9E7", ec=RED, lw=1.6))
    ax.text(5, 7.65, "Canonicalisation control", ha="center", fontsize=11, fontweight="bold", color=RED)
    ax.text(5, 7.3, "the only factor varied; all other stages fixed", ha="center", fontsize=9, style="italic", color=RED)
    for i, (r, d) in enumerate([("R1 raw", "URL as shipped"), ("R2 canonical", "scheme, www., trailing\nseparators removed"),
                                ("R3 host-only", "registrable host;\npath dropped")]):
        box(ax, 1.25 + i * 2.55, 5.8, 2.4, 1.25, r, d, fc="white", ec=RED, tc="black")
    box(ax, 1, 4.0, 8, 1.1, "Feature extraction", "24 lexical features  |  character n-gram TF-IDF (300k)")
    box(ax, 1, 2.4, 8, 1.1, "Partitioning", "stratified random  |  registrable-domain-disjoint (70/30, seed 42)")
    box(ax, 1, 0.5, 8, 1.4, "Models and evaluation",
        "LR  |  RF  |  XGBoost  |  n-gram LR\naccuracy, F1, ROC-AUC, bootstrap CI, McNemar, base-rate precision")
    for y1, y2 in [(8.6, 8.0), (5.6, 5.1), (4.0, 3.5), (2.4, 1.9)]:
        arrow(ax, 5, y1, 5, y2)
    save(fig, "fig_pipeline.png")


# ----------------------------------------------------------- 3 formatting (Table II)
def fig_formatting():
    t = R["phiusiil"]["formatting"]
    names = list(t)
    leg = [t[n]["legitimate"] for n in names]
    ph = [t[n]["phishing"] for n in names]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    yy = np.arange(len(names))
    ax.barh(yy - 0.2, leg, 0.38, color=BLUE, label="Legitimate")
    ax.barh(yy + 0.2, ph, 0.38, color=RED, label="Phishing")
    for i, (a, b) in enumerate(zip(leg, ph)):
        ax.text(a + 1, i - 0.2, f"{a:.2f}%", va="center", fontsize=9)
        ax.text(b + 1, i + 0.2, f"{b:.2f}%", va="center", fontsize=9)
    ax.set_yticks(yy, names); ax.invert_yaxis(); ax.set_xlim(0, 115)
    ax.set_xlabel("Share of URLs in class (%)"); ax.legend(loc="lower right", frameon=False)
    save(fig, "fig_formatting.png")


# ----------------------------------------------------------- 4 shipped feature stumps
def fig_stumps():
    s = R["shipped_features"]["stump_accuracy"]
    items = list(s.items())[:15]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    names = [k for k, _ in items]; vals = [v for _, v in items]
    cols = [RED if v > 0.9 else BLUE for v in vals]
    ax.barh(names, vals, color=cols)
    ax.axvline(R["shipped_features"]["random_split_baseline"], color="black", ls="--", lw=1)
    ax.text(R["shipped_features"]["random_split_baseline"] + 0.003, len(names) - 0.6,
            "majority baseline", fontsize=9)
    ax.axvline(0.9, color=RED, ls=":", lw=1)
    for i, v in enumerate(vals):
        ax.text(v + 0.003, i, f"{v:.4f}", va="center", fontsize=8)
    ax.invert_yaxis(); ax.set_xlim(0.5, 1.03); ax.set_xlabel("Single-feature (depth-1 stump) test accuracy")
    save(fig, "fig_stumps.png")


# ----------------------------------------------------------- 5 accuracy by representation
def fig_accuracy():
    m = R["main"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), sharey=True)
    x = np.arange(3)
    for ax, split, title in zip(axes, ("random", "domain"), ("Stratified random split", "Registrable-domain-disjoint split")):
        for k, model in enumerate(("LR", "RF", "XGB")):
            vals = [m[f"{r}|{split}|{model}"]["accuracy"] for r in RUNGS]
            ax.bar(x + (k - 1) * 0.26, vals, 0.25, color=MODEL_COLOURS[model], label=model)
            for xi, v in zip(x + (k - 1) * 0.26, vals):
                ax.text(xi, v + 0.004, f"{v:.3f}", ha="center", fontsize=7, rotation=90)
        ax.set_xticks(x, [f"{r}\n{RUNG_NAMES[r]}" for r in RUNGS]); ax.set_title(title, fontsize=11)
        ax.set_ylim(0.6, 1.05)
    axes[0].set_ylabel("Test accuracy"); axes[1].legend(frameon=False, loc="upper right")
    save(fig, "fig_accuracy_by_representation.png")


# ----------------------------------------------------------- 6 ROC
def fig_roc():
    d = np.load(ROOT / "results" / "phiusiil_domain_test.npz")
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    for r in RUNGS:
        fpr, tpr, _ = roc_curve(d["y"], d[f"{r}_XGB"])
        auc = R["main"][f"{r}|domain|XGB"]["roc_auc"]
        ax.plot(fpr, tpr, color=RUNG_COLOURS[r], lw=1.8, label=f"{r} {RUNG_NAMES[r]} (AUC {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#BBB", ls="--", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    save(fig, "fig_roc.png")


# ----------------------------------------------------------- 7 McNemar
def fig_mcnemar():
    mc = R["mcnemar"]
    pairs = [("R1_vs_R2", "R1 vs R2"), ("R2_vs_R3", "R2 vs R3"), ("R1_vs_R3", "R1 vs R3")]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    x = np.arange(len(pairs))
    a = [mc[k]["a_right_b_wrong"] for k, _ in pairs]
    b = [mc[k]["a_wrong_b_right"] for k, _ in pairs]
    ax.bar(x - 0.2, a, 0.38, color=NAVY, label="first right, second wrong")
    ax.bar(x + 0.2, b, 0.38, color=LIGHT, label="first wrong, second right")
    for xi, v in zip(x - 0.2, a): ax.text(xi, v + 80, f"{v:,}", ha="center", fontsize=9)
    for xi, v in zip(x + 0.2, b): ax.text(xi, v + 80, f"{v:,}", ha="center", fontsize=9)
    ax.set_xticks(x, [l for _, l in pairs]); ax.set_ylabel("Discordant test URLs")
    ax.legend(frameon=False)
    save(fig, "fig_mcnemar.png")


# ----------------------------------------------------------- 8 n-gram vs XGB
def fig_ngram():
    if "ngram" not in R:
        return
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    x = np.arange(3)
    xgb = [R["main"][f"{r}|domain|XGB"]["accuracy"] for r in RUNGS]
    ng = [R["ngram"][r]["accuracy"] for r in RUNGS]
    ax.plot(x, xgb, "o-", color=NAVY, lw=2, label="XGBoost, 24 lexical features")
    ax.plot(x, ng, "s--", color=RED, lw=2, label="Char n-gram TF-IDF + LR (300k)")
    for xi, v in zip(x, xgb): ax.text(xi + 0.04, v - 0.012, f"{v:.4f}", fontsize=9, color=NAVY)
    for xi, v in zip(x, ng): ax.text(xi + 0.04, v + 0.006, f"{v:.4f}", fontsize=9, color=RED)
    ax.set_xticks(x, [f"{r} {RUNG_NAMES[r]}" for r in RUNGS]); ax.set_ylabel("Accuracy (domain split)")
    ax.set_ylim(0.75, 1.02); ax.legend(frameon=False, loc="upper right")
    save(fig, "fig_ngram.png")


# ----------------------------------------------------------- 9 cross corpus
def fig_cross_corpus():
    cc = R["cross_corpus"]
    names = list(cc)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), gridspec_kw={"width_ratios": [1.6, 1]})
    x = np.arange(len(names))
    for k, r in enumerate(RUNGS):
        axes[0].bar(x + (k - 1) * 0.26, [cc[n][r] for n in names], 0.25, color=RUNG_COLOURS[r],
                    label=f"{r} {RUNG_NAMES[r]}")
    axes[0].set_xticks(x, names, fontsize=9); axes[0].set_ylim(0.6, 1.02)
    axes[0].set_ylabel("XGBoost accuracy (domain split)")
    axes[0].legend(frameon=False, fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    dc = [cc[n]["delta_cos"] for n in names]
    axes[1].bar(x, dc, color=[RED if v > 0.05 else GREY for v in dc])
    for xi, v in zip(x, dc): axes[1].text(xi, v + (0.004 if v >= 0 else -0.012), f"{v:+.4f}", ha="center", fontsize=8)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xticks(x, names, fontsize=8, rotation=15); axes[1].set_ylabel("Δcos = R1 − R2 accuracy")
    save(fig, "fig_cross_corpus.png")


# ----------------------------------------------------------- 10 base rate
def fig_base_rate():
    br = R["base_rate"]
    prev = [float(p) for p in br["R2"]["precision"]]
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    for r in ("R2", "R3"):
        vals = [br[r]["precision"][str(p)] for p in prev]
        ax.plot(prev, vals, "o-", color=RUNG_COLOURS[r], lw=2, label=f"{r} {RUNG_NAMES[r]}")
        for p, v in zip(prev, vals):
            ax.text(p, v * 1.25, f"{v:.4f}", fontsize=8, ha="center", color=RUNG_COLOURS[r])
    ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
    ax.set_xlabel("Simulated phishing prevalence"); ax.set_ylabel("Precision at 90% recall")
    ax.set_xticks(prev, ["50%", "10%", "5%", "0.05%"]); ax.legend(frameon=False)
    save(fig, "fig_base_rate.png")


# ----------------------------------------------------------- 11 feature importance
def fig_importance():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, r in zip(axes, ("R1", "R2")):
        mdl = joblib.load(ROOT / "models" / f"xgb_{r}.joblib")
        imp = mdl.get_booster().get_score(importance_type="gain")
        vals = np.array([imp.get(f"f{i}", imp.get(n, 0.0)) for i, n in enumerate(FEATURE_NAMES)])
        vals = vals / vals.sum()
        order = np.argsort(vals)[::-1][:10]
        ax.barh([FEATURE_NAMES[i] for i in order], vals[order], color=RUNG_COLOURS[r])
        ax.invert_yaxis(); ax.set_title(f"{r} {RUNG_NAMES[r]}", fontsize=11)
        ax.set_xlabel("Share of total gain")
    save(fig, "fig_importance.png")


# ----------------------------------------------------------- 12 score distributions
def fig_scores():
    d = np.load(ROOT / "results" / "phiusiil_domain_test.npz")
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    bins = np.linspace(0, 1, 41)
    for ax, r in zip(axes, RUNGS):
        p = d[f"{r}_XGB"]
        ax.hist(p[d["y"] == 0], bins, color=BLUE, alpha=0.75, label="Legitimate")
        ax.hist(p[d["y"] == 1], bins, color=RED, alpha=0.6, label="Phishing")
        ax.set_yscale("log"); ax.set_title(f"{r} {RUNG_NAMES[r]}", fontsize=11); ax.set_xlabel("P(phishing)")
    axes[0].set_ylabel("Test URLs (log)"); axes[2].legend(frameon=False, fontsize=8)
    save(fig, "fig_scores.png")


# ----------------------------------------------------------- 13 dataset composition
def fig_datasets():
    from audit import data
    names = data.CORPORA
    leg, ph = [], []
    for n in names:
        d = data.load(n); leg.append(int((d.y == 0).sum())); ph.append(int(d.y.sum()))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(names))
    ax.bar(x, leg, color=BLUE, label="Legitimate")
    ax.bar(x, ph, bottom=leg, color=RED, label="Phishing")
    for xi, a, b in zip(x, leg, ph):
        ax.text(xi, a + b + 2500, f"{a + b:,}", ha="center", fontsize=9)
    ax.set_xticks(x, names); ax.set_ylabel("URLs"); ax.legend(frameon=False)
    ax.set_ylim(0, max(np.add(leg, ph)) * 1.12)
    save(fig, "fig_datasets.png")


# ----------------------------------------------------------- 14 working-model flow
def fig_app_flow():
    fig, ax = plt.subplots(figsize=(9.5, 3.2))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 3.4)
    steps = [("User input", "URL or list\nof URLs"), ("Canonicalise", "R1, R2, R3\nforms"),
             ("Featurise", "24 lexical\nfeatures each"), ("Three XGBoost\nmodels", "trained per rung,\ndomain split"),
             ("Verdict", "P(phishing),\nstability warning")]
    w = 1.7
    for i, (t, b) in enumerate(steps):
        x = 0.1 + i * 2.0
        box(ax, x, 0.8, w, 1.9, t, b, fc="#FBE9E7" if i == 1 else "#EEF3F9", ec=RED if i == 1 else NAVY)
        if i < len(steps) - 1:
            arrow(ax, x + w + 0.02, 1.75, x + 2.0 - 0.02, 1.75)
    save(fig, "fig_app_flow.png")


if __name__ == "__main__":
    for f in (fig_ladder, fig_pipeline, fig_formatting, fig_stumps, fig_accuracy, fig_roc,
              fig_mcnemar, fig_ngram, fig_cross_corpus, fig_base_rate, fig_importance,
              fig_scores, fig_datasets, fig_app_flow):
        f()
