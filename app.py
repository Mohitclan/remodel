"""Working model: interactive canonicalisation-control checker.

Run:  .venv/bin/streamlit run app.py --server.headless true
URL options:  ?url=<address> pre-fills the checker;  ?tab=compare|results|how opens that tab first
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from audit.canon import RUNGS, canonicalise, matches_template
from audit.features import FEATURE_NAMES, featurise

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Phishing Dataset Shortcut Checker", layout="wide")

# Plain-language names for the three models (one XGBoost model per URL representation)
MODEL = {
    "R1": {"name": "Model A: Raw URL", "sees": "the URL exactly as typed"},
    "R2": {"name": "Model B: Cleaned URL", "sees": "the URL with `https://`, `www.` and the trailing `/` removed"},
    "R3": {"name": "Model C: Name only", "sees": "only the website name, nothing after it"},
}
FEATURE_HELP = {
    "url_len": "length of the text", "host_len": "length of the website name",
    "path_len": "length of everything after the website name", "n_dots": "number of dots",
    "n_hyphens": "number of hyphens", "n_underscores": "number of underscores",
    "n_slashes": "number of slashes", "n_qmarks": "number of ?", "n_equals": "number of =",
    "n_ampersands": "number of &", "n_at": "number of @", "n_digits": "number of digits",
    "n_letters": "number of letters", "n_special": "number of other symbols",
    "digit_ratio": "share of digits", "letter_ratio": "share of letters",
    "special_ratio": "share of symbols", "url_entropy": "randomness of the text",
    "host_entropy": "randomness of the website name", "n_host_tokens": "parts in the website name",
    "n_path_tokens": "parts after the website name", "max_token_len": "longest word",
    "host_is_ip": "website name is an IP address", "host_digit_ratio": "share of digits in website name",
}
EXAMPLES = ["wikipedia.org", "https://www.wikipedia.org",
            "https://www.paypal-account-verify.com", "paypal-account-verify.com"]


@st.cache_resource
def load_models():
    return {r: joblib.load(ROOT / "models" / f"xgb_{r}.joblib") for r in RUNGS}


@st.cache_data
def load_samples():
    """Real PhiUSIIL URLs from the test portion (never seen in training), with true labels."""
    p = ROOT / "results" / "demo_samples.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data
def load_results():
    p = ROOT / "results" / "results.json"
    return json.loads(p.read_text()) if p.exists() else None


def score(url):
    """Phishing probability from each of the three models."""
    models = load_models()
    return {r: float(models[r].predict_proba(featurise([canonicalise(url, r)]))[0, 1]) for r in RUNGS}


def verdict(p):
    return "PHISHING" if p >= 0.5 else "SAFE"


def coloured(p):
    return f":red[**{verdict(p)}**]" if p >= 0.5 else f":green[**{verdict(p)}**]"


# ------------------------------------------------------------------ header
st.title("Phishing Dataset Shortcut Checker")
st.markdown(
    "Three AI models were trained on the same phishing dataset (**PhiUSIIL**, 235,795 URLs). "
    "They are identical except for **how much of the URL each one is allowed to see**. "
    "If their answers disagree for the same website, the raw model is judging the URL by its "
    "**formatting** instead of the **website itself**. That shortcut is what this project exposes.")
st.caption("Minor Project · IEEE ICCSAI 2026 Paper 3112 · Domain-Disjoint Splitting Is Not Enough")

with st.sidebar:
    st.header("Quick guide")
    st.markdown(
        "1. In **Check a URL**, type any web address in the box at the top and press Enter.\n"
        "2. Read the three model cards: each shows what that model saw and its answer.\n"
        "3. Read **What this means** for a plain explanation.\n"
        "4. **Dataset & Model** shows the data and how the models work.\n\n"
        "**Try this:** check `wikipedia.org`, then `https://www.wikipedia.org`. "
        "Same website, different answer from Model A.")
    st.divider()
    st.markdown("**Verdict colours**  \n:green[**SAFE**]: model thinks it is legitimate  \n"
                ":red[**PHISHING**]: model thinks it is a scam")
    st.markdown("**Phishing score**: how sure the model is that the URL is a scam, from 0% to 100%.")

params = st.query_params
order = ["check", "compare", "dataset", "results", "how"]
labels = {"check": "Check a URL", "compare": "Compare URLs", "dataset": "Dataset & Model",
          "results": "Project results", "how": "How it works"}
first = params.get("tab", "check")
if first in order:
    order.remove(first)
    order.insert(0, first)
tabs = dict(zip(order, st.tabs([labels[k] for k in order])))

# ------------------------------------------------------------------ check a URL
with tabs["check"]:
    if "url" not in st.session_state:
        st.session_state.url = params.get("url", "wikipedia.org")

    def use_example(u):
        st.session_state.url = u

    samples = load_samples()
    url = st.text_input("**Type a web address here and press Enter**", key="url",
                        placeholder="for example: google.com  or  https://www.google.com").strip()
    st.caption("Click in the box, select the old text (Cmd + A), type a new URL and press Enter.")
    groups = []
    if samples is not None:
        allp = samples[["p_R1", "p_R2", "p_R3"]]
        legit = samples[(samples.label == "Legitimate") & (allp < 0.5).all(axis=1)].url.head(3).tolist()
        phish = samples[(samples.label == "Phishing") & (allp >= 0.5).all(axis=1)].url.head(3).tolist()
        groups += [(":green[**Valid:**] real legitimate websites from the dataset", legit),
                   (":red[**Invalid:**] real phishing websites from the dataset", phish)]
    groups.append(("**Same website, different format** (shows the shortcut)", EXAMPLES))
    with st.expander("Or click a ready-made example"):
        for title, items in groups:
            st.markdown(title)
            for c, ex in zip(st.columns(4), items):
                c.button(ex, on_click=use_example, args=(ex,), width="stretch", key=f"ex-{ex}")

    if url:
        p = score(url)
        st.subheader("What each model sees and decides")
        for c, r in zip(st.columns(3), RUNGS):
            with c.container(border=True):
                st.markdown(f"#### {MODEL[r]['name']}")
                st.caption(f"Sees {MODEL[r]['sees']}:")
                st.code(canonicalise(url, r) or "(empty)", language=None, wrap_lines=True)
                st.markdown(f"### {coloured(p[r])}")
                st.progress(p[r], text=f"Phishing score: {p[r] * 100:.1f}%")

        known = samples[samples.url == url] if samples is not None else None
        if known is not None and len(known):
            truth = known.iloc[0].label
            right = [MODEL[r]["name"].split(":")[0] for r in RUNGS
                     if (p[r] >= 0.5) == (truth == "Phishing")]
            st.info(f"**This URL is in the PhiUSIIL test data. Its true label is {truth.upper()}.** "
                    f"Correct here: {', '.join(right) if right else 'none of the models'}. "
                    "The models never saw this URL during training.")

        st.subheader("What this means")
        a, b = verdict(p["R1"]), verdict(p["R2"])
        if a != b:
            st.warning(
                f"**Model A says {a}, but Model B says {b} for the same website.**  \n"
                "The only difference between what they saw is the formatting "
                "(`https://`, `www.` and the trailing `/`). So Model A's answer is based on **how the "
                "URL is written, not on the website itself**. It learned a shortcut from the dataset, "
                "because every legitimate URL in PhiUSIIL is written as `https://www.<site>`.")
        else:
            st.success(
                f"**Models A and B agree ({a}).** For this URL the answer does not depend on the "
                "formatting, so Model A is not relying on the shortcut here.")

        tpl = matches_template(url)
        with st.container(border=True):
            st.markdown("**The dataset shortcut, in one rule (no AI at all)**")
            st.markdown(
                "Every legitimate URL in PhiUSIIL looks exactly like `https://www.<site>` with nothing "
                "after it, and almost no phishing URL does. A one-line rule that only checks this "
                "format is **99.73% accurate** on the whole dataset.")
            st.markdown(
                f"This URL **{'matches' if tpl else 'does not match'}** that format, so the rule "
                f"would call it {':green[**SAFE**]' if tpl else ':red[**PHISHING**]'}.")

        with st.expander("Show the 24 numbers each model looked at"):
            feats = featurise([canonicalise(url, r) for r in RUNGS]).T
            feats.columns = ["Model A", "Model B", "Model C"]
            feats.insert(0, "What it measures", [FEATURE_HELP[f] for f in FEATURE_NAMES])
            st.dataframe(feats.round(3), width="stretch")
            st.caption("Each model turns its version of the URL into these 24 numbers, then decides. "
                       "Notice how path_len and n_slashes change between Model A and Model B.")

# ------------------------------------------------------------------ compare URLs
with tabs["compare"]:
    st.markdown(
        "Paste several web addresses, one per line. Pairs that point to the **same website** but are "
        "written differently show the shortcut clearly: Model A changes its answer, while Models B "
        "and C give both versions the same score.")
    text = st.text_area("Web addresses (one per line)",
                        "https://www.wikipedia.org\nwikipedia.org\nhttps://www.sbi.co.in\nhttp://sbi.co.in/\n"
                        "https://www.paypal-account-verify.com\npaypal-account-verify.com", height=170)
    urls = [u.strip() for u in text.splitlines() if u.strip()]
    if urls:
        rows = []
        for u in urls:
            p = score(u)
            rows.append({"Web address": u,
                         "Looks like https://www.<site>?": "Yes" if matches_template(u) else "No",
                         "Model A (raw)": f"{verdict(p['R1'])}  {p['R1'] * 100:.0f}%",
                         "Model B (cleaned)": f"{verdict(p['R2'])}  {p['R2'] * 100:.0f}%",
                         "Model C (name only)": f"{verdict(p['R3'])}  {p['R3'] * 100:.0f}%",
                         "Shortcut?": "Yes" if verdict(p["R1"]) != verdict(p["R2"]) else "No"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Percentages are phishing scores. \"Shortcut? Yes\" means Model A disagrees with "
                   "Model B, so Model A's answer depends only on the formatting.")

# ------------------------------------------------------------------ dataset & model
with tabs["dataset"]:
    res = load_results()
    samples = load_samples()
    st.subheader("The dataset")
    if res is not None:
        P = res["phiusiil"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total URLs", f"{P['n']:,}")
        c2.metric("Legitimate (valid)", f"{P['n_legitimate']:,}")
        c3.metric("Phishing (invalid)", f"{P['n_phishing']:,}")
        c4.metric("Different websites", f"{P['distinct_domains']:,}")
    st.markdown(
        "**PhiUSIIL Phishing URL Dataset** (2024), downloaded from the UCI Machine Learning Repository "
        "(dataset 967, free licence CC BY 4.0). Each row is one URL with a label: legitimate or phishing. "
        "We also tested three more public datasets: LegitPhish and URL-Phish (Mendeley Data) and "
        "Mowar and Jain (Zenodo).")
    if res is not None:
        cc = pd.DataFrame(res["cross_corpus"]).T
        st.dataframe(pd.DataFrame({"Dataset": cc.index, "URLs": cc["n"].astype(int).map("{:,}".format),
                                   "Used for": ["Main dataset: trains the app's models",
                                                "Comparison", "Comparison", "Comparison"]}),
                     hide_index=True, width="stretch")

    st.markdown("**Real examples from the test data, with the true label and each model's answer**")
    if samples is not None:
        if "shuffle" not in st.session_state:
            st.session_state.shuffle = 0
        if st.button("Show different examples"):
            st.session_state.shuffle += 1
        k = st.session_state.shuffle
        show = pd.concat([samples[samples.label == "Legitimate"].sample(5, random_state=k),
                          samples[samples.label == "Phishing"].sample(5, random_state=k)])

        def mark(row, r):
            v = verdict(row[f"p_{r}"])
            ok = (v == "PHISHING") == (row.label == "Phishing")
            return f"{'✓' if ok else '✗'} {v}"

        st.dataframe(pd.DataFrame({
            "Web address": show.url, "True label": show.label.str.upper(),
            "Model A (raw)": [mark(r, "R1") for _, r in show.iterrows()],
            "Model B (cleaned)": [mark(r, "R2") for _, r in show.iterrows()],
            "Model C (name only)": [mark(r, "R3") for _, r in show.iterrows()],
        }), hide_index=True, width="stretch")
        st.caption("✓ = correct, ✗ = wrong. These URLs come from the test portion, from websites the "
                   "models never saw in training. Model A is almost always right because every "
                   "legitimate URL starts with `https://www.`; Models B and C cannot use that shortcut.")

    st.subheader("The model")
    st.markdown(
        "Each of the three models is an **XGBoost** classifier (gradient-boosted decision trees: "
        "300 trees, depth 6). It does not read the URL as text: it first turns the URL into **24 "
        "numbers** (length, number of dots, digits, symbols, randomness and so on) and decides from those.")
    if res is not None:
        m = res["main"]
        st.dataframe(pd.DataFrame({
            "Model": [MODEL[r]["name"] for r in RUNGS],
            "What it sees": [MODEL[r]["sees"].replace("`", "") for r in RUNGS],
            "Accuracy on unseen websites": [f"{m[f'{r}|domain|XGB']['accuracy'] * 100:.2f}%" for r in RUNGS],
            "Saved model file": [f"models/xgb_{r}.joblib" for r in RUNGS],
        }), hide_index=True, width="stretch")
        st.markdown(
            f"**Training:** {P['domain_split_train']:,} URLs.  **Testing:** {P['domain_split_test']:,} URLs "
            "from websites that do not appear in training (70/30 split, fixed random seed 42).")
    with st.expander("The 24 numbers (features) every model uses"):
        st.dataframe(pd.DataFrame({"Feature": FEATURE_NAMES,
                                   "What it measures": [FEATURE_HELP[f] for f in FEATURE_NAMES]}),
                     hide_index=True, width="stretch")
    with st.expander("Where everything is on this laptop"):
        st.markdown(
            f"Project folder: `{ROOT}`\n\n"
            "- `data/raw/`: the four downloaded datasets (PhiUSIIL is "
            "`data/raw/phiusiil/PhiUSIIL_Phishing_URL_Dataset.csv`)\n"
            "- `models/`: the three trained models used by this app\n"
            "- `audit/`: the code (`canon.py` rewrites URLs, `features.py` computes the 24 numbers, "
            "`evaluate.py` trains and tests)\n"
            "- `run_audit.py`: runs every experiment; `results/results.json` holds every number\n"
            "- `figures/`: all charts used in the report")

# ------------------------------------------------------------------ results
with tabs["results"]:
    res = load_results()
    if res is None:
        st.info("Run `python run_audit.py` first.")
    else:
        m = res["main"]
        acc = lambda r, s: m[f"{r}|{s}|XGB"]["accuracy"] * 100
        st.markdown("**Key findings on PhiUSIIL (XGBoost model)**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Raw URLs, normal split", f"{acc('R1', 'random'):.2f}%")
        c2.metric("Raw URLs, strict domain split", f"{acc('R1', 'domain'):.2f}%",
                  f"{acc('R1', 'domain') - acc('R1', 'random'):+.2f} pts")
        c3.metric("After removing `https://www.`", f"{acc('R2', 'domain'):.2f}%",
                  f"{acc('R2', 'domain') - acc('R1', 'domain'):+.2f} pts")
        c4.metric("One rule, no AI", f"{res['phiusiil']['template_rule']['accuracy'] * 100:.2f}%")
        st.markdown(
            "- The usual fix for leakage (keeping each website on only one side of the train/test "
            "split) **does not lower accuracy at all**, so it cannot catch this shortcut.\n"
            "- Simply deleting `https://`, `www.` and the trailing `/` from **every** URL removes about "
            "**16 points** of accuracy. Those characters carry no real phishing information, so that "
            "accuracy came from the shortcut.\n"
            f"- At a realistic phishing rate (1 in 2,000 URLs), the cleaned model would raise about "
            f"**{(1 - res['base_rate']['R2']['precision']['0.0005']) / res['base_rate']['R2']['precision']['0.0005']:,.0f} "
            "false alarms for every real phishing URL it catches**.")
        img = ROOT / "figures" / "fig_accuracy_by_representation.png"
        if img.exists():
            st.image(str(img), caption="Accuracy of three model types for each URL version and split")
        with st.expander("Full results tables"):
            tbl = pd.DataFrame([{"URL version": {"R1": "A: raw", "R2": "B: cleaned", "R3": "C: name only"}[r],
                                 "Split": s, **{k: round(m[f'{r}|{s}|{k}']['accuracy'], 4) for k in ("LR", "RF", "XGB")}}
                                for r in RUNGS for s in ("random", "domain")])
            st.dataframe(tbl, hide_index=True, width="stretch")
            cc = pd.DataFrame(res["cross_corpus"]).T[["n", "R1", "R2", "R3", "delta_cos", "rule", "base"]]
            cc.columns = ["URLs", "A: raw", "B: cleaned", "C: name only", "Drop A to B", "One rule", "Baseline"]
            st.markdown("**Same test on four datasets** (a large *Drop A to B* means the shortcut is present)")
            st.dataframe(cc.astype(float).round(4), width="stretch")

# ------------------------------------------------------------------ how it works
with tabs["how"]:
    st.markdown(
        "#### The idea\n"
        "If a dataset's two classes are written in different styles, an AI model can learn the style "
        "instead of the real signal. To test for this, we rewrite every URL in three ways, **the same way "
        "for safe and phishing URLs**, and train one model on each version:")
    img = ROOT / "figures" / "fig_ladder.png"
    if img.exists():
        st.image(str(img), width=650,
                 caption="The three URL versions: R1 = Model A (raw), R2 = Model B (cleaned), R3 = Model C (name only)")
    st.markdown(
        "#### The steps\n"
        "1. **Data:** four public phishing datasets, with PhiUSIIL as the main one.\n"
        "2. **Rewrite:** create the three URL versions above.\n"
        "3. **Measure:** turn each version into 24 numbers (length, dots, digits and so on).\n"
        "4. **Split:** keep every website on only one side of the train/test split, so no model can "
        "memorise website names.\n"
        "5. **Train and test:** train an XGBoost model on each version and compare the accuracies.\n\n"
        "If accuracy falls sharply from version A to version B, which only deletes formatting, the "
        "dataset contains a shortcut. This app uses the three trained models from step 5.")
    img = ROOT / "figures" / "fig_pipeline.png"
    if img.exists():
        st.image(str(img), caption="Full audit pipeline", width=700)
