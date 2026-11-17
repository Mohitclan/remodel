"""Terminal demo of the working model (start it with:  run cli).

Type any URL to see, step by step, what each of the three models receives,
the 24 numbers it computes, which numbers pushed its decision, and its answer.
Type  help  for the other commands (valid, invalid, pairs, compare, sample,
dataset, model, quit).
"""
import json
import random
import readline  # noqa: F401  (arrow keys and history at the prompt)
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from audit.canon import RUNGS, canonicalise, matches_template
from audit.features import FEATURE_NAMES, featurise

ROOT = Path(__file__).resolve().parent
con = Console(highlight=False)

NAME = {"R1": "Model A", "R2": "Model B", "R3": "Model C"}
SHORT = {"R1": "raw URL", "R2": "cleaned URL", "R3": "name only"}
SEES = {"R1": "the URL exactly as typed",
        "R2": "https://, http://, www. and trailing / removed",
        "R3": "only the website name"}
HELP = {
    "url_len": "length of the text", "host_len": "length of the website name",
    "path_len": "length after the website name", "n_dots": "number of dots",
    "n_hyphens": "number of hyphens", "n_underscores": "number of underscores",
    "n_slashes": "number of slashes", "n_qmarks": "number of ?", "n_equals": "number of =",
    "n_ampersands": "number of &", "n_at": "number of @", "n_digits": "number of digits",
    "n_letters": "number of letters", "n_special": "number of symbols",
    "digit_ratio": "share of digits", "letter_ratio": "share of letters",
    "special_ratio": "share of symbols", "url_entropy": "randomness of the text",
    "host_entropy": "randomness of website name", "n_host_tokens": "parts in website name",
    "n_path_tokens": "parts after website name", "max_token_len": "longest word",
    "host_is_ip": "website name is an IP", "host_digit_ratio": "digits in website name",
}
PAIRS = [("https://www.wikipedia.org", "wikipedia.org"),
         ("https://www.sbi.co.in", "http://sbi.co.in/"),
         ("https://www.paypal-account-verify.com", "paypal-account-verify.com")]


# ------------------------------------------------------------------ loading
def load():
    with con.status("[bold]Loading the three trained models and the dataset..."):
        models = {r: joblib.load(ROOT / "models" / f"xgb_{r}.joblib") for r in RUNGS}
        lookup = pd.read_csv(ROOT / "results" / "url_lookup.csv.gz")
        res = json.loads((ROOT / "results" / "results.json").read_text())
        samples = pd.read_csv(ROOT / "results" / "demo_samples.csv")
    return models, lookup, res, samples


MODELS, LOOKUP, RES, SAMPLES = load()
BY_URL = dict(zip(LOOKUP.url, zip(LOOKUP.label, LOOKUP.split)))


def score(url):
    """Everything each model does with one URL."""
    out = {}
    for r in RUNGS:
        seen = canonicalise(url, r)
        X = featurise([seen])
        booster = MODELS[r].get_booster()
        contrib = booster.predict(xgb.DMatrix(X), pred_contribs=True)[0][:-1]
        out[r] = {"seen": seen, "X": X.iloc[0], "contrib": contrib,
                  "p": float(MODELS[r].predict_proba(X)[0, 1])}
    return out


def verdict(p):
    return ("PHISHING", "bold white on red") if p >= 0.5 else ("SAFE", "bold white on green")


def bar(p, width=24):
    n = round(p * width)
    colour = "red" if p >= 0.5 else "green"
    return Text("█" * n, style=colour) + Text("░" * (width - n), style="grey50")


def fmt(v):
    return f"{v:.3f}" if v != int(v) else f"{int(v)}"


# ------------------------------------------------------------------ one URL
def check(url, detail=True):
    url = url.strip()
    if not url:
        return
    s = score(url)
    con.print(Rule(f"[bold]Checking: [cyan]{url}"))

    # step 1: input to each model
    t = Table(box=box.SIMPLE_HEAVY, title="STEP 1  What each model receives", title_justify="left",
              title_style="bold yellow")
    t.add_column("Model", style="bold")
    t.add_column("Rule")
    t.add_column("Text the model actually sees", style="cyan")
    for r in RUNGS:
        t.add_row(NAME[r], SEES[r], s[r]["seen"])
    con.print(t)

    # step 2: 24 features
    if detail:
        t = Table(box=box.SIMPLE, title="STEP 2  The 24 numbers each model computes from its text "
                  "(rows in [magenta]magenta[/] differ between models)",
                  title_justify="left", title_style="bold yellow")
        t.add_column("#", justify="right", style="grey50")
        t.add_column("Feature")
        t.add_column("Meaning", style="grey62")
        for r in RUNGS:
            t.add_column(NAME[r], justify="right")
        for i, f in enumerate(FEATURE_NAMES, 1):
            vals = [s[r]["X"][f] for r in RUNGS]
            diff = len({round(v, 6) for v in vals}) > 1
            st = "magenta" if diff else None
            t.add_row(str(i), Text(f, style=st), HELP[f], *[Text(fmt(v), style=st) for v in vals])
        con.print(t)

    # step 3: why (feature contributions)
    if detail:
        panels = []
        for r in RUNGS:
            c = s[r]["contrib"]
            order = np.argsort(-np.abs(c))[:5]
            t = Table(box=box.MINIMAL, show_header=True, pad_edge=False)
            t.add_column("Feature")
            t.add_column("Value", justify="right")
            t.add_column("Push", justify="right")
            for k in order:
                f = FEATURE_NAMES[k]
                push = Text(f"{c[k]:+.2f}", style="red" if c[k] > 0 else "green")
                t.add_row(f, fmt(s[r]["X"][f]), push)
            panels.append(Panel(t, title=f"[bold]{NAME[r]}", width=40))
        con.print(Text("STEP 3  Which numbers pushed each decision most "
                       "(red = towards phishing, green = towards safe)", style="bold yellow"))
        con.print(Columns(panels))

    # step 4: decision
    t = Table(box=box.ROUNDED, title="STEP 4  Decision", title_justify="left", title_style="bold yellow")
    t.add_column("Model", style="bold")
    t.add_column("Phishing score", justify="right")
    t.add_column("")
    t.add_column("Answer", justify="center")
    for r in RUNGS:
        v, st = verdict(s[r]["p"])
        t.add_row(NAME[r], f"{s[r]['p'] * 100:.1f}%", bar(s[r]["p"]), Text(f" {v} ", style=st))
    con.print(t)

    # dataset check + interpretation
    lines = []
    if url in BY_URL:
        label, part = BY_URL[url]
        right = [NAME[r] for r in RUNGS if (s[r]["p"] >= 0.5) == (label == "Phishing")]
        seen_msg = ("[green]was NOT seen in training (test set)[/]" if part == "test"
                    else "[yellow]was in the training set[/]")
        lines.append(f"[bold]In the PhiUSIIL dataset:[/] true label [bold]{label.upper()}[/], "
                     f"and this website {seen_msg}.")
        lines.append(f"Correct: {', '.join(right) if right else 'none'}.")
    else:
        lines.append("[bold]Not in the dataset:[/] a new URL, so no true label is available.")
    tpl = matches_template(url)
    lines.append(f"[bold]One-line format rule (no AI):[/] looks like https://www.<site> with nothing after it? "
                 f"{'[green]YES -> rule says SAFE' if tpl else '[red]NO -> rule says PHISHING'}[/]")
    a, b = s["R1"]["p"] >= 0.5, s["R2"]["p"] >= 0.5
    if a != b:
        lines.append("[bold red]Model A and Model B disagree.[/] The only difference between them is the "
                     "formatting (https://, www., /), so Model A is judging the [bold]format[/], not the "
                     "website. This is the dataset shortcut the project exposes.")
        if s["R1"]["seen"] == s["R2"]["seen"]:
            lines.append("Here all models receive the [bold]same text[/] and the same 24 numbers, yet Model A "
                         "says phishing: in training, every legitimate URL started with https://www., so "
                         "Model A learned that a short URL without it must be phishing (see url_len in Step 3).")
    else:
        lines.append("[bold]Model A and Model B agree[/], so for this URL the answer does not depend "
                     "on the formatting.")
    con.print(Panel("\n".join(lines), title="[bold]What this means", border_style="blue"))


# ------------------------------------------------------------------ other commands
def compare(urls):
    t = Table(box=box.ROUNDED, title="Side-by-side comparison", title_style="bold yellow")
    t.add_column("URL", style="cyan")
    for r in RUNGS:
        t.add_column(f"{NAME[r]}\n({SHORT[r]})", justify="center")
    t.add_column("True label")
    for u in urls:
        s = score(u)
        cells = []
        for r in RUNGS:
            v, st = verdict(s[r]["p"])
            cells.append(Text(f"{v} {s[r]['p'] * 100:.1f}%", style=st.replace("bold white on ", "bold ")))
        t.add_row(u, *cells, BY_URL.get(u, ("not in dataset",))[0])
    con.print(t)


def sample(n=10):
    rows = pd.concat([SAMPLES[SAMPLES.label == "Legitimate"].sample(n // 2),
                      SAMPLES[SAMPLES.label == "Phishing"].sample(n // 2)])
    t = Table(box=box.ROUNDED, title="Random real URLs from the TEST set (never seen in training)",
              title_style="bold yellow")
    t.add_column("URL", style="cyan", max_width=48)
    t.add_column("True label")
    for r in RUNGS:
        t.add_column(NAME[r], justify="center")
    for _, row in rows.iterrows():
        cells = []
        for r in RUNGS:
            p = row[f"p_{r}"]
            ok = (p >= 0.5) == (row.label == "Phishing")
            cells.append(Text(f"{'✓' if ok else '✗'} {p * 100:.0f}%", style="green" if ok else "red"))
        t.add_row(row.url, row.label.upper(), *cells)
    con.print(t)
    con.print("[grey62]Type  check <url>  or paste any of these URLs to see the full breakdown.")


def dataset():
    P = RES["phiusiil"]
    t = Table(box=box.ROUNDED, title="Dataset: PhiUSIIL Phishing URL Dataset (UCI, 2024)",
              title_style="bold yellow", show_header=False)
    t.add_column(style="bold")
    t.add_column()
    t.add_row("File", "data/raw/phiusiil/PhiUSIIL_Phishing_URL_Dataset.csv")
    t.add_row("Total URLs", f"{P['n']:,}")
    t.add_row("Legitimate", f"{P['n_legitimate']:,}")
    t.add_row("Phishing", f"{P['n_phishing']:,}")
    t.add_row("Different websites", f"{P['distinct_domains']:,}")
    t.add_row("Training set", f"{P['domain_split_train']:,} URLs")
    t.add_row("Test set", f"{P['domain_split_test']:,} URLs (websites never seen in training)")
    t.add_row("Legit URLs in https://www.<site> format", "100%")
    t.add_row("Phishing URLs in that format", "0.62%")
    con.print(t)
    t = Table(box=box.SIMPLE, title="Other datasets checked", title_style="bold yellow")
    t.add_column("Dataset")
    t.add_column("URLs", justify="right")
    for k, v in RES["cross_corpus"].items():
        t.add_row(k, f"{int(v['n']):,}")
    con.print(t)
    con.print("[bold]Random rows from the file (label and which part of the split):")
    head = LOOKUP.sample(6, random_state=random.randint(0, 9999))[["url", "label", "split"]]
    t = Table(box=box.SIMPLE)
    for c in ["url", "label", "split"]:
        t.add_column(c)
    for _, r in head.iterrows():
        t.add_row(r.url, r.label, r.split)
    con.print(t)


def model():
    m = RES["main"]
    t = Table(box=box.ROUNDED, title="The model: XGBoost (300 trees, depth 6, learning rate 0.1)",
              title_style="bold yellow")
    t.add_column("Model", style="bold")
    t.add_column("What it sees")
    t.add_column("Accuracy (unseen websites)", justify="right")
    t.add_column("File")
    for r in RUNGS:
        t.add_row(NAME[r], SEES[r], f"{m[f'{r}|domain|XGB']['accuracy'] * 100:.2f}%",
                  f"models/xgb_{r}.joblib")
    con.print(t)
    con.print("Each model turns its text into [bold]24 numbers[/] (type a URL to see them) "
              "and 300 decision trees vote on those numbers.\n"
              "Code: audit/canon.py (rewrites the URL), audit/features.py (24 numbers), "
              "audit/evaluate.py (training and testing), run_audit.py (all experiments).\n"
              "A one-line format rule with no AI scores [bold]99.73%[/]: Model A's 99.60% "
              "comes from that shortcut.")


def examples(kind):
    allp = SAMPLES[["p_R1", "p_R2", "p_R3"]]
    if kind == "valid":
        urls = SAMPLES[(SAMPLES.label == "Legitimate") & (allp < 0.5).all(axis=1)].url.head(3).tolist()
    else:
        urls = SAMPLES[(SAMPLES.label == "Phishing") & (allp >= 0.5).all(axis=1)].url.head(3).tolist()
    for u in urls:
        check(u, detail=False)
    con.print(f"[grey62]Showed short results. Type  check {urls[0]}  for the full 4-step breakdown.")


def help_():
    t = Table(box=box.SIMPLE, show_header=False)
    t.add_column(style="bold cyan")
    t.add_column()
    for c, d in [("<any url>", "full 4-step breakdown: input, 24 features, reasons, decision"),
                 ("valid", "3 real legitimate URLs from the test set"),
                 ("invalid", "3 real phishing URLs from the test set"),
                 ("pairs", "same website in two formats: shows the shortcut"),
                 ("compare <url> <url> ...", "several URLs side by side"),
                 ("sample", "10 random real URLs with true label and each model's answer"),
                 ("dataset", "dataset size, split and file location"),
                 ("model", "model details, accuracy and file locations"),
                 ("clear", "clear the screen"),
                 ("quit", "exit")]:
        t.add_row(c, d)
    con.print(Panel(t, title="[bold]Commands", border_style="cyan"))


def banner():
    con.print(Panel.fit(
        "[bold]PHISHING DATASET SHORTCUT CHECKER[/]  ·  terminal demo of the working model\n"
        "IEEE ICCSAI 2026 Paper 3112 · Domain-Disjoint Splitting Is Not Enough\n\n"
        f"Three XGBoost models trained on PhiUSIIL ({RES['phiusiil']['n']:,} URLs). Identical except for how "
        "much of the URL each sees:\n"
        "  [bold]Model A[/] raw URL   [bold]Model B[/] cleaned URL   [bold]Model C[/] website name only\n"
        "If A and B disagree on the same website, A is judging the format, not the website.",
        border_style="green"))
    help_()


def main():
    banner()
    while True:
        try:
            line = input("\nurl or command > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        cmd, *rest = line.split()
        c = cmd.lower()
        try:
            if c in ("quit", "exit", "q"):
                break
            elif c in ("help", "h", "?"):
                help_()
            elif c == "clear":
                con.clear()
            elif c == "valid":
                examples("valid")
            elif c == "invalid":
                examples("invalid")
            elif c == "pairs":
                for a, b in PAIRS:
                    compare([a, b])
                con.print("[bold]Same website each time.[/] Only Model A changes its answer, because only "
                          "it sees https://www. That is the shortcut.")
            elif c == "compare":
                compare(rest) if rest else con.print("Usage: compare <url> <url> ...")
            elif c == "sample":
                sample()
            elif c == "dataset":
                dataset()
            elif c == "model":
                model()
            elif c == "check":
                check(" ".join(rest))
            else:
                check(line)
        except Exception as e:  # keep the demo alive whatever is typed
            con.print(f"[red]Could not process that input: {e}")
    con.print("Bye.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        check(" ".join(sys.argv[1:]))
    else:
        main()
