"""Splits, models and statistics used by the audit (paper Sections IV-V)."""
import numpy as np
from scipy.stats import binomtest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

SEED = 42
TEST_SIZE = 0.30


# ---------------------------------------------------------------- splits
def random_split(y):
    idx = np.arange(len(y))
    return train_test_split(idx, test_size=TEST_SIZE, stratify=y, random_state=SEED)


def domain_split(groups):
    """Registrable-domain-disjoint split: no eTLD+1 occurs in both partitions."""
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    tr, te = next(gss.split(np.zeros(len(groups)), groups=groups))
    return tr, te


# ---------------------------------------------------------------- models
def make_model(name):
    if name == "LR":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, random_state=SEED))
    if name == "RF":
        return RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED)
    if name == "XGB":
        return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                             n_jobs=-1, random_state=SEED, eval_metric="logloss")
    raise ValueError(name)


def make_ngram_model():
    """char_wb 3-5 grams, TF-IDF, min_df 3, 300k features, L2 logistic regression."""
    return make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                        max_features=300_000, sublinear_tf=True),
        LogisticRegression(max_iter=2000, C=1.0, solver="liblinear",
                           random_state=SEED))


# ---------------------------------------------------------------- metrics
def scores(y_true, y_pred, y_prob):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def bootstrap_ci(y_true, y_pred, n=1000, alpha=0.05, seed=SEED):
    """95% percentile bootstrap CI of accuracy over test-set resamples."""
    rng = np.random.default_rng(seed)
    correct = (np.asarray(y_true) == np.asarray(y_pred)).astype(float)
    m = len(correct)
    accs = np.array([correct[rng.integers(0, m, m)].mean() for _ in range(n)])
    return float(np.quantile(accs, alpha / 2)), float(np.quantile(accs, 1 - alpha / 2))


def mcnemar(y_true, pred_a, pred_b):
    """Paired McNemar test on correctness vectors (same test instances).
    b = A right & B wrong, c = A wrong & B right."""
    a_ok = np.asarray(pred_a) == np.asarray(y_true)
    b_ok = np.asarray(pred_b) == np.asarray(y_true)
    b = int(np.sum(a_ok & ~b_ok))
    c = int(np.sum(~a_ok & b_ok))
    stat = (abs(b - c) - 1) ** 2 / (b + c) if b + c else 0.0
    return {"a_right_b_wrong": b, "a_wrong_b_right": c, "chi2": float(stat),
            "p_chi2": float(chi2.sf(stat, 1)),
            "p_exact": float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0}


def precision_at_recall(y_true, y_prob, prevalences, recall=0.90):
    """Precision at a fixed recall re-weighted to a target prevalence pi via the
    likelihood ratio: P = pi*TPR / (pi*TPR + (1-pi)*FPR)."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    i = int(np.argmax(tpr >= recall))
    t, f = float(tpr[i]), float(fpr[i])
    out = {}
    for pi in prevalences:
        out[str(pi)] = 1.0 if f == 0 else float(pi * t / (pi * t + (1 - pi) * f))
    return {"tpr": t, "fpr": f, "precision": out}
