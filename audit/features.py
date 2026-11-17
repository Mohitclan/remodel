"""24 lexical features recomputed from the URL string (paper Section IV-C).

The shipped dataset columns are never used; every feature here is a length,
count, ratio or entropy over the (canonicalised) string and its host, so the
representation is the only factor that differs between rungs.
"""
import math
import re
from collections import Counter

import numpy as np
import pandas as pd

from .canon import host_of

FEATURE_NAMES = [
    "url_len", "host_len", "path_len",
    "n_dots", "n_hyphens", "n_underscores", "n_slashes", "n_qmarks",
    "n_equals", "n_ampersands", "n_at", "n_digits", "n_letters", "n_special",
    "digit_ratio", "letter_ratio", "special_ratio",
    "url_entropy", "host_entropy",
    "n_host_tokens", "n_path_tokens", "max_token_len",
    "host_is_ip", "host_digit_ratio",
]

_IP = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
_TOKEN_SPLIT = re.compile(r"[/?#&=._\-:]+")


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in Counter(s).values())


def url_features(url: str) -> list:
    host = host_of(url)
    after_host = url.split(host, 1)[1] if host and host in url else ""
    n = max(len(url), 1)
    digits = sum(ch.isdigit() for ch in url)
    letters = sum(ch.isalpha() for ch in url)
    special = len(url) - digits - letters
    tokens = [t for t in _TOKEN_SPLIT.split(url) if t]
    path_tokens = [t for t in re.split(r"[/?#&=]+", after_host) if t]
    return [
        len(url), len(host), len(after_host),
        url.count("."), url.count("-"), url.count("_"), url.count("/"),
        url.count("?"), url.count("="), url.count("&"), url.count("@"),
        digits, letters, special,
        digits / n, letters / n, special / n,
        _entropy(url), _entropy(host),
        len([t for t in host.split(".") if t]), len(path_tokens),
        max((len(t) for t in tokens), default=0),
        int(bool(_IP.match(host))),
        sum(ch.isdigit() for ch in host) / max(len(host), 1),
    ]


def featurise(urls) -> pd.DataFrame:
    return pd.DataFrame(np.array([url_features(u) for u in urls], dtype=float),
                        columns=FEATURE_NAMES)
