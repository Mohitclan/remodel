"""URL canonicalisation ladder (paper Section IV-C).

R1  raw        the URL exactly as shipped
R2  canonical  scheme, a leading ``www.`` and trailing separators removed;
               path and query retained
R3  host-only  the host alone (path, query, port and a leading www. dropped)

The registrable domain (eTLD+1) is used only to group URLs for the
domain-disjoint split.
"""
import re
from functools import lru_cache

import tldextract

# Offline extractor: uses the Public Suffix List snapshot bundled with
# tldextract, so results are reproducible and need no network access.
_EXTRACT = tldextract.TLDExtract(suffix_list_urls=())

RUNGS = ("R1", "R2", "R3")
RUNG_NAMES = {"R1": "Raw", "R2": "Canonical", "R3": "Host-only"}

_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")
_WWW = re.compile(r"^www\d*\.", re.IGNORECASE)
_TRAILING = re.compile(r"[/?#&.]+$")
_IPV4 = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


def strip_scheme(url: str) -> str:
    return _SCHEME.sub("", url.strip())


def host_of(url: str) -> str:
    """Host part of a URL with or without a scheme (no port, no userinfo)."""
    rest = strip_scheme(url)
    host = re.split(r"[/?#]", rest, maxsplit=1)[0]
    host = host.rsplit("@", 1)[-1]
    return host.split(":", 1)[0].lower()


@lru_cache(maxsize=None)
def registrable_domain(host: str) -> str:
    """eTLD+1 of a host via the Public Suffix List; falls back to the host
    itself for IP addresses and hosts without a recognised suffix."""
    if not host or _IPV4.match(host):
        return host
    ext = _EXTRACT(host)
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}".lower()
    return _WWW.sub("", host)


def canonicalise(url: str, rung: str) -> str:
    url = url.strip()
    if rung == "R1":
        return url
    if rung == "R2":
        s = _WWW.sub("", strip_scheme(url))
        return _TRAILING.sub("", s)
    if rung == "R3":
        # host only: path, query, port, userinfo and a leading www. dropped
        return _WWW.sub("", host_of(url))
    raise ValueError(f"unknown rung {rung!r}")


# The class-conditional template found in PhiUSIIL (paper Table II):
# https, a www. prefix and nothing after the host.
TEMPLATE = re.compile(r"^https://www\.[^/?#]+$")


def matches_template(url: str) -> bool:
    return bool(TEMPLATE.match(url.strip()))
