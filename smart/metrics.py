"""Age-state definitions used in the manuscript; all age quantities are in years."""
from __future__ import annotations

import numpy as np

OVERALL_DOMAIN = "overall sleep biological age"


def _finite(x, name):
    a = np.asarray(x, dtype=float)
    if not np.isfinite(a).all():
        raise ValueError(f"{name} must contain finite values; resolve missing observations explicitly.")
    return a


def age_gap(predicted_age, chronological_age):
    """Predicted age minus chronological age, preserving sign.

    For an (n, m) age matrix, pass chronological ages as shape (n, 1).
    """
    return _finite(predicted_age, "predicted_age") - _finite(chronological_age, "chronological_age")


def organ_asynchrony(organ_gaps):
    """Sample SD across the final axis (M_org - 1), excluding overall sleep age.

    The caller supplies organ-specific columns only. At least two are required.
    """
    g = _finite(organ_gaps, "organ_gaps")
    if g.ndim < 1 or g.shape[-1] < 2:
        raise ValueError("Asynchrony requires at least two organ-specific domains.")
    return np.std(g, axis=-1, ddof=1)


def summarize_predictions(predictions, chronological_age, domains, overall_domain=OVERALL_DOMAIN):
    """Summarize a complete (participant, decoding, domain) tensor.

    Mean ages are computed before gaps and asynchrony. Instability is the mean
    within-organ decoding SD, not the SD of the pooled age vector. MCSE equals
    decoding SD / sqrt(K). Single-decoding SD, MCSE and instability are undefined.
    Missing values are rejected rather than silently using different denominators.
    """
    a = _finite(predictions, "predictions")
    c = _finite(chronological_age, "chronological_age")
    domains = list(domains)
    if a.ndim != 3 or min(a.shape) < 1:
        raise ValueError("predictions must have nonempty shape (participants, decodings, domains).")
    n, k, m = a.shape
    if c.shape != (n,) or len(domains) != m or len(set(domains)) != m:
        raise ValueError("Chronological ages and unique domain labels must match the tensor.")
    if (a < 0).any() or (c < 0).any():
        raise ValueError("Predicted and chronological ages must be nonnegative.")
    organ_idx = [j for j, name in enumerate(domains) if name != overall_domain]
    if len(organ_idx) < 2:
        raise ValueError("At least two organ-specific outputs are required for this summary.")
    means = a.mean(axis=1)
    gaps = age_gap(means, c[:, None])
    sd = a.std(axis=1, ddof=1) if k > 1 else np.full((n, m), np.nan)
    return {
        "mean_age": means,
        "age_gap": gaps,
        "decoding_sd": sd,
        "mcse": sd / np.sqrt(k),
        "asynchrony": organ_asynchrony(gaps[:, organ_idx]),
        "instability": sd[:, organ_idx].mean(axis=1),
        "n_decodings": k,
    }


def concordance_correlation(x, y):
    """Lin's concordance correlation coefficient using population moments.

    Equal constant vectors have undefined CCC and return NaN.
    """
    x, y = _finite(x, "x"), _finite(y, "y")
    if x.ndim != 1 or y.shape != x.shape or x.size < 2:
        raise ValueError("CCC requires matching vectors with at least two observations.")
    denominator = x.var() + y.var() + (x.mean() - y.mean()) ** 2
    return float(2 * np.mean((x - x.mean()) * (y - y.mean())) / denominator) if denominator else np.nan
