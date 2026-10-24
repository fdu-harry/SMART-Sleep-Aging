"""Reference deviations and entropy from the SHHS and MrOS analyses."""
import numpy as np


def reference_deviation(values, *, kind, lower=None, upper=None, threshold=None, endpoint=None):
    """Scale departures by half the reference span; retain missing values.

    For an interval, lower and upper are its two boundaries. For a single
    threshold, endpoint is the fixed boundary on its normal side.
    """
    x = np.asarray(values, dtype=float)
    if np.isinf(x).any():
        raise ValueError("Measurements must be finite or missing.")
    if kind == "interval":
        lo, hi = lower, upper
    elif kind == "upper":
        lo, hi = endpoint, threshold
    elif kind == "lower":
        lo, hi = threshold, endpoint
    else:
        raise ValueError("kind must be interval, upper, or lower.")
    if lo is None or hi is None or not np.isfinite([lo, hi]).all() or hi <= lo:
        raise ValueError("Reference boundaries must be finite and ordered.")
    scale = (hi - lo) / 2
    if scale < 1e-10:
        delta = np.zeros_like(x)
    else:
        delta = np.where(x > hi, (x - hi) / scale,
                         np.where(x < lo, (x - lo) / scale, 0.0))
        if kind == "upper":
            delta = np.maximum(delta, 0)
        elif kind == "lower":
            delta = np.minimum(delta, 0)
    return np.where(np.isnan(x), np.nan, delta)


def fit_entropy_cutpoints(deviations):
    """Return the complete bin boundaries used by the source notebooks.

    Use nonzero-deviation tertiles when at least 80 values are present.
    Smaller sets use the original fixed boundaries, 0.5, 1.5 and 3.0.
    Tied quantiles are collapsed before rounding to six decimal places.
    """
    d = np.abs(np.asarray(deviations, dtype=float))
    if np.isinf(d).any():
        raise ValueError("Deviations must be finite or missing.")
    nonzero = d[np.isfinite(d) & (d > 1e-9)]
    if len(nonzero) < 80:
        return np.array([0.5, 1.5, 3.0])
    cuts = [1e-9]
    previous = 0.0
    for value in np.sort(np.percentile(nonzero, np.linspace(0, 100, 4)[1:-1])):
        if value > previous + 1e-9:
            cuts.append(round(float(value), 6))
            previous = value
    return np.asarray(cuts)


def homeostasis_entropy(deviations, cutpoints, min_indicators=5, normalize=False):
    """Shannon entropy across departure bins, with the cohort observation filter.

    Rows are participants and columns are indicators. Pass the complete
    boundaries from fit_entropy_cutpoints. Exact ties enter the higher bin.
    """
    d = np.abs(np.asarray(deviations, dtype=float))
    cuts = np.asarray(cutpoints, dtype=float)
    if d.ndim != 2 or np.isinf(d).any() or cuts.ndim != 1 or not np.isfinite(cuts).all():
        raise ValueError("Supply a deviation matrix and finite bin boundaries.")
    nonmonotonic = np.any(np.diff(cuts) < 0) and np.any(np.diff(cuts) > 0)
    if len(cuts) < 1 or nonmonotonic or min_indicators < 3:
        raise ValueError("Bin boundaries must be ordered and at least three indicators are required.")
    out = np.full(len(d), np.nan)
    n_bins = len(cuts) + 1
    for i, row in enumerate(d):
        row = row[np.isfinite(row)]
        if len(row) < min_indicators:
            continue
        bins = np.digitize(row, cuts)
        counts = np.bincount(bins, minlength=n_bins)
        proportions = counts[counts > 0] / len(row)
        out[i] = -np.sum(proportions * np.log2(proportions))
        if normalize:
            out[i] /= np.log2(n_bins)
    return out
