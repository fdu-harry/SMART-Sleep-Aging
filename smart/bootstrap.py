"""Paired participant bootstrap for the three decoding summaries.

All predictions for a participant travel together in each resample. The middle
strategy averages separately computed single-decoding performances. It is not
the performance of averaged predictions. Scores are held fixed: this procedure
does not refit or cross-validate any predictive model.
"""
from dataclasses import dataclass
import numpy as np

STRATEGIES = ("Greedy decoding", "Single stochastic decoding (mean performance)",
              "Mean stochastic prediction")


@dataclass
class BootstrapResult:
    estimates: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    samples: np.ndarray
    n_participants: int
    rejected_draws: int
    seed: int

    def rows(self):
        return [dict(strategy=name, estimate=float(self.estimates[j]),
                     lower=float(self.lower[j]), upper=float(self.upper[j]),
                     n=self.n_participants, bootstrap_resamples=len(self.samples),
                     rejected_draws=self.rejected_draws, seed=self.seed)
                for j, name in enumerate(STRATEGIES)]


def paired_decoding_bootstrap(outcome, greedy, stochastic, *, participant_ids,
                              metric="auc", event=None, ensemble_score=None,
                              n_resamples=2000, seed=42):
    """Compute original-sample statistics and paired percentile 95% CIs.

    outcome: binary labels for AUC or nonnegative follow-up times for C-index.
    greedy: length-n score vector; stochastic: n x K scores (K >= 2).
    Higher scores must mean greater event risk for both metrics.
    ensemble_score: optional independently prepared mean-age-model risk score.
        By default use the mean of stochastic scores, appropriate for raw age
        gaps/AUC. For fitted Cox risk scores, provide the score from the model
        fitted using mean ages; averaging the run-specific risks is not equivalent.

    Participants must have one aligned, complete row each. Resolve inclusion,
    timing, covariates and model fitting before calling this metric function.
    A resample with no AUC classes or no comparable survival pairs is rejected
    jointly for all three strategies. The number of rejections is returned.
    """
    y = np.asarray(outcome, dtype=float)
    g = np.asarray(greedy, dtype=float)
    s = np.asarray(stochastic, dtype=float)
    ids = np.asarray(participant_ids, dtype=str)
    if y.ndim != 1 or len(y) < 2 or g.shape != y.shape or s.ndim != 2 or s.shape[0] != len(y) or s.shape[1] < 2:
        raise ValueError("Use matching participant vectors and at least two stochastic decodings.")
    if ids.shape != y.shape or len(np.unique(ids)) != len(y) or np.any(np.char.strip(ids) == ""):
        raise ValueError("Participant identifiers must be unique, nonempty and aligned.")
    ens = s.mean(axis=1) if ensemble_score is None else np.asarray(ensemble_score, dtype=float)
    if ens.shape != y.shape or not all(np.isfinite(a).all() for a in (y, g, s, ens)):
        raise ValueError("Use a common complete sample across all strategies; no NaN/inf allowed.")
    if not isinstance(n_resamples, int) or n_resamples < 2:
        raise ValueError("n_resamples must be an integer of at least two.")
    if metric == "auc":
        from sklearn.metrics import roc_auc_score
        if set(np.unique(y)) != {0.0, 1.0}:
            raise ValueError("AUC requires both binary classes, encoded 0 and 1.")
        def evaluate(idx, scores):
            if np.unique(y[idx]).size != 2:
                return np.nan
            return roc_auc_score(y[idx], scores[idx])
    elif metric == "cindex":
        from lifelines.utils import concordance_index
        e = np.asarray(event, dtype=float)
        if e.shape != y.shape or not np.isfinite(e).all() or not set(np.unique(e)).issubset({0.0, 1.0}) or (y < 0).any():
            raise ValueError("C-index requires nonnegative times and matching binary event indicators.")
        def evaluate(idx, scores):
            try:
                return concordance_index(y[idx], -scores[idx], e[idx])
            except ZeroDivisionError:
                return np.nan
    else:
        raise ValueError("metric must be auc or cindex.")

    def statistics(idx):
        values = [evaluate(idx, g)] + [evaluate(idx, s[:, j]) for j in range(s.shape[1])] + [evaluate(idx, ens)]
        return np.array([values[0], np.mean(values[1:-1]), values[-1]])

    point = statistics(np.arange(len(y)))
    if not np.isfinite(point).all():
        raise ValueError("The original sample has no estimable metric.")
    rng = np.random.default_rng(seed)
    samples, rejected, attempts = [], 0, 0
    while len(samples) < n_resamples:
        attempts += 1
        if attempts > 10 * n_resamples:
            raise ValueError("Too few valid paired resamples; inspect sample size and event counts.")
        value = statistics(rng.integers(0, len(y), size=len(y)))
        if np.isfinite(value).all():
            samples.append(value)
        else:
            rejected += 1
    samples = np.array(samples)
    low, high = np.quantile(samples, [0.025, 0.975], axis=0)
    return BootstrapResult(point, low, high, samples, len(y), rejected, seed)
