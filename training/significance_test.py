"""
Phase 2 — statistical significance vs. the 75.8%/84.5% baselines.

Per PROJECT.md sec 2 (academic bar) and sec 4 step 6: report accuracy WITH a
significance test, not accuracy alone. A McNemar's test needs the baseline's
PAIRED per-subject predictions on the same test set -- we don't have those,
only Rohani et al.'s reported aggregate accuracy. So the correct tool here is
a bootstrap confidence interval on OUR accuracy, checked against the fixed
baseline value -- not a McNemar's test run on data we don't actually have.
"""

import numpy as np
import pandas as pd


def bootstrap_accuracy_ci(subj_df: pd.DataFrame, n_bootstrap: int = 2000,
                           ci: float = 0.95, seed: int = 42) -> dict:
    """
    subj_df: subject-level results with 'true_label' and 'pred_label' columns
    (the same format compute_metrics() in train_yolo_cls.py consumes).

    Resamples SUBJECTS with replacement (not epochs -- consistent with every
    other subject-wise-CV rule in this project) n_bootstrap times, computing
    accuracy each time, to get an empirical confidence interval.
    """
    rng = np.random.default_rng(seed)
    n = len(subj_df)
    correct = (subj_df["true_label"].values == subj_df["pred_label"].values).astype(float)

    boot_accs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_accs[i] = correct[idx].mean()

    alpha = 1 - ci
    lower, upper = np.percentile(boot_accs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "observed_accuracy": float(correct.mean()),
        "ci_level": ci,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "n_subjects": n,
        "n_bootstrap": n_bootstrap,
    }


def compare_to_baseline(subj_df: pd.DataFrame, baseline_accuracy: float,
                         n_bootstrap: int = 2000, seed: int = 42) -> dict:
    """
    baseline_accuracy: e.g. 0.758 or 0.845 -- the fixed reference point, not
    a distribution (we don't have per-subject baseline predictions).

    'significantly_better': the baseline falls BELOW our bootstrap CI's lower
    bound -- i.e. even the unlucky 2.5% tail of our resampled accuracy still
    beats it. This is a real but conservative claim; report the CI itself in
    the paper, not just this boolean, since the boolean hides how close it was.
    """
    ci_result = bootstrap_accuracy_ci(subj_df, n_bootstrap=n_bootstrap, seed=seed)
    ci_result["baseline_accuracy"] = baseline_accuracy
    ci_result["significantly_better"] = ci_result["ci_lower"] > baseline_accuracy
    ci_result["significantly_worse"] = ci_result["ci_upper"] < baseline_accuracy
    return ci_result