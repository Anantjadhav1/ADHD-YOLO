"""
Tests for training/positive_control.py (Rest vs Task, added 2026-10-16).

What these pin down:
  * The paired table keeps the split SUBJECT-WISE: every row of a child carries
    that child's split, so rest and task can never land on opposite sides of a
    fold boundary.
  * The held-out test split is excluded, and features measured in only one
    condition (coherence) are excluded -- otherwise the classifier could
    separate the classes by which columns are missing.
  * The statistics respect the pairing: the within-subject permutation p is
    small when task beats rest inside every subject, and NOT small when the
    labels carry no information.

Built on small synthetic tables, so these run in CI without the dataset.
Run:  py -m pytest tests/test_positive_control.py
"""

import numpy as np
import pandas as pd

from training.positive_control import (
    REST_LABEL, TASK_LABEL, build_paired_table, cluster_bootstrap_ci,
    paired_win_rate, within_subject_permutation_p, _arrays_by_subject, _pooled_auc,
)


def _features_frame():
    rows = []
    for i, split in enumerate(["fold_0", "fold_0", "fold_1", "fold_1", "test"]):
        rows.append({
            "subject_id": f"S{i}", "group": "ADHD" if i % 2 else "Control", "split": split,
            "relpow_Pz_alpha_eo": 0.30 + i / 100, "relpow_Pz_alpha_ec": 0.50,
            "relpow_Pz_alpha_vcpt": 0.20 + i / 100,
            "tbr_eo": 2.0, "tbr_ec": 2.5, "tbr_vcpt": 1.8,
            "coh_f_p_alpha_eo": 0.1, "coh_f_p_alpha_ec": 0.1,   # rest-only family
        })
    df = pd.DataFrame(rows)
    df.loc[df["subject_id"] == "S1", ["relpow_Pz_alpha_vcpt", "tbr_vcpt"]] = np.nan  # no VCPT
    return df


def test_paired_table_is_subject_wise_and_excludes_test_and_one_sided_features():
    table, feats = build_paired_table(_features_frame(), ["eo"])
    assert sorted(feats) == ["relpow_Pz_alpha", "tbr"]           # coherence dropped
    assert "S4" not in set(table["subject_id"])                   # test split dropped
    assert "S1" not in set(table["subject_id"])                   # no VCPT recording
    for _, g in table.groupby("subject_id"):
        assert g["split"].nunique() == 1                          # never split across folds
        assert sorted(g["group"]) == sorted([REST_LABEL, TASK_LABEL])
    # Feature values come from the right condition.
    s0 = table[table["subject_id"] == "S0"].set_index("condition")
    assert s0.loc["eo", "relpow_Pz_alpha"] == 0.30
    assert s0.loc["vcpt", "relpow_Pz_alpha"] == 0.20


def test_ec_plus_eo_gives_two_rest_rows_per_subject():
    table, _ = build_paired_table(_features_frame(), ["ec", "eo"])
    counts = table.groupby("subject_id")["group"].value_counts().unstack()
    assert (counts[REST_LABEL] == 2).all() and (counts[TASK_LABEL] == 1).all()


def _oof(task_probs, rest_probs):
    rows = []
    for i, (t, r) in enumerate(zip(task_probs, rest_probs)):
        rows.append({"subject_id": f"S{i}", "true_label": TASK_LABEL, "pred_prob_adhd": t})
        rows.append({"subject_id": f"S{i}", "true_label": REST_LABEL, "pred_prob_adhd": r})
    return pd.DataFrame(rows)


def test_permutation_p_is_small_when_task_wins_inside_every_subject():
    rng = np.random.default_rng(0)
    base = rng.uniform(0.2, 0.8, 30)          # large between-subject spread
    oof = _oof(base + 0.05, base)              # but task always a little higher
    observed = _pooled_auc(_arrays_by_subject(oof))
    p = within_subject_permutation_p(oof, observed, n=999, seed=1)
    assert p < 0.01
    assert paired_win_rate(oof) == (30, 30)


def test_permutation_p_is_not_small_when_labels_carry_nothing():
    rng = np.random.default_rng(0)
    oof = _oof(rng.uniform(0, 1, 30), rng.uniform(0, 1, 30))
    observed = _pooled_auc(_arrays_by_subject(oof))
    p = within_subject_permutation_p(oof, observed, n=999, seed=1)
    assert p > 0.05


def test_cluster_bootstrap_ci_contains_the_point_estimate():
    rng = np.random.default_rng(0)
    base = rng.uniform(0.2, 0.8, 40)
    oof = _oof(base + rng.normal(0.05, 0.05, 40), base)
    est = _pooled_auc(_arrays_by_subject(oof))
    lo, hi = cluster_bootstrap_ci(oof, n=500, seed=2)
    assert lo <= est <= hi and hi - lo > 0
