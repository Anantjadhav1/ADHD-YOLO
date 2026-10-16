"""
Tests for training/results_table.py, which regenerates the Phase 2 headline
table from the committed out-of-fold files (added 2026-10-16).

What these pin down:
  * The arms really do share one cohort: 84 subjects, 42 ADHD / 42 Control.
    The README table claimed this; the classical rows turned out to come from
    91 subjects (44 / 47), so the claim is now enforced rather than assumed.
  * The headline AUCs on that cohort.
  * results_table's procedure is classical_baseline's procedure: run on the
    551-feature model's own 91 subjects, it reproduces the numbers
    classical_baseline.py has always printed (0.538, [0.416, 0.660], 0.268)
    to three decimals. If either script's procedure drifts, this fails.

Reads only the OOF CSVs under runs/, which are tracked in git, so it runs in
CI without the dataset.
Run:  py -m pytest tests/test_results_table.py
"""

import pandas as pd

from training.classical_baseline import _auc
from training.results_table import ARMS, pooled_stats


def _oofs():
    return {name: pd.read_csv(path) for name, path in ARMS}


def _common():
    return set.intersection(*(set(d["subject_id"]) for d in _oofs().values()))


def _auc_of(d):
    y = (d["true_label"] == "ADHD").to_numpy()
    p = d["pred_prob_adhd"].to_numpy()
    return _auc(p[y], p[~y])


def test_every_arm_is_one_row_per_subject():
    for name, d in _oofs().items():
        assert not d["subject_id"].duplicated().any(), name


def test_the_arms_share_an_84_subject_balanced_cohort():
    common = _common()
    assert len(common) == 84
    for d in _oofs().values():
        on_common = d[d["subject_id"].isin(common)]
        assert int((on_common["true_label"] == "ADHD").sum()) == 42


def test_headline_aucs_on_the_common_cohort():
    common = _common()
    got = {name: round(_auc_of(d[d["subject_id"].isin(common)]), 3) for name, d in _oofs().items()}
    assert got == {
        "CNN scalogram, frozen backbone": 0.522,
        "CNN topomap, frozen backbone": 0.523,
        "Classical SVM, 301 power features": 0.522,
        "Classical SVM, 551 features with coherence": 0.556,
    }


def test_classical_rows_previously_reported_were_on_91_subjects():
    # The published 0.517 (301) and 0.538 (551) are the all-91 values, which is
    # why the table's classical CIs could not be paired with 84-subject AUCs.
    oofs = _oofs()
    assert len(oofs["Classical SVM, 301 power features"]) == 91
    assert round(_auc_of(oofs["Classical SVM, 301 power features"]), 3) == 0.517
    assert round(_auc_of(oofs["Classical SVM, 551 features with coherence"]), 3) == 0.538


def test_procedure_reproduces_classical_baselines_own_output():
    s = pooled_stats(_oofs()["Classical SVM, 551 features with coherence"], seed=42)
    assert (s["n"], s["n_adhd"]) == (91, 44)
    assert round(s["auc"], 3) == 0.538
    assert (round(s["ci_low"], 3), round(s["ci_high"], 3)) == (0.416, 0.660)
    assert round(s["perm_p"], 3) == 0.268
