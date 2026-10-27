"""
Tests for training/h3_riemann.py on SYNTHETIC data -- checked before the
registered analysis is ever run. Skipped where pyriemann is not installed.

Pins: one OAS covariance per condition, symmetric positive definite even when
ICA has removed components (rank-deficient data); every subject scored exactly
once; the tangent-space reference is fitted on training folds only, so changing
one subject's matrix changes no other subject's prediction outside its fold;
a common scale factor on all data changes nothing (affine invariance).

Run:  py -m pytest tests/test_h3_riemann.py
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyriemann")

from training import h3_riemann as h3


class _FakeEpochs:
    def __init__(self, data):
        self._d = data

    def get_data(self):
        return self._d


def _spd(rng, n=19):
    a = rng.normal(size=(n, n))
    return a @ a.T + n * np.eye(n)


def _labels(n):
    return pd.DataFrame({"subject_id": [f"S{i}" for i in range(n)],
                         "group": ["ADHD" if i % 2 else "Control" for i in range(n)],
                         "split": [f"fold_{i % 5}" for i in range(n)]})


def test_covariance_is_spd_even_for_rank_deficient_data():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 20e-6, (30, 19, 192))
    x[:, -1, :] = x[:, 0, :]                      # one channel duplicated: rank 18
    c = h3.subject_covariance(_FakeEpochs(x))
    assert c.shape == (19, 19) and np.allclose(c, c.T)
    assert np.linalg.eigvalsh(c).min() > 0


def test_every_subject_scored_once_and_scale_invariant():
    rng = np.random.default_rng(1)
    lab = _labels(30)
    covs = {c: np.stack([_spd(rng) for _ in range(30)]) for c in h3.CONDITIONS}
    a = h3.cross_validate(covs, lab)
    b = h3.cross_validate({c: v * 1e6 for c, v in covs.items()}, lab)
    assert len(a) == 30 and a["subject_id"].is_unique and a["pred_prob_adhd"].notna().all()
    assert np.allclose(a["pred_prob_adhd"], b["pred_prob_adhd"], atol=1e-6)


def test_reference_mean_uses_training_folds_only():
    rng = np.random.default_rng(2)
    lab = _labels(30)
    covs = {c: np.stack([_spd(rng) for _ in range(30)]) for c in h3.CONDITIONS}
    base = h3.cross_validate(covs, lab)
    changed = {c: v.copy() for c, v in covs.items()}
    changed["EC"][0] = _spd(np.random.default_rng(99)) * 50   # S0 is in fold_0
    after = h3.cross_validate(changed, lab)
    same_fold = lab["split"] == "fold_0"
    # Other subjects in S0's own fold are scored by a model that never saw S0.
    assert np.allclose(base.loc[same_fold & (lab.index != 0), "pred_prob_adhd"],
                       after.loc[same_fold & (lab.index != 0), "pred_prob_adhd"])
