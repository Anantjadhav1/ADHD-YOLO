"""
Tests for training/linear_sensitivity.py (post-hoc, added 2026-10-16).

The anchor: run with the published settings, the harness must reproduce
runs/fusion/fusion_cv_results.csv fold for fold. If it does not, nothing the
other variants report can be attributed to the one change they make.

Then the two findings the README states, pinned so they cannot drift silently:
standardising moves the pooled AUC from 0.537 to 0.626, and dropping the CNN
probability leaves it at 0.627 -- the gain comes from the model, not the fusion.

Reads only tracked files under runs/ and docs/, so it runs in CI.
Run:  py -m pytest tests/test_linear_sensitivity.py
"""

import numpy as np
import pandas as pd
import pytest

from training.linear_sensitivity import CNN_OOF, FEATURES, run_variant

MANIFEST = "data_pipeline/splits/subject_splits.csv"


@pytest.fixture(autouse=True)
def _placeholder_data_root(monkeypatch, tmp_path):
    """run_fusion_cv loads the manifest through load_manifest(), which resolves
    its relative EDF paths against ADHD_YOLO_DATA_ROOT and exits if it is unset.
    Fusion never opens those files -- it reads only `split` and `group` -- so any
    directory will do. Without this the tests passed only on a machine where the
    variable happened to be set, and CI was red from 2026-10-16 to 2026-10-21."""
    monkeypatch.setenv("ADHD_YOLO_DATA_ROOT", str(tmp_path))


def _run(standardise, use_cnn):
    return run_variant(MANIFEST, pd.read_csv(CNN_OOF), pd.read_csv(FEATURES), standardise, use_cnn)


def test_published_settings_reproduce_the_committed_fusion_folds_exactly():
    folds, oof, s = _run(standardise=False, use_cnn=True)
    published = pd.read_csv("runs/fusion/fusion_cv_results.csv")
    assert np.allclose(folds["auc"].to_numpy(), published["auc"].to_numpy(), atol=1e-12)
    assert s["n"] == 84 and round(s["auc"], 3) == 0.537


def test_standardising_and_dropping_the_cnn():
    _, _, with_cnn = _run(standardise=True, use_cnn=True)
    _, _, without = _run(standardise=True, use_cnn=False)
    assert round(with_cnn["auc"], 3) == 0.626
    assert round(without["auc"], 3) == 0.627
    assert abs(with_cnn["auc"] - without["auc"]) < 0.01   # the CNN adds nothing
