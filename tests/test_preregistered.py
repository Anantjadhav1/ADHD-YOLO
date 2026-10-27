"""
Tests for training/preregistered.py on SYNTHETIC tables -- no real data -- so
the analysis code is checked before it is ever run on the registered data.

Pins: H1a scores every subject exactly once; H1b standardises each dataset with
its own statistics, so a pure gain/offset difference between datasets does not
change its predictions; H2 trains on development subjects only and scores only
the test split; the decision rule needs BOTH p < 0.05/3 and CI lower bound > 0.5.

Run:  py -m pytest tests/test_preregistered.py
"""

import numpy as np
import pandas as pd

from training import preregistered as pr

COLS = [f"f{i}_vcpt" for i in range(6)]


def _table(n, rng, signal=0.0, gain=1.0, offset=0.0):
    y = np.array([i % 2 for i in range(n)])
    X = rng.normal(0, 1, (n, len(COLS))) + signal * y[:, None]
    df = pd.DataFrame(X * gain + offset, columns=COLS)
    df.insert(0, "group", np.where(y == 1, "ADHD", "Control"))
    df.insert(0, "subject_id", [f"S{i}" for i in range(n)])
    return df


def test_h1a_scores_every_subject_once():
    nas = _table(40, np.random.default_rng(0))
    oof = pr.h1a(nas)
    assert len(oof) == 40 and oof["subject_id"].is_unique and oof["pred_prob_adhd"].notna().all()


def test_h1b_is_invariant_to_a_gain_and_offset_between_datasets():
    rng = np.random.default_rng(1)
    roh = _table(60, rng, signal=1.0)
    roh["split"] = ["fold_%d" % (i % 5) for i in range(60)]
    nas = _table(40, np.random.default_rng(2), signal=1.0)
    scaled = nas.copy()
    scaled[COLS] = scaled[COLS] * 7.0 + 3.0          # a different unit and baseline
    a, b = pr.h1b(roh, nas), pr.h1b(roh, scaled)
    assert np.allclose(a["pred_prob_adhd"], b["pred_prob_adhd"])


def test_h2_trains_on_development_and_scores_only_the_test_split():
    rng = np.random.default_rng(3)
    roh = _table(50, rng)
    roh["split"] = ["test" if i < 10 else "fold_%d" % (i % 5) for i in range(50)]
    roh["all_missing"] = np.nan                      # dropped, like P300 columns
    oof = pr.h2(roh)
    assert set(oof["subject_id"]) == {f"S{i}" for i in range(10)}
    assert "all_missing" not in pr.engineered_columns(roh)


def test_decision_needs_both_conditions():
    assert pr.decide({"perm_p": 0.001, "ci_low": 0.55}) == "POSITIVE"
    assert pr.decide({"perm_p": 0.02, "ci_low": 0.55}) == "negative"    # p above 0.0167
    assert pr.decide({"perm_p": 0.001, "ci_low": 0.49}) == "negative"   # CI touches 0.5
