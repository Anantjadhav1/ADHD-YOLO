"""
Tests for training/positive_control_cnn.py's data assembly (added 2026-10-16).

No training happens here: the image files are empty placeholders, because the
assembly only reads names and folders. What these pin down:
  * Images are re-filed by CONDITION (task -> "ADHD", rest -> "Control"), as
    two units per child, "<id>-TASK" and "<id>-REST", in the child's own split
    -- so folds stay subject-wise and run_cv's one-row-per-subject OOF assert
    is satisfied by a legitimate design rather than switched off.
  * Rest = EO excludes EC images; rest = EC+EO includes both.
  * Children outside the cohort, and children missing one side, are dropped.
  * The unit manifest passes subject_split.verify_no_leakage, which run_cv
    calls on it.
  * A child whose image sits in a different split from the manifest is an
    error, not something to file silently.

Run:  py -m pytest tests/test_positive_control_cnn.py
"""

from pathlib import Path

import pandas as pd
import pytest

from data_pipeline.subject_split import verify_no_leakage
from training.positive_control import REST_LABEL, TASK_LABEL
from training.positive_control_cnn import (
    assemble_condition_tree, parse_image_name, to_child_level,
)

REP = "scalogram"


def _touch(root, split, diag, name):
    p = Path(root) / REP / split / diag / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")


def _source_tree(root):
    # A: full set in fold_0. B: full set in fold_1. C: no task recording.
    # D: has images but is outside the cohort.
    for i in range(2):
        for cond in ("EC", "EO", "VCPT"):
            _touch(root, "fold_0", "ADHD", f"A0000001_{cond}_{i:04d}.png")
            _touch(root, "fold_1", "Control", f"B0000002_{cond}_{i:04d}.png")
            _touch(root, "fold_1", "Control", f"D0000004_{cond}_{i:04d}.png")
        for cond in ("EC", "EO"):
            _touch(root, "fold_0", "Control", f"C0000003_{cond}_{i:04d}.png")


COHORT = {"A0000001": "fold_0", "B0000002": "fold_1", "C0000003": "fold_0"}


def test_parse_image_name():
    assert parse_image_name("F08080102_EO_0007.png") == ("F08080102", "EO")


def test_refiled_by_condition_with_two_units_per_child(tmp_path):
    _source_tree(tmp_path / "src")
    units = assemble_condition_tree(str(tmp_path / "src"), REP, COHORT, ["EO"], str(tmp_path / "out"))

    assert set(units["subject_id"]) == {"A0000001-TASK", "A0000001-REST",
                                        "B0000002-TASK", "B0000002-REST"}  # C: no task; D: not in cohort
    assert dict(zip(units["subject_id"], units["group"]))["A0000001-TASK"] == TASK_LABEL
    assert dict(zip(units["subject_id"], units["group"]))["A0000001-REST"] == REST_LABEL
    assert (units.loc[units["subject_id"].str.startswith("B"), "split"] == "fold_1").all()

    out = tmp_path / "out" / REP
    assert sorted(p.name for p in (out / "fold_0" / TASK_LABEL).iterdir()) == [
        "A0000001-TASK_VCPT_0000.png", "A0000001-TASK_VCPT_0001.png"]
    rest_files = sorted(p.name for p in (out / "fold_0" / REST_LABEL).iterdir())
    assert rest_files == ["A0000001-REST_EO_0000.png", "A0000001-REST_EO_0001.png"]  # no EC

    verify_no_leakage(units[["subject_id", "group", "split"]])


def test_ec_plus_eo_rest_includes_both(tmp_path):
    _source_tree(tmp_path / "src")
    units = assemble_condition_tree(str(tmp_path / "src"), REP, COHORT, ["EC", "EO"], str(tmp_path / "out"))
    assert dict(zip(units["subject_id"], units["n_images"]))["A0000001-REST"] == 4


def test_image_in_the_wrong_split_is_an_error(tmp_path):
    _source_tree(tmp_path / "src")
    with pytest.raises(AssertionError, match="split"):
        assemble_condition_tree(str(tmp_path / "src"), REP, {"A0000001": "fold_3"},
                                ["EO"], str(tmp_path / "out"))


def test_to_child_level_strips_the_unit_suffix():
    oof = pd.DataFrame({"subject_id": ["A0000001-TASK", "A0000001-REST"],
                        "true_label": [TASK_LABEL, REST_LABEL], "pred_prob_adhd": [0.7, 0.4]})
    assert to_child_level(oof)["subject_id"].tolist() == ["A0000001", "A0000001"]
