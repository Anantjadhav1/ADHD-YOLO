"""
Tests for root-relative manifest paths.

The manifest is meant to be the single source of truth for which child is in
which fold, and subject_split.py's design says it must never be regenerated --
regenerating reshuffles folds and invalidates any result computed against the
old one. That only works if the manifest can actually be committed, which it
could not while every row embedded one machine's drive letter.

So paths are stored relative to $ADHD_YOLO_DATA_ROOT and resolved on load.
These cover the round trip, the back-compatibility path for manifests that
still hold absolute paths, and the NaN vcpt_path case.

    py -m pytest tests/test_manifest_paths.py
    py tests/test_manifest_paths.py
"""

import os

import pandas as pd

from data_pipeline.subject_split import (
    DATA_ROOT_ENV,
    load_manifest,
    to_absolute,
    to_relative,
)


def _write_manifest(tmpdir, eoec, vcpt=None):
    path = os.path.join(tmpdir, "manifest.csv")
    pd.DataFrame([{
        "subject_id": "C09110104", "group": "Control", "split": "fold_2",
        "eoec_path": eoec, "vcpt_path": vcpt,
    }]).to_csv(path, index=False)
    return path


def test_to_relative_strips_the_root():
    rel = to_relative(os.path.join("D:", "data", "edf (all)", "x-EOEC.edf"), os.path.join("D:", "data"))
    assert rel == "edf (all)/x-EOEC.edf"


def test_to_relative_uses_posix_separators():
    """A manifest written on Windows must still be readable on Linux."""
    rel = to_relative(os.path.join("D:", "data", "sub", "x.edf"), os.path.join("D:", "data"))
    assert "\\" not in rel


def test_round_trip_returns_the_original_path():
    root = os.path.join("D:", "data")
    original = os.path.join(root, "edf (all)", "x-EOEC.edf")
    assert os.path.normcase(to_absolute(to_relative(original, root), root)) == os.path.normcase(original)


def test_load_manifest_resolves_relative_paths(tmp_path):
    root = str(tmp_path / "dataroot")
    os.makedirs(os.path.join(root, "edf (all)"), exist_ok=True)
    manifest_path = _write_manifest(str(tmp_path), "edf (all)/x-EOEC.edf")

    manifest = load_manifest(manifest_path, data_root=root)

    assert os.path.isabs(manifest.loc[0, "eoec_path"])
    assert manifest.loc[0, "eoec_path"].endswith("x-EOEC.edf")


def test_absolute_paths_still_work(tmp_path):
    """Manifests generated before this change must keep loading unchanged."""
    absolute = os.path.abspath(os.path.join(str(tmp_path), "x-EOEC.edf"))
    manifest_path = _write_manifest(str(tmp_path), absolute)

    manifest = load_manifest(manifest_path, data_root=str(tmp_path / "unrelated"))

    assert manifest.loc[0, "eoec_path"] == absolute


def test_missing_vcpt_is_not_turned_into_a_path(tmp_path):
    """
    NaN is truthy as a float -- the trap already recorded as solved problem #11,
    where a missing vcpt_path became a fake path. It must survive resolution as
    a null, not become '<root>/nan'.
    """
    manifest_path = _write_manifest(str(tmp_path), "edf/x-EOEC.edf", vcpt=None)

    manifest = load_manifest(manifest_path, data_root=str(tmp_path))

    assert pd.isna(manifest.loc[0, "vcpt_path"])


def test_env_var_is_used_when_no_root_passed(tmp_path, monkeypatch=None):
    manifest_path = _write_manifest(str(tmp_path), "edf/x-EOEC.edf")
    previous = os.environ.get(DATA_ROOT_ENV)
    os.environ[DATA_ROOT_ENV] = str(tmp_path)
    try:
        manifest = load_manifest(manifest_path)
        assert os.path.isabs(manifest.loc[0, "eoec_path"])
    finally:
        if previous is None:
            os.environ.pop(DATA_ROOT_ENV, None)
        else:
            os.environ[DATA_ROOT_ENV] = previous


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            if "tmp_path" in fn.__code__.co_varnames[: fn.__code__.co_argcount]:
                with tempfile.TemporaryDirectory() as td:
                    fn(Path(td))
            else:
                fn()
            print(f"PASS  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {name}: {exc}")

    print(f"\n{'ALL PASSED' if failures == 0 else f'{failures} FAILURES'}")
    raise SystemExit(1 if failures else 0)
