"""
Tests for data_pipeline/nasrabadi.py (added for the pre-registered H1a/H1b).

Synthetic .mat files only -- no part of the real dataset is read -- so these run
in CI. What they pin down:
  * group comes from the top-level folder; the subject ID is the file stem;
  * the 10-10 names T7/T8/P7/P8 become T3/T4/T5/T6 and every channel gets a position;
  * int16 and float64 files load to the same volts (D2.3);
  * the threshold stage's epochs and rejection rule are epoch_signal's own: the
    number of epochs epoch_signal keeps at a threshold equals the number the
    threshold search counts as kept;
  * solve_threshold returns the smallest threshold meeting the target.

Run:  py -m pytest tests/test_nasrabadi.py
"""

import numpy as np
import pytest
import scipy.io as sio

from data_pipeline import nasrabadi as nas
from data_pipeline.preprocessing import CHANNELS_19, epoch_signal

N = 128 * 30  # 30 s at 128 Hz


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(str(path), {path.stem: data})


@pytest.fixture
def fake_root(tmp_path):
    rng = np.random.default_rng(0)
    signal = rng.normal(0, 40, (N, 19))
    _write(tmp_path / "ADHD_part1" / "v1p.mat", np.round(signal).astype(np.int16))
    _write(tmp_path / "Control_part2" / "Control_part2" / "v2.mat", np.round(signal).astype(float))
    return tmp_path


def test_discover_takes_group_from_the_top_folder(fake_root):
    df = nas.discover(str(fake_root)).set_index("subject_id")
    assert df.loc["v1p", "group"] == "ADHD"
    assert df.loc["v2", "group"] == "Control"


def test_load_mat_renames_scales_and_positions(fake_root):
    df = nas.discover(str(fake_root)).set_index("subject_id")
    a = nas.load_mat(df.loc["v1p", "path"])
    b = nas.load_mat(df.loc["v2", "path"])
    assert sorted(a.ch_names) == sorted(CHANNELS_19)
    assert not {"T7", "T8", "P7", "P8"} & set(a.ch_names)
    assert a.info["sfreq"] == 128.0
    assert np.allclose(a.get_data(), b.get_data())            # int16 == float64 scale
    assert 10e-6 < np.abs(a.get_data()).std() < 100e-6       # nominal uV -> volts
    pos = np.array([ch["loc"][:3] for ch in a.info["chs"]])
    assert np.isfinite(pos).all() and (np.abs(pos).sum(axis=1) > 0).all()


def test_threshold_search_counts_what_epoch_signal_keeps(fake_root):
    raw = nas.load_mat(nas.discover(str(fake_root))["path"].iloc[0])
    raw._data[:, 128 * 3: 128 * 3 + 20] += 400e-6             # a few artifact epochs
    worst, flat = nas.epoch_peak_to_peak(raw)
    for t in (150e-6, 300e-6, 600e-6):
        _, diag = epoch_signal(raw, reject_uv=t, return_diagnostics=True)
        assert diag["n_kept"] == int(np.sum((worst <= t) & ~flat))


def test_solve_threshold_is_the_smallest_meeting_the_target():
    rng = np.random.default_rng(1)
    subjects = [(rng.uniform(0, 100, 200), np.zeros(200, bool)) for _ in range(9)]
    t, rate = nas.solve_threshold(subjects, target=0.20)
    assert rate <= 0.20
    below = np.unique(np.concatenate([w for w, _ in subjects]))
    below = below[below < t]
    assert nas.median_rejection(subjects, below[-1]) > 0.20
