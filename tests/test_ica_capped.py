"""
Regression test for the ICA `capped` diagnostic (bug found 2026-10-16).

remove_artifacts_ica() caps how many components muscle detection may remove,
then runs a topography veto that can strip most of what survived the cap.
`capped` used to be computed as len(ica.exclude) >= MAX_ICA_COMPONENTS_EXCLUDED,
i.e. AFTER the veto, so it could not say whether the cap had fired. On the real
cohort it was False for all 108 subjects -- including C12031144, where the cap
fired and the function itself warned "FLAG THIS SUBJECT FOR MANUAL QC".

The detectors and the veto are replaced with fakes that reproduce
C12031144's exact shape (muscle flags 8 -> capped -> veto keeps 1), so this
runs on a synthetic Raw in seconds, without the dataset.

Run:  py -m pytest tests/test_ica_capped.py
"""

import warnings

import mne
import numpy as np
import pytest

from data_pipeline import preprocessing as pp

mne.set_log_level("ERROR")


def _synthetic_raw(seconds=20.0, sfreq=500.0, seed=0):
    rng = np.random.default_rng(seed)
    info = mne.create_info(list(pp.CHANNELS_19), sfreq=sfreq, ch_types="eeg")
    data = rng.standard_normal((len(pp.CHANNELS_19), int(seconds * sfreq))) * 20e-6
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="ignore")
    return raw


def _install_fakes(monkeypatch, n_muscle, keep_after_veto):
    """Muscle detection flags `n_muscle` components; EOG flags none; the veto
    keeps only the first `keep_after_veto` of whatever survived the cap."""
    n = len(pp.CHANNELS_19)

    def fake_muscle(self, inst, *args, **kwargs):
        scores = np.zeros(n)
        scores[:n_muscle] = np.linspace(1.0, 0.5, n_muscle)  # strongest first
        return list(range(n_muscle)), scores

    def fake_eog(self, inst, *args, **kwargs):
        return [], np.zeros(n)

    def fake_veto(ica, fitted_ch, exclude, reasons):
        ranked = sorted(exclude)
        kept = set(ranked[:keep_after_veto])
        return kept, {i: "vetoed by test" for i in ranked[keep_after_veto:]}

    monkeypatch.setattr(mne.preprocessing.ICA, "find_bads_muscle", fake_muscle)
    monkeypatch.setattr(mne.preprocessing.ICA, "find_bads_eog", fake_eog)
    monkeypatch.setattr(pp, "_topography_veto", fake_veto)


def _run(raw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, diag = pp.remove_artifacts_ica(raw)
    return diag, [str(w.message) for w in caught]


def test_capped_is_true_when_the_cap_fires_even_if_the_veto_leaves_one(monkeypatch):
    # C12031144's shape: muscle flags 8, cap keeps 5, veto leaves 1.
    _install_fakes(monkeypatch, n_muscle=8, keep_after_veto=1)
    diag, msgs = _run(_synthetic_raw())

    assert any("capped to" in m and "MANUAL QC" in m for m in msgs), "the cap should have fired"
    assert diag["n_excluded"] == 1
    # The bug: this was False, because 1 < MAX_ICA_COMPONENTS_EXCLUDED.
    assert diag["capped"] is True
    assert diag["n_muscle_flagged"] == 8


def test_capped_is_false_when_muscle_detection_stays_under_the_cap(monkeypatch):
    _install_fakes(monkeypatch, n_muscle=2, keep_after_veto=2)
    diag, msgs = _run(_synthetic_raw())

    assert not any("capped to" in m for m in msgs)
    assert diag["capped"] is False
    assert diag["n_muscle_flagged"] == 2
    assert diag["n_excluded"] == 2


@pytest.mark.parametrize("keep", [0, 1, 5])
def test_capped_does_not_depend_on_what_the_veto_leaves(monkeypatch, keep):
    _install_fakes(monkeypatch, n_muscle=8, keep_after_veto=keep)
    diag, _ = _run(_synthetic_raw())
    assert diag["capped"] is True
    assert diag["n_excluded"] == keep
