"""
Regression tests for artifact rejection in preprocessing.epoch_signal().

Covers the bug found on 2026-08-24 when build_dataset.py was first pointed at
the real cohort: the LABEL channel is a digital marker that is constant by
construction, MNE types it as `eeg`, and `flat` drops an epoch if ANY eeg
channel falls below threshold. With no `picks` restriction that rejected 100%
of epochs on 100% of subjects, so the pipeline could not produce a single
image. The warning blamed the peak-to-peak threshold, which sent the first
investigation in the wrong direction -- measurement showed 250 uV was keeping
57-92% of epochs, not none.

Built on a synthetic Raw so these run without the dataset present. Run either
way:
    py -m pytest tests/test_epoch_rejection.py
    py tests/test_epoch_rejection.py
"""

import warnings

import mne
import numpy as np

from data_pipeline.preprocessing import CHANNELS_19, LABEL_CHANNEL, epoch_signal

mne.set_log_level("ERROR")

SFREQ = 500.0
DURATION_SEC = 30.0
CLEAN_AMPLITUDE_V = 20e-6  # ~120 uV peak-to-peak, comfortably under the 250 uV threshold


def _synthetic_raw(with_label: bool = True) -> mne.io.RawArray:
    """19 clean EEG channels, plus the constant LABEL channel the real files carry."""
    rng = np.random.default_rng(0)
    n_samples = int(SFREQ * DURATION_SEC)
    data = rng.normal(0, CLEAN_AMPLITUDE_V, (len(CHANNELS_19), n_samples))
    names = list(CHANNELS_19)

    if with_label:
        data = np.vstack([data, np.zeros((1, n_samples))])
        names.append(LABEL_CHANNEL)

    info = mne.create_info(names, SFREQ, ch_types="eeg")
    return mne.io.RawArray(data, info)


def test_flat_label_channel_does_not_reject_every_epoch():
    """The bug itself: a permanently-flat marker channel rejected everything."""
    epochs = epoch_signal(_synthetic_raw())
    assert len(epochs) > 0, "flat LABEL channel rejected every epoch"


def test_label_excluded_from_epochs_but_kept_on_raw():
    """
    LABEL must not reach the epochs, but must stay on the Raw --
    extract_vcpt_behavioral_proxy() reads it from there.
    """
    raw = _synthetic_raw()
    epochs = epoch_signal(raw)

    assert LABEL_CHANNEL not in epochs.ch_names
    assert LABEL_CHANNEL in raw.ch_names
    assert epochs.ch_names == [ch for ch in CHANNELS_19 if ch in raw.ch_names]


def test_clean_data_is_not_rejected():
    _, diagnostics = epoch_signal(_synthetic_raw(), return_diagnostics=True)
    assert diagnostics["rejection_rate"] < 0.05


def test_real_artifact_is_still_rejected():
    """
    The fix must not amount to switching rejection off. A channel driven far
    past the threshold still has to be caught.
    """
    # NOTE: MNE emits RuntimeWarning("All epochs were dropped!") when this
    # passes, so do NOT run the suite with -W error -- it turns a correct
    # result into a failure.
    #
    # The stronger assertion is that CLEAN epochs survive alongside the dirty
    # ones. Rejecting everything also satisfies "artifact was rejected", and
    # that is not hypothetical: on 2026-08-25 the LABEL channel caused exactly
    # that on every subject in the cohort, and it looked like working rejection.
    raw = _synthetic_raw()
    data = raw.get_data()
    data[0, :] *= 100  # ~12,000 uV peak-to-peak on one channel
    loud = mne.io.RawArray(data, raw.info)

    _, diagnostics = epoch_signal(loud, return_diagnostics=True)
    assert diagnostics["rejection_rate"] > 0.9


def test_rejection_can_still_be_disabled():
    """reject_uv=None restores pre-rejection behaviour, per the docstring."""
    raw = _synthetic_raw()
    n_events = len(mne.make_fixed_length_events(raw, duration=1.5))

    assert len(epoch_signal(raw, reject_uv=None)) == n_events


def test_warning_names_the_responsible_channel():
    """
    The original message blamed the peak-to-peak threshold unconditionally,
    even when `flat` did the rejecting. It must name the real culprit.
    """
    raw = _synthetic_raw(with_label=False)
    data = raw.get_data()
    data[:, :] *= 1000  # drive every channel past the threshold
    loud = mne.io.RawArray(data, raw.info)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        epoch_signal(loud)

    messages = [str(w.message) for w in caught]
    total_rejection = [m for m in messages if "ALL" in m and "rejected" in m]
    assert total_rejection, f"expected a total-rejection warning, got {messages}"
    assert "Channels responsible" in total_rejection[0]


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {name}: {exc}")

    print(f"\n{'ALL PASSED' if failures == 0 else f'{failures} FAILURES'}")
    raise SystemExit(1 if failures else 0)
