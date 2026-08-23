"""
Phase 2 — classical EEG biomarker features, for fusion with the CNN.

The classical/fusion route remains the most promising path past 75.8%/84.5%
-- the paper's own Table 2 hit 84.5% with feature selection + Logistic
Regression and no deep learning. But note they used 113 selected features,
not three.

TBR SPECIFICALLY DOES NOT WORK on this dataset. Tested on 20 subjects
(10/10), subject-level AUC was 0.43 (EC, reversed direction), 0.49 (EO) and
0.55 (VCPT), every CI containing 0.5. This replicates a well-established
negative finding -- Arns et al. (2013) meta-analysis, the five-algorithm null
on iSPOT-A/ICAN (2020), and the 2026 multiverse analysis, which identifies
individual alpha peak frequency and aperiodic activity as the confounds that
limit TBR. It is kept here as a reported negative result and for
comparability with the source paper.

Getting the classical half to contribute requires features TBR is not:
aperiodic exponent/offset (specparam), individual alpha peak frequency,
relative band power per channel, frontal alpha asymmetry, coherence
summaries. See PROGRESS.md 2026-08-23 and docs/STUDY_GUIDE.md Module 3.

SCOPE, matching current data reality (see PROJECT.md limitations): TBR only
needs resting-state power spectra, so it's fully computable right now. P300
latency/amplitude and per-condition behavioral features (omission/commission/
reaction time) are NOT included here -- those need stimulus-locked triggers
the VCPT LABEL channel doesn't reliably provide (see preprocessing.py
module docstring). compute_classical_features() returns NaN for anything
not yet computable rather than fabricating a value -- the fusion classifier
downstream must handle NaN inputs, not silently get a fake number instead.
"""

import numpy as np
from scipy.signal import welch

from data_pipeline.preprocessing import CHANNELS_19

# np.trapz was REMOVED in NumPy 2.0 in favour of np.trapezoid. This project runs
# on NumPy 2.4, so np.trapz would raise AttributeError.
_trapz = getattr(np, "trapezoid", None) or np.trapz

FRONTAL_CHANNELS = ["F3", "F4", "Fz"]  # matches the paper's TBR site exactly
THETA_BAND_HZ = (4, 8)
BETA_BAND_HZ = (12, 30)
# Welch segment length. Clamped to epoch length downstream, so on 1.5 s epochs
# (751 samples) this yields df = 0.67 Hz -> 6 theta bins. The previous value of
# 256 gave df = 1.95 Hz and only TWO theta bins, which is not enough points to
# estimate a band ratio from. Epoch length is the real ceiling here; see
# PROGRESS.md 2026-08-23.
TBR_NPERSEG = 1000

def _band_power(psd: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> float:
    """
    Power in a frequency band = the INTEGRAL of the PSD across it.

    The PSD is a density (uV^2/Hz), so taking .mean() across the band's bins
    gives average density, i.e. power divided by bandwidth. Since theta spans
    4 Hz and beta spans 18 Hz, those bandwidths do NOT cancel in a ratio --
    they inflate it by 18/4 = 4.5x. Measured inflation on 20 real subjects:
    4.88x. This is what put TBR at 9-16 instead of the published 1.5-3.5.

    Returns NaN rather than a number if the band holds fewer than 2 bins --
    a trapezoidal integral over one point is meaningless, and surfacing that
    is better than silently returning a value derived from a single sample.
    """
    mask = (freqs >= lo) & (freqs <= hi)
    if int(mask.sum()) < 2:
        return float("nan")
    return float(_trapz(psd[mask], freqs[mask]))

def compute_tbr(epochs, frontal_channels: list = FRONTAL_CHANNELS) -> float:
    """
        Theta/Beta Ratio at frontal channels:
    TBR = Power(theta 4-8Hz) / Power(beta 12-30Hz), where "power" is the
    INTEGRAL of the PSD across the band (see _band_power).

    One value per subject per condition: PSDs from every epoch and frontal
    channel are averaged into a single spectrum BEFORE the ratio is taken.

    CAVEAT: the spectrum is still estimated from 1.5 s epochs, which caps
    frequency resolution at 0.67 Hz no matter what TBR_NPERSEG is set to.
    Computing this on the continuous segment instead would give both finer
    resolution and more averaging. Tracked as a separate change.

    NOTE ON THE FEATURE ITSELF: TBR did not separate ADHD from Control on the
    20 subjects tested here (AUC 0.43-0.55, all CIs containing 0.5), matching
    Arns et al. (2013) and later nulls. It is retained as a reported negative
    result and for comparability with the source paper, NOT as a feature
    expected to carry the fusion classifier. See PROGRESS.md 2026-08-23.

    epochs: an mne.Epochs object for ONE condition (EC, EO, or VCPT) for one
    subject -- pass result["ec_epochs"] etc. from preprocess_subject().
    Returns a single float, or NaN if none of the frontal channels are present.
    """
    available = [ch for ch in frontal_channels if ch in epochs.ch_names]
    if not available:
        return float("nan")

    data = epochs.get_data(picks=available)  # (n_epochs, n_channels, n_samples)
    sfreq = epochs.info["sfreq"]
    # Average PSD across all epochs and the frontal channels for one stable
    # subject/condition-level estimate, not one value per epoch.
    flat = data.reshape(-1, data.shape[-1])
    freqs, psd = welch(flat, fs=sfreq, nperseg=min(TBR_NPERSEG, flat.shape[-1]), axis=-1)
    mean_psd = psd.mean(axis=0)

    theta_power = _band_power(mean_psd, freqs, *THETA_BAND_HZ)
    beta_power = _band_power(mean_psd, freqs, *BETA_BAND_HZ)

    if not np.isfinite(beta_power) or beta_power <= 0:
        return float("nan")  # avoid a divide-by-zero producing a fake infinite TBR
    return float(theta_power / beta_power)


def compute_classical_features(epochs_by_task: dict) -> dict:
    """
    epochs_by_task: {"EC": ec_epochs, "EO": eo_epochs, "VCPT": vcpt_epochs}
    for one subject, as returned by preprocessing.preprocess_subject().

    Returns one row's worth of classical features for the fusion classifier.
    TBR computed separately per condition (matching the paper's own "EC/EO/VCPT"
    feature-group structure in Table 1, not a single collapsed value) --
    ADHD-related frontal slowing may show up differently at rest vs. under
    task load, so collapsing to one number would throw away real signal.

    p300_latency_ms, p300_amplitude, omission_errors, commission_errors,
    reaction_time_ms are explicitly NaN -- NOT fabricated -- pending the
    trigger/condition-coding confirmation documented in PROJECT.md limitations.
    """
    features = {}
    for task in ["EC", "EO", "VCPT"]:
        key = f"tbr_{task.lower()}"
        features[key] = compute_tbr(epochs_by_task[task]) if task in epochs_by_task else float("nan")

    # Explicitly present as NaN, not omitted -- so the fusion classifier's
    # input schema is stable regardless of whether these become available
    # later, and so nobody mistakes "not in the dict" for "computed as zero".
    features["p300_latency_ms"] = float("nan")
    features["p300_amplitude"] = float("nan")
    features["omission_errors"] = float("nan")
    features["commission_errors"] = float("nan")
    features["reaction_time_ms"] = float("nan")

    return features