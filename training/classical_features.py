"""
Phase 2 — classical EEG biomarker features, for fusion with the CNN.

Per PROJECT.md sec 5a #1: this is the SINGLE HIGHEST-LEVERAGE item for
accuracy on this dataset -- the paper's own Table 2 shows a feature-selection
+ Logistic Regression combo hit 84.5% using engineered features and no deep
learning at all. TBR here, not the CNN, is what's most likely to move the
needle past both 75.8% and 84.5% when fused.

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

FRONTAL_CHANNELS = ["F3", "F4", "Fz"]  # matches the paper's TBR site exactly
THETA_BAND_HZ = (4, 8)
BETA_BAND_HZ = (12, 30)
TBR_HYPOARQUOSAL_THRESHOLD = 3.0  # from the original blueprint's biomarker
# engine concept -- kept as an informational flag, NOT used to make the final
# decision. The meta-classifier (fusion_classifier.py) is what actually
# decides; this threshold is for the dashboard's plain-language explanation
# later in Phase 5, not a hard rule here.


def compute_tbr(epochs, frontal_channels: list = FRONTAL_CHANNELS) -> float:
    """
    Theta/Beta Ratio at frontal channels, matching the paper's formula:
    TBR = Power(theta 4-8Hz) / Power(beta 12-30Hz), averaged across frontal
    channels, computed from the full recording's power spectrum (not
    per-epoch -- like coherence, TBR is a subject/condition-level summary
    feature in the source paper's own methodology, not a per-epoch value).

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
    freqs, psd = welch(flat, fs=sfreq, nperseg=min(256, flat.shape[-1]), axis=-1)
    mean_psd = psd.mean(axis=0)

    theta_power = mean_psd[(freqs >= THETA_BAND_HZ[0]) & (freqs <= THETA_BAND_HZ[1])].mean()
    beta_power = mean_psd[(freqs >= BETA_BAND_HZ[0]) & (freqs <= BETA_BAND_HZ[1])].mean()

    if beta_power <= 0:
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