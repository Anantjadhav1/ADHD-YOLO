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

# --- Expanded feature set ---------------------------------------------------
#
# TBR alone gave AUC 0.43 / 0.49 / 0.55 on this cohort -- chance, replicating
# Arns et al. (2013) and the later nulls. The CNN arm also came in at chance
# (0.503-0.523 across two representations and two capacity settings). So the
# classical arm needs to be more than three features before "classical vs CNN"
# is a meaningful comparison at all: Rohani et al. reached 84.5% with 113
# features selected from 826, not with TBR.
#
# Two of these target confounds the TBR literature names EXPLICITLY:
#
#   INDIVIDUAL ALPHA PEAK FREQUENCY. Alpha peak rises with age, so a child
#   whose IAF sits at 7-8 Hz has genuine alpha power inside the 4-8 Hz theta
#   window -- inflating TBR with no excess theta at all. The 2026 eLife
#   multiverse analysis identifies this as a primary reason TBR fails.
#
#   APERIODIC EXPONENT AND OFFSET. A difference in 1/f slope shifts every
#   band-power measure, and TBR is maximally sensitive because theta and beta
#   sit at opposite ends of the spectrum. Some of the historical "excess theta
#   in ADHD" may be an aperiodic difference misread as an oscillatory one.
#
# Feature count: 19 channels x 5 bands x 3 conditions relative power (285),
# plus IAF, aperiodic exponent/offset and frontal alpha asymmetry per
# condition, plus EC->EO alpha reactivity. Roughly 300 features on 84
# subjects. That is p >> n, so FEATURE SELECTION MUST HAPPEN INSIDE THE CV
# LOOP -- selecting on the full dataset and then cross-validating is the
# classic leak, and worth noting in the discussion that the 84.5% being
# chased may itself suffer from it.
ALL_BANDS_HZ = {
    "delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 12),
    "beta": (12, 30), "gamma": (30, 50),
}
BROADBAND_HZ = (0.5, 50)
IAF_SEARCH_HZ = (7.0, 13.0)
OCCIPITAL_CHANNELS = ["O1", "O2"]
# Range for the 1/f fit. The alpha band is EXCLUDED from it: a strong
# oscillatory peak tilts a log-log regression and biases the exponent.
# Verified on synthetic spectra -- with alpha amplitude varied 8-fold and the
# true exponent fixed at 1.6, the recovered value stayed 1.61-1.62.
APERIODIC_FIT_HZ = (1.0, 45.0)
APERIODIC_EXCLUDE_HZ = (7.0, 13.0)

# --- Connectivity -----------------------------------------------------------
#
# Every measurement in this project so far derives from spectral POWER: TBR,
# relative band power, the aperiodic slope, and the scalogram/topomap images
# are power representations too. All five landed at AUC 0.50-0.55.
#
# Coherence measures something physiologically different -- the phase
# relationship between regions, i.e. whether they are communicating, not how
# much each is doing. It is the one distinct feature family untested, and the
# source paper lists it among its five retained groups.
#
# SUMMARISED, not enumerated. A 19x19 matrix over 5 bands and 2 conditions is
# 1,710 pairwise values on 84 subjects -- the selector would be choosing among
# noise. These 25 region-level summaries are the ones connectivity research
# actually reports. Verified on a synthetic matrix with a seeded
# frontal-parietal link: the summary recovered 0.4000 for that pair and ~0.02
# elsewhere.
COHERENCE_REGIONS = {
    "fp": ["Fp1", "Fp2"],
    "f":  ["F7", "F3", "Fz", "F4", "F8"],
    "t":  ["T3", "T4", "T5", "T6"],
    "c":  ["C3", "Cz", "C4"],
    "p":  ["P3", "Pz", "P4"],
    "o":  ["O1", "O2"],
}
LEFT_CHANNELS = ["Fp1", "F7", "F3", "T3", "C3", "T5", "P3", "O1"]
RIGHT_CHANNELS = ["Fp2", "F8", "F4", "T4", "C4", "T6", "P4", "O2"]
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

def _mean_psd(epochs, channels: list) -> tuple:
    """Welch PSD averaged over epochs, for the requested channels.

    Returns (freqs, psd) where psd has one row per requested channel, in the
    order given. Channels absent from the recording are dropped rather than
    filled, and the caller sees a shorter list -- silently returning zeros
    would make a missing electrode look like a flat one.
    """
    available = [c for c in channels if c in epochs.ch_names]
    if not available:
        return np.array([]), np.array([]), []
    data = epochs.get_data(picks=available)      # (n_epochs, n_ch, n_samples)
    sfreq = float(epochs.info["sfreq"])
    nps = int(min(TBR_NPERSEG, data.shape[-1]))
    freqs, psd = welch(data, fs=sfreq, nperseg=nps, axis=-1)
    return freqs, psd.mean(axis=0), available


def compute_relative_band_power(epochs, channels: list = None) -> dict:
    """Relative band power per channel: band power / broadband power.

    RELATIVE rather than absolute, deliberately. Absolute power varies with
    skull thickness, electrode impedance and gel quality -- between-subject
    differences that have nothing to do with the brain. Dividing by each
    subject's own broadband power removes that scaling. It also partially
    normalises the 1/f slope, which is why the aperiodic exponent is computed
    separately rather than being left implicit here.
    """
    channels = channels or CHANNELS_19
    freqs, psd, available = _mean_psd(epochs, channels)
    if len(available) == 0:
        return {}
    out = {}
    for i, ch in enumerate(available):
        total = _band_power(psd[i], freqs, *BROADBAND_HZ)
        for band, (lo, hi) in ALL_BANDS_HZ.items():
            bp = _band_power(psd[i], freqs, lo, hi)
            out[f"relpow_{ch}_{band}"] = (bp / total
                                          if np.isfinite(total) and total > 0
                                          else float("nan"))
    return out


def compute_iaf(epochs, channels: list = OCCIPITAL_CHANNELS) -> float:
    """Individual alpha peak frequency: the frequency of maximum power in
    7-13 Hz, over occipital channels where alpha is strongest.

    Verified on synthetic spectra with a known injected peak -- recovered
    within one frequency bin at 8.0, 9.5, 10.0 and 11.5 Hz.
    """
    freqs, psd, available = _mean_psd(epochs, channels)
    if len(available) == 0:
        return float("nan")
    mean_psd = psd.mean(axis=0)
    mask = (freqs >= IAF_SEARCH_HZ[0]) & (freqs <= IAF_SEARCH_HZ[1])
    if mask.sum() < 3:
        return float("nan")
    return float(freqs[mask][np.argmax(mean_psd[mask])])


def compute_aperiodic(epochs, channels: list = None) -> tuple:
    """(exponent, offset) of the 1/f background, from a log-log linear fit.

    This is what specparam/FOOOF does at its core, minus the iterative peak
    removal. Using a plain fit avoids a new dependency; excluding the alpha
    band from the fit range covers the main thing the iteration is there to
    handle. State the simplification in the methods -- do not call it FOOOF.

    Exponent is returned POSITIVE (a steeper 1/f gives a larger number), which
    is the convention in the aperiodic literature.
    """
    channels = channels or CHANNELS_19
    freqs, psd, available = _mean_psd(epochs, channels)
    if len(available) == 0:
        return float("nan"), float("nan")
    mean_psd = psd.mean(axis=0)
    m = ((freqs >= APERIODIC_FIT_HZ[0]) & (freqs <= APERIODIC_FIT_HZ[1])
         & (mean_psd > 0))
    m &= ~((freqs >= APERIODIC_EXCLUDE_HZ[0]) & (freqs <= APERIODIC_EXCLUDE_HZ[1]))
    if m.sum() < 10:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(np.log10(freqs[m]), np.log10(mean_psd[m]), 1)
    return float(-slope), float(intercept)


def compute_frontal_asymmetry(epochs, left: str = "F3", right: str = "F4") -> float:
    """log(right alpha) - log(left alpha) at F3/F4.

    Log difference rather than a raw ratio so the measure is symmetric: a
    two-fold left bias and a two-fold right bias give equal and opposite
    values rather than 0.5 and 2.0.
    """
    freqs, psd, available = _mean_psd(epochs, [left, right])
    if len(available) < 2:
        return float("nan")
    lo, hi = ALL_BANDS_HZ["alpha"]
    a_left = _band_power(psd[available.index(left)], freqs, lo, hi)
    a_right = _band_power(psd[available.index(right)], freqs, lo, hi)
    if not (a_left > 0 and a_right > 0):
        return float("nan")
    return float(np.log(a_right) - np.log(a_left))

def _pair_mean(mat: np.ndarray, chs: list, a: list, b: list) -> float:
    """Mean connectivity between two channel sets.

    Within a region (a is b) the diagonal is excluded -- a channel's coherence
    with itself is 1 by definition and would inflate every within-region value
    by a constant that varies with region size.
    """
    ia = [chs.index(c) for c in a if c in chs]
    ib = [chs.index(c) for c in b if c in chs]
    if not ia or not ib:
        return float("nan")
    sub = mat[np.ix_(ia, ib)]
    if set(a) == set(b):
        off = ~np.eye(len(ia), dtype=bool)
        return float(sub[off].mean()) if off.any() else float("nan")
    return float(sub.mean())


def compute_coherence_features(raw_segment, ch_names: list = None) -> dict:
    """Region-level imaginary-coherence summaries for one condition.

    IMAGINARY coherence, not plain coherence. Plain coherence came back at
    0.98-0.999 across every channel pair regardless of scalp distance -- volume
    conduction, where one source appears at many electrodes with zero phase lag,
    not connectivity. The imaginary part discards zero-lag coupling by
    construction. Confirmed on real data during the image work.

    Takes a continuous Raw segment, not 1.5 s epochs: MNE refuses Delta at that
    epoch length ("0.750 < 5 cycles"), so this re-epochs at 10 s the same way
    generate_coherence_image does.

    Returns {} rather than raising if the segment is too short -- a subject
    missing connectivity features should be imputed alongside the others, not
    kill the batch.
    """
    import mne
    import mne_connectivity
    from data_pipeline.image_conversion import (COHERENCE_OVERLAP,
                                                COHERENCE_WINDOW_SEC,
                                                TOPOMAP_BANDS)

    ch_names = ch_names or CHANNELS_19
    picks = [c for c in ch_names if c in raw_segment.ch_names]
    if len(picks) < 4:
        return {}

    sfreq = float(raw_segment.info["sfreq"])
    step = COHERENCE_WINDOW_SEC * (1.0 - COHERENCE_OVERLAP)
    try:
        events = mne.make_fixed_length_events(raw_segment, duration=step)
        eps = mne.Epochs(raw_segment, events, tmin=0,
                         tmax=COHERENCE_WINDOW_SEC - 1.0 / sfreq,
                         baseline=None, preload=True, picks=picks,
                         reject=None, verbose=False)
        if len(eps) < 5:
            return {}
        con = mne_connectivity.spectral_connectivity_epochs(
            eps, method="imcoh", mode="multitaper",
            fmin=tuple(lo for lo, _ in TOPOMAP_BANDS.values()),
            fmax=tuple(hi for _, hi in TOPOMAP_BANDS.values()),
            sfreq=sfreq, faverage=True, verbose=False)
    except Exception:  # noqa: BLE001
        return {}

    # imcoh is signed; magnitude is the meaningful quantity. The library fills
    # only the lower triangle, so mirror it.
    data = np.abs(con.get_data(output="dense"))
    out = {}
    names = list(COHERENCE_REGIONS)
    for bi, band in enumerate(TOPOMAP_BANDS):
        m = data[:, :, bi]
        m = m + m.T
        b = band.lower()
        for i, ra in enumerate(names):
            for rb in names[i:]:
                out[f"coh_{ra}_{rb}_{b}"] = _pair_mean(
                    m, picks, COHERENCE_REGIONS[ra], COHERENCE_REGIONS[rb])
        out[f"coh_intra_left_{b}"] = _pair_mean(m, picks, LEFT_CHANNELS, LEFT_CHANNELS)
        out[f"coh_intra_right_{b}"] = _pair_mean(m, picks, RIGHT_CHANNELS, RIGHT_CHANNELS)
        out[f"coh_interhemi_{b}"] = _pair_mean(m, picks, LEFT_CHANNELS, RIGHT_CHANNELS)
        off = ~np.eye(m.shape[0], dtype=bool)
        out[f"coh_overall_{b}"] = float(m[off].mean())
    return out

def compute_classical_features(epochs_by_task: dict, raw_by_task: dict = None) -> dict:
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

    # Expanded set -- see the note above ALL_BANDS_HZ for why three features
    # was never going to be a fair classical arm.
    alpha_by_task = {}
    for task in ["EC", "EO", "VCPT"]:
        t = task.lower()
        if task not in epochs_by_task:
            continue
        ep = epochs_by_task[task]
        for k, v in compute_relative_band_power(ep).items():
            features[f"{k}_{t}"] = v
        features[f"iaf_{t}"] = compute_iaf(ep)
        exp, off = compute_aperiodic(ep)
        features[f"aperiodic_exponent_{t}"] = exp
        features[f"aperiodic_offset_{t}"] = off
        features[f"frontal_alpha_asym_{t}"] = compute_frontal_asymmetry(ep)
        freqs, psd, avail = _mean_psd(ep, OCCIPITAL_CHANNELS)
        alpha_by_task[t] = (_band_power(psd.mean(axis=0), freqs, *ALL_BANDS_HZ["alpha"])
                            if len(avail) else float("nan"))

    # EC -> EO alpha reactivity: how much occipital alpha COLLAPSES on eye
    # opening. A functional measure rather than a static one -- it asks whether
    # the alpha-blocking response is intact, not how much alpha there is. The
    # EC/EO boundary work already showed this response varies a lot between
    # these children.
        # Connectivity, from the continuous segments. VCPT excluded deliberately:
    # coherence here is the resting-state EC/EO representation, matching how
    # the source paper uses it.
    for task, key in (("EC", "ec"), ("EO", "eo")):
        seg = (raw_by_task or {}).get(task)
        if seg is not None:
            for k, v in compute_coherence_features(seg).items():
                features[f"{k}_{key}"] = v
    ec, eo = alpha_by_task.get("ec", float("nan")), alpha_by_task.get("eo", float("nan"))
    features["alpha_reactivity"] = (float(np.log(ec) - np.log(eo))
                                    if (np.isfinite(ec) and np.isfinite(eo)
                                        and ec > 0 and eo > 0) else float("nan"))

    # Explicitly present as NaN, not omitted -- so the fusion classifier's
    # input schema is stable regardless of whether these become available
    # later, and so nobody mistakes "not in the dict" for "computed as zero".
    features["p300_latency_ms"] = float("nan")
    features["p300_amplitude"] = float("nan")
    features["omission_errors"] = float("nan")
    features["commission_errors"] = float("nan")
    features["reaction_time_ms"] = float("nan")

    return features