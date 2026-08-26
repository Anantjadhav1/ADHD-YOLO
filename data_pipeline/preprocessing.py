"""
Phase 1 — EEG preprocessing.

Built against the REAL Rohani et al. dataset structure (confirmed by inspecting
actual .edf files, not just the paper text). Key differences from a naive
reading of the paper:

- Two separate files per subject per session: "<ID>-<date>-<time>-EOEC.edf"
  (resting state) and "<ID>-<date>-<time>-VCPT.edf" (Go/NoGo task).
- Files contain 22 channels: the real 19 EEG channels + X1, X2 (confirmed
  dead/unused — near-zero flat signal, not usable EOG) + LABEL (a digital
  marker channel, not standard EDF annotations).
- EOEC files have NO event markers at all. EC/EO segmentation is done via
  the alpha-blocking effect (occipital alpha power is much higher during
  eyes-closed rest) instead of an external timing file — validated on 5 real
  subjects, works cleanly on 4/5 (the 5th just gets flagged for manual QC,
  not silently trusted).
- VCPT LABEL channel does NOT reliably encode the paper's 4 trial conditions
  (A-A/A-P/P-P/P-H): pulse count varies 100-175 across subjects and pulse
  width is continuous, not clustered — consistent with a behavioral response
  marker (e.g. button press duration), not a stimulus/condition code. P300
  latency/amplitude and per-condition behavioral features are NOT reliably
  recoverable from this alone. Treated as pending until confirmed otherwise
  (see PROJECT.md limitations).
"""

import re
import warnings
from collections import Counter
from dataclasses import dataclass

import mne
import numpy as np
from scipy.signal import welch


# --- Recording parameters (confirmed against real files) ---
SAMPLING_RATE_HZ = 500
BANDPASS_LOW_HZ = 0.5
BANDPASS_HIGH_HZ = 50.0
NOTCH_FREQ_HZ = 50.0

CHANNELS_19 = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3", "C3", "Cz", "C4", "T4", "T5",
    "P3", "Pz", "P4", "T6", "O1", "O2",
]
DEAD_CHANNELS = ["X1", "X2"]  # confirmed near-zero, flat, unusable — dropped
LABEL_CHANNEL = "LABEL"

EPOCH_LENGTH_SEC = 1.5
ALPHA_BAND_HZ = (8, 12)

# --- Artifact rejection ---
# No real EOG channel exists in this dataset (X1/X2 confirmed dead), so blink
# detection uses the frontopolar channels as proxies. Blinks dominate Fp1/Fp2,
# so the correlation is driven by ocular activity -- but note these ARE real
# EEG channels, so some genuine frontopolar brain signal is removed with them.
# Acceptable here because TBR is computed at F3/F4/Fz, not Fp1/Fp2. State this
# limitation in the methods section.
EOG_PROXY_CHANNELS = ["Fp1", "Fp2"]

# ICA decomposition degrades with low-frequency drift, so ICA is FIT on a 1 Hz
# high-passed copy and the resulting solution APPLIED to the 0.5 Hz data. This
# is standard MNE practice, not a deviation.
ICA_HIGHPASS_HZ = 1.0

# Peak-to-peak epoch rejection. 150 uV is a conventional starting point for
# pediatric EEG after ICA has removed blinks. Rejection RATE is reported per
# recording so unusually bad subjects surface instead of silently contributing
# fewer images.
# Set from the measured amplitude distribution on C09090107 EC (post-ICA), not
# from a literature convention -- 150 uV rejected 100% of epochs pre-ICA and
# 15% post-ICA, cutting into the bulk of the distribution rather than its tail.
#
# Post-ICA per-channel medians run 33-103 uV, with O1/O2/Pz highest, which is
# genuine eyes-closed occipital alpha, not artifact. The worst-channel
# distribution has p90 = 167 uV and p95 = 544 uV -- a sharp discontinuity, so
# real artifact begins somewhere past 500.
#
# 250 uV sits in that gap: above O2's p90 (146 uV) so strong-alpha epochs
# survive, below the outlier population so transients are caught. Keeps 92%.
# A tighter threshold would preferentially reject high-alpha epochs, biasing
# the EC condition against its own dominant physiological feature.
# Set from the measured post-ICA amplitude distribution on C09090107 EC, not a
# literature convention. 150 uV rejected 100% of epochs pre-ICA and 15%
# post-ICA, cutting into the bulk of the distribution rather than its tail.
# Post-ICA per-channel medians run 33-103 uV, highest at O1/O2/Pz -- that is
# genuine eyes-closed occipital alpha, not artifact. Worst-channel p90 = 167 uV,
# p95 = 544 uV: a sharp discontinuity, so real artifact starts past ~500.
# 250 uV sits in that gap and keeps 92% of epochs. A tighter threshold would
# preferentially reject high-alpha epochs, biasing EC against its own dominant
# physiological feature.
REJECT_PEAK_TO_PEAK_V = 250e-6

# Muscle-component detection threshold. MNE's default (0.5) flagged 5 of 19
# components on C09090107 and 14 of 19 on F09080101 -- destroying the
# decomposition. A sweep across 0.5-1.0 on 4 subjects (see
# training/sweep_muscle_threshold.py) showed TBR moving 8-311% with this
# parameter alone, in INCONSISTENT directions across subjects. 0.9 keeps 3 of 4
# subjects inside the 1-4 component target; 1.0 disables muscle detection
# entirely. Report this sensitivity in the methods -- do not present a single
# TBR value as if the parameter were fixed by anything but judgement.
MUSCLE_THRESHOLD = 0.9

# Hard cap on components removed. If detection wants more than a quarter of the
# decomposition, the recording is the problem -- flag the subject rather than
# silently reconstructing it from a handful of components. EOG is never capped
# (unambiguous, consistently 2 across every subject and threshold tested);
# muscle is capped by score, strongest kept.
MAX_ICA_COMPONENTS_EXCLUDED = 5

# --- Topography veto -------------------------------------------------------
#
# find_bads_eog and find_bads_muscle judge components on SPECTRAL criteria
# alone. A spatially diffuse component whose frequency content happens to match
# gets removed -- and removing a diffuse component subtracts signal from every
# channel proportionally, occipital alpha included. Measured on all 108
# subjects (training/audit_ica_alpha.py): median occipital alpha retained
# across ICA was 0.643, with 34/108 subjects losing more than half.
#
# Measured on the 313 components that were being removed, by peak region:
#
#     peak_region   central  frontal  frontopolar  occipital  parietal  temporal
#     eog                17       13           77         20        17        12
#     muscle             11       30           30         15        11        57
#
# An eog component should peak frontopolar; 58% do (incl. lateral frontal).
# A muscle component should peak temporal; 37% do. Overall 48%. Both
# detectors are wrong more often than right.
#
# So: a spectral detector needs a spatial sanity check, exactly as the EC/EO
# changepoint detector needed a physical plausibility check on top of its
# statistics. Same error, different domain.
REGIONS = {
    "frontopolar": ["Fp1", "Fp2"],            # blinks, vertical eye movement
    "frontal":     ["F7", "F3", "Fz", "F4", "F8"],
    "temporal":    ["T3", "T4", "T5", "T6"],  # EMG lives here
    "central":     ["C3", "Cz", "C4"],
    "parietal":    ["P3", "Pz", "P4"],
    "occipital":   ["O1", "O2"],              # alpha
}
EXPECTED_PEAK_REGION = {
    "eog": {"frontopolar", "frontal"},
    "muscle": {"temporal"},
}
# How far the strongest region must stand above the scalp mean. 1.0 is
# perfectly uniform. Reference topographies score 4.05 (genuine ocular) and
# ~2.5+ (temporal muscle); the median component actually being removed scored
# 1.18 for eog.
#
# 2.0 rather than 1.5 deliberately. The two rules keep 0.69 vs 0.96 components
# per subject with estimated retention 0.996 vs 0.934. The errors are not
# symmetric: losing alpha is DEMONSTRATED and damages the EC/EO split plus
# every generated image, while surviving blinks inflate TBR's numerator on a
# biomarker already measured at AUC ~0.5 -- and 250 uV epoch rejection catches
# gross ocular artifact regardless. At 0.69/subject roughly a third of subjects
# get zero exclusions, which is ICA effectively off for them; that is only
# acceptable because the detectors are right 48% of the time.
MIN_COMPONENT_FOCALITY = 2.0


def _region_weight(topo: np.ndarray, ch_names: list, region: list) -> float:
    """Mean |weight| over a region, divided by the mean over all channels.

    Scale-free because ICA component sign and magnitude are arbitrary -- only
    the SHAPE of a topography carries meaning. 1.0 means the region is exactly
    average; above 1.0 means it dominates.
    """
    a = np.abs(topo)
    m = a.mean()
    if m <= 0:
        return float("nan")
    idx = [ch_names.index(c) for c in region if c in ch_names]
    return float(a[idx].mean() / m) if idx else float("nan")


def _topography_veto(ica, fitted_ch: list, exclude: set, reasons: dict) -> tuple:
    """Drop components whose scalp topography contradicts their detection reason.

    Returns (kept, vetoed_detail). A component survives only if it peaks in a
    region consistent with why it was flagged AND is focal enough there.
    """
    mixing = ica.get_components()
    kept, vetoed = set(), {}
    for i in sorted(exclude):
        topo = mixing[:, i]
        weights = {n: _region_weight(topo, fitted_ch, ch) for n, ch in REGIONS.items()}
        finite = {n: w for n, w in weights.items() if np.isfinite(w)}
        if not finite:
            vetoed[i] = "no valid topography"
            continue
        peak = max(finite, key=finite.get)
        focality = finite[peak]

        reason = "+".join(reasons.get(i, ["?"]))
        expected = set()
        for r in reason.split("+"):
            expected |= EXPECTED_PEAK_REGION.get(r, set())

        if expected and peak not in expected:
            vetoed[i] = f"{reason} but peaks {peak} (expected {'/'.join(sorted(expected))})"
        elif focality < MIN_COMPONENT_FOCALITY:
            vetoed[i] = f"{reason} peaks {peak} but focality {focality:.2f} < {MIN_COMPONENT_FOCALITY}"
        else:
            kept.add(i)
    return kept, vetoed
FLAT_THRESHOLD_V = 1e-7  # catches dead/disconnected channel segments
HIGH_REJECTION_RATE_WARN = 0.30
ALPHA_AMBIGUOUS_RATIO_RANGE = (0.7, 1.4)  # ratio inside this range -> flag for manual QC


@dataclass
class SubjectFile:
    subject_id: str
    group: str  # "ADHD" or "Control"
    date: str
    time: str
    task: str  # "EOEC" or "VCPT"
    filepath: str


def parse_filename(filepath: str) -> SubjectFile:
    """
        Parse the real filename convention: <ID>-<YYYY.MM.DD>-<HH.MM.SS>-<TASK>.edf
    e.g. C09090107-2019.12.29-15.25.17-EOEC.edf
    Dots, underscores and hyphens are all accepted as date/time separators.
    """
    import os
    fname = os.path.basename(filepath)
    m = re.match(
    r"([A-Za-z0-9]+)-(\d{4}[._-]\d{2}[._-]\d{2})-(\d{2}[._-]\d{2}[._-]\d{2})-(\w+)\.edf",
    fname, re.IGNORECASE
 )
    if not m:
        raise ValueError(f"Filename doesn't match expected pattern: {fname}")
    subject_id, date, time, task = m.groups()
    group = "ADHD" if subject_id.upper().startswith("F") else "Control"
    return SubjectFile(subject_id=subject_id, group=group, date=date, time=time, task=task.upper(), filepath=filepath)


def load_raw(filepath: str) -> mne.io.Raw:
    """Load a raw recording, drop the confirmed-dead X1/X2 channels, and
    attach a standard 10-20 montage — the files themselves carry no channel
    position data, which topomap plotting needs downstream."""
    with warnings.catch_warnings():
        # These EDFs declare per-channel highpass/lowpass values that are not
        # all identical, so MNE warns twice per file that it is storing the
        # most conservative of each. That is the correct behaviour and there is
        # nothing to act on -- but it is 4 lines per subject, and across 108
        # subjects it buries the warnings that DO matter (rejection rates, ICA
        # caps, EC/EO ambiguity). Suppressed by exact message so anything else
        # RuntimeWarning still surfaces.
        warnings.filterwarnings(
            "ignore",
            message=r".*Channels contain different (highpass|lowpass) filters.*",
            category=RuntimeWarning,
        )
        raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
    drop = [ch for ch in DEAD_CHANNELS if ch in raw.ch_names]
    if drop:
        raw.drop_channels(drop)
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, on_missing="ignore")
    return raw


def filter_raw(raw: mne.io.Raw) -> mne.io.Raw:
    """
    Bandpass + notch, applied ONLY to the 19 EEG channels — the LABEL
    channel is a digital marker, not a physiological signal, and must not
    be smoothed by the same filter or its edges become unusable for event
    detection.
    """
    eeg_picks = [ch for ch in CHANNELS_19 if ch in raw.ch_names]
    raw = raw.copy().filter(
        l_freq=BANDPASS_LOW_HZ, h_freq=BANDPASS_HIGH_HZ,
        picks=eeg_picks, method="fir", phase="zero", verbose=False,
    )
    raw = raw.notch_filter(freqs=NOTCH_FREQ_HZ, picks=eeg_picks, verbose=False)
    return raw


def remove_artifacts_ica(raw: mne.io.Raw, n_components: int = 19,
                         eog_proxy: list = EOG_PROXY_CHANNELS,
                         return_ica: bool = False) -> tuple:
    """
    ICA-based artifact removal on the 19 EEG channels.

    PREVIOUSLY A NO-OP: this function used to fit an ICA and then call
    ica.apply() with an empty exclude list, which reconstructs the signal
    bit-identically. It cost a full ICA fit per recording and changed nothing.
    Every image and feature produced before 2026-08-24 came from unrejected
    data. See PROGRESS.md.

    Two detectors run:
      - EOG (blinks/saccades) via correlation with Fp1/Fp2 as proxies
      - Muscle (EMG) via ica.find_bads_muscle

    Muscle detection matters specifically here: EMG contaminates 20 Hz and
    above, which is the beta band -- the DENOMINATOR of TBR. Blinks contaminate
    delta/theta, the numerator. So artifact hits the primary biomarker from
    both directions.

    Returns (cleaned_raw, diagnostics). Diagnostics are returned rather than
    logged so callers can record them per subject -- silent cleaning would hide
    a recording where half the components were rejected.

    Never raises on detection failure: warns and returns the data uncleaned,
    so one bad subject can't kill a 103-subject batch.
    """
    eeg_picks = [ch for ch in CHANNELS_19 if ch in raw.ch_names]

    # Fit on a 1 Hz high-passed copy; apply to the original.
    raw_for_fit = raw.copy().filter(
        l_freq=ICA_HIGHPASS_HZ, h_freq=None, picks=eeg_picks,
        method="fir", phase="zero", verbose=False,
    )

    ica = mne.preprocessing.ICA(
        n_components=min(n_components, len(eeg_picks)),
        random_state=42, max_iter="auto",
    )
    ica.fit(raw_for_fit, picks=eeg_picks, verbose=False)

    exclude, reasons = set(), {}

    proxies = [ch for ch in eog_proxy if ch in raw.ch_names]
    if proxies:
        try:
            eog_idx, _ = ica.find_bads_eog(raw_for_fit, ch_name=proxies, verbose=False)
            for i in eog_idx:
                reasons.setdefault(i, []).append("eog")
            exclude.update(eog_idx)
        except Exception as e:  # noqa: BLE001 - never kill the batch
            warnings.warn(f"EOG component detection failed: {e}")
    else:
        warnings.warn("No EOG proxy channels present; skipping blink detection.")

        muscle_idx = []
    try:
        muscle_idx, muscle_scores = ica.find_bads_muscle(
            raw_for_fit, threshold=MUSCLE_THRESHOLD, verbose=False)
        # Cap by score, keeping the strongest. EOG is never capped.
        allowed = max(0, MAX_ICA_COMPONENTS_EXCLUDED - len(exclude))
        if len(muscle_idx) > allowed:
            ranked = sorted(muscle_idx, key=lambda i: abs(muscle_scores[i]), reverse=True)
            capped_out = ranked[allowed:]
            muscle_idx = ranked[:allowed]
            warnings.warn(
                f"Muscle detection flagged {len(ranked)} of {ica.n_components_} "
                f"components; capped to {allowed}. Dropped from the exclude list: "
                f"{sorted(capped_out)}. FLAG THIS SUBJECT FOR MANUAL QC -- a "
                f"decomposition this contaminated is a recording problem, and "
                f"reconstructing from the remainder would not be trustworthy."
            )
        for i in muscle_idx:
            reasons.setdefault(i, []).append("muscle")
        exclude.update(muscle_idx)
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"Muscle component detection failed: {e}")

    # Spatial sanity check on a spectral detector -- see _topography_veto.
    fitted_ch = [c for c in CHANNELS_19 if c in raw.ch_names]
    detected = dict(reasons)
    exclude, vetoed = _topography_veto(ica, fitted_ch, exclude, reasons)
    reasons = {i: r for i, r in reasons.items() if i in exclude}
    if vetoed:
        warnings.warn(
            f"Topography veto blocked {len(vetoed)} of {len(detected)} detected "
            f"components: {'; '.join(f'{i}: {r}' for i, r in sorted(vetoed.items()))}"
        )

    ica.exclude = sorted(exclude)
    raw_clean = ica.apply(raw.copy(), verbose=False)

    n_flagged = len(proxies and exclude or exclude)
    diagnostics = {
        "n_components": int(ica.n_components_),
        "n_excluded": len(ica.exclude),
        "excluded": [int(i) for i in ica.exclude],
        "reasons": {int(i): "+".join(r) for i, r in sorted(reasons.items())},
        "eog_proxy_used": proxies,
        "muscle_threshold": MUSCLE_THRESHOLD,
        # True when the cap fired -- carry this into the audit log so
        # over-contaminated subjects can be reviewed before Phase 2.
        "capped": len(ica.exclude) >= MAX_ICA_COMPONENTS_EXCLUDED,
        "n_detected": len(detected),
        "n_vetoed": len(vetoed),
        "vetoed": {int(i): r for i, r in sorted(vetoed.items())},
        "focality_threshold": MIN_COMPONENT_FOCALITY,
    }
    if return_ica:
        # For diagnostics that need the component TOPOGRAPHIES, not just which
        # indices were excluded -- e.g. checking whether an "ocular" component
        # actually loads on O1/O2, which would mean it carries alpha rather
        # than eye movement. Off by default so the normal return shape is
        # unchanged; the alternative was refitting ICA in the audit, doubling
        # the slowest step in the pipeline.
        diagnostics["ica"] = ica
    return raw_clean, diagnostics


def split_eoec_by_alpha(raw: mne.io.Raw, trim_sec: float = 15.0,
                        boundary_frac: float | None = None) -> dict:
    """
    boundary_frac: split at this fraction of the recording instead of the
        midpoint. None keeps the existing `half = n // 2` behaviour.

    WHY THIS EXISTS. The midpoint is an assumption nobody checked until
    2026-08-25. A changepoint detector run on all 108 subjects post-ICA
    (training/find_ec_eo_boundary.py) found the trusted-only median boundary
    at 0.537, not 0.500 -- Wilcoxon p < 0.00001. So the midpoint misplaces
    roughly 18 s of an 8-minute recording, contaminating the EO segment with
    eyes-closed data.

    Real, but small, and the detector only validates for 64/108 subjects, so
    it is NOT applied wholesale -- regenerating 122k images to move a boundary
    by 3.7% was not worth 5 hours. It is used only for the five subjects the
    midpoint rule flagged as ambiguous AND whose detected boundary passes all
    three checks (permutation p<0.05, split-half |O1-O2|<=0.08, and inside a
    physically plausible 0.25-0.75 range).

    That third check is load-bearing: 19 subjects passed both STATISTICAL
    tests with boundaries at 0.10 or 0.89, putting one condition under a
    minute. A statistical test needs a physical sanity check on top of it.

    Callers must record which rule was used -- see build_dataset.py.
    """
    """
    Split a resting-state EOEC recording into eyes-closed (EC) and
    eyes-open (EO) segments using the alpha-blocking effect: occipital
    alpha power (8-12 Hz at O1/O2) is much higher during eyes-closed rest.

    Validated on 5 real subjects: 4/5 show a clean >1.9x ratio consistent
    with EC-first-then-EO ordering. Subjects with an ambiguous ratio are
    flagged, not silently trusted — check those manually before using them.

    Returns dict with 'ec' and 'eo' Raw segments, the alpha ratio, and an
    'ambiguous' flag.
    """
    sf = raw.info["sfreq"]
    data = raw.get_data(picks=["O1", "O2"]).mean(axis=0)
    n = len(data)
    trim = int(trim_sec * sf)
    half = n // 2 if boundary_frac is None else int(round(boundary_frac * n))
    # Guard: a caller-supplied fraction must still leave both segments usable.
    if not (trim < half < n - trim):
        raise ValueError(
            f"boundary_frac={boundary_frac} puts the split at sample {half} of "
            f"{n}, outside the usable range after trimming {trim_sec}s from each "
            "end. Fall back to the midpoint for this subject."
        )

    def alpha_power(seg):
        freqs, psd = welch(seg, fs=sf, nperseg=int(4 * sf))
        band = (freqs >= ALPHA_BAND_HZ[0]) & (freqs <= ALPHA_BAND_HZ[1])
        return psd[band].mean()

    a_first = alpha_power(data[trim:half])
    a_second = alpha_power(data[half:n - trim])
    ratio = a_first / a_second
    ambiguous = ALPHA_AMBIGUOUS_RATIO_RANGE[0] < ratio < ALPHA_AMBIGUOUS_RATIO_RANGE[1]

    # first half = higher alpha = eyes closed (standard ordering seen in 4/5 subjects)
    last_t = raw.times[-1]  # actual last valid timestamp — n/sf can overshoot by a sample due to float rounding
    ec_raw, eo_raw = (raw.copy().crop(tmin=0, tmax=half / sf),
                       raw.copy().crop(tmin=half / sf, tmax=last_t))
    if ratio < 1:
        # reversed order for this subject — swap
        ec_raw, eo_raw = eo_raw, ec_raw

    return {"ec": ec_raw, "eo": eo_raw, "alpha_ratio": ratio,
            "ambiguous": ambiguous,
            "boundary_frac": half / n,
            "split_rule": "midpoint" if boundary_frac is None else "detected"}

def epoch_signal(raw: mne.io.Raw, epoch_length_sec: float = EPOCH_LENGTH_SEC,
                 reject_uv: float = REJECT_PEAK_TO_PEAK_V,
                 return_diagnostics: bool = False):
    """
    Fixed-length sliding-window epochs with peak-to-peak artifact rejection.

    Rejection was previously ABSENT -- no reject parameter was passed, so every
    artifact-laden segment became a training image. Pass reject_uv=None to
    restore the old behaviour for comparison.

    tmax is epoch_length - 1/sfreq so epochs are exactly the intended length;
    tmax=epoch_length yields one extra sample and a 1-sample overlap.

    Set return_diagnostics=True to also get the rejection rate. Default is
    False so existing callers keep working unchanged.
    """
    events = mne.make_fixed_length_events(raw, duration=epoch_length_sec)
    sfreq = float(raw.info["sfreq"])

    reject = flat = None
    if reject_uv is not None:
        reject = dict(eeg=reject_uv)
        flat = dict(eeg=FLAT_THRESHOLD_V)

    # Restrict to the real 19 EEG channels, the same way filter_raw() and
    # remove_artifacts_ica() already do. Without this, reject/flat are applied
    # to every channel MNE types as eeg -- which includes LABEL, a digital
    # marker channel that is constant by construction. `flat` drops an epoch if
    # ANY channel falls below threshold, so a permanently-flat LABEL rejected
    # 100% of epochs on 100% of subjects, and the pipeline could not produce a
    # single image. LABEL stays on the Raw, which is what
    # extract_vcpt_behavioral_proxy() reads -- it is only excluded from epochs.
    eeg_picks = [ch for ch in CHANNELS_19 if ch in raw.ch_names]

    epochs = mne.Epochs(
        raw, events, tmin=0, tmax=epoch_length_sec - 1.0 / sfreq,
        baseline=None, preload=True, reject=reject, flat=flat,
        picks=eeg_picks, verbose=False,
    )

    n_total, n_kept = len(events), len(epochs)
    rate = 1.0 - (n_kept / n_total) if n_total else 0.0

    if n_kept == 0:
        # Name the channel that actually caused the drops. The previous message
        # blamed the peak-to-peak threshold unconditionally, which is what sent
        # an earlier investigation after the threshold value while the real
        # cause was a flat marker channel.
        culprits = Counter(ch for entry in epochs.drop_log for ch in entry)
        blamed = ", ".join(f"{ch} ({n})" for ch, n in culprits.most_common(3)) or "unknown"
        warnings.warn(
            f"ALL {n_total} epochs rejected (p2p > {reject_uv*1e6:.0f} uV or flat < "
            f"{FLAT_THRESHOLD_V*1e6:.2f} uV). Channels responsible, most frequent first: "
            f"{blamed}. This recording is unusable at these thresholds -- inspect it "
            "before including the subject."
        )
    elif rate > HIGH_REJECTION_RATE_WARN:
        warnings.warn(
            f"High rejection rate: {rate:.1%} ({n_total - n_kept}/{n_total} epochs). "
            "Flag this subject for manual QC."
        )

    if return_diagnostics:
        return epochs, {"n_events": n_total, "n_kept": n_kept,
                        "rejection_rate": float(rate),
                        "reject_uv": reject_uv}
    return epochs


def extract_vcpt_behavioral_proxy(raw: mne.io.Raw) -> dict:
    """
    Approximate behavioral summary stats from the LABEL channel pulses.
    NOT equivalent to the paper's true per-condition omission/commission/RT
    features — those need stimulus-locked triggers we don't have. This is a
    documented approximation (event count + duration stats) until/unless
    confirmed trigger info becomes available.
    """
    sf = raw.info["sfreq"]
    label = raw.get_data(picks=[LABEL_CHANNEL])[0]
    rising = np.where((label[:-1] == 0) & (label[1:] == 1))[0]
    falling = np.where((label[:-1] == 1) & (label[1:] == 0))[0]
    n = min(len(rising), len(falling))
    widths = (falling[:n] - rising[:n]) / sf
    return {
        "n_events": len(rising),
        "mean_event_duration_s": float(widths.mean()) if n else None,
        "std_event_duration_s": float(widths.std()) if n else None,
        "note": "approximation pending confirmed trigger/condition coding",
    }


def preprocess_subject(eoec_filepath: str, vcpt_filepath: str | None = None,
                       boundary_frac: float | None = None) -> dict:
    """Full pipeline for one subject's session.

    boundary_frac: forwarded to split_eoec_by_alpha. None keeps the midpoint.
        Only supplied for subjects whose detected boundary passed all three
        validation checks -- see split_eoec_by_alpha's docstring.
    """
    result = {}

    raw_eoec = load_raw(eoec_filepath)
    raw_eoec = filter_raw(raw_eoec)
    raw_eoec, ica_diag_eoec = remove_artifacts_ica(raw_eoec)
    result["ica_eoec"] = ica_diag_eoec
    split = split_eoec_by_alpha(raw_eoec, boundary_frac=boundary_frac)
    result["split_rule"] = split["split_rule"]
    result["boundary_frac"] = split["boundary_frac"]
    # Cleaned CONTINUOUS segments, kept alongside the epoched versions.
    # Coherence needs a different window length than the image pipeline (see
    # image_conversion.COHERENCE_WINDOW_SEC): at 1.5 s, MNE reports
    # "fmin=0.500 Hz corresponds to 0.750 < 5 cycles" -- less than one full
    # cycle of Delta, so that panel is not an estimate at all. Re-epoching from
    # continuous data is the only correct route: concatenating the 1.5 s epochs
    # would splice non-contiguous segments (they are post-rejection), and every
    # join is a step discontinuity whose broadband splatter lands hardest in
    # the low frequencies this is meant to fix.
    result["ec_raw"] = split["ec"]
    result["eo_raw"] = split["eo"]
    result["ec_epochs"], result["reject_ec"] = epoch_signal(split["ec"], return_diagnostics=True)
    result["eo_epochs"], result["reject_eo"] = epoch_signal(split["eo"], return_diagnostics=True)
    result["alpha_ratio"] = split["alpha_ratio"]
    result["eoec_ambiguous"] = split["ambiguous"]

    if vcpt_filepath:
        raw_vcpt = load_raw(vcpt_filepath)
        raw_vcpt = filter_raw(raw_vcpt)
        raw_vcpt, ica_diag_vcpt = remove_artifacts_ica(raw_vcpt)
        result["ica_vcpt"] = ica_diag_vcpt
        result["vcpt_epochs"], result["reject_vcpt"] = epoch_signal(raw_vcpt, return_diagnostics=True)
        result["vcpt_behavioral_proxy"] = extract_vcpt_behavioral_proxy(raw_vcpt)

    return result


if __name__ == "__main__":
    # Quick manual test once you point at real files, e.g.:
    # result = preprocess_subject(
    #     "F08080102-2019_09_08-10_42_29-EOEC.EDF",
    #     "F08080102-2019_09_08-10_57_53-VCPT.EDF",
    # )
    # print(result["alpha_ratio"], result["eoec_ambiguous"])
    # print(result["vcpt_behavioral_proxy"])
    pass