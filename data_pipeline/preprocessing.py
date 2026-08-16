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
    Parse the real filename convention: <ID>-<YYYY_MM_DD>-<HH_MM_SS>-<TASK>.edf
    e.g. F08080102-2019_09_08-10_42_29-EOEC.EDF
    """
    import os
    fname = os.path.basename(filepath)
    m = re.match(r"([A-Za-z0-9]+)-(\d{4}_\d{2}_\d{2})-(\d{2}_\d{2}_\d{2})-(\w+)\.edf", fname, re.IGNORECASE)
    if not m:
        raise ValueError(f"Filename doesn't match expected pattern: {fname}")
    subject_id, date, time, task = m.groups()
    group = "ADHD" if subject_id.upper().startswith("F") else "Control"
    return SubjectFile(subject_id=subject_id, group=group, date=date, time=time, task=task.upper(), filepath=filepath)


def load_raw(filepath: str) -> mne.io.Raw:
    """Load a raw recording, drop the confirmed-dead X1/X2 channels, and
    attach a standard 10-20 montage — the files themselves carry no channel
    position data, which topomap plotting needs downstream."""
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
        picks=eeg_picks, method="fir", phase="zero",
    )
    raw = raw.notch_filter(freqs=NOTCH_FREQ_HZ, picks=eeg_picks)
    return raw


def remove_artifacts_ica(raw: mne.io.Raw, n_components: int = 19) -> mne.io.Raw:
    """
    ICA on the 19 EEG channels only. No usable EOG channel exists in this
    dataset (X1/X2 confirmed dead), so ocular component identification must
    rely on frontal-channel correlation (Fp1/Fp2) or manual/automatic
    component inspection — spot-check on real subjects before trusting it
    across all 103.
    """
    eeg_picks = [ch for ch in CHANNELS_19 if ch in raw.ch_names]
    ica = mne.preprocessing.ICA(n_components=min(n_components, len(eeg_picks)), random_state=42)
    ica.fit(raw, picks=eeg_picks)
    # TODO: identify ocular components via correlation with Fp1/Fp2 (proxy
    # for EOG since no real EOG channel exists), or manual inspection on a
    # subset, before applying automatically to all subjects.
    raw_clean = ica.apply(raw.copy())
    return raw_clean


def split_eoec_by_alpha(raw: mne.io.Raw, trim_sec: float = 15.0) -> dict:
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
    half = n // 2

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

    return {"ec": ec_raw, "eo": eo_raw, "alpha_ratio": ratio, "ambiguous": ambiguous}


def epoch_signal(raw: mne.io.Raw, epoch_length_sec: float = EPOCH_LENGTH_SEC) -> mne.Epochs:
    """Fixed-length sliding-window epochs. Used for EOEC (EC/EO) and for the
    full VCPT recording, since the classification-only pipeline doesn't
    require trial-locked epochs — only Grad-CAM/interpretability would ever
    want stimulus-locked windows, and that's blocked pending trigger info."""
    events = mne.make_fixed_length_events(raw, duration=epoch_length_sec)
    epochs = mne.Epochs(raw, events, tmin=0, tmax=epoch_length_sec, baseline=None, preload=True, verbose=False)
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


def preprocess_subject(eoec_filepath: str, vcpt_filepath: str | None = None) -> dict:
    """Full pipeline for one subject's session."""
    result = {}

    raw_eoec = load_raw(eoec_filepath)
    raw_eoec = filter_raw(raw_eoec)
    raw_eoec = remove_artifacts_ica(raw_eoec)
    split = split_eoec_by_alpha(raw_eoec)
    result["ec_epochs"] = epoch_signal(split["ec"])
    result["eo_epochs"] = epoch_signal(split["eo"])
    result["alpha_ratio"] = split["alpha_ratio"]
    result["eoec_ambiguous"] = split["ambiguous"]

    if vcpt_filepath:
        raw_vcpt = load_raw(vcpt_filepath)
        raw_vcpt = filter_raw(raw_vcpt)
        raw_vcpt = remove_artifacts_ica(raw_vcpt)
        result["vcpt_epochs"] = epoch_signal(raw_vcpt)
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