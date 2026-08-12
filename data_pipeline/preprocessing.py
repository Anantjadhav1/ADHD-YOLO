"""
Phase 1 — EEG preprocessing.

Filter values below match the Rohani et al. (2022) protocol so results stay
comparable to the 75.8% SVM baseline. Change these only if you have a
specific, documented reason to diverge.

Pipeline: raw -> bandpass -> notch -> ICA artifact removal -> epoch slicing.
Each step is a separate function so you can inspect intermediate output in
a notebook before trusting the full pipeline on all 103 subjects.
"""

import mne


# --- Recording parameters (from the paper / dataset README) ---
SAMPLING_RATE_HZ = 500
BANDPASS_LOW_HZ = 0.5
BANDPASS_HIGH_HZ = 50.0
NOTCH_FREQ_HZ = 50.0  # 45-55 Hz notch in the paper; MNE notch_filter takes the center freq
CHANNELS_19 = [
    "Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz",
    "C3", "C4", "Cz", "T3", "T4", "T5", "T6",
    "P3", "P4", "Pz", "O1", "O2",
]
EPOCH_LENGTH_SEC = 1.5  # paper uses 1-2s sliding windows; 1.5s is a reasonable starting point


def load_raw(filepath: str) -> mne.io.Raw:
    """
    Load a raw EEG recording. Update this once you confirm the actual file
    format from the dataset (.edf is assumed here — change if it's .mat, .csv,
    or something else once you have the real files).
    """
    raw = mne.io.read_raw_edf(filepath, preload=True)
    return raw


def filter_raw(raw: mne.io.Raw) -> mne.io.Raw:
    """Zero-phase bandpass + notch filter, matching the paper's protocol."""
    raw = raw.copy().filter(
        l_freq=BANDPASS_LOW_HZ,
        h_freq=BANDPASS_HIGH_HZ,
        method="fir",
        phase="zero",
    )
    raw = raw.notch_filter(freqs=NOTCH_FREQ_HZ)
    return raw


def remove_artifacts_ica(raw: mne.io.Raw, n_components: int = 19) -> mne.io.Raw:
    """
    ICA to isolate and drop ocular (EOG) and muscular (EMG) components.

    NOTE: automatic component classification (e.g. via ICLabel) still needs
    manual spot-checking on a few subjects before trusting it on all 103 —
    don't skip that check even if it "looks fine" on the first few.
    """
    ica = mne.preprocessing.ICA(n_components=n_components, random_state=42)
    ica.fit(raw)
    # TODO once real data is loaded: identify EOG/EMG components (via
    # mne.preprocessing.ICA.find_bads_eog / find_bads_muscle, or manual
    # inspection) and exclude them before apply().
    raw_clean = ica.apply(raw.copy())
    return raw_clean


def epoch_signal(raw: mne.io.Raw, epoch_length_sec: float = EPOCH_LENGTH_SEC) -> mne.Epochs:
    """Slice continuous signal into fixed-length windows for downstream conversion."""
    events = mne.make_fixed_length_events(raw, duration=epoch_length_sec)
    epochs = mne.Epochs(raw, events, tmin=0, tmax=epoch_length_sec, baseline=None, preload=True)
    return epochs


def preprocess_subject(filepath: str) -> mne.Epochs:
    """Full pipeline for one subject, start to finish."""
    raw = load_raw(filepath)
    raw = filter_raw(raw)
    raw = remove_artifacts_ica(raw)
    epochs = epoch_signal(raw)
    return epochs


if __name__ == "__main__":
    # Quick manual test once you have a real file path to point at.
    # epochs = preprocess_subject("/path/to/one/subject.edf")
    # print(epochs)
    pass
