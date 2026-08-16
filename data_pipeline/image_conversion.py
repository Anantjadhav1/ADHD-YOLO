"""
Phase 1 — 1D EEG -> 2D image conversion.

Two representations per epoch, per PROJECT.md sec 4:
  1. CWT scalograms (Complex Morlet) on Fz/Cz/Pz/F3/F4 -> composite image
  2. Topographic power heatmaps across 5 bands -> composite image

Both saved as 224x224 RGB PNGs, organized into class folders matching the
Ultralytics yolov8n-cls expected dataset layout:
  output_dir/<representation>/<split>/<class>/<filename>.png

NOTE: split ("train"/"val"/"test") is NOT assigned here — that must happen
at the SUBJECT level before this function is ever called, using subject-wise
grouped CV (see PROJECT.md sec 6/Phase 2). Do not shuffle epochs into splits
independently, or you leak the same subject's epochs across train and test.
"""

import os

import matplotlib
matplotlib.use("Agg")  # no display backend needed, we're just saving files
import matplotlib.pyplot as plt
import mne
import numpy as np
import pywt
from PIL import Image

IMG_SIZE = 224

SCALOGRAM_CHANNELS = ["Fz", "Cz", "Pz", "F3", "F4"]
TOPOMAP_BANDS = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 12),
    "Beta": (12, 30),
    "Gamma": (30, 50),
}
CWT_FREQ_RANGE_HZ = (0.5, 50)
CWT_N_FREQS = 40
CWT_WAVELET = "cmor1.5-1.0"


def _fig_to_rgb_array(fig) -> np.ndarray:
    """Render a matplotlib figure to an RGB numpy array, resized to IMG_SIZE."""
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    img = Image.fromarray(buf).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return np.array(img)


def generate_scalogram_image(epoch_data: np.ndarray, ch_names: list, sfreq: float) -> np.ndarray:
    """
    epoch_data: shape (n_channels, n_samples) for ONE epoch, already picked
                to the full 19-channel set (we select SCALOGRAM_CHANNELS here).
    Returns a 224x224x3 uint8 RGB array.
    """
    available = [ch for ch in SCALOGRAM_CHANNELS if ch in ch_names]
    fig, axes = plt.subplots(len(available), 1, figsize=(4, 6))
    if len(available) == 1:
        axes = [axes]

    freqs = np.linspace(CWT_FREQ_RANGE_HZ[0], CWT_FREQ_RANGE_HZ[1], CWT_N_FREQS)
    scales = pywt.frequency2scale(CWT_WAVELET, freqs / sfreq)

    for ax, ch in zip(axes, available):
        idx = ch_names.index(ch)
        signal = epoch_data[idx]
        coeffs, _ = pywt.cwt(signal, scales, CWT_WAVELET, sampling_period=1 / sfreq)
        power = np.abs(coeffs)
        # EEG power follows a 1/f trend — theta-band power is ~20x beta-band
        # power in practice, confirmed on real data. A single global color
        # scale (even log) gets dominated by the strongest band and visually
        # flattens everything else, including the beta band TBR depends on.
        # Normalize each frequency ROW independently (z-score across time) so
        # relative temporal structure is visible at every frequency, not just
        # the dominant one.
        row_mean = power.mean(axis=1, keepdims=True)
        row_std = power.std(axis=1, keepdims=True) + 1e-12
        power_norm = (power - row_mean) / row_std
        ax.imshow(power_norm, aspect="auto", cmap="viridis", origin="lower",
                  vmin=-3, vmax=3)
        ax.axis("off")

    plt.subplots_adjust(hspace=0.05, wspace=0, left=0, right=1, top=1, bottom=0)
    arr = _fig_to_rgb_array(fig)
    plt.close(fig)
    return arr


def generate_topomap_image(epoch_data: np.ndarray, info: mne.Info) -> np.ndarray:
    """
    epoch_data: shape (n_channels, n_samples) for ONE epoch, full 19-channel set.
    info: the mne.Info object from the epochs (has channel positions via montage).
    Returns a 224x224x3 uint8 RGB array, one topomap per band arranged in a row.
    """
    from scipy.signal import welch

    sfreq = info["sfreq"]
    fig, axes = plt.subplots(1, len(TOPOMAP_BANDS), figsize=(10, 10))
    # NOTE: figure must be SQUARE (not wide-and-short) even though the layout
    # is a single row — resizing a wide canvas down to a square IMG_SIZE x
    # IMG_SIZE output stretches the round head shapes into ovals. MNE already
    # draws each head as a true circle within its own axes; keeping the
    # overall canvas square is what prevents the final resize from distorting it.

    freqs, psd = welch(epoch_data, fs=sfreq, nperseg=min(256, epoch_data.shape[1]), axis=-1)

    for ax, (band_name, (lo, hi)) in zip(axes, TOPOMAP_BANDS.items()):
        band_mask = (freqs >= lo) & (freqs <= hi)
        band_power = psd[:, band_mask].mean(axis=1)
        mne.viz.plot_topomap(band_power, info, axes=ax, show=False, cmap="jet", contours=0)
        ax.set_title("")

    plt.subplots_adjust(hspace=0, wspace=0.05, left=0, right=1, top=1, bottom=0)
    arr = _fig_to_rgb_array(fig)
    plt.close(fig)
    return arr


def process_epochs_to_images(epochs: mne.Epochs, subject_id: str, label: str,
                               task: str, split: str, output_dir: str,
                               max_epochs: int | None = None):
    """
    Generate both representations for epochs from one subject/task, save to
    disk in the class-folder layout yolov8n-cls expects:
        output_dir/<representation>/<split>/<class>/<filename>.png

    label: "ADHD" or "Control" — becomes the class folder name.
    split: "test", "fold_0".."fold_4" (or whatever data_pipeline/subject_split.py
        assigned this subject) — becomes a folder level so every image on disk
        traces back to the manifest that produced it. Get this via
        subject_split.get_split_for_subject(manifest, subject_id) — see
        process_subject_from_manifest() below — never hand-assign it here,
        or you risk the exact subject-leakage bug subject_split.py exists to prevent.
    max_epochs: cap for quick testing; None processes all epochs.
    """
    ch_names = epochs.ch_names
    sfreq = epochs.info["sfreq"]
    data = epochs.get_data()  # shape (n_epochs, n_channels, n_samples)

    n = len(data) if max_epochs is None else min(max_epochs, len(data))

    for rep_name in ["scalogram", "topomap"]:
        out_dir = os.path.join(output_dir, rep_name, split, label)
        os.makedirs(out_dir, exist_ok=True)

    for i in range(n):
        epoch = data[i]

        scal_img = generate_scalogram_image(epoch, ch_names, sfreq)
        scal_path = os.path.join(output_dir, "scalogram", split, label, f"{subject_id}_{task}_{i:04d}.png")
        Image.fromarray(scal_img).save(scal_path)

        topo_img = generate_topomap_image(epoch, epochs.info)
        topo_path = os.path.join(output_dir, "topomap", split, label, f"{subject_id}_{task}_{i:04d}.png")
        Image.fromarray(topo_img).save(topo_path)

    return n


def process_subject_from_manifest(epochs_by_task: dict, subject_id: str,
                                    manifest, output_dir: str,
                                    max_epochs: int | None = None) -> dict:
    """
    Convenience wrapper that actually enforces the "split must happen at the
    subject level before this runs" rule from the module docstring, instead
    of just documenting it.

    epochs_by_task: e.g. {"EC": ec_epochs, "EO": eo_epochs, "VCPT": vcpt_epochs}
        for ONE subject, as returned by preprocessing.preprocess_subject().
    manifest: the DataFrame from subject_split.load_manifest() — split and
        label are looked up from here, never passed by hand, so there's no
        way for a typo to put a subject's images in the wrong split folder.

    Returns {task: n_epochs_processed}.
    """
    row = manifest.loc[manifest["subject_id"] == subject_id]
    if row.empty:
        raise KeyError(
            f"subject_id {subject_id!r} not found in the split manifest. "
            "Run data_pipeline/subject_split.py first and pass its output here."
        )
    split = row.iloc[0]["split"]
    label = row.iloc[0]["group"]

    counts = {}
    for task, epochs in epochs_by_task.items():
        counts[task] = process_epochs_to_images(
            epochs, subject_id, label, task, split, output_dir, max_epochs=max_epochs,
        )
    return counts