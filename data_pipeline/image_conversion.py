"""
Phase 1 — 1D EEG -> 2D image conversion.

Three representations, per PROJECT.md sec 4:
  1. CWT scalograms (Complex Morlet) on Fz/Cz/Pz/F3/F4 -> composite image, per epoch
  2. Topographic power heatmaps across 5 bands -> composite image, per epoch
  3. EC/EO coherence (functional connectivity) -> composite image, per subject/condition

Saved as 224x224 RGB PNGs, organized into class folders matching the
Ultralytics yolov8n-cls expected dataset layout:
  output_dir/<representation>/<split>/<class>/<filename>.png

NOTE: split ("train"/"val"/"test") is NOT assigned here — that must happen
at the SUBJECT level before this function is ever called, using subject-wise
grouped CV (see PROJECT.md sec 6/Phase 2). Do not shuffle epochs into splits
independently, or you leak the same subject's epochs across train and test.
"""

import math
import os

import matplotlib
matplotlib.use("Agg")  # no display backend needed, we're just saving files
import matplotlib.pyplot as plt
import mne
import mne_connectivity
import numpy as np
import pywt
from PIL import Image

from data_pipeline.preprocessing import CHANNELS_19  # the real 19 EEG channels —
# reused here instead of redefining, so this module can never drift out of sync
# with preprocessing.py about which channels are actually EEG vs. LABEL/dead.

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


def _make_panel_grid(n_panels: int, figsize: tuple = (10, 10)):
    """
    Lay out n_panels equal cells on a SQUARE canvas, as close to square as
    n_panels allows, and return (fig, axes) where axes is a flat list of
    exactly n_panels axes. Any leftover cells are switched off so they render
    as blank background rather than empty framed boxes.

    Why this exists rather than plt.subplots(1, n): both topomaps and
    coherence matrices are drawn with equal aspect, so the plotted content is
    a circle/square limited by the SHORTER side of its cell. A 1xN row on a
    square canvas gives each panel a cell of (canvas/N) wide by (canvas) tall,
    so the content is capped by that narrow width and the rest of the cell is
    whitespace.

    Measured for the 5 bands used here, at IMG_SIZE=224 (22.4 px per canvas
    inch):
        1x5 -> cell 2.00 x 10.00 in -> content  ~45 px, ~16% of canvas inked
        2x3 -> cell 3.33 x  5.00 in -> content  ~75 px, ~44% of canvas inked
    i.e. ~1.7x linear / ~2.8x area resolution for the same pixels and the same
    compute. (A 2x2 grid would give ~112 px, but only holds 4 panels -- that
    is a decision about DROPPING a band, not a layout change, so it is
    deliberately not done here.)

    The canvas stays square: MNE draws each head as a true circle, and
    resizing a non-square canvas down to IMG_SIZE x IMG_SIZE is what stretched
    the heads into ovals originally. Keeping the canvas square is load-bearing
    for that fix -- only the cell arrangement inside it changes.
    """
    n_rows = max(1, int(math.floor(math.sqrt(n_panels))))
    n_cols = int(math.ceil(n_panels / n_rows))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_1d(np.asarray(axes)).ravel()
    for extra in axes[n_panels:]:
        extra.axis("off")  # leftover cells: blank, not an empty framed box
    return fig, list(axes[:n_panels])


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
    # Square canvas (see _make_panel_grid): non-square canvases get stretched
    # into ovals by the final resize to IMG_SIZE x IMG_SIZE. The grid packs the
    # bands into roughly-square cells instead of one thin row, which is what
    # makes each head ~75 px instead of ~45 px for the same output size.
    fig, axes = _make_panel_grid(len(TOPOMAP_BANDS))

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
    Generate both per-epoch representations for one subject/task, save to
    disk in the class-folder layout yolov8n-cls expects:
        output_dir/<representation>/<split>/<class>/<filename>.png

    label: "ADHD" or "Control" — becomes the class folder name.
    split: "test", "fold_0".."fold_4" (or whatever data_pipeline/subject_split.py
        assigned this subject) — becomes a folder level so every image on disk
        traces back to the manifest that produced it. Get this via
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


def generate_coherence_image(epochs: mne.Epochs, ch_names: list) -> np.ndarray:
    """
    EC/EO functional connectivity representation, per PROJECT.md sec 4 step 3
    ("New: add an EC/EO coherence representation... the source paper
    specifically flags coherence as one of its five retained feature groups").

    IMPORTANT DESIGN POINT, different from scalogram/topomap: this produces
    ONE image per subject per condition (EC or EO), not one per epoch.
    Coherence needs averaging across many trials for a stable estimate --
    computing it from a single 1.5s epoch isn't meaningful. Call this once
    on the full Epochs object for a condition, not per-epoch in a loop.

    METHOD CHOICE, confirmed necessary by testing on real data: plain
    coherence ('coh') came back 0.98-0.999 across EVERY channel pair
    regardless of scalp distance, with almost no variance between bands --
    this is volume conduction / common-reference inflation, a well-documented
    EEG artifact, not real connectivity structure. It didn't improve with
    longer epochs either (ruled out small-sample bias as the cause). Switched
    to imaginary coherence ('imcoh'), which removes the zero-lag
    volume-conduction component -- confirmed on real data this produces
    genuine relative structure (values still small in absolute terms, but
    with real variation across channel pairs, which is what a CNN can
    actually learn from).
    """
    eeg_epochs = epochs.copy().pick(ch_names)  # exclude LABEL etc — confirmed necessary,
    # LABEL was riding along uncounted (20 "channels" instead of 19) before this pick was added.

    con = mne_connectivity.spectral_connectivity_epochs(
        eeg_epochs, method="imcoh", mode="multitaper",
        fmin=tuple(lo for lo, hi in TOPOMAP_BANDS.values()),
        fmax=tuple(hi for lo, hi in TOPOMAP_BANDS.values()),
        sfreq=eeg_epochs.info["sfreq"], faverage=True, verbose=False,
    )
    data = np.abs(con.get_data(output="dense"))  # imcoh can be negative; magnitude is what's meaningful

    # Same grid + square-canvas reasoning as generate_topomap_image: a 19x19
    # coherence matrix is drawn with equal aspect too, so a 1xN row wastes the
    # same ~80% of the canvas on whitespace.
    fig, axes = _make_panel_grid(len(TOPOMAP_BANDS))

    for ax, (band_name, _) in zip(axes, TOPOMAP_BANDS.items()):
        band_idx = list(TOPOMAP_BANDS.keys()).index(band_name)
        mat = data[:, :, band_idx]
        mat = mat + mat.T  # library only fills the lower triangle; mirror it for a full symmetric matrix
        # Per-band min-max normalization for visibility: imcoh magnitudes are small
        # (~0.001-0.01) and vary a lot in overall scale between subjects/bands, so a
        # fixed global vmin/vmax would wash out real structure the same way the
        # unnormalized scalogram did earlier in this project.
        vmax = mat.max() if mat.max() > 0 else 1.0
        ax.imshow(mat, cmap="viridis", vmin=0, vmax=vmax)
        ax.axis("off")

    plt.subplots_adjust(hspace=0, wspace=0.05, left=0, right=1, top=1, bottom=0)
    arr = _fig_to_rgb_array(fig)
    plt.close(fig)
    return arr


def process_coherence_for_subject(epochs_by_task: dict, subject_id: str, label: str,
                                    split: str, output_dir: str) -> dict:
    """Generate the coherence image for EC and/or EO (whichever are present in
    epochs_by_task — VCPT is deliberately excluded, coherence here is
    specifically the EC/EO resting-state representation PROJECT.md calls for).
    Returns {task: 1} for each generated, so callers can log counts the same
    way as process_epochs_to_images."""
    counts = {}
    out_dir = os.path.join(output_dir, "coherence", split, label)
    os.makedirs(out_dir, exist_ok=True)

    for task in ["EC", "EO"]:
        if task not in epochs_by_task:
            continue
        img = generate_coherence_image(epochs_by_task[task], CHANNELS_19)
        path = os.path.join(out_dir, f"{subject_id}_{task}.png")
        Image.fromarray(img).save(path)
        counts[task] = 1

    return counts


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

    Returns {task: n_epochs_processed}, plus a "coherence" key with the
    EC/EO coherence image counts.
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

    counts["coherence"] = process_coherence_for_subject(epochs_by_task, subject_id, label, split, output_dir)
    return counts