"""
Phase 3 — clinical-plausibility check for Grad-CAM attention.

Per PROJECT.md sec 4 step 7: "confirm Grad-CAM attention concentrates near
known ADHD-relevant sites (frontal Fz/F3/F4 for theta/beta, Cz/Pz for P300)
before presenting it as a clinical-plausibility result." This is a sanity
check, not a metric to optimize -- if it fails, that's a real finding to
report (the model may be learning something not neurologically meaningful),
not something to quietly drop from the paper.

Relies on a structural fact about generate_scalogram_image() in
image_conversion.py: the composite image is 5 equal-height horizontal strips
stacked top-to-bottom in SCALOGRAM_CHANNELS order (Fz, Cz, Pz, F3, F4). This
mapping breaks if that image layout ever changes -- see the assertion in
compute_channel_strip_attention().
"""

import numpy as np

from data_pipeline.image_conversion import SCALOGRAM_CHANNELS, IMG_SIZE

FRONTAL_CHANNELS = ["Fz", "F3", "F4"]  # theta/beta (TBR) relevant sites, per PROJECT.md
CENTRAL_PARIETAL_CHANNELS = ["Cz", "Pz"]  # P300 relevant sites, per PROJECT.md

# Which hypothesis applies depends on the task the image came from -- P300 is
# a Go/NoGo-task-evoked ERP component, meaningless for resting EOEC images.
TASK_HYPOTHESIS = {
    "EC": "frontal",
    "EO": "frontal",
    "VCPT": "central_parietal",
}


def compute_channel_strip_attention(heatmap: np.ndarray, channel_order: list = SCALOGRAM_CHANNELS) -> dict:
    """
    heatmap: (IMG_SIZE, IMG_SIZE) float array from GradCAM.generate()['heatmap'].
    Returns {channel_name: mean_attention} by dividing the image height into
    len(channel_order) equal strips, top to bottom, matching how
    generate_scalogram_image() actually stacks channels via plt.subplots.
    """
    assert heatmap.shape == (IMG_SIZE, IMG_SIZE), \
        f"expected {(IMG_SIZE, IMG_SIZE)}, got {heatmap.shape} -- wrong image size or not a scalogram heatmap"

    n = len(channel_order)
    strip_height = IMG_SIZE // n
    result = {}
    for i, ch in enumerate(channel_order):
        strip = heatmap[i * strip_height:(i + 1) * strip_height, :]
        result[ch] = float(strip.mean())
    return result


def clinical_plausibility_score(channel_attention: dict, task: str) -> dict:
    """
    Given per-channel mean attention (from compute_channel_strip_attention)
    and which task the image came from, check whether attention concentrates
    on the neurologically-expected channels for that task.

    Returns the group means and a boolean flag -- NOT a pass/fail verdict on
    the model. A False result is a real, reportable finding (the "Interpretability
    sanity check" PROJECT.md sec 4 step 7 calls for), not a bug to hide.
    """
    hypothesis = TASK_HYPOTHESIS.get(task)
    if hypothesis is None:
        raise ValueError(f"no plausibility hypothesis defined for task {task!r}")

    frontal_vals = [v for k, v in channel_attention.items() if k in FRONTAL_CHANNELS]
    cp_vals = [v for k, v in channel_attention.items() if k in CENTRAL_PARIETAL_CHANNELS]
    other_vals = [v for k, v in channel_attention.items() if k not in FRONTAL_CHANNELS + CENTRAL_PARIETAL_CHANNELS]

    frontal_mean = float(np.mean(frontal_vals)) if frontal_vals else float("nan")
    cp_mean = float(np.mean(cp_vals)) if cp_vals else float("nan")
    other_mean = float(np.mean(other_vals)) if other_vals else float("nan")

    if hypothesis == "frontal":
        expected_dominant = frontal_mean > cp_mean
        if other_vals:  # only compare against "other" if that group actually exists --
            expected_dominant = expected_dominant and frontal_mean > other_mean
            # BUG FOUND IN TESTING: with the current SCALOGRAM_CHANNELS (Fz,Cz,Pz,F3,F4),
            # every channel is already either frontal or central_parietal, so other_vals
            # is always empty and other_mean is always NaN. Comparing anything to NaN is
            # always False in Python, which silently made this flag always False
            # regardless of the real attention pattern -- caught by testing with a
            # synthetic heatmap that SHOULD have matched, and didn't.
    else:
        expected_dominant = cp_mean > frontal_mean
        if other_vals:
            expected_dominant = expected_dominant and cp_mean > other_mean

    return {
        "task": task,
        "hypothesis": hypothesis,
        "frontal_mean": frontal_mean,
        "central_parietal_mean": cp_mean,
        "other_mean": other_mean,
        "matches_expected_site": expected_dominant,
    }


def aggregate_plausibility(per_image_scores: list) -> dict:
    """
    per_image_scores: list of dicts from clinical_plausibility_score(), across
    many test-set images. Returns the fraction where attention matched the
    expected site, broken down by task -- this fraction is what actually goes
    in the paper's interpretability section, not any single image's result.
    """
    import pandas as pd
    df = pd.DataFrame(per_image_scores)
    summary = df.groupby("task")["matches_expected_site"].agg(["mean", "count"])
    summary = summary.rename(columns={"mean": "fraction_matching_expected_site", "count": "n_images"})
    return summary.to_dict(orient="index")