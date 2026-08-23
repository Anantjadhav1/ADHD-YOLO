"""
Phase 5 — inference logic for the /predict endpoint.

Kept separate from main.py deliberately: this is plain Python, testable
directly without spinning up an HTTP server, the same way every other piece
of this project has been verified against real data before being wired in.

MODEL_PATH must point at a real trained yolov8n-cls checkpoint (Phase 2's
output). No trained model ships with this repo -- models/ is gitignored on
purpose (large binary files, and a real one doesn't exist yet pending the
full 103-subject training run). run_inference() fails loudly with a clear
error if the path doesn't exist, rather than silently falling back to
untrained ImageNet weights and returning a meaningless prediction.
"""

import base64
import io
import os

import numpy as np
from PIL import Image

from training.classical_features import compute_classical_features
from data_pipeline.image_conversion import generate_scalogram_image, SCALOGRAM_CHANNELS
from data_pipeline.preprocessing import preprocess_subject
from interpretability.gradcam import GradCAM

DISCLAIMER = (
    "This is a research decision-support output, not a medical diagnosis. "
    "It must not be used as a substitute for clinical evaluation by a qualified professional."
)


def _image_to_base64(img_array: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(img_array).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def run_inference(eoec_path: str, vcpt_path: str | None, model_path: str) -> dict:
    """
    Runs one subject's raw EDF file(s) through the full pipeline: preprocess
    -> generate scalogram images -> classify each -> average to one
    subject-level prediction -> Grad-CAM on one representative image ->
    classical TBR features. Returns a plain dict, ready to become the
    /predict endpoint's JSON response.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model at {model_path}. This project's models/ directory is "
            "gitignored and no real trained checkpoint exists yet -- Phase 2's full "
            "103-subject training run hasn't happened. Do not point this at an "
            "untrained/ImageNet-only checkpoint; the prediction would be meaningless."
        )

    from ultralytics import YOLO  # deferred import -- keeps a missing model_path
    # error fast and clear above, instead of failing on the (slower) YOLO import first
    model = YOLO(model_path)

    result = preprocess_subject(eoec_path, vcpt_path)
    epochs_by_task = {"EC": result["ec_epochs"], "EO": result["eo_epochs"]}
    if "vcpt_epochs" in result:
        epochs_by_task["VCPT"] = result["vcpt_epochs"]

    # Classify every EC/EO epoch (same representation train_yolo_cls.py trains
    # on), average to one subject-level probability -- consistent with
    # aggregate_to_subject_level()'s logic elsewhere in this project, kept
    # inline here since this is single-subject live inference, not a CV fold.
    probs_adhd = []
    last_image_path = None
    for task in ["EC", "EO"]:
        epochs = epochs_by_task[task]
        ch_names = epochs.ch_names
        sfreq = epochs.info["sfreq"]
        data = epochs.get_data()
        for i in range(len(data)):
            img = generate_scalogram_image(data[i], ch_names, sfreq)
            tmp_path = f"/tmp/_infer_{task}_{i}.png"
            Image.fromarray(img).save(tmp_path)
            pred = model.predict(source=tmp_path, verbose=False)[0]
            names = pred.names
            adhd_idx = [k for k, v in names.items() if v == "ADHD"][0]
            probs_adhd.append(float(pred.probs.data[adhd_idx]))
            last_image_path = tmp_path  # keep the most recent for Grad-CAM below

    if not probs_adhd:
        raise ValueError("No epochs were generated from the provided EEG file -- check the recording length/quality.")

    mean_prob_adhd = float(np.mean(probs_adhd))
    predicted_class = "ADHD" if mean_prob_adhd >= 0.5 else "Control"

    gradcam = GradCAM(model)
    gc_result = gradcam.generate(last_image_path, target_class=predicted_class)

    classical = compute_classical_features(epochs_by_task)
    # NaN is not valid JSON (json.dumps with allow_nan=False, which Starlette's
    # JSONResponse uses, raises ValueError) -- found by testing an actual HTTP
    # request, not by testing run_inference() as a plain Python function, where
    # a NaN float is perfectly valid and this bug stays invisible. Since
    # p300/behavioral fields are CURRENTLY ALWAYS NaN (see classical_features.py),
    # every single /predict request would 500 without this fix. None -> JSON
    # null is also the semantically correct representation of "not available"
    # anyway, more so than a non-standard NaN token.
    classical = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in classical.items()}

    return {
        "predicted_class": predicted_class,
        "confidence": mean_prob_adhd if predicted_class == "ADHD" else 1 - mean_prob_adhd,
        "n_epochs_analyzed": len(probs_adhd),
        "classical_biomarkers": classical,
        "gradcam_overlay_png_base64": _image_to_base64(gc_result["overlay"]),
        "disclaimer": DISCLAIMER,
    }