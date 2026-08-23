"""
FastAPI backend — Phase 5.

/predict is wired up now that inference.py exists and is tested against real
data — but it still requires a REAL trained model at MODEL_PATH (set via
env var). No trained checkpoint ships with this repo (models/ is gitignored,
and a real one doesn't exist until Phase 2's full 103-subject run happens).
Calling /predict before then returns a clear 503, not a silent bad prediction.
"""

import os
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile

from backend.app.inference import run_inference

app = FastAPI(title="ADHD-YOLO API")

MODEL_PATH = os.environ.get("ADHD_YOLO_MODEL_PATH", "models/yolov8n-cls-trained.pt")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(eoec_file: UploadFile = File(...), vcpt_file: UploadFile | None = File(None)):
    """
    Upload one subject's EOEC recording (required) and VCPT recording
    (optional) as .edf files. Returns prediction, confidence, TBR biomarkers,
    and a Grad-CAM overlay (base64 PNG) -- NOT a diagnosis, see the
    disclaimer field in the response.
    """
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(
            status_code=503,
            detail=f"No trained model available at {MODEL_PATH}. Phase 2's real training "
                    "run hasn't been completed yet — see PROJECT.md roadmap.",
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        eoec_path = os.path.join(tmpdir, eoec_file.filename)
        with open(eoec_path, "wb") as f:
            f.write(await eoec_file.read())

        vcpt_path = None
        if vcpt_file is not None:
            vcpt_path = os.path.join(tmpdir, vcpt_file.filename)
            with open(vcpt_path, "wb") as f:
                f.write(await vcpt_file.read())

        try:
            result = run_inference(eoec_path, vcpt_path, MODEL_PATH)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

    return result


# TODO (Phase 5, after Phase 2 baseline is validated):
# - Wire up the fusion meta-classifier (training/fusion_classifier.py) as an
#   optional second prediction alongside the CNN-only one above
# - Cap epochs analyzed per request for response time (currently processes
#   every epoch in the file -- 505 on a real 12-minute EOEC recording in
#   testing, which is thorough but slow for a live API)