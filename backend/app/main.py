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
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend.app.inference import run_inference

app = FastAPI(title="ADHD-YOLO API")

# The dashboard is opened as a file:// page during demos, which sends
# Origin: null. Permissive CORS is acceptable here because this is a local
# research tool with no authentication and no stored data; it must be
# tightened before any public deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
MODEL_PATH = os.environ.get("ADHD_YOLO_MODEL_PATH", "models/yolov8n-cls-trained.pt")
REFERENCE_CSV = "docs/reference_scores.csv"
STUDY_CSV = "docs/preregistered_results.csv"
APP_DIR = Path("frontend/app/dist")   # the built React dashboard (npm run build)

@app.get("/")
def root():
    return {
        "service": "ADHD-YOLO API",
        "docs": "/docs",
        "endpoints": ["/health", "/predict", "/reference", "/study", "/app/"],
    }


@app.get("/reference")
def reference():
    """The children a live prediction is placed among: scored through this same
    /predict path by the same checkpoint, which never trained or selected on any
    of them (training/score_reference_set.py)."""
    d = pd.read_csv(REFERENCE_CSV)
    d = d[d["status"] == "ok"]
    return [{"subject_id": r["subject_id"], "group": r["group"], "split": r["split"],
             "p_adhd": float(r["p_adhd"])} for _, r in d.iterrows()]


@app.get("/study")
def study():
    """The pre-registered results exactly as committed (docs/PREREGISTRATION.md).
    NaN becomes null, which is valid JSON; H2 has no p-value by design."""
    d = pd.read_csv(STUDY_CSV)
    return d.astype(object).where(d.notna(), None).to_dict("records")

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


# Mounted last so it cannot shadow the API routes above. Absent until the
# dashboard has been built, in which case the API still runs on its own.
if APP_DIR.exists():
    app.mount("/app", StaticFiles(directory=APP_DIR, html=True), name="app")


# TODO (Phase 5, after Phase 2 baseline is validated):
# - Wire up the fusion meta-classifier (training/fusion_classifier.py) as an
#   optional second prediction alongside the CNN-only one above
# - Cap epochs analyzed per request for response time (currently processes
#   every epoch in the file -- 505 on a real 12-minute EOEC recording in
#   testing, which is thorough but slow for a live API)