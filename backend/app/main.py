"""
FastAPI backend — Phase 5.

Deliberately minimal right now: a health check only. Don't build prediction
endpoints until Phase 2 has a working, validated model to serve — an API in
front of an unvalidated model is worse than no API, since it invites you to
skip the validation step.
"""

from fastapi import FastAPI

app = FastAPI(title="ADHD-YOLO API")


@app.get("/health")
def health():
    return {"status": "ok"}


# TODO (Phase 5, after Phase 2 baseline is validated):
# - POST /predict — accept an EEG file or precomputed image, return
#   classification + Grad-CAM overlay + biomarker values
# - Wire up the fusion meta-classifier from PROJECT.md §4 step 5
