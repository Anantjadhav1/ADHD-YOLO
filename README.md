# ADHD-YOLO

YOLO-based image classification framework for pediatric ADHD screening from EEG/ERP signals — converted to 2D scalograms and topographic heatmaps, classified with `yolov8n-cls`, explained with Grad-CAM, and fused with classical EEG biomarkers. Built as both a research project (thesis-grade methodology) and an engineering portfolio piece (FastAPI backend, Docker, AWS deployment).

Full methodology, decisions, and roadmap: see `PROJECT.md`. Session-by-session log: see `PROGRESS.md`. This file is the high-level orientation — what the project is, where it stands, and what's next.

---

## What this project actually does

Takes raw EEG recordings from children (resting-state eyes-open/eyes-closed, plus a Go/NoGo attention task), converts the 1D signal into 2D image representations, and classifies ADHD vs. Control using a YOLO image classifier — with the model's reasoning made visible via Grad-CAM, and its output fused with classical hand-engineered EEG biomarkers (theta/beta ratio, etc.) for a stronger combined prediction.

**This is a decision-support research tool, not a diagnostic device.** That framing is deliberate and stated everywhere in the project — see `PROJECT.md` §2.

## Why YOLO, and why classification instead of detection

The original idea was YOLO *object detection* — drawing bounding boxes around EEG anomalies like theta bursts or P300 latency drops. That was dropped early: those boxes would have to be generated from the same threshold rules (TBR > 3.0, P300 > 380ms) the project's own biomarker engine already uses, which means the detector would just be re-learning a rule that was already written down — no independent signal, and a guaranteed first question from any reviewer. Classification (`yolov8n-cls`) + Grad-CAM for localization avoids that circularity entirely while keeping the same "is this novel" angle: YOLO-on-spectrograms exists in other domains (confirmed against sleep-apnea EEG literature) but not, as far as we found, for pediatric ADHD specifically.

## The baseline we're measuring against

Rohani et al. (2022) — the paper this dataset comes from — got **75.8% accuracy** with an SVM on 113 hand-selected features (from 826 originally extracted), and, notably, a **feature-selection + Logistic Regression combination hit 84.5%** in their own results table, though it was excluded from their headline result for clinical-interpretability reasons, not accuracy. Both numbers are real targets here — see `PROJECT.md` §5a for why the fusion model (CNN + classical biomarkers), not the CNN alone, is the highest-leverage path to beating them.

---

## Current status

| Phase | Status |
|---|---|
| 0 — Setup (repo, GitHub, Jira, Docker) | ✅ Done |
| 1 — Data pipeline | 🟡 In progress — see below |
| 2 — Baseline model + fusion (the make-or-break checkpoint) | ⬜ Not started |
| 3 — Grad-CAM + clinical-plausibility check | ⬜ Not started |
| 4 — Literature review (PRISMA) + paper writing | ⬜ Not started |
| 5 — Backend, dashboard, AWS deployment | ⬜ Not started |

### What's actually built and tested in Phase 1

- **`data_pipeline/preprocessing.py`** — loads real `.edf` files, applies the paper's exact filter protocol (0.5–50 Hz bandpass, 45–55 Hz notch), runs ICA artifact removal, and splits resting-state recordings into eyes-closed/eyes-open segments.
- **`data_pipeline/image_conversion.py`** — converts EEG epochs into the two image representations (CWT scalograms + topographic band-power heatmaps) that the classifier will train on.
- Both have been **run end-to-end on real subject files** (not synthetic test data) — one ADHD subject, one Control subject, all three recording types (resting eyes-closed, eyes-open, and the Go/NoGo task).

### Real problems found and solved while building this

The dataset didn't match the paper's description exactly, and figuring out the actual structure was most of Phase 1's real work:

1. **No channel position data in the raw files** — broke topomap plotting outright. Fixed by attaching a standard 10-20 montage on load.
2. **EOEC (resting-state) files have zero event markers** — no way to know where "eyes closed" ends and "eyes open" begins from metadata alone. Solved using the alpha-blocking effect (occipital alpha power is ~2–19x higher during eyes-closed rest, confirmed on 5 real subjects) instead of needing an external timing file.
3. **VCPT (Go/NoGo) trigger channel doesn't encode trial conditions** — pulse count varies 100–175 across subjects with continuously-varying pulse width, consistent with a behavioral response marker (e.g. button-press duration), not the 4-condition stimulus code (A-A/A-P/P-P/P-H) the paper describes. **This means true P300 latency/amplitude and per-condition behavioral features (omission/commission/reaction time) can't currently be recovered.** Checked for companion marker files (none exist) and a `metadata.csv` (not present in what was extracted — may not exist at all; pending confirmation from the dataset's corresponding author). Fallback in place: fixed-window epoching for the CNN image pipeline (doesn't need condition labels, consistent with the classification-only decision above) and an approximate behavioral-proxy summary stat, both clearly flagged as approximations in the code and in `PROJECT.md`'s limitations.
4. **Two real visualization bugs**, caught by actually looking at generated output images rather than trusting "the code ran without errors": scalogram images came out visually flat because EEG's natural 1/f power trend (theta band ~20x stronger than beta on real data) made a single global color scale drown out everything except the dominant band — fixed with per-frequency-row normalization. Topomap head shapes came out as stretched ovals because the pre-resize canvas wasn't square — fixed by keeping the composite figure square before the final resize.
5. **One float-rounding edge case** in the EC/EO crop boundary that would have crashed on the full 103-subject batch — caught by testing on real files instead of assuming the logic was correct.

None of this was visible from reading the paper or the dataset README alone — it only surfaced by loading and inspecting the actual `.edf` files.

---

## What's next

1. **Subject-wise train/val/test split** — must happen at the subject level, before any image generation at scale, or epochs from the same child leak across train and test.
2. **Run the full pipeline on all 103 subjects** once the complete raw dataset is available locally (currently working from a handful of sample subjects).
3. **Phase 2 — the critical checkpoint:** train the `yolov8n-cls` baseline with subject-wise 5-fold CV, replicate the classical TBR-based feature pipeline, build the fusion meta-classifier, and report accuracy/sensitivity/specificity/AUC with significance tests against both 75.8% and 84.5%. Nothing downstream gets built until this is done and understood.
4. **Phase 3** — Grad-CAM, EC/EO coherence channel, clinical-plausibility check against known ADHD electrode/frequency sites.
5. **Phase 4** — PRISMA-documented literature review (template already in `docs/literature_review/PRISMA_flow.md`), paper writing.
6. **Phase 5** — FastAPI backend, dashboard, AWS deployment.

---

## Repo layout
adhd-yolo/
├── PROJECT.md # full methodology, decisions, roadmap
├── PROGRESS.md # session-by-session log
├── data_pipeline/
│ ├── preprocessing.py # EEG loading, filtering, ICA, EC/EO split
│ └── image_conversion.py # CWT scalograms + topomap generation
├── models/ # trained weights (gitignored — large files)
├── backend/ # FastAPI serving the model (Phase 5)
├── frontend/ # dashboard (Phase 5)
├── notebooks/ # exploratory work
├── docs/
│ ├── jira_board.md # epic/story breakdown
│ └── literature_review/
│ └── PRISMA_flow.md # literature search log template
└── docker-compose.yml # local dev, portable to EC2 later

## Tech stack

Signal processing: MNE-Python, PyWavelets, SciPy. ML: PyTorch, Ultralytics YOLOv8/v11, scikit-learn, XGBoost. Backend: FastAPI. Containers: Docker + docker-compose. Cloud: AWS S3 + EC2 (g4dn.xlarge). Version control: GitHub with branch-per-feature + PR workflow. Project tracking: Jira.

## Setup

```powershell
git init
git checkout -b main
docker compose build
docker compose up
```

Health check: `http://localhost:8000/health`

## Workflow

`main` is always deployable — never commit directly to it. One branch per feature (`git checkout -b feature/data-conversion`), small commits, PR into `main` even solo, delete the branch after merge. See `docs/jira_board.md` for the epic breakdown.

## Known limitations (stated upfront, not hidden)

- Dataset is 103 subjects — small for a deep classifier; subject-wise validation and transfer learning are mandatory, not optional, because of this.
- P300 latency/amplitude and per-condition behavioral features are currently unavailable pending clarification from the dataset source (see "Real problems found" above).
- This is a research/decision-support tool. It does not diagnose ADHD and is not a replacement for clinical evaluation.