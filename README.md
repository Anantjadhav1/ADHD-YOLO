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
| 1 — Data pipeline | ✅ Built and verified against real subjects |
| 2 — Baseline model + fusion (the make-or-break checkpoint) | 🟡 Training driver built and smoke-tested; real run pending full dataset + GPU |
| 3 — Grad-CAM + clinical-plausibility check | ⬜ Not started |
| 4 — Literature review + paper writing | ⬜ Not started |
| 5 — Backend, dashboard, AWS deployment | ⬜ Not started |

### What's built and tested

- **`data_pipeline/preprocessing.py`** — loads real `.edf` files, applies the paper's exact filter protocol (0.5–50 Hz bandpass, 45–55 Hz notch), runs ICA artifact removal, splits resting-state recordings into eyes-closed/eyes-open segments via the alpha-blocking effect.
- **`data_pipeline/image_conversion.py`** — converts EEG epochs into CWT scalograms and topographic band-power heatmaps.
- **`data_pipeline/subject_split.py`** — subject-level stratified holdout test set + stratified 5-fold CV, writing one manifest CSV every downstream script reads from, so train/test leakage is structurally prevented rather than relying on discipline.
- **`data_pipeline/build_dataset.py`** — batch driver: runs preprocessing → split lookup → image conversion across the whole cohort in one command, with per-subject error handling and an audit log (`build_log.csv`) so one bad file can't crash a 103-subject run.
- **`training/train_yolo_cls.py`** — subject-wise CV training driver for `yolov8n-cls`. Aggregates epoch-level predictions back to one prediction per subject before computing accuracy/sensitivity/specificity/AUC, since the baselines being compared against (75.8%/84.5%) are subject-level numbers.

All of the above have been **run end-to-end on real subject files**, not synthetic test data — including a full real training pass through the Ultralytics API (pretrained weights → train → predict → aggregate → metrics → CSV).

### Real problems found and solved while building this

1. **No channel position data in the raw files** — broke topomap plotting outright. Fixed by attaching a standard 10-20 montage on load.
2. **EOEC (resting-state) files have zero event markers.** Solved using the alpha-blocking effect (occipital alpha power ~2–19x higher during eyes-closed rest, confirmed on 5 real subjects) instead of needing an external timing file. Ambiguous-ratio subjects are flagged for manual QC, not silently trusted.
3. **VCPT (Go/NoGo) trigger channel doesn't encode trial conditions** — pulse count varies 100–175 across subjects with continuously-varying pulse width, consistent with a behavioral response marker, not the 4-condition stimulus code the paper describes. **True P300 latency/amplitude and per-condition behavioral features can't currently be recovered** — no companion marker file or metadata.csv exists in what was extracted. Fallback in place: fixed-window epoching for the CNN pipeline (doesn't need condition labels) + an approximate behavioral-proxy stat, both clearly flagged as approximations. Pending confirmation from the dataset's corresponding author.
4. **Scalogram images came out visually flat** — EEG's 1/f trend (theta ~20x beta power on real data) let a single global color scale drown out everything but the dominant band. Fixed with per-frequency-row normalization.
5. **Topomap heads came out as stretched ovals** — pre-resize canvas wasn't square. Fixed by keeping the composite figure square before the final resize.
6. **A float-rounding edge case** in the EC/EO crop boundary that would have crashed the full 103-subject batch.
7. **A false-positive bug in the leakage detector itself** — an earlier filename-prefixing choice broke the subject-ID parser and would have flagged every subject as leaking, blocking all training. Found by testing the safety check with a deliberately-injected real leak, not by assuming it worked.

None of this was visible from reading the paper or the dataset README alone — it surfaced only by loading and running against the actual `.edf` files.

---

## What's next

1. **Run the full pipeline on all 103 subjects** once the complete raw dataset is available locally (currently verified against 5 sample subjects).
2. **Phase 2 — the critical checkpoint:** real `yolov8n-cls` training with subject-wise 5-fold CV (30 epochs, GPU-backed), classical TBR-based feature replication, the fusion meta-classifier, and significance tests against both 75.8% and 84.5%. Nothing downstream gets built until this is done and understood.
3. **Phase 3** — Grad-CAM, EC/EO coherence channel, clinical-plausibility check against known ADHD electrode/frequency sites.
4. **Phase 4** — literature review, paper writing.
5. **Phase 5** — FastAPI backend, dashboard, AWS deployment.

---

## Repo layout
adhd-yolo/
├── PROJECT.md # full methodology, decisions, roadmap
├── PROGRESS.md # session-by-session log
├── data_pipeline/
│ ├── preprocessing.py # EEG loading, filtering, ICA, EC/EO split
│ ├── image_conversion.py # CWT scalograms + topomap generation
│ ├── subject_split.py # subject-wise train/test/CV manifest
│ └── build_dataset.py # batch driver across the whole cohort
├── training/
│ └── train_yolo_cls.py # yolov8n-cls training + subject-level evaluation
├── models/ # trained weights (gitignored — large files)
├── backend/ # FastAPI serving the model (Phase 5)
├── frontend/ # dashboard (Phase 5)
├── notebooks/ # exploratory work
├── docs/
│ └── jira_board.md # epic/story breakdown
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