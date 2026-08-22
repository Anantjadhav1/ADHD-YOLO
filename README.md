# ADHD-YOLO

YOLO-based image classification framework for pediatric ADHD screening from EEG/ERP signals — converted to 2D scalograms, topographic heatmaps, and EC/EO coherence maps, classified with `yolov8n-cls`, explained with Grad-CAM, and fused with classical EEG biomarkers (theta/beta ratio) via a logistic-regression meta-classifier. Built as both a research project (thesis-grade methodology) and an engineering portfolio piece (FastAPI backend, Docker, AWS deployment).

Full methodology, decisions, and roadmap: see `PROJECT.md`. Session-by-session log: see `PROGRESS.md`. This file is the high-level orientation — what the project is, where it stands, and what's next.

---

## What this project actually does

Takes raw EEG recordings from children (resting-state eyes-open/eyes-closed, plus a Go/NoGo attention task), converts the 1D signal into 2D image representations (scalograms, topomaps, EC/EO coherence maps), and classifies ADHD vs. Control three ways: a `yolov8n-cls` image classifier alone, classical hand-engineered biomarkers (theta/beta ratio) alone, and a fusion of both via a meta-classifier — reported side by side. The CNN's reasoning is made visible via Grad-CAM and checked against known ADHD-relevant electrode sites for clinical plausibility.

**This is a decision-support research tool, not a diagnostic device.** That framing is deliberate and stated everywhere in the project — see `PROJECT.md` §2.

## Why YOLO, and why classification instead of detection

The original idea was YOLO *object detection* — drawing bounding boxes around EEG anomalies like theta bursts or P300 latency drops. That was dropped early: those boxes would have to be generated from the same threshold rules (TBR > 3.0, P300 > 380ms) the project's own biomarker engine already uses, which means the detector would just be re-learning a rule that was already written down — no independent signal, and a guaranteed first question from any reviewer. Classification (`yolov8n-cls`) + Grad-CAM for localization avoids that circularity entirely while keeping the same "is this novel" angle: YOLO-on-spectrograms exists in other domains (confirmed against sleep-apnea EEG literature) but not, as far as we found, for pediatric ADHD specifically.

## The baseline we're measuring against

Rohani et al. (2022) — the paper this dataset comes from — got **75.8% accuracy** with an SVM on 113 hand-selected features (from 826 originally extracted), and, notably, a **feature-selection + Logistic Regression combination hit 84.5%** in their own results table, though it was excluded from their headline result for clinical-interpretability reasons, not accuracy. Both numbers are real targets here. Per `PROJECT.md` §5a, the classical-biomarker + fusion path (not the CNN alone) is the highest-leverage route to beating them — which is why `training/classical_features.py` and `training/fusion_classifier.py` exist as first-class pieces of this project, not an afterthought.

---

## Current status

| Phase | Status |
|---|---|
| 0 — Setup (repo, GitHub, Jira, Docker) | ✅ Done |
| 1 — Data pipeline | ✅ Done and verified against real subjects (3 image representations: scalogram, topomap, coherence) |
| 2 — Baseline model + classical features + fusion | 🟡 **All three pipelines built and verified end-to-end on real sample data; real full-scale run pending full dataset + GPU** |
| 3 — Grad-CAM + clinical-plausibility check | 🟡 Code written and internally tested; blocked on Phase 2's real trained model before it can be exercised for real |
| 4 — Literature review + paper writing | ⬜ Not started |
| 5 — Backend, dashboard, AWS deployment | ⬜ Not started |

### What's built and tested

- **`data_pipeline/preprocessing.py`** — loads real `.edf` files, applies the paper's exact filter protocol (0.5–50 Hz bandpass, 45–55 Hz notch), runs ICA artifact removal, splits resting-state recordings into eyes-closed/eyes-open segments via the alpha-blocking effect.
- **`data_pipeline/image_conversion.py`** — converts EEG epochs into three representations: CWT scalograms, topographic band-power heatmaps, and EC/EO functional-connectivity (imaginary coherence) maps.
- **`data_pipeline/subject_split.py`** — subject-level stratified holdout test set + stratified 5-fold CV, writing one manifest CSV every downstream script reads from, so train/test leakage is structurally prevented rather than relying on discipline.
- **`data_pipeline/build_dataset.py`** — batch driver: runs preprocessing → split lookup → image conversion across the whole cohort in one command, with per-subject error handling and an audit log (`build_log.csv`) so one bad file can't crash a 103-subject run.
- **`training/classical_features.py`** — computes theta/beta ratio (TBR) at frontal channels (F3/F4/Fz), separately for EC/EO/VCPT, matching the paper's own formula and feature-group structure. P300 and behavioral fields are present but explicitly `NaN`, not fabricated, pending the outstanding trigger-coding confirmation.
- **`data_pipeline/build_classical_features.py`** — batch driver producing one CSV row of classical features per subject, mirroring `build_dataset.py`'s manifest-driven, per-subject-error-handled structure.
- **`training/train_yolo_cls.py`** — subject-wise CV training driver for `yolov8n-cls`. Aggregates epoch-level predictions back to one prediction per subject before computing accuracy/sensitivity/specificity/AUC, since the baselines being compared against (75.8%/84.5%) are subject-level numbers.
- **`training/fusion_classifier.py`** — merges the CNN's out-of-fold subject-level probability with the classical TBR features into a logistic-regression meta-classifier, using the same CV folds and metrics function as `train_yolo_cls.py` so all three approaches (CNN alone / classical alone / fused) land in one comparable results table.
- **`interpretability/gradcam.py`** — Grad-CAM implementation hooked into the real, inspected `yolov8n-cls` architecture (final C2f block before the classify head), producing attention heatmaps and image overlays.
- **`interpretability/clinical_plausibility.py`** — checks whether Grad-CAM attention concentrates on the electrode sites known to matter for ADHD (frontal for theta/beta, central-parietal for P300), per task. Reports a real finding either way — not a metric to force-pass.

All data-pipeline, classical-feature, and training code above has been **run end-to-end on real subject files**, not synthetic test data. The interpretability code has been built and internally bug-tested but not yet run against a real trained model, since that depends on Phase 2's full run.

### Real problems found and solved while building this

1. **No channel position data in the raw files** — broke topomap plotting outright. Fixed by attaching a standard 10-20 montage on load.
2. **EOEC (resting-state) files have zero event markers.** Solved using the alpha-blocking effect (occipital alpha power ~2–19x higher during eyes-closed rest, confirmed on 5 real subjects) instead of needing an external timing file. Ambiguous-ratio subjects are flagged for manual QC, not silently trusted.
3. **VCPT (Go/NoGo) trigger channel doesn't encode trial conditions** — pulse count varies 100–175 across subjects with continuously-varying pulse width, consistent with a behavioral response marker, not the 4-condition stimulus code the paper describes. **True P300 latency/amplitude and per-condition behavioral features can't currently be recovered** — no companion marker file or metadata.csv exists in what was extracted. Fallback in place: fixed-window epoching for the CNN pipeline (doesn't need condition labels) + an approximate behavioral-proxy stat, both clearly flagged as approximations. Pending confirmation from the dataset's corresponding author.
4. **Scalogram images came out visually flat** — EEG's 1/f trend (theta ~20x beta power on real data) let a single global color scale drown out everything but the dominant band. Fixed with per-frequency-row normalization.
5. **Topomap heads came out as stretched ovals** — pre-resize canvas wasn't square. Fixed by keeping the composite figure square before the final resize.
6. **A float-rounding edge case** in the EC/EO crop boundary that would have crashed the full 103-subject batch.
7. **A false-positive bug in the leakage detector itself** — an earlier filename-prefixing choice broke the subject-ID parser and would have flagged every subject as leaking, blocking all training. Found by testing the safety check with a deliberately-injected real leak, not by assuming it worked.
8. **Plain coherence came back saturated (0.98–0.999) across every channel pair regardless of scalp distance** — a known EEG artifact (volume conduction / common-reference inflation), not real connectivity. Switched to imaginary coherence (`imcoh`), which removes the zero-lag component and shows genuine structure.
9. **LABEL channel was silently riding along as a 20th "channel"** into the coherence calculation before an explicit channel pick was added to exclude it.
10. **A NaN-comparison bug in the clinical-plausibility check** — with the current 5 scalogram channels, every channel is already frontal or central-parietal, so the "other channels" group was always empty, making a mean-of-empty-list comparison silently always False regardless of the real attention pattern. Caught by testing with a synthetic heatmap that should have passed and didn't.
11. **`NaN` is truthy in Python** — a missing `vcpt_path` in the manifest is stored as float `NaN`, not `None`, so `row.get("vcpt_path") or None` silently passed `NaN` through as a fake file path, crashing subjects that had no VCPT file with a cryptic error. Fixed with an explicit `pd.notna()` check.
12. **A column-collision bug in the fusion classifier** — the classical features CSV already carried a `split` column, and re-merging the manifest's `split` on top of it produced `split_x`/`split_y` instead of one usable `split` column, crashing with `KeyError: 'split'`. Fixed by removing the redundant merge.
13. **A single-class CV fold** — a training fold can end up with only one class present, which logistic regression can't fit. Confirmed to actually happen with a tiny sample; should be rare with the real stratified 103-subject split, but now skips that fold with a warning instead of crashing the whole run.

None of this was visible from reading the paper or the dataset README alone — it surfaced only by loading and running against the actual `.edf` files, or by deliberately testing edge cases.

### An early, informal signal (not a result)

On the 5 real sample subjects, TBR came out **consistently higher for ADHD than Control across every condition** — the correct hypothesized direction. Absolute values (9–16) run higher than typical published ranges (1.5–3.5), which is flagged as worth validating further once more subjects are available, not treated as confirmed. With only 5 subjects this is not statistically meaningful — it's a sanity check that the feature is computing something real, not a preview of the final result.

---

## What's next

1. **Run the full pipeline on all 103 subjects** once the complete raw dataset is available locally (currently verified against 5 sample subjects).
2. **Phase 2 — the critical checkpoint:** real subject-wise 5-fold CV runs (GPU-backed) for all three approaches — `yolov8n-cls` alone, classical TBR alone, and the fusion classifier — with significance tests against both 75.8% and 84.5%. Nothing downstream gets built until this is done and understood.
3. **Phase 3** — run Grad-CAM and the clinical-plausibility check against the real trained model once Phase 2 produces one.
4. **Phase 4** — literature review, paper writing.
5. **Phase 5** — FastAPI backend, dashboard, AWS deployment.

---
```
## Repo layout
adhd-yolo/
├── PROJECT.md # full methodology, decisions, roadmap
├── PROGRESS.md # session-by-session log
├── data_pipeline/
│ ├── preprocessing.py # EEG loading, filtering, ICA, EC/EO split
│ ├── image_conversion.py # CWT scalograms + topomaps + EC/EO coherence maps
│ ├── subject_split.py # subject-wise train/test/CV manifest
│ ├── build_dataset.py # batch driver: images across the whole cohort
│ └── build_classical_features.py # batch driver: classical features CSV per subject
├── training/
│ ├── classical_features.py # TBR (theta/beta ratio) biomarker computation
│ ├── train_yolo_cls.py # yolov8n-cls training + subject-level evaluation
│ └── fusion_classifier.py # CNN + classical TBR fusion meta-classifier
├── interpretability/
│ ├── gradcam.py # Grad-CAM for the trained yolov8n-cls model
│ └── clinical_plausibility.py # checks Grad-CAM attention against known ADHD sites
├── models/ # trained weights (gitignored — large files)
├── backend/ # FastAPI serving the model (Phase 5)
├── frontend/ # dashboard (Phase 5)
├── notebooks/ # exploratory work
├── docs/
│ └── jira_board.md # epic/story breakdown
└── docker-compose.yml # local dev, portable to EC2 later
```

## Tech stack

Signal processing: MNE-Python, MNE-Connectivity, PyWavelets, SciPy. ML: PyTorch, Ultralytics YOLOv8/v11, scikit-learn (Logistic Regression fusion classifier), XGBoost. Backend: FastAPI. Containers: Docker + docker-compose. Cloud: AWS S3 + EC2 (g4dn.xlarge). Version control: GitHub with branch-per-feature + PR workflow. Project tracking: Jira.

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
- The TBR pattern seen so far (ADHD > Control) is directionally correct but based on only 5 subjects — not yet a validated result.
- Grad-CAM and the clinical-plausibility check are built and internally tested but have not yet been run against a real trained model — that depends on Phase 2's real training run.
- This is a research/decision-support tool. It does not diagnose ADHD and is not a replacement for clinical evaluation.
