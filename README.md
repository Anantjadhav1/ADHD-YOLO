# ADHD-YOLO

YOLO-based image classification framework for pediatric ADHD screening from EEG/ERP signals — converted to 2D scalograms, topographic heatmaps, and EC/EO coherence maps, classified with `yolov8n-cls`, explained with Grad-CAM, and fused with classical EEG biomarkers via a logistic-regression meta-classifier. Built as both a research project (thesis-grade methodology) and an engineering portfolio piece (FastAPI backend, Docker, AWS deployment).

Full methodology, decisions, and roadmap: see `PROJECT.md`. Session-by-session log: see `PROGRESS.md`. Background science — EEG, spectral analysis, the ADHD biomarker literature, evaluation methodology: see `docs/STUDY_GUIDE.md`. This file is the high-level orientation — what the project is, where it stands, and what's next.

> **`PROJECT.md` is currently broken.** The file in the repo is a truncated copy of `PROGRESS.md`, not the methodology document. Every module cites sections of it (`sec 4 step 3`, `sec 5a`, `sec 6 Phase 2`) that don't exist in the file as committed. Recover it from git history before relying on any cross-reference. Tracked in `PROGRESS.md` (2026-08-23).

---

## What this project actually does

Takes raw EEG recordings from children (resting-state eyes-open/eyes-closed, plus a Go/NoGo attention task), converts the 1D signal into 2D image representations (scalograms, topomaps, EC/EO coherence maps), and classifies ADHD vs. Control three ways: a `yolov8n-cls` image classifier alone, classical hand-engineered biomarkers alone, and a fusion of both via a meta-classifier — reported side by side. The CNN's reasoning is made visible via Grad-CAM and checked against known ADHD-relevant electrode sites for clinical plausibility.

**This is a decision-support research tool, not a diagnostic device.** That framing is deliberate and stated everywhere in the project.

## Why YOLO, and why classification instead of detection

The original idea was YOLO *object detection* — drawing bounding boxes around EEG anomalies like theta bursts or P300 latency drops. That was dropped early: those boxes would have to be generated from the same threshold rules (TBR > 3.0, P300 > 380ms) the project's own biomarker engine already uses, which means the detector would just be re-learning a rule that was already written down — no independent signal, and a guaranteed first question from any reviewer. Classification (`yolov8n-cls`) + Grad-CAM for localization avoids that circularity entirely while keeping the same "is this novel" angle: YOLO-on-spectrograms exists in other domains (confirmed against sleep-apnea EEG literature) but not, as far as we found, for pediatric ADHD specifically.

## The baseline we're measuring against

Rohani et al. (2022) — the paper this dataset comes from — got **75.8% accuracy** with an SVM on 113 hand-selected features (from 826 originally extracted), and a **feature-selection + Logistic Regression combination hit 84.5%** in their own results table, though it was excluded from their headline result for clinical-interpretability reasons, not accuracy. Both numbers are real targets here.

**Note the feature count: 113, not 3.** The project's working assumption has been that the classical-biomarker + fusion path is the highest-leverage route to beating them. As of 2026-08-23 that path is carrying only three real features (TBR per condition), and those three do not separate the groups (see below). The assumption isn't refuted, but it now depends on expanding the feature set rather than on TBR alone.

---

## Current status

| Phase | Status |
|---|---|
| 0 — Setup (repo, GitHub, Jira, Docker) | ✅ Done |
| 1 — Data pipeline | 🟡 Built and verified on real subjects, but three correctness issues found on 2026-08-23 (see below) must be resolved before generating images at scale |
| 2 — Baseline model + classical features + fusion | 🟡 All three pipelines built; **CNN↔fusion plumbing is missing**; classical half currently at chance; real full-scale run pending |
| 3 — Grad-CAM + clinical-plausibility check | 🟡 Code written and internally tested; blocked on Phase 2's real trained model |
| 4 — Literature review + paper writing | ⬜ Not started |
| 5 — Backend, dashboard, AWS deployment | 🟡 `/predict` endpoint built and tested against a toy model; everything else not started |

**Full 103-subject dataset located** (`D:\ADHD-Faezeh Rohani-edf\edf (all)\`, 109 EOEC files) — the single biggest blocker since mid-August is resolved. Two discovery issues need clearing first: the 109-vs-103 file count discrepancy, and case-insensitive globbing on Windows. Both trip `discover_subjects`'s duplicate-ID guard.

### What's built

- **`data_pipeline/preprocessing.py`** — loads real `.edf` files, applies the paper's filter protocol (0.5–50 Hz bandpass, 45–55 Hz notch), splits resting-state recordings into eyes-closed/eyes-open via the alpha-blocking effect. *(ICA function present but currently a no-op — see below.)*
- **`data_pipeline/image_conversion.py`** — CWT scalograms, topographic band-power heatmaps, EC/EO imaginary-coherence maps.
- **`data_pipeline/subject_split.py`** — subject-level stratified holdout test set + stratified 5-fold CV, writing one manifest CSV every downstream script reads from, so train/test leakage is structurally prevented rather than relying on discipline.
- **`data_pipeline/build_dataset.py`** — batch driver with per-subject error handling and an audit log.
- **`training/classical_features.py`** — theta/beta ratio at frontal channels, separately for EC/EO/VCPT. P300 and behavioral fields explicitly `NaN`, not fabricated.
- **`data_pipeline/build_classical_features.py`** — batch driver, one CSV row per subject.
- **`training/train_yolo_cls.py`** — subject-wise CV training driver. Aggregates epoch-level predictions to one prediction per subject before computing metrics, since the baselines being compared against are subject-level numbers.
- **`training/fusion_classifier.py`** — merges the CNN's subject-level probability with classical features into a logistic-regression meta-classifier.
- **`training/significance_test.py`** — bootstrap CI on subject-level accuracy vs. the 75.8%/84.5% baselines.
- **`training/verify_tbr.py`** — read-only diagnostic comparing four TBR computation variants for both magnitude and group separation. Changes no pipeline code.
- **`interpretability/gradcam.py`**, **`interpretability/clinical_plausibility.py`** — Grad-CAM hooked into the inspected `yolov8n-cls` architecture, plus an attention-vs-known-ADHD-sites sanity check.

---

## Open correctness issues

These were found by running code against real data, not by reading it. All are unresolved as of this commit.

1. **`remove_artifacts_ica()` is a no-op.** `ica.apply()` is called with an empty `exclude` list, which reconstructs the signal bit-identically. **There is currently zero artifact rejection anywhere in the pipeline** — no ICA rejection, no epoch amplitude threshold — while costing a full ICA fit per recording. Every image and every feature produced so far comes from unrejected data.
2. **TBR is computed on 1.5 s epochs, which physically caps frequency resolution** at 0.67 Hz. TBR is a subject-level summary and should be computed on the continuous segment instead, which would give both finer resolution and more averaging. *(The band-power units bug in the same function is now fixed — see below.)*
4. **Missing Phase 2 plumbing.** `train_yolo_cls.run_cv()` computes per-subject out-of-fold probabilities and discards them, but `fusion_classifier.run_fusion_cv()` requires them as input. No code path connects the two halves.
5. **The held-out test split is never evaluated.** `run_cv` filters it out and no `evaluate_on_test()` exists — the two-stage design in `subject_split.py` has no consumer.
6. **CV accuracy will be optimistically biased.** Ultralytics selects `best.pt` by accuracy on the val fold, and evaluation then scores on that same fold.
7. **Windows discovery bugs.** `discover_subjects` globs both `.edf` and `.EDF`; on a case-insensitive filesystem both match the same files, producing a duplicate for every subject and tripping the duplicate-ID guard.

## Real problems found and solved

1. **No channel position data in the raw files** — broke topomap plotting. Fixed by attaching a standard 10-20 montage on load.
2. **EOEC files have zero event markers.** Solved via the alpha-blocking effect instead of an external timing file. Ambiguous-ratio subjects are flagged for manual QC, not silently trusted.
3. **VCPT trigger channel doesn't encode trial conditions** — pulse count varies 100–175 across subjects with continuously-varying pulse width, consistent with a behavioral response marker, not the 4-condition stimulus code the paper describes. **True P300 latency/amplitude and per-condition behavioral features can't currently be recovered.** Pending confirmation from the dataset's corresponding author — outstanding since mid-August.
4. **Scalogram images came out visually flat** — EEG's 1/f trend let one band dominate a global color scale. Fixed with per-frequency-row normalization. *(Caveat: this fix removes absolute band-power information from the images — see "Known limitations".)*
5. **Topomap heads came out as stretched ovals** — pre-resize canvas wasn't square.
6. **A float-rounding edge case** in the EC/EO crop boundary that would have crashed the full batch.
7. **A false-positive bug in the leakage detector itself** — filename prefixing broke the subject-ID parser and would have flagged every subject as leaking. Found by testing the check with a deliberately-injected real leak.
8. **Plain coherence saturated at 0.98–0.999 across every channel pair** — volume conduction, not real connectivity. Switched to imaginary coherence.
9. **LABEL channel was silently riding along as a 20th "channel"** into the coherence calculation.
10. **A NaN-comparison bug in the clinical-plausibility check** made its pass/fail flag always `False` regardless of the real attention pattern.
11. **`NaN` is truthy in Python** — a missing `vcpt_path` is stored as float `NaN`, so `row.get(...) or None` passed it through as a fake path.
12. **A column-collision bug in the fusion classifier** — re-merging `split` produced `split_x`/`split_y`.
13. **A single-class CV fold** crashed logistic regression; now skipped with a warning.
14. **`NaN` is not valid JSON** — would have made every `/predict` request return 500. Found via a real HTTP request through `TestClient`, not by calling the Python function directly.
15. **Band power was computed as the mean of the PSD, not its integral** — inflating TBR by 4.88× on real subjects. Fixed with `np.trapezoid` (note: `np.trapz` was *removed* in NumPy 2.0, so the naive fix raises `AttributeError` on this environment).
16. **`parse_filename`'s regex matched zero real files** — it required underscores in the date/time; the dataset uses dots. Would have raised on the first file of the 103-subject run. The docstring documented the wrong convention, which is how it survived. Fixed to accept `[._-]`.

None of this was visible from reading the paper or the dataset README — it surfaced only by loading and running against the actual `.edf` files, or by deliberately testing edge cases.

---

## Findings on the classical biomarker (2026-08-23)

Tested on **20 subjects (10 ADHD / 10 Control)**, not the 5 used previously.

**The TBR magnitude anomaly was a units bug, and it's resolved.** Band power was being computed as the *mean* of the PSD across each band — average spectral density, not power. Theta spans 4 Hz and beta spans 18 Hz, so the ratio was inflated by ~4.5×. Measured inflation on real subjects: **4.88×**. Corrected group means land inside published ranges (EC 2.66/2.87, EO 1.92/1.95, VCPT 2.48/2.41 for ADHD/Control).

**TBR does not separate ADHD from Control at this sample size.** Bootstrapped subject-level AUC:

| Condition | AUC | 95% CI | Direction |
|---|---|---|---|
| EC | 0.430 | [0.18, 0.70] | **reversed** — Control higher than ADHD |
| EO | 0.490 | [0.24, 0.75] | none |
| VCPT | 0.550 | [0.28, 0.80] | correct but negligible |

All three CIs contain 0.5. **This overturns the earlier 5-subject observation** that TBR was "consistently higher for ADHD than Control across every condition" — that was noise, and it was reported as an encouraging signal in two prior documents. Correcting the units does not change separation (AUC moves 0.420 → 0.430 on EC), so the fix is about correctness and comparability to literature, not accuracy.

**This replicates the current scientific consensus.** A literature review (2026-08-24) found the negative result is well established, not an anomaly of this dataset:

- **Arns, Conners & Kraemer (2013)** — meta-analysis of 9 studies (1253 ADHD / 517 non-ADHD). The group difference *shrank across publication years*, because TBR was rising in the **control** groups. Concluded TBR is not a reliable diagnostic measure.
- **(2020, iSPOT-A/ICAN)** — five different spectral-analysis algorithms computed TBR across two multi-centre clinical datasets. They produced significantly different values, and **none distinguished ADHD from controls.** This is essentially the experiment `verify_tbr.py` runs, published on far more subjects.
- **Arns et al. (2024)**, N=417 — "TBR has no diagnostic value for ADHD."
- **(2026) eLife multiverse analysis**, N=1499+381 — identifies **individual alpha peak frequency and aperiodic neural activity** as the mechanisms that limit TBR's value.
- **Coolidge et al. (2007)** — separating ADHD from *other* psychological disorders: sensitivity 50%, specificity 36%.

Caveats in both directions: n=20 is small and the CIs are wide, so this is *no evidence of separation* rather than *evidence of no separation*.

**What this means for the project.** The negative result is a finding with citations, not an absence of results — and it sharpens the comparison rather than weakening it. If the CNN succeeds where the classical marker fails, the interesting question becomes *what it is seeing*, which is exactly what Grad-CAM and `clinical_plausibility.py` are built to answer. The literature also names two specific confounds that point directly at better features: **aperiodic exponent/offset** (a slope difference shifts every band-power measure, and TBR is maximally sensitive since theta and beta sit at opposite ends) and **individual alpha peak frequency** (a child with IAF at 7-8 Hz has genuine alpha power inside the theta window). Both are computable from data already in hand.

**A useful side-effect:** EC > EO holds in 15/20 subjects, the physiologically expected direction. The five that don't (`F09081100`, `F09101156`, `F10011103`, `C10011101`, `C10020106`) are candidates for a flipped EC/EO assignment — suggesting a cheap QC rule where TBR direction must agree with `alpha_ratio`.

**Measured for Phase 2 planning:** VCPT accounts for **65% of all epoch-images** (~870–920 per subject, vs ~200–370 each for EC and EO). Training on all three conditions mixed means two thirds of the training signal comes from one condition.

---

## What's next

1. **Clear the discovery blockers** — resolve the 109-vs-103 file count and the case-insensitive glob duplicate, so `subject_split.py` can run on the real cohort.
2. **Recover `PROJECT.md`** from git history — it holds the locked design decisions and limitations the paper depends on.
3. **Fix the pipeline correctness issues** — ICA no-op, TBR units + continuous-signal PSD, scalogram normalization — *before* generating images at scale, since each one invalidates already-generated output.
4. **Expand the classical feature set** — relative band power per channel per condition, aperiodic exponent/offset, individual alpha peak frequency, frontal alpha asymmetry, coherence summaries. Now critical path.
5. **Close the Phase 2 plumbing gaps** — save out-of-fold CNN probabilities, write `evaluate_on_test()`, add an inner validation split so epoch selection stops leaking.
6. **Phase 2 — the critical checkpoint:** real subject-wise 5-fold CV for all three approaches, with significance tests against 75.8% and 84.5%.
7. **Phase 3** — Grad-CAM and clinical-plausibility against the real trained model.
8. **Phase 4** — literature review, paper writing.
9. **Phase 5** — dashboard, AWS deployment.

---

```
## Repo layout
adhd-yolo/
├── PROJECT.md                       # full methodology, decisions, roadmap  [BROKEN — see note at top]
├── PROGRESS.md                      # session-by-session log
├── data_pipeline/
│   ├── preprocessing.py             # EEG loading, filtering, ICA, EC/EO split
│   ├── image_conversion.py          # CWT scalograms + topomaps + EC/EO coherence maps
│   ├── subject_split.py             # subject-wise train/test/CV manifest
│   ├── build_dataset.py             # batch driver: images across the whole cohort
│   └── build_classical_features.py  # batch driver: classical features CSV per subject
├── training/
│   ├── classical_features.py        # TBR (theta/beta ratio) biomarker computation
│   ├── verify_tbr.py                # diagnostic: TBR variant comparison (read-only)
│   ├── train_yolo_cls.py            # yolov8n-cls training + subject-level evaluation
│   ├── fusion_classifier.py         # CNN + classical fusion meta-classifier
│   └── significance_test.py         # bootstrap CI vs. published baselines
├── interpretability/
│   ├── gradcam.py                   # Grad-CAM for the trained yolov8n-cls model
│   └── clinical_plausibility.py     # checks Grad-CAM attention against known ADHD sites
├── models/                          # trained weights (gitignored — large files)
├── backend/                         # FastAPI serving the model (Phase 5)
├── frontend/                        # dashboard (Phase 5)
├── notebooks/                       # exploratory work
├── docs/
│   ├── STUDY_GUIDE.md               # the science: EEG, spectral analysis, ML, stats
│   └── jira_board.md                # epic/story breakdown
└── docker-compose.yml               # local dev, portable to EC2 later
```

## Tech stack

Signal processing: MNE-Python, MNE-Connectivity, PyWavelets, SciPy. ML: PyTorch, Ultralytics YOLOv8/v11, scikit-learn, XGBoost. Backend: FastAPI. Containers: Docker + docker-compose. Cloud: AWS S3 + EC2 (g4dn.xlarge). Version control: GitHub with branch-per-feature + PR workflow. Project tracking: Jira.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
```

Or via Docker:

```powershell
docker compose build
docker compose up
```

Health check: `http://localhost:8000/health`

**Windows note:** if `python`/`python3` resolve to an MSYS2 install (`C:\msys64\...`), packages installed by `pip` won't be visible. Use `py -m ...` or activate a venv. `where` in PowerShell is an alias for `Where-Object` — use `Get-Command python -All` to inspect.

**Import note:** `data_pipeline` and `training` scripts use package-relative imports and must be run as `python -m data_pipeline.<script>` from the repo root, not `python data_pipeline/<script>.py`.

## Workflow

`main` is always deployable — never commit directly to it. One branch per feature, small commits, PR into `main` even solo, delete the branch after merge. See `docs/jira_board.md` for the epic breakdown.

## Known limitations (stated upfront, not hidden)

- Dataset is ~103 subjects — small for a deep classifier; subject-wise validation and transfer learning are mandatory, not optional.
- **The classical biomarker currently at the centre of the fusion design (TBR) does not separate the groups** on the 20 subjects tested (all AUC CIs contain 0.5). Sample size is small, so this is absence of evidence rather than evidence of absence — but the fusion path needs more than three features to be viable.
- **There is currently no artifact rejection in the pipeline** — the ICA step is a no-op and no epoch amplitude threshold is applied. All results and images to date come from unrejected data. This is a *named threat to validity* for this population specifically: the literature reports that children with ADHD move more than controls, so artifact contamination is differential between the groups rather than random.
- P300 latency/amplitude and per-condition behavioral features are unavailable pending clarification from the dataset source.
- **The per-frequency-row normalization applied to scalograms removes absolute band-power relationships**, which means the theta/beta ratio is not recoverable from the scalogram images by design. This was a fix for a visual problem that has a signal-content cost, and needs revisiting.
- Grad-CAM and the clinical-plausibility check have not been run against a real trained model.
- No Phase 2 result exists yet. All accuracy figures in this repo are from toy smoke tests and are not results.
- This is a research/decision-support tool. It does not diagnose ADHD and is not a replacement for clinical evaluation.