# Progress log

Update this at the end of every working session — a few lines is enough. This is what lets a new chat pick up exactly where you left off.

## Format
YYYY-MM-DD
What was done
What broke / what you learned
Next step


---

### 2026-08-12
- Project scoped and PROJECT.md written: classification-only + Grad-CAM (no bounding-box detection), subject-wise CV, transfer learning plan, biomarker fusion layer, PRISMA lit review plan.
- Decisions locked: local dev with Docker from day one, classification over detection.
- Repo skeleton created (this commit).
- **Next:** get raw dataset from IEEE DataPort / your copy, confirm file format (.edf/.mat/.csv), start `data_pipeline/preprocessing.py` on a handful of subjects.

### 2026-08-1x (backfilled — Phase 1 preprocessing + image conversion)
- `data_pipeline/preprocessing.py` and `data_pipeline/image_conversion.py` built and run end-to-end on real subject files (one ADHD, one Control, all three recording types: EC/EO resting, Go/NoGo VCPT) — not synthetic test data.
- Real problems found by inspecting actual files/output, not by assuming the code worked:
  1. Raw files carry no channel position data — broke topomap plotting. Fixed by attaching a standard 10-20 montage on load.
  2. EOEC files have zero event markers — no metadata way to locate the EC/EO boundary. Solved via the alpha-blocking effect (occipital alpha power ~2-19x higher eyes-closed, confirmed on 5 real subjects); ambiguous-ratio subjects are flagged for manual QC rather than trusted silently.
  3. VCPT LABEL channel doesn't encode the paper's 4 trial conditions — pulse count varies 100-175/subject with continuous pulse width, consistent with a behavioral response marker, not a stimulus code. True P300 latency/amplitude and per-condition behavioral features are **not currently recoverable**. No companion marker file or metadata.csv found. Fallback: fixed-window epoching (doesn't need condition labels) + an approximate behavioral-proxy summary stat, both flagged as approximations in code and in PROJECT.md limitations. **Still pending: confirmation from the dataset's corresponding author.**
  4. Scalogram images came out visually flat — EEG's 1/f trend (theta ~20x beta) let a single global color scale drown out everything but the dominant band. Fixed with per-frequency-row normalization.
  5. Topomap heads came out as stretched ovals — pre-resize canvas wasn't square. Fixed by keeping the composite figure square before the final resize.
  6. One float-rounding edge case in the EC/EO crop boundary that would have crashed on the full 103-subject batch — caught by testing on real files, not by code review alone.
- **Next:** subject-wise train/val/test split, before generating images at scale (epochs from the same child must not leak across train/test).

### 2026-08-16
- Built and tested `data_pipeline/subject_split.py`: stratified subject-level holdout test set + stratified 5-fold CV over the rest, writing one manifest CSV (`subject_id, group, split, eoec_path, vcpt_path`) that downstream scripts read from instead of re-deriving splits.
  - Tested: dry-run on synthetic subjects (49 ADHD / 54 Control, matching the real cohort balance); real-filename discovery against mixed-case `.edf`/`.EDF` files with/without a matching VCPT file; duplicate `subject_id` and too-few-subjects-for-n_folds both raise instead of silently producing a bad split; manifest round-trips through CSV; unknown-subject lookup raises `KeyError`.
- **Bug found in `image_conversion.py`:** its own docstring documented the output layout as `output_dir/<representation>/<split>/<class>/<filename>.png`, but `process_epochs_to_images()` never actually wrote a `<split>` folder — every image landed in `output_dir/<representation>/<class>/...` regardless of train/val/test/fold. (Not a surprise — `subject_split.py` didn't exist yet when this was written, so there was nothing to wire it to.)
  - Fixed: `process_epochs_to_images()` now takes a required `split` argument and uses it in the path. Added `process_subject_from_manifest()`, which looks up a subject's `split` and `group` directly from the `subject_split.py` manifest, so callers can't hand-type the wrong split and silently reintroduce leakage.
  - Verified end-to-end with synthetic EEG → epochs → manifest lookup → saved PNGs: confirmed on disk as `scalogram/fold_2/ADHD/...` and `topomap/fold_2/ADHD/...`; visually checked both image types render correctly (5 distinct scalogram rows, round topomap heads, not stretched).
- **Next:** run the full pipeline (preprocessing → split → image_conversion) on all 103 subjects once the complete raw dataset is available locally. Then Phase 2: `yolov8n-cls` baseline + classical TBR/biomarker replication + fusion meta-classifier, subject-wise 5-fold CV, report accuracy/sensitivity/specificity/AUC with significance tests against 75.8% and 84.5%.

### 2026-08-16 (cont'd) — batch pipeline driver
- Built `data_pipeline/build_dataset.py`: the missing piece that actually runs preprocessing → subject_split manifest lookup → image_conversion across a whole cohort in one command, instead of each module only working in isolation.
  - Reads an existing manifest, never regenerates one (regenerating on every batch run would silently reshuffle folds and invalidate anything already computed against the old split).
  - Respects the "ambiguous EC/EO split → don't silently trust it" rule already stated in `preprocessing.py`: ambiguous subjects are skipped by default and logged with their `alpha_ratio`, not force-processed. `--include-ambiguous` overrides per-run, `--subjects <id>` re-runs a specific subset.
  - One subject failing doesn't crash the batch — every subject wrapped in try/except, real exception text logged to `build_log.csv`, not swallowed.
- Tested against real synthetic `.edf` files on disk (not just in-memory objects) covering: a clean subject with a VCPT file, a clean subject without one, a genuinely ambiguous subject (alpha_ratio ≈ 1.01, confirmed flagged and skipped), an unknown subject_id (confirmed logged as failed, batch continued), and the `--include-ambiguous` override (confirmed it forces processing when asked). Verified output images land under the correct `<representation>/<split>/<label>/` path.
- **Gotcha found and documented:** `subject_split.py` and `build_dataset.py` both do package-relative imports (`from data_pipeline import ...`), so they must be run as `python3 -m data_pipeline.<script>` from the repo root — `python3 data_pipeline/<script>.py` breaks with `ModuleNotFoundError: No module named 'data_pipeline'` as soon as they import each other. Worth remembering before running this on the real 103-subject batch.
- **Next:** once the real dataset is available, run `subject_split.py` for the real manifest, then `build_dataset.py` on all 103 subjects. Review anyone flagged `skipped_ambiguous` manually before deciding whether to re-run them with `--include-ambiguous`.

### 2026-08-17
- Verified the uploaded repo state end-to-end: ran `subject_split.py` and `build_dataset.py` for real against all 5 sample subjects (not synthetic) — 5/5 processed clean, correct fold/class folder structure, VCPT-less subjects handled correctly.
- Built `training/train_yolo_cls.py` — subject-wise CV training driver for `yolov8n-cls`. Key design point: images are per-epoch but 75.8%/84.5% are per-subject accuracy, so evaluation aggregates epoch-level predictions back to one prediction per subject (mean probability) before computing metrics — implemented in `aggregate_to_subject_level()`, not left as a manual step.
- Found and fixed a real bug via testing, not code review: the fold-assembly step prefixed symlinked filenames with the fold name for traceability, which broke the subject-ID parser and made the leakage-check flag every subject as leaking (a false positive that would have blocked all training). Fixed by dropping the unnecessary prefix — filenames are already unique across folds since each subject belongs to exactly one fold.
- Verified the leakage check works both ways: confirmed it passes clean on a correct fold assembly, and confirmed it correctly catches a deliberately-injected real leak (same subject's file symlinked into both train and val).
- Ran one real (if statistically meaningless — 1 epoch, ~10 images, one fold with a single subject) end-to-end training pass against real generated images: confirmed the full chain (YOLO pretrained weights load → train → predict → aggregate → metrics → CSV) runs without crashing. Fixed one cosmetic bug (Ultralytics nesting output under its own default `runs/classify/` instead of the given path) by resolving to an absolute path.
- **Next:** once the full 103-subject dataset and a GPU-backed machine are available, run `subject_split.py` for the real manifest (5-fold default), `build_dataset.py` on all subjects, then `train_yolo_cls.py` for a real Phase 2 baseline result — this is the make-or-break checkpoint per PROJECT.md.

### 2026-08-1x (backfilled — EC/EO coherence representation)
- Extended `data_pipeline/image_conversion.py` with a third image representation: EC/EO functional connectivity (coherence) maps, per PROJECT.md sec 4 step 3 — the source paper specifically flags coherence as one of its five retained feature groups.
  - Design point: unlike scalogram/topomap, this is computed once per subject per condition (EC or EO) across the whole `Epochs` object, not per individual epoch — coherence needs averaging across many trials for a stable estimate.
  - **Real problem found:** plain coherence (`'coh'`) came back saturated at 0.98–0.999 across every channel pair regardless of scalp distance, with almost no variance between bands. Confirmed this is volume conduction / common-reference inflation (a known EEG artifact), not real connectivity — ruled out small-sample bias by testing with longer epochs too. Fixed by switching to imaginary coherence (`'imcoh'`), which removes the zero-lag component and shows genuine variation across channel pairs.
  - **Second bug found:** the LABEL channel was being silently included as a 20th "channel" in the coherence calculation before an explicit `.pick(CHANNELS_19)` was added to exclude it.
  - `image_conversion.py` now imports `CHANNELS_19` directly from `preprocessing.py` instead of redefining the channel list, so the two modules can't drift out of sync about which channels are real EEG.
- Updated `backend/requirements.txt`: added `mne-connectivity` (coherence computation) and `xgboost` (planned for the fusion meta-classifier in Phase 2).
- **Next:** re-run `build_dataset.py` on the 5 sample subjects to confirm the new coherence images generate correctly end-to-end alongside the existing scalogram/topomap outputs, before scaling to the full cohort.

### 2026-08-1x (backfilled — interpretability scaffold)
- Built `interpretability/gradcam.py`: Grad-CAM for the trained `yolov8n-cls` model. Hooks into `model.model.model[8]` — confirmed by inspecting the real loaded model architecture (not assumed from documentation) — the final C2f block, immediately before the Classify head at index `[9]`.
  - Confirmed by inspecting real model output that the forward pass returns a `(probs, logits)` tuple, not a plain tensor. Backprop is done on the raw logit rather than the softmaxed probability, since backpropping through softmax risks flattened gradients once a class is already confident — verified `probs[i] == softmax(logits)[i]` on real output before relying on this.
- Built `interpretability/clinical_plausibility.py`: checks whether Grad-CAM attention concentrates on the electrode sites known to matter for ADHD (frontal Fz/F3/F4 for theta/beta, central-parietal Cz/Pz for P300), per PROJECT.md sec 4 step 7. Designed as a sanity check that reports a real finding either way, not a metric to force-pass.
  - **Bug found in testing:** with the current 5 scalogram channels (Fz, Cz, Pz, F3, F4), every channel is already either "frontal" or "central-parietal," so the "other channels" comparison group is always empty. Comparing against an empty group's mean (`NaN`) is always `False` in Python, which silently made the plausibility flag always `False` regardless of the actual attention pattern. Caught by testing with a synthetic heatmap that should have passed and didn't — not caught by code review alone.
- Neither module has been run against a real trained model yet — both depend on `training/train_yolo_cls.py` producing real trained weights first, which is still blocked on the full dataset + GPU (see 2026-08-17 entry).
- **Next:** once Phase 2's real training run produces a trained model, run Grad-CAM + the plausibility check against real test-set predictions for the first time.

### 2026-08-19 — Phase 2: classical TBR features + fusion meta-classifier
- Verified previous upload matches exactly (only trailing-newline/cosmetic diffs) — synced the cleaner `image_conversion.py` version from the upload back into the working copy.
- Built `data_pipeline/classical_features.py`: TBR (theta/beta ratio) at frontal channels (F3/F4/Fz), matching the paper's formula, computed separately for EC/EO/VCPT (matching Table 1's "EC/EO/VCPT" feature-group structure rather than collapsing to one number). P300/behavioral fields present but explicitly NaN — not fabricated — pending the trigger-coding confirmation still outstanding.
  - Tested on real ADHD and Control subjects: TBR shows a real, consistent, discriminative pattern (ADHD higher than Control across every condition) — the correct hypothesized direction, though absolute magnitude (9-16) is higher than typical published ranges (1.5-3.5) and worth validating further once more subjects are available. Not a code bug found on inspection — formula is a straightforward theta/beta PSD ratio — but flagged rather than presented as validated.
- Built `data_pipeline/build_classical_features.py`: batch driver mirroring `build_dataset.py`'s manifest-driven, per-subject-error-handled structure.
  - **Bug found and fixed:** pandas stores a missing `vcpt_path` as float `NaN`, not `None` — and `NaN` is truthy in Python, so `row.get("vcpt_path") or None` silently passed `NaN` through as if it were a real file path, crashing 2/5 subjects with a cryptic path-type error. Fixed with an explicit `pd.notna()` check.
- Built `training/fusion_classifier.py`: merges CNN subject-level probability with classical TBR features into a logistic regression, reusing the same subject-wise CV folds and `compute_metrics()` as `train_yolo_cls.py` so results land in the same comparison table. All-NaN feature columns (currently P300/behavioral) are dropped automatically rather than imputed as noise; once those become available, they're included with no code change needed.
  - **Bug found and fixed:** `classical_features.csv` already carries a `split` column (from `build_classical_features_table`); re-merging the manifest's `split` in `run_fusion_cv` created a `split_x`/`split_y` collision instead of a usable `split` column, crashing with `KeyError: 'split'`. Fixed by removing the redundant merge.
  - **Real edge case found and handled:** a training fold can end up with only one class present (hit this for real with the tiny 5-subject test sample) — `LogisticRegression` can't fit that. Should be rare-to-never with the real stratified 103-subject 5-fold split, but now skips that fold with a clear warning instead of crashing the whole CV run, matching `build_dataset.py`'s per-subject error philosophy.
  - Verified end-to-end with real out-of-fold CNN predictions (collected properly across both toy folds, not just one) merged with real classical features — full chain runs, correctly skips the single-class fold, correctly evaluates the valid one.
- **Next:** once the full 103-subject dataset is available, run all three pipelines for real (CNN alone via `train_yolo_cls.py`, classical alone is directly readable from `classical_features.csv`, fused via `fusion_classifier.py`) and report all three in the Phase 2 comparison table against 75.8%/84.5% — this is the actual make-or-break checkpoint PROJECT.md calls for. Phase 2 and Phase 3 are now both functionally complete pending that real run.

### 2026-08-20 — significance testing vs. baselines + a real bug from the file reorg
- Noticed `classical_features.py` was moved from `data_pipeline/` to `training/` in the last upload, but `data_pipeline/build_classical_features.py` still imported from the old path (`from data_pipeline.classical_features import ...`) — would have crashed with `ModuleNotFoundError` on next run. Fixed to `from training.classical_features import ...`.
- Built `training/significance_test.py`: bootstrap confidence interval on subject-level accuracy, checked against 75.8%/84.5%. Not a McNemar's test — that needs the baseline's paired per-subject predictions on the same test set, which we don't have, only Rohani et al.'s reported aggregate accuracy. Bootstrap CI is the correct tool for comparing our accuracy against a fixed reference number.
  - Tested on a realistic 20-subject case (85% accuracy): CI came back wide enough (70–100%) that it correctly does NOT claim significance over 75.8% — appropriately conservative rather than overclaiming.
  - Tested the tiny-N edge case matching our current real sample size (3 subjects): CI degenerates to nearly [0,1] instead of falsely looking precise — confirms it won't produce a misleadingly narrow interval when there isn't enough data to support one.
- **Next:** once Phase 2's real training run (full 103 subjects) produces subject-level results for CNN-alone, classical-alone, and fused, run `compare_to_baseline()` on all three against both 75.8% and 84.5% — these numbers, plus the CIs themselves (not just the significant/not-significant flag), go directly in the paper's results section.

### 2026-08-21 — Phase 5: /predict endpoint
- Built `backend/app/inference.py`: ties preprocessing → CNN prediction (averaged across EC/EO epochs) → Grad-CAM → TBR features into one function, kept separate from the FastAPI layer so it's testable directly.
- Wired it into `backend/app/main.py` as `POST /predict`. Returns 503 with a clear message if no real trained model is configured, rather than silently using untrained weights.
- **Bug found (same class as before):** copied the old `data_pipeline.classical_features` import path into the new file — fixed to `training.classical_features`.
- **Bug found (new, more serious):** `NaN` isn't valid JSON. Since P300/behavioral biomarker fields are currently always `NaN`, this would have made **every single `/predict` request return a 500 error** — only surfaced by testing an actual HTTP request through FastAPI's `TestClient`, not by testing `run_inference()` as a plain Python function (where `NaN` floats are perfectly valid). Fixed by converting `NaN` → `None` (JSON `null`) before returning, which is also the more correct representation of "not available."
- Verified full real request/response cycle: real EEG files, real (toy) trained checkpoint, through `TestClient` — 200 response, correct prediction/confidence/TBR/Grad-CAM/disclaimer fields, P300 fields correctly `null` not `nan`.
- **Next:** cap epochs-per-request for response time (currently processes all 505 epochs in a real 12-minute file), then wire the fusion classifier in as an optional second prediction once Phase 2's real model exists.

### 2026-08-23 — TBR units bug confirmed; TBR fails to separate groups at n=20; full dataset located

Session goal was narrow: test whether the unexplained TBR magnitude (9-16 vs published 1.5-3.5, flagged on 2026-08-19) was a units bug. It was. But testing it on 20 subjects instead of 5 overturned a bigger claim.

**Built `training/verify_tbr.py`** — a read-only diagnostic that computes TBR four ways side by side (`.mean()` vs `np.trapezoid` band power × `nperseg` 256 vs 1000) and reports both magnitude and group separation. Changes no pipeline code; reuses `preprocessing.py`'s real functions so numbers are directly comparable to what the pipeline produces. Skips ICA deliberately (see below).

**CONFIRMED — the TBR magnitude anomaly is a units bug.**
- `classical_features.compute_tbr()` takes `.mean()` of the PSD across each band. That is average spectral *density*, not band *power*. Published TBR uses the integral. Theta spans 4 Hz, beta spans 18 Hz, so the ratio is inflated by ~18/4 = 4.5x.
- **Measured inflation on 20 real subjects: 4.88x** (predicted ~4.5x).
- Group means under the corrected method land inside the published range: EC 2.66 (ADHD) / 2.87 (Control), EO 1.92 / 1.95, VCPT 2.48 / 2.41. The 9-16 anomaly is fully explained. No data problem.

**OVERTURNED — TBR does not separate ADHD from Control.** This contradicts the 2026-08-19 entry's claim of "a real, consistent, discriminative pattern (ADHD higher than Control across every condition)" from 5 subjects. On 20 subjects (10/10), bootstrapped subject-level AUC:

| Condition | AUC | 95% CI | Direction |
|---|---|---|---|
| EC | 0.430 | [0.18, 0.70] | **reversed** — Control higher than ADHD |
| EO | 0.490 | [0.24, 0.75] | none |
| VCPT | 0.550 | [0.28, 0.80] | correct but negligible |

All three CIs contain 0.5. The 5-subject signal was noise. Recorded here rather than quietly dropped, because it was written up as an encouraging finding in two prior documents.

**The units fix does not change separation** — AUC moves 0.420 -> 0.430 (EC), 0.460 -> 0.450 (EO), 0.580 -> 0.550 (VCPT). It is a correctness fix, not an accuracy fix. Worth adopting so the number is defensible and comparable to literature, but it will not move the Phase 2 result.

**Implication for PROJECT.md 5a:** the premise that classical biomarkers + fusion are "the single highest-leverage item for accuracy on this dataset" currently rests on 3 features, and those 3 are at chance. The premise isn't dead — Rohani et al. reached 84.5% with 113 selected features, not 3 — but expanding the classical feature set is now the critical path for that claim, not an optional enhancement. TBR alone was never going to carry it.

**Other real problems found this session:**

1. **`parse_filename`'s regex matched ZERO real files.** It required underscores in the date/time (`2019_09_08`); the actual dataset uses dots (`C09090107-2019.12.29-15.25.17-EOEC.edf`). `discover_subjects` would have raised `ValueError` on the first file of the 103-subject run. The docstring documented the wrong convention, which is how this survived. Fixed to accept `[._-]` as separator.
2. **`nperseg=1000` is silently capped at 751 by the 1.5 s epoch length** — confirmed by the bin counts in the output (6 theta bins, not the 9 that 0.5 Hz resolution would give). Epoch length physically caps frequency resolution. TBR is a subject-level summary and has no reason to be computed on epochs at all; computing it on the continuous segment would give both better resolution and far more averaging, for free.
3. **`remove_artifacts_ica()` is a no-op.** `ica.apply()` is called with an empty `exclude` list, which reconstructs the signal bit-identically. There is currently zero artifact rejection anywhere in the pipeline, and it costs a full ICA fit per recording (~206 fits on the full cohort) for no change in output.
4. **5/20 subjects show EC < EO**, against the expected direction (theta/beta is higher eyes-closed). `F09081100, F09101156, F10011103, C10011101, C10020106`. These are candidates for a flipped EC/EO assignment. Suggests a cheap QC rule: TBR direction should agree with `alpha_ratio`, and disagreement flags a subject — stronger than the alpha ratio alone.
5. **`PROJECT.md` does not contain the methodology document.** It is a 38-line truncated copy of PROGRESS.md ending mid-August. Every module in the repo cites "PROJECT.md sec 4 step 3", "sec 5a", "sec 6 Phase 2" — none of those sections exist in the file. Needs recovering from git history. **Blocking for the paper**, since it holds the locked design decisions and the limitations list.
6. **Environment gotcha:** on this Windows box, `python` and `python3` both resolve to MSYS2's Python (`C:\msys64\ucrt64\bin\`), which has no packages. `pip` installs into `C:\Users\<user>\AppData\Local\Programs\Python\Python313\`. Use `py -m ...`, or set up a venv. Also note `where` in PowerShell is an alias for `Where-Object`, not `where.exe`.

**FULL DATASET LOCATED** — `D:\ADHD-Faezeh Rohani-edf\edf (all)\`, 109 EOEC files. This was the single biggest blocker since 2026-08-17 and it is gone. Two things to resolve before the real run:
- 109 files vs 103 subjects in the paper. `discover_subjects` raises on duplicate subject IDs, so it will halt. Need to determine whether these are genuine re-recorded sessions or the same files copied into both `edf (all)` and the `edf (just c)` / `edf (just f)` subfolders. **Check not yet run.**
- `discover_subjects` globs both `*-EOEC.edf` and `*-EOEC.EDF`. Windows filesystems are case-insensitive, so both patterns likely match the same files, producing a duplicate for every subject and tripping the same guard. Needs `set()` dedup. **Prediction, not yet verified.**

**Measured, not assumed — inputs for Phase 2 planning:**
- VCPT accounts for **65% of all epoch-images** (VCPT ~870-920 epochs/subject vs EC/EO ~200-370 each). Training on all three conditions mixed means two thirds of the training signal is one condition.
- Per-subject total epoch count varies 1275-1616 (1.27x). Milder than feared, but still argues for capping epochs per subject so recording length can't become a learnable feature.

**Not changed this session** (deliberately — verify first, patch second): `classical_features.py` still uses `.mean()`. The patch is understood and small, but should land together with the move off epochs onto the continuous signal rather than as two separate edits.

- **Next:** (1) run the duplicate-subject-ID check on the 109 files; (2) recover `PROJECT.md` from git history; (3) patch `classical_features.py` — trapz + continuous-signal PSD; (4) expand the classical feature set beyond TBR (relative band power per channel, aperiodic exponent/offset, individual alpha peak frequency, frontal alpha asymmetry, coherence summaries) — now critical path, not optional; (5) resolve the ICA no-op before any real training run, since every image and every feature currently comes from unrejected data.


### 2026-08-24 — TBR band power fixed to integral; literature review confirms the negative result

Branch: `fix/tbr-band-power`. Implements the variant validated on 2026-08-23; adds no new hypothesis.

**Changed `training/classical_features.py`:**
- Added `_band_power()` — band power is now the INTEGRAL of the PSD across the band (`np.trapezoid`), not the mean. Returns NaN if a band holds fewer than 2 bins rather than silently returning a value derived from one sample.
- `TBR_NPERSEG` 256 -> 1000 (clamped to epoch length, so 751 in practice). Frequency resolution 1.95 Hz -> 0.67 Hz; theta bins 2 -> 6.
- Added a `np.trapezoid` shim. `np.trapz` was REMOVED in NumPy 2.0 and this environment runs NumPy 2.4, so the naive fix would have raised `AttributeError`.
- Rewrote the module and function docstrings. The old module docstring asserted TBR was "the SINGLE HIGHEST-LEVERAGE item for accuracy" — contradicted by our own measurement. The old function docstring claimed TBR was computed "from the full recording's power spectrum (not per-epoch)", which was never true: it is computed from epoch PSDs averaged together, and the epoch length caps resolution.

Verified: patched logic reproduces the `trapz/1000` column from `verify_tbr.py` exactly (6θ/27β bins on real data). Known-answer test on a flat spectrum returns theta=4.0, beta=18.0, ratio=0.2222 = 4/18, as it must.

**Literature review — the negative TBR result is the current consensus, not an anomaly.** Searched properly for the first time; should have been done before treating a 5-subject signal as encouraging:

- **Arns, Conners & Kraemer (2013)**, *J Atten Disord* 17(5):374-383 — meta-analysis, 9 studies, 1253 ADHD / 517 non-ADHD, TBR at Cz eyes-open. Grand-mean ES 0.75 (6-13y), 0.62 (6-18y), but significant heterogeneity means these are overestimates. The group difference **shrank across publication years because TBR rose in the CONTROL groups**. Conclusion: excessive TBR is not a reliable diagnostic measure; may have prognostic value in a subgroup.
- **Arns et al. (2016)**, *JCPP* editorial — "How should child psychologists and psychiatrists interpret FDA device approval? Caveat emptor", written in response to the 2013 FDA clearance of the NEBA device.
- **(2020)** *Appl Psychophysiol Biofeedback* — five different spectral-analysis algorithms applied to TBR across iSPOT-A and ICAN. The methods produced significantly different TBR values and **none distinguished ADHD from controls**. This is essentially our `verify_tbr.py` experiment, published, on far more subjects.
- **Arns et al. (2024)**, *Appl Psychophysiol Biofeedback*, N=417 — subtyping meta-analysis. Grand-mean effect sizes -0.212 < d < 0.218, non-significant. "TBR has no diagnostic value for ADHD."
- **(2026)** *eLife* multiverse analysis, N=1499 + 381 — varied every reasonable methodological choice. **Individual alpha peak frequency and aperiodic neural activity shape TBR estimates, limiting their value as a biomarker.**
- **Coolidge et al. (2007)** — TBR separating ADHD from OTHER psychological disorders (the clinically real task): sensitivity 50%, specificity 36%.

**Two named mechanisms, both actionable:**
1. **Slow alpha contaminating theta.** Alpha peak frequency rises with age; a child with IAF at 7-8 Hz has genuine alpha power inside the 4-8 Hz theta window, inflating TBR with no excess theta. Fix: compute individual alpha peak frequency per subject, use it as a feature, and optionally define theta relative to each child's own alpha rather than fixed edges.
2. **Aperiodic (1/f) activity.** A difference in spectral slope shifts every band-power measure, and TBR is maximally sensitive since theta and beta sit at opposite ends. Fix: fit `specparam`/FOOOF, use exponent and offset as features.

**A confound now confirmed as live in this pipeline:** the literature specifically notes children with ADHD move more than controls, which biases spectral estimates. With ICA a no-op and no epoch rejection, this confound is fully active AND differential between our groups. This moves `fix/ica-artifact-rejection` up in priority — it is not cleanup, it is a named threat to validity.

**Reframing for the paper:** the negative result is a finding with citations, not an absence of results. If the CNN succeeds where the classical marker fails, the interesting question becomes what it is seeing — which is exactly what Grad-CAM and `clinical_plausibility.py` are built to answer. This strengthens the thesis rather than weakening it.

**Added `docs/STUDY_GUIDE.md`** — 12 modules covering EEG physiology, spectral analysis, the ADHD/TBR literature, preprocessing, wavelets, connectivity, image conversion, CNNs, evaluation methodology, small-sample statistics, and interpretability, with a 30-question self-test.

**Still outstanding from the previous session, unchanged:** the duplicate-subject-ID check on the 109 EOEC files has still not been run, and `PROJECT.md` has still not been recovered from git history. Both block the full-cohort run.

- **Next:** (1) run the duplicate-ID check; (2) recover `PROJECT.md`; (3) `fix/ica-artifact-rejection` — promoted above the augmentation and layout fixes now that the movement confound is confirmed relevant to this population; (4) TBR on the continuous segment rather than epochs; (5) expand the classical feature set, starting with aperiodic exponent and IAF since the literature names both as the specific mechanisms TBR misses.


### 2026-08-24b — ICA no-op fixed; artifact rejection now actually exists

Branch: `fix/ica-artifact-rejection`. Promoted above the augmentation and image-layout fixes: the movement confound named in the TBR literature (ADHD children move more than controls) means missing artifact rejection is differential between the groups, not random noise. That is a threat to validity, not cleanup.

**`remove_artifacts_ica()` was a no-op.** It fit an ICA and then called `ica.apply()` with an empty `exclude` list, which reconstructs the signal bit-identically — a full ICA fit per recording (~206 across the cohort) for zero change in output. Every image and every feature produced before this commit came from unrejected data.

**Now does three things it didn't:**
- **EOG detection** via `find_bads_eog` using Fp1/Fp2 as proxies. No real EOG channel exists (X1/X2 confirmed dead). Blinks dominate the frontopolar channels so the correlation is driven by ocular activity, but these ARE real EEG channels, so some genuine frontopolar brain signal is removed with them. Acceptable because TBR is computed at F3/F4/Fz — **must be stated as a limitation in the methods section.**
- **Muscle detection** via `find_bads_muscle`. Relevant specifically here: EMG contaminates 20 Hz and up, which is the beta band, the DENOMINATOR of TBR. Blinks contaminate delta/theta, the numerator. Artifact was hitting the primary biomarker from both directions.
- **Fits on a 1 Hz high-passed copy**, applies the solution to the 0.5 Hz data. Low-frequency drift degrades ICA decomposition; standard MNE practice, not a deviation.

**`epoch_signal()` had no rejection at all.** Added peak-to-peak rejection at 150 uV plus a flat-channel threshold. `reject_uv=None` restores the old behaviour for comparison. Also fixed `tmax`: was `epoch_length`, which yields one extra sample and a 1-sample overlap between consecutive epochs; now `epoch_length - 1/sfreq`.

**Design decisions worth recording:**
- Both functions **return diagnostics** rather than cleaning silently. A recording where half the components were rejected, or a third of epochs dropped, should surface — not quietly contribute fewer images. Consistent with the project's flag-don't-silently-trust norm from the EC/EO ambiguity handling.
- Warnings fire at >30% rejection and at zero surviving epochs. Neither raises — one bad subject must not kill a 103-subject batch.
- Detection failures are caught and warned, returning uncleaned data rather than propagating.
- `preprocess_subject()` now threads `ica_eoec`, `ica_vcpt`, `reject_ec`, `reject_eo`, `reject_vcpt` into its result dict.
- Chose a fixed 150 uV threshold over a per-recording percentile. A percentile guarantees dropping a fixed fraction regardless of quality — losing good data on clean recordings and keeping bad data on poor ones. A fixed threshold plus a reported rejection rate is honest and defensible.
- `autoreject` was considered and deferred: it adds a dependency, is slow, and the simpler threshold is reportable in a methods section without further explanation. Revisit if rejection rates come out extreme.

**Consequences to handle before the full run:**
- **Every image and classical feature generated so far is now stale.** They came from unrejected data. Both must be regenerated after this lands.
- Epoch counts per subject will drop, and by different amounts per subject. The VCPT-dominance figure (65% of all images) will shift and needs re-measuring.
- Subjects with high rejection rates need a QC decision: include, exclude, or flag. No policy exists yet — needs one before Phase 2.
- TBR values will change. The 2026-08-23 measurements (group means 1.9-2.9, AUC 0.43/0.49/0.55) were computed on unrejected data and must be re-run. If TBR separation changes materially once artifact is removed, that is itself a finding worth reporting — the literature specifically raises movement artifact as a source of biased TBR estimates.

**Still outstanding, unchanged across three sessions:** the duplicate-subject-ID check on the 109 EOEC files, and recovering `PROJECT.md` from git history. Both block the full-cohort run and are one command each.

- **Next:** (1) re-run the 20-subject TBR check on ARTIFACT-REJECTED data and compare against the 2026-08-23 numbers; (2) duplicate-ID check; (3) recover `PROJECT.md`; (4) `fix/yolo-augmentation-flags`; (5) `fix/topomap-grid-layout`.


### 2026-08-24c — artifact rejection thresholds set from measured data, not convention

Completes `fix/ica-artifact-rejection`. Both thresholds in the previous commit were guesses from literature; both were wrong, and measuring produced a finding.

**REJECT_PEAK_TO_PEAK_V: 150 -> 250 uV.** 150 uV rejected 100% of epochs on C09090107. Measuring the actual distribution showed why:
- Pre-ICA the worst-channel median was 260.7 uV against a pooled median of 142.3 — a 2x gap, i.e. driven by a subset of channels. Post-ICA that gap is gone (per-channel medians 33-103 uV, uniform), confirming it was blinks and that ICA removed them.
- The highest post-ICA channels are O2 (103), O1 (91), Pz (85) — occipital/parietal. That is genuine eyes-closed alpha, not artifact.
- Worst-channel p90 = 167 uV, p95 = 544 uV. A sharp discontinuity, so real artifact begins past ~500. 250 uV sits in that gap and keeps 92% of epochs.
- **A 150 uV threshold would have been systematically biased**, preferentially rejecting high-alpha epochs — i.e. discarding the EC condition's dominant physiological feature. Worth remembering: a threshold that cuts into the bulk of a distribution rather than its tail is rejecting signal, not noise.

**MUSCLE_THRESHOLD: 0.5 (MNE default) -> 0.9, plus a hard cap.** Built `training/sweep_muscle_threshold.py` to measure the parameter's effect rather than guess again. Swept 0.5-1.0 across 2 ADHD + 2 Control:

| Subject | TBR_EC range | Swing | Components excluded @0.5 |
|---|---|---|---|
| F08080102 (ADHD) | 2.356 - 2.551 | 8.3% | 5 of 19 |
| F09080101 (ADHD) | 0.517 - 2.126 | **311%** | **14 of 19** |
| C09090107 (Control) | 1.176 - 1.314 | 11.8% | 7 of 19 |
| C09091102 (Control) | 1.838 - 2.342 | 27.4% | 6 of 19 |

Three findings from this:

1. **The effect is not directionally consistent.** Removing more "muscle" RAISES TBR in three subjects and LOWERS it in one. So it is not cleanly stripping beta — it is a per-subject perturbation with no consistent sign.
2. **F09080101 is a broken recording, not a threshold problem.** 14 of 19 components flagged as muscle at the default, still 9 at 0.9. Its decomposition is dominated by high-frequency structure. Needs manual inspection before inclusion.
3. **EOG detection is completely stable** — exactly 2 components for every subject at every threshold. Blink detection works; muscle detection is the unreliable half.

Chose 0.9 (3 of 4 subjects inside the 1-4 target) plus `MAX_ICA_COMPONENTS_EXCLUDED = 5`. The cap is the important part: if detection wants more than a quarter of the decomposition, flag the subject rather than silently reconstructing from a handful of components. EOG is never capped; muscle is capped by score, strongest kept. `diagnostics["capped"]` records when it fired, for the audit log. Threshold 1.0 was rejected — all four subjects land in range there, but only because muscle detection is entirely disabled.

**This is a pipeline-wide issue, not a TBR issue.** The same ICA cleaning feeds the scalograms and topomaps. F09080101's images would have been built from 5 surviving components out of 19.

**Third measured sensitivity on the same feature.** TBR now has three documented dependencies on implementation choice:

| Choice | Shift in TBR |
|---|---|
| mean vs integral band power | 4.88x |
| 751 vs 750 samples per epoch | 27% |
| ICA muscle threshold | 8-311% |

The middle one was discovered accidentally: the `tmax` off-by-one fix changed epochs from 751 to 750 samples, moving df from 0.6658 to 0.6667 Hz. That tiny change made FFT bins land exactly on the band edges (4.0/8.0/12.0/30.0) instead of straddling them — theta span went from 3.33 Hz to the full 4.00 Hz. **A one-sample change in epoch length moved the primary biomarker by 27%.**

General rule adopted: choose `nperseg` so `fs/nperseg` divides evenly into the band edges. At 500 Hz, 750/1000/2000 all work; 751 does not.

Taken together this is an independent replication, on our own data, of the 2020 five-algorithm null and the 2026 multiverse analysis. **Belongs in the discussion section** — it is stronger evidence than citing theirs, because it is ours.

**Open decisions before the full run:**
- No QC policy yet for capped/high-rejection subjects. Include, exclude, or flag? Needed before Phase 2.
- F09080101 specifically needs manual inspection.
- All images and classical features remain stale pending regeneration.

**Still outstanding across four sessions:** duplicate-subject-ID check on the 109 EOEC files; recovering `PROJECT.md` from git history. One command each, both blocking the full-cohort run.

- **Next:** (1) duplicate-ID check and `PROJECT.md` recovery — these keep slipping and block everything; (2) re-run the 20-subject TBR check on artifact-rejected data; (3) `fix/yolo-augmentation-flags`; (4) `fix/topomap-grid-layout`.


### 2026-08-24d — both four-session blockers cleared; first real 108-subject manifest exists

Both items that kept slipping across four sessions turned out to be genuinely one command each, but the duplicate-ID "check" surfaced a real bug rather than just confirming or refuting the prediction.

**`PROJECT.md` recovered.** It was never edited down — commit `78b1b1e` (2026-08-16) accidentally overwrote it with `PROGRESS.md`'s content (both are 38 lines, identical text). The real 125-line v2 guideline was intact at the initial commit (`971d71a`) the whole time. Restored via `git show 971d71a:PROJECT.md > PROJECT.md`; no content was lost or needed to be reconstructed.

**Duplicate-ID prediction confirmed, and it's worse than "prediction, not yet verified" suggested.** Tested `discover_subjects`'s actual glob calls directly against the real 109-file dataset at `D:\ADHD-Faezeh Rohani-edf\edf (all)\`:
- `glob("*-EOEC.edf")` and `glob("*-EOEC.EDF")` each returned the **same 108 paths** — Windows resolves both patterns against the same case-insensitive filesystem, so this isn't "108 lowercase + 108 uppercase distinct files," it's 108 real files each matched twice. `discover_subjects` concatenated both lists without deduping, so the loop would hit `sf.subject_id in subjects` on the very first repeated path and raise `ValueError` immediately — confirmed by direct reproduction, not just code reading. **Fixed** in `data_pipeline/subject_split.py::discover_subjects` by deduping matches on `os.path.normcase(os.path.abspath(path))` before the loop. Kept the double-glob (not simplified to one call) because on a genuinely case-sensitive filesystem (Linux), `.edf` and `.EDF` can be real distinct files and both still need to be found.
- **Second, unrelated bug found by the same investigation:** one real file, `C11121140-2019.08.26-10.02.39-EOEC..edf`, has a double dot before the extension (a naming artifact in the source dataset, not our code). Neither glob pattern matches it, so `discover_subjects` was silently dropping subject C11121140 from the cohort entirely — not crashing, not logging, just absent. Caught by diffing every EOEC-ish filename on disk against what the glob actually matched. **Fixed:** `discover_subjects` now also walks `data_dir` and `warnings.warn()`s on any file containing the task token (`EOEC`) that didn't match the strict pattern, so a malformed filename is flagged for a manual naming decision instead of silently shrinking the cohort. Did not rename the source file — it's raw data outside the repo, and the fix belongs in code, not in the dataset.

**Ran the real split for the first time.** `py -m data_pipeline.subject_split --data-dir "D:/ADHD-Faezeh Rohani-edf/edf (all)" --output data_pipeline/splits/subject_splits.csv`:
- 108 subjects discovered (C11121140 correctly flagged and excluded, warning printed), 0 crashes.
- Stratified holdout test: n=17 (9 Control / 8 ADHD). 5-fold CV over the rest: 18-19 subjects/fold, balanced within 1 subject per class per fold.
- 108 rows in the manifest, 108 unique `subject_id` (no leakage at the discovery stage), 8 subjects missing a `vcpt_path` (expected — VCPT is optional per `SubjectRecord`, not every EOEC subject has a matching VCPT recording on disk).
- This is the first real subject-level manifest this project has ever had against the full cohort — everything before this was 3-5 real subjects or synthetic data.

**Not done this session (deliberately out of scope):** did not run `build_dataset.py` or generate images/features from this manifest. That's a long-running, GPU-adjacent step and every image/feature is already known-stale pending artifact-rejection regeneration (per 2026-08-24b/c) — running it now would produce output that needs regenerating anyway.

- **Next:** (1) `build_dataset.py` on all 108 subjects using this manifest, now that both blockers are clear — the actual Phase 2 make-or-break run; (2) decide what to do about C11121140 (inspect the malformed file / ask the dataset source / accept 108 instead of 109) before that run; (3) re-run the 20-subject TBR check on artifact-rejected data; (4) `fix/yolo-augmentation-flags`; (5) `fix/topomap-grid-layout`.


### 2026-08-24e — augmentation flags disabled; the environment was never real

Goal was `fix/yolo-augmentation-flags` (one small change). Installing the dependencies to verify it exposed that the declared environment and the actual environment had nothing in common, and that Phase 2 was blocked by something not on any list.

**`fix/yolo-augmentation-flags` — done and verified against a real training run.** `model.train()` ran with Ultralytics' photograph defaults: `fliplr=0.5` mirrors topomap hemispheres (destroying F3/F4 asymmetry) and reverses the scalogram time axis, `hsv_*` recolours a colormap where colour IS the measurement, `scale`/`erasing` crop or blank channels out of the composite. Added `DISABLE_AUGMENTATION`, spread into `model.train()`.

**The flag list in the handoff doc was wrong in two ways, both caught by checking against the installed package rather than copying:**
- It included `cutmix`, which is **not a valid key** in the pinned version — `check_dict_alignment` raises on unknown keys, so the training call would have crashed on the first fold.
- It included `degrees`, `translate`, `shear`, `perspective`, `mixup` — all valid config keys, none of which `ClassificationDataset.__init__` ever reads. Setting them to 0.0 looks like a fix and does nothing. Confirmed by parsing that method's source for `args.<key>`; the classification path reads exactly 8 augmentation keys and no more.

Method used, and worth reusing: derive the list from the source rather than from `default.yaml`, because `default.yaml` is the union across tasks. Recorded in the comment on the constant along with the one-liner to re-derive it.

**Verified the way §6O asked for** — ran a real 1-epoch train on a toy dataset and read back the `args.yaml` Ultralytics writes: all 8 flags recorded exactly as intended. Also re-confirmed the relative-`project`-path nesting from 2026-08-17 still happens on the new version (the smoke test reproduced it), so `run_cv`'s `.resolve()` is still required.

**BLOCKER FOUND, not on any prior list: the pinned Ultralytics cannot load pretrained weights on any installable torch.**
- `ultralytics==8.2.31` calls `torch.load(file, map_location="cpu")` with no `weights_only` argument. PyTorch **2.6** flipped that default from `False` to `True`. So `YOLO('yolov8n-cls.pt')` — the first line of `run_cv` — raises `UnpicklingError`.
- PROJECT.md calls ImageNet transfer learning "mandatory, not optional" at N=103, so this blocked Phase 2 completely and would have surfaced only at the first real run.
- **Downgrading torch is not an escape.** I initially thought torch 2.5.1 was an option — that was a bad check: I grepped the wheel index without filtering for `win_amd64`, so I was reading Linux wheels. On Windows + Python 3.13 the oldest installable torch is **2.6.0**, i.e. the first version with the new default. Recorded because the wrong version of this claim is easy to re-derive.
- **Resolved by upgrading to `ultralytics==8.3.253`**, which routes through a `torch_load` wrapper that handles it. Chose the last 8.3.x over 8.4.127 to stay closer to the 8.2.x the code was written against.

**Re-verified the three things the upgrade put at risk** (all previously confirmed empirically against 8.2.31, all still hold on 8.3.253):
1. `DISABLE_AUGMENTATION` — **did NOT survive**: `crop_fraction` was valid in 8.2.31 and is **removed** in 8.3.x, where passing it raises. Dropped it. (`cutmix` becomes valid here but classification still never reads it, so it stays out.) This is the concrete instance of §6O's own warning that arg names shift between versions.
2. `gradcam.py`'s `model.model.model[8]` hook — still the `C2f` block, still immediately before `Classify` at `[9]`. Confirmed by loading the real model, not assumed.
3. `gradcam.py`'s `(probs, logits)` tuple assumption — still a 2-tuple, and `probs == softmax(logits)` still holds.

**`backend/requirements.txt` described an environment that has never existed.** Every single pin differed from what is installed — mne 1.7.1 vs 1.12.1, numpy 1.26.4 vs 2.4.4, pandas 2.2.2 vs 3.0.2, torch 2.3.1 vs 2.13.0, and so on for all 17 packages. Nothing was ever installed from this file; the working environment was assembled ad hoc and drifted completely away from it. This is not cosmetic: `classical_features.py` carries an `np.trapezoid` shim *because* `np.trapz` was removed in NumPy 2.0, which is only coherent against NumPy 2.x — the file claimed 1.26.4, where `np.trapz` still exists. Rewritten to the versions actually in use, with the reasoning inline.

**`PyWavelets` was not installed at all** — `image_conversion.py` does `import pywt` for every scalogram, so `build_dataset.py` would have failed immediately on the full run. Installed (1.9.0). Also installed the missing `uvicorn` and `python-multipart`.

**Housekeeping:** Ultralytics downloads pretrained weights into the CWD, so `yolov8n-cls.pt` landed in the repo root where `.gitignore`'s `models/*.pt` did not cover it. Added `*.pt`/`*.pth`.

**Deliberately not done:** the topomap grid fix (§6J) — explained but not written, pending a decision on 2×3-with-Gamma vs 2×2-without. While reading that code I found two adjacent issues, flagged but not bundled: `nperseg=min(256, ...)` in `generate_topomap_image` violates the "`fs/nperseg` must divide evenly into band edges" rule adopted on 2026-08-24c (256 gives df=1.953 Hz; band edges 4/8/12/30 all straddle bins — the same class as the 27% TBR shift), and `(freqs >= lo) & (freqs <= hi)` puts the shared edge bin in both adjacent bands. Separately confirmed that the `.mean()` on line 116 is **not** the mean-vs-integral bug from #17 — each band is drawn with its own auto-scaled colormap, so a constant per-band factor cancels exactly. Stated so nobody re-fixes it.

**Environment note:** CPU-only, no NVIDIA GPU. At ~1,300 epoch-images per subject × 108 subjects ≈ 140,000 images, trained 5× over for 5-fold CV, a full 30-epoch run is days on CPU. **This promotes §6T (cap epochs per subject per condition) from optimization to prerequisite** — and it independently improves the VCPT-dominance problem (65% of images) rather than just being a speed hack.

- **Next:** (1) `fix/topomap-grid-layout` (§6J) once the band decision is made; (2) §6T epoch capping — now a prerequisite for any CPU run, not an optimization; (3) `feat/save-oof-probabilities` (§6P) and `feat/evaluate-on-test` (§6Q), the two gaps that make Phase 2 unable to run end to end; (4) a reduced smoke run (2 folds, 3 epochs, capped images) to prove the chain before committing to a real one; (5) `fix/scalogram-normalization` (§6H).


### 2026-08-24f — topomap/coherence grid layout; recovering an orphaned commit

**First, a git recovery worth recording so it isn't repeated.** The previous session's work (PROJECT.md recovery, glob dedup, augmentation flags, requirements rewrite) was committed onto branch `fix/duplicate-ultralytics-requirement` as `5d72788`, but PR #1 merged only the earlier commit on that branch (`5940b9d`, the duplicate-pin removal). `main` was then fast-forwarded and the branch deleted, leaving `5d72788` orphaned — the files on disk silently reverted to their pre-session state. Recovered with `git cherry-pick 5d72788`; no conflict, since the rewritten `requirements.txt` already satisfied the PR's intent (ultralytics listed once). Found via `git reflog`, which still held the commit. **Lesson: check `git log` shows the expected commits after a PR merge + pull; a fast-forward past your own branch discards it silently.**

**`yolov8n-cls.pt` (5.5 MB) is now tracked** as of `9cae0e4`, and pushed. `.gitignore` gained `*.pt` in the recovered commit, but gitignore does not untrack an already-committed file, so the two now disagree. Ultralytics re-downloads this file on demand, so it does not need to be in the repo. Left tracked pending a decision — `git rm --cached yolov8n-cls.pt` removes it going forward, though it stays in history.

**`fix/topomap-grid-layout` (§6J) — done, measured.** `generate_topomap_image` and `generate_coherence_image` both drew their 5 bands as `plt.subplots(1, 5, figsize=(10, 10))` — a single row on a square canvas. Both topomap heads and 19x19 coherence matrices render with equal aspect, so content is capped by the SHORTER side of its cell: 2 in wide in a 2 x 10 in cell, leaving the other 8 inches as whitespace.

Added `_make_panel_grid(n_panels)`, shared by both functions: picks `rows = floor(sqrt(n))`, `cols = ceil(n/rows)` (5 -> 2x3), keeps the canvas square, and switches off leftover cells so they render blank rather than as empty framed boxes. Exercised at n = 1, 2, 4, 5, 6, 9.

The square canvas is unchanged and load-bearing — it is what stops the final resize from stretching heads into ovals (the 2026-08-1x fix). Only the arrangement inside it changed.

**Measured on a real generated image rather than asserted:**

| Layout | Cell | Head Ø | Canvas inked |
|---|---|---|---|
| 1x5 (old) | 2.00 x 10.00 in | 44 px | 15.0% |
| 2x3 (new) | 3.33 x 5.00 in | **74 px** | **41.8%** |

1.68x linear, 2.83x area, at identical output size and compute. Regression check: head width 74 px vs height 75 px, aspect 0.987 — still round, oval fix intact. Coherence images went from the same ~1x5 waste to 54.1% inked.

**The handoff's §6J figures were wrong and are corrected here.** It claimed 2x3 gives "~110 px, 4x effective resolution." ~110 px is the **2x2** number (5 in cells), and 2x2 holds only four panels. Real 2x3 result is 74 px / 2.83x area. Still clearly worth doing — but the arithmetic had been attached to the wrong layout.

**Deliberately not done: the 2x2 variant.** It would give ~112 px, but requires dropping a band to four. The obvious candidate is Gamma, which §6N already flags as filter-shaped (50 Hz low-pass and 50 Hz notch overlap, transition band ~44-56 Hz). That is a decision about which bands the model sees, not a layout change, so it stays open rather than being bundled into a layout commit.

**Found while testing, not fixed here:** `spectral_connectivity_epochs` emits `fmin=0.500 Hz corresponds to 0.750 < 5 cycles based on the epoch length 1.500 sec, need at least 10.000 sec epochs or fmin=3.333. Spectrum estimate will be unreliable.` The **Delta-band coherence panel is not a trustworthy estimate at 1.5 s epochs** — the library says so explicitly. Same root cause as §6M (cone of influence): 1.5 s epochs cannot support low-frequency estimates. Either lengthen epochs for the coherence representation specifically (it is already computed per subject per condition, not per epoch, so it is free to use a different windowing) or drop Delta from the coherence panels. Not bundled here.

- **Next:** (1) §6T epoch capping — prerequisite for any CPU run; (2) `feat/save-oof-probabilities` (§6P) and `feat/evaluate-on-test` (§6Q); (3) reduced smoke run (2 folds, 3 epochs, capped) to prove the chain end to end; (4) decide the Gamma/2x2 question and the Delta-coherence question above; (5) `fix/scalogram-normalization` (§6H).


### 2026-08-24g — the CNN and classical halves of Phase 2 are finally connected

**`feat/save-oof-probabilities` (§6P) — done.** `run_cv()` computed per-subject validation probabilities on every fold, used them once for that fold's metrics, and dropped them. `fusion_classifier.run_fusion_cv()` takes exactly that table as its `cnn_subject_probs` argument. Nothing connected the two, so **the fused arm of the Phase 2 comparison could not be produced at all** — the headline result is CNN-alone vs classical-alone vs fused, and one of the three had no code path.

Added `collect_oof_predictions(fold_frames, manifest)`; `run_cv()` now accumulates each fold's `subj_df` and returns `(results_df, oof_df)`; `main()` writes `<representation>_oof_cnn_probs.csv` alongside the existing CV results.

**Why concatenated validation folds are legitimately out-of-fold**, recorded because it is the property the whole fusion step depends on: every subject belongs to exactly one fold, and each fold's predictions come from a model that never trained on that fold's subjects. So each row is a prediction on a subject unseen by the model that produced it. Fitting the meta-classifier on in-sample CNN probabilities instead would let it learn to trust an over-confident feature and inflate the fused number.

Written per-representation (`scalogram_oof_cnn_probs.csv`, `topomap_oof_cnn_probs.csv`) because the CNN probability differs between the two models and fusing against the wrong file would silently mismatch rather than error.

**Safety properties, tested both ways rather than assumed:**
- Duplicate `subject_id` across folds **asserts** — that would mean a subject was validated in two folds, i.e. the split itself leaked. Verified it fires by deliberately injecting a duplicated fold, not just by reading the code.
- A dev subject with no OOF row **warns and continues**. This happens for real when a subject produced no images (preprocessing failure, or skipped as EC/EO-ambiguous by `build_dataset.py`). It matters because that subject is then absent from the fusion table too, so the CNN-alone and fused arms would be scored on different cohorts — exactly the kind of silent mismatch that makes a comparison table wrong without looking wrong.
- Empty input raises instead of returning an empty table.

**Verified the actual handoff, not just the function.** Fed the produced table straight into `build_fusion_table()` and then `run_fusion_cv()` end to end on synthetic subjects: merge produces the required `pred_prob_adhd` / `split` / `group` columns, no `split_x`/`split_y` collision (the bug fixed on 2026-08-19 stays fixed), the all-NaN P300 column is dropped as designed, and all three folds train and score. The metrics from that run are meaningless — synthetic data with planted separation — but the chain is real.

**Also:** `main()` now `makedirs(output_dir)` before writing, since it writes two files and no longer relies on Ultralytics having created the directory first.

- **Next:** (1) `feat/evaluate-on-test` (§6Q) — the held-out test split still has no consumer, so `subject_split.py`'s two-stage design remains half-unused; (2) §6T epoch capping, prerequisite for any CPU run; (3) reduced smoke run (2 folds, 3 epochs, capped) against real images; (4) the Gamma/2x2 and Delta-coherence decisions; (5) `fix/scalogram-normalization` (§6H).