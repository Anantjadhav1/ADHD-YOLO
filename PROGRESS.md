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