# Progress log

Update this at the end of every working session — a few lines is enough. This is what lets a new chat pick up exactly where you left off.

## Format

```
### YYYY-MM-DD
- What was done
- What broke / what you learned
- Next step
```

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