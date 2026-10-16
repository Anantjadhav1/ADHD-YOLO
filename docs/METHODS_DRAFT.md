# Methods — working draft

**Status.** Drafted 2026-10-16 from `PROGRESS.md`, `README.md`, `PROJECT.md` and the code itself. Every parameter below was read from the code, not from memory or the logs; where the record is incomplete the text says **[verify]**. Citations still need checking against primary sources before submission. This is a scaffold for the thesis chapter, not finished prose.

---

## 1. Participants and recordings

EEG from Rohani et al. (2022), recorded from children diagnosed with ADHD and typically developing controls. Each child has up to two recordings:

| file | content | approx. length |
|---|---|---|
| `<ID>-<date>-<time>-EOEC.edf` | resting state, eyes closed then eyes open, in one file | ~8 min |
| `<ID>-<date>-<time>-VCPT.edf` | visual continuous performance task (Go/NoGo) | ~12 min |

19 scalp channels (10–20 system) at 500 Hz, plus two channels confirmed flat (`X1`, `X2`, dropped) and a digital marker channel (`LABEL`). Diagnosis is encoded in the subject-ID prefix (`F` = ADHD, `C` = Control).

The distributed files yield **108 subjects (52 ADHD, 56 Control)**, against the 103 (49 / 54) described in the source paper; the difference was not resolved and should be stated. One further EOEC file carries a malformed name (`C11121140-…-EOEC..edf`) and is excluded by discovery with a warning. 100 of the 108 have a VCPT recording.

**Not available:** the VCPT `LABEL` channel does not encode the four stimulus conditions (pulse counts vary 100–175 per subject with continuous widths, consistent with response markers). P300 and per-condition behavioural measures — part of Rohani et al.'s 826 features — therefore could not be computed.

## 2. Subject-wise partitioning

All partitioning is at the subject level; no epoch, image or feature from one child appears on both sides of any split (`data_pipeline/subject_split.py`, seed 42).

- A stratified held-out **test split of 15%** (17 subjects: 8 ADHD, 9 Control).
- The remaining 91 subjects form **five stratified development folds**.
- One manifest records each subject's split; every downstream script reads it rather than re-deriving splits. Fold assembly and the out-of-fold table both assert that no subject crosses a boundary.

**The held-out test split was never used for a one-shot evaluation.** `evaluate_on_test()` exists, but no run is recorded; the split was later used only to build the reference distribution shown by the demonstration interface (scored by a checkpoint that neither trained nor selected on it). All results below are development-fold cross-validation.

## 3. Preprocessing (`data_pipeline/preprocessing.py`)

1. **Montage and filtering.** Standard 10–20 positions attached (the files carry none). Band-pass 0.5–50 Hz and a 50 Hz notch, on EEG channels only; `LABEL` stays unfiltered.
2. **ICA artifact removal.** 19 components (`random_state=42`), fitted on a 1 Hz high-passed copy and applied to the 0.5 Hz data.
   - *Ocular:* `find_bads_eog` with Fp1/Fp2 as proxies — there is no EOG channel. Genuine frontopolar activity is removed along with blinks; the main biomarkers are computed at F3/F4/Fz.
   - *Muscle:* `find_bads_muscle`, threshold 0.9 (MNE's default of 0.5 removed 14 of 19 components in one subject; chosen by sweeping 0.5–1.0).
   - *Cap:* at most 5 components excluded; ocular components are never capped, muscle components are kept strongest-first. When the cap fires the subject is flagged for manual review.
   - *Topography veto:* a flagged component is kept for exclusion only if its spatial pattern peaks in the expected region (ocular: frontal/frontopolar; muscle: temporal) with focality ≥ 2.0.
3. **Eyes-closed / eyes-open segmentation.** The EOEC files contain no event markers. The boundary is located from occipital alpha (8–12 Hz) blocking; for 64 subjects a detected boundary passed validation and is used (`docs/ec_eo_boundaries_postica.csv`), otherwise the midpoint. A subject whose EC/EO alpha ratio falls in 0.7–1.4 is flagged ambiguous; 7 development subjects were excluded from the image-based arms on this ground. Because the boundary is alpha-derived, EC vs EO is never used as a classification target.
4. **Epoching and rejection.** Non-overlapping 1.5 s epochs (750 samples). An epoch is rejected if any EEG channel exceeds 250 µV peak-to-peak or is flat below 0.1 µV. Both thresholds were set from measured distributions: 150 µV, the common default, rejected every epoch of at least one subject and preferentially removed high-alpha eyes-closed epochs. A rejection rate above 30% is flagged.

## 4. Image representations (`data_pipeline/image_conversion.py`)

Images are 224 × 224. For the ADHD analysis at most 300 epochs per subject per condition were imaged, sampled evenly across the recording, so recording length cannot become a learnable feature and VCPT (otherwise ~65% of images) does not dominate.

- **Scalogram.** Continuous wavelet transform (complex Morlet `cmor1.5-1.0`, 40 frequencies, 0.5–50 Hz) of Fz, Cz, Pz, F3 and F4, stacked vertically. Encoded as three distinct views rather than a colormap: **R** log10 power on a fixed global scale (preserves absolute band power, so the theta/beta ratio is recoverable — r = 1.000 on synthetic epochs spanning TBR 1–5, against r = −0.09 for the earlier row-normalised encoding); **G** aperiodic-corrected power; **B** per-frequency z-score, clipped at ±3.
- **Topomap.** Welch band power per channel (nperseg 1000, bounded by the epoch length) for delta 0.5–4, theta 4–8, alpha 8–12, beta 12–30 and gamma 30–50 Hz, drawn as five scalp maps in a 2 × 3 grid on a square canvas (a single row left each head 44 px wide; the grid gives 74 px).
- **Coherence.** Imaginary coherence (plain coherence was 0.98–0.999 for every pair regardless of distance, i.e. volume conduction), 10 s windows with 50% overlap. Used only as summarised features (§5): one map per subject per condition is too few images for a CNN.

## 5. Engineered features (`training/classical_features.py`) — 551 per subject

Computed for EC, EO and VCPT separately unless noted.

| family | detail | per condition |
|---|---|---|
| relative band power | 19 channels × 5 bands | 95 |
| theta/beta ratio | F3/F4/Fz; theta 4–8, beta 12–30 Hz; band power as the **integral** of the Welch PSD (a mean of the PSD inflated TBR 4.88×) | 1 |
| individual alpha frequency | peak in 7–13 Hz at O1/O2 | 1 |
| aperiodic exponent and offset | linear fit of log power vs log frequency over 1–45 Hz excluding 7–13 Hz. **A simplification of specparam/FOOOF without iterative peak removal — do not call it FOOOF.** | 2 |
| frontal alpha asymmetry | F3 vs F4 | 1 |
| coherence summaries | 25 region-level imaginary-coherence summaries × 5 bands; EC and EO only | 125 |

Plus EC→EO alpha reactivity (1). Totals: 225 (EC) + 225 (EO) + 100 (VCPT) + 1 = 551.

Spectral features are computed from 1.5 s epochs, which bounds frequency resolution at 0.67 Hz; individual alpha frequency is quantised accordingly (105 of 108 subjects fall on five values).

## 6. Classifiers

**CNN** (`training/train_yolo_cls.py`). YOLOv8n-cls (Ultralytics; 8.3.253 is pinned locally, the version on the Colab runs is **[verify]**), ImageNet-pretrained, first 8 modules frozen; 224 px input; up to 30 epochs with early stopping (patience 5); seed 42, deterministic. All photographic augmentations disabled — horizontal flips reverse time on scalograms and mirror hemispheres on topomaps, and colour jitter alters a colour channel that *is* the measurement. The disabled set was derived from the classification dataloader's source, not from the default configuration (`fliplr`, `flipud`, `scale`, `hsv_h/s/v`, `erasing`, `auto_augment`). Scalogram and topomap are separate arms. A fully fine-tuned scalogram model is also reported.

**Engineered-feature SVM** (`training/classical_baseline.py`). Median imputation → standardisation → `SelectKBest` (ANOVA F) with k ∈ {10, 20, 40, 80} → RBF SVM with C ∈ {0.1, 1, 10}, balanced class weights. k and C are chosen on the inner fold and the model is refitted on training + inner fold before scoring the outer fold, so feature selection never sees the subjects it is scored on.

**Fusion** (`training/fusion_classifier.py`). Logistic regression (scikit-learn defaults: L2, C = 1, `max_iter` 1000) on the **scalogram** CNN's out-of-fold probability plus the 551 engineered features, median-imputed. (Which arm was used had not been recorded; the scalogram file reproduces the committed fusion fold AUCs exactly and the topomap file does not.) Two limitations to state: features are **not standardised** before the penalised regression, so the penalty acts unevenly across units; and the standard stacking leak applies (the CNN probability for a subject in fold *k* comes from a model trained without fold *k*, while the fusion model's training subjects have probabilities from models that included fold *k*).

**Three-way split for the CNN and SVM.** Within each outer fold the next development fold (rotating) is the inner fold, used only for checkpoint or hyperparameter selection, and the remaining three folds train. Selecting the epoch on the same fold that is scored inflates the estimate; this design prevents it at the cost of one fold of training data. Fusion uses a two-way split, acceptable because a logistic regression involves no checkpoint selection.

## 7. Evaluation

- **Subject-level aggregation.** CNN images are per epoch; each subject's prediction is the mean probability over its epochs, thresholded at 0.5. All metrics are per subject, so the effective sample size is the number of children.
- **Pooled AUC.** Computed over all out-of-fold subject predictions, not as a mean of fold AUCs (~17 subjects per fold gives unstable fold-level AUCs).
- **Uncertainty.** 95% CI from 10,000 bootstrap resamples of subjects; one-sided permutation p from 5,000 label permutations (fraction of permuted AUCs ≥ observed); a fresh seed-42 generator per arm.
- **Common cohort.** The headline table restricts every arm to the 84 development subjects all arms cover (42 ADHD / 42 Control), regenerated from the committed out-of-fold files by `training/results_table.py` and pinned by a test. The engineered-feature arm alone covers 91 (44 / 47).
- **Paired comparisons** (`training/paired_comparison.py`, n = 84): bootstrap CI of the AUC difference, McNemar's test on the thresholded calls, and Cohen's κ for agreement.
- Fusion is reported as a pooled AUC like every other arm (0.537). The figure previously quoted, 0.548, was the mean of five fold AUCs.
- Accuracy, sensitivity and specificity at 0.5 are reported alongside. Rohani et al.'s 75.8% and 84.5% are **accuracies**; AUCs here should not be compared with them directly.

## 7b. Post-hoc sensitivity analysis — exploratory (`training/linear_sensitivity.py`)

Specified after the results above were known, so reported separately and not in the main table. Re-fitting fusion with standardised inputs raised the pooled AUC from 0.537 to 0.626; the same standardised logistic regression *without* the CNN probability gave 0.627 (95% CI 0.504–0.745, unadjusted permutation p = 0.024). The change is attributable to the model class — a linear model on all 551 standardised features rather than an RBF SVM on ≤80 selected ones — not to fusion. State: post hoc; at least eight prior analyses of the same subjects; uncorrected p; fold AUCs 0.347–0.764; largest weights on eyes-open coherence and temporal/occipital gamma. Not confirmable on the 17-subject held-out split (CI ≈ ±0.25).

## 8. Positive control (`training/positive_control.py`)

To establish that a null result reflects the data rather than the pipeline, the engineered-feature arm's own `run_cv` was applied unmodified to **rest vs task** in the same children: one row per child per condition, both rows in the child's fold, development subjects only, the 100 features measured in both conditions (coherence is rest-only and would let the classifier separate the classes by missing columns). Primary contrast EO vs VCPT — both eyes-open, both labelled by recording rather than by alpha; EC+EO vs VCPT reported as secondary. The bootstrap resamples children and the permutation null flips rest/task within each child.

The CNN arm's positive control (`training/positive_control_cnn.py`) uses the same construction through `train_yolo_cls.run_cv`, on the 84-subject CNN cohort, with images built at 40 epochs per condition **[result pending]**.

## 9. Quality control

Per subject: rejection rate per condition, whether the ICA muscle cap fired and how many components muscle detection flagged, and EC/EO ambiguity — recorded in `build_log.csv` and returned per upload by the `/predict` endpoint. No include/exclude rule was applied: **flagged subjects are included in every analysis**. Across all 108 subjects the muscle cap fired for 27, 65 lost more than 30% of epochs in at least one condition (eyes-closed median 31%; 90th percentiles EC 63%, EO 45%, VCPT 45%), and 78 were flagged on at least one ground; flag rates did not differ by group (cap: Fisher p = 0.39). As a sensitivity analysis, four exclusion policies were applied to the fixed out-of-fold predictions (`training/qc_sensitivity.py`; 84 → 52 subjects): every arm's 95% CI contained 0.5 under every policy, so the conclusion does not depend on the QC decision. Before 2026-10-16 the recorded `capped` flag was computed after the topography veto and was false for all 108 subjects; the cap warnings in the logs were correct throughout, and no exclude list or result depended on the flag.

## 10. Software

Python 3.13; MNE 1.12.1; NumPy 2.4.4; SciPy 1.17.1; scikit-learn 1.8.0; PyWavelets 1.9.0; mne-connectivity 0.8.1; Ultralytics 8.3.253; PyTorch 2.13 (CPU, local). The CNN development runs were made on Google Colab GPUs **[verify the library versions used there]**. `backend/requirements.txt` pins the local versions.

## 11. Limitations to state

1. 108 subjects in the distributed data against 103 in the source paper; reason unknown.
2. No P300 or behavioural measures (§1) — the most plausible single reason the engineered-feature arm falls short of Rohani et al.
3. EC/EO boundaries are alpha-derived; 7 ambiguous subjects excluded from the image arms.
4. Frontopolar channels double as EOG proxies.
5. Gamma (30–50 Hz) lies in the transition band of the 50 Hz low-pass and notch and carries residual EMG; frontopolar gamma is among the most "discriminative" features and is artifact.
6. 1.5 s epochs limit spectral resolution (TBR, IAF).
7. No QC-based exclusion; flagged recordings are included.
8. Fusion features unstandardised (standardising changes the result — §7b); stacking leak.
9. The held-out test split was never evaluated one-shot.
10. n = 84 gives 95% CIs of roughly ±0.13 AUC; the study can exclude large effects, not small ones.
