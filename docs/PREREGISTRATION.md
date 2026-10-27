# Pre-registration: confirmatory and new analyses after Phase 2

**Registered:** 2026-10-21, by the commit that adds this file. The analysis code at the parent commit `12d0d3e` is the code these analyses use, unless a deviation is recorded below.
**Rule:** this file is append-only. The text above "Deviations" is never edited; any change to the plan is added under "Deviations" with its date and reason, before or after the result is known — and says which.

**On the strength of this record.** It is a git commit in a public repository. Git history can be rewritten — this repository's was, on 2026-10-21, to remove an attribution line from 12 commits — so the record is only as good as this commit staying unrewritten. It has not been registered on an immutable platform (OSF, AsPredicted).

---

## 1. What has already been seen

Stated so that nothing below is mistaken for a blind analysis.

| data | status at registration |
|---|---|
| Rohani et al. development folds (91 subjects; 84 in the CNN cohort) | **Extensively analysed.** Seven pre-planned measurements (all AUC 0.50–0.56; every reported CI contains 0.5), three positive controls, a QC sensitivity analysis, and one **post-hoc** model: a standardised L2 logistic regression on all 551 engineered features, pooled AUC 0.627 (95% CI 0.504–0.745, unadjusted p = 0.024), found after the others were known. |
| Rohani et al. held-out test split (17 subjects: 8 ADHD, 9 Control) | **Never used to evaluate any model.** Engineered features were computed for these subjects but no classical model was ever scored on them; their feature values were viewed only in checks unrelated to diagnosis (correlation with age, QC rates across all 108). A CNN checkpoint's predictions for them were viewed when building the dashboard's reference distribution. |
| Nasrabadi et al. (IEEE DataPort; 61 ADHD / 60 Control as described by its documentation) | **Not downloaded. Never seen.** |

Consequence: only analyses on the test split and on the Nasrabadi data can be confirmatory. New analyses on the development folds are pre-specified but not blind, and are labelled secondary.

## 2. Hypotheses and analyses

Three analyses form the **confirmatory family** (H1a, H1b, H3). One is an **estimate only** (H2) and makes no significance claim.

### H1a — Replication: ADHD is distinguishable in an independent cohort

- **Data:** every Nasrabadi subject for whom features can be computed.
- **Features:** the 100 task-recording features this project computes for the VCPT recording — relative band power (19 channels × delta 0.5–4, theta 4–8, alpha 8–12, beta 12–30, gamma 30–50 Hz), theta/beta ratio, individual alpha frequency, aperiodic exponent and offset, frontal alpha asymmetry — computed by `training/classical_features.py` at the registered code, from the Nasrabadi task recording. If any of the 19 channels is absent, every feature involving it is dropped from both datasets before any model is fitted.
- **Preprocessing:** `data_pipeline/preprocessing.py` at the registered code (0.5–50 Hz band-pass, 50 Hz notch, ICA with the same detectors, cap and veto, 1.5 s epochs, 250 µV / 0.1 µV rejection), with the sampling rate read from the files. Amplitude units are checked against the dataset documentation before any analysis, so thresholds apply in µV.
- **Model:** median imputation → standardisation → logistic regression (L2, C = 1, `max_iter` = 1000), all fitted on training folds only. No hyperparameter is tuned.
- **Validation:** subject-wise stratified 5-fold cross-validation, seed 42. No subject's data in both training and scored folds.

### H1b — Transfer: a model trained on Rohani et al. works on Nasrabadi et al.

- **Training data:** the 85 Rohani development subjects with a VCPT recording. The test split is not used.
- **Features and model:** as H1a.
- **Domain normalisation:** features are standardised **within each dataset separately** (Rohani scaler fitted on the 85 training subjects; Nasrabadi scaler fitted on all Nasrabadi subjects). This uses no labels.
- **Evaluation:** the trained model is applied **once** to every Nasrabadi subject.

### H3 — New representation on the development folds (secondary, not blind)

- **Data:** the 84-subject CNN cohort, so the result sits beside the headline table.
- **Features:** Riemannian tangent-space features. Per subject and per condition (EC, EO), one spatial covariance matrix of the 19 channels over all surviving 1.5 s epochs (OAS shrinkage), projected to the tangent space at the Riemannian mean of the **training folds**; EC and EO vectors concatenated (2 × 190 = 380 features). Implemented with `pyriemann`, which is not yet a dependency.
- **Model:** standardisation → logistic regression (L2, C = 1, `max_iter` = 1000), fitted on training folds only.
- **Validation:** the existing five development folds. No hyperparameter is tuned, so no inner selection fold is needed; each outer fold is scored by a model trained on the other four.

### H2 — The post-hoc model on untouched data (estimate only)

- **Model:** exactly the post-hoc specification — all 551 engineered features, median imputation → standardisation → logistic regression (L2, C = 1, `max_iter` = 1000) — trained once on all 91 development subjects.
- **Evaluation:** applied **once** to the 17-subject held-out test split.
- **Reported as an estimate with its 95% CI only.** At n = 17 the interval spans roughly ±0.25, so it can neither confirm nor refute the 0.627; it enters no significance test. This uses up the held-out split.

## 3. Outcome measures and decision rule

- **Primary metric:** AUC over all scored subjects (pooled out-of-fold for H1a and H3).
- **Uncertainty:** 95% CI from 10,000 bootstrap resamples of subjects; one-sided permutation p from 5,000 label permutations; seed 42. The procedure of `training/results_table.pooled_stats`.
- **Multiplicity:** Bonferroni over the confirmatory family {H1a, H1b, H3}: **α = 0.05 / 3 ≈ 0.0167.**
- **A result counts as positive only if** p < 0.0167 **and** the 95% CI lower bound is above 0.5. Anything else is reported as negative. No "trend", "approaching significance" or "marginal" language.
- Accuracy, sensitivity and specificity at a 0.5 threshold are reported alongside, not used for the decision.
- **Every result is reported, whatever it is.**

## 4. Exclusions

- Rohani data: none; QC-flagged subjects stay in, as in every Phase 2 analysis.
- Nasrabadi data: a subject is excluded only if no features can be computed (no recording loads, or zero epochs survive rejection). Every excluded subject is listed with the reason. No other exclusion, QC-based or otherwise.

## 5. Not covered by this registration

These are exploratory if done, and are reported as such:
- Fixing open issue 3 (IAF and TBR from the continuous signal rather than 1.5 s epochs). The analyses above use the features as the registered code computes them.
- New feature families (Hjorth parameters, entropy, microstates), EEG-native networks (EEGNet, ShallowConvNet), EEG foundation models, and pooling the two datasets.
- P300 and behavioural features, should the dataset author supply the VCPT trigger coding. Those would need their own registration before the coding is received.
- Any CNN analysis on the Nasrabadi data.

---

## Deviations

*Append below; do not edit the text above.*

### D1 — 2026-10-27, before any analysis: how the Nasrabadi files are read

Recorded after inspecting only file structure, documentation and storage types — no signal values, plots or statistics — and before any feature was computed.

1. **Channel names.** The files use 10-10 names for four sites: T7, T8, P7, P8. These are the same electrodes as this pipeline's T3, T4, T5, T6, and are renamed accordingly (T7→T3, T8→T4, P7→T5, P8→T6). They are therefore *not* treated as absent, and no feature is dropped on that account. Column order is taken from `Channel_Labels.docx`: Fp1 Fp2 F3 F4 C3 C4 P3 P4 O1 O2 F7 F8 T7 T8 P7 P8 Fz Cz Pz.
2. **Sampling rate.** Not stored in the files. Taken from the dataset documentation: [FILL: e.g. 128 Hz].
3. **Units and storage.** 32 files are stored as float64 and 89 as int16. Storage type is balanced across groups (float64: ADHD 16/61, Control 16/60), so it cannot stand in for diagnosis. Units per the dataset documentation: [FILL: e.g. µV, and whether int16 and float64 files share a scale]. All data are converted to float and to volts before preprocessing, so the 250 µV and 0.1 µV thresholds apply as registered.
4. **File-name suffix.** 49 files end in "p" (ADHD 29/61, Control 20/60). The documentation gives no meaning for it; it is ignored, and the subject ID is the file stem.
5. **Labels.** Diagnosis is taken from the top-level folder name (ADHD_part1/2, Control_part1/2): 61 ADHD, 60 Control.
