# ADHD-YOLO

YOLO-based image classification framework for pediatric ADHD screening from EEG/ERP signals — converted to 2D scalograms, topographic heatmaps, and EC/EO coherence maps, classified with `yolov8n-cls`, explained with Grad-CAM, and fused with classical EEG biomarkers via a logistic-regression meta-classifier. Built as both a research project (thesis-grade methodology) and an engineering portfolio piece (FastAPI backend, Docker, AWS deployment).

Full methodology, decisions, and roadmap: see `PROJECT.md` (its §6a indexes the §6X work IDs used throughout the log). Session-by-session log: see `PROGRESS.md`. This file is the high-level orientation — what the project is, where it stands, and what's next. A `docs/STUDY_GUIDE.md` covering the background science — EEG, spectral analysis, the ADHD biomarker literature, evaluation methodology — is referenced in places but **has not been written yet**.

> **`PROJECT.md` section 6 uses identifiers that were never defined.** The methodology document itself is intact (recovered in `93cfcc5`) — the earlier "truncated copy of PROGRESS.md" warning no longer applies. What is still missing is the lettered work-item list: `PROGRESS.md` cites §6H, §6J, §6M, §6N, §6O, §6P, §6Q, §6R and §6T as its canonical work IDs, and none of them had a definition anywhere in the repo. `PROJECT.md` §6a now carries a table reconstructed from how each ID is used in the log — accurate to the work, but not recovered from the original wording.

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
| 1 — Data pipeline | 🟡 **Regenerated as `dataset_v2` with the topography veto; per-channel interpolation was not bundled in.** The full-cohort audit (108 subjects) confirms the veto restores median alpha retention to **1.000** with no differential loss between groups (p = 0.86). `dataset_v2` was then built: 98/108 subjects processed, 10 skipped as EC/EO-ambiguous. A post-veto rejection audit surfaced a **new, separate issue**: 250 µV epoch rejection removes EC epochs at roughly 1.5× the rate of EO (32.0% vs 20.5%, systematic across 87/108 subjects) — a threshold problem that also erodes the alpha-blocking signal the EC/EO splitter depends on. Per-channel interpolation (to stop discarding ~17 clean channels per rejected epoch) is still outstanding |
| 2 — Baseline model + classical features + fusion | 🟡 All three pipelines built and connected end to end (§6P, §6Q). A first real (short, capped) CV run against `dataset_v2` — scalogram representation, 5 folds × 3 epochs — landed at **63.0% ± 11.6% accuracy** (sensitivity 45.6%, specificity 80.7%, AUC 0.669), below both the 75.8%/84.5% baselines. This is a proof-of-chain run, not a Phase 2 result: 3 epochs is far short of convergence, only one of three image representations has been tried, and CV accuracy is still optimistically biased (§6R) |
| 3 — Grad-CAM + clinical-plausibility check | 🟡 Code written and internally tested; blocked on Phase 2's real trained model |
| 4 — Literature review + paper writing | ⬜ Not started |
| 5 — Backend, dashboard, AWS deployment | 🟡 `/predict` endpoint built and tested against a toy model; everything else not started |

**Full dataset located, discovered and processed** (`D:\ADHD-Faezeh Rohani-edf\edf (all)\`, 109 EOEC files). The 109-vs-103 count resolved to **108 usable subjects** (52 ADHD / 56 Control), with `C11121140` excluded for a malformed filename. The cohort still exceeds the paper's 103 (49/54) by five subjects, unexplained and with no demographics file shipped; results are reported on 108 with the deviation stated, noting the 75.8%/84.5% baseline was computed on 103.

**Subject ages were recovered from the ID encoding.** IDs decompose as `[C|F]YYMMDDNN` — birth date plus serial — and the recording date is in the filename, so age is derivable despite the missing demographics file. All 108 parse and every age falls in 6.1–11.1 years. **Groups are age-matched: ADHD 8.29 ± 1.22 y, Control 8.43 ± 1.04 y, Mann-Whitney p = 0.50.** This matters: slow-wave power falls and alpha peak frequency rises with age, so an age difference would have made "elevated theta in ADHD" partly "younger children". Ages are *derived* from an inferred scheme, not read from a file — stated as such in the methods.

**Test split is 14 of 17 usable.** Fourteen subjects were skipped by the EC/EO ambiguity rule, four of them in the test split; five were recovered using validated changepoint boundaries (see below). At n=14 the 95% CI on accuracy is roughly ±24 points, which constrains what the final result can claim.

> 🟡 **ICA was removing occipital alpha. Fixed — but the existing images predate the fix.**
>
> An audit of all 108 subjects found ICA removing genuine occipital alpha: **median retention 0.643**, 68/108 subjects losing >25%, 34/108 losing >50% (worst 0.02×). Alpha blocking is among the most robust effects in electrophysiology, so losing a third of it is preprocessing damage, not variation.
>
> The mechanism was **not** occipital-dominant components — only 9 of 313 (2.9%) were. It was that `find_bads_eog` and `find_bads_muscle` judge components on *spectral* criteria alone, so spatially diffuse components got removed, and removing a diffuse component subtracts signal from every channel proportionally. Measured by peak region, an `eog` component peaks frontopolar only 58% of the time and a `muscle` component peaks temporal only 37% — **48% overall.** Both detectors are wrong more often than right.
>
> **The fix is a topography veto** — a spatial sanity check on a spectral detector, the same shape as the physical plausibility check the EC/EO changepoint detector needed. A component is excluded only if it peaks in the region its detection reason implies *and* is focal there (focality > 2.0). Validated on 20 subjects: median alpha retention **0.643 → 1.000**, range 0.977–1.015, and the components still removed now match the reference profile for genuine ocular artifact (occipital 0.259, frontal 3.379 against a reference 0.09 / 4.05).
>
> **Update: confirmed on the full cohort and regenerated.** `training/audit_ica_alpha.py` re-run on all 108 subjects: 71 components excluded, median occipital weight 0.237 vs. median frontal weight 3.412 (0/71 occipital-dominant), matching the reference ocular profile. Retention loss is **not differential between groups** at full scale either (ADHD median excluded 1.0, Control 0.5, Mann-Whitney p = 0.86; corr(components removed, alpha retained) = +0.12) — the earlier n=20 wobble (p = 0.096) was noise. Verdict logged in `audit_veto.log`: alpha is preserved, proceed to training. `dataset_v2` was then built from the vetoed pipeline (98/108 subjects imaged, 10 skipped as EC/EO-ambiguous — see `build_v2.log`).
>
> **Per-channel interpolation was *not* bundled into that regeneration run**, and a follow-up audit (`training/audit_rejection.py` → `audit_rejection_postveto.log`) found a second, independent problem at the 250 µV epoch-rejection threshold: it removes EC epochs systematically more than EO (mean 32.0% vs. 20.5%, EC > EO in 87/108 subjects), which both discards more of the condition the biomarkers most rely on and quietly damages the alpha-blocking contrast the EC/EO splitter uses. Rejection is dominated by a small number of frontal/frontopolar channels (Fp1, Fp2, F7, F8) plus O1/O2, with a mean of only 3.1/19 channels driving rejection per subject — i.e., largely localized, which is exactly the case per-channel interpolation is meant for. This still needs doing before `dataset_v2` is treated as final.

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

These were found by running code against real data, not by reading it. Items resolved since are listed under "Real problems found and solved" below rather than deleted, so the record of what broke stays intact.

1. ~~All generated images and classical features are stale~~ **Superseded** — `dataset_v2` regenerates images under the topography veto, but per-channel interpolation is still outstanding (see "Known limitations"), and classical features have not yet been rebuilt against it.
1b. **`dataset_v2` build skipped 10 subjects as EC/EO-ambiguous** (`build_v2.log`), and the scalogram smoke CV run separately reported 7 subjects with no CNN prediction at all (partial overlap expected, not yet reconciled) — `run_fusion_cv()` will silently train on fewer subjects than the full cohort until this is resolved.
2. **No QC policy for problem subjects.** The pipeline now warns above 30% epoch rejection and flags subjects where ICA component removal hits the cap, but there's no rule for whether to include, exclude, or flag them. **F09080101 specifically needs manual inspection** — muscle detection flags 14 of its 19 components at MNE's default threshold, meaning its decomposition is dominated by high-frequency structure. Needs deciding before Phase 2.
3. **TBR is computed on 1.5 s epochs, which physically caps frequency resolution** at 0.67 Hz. TBR is a subject-level summary and should be computed on the continuous segment instead, which would give both finer resolution and more averaging. *(The band-power units bug in the same function is now fixed — see below.)*
4. **CV accuracy will be optimistically biased (§6R).** Ultralytics selects `best.pt` by accuracy on the val fold, and evaluation then scores on that same fold. `evaluate_on_test()` already avoids this for the final model via an inner validation fold; the CV loop itself still needs the same treatment.

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
16. **Both artifact thresholds were guesses from literature, and both were wrong.** 150 µV peak-to-peak rejected 100% of epochs; measuring the actual post-ICA distribution showed the bulk ends near 170 µV with real artifact past 500, so 250 µV is the defensible choice. MNE's default muscle-detection threshold (0.5) removed 14 of 19 ICA components on one subject. Fixed by measuring rather than re-guessing (`training/sweep_muscle_threshold.py`).
17. **`remove_artifacts_ica()` was a no-op** — `ica.apply()` with an empty exclude list reconstructs the signal bit-identically, so a full ICA fit per recording changed nothing. Combined with a missing `reject` parameter in epoching, the pipeline had *zero* artifact rejection. Fixed with EOG detection (Fp1/Fp2 proxies), muscle detection, and peak-to-peak epoch rejection.
18. **`parse_filename`'s regex matched zero real files** — it required underscores in the date/time; the dataset uses dots. Would have raised on the first file of the 103-subject run. The docstring documented the wrong convention, which is how it survived. Fixed to accept `[._-]`.
19. **`run_cv()` computed out-of-fold probabilities and threw them away** — `fusion_classifier.run_fusion_cv()` required exactly those as input, so the CNN and fusion halves of Phase 2 had no connecting code path at all. Fixed with `collect_oof_predictions()`, written per-representation because the CNN probability differs between the scalogram and topomap models and fusing against the wrong file would mismatch silently rather than error.
20. **The held-out test split had no consumer.** `subject_split.py` had carved out a stratified test set since 2026-08-16; `run_cv()` only ever filtered it out. Fixed with `evaluate_on_test()`. The trap it had to avoid: Ultralytics picks `best.pt` by accuracy on whatever it gets as `val`, so passing it the test split would be selection-on-test — the final model takes an inner validation fold from dev instead, and the function raises if asked to use `test` for it.
21. **Case-insensitive globbing produced a duplicate for every subject.** `discover_subjects` globbed both `.edf` and `.EDF`; on Windows both match the same files, tripping the duplicate-ID guard and blocking the cohort run for four sessions. Fixed by deduping on `os.path.normcase(os.path.abspath(path))`.
22. **The `LABEL` channel rejected 100% of epochs on 100% of subjects.** These files carry a digital marker channel that is constant by construction, MNE types it as `eeg`, and `flat` drops an epoch if *any* eeg channel falls below threshold — so `epoch_signal` produced empty `Epochs` for every recording and the pipeline could not generate a single image. Latent until `build_dataset.py` was first run against the real cohort. The warning blamed the 250 µV peak-to-peak threshold, which sent the first investigation after the threshold value; measuring showed 250 µV was keeping 57–92% of epochs, not none. Same channel as #9, one function upstream, where it was fatal rather than merely wrong. Fixed by restricting epoching to the 19 real EEG channels, as `filter_raw` and the ICA already did; the warning now names the channels actually responsible.

23. **The EC/EO midpoint split was an unchecked assumption.** `split_eoec_by_alpha` always used `half = n // 2`. A changepoint detector validated post-ICA on all 108 subjects found the trusted-only median boundary at **0.537, not 0.500** (Wilcoxon p < 0.00001) — so the midpoint misplaces ~18 s of an 8-minute recording, and the bias runs one way: EO is contaminated with eyes-closed data, never the reverse. Real but small, and the detector validates for only 64/108, so it was **not** applied wholesale. `split_eoec_by_alpha` now accepts an explicit `boundary_frac`; it is supplied only for subjects the midpoint rule flagged as ambiguous *and* whose boundary passes all three checks. Five such subjects were rebuilt, taking the test split from 13 to 14.
24. **`build_log.csv` truncated instead of merging.** Safe for a full-cohort run, destructive for a partial one — a 5-subject `--subjects` rebuild wiped the record of the other 103. Images survived; provenance did not. The general rule this is an instance of: *any writer a partial run can touch must merge by default.* `--subjects` exists to make partial runs cheap, so a truncating writer behind it is a trap. Same shape as an earlier double-build collision.

None of this was visible from reading the paper or the dataset README — it surfaced only by loading and running against the actual `.edf` files, or by deliberately testing edge cases.

### Four diagnostics, each catching an error in the one before it

Resolving the EC/EO question took four attempts, and the sequence is worth recording because three of the four *looked* conclusive:

1. **A second vote from frontal ocular artifact — abandoned.** 72.2% agreement against a **base rate of 92.2%**: 20 points worse than always guessing EC-first, so it carried no information. The pass threshold had been compared against 50% instead of the base rate. Kept as a group-level finding (EC halves 150.2 µV vs EO 129.6 µV, Wilcoxon p < 0.0001) — a real mechanism, too noisy to classify individuals.
2. **Changepoint detection, pre-ICA — not evidence.** Median boundary 0.564 with a 1.44× "gain" over the midpoint. But `best_score` is the maximum over every candidate by construction, so it beats the midpoint automatically: a synthetic series with *no* changepoint scored gain 1.46×.
3. **Two real validation tests.** A permutation null on the time-shuffled alpha profile (real p = 0.000, no-boundary p = 0.350) and split-half agreement between O1 and O2 (real 0.000–0.033, noise 0.250–0.258). Both checked on synthetic data before being trusted. The obvious version of the split-half test — "is the ratio more extreme at the detected boundary" — passed on pure noise too.
4. **Pre-checks that fired.** ICA moves the boundary, so the validation was re-run post-ICA to measure the signal the pipeline actually splits. And six "pinned" boundaries at `MIN_SEGMENT_FRAC` turned out to be *where the search stopped*, not where the transition was.

**The finding that mattered most: a statistical test needs a physical sanity check on top of it.** Four subjects passed *every* statistical test — permutation p = 0.005, split-half difference 0.000 — with boundaries putting one condition at 47–70 seconds of an 8-minute recording. Adding a plausibility range of 0.25–0.75 as an AND (not a tiebreaker) rejected **19 subjects that passed both statistical tests**.

The same shape recurred immediately afterwards in the ICA audit: a *spectral* detector (`find_bads_eog`, `find_bads_muscle`) needs a *spatial* sanity check on top of it. Both detectors flag components on frequency-domain criteria alone, and components with no focal topography — not eye movement, not muscle — satisfy those criteria and get removed. It is the same error in a different domain.

Related: of four thresholds set in this project, the two taken from convention (150 µV peak-to-peak, MNE's muscle threshold of 0.5) were both wrong, and the two set from measured distributions were both right. **Copy conventions for physics; measure anything data-dependent.**

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

**TBR is unstable under reasonable methodological variation — measured, not assumed.** Three implementation choices, each moving the same feature on the same data:

| Choice | Shift in TBR |
|---|---|
| Mean vs. integral band power | **4.88×** |
| 751 vs. 750 samples per epoch | **27%** |
| ICA muscle-detection threshold | **8–311%** |

The middle one was accidental: fixing a one-sample `tmax` off-by-one changed `df` from 0.6658 to 0.6667 Hz, which made FFT bins land exactly on the band edges (4.0/8.0/12.0/30.0) rather than straddling them. Theta span went from 3.33 Hz to the full 4.00 Hz. *A one-sample change in epoch length moved the primary biomarker by 27%.*

The third is worse than a magnitude shift — the muscle threshold moves TBR in **inconsistent directions** across subjects (up in three, down in one), so it isn't cleanly stripping beta; it's a per-subject perturbation with no consistent sign.

This is an independent replication, on this dataset, of the 2020 five-algorithm null and the 2026 multiverse analysis. It is stronger evidence than citing theirs.

**A useful side-effect:** EC > EO holds in 15/20 subjects, the physiologically expected direction. The five that don't (`F09081100`, `F09101156`, `F10011103`, `C10011101`, `C10020106`) are candidates for a flipped EC/EO assignment — suggesting a cheap QC rule where TBR direction must agree with `alpha_ratio`.

**Measured for Phase 2 planning:** VCPT accounts for **65% of all epoch-images** (~870–920 per subject, vs ~200–370 each for EC and EO). Training on all three conditions mixed means two thirds of the training signal comes from one condition.

---

## What's next

1. ~~Confirm the veto on the full cohort~~ **Done** — full-cohort audit confirms median retention 1.000 with no differential group loss (p = 0.86).
2. ~~Regenerate the cohort~~ **Done, partially** — `dataset_v2` was built with the topography veto, but *without* per-channel interpolation. A follow-up audit found the 250 µV rejection threshold removes EC epochs systematically more than EO (32.0% vs 20.5%), so this isn't the final dataset yet.
3. **Add per-channel interpolation for epoch rejection** and rebuild once more (`dataset_v3`, or re-tag in place) — rejection is localized to a handful of channels (mean 3.1/19), which is exactly what interpolation is for. Do this before treating any accuracy number below as meaningful.
4. ~~Prove the chain with a reduced smoke run~~ **Done, for one representation** — 5-fold CV, 3 epochs, scalogram only, against `dataset_v2`: 63.0% ± 11.6% accuracy, 45.6% sensitivity, 80.7% specificity, AUC 0.669. Below both baselines, as expected from a 3-epoch capped run; not a Phase 2 result. Repeat once per representation (topomap, coherence) once interpolation lands.
5. **Compute TBR on the continuous segment** rather than 1.5 s epochs, which cap frequency resolution at 0.67 Hz.
6. **Expand the classical feature set** — relative band power per channel per condition, aperiodic exponent/offset, individual alpha peak frequency, frontal alpha asymmetry, coherence summaries. Now critical path.
7. **Close the last Phase 2 gap** — add an inner validation split to the CV loop (§6R) so epoch selection stops leaking. Out-of-fold probabilities (§6P) and held-out test evaluation (§6Q) are both done.
8. **Phase 2 — the critical checkpoint:** real subject-wise 5-fold CV, full epoch budget, for all three approaches, with significance tests against 75.8% and 84.5%.
9. **Phase 3** — Grad-CAM and clinical-plausibility against the real trained model.
10. **Phase 4** — literature review, paper writing.
11. **Phase 5** — dashboard, AWS deployment.

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
│   ├── sweep_muscle_threshold.py    # diagnostic: ICA muscle threshold sensitivity
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
│   ├── STUDY_GUIDE.md               # the science: EEG, spectral analysis, ML, stats  [NOT WRITTEN YET]
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

**Data root:** the raw `.edf` files are not in the repo. Point `$ADHD_YOLO_DATA_ROOT` at them once per machine:

```powershell
$env:ADHD_YOLO_DATA_ROOT = "D:\ADHD-Faezeh Rohani-edf"
```

`data_pipeline/splits/subject_splits.csv` stores paths **relative** to that root and is tracked in git — it is the single source of truth for which child is in which fold, and `subject_split.py` is designed never to regenerate it (regenerating reshuffles folds and invalidates any result computed against the old one). `load_manifest()` resolves those paths back to absolute ones at read time, so every downstream script keeps receiving openable paths. A manifest that still contains absolute paths continues to load unchanged; it is just not portable to another machine.

**Import note:** `data_pipeline` and `training` scripts use package-relative imports and must be run as `python -m data_pipeline.<script>` from the repo root, not `python data_pipeline/<script>.py`.

## Workflow

`main` is always deployable — never commit directly to it. One branch per feature, small commits, PR into `main` even solo, delete the branch after merge. See `docs/jira_board.md` for the epic breakdown.

## Known limitations (stated upfront, not hidden)

- Dataset is ~103 subjects — small for a deep classifier; subject-wise validation and transfer learning are mandatory, not optional.
- **The classical biomarker currently at the centre of the fusion design (TBR) does not separate the groups** on the 20 subjects tested (all AUC CIs contain 0.5). Sample size is small, so this is absence of evidence rather than evidence of absence — but the fusion path needs more than three features to be viable.
- **Blink removal uses Fp1/Fp2 as EOG proxies**, since no real EOG channel exists in this dataset (X1/X2 are confirmed dead). Those are genuine EEG channels, so some real frontopolar brain activity is removed alongside ocular artifact. Acceptable here because TBR is computed at F3/F4/Fz, but it is a stated limitation.
- **All TBR figures reported below predate artifact rejection.** They were computed on unrejected data and need re-running. Since the literature names movement artifact as a source of biased TBR estimates, a material change after rejection would itself be a finding.
- P300 latency/amplitude and per-condition behavioral features are unavailable pending clarification from the dataset source.
- **The per-frequency-row normalization applied to scalograms removes absolute band-power relationships**, which means the theta/beta ratio is not recoverable from the scalogram images by design. This was a fix for a visual problem that has a signal-content cost, and needs revisiting.
- Grad-CAM and the clinical-plausibility check have not been run against a real trained model.
- **The original 122k-image dataset must not be trained on** (superseded by `dataset_v2`, built with the topography veto). It was generated before the veto, when ICA was removing a median 36% of occipital alpha.
- **`dataset_v2` is itself not final.** It was built with the veto but without per-channel interpolation for epoch rejection, and a post-veto audit found the 250 µV threshold rejects EC epochs systematically more than EO (32.0% vs 20.5% mean, 87/108 subjects) — a threshold problem, not a subject problem, and one that erodes the alpha-blocking contrast the EC/EO splitter relies on.
- **The ICA component detectors are unreliable and the veto is a guard, not a repair.** `find_bads_eog` and `find_bads_muscle` place artifact correctly only 48% of the time. The veto discards their bad output, but roughly a third of subjects consequently receive zero component exclusions — ICA effectively off, with epoch rejection as the only artifact control. `mne-icalabel`, which classifies from topography and spectrum together, is the principled replacement and has not been evaluated.
- **No Phase 2 result exists yet.** A first real (but short, capped) CV run against `dataset_v2` — scalogram representation only, 5 folds × 3 epochs — scored 63.0% ± 11.6% accuracy, 45.6% sensitivity, 80.7% specificity, AUC 0.669, below both the 75.8%/84.5% baselines. This is a proof-of-chain smoke result, not a Phase 2 claim: 3 epochs, one representation, no inner-validation fix for §6R, and it predates the interpolation fix above.
- This is a research/decision-support tool. It does not diagnose ADHD and is not a replacement for clinical evaluation.