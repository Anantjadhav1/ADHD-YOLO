# ADHD-YOLO

YOLO-based image classification framework for pediatric ADHD screening from EEG/ERP signals — converted to 2D scalograms, topographic heatmaps, and EC/EO coherence maps, classified with `yolov8n-cls`, explained with Grad-CAM, and fused with classical EEG biomarkers via a logistic-regression meta-classifier. Built as both a research project (thesis-grade methodology) and an engineering portfolio piece (FastAPI backend, web dashboard, Docker).

Full methodology, decisions, and roadmap: see `PROJECT.md` (its §6a indexes the §6X work IDs used throughout the log). Session-by-session log: see `PROGRESS.md`. This file is the high-level orientation — what the project is, where it stands, and what's next. A `docs/STUDY_GUIDE.md` covering the background science — EEG, spectral analysis, the ADHD biomarker literature, evaluation methodology — is referenced in places but **has not been written yet**.

> **`PROJECT.md` section 6 uses identifiers that were never defined.** The methodology document itself has been overwritten twice by accidental pastes — with a copy of the progress log (`78b1b1e`, recovered in `93cfcc5`), and later with a copy of this README (`d229351`, restored from `b3317f3` on 2026-09-24; see solved issue 26). If its first line is not `# ADHD-YOLO — Project Guideline`, it has happened again. What is still missing is the lettered work-item list: `PROGRESS.md` cites §6H, §6J, §6M, §6N, §6O, §6P, §6Q, §6R and §6T as its canonical work IDs, and none of them had a definition anywhere in the repo. `PROJECT.md` §6a now carries a table reconstructed from how each ID is used in the log — accurate to the work, but not recovered from the original wording.

---

## What this project actually does

Takes raw EEG recordings from children (resting-state eyes-open/eyes-closed, plus a Go/NoGo attention task), converts the 1D signal into 2D image representations (scalograms, topomaps, EC/EO coherence maps), and classifies ADHD vs. Control three ways: a `yolov8n-cls` image classifier alone, classical hand-engineered biomarkers alone, and a fusion of both via a meta-classifier — reported side by side. Grad-CAM overlays show where the CNN looked; the planned clinical-plausibility check against known ADHD-relevant electrode sites was cut once every arm measured at chance (see Current status).

**This is a decision-support research tool, not a diagnostic device.** That framing is deliberate and stated everywhere in the project.

## Why YOLO, and why classification instead of detection

The original idea was YOLO *object detection* — drawing bounding boxes around EEG anomalies like theta bursts or P300 latency drops. That was dropped early: those boxes would have to be generated from the same threshold rules (TBR > 3.0, P300 > 380ms) the project's own biomarker engine already uses, which means the detector would just be re-learning a rule that was already written down — no independent signal, and a guaranteed first question from any reviewer. Classification (`yolov8n-cls`) + Grad-CAM for localization avoids that circularity entirely while keeping the same "is this novel" angle: YOLO-on-spectrograms exists in other domains (confirmed against sleep-apnea EEG literature) but not, as far as we found, for pediatric ADHD specifically.

## The baseline we're measuring against

Rohani et al. (2022) — the paper this dataset comes from — got **75.8% accuracy** with an SVM on 113 hand-selected features (from 826 originally extracted), and a **feature-selection + Logistic Regression combination hit 84.5%** in their own results table, though it was excluded from their headline result for clinical-interpretability reasons, not accuracy. Both numbers are real targets here.

**Note the feature count: 113, not 3.** The project's working assumption was that the classical-biomarker + fusion path is the highest-leverage route to beating them. On 2026-08-23 that path carried only three features (TBR per condition). It has since been expanded to 551 (relative band power, individual alpha frequency, aperiodic exponent and offset, frontal asymmetry, alpha reactivity, coherence) and still measures at chance. What remains untested is the P300 and behavioural features, which need the VCPT trigger coding.

---

## Current status

| Phase | Status |
|---|---|
| 0 — Setup (repo, GitHub, Jira, Docker) | ✅ Done |
| 1 — Data pipeline | ✅ **Complete.** `dataset_v2_small` regenerated with the topography veto active — 98 subjects, ~60,000 images, occipital alpha retention 1.000. Every known correctness issue fixed or documented |
| 2 — Baseline model + classical features + fusion | ✅ **Complete. Seven measurements, four arms, three method families — all at chance.** AUC 0.50–0.56, every CI containing 0.5, all paired differences non-significant |
| 3 — Grad-CAM + clinical-plausibility check | ⬜ **Cut.** Attribution on a model at AUC 0.52 would show where a coin flip looks. A Grad-CAM figure over a null result would imply the model had learned something. The live dashboard still shows the overlay, captioned as attention rather than evidence |
| 4 — Literature review + paper writing | 🟡 Literature review drafted, including 2026 work that independently reaches the same TBR conclusion. Paper not started |
| 5 — Backend, dashboard, deployment | 🟡 **Runs end to end on a real recording** (2026-09-24): upload → preprocessing → CNN → Grad-CAM → biomarkers → dashboard. The dashboard's chance band needs fixing (open issue 4). Public deployment not started |

**Full dataset located, discovered and processed** (`D:\ADHD-Faezeh Rohani-edf\edf (all)\`, 109 EOEC files). The 109-vs-103 count resolved to **108 usable subjects** (52 ADHD / 56 Control), with `C11121140` excluded for a malformed filename. The cohort still exceeds the paper's 103 (49/54) by five subjects, unexplained and with no demographics file shipped; results are reported on 108 with the deviation stated, noting the 75.8%/84.5% baseline was computed on 103.

**Subject ages were recovered from the ID encoding.** IDs decompose as `[C|F]YYMMDDNN` — birth date plus serial — and the recording date is in the filename, so age is derivable despite the missing demographics file. All 108 parse and every age falls in 6.1–11.1 years. **Groups are age-matched: ADHD 8.29 ± 1.22 y, Control 8.43 ± 1.04 y, Mann-Whitney p = 0.50.** This matters: slow-wave power falls and alpha peak frequency rises with age, so an age difference would have made "elevated theta in ADHD" partly "younger children". Ages are *derived* from an inferred scheme, not read from a file — stated as such in the methods.

**Test split is 14 of 17 usable.** Fourteen subjects were skipped by the EC/EO ambiguity rule, four of them in the test split; five were recovered using validated changepoint boundaries (see below). At n=14 the 95% CI on accuracy is roughly ±24 points, which constrains what the final result can claim.

> ✅ **ICA was removing occipital alpha. Fixed, confirmed on the full cohort, and the dataset regenerated.**
>
> An audit of all 108 subjects found ICA removing genuine occipital alpha: **median retention 0.643**, 68/108 subjects losing >25%, 34/108 losing >50% (worst 0.02×). Alpha blocking is among the most robust effects in electrophysiology, so losing a third of it is preprocessing damage, not variation.
>
> The mechanism was **not** occipital-dominant components — only 9 of 313 (2.9%) were. It was that `find_bads_eog` and `find_bads_muscle` judge components on *spectral* criteria alone, so spatially diffuse components got removed, and removing a diffuse component subtracts signal from every channel proportionally. Measured by peak region, an `eog` component peaks frontopolar only 58% of the time and a `muscle` component peaks temporal only 37% — **48% overall.** Both detectors are wrong more often than right.
>
> **The fix is a topography veto** — a spatial sanity check on a spectral detector, the same shape as the physical plausibility check the EC/EO changepoint detector needed. A component is excluded only if it peaks in the region its detection reason implies *and* is focal there (focality > 2.0). Validated on 20 subjects: median alpha retention **0.643 → 1.000**, range 0.977–1.015, and the components still removed now match the reference profile for genuine ocular artifact (occipital 0.259, frontal 3.379 against a reference 0.09 / 4.05).
>
> **The dataset was regenerated with the veto active** (`dataset_v2`, and the smaller `dataset_v2_small`). The original ~122k images in `dataset/` predate the fix and must not be used.
>
> The loss was **not differential between groups** (ADHD 0.629, Control 0.651, p = 0.68), so preprocessing was not manufacturing a between-group difference. Re-confirmed after the veto on the full cohort: median retention 1.000, between-group p = 0.86.

### What's built

- **`data_pipeline/preprocessing.py`** — loads real `.edf` files, applies the paper's filter protocol (0.5–50 Hz bandpass, 45–55 Hz notch), splits resting-state recordings into eyes-closed/eyes-open via the alpha-blocking effect, removes ocular and muscle artifacts with ICA plus a topography veto, and rejects epochs at a measured 250 µV peak-to-peak threshold.
- **`data_pipeline/image_conversion.py`** — CWT scalograms (three views in RGB: global log power, aperiodic-corrected, row z-score), topographic band-power heatmaps, EC/EO imaginary-coherence maps.
- **`data_pipeline/subject_split.py`** — subject-level stratified holdout test set + stratified 5-fold CV, writing one manifest CSV every downstream script reads from, so train/test leakage is structurally prevented rather than relying on discipline.
- **`data_pipeline/build_dataset.py`** — batch driver with per-subject error handling and an audit log.
- **`training/classical_features.py`** — theta/beta ratio at frontal channels plus relative band power, individual alpha frequency, aperiodic exponent/offset, frontal asymmetry, alpha reactivity and coherence summaries, separately for EC/EO/VCPT (551 usable features). P300 and behavioral fields explicitly `NaN`, not fabricated.
- **`data_pipeline/build_classical_features.py`** — batch driver, one CSV row per subject.
- **`training/train_yolo_cls.py`** — subject-wise CV training driver. Aggregates epoch-level predictions to one prediction per subject before computing metrics, since the baselines being compared against are subject-level numbers.
- **`training/fusion_classifier.py`** — merges the CNN's subject-level probability with classical features into a logistic-regression meta-classifier.
- **`training/significance_test.py`** — bootstrap CI on subject-level accuracy vs. the 75.8%/84.5% baselines.
- **`training/verify_tbr.py`** — read-only diagnostic comparing four TBR computation variants for both magnitude and group separation. Changes no pipeline code.
- **`interpretability/gradcam.py`**, **`interpretability/clinical_plausibility.py`** — Grad-CAM hooked into the inspected `yolov8n-cls` architecture, plus an attention-vs-known-ADHD-sites sanity check.
- **`training/classical_baseline.py`**, **`training/paired_comparison.py`** — the SVM arm with feature selection inside the CV loop, and McNemar + paired-bootstrap comparisons between arms.
- **`backend/app/`** — FastAPI. `/predict` takes an EOEC (and optional VCPT) `.edf` and returns a subject-level probability, TBR biomarkers and a Grad-CAM overlay.
- **`frontend/index.html`** — single-file dashboard over `/predict`, with a worked-example mode that needs no backend.

---

## Open correctness issues

These were found by running code against real data, not by reading it. Items resolved since are listed under "Real problems found and solved" below rather than deleted, so the record of what broke stays intact.

1. **Seven subjects sit outside the intersection of all arms.** The classical arm covers 91 development subjects; the CNN arms cover 84, because 7 EC/EO-ambiguous subjects have features but no generated images. Paired comparisons use the 84-subject intersection and report what was dropped.
2. **No QC policy for problem subjects.** The pipeline now warns above 30% epoch rejection and flags subjects where ICA component removal hits the cap, but there's no rule for whether to include, exclude, or flag them. **F09080101 specifically needs manual inspection** — muscle detection flags 14 of its 19 components at MNE's default threshold, meaning its decomposition is dominated by high-frequency structure. Still undecided.
3. **TBR is computed on 1.5 s epochs, which physically caps frequency resolution** at 0.67 Hz. TBR is a subject-level summary and should be computed on the continuous segment instead, which would give both finer resolution and more averaging. *(The band-power units bug in the same function is now fixed — see below.)*
4. **The dashboard's chance band conflates two quantities.** The shaded 0.397–0.641 is the 95% CI of the pooled AUC, but it is drawn on the axis of a single child's predicted probability, captioned as if predictions inside it are uninformative and those outside are not. On the 84 out-of-fold predictions, outside-band predictions are correct 8/15 (53%) against 38/69 (55%) inside. Fix: replace it with ADHD and Control dot strips from the 31 subjects the demo checkpoint never trained or selected on, scored through the live path. *(§6R, previously listed here, is fixed; see the results section.)*

## Real problems found and solved

1. **No channel position data in the raw files** — broke topomap plotting. Fixed by attaching a standard 10-20 montage on load.
2. **EOEC files have zero event markers.** Solved via the alpha-blocking effect instead of an external timing file. Ambiguous-ratio subjects are flagged for manual QC, not silently trusted.
3. **VCPT trigger channel doesn't encode trial conditions** — pulse count varies 100–175 across subjects with continuously-varying pulse width, consistent with a behavioral response marker, not the 4-condition stimulus code the paper describes. **True P300 latency/amplitude and per-condition behavioral features can't currently be recovered.** Pending confirmation from the dataset's corresponding author — outstanding since mid-August.
4. **Scalogram images came out visually flat** — EEG's 1/f trend let one band dominate a global color scale. Fixed with per-frequency-row normalization, which removed absolute band-power information from the images. *(That cost was removed on 2026-08-25, §6H: scalograms now carry three views in RGB, with global log power in the red channel, so TBR is recoverable from the image — r = 1.000 against the signal on synthetic epochs, where row z-scoring gave r = −0.09.)*
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
25. **`/predict` would have crashed on Windows on the first real request.** `inference.py` wrote per-epoch PNGs to a hard-coded `/tmp/...`, which Windows resolves to `D:\tmp\`, which does not exist, so every request would raise `FileNotFoundError` and return 500. Latent because no checkpoint had ever been present locally, and `main.py` returns 503 before inference runs. Fixed with `tempfile.mkdtemp()`. Same shape as #22: invisible until the first real run.
26. **`PROJECT.md` was overwritten by accidental pastes, twice.** On 2026-08-16 (`78b1b1e`) it was replaced with a copy of the progress log, and on 2026-08-26 (`d229351`) with a copy of this README, both inside routine doc-update commits. Each time the methodology document silently disappeared while every reference to it kept pointing at it. Restored from `b3317f3` on 2026-09-24. A one-line check that each doc still starts with its own title would catch a third occurrence.

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

## The result: seven measurements, all at chance

Four arms across three method families, on the same 84 development subjects
(42 ADHD / 42 Control, so 0.5 is a genuine chance baseline):

| arm | AUC | 95% CI | permutation p |
|---|---|---|---|
| TBR (best condition, VCPT) | 0.550 | — | — |
| CNN scalogram, full fine-tune | 0.503 | — | — |
| CNN scalogram, frozen backbone | 0.522 | [0.397, 0.641] | 0.311 |
| CNN topomap, frozen backbone | 0.523 | [0.393, 0.647] | 0.368 |
| Classical SVM, 301 power features | 0.517 | [0.398, 0.640] | 0.400 |
| Classical SVM, 551 features with coherence | **0.556** | [0.416, 0.660] | 0.268 |
| Fusion (CNN + classical) | 0.548 | — | — |

Every confidence interval contains 0.5. Baselines to beat were 0.758 and 0.845.

Paired tests between arms — far more powerful than comparing intervals, because
both models are scored on the same subjects — find no significant difference
anywhere: scalogram vs topomap p = 0.992, scalogram vs classical p = 0.639,
topomap vs classical p = 0.659.

### The bias mattered more than the model

Before §6R was fixed, a 3-epoch run reported AUC 0.669. After the three-way
split removed the selection bias, 30 epochs reports 0.503. Same data, same
architecture. That gap is what a reviewer would have caught, and it is the
strongest argument for having done the methodological work before the
modelling work.

### Capacity was a real problem, but not the problem

The full fine-tune reached training loss 0.0004 by epoch 30 — all 14,470 images
memorised — while validation accuracy never beat epoch 1. With 1.44M parameters
against ~50 training subjects per fold, the network learns *which subject* an
image came from, and subject identity does not transfer.

Freezing the backbone fixed that mechanically: loss plateaued at 0.060 and
validation rose across epochs rather than decaying. Out-of-sample discrimination
stayed at chance regardless.

### Fold-level results are unstable at this sample size

Topomap, per fold:

| fold_0 | fold_1 | fold_2 | fold_3 | fold_4 |
|---|---|---|---|---|
| **0.833** | 0.347 | **0.750** | 0.365 | 0.375 |

Fold-mean 0.534, pooled 0.523. Five folds of ~17 subjects produced AUCs from
0.35 to 0.83 when the true value is near chance. **Reporting `fold_0` alone
would have given "AUC 0.833, comparable to published work"** — and a single
train/test split on this cohort produces exactly that.

### Fusion reduces variance without adding signal

Fusion lands between its inputs and below the classical arm alone. Its
fold-to-fold variance is much smaller than any single arm — ±0.027 against
±0.106 to ±0.237 — because averaging uncorrelated predictors reduces variance.
That is real and expected. It adds no signal, because there was none to add.
A stable-looking number is not the same as an informative one.

### The most discriminative features are arousal and artifact

Of the 20 features the selector ranks highest, **8 are eyes-open alpha** spread
across the whole scalp (Fp1, Fp2, F4, Cz, Pz, T6, O1, O2) and **4 are gamma**
at frontopolar and occipital sites. Coherence is 45% of the feature space and
only 15% of the top 20.

Eyes-open alpha at every electrode is not an ADHD marker. It measures how well a
child suppressed alpha when they opened their eyes — arousal, drowsiness,
compliance with the instruction. Twelve subjects in this cohort show *reversed*
alpha reactivity, meaning no blocking response at all. And gamma sits inside the
50 Hz filter transition band and carries residual EMG; at Fp1/Fp2 it is eye and
forehead muscle.

This matters for interpreting published positive results. A classifier trained
on a cohort where the ADHD children are more restless will learn restlessness —
and subject-wise cross-validation will reward it, as long as restlessness is
stable within a subject. That is a more useful explanation than assuming
carelessness.

### What this establishes, and what it does not

**Establishes:** on 84 subjects, neither a transfer-learned CNN on EEG-derived
images nor an SVM on 551 engineered features discriminates ADHD from Control.
Two image representations, two capacity settings, and three feature families
spanning spectral power, aperiodic activity and connectivity.

**Does not establish:** that EEG carries no signal, that a different
architecture would fail, or that this holds at n=500. It is a bounded negative
result about specific approaches at a specific sample size — better evidenced
than most positive results in this literature, because the memorisation curve,
the fold spread, the balanced classes, the permutation tests and the paired
comparisons all agree.

**Still untested and worth pursuing:** P300 and behavioural measures. Rohani et
al. used 826 features including event-related potentials and task performance;
the VCPT trigger channel in this dataset does not encode the stimulus conditions
needed to compute them. That remains the single most plausible explanation for
the gap between 0.556 and 0.758.

## What's next

Phase 2 is closed and the pipeline runs end to end. What remains is aimed at the two things that could still change the result, and at the product.

1. **Fix the dashboard's chance band** (open issue 4).
2. **Email the dataset author about VCPT trigger coding.** Outstanding since mid-August. Rohani et al. used 826 features including P300 and behavioural measures, which need the trigger coding. This is the only route where a large, legitimate jump is plausible.
3. **Positive control: Rest vs Task** on the existing images (EC+EO vs VCPT, same folds, same §6R split). If the pipeline that scores 0.52 on ADHD scores high here, the null result is about the data, not the code. *Not* EC vs EO: 64 of 108 EC/EO boundaries were located using alpha power, so that task is circular.
4. **EEG-native architectures:** EEGNet, ShallowConvNet and Deep4Net via `braindecode`, on raw epochs instead of ImageNet transfer.
5. **A second public dataset and cross-dataset validation:** Nasrabadi et al. (IEEE DataPort, 61 ADHD / 60 Control, already named in `PROJECT.md` §3). Train on one cohort, test on the other.
6. **Web:** per-upload EEG quality-control report, PDF export, public deployment.
7. **Paper.**

---

## Repo layout

```
adhd-yolo/
├── PROJECT.md                       # full methodology and decisions; §6a indexes the §6X work IDs
├── PROGRESS.md                      # session-by-session log
├── data_pipeline/
│   ├── preprocessing.py             # EEG loading, filtering, ICA, EC/EO split
│   ├── image_conversion.py          # CWT scalograms + topomaps + EC/EO coherence maps
│   ├── subject_split.py             # subject-wise train/test/CV manifest
│   ├── build_dataset.py             # batch driver: images across the whole cohort
│   └── build_classical_features.py  # batch driver: classical features CSV per subject
├── training/
│   ├── classical_features.py        # TBR (theta/beta ratio) biomarker computation
│   ├── classical_baseline.py        # SVM arm, selection inside the CV loop
│   ├── paired_comparison.py         # McNemar + paired bootstrap between arms
│   ├── recover_fold_metrics.py      # rescore folds from saved weights, no retraining
│   ├── verify_tbr.py                # diagnostic: TBR variant comparison (read-only)
│   ├── sweep_muscle_threshold.py    # diagnostic: ICA muscle threshold sensitivity
│   ├── train_yolo_cls.py            # yolov8n-cls training + subject-level evaluation
│   ├── fusion_classifier.py         # CNN + classical fusion meta-classifier
│   └── significance_test.py         # bootstrap CI vs. published baselines
├── interpretability/
│   ├── gradcam.py                   # Grad-CAM for the trained yolov8n-cls model
│   └── clinical_plausibility.py     # checks Grad-CAM attention against known ADHD sites
├── models/                          # trained weights (gitignored); the demo needs yolov8n-cls-trained.pt
├── backend/                         # FastAPI: main.py (routes), inference.py (EDF → prediction)
├── frontend/
│   └── index.html                   # dashboard — single file, no build step
├── notebooks/                       # exploratory work
├── docs/
│   ├── STUDY_GUIDE.md               # the science: EEG, spectral analysis, ML, stats  [NOT WRITTEN YET]
│   └── jira_board.md                # epic/story breakdown
└── docker-compose.yml               # local dev, portable to EC2 later
```

## Tech stack

Signal processing: MNE-Python, MNE-Connectivity, PyWavelets, SciPy. ML: PyTorch, Ultralytics YOLOv8/v11, scikit-learn, XGBoost. Backend: FastAPI. Containers: Docker + docker-compose. Training: Google Colab (NVIDIA T4). Deployment: not started. Version control: GitHub with branch-per-feature + PR workflow. Project tracking: Jira.

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

### Running the demo (backend + dashboard)

The API needs a trained checkpoint at `models/yolov8n-cls-trained.pt` (gitignored). Use the frozen scalogram fold-0 checkpoint from the Colab run, `adhd_runs_frozen/scalogram_fold_0/weights/best.pt`, **2,963,912 bytes**. It must be a *scalogram* checkpoint: `inference.py` generates scalograms, and a topomap model would score the wrong representation without raising an error.

```powershell
cd D:\adhd-yolo
py -m uvicorn backend.app.main:app --port 8000 --reload
```

Then open `frontend\index.html` in a browser and upload an EOEC `.edf`. `http://127.0.0.1:8000/docs` lists the endpoints.

- **Demo only on held-out test subjects** (listed as `test` in `data_pipeline/splits/subject_splits.csv`). This checkpoint trained on folds 2–4 and selected its epoch on fold 1; test subjects are unseen by every model in the project. Its own held-out AUC on fold 0 was 0.486.
- **Leave VCPT empty for a fast demo.** The CNN uses EC/EO only; a VCPT upload adds a second ICA pass. `C12031144` (Control, no VCPT recording) is the fastest test subject.
- Each request classifies 30 evenly spaced epochs per condition.
- **Any single prediction is uninformative.** Pooled AUC is 0.52. Do not show the dashboard until open issue 4 is fixed.

## Workflow

`main` is always deployable — never commit directly to it. One branch per feature, small commits, PR into `main` even solo, delete the branch after merge. See `docs/jira_board.md` for the epic breakdown.

## Known limitations (stated upfront, not hidden)

- **108 subjects processed: 84 in the development set, 14 in the held-out test set** — small for a deep classifier; subject-wise validation and transfer learning are mandatory, not optional.
- **TBR does not separate the groups.** Best condition (VCPT) AUC 0.550 on the 84 development subjects, and expanding the classical arm to 551 features did not change the picture.
- **Blink removal uses Fp1/Fp2 as EOG proxies**, since no real EOG channel exists in this dataset (X1/X2 are confirmed dead). Those are genuine EEG channels, so some real frontopolar brain activity is removed alongside ocular artifact. Acceptable here because TBR is computed at F3/F4/Fz, but it is a stated limitation.
- **The TBR figures in the 2026-08-23 section predate artifact rejection.** The Phase 2 results table was computed after it.
- P300 latency/amplitude and per-condition behavioral features are unavailable pending clarification from the dataset source.
- **Grad-CAM runs against a real checkpoint only in the live demo, and the clinical-plausibility check was never run** (Phase 3 cut). The overlay shows where the network looked, not evidence.
- **`dataset/` (the original ~122k images) must not be trained on.** It predates the topography veto, when ICA was removing a median 36% of occipital alpha. All reported results use the regenerated datasets (`dataset_v2`, `dataset_v2_small`).
- **The ICA component detectors are unreliable and the veto is a guard, not a repair.** `find_bads_eog` and `find_bads_muscle` place artifact correctly only 48% of the time. The veto discards their bad output, but roughly a third of subjects consequently receive zero component exclusions — ICA effectively off, with epoch rejection as the only artifact control. `mne-icalabel`, which classifies from topography and spectrum together, is the principled replacement and has not been evaluated.
- **Every arm is at chance.** Seven measurements across four arms and three method families, AUC 0.50–0.56, every CI containing 0.5. Clean, unbiased measurements, reported as the results they are.
- **Fold-level metrics are unstable at this sample size.** ~17 subjects per fold produced AUCs from 0.35 to 0.83 in a single run. Report pooled out-of-fold values, never a single fold.
- **The most discriminative features are arousal and muscle artifact**, not pathology — see the results section. This is a property of the data, not a bug, and it is the most likely reason published positive results may not replicate.
- **n = 84 development subjects, ~17 per fold.** Sensitivity varies ±0.298 between folds and one fold scored below chance, so most fold-to-fold spread is which subjects landed where rather than model skill. Any accuracy figure from this cohort carries confidence intervals wide enough to overlap the published baselines.
- **The live demo uses one fold's checkpoint** (fold 0, held-out AUC 0.486) and scores 30 epochs per condition rather than every epoch. Any single prediction it makes is uninformative.
- This is a research/decision-support tool. It does not diagnose ADHD and is not a replacement for clinical evaluation.