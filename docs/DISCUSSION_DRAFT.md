# Discussion — working draft

**Status.** Drafted 2026-10-28. Argument and structure only; prose still to write. Results are cited by section of `docs/RESULTS_DRAFT.md` (R1–R6), methods by section of `docs/METHODS_DRAFT.md`. Every citation below is one already recorded in `PROGRESS.md` (2026-08-24 and the Review 1 entry) and must be checked against the primary source; entries marked **[verify]** have details that disagree between notes.

---

## D1. Principal finding

One paragraph, stated without hedging and without overreach:
- On Rohani et al., no method separated ADHD from Control: five arms and four representations, AUC 0.50–0.57, every interval containing 0.5 (R1, R5-H3).
- On Nasrabadi et al., a model fixed in advance did separate them modestly (AUC 0.637, R5-H1a), but the model trained on one cohort failed on the other (R5-H1b).
- So the signal is cohort-specific. No EEG marker of ADHD that holds across these two cohorts was found.

## D2. Why the null on Rohani et al. is credible

The argument a reader needs before accepting a negative result:
1. **The code works.** Every arm separates rest from task in the same children (R2). A pipeline fault would not pass these controls and then fail only on diagnosis.
2. **Not one method.** Spectral power, time-frequency images, scalp maps, connectivity and covariance geometry; linear and non-linear models; frozen and fully fine-tuned networks (R1, R5-H3).
3. **Not QC.** No exclusion policy moves any interval off 0.5 (R3).
4. **Not a selection artefact.** The three-way split removed the bias that had produced 0.669 (Methods §6; R1).
5. **Paired tests.** The arms do not differ from one another, and they agree with each other near chance (κ ≤ 0.28). They are not each finding a weak version of one signal (R1).

What it does not establish: that EEG carries no ADHD information; that a different architecture or n = 500 would fail; that P300 and behaviour would fail (D5).

## D3. Agreement with the literature on theta/beta ratio and spectral markers

- **Arns, Conners & Kraemer (2013)**, *J Atten Disord* 17(5):374–383. A meta-analysis of 9 studies. The TBR group difference shrank across publication years as TBR rose in control groups; the authors concluded TBR is not a reliable diagnostic measure.
- **Arns et al. (2016)**, *JCPP* editorial on the FDA clearance of the NEBA device. This is the clinical context of the claim being tested.
- **(2020)**, *Appl Psychophysiol Biofeedback*, iSPOT-A and ICAN: five spectral algorithms gave different TBR values, and none separated ADHD from controls **[verify authors]**.
- **Arns et al. (2024)**, *Appl Psychophysiol Biofeedback*, N = 417: no diagnostic value of TBR.
- **Strzelczyk, Vetsch & Langer (2026)**, *eLife*: a multiverse analysis that found a TBR difference in 1.91% of 576 specifications, attributing reported elevations to aperiodic activity and individual alpha frequency **[verify N: PROGRESS records both "1,499 + 381" and "primary sample 1,122"]**.
- **Coolidge et al. (2007)**: against other disorders, sensitivity 50% and specificity 36%.

This project's TBR (best condition, AUC 0.550) and its aperiodic and IAF features (Methods §5) are consistent with that consensus. The point to make: the consensus formed around one marker; this project extends it to learned representations on the same data.

## D4. Why published accuracies are higher

- **Window-level leakage.** Aggul (2026), *Frontiers in Neuroinformatics*: 95.59% accuracy with random-window splits against 60.98% with participant-disjoint evaluation on the same cohort **[verify]**. Every split here is subject-wise (Methods §2).
- **Selection on the scored fold.** Here it inflated the AUC from 0.503 to 0.669 (R1). It is invisible unless a three-way split is used.
- **Single splits at small n.** The fold AUCs here run from 0.347 to 0.833 (R1). A single split could have reported 0.833.
- **Accuracy vs AUC.** Rohani et al.'s 75.8% and 84.5% are accuracies, on a distribution of 103 subjects; the files distributed give 108 (Methods §1). The comparison is not like for like.
- **Features this project could not compute.** Rohani et al. used P300 and behavioural measures (part of 826 features). The distributed VCPT marker channel does not encode stimulus conditions (Methods §1). This is the single most plausible legitimate reason for the gap, and it remains untested.

## D5. What the Nasrabadi result means, and what it does not

- **For it:** pre-registered, the model fixed, α corrected, the CI above 0.5 (R5-H1a). This is the most defensible positive in the project.
- **Against reading it as ADHD physiology:**
  - Task, site, amplifier, sampling rate (128 vs 500 Hz) and protocol all differ between cohorts.
  - The amplitude units are unknown, and the rejection threshold was matched by behaviour (Methods §9b).
  - Within one cohort, any systematic difference in how the two groups were recorded is indistinguishable from diagnosis.
  - The R2 controls show artifact (frontopolar/temporal muscle) is learnable by these models, and the ADHD-relevant features on Rohani et al. were arousal and artifact (R1).
- **Transfer failed (R5-H1b),** which is what a cohort-specific signal predicts. A shared physiological marker would predict otherwise, even with attenuation.
- **Open question for the text:** whether H1b failed because the marker is absent in Rohani et al., or because the 100 task features differ in meaning between a Go/NoGo task and the Nasrabadi task **[verify task]**. Both readings forbid a cross-cohort claim.

## D6. Methodological contributions

These hold whatever the result:
1. Positive controls as a standard for negative EEG classification results (R2).
2. The three-way split, with the size of the bias it removes measured (R1).
3. Pre-registration with append-only deviations, written before a second dataset was opened (Methods §9b).
4. Units-agnostic threshold matching across datasets, decided without labels (Methods §9b, D2–D3 of the registration).
5. Image encodings that preserve the measurement: an absolute-power channel, TBR recoverable (r = 1.000), and augmentations that would distort physiology disabled (Methods §4, §6).

## D7. Limitations

Carry over Methods §11 and group them for the reader:
- **Sample:** n = 84/121, intervals of about ±0.13 AUC; only large effects can be excluded.
- **Data:** no P300 or behaviour; alpha-derived EC/EO boundaries; frontopolar channels doubling as EOG; the 108 vs 103 discrepancy; Nasrabadi units undocumented.
- **Analysis:** gamma in the filter transition band; 1.5 s epochs; no QC exclusion; fusion unstandardised; post-hoc R4.
- **Record:** pre-registration as a git commit, not an immutable registry.

## D8. Future work

1. P300 and behavioural features, once the dataset author supplies the VCPT stimulus coding. This needs its own registration before the coding is received (registration §5).
2. Multi-site data with harmonised acquisition, which is the only design that can separate diagnosis from site.
3. Models that can be audited for artifact. Attribution was cut here because it is meaningless at chance (README Phase 3). It becomes meaningful only once a model works.

## D9. Conclusion

Three or four sentences, mirroring D1. End on the bounded claim: on two public paediatric cohorts, under subject-wise validation and pre-registration, no cross-cohort EEG marker of ADHD was found. The within-cohort signal that was found is not separable from how that cohort was recorded.