# Results — working draft

**Status.** Drafted 2026-10-28. Every number is copied from a committed file named beside it; regenerate from that file rather than editing a number here. Structure follows the order in which a reader needs the evidence: the null, then why it is not the code, then why it is not QC, then the pre-registered phase. A scaffold for the thesis chapter, not finished prose. Methods: `docs/METHODS_DRAFT.md`.

---

## R1. ADHD vs Control on Rohani et al.: every arm at chance

Common cohort: 84 development subjects (42 ADHD / 42 Control), pooled out-of-fold AUC, 95% bootstrap CI, one-sided permutation p. Source: `docs/results_table.csv` (`py -m training.results_table`), fusion from `docs/linear_sensitivity.csv`.

| arm | AUC | 95% CI | p |
|---|---|---|---|
| CNN scalogram, frozen backbone | 0.522 | 0.397–0.650 | 0.364 |
| CNN topomap, frozen backbone | 0.523 | 0.396–0.650 | 0.360 |
| Classical SVM, 301 power features | 0.522 | 0.399–0.647 | 0.375 |
| Classical SVM, 551 features with coherence | 0.556 | 0.431–0.682 | 0.201 |
| Fusion (scalogram CNN + 551 features) | 0.537 | 0.406–0.667 | 0.277 |

Without intervals: theta/beta ratio, best condition (VCPT), 0.550; CNN scalogram fully fine-tuned, 0.503.

**To state:** every interval contains 0.5. On its own 91 subjects the 551-feature arm scores 0.538 (0.416–0.660, p = 0.268). Rohani et al.'s 75.8% and 84.5% are accuracies, not AUCs.

**Paired comparisons** (`docs/paired_comparison.csv`, n = 84; "classical" is the 551-feature arm):

| | ΔAUC | 95% CI | p | McNemar p | κ |
|---|---|---|---|---|---|
| scalogram − topomap | −0.001 | −0.160 to 0.163 | 0.992 | 1.00 | 0.02 |
| scalogram − classical | −0.035 | −0.182 to 0.110 | 0.639 | 0.86 | 0.28 |
| topomap − classical | −0.034 | −0.187 to 0.114 | 0.659 | 0.74 | 0.12 |

No arm differs from another. Agreement between arms is near chance (κ 0.02–0.28): they are not finding the same weak signal.

**Supporting observations (figures to make):**
- Fold AUCs for the topomap arm are 0.833, 0.347, 0.750, 0.365, 0.375. A single split would have reported 0.833.
- The full fine-tune memorised its training set (loss 0.0004) while validation never beat epoch 1. Freezing stopped the memorisation, but not the null.
- Selection bias: before the three-way split, a 3-epoch run reported 0.669; after it, 0.503.
- The highest-ranked features are eyes-open alpha (8 of the top 20) and gamma at frontopolar/occipital sites (4 of 20): arousal and artifact.

## R2. Positive controls: the same code finds a difference that exists

Rest (EO) vs Task (VCPT) in the same children; within-child permutation. Sources: `docs/positive_control_eo_summary.csv`, `docs/positive_control_cnn_scalogram_eo_summary.csv`, `docs/positive_control_cnn_topomap_eo_summary.csv`.

| Rest (EO) vs Task | classical | CNN scalogram | CNN topomap |
|---|---|---|---|
| children | 85 | 79 | 79 |
| AUC | 0.686 | 0.633 | 0.747 |
| 95% CI | 0.637–0.741 | 0.582–0.686 | 0.693–0.804 |
| p | 0.0002 | 0.0002 | 0.0002 |
| task above own rest | 80% | 81% | 86% |
| same code, ADHD vs Control | 0.538 (91) | 0.522 | 0.523 |

Secondary contrast, classical EC+EO vs Task: 0.771 (0.723–0.815), `docs/positive_control_eceo_summary.csv`.

**To state:** every arm separates rest from task, and none separates ADHD from Control. The separating features are partly physiology (parietal alpha suppression) and partly artifact (frontopolar beta/gamma). The arm with the most access to frontopolar and temporal sites, the topomap, scores highest.

## R3. Quality control does not change the conclusion

Fixed out-of-fold predictions, rescored under four exclusion policies. Source: `docs/qc_sensitivity.csv`.

| policy | n (ADHD) | scalogram | topomap | classical 551 |
|---|---|---|---|---|
| none (as reported) | 84 (42) | 0.522 | 0.523 | 0.556 |
| drop ICA-cap subjects | 63 (31) | 0.549 | 0.514 | 0.498 |
| drop top-decile rejection | 65 (30) | 0.540 | 0.562 | 0.621 |
| drop either | 52 (24) | 0.598 | 0.561 | 0.586 |

**To state:** every interval contains 0.5 under every policy. One of the twelve cells has p = 0.048, which is what chance gives.

## R4. Post hoc and exploratory: a standardised linear model

Source: `docs/linear_sensitivity.csv`. Reported apart from R1 because it was specified after R1 was known.

| on the same 84 | AUC | 95% CI | p (unadjusted) |
|---|---|---|---|
| fusion as published (unscaled) | 0.537 | 0.406–0.667 | 0.277 |
| fusion, standardised | 0.626 | 0.502–0.745 | 0.025 |
| 551 features only, standardised | 0.627 | 0.504–0.745 | 0.024 |

**To state:** it was specified after at least eight analyses of these children; its fold AUCs range 0.347–0.764; its weights fall on eyes-open coherence and temporal/occipital gamma. It was then tested once on untouched data (R5, H2).

## R5. Pre-registered analyses

Registered 2026-10-21 (`docs/PREREGISTRATION.md`). Bonferroni α = 0.0167 over {H1a, H1b, H3}. Source: `docs/preregistered_results.csv`.

| | data | n (ADHD) | AUC | 95% CI | p | decision |
|---|---|---|---|---|---|---|
| H1a | within Nasrabadi et al. | 121 (61) | 0.637 | 0.533–0.734 | 0.0026 | **positive** |
| H1b | Rohani → Nasrabadi | 121 (61) | 0.432 | 0.328–0.534 | 0.904 | negative |
| H3 | Riemannian, Rohani | 84 (42) | 0.566 | 0.440–0.688 | 0.149 | negative |
| H2 | R4 model, Rohani test split | 17 (8) | 0.514 | 0.214–0.808 | — | estimate only |

**To state:**
- H1a: a group difference exists within a second cohort and was found by a model fixed in advance. It is a cohort difference, not an ADHD marker: task, site, equipment and artifact are not separable.
- H1b: a model trained on one cohort does not transfer to the other.
- H3: a fourth representation, covariance geometry, is also at chance on Rohani et al.
- H2: neither confirms nor refutes R4.

## R6. Summary for the discussion

1. On Rohani et al., five arms, four representations and two capacity settings all give AUCs of 0.50–0.57, each with an interval containing 0.5 (R1, R5-H3).
2. The code is not the reason: every arm passes its positive control (R2).
3. QC is not the reason: no exclusion policy changes the conclusion (R3).
4. A modest pre-registered signal exists inside a second cohort (R5-H1a) but does not transfer (R5-H1b).
5. The defensible conclusion is that the signal is cohort-specific; no EEG marker of ADHD that holds across these cohorts was found.
6. Untested: P300 and behavioural measures, which need the VCPT stimulus coding (Methods §1).