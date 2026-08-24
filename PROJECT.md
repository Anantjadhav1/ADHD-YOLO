# ADHD-YOLO — Project Guideline (v2, post-review)

Reference document. Keep this in the project root and update PROGRESS.md alongside it as work happens. This supersedes the original blueprint where the two disagree — the changes below exist because the original design had specific, fixable weaknesses.

---

## 1. What changed from the original blueprint, and why

| Original plan | Problem | Fix |
|---|---|---|
| YOLO detects bounding boxes around "theta bursts" / "P300 drops" | No independent ground truth for boxes — if boxes are generated from the same TBR/latency thresholds the biomarker engine already uses, the detector just re-learns a rule you already wrote. A reviewer's first question will be "what does detection add over the rule?" | **Default: drop true object detection for the diagnostic claim.** Use `yolov8n-cls` / `yolov11n-cls` for classification, Grad-CAM for localization/interpretability. Keep bounding-box detection as an optional exploratory add-on (see §5), not a headline result. |
| 80/10/10 random split | With 103 subjects, a single random split can swing reported accuracy 5–10 points depending on who lands in test. Random epoch-level splits also risk subject leakage across train/test. | Subject-wise grouped k-fold (5-fold minimum) or leave-one-subject-out. Never split at the epoch level. |
| Train YOLO from scratch on ~103 subjects | Deep detectors are typically trained on thousands of images; 103 subjects is a real overfitting risk for a from-scratch CNN backbone. | Transfer learning is mandatory, not optional: pretrain/fine-tune from ImageNet weights, and ideally warm-start further using the public Nasrabadi ADHD-EEG dataset (61 ADHD / 60 control, IEEE DataPort) before fine-tuning on your primary cohort. |
| "Exceed 75.8%" as the target | 75.8% is a soft bar — other studies in this exact literature report 90–100% on different (often smaller, less generalizable) cohorts. Beating it alone isn't a strong result. | Report accuracy **and** sensitivity, specificity, AUC, plus a significance test (McNemar's or bootstrap CI) against the 75.8% SVM baseline. Discuss *why* prior studies vary so much (task design, age range, sample size) rather than present 75.8% as the one number to beat. |
| Two isolated XAI outputs (Grad-CAM heatmaps + biomarker math engine) | Running in parallel without combining them wastes the paper's strongest finding from the source literature: combinatorial biomarkers outperform individual ones. | Late-fusion: feed the CNN's output probability alongside TBR, P300 latency, and behavioral features (omission/commission errors, reaction time) into a small meta-classifier (logistic regression or XGBoost). This is your hybrid model, and it directly answers a claim from the original paper's own conclusion. |
| Diagnostic framing | Overclaiming "diagnoses ADHD" is a red flag for both a paper reviewer and basic clinical-safety judgment. | Frame everywhere as a decision-support / research tool, not a diagnostic replacement. State this explicitly in the paper's limitations section. |

---

## 2. Project goals and how "done" is defined

This serves two audiences at once — keep both bars in view, don't let one crowd out the other.

**Academic bar (satisfies your professor):**
- PRISMA-documented literature review (real search, real screening numbers, real included-study count)
- Subject-wise validated results reported with accuracy, sensitivity, specificity, AUC
- Statistical comparison against the 75.8% SVM baseline
- Honest limitations section (sample size, generalizability, non-diagnostic framing)
- Novelty statement that's accurate: "first application of YOLO-based image classification/detection to pediatric ADHD EEG/ERP-derived representations, combined with quantitative biomarker fusion" — not "novel technique," since YOLO-on-spectrograms exists in other domains (confirmed against sleep-apnea EEG literature).

**Engineering bar (satisfies the portfolio goal):**
- Working FastAPI backend serving the trained model
- A dashboard showing prediction + Grad-CAM overlay + biomarker values with plain-language explanation
- Dockerized, deployed (even a minimal single EC2 instance counts — don't over-invest here before the model works)
- Clean GitHub history: branches, PRs, real commit messages
- Jira board reflecting actual sprint work, not filled in retroactively

**The single most important checkpoint:** does the classifier beat 75.8% with subject-wise validation, and is that result statistically meaningful? If no, this becomes a negative/exploratory result paper (still publishable, different framing) — figure this out early (see roadmap, Phase 2) before building anything downstream of it.

---

## 3. Data plan

- **Primary dataset:** the Rohani et al. cohort you'll provide — 103 children (49 ADHD, 54 control), 19-channel 10–20 system, 500 Hz, EC/EO resting state + 400-trial Go/NoGo ERP task. Cite as: Rohani, F., Khoshhal Roudposhti, K., Taheri, H., Mashhadi, A., & Mueller, A. (2022). *Journal of Computer and Knowledge Engineering, 5*(2), 1–10.
- **Secondary/validation dataset:** Nasrabadi et al., "EEG data for ADHD/Control children," IEEE DataPort, doi: 10.21227/rzfh-zn36 — 61 ADHD / 60 control, 19-channel, 128 Hz, open access. Use for: (a) pretraining the CNN backbone before fine-tuning on the primary cohort, (b) an external validation check — if your model trained on Rohani generalizes reasonably to Nasrabadi's differently-recorded cohort, that's a real strength to report.
- Confirm before Sprint 1: file format (.edf, .mat, .csv?), whether it's raw continuous signal or pre-epoched, and whether channel/trial-level ADHD/control labels are included per the metadata.csv structure described in the dataset README.

---

## 4. Revised technical pipeline

1. **Preprocessing** (MNE-Python): 0.5–50 Hz bandpass, 50 Hz notch, ICA artifact removal, 1–2s epoch windows. (Unchanged from original blueprint — this part was sound.)
2. **Augmentation — signal domain, before image conversion:** time-jitter, amplitude scaling, small time-warps, channel dropout. Do NOT augment after converting to images (e.g. rotating a scalogram) — that destroys the frequency-axis meaning.
3. **1D→2D conversion:**
   - CWT scalograms (Complex Morlet, Fz/Cz/Pz/F3/F4) — unchanged.
   - Topographic heatmaps across 5 bands — unchanged.
   - **New:** add an EC/EO coherence (functional connectivity) representation as an additional input channel — the source paper specifically flags coherence as one of its five retained feature groups; it's currently missing from the image pipeline.
4. **Model:**
   - `yolov8n-cls` / `yolov11n-cls`, ImageNet-pretrained, fine-tuned. Pretrain further on Nasrabadi dataset if time allows.
   - Grad-CAM on the final conv block for visual explainability.
   - *Optional stretch goal only, not core deliverable:* true YOLO object detection, but only if you get independent expert-labeled bounding boxes (a neuroscientist annotating a subset) — otherwise skip it and rely on Grad-CAM for localization.
5. **Fusion layer:** meta-classifier (logistic regression/XGBoost) combining CNN output probability + TBR + P300 latency/amplitude + behavioral features.
6. **Validation:** subject-wise grouped 5-fold CV or leave-one-subject-out. Report mean ± std across folds (matching the source paper's 10-trial averaging convention for comparability).
7. **Interpretability sanity check:** confirm Grad-CAM attention concentrates near known ADHD-relevant sites (frontal Fz/F3/F4 for theta/beta, Cz/Pz for P300) before presenting it as a clinical-plausibility result.

---

## 5. Literature review (PRISMA)

Run real searches on Google Scholar, IEEE Xplore, and PubMed with terms like: `ADHD EEG deep learning`, `ADHD EEG spectrogram CNN`, `EEG object detection YOLO`, `EEG image classification ADHD children`. Track real counts at each stage (identified → screened → eligible → included). Target ~15–25 included studies, incorporating the six studies already summarized in Table 3 of the Rohani paper plus newer 2023–2026 work. This becomes the paper's related-work section and the justification for why 75.8% is your comparison point.

---

## 5a. Accuracy priority — ranked by actual expected impact

Not everything in §4 moves accuracy equally. Ranked by leverage for this specific dataset (small N, existing strong classical baselines):

1. **Classical biomarker fusion — highest leverage, do this in Phase 2, not Phase 3.** The paper's own Table 2 shows HSSL + Logistic Regression hit 84.5% — higher than the 75.8% "approved" baseline you're benchmarking against, using engineered features and no deep learning at all. This means the ceiling for accuracy on this exact dataset is already known to be north of 75.8% through relatively simple means. A model that fuses CNN output with TBR, P300 latency/amplitude, and behavioral features has a real shot at beating both 75.8% and 84.5% — build and evaluate this alongside the raw CNN in Phase 2, not as a later add-on.
2. **Transfer learning / pretraining** — with 103 subjects, an ImageNet-pretrained (and ideally Nasrabadi-pretrained) backbone will outperform training from scratch by a wide margin. Non-negotiable, not optional.
3. **Multi-representation input** (scalogram + topomap + EC/EO coherence) — more signal in, generally more accuracy out, at low implementation cost since you're already generating two of the three.
4. **Signal-domain augmentation** — helps generalization on small N, moderate impact, cheap to add.
5. **Ensembling multiple backbones** (e.g. yolov8n-cls + yolov11n-cls, or CNN + classical fused separately then averaged) — real but smaller gains, worth it only after 1–4 are done and you have time left.
6. **NOT bounding-box detection.** Confirmed again here: it doesn't help accuracy, adds a circular-label risk (§1), and spends model capacity on a task the paper's classical methods already solve better with less complexity.

**One structural nuance to get right:** the classical features (826 → 113 selected) are computed **per subject** (one feature vector per person), while the CNN operates on **per-epoch images** (many images per subject). Before comparing CNN or fused accuracy against 75.8%/84.5%, aggregate epoch-level predictions up to one prediction per subject — majority vote or averaged probability across that subject's epochs — otherwise you're comparing apples to oranges and any number you report isn't actually comparable to the baseline you're citing.

## 6. Revised phased roadmap (reordered — validate the core idea before building infrastructure)

**Phase 0 — Setup (now):** repo skeleton, GitHub branch workflow, Jira board with 5 epics, Docker skeleton, PROJECT.md/PROGRESS.md.

**Phase 1 — Preprocessing pipeline:** get raw data loading, filtering, ICA, epoching working end-to-end on a handful of subjects first, then scale to all 103.

**Phase 2 — Baseline result (the make-or-break phase, now includes fusion):**
- CWT + topomap generation → `yolov8n-cls` baseline, subject-wise CV, epoch-to-subject aggregation.
- In parallel: replicate the classical feature pipeline (TBR, P300, behavioral features) — this also validates your own preprocessing against the paper's reported 75.8%/84.5% before you trust anything built on top of it.
- Fuse CNN output + classical features in a meta-classifier.
- Report all three (CNN alone, classical alone, fused) with accuracy, sensitivity, specificity, AUC, and significance tests against both the 75.8% and 84.5% reference points.
- Do not proceed to Phase 3 until this is done and understood — whichever of the three wins, or if none beat the baselines, that's still a real, reportable result.

**Phase 3 — XAI:** Grad-CAM, EC/EO coherence channel, clinical-plausibility sanity check against known ADHD electrode/frequency sites.

**Phase 4 — Literature review + paper writing:** run concurrently with Phase 3 once you know your own numbers to discuss against the literature.

**Phase 5 — Engineering/deployment:** FastAPI backend, dashboard, Docker, AWS deployment, Claude API report generation.

---

## 7. Immediate next steps (this week)

1. Share the raw dataset (or IEEE DataPort link/DOI) so preprocessing can actually start.
2. Confirm dev environment (local machine + OS, or cloud from day one) — determines the Docker/dependency skeleton.
3. Set up repo: `main` branch protected, `feature/*` branch convention, PR-per-feature even solo.
4. Set up Jira board: Epic 1–5 as listed in the kickoff doc, Phase 0–2 tasks entered as the first sprint.
5. Decide now, in writing in this doc: are we doing classification-only + Grad-CAM (recommended default), or investing in real expert-labeled bounding boxes for true detection? This changes what Phase 2/3 actually build.

---

## 8. Decisions (locked in)

- **Detection framing: classification-only + Grad-CAM.** No object detection, no bounding-box labeling needed. `yolov8n-cls` is the model; Grad-CAM handles localization/interpretability. This removes the circularity problem in §1 entirely — nothing to revisit unless Phase 2 results specifically motivate revisiting it.
- **Dev environment: local machine, Docker from day one.** Everything runs in containers locally now, so the move to AWS EC2 later is a deployment target change, not a rewrite. See `docker-compose.yml` in the repo skeleton.

## 9. Still open

- Whether Phase 4 (paper writing) is drafted together here, or handled by you separately once results are in.
