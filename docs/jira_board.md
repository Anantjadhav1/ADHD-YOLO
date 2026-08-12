# Jira board setup

I can't create a real Jira board directly (no API access to your workspace), so here's the exact breakdown to enter manually — takes about 10 minutes. Link each story to its GitHub branch/PR once you open it.

## Epic 1: Data Pipeline
- [ ] Confirm raw data format and load one subject successfully
- [ ] Implement bandpass + notch filtering (`data_pipeline/preprocessing.py`)
- [ ] Implement ICA artifact removal, spot-check on 5 subjects
- [ ] Implement epoch slicing
- [ ] Implement CWT scalogram generation (Fz/Cz/Pz/F3/F4)
- [ ] Implement topographic heatmap generation (5 bands)
- [ ] Add EC/EO coherence channel
- [ ] Run full pipeline on all subjects, organize subject-wise train/val/test folders

## Epic 2: Model Training (Phase 2 — the critical checkpoint)
- [ ] Train `yolov8n-cls` baseline, ImageNet-pretrained
- [ ] Implement subject-wise grouped 5-fold CV
- [ ] Report accuracy, sensitivity, specificity, AUC per fold
- [ ] Run significance test vs. 75.8% SVM baseline
- [ ] Decision point: does baseline clear the bar? Document result either way before continuing.

## Epic 3: XAI + Fusion
- [ ] Attach Grad-CAM to final conv block
- [ ] Validate Grad-CAM attention against known ADHD electrode/frequency sites
- [ ] Build biomarker fusion meta-classifier (CNN output + TBR + P300 + behavioral features)

## Epic 4: Backend + Frontend
- [ ] FastAPI `/predict` endpoint (only after Epic 2 checkpoint passes)
- [ ] Dashboard: prediction + Grad-CAM overlay + biomarker values

## Epic 5: DevOps/Deployment
- [x] Docker skeleton (done — this repo)
- [ ] GitHub Actions CI (skeleton done, tests pending)
- [ ] AWS S3 for dataset/model storage
- [ ] EC2 (g4dn.xlarge) deployment

## Epic 6: Literature Review / Paper (new — run alongside Epic 2/3)
- [ ] PRISMA search across Scholar/IEEE Xplore/PubMed
- [ ] Screen and log identification/screening/eligibility/included counts
- [ ] Draft related-work section
- [ ] Draft methodology, results, limitations sections once Epic 2/3 results exist
