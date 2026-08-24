"""
Phase 2 — yolov8n-cls training with subject-wise CV.

Reads the manifest from subject_split.py and the images from build_dataset.py.
Never re-derives splits or re-reads raw EEG — this script's only job is
train -> evaluate -> aggregate, per PROJECT.md sec 6 Phase 2.

THE ONE RULE THIS FILE EXISTS TO ENFORCE: images are per-EPOCH (many per
subject), but 75.8%/84.5% are per-SUBJECT accuracy (see PROJECT.md sec 5a
"structural nuance"). Training is naturally epoch-level -- that's fine, more
training examples is good. But evaluation MUST aggregate epoch-level
predictions back to one prediction per subject (majority vote / mean
probability) before computing accuracy/sensitivity/specificity/AUC, or the
number reported isn't actually comparable to the baselines being cited.
aggregate_to_subject_level() below is not optional postprocessing -- skipping
it silently produces a number that looks like a result but isn't one.
"""

import argparse
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

from data_pipeline import subject_split

CLASSES = ["ADHD", "Control"]  # fixed order -> fixed index mapping everywhere below

# Ultralytics' classification defaults (pinned to 8.3.253 in
# backend/requirements.txt) are tuned for photographs and actively corrupt
# these images if left on:
#   fliplr=0.5   reverses the time axis on scalograms; mirrors left/right
#                hemisphere on topomaps, destroying F3/F4 asymmetry
#   scale=0.5    RandomResizedCrop-style zoom -> crops channels out of the
#                composite image (maps to crop range (1-scale, 1.0))
#   hsv_h/s/v    recolor a colormap where color IS the measurement
#   erasing=0.4  blanks regions of a 5-channel stack
#   auto_augment applies photograph-tuned policies (randaugment/etc) on top
#
# These are EXACTLY the augmentation keys ClassificationDataset.__init__
# reads -- derived by parsing that method's source for `args.<key>` against
# the installed version, not copied from default.yaml. The distinction
# matters: default.yaml also defines degrees/translate/shear/perspective/
# mixup/mosaic/copy_paste/bgr/cutmix, all of which are valid config keys the
# classification path never reads. Setting those to 0.0 would look like a
# fix and do nothing.
#
# This list is VERSION-SENSITIVE and must be re-derived on any Ultralytics
# upgrade. Already bitten once: `crop_fraction` was valid in 8.2.31 but was
# REMOVED by 8.3.x, where passing it raises instead of being ignored. Re-derive with:
#   import inspect, re; from ultralytics.data.dataset import ClassificationDataset
#   re.findall(r'args\.(\w+)', inspect.getsource(ClassificationDataset.__init__))
DISABLE_AUGMENTATION = dict(
    fliplr=0.0, flipud=0.0, scale=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
    erasing=0.0, auto_augment=None,
)


def subject_id_from_filename(fname: str) -> str:
    """'F08080102_EC_0000.png' -> 'F08080102'. Subject IDs are alnum-only
    (see preprocessing.parse_filename's regex), so splitting on the first
    underscore is safe and won't accidentally cut a subject ID in half."""
    return Path(fname).stem.split("_")[0]


def build_fold_dataset(images_root: str, representation: str, val_fold: str,
                        all_dev_folds: list, workdir: str) -> str:
    """
    Assemble a temp directory in the layout Ultralytics classification
    training expects (train/<class>/*.png, val/<class>/*.png), using
    symlinks -- not copies -- so this is fast and never duplicates the
    actual image data on disk.

    val_fold: the one fold held out as validation this round, e.g. "fold_2".
    all_dev_folds: every fold name EXCEPT the held-out TEST set (that's the
        whole point of subject_split.py's two-stage design -- test is never
        touched here, only used once in Phase 2's final evaluation).
    """
    ds_root = Path(workdir) / "dataset"
    if ds_root.exists():
        shutil.rmtree(ds_root)

    for cls in CLASSES:
        (ds_root / "train" / cls).mkdir(parents=True, exist_ok=True)
        (ds_root / "val" / cls).mkdir(parents=True, exist_ok=True)

    train_folds = [f for f in all_dev_folds if f != val_fold]

    for split_name, dest_split in [(val_fold, "val")] + [(f, "train") for f in train_folds]:
        for cls in CLASSES:
            src_dir = Path(images_root) / representation / split_name / cls
            if not src_dir.exists():
                continue
            for img_path in src_dir.glob("*.png"):
                # No split-name prefix here (deliberately): each subject belongs to
                # exactly ONE fold (enforced by subject_split.py's verify_no_leakage),
                # so filenames can never collide across folds even when several folds
                # land in the same "train" bucket. An earlier version prefixed
                # filenames with the fold name for traceability, which broke
                # subject_id_from_filename's first-underscore parsing and caused
                # verify_fold_dataset() to flag every subject as leaking — a false
                # positive that would have silently blocked every training run.
                dest = ds_root / dest_split / cls / img_path.name
                if not dest.exists():
                    os.symlink(img_path.resolve(), dest)

    return str(ds_root)


def verify_fold_dataset(ds_root: str, val_fold: str, images_root: str, representation: str) -> None:
    """Assert the assembled fold has no leakage: every subject_id present in
    val/ must be absent from train/, and nothing from the held-out TEST split
    should be present anywhere. Run this before every training call, not just
    when the code was written -- a future edit to build_fold_dataset() could
    silently reintroduce leakage otherwise."""
    def subjects_in(split_dir):
        ids = set()
        for cls in CLASSES:
            d = Path(ds_root) / split_dir / cls
            if d.exists():
                ids.update(subject_id_from_filename(p.name) for p in d.glob("*.png"))
        return ids

    train_subjects = subjects_in("train")
    val_subjects = subjects_in("val")
    overlap = train_subjects & val_subjects
    assert not overlap, f"LEAKAGE: subject(s) {overlap} appear in both train and val for fold {val_fold}"

    test_dir = Path(images_root) / representation / "test"
    if test_dir.exists():
        test_subjects = set()
        for cls in CLASSES:
            d = test_dir / cls
            if d.exists():
                test_subjects.update(subject_id_from_filename(p.name) for p in d.glob("*.png"))
        assert not (test_subjects & (train_subjects | val_subjects)), \
            "LEAKAGE: a TEST-split subject appears in a training fold — test must never be touched until final evaluation"


def aggregate_to_subject_level(per_image_results: list) -> pd.DataFrame:
    """
    per_image_results: list of dicts {filename, subject_id, true_label, pred_prob_adhd}
    one row per epoch-image. Returns one row per SUBJECT: mean probability
    across that subject's epochs, thresholded at 0.5 for the predicted class.
    THIS is what accuracy/sensitivity/specificity/AUC must be computed from
    -- not the raw per-image predictions -- to be comparable to 75.8%/84.5%.
    """
    df = pd.DataFrame(per_image_results)
    subj = df.groupby(["subject_id", "true_label"], as_index=False)["pred_prob_adhd"].mean()
    subj["pred_label"] = np.where(subj["pred_prob_adhd"] >= 0.5, "ADHD", "Control")
    return subj


def compute_metrics(subj_df: pd.DataFrame) -> dict:
    """Accuracy, sensitivity (recall on ADHD), specificity (recall on
    Control), AUC -- matching the reporting format PROJECT.md sec 4 step 6
    requires, at the subject level."""
    y_true = (subj_df["true_label"] == "ADHD").astype(int)
    y_pred = (subj_df["pred_label"] == "ADHD").astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")  # ADHD recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")  # Control recall
    try:
        auc = roc_auc_score(y_true, subj_df["pred_prob_adhd"])
    except ValueError:
        auc = float("nan")  # only one class present in this fold -- can't compute AUC, not a crash

    return {"n_subjects": len(subj_df), "accuracy": accuracy, "sensitivity": sensitivity,
            "specificity": specificity, "auc": auc}


def predict_val_fold(model, ds_root: str) -> list:
    """Run the trained model on every val image, return per-image results
    ready for aggregate_to_subject_level(). Kept separate from training so
    it's independently testable."""
    results = []
    for cls in CLASSES:
        val_dir = Path(ds_root) / "val" / cls
        if not val_dir.exists():
            continue
        for img_path in val_dir.glob("*.png"):
            pred = model.predict(source=str(img_path), verbose=False)[0]
            names = pred.names  # index -> class name, as the model learned it
            adhd_idx = [k for k, v in names.items() if v == "ADHD"][0]
            prob_adhd = float(pred.probs.data[adhd_idx])
            results.append({
                "filename": img_path.name,
                "subject_id": subject_id_from_filename(img_path.name),
                "true_label": cls,
                "pred_prob_adhd": prob_adhd,
            })
    return results


def run_cv(manifest_path: str, images_root: str, representation: str,
           output_dir: str, epochs: int = 30, imgsz: int = 224) -> pd.DataFrame:
    """Full subject-wise CV loop: for each dev fold, assemble the dataset,
    verify no leakage, train, evaluate at the subject level. TEST split is
    never touched here -- see PROJECT.md sec 6, final test evaluation is a
    separate one-shot step after model selection is done via these folds."""
    from ultralytics import YOLO

    manifest = subject_split.load_manifest(manifest_path)
    subject_split.verify_no_leakage(manifest)

    dev_folds = sorted(f for f in manifest["split"].unique() if f != "test")
    fold_metrics = []

    with tempfile.TemporaryDirectory() as workdir:
        for val_fold in dev_folds:
            ds_root = build_fold_dataset(images_root, representation, val_fold, dev_folds, workdir)
            verify_fold_dataset(ds_root, val_fold, images_root, representation)

            model = YOLO("yolov8n-cls.pt")  # ImageNet-pretrained -- see PROJECT.md sec 5a #2, non-negotiable given N=103
            # Resolve to an absolute path: Ultralytics treats a relative `project`
            # path as relative to ITS OWN default runs/ folder, not the CWD --
            # passing it relative silently nests output under runs/classify/<project>/
            # instead of exactly where you asked. Absolute path avoids that.
            abs_output_dir = str(Path(output_dir).resolve())
            model.train(data=ds_root, epochs=epochs, imgsz=imgsz,
                        project=abs_output_dir, name=f"{representation}_{val_fold}", exist_ok=True,
                        **DISABLE_AUGMENTATION)

            per_image = predict_val_fold(model, ds_root)
            subj_df = aggregate_to_subject_level(per_image)
            metrics = compute_metrics(subj_df)
            metrics["fold"] = val_fold
            fold_metrics.append(metrics)
            print(f"[{val_fold}] n_subjects={metrics['n_subjects']} "
                  f"acc={metrics['accuracy']:.3f} sens={metrics['sensitivity']:.3f} "
                  f"spec={metrics['specificity']:.3f} auc={metrics['auc']:.3f}")

    results_df = pd.DataFrame(fold_metrics)
    print("\nMean ± std across folds:")
    for col in ["accuracy", "sensitivity", "specificity", "auc"]:
        print(f"  {col}: {results_df[col].mean():.3f} ± {results_df[col].std():.3f}")

    return results_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=str, default="data_pipeline/splits/subject_splits.csv")
    parser.add_argument("--images-root", type=str, default="data_pipeline/images")
    parser.add_argument("--representation", type=str, default="scalogram", choices=["scalogram", "topomap"])
    parser.add_argument("--output-dir", type=str, default="training/runs")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    results = run_cv(args.manifest, args.images_root, args.representation, args.output_dir, epochs=args.epochs)
    results.to_csv(os.path.join(args.output_dir, f"{args.representation}_cv_results.csv"), index=False)


if __name__ == "__main__":
    main()