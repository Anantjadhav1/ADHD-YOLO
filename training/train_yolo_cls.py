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
import warnings
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


def collect_oof_predictions(fold_frames: list, manifest: pd.DataFrame) -> pd.DataFrame:
    """
    Stack each fold's subject-level validation predictions into ONE
    out-of-fold (OOF) table: one row per development subject, holding the
    probability predicted by the fold in which that subject was held out.

    This is what fusion_classifier.run_fusion_cv() needs as its
    cnn_subject_probs argument. Without it the CNN half and the classical
    half of Phase 2 never meet -- run_cv() used to compute exactly these
    numbers per fold and then discard them, so the fused arm of the
    CNN-alone / classical-alone / fused comparison could not be produced at
    all.

    Why concatenating val folds is legitimately "out-of-fold": every subject
    belongs to exactly one fold (enforced by subject_split.verify_no_leakage),
    and a fold's predictions come from a model that never saw that fold's
    subjects during training. So each row is a prediction on a subject unseen
    by the model that produced it, which is the property the fusion
    meta-classifier needs -- fitting it on in-sample CNN probabilities would
    let it learn to trust an over-confident feature.

    Asserts one row per subject: a duplicate means a subject was validated in
    two different folds, i.e. the split itself leaked. That is worth crashing
    on rather than silently averaging away.
    """
    if not fold_frames:
        raise ValueError("No fold predictions collected -- cannot build an OOF table.")

    oof = pd.concat(fold_frames, ignore_index=True)

    dupes = oof["subject_id"][oof["subject_id"].duplicated()].unique()
    assert len(dupes) == 0, (
        f"LEAKAGE: subject(s) {list(dupes)} were validated in more than one fold. "
        "Each subject must belong to exactly one fold -- check subject_split.py's manifest."
    )

    # Not an error, but must not pass silently: a dev subject with no OOF row
    # produced no images (preprocessing failure, or skipped as EC/EO-ambiguous
    # by build_dataset.py). Those subjects are absent from the fusion table
    # too, so the arms of the comparison would be scored on different cohorts.
    dev_subjects = set(manifest.loc[manifest["split"] != "test", "subject_id"])
    missing = dev_subjects - set(oof["subject_id"])
    if missing:
        warnings.warn(
            f"{len(missing)} development subject(s) have no CNN prediction and will be "
            f"absent from the fusion table: {sorted(missing)}. Check build_log.csv for "
            "preprocessing failures or subjects skipped as EC/EO-ambiguous."
        )
    return oof


def predict_class_dirs(model, root) -> list:
    """Run the model over a directory laid out as <root>/<class>/*.png and
    return per-image results ready for aggregate_to_subject_level().

    Split out from predict_val_fold so the exact same prediction path serves
    both the CV folds (assembled temp dir) and the held-out test split (read
    straight from images_root) -- two prediction code paths that could drift
    apart is precisely how a final test number stops being comparable to the
    CV numbers it is reported beside."""
    results = []
    for cls in CLASSES:
        cls_dir = Path(root) / cls
        if not cls_dir.exists():
            continue
        for img_path in cls_dir.glob("*.png"):
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


def predict_val_fold(model, ds_root: str) -> list:
    """Run the trained model on every val image of an assembled fold dataset."""
    return predict_class_dirs(model, Path(ds_root) / "val")


def evaluate_on_test(manifest_path: str, images_root: str, representation: str,
                     output_dir: str, epochs: int = 30, imgsz: int = 224,
                     inner_val_fold: str = None) -> tuple:
    """
    The one-shot final evaluation on the held-out TEST split, which
    subject_split.py carves out and which nothing consumed until now -- the
    two-stage design existed but only half of it was wired up.

    Run this ONCE, after run_cv() has settled the configuration. Every time
    the test set informs a decision (architecture, epochs, thresholds) it
    stops being a held-out set and starts being a second validation set, and
    the number it produces stops being an estimate of generalisation.

    Design, and the alternatives rejected:
      * Trains ONE final model on the development subjects and evaluates it
        on test. This is the conventional choice, and it also produces the
        single trained checkpoint that Phase 3 (Grad-CAM) and Phase 5
        (/predict) both need and neither currently has.
      * Ensembling the k fold models was considered -- it costs no extra
        training, which is attractive on CPU -- but "the model" then becomes
        an ensemble, which muddies the Grad-CAM interpretability claim that
        is the whole point of Phase 3.
      * Picking the best-scoring fold model was rejected outright: those
        models were selected using their own validation folds, so choosing
        among them by that score and then reporting a test number carries
        the selection bias straight through.

    inner_val_fold: one development fold held out purely so Ultralytics can
        pick `best.pt` by validation accuracy. It must NOT be the test split.
        Ultralytics always selects a checkpoint using whatever it is given as
        val, so handing it the test split would be selection-on-test -- the
        same optimistic bias flagged for the CV loop, but on the one number
        that is supposed to be clean. Defaults to the last dev fold.

    Returns (metrics_dict, per_subject_df).
    """
    from ultralytics import YOLO

    manifest = subject_split.load_manifest(manifest_path)
    subject_split.verify_no_leakage(manifest)

    dev_folds = sorted(f for f in manifest["split"].unique() if f != "test")
    if not dev_folds:
        raise ValueError("Manifest has no development folds -- nothing to train on.")
    if "test" not in set(manifest["split"]):
        raise ValueError("Manifest has no 'test' split -- nothing to evaluate against.")

    if inner_val_fold is None:
        inner_val_fold = dev_folds[-1]
    if inner_val_fold not in dev_folds:
        raise ValueError(
            f"inner_val_fold={inner_val_fold!r} is not a development fold "
            f"(dev folds: {dev_folds}). It must never be the test split."
        )

    test_dir = Path(images_root) / representation / "test"
    if not test_dir.exists():
        raise FileNotFoundError(
            f"No test images at {test_dir} -- run build_dataset.py so the test "
            "split's images exist before the final evaluation."
        )

    with tempfile.TemporaryDirectory() as workdir:
        # Reuses the CV assembler, so the final model's training data is built
        # by exactly the same code (and passes the same leakage check) as every
        # CV fold. build_fold_dataset only ever reads dev folds, so the test
        # split cannot be pulled in by construction.
        ds_root = build_fold_dataset(images_root, representation, inner_val_fold, dev_folds, workdir)
        verify_fold_dataset(ds_root, inner_val_fold, images_root, representation)

        model = YOLO("yolov8n-cls.pt")
        abs_output_dir = str(Path(output_dir).resolve())
        model.train(data=ds_root, epochs=epochs, imgsz=imgsz,
                    project=abs_output_dir, name=f"{representation}_final", exist_ok=True,
                    **DISABLE_AUGMENTATION)

        per_image = predict_class_dirs(model, test_dir)

    if not per_image:
        raise ValueError(f"No test images found under {test_dir} -- cannot evaluate.")

    subj_df = aggregate_to_subject_level(per_image)

    # Belt-and-braces: the assembled training set must share no subject with
    # test. verify_fold_dataset already checks this, but this is the one
    # number in the project that gets reported as a generalisation estimate,
    # so it is worth asserting against the manifest independently rather than
    # trusting a single upstream check.
    train_subjects = set(manifest.loc[manifest["split"].isin(dev_folds), "subject_id"])
    overlap = train_subjects & set(subj_df["subject_id"])
    assert not overlap, (
        f"LEAKAGE: subject(s) {sorted(overlap)} are in both the training folds and the "
        "test split. The final test number would be meaningless."
    )

    metrics = compute_metrics(subj_df)
    print(f"\n=== HELD-OUT TEST ({representation}) — one-shot, do not tune against this ===")
    print(f"  n_subjects  {metrics['n_subjects']}")
    for k in ["accuracy", "sensitivity", "specificity", "auc"]:
        print(f"  {k:11} {metrics[k]:.3f}")
    print(f"  (final model trained on {len(dev_folds) - 1} dev folds, "
          f"inner val = {inner_val_fold})")

    return metrics, subj_df


def run_cv(manifest_path: str, images_root: str, representation: str,
           output_dir: str, epochs: int = 30, imgsz: int = 224) -> tuple:
    """Full subject-wise CV loop: for each dev fold, assemble the dataset,
    verify no leakage, train, evaluate at the subject level. TEST split is
    never touched here -- see PROJECT.md sec 6, final test evaluation is a
    separate one-shot step after model selection is done via these folds.

    Returns (fold_metrics_df, oof_df). The second element is the per-subject
    out-of-fold probability table the fusion meta-classifier consumes -- see
    collect_oof_predictions(). It is returned rather than only printed
    because it is an INPUT to the next stage, not a report artifact."""
    from ultralytics import YOLO

    manifest = subject_split.load_manifest(manifest_path)
    subject_split.verify_no_leakage(manifest)

    dev_folds = sorted(f for f in manifest["split"].unique() if f != "test")
    fold_metrics = []
    oof_frames = []

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
            # Keep this fold's subject-level predictions -- these ARE the
            # out-of-fold probabilities the fusion classifier needs. They used
            # to be computed here, used once for metrics, and dropped.
            oof_frames.append(subj_df.assign(fold=val_fold))
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

    oof_df = collect_oof_predictions(oof_frames, manifest)
    print(f"\nCollected out-of-fold predictions for {len(oof_df)} subjects.")

    return results_df, oof_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=str, default="data_pipeline/splits/subject_splits.csv")
    parser.add_argument("--images-root", type=str, default="data_pipeline/images")
    parser.add_argument("--representation", type=str, default="scalogram", choices=["scalogram", "topomap"])
    parser.add_argument("--output-dir", type=str, default="training/runs")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--evaluate-on-test", action="store_true",
        help="Run the ONE-SHOT final evaluation on the held-out test split. Off by "
             "default and deliberately opt-in: this is not a metric to iterate against. "
             "Every run that informs a decision turns the test set into a second "
             "validation set. Run it once, when the configuration is settled.",
    )
    parser.add_argument(
        "--inner-val-fold", type=str, default=None,
        help="Dev fold used only so Ultralytics can select best.pt when training the "
             "final model (default: last dev fold). Never the test split.",
    )
    parser.add_argument(
        "--skip-cv", action="store_true",
        help="Skip the CV loop. Only meaningful together with --evaluate-on-test, "
             "when CV has already been run and its outputs are on disk.",
    )
    args = parser.parse_args()

    if args.skip_cv and not args.evaluate_on_test:
        parser.error("--skip-cv skips everything unless --evaluate-on-test is also given.")

    os.makedirs(args.output_dir, exist_ok=True)

    if not args.skip_cv:
        results, oof = run_cv(args.manifest, args.images_root, args.representation,
                              args.output_dir, epochs=args.epochs)
        results.to_csv(os.path.join(args.output_dir, f"{args.representation}_cv_results.csv"), index=False)

        # The fusion stage's actual input. Written per-representation because the
        # CNN probability differs between scalogram and topomap models, and fusing
        # against the wrong one would silently mismatch.
        oof_path = os.path.join(args.output_dir, f"{args.representation}_oof_cnn_probs.csv")
        oof.to_csv(oof_path, index=False)
        print(f"Saved out-of-fold CNN probabilities -> {oof_path}")
        print("  Feed this to fusion_classifier.run_fusion_cv() as cnn_subject_probs.")

    if args.evaluate_on_test:
        test_metrics, test_subj = evaluate_on_test(
            args.manifest, args.images_root, args.representation, args.output_dir,
            epochs=args.epochs, inner_val_fold=args.inner_val_fold,
        )
        # Per-subject rows, not just the summary: significance_test.py's
        # bootstrap needs the individual correct/incorrect outcomes, and a
        # fused test number needs these probabilities alongside the classical
        # features. Writing only the metrics dict would force a retrain to
        # recover them.
        test_subj.to_csv(
            os.path.join(args.output_dir, f"{args.representation}_test_subject_preds.csv"), index=False)
        pd.DataFrame([test_metrics]).to_csv(
            os.path.join(args.output_dir, f"{args.representation}_test_metrics.csv"), index=False)
        print(f"Saved held-out test predictions and metrics -> {args.output_dir}")


if __name__ == "__main__":
    main()