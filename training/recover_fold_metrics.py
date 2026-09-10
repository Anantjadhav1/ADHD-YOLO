"""
Recover subject-level CV metrics from saved fold weights, without retraining.

WHAT HAPPENED
-------------
A Colab session was recycled after `run_cv` finished all five topomap folds but
before it returned. The per-fold backup added on 2026-09-05 saved each fold's
Ultralytics output directory -- including `weights/best.pt` -- but
`fold_metrics` and `oof_frames` live in memory until the loop ends, so every
subject-level number was lost while all five trained models survived.

The expensive part (training) completed; only the cheap part (scoring) was
lost. This recovers it.

WHY THIS IS NOT A SHORTCUT THAT CHANGES THE RESULT
--------------------------------------------------
It reproduces exactly what `run_cv` does after training: load the fold's
`best.pt`, run `predict_class_dirs` on that fold's OUTER images -- the ones
excluded from both train/ and val/ by the section 6R three-way split --
aggregate to subject level, and compute metrics. No model is retrained, no
checkpoint is reselected, and the outer fold is still data the model never saw.

The one thing it CANNOT verify is that each `best.pt` was selected on the inner
fold rather than the outer one. That is guaranteed by how the run was
configured, not by anything visible in the saved output, so this script asserts
the run directory naming matches the expected `<representation>_<outer_fold>`
pattern and refuses to guess if it does not.

    python -m training.recover_fold_metrics \\
        --runs-dir /content/drive/MyDrive/adhd_runs_topomap \\
        --images-root /content/dataset_v2_small \\
        --representation topomap
"""

import argparse
import re
from pathlib import Path

import pandas as pd

from data_pipeline import subject_split
from training.train_yolo_cls import (aggregate_to_subject_level,
                                     collect_oof_predictions, compute_metrics,
                                     predict_class_dirs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True,
                    help="directory holding <representation>_<fold> subdirectories")
    ap.add_argument("--images-root", required=True)
    ap.add_argument("--representation", required=True, choices=["scalogram", "topomap"])
    ap.add_argument("--manifest", default="data_pipeline/splits/subject_splits.csv")
    ap.add_argument("--output-dir", default=None,
                    help="where to write the CSVs (defaults to --runs-dir)")
    args = ap.parse_args()

    from ultralytics import YOLO

    runs = Path(args.runs_dir)
    out_dir = Path(args.output_dir or args.runs_dir)
    manifest = subject_split.load_manifest(args.manifest)

    pattern = re.compile(rf"^{re.escape(args.representation)}_(fold_\d+)$")
    fold_dirs = sorted(d for d in runs.iterdir()
                       if d.is_dir() and pattern.match(d.name))
    if not fold_dirs:
        raise FileNotFoundError(
            f"No '{args.representation}_fold_N' directories under {runs}. "
            "This script identifies the OUTER fold from the directory name, so a "
            "different naming scheme means it cannot tell which subjects each "
            "model was scored against -- and guessing would silently produce "
            "in-sample numbers."
        )
    print(f"Found {len(fold_dirs)} fold directories under {runs}\n")

    fold_metrics, oof_frames = [], []
    for d in fold_dirs:
        outer_fold = pattern.match(d.name).group(1)
        ckpt = d / "weights" / "best.pt"
        if not ckpt.exists():
            print(f"  {d.name}: no weights/best.pt -- skipping")
            continue

        outer_images = Path(args.images_root) / args.representation / outer_fold
        if not outer_images.exists():
            raise FileNotFoundError(
                f"{outer_images} does not exist. The images must be the SAME ones "
                "the run scored against; recovering metrics from a different "
                "dataset build would produce numbers that look valid and are not."
            )

        print(f"  {d.name}: scoring outer fold {outer_fold} ... ", end="", flush=True)
        model = YOLO(str(ckpt))
        per_image = predict_class_dirs(model, outer_images)
        if not per_image:
            raise ValueError(f"No predictions produced for {outer_images}.")

        subj_df = aggregate_to_subject_level(per_image)
        oof_frames.append(subj_df.assign(fold=outer_fold))
        m = compute_metrics(subj_df)
        m["fold"] = outer_fold
        # inner_val_fold is NOT recoverable from the saved output -- it was only
        # ever in the lost fold_metrics. Recorded as unknown rather than
        # reconstructed from the rotation rule, which would be an assumption
        # dressed as data.
        m["inner_val_fold"] = "unknown (recovered run)"
        m["recovered"] = True
        fold_metrics.append(m)
        print(f"n={m['n_subjects']} acc={m['accuracy']:.3f} auc={m['auc']:.3f}")

    if not fold_metrics:
        print("\nNothing recovered.")
        return

    results = pd.DataFrame(fold_metrics)
    print("\nMean +/- std across folds:")
    for col in ["accuracy", "sensitivity", "specificity", "auc"]:
        print(f"  {col}: {results[col].mean():.3f} +/- {results[col].std():.3f}")

    oof = collect_oof_predictions(oof_frames, manifest)

    out_dir.mkdir(parents=True, exist_ok=True)
    res_path = out_dir / f"{args.representation}_cv_results.csv"
    oof_path = out_dir / f"{args.representation}_oof_cnn_probs.csv"
    results.to_csv(res_path, index=False)
    oof.to_csv(oof_path, index=False)
    print(f"\nRecovered {len(oof)} subject-level predictions")
    print(f"  -> {res_path}")
    print(f"  -> {oof_path}")
    print("\nPool the OOF table across all folds for the headline AUC rather than "
          "averaging the per-fold values: 17 subjects per fold is too few for a "
          "stable fold-level AUC, and the mean of five noisy estimates is noisier "
          "than one estimate over 84.")


if __name__ == "__main__":
    main()