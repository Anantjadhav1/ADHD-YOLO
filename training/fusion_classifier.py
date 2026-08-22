"""
Phase 2 — fusion meta-classifier: CNN output probability + classical
biomarkers, per PROJECT.md sec 4 step 5 and sec 5a #1 (highest accuracy
leverage on this dataset).

Reuses the SAME subject-wise CV folds as train_yolo_cls.py -- never
re-derives splits -- and the same compute_metrics() so fusion results are
directly comparable to the CNN-alone and classical-alone numbers in the
Phase 2 report table PROJECT.md sec 6 calls for.
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from data_pipeline import subject_split
from training.train_yolo_cls import compute_metrics


def drop_all_nan_columns(df: pd.DataFrame, feature_cols: list) -> list:
    """
    Currently p300_latency_ms, p300_amplitude, omission_errors,
    commission_errors, reaction_time_ms are 100% NaN for every subject
    (blocked pending trigger confirmation -- see classical_features.py). A
    100%-NaN column contributes nothing but noise if imputed. Drop columns
    that are ENTIRELY missing, keep columns that are only PARTIALLY missing
    (those get imputed per-fold below). This means once P300/behavioral
    features become available, they're automatically included next run --
    no code change needed here.
    """
    return [c for c in feature_cols if df[c].notna().any()]


def build_fusion_table(cnn_subject_probs: pd.DataFrame, classical_features: pd.DataFrame) -> pd.DataFrame:
    """
    cnn_subject_probs: columns [subject_id, true_label, pred_prob_adhd] --
        the output of train_yolo_cls.aggregate_to_subject_level(), one row
        per subject (already averaged across that subject's epochs).
    classical_features: the DataFrame from build_classical_features.py.

    Returns one merged row per subject. Uses an inner join deliberately --
    a subject missing from either side can't be used for fusion, and
    silently dropping them is safer than silently filling in fake data.
    """
    merged = cnn_subject_probs.merge(
        classical_features, on="subject_id", how="inner", suffixes=("", "_clf")
    )
    dropped = set(cnn_subject_probs["subject_id"]) ^ set(merged["subject_id"])
    if dropped:
        print(f"Note: {len(dropped)} subject(s) dropped from fusion (missing from one side): {dropped}")
    return merged


def train_and_evaluate_fold(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: list) -> dict:
    """
    One fold: fit the imputer and logistic regression on TRAIN ONLY, apply
    to val. Fitting the imputer on val (or on train+val combined) would leak
    val-set statistics into training -- the same leakage discipline
    subject_split.py enforces at the split level applies here too.
    """
    active_cols = drop_all_nan_columns(pd.concat([train_df, val_df]), feature_cols)
    X_train_raw = train_df[["pred_prob_adhd"] + active_cols].values
    X_val_raw = val_df[["pred_prob_adhd"] + active_cols].values

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train_raw)
    X_val = imputer.transform(X_val_raw)

    y_train = (train_df["group"] == "ADHD").astype(int).values
    y_val_true_label = val_df["group"].values

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    val_prob_adhd = clf.predict_proba(X_val)[:, 1]

    result_df = pd.DataFrame({
        "subject_id": val_df["subject_id"].values,
        "true_label": y_val_true_label,
        "pred_prob_adhd": val_prob_adhd,
        "pred_label": np.where(val_prob_adhd >= 0.5, "ADHD", "Control"),
    })
    return compute_metrics(result_df)


def run_fusion_cv(manifest_path: str, cnn_subject_probs: pd.DataFrame,
                   classical_features: pd.DataFrame) -> pd.DataFrame:
    """Full subject-wise CV for the fusion model, mirroring train_yolo_cls.run_cv()'s
    fold loop structure so results line up in the same report table."""
    manifest = subject_split.load_manifest(manifest_path)
    subject_split.verify_no_leakage(manifest)  # same safety check train_yolo_cls.run_cv() does
    fused = build_fusion_table(cnn_subject_probs, classical_features)
    # NOTE: no separate merge for "split" here -- classical_features already
    # carries it (build_classical_features_table records it from this same
    # manifest), so merging it in again created a split_x/split_y collision
    # instead of a usable "split" column. Found by actually running this,
    # not by reading the merge logic.

    feature_cols = [c for c in classical_features.columns if c not in ("subject_id", "group", "split", "error")]
    dev_folds = sorted(f for f in fused["split"].unique() if f != "test")

    fold_metrics = []
    for val_fold in dev_folds:
        train_df = fused[fused["split"].isin([f for f in dev_folds if f != val_fold])]
        val_df = fused[fused["split"] == val_fold]
        if train_df.empty or val_df.empty:
            continue
        if train_df["group"].nunique() < 2:
            # Real edge case, hit during testing with a tiny sample: a training
            # fold can end up with only one class present (e.g. a fold that's
            # nearly all one group). LogisticRegression can't fit that. With the
            # real, properly-stratified 103-subject 5-fold split this should be
            # rare-to-never, but one bad fold still shouldn't crash the whole
            # CV run -- skip it with a clear warning instead, same philosophy
            # as build_dataset.py's per-subject error handling.
            print(f"[{val_fold}] SKIPPED: training fold has only one class present "
                  f"({train_df['group'].unique()}) -- can't fit a binary classifier.")
            continue
        metrics = train_and_evaluate_fold(train_df, val_df, feature_cols)
        metrics["fold"] = val_fold
        fold_metrics.append(metrics)
        print(f"[{val_fold}] n={metrics['n_subjects']} acc={metrics['accuracy']:.3f} "
              f"sens={metrics['sensitivity']:.3f} spec={metrics['specificity']:.3f} auc={metrics['auc']:.3f}")

    return pd.DataFrame(fold_metrics)