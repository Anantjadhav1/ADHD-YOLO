"""
Post-hoc sensitivity analysis: does the fusion result depend on feature scaling,
and if so, does the CNN have anything to do with it?

EXPLORATORY, NOT CONFIRMATORY. Read this before quoting any number below.

Origin (2026-10-16). Writing the methods found that fusion_classifier.py fits
an L2-penalised logistic regression on the CNN probability plus 551 engineered
features WITHOUT standardising them, so the penalty acts unevenly across units
(the CNN probability is 0-1; aperiodic offsets are around -10). The engineered
SVM already standardises; fusion was the only model that did not. This script
re-runs fusion with standardisation, and -- because a jump in the number could
come from the model class rather than from fusing -- the same standardised
linear model WITHOUT the CNN probability.

Nothing here replaces the Phase 2 table. These specifications were chosen
AFTER the earlier results were known, after at least eight analyses of the same
84 subjects; the permutation p values are unadjusted for that. Fold-level AUCs
are unstable. A confirmatory test needs a model fixed in advance and data it
has never touched -- and the 17-subject held-out split is too small to settle
an AUC near 0.6 (its CI would span roughly +/-0.25).

Held identical to the published fusion: the same function (run_fusion_cv),
folds, inner join with the scalogram CNN's out-of-fold file (which reproduces
runs/fusion/fusion_cv_results.csv exactly), median imputation fitted on the
training folds, LogisticRegression(max_iter=1000). The only changes are the
StandardScaler (fitted on training folds only) and, for the third row, dropping
the CNN column. Pooled AUC, CI and p use results_table.pooled_stats, the same
procedure as the headline table.

Run:  py -m training.linear_sensitivity
"""

import argparse
import contextlib
import io

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from training import fusion_classifier as fc
from training.results_table import pooled_stats
from training.train_yolo_cls import compute_metrics

CNN_OOF = "runs/frozen/scalogram_oof_cnn_probs.csv"
FEATURES = "docs/classical_features_v3.csv"

VARIANTS = [
    # (label, standardise, include the CNN probability)
    ("fusion as published (unscaled)", False, True),
    ("fusion, standardised", True, True),
    ("engineered features only, standardised", True, False),
]


def _fold_fn(standardise: bool, use_cnn: bool, store: list):
    def fold(train_df, val_df, feature_cols):
        active = fc.drop_all_nan_columns(pd.concat([train_df, val_df]), feature_cols)
        cols = (["pred_prob_adhd"] if use_cnn else []) + active
        imp = SimpleImputer(strategy="median")
        Xtr = imp.fit_transform(train_df[cols].values)
        Xva = imp.transform(val_df[cols].values)
        if standardise:
            sc = StandardScaler()
            Xtr, Xva = sc.fit_transform(Xtr), sc.transform(Xva)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(Xtr, (train_df["group"] == "ADHD").astype(int).values)
        p = clf.predict_proba(Xva)[:, 1]
        out = pd.DataFrame({"subject_id": val_df["subject_id"].values,
                            "true_label": val_df["group"].values, "pred_prob_adhd": p,
                            "pred_label": np.where(p >= 0.5, "ADHD", "Control")})
        store.append(out)
        return compute_metrics(out)
    return fold


def run_variant(manifest: str, cnn: pd.DataFrame, feats: pd.DataFrame,
                standardise: bool, use_cnn: bool, seed: int = 42) -> tuple:
    store = []
    original = fc.train_and_evaluate_fold
    fc.train_and_evaluate_fold = _fold_fn(standardise, use_cnn, store)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            folds = fc.run_fusion_cv(manifest, cnn, feats)
    finally:
        fc.train_and_evaluate_fold = original
    oof = pd.concat(store, ignore_index=True)
    return folds, oof, pooled_stats(oof, seed=seed)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data_pipeline/splits/subject_splits.csv")
    ap.add_argument("--csv", default="docs/linear_sensitivity.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cnn, feats = pd.read_csv(CNN_OOF), pd.read_csv(FEATURES)
    rows = []
    print("EXPLORATORY post-hoc sensitivity analysis -- see the module docstring.\n")
    for label, standardise, use_cnn in VARIANTS:
        folds, oof, s = run_variant(args.manifest, cnn, feats, standardise, use_cnn, args.seed)
        acc = float(((oof["pred_prob_adhd"] >= 0.5) == (oof["true_label"] == "ADHD")).mean())
        rows.append({"variant": label, "standardised": standardise, "uses_cnn": use_cnn,
                     "n": s["n"], "pooled_auc": s["auc"], "ci_low": s["ci_low"],
                     "ci_high": s["ci_high"], "perm_p_unadjusted": s["perm_p"],
                     "accuracy": acc, "fold_aucs": " ".join(f"{a:.3f}" for a in folds["auc"])})
        print(f"{label:40} n={s['n']}  AUC {s['auc']:.3f}  CI [{s['ci_low']:.3f}, {s['ci_high']:.3f}]  "
              f"p {s['perm_p']:.3f} (unadjusted)  folds {rows[-1]['fold_aucs']}")
    pd.DataFrame(rows).to_csv(args.csv, index=False)
    print(f"\n-> {args.csv}")


if __name__ == "__main__":
    main()
