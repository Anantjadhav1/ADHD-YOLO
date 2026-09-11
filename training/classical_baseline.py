"""
The classical arm: SVM on engineered EEG features, on the SAME folds as the CNN.

WHY THIS EXISTS
---------------
significance_test.py compares a bootstrap CI against a fixed 75.8% taken from a
DIFFERENT split of the data. That is not a hypothesis test, and with ~17 test
subjects the CI spans about +/-24 points -- it cannot conclude anything either
way.

A PAIRED comparison can. The fold-level AUCs from the topomap run ranged 0.347
to 0.833 across five folds of ~17 subjects, and almost all of that spread is
WHICH SUBJECTS landed in which fold rather than model quality. Pairing cancels
it, because both models are scored on exactly the same subjects.

So this deliberately mirrors train_yolo_cls.run_cv():
  - the same subject_split.py manifest and folds
  - the same section 6R three-way split (outer fold excluded entirely; an inner
    fold used for model selection; the rest trains)
  - the same output table shape, so the two OOF files join on subject_id

THE LEAK THIS AVOIDS
--------------------
With ~301 features and 84 development subjects, p >> n, so feature selection is
not optional -- and it is exactly where classical EEG pipelines leak. Selecting
features on the full dataset and THEN cross-validating lets the test subjects
influence which features exist, which inflates accuracy while looking rigorous.

Here the imputer, the scaler, the selector and the SVM are all fitted inside a
single sklearn Pipeline on the training folds ONLY, so nothing about the outer
fold can reach any of them. Worth a sentence in the discussion that the 84.5%
being chased may itself suffer from this -- selection-before-CV is common in
this literature and rarely stated either way.

WHY IMPUTATION IS ACCEPTABLE HERE
---------------------------------
Missingness is structural, not scattered: 5 columns are 100% missing (the
P300/behavioural placeholders, never computed) and 100 VCPT columns are 7.4%
missing because 8 subjects have no VCPT recording. Nothing sits in between.

Those 8 split 5 ADHD / 3 Control across four folds and test -- not concentrated
in either group, so imputing them cannot hand the classifier a group-correlated
artefact. Checked before choosing this rather than assumed. (fold_2 holds 3 of
the 8, so it leans on imputed values more than the others; noted rather than
corrected.)

    python -m training.classical_baseline
    python -m training.classical_baseline --features docs/classical_features_v2.csv
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data_pipeline import subject_split

warnings.filterwarnings("ignore", category=UserWarning)

# Candidate feature counts. Rohani et al. selected 113 from 826 on 103
# subjects; this cohort has ~301 from 84, a similar regime. k is chosen on the
# INNER fold, never on the outer one.
K_CANDIDATES = [10, 20, 40, 80]
# Matching the source paper's classifier family. An RBF SVM on ~60 training
# subjects is already generous; anything larger would be fitting noise.
C_CANDIDATES = [0.1, 1.0, 10.0]

NON_FEATURE_COLS = {"subject_id", "group", "split", "error"}


def load_features(path: str) -> tuple:
    df = pd.read_csv(path)
    num = df.select_dtypes("number")
    # Drop columns that are entirely missing. Keeping them would make the
    # imputer invent a constant for every subject -- a feature carrying no
    # information but taking up a slot in the selector.
    all_nan = [c for c in num.columns if num[c].isna().all()]
    feat_cols = [c for c in num.columns
                 if c not in NON_FEATURE_COLS and c not in all_nan]
    print(f"{len(df)} subjects, {len(feat_cols)} usable features "
          f"({len(all_nan)} dropped as entirely missing: {all_nan})")
    return df, feat_cols


def run_cv(df: pd.DataFrame, feat_cols: list, manifest: pd.DataFrame,
           seed: int = 42) -> tuple:
    dev_folds = sorted(f for f in manifest["split"].unique() if f != "test")
    if len(dev_folds) < 3:
        raise ValueError(f"Need >=3 development folds for the three-way split; got {dev_folds}")

    dev = df[df["split"] != "test"].copy()
    dev["y"] = (dev["group"] == "ADHD").astype(int)

    fold_rows, oof_rows = [], []
    for i, outer in enumerate(dev_folds):
        inner = dev_folds[(i + 1) % len(dev_folds)]

        te = dev[dev["split"] == outer]
        va = dev[dev["split"] == inner]
        tr = dev[~dev["split"].isin([outer, inner])]
        if te.empty or va.empty or tr.empty:
            print(f"  {outer}: empty split, skipping")
            continue

        Xtr, ytr = tr[feat_cols].to_numpy(), tr["y"].to_numpy()
        Xva, yva = va[feat_cols].to_numpy(), va["y"].to_numpy()
        Xte, yte = te[feat_cols].to_numpy(), te["y"].to_numpy()

        # Select k and C on the INNER fold. The outer fold contributes to
        # neither, which is what makes its score an out-of-sample estimate
        # rather than a best-of-N maximum.
        best = (-1.0, None, None)
        for k in K_CANDIDATES:
            if k > len(feat_cols):
                continue
            for C in C_CANDIDATES:
                pipe = Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    ("select", SelectKBest(f_classif, k=k)),
                    ("svm", SVC(C=C, kernel="rbf", probability=True,
                                random_state=seed, class_weight="balanced")),
                ])
                pipe.fit(Xtr, ytr)
                score = pipe.score(Xva, yva)
                if score > best[0]:
                    best = (score, k, C)

        _, k, C = best
        # Refit on train + inner val, so the final model uses everything except
        # the outer fold -- the same data budget run_cv's CNN folds get.
        Xfit = np.vstack([Xtr, Xva])
        yfit = np.concatenate([ytr, yva])
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("select", SelectKBest(f_classif, k=k)),
            ("svm", SVC(C=C, kernel="rbf", probability=True,
                        random_state=seed, class_weight="balanced")),
        ])
        pipe.fit(Xfit, yfit)

        prob = pipe.predict_proba(Xte)[:, 1]
        pred = (prob >= 0.5).astype(int)

        for sid, yt, p, pr in zip(te["subject_id"], yte, prob, pred):
            oof_rows.append({
                "subject_id": sid,
                "true_label": "ADHD" if yt else "Control",
                "pred_prob_adhd": float(p),
                "pred_label": "ADHD" if pr else "Control",
                "fold": outer,
            })

        acc = float((pred == yte).mean())
        sens = float(pred[yte == 1].mean()) if (yte == 1).any() else float("nan")
        spec = float(1 - pred[yte == 0].mean()) if (yte == 0).any() else float("nan")
        auc = _auc(prob[yte == 1], prob[yte == 0])
        fold_rows.append({"fold": outer, "inner_val_fold": inner,
                          "n_subjects": len(te), "k_selected": k, "C_selected": C,
                          "accuracy": acc, "sensitivity": sens,
                          "specificity": spec, "auc": auc})
        print(f"[{outer}] inner_val={inner} n={len(te)} k={k} C={C} "
              f"acc={acc:.3f} sens={sens:.3f} spec={spec:.3f} auc={auc:.3f}")

    return pd.DataFrame(fold_rows), pd.DataFrame(oof_rows)


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (len(pos) * len(neg)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="docs/classical_features_v2.csv")
    ap.add_argument("--manifest", default="data_pipeline/splits/subject_splits.csv")
    ap.add_argument("--output-dir", default="runs/classical")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df, feat_cols = load_features(args.features)
    manifest = subject_split.load_manifest(args.manifest)
    subject_split.verify_no_leakage(manifest)

    results, oof = run_cv(df, feat_cols, manifest, seed=args.seed)
    if results.empty:
        print("\nNo folds completed.")
        return

    print("\nMean +/- std across folds:")
    for col in ["accuracy", "sensitivity", "specificity", "auc"]:
        print(f"  {col}: {results[col].mean():.3f} +/- {results[col].std():.3f}")

    # Pooled across all development subjects. Report THIS, not the fold mean:
    # ~17 subjects per fold is too few for a stable fold-level AUC, and
    # averaging five noisy estimates is noisier than one estimate over 84.
    y = (oof["true_label"] == "ADHD").to_numpy()
    p = oof["pred_prob_adhd"].to_numpy()
    pooled = _auc(p[y], p[~y])
    rng = np.random.default_rng(args.seed)
    boots = []
    for _ in range(10000):
        idx = rng.integers(0, len(p), len(p))
        if 0 < y[idx].sum() < len(idx):
            boots.append(_auc(p[idx][y[idx]], p[idx][~y[idx]]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    perm = np.array([_auc(p[m], p[~m]) for m in (rng.permutation(y) for _ in range(5000))])

    print(f"\nPOOLED over {len(oof)} subjects ({y.sum()} ADHD / {(~y).sum()} Control)")
    print(f"  AUC            {pooled:.4f}")
    print(f"  95% CI         [{lo:.3f}, {hi:.3f}]  "
          f"{'contains 0.5' if lo <= 0.5 <= hi else 'EXCLUDES 0.5'}")
    print(f"  permutation p  {(perm >= pooled).mean():.3f}")
    print(f"  accuracy @0.5  {((p >= 0.5) == y).mean():.3f}")

    print(f"\n  For reference on the same 84 subjects:")
    print(f"    CNN scalogram, frozen   AUC 0.521  CI [0.397, 0.641]  p=0.311")
    print(f"    CNN topomap, frozen     AUC 0.523  CI [0.393, 0.647]  p=0.368")
    print("\n  These are directly comparable -- same folds, same subjects, same")
    print("  three-way split. Run paired_comparison.py for a test of the")
    print("  DIFFERENCE, which is far more powerful than comparing two CIs.")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / "classical_cv_results.csv", index=False)
    oof.to_csv(out / "classical_oof_probs.csv", index=False)
    print(f"\n  -> {out / 'classical_cv_results.csv'}")
    print(f"  -> {out / 'classical_oof_probs.csv'}")


if __name__ == "__main__":
    main()