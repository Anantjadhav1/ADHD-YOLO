"""
Positive control: Rest vs Task through the classical arm's OWN cross-validation.

Every arm of Phase 2 scored at chance on ADHD vs Control. That is only a
finding about the data if the pipeline can find a difference that is known to
exist. Rest vs Task is that difference: the same children, recorded at rest
and while doing the VCPT Go/NoGo task, differ in arousal and engagement in
ways spectral power is known to track. If this pipeline scores high here, the
ADHD null is about the data; if it also scores at chance, the code is suspect.

What is held identical to the ADHD analysis, on purpose:
  * the SAME function: training.classical_baseline.run_cv, imported, unmodified
    -- imputer, scaler, SelectKBest, RBF SVM, k and C chosen on an inner fold;
  * the SAME 6R three-way split (outer test / inner val / train) and the SAME
    subject folds from the manifest. Both rows of a subject carry that
    subject's split, so a child's rest and task rows are always in the same
    fold: the split stays subject-wise, never row-wise;
  * development subjects only. The held-out test split is not touched.

What differs, and why:
  * Only features measured in BOTH conditions (100: relative band power,
    TBR, IAF, aperiodic exponent/offset, frontal alpha asymmetry). Coherence
    exists for rest only; including it would let the classifier separate the
    classes by which columns are missing rather than by physiology.
  * Rest is EO by default. EO and VCPT are both eyes-open, and both labels
    come from which FILE a row came from, not from alpha. EC vs EO is NOT used:
    64 of 108 EC/EO boundaries were located with alpha power, so that task is
    circular. EC+EO vs VCPT (README's wording) is available as --rest ec+eo,
    but it lets eye state carry the separation, so it is the weaker control.
  * Statistics respect the pairing: the bootstrap resamples SUBJECTS, and the
    permutation null flips rest/task labels WITHIN each subject.

run_cv hard-codes its positive class as group == "ADHD". Rather than modify
the function under test, task rows are labelled "ADHD" and rest rows
"Control" on the way in, and relabelled on the way out. Every output of this
script says task/rest; nothing it writes should be read as a diagnosis.

Run:  py -m training.positive_control
      py -m training.positive_control --rest ec+eo
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer

from training.classical_baseline import _auc, run_cv

# run_cv's label strings. HERE they mean task / rest, not a diagnosis.
TASK_LABEL, REST_LABEL = "ADHD", "Control"


def _stems(df: pd.DataFrame, cond: str) -> set:
    suffix = f"_{cond}"
    return {c[: -len(suffix)] for c in df.columns if c.endswith(suffix)}


def build_paired_table(df: pd.DataFrame, rest_conds: list, task_cond: str = "vcpt"):
    """One row per subject per condition, on the features shared by all of them."""
    stems = sorted(set.intersection(*[_stems(df, c) for c in rest_conds + [task_cond]]))
    dev = df[df["split"] != "test"]
    task_cols = [f"{s}_{task_cond}" for s in stems]
    dev = dev[dev[task_cols].notna().any(axis=1)]  # subjects with a VCPT recording

    rows = []
    for _, r in dev.iterrows():
        for cond, label in [(c, REST_LABEL) for c in rest_conds] + [(task_cond, TASK_LABEL)]:
            row = {"subject_id": r["subject_id"], "split": r["split"], "group": label,
                   "condition": cond, "diagnosis": r["group"]}
            row.update({s: r[f"{s}_{cond}"] for s in stems})
            rows.append(row)
    return pd.DataFrame(rows), stems


def _arrays_by_subject(oof: pd.DataFrame):
    return [(g["pred_prob_adhd"].to_numpy(), (g["true_label"] == TASK_LABEL).to_numpy())
            for _, g in oof.groupby("subject_id")]


def _pooled_auc(groups) -> float:
    p = np.concatenate([g[0] for g in groups])
    y = np.concatenate([g[1] for g in groups])
    return _auc(p[y], p[~y])


def cluster_bootstrap_ci(oof: pd.DataFrame, n: int = 2000, seed: int = 42) -> tuple:
    """Resample SUBJECTS, not rows: a child's rest and task rows are not independent."""
    rng = np.random.default_rng(seed)
    groups = _arrays_by_subject(oof)
    stats = [_pooled_auc([groups[i] for i in rng.integers(0, len(groups), len(groups))])
             for _ in range(n)]
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def within_subject_permutation_p(oof: pd.DataFrame, observed: float,
                                 n: int = 5000, seed: int = 42) -> float:
    """Null: within each subject, which row is 'task' is arbitrary. Labels are
    shuffled inside each subject, never across subjects."""
    rng = np.random.default_rng(seed)
    groups = _arrays_by_subject(oof)
    hits = 0
    for _ in range(n):
        perm = [(p, rng.permutation(y)) for p, y in groups]
        hits += _pooled_auc(perm) >= observed
    return (hits + 1) / (n + 1)


def paired_win_rate(oof: pd.DataFrame) -> tuple:
    """Number of subjects whose task row scores above their own rest row(s)."""
    wins = []
    for _, g in oof.groupby("subject_id"):
        t = g.loc[g["true_label"] == TASK_LABEL, "pred_prob_adhd"]
        r = g.loc[g["true_label"] == REST_LABEL, "pred_prob_adhd"]
        if len(t) and len(r):
            wins.append(t.mean() > r.mean())
    return int(np.sum(wins)), len(wins)


def describe_top_features(table: pd.DataFrame, feats: list, top: int = 10) -> pd.DataFrame:
    """DESCRIPTIVE ONLY, on all development rows, ignoring the pairing. Says
    which measurements differ most between conditions; feeds no metric."""
    X = SimpleImputer(strategy="median").fit_transform(table[feats].to_numpy())
    y = (table["group"] == TASK_LABEL).to_numpy()
    F, _ = f_classif(X, y)
    diff = X[y].mean(axis=0) - X[~y].mean(axis=0)
    out = pd.DataFrame({"feature": feats, "F": F, "task_minus_rest": diff})
    return out.sort_values("F", ascending=False).head(top).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="docs/classical_features_v3.csv")
    ap.add_argument("--manifest", default="data_pipeline/splits/subject_splits.csv")
    ap.add_argument("--rest", default="eo", choices=["eo", "ec", "ec+eo"])
    ap.add_argument("--output-dir", default="docs")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.features)
    manifest = pd.read_csv(args.manifest)
    rest_conds = args.rest.split("+")
    table, feats = build_paired_table(df, rest_conds)
    n_subj = table["subject_id"].nunique()
    print(f"Positive control: rest ({args.rest.upper()}) vs task (VCPT)")
    print(f"  {n_subj} development subjects, {len(table)} rows, {len(feats)} shared features")
    print("  run_cv's per-fold 'sens' = task recall, 'spec' = rest recall\n")

    folds, oof = run_cv(table, feats, manifest, seed=args.seed)

    groups = _arrays_by_subject(oof)
    pooled = _pooled_auc(groups)
    lo, hi = cluster_bootstrap_ci(oof, seed=args.seed)
    p = within_subject_permutation_p(oof, pooled, seed=args.seed)
    wins, n_pairs = paired_win_rate(oof)

    print(f"\n=== Rest ({args.rest.upper()}) vs Task, pooled over {len(oof)} out-of-fold rows ===")
    print(f"  pooled AUC        {pooled:.3f}")
    print(f"  95% CI (subjects) [{lo:.3f}, {hi:.3f}]")
    print(f"  permutation p     {p:.4f}  (labels flipped within subject, 5000 reps)")
    print(f"  task > own rest   {wins}/{n_pairs} subjects ({wins / n_pairs:.0%})")
    print(f"  fold AUCs         {', '.join(f'{a:.3f}' for a in folds['auc'])}")
    print(f"  k selected        {', '.join(str(k) for k in folds['k_selected'])}")

    top = describe_top_features(table, feats)
    print("\n  Most different between conditions (descriptive, all dev rows, unpaired F):")
    for _, r in top.iterrows():
        print(f"    {r['feature']:28} F={r['F']:7.1f}   task - rest = {r['task_minus_rest']:+.4f}")

    tag = args.rest.replace("+", "")
    os.makedirs(args.output_dir, exist_ok=True)
    out_oof = oof.copy()
    out_oof["true_label"] = np.where(out_oof["true_label"] == TASK_LABEL, "task", "rest")
    out_oof = out_oof.rename(columns={"pred_prob_adhd": "pred_prob_task"}).drop(columns="pred_label")
    out_oof.to_csv(os.path.join(args.output_dir, f"positive_control_{tag}_oof.csv"), index=False)
    folds.rename(columns={"sensitivity": "task_recall", "specificity": "rest_recall"}).to_csv(
        os.path.join(args.output_dir, f"positive_control_{tag}_folds.csv"), index=False)
    pd.DataFrame([{"rest": args.rest, "n_subjects": n_subj, "n_features": len(feats),
                   "pooled_auc": pooled, "ci_low": lo, "ci_high": hi, "perm_p": p,
                   "task_gt_rest": wins, "n_pairs": n_pairs}]).to_csv(
        os.path.join(args.output_dir, f"positive_control_{tag}_summary.csv"), index=False)
    print(f"\n  wrote {args.output_dir}/positive_control_{tag}_{{oof,folds,summary}}.csv")


if __name__ == "__main__":
    main()
