"""
The pre-registered analyses H1a, H1b and H2 (docs/PREREGISTRATION.md, with
deviations D1-D3). H3 is separate: it needs the Rohani recordings re-processed
for covariance matrices and a new dependency.

This file was committed BEFORE it was run on the real data, so its code is
fixed ahead of its results. It implements the registration and nothing else;
every choice below is quoted from it.

  H1a  replication: subject-wise stratified 5-fold CV within Nasrabadi et al.
  H1b  transfer: trained on the 85 Rohani development subjects with a task
       recording, applied once to Nasrabadi et al.; features standardised
       within each dataset (no labels used).
  H2   estimate only: the post-hoc model (all engineered features) trained on
       the 91 Rohani development subjects, applied once to the 17-subject
       held-out test split. No significance test; no p value is reported.

Model everywhere: median imputation -> standardisation -> logistic regression
(L2, C = 1, max_iter = 1000), fitted on training data only. Nothing is tuned.

Decision rule for the confirmatory family {H1a, H1b, H3}: positive only if the
one-sided permutation p < 0.05 / 3 AND the 95% CI lower bound > 0.5. Anything
else is negative. Statistics: training.results_table.pooled_stats (10,000
subject bootstraps, 5,000 permutations, seed 42).

Run:  py -m training.preregistered
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from training.results_table import pooled_stats

ALPHA = 0.05 / 3
SEED = 42
ROHANI_FEATURES = "docs/classical_features_v3.csv"
NASRABADI_FEATURES = "D:/adhd-yolo-work/nasrabadi/features.csv"
NON_FEATURE = {"subject_id", "group", "split", "error", "n_epochs", "n_kept", "rejection_rate"}


def model() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, max_iter=1000)),
    ])


def task_columns(df: pd.DataFrame) -> list:
    return sorted(c for c in df.columns if c.endswith("_vcpt"))


def _oof_frame(ids, groups, prob) -> pd.DataFrame:
    return pd.DataFrame({"subject_id": list(ids), "true_label": list(groups),
                         "pred_prob_adhd": np.asarray(prob, dtype=float)})


def h1a(nas: pd.DataFrame) -> pd.DataFrame:
    """Subject-wise stratified 5-fold CV within Nasrabadi (one row per subject)."""
    cols = task_columns(nas)
    X, y = nas[cols].to_numpy(), (nas["group"] == "ADHD").astype(int).to_numpy()
    prob = np.full(len(nas), np.nan)
    for tr, te in StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y):
        prob[te] = model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    assert np.isfinite(prob).all(), "every subject must be scored exactly once"
    return _oof_frame(nas["subject_id"], nas["group"], prob)


def rohani_task_training_set(roh: pd.DataFrame) -> pd.DataFrame:
    """The 85 development subjects with a task recording (H1b)."""
    dev = roh[roh["split"] != "test"]
    return dev[dev[task_columns(roh)].notna().any(axis=1)]


def h1b(roh: pd.DataFrame, nas: pd.DataFrame) -> pd.DataFrame:
    """Train on Rohani, apply once to Nasrabadi. Each dataset is imputed and
    standardised with its OWN statistics (no labels), then the classifier
    trained on standardised Rohani features scores standardised Nasrabadi ones."""
    cols = task_columns(nas)
    if sorted(cols) != task_columns(roh):
        raise ValueError("Rohani and Nasrabadi task features do not line up")
    train = rohani_task_training_set(roh)

    def standardise(frame):
        x = SimpleImputer(strategy="median").fit_transform(frame[cols].to_numpy())
        return StandardScaler().fit_transform(x)

    lr = LogisticRegression(C=1.0, max_iter=1000)
    lr.fit(standardise(train), (train["group"] == "ADHD").astype(int).to_numpy())
    prob = lr.predict_proba(standardise(nas))[:, 1]
    return _oof_frame(nas["subject_id"], nas["group"], prob)


def engineered_columns(roh: pd.DataFrame) -> list:
    """All engineered features, as in the post-hoc model: numeric, not
    bookkeeping, not missing for every development subject."""
    dev = roh[roh["split"] != "test"]
    num = [c for c in roh.select_dtypes("number").columns if c not in NON_FEATURE]
    return [c for c in num if dev[c].notna().any()]


def h2(roh: pd.DataFrame) -> pd.DataFrame:
    """The post-hoc model trained on the 91 development subjects, applied once
    to the held-out test split."""
    cols = engineered_columns(roh)
    dev, test = roh[roh["split"] != "test"], roh[roh["split"] == "test"]
    m = model().fit(dev[cols].to_numpy(), (dev["group"] == "ADHD").astype(int).to_numpy())
    return _oof_frame(test["subject_id"], test["group"], m.predict_proba(test[cols].to_numpy())[:, 1])


def decide(s: dict) -> str:
    return "POSITIVE" if (s["perm_p"] < ALPHA and s["ci_low"] > 0.5) else "negative"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nasrabadi", default=NASRABADI_FEATURES)
    ap.add_argument("--rohani", default=ROHANI_FEATURES)
    ap.add_argument("--csv", default="docs/preregistered_results.csv")
    ap.add_argument("--oof-dir", default="D:/adhd-yolo-work/nasrabadi")
    args = ap.parse_args()

    roh, nas = pd.read_csv(args.rohani), pd.read_csv(args.nasrabadi)
    runs = {"H1a": h1a(nas), "H1b": h1b(roh, nas), "H2": h2(roh)}
    rows = []
    print(f"Pre-registered analyses (alpha = {ALPHA:.4f}, one-sided permutation p)\n")
    for name, oof in runs.items():
        s = pooled_stats(oof, seed=SEED)
        acc = float(((oof["pred_prob_adhd"] >= 0.5) == (oof["true_label"] == "ADHD")).mean())
        row = {"hypothesis": name, "n": s["n"], "n_adhd": s["n_adhd"], "auc": s["auc"],
               "ci_low": s["ci_low"], "ci_high": s["ci_high"], "accuracy": acc}
        if name == "H2":   # estimate only: no p value, no decision (registration sec. 2)
            row.update(perm_p=np.nan, decision="estimate only")
            print(f"{name}  n={s['n']:3d}  AUC {s['auc']:.3f}  95% CI [{s['ci_low']:.3f}, {s['ci_high']:.3f}]"
                  f"  accuracy {acc:.3f}  (estimate only)")
        else:
            row.update(perm_p=s["perm_p"], decision=decide(s))
            print(f"{name}  n={s['n']:3d}  AUC {s['auc']:.3f}  95% CI [{s['ci_low']:.3f}, {s['ci_high']:.3f}]"
                  f"  p {s['perm_p']:.4f}  accuracy {acc:.3f}  -> {row['decision']}")
        rows.append(row)
        os.makedirs(args.oof_dir, exist_ok=True)
        oof.to_csv(os.path.join(args.oof_dir, f"preregistered_{name}_oof.csv"), index=False)
    pd.DataFrame(rows).to_csv(args.csv, index=False)
    print(f"\n-> {args.csv}   (H3 is run separately)")


if __name__ == "__main__":
    main()
