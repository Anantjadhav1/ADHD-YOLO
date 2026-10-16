"""
Regenerate the Phase 2 headline table from the committed out-of-fold files:
one cohort, one procedure, reproducible from the repo.

Why this exists (2026-10-16). The table in README and PROGRESS says its arms
are measured "on the same 84 development subjects (42 ADHD / 42 Control)".
Checking that claim found it held for the AUCs but not for the intervals:

  * Both classical rows paired an AUC from one cohort with a CI and p from
    another. classical_baseline.py pools over all 91 development subjects
    (44 ADHD / 47 Control) -- 0.538 for the 551-feature model -- while the
    0.556 in the table is the same model on the 84-subject intersection.
    The CI [0.416, 0.660] and p = 0.268 printed beside 0.556 belong to 0.538.
  * The CNN rows' CI and p were computed on Colab by code that is not in the
    repo, so they cannot be regenerated, and the committed procedure below
    does not reproduce them (scalogram p 0.311 published vs 0.364 here; at
    5000 permutations the Monte Carlo error is ~0.007, so that is a different
    procedure, not noise).

No conclusion changes: every AUC is identical, every CI still contains 0.5,
every p is still far from significance. What changes is that every number in
the table now comes from the same subjects and from code anyone can run.

Procedure: the one classical_baseline.main() already uses, so the classical
rows are its own numbers on the right cohort -- AUC over all subjects;
10,000 bootstrap resamples (one row per subject, so resampling rows is
resampling subjects); 5,000 label permutations; a fresh RNG seeded 42 per
arm, so each row is reproducible on its own and independent of row order.

Run:  py -m training.results_table
"""

import argparse
import os

import numpy as np
import pandas as pd

from training.classical_baseline import _auc

ARMS = [
    ("CNN scalogram, frozen backbone", "runs/frozen/scalogram_oof_cnn_probs.csv"),
    ("CNN topomap, frozen backbone", "runs/frozen_topomap/topomap_oof_cnn_probs.csv"),
    ("Classical SVM, 301 power features", "runs/classical/classical_oof_probs.csv"),
    ("Classical SVM, 551 features with coherence", "runs/classical_v3/classical_oof_probs.csv"),
]


def pooled_stats(oof: pd.DataFrame, seed: int = 42, n_boot: int = 10000, n_perm: int = 5000) -> dict:
    """classical_baseline.main()'s exact procedure, including the order in
    which it draws from the RNG (bootstrap first, then permutations)."""
    y = (oof["true_label"] == "ADHD").to_numpy()
    p = oof["pred_prob_adhd"].to_numpy()
    auc = _auc(p[y], p[~y])
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(p), len(p))
        if 0 < y[idx].sum() < len(idx):
            boots.append(_auc(p[idx][y[idx]], p[idx][~y[idx]]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    perm = np.array([_auc(p[m], p[~m]) for m in (rng.permutation(y) for _ in range(n_perm))])
    return {"n": len(oof), "n_adhd": int(y.sum()), "auc": auc,
            "ci_low": float(lo), "ci_high": float(hi), "perm_p": float((perm >= auc).mean())}


def build_table(arms=ARMS, seed: int = 42) -> tuple:
    oofs = {name: pd.read_csv(path) for name, path in arms}
    for name, d in oofs.items():
        if d["subject_id"].duplicated().any():
            raise ValueError(f"{name}: duplicate subject_id -- not one row per subject")
    common = set.intersection(*(set(d["subject_id"]) for d in oofs.values()))
    rows = []
    for name, d in oofs.items():
        on_common = pooled_stats(d[d["subject_id"].isin(common)], seed=seed)
        everyone = pooled_stats(d, seed=seed)
        rows.append({"arm": name, **on_common,
                     "n_all": everyone["n"], "auc_all": everyone["auc"],
                     "ci_low_all": everyone["ci_low"], "ci_high_all": everyone["ci_high"],
                     "perm_p_all": everyone["perm_p"]})
    return pd.DataFrame(rows), sorted(common)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--csv", default="docs/results_table.csv")
    args = ap.parse_args()

    table, common = build_table(seed=args.seed)
    n_adhd = int(table["n_adhd"].iloc[0])
    print(f"Common cohort: {len(common)} subjects ({n_adhd} ADHD / {len(common) - n_adhd} Control)\n")
    print(f"{'arm':44}{'AUC':>7}   95% CI           perm p   | own cohort (n, AUC, CI, p)")
    for _, r in table.iterrows():
        print(f"{r['arm']:44}{r['auc']:7.3f}   [{r['ci_low']:.3f}, {r['ci_high']:.3f}]   "
              f"{r['perm_p']:.3f}    | {r['n_all']:3d}  {r['auc_all']:.3f}  "
              f"[{r['ci_low_all']:.3f}, {r['ci_high_all']:.3f}]  {r['perm_p_all']:.3f}")
    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    table.to_csv(args.csv, index=False)
    print(f"\n-> {args.csv}")


if __name__ == "__main__":
    main()
