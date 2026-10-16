"""
Does the Phase 2 result depend on which recordings are excluded for quality?

README open issue 2 has been open since August: the pipeline flags problem
recordings, but there was no rule for whether to include, exclude or flag them.
That decision should be made on principle, not by seeing which choice gives the
best number -- so this script answers the prior question instead: does the
choice change the conclusion at all?

Inputs:
  * docs/qc_audit.csv -- per-subject QC for all 108, written from the 2026-10-16
    build_log.csv columns (rejection rate per condition, whether the ICA muscle
    cap fired in either pass, EC/EO ambiguity).
  * the committed out-of-fold predictions behind the headline table.

Policies are applied to FIXED out-of-fold predictions: every model was trained
with every subject, and only the scored set changes. That answers "is the
result different on cleaner recordings?", not "would models trained without
bad recordings do better?" -- a rerun question, and a different one.

Tail thresholds are measured from this cohort (90th percentile per condition),
not taken from the pipeline's 30% warning: in EC the median subject already
loses 31% of epochs, so 30% flags the middle of the distribution, not its tail.

Run:  py -m training.qc_sensitivity
"""

import argparse

import pandas as pd

from training.results_table import ARMS, pooled_stats

QC_AUDIT = "docs/qc_audit.csv"
REJ_COLS = ("rej_ec", "rej_eo", "rej_vcpt")
SENSITIVITY_ARMS = [a for a in ARMS if a[0] != "Classical SVM, 301 power features"]


def _true(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().eq("true")


def policy_masks(qc: pd.DataFrame, tail_q: float = 0.90) -> tuple:
    """Return ({policy: boolean Series of subjects to DROP}, {column: threshold})."""
    qc = qc.set_index("subject_id")
    cap = _true(qc["capped_eoec"]) | _true(qc["capped_vcpt"])
    thresholds = {c: float(qc[c].quantile(tail_q)) for c in REJ_COLS}
    tail = pd.concat([qc[c] > t for c, t in thresholds.items()], axis=1).any(axis=1)
    none = pd.Series(False, index=qc.index)
    return ({"none (as reported)": none,
             "drop ICA-cap subjects": cap,
             "drop top-decile rejection": tail,
             "drop either": cap | tail}, thresholds)


def sensitivity_table(qc: pd.DataFrame, arms=SENSITIVITY_ARMS, seed: int = 42,
                      n_boot: int = 10000, n_perm: int = 5000) -> tuple:
    masks, thresholds = policy_masks(qc)
    oofs = {name: pd.read_csv(path) for name, path in arms}
    common = set.intersection(*(set(d["subject_id"]) for d in oofs.values()))
    rows = []
    for policy, drop in masks.items():
        keep = {s for s in common if not bool(drop.get(s, False))}
        for name, d in oofs.items():
            s = pooled_stats(d[d["subject_id"].isin(keep)], seed=seed, n_boot=n_boot, n_perm=n_perm)
            rows.append({"policy": policy, "arm": name, **s})
    return pd.DataFrame(rows), thresholds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="docs/qc_sensitivity.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    table, thresholds = sensitivity_table(pd.read_csv(QC_AUDIT), seed=args.seed)
    print("Top-decile thresholds measured from the cohort: " +
          ", ".join(f"{c[4:].upper()} > {t:.0%}" for c, t in thresholds.items()))
    print("Applied to fixed out-of-fold predictions; only the scored set changes.\n")
    for policy, g in table.groupby("policy", sort=False):
        n, na = int(g["n"].iloc[0]), int(g["n_adhd"].iloc[0])
        cells = "  ".join(f"{r.auc:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}] p={r.perm_p:.3f}"
                          for r in g.itertuples())
        print(f"{policy:28} n={n:3d} ({na} ADHD)  {cells}")
    print("\ncolumns: " + " | ".join(a for a, _ in SENSITIVITY_ARMS))
    table.to_csv(args.csv, index=False)
    print(f"-> {args.csv}")


if __name__ == "__main__":
    main()
