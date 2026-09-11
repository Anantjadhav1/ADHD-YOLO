"""
Paired comparison between arms, on the same subjects.

WHY A PAIRED TEST AND NOT A CI COMPARISON
-----------------------------------------
significance_test.py bootstraps one arm's accuracy and compares the interval
against a fixed 75.8% from a DIFFERENT split of the data. That is not a
hypothesis test. With ~17 subjects per fold the interval spans roughly +/-24
points, so it cannot separate any two plausible values.

The fold-level AUCs make the reason concrete: the topomap run produced 0.347,
0.365, 0.375, 0.750 and 0.833 across five folds of ~17 subjects. Almost all of
that spread is WHICH SUBJECTS landed in which fold, not model quality. A paired
test cancels it exactly, because both arms are scored on the same people --
subject difficulty enters both predictions and subtracts out.

THE INTERSECTION MATTERS
------------------------
The arms do not cover identical subjects. The classical arm has 91 development
subjects; the CNN arms have 84, because 7 EC/EO-ambiguous subjects have
classical features but no generated images. Comparing 91 against 84 would be
comparing two models on partly different people and calling the difference a
result. This script intersects on subject_id and reports how many were dropped.

WHAT IT REPORTS
---------------
  McNemar -- on the DISCORDANT pairs only (one arm right, the other wrong).
      Concordant pairs carry no information about which arm is better and are
      excluded by construction. With n=84 the discordant count may be small, in
      which case the exact binomial is used and the power is stated plainly
      rather than a p-value being quoted as though it settled something.

  Paired bootstrap on the AUC difference -- resamples SUBJECTS, not
      predictions, so both arms are re-scored on the same resampled people and
      the pairing is preserved. Resampling predictions independently would
      destroy exactly the correlation this test exists to exploit.

  Agreement -- Cohen's kappa, plus the raw confusion of the two arms against
      each other. Two chance-level models that disagree completely are a
      different situation from two that agree on the same wrong answers.

    python -m training.paired_comparison \\
        --arm CNN_scalogram runs/frozen/scalogram_oof_cnn_probs.csv \\
        --arm CNN_topomap runs/frozen_topomap/topomap_oof_cnn_probs.csv \\
        --arm Classical runs/classical/classical_oof_probs.csv
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (len(pos) * len(neg)))


def load_arm(path: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    need = {"subject_id", "true_label", "pred_prob_adhd", "pred_label"}
    missing = need - set(d.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    return d[["subject_id", "true_label", "pred_prob_adhd", "pred_label"]]


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Exact McNemar on the discordant pairs.

    b = A right, B wrong.  c = A wrong, B right.  Concordant pairs are
    uninformative about which arm is better and play no part.

    Uses the exact binomial rather than the chi-square approximation: with
    n=84 the discordant count is often under 25, where the approximation is
    unreliable in the direction that matters (too small a p-value).
    """
    b = int((correct_a & ~correct_b).sum())
    c = int((~correct_a & correct_b).sum())
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "n_discordant": 0, "p": float("nan"),
                "note": "arms are identical on every subject"}
    from scipy.stats import binomtest
    p = binomtest(b, n, 0.5).pvalue
    return {"b": b, "c": c, "n_discordant": n, "p": float(p), "note": ""}


def paired_bootstrap_auc(y: np.ndarray, pa: np.ndarray, pb: np.ndarray,
                         n_boot: int = 10000, seed: int = 42) -> dict:
    """Bootstrap the AUC difference by resampling SUBJECTS.

    Both arms are re-scored on the same resampled subjects each iteration, so
    the correlation between them survives. Resampling each arm's predictions
    independently would break the pairing and inflate the variance of the
    difference -- which is the whole quantity of interest.
    """
    rng = np.random.default_rng(seed)
    obs = auc(pa[y], pa[~y]) - auc(pb[y], pb[~y])
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        yi = y[idx]
        if not (0 < yi.sum() < len(yi)):
            continue
        d = auc(pa[idx][yi], pa[idx][~yi]) - auc(pb[idx][yi], pb[idx][~yi])
        if np.isfinite(d):
            diffs.append(d)
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # Two-sided p: fraction of the bootstrap distribution on the far side of
    # zero, doubled. Not a permutation test -- it asks whether the observed
    # difference is distinguishable from zero given resampling variability.
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"observed": float(obs), "ci_lo": float(lo), "ci_hi": float(hi),
            "p": float(min(p, 1.0))}


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    po = (a == b).mean()
    pe = (a.mean() * b.mean()) + ((1 - a.mean()) * (1 - b.mean()))
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", nargs=2, action="append", metavar=("NAME", "CSV"),
                    required=True, help="repeat for each arm")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--csv", default="docs/paired_comparison.csv")
    args = ap.parse_args()

    arms = {}
    for name, path in args.arm:
        if not Path(path).exists():
            print(f"  {name}: {path} not found, skipping")
            continue
        arms[name] = load_arm(path)
        print(f"  {name}: {len(arms[name])} subjects from {path}")
    if len(arms) < 2:
        raise SystemExit("Need at least two arms to compare.")

    common = set.intersection(*(set(d.subject_id) for d in arms.values()))
    common = sorted(common)
    print(f"\nIntersection: {len(common)} subjects present in all arms")
    for name, d in arms.items():
        dropped = len(d) - len(common)
        if dropped:
            missing = sorted(set(d.subject_id) - set(common))
            print(f"  {name}: {dropped} not in the intersection -> {missing}")
    print("  Comparing arms on different subject sets would attribute a")
    print("  difference in COHORT to a difference in METHOD.")

    aligned = {}
    for name, d in arms.items():
        s = d[d.subject_id.isin(common)].set_index("subject_id").loc[common]
        aligned[name] = s
    truth = (aligned[next(iter(aligned))]["true_label"] == "ADHD").to_numpy()
    for name, s in aligned.items():
        t = (s["true_label"] == "ADHD").to_numpy()
        if not np.array_equal(t, truth):
            raise ValueError(f"{name} disagrees with another arm on the true labels.")

    print(f"\n{'=' * 74}\nPER-ARM, ON THE SHARED {len(common)} SUBJECTS "
          f"({truth.sum()} ADHD / {(~truth).sum()} Control)\n{'=' * 74}")
    print(f"  {'arm':<20}{'AUC':>8}{'accuracy':>10}{'sens':>8}{'spec':>8}")
    for name, s in aligned.items():
        p = s["pred_prob_adhd"].to_numpy()
        pred = (s["pred_label"] == "ADHD").to_numpy()
        print(f"  {name:<20}{auc(p[truth], p[~truth]):>8.3f}"
              f"{(pred == truth).mean():>10.3f}"
              f"{pred[truth].mean():>8.3f}{1 - pred[~truth].mean():>8.3f}")

    rows = []
    print(f"\n{'=' * 74}\nPAIRWISE\n{'=' * 74}")
    for a, b in itertools.combinations(aligned, 2):
        sa, sb = aligned[a], aligned[b]
        pa, pb = sa["pred_prob_adhd"].to_numpy(), sb["pred_prob_adhd"].to_numpy()
        preda = (sa["pred_label"] == "ADHD").to_numpy()
        predb = (sb["pred_label"] == "ADHD").to_numpy()
        ca, cb = preda == truth, predb == truth

        mc = mcnemar(ca, cb)
        bs = paired_bootstrap_auc(truth, pa, pb, seed=args.seed)
        k = kappa(preda, predb)

        print(f"\n  {a}  vs  {b}")
        print(f"    AUC difference   {bs['observed']:+.4f}   "
              f"95% CI [{bs['ci_lo']:+.3f}, {bs['ci_hi']:+.3f}]   p = {bs['p']:.3f}")
        if mc["n_discordant"] == 0:
            print(f"    McNemar          {mc['note']}")
        else:
            print(f"    McNemar          b={mc['b']} (only {a} right), "
                  f"c={mc['c']} (only {b} right), p = {mc['p']:.3f}")
            if mc["n_discordant"] < 25:
                print(f"      NOTE: only {mc['n_discordant']} discordant pairs. Exact binomial")
                print("      used, but the test has little power at this count -- a")
                print("      non-significant result here does NOT establish equivalence.")
        print(f"    agreement        kappa = {k:+.3f}  "
              f"({(preda == predb).mean():.1%} of predictions match)")
        rows.append({"arm_a": a, "arm_b": b, "n": len(common),
                     "auc_diff": bs["observed"], "auc_diff_ci_lo": bs["ci_lo"],
                     "auc_diff_ci_hi": bs["ci_hi"], "auc_diff_p": bs["p"],
                     "mcnemar_b": mc["b"], "mcnemar_c": mc["c"],
                     "mcnemar_p": mc["p"], "kappa": k})

    print(f"\n{'=' * 74}\nHOW TO READ THIS\n{'=' * 74}")
    print("  A non-significant difference between two arms that are BOTH at chance")
    print("  says they are equivalently uninformative -- not that they are")
    print("  equivalently good. Report the per-arm AUCs alongside, never the")
    print("  comparison alone.")
    print("\n  Low kappa between two chance-level arms means they are wrong about")
    print("  DIFFERENT subjects, which leaves room for fusion to help. High kappa")
    print("  means they fail on the same subjects, and fusion will not.")

    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.csv, index=False)
    print(f"\n  -> {args.csv}")


if __name__ == "__main__":
    main()