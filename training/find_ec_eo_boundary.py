"""
Is the detected EC/EO boundary real, or is the detector fitting noise?

WHAT THE FIRST VERSION FOUND
----------------------------
Sliding the split point and maximising |log(alpha_before / alpha_after)| across
108 subjects gave a median boundary at 0.564 -- not 0.50 -- with only 34/108
inside 0.45-0.55 and a median 1.44x "gain" over the midpoint. That suggested
split_eoec_by_alpha's hard-coded `half = n // 2` is wrong for most subjects.

WHY THAT WAS NOT ENOUGH
-----------------------
Three problems, all found by testing the method on synthetic data where the
answer is known:

1. GAIN IS MEANINGLESS ALONE. best_score is the MAXIMUM over every candidate
   boundary, so it is >= the midpoint score by construction. A pure-noise
   series with no changepoint at all scored gain = 1.46x at p = 0.350.
   Reporting gain without a null reports the optimiser, not the data.

2. THE OBVIOUS SPLIT-HALF TEST DOES NOT WORK. Asking "is the alpha ratio at
   the detected boundary more extreme in the other channel than at the
   midpoint" passed on pure noise too -- same selection problem.

3. THE VERDICT LOGIC PICKED THE WRONG EXPLANATION. It reported "mostly bad
   split point" (p=0.0358) when weak alpha blocking was the stronger signal
   (p=0.0002), because the branch was gated on an arbitrary threshold only
   4/14 subjects crossed.

WHAT THIS VERSION DOES INSTEAD
------------------------------
Two tests that DO separate signal from noise, both validated on synthetic data
before being applied here:

  SPLIT-HALF BY CHANNEL. Detect the boundary using O1 alone, then O2 alone. A
  physically real transition appears in both; noise-fitting does not.
  Median |k(O1) - k(O2)| over 30 simulations:
        real, strong blocking   0.000
        real, moderate          0.000
        real, weak              0.033
        no boundary             0.258
        no boundary, high var   0.250

  PERMUTATION NULL. Shuffle the alpha profile in time and re-detect, 200x.
  This destroys temporal STRUCTURE while keeping the values, so it asks "is
  there a changepoint" rather than "is there variance".
        real, strong   p = 0.000
        real, weak     p = 0.000
        no boundary    p = 0.350   (despite gain 1.46x)

A subject's boundary is trusted only if BOTH pass.

Read-only. Changes no pipeline code, regenerates no images.

    py -m training.find_ec_eo_boundary
    py -m training.find_ec_eo_boundary --n-perm 500 --csv docs/ec_eo_boundaries.csv
"""

import argparse
import warnings
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import welch

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

from data_pipeline import subject_split
from data_pipeline.preprocessing import (ALPHA_AMBIGUOUS_RATIO_RANGE,
                                         ALPHA_BAND_HZ, filter_raw, load_raw,
                                         remove_artifacts_ica)

WINDOW_SEC = 4.0
TRIM_SEC = 15.0

# Was 0.20. Lowered so the search is not artificially clipped: a boundary whose
# true optimum lies below the constraint silently reports AS the constraint,
# which looks like a detection rather than a failure. Six subjects sat at 0.20
# in the first run; relaxing to 0.10 moved every one of them (0.137-0.146, and
# F10050108 to 0.097), proving those values were clipped, not detected.
# Rejection is now the plausibility guard's job, not the constraint's.
MIN_SEGMENT_FRAC = 0.10

# A statistical test needs a physical sanity check on top of it.
#
# The four "pinned but trusted" subjects passed EVERY statistical test --
# permutation p = 0.005, split-half difference 0.000 -- and were still wrong.
# Their boundaries put one condition at 47-70 s of an 8-minute recording. The
# detector was finding something real (an artifact burst, or a brief
# eye-opening) but it was not the EC/EO transition, and nothing in the
# statistics could tell the difference.
#
# 0.25-0.75 spans roughly a 2-min/6-min split either way -- wide enough for
# genuine protocol variation, narrow enough to reject a 70-second "condition".
# Boundaries outside it fall back to the midpoint REGARDLESS of how well they
# validate.
PLAUSIBLE_BOUNDARY_RANGE = (0.25, 0.75)

N_PERM_DEFAULT = 200
PERM_ALPHA = 0.05
# Max |k(O1) - k(O2)| for the channels to count as agreeing. 0.08 of the
# recording is ~20 s on an 8-minute file. Set from the synthetic separation
# above (real <= 0.033, noise >= 0.25) -- between the two, not tuned on the
# real data.
SPLIT_HALF_TOL = 0.08

# NOTE ON ICA: computed on filtered-but-not-ICA-cleaned data. The cohort run
# splits POST-ICA, and pre- vs post-ICA alpha ratios differ materially on 5 of
# 14 ambiguous subjects (C10071110 0.79->2.91, C10061115 1.02->2.14,
# F09080101 0.96->0.47, F11101129 0.92->1.33, F12111128 1.03->0.84). Numbers
# here are NOT directly comparable to build_log.csv. Both reported so the
# difference stays visible rather than being averaged away.


def alpha_profile(raw: mne.io.Raw, picks: list) -> np.ndarray:
    """Alpha power in consecutive non-overlapping windows, for the given channels."""
    sfreq = float(raw.info["sfreq"])
    sig = raw.get_data(picks=picks).mean(axis=0)
    trim = int(TRIM_SEC * sfreq)
    sig = sig[trim:len(sig) - trim]
    win = int(WINDOW_SEC * sfreq)
    n_win = len(sig) // win
    if n_win < 10:
        raise ValueError(f"Only {n_win} windows after trimming -- recording too short.")
    out = np.empty(n_win)
    for i in range(n_win):
        seg = sig[i * win:(i + 1) * win]
        freqs, psd = welch(seg, fs=sfreq, nperseg=len(seg))
        band = (freqs >= ALPHA_BAND_HZ[0]) & (freqs <= ALPHA_BAND_HZ[1])
        out[i] = psd[band].mean()
    return out


def detect(powers: np.ndarray) -> tuple:
    """(boundary_fraction, score) for the split maximising |log alpha ratio|.

    Maximising that specific quantity, not a generic changepoint statistic,
    because it is exactly what split_eoec_by_alpha thresholds on. Optimising
    anything else would find a boundary the pipeline then fails to act on.

    MIN_SEGMENT_FRAC prevents the degenerate answer where one side is a handful
    of windows and the ratio is noise. Subjects whose optimum sits AT this bound
    are reported separately -- the optimiser wanted to go further, which is
    either a genuinely extreme boundary or edge-running on noise.
    """
    n = len(powers)
    lo, hi = int(n * MIN_SEGMENT_FRAC), int(n * (1 - MIN_SEGMENT_FRAC))
    scores = np.full(n, np.nan)
    for k in range(lo, hi):
        a, b = powers[:k].mean(), powers[k:].mean()
        if a > 0 and b > 0:
            scores[k] = abs(np.log(a / b))
    if np.all(np.isnan(scores)):
        return float("nan"), float("nan")
    k = int(np.nanargmax(scores))
    return k / n, float(scores[k])


def permutation_p(powers: np.ndarray, observed: float, n_perm: int, seed: int) -> tuple:
    """Fraction of time-shuffled profiles reaching a score >= observed.

    Shuffling destroys temporal ordering while preserving the value
    distribution, so this asks specifically "is there a CHANGEPOINT here", not
    "is there variance here". A profile with no transition scores no better
    than its own shuffles.
    """
    rng = np.random.default_rng(seed)
    null = np.array([detect(rng.permutation(powers))[1] for _ in range(n_perm)])
    # +1 top and bottom: unbiased, and can never return exactly 0, which would
    # overstate the evidence.
    p = (np.sum(null >= observed) + 1) / (n_perm + 1)
    return float(p), float(np.median(null))


def ratio_at(powers: np.ndarray, frac: float) -> float:
    k = max(1, min(len(powers) - 1, int(round(frac * len(powers)))))
    a, b = powers[:k].mean(), powers[k:].mean()
    return float(a / b) if b > 0 else float("nan")


def analyse(eoec_path: str, n_perm: int, seed: int, post_ica: bool = True) -> dict:
    """
    post_ica=True by default, because the pipeline calls split_eoec_by_alpha
    AFTER remove_artifacts_ica. The first version of this script measured
    pre-ICA and the two are not the same signal: on a 20-subject check, ICA
    moved the boundary by more than the split-half tolerance for 2 of the 17
    subjects whose boundary was detectable at all (F11121121 0.197->0.628,
    F12070126 0.752->0.559). Validating on data the pipeline never sees would
    mean regenerating 122k images on the wrong measurement.
    """
    raw = filter_raw(load_raw(eoec_path))
    if post_ica:
        raw, _ = remove_artifacts_ica(raw)
    have = [c for c in ("O1", "O2") if c in raw.ch_names]
    if not have:
        raise ValueError("No occipital channels (O1/O2).")

    both = alpha_profile(raw, have)
    frac, score = detect(both)
    mid_ratio = ratio_at(both, 0.5)
    best_ratio = ratio_at(both, frac)
    mid_score = abs(np.log(mid_ratio)) if mid_ratio > 0 else float("nan")

    p_perm, null_med = permutation_p(both, score, n_perm, seed)

    if len(have) == 2:
        f1, _ = detect(alpha_profile(raw, ["O1"]))
        f2, _ = detect(alpha_profile(raw, ["O2"]))
        split_diff = abs(f1 - f2)
    else:
        f1 = f2 = split_diff = float("nan")

    at_bound = (frac <= MIN_SEGMENT_FRAC + 1e-6) or (frac >= 1 - MIN_SEGMENT_FRAC - 1e-6)

    return {
        "boundary_frac": frac, "best_score": score, "midpoint_score": mid_score,
        "gain": score / mid_score if mid_score > 0 else float("nan"),
        "midpoint_ratio": mid_ratio, "best_ratio": best_ratio,
        "perm_p": p_perm, "perm_null_median": null_med,
        "o1_frac": f1, "o2_frac": f2, "split_half_diff": split_diff,
        "pinned_at_bound": at_bound, "n_windows": len(both),
        "passes_perm": p_perm < PERM_ALPHA,
        "passes_split_half": bool(split_diff <= SPLIT_HALF_TOL),
        "plausible": bool(PLAUSIBLE_BOUNDARY_RANGE[0] <= frac
                          <= PLAUSIBLE_BOUNDARY_RANGE[1]),
        "post_ica": post_ica,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data_pipeline/splits/subject_splits.csv")
    ap.add_argument("--build-log", default="dataset/build_log.csv")
    ap.add_argument("--csv", default="docs/ec_eo_boundaries.csv")
    ap.add_argument("--n-perm", type=int, default=N_PERM_DEFAULT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pre-ica", action="store_true",
                    help="skip ICA (fast, but NOT what the pipeline splits on)")
    args = ap.parse_args()
    post_ica = not args.pre_ica

    man = subject_split.load_manifest(args.manifest)
    if args.limit:
        man = man.head(args.limit)

    skipped = set()
    if Path(args.build_log).exists():
        bl = pd.read_csv(args.build_log)
        skipped = set(bl.loc[bl.status.astype(str).str.contains("ambiguous"), "subject_id"])
        print(f"Cohort run skipped {len(skipped)} subjects as ambiguous.")
    print(f"Permutations per subject: {args.n_perm}")
    print(f"ICA: {'APPLIED (matches the pipeline)' if post_ica else 'SKIPPED (--pre-ica)'}\n")

    rows = []
    for i, r in man.reset_index(drop=True).iterrows():
        print(f"[{i+1}/{len(man)}] {r.subject_id} ... ", end="", flush=True)
        try:
                res = analyse(r.eoec_path, args.n_perm, args.seed, post_ica=post_ica)
        except Exception as e:                       # noqa: BLE001
            print(f"FAILED: {e}")
            continue
        res.update(subject_id=r.subject_id, group=r.group, split=r.split,
                   skipped_by_build=r.subject_id in skipped)
        rows.append(res)
        marks = ("P" if res["passes_perm"] else "-") \
              + ("S" if res["passes_split_half"] else "-") \
              + ("L" if res["plausible"] else "-")
        print(f"k={res['boundary_frac']:.2f} score={res['best_score']:.2f} "
              f"p={res['perm_p']:.3f} halfdiff={res['split_half_diff']:.2f} [{marks}]"
              + ("  PINNED" if res["pinned_at_bound"] else ""))

    df = pd.DataFrame(rows)
    if df.empty:
        print("\nNothing analysed.")
        return
        # Plausibility is an AND, not a tiebreaker: four subjects passed both
    # statistical tests with a boundary that put one condition at ~60 seconds.
    df["trusted"] = df.passes_perm & df.passes_split_half & df.plausible
    print("\n" + "=" * 78)
    print("VALIDATION -- is the detected boundary real?")
    print("=" * 78)
    print(f"  passes permutation test (p < {PERM_ALPHA}):    "
          f"{df.passes_perm.sum():>3}/{len(df)}  ({df.passes_perm.mean():.1%})")
    print(f"  passes split-half (|O1-O2| <= {SPLIT_HALF_TOL}):   "
          f"{df.passes_split_half.sum():>3}/{len(df)}  ({df.passes_split_half.mean():.1%})")
    print(f"  plausible boundary {PLAUSIBLE_BOUNDARY_RANGE}:        "
          f"{df.plausible.sum():>3}/{len(df)}  ({df.plausible.mean():.1%})")
    stat_ok = df.passes_perm & df.passes_split_half
    print(f"    of which passed BOTH stats but are implausible: "
          f"{(stat_ok & ~df.plausible).sum()}")
    print(f"  passes BOTH -> boundary trusted:         "
          f"{df.trusted.sum():>3}/{len(df)}  ({df.trusted.mean():.1%})")
    print(f"\n  median split-half difference: {df.split_half_diff.median():.3f}")
    print(f"  median permutation p:         {df.perm_p.median():.4f}")
    print("\n  Reminder: median gain was 1.44x in the unvalidated version. A synthetic")
    print("  series with NO changepoint scored gain 1.46x at p = 0.350, so gain alone")
    print("  proves nothing. These two columns are the evidence.")

    tr = df[df.trusted]
    if not tr.empty:
        print("\n" + "=" * 78)
        print(f"WHERE IS THE BOUNDARY -- trusted subjects only (n={len(tr)})")
        print("=" * 78)
        bf = tr.boundary_frac
        print(f"  median {bf.median():.3f}   mean {bf.mean():.3f}   sd {bf.std():.3f}")
        print(f"  within 0.45-0.55: {((bf >= .45) & (bf <= .55)).sum()}/{len(tr)}")
        print(f"  median |boundary - 0.5|: {(bf - 0.5).abs().median():.3f}")
        try:
            from scipy.stats import wilcoxon
            print(f"  Wilcoxon vs 0.5: p = {wilcoxon(bf - 0.5).pvalue:.5f}")
        except Exception:                            # noqa: BLE001
            pass
        print("\n  If the median is reliably above 0.5, the EC segment is systematically")
        print("  LONGER than half and `half = n // 2` mislabels EO data as EC in most")
        print("  subjects -- not only the flagged ones.")

    print("\n" + "=" * 78)
    print(f"PINNED AT MIN_SEGMENT_FRAC ({MIN_SEGMENT_FRAC})")
    print("=" * 78)
    pin = df[df.pinned_at_bound]
    print(f"  {len(pin)}/{len(df)} subjects. The optimiser wanted to exceed the limit --")
    print("  either a genuinely extreme boundary or edge-running on noise.")
    if not pin.empty:
        print(f"  of these, trusted by both tests: {pin.trusted.sum()}/{len(pin)}")
        print("\n" + pin[["subject_id", "split", "boundary_frac", "best_score", "perm_p",
                          "split_half_diff", "trusted"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    if df.skipped_by_build.any():
        amb, ok = df[df.skipped_by_build], df[~df.skipped_by_build]
        print("\n" + "=" * 78)
        print("TWO EXPLANATIONS FOR THE FLAGGED SUBJECTS -- ranked by evidence")
        print("=" * 78)
        try:
            from scipy.stats import mannwhitneyu
            p_a = mannwhitneyu((amb.boundary_frac - .5).abs(),
                               (ok.boundary_frac - .5).abs()).pvalue
            p_b = mannwhitneyu(amb.best_score, ok.best_score).pvalue
        except Exception:                            # noqa: BLE001
            p_a = p_b = float("nan")

        ev = sorted([
            ("A: bad split point (boundary off-centre)", p_a,
             (amb.boundary_frac - .5).abs().median(), (ok.boundary_frac - .5).abs().median()),
            ("B: weak alpha blocking (low score)", p_b,
             amb.best_score.median(), ok.best_score.median()),
        ], key=lambda t: t[1])

        print(f"  {'explanation':<44} {'p':>9} {'flagged':>9} {'rest':>8}")
        for name, p, m_a, m_o in ev:
            print(f"  {name:<44} {p:>9.4f} {m_a:>9.3f} {m_o:>8.3f}")
        print("\n  Both reported and ranked rather than one declared the winner: an")
        print("  earlier version forced a single answer via an arbitrary threshold")
        print("  and named the WEAKER explanation.")
        print(f"\n  Flagged subjects whose boundary is trusted: {amb.trusted.sum()}/{len(amb)}")
        if amb.trusted.sum():
            rec = amb[amb.trusted]
            esc = rec[(rec.best_ratio <= ALPHA_AMBIGUOUS_RATIO_RANGE[0]) |
                      (rec.best_ratio >= ALPHA_AMBIGUOUS_RATIO_RANGE[1])]
            print(f"  ... and escaping the ambiguous band at that boundary: {len(esc)}")
            if len(esc):
                print(f"    {', '.join(esc.subject_id)}")
                print(f"    (test split: {(esc.split == 'test').sum()})")

        print("\n  Per flagged subject:")
        print(amb[["subject_id", "group", "split", "boundary_frac", "best_score",
                   "perm_p", "split_half_diff", "midpoint_ratio", "best_ratio", "trusted"]]
              .sort_values("split").to_string(index=False,
                                              float_format=lambda v: f"{v:.3f}"))

    print("\n" + "=" * 78)
    print("WHAT TO DO")
    print("=" * 78)
    if df.trusted.mean() > 0.7:
        print("  Most boundaries validate. Replacing `half = n // 2` in")
        print("  split_eoec_by_alpha with this detector is justified -- but it means")
        print("  regenerating all ~122k images. Do that on this evidence, not on gain.")
    elif df.trusted.mean() > 0.3:
        print("  Mixed. Boundary detection works for some subjects and not others.")
        print("  Safer: keep the midpoint by default, use the detected boundary only")
        print("  where BOTH tests pass, and record which subjects got which.")
    else:
        print("  Most boundaries do NOT validate. The detector is largely fitting")
        print("  noise, and the 0.564 median from the unvalidated version should not")
        print("  be acted on. Keep the midpoint split and report the flagged subjects")
        print("  as excluded.")

    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv, index=False)
    print(f"\nSaved -> {args.csv}")


if __name__ == "__main__":
    main()