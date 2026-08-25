"""
Two checks that must pass BEFORE split_eoec_by_alpha is changed.

Boundary detection validated on 85/108 subjects (permutation p<0.05 AND
split-half |O1-O2|<=0.08), trusted-only median 0.567, Wilcoxon vs 0.5
p<0.00001. That justifies replacing `half = n // 2`. But two things about the
validation itself have to be checked first, or the change would rest on
evidence that does not apply to the pipeline as it actually runs.

CHECK 1 -- DOES THE BOUNDARY SURVIVE ICA?
-----------------------------------------
find_ec_eo_boundary.py computed everything on filtered-but-NOT-ICA-cleaned
data. The pipeline calls split_eoec_by_alpha AFTER remove_artifacts_ica. Those
are not the same signal, and we already know the difference matters: pre- vs
post-ICA alpha ratios diverge materially on 5 of the 14 flagged subjects
(C10071110 0.79->2.91, C10061115 1.02->2.14, F09080101 0.96->0.47,
F11101129 0.92->1.33, F12111128 1.03->0.84).

If ICA moves the BOUNDARY as much as it moves the RATIO, then a validation run
pre-ICA says nothing about what the pipeline will do post-ICA, and regenerating
122k images on that basis would be building on the wrong measurement.

Passes if the post-ICA boundary lands within SPLIT_HALF_TOL of the pre-ICA one
for the large majority of subjects -- the same tolerance the split-half test
uses, so "ICA moves it" is judged against the same yardstick as "the channels
disagree".

CHECK 2 -- ARE THE PINNED BOUNDARIES REAL, OR JUST CLIPPED?
-----------------------------------------------------------
Six subjects landed at MIN_SEGMENT_FRAC = 0.20, meaning the optimiser wanted to
go further and was stopped. Four of them (C12030157, F08080102, F10050108,
F12081133) passed BOTH validation tests with split-half difference 0.000 -- so
there is a real transition, but its reported position is an artefact of where
the search was cut off, not where the transition is.

Re-runs those with MIN_SEGMENT_FRAC = 0.10. If the boundary moves substantially
lower, 0.20 was a clipped value and using it would mislabel a large block of
each recording. If it stays near 0.20, the constraint was not binding after all
and the value stands.

Read-only. Changes no pipeline code.

    py -m training.precheck_boundary
    py -m training.precheck_boundary --n-subjects 30 --csv docs/precheck_boundary.csv
"""

import argparse
import warnings
from pathlib import Path

import mne
import numpy as np
import pandas as pd

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

from data_pipeline import subject_split
from data_pipeline.preprocessing import filter_raw, load_raw, remove_artifacts_ica
from training.find_ec_eo_boundary import (SPLIT_HALF_TOL, alpha_profile,
                                          permutation_p)

# Subjects that hit the MIN_SEGMENT_FRAC bound in the validation run. The first
# four passed both tests; the last two did not and are included as controls --
# if the relaxed search behaves the same way for trusted and untrusted pinned
# subjects, that itself is informative.
PINNED_SUBJECTS = ["C12030157", "F08080102", "F10050108", "F12081133",
                   "F11121121", "F11101129"]
PINNED_TRUSTED = {"C12030157", "F08080102", "F10050108", "F12081133"}
RELAXED_MIN_FRAC = 0.10
CONSISTENCY_PASS_RATE = 0.80


def detect_with_frac(powers: np.ndarray, min_frac: float) -> tuple:
    """detect() with the segment constraint as an argument.

    Mirrors find_ec_eo_boundary.detect exactly; the constraint is a parameter
    here only so check 2 can relax it. Kept as a separate function rather than
    monkey-patching the module constant, which would silently change behaviour
    for anything else importing it in the same process.
    """
    n = len(powers)
    lo, hi = int(n * min_frac), int(n * (1 - min_frac))
    scores = np.full(n, np.nan)
    for k in range(lo, hi):
        a, b = powers[:k].mean(), powers[k:].mean()
        if a > 0 and b > 0:
            scores[k] = abs(np.log(a / b))
    if np.all(np.isnan(scores)):
        return float("nan"), float("nan")
    k = int(np.nanargmax(scores))
    return k / n, float(scores[k])


def check_ica_consistency(man: pd.DataFrame, n_subjects: int, n_perm: int,
                          seed: int) -> pd.DataFrame:
    # Stratify: ICA behaviour could plausibly differ by group, and taking the
    # first N would return mostly Controls (IDs sort C before F, and that first
    # letter IS the group label).
    per = max(1, n_subjects // 2)
    sel = pd.concat([g.head(per) for _, g in man.groupby("group")]).reset_index(drop=True)

    rows = []
    for i, r in sel.iterrows():
        print(f"[{i+1}/{len(sel)}] {r.subject_id} ... ", end="", flush=True)
        try:
            raw = filter_raw(load_raw(r.eoec_path))
            have = [c for c in ("O1", "O2") if c in raw.ch_names]
            pre = alpha_profile(raw, have)
            k_pre, s_pre = detect_with_frac(pre, 0.20)
            p_pre, _ = permutation_p(pre, s_pre, n_perm, seed)

            raw_ica, _ = remove_artifacts_ica(raw)          # the slow part
            post = alpha_profile(raw_ica, have)
            k_post, s_post = detect_with_frac(post, 0.20)
            p_post, _ = permutation_p(post, s_post, n_perm, seed)
        except Exception as e:                              # noqa: BLE001
            print(f"FAILED: {e}")
            continue

        d = abs(k_pre - k_post)
        rows.append({"subject_id": r.subject_id, "group": r.group,
                     "k_pre": k_pre, "k_post": k_post, "abs_diff": d,
                     "score_pre": s_pre, "score_post": s_post,
                     "p_pre": p_pre, "p_post": p_post,
                     "agrees": d <= SPLIT_HALF_TOL,
                     "both_significant": (p_pre < 0.05) and (p_post < 0.05)})
        print(f"pre {k_pre:.2f} -> post {k_post:.2f}  diff {d:.3f}  "
              f"{'OK' if d <= SPLIT_HALF_TOL else 'MOVED'}")
    return pd.DataFrame(rows)


def check_pinned(man: pd.DataFrame, n_perm: int, seed: int) -> pd.DataFrame:
    rows = []
    for sid in PINNED_SUBJECTS:
        hit = man.loc[man.subject_id == sid]
        if hit.empty:
            print(f"  {sid}: not in manifest, skipping")
            continue
        print(f"  {sid} ... ", end="", flush=True)
        try:
            raw = filter_raw(load_raw(hit.iloc[0].eoec_path))
            have = [c for c in ("O1", "O2") if c in raw.ch_names]
            prof = alpha_profile(raw, have)
            k_20, s_20 = detect_with_frac(prof, 0.20)
            k_10, s_10 = detect_with_frac(prof, RELAXED_MIN_FRAC)
            p_10, _ = permutation_p(prof, s_10, n_perm, seed)
            f1, _ = detect_with_frac(alpha_profile(raw, ["O1"]), RELAXED_MIN_FRAC)
            f2, _ = detect_with_frac(alpha_profile(raw, ["O2"]), RELAXED_MIN_FRAC)
        except Exception as e:                              # noqa: BLE001
            print(f"FAILED: {e}")
            continue

        moved = abs(k_10 - k_20) > 0.02
        rows.append({"subject_id": sid, "split": hit.iloc[0].split,
                     "was_trusted": sid in PINNED_TRUSTED,
                     "k_at_0.20": k_20, "k_at_0.10": k_10,
                     "score_0.20": s_20, "score_0.10": s_10,
                     "perm_p_0.10": p_10, "split_half_0.10": abs(f1 - f2),
                     "moved_when_relaxed": moved,
                     "still_pinned": k_10 <= RELAXED_MIN_FRAC + 1e-6})
        print(f"0.20 -> {k_10:.3f}  {'MOVED (was clipped)' if moved else 'stayed'}")
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data_pipeline/splits/subject_splits.csv")
    ap.add_argument("--n-subjects", type=int, default=20)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--csv", default="docs/precheck_boundary.csv")
    args = ap.parse_args()

    man = subject_split.load_manifest(args.manifest)

    print("=" * 78)
    print(f"CHECK 1 -- does the boundary survive ICA?  ({args.n_subjects} subjects)")
    print("=" * 78)
    print("ICA is fitted per recording, so this is the slow part (~30 s/subject).\n")
    ica = check_ica_consistency(man, args.n_subjects, args.n_perm, args.seed)

    if ica.empty:
        print("\n  No subjects analysed -- cannot proceed.")
        return

    rate = ica.agrees.mean()
    print(f"\n  agree within {SPLIT_HALF_TOL}: {ica.agrees.sum()}/{len(ica)}  ({rate:.1%})")
    print(f"  median |k_pre - k_post|: {ica.abs_diff.median():.4f}")
    print(f"  max    |k_pre - k_post|: {ica.abs_diff.max():.4f}")
    print(f"  significant both pre and post: {ica.both_significant.sum()}/{len(ica)}")

    if not ica.agrees.all():
        print("\n  Subjects whose boundary MOVED:")
        print(ica.loc[~ica.agrees, ["subject_id", "group", "k_pre", "k_post",
                                    "abs_diff", "p_pre", "p_post"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    check1 = rate >= CONSISTENCY_PASS_RATE
    print(f"\n  CHECK 1: {'PASS' if check1 else 'FAIL'}")
    if check1:
        print("  ICA does not materially move the boundary. The pre-ICA validation")
        print("  transfers to the pipeline, which splits post-ICA.")
    else:
        print(f"  ICA moves the boundary on {1-rate:.0%} of subjects. The pre-ICA")
        print("  validation does NOT describe what the pipeline would do. Do not")
        print("  regenerate on it. Either re-run the full validation post-ICA, or")
        print("  move the split before ICA -- but decide deliberately, and note that")
        print("  splitting pre-ICA changes what ICA is fitted on, which is its own")
        print("  question rather than a free fix.")

    print("\n" + "=" * 78)
    print(f"CHECK 2 -- pinned boundaries, re-run at MIN_SEGMENT_FRAC = {RELAXED_MIN_FRAC}")
    print("=" * 78)
    pin = check_pinned(man, args.n_perm, args.seed)

    if not pin.empty:
        print("\n" + pin.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        tr = pin[pin.was_trusted]
        print(f"\n  Of the {len(tr)} trusted pinned subjects, "
              f"{tr.moved_when_relaxed.sum()} moved when the constraint was relaxed.")
        if tr.moved_when_relaxed.any():
            print("  Their 0.20 values were CLIPPED, not detected. Using them would")
            print("  mislabel a large block of each recording. Use the 0.10 values, or")
            print("  exclude these subjects -- do not use 0.20.")
        if tr.still_pinned.any():
            print(f"\n  Still at the bound even at {RELAXED_MIN_FRAC}: "
                  f"{', '.join(tr.loc[tr.still_pinned, 'subject_id'])}")
            print("  The optimum is outside any plausible protocol split. Treat as")
            print("  edge-running on noise and fall back to the midpoint for these.")

    print("\n" + "=" * 78)
    print("GATE")
    print("=" * 78)
    if check1:
        print("  Check 1 passed -- proceed to patching split_eoec_by_alpha.")
        print("  Apply check 2's findings to the pinned subjects: use the relaxed")
        print("  boundary where it is stable, midpoint where it is not.")
    else:
        print("  Check 1 FAILED -- do NOT patch split_eoec_by_alpha yet, and do not")
        print("  regenerate. Resolve the pre/post-ICA discrepancy first.")

    out = ica.assign(check="ica_consistency")
    if not pin.empty:
        out = pd.concat([out, pin.assign(check="pinned_relaxed")], ignore_index=True)
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.csv, index=False)
    print(f"\nSaved -> {args.csv}")


if __name__ == "__main__":
    main()