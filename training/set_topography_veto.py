"""
Set the ICA topography veto thresholds from measured data, not convention.

THE PROBLEM THIS FIXES
----------------------
audit_ica_alpha.py found ICA removing a median 36% of occipital alpha across
the cohort -- 68/108 subjects losing >25%, 34/108 losing >50%, worst 0.02x.

The cause is not occipital-dominant components (only 9 of 313 are). It is that
the median REMOVED component has a near-uniform topography:

                                    occipital   frontal
    a genuine ocular component           0.09      4.05
    a temporal muscle component          0.10      0.22
    the median component removed        0.951     1.000

1.0 everywhere means focal nowhere. find_bads_eog and find_bads_muscle judge
components on SPECTRAL criteria alone, so a spatially diffuse component whose
frequency content happens to match gets removed -- and removing a diffuse
component subtracts signal from every channel proportionally, occipital alpha
included.

The fix is a spatial sanity check on a spectral detector, which is the same
shape as the plausibility guard the boundary detector needed: a statistical
test needed a physical check, and a spectral test needs a spatial one.

WHY THIS SCRIPT EXISTS RATHER THAN JUST PICKING NUMBERS
-------------------------------------------------------
Four thresholds have been set in this project. The two taken from convention
-- 150 uV peak-to-peak, and MNE's muscle threshold of 0.5 -- were both wrong
(150 uV rejected 100% of epochs; 0.5 removed 14 of 19 components on one
subject). The two set from measured distributions were both right.

So: measure first. docs/ica_components_audit.csv already holds occipital and
frontal weights for all 313 components that were removed. This reads them,
shows what any candidate threshold would keep and reject, and estimates the
alpha that would be recovered -- before a line of pipeline code changes.

WHAT IT DOES NOT DO
-------------------
Changes no pipeline code. It proposes thresholds and shows their consequences;
applying them is a separate, deliberate step.

    py -m training.set_topography_veto
    py -m training.set_topography_veto --components docs/ica_components_audit.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# What a real artifact looks like, from synthetic topographies with a known
# answer (see audit_ica_alpha.py's docstring). These are reference points for
# reading the measured distribution, not thresholds in themselves.
REFERENCE = {
    "genuine ocular":   {"occipital": 0.09, "frontal": 4.05},
    "temporal muscle":  {"occipital": 0.10, "frontal": 0.22},
    "uniform (no focus)": {"occipital": 1.00, "frontal": 1.00},
}

# Candidate rules to evaluate. Each is (name, predicate) where the predicate
# says "this component may be excluded".
def build_rules():
    return {
        "current (no veto)":
            lambda d: pd.Series(True, index=d.index),
        "veto occipital > 1.5":
            lambda d: d.occipital_weight <= 1.5,
        "eog must be frontal > 2.0":
            lambda d: ~d.is_eog | (d.frontal_weight > 2.0),
        "eog frontal > 2.0 AND veto occipital > 1.5":
            lambda d: (~d.is_eog | (d.frontal_weight > 2.0)) & (d.occipital_weight <= 1.5),
        "require focality (max region weight > 1.5)":
            lambda d: d.max_region_weight > 1.5,
        "focality > 1.5 AND veto occipital > 1.5":
            lambda d: (d.max_region_weight > 1.5) & (d.occipital_weight <= 1.5),
        "focality > 2.0 AND veto occipital > 1.5":
            lambda d: (d.max_region_weight > 2.0) & (d.occipital_weight <= 1.5),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--components", default="docs/ica_components_audit.csv")
    ap.add_argument("--subjects", default="docs/ica_alpha_audit.csv")
    ap.add_argument("--csv", default="docs/topography_veto_options.csv")
    args = ap.parse_args()

    d = pd.read_csv(args.components)
    print(f"{len(d)} removed components across {d.subject_id.nunique()} subjects\n")

    d["is_eog"] = d.reason.astype(str).str.contains("eog")
    d["is_muscle"] = d.reason.astype(str).str.contains("muscle")
    # "Focality" = how much the strongest region stands out. A genuine artifact
    # dominates SOME region; a diffuse component dominates none. Only the two
    # regions measured in the audit are available here, so this is a lower
    # bound on true focality -- a component focal at, say, Cz would look
    # diffuse by this measure. Stated so the number is not over-read.
    d["max_region_weight"] = d[["occipital_weight", "frontal_weight"]].max(axis=1)

    print("=" * 76)
    print("WHAT IS ACTUALLY BEING REMOVED")
    print("=" * 76)
    print(f"  {'':<26} {'occipital':>10} {'frontal':>9} {'focality':>10}")
    for name, r in REFERENCE.items():
        print(f"  {name:<26} {r['occipital']:>10.2f} {r['frontal']:>9.2f} "
              f"{max(r['occipital'], r['frontal']):>10.2f}")
    print("  " + "-" * 58)
    for lbl, sub in [("ALL removed", d),
                     ("removed as eog", d[d.is_eog]),
                     ("removed as muscle", d[d.is_muscle])]:
        if sub.empty:
            continue
        print(f"  {lbl + ' (median)':<26} {sub.occipital_weight.median():>10.3f} "
              f"{sub.frontal_weight.median():>9.3f} {sub.max_region_weight.median():>10.3f}")

    print(f"\n  Focality below ~1.5 means the component is not clearly dominant in")
    print(f"  either measured region. Fraction of removed components below 1.5: "
          f"{(d.max_region_weight < 1.5).mean():.1%}")

    print("\n" + "=" * 76)
    print("EOG COMPONENTS -- are they frontal at all?")
    print("=" * 76)
    eog = d[d.is_eog]
    if not eog.empty:
        print(f"  n = {len(eog)}   median frontal weight {eog.frontal_weight.median():.3f}")
        print(f"  a genuine ocular component scores ~4.05")
        for t in (1.5, 2.0, 2.5, 3.0):
            print(f"    frontal weight > {t}: {(eog.frontal_weight > t).sum():>3}/{len(eog)}"
                  f"  ({(eog.frontal_weight > t).mean():.1%})")
        print("\n  If most 'eog' components are not frontally dominant, the EOG detector")
        print("  is not finding eye movement. Fp1/Fp2 are real EEG channels, so")
        print("  correlating against them can flag anything that happens to covary.")

    print("\n" + "=" * 76)
    print("CANDIDATE RULES -- what each would keep and reject")
    print("=" * 76)
    print(f"  {'rule':<44} {'kept':>7} {'vetoed':>8} {'%veto':>7}")
    print("  " + "-" * 68)
    rows = []
    for name, rule in build_rules().items():
        keep = rule(d)
        rows.append({"rule": name, "kept": int(keep.sum()),
                     "vetoed": int((~keep).sum()), "pct_vetoed": float((~keep).mean())})
        print(f"  {name:<44} {keep.sum():>7} {(~keep).sum():>8} {(~keep).mean():>6.1%}")

    subj_path = Path(args.subjects)
    if subj_path.exists():
        subj = pd.read_csv(subj_path)
        print("\n" + "=" * 76)
        print("ESTIMATED EFFECT ON ALPHA RETENTION")
        print("=" * 76)
        print("  Rough, and deliberately labelled so: it assumes each removed component")
        print("  costs roughly its share of the observed loss, which ignores that")
        print("  components differ in how much alpha they carry. The real number comes")
        print("  from re-running audit_ica_alpha.py after the veto is in. Use this to")
        print("  RANK the rules, not to predict the outcome.\n")

        merged = d.merge(subj[["subject_id", "alpha_retained", "n_excluded"]],
                         on="subject_id", how="left")
        print(f"  {'rule':<44} {'est. median retained':>21}")
        print("  " + "-" * 68)
        base = subj.alpha_retained.median()
        print(f"  {'current (measured, not estimated)':<44} {base:>21.3f}")
        for name, rule in build_rules().items():
            if name.startswith("current"):
                continue
            keep = rule(merged)
            per_subj = []
            for sid, g in merged.assign(keep=keep).groupby("subject_id"):
                r = g.alpha_retained.iloc[0]
                n = g.n_excluded.iloc[0]
                if not np.isfinite(r) or n <= 0:
                    continue
                frac_kept = g.keep.mean()
                # loss scales with the fraction of components still removed
                per_subj.append(1.0 - (1.0 - r) * frac_kept)
            if per_subj:
                print(f"  {name:<44} {np.median(per_subj):>21.3f}")

    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.csv, index=False)

    print("\n" + "=" * 76)
    print("HOW TO CHOOSE")
    print("=" * 76)
    print("  Two failure modes, and they pull in opposite directions:")
    print("    veto too little -> alpha keeps being removed, the original problem")
    print("    veto too much   -> real blinks stay in, inflating delta/theta and")
    print("                       therefore TBR's numerator")
    print("\n  Prefer a rule that vetoes the DIFFUSE components (the mechanism found)")
    print("  while keeping the frontally-focal ones (real blinks). If a rule vetoes")
    print("  nearly everything, the detectors are not finding artifact at all and the")
    print("  answer is to replace them -- mne-icalabel is the obvious candidate --")
    print("  rather than to tune a veto around them.")
    print(f"\n  Options written to {args.csv}")


if __name__ == "__main__":
    main()