"""
Choose the ICA topography veto rule from the full-region measurements.

An earlier version of this analysis measured only occipital and frontal
weights. That was a blind spot: muscle artifact is TEMPORAL, so every genuine
muscle component looked diffuse by construction, and a focality veto built on
those two regions would have rejected the muscle detector's entire output on a
measurement artifact rather than on evidence.

With all six regions measured, the picture is:

    peak_region   central  frontal  frontopolar  occipital  parietal  temporal
    eog                17       13           77         20        17        12
    muscle             11       30           30         15        11        57

An `eog` component should peak frontopolar; 49% do. A `muscle` component
should peak temporal; 37% do. Both detectors are wrong more often than right,
because both judge on SPECTRAL criteria with no spatial check.

Focality (how far the strongest region stands above the mean) is bimodal for
eog -- 25th percentile 1.01, i.e. perfectly uniform, but 75th percentile 3.31.
Real blinks and diffuse junk, mixed together. That is what makes a veto
viable rather than a blunt instrument.

    py -m training.choose_ica_veto
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Where each detector's artifact actually lives on the scalp. Blinks and
# saccades are frontopolar and lateral-frontal; EMG is temporal. A component
# flagged for one reason but peaking elsewhere is not that artifact.
EXPECTED_REGIONS = {
    "eog": {"frontopolar", "frontal"},
    "muscle": {"temporal"},
    "eog+muscle": {"frontopolar", "frontal", "temporal"},
}


def region_matches(row) -> bool:
    exp = EXPECTED_REGIONS.get(str(row.reason))
    return True if exp is None else row.peak_region in exp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--components", default="docs/ica_components_audit_v2.csv")
    ap.add_argument("--subjects", default="docs/ica_alpha_audit_v2.csv")
    ap.add_argument("--csv", default="docs/ica_veto_rules.csv")
    args = ap.parse_args()

    d = pd.read_csv(args.components)
    d["region_ok"] = d.apply(region_matches, axis=1)
    print(f"{len(d)} removed components across {d.subject_id.nunique()} subjects\n")

    print("=" * 74)
    print("DOES THE PEAK REGION MATCH THE DETECTION REASON?")
    print("=" * 74)
    for reason, g in d.groupby("reason"):
        exp = EXPECTED_REGIONS.get(reason, set())
        print(f"  {reason:<12} n={len(g):<4} peaks in {sorted(exp)}: "
              f"{g.region_ok.sum():>3}/{len(g)}  ({g.region_ok.mean():.0%})")
    print(f"\n  overall: {d.region_ok.sum()}/{len(d)}  ({d.region_ok.mean():.0%})")

    rules = {
        "current (no veto)": lambda x: pd.Series(True, index=x.index),
        "region must match reason": lambda x: x.region_ok,
        "focality > 1.5": lambda x: x.focality > 1.5,
        "focality > 2.0": lambda x: x.focality > 2.0,
        "region match AND focality > 1.5": lambda x: x.region_ok & (x.focality > 1.5),
        "region match AND focality > 2.0": lambda x: x.region_ok & (x.focality > 2.0),
        "region match AND focality > 1.5 AND occipital <= 1.5":
            lambda x: x.region_ok & (x.focality > 1.5) & (x.w_occipital <= 1.5),
    }

    print("\n" + "=" * 74)
    print("CANDIDATE RULES")
    print("=" * 74)
    print(f"  {'rule':<52} {'kept':>6} {'/subj':>7} {'%veto':>7}")
    print("  " + "-" * 72)
    n_subj = d.subject_id.nunique()
    rows = []
    for name, fn in rules.items():
        k = fn(d)
        rows.append({"rule": name, "kept": int(k.sum()),
                     "per_subject": k.sum() / n_subj, "pct_vetoed": float((~k).mean())})
        print(f"  {name:<52} {k.sum():>6} {k.sum()/n_subj:>7.2f} {(~k).mean():>6.0%}")

    print("\n  Typical clean EEG has 1-3 genuine ocular components per subject and")
    print("  often 1-2 muscle. A rule keeping far fewer is removing too little;")
    print("  far more means it is not filtering.")

    sp = Path(args.subjects)
    if sp.exists():
        subj = pd.read_csv(sp)
        m = d.merge(subj[["subject_id", "alpha_retained", "n_excluded"]],
                    on="subject_id", how="left")
        print("\n" + "=" * 74)
        print("ESTIMATED ALPHA RETENTION")
        print("=" * 74)
        print("  Rough by construction: it assumes each removed component costs its")
        print("  share of the observed loss, which is false in detail -- components")
        print("  differ in how much alpha they carry. Use it to RANK, not predict.")
        print("  The real number comes from re-running audit_ica_alpha.py.\n")
        print(f"  {'rule':<52} {'est. median':>12}")
        print("  " + "-" * 68)
        print(f"  {'current (MEASURED, not estimated)':<52} "
              f"{subj.alpha_retained.median():>12.3f}")
        for name, fn in rules.items():
            if name.startswith("current"):
                continue
            keep = fn(m)
            est = []
            for _, g in m.assign(keep=keep).groupby("subject_id"):
                r, n = g.alpha_retained.iloc[0], g.n_excluded.iloc[0]
                if np.isfinite(r) and n > 0:
                    est.append(1.0 - (1.0 - r) * g.keep.mean())
            if est:
                print(f"  {name:<52} {np.median(est):>12.3f}")

    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.csv, index=False)

    print("\n" + "=" * 74)
    print("HOW TO READ THIS")
    print("=" * 74)
    print("  The two failure modes pull opposite ways:")
    print("    veto too little -> alpha keeps being removed")
    print("    veto too much   -> real blinks survive, inflating delta/theta and")
    print("                       therefore TBR's NUMERATOR")
    print("\n  A rule keeping ~1-2 components per subject, with estimated retention")
    print("  near 1.0, is the target. If every rule keeps under ~0.5 per subject,")
    print("  the detectors are not finding artifact at all and the answer is to")
    print("  REPLACE them -- mne-icalabel classifies components from topography and")
    print("  spectrum together -- rather than to keep tuning a guard around them.")
    print(f"\n  Written to {args.csv}")


if __name__ == "__main__":
    main()