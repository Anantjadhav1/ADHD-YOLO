"""
Is ICA removing genuine occipital alpha, or only artifact?

WHY THIS MATTERS ENOUGH TO GATE TRAINING
----------------------------------------
Alpha blocking -- posterior 8-12 Hz power collapsing when the eyes open --
is among the most robust phenomena in electrophysiology. Berger described it
in 1929 and it has replicated ever since. If it disappears from a recording
after ICA, ICA removed it.

Measured while validating the EC/EO boundary detector, comparing alpha
contrast (|log(alpha_first_half / alpha_second_half)|) before and after ICA:

        C10050113   3.60 -> 0.66     0.18x
        F09080101   1.44 -> 0.15     0.10x
        C12091154   3.69 -> 1.25     0.34x
        C12021125   6.22 -> 170.74    27x

F09080101 loses 90% of its alpha blocking. C12021125 gains 27-fold, which is
its own kind of wrong. Whatever is happening is not subtle.

If ICA is stripping occipital alpha, the damage is not confined to the EC/EO
split. It reaches every scalogram, every topomap, the coherence maps, and TBR
-- because all of them are computed on the ICA-cleaned signal. That is the
whole dataset, which is why this runs before training rather than after.

WHAT THIS MEASURES
------------------
1. ALPHA RETENTION. Occipital (O1/O2) alpha power before and after ICA, per
   half of the recording, and the ratio. Below 0.5 means over half the alpha
   is gone.

2. WHERE THE REMOVED COMPONENTS SIT. For each excluded component, the mean
   |topography weight| at O1/O2 relative to the mean across all 19 channels.
   Validated on synthetic topographies:

        genuine ocular component     occipital weight 0.09
        alpha-carrying component     occipital weight 6.16
        temporal muscle component    occipital weight 0.10

   Two orders of magnitude apart, so the 0.5 / 1.5 thresholds are not finely
   balanced -- a value near 1.5 is a real warning, not rounding noise.

3. WHETHER IT IS DIFFERENTIAL BETWEEN GROUPS. If ADHD subjects lose more
   alpha than Controls, that is a confound rather than noise: it would create
   a systematic between-group difference manufactured by preprocessing. The
   literature already reports that children with ADHD move more, so
   artifact-driven differences between these groups are a live risk.

Read-only. Changes no pipeline code, regenerates no images.

    py -m training.audit_ica_alpha
    py -m training.audit_ica_alpha --limit 20 --csv docs/ica_alpha_audit.csv
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
from data_pipeline.preprocessing import (ALPHA_BAND_HZ, CHANNELS_19,
                                         EOG_PROXY_CHANNELS, filter_raw,
                                         load_raw, remove_artifacts_ica)

OCCIPITAL = ["O1", "O2"]

# Focality has to be judged over regions where artifact ACTUALLY LIVES, not
# just the two the alpha question needed. Measuring only occipital and frontal
# made every temporal muscle component look diffuse -- the synthetic reference
# scores genuine temporal muscle at 0.22 on that pair -- so a focality veto
# built on it would have rejected the muscle detector's entire output on a
# measurement artifact rather than on evidence.
REGIONS = {
    "frontopolar": ["Fp1", "Fp2"],          # blinks, vertical eye movement
    "frontal":     ["F7", "F3", "Fz", "F4", "F8"],
    "temporal":    ["T3", "T4", "T5", "T6"],  # muscle (EMG) lives here
    "central":     ["C3", "Cz", "C4"],
    "parietal":    ["P3", "Pz", "P4"],
    "occipital":   ["O1", "O2"],            # alpha
}
TRIM_SEC = 15.0
# Below this fraction of alpha retained, ICA has taken more than it should.
ALPHA_LOSS_FLAG = 0.5
# Occipital topography weight above which an excluded component is carrying
# posterior brain signal rather than artifact.
OCCIPITAL_WEIGHT_FLAG = 1.5


def alpha_power(raw: mne.io.Raw, picks: list, lo_frac: float, hi_frac: float) -> float:
    """Mean 8-12 Hz power over a fractional span of the recording."""
    sfreq = float(raw.info["sfreq"])
    sig = raw.get_data(picks=picks).mean(axis=0)
    trim = int(TRIM_SEC * sfreq)
    sig = sig[trim:len(sig) - trim]
    n = len(sig)
    seg = sig[int(lo_frac * n):int(hi_frac * n)]
    if len(seg) < int(4 * sfreq):
        return float("nan")
    freqs, psd = welch(seg, fs=sfreq, nperseg=int(4 * sfreq))
    band = (freqs >= ALPHA_BAND_HZ[0]) & (freqs <= ALPHA_BAND_HZ[1])
    return float(psd[band].mean())


def region_weight(topo: np.ndarray, ch_names: list, region: list) -> float:
    """
    Mean |topography weight| over `region`, divided by the mean over all
    channels. Scale-free, because ICA components have arbitrary sign and
    magnitude -- only the SHAPE of a topography is meaningful, so an absolute
    weight would compare nothing.

      ~1.0  uniform across the scalp
      <0.5  the region is not involved
      >1.5  the region dominates
    """
    a = np.abs(topo)
    m = a.mean()
    if m <= 0:
        return float("nan")
    idx = [ch_names.index(c) for c in region if c in ch_names]
    return float(a[idx].mean() / m) if idx else float("nan")


def analyse(eoec_path: str) -> dict:
    raw = filter_raw(load_raw(eoec_path))
    occ = [c for c in OCCIPITAL if c in raw.ch_names]
    if not occ:
        raise ValueError("No occipital channels (O1/O2).")

    pre_1 = alpha_power(raw, occ, 0.0, 0.5)
    pre_2 = alpha_power(raw, occ, 0.5, 1.0)
    pre_all = alpha_power(raw, occ, 0.0, 1.0)

    clean, diag = remove_artifacts_ica(raw, return_ica=True)
    ica = diag.pop("ica")

    post_1 = alpha_power(clean, occ, 0.0, 0.5)
    post_2 = alpha_power(clean, occ, 0.5, 1.0)
    post_all = alpha_power(clean, occ, 0.0, 1.0)

    # Component topographies: (n_channels, n_components), rows in the order
    # ICA was fitted on -- the 19 EEG channels.
    mixing = ica.get_components()
    fitted_ch = [c for c in CHANNELS_19 if c in raw.ch_names]

    comps = []
    for i in diag["excluded"]:
        topo = mixing[:, i]
        rec = {
            "component": int(i),
            "reason": diag["reasons"].get(int(i), "?"),
            "occipital_weight": region_weight(topo, fitted_ch, OCCIPITAL),
            "frontal_weight": region_weight(topo, fitted_ch, EOG_PROXY_CHANNELS),
        }
        for name, chans in REGIONS.items():
            rec[f"w_{name}"] = region_weight(topo, fitted_ch, chans)
        # Focality over ALL regions: how much the strongest region stands out.
        # A genuine artifact dominates SOME region; a diffuse component
        # dominates none. This is the number the veto should use.
        rvals = [rec[f"w_{n}"] for n in REGIONS if np.isfinite(rec[f"w_{n}"])]
        rec["focality"] = max(rvals) if rvals else float("nan")
        rec["peak_region"] = max(REGIONS, key=lambda n: rec[f"w_{n}"]) if rvals else "?"
        comps.append(rec)

    worst_occ = max((c["occipital_weight"] for c in comps
                     if np.isfinite(c["occipital_weight"])), default=float("nan"))
    n_occ_dominant = sum(1 for c in comps
                         if np.isfinite(c["occipital_weight"])
                         and c["occipital_weight"] > OCCIPITAL_WEIGHT_FLAG)

    def contrast(a, b):
        return abs(np.log(a / b)) if (a > 0 and b > 0) else float("nan")

    return {
        "alpha_pre": pre_all, "alpha_post": post_all,
        "alpha_retained": post_all / pre_all if pre_all > 0 else float("nan"),
        "contrast_pre": contrast(pre_1, pre_2),
        "contrast_post": contrast(post_1, post_2),
        "n_excluded": diag["n_excluded"],
        "excluded": ",".join(str(i) for i in diag["excluded"]),
        "reasons": ";".join(f"{k}:{v}" for k, v in sorted(diag["reasons"].items())),
        "worst_occipital_weight": worst_occ,
        "n_occipital_dominant": n_occ_dominant,
        "capped": diag.get("capped", False),
        "_components": comps,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data_pipeline/splits/subject_splits.csv")
    ap.add_argument("--csv", default="docs/ica_alpha_audit.csv")
    ap.add_argument("--components-csv", default="docs/ica_components_audit.csv")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    man = subject_split.load_manifest(args.manifest)
    if args.limit:
        # Stratify: a plain head() returns only Controls, since IDs sort
        # C-before-F and that first letter IS the group label.
        per = max(1, args.limit // man["group"].nunique())
        man = pd.concat([g.head(per) for _, g in man.groupby("group")]).reset_index(drop=True)

    rows, comp_rows = [], []
    for i, r in man.reset_index(drop=True).iterrows():
        print(f"[{i+1}/{len(man)}] {r.subject_id} ... ", end="", flush=True)
        try:
            res = analyse(r.eoec_path)
        except Exception as e:                          # noqa: BLE001
            print(f"FAILED: {e}")
            continue
        for c in res.pop("_components"):
            c.update(subject_id=r.subject_id, group=r.group)
            comp_rows.append(c)
        res.update(subject_id=r.subject_id, group=r.group, split=r.split)
        rows.append(res)
        flag = ""
        if res["alpha_retained"] < ALPHA_LOSS_FLAG:
            flag += "  ALPHA LOSS"
        if res["n_occipital_dominant"]:
            flag += f"  {res['n_occipital_dominant']} OCCIPITAL COMPONENT(S) REMOVED"
        print(f"alpha {res['alpha_retained']:.2f}x  "
              f"contrast {res['contrast_pre']:.2f}->{res['contrast_post']:.2f}  "
              f"excl={res['n_excluded']} worst_occ={res['worst_occipital_weight']:.2f}{flag}")

    df, cdf = pd.DataFrame(rows), pd.DataFrame(comp_rows)
    if df.empty:
        print("\nNothing analysed.")
        return

    print("\n" + "=" * 78)
    print("ALPHA RETENTION ACROSS ICA")
    print("=" * 78)
    ret = df.alpha_retained
    print(f"  median {ret.median():.3f}   mean {ret.mean():.3f}   "
          f"range {ret.min():.3f} - {ret.max():.3f}")
    for thr, label in [(0.5, "lost >50%"), (0.75, "lost >25%"), (1.5, "GAINED >50%")]:
        n = (ret < thr).sum() if thr <= 1 else (ret > thr).sum()
        print(f"  {label:<14} {n:>3}/{len(df)}")
    print("\n  A ratio far from 1.0 in EITHER direction is a problem. Removing a")
    print("  component cannot ADD alpha; a large gain means the removed component")
    print("  was cancelling posterior signal, which is its own kind of wrong.")

    print("\n" + "=" * 78)
    print("WHERE THE REMOVED COMPONENTS SIT")
    print("=" * 78)
    if not cdf.empty:
        print(f"  {len(cdf)} excluded components across {df.shape[0]} subjects")
        print(f"  median occipital weight: {cdf.occipital_weight.median():.3f}")
        print(f"  median frontal weight:   {cdf.frontal_weight.median():.3f}")
        bad = cdf[cdf.occipital_weight > OCCIPITAL_WEIGHT_FLAG]
        print(f"\n  occipital-dominant (>{OCCIPITAL_WEIGHT_FLAG}): {len(bad)}/{len(cdf)}"
              f"  ({len(bad)/len(cdf):.1%})")
        print("  A genuine ocular component scores ~0.09 occipitally and ~4.0 frontally.")
        if not bad.empty:
            print("\n  Removed components loading on O1/O2 -- these carry brain signal:")
            print(bad.sort_values("occipital_weight", ascending=False)
                  .head(20)[["subject_id", "component", "reason",
                             "occipital_weight", "frontal_weight"]]
                  .to_string(index=False, float_format=lambda v: f"{v:.2f}"))
            print("\n  By detection reason:")
            print(bad.groupby("reason").size().to_string())

    print("\n" + "=" * 78)
    print("IS ALPHA LOSS DIFFERENTIAL BETWEEN GROUPS?")
    print("=" * 78)
    for g, sub in df.groupby("group"):
        print(f"  {g:<8} n={len(sub):<4} median retained {sub.alpha_retained.median():.3f}   "
              f"median excluded {sub.n_excluded.median():.1f}")
    groups = df.group.unique()
    if len(groups) == 2:
        try:
            from scipy.stats import mannwhitneyu
            a = df[df.group == groups[0]].alpha_retained.dropna()
            b = df[df.group == groups[1]].alpha_retained.dropna()
            pv = mannwhitneyu(a, b).pvalue
            print(f"\n  Mann-Whitney p = {pv:.4f}")
            if pv < 0.05:
                print("  *** DIFFERENTIAL LOSS -- this is a CONFOUND, not noise. ***")
                print("  Preprocessing would be manufacturing a between-group difference")
                print("  in exactly the band the biomarkers are computed from.")
            else:
                print("  No differential loss between groups.")
        except Exception:                               # noqa: BLE001
            pass

    fin = df.dropna(subset=["alpha_retained"])
    if len(fin) > 5:
        c = np.corrcoef(fin.n_excluded, fin.alpha_retained)[0, 1]
        print(f"\n  corr(components removed, alpha retained) = {c:+.3f}")
        print("  Strongly negative would mean removing more components costs more")
        print("  alpha -- i.e. the detectors are indiscriminate rather than targeted.")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    frac_lost = (ret < ALPHA_LOSS_FLAG).mean()
    frac_bad = (len(cdf[cdf.occipital_weight > OCCIPITAL_WEIGHT_FLAG]) / len(cdf)
                if not cdf.empty else 0.0)
    if frac_lost < 0.05 and frac_bad < 0.05:
        print("  Alpha is preserved and the removed components are not occipital.")
        print("  ICA is doing what it should. Close this question and proceed to")
        print("  training -- the boundary shifts came from something else.")
    elif frac_lost < 0.20:
        print(f"  Confined: {frac_lost:.0%} of subjects lose >50% of occipital alpha,")
        print(f"  and {frac_bad:.0%} of removed components are occipital-dominant.")
        print("  Add an occipital-weight veto to remove_artifacts_ica -- refuse to")
        print("  exclude a component whose occipital weight exceeds "
              f"{OCCIPITAL_WEIGHT_FLAG}, whatever the")
        print("  detector says -- then regenerate only the affected subjects.")
    else:
        print(f"  WIDESPREAD: {frac_lost:.0%} of subjects lose >50% of occipital alpha.")
        print("  The ICA step needs rethinking before ANY training. Every image in")
        print("  the dataset was produced from this signal. Do not train on it.")

    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=[c for c in df.columns if c.startswith("_")]).to_csv(args.csv, index=False)
    if not cdf.empty:
        cdf.to_csv(args.components_csv, index=False)
        print(f"\nSaved -> {args.csv}\n         {args.components_csv}")
    else:
        print(f"\nSaved -> {args.csv}")


if __name__ == "__main__":
    main()