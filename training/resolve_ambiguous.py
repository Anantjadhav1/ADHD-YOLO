"""
Resolve the EC/EO-ambiguous subjects with a SECOND, independent vote.

THE PROBLEM
-----------
split_eoec_by_alpha() decides which half of an EOEC recording is eyes-closed by
comparing occipital alpha power between halves. When the ratio lands inside
ALPHA_AMBIGUOUS_RATIO_RANGE = (0.7, 1.4) it refuses to guess and flags the
subject. On the 108-subject cohort run that fired 14 times -- including 4 of
the 17 test subjects, leaving 13.

n=13 gives a 95% CI on accuracy of roughly +/-27 points. An observed 80% would
be compatible with 53-100%, which cannot be distinguished from the 75.8%
baseline or from chance. That is the number that goes in the paper, so
recovering these subjects is not optional.

WHY NOT JUST WIDEN THE THRESHOLD
--------------------------------
8 of the 14 sit at alpha ratio 0.92-1.03. Alpha genuinely cannot decide there.
Widening would relabel a coin flip as a decision and hide the uncertainty
rather than resolve it.

THE SECOND VOTE
---------------
The rejection audit (2026-08-25) found EC rejects more epochs than EO in 86 of
108 subjects, and that 87% of rejections are driven by Fp1/Fp2/F7/F8 -- frontal
and frontopolar, i.e. OCULAR artifact, not occipital alpha. The likely
mechanism is drowsiness during several minutes of eyes-closed rest, which
produces slow roving eye movements and eyelid flutter.

So: the half with more frontal artifact is probably eyes-closed.

This is genuinely independent of vote 1 -- different physiological mechanism,
different channels, different frequency content. Two independent votes that
agree is a real argument; two correlated votes agreeing would not be.

DELIBERATELY SKIPS ICA
----------------------
remove_artifacts_ica() removes exactly the ocular components vote 2 measures.
Running it first would erase the signal. Both votes are therefore computed on
filtered-but-not-ICA-cleaned data, so the alpha ratios here may differ slightly
from the pipeline's (which splits post-ICA). Reported alongside so the
difference is visible rather than assumed away.

CALIBRATES ITSELF
-----------------
Also runs on the subjects alpha decided CLEARLY (ratio outside the ambiguous
band). If vote 2 agrees with vote 1 on the large majority of those, it is
trustworthy on the ambiguous ones. If it does not, this whole approach fails
and should be abandoned rather than patched -- the script says so explicitly
rather than reporting numbers that look usable.

PROPOSES, DOES NOT APPLY
------------------------
Prints the subjects that could be recovered. Recovering them is a separate,
manual step:
    py -m data_pipeline.build_dataset --output-dir dataset \
        --subjects <id> [<id> ...] --include-ambiguous

    py -m training.resolve_ambiguous
    py -m training.resolve_ambiguous --csv docs/ambiguity_resolution.csv
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
                                         ALPHA_BAND_HZ, filter_raw, load_raw)

# The four channels the rejection audit found responsible for 87% of rejections.
FRONTAL_ARTIFACT_CHANNELS = ["Fp1", "Fp2", "F7", "F8"]
OCCIPITAL_CHANNELS = ["O1", "O2"]
TRIM_SEC = 15.0          # matches split_eoec_by_alpha: clears the 6.6 s filter edge
ARTIFACT_WINDOW_SEC = 1.5  # matches the epoch length rejection operates on


def _alpha_power(seg: np.ndarray, sfreq: float) -> float:
    freqs, psd = welch(seg, fs=sfreq, nperseg=int(4 * sfreq))
    band = (freqs >= ALPHA_BAND_HZ[0]) & (freqs <= ALPHA_BAND_HZ[1])
    return float(psd[band].mean())


def _frontal_artifact(seg: np.ndarray, sfreq: float) -> float:
    """
    Median peak-to-peak amplitude (uV) across non-overlapping windows, taking
    the WORST frontal channel per window.

    Worst-channel, not mean, because that is how epoch rejection actually
    behaves: one channel over threshold rejects the whole epoch. Median rather
    than mean so a handful of huge movement transients cannot dominate -- we
    want the typical level of ocular activity, not the extremes.
    """
    win = int(ARTIFACT_WINDOW_SEC * sfreq)
    n_win = seg.shape[1] // win
    if n_win < 5:
        return float("nan")
    trimmed = seg[:, :n_win * win].reshape(seg.shape[0], n_win, win)
    p2p = (trimmed.max(axis=2) - trimmed.min(axis=2)) * 1e6   # volts -> uV
    return float(np.median(p2p.max(axis=0)))


def analyse(eoec_path: str) -> dict:
    raw = filter_raw(load_raw(eoec_path))   # NO ICA -- see module docstring
    sfreq = float(raw.info["sfreq"])

    occ = [c for c in OCCIPITAL_CHANNELS if c in raw.ch_names]
    fro = [c for c in FRONTAL_ARTIFACT_CHANNELS if c in raw.ch_names]
    if not occ or not fro:
        raise ValueError(f"Missing channels: occipital={occ}, frontal={fro}")

    trim = int(TRIM_SEC * sfreq)
    occ_data = raw.get_data(picks=occ).mean(axis=0)
    fro_data = raw.get_data(picks=fro)
    half = occ_data.shape[0] // 2

    a1 = _alpha_power(occ_data[trim:half], sfreq)
    a2 = _alpha_power(occ_data[half:-trim], sfreq)
    f1 = _frontal_artifact(fro_data[:, trim:half], sfreq)
    f2 = _frontal_artifact(fro_data[:, half:-trim], sfreq)

    alpha_ratio = a1 / a2 if a2 > 0 else float("nan")
    frontal_ratio = f1 / f2 if f2 > 0 else float("nan")

    return {
        "alpha_ratio": alpha_ratio,
        "frontal_ratio": frontal_ratio,
        "vote_alpha_ec_first": bool(alpha_ratio > 1.0),
        "vote_frontal_ec_first": bool(frontal_ratio > 1.0),
        "alpha_ambiguous": bool(ALPHA_AMBIGUOUS_RATIO_RANGE[0] < alpha_ratio
                                < ALPHA_AMBIGUOUS_RATIO_RANGE[1]),
        "frontal_uv_first": f1,
        "frontal_uv_second": f2,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data_pipeline/splits/subject_splits.csv")
    ap.add_argument("--csv", default="docs/ambiguity_resolution.csv")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    man = subject_split.load_manifest(args.manifest)
    if args.limit:
        man = man.head(args.limit)

    rows = []
    for i, r in man.reset_index(drop=True).iterrows():
        print(f"[{i+1}/{len(man)}] {r.subject_id} ... ", end="", flush=True)
        try:
            res = analyse(r.eoec_path)
        except Exception as e:                       # noqa: BLE001
            print(f"FAILED: {e}")
            continue
        res.update(subject_id=r.subject_id, group=r.group, split=r.split)
        rows.append(res)
        print(f"alpha {res['alpha_ratio']:.2f}  frontal {res['frontal_ratio']:.2f}")

    df = pd.DataFrame(rows)
    if df.empty:
        print("\nNothing analysed.")
        return
    df["votes_agree"] = df.vote_alpha_ec_first == df.vote_frontal_ec_first

    clear = df[~df.alpha_ambiguous]
    amb = df[df.alpha_ambiguous]

    print("\n" + "=" * 74)
    print("CALIBRATION -- does the frontal vote agree where alpha is CLEAR?")
    print("=" * 74)
    if len(clear) < 10:
        print(f"  Only {len(clear)} clearly-decided subjects; too few to calibrate.")
        agree_rate = float("nan")
    else:
        agree_rate = clear.votes_agree.mean()
        print(f"  {clear.votes_agree.sum()}/{len(clear)} agree  ({agree_rate:.1%})")
        print(f"  EC-first by alpha: {clear.vote_alpha_ec_first.sum()}/{len(clear)}")
        print("\n  Frontal artifact, EC half vs EO half (as alpha labelled them):")
        ec_first = clear[clear.vote_alpha_ec_first]
        eo_first = clear[~clear.vote_alpha_ec_first]
        ec_uv = pd.concat([ec_first.frontal_uv_first, eo_first.frontal_uv_second])
        eo_uv = pd.concat([ec_first.frontal_uv_second, eo_first.frontal_uv_first])
        print(f"    EC halves  median {ec_uv.median():6.1f} uV")
        print(f"    EO halves  median {eo_uv.median():6.1f} uV")
        try:
            from scipy.stats import wilcoxon
            print(f"    Wilcoxon p = {wilcoxon(ec_uv.values, eo_uv.values).pvalue:.4f}")
        except Exception:                            # noqa: BLE001
            pass

        if agree_rate < 0.70:
            print("\n  *** VOTE 2 IS NOT RELIABLE ON THIS COHORT. ***")
            print("  It disagrees with alpha too often where alpha is unambiguous, so")
            print("  its agreement on the ambiguous subjects would mean nothing. Do NOT")
            print("  use it to recover them. Abandon this approach rather than tuning it")
            print("  until it agrees -- that would be fitting the method to the answer.")
        elif agree_rate < 0.85:
            print("\n  Vote 2 is moderately reliable. Usable as corroboration, but the")
            print("  recovered subjects should be reported as recovered, and ideally")
            print("  checked with a leave-them-out sensitivity analysis.")
        else:
            print("\n  Vote 2 is reliable on this cohort. Agreement on an ambiguous")
            print("  subject is a real argument for its EC/EO ordering.")

    print("\n" + "=" * 74)
    print(f"AMBIGUOUS SUBJECTS ({len(amb)})")
    print("=" * 74)
    if not amb.empty:
        show = amb[["subject_id", "group", "split", "alpha_ratio", "frontal_ratio",
                    "vote_alpha_ec_first", "vote_frontal_ec_first", "votes_agree"]]
        print(show.sort_values("split").to_string(index=False,
                                                  float_format=lambda v: f"{v:.2f}"))

        rec = amb[amb.votes_agree]
        keep = amb[~amb.votes_agree]
        print(f"\n  Both votes agree -> recoverable: {len(rec)}")
        print(f"  Votes disagree    -> stay excluded: {len(keep)}")

        n_test_rec = (rec.split == "test").sum()
        n_test_out = (keep.split == "test").sum()
        print(f"\n  Test split: {n_test_rec} recoverable, {n_test_out} would remain excluded")
        print("  (13 test subjects currently; each one recovered narrows the CI)")

        if not rec.empty and (np.isnan(agree_rate) or agree_rate >= 0.70):
            ids = " ".join(rec.subject_id)
            print("\n  To recover them:")
            print(f"    py -m data_pipeline.build_dataset --output-dir dataset \\")
            print(f"        --subjects {ids} --include-ambiguous")
            print("\n  Then state in the methods that these subjects were EC/EO-ambiguous")
            print("  by alpha and were resolved using frontal ocular artifact as an")
            print("  independent second criterion. A leave-them-out sensitivity analysis")
            print("  is the honest way to show the result does not hinge on them.")
        if not keep.empty:
            print(f"\n  Remain excluded (votes disagree): {', '.join(keep.subject_id)}")
            print("  Report as a limitation -- excluded, not silently dropped.")

    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv, index=False)
    print(f"\nSaved -> {args.csv}")


if __name__ == "__main__":
    main()