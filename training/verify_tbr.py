"""
Step 1 verification — is the TBR magnitude anomaly (9-16 vs published 1.5-3.5)
a units bug rather than a data problem?

HYPOTHESIS UNDER TEST
---------------------
training/classical_features.py computes band power as the MEAN of the PSD
across the band's frequency bins:

    theta_power = mean_psd[(freqs >= 4) & (freqs <= 8)].mean()
    beta_power  = mean_psd[(freqs >= 12) & (freqs <= 30)].mean()

That is average spectral DENSITY, not band POWER. Published TBR uses the
integral over the band. Because theta spans 4 Hz and beta spans 18 Hz, the
ratio is inflated by roughly 18/4 = 4.5x. If that's the whole story,
9-16 / 4.5 = 2.0-3.6, which sits inside the published 1.5-3.5 range.

Secondary hypothesis: nperseg=min(256, ...) at 500 Hz gives df = 1.95 Hz, so
the theta band contains only TWO frequency bins. TBR is being estimated from
two numbers, and its value depends heavily on exactly where those bins land
relative to the 4 Hz and 8 Hz edges.

WHAT THIS SCRIPT DOES NOT DO
----------------------------
It does not change any pipeline code. It computes TBR four ways side by side
and reports them, so the decision to change classical_features.py is made from
real numbers on real subjects rather than from the arithmetic argument alone.

Consistent with this project's norms: it reuses preprocessing.py's real
functions rather than reimplementing loading/filtering, so the numbers are
directly comparable to what the pipeline produces.

USAGE
-----
    python3 -m training.verify_tbr --manifest data_pipeline/splits/subject_splits.csv
    python3 -m training.verify_tbr --data-dir /path/to/edf/files
    python3 -m training.verify_tbr --data-dir ./data --average-ref --csv out.csv

Run from the repo root as a module (`python3 -m ...`), not as a bare script --
preprocessing.py uses package-relative imports.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# np.trapz was REMOVED in NumPy 2.0 in favour of np.trapezoid. Shim so this
# works on both 1.x and 2.x -- worth carrying into classical_features.py too.
_trapz = getattr(np, "trapezoid", None) or np.trapz

import warnings

import mne
from scipy.signal import welch

# Diagnostic script only -- fine to mute here, do NOT copy into pipeline code.
mne.set_log_level("ERROR")
warnings.filterwarnings("ignore", category=RuntimeWarning)

from data_pipeline.preprocessing import (
    CHANNELS_19,
    epoch_signal,
    filter_raw,
    load_raw,
    parse_filename,
    split_eoec_by_alpha,
)

FRONTAL_CHANNELS = ["F3", "F4", "Fz"]
THETA_BAND_HZ = (4, 8)
BETA_BAND_HZ = (12, 30)

# The four variants under test: (band-power method, welch nperseg)
VARIANTS = [
    ("mean", 256),    # what classical_features.py does today
    ("mean", 1000),
    ("trapz", 256),
    ("trapz", 1000),  # the proposed fix
]
CURRENT = ("mean", 256)
PROPOSED = ("trapz", 1000)


def variant_label(method: str, nperseg: int) -> str:
    return f"{method}/{nperseg}"


def band_power(psd: np.ndarray, freqs: np.ndarray, lo: float, hi: float,
               method: str) -> float:
    """
    method="mean"  -> average PSD density across the band (current behaviour)
    method="trapz" -> integral of PSD over the band (physical band power)

    Returns NaN if the band contains fewer than 2 bins, since a trapezoidal
    integral over a single point is meaningless -- better to surface that than
    to silently return a number derived from one sample.
    """
    mask = (freqs >= lo) & (freqs <= hi)
    n_bins = int(mask.sum())
    if n_bins == 0:
        return float("nan")
    if method == "mean":
        return float(psd[mask].mean())
    if n_bins < 2:
        return float("nan")
    return float(_trapz(psd[mask], freqs[mask]))


def compute_tbr_variant(epochs, method: str, nperseg: int,
                        frontal_channels: list = FRONTAL_CHANNELS) -> dict:
    """
    One TBR value for one condition, using one (method, nperseg) combination.
    Mirrors classical_features.compute_tbr()'s structure exactly -- same
    channels, same averaging order (PSD averaged across epochs and frontal
    channels first, then the ratio taken) -- so the only thing that differs
    between variants is the thing under test.
    """
    available = [ch for ch in frontal_channels if ch in epochs.ch_names]
    if not available:
        return {"tbr": float("nan"), "n_theta_bins": 0, "n_beta_bins": 0}

    data = epochs.get_data(picks=available)          # (n_epochs, n_ch, n_samples)
    sfreq = float(epochs.info["sfreq"])
    flat = data.reshape(-1, data.shape[-1])

    nps = int(min(nperseg, flat.shape[-1]))
    freqs, psd = welch(flat, fs=sfreq, nperseg=nps, axis=-1)
    mean_psd = psd.mean(axis=0)

    theta = band_power(mean_psd, freqs, *THETA_BAND_HZ, method=method)
    beta = band_power(mean_psd, freqs, *BETA_BAND_HZ, method=method)

    tbr = float(theta / beta) if (beta and beta > 0 and np.isfinite(beta)) else float("nan")
    return {
        "tbr": tbr,
        "n_theta_bins": int(((freqs >= THETA_BAND_HZ[0]) & (freqs <= THETA_BAND_HZ[1])).sum()),
        "n_beta_bins": int(((freqs >= BETA_BAND_HZ[0]) & (freqs <= BETA_BAND_HZ[1])).sum()),
        "df_hz": float(freqs[1] - freqs[0]),
        "nperseg_used": nps,
    }


def prepare_subject(eoec_path: str, vcpt_path: str | None,
                    average_ref: bool = False) -> dict:
    """
    Load -> filter -> (optional average reference) -> EC/EO split -> epoch.

    DELIBERATELY SKIPS ICA. remove_artifacts_ica() currently fits an ICA and
    then calls ica.apply() with an empty exclude list, which reconstructs the
    signal bit-identically -- it is a no-op that costs a full ICA fit per
    recording. Skipping it changes nothing about the output and makes this
    check run in seconds instead of many minutes. If that changes (i.e. once
    components are actually excluded), revisit this.
    """
    out = {}
    raw = load_raw(eoec_path)
    raw = filter_raw(raw)
    if average_ref:
        # Not the current pipeline behaviour -- exposed here to test whether
        # referencing is a second contributor to the TBR magnitude.
        raw = raw.copy().set_eeg_reference("average", projection=False, verbose=False)

    split = split_eoec_by_alpha(raw)
    out["EC"] = epoch_signal(split["ec"])
    out["EO"] = epoch_signal(split["eo"])
    out["alpha_ratio"] = float(split["alpha_ratio"])
    out["ambiguous"] = bool(split["ambiguous"])

    if vcpt_path:
        raw_v = load_raw(vcpt_path)
        raw_v = filter_raw(raw_v)
        if average_ref:
            raw_v = raw_v.copy().set_eeg_reference("average", projection=False, verbose=False)
        out["VCPT"] = epoch_signal(raw_v)

    return out


def discover_from_dir(data_dir: str) -> pd.DataFrame:
    """Build a minimal manifest-shaped table straight from filenames, so this
    script works before subject_split.py has been run."""
    rows = {}
    root = Path(data_dir)
    for pattern in ("**/*-EOEC.edf", "**/*-EOEC.EDF"):
        for path in sorted(root.glob(pattern)):
            sf = parse_filename(str(path))
            if sf.subject_id in rows:
                continue
            vcpt = sorted(root.glob(f"**/{sf.subject_id}-*-VCPT.*"))
            rows[sf.subject_id] = {
                "subject_id": sf.subject_id,
                "group": sf.group,
                "eoec_path": str(path),
                "vcpt_path": str(vcpt[0]) if vcpt else None,
            }
    return pd.DataFrame(list(rows.values()))


def load_subjects(manifest_path: str | None, data_dir: str | None) -> pd.DataFrame:
    if manifest_path:
        from data_pipeline import subject_split
        df = subject_split.load_manifest(manifest_path)
    else:
        df = discover_from_dir(data_dir)
    if df.empty:
        raise SystemExit("No subjects found. Check --manifest / --data-dir.")
    return df


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Standardised mean difference. NaN if either group has <2 finite values."""
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled_var = ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)
    if pooled_var <= 0:
        return float("nan")
    return float((a.mean() - b.mean()) / np.sqrt(pooled_var))


def rank_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """
    Mann-Whitney AUC: P(a random ADHD subject has higher TBR than a random
    Control). Scale-free, so it is directly comparable ACROSS the four
    variants even though their absolute TBR values differ by ~4.5x. This is
    the number that actually matters -- matching published magnitudes is
    cosmetic, separating the two groups is not.
    """
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (len(pos) * len(neg)))


def run(subjects: pd.DataFrame, average_ref: bool) -> pd.DataFrame:
    records = []
    total = len(subjects)

    for i, row in subjects.reset_index(drop=True).iterrows():
        sid, group = row["subject_id"], row["group"]
        print(f"[{i+1}/{total}] {sid} ({group}) ... ", end="", flush=True)
        try:
            vcpt = row.get("vcpt_path")
            vcpt = vcpt if pd.notna(vcpt) else None      # NaN is truthy -- see build_classical_features.py
            prepared = prepare_subject(row["eoec_path"], vcpt, average_ref=average_ref)
        except Exception as e:
            print(f"FAILED: {e}")
            continue

        for cond in ("EC", "EO", "VCPT"):
            if cond not in prepared:
                continue
            rec = {
                "subject_id": sid,
                "group": group,
                "condition": cond,
                "n_epochs": len(prepared[cond]),
                "alpha_ratio": prepared["alpha_ratio"],
                "ambiguous": prepared["ambiguous"],
            }
            for method, nperseg in VARIANTS:
                res = compute_tbr_variant(prepared[cond], method, nperseg)
                rec[variant_label(method, nperseg)] = res["tbr"]
                rec[f"bins_{variant_label(method, nperseg)}"] = (
                    f"{res['n_theta_bins']}θ/{res['n_beta_bins']}β"
                )
            records.append(rec)
        print("ok")

    return pd.DataFrame(records)


def report(df: pd.DataFrame) -> None:
    if df.empty:
        print("\nNo results — every subject failed to process.")
        return

    cols = [variant_label(m, n) for m, n in VARIANTS]
    cur, prop = variant_label(*CURRENT), variant_label(*PROPOSED)

    print("\n" + "=" * 78)
    print("PER-SUBJECT TBR, ALL FOUR VARIANTS")
    print("=" * 78)
    show = df[["subject_id", "group", "condition", "n_epochs"] + cols].copy()
    print(show.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

    print("\n" + "=" * 78)
    print("FREQUENCY RESOLUTION PER VARIANT  (theta bins / beta bins)")
    print("=" * 78)
    for m, n in VARIANTS:
        lbl = variant_label(m, n)
        bins = df[f"bins_{lbl}"].dropna().unique()
        print(f"  {lbl:<14} {', '.join(map(str, bins))}")
    print("\n  Two theta bins is not enough to estimate a band ratio from.")

    print("\n" + "=" * 78)
    print("GROUP MEANS BY CONDITION")
    print("=" * 78)
    grouped = df.groupby(["condition", "group"])[cols].mean().round(3)
    print(grouped.to_string())

    print("\n" + "=" * 78)
    print("SEPARATION: does the variant still tell ADHD from Control?")
    print("=" * 78)
    print("  Cohen's d > 0 and AUC > 0.5 both mean ADHD > Control (expected direction).")
    print("  AUC is scale-free, so it is comparable across variants; TBR magnitude is not.\n")
    header = f"  {'condition':<10} {'variant':<14} {'ADHD':>9} {'Control':>9} {'d':>7} {'AUC':>6}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for cond in df["condition"].unique():
        sub = df[df["condition"] == cond]
        adhd = sub[sub["group"] == "ADHD"]
        ctrl = sub[sub["group"] == "Control"]
        for m, n in VARIANTS:
            lbl = variant_label(m, n)
            a, c = adhd[lbl].to_numpy(float), ctrl[lbl].to_numpy(float)
            print(f"  {cond:<10} {lbl:<14} {np.nanmean(a):>9.3f} {np.nanmean(c):>9.3f} "
                  f"{cohens_d(a, c):>7.2f} {rank_auc(a, c):>6.3f}")
        print()

    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    cur_vals = df[cur].to_numpy(float)
    prop_vals = df[prop].to_numpy(float)
    finite = np.isfinite(cur_vals) & np.isfinite(prop_vals) & (prop_vals != 0)
    if finite.any():
        factor = float(np.median(cur_vals[finite] / prop_vals[finite]))
        print(f"  Current  ({cur:<12}) range: {np.nanmin(cur_vals):.2f} - {np.nanmax(cur_vals):.2f}")
        print(f"  Proposed ({prop:<12}) range: {np.nanmin(prop_vals):.2f} - {np.nanmax(prop_vals):.2f}")
        print(f"  Median inflation factor: {factor:.2f}x   (arithmetic prediction: ~4.5x)")
                # Published TBR ranges describe GROUP MEANS, not the per-subject spread.
        # An earlier version of this check compared min/max of every individual
        # value against 1.0-5.0 and reported a false failure -- individual
        # subjects vary far more widely than any published mean does.
        print("\n  Group means under the proposed method (compare to ~1.5-3.5):")
        means = df.groupby(["condition", "group"])[prop].mean()
        ok = True
        for (cond, grp), val in means.items():
            flag = "" if 1.0 <= val <= 4.5 else "   <-- outside expected range"
            ok = ok and not flag
            print(f"    {cond:<6} {grp:<8} {val:6.2f}{flag}")
        if not ok:
            print("\n  -> Units are not the whole story. Next suspects, in order:")
            print("     artifact rejection (ICA is currently a no-op), then referencing")
            print("     (try --average-ref), then the EC/EO split boundary.")
    print("\n  Whatever the magnitude, check the AUC column above: if separation")
    print("  holds or improves under trapz/1000, adopt it. If separation COLLAPSES,")
    print("  that is the more important finding and this needs a closer look before")
    print("  changing classical_features.py.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", type=str, help="subject_splits.csv from subject_split.py")
    src.add_argument("--data-dir", type=str, help="directory of raw EDF files (no manifest needed)")
    p.add_argument("--average-ref", action="store_true",
                   help="apply average reference before computing TBR (not current pipeline behaviour)")
    p.add_argument("--csv", type=str, default=None, help="write the per-subject table here")
    p.add_argument("--limit", type=int, default=None, help="only process the first N subjects")
    args = p.parse_args()

    subjects = load_subjects(args.manifest, args.data_dir)
    if args.limit:
        # Sample evenly from BOTH groups. A plain .head() returns only Controls,
        # because subject IDs sort C-before-F and that first letter IS the group
        # label -- so the separation stats came back empty on the first run.
        per_group = max(1, args.limit // max(1, subjects["group"].nunique()))
        subjects = pd.concat(
            [g.head(per_group) for _, g in subjects.groupby("group")]
        ).reset_index(drop=True)

    print(f"Found {len(subjects)} subject(s). "
          f"Average reference: {'ON' if args.average_ref else 'off (pipeline default)'}")
    print("ICA is skipped — it is currently a no-op (empty exclude list).\n")

    df = run(subjects, average_ref=args.average_ref)
    report(df)

    if args.csv and not df.empty:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.csv, index=False)
        print(f"\nSaved -> {args.csv}")


if __name__ == "__main__":
    main()