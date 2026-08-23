"""
Diagnostic — how sensitive is TBR to the ICA muscle-detection threshold?

WHY THIS MATTERS
----------------
find_bads_muscle targets components with power concentrated above ~20 Hz.
That IS the beta band, which is the DENOMINATOR of TBR. So the threshold
choice directly moves the primary biomarker:

    over-reject muscle  ->  beta falls  ->  TBR rises
    under-reject muscle ->  EMG inflates beta  ->  TBR falls

The literature reports that children with ADHD move more than controls. If
that means they also have more EMG, then an arbitrary threshold choice could
manufacture or erase a group difference. Before picking a number, measure how
much it actually moves.

The default (0.5) flagged 5 of 19 components on C09090107 -- roughly 37% of
the decomposition, which looked over-aggressive.

WHAT THIS DOES NOT DO
---------------------
Changes no pipeline code. It monkey-patches find_bads_muscle's default inside
this process only, runs the real remove_artifacts_ica() for each threshold,
and reports components excluded plus the resulting TBR.

USAGE
-----
    py -m training.sweep_muscle_threshold --file "D:\\path\\to\\X-EOEC.edf"
    py -m training.sweep_muscle_threshold --data-dir "D:\\..." --limit 4

Run from the repo root as a module.
"""

import argparse
import warnings
from pathlib import Path

import mne
import numpy as np

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

import data_pipeline.preprocessing as P
from training.classical_features import compute_tbr

THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

_ORIGINAL_FIND_BADS_MUSCLE = mne.preprocessing.ICA.find_bads_muscle


def _patch_muscle_threshold(value: float) -> None:
    """Force find_bads_muscle to use `value`, whatever the caller passes."""
    def patched(self, inst, **kwargs):
        kwargs["threshold"] = value
        kwargs.setdefault("verbose", False)
        return _ORIGINAL_FIND_BADS_MUSCLE(self, inst, **kwargs)
    mne.preprocessing.ICA.find_bads_muscle = patched


def _restore() -> None:
    mne.preprocessing.ICA.find_bads_muscle = _ORIGINAL_FIND_BADS_MUSCLE


def sweep_one(eoec_path: str) -> list:
    sf = P.parse_filename(eoec_path)
    print(f"\n{sf.subject_id} ({sf.group})")
    print(f"  {'thresh':>7} {'excl':>5} {'eog':>4} {'musc':>5} "
          f"{'TBR_EC':>8} {'TBR_EO':>8}  components")
    print("  " + "-" * 68)

    raw = P.filter_raw(P.load_raw(eoec_path))
    rows = []
    for t in THRESHOLDS:
        _patch_muscle_threshold(t)
        try:
            clean, diag = P.remove_artifacts_ica(raw)
            split = P.split_eoec_by_alpha(clean)
            # reject_uv=None: isolate the ICA threshold's effect. Epoch
            # rejection is a separate knob and would confound this.
            tbr_ec = compute_tbr(P.epoch_signal(split["ec"], reject_uv=None))
            tbr_eo = compute_tbr(P.epoch_signal(split["eo"], reject_uv=None))
        except Exception as e:  # noqa: BLE001
            print(f"  {t:>7.1f}  FAILED: {e}")
            continue
        finally:
            _restore()

        n_eog = sum(1 for r in diag["reasons"].values() if "eog" in r)
        n_mus = sum(1 for r in diag["reasons"].values() if "muscle" in r)
        print(f"  {t:>7.1f} {diag['n_excluded']:>5} {n_eog:>4} {n_mus:>5} "
              f"{tbr_ec:>8.4f} {tbr_eo:>8.4f}  {diag['excluded']}")
        rows.append({"subject_id": sf.subject_id, "group": sf.group,
                     "threshold": t, "n_excluded": diag["n_excluded"],
                     "n_eog": n_eog, "n_muscle": n_mus,
                     "tbr_ec": tbr_ec, "tbr_eo": tbr_eo})
    return rows


def report(rows: list) -> None:
    if not rows:
        return
    print("\n" + "=" * 72)
    print("HOW MUCH DOES THE THRESHOLD MOVE TBR?")
    print("=" * 72)
    for sid in dict.fromkeys(r["subject_id"] for r in rows):
        vals = [r["tbr_ec"] for r in rows if r["subject_id"] == sid
                and np.isfinite(r["tbr_ec"])]
        if len(vals) < 2:
            continue
        lo, hi = min(vals), max(vals)
        print(f"  {sid:<12} TBR_EC ranges {lo:.3f} - {hi:.3f}  "
              f"({(hi - lo) / lo * 100:5.1f}% swing across thresholds)")

    print("\n  Reading this:")
    print("   <5% swing   -> threshold barely matters. Pick 0.8, move on.")
    print("   5-20% swing -> matters. Pick a value, and report the sensitivity")
    print("                  in your methods rather than presenting one number.")
    print("   >20% swing  -> muscle rejection is eating real beta. Use a")
    print("                  stricter threshold and treat TBR with suspicion.")
    print("\n  Target for n_excluded: 1-4 of 19 components.")
    print("  If TBR rises steadily as the threshold LOOSENS (lower value, more")
    print("  components removed), that is beta being stripped out, not artifact.")


def main() -> None:
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", type=str, help="one EOEC .edf")
    src.add_argument("--data-dir", type=str, help="directory to scan")
    p.add_argument("--limit", type=int, default=2,
                   help="subjects to test when using --data-dir (default 2)")
    p.add_argument("--csv", type=str, default=None)
    args = p.parse_args()

    if args.file:
        paths = [args.file]
    else:
        found = sorted(Path(args.data_dir).glob("**/*-EOEC.edf"))
        by_group = {"ADHD": [], "Control": []}
        for path in found:
            try:
                by_group[P.parse_filename(str(path)).group].append(str(path))
            except ValueError:
                continue
        # Take from both groups: IDs sort C-before-F and that first letter is
        # the group label, so a plain slice returns only Controls.
        per = max(1, args.limit // 2)
        paths = by_group["ADHD"][:per] + by_group["Control"][:per]

    rows = []
    for path in paths:
        try:
            rows.extend(sweep_one(path))
        except Exception as e:  # noqa: BLE001
            print(f"\nFAILED on {Path(path).name}: {e}")

    report(rows)

    if args.csv and rows:
        import pandas as pd
        pd.DataFrame(rows).to_csv(args.csv, index=False)
        print(f"\nSaved -> {args.csv}")


if __name__ == "__main__":
    main()