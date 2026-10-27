"""
Loader and preparation stages for the Nasrabadi et al. dataset (IEEE DataPort),
for the pre-registered analyses H1a and H1b (docs/PREREGISTRATION.md, D1 and D2).

Run the stages in this order, from the repo root:

    py -m data_pipeline.nasrabadi clean       # filter + ICA every recording; label-free
    py -m data_pipeline.nasrabadi threshold   # the D2 rejection threshold; label-free
    py -m data_pipeline.nasrabadi features    # epochs + the 100 task-recording features

`clean` and `threshold` never use diagnosis. `threshold` does not even open the
dataset folders -- it lists the cleaned files only -- so the threshold cannot
depend on which group a recording came from. Its value must be appended to D2
before `features` is run and before any model sees the result.

Every rule here comes from the registration:
  * 128 Hz, from the dataset's public page (D2.1);
  * column order from Channel_Labels.docx, with the 10-10 names T7/T8/P7/P8
    renamed to this pipeline's T3/T4/T5/T6 -- the same electrodes (D1.1);
  * values treated as nominal microvolts for the conversion to volts (D1.3, D2.4);
  * the Rohani pipeline unchanged: 0.5-50 Hz band-pass, 50 Hz notch, ICA with
    the same detectors, cap and veto, 1.5 s epochs, 0.1 uV flat rule;
  * the peak-to-peak threshold matched by behaviour, not by number: the value at
    which the median per-subject rejection rate equals the Rohani task
    recording's, 18.26% (D2.4);
  * the task recording is filed under the "VCPT" key, so the features carry the
    same `_vcpt` names as Rohani et al.'s and line up column for column (H1b).
"""

import argparse
import csv
import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import scipy.io as sio

from data_pipeline.preprocessing import (
    CHANNELS_19, EPOCH_LENGTH_SEC, FLAT_THRESHOLD_V,
    epoch_signal, filter_raw, remove_artifacts_ica,
)

SFREQ_HZ = 128.0
FILE_CHANNEL_ORDER = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
                      "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz"]
RENAME_10_10_TO_10_20 = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}
NOMINAL_UNIT_TO_VOLT = 1e-6
TARGET_MEDIAN_REJECTION = 0.1826   # median rej_vcpt, docs/qc_audit.csv (D2.4)

DEFAULT_ROOT = "nasrawadi"
DEFAULT_WORK = "D:/adhd-yolo-work/nasrabadi"
CLEAN_LOG_FIELDS = ["subject_id", "status", "duration_s", "n_excluded", "capped",
                    "n_muscle_flagged", "detail"]


def discover(root: str = DEFAULT_ROOT) -> pd.DataFrame:
    """One row per .mat file: subject_id (file stem), group (from the top-level
    folder name, D1.5) and path."""
    rows = []
    for p in sorted(Path(root).rglob("*.mat")):
        top = p.relative_to(root).parts[0]
        if top.startswith("ADHD"):
            group = "ADHD"
        elif top.startswith("Control"):
            group = "Control"
        else:
            raise ValueError(f"Cannot tell the group of {p} from its folder {top!r}")
        rows.append({"subject_id": p.stem, "group": group, "path": str(p)})
    df = pd.DataFrame(rows, columns=["subject_id", "group", "path"])
    dupes = df.loc[df["subject_id"].duplicated(), "subject_id"].tolist()
    if dupes:
        raise ValueError(f"Duplicate subject IDs across folders: {dupes}")
    return df


def load_mat(path: str) -> mne.io.RawArray:
    """One recording as an MNE Raw: 19 EEG channels, 128 Hz, volts, renamed to
    this pipeline's channel names, standard 10-20 positions attached."""
    m = sio.loadmat(path)
    keys = [k for k in m if not k.startswith("__")]
    if len(keys) != 1:
        raise ValueError(f"{path}: expected one variable, found {keys}")
    x = np.asarray(m[keys[0]], dtype=float)  # int16 and float64 files share one scale (D2.3)
    if x.ndim != 2 or x.shape[1] != len(FILE_CHANNEL_ORDER):
        raise ValueError(f"{path}: expected (samples, 19), got {x.shape}")
    names = [RENAME_10_10_TO_10_20.get(c, c) for c in FILE_CHANNEL_ORDER]
    info = mne.create_info(names, SFREQ_HZ, ch_types="eeg")
    raw = mne.io.RawArray(x.T * NOMINAL_UNIT_TO_VOLT, info, verbose=False)
    raw.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="raise")
    return raw


def _append_row(path: Path, row: dict, fields: list) -> None:
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow(row)


def clean_all(root: str, work: str) -> None:
    """Filter + ICA every recording with the registered pipeline; cache the
    result. Resumable: subjects already cleaned are skipped. Label-free -- the
    manifest with groups is written for the features stage only."""
    work = Path(work)
    (work / "clean").mkdir(parents=True, exist_ok=True)
    man = discover(root)
    man.to_csv(work / "manifest.csv", index=False)
    log = work / "clean_log.csv"
    done = set(pd.read_csv(log)["subject_id"]) if log.exists() else set()
    print(f"{len(man)} recordings; {len(done)} already cleaned")
    for i, r in enumerate(man.itertuples(), 1):
        out = work / "clean" / f"{r.subject_id}_raw.fif"
        if r.subject_id in done and out.exists():
            continue
        print(f"[{i}/{len(man)}] {r.subject_id} ... ", end="", flush=True)
        row = {k: "" for k in CLEAN_LOG_FIELDS}
        row["subject_id"] = r.subject_id
        try:
            raw = filter_raw(load_mat(r.path))
            raw, diag = remove_artifacts_ica(raw)
            raw.save(out, overwrite=True, verbose=False)
            row.update(status="ok", duration_s=round(raw.times[-1], 1),
                       n_excluded=diag["n_excluded"], capped=diag["capped"],
                       n_muscle_flagged=diag.get("n_muscle_flagged", 0))
        except Exception as e:  # noqa: BLE001 - one bad file must not stop the batch
            row.update(status="failed", detail=repr(e))
        _append_row(log, row, CLEAN_LOG_FIELDS)
        print(row["status"] + (f" ({row['detail']})" if row["detail"] else ""))


def epoch_peak_to_peak(raw: mne.io.BaseRaw) -> tuple:
    """Per 1.5 s epoch: the worst channel's peak-to-peak, and whether any channel
    is flat -- exactly what epoch_signal's reject and flat rules test, on exactly
    its epochs (same events, same picks, same tmax)."""
    picks = [c for c in CHANNELS_19 if c in raw.ch_names]
    events = mne.make_fixed_length_events(raw, duration=EPOCH_LENGTH_SEC)
    ep = mne.Epochs(raw, events, tmin=0, tmax=EPOCH_LENGTH_SEC - 1.0 / raw.info["sfreq"],
                    baseline=None, preload=True, reject=None, flat=None,
                    picks=picks, verbose=False)
    d = ep.get_data()
    p2p = d.max(axis=2) - d.min(axis=2)
    return p2p.max(axis=1), (p2p < FLAT_THRESHOLD_V).any(axis=1)


def median_rejection(per_subject: list, threshold_v: float) -> float:
    return float(np.median([np.mean((w > threshold_v) | f) for w, f in per_subject]))


def solve_threshold(per_subject: list, target: float = TARGET_MEDIAN_REJECTION) -> tuple:
    """The smallest peak-to-peak threshold at which the median per-subject
    rejection rate is at or below `target`. Rejection falls as the threshold
    rises, so a binary search over the observed peak-to-peak values finds it."""
    cand = np.unique(np.concatenate([w for w, _ in per_subject]))
    if median_rejection(per_subject, cand[-1]) > target:
        raise ValueError("Flat epochs alone exceed the target rejection rate.")
    lo, hi = 0, len(cand) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if median_rejection(per_subject, cand[mid]) <= target:
            hi = mid
        else:
            lo = mid + 1
    return float(cand[lo]), median_rejection(per_subject, cand[lo])


def threshold_all(work: str) -> dict:
    """The D2 threshold. Reads only the cleaned files -- never manifest.csv."""
    files = sorted((Path(work) / "clean").glob("*_raw.fif"))
    if not files:
        raise SystemExit(f"No cleaned files in {Path(work) / 'clean'}; run the clean stage first.")
    per_subject = [epoch_peak_to_peak(mne.io.read_raw_fif(f, preload=True, verbose=False))
                   for f in files]
    t, achieved = solve_threshold(per_subject)
    out = {"threshold_v": t, "threshold_nominal_uv": round(t / NOMINAL_UNIT_TO_VOLT, 1),
           "target_median_rejection": TARGET_MEDIAN_REJECTION,
           "achieved_median_rejection": round(achieved, 4),
           "n_subjects": len(files), "n_epochs": int(sum(len(w) for w, _ in per_subject))}
    (Path(work) / "threshold.json").write_text(json.dumps(out, indent=2))
    return out


def features_all(work: str) -> None:
    """Epoch each cleaned recording with the D2 threshold and compute the task
    features. Subjects with no surviving epoch are excluded and listed (sec. 4)."""
    from training.classical_features import compute_classical_features

    work = Path(work)
    tpath = work / "threshold.json"
    if not tpath.exists():
        raise SystemExit("No threshold.json: run the threshold stage, and append its value "
                         "to D2 in docs/PREREGISTRATION.md, before computing features.")
    threshold_v = json.loads(tpath.read_text())["threshold_v"]
    man = pd.read_csv(work / "manifest.csv")
    rows, excluded = [], []
    for i, r in enumerate(man.itertuples(), 1):
        fif = work / "clean" / f"{r.subject_id}_raw.fif"
        if not fif.exists():
            excluded.append({"subject_id": r.subject_id, "reason": "cleaning failed"})
            continue
        raw = mne.io.read_raw_fif(fif, preload=True, verbose=False)
        ep, diag = epoch_signal(raw, reject_uv=threshold_v, return_diagnostics=True)
        if diag["n_kept"] == 0:
            excluded.append({"subject_id": r.subject_id, "reason": "no epoch survived rejection"})
            continue
        feats = compute_classical_features({"VCPT": ep})
        row = {"subject_id": r.subject_id, "group": r.group,
               "n_epochs": diag["n_events"], "n_kept": diag["n_kept"],
               "rejection_rate": round(diag["rejection_rate"], 4)}
        row.update({k: v for k, v in feats.items() if k.endswith("_vcpt")})
        rows.append(row)
        print(f"[{i}/{len(man)}] {r.subject_id} done")
    pd.DataFrame(rows).to_csv(work / "features.csv", index=False)
    pd.DataFrame(excluded, columns=["subject_id", "reason"]).to_csv(work / "excluded.csv", index=False)
    print(f"\n{len(rows)} subjects with features, {len(excluded)} excluded -> {work}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["clean", "threshold", "features"])
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--work-dir", default=DEFAULT_WORK)
    args = ap.parse_args()
    if args.stage == "clean":
        clean_all(args.root, args.work_dir)
    elif args.stage == "threshold":
        out = threshold_all(args.work_dir)
        print(json.dumps(out, indent=2))
        print("\nAppend threshold_nominal_uv and achieved_median_rejection to D2 in "
              "docs/PREREGISTRATION.md, and commit, BEFORE running the features stage.")
    else:
        features_all(args.work_dir)


if __name__ == "__main__":
    main()
