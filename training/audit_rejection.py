"""
Read-only audit: is the 250 uV rejection threshold discarding artifact, or
discarding signal?

Motivation. The first real cohort run showed C09110104 rejecting 69.2% of its
EC epochs. That has two opposite fixes -- exclude the subject as a bad
recording, or loosen a threshold that is too aggressive -- and picking wrong is
expensive, so this measures before deciding. A deep-dive on that subject found
the rejection was almost entirely eyes-closed (keep 30.8% EC vs 83.3% EO),
driven by O1 and Pz with only 2 of 19 channels involved: the signature of large
occipital alpha, which is exactly what eyes-closed is supposed to produce.

This script tests whether that generalises. If EC rejects systematically harder
than EO across the cohort, the threshold is condition-blind and is discarding
the alpha-blocking effect that split_eoec_by_alpha() itself relies on -- a
threshold problem, not a subject problem. If instead a handful of subjects
reject heavily in every condition, those are genuinely bad recordings and the
fix is a QC exclusion rule.

Changes no pipeline code and writes no images. Mirrors sweep_muscle_threshold.py
in being a diagnostic that reports numbers rather than altering behaviour.

    py -m training.audit_rejection
"""

import argparse
import os
import warnings

import mne
import numpy as np
import pandas as pd

from data_pipeline import subject_split
from data_pipeline.preprocessing import (
    CHANNELS_19,
    EPOCH_LENGTH_SEC,
    REJECT_PEAK_TO_PEAK_V,
    filter_raw,
    load_raw,
    remove_artifacts_ica,
    split_eoec_by_alpha,
)

THRESHOLDS_UV = [150, 200, 250, 300, 400, 500, 750, 1000]
CURRENT_UV = int(round(REJECT_PEAK_TO_PEAK_V * 1e6))
KEEP_CURRENT = "keep_{}".format(CURRENT_UV)


def epoch_p2p(raw):
    """Per-epoch, per-channel peak-to-peak in volts, on the real EEG channels only."""
    events = mne.make_fixed_length_events(raw, duration=EPOCH_LENGTH_SEC)
    sfreq = float(raw.info["sfreq"])
    picks = [ch for ch in CHANNELS_19 if ch in raw.ch_names]
    epochs = mne.Epochs(
        raw, events, tmin=0, tmax=EPOCH_LENGTH_SEC - 1.0 / sfreq, baseline=None,
        preload=True, reject=None, flat=None, picks=picks, verbose=False,
    )
    data = epochs.get_data()
    return data.max(axis=2) - data.min(axis=2), epochs.ch_names


def audit_condition(raw, subject_id, group, cond):
    p2p, ch_names = epoch_p2p(raw)
    if p2p.size == 0:
        return {"subject_id": subject_id, "group": group, "condition": cond, "n_epochs": 0}

    # MNE rejects on the worst channel, so the worst channel is what decides.
    worst = p2p.max(axis=1)
    row = {
        "subject_id": subject_id, "group": group, "condition": cond,
        "n_epochs": int(len(worst)),
        "median_p2p_uv": float(np.median(worst) * 1e6),
        "p95_p2p_uv": float(np.percentile(worst, 95) * 1e6),
    }
    for t in THRESHOLDS_UV:
        row["keep_{}".format(t)] = float((worst <= t * 1e-6).mean())

    # Whole-head elevation is a bad recording; two occipital channels is alpha.
    over = (p2p > REJECT_PEAK_TO_PEAK_V).mean(axis=0)
    order = np.argsort(-over)
    row["top_channel"] = ch_names[order[0]]
    row["top_channel_over_rate"] = float(over[order[0]])
    row["n_channels_over_20pct"] = int((over > 0.20).sum())
    return row


def audit_subject(subject_id, manifest_row):
    rows = []
    raw = remove_artifacts_ica(filter_raw(load_raw(manifest_row["eoec_path"])))[0]
    split = split_eoec_by_alpha(raw)
    rows.append(audit_condition(split["ec"], subject_id, manifest_row["group"], "EC"))
    rows.append(audit_condition(split["eo"], subject_id, manifest_row["group"], "EO"))

    vcpt = manifest_row["vcpt_path"]
    if isinstance(vcpt, str) and vcpt:
        raw_vcpt = remove_artifacts_ica(filter_raw(load_raw(vcpt)))[0]
        rows.append(audit_condition(raw_vcpt, subject_id, manifest_row["group"], "VCPT"))
    return rows


def summarize(df):
    out = ["", "=" * 74, "REJECTION AUDIT -- current threshold {} uV".format(CURRENT_UV), "=" * 74]

    out.append("\nPer-condition keep-rate by threshold (mean across subjects):")
    out.append("  {:6s} {:>6s} {:>8s} ".format("cond", "n_subj", "med uV")
               + " ".join("{:>6d}".format(t) for t in THRESHOLDS_UV))
    for cond in ("EC", "EO", "VCPT"):
        d = df[df.condition == cond]
        if d.empty:
            continue
        keeps = " ".join("{:6.1%}".format(d["keep_{}".format(t)].mean()) for t in THRESHOLDS_UV)
        out.append("  {:6s} {:6d} {:8.1f} {}".format(cond, len(d), d.median_p2p_uv.median(), keeps))

    # The decisive comparison: is EC systematically worse than EO?
    ec = df[df.condition == "EC"].set_index("subject_id")
    eo = df[df.condition == "EO"].set_index("subject_id")
    common = ec.index.intersection(eo.index)
    if len(common):
        rej_ec = 1 - ec.loc[common, KEEP_CURRENT]
        rej_eo = 1 - eo.loc[common, KEEP_CURRENT]
        worse = float((rej_ec > rej_eo).mean())
        out.append(
            "\nEC vs EO rejection at {} uV:"
            "\n  mean rejection   EC {:.1%}   vs   EO {:.1%}"
            "\n  EC rejects more than EO in {:.0%} of subjects ({}/{})".format(
                CURRENT_UV, rej_ec.mean(), rej_eo.mean(), worse, int(worse * len(common)), len(common))
        )
        if worse > 0.70:
            out.append("  -> SYSTEMATIC. The threshold is penalising eyes-closed alpha -- a threshold"
                       "\n     problem, not a subject problem. Note this also removes the alpha-blocking"
                       "\n     effect that split_eoec_by_alpha() depends on.")
        else:
            out.append("  -> No systematic EC/EO asymmetry; high rejection is subject-specific,"
                       "\n     which points at a QC exclusion rule rather than a threshold change.")

    # Subject-level triage.
    rej = df.groupby("subject_id")[KEEP_CURRENT].mean().rsub(1)
    out.append("\nSubject-level rejection at {} uV (mean across that subject's conditions):".format(CURRENT_UV))
    for label, mask in (
        ("  >50% rejected", rej > 0.5),
        ("  30-50% rejected", (rej > 0.3) & (rej <= 0.5)),
        ("  <30% rejected", rej <= 0.3),
    ):
        out.append("{:20s} {:4d} subjects".format(label, int(mask.sum())))

    heavy = rej[rej > 0.3].sort_values(ascending=False)
    if len(heavy):
        out.append("\n  Worst: " + ", ".join("{} {:.0%}".format(s, v) for s, v in heavy.head(10).items()))

    # Localised (alpha) or whole-head (bad recording)?
    out.append("\nChannels driving rejection:")
    top = df[df.n_epochs > 0].top_channel.value_counts()
    out.append("  most frequent worst channel: " + ", ".join("{} ({})".format(c, n) for c, n in top.head(6).items()))
    out.append("  mean channels over threshold in >20% of epochs: {:.1f} / 19".format(df.n_channels_over_20pct.mean()))
    out.append("  -> low means localised (alpha at O1/O2/Pz); high means a genuinely noisy recording.")

    out.append("\nCohort keep-rate by threshold (all conditions pooled):")
    for t in THRESHOLDS_UV:
        col = "keep_{}".format(t)
        flag = "   <-- current" if t == CURRENT_UV else ""
        out.append("  {:5d} uV: mean keep {:6.1%}   condition-runs below 70% keep: {:4d}{}".format(
            t, df[col].mean(), int((df[col] < 0.7).sum()), flag))

    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data_pipeline/splits/subject_splits.csv")
    ap.add_argument("--out", default="data_pipeline/splits/rejection_audit.csv")
    ap.add_argument("--limit", type=int, default=None, help="audit only the first N subjects (quick check)")
    args = ap.parse_args()

    manifest = subject_split.load_manifest(args.manifest)
    ids = manifest["subject_id"].tolist()
    if args.limit:
        ids = ids[: args.limit]

    rows, failed = [], []
    for i, sid in enumerate(ids, 1):
        row = manifest.loc[manifest["subject_id"] == sid].iloc[0]
        print("[{}/{}] {} ... ".format(i, len(ids), sid), end="", flush=True)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                subject_rows = audit_subject(sid, row)
            rows.extend(subject_rows)
            print(" ".join("{}={:.0%}rej".format(r["condition"], 1 - r.get(KEEP_CURRENT, 1))
                           for r in subject_rows), flush=True)
        except Exception as e:
            failed.append((sid, repr(e)))
            print("FAILED {!r}".format(e), flush=True)

        # Write incrementally -- an hour-long audit must not lose everything to
        # one bad subject near the end.
        if rows:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            pd.DataFrame(rows).to_csv(args.out, index=False)

    df = pd.DataFrame(rows)
    print(summarize(df))
    if failed:
        print("\nFAILED subjects ({}): {}".format(len(failed), failed))
    print("\nPer-condition rows -> {}".format(args.out))


if __name__ == "__main__":
    main()
