"""
Pre-registered H3: Riemannian tangent-space features on the Rohani development
folds (docs/PREREGISTRATION.md, sec. 2). Committed before it is run.

Registered specification, implemented here and nowhere else:
  * cohort: the 84-subject CNN cohort (read from the committed CNN out-of-fold
    file), so the result sits beside the headline table;
  * per subject and per condition (EC, EO): ONE spatial covariance matrix of the
    19 channels over all surviving 1.5 s epochs, OAS shrinkage;
  * projected to the tangent space at the Riemannian mean of the TRAINING folds;
    EC and EO vectors concatenated (2 x 190 = 380 features);
  * standardisation -> logistic regression (L2, C = 1, max_iter = 1000), fitted
    on training folds only;
  * the five existing development folds; nothing is tuned, so each outer fold is
    scored by a model trained on the other four;
  * decision: positive only if one-sided permutation p < 0.05/3 AND the 95% CI
    lower bound > 0.5 (training.preregistered.decide).

Implementation choices the registration does not fix, stated here:
  * preprocessing is preprocess_subject() unchanged, with the EC/EO boundary
    chosen exactly as build_dataset.py chose it for the CNN images (trusted
    detected boundary where one exists, else midpoint, falling back to the
    midpoint if a detected boundary is rejected);
  * samples are converted to microvolts before the covariance is estimated. A
    common scale factor changes no tangent-space feature (the affine-invariant
    metric is invariant to it); it only keeps the matrices well conditioned in
    floating point;
  * a subject with no surviving EC or EO epoch cannot have that covariance and is
    excluded and listed (registration sec. 4 anticipates exclusion only when
    features cannot be computed).

Stages, from the repo root:
    py -m training.h3_riemann covariances   # label-free; ~84 ICA runs, resumable
    py -m training.h3_riemann analyse       # the one registered analysis
"""

import argparse
import csv
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from training.results_table import pooled_stats

COHORT_OOF = "runs/frozen/scalogram_oof_cnn_probs.csv"
MANIFEST = "data_pipeline/splits/subject_splits.csv"
DEFAULT_WORK = "D:/adhd-yolo-work/h3_riemann"
RESULTS_CSV = "docs/preregistered_results.csv"
SEED = 42
CONDITIONS = ("EC", "EO")
LOG_FIELDS = ["subject_id", "status", "n_ec", "n_eo", "detail"]


def subject_covariance(epochs) -> np.ndarray:
    """One OAS covariance over all surviving epochs, samples in microvolts."""
    from pyriemann.utils.covariance import covariances

    x = epochs.get_data() * 1e6                      # (n_epochs, 19, n_times)
    x = np.concatenate(list(x), axis=1)              # (19, n_epochs * n_times)
    return covariances(x[np.newaxis], estimator="oas")[0]


def cohort_ids() -> list:
    return sorted(pd.read_csv(COHORT_OOF)["subject_id"])


def covariances_all(work: str) -> None:
    """Label-free: preprocessing and covariances only; diagnosis is never read."""
    from data_pipeline import subject_split
    from data_pipeline.build_dataset import DEFAULT_BOUNDARIES_CSV, load_boundaries
    from data_pipeline.preprocessing import CHANNELS_19, preprocess_subject

    work = Path(work)
    (work / "cov").mkdir(parents=True, exist_ok=True)
    man = subject_split.load_manifest(MANIFEST).set_index("subject_id")
    boundaries = load_boundaries(DEFAULT_BOUNDARIES_CSV)
    log = work / "cov_log.csv"
    done = set(pd.read_csv(log)["subject_id"]) if log.exists() else set()
    ids = cohort_ids()
    for i, sid in enumerate(ids, 1):
        if sid in done:
            continue
        print(f"[{i}/{len(ids)}] {sid} ... ", end="", flush=True)
        row = {"subject_id": sid, "status": "", "n_ec": "", "n_eo": "", "detail": ""}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    res = preprocess_subject(man.loc[sid, "eoec_path"], None,
                                             boundary_frac=boundaries.get(sid))
                except ValueError:
                    res = preprocess_subject(man.loc[sid, "eoec_path"], None, boundary_frac=None)
            ec, eo = res["ec_epochs"], res["eo_epochs"]
            row.update(n_ec=len(ec), n_eo=len(eo))
            if len(ec) == 0 or len(eo) == 0:
                row.update(status="excluded", detail="no surviving EC or EO epoch")
            else:
                assert ec.ch_names == list(CHANNELS_19) or sorted(ec.ch_names) == sorted(CHANNELS_19)
                np.savez(work / "cov" / f"{sid}.npz", ec=subject_covariance(ec),
                         eo=subject_covariance(eo), ch_names=np.array(ec.ch_names))
                row["status"] = "ok"
        except Exception as e:  # noqa: BLE001 - one subject must not stop the batch
            row.update(status="failed", detail=repr(e))
        new = not log.exists()
        with open(log, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
        print(row["status"] + (f" ({row['detail']})" if row["detail"] else ""))


def cross_validate(covs: dict, labels: pd.DataFrame) -> pd.DataFrame:
    """covs: {condition: array (n_subjects, 19, 19)} in the row order of `labels`
    (columns subject_id, group, split). Returns one out-of-fold row per subject."""
    from pyriemann.tangentspace import TangentSpace
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    split = labels["split"].to_numpy()
    y = (labels["group"] == "ADHD").astype(int).to_numpy()
    prob = np.full(len(labels), np.nan)
    for fold in sorted(set(split)):
        te, tr = split == fold, split != fold
        spaces = {c: TangentSpace(metric="riemann").fit(covs[c][tr]) for c in CONDITIONS}
        feats = lambda idx: np.hstack([spaces[c].transform(covs[c][idx]) for c in CONDITIONS])
        clf = Pipeline([("scale", StandardScaler()),
                        ("lr", LogisticRegression(C=1.0, max_iter=1000))])
        clf.fit(feats(tr), y[tr])
        prob[te] = clf.predict_proba(feats(te))[:, 1]
    assert np.isfinite(prob).all(), "every subject must be scored exactly once"
    return pd.DataFrame({"subject_id": labels["subject_id"].to_numpy(),
                         "true_label": labels["group"].to_numpy(), "pred_prob_adhd": prob})


def analyse(work: str) -> dict:
    from data_pipeline import subject_split
    from training.preregistered import ALPHA, decide

    work = Path(work)
    log = pd.read_csv(work / "cov_log.csv")
    ok = sorted(log.loc[log["status"] == "ok", "subject_id"])
    not_ok = log[log["status"] != "ok"]
    man = subject_split.load_manifest(MANIFEST).set_index("subject_id")
    labels = man.loc[ok, ["group", "split"]].reset_index()
    if (labels["split"] == "test").any():
        raise SystemExit("A test-split subject is in the H3 cohort; it must not be.")
    loaded = [np.load(work / "cov" / f"{sid}.npz") for sid in ok]
    covs = {c: np.stack([d[c.lower()] for d in loaded]) for c in CONDITIONS}
    oof = cross_validate(covs, labels)
    s = pooled_stats(oof, seed=SEED)
    acc = float(((oof["pred_prob_adhd"] >= 0.5) == (oof["true_label"] == "ADHD")).mean())
    decision = decide(s)
    print(f"H3  n={s['n']:3d} ({s['n_adhd']} ADHD)  AUC {s['auc']:.3f}  "
          f"95% CI [{s['ci_low']:.3f}, {s['ci_high']:.3f}]  p {s['perm_p']:.4f}  "
          f"accuracy {acc:.3f}  -> {decision}   (alpha {ALPHA:.4f})")
    if len(not_ok):
        print("not analysed:", not_ok[["subject_id", "status", "detail"]].to_dict("records"))
    oof.to_csv(work / "preregistered_H3_oof.csv", index=False)
    row = {"hypothesis": "H3", "n": s["n"], "n_adhd": s["n_adhd"], "auc": s["auc"],
           "ci_low": s["ci_low"], "ci_high": s["ci_high"], "accuracy": acc,
           "perm_p": s["perm_p"], "decision": decision}
    res = pd.read_csv(RESULTS_CSV) if os.path.exists(RESULTS_CSV) else pd.DataFrame()
    if len(res):
        res = res[res["hypothesis"] != "H3"]
    pd.concat([res, pd.DataFrame([row])], ignore_index=True).to_csv(RESULTS_CSV, index=False)
    print(f"-> {RESULTS_CSV}")
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["covariances", "analyse"])
    ap.add_argument("--work-dir", default=DEFAULT_WORK)
    args = ap.parse_args()
    if args.stage == "covariances":
        covariances_all(args.work_dir)
    else:
        analyse(args.work_dir)


if __name__ == "__main__":
    main()
