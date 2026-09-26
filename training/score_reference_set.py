"""
Score the dashboard's reference set through the live /predict path.

Issue 4: the dashboard's chance band (0.397-0.641) is the 95% CI of the pooled
AUC, drawn on the axis of a single child's probability. It is being replaced
with ADHD and Control dot strips of real predictions.

Those reference predictions must come from the SAME scoring path as a live
request -- same checkpoint, same 30-epochs-per-condition subset, same
preprocessing -- so this calls run_inference() directly rather than reusing
the stored out-of-fold CSVs (five different checkpoints, every epoch scored).

Only subjects the demo checkpoint never trained or selected on are valid:
it trained on fold_2..fold_4 and selected its epoch on fold_1, so fold_0 and
test are the reference set. Nothing here touches splits or validation.

Writes after every subject and skips subjects already in the output, so an
interrupted run resumes instead of starting over (same rule as #24: any
writer a partial run can touch must merge, not truncate).

Run from the repo root:
    py -m training.score_reference_set
"""

import argparse
import os
import time

import pandas as pd

from backend.app.inference import run_inference
from data_pipeline.subject_split import load_manifest

REFERENCE_SPLITS = ("fold_0", "test")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data_pipeline/splits/subject_splits.csv")
    parser.add_argument("--model", default="models/yolov8n-cls-trained.pt")
    parser.add_argument("--output", default="docs/reference_scores.csv")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    ref = manifest[manifest["split"].isin(REFERENCE_SPLITS)].reset_index(drop=True)
    print(f"Reference set: {len(ref)} subjects in {REFERENCE_SPLITS}")

    done = set()
    rows = []
    if os.path.exists(args.output):
        prev = pd.read_csv(args.output)
        rows = prev.to_dict("records")
        done = set(prev.loc[prev["status"] == "ok", "subject_id"])
        print(f"Resuming: {len(done)} already scored")

    for _, r in ref.iterrows():
        sid = r["subject_id"]
        if sid in done:
            continue
        rows = [x for x in rows if x["subject_id"] != sid]  # replace old failures
        t0 = time.time()
        try:
            # VCPT deliberately omitted: the CNN scores EC/EO only, so the
            # probability is identical and one ICA pass per subject is saved.
            out = run_inference(r["eoec_path"], None, args.model)
            p_adhd = out["confidence"] if out["predicted_class"] == "ADHD" else 1 - out["confidence"]
            rows.append({
                "subject_id": sid, "group": r["group"], "split": r["split"],
                "p_adhd": round(p_adhd, 4), "n_epochs": out["n_epochs_analyzed"],
                "status": "ok", "error": "",
            })
            print(f"  {sid} ({r['group']}, {r['split']}): p_adhd={p_adhd:.3f}  [{time.time() - t0:.0f}s]")
        except Exception as e:
            rows.append({
                "subject_id": sid, "group": r["group"], "split": r["split"],
                "p_adhd": None, "n_epochs": 0, "status": "failed", "error": str(e)[:300],
            })
            print(f"  {sid}: FAILED -- {e}")
        pd.DataFrame(rows).to_csv(args.output, index=False)

    df = pd.DataFrame(rows)
    ok = df[df["status"] == "ok"]
    print(f"\nScored {len(ok)}/{len(df)} -> {args.output}")
    print(ok.groupby("group")["p_adhd"].describe().round(3))


if __name__ == "__main__":
    main()