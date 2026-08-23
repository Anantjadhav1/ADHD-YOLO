"""
Phase 2 — batch driver for classical features, mirroring build_dataset.py's
structure exactly (same manifest-driven, per-subject-error-handled pattern)
but producing one CSV row per subject instead of images.
"""

import argparse
import os

import pandas as pd

from data_pipeline import subject_split
from training.classical_features import compute_classical_features
from data_pipeline.preprocessing import preprocess_subject


def build_classical_features_table(manifest_path: str) -> pd.DataFrame:
    manifest = subject_split.load_manifest(manifest_path)
    subject_split.verify_no_leakage(manifest)

    rows = []
    for i, row in manifest.iterrows():
        subject_id = row["subject_id"]
        print(f"[{i+1}/{len(manifest)}] {subject_id} ({row['group']}, split={row['split']}) ... ", end="")
        try:
            # pandas stores a missing vcpt_path as float NaN, not None or "" --
            # and NaN is truthy in Python, so `row.get(...) or None` silently
            # passed NaN through as if it were a real file path. Must check
            # pd.notna() explicitly.
            vcpt_path = row.get("vcpt_path")
            vcpt_path = vcpt_path if pd.notna(vcpt_path) else None
            result = preprocess_subject(row["eoec_path"], vcpt_path)
            epochs_by_task = {"EC": result["ec_epochs"], "EO": result["eo_epochs"]}
            if "vcpt_epochs" in result:
                epochs_by_task["VCPT"] = result["vcpt_epochs"]

            feats = compute_classical_features(epochs_by_task)
            feats.update({"subject_id": subject_id, "group": row["group"], "split": row["split"]})
            rows.append(feats)
            print("ok")
        except Exception as e:
            print(f"FAILED: {e}")
            rows.append({"subject_id": subject_id, "group": row["group"], "split": row["split"], "error": str(e)})

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default="data_pipeline/splits/subject_splits.csv")
    parser.add_argument("--output", type=str, default="data_pipeline/splits/classical_features.csv")
    args = parser.parse_args()

    df = build_classical_features_table(args.manifest)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()