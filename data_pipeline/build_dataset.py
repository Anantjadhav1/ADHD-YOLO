"""
Phase 1 — batch pipeline driver: run preprocessing -> split -> image_conversion
across every subject in the cohort, in one command, producing an auditable
per-subject log.

Design choices, matching the values already established in preprocessing.py /
subject_split.py / image_conversion.py:

- Never silently trust a subject flagged "ambiguous" by the EC/EO alpha-blocking
  split (preprocessing.split_eoec_by_alpha). By default these subjects are
  SKIPPED from image generation and logged, not force-processed -- the same
  "flagged for manual QC, not silently trusted" rule preprocessing.py already
  states. Override with --include-ambiguous only after manually reviewing which
  subjects that means.

- One subject failing (corrupt file, ICA non-convergence, missing channels,
  whatever) must not crash a 103-subject batch run. Every subject is wrapped in
  try/except; failures are logged to build_log.csv with the real exception, not
  swallowed or silently skipped.

- The subject-wise split (subject_split.py) is READ here, never regenerated.
  Regenerating it on every batch run would silently reshuffle who's in test /
  which fold, invalidating any result already computed against the existing
  manifest. If no manifest exists yet, this script tells you to run
  subject_split.py first rather than quietly creating one with different
  defaults (different --seed, different --test-size, etc.).
"""

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path

from data_pipeline import subject_split
from data_pipeline.image_conversion import process_subject_from_manifest
from data_pipeline.preprocessing import preprocess_subject


@dataclass
class SubjectResult:
    subject_id: str
    group: str
    split: str
    status: str  # "ok" | "skipped_ambiguous" | "failed"
    n_ec: int = 0
    n_eo: int = 0
    n_vcpt: int = 0
    detail: str = ""


def run_subject(
    subject_id: str,
    row,
    manifest,
    output_dir: str,
    max_epochs_per_task: int | None,
    include_ambiguous: bool,
) -> SubjectResult:
    group, split = row["group"], row["split"]
    eoec_path = row["eoec_path"]
    vcpt_path = row["vcpt_path"] if isinstance(row["vcpt_path"], str) and row["vcpt_path"] else None

    try:
        result = preprocess_subject(eoec_path, vcpt_path)
    except Exception as e:
        return SubjectResult(subject_id, group, split, "failed", detail=f"preprocessing failed: {e!r}")

    if result["eoec_ambiguous"] and not include_ambiguous:
        return SubjectResult(
            subject_id, group, split, "skipped_ambiguous",
            detail=f"alpha_ratio={result['alpha_ratio']:.2f} -- flagged for manual QC, not auto-processed",
        )

    epochs_by_task = {"EC": result["ec_epochs"], "EO": result["eo_epochs"]}
    if "vcpt_epochs" in result:
        epochs_by_task["VCPT"] = result["vcpt_epochs"]

    try:
        counts = process_subject_from_manifest(
            epochs_by_task, subject_id, manifest, output_dir, max_epochs=max_epochs_per_task,
        )
    except Exception as e:
        return SubjectResult(subject_id, group, split, "failed", detail=f"image generation failed: {e!r}")

    return SubjectResult(
        subject_id, group, split, "ok",
        n_ec=counts.get("EC", 0), n_eo=counts.get("EO", 0), n_vcpt=counts.get("VCPT", 0),
    )


def run_batch(
    manifest_path: str,
    output_dir: str,
    max_epochs_per_task: int | None = None,
    include_ambiguous: bool = False,
    subjects_filter: list[str] | None = None,
) -> list[SubjectResult]:
    manifest = subject_split.load_manifest(manifest_path)
    subject_split.verify_no_leakage(manifest)  # re-check even though subject_split.py already checked once --
                                                # catches a hand-edited manifest before it burns a batch run

    subject_ids = subjects_filter or manifest["subject_id"].tolist()
    results = []
    for i, subject_id in enumerate(subject_ids, 1):
        matches = manifest.loc[manifest["subject_id"] == subject_id]
        if matches.empty:
            results.append(SubjectResult(subject_id, "?", "?", "failed", detail="not found in manifest"))
            print(f"[{i}/{len(subject_ids)}] {subject_id} ... failed (not in manifest)")
            continue
        row = matches.iloc[0]
        print(f"[{i}/{len(subject_ids)}] {subject_id} ({row['group']}, split={row['split']}) ... ", end="", flush=True)
        result = run_subject(subject_id, row, manifest, output_dir, max_epochs_per_task, include_ambiguous)
        print(result.status + (f" ({result.detail})" if result.status != "ok" else ""))
        results.append(result)

    return results


def save_log(results: list[SubjectResult], output_dir: str) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log_path = os.path.join(output_dir, "build_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject_id", "group", "split", "status", "n_ec", "n_eo", "n_vcpt", "detail"])
        for r in results:
            writer.writerow([r.subject_id, r.group, r.split, r.status, r.n_ec, r.n_eo, r.n_vcpt, r.detail])
    return log_path


def summarize(results: list[SubjectResult]) -> str:
    ok = [r for r in results if r.status == "ok"]
    skipped = [r for r in results if r.status == "skipped_ambiguous"]
    failed = [r for r in results if r.status == "failed"]

    lines = [
        f"Processed: {len(results)} subjects",
        f"  ok:                   {len(ok)}",
        f"  skipped (ambiguous):  {len(skipped)}" + (f"  -> {[r.subject_id for r in skipped]}" if skipped else ""),
        f"  failed:               {len(failed)}" + (f"  -> {[(r.subject_id, r.detail) for r in failed]}" if failed else ""),
    ]
    if ok:
        total_ec = sum(r.n_ec for r in ok)
        total_eo = sum(r.n_eo for r in ok)
        total_vcpt = sum(r.n_vcpt for r in ok)
        lines.append(f"Epochs imaged: EC={total_ec} EO={total_eo} VCPT={total_vcpt} (scalogram + topomap generated for each)")
    if skipped:
        lines.append(
            "\nAmbiguous subjects were NOT processed -- review their alpha_ratio in build_log.csv, "
            "manually confirm EC/EO ordering, then re-run with --subjects <id> --include-ambiguous "
            "if they check out."
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--manifest", type=str, default="data_pipeline/splits/subject_splits.csv",
        help="Output of subject_split.py. Must already exist -- this script never generates it.",
    )
    parser.add_argument("--output-dir", type=str, default="data_pipeline/images")
    parser.add_argument(
        "--max-epochs-per-task", type=int, default=None,
        help="Cap epochs per subject/task -- use this for a quick smoke-test run on a few subjects "
             "before committing to the full batch.",
    )
    parser.add_argument(
        "--include-ambiguous", action="store_true",
        help="Process subjects flagged ambiguous by the EC/EO alpha-blocking split anyway. "
             "Off by default -- review flagged subjects manually first.",
    )
    parser.add_argument(
        "--subjects", type=str, nargs="*", default=None,
        help="Optional: process only these subject_ids (e.g. to re-run subjects that failed last time).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.manifest):
        raise SystemExit(
            f"No manifest found at {args.manifest}. Run data_pipeline/subject_split.py "
            "first -- this script reads a split, it never creates one."
        )

    results = run_batch(
        args.manifest, args.output_dir,
        max_epochs_per_task=args.max_epochs_per_task,
        include_ambiguous=args.include_ambiguous,
        subjects_filter=args.subjects,
    )
    print()
    print(summarize(results))
    log_path = save_log(results, args.output_dir)
    print(f"\nPer-subject log -> {log_path}")


if __name__ == "__main__":
    main()