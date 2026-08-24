"""
Phase 1 — Subject-wise train/val/test + cross-validation split.

Why this exists as its own step, before image_conversion.py runs at scale:
this dataset produces MANY epoch-level images per subject (multiple EC/EO
windows, multiple VCPT windows). If the train/test split happens after
images exist -- e.g. by randomly splitting the image files themselves --
epochs from the same child end up on both sides of the split. The model
then partly learns "this is child #14's brainwave signature" rather than
"this is what ADHD looks like," and reported accuracy is inflated in a way
that won't replicate on a new child. Splitting at the SUBJECT level first,
and only generating images after every subject already has a split label,
makes that leakage structurally impossible instead of relying on discipline.

Two-stage design, matching PROJECT.md §4.6 and §6 (Phase 2):
  1. A stratified holdout TEST set (subject-level) carved out once and never
     touched for model selection, threshold tuning, or fold-based decisions --
     only for the one-shot final evaluation against the 75.8% / 84.5%
     baselines in Phase 2.
  2. Stratified 5-fold CV over the remaining subjects, used during
     development (model selection, early stopping, hyperparameter tuning).

Every downstream script (image_conversion.py, training, fusion) should read
the manifest this produces via load_manifest() rather than re-deriving
splits -- one manifest, one source of truth for which child is in which
bucket.
"""

import argparse
import glob
import os
import random
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

RANDOM_SEED_DEFAULT = 42
DEFAULT_TEST_SIZE = 0.15
DEFAULT_N_FOLDS = 5


@dataclass
class SubjectRecord:
    subject_id: str
    group: str  # "ADHD" | "Control"
    eoec_path: str
    vcpt_path: str | None = None


def discover_subjects(data_dir: str, pattern: str = "*-EOEC.edf") -> list[SubjectRecord]:
    """
    Scan data_dir recursively for EOEC files and build one record per unique
    subject_id. Every subject has exactly one EOEC (resting-state) recording,
    so scanning EOEC files -- not VCPT -- gives one row per subject cleanly.
    The matching VCPT file, if present, is attached to the same record.

    Lazily imports preprocessing.parse_filename so this module (and its
    dry-run mode below) doesn't require mne just to touch filenames.
    """
    from data_pipeline.preprocessing import parse_filename  # lazy: avoid mne dependency for dry-run/tests

    subjects: dict[str, SubjectRecord] = {}
    raw_matches = sorted(glob.glob(os.path.join(data_dir, "**", pattern), recursive=True))
    raw_matches += sorted(glob.glob(os.path.join(data_dir, "**", pattern.upper()), recursive=True))
    # On case-insensitive filesystems (Windows), the lower- and upper-case glob
    # above both resolve to the SAME files, so raw_matches contains every real
    # file twice with identical paths -- not two distinct files. Deduping by
    # normalized path keeps the double-glob (needed on case-sensitive
    # filesystems, e.g. Linux, where .edf and .EDF really are different files)
    # without raising a spurious "duplicate subject" error on every subject.
    deduped: dict[str, str] = {}
    for path in raw_matches:
        deduped.setdefault(os.path.normcase(os.path.abspath(path)), path)
    matches = sorted(deduped.values())

    # Files that look like they belong to this task but didn't match the strict
    # naming pattern (e.g. a stray double dot before the extension) would
    # otherwise be silently dropped from the cohort instead of raising or
    # appearing in the manifest. Flag them instead of trusting the glob silently.
    task_hint = pattern.strip("*").lstrip("-").split(".")[0]
    matched_paths = set(deduped.keys())
    for root, _dirs, files in os.walk(data_dir):
        for fname in files:
            if task_hint.lower() not in fname.lower():
                continue
            full = os.path.normcase(os.path.abspath(os.path.join(root, fname)))
            if full not in matched_paths:
                warnings.warn(
                    f"File looks like a {task_hint} recording but didn't match the "
                    f"expected naming pattern and was SKIPPED (missing from the "
                    f"manifest): {os.path.join(root, fname)}"
                )

    for path in matches:
        sf = parse_filename(path)
        if sf.subject_id in subjects:
            raise ValueError(
                f"Duplicate EOEC file for subject {sf.subject_id}: "
                f"{subjects[sf.subject_id].eoec_path} and {path}. "
                "Expected exactly one EOEC recording per subject -- check for "
                "re-recorded/duplicate sessions before splitting."
            )
        vcpt_candidates = sorted(
            glob.glob(os.path.join(data_dir, "**", f"{sf.subject_id}-*-VCPT.*"), recursive=True)
        )
        subjects[sf.subject_id] = SubjectRecord(
            subject_id=sf.subject_id,
            group=sf.group,
            eoec_path=path,
            vcpt_path=vcpt_candidates[0] if vcpt_candidates else None,
        )
    return list(subjects.values())


def subjects_to_frame(subjects: list[SubjectRecord]) -> pd.DataFrame:
    return pd.DataFrame([asdict(s) for s in subjects])


def make_splits(
    df: pd.DataFrame,
    test_size: float = DEFAULT_TEST_SIZE,
    n_folds: int = DEFAULT_N_FOLDS,
    seed: int = RANDOM_SEED_DEFAULT,
) -> pd.DataFrame:
    """
    Every row in `df` is one child, so there is no epoch-level leakage risk
    here by construction -- that risk only reappears later if image_conversion.py
    generates images without joining each one back to its subject_id's split.
    """
    if df["subject_id"].duplicated().any():
        dupes = df.loc[df["subject_id"].duplicated(), "subject_id"].tolist()
        raise ValueError(f"Duplicate subject_id(s), cannot split: {dupes}")

    for grp, n in df["group"].value_counts().items():
        if n < n_folds:
            raise ValueError(
                f"Group '{grp}' has only {n} subjects, fewer than n_folds={n_folds}. "
                "Reduce n_folds or gather more subjects before splitting."
            )
        if n < 2:
            raise ValueError(f"Group '{grp}' has too few subjects ({n}) to stratify a test holdout.")

    dev_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df["group"], random_state=seed,
    )
    dev_df = dev_df.reset_index(drop=True).copy()
    test_df = test_df.copy()
    test_df["split"] = "test"

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    dev_df["split"] = ""
    for fold_idx, (_, val_idx) in enumerate(skf.split(dev_df, dev_df["group"])):
        dev_df.loc[val_idx, "split"] = f"fold_{fold_idx}"

    manifest = pd.concat([dev_df, test_df], ignore_index=True)
    return manifest[["subject_id", "group", "split", "eoec_path", "vcpt_path"]]


def verify_no_leakage(manifest: pd.DataFrame) -> None:
    """Cheap assertions, run right after make_splits() and again right before
    training in case the manifest ever gets hand-edited. Catches a silent bug
    here instead of burning a training run on it."""
    assert manifest["subject_id"].is_unique, "a subject_id appears more than once in the manifest"
    assert (manifest["split"].astype(str).str.len() > 0).all(), "every subject needs a split assigned"
    test_ids = set(manifest.loc[manifest["split"] == "test", "subject_id"])
    fold_ids = set(manifest.loc[manifest["split"] != "test", "subject_id"])
    assert test_ids.isdisjoint(fold_ids), "a subject appears in both the test set and a CV fold"


def summarize(manifest: pd.DataFrame) -> str:
    lines = ["Split summary (subject-level, not epoch-level):"]
    order = sorted(manifest["split"].unique(), key=lambda s: (s != "test", s))
    for split_name in order:
        sub = manifest[manifest["split"] == split_name]
        counts = sub["group"].value_counts().to_dict()
        lines.append(f"  {split_name:>8}: n={len(sub):3d}  {counts}")
    return "\n".join(lines)


def save_manifest(manifest: pd.DataFrame, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)


def load_manifest(path: str) -> pd.DataFrame:
    """Downstream scripts (image_conversion.py, training, fusion) should call
    this rather than re-deriving splits, so every representation and every
    model uses the same subject -> split mapping."""
    return pd.read_csv(path)


def get_split_for_subject(manifest: pd.DataFrame, subject_id: str) -> str:
    row = manifest.loc[manifest["subject_id"] == subject_id]
    if row.empty:
        raise KeyError(f"subject_id {subject_id!r} not found in manifest")
    return row.iloc[0]["split"]


def _make_synthetic_subjects(n_adhd: int = 49, n_control: int = 54, seed: int = RANDOM_SEED_DEFAULT) -> pd.DataFrame:
    """
    Dry-run helper matching the real cohort's class balance (49 ADHD / 54
    Control per PROJECT.md), so the split logic can be exercised and tested
    before the real dataset is available locally (see PROJECT.md's open
    items -- "share the raw dataset" is still pending). Not used once
    discover_subjects() has real files to scan.
    """
    rng = random.Random(seed)
    seen = set()

    def unique_id(prefix: str) -> str:
        while True:
            sid = f"{prefix}{rng.randint(10_000_000, 99_999_999)}"
            if sid not in seen:
                seen.add(sid)
                return sid

    rows = []
    for _ in range(n_adhd):
        sid = unique_id("F")
        rows.append({"subject_id": sid, "group": "ADHD", "eoec_path": f"synthetic/{sid}-EOEC.edf", "vcpt_path": None})
    for _ in range(n_control):
        sid = unique_id("C")
        rows.append({"subject_id": sid, "group": "Control", "eoec_path": f"synthetic/{sid}-EOEC.edf", "vcpt_path": None})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Directory containing raw .edf files. Omit to run a DRY RUN on synthetic "
             "subject IDs (49 ADHD / 54 Control, matching the real cohort's class balance) "
             "to sanity-check the split logic before the real dataset is available.",
    )
    parser.add_argument("--output", type=str, default="data_pipeline/splits/subject_splits.csv")
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED_DEFAULT)
    args = parser.parse_args()

    if args.data_dir:
        subjects = discover_subjects(args.data_dir)
        if not subjects:
            raise SystemExit(f"No *-EOEC.edf files found under {args.data_dir}")
        df = subjects_to_frame(subjects)
        print(f"Discovered {len(df)} subjects from {args.data_dir}")
    else:
        print("No --data-dir given -- running a DRY RUN on synthetic subject IDs.")
        print("This validates the split logic only. Re-run with --data-dir once the real dataset is available.\n")
        df = _make_synthetic_subjects(seed=args.seed)

    manifest = make_splits(df, test_size=args.test_size, n_folds=args.n_folds, seed=args.seed)
    verify_no_leakage(manifest)
    print(summarize(manifest))

    save_manifest(manifest, args.output)
    print(f"\nSaved manifest -> {args.output}")


if __name__ == "__main__":
    main()