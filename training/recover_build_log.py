"""
Rebuild the 103 build_log.csv rows destroyed by a partial run.

save_log() used to truncate. A 5-subject --subjects rebuild wiped the other
103. The images were never touched, so the record is reconstructible -- and
from better sources than the terminal log:

  subject_id/group/split  <- the manifest, authoritative for all 108
  n_ec/n_eo/n_vcpt        <- COUNTED from images on disk, which reflects what
                             actually exists rather than what was printed
  status                  <- images present => "ok", absent => "skipped_ambiguous"
                             (the cohort run reported failed: 0)

An earlier version parsed build_cohort.log and got 0 rows: PowerShell's
Tee-Object writes UTF-16LE, and reading it as UTF-8 with errors="ignore"
silently produced garbage rather than failing. The log is now used only for
the optional detail field, with encoding sniffing, and its absence is not
fatal.

Rows already in build_log.csv are KEPT -- those come from the newer
5-subject rebuild and carry the detected boundaries.

    py -m training.recover_build_log
"""
import argparse, csv, os, re
from collections import defaultdict
from pathlib import Path

LINE = re.compile(r"^\[(\d+)/(\d+)\]\s+(\S+)\s+\((\w+),\s*split=(\S+?)\)\s+\.\.\.\s+(\S+)(?:\s+\((.*)\))?\s*$")
FIELDS = ["subject_id","group","split","status","n_ec","n_eo","n_vcpt",
          "split_rule","boundary_frac","detail"]


def count_images(dataset_dir):
    """Per-subject, per-task image counts from filenames: <sid>_<task>_<n>.png"""
    counts = defaultdict(lambda: defaultdict(int))
    for p in (Path(dataset_dir) / "scalogram").rglob("*.png"):
        parts = p.stem.split("_")
        if len(parts) >= 2:
            counts[parts[0]][parts[1]] += 1
    return counts


def read_details(log_path):
    """Optional detail strings. Tries several encodings -- Tee-Object writes
    UTF-16LE, which read as UTF-8 yields garbage instead of an error."""
    if not os.path.exists(log_path):
        return {}
    for enc in ("utf-16", "utf-16-le", "utf-8-sig", "utf-8", "latin-1"):
        try:
            text = Path(log_path).read_text(encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
        out = {}
        for line in text.splitlines():
            m = LINE.match(line.strip())
            if m:
                out[m.group(3)] = m.group(7) or ""
        if out:
            print(f"  read {len(out)} detail lines from {log_path} (encoding: {enc})")
            return out
    print(f"  could not parse {log_path} in any encoding -- detail left blank")
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data_pipeline/splits/subject_splits.csv")
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--log", default="build_cohort.log")
    args = ap.parse_args()

    from data_pipeline import subject_split
    man = subject_split.load_manifest(args.manifest)
    print(f"Manifest: {len(man)} subjects")

    counts = count_images(args.dataset)
    print(f"Images on disk for {len(counts)} subjects")
    details = read_details(args.log)

    rebuilt = {}
    for _, r in man.iterrows():
        sid = r["subject_id"]
        c = counts.get(sid, {})
        n_ec, n_eo, n_vcpt = c.get("EC", 0), c.get("EO", 0), c.get("VCPT", 0)
        has = (n_ec + n_eo + n_vcpt) > 0
        rebuilt[sid] = {
            "subject_id": sid, "group": r["group"], "split": r["split"],
            "status": "ok" if has else "skipped_ambiguous",
            "n_ec": n_ec, "n_eo": n_eo, "n_vcpt": n_vcpt,
            "split_rule": "midpoint" if has else "",
            "boundary_frac": "", "detail": details.get(sid, ""),
        }

    log_path = os.path.join(args.dataset, "build_log.csv")
    existing = {}
    if os.path.exists(log_path):
        with open(log_path, newline="") as f:
            for row in csv.DictReader(f):
                existing[row["subject_id"]] = {k: row.get(k, "") for k in FIELDS}
        print(f"Keeping {len(existing)} newer rows already in build_log.csv")

    merged = {**rebuilt, **existing}   # existing wins -- it carries the boundaries
    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for sid in sorted(merged):
            w.writerow(merged[sid])

    n_ok = sum(1 for v in merged.values() if v["status"] == "ok")
    n_skip = sum(1 for v in merged.values() if v["status"] == "skipped_ambiguous")
    print(f"\nbuild_log.csv now holds {len(merged)} subjects: {n_ok} ok, {n_skip} skipped")
    print(f"  -> {log_path}")


if __name__ == "__main__":
    main()