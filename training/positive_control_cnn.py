"""
Positive control for the CNN arm: Rest vs Task through train_yolo_cls.run_cv,
imported and called UNMODIFIED.

The classical arm passed its positive control (training/positive_control.py,
AUC 0.686 on EO vs VCPT). The CNN arm is the one more worth checking: its
pipeline has more places to go quietly wrong -- image encoding, fold assembly,
Ultralytics' class indexing, epoch-to-subject aggregation -- and it scored
0.52 on ADHD. If it separates rest from task, its null is about the data.

How the images are fed to an unmodified run_cv:
  * Images come from build_dataset.py, named <subject>_<COND>_<n>.png under
    <root>/<representation>/<split>/<diagnosis>/. They are hard-linked (copied
    if linking fails) into a second tree where the class folder is the
    CONDITION: task images under "ADHD", rest images under "Control" -- the two
    class names run_cv knows. Every output here says task/rest.
  * Each child becomes two aggregation units, "<subject>-TASK" and
    "<subject>-REST". run_cv's OOF table asserts one row per subject_id (a
    duplicate would mean a subject was validated in two folds), and a child
    legitimately contributes a rest row and a task row to the SAME fold.
  * Both units keep the child's split, so folds stay subject-wise by
    construction; this is asserted again before training.
  * The cohort is the 84 subjects the CNN arms were actually evaluated on
    (read from the committed OOF file), restricted to those with a task
    recording, development folds only. Same children, same folds, same 6R
    three-way split, same frozen backbone, patience and disabled augmentation.

The one deliberate difference from the ADHD run is the epoch cap: these images
were built with --max-epochs-per-task 40 (the ADHD images used 300), because
regenerating at 300 on CPU is many hours. Fewer images per child should make
the control harder to pass, not easier, so a pass still counts; a failure would
need a rerun at the full cap before being read as a pipeline fault.

Run:  py -m training.positive_control_cnn --images-root D:/adhd-yolo-work/images_all
"""

import argparse
import os
import shutil
from pathlib import Path

import pandas as pd

from training.positive_control import (
    REST_LABEL, TASK_LABEL, _arrays_by_subject, _pooled_auc,
    cluster_bootstrap_ci, paired_win_rate, within_subject_permutation_p,
)

CNN_COHORT_OOF = "runs/frozen/scalogram_oof_cnn_probs.csv"
TASK_COND = "VCPT"


def parse_image_name(fname: str) -> tuple:
    """'F08080102_EO_0007.png' -> ('F08080102', 'EO')."""
    parts = Path(fname).stem.split("_")
    return parts[0], parts[1]


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def assemble_condition_tree(images_root: str, representation: str, cohort: dict,
                            rest_conds: list, out_root: str) -> pd.DataFrame:
    """Re-file `cohort` children's images by condition. cohort: {child: split}.
    Returns the per-unit manifest run_cv will read."""
    src_rep = Path(images_root) / representation
    dst_rep = Path(out_root) / representation
    if dst_rep.exists():
        shutil.rmtree(dst_rep)  # never mix a stale tree into a new run

    # Pass 1: decide what goes in. Pass 2: link it. Linking while scanning and
    # filtering the manifest afterwards left one-sided children's images in the
    # tree -- and run_cv reads every image in a split folder, so they were
    # trained on and scored as unpaired units the manifest knew nothing about.
    # Caught by tests/test_positive_control_cnn.py.
    planned = []
    for img in src_rep.glob("*/*/*.png"):
        child, cond = parse_image_name(img.name)
        if child not in cohort:
            continue
        if cond == TASK_COND:
            unit, cls = f"{child}-TASK", TASK_LABEL
        elif cond in rest_conds:
            unit, cls = f"{child}-REST", REST_LABEL
        else:
            continue
        split = img.parent.parent.name
        assert split == cohort[child], (
            f"{img} sits in split {split!r} but the manifest puts {child} in {cohort[child]!r}")
        planned.append((img, child, unit, cls, split))

    # Only children with BOTH a rest and a task side: a one-sided child adds
    # nothing to a paired comparison and would unbalance the classes.
    sides = {}
    for _, child, _, cls, _ in planned:
        sides.setdefault(child, set()).add(cls)
    paired = {c for c, s in sides.items() if s == {TASK_LABEL, REST_LABEL}}

    counts = {}
    for img, child, unit, cls, split in planned:
        if child in paired:
            _link(img, dst_rep / split / cls / img.name.replace(child, unit, 1))
            counts[unit] = counts.get(unit, 0) + 1

    rows = []
    for child in sorted(paired):
        for unit, cls in ((f"{child}-TASK", TASK_LABEL), (f"{child}-REST", REST_LABEL)):
            rows.append({"subject_id": unit, "group": cls, "split": cohort[child],
                         "n_images": counts[unit]})
    units = pd.DataFrame(rows, columns=["subject_id", "group", "split", "n_images"])
    # The structural guarantee, checked rather than assumed.
    per_child = units.assign(child=units["subject_id"].str.rsplit("-", n=1).str[0])
    assert (per_child.groupby("child")["split"].nunique() == 1).all(), "a child spans two folds"
    return units


def to_child_level(oof: pd.DataFrame) -> pd.DataFrame:
    """run_cv's OOF (one row per unit) -> rows keyed by CHILD, for the paired stats."""
    out = oof.copy()
    out["subject_id"] = out["subject_id"].str.rsplit("-", n=1).str[0]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-root", required=True)
    ap.add_argument("--manifest", default="data_pipeline/splits/subject_splits.csv")
    ap.add_argument("--representation", default="scalogram", choices=["scalogram", "topomap"])
    ap.add_argument("--rest", default="eo", choices=["eo", "ec+eo"])
    ap.add_argument("--work-dir", default="D:/adhd-yolo-work/positive_control_cnn")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--images-epoch-cap", type=int, default=40,
                    help="The --max-epochs-per-task the images were built with; recorded in the summary.")
    args = ap.parse_args()

    from training import train_yolo_cls  # deferred: keeps --help fast

    manifest = pd.read_csv(args.manifest)
    cnn_cohort = set(pd.read_csv(CNN_COHORT_OOF)["subject_id"])
    dev = manifest[(manifest["split"] != "test") & manifest["subject_id"].isin(cnn_cohort)]
    cohort = dict(zip(dev["subject_id"], dev["split"]))
    rest_conds = [c.upper() for c in args.rest.split("+")]
    tag = f"{args.representation}_{args.rest.replace('+', '')}"

    work = Path(args.work_dir) / tag
    units = assemble_condition_tree(args.images_root, args.representation, cohort,
                                    rest_conds, str(work / "images"))
    n_children = units["subject_id"].str.rsplit("-", n=1).str[0].nunique()
    unit_manifest = work / "unit_manifest.csv"
    units[["subject_id", "group", "split"]].to_csv(unit_manifest, index=False)
    print(f"CNN positive control: rest ({args.rest.upper()}) vs task (VCPT), {args.representation}")
    print(f"  {n_children} children of the CNN arm's 84 (dev, with a task recording), "
          f"{len(units)} units, {int(units['n_images'].sum())} images")
    print("  run_cv's 'sens' = task recall, 'spec' = rest recall; 'n_subjects' counts units\n")

    folds, oof = train_yolo_cls.run_cv(str(unit_manifest), str(work / "images"),
                                       args.representation, str(work / "runs"),
                                       epochs=args.epochs)
    kids = to_child_level(oof)
    groups = _arrays_by_subject(kids)
    pooled = _pooled_auc(groups)
    lo, hi = cluster_bootstrap_ci(kids, seed=args.seed)
    p = within_subject_permutation_p(kids, pooled, seed=args.seed)
    wins, n_pairs = paired_win_rate(kids)

    print(f"\n=== CNN {args.representation}: Rest ({args.rest.upper()}) vs Task, "
          f"{n_children} children ===")
    print(f"  pooled AUC        {pooled:.3f}")
    print(f"  95% CI (children) [{lo:.3f}, {hi:.3f}]")
    print(f"  permutation p     {p:.4f}  (labels flipped within child, 5000 reps)")
    print(f"  task > own rest   {wins}/{n_pairs} children ({wins / n_pairs:.0%})")
    print(f"  fold AUCs         {', '.join(f'{a:.3f}' for a in folds['auc'])}")

    os.makedirs("docs", exist_ok=True)
    out = kids.copy()
    out["true_label"] = out["true_label"].map({TASK_LABEL: "task", REST_LABEL: "rest"})
    out = out.rename(columns={"pred_prob_adhd": "pred_prob_task"}).drop(columns="pred_label")
    out.to_csv(f"docs/positive_control_cnn_{tag}_oof.csv", index=False)
    folds.rename(columns={"sensitivity": "task_recall", "specificity": "rest_recall",
                          "n_subjects": "n_units"}).to_csv(
        f"docs/positive_control_cnn_{tag}_folds.csv", index=False)
    pd.DataFrame([{"representation": args.representation, "rest": args.rest,
                   "n_children": n_children, "epochs_per_condition_cap": args.images_epoch_cap,
                   "pooled_auc": pooled, "ci_low": lo, "ci_high": hi, "perm_p": p,
                   "task_gt_rest": wins, "n_pairs": n_pairs}]).to_csv(
        f"docs/positive_control_cnn_{tag}_summary.csv", index=False)
    print(f"\n  wrote docs/positive_control_cnn_{tag}_{{oof,folds,summary}}.csv")


if __name__ == "__main__":
    main()
