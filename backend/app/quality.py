"""
Per-upload EEG quality control for /predict.

Kept out of inference.py on purpose: inference.py pulls in torch, pywt and
mne_connectivity transitively, none of which CI installs, so anything defined
there cannot be unit-tested in CI. This module depends only on
data_pipeline.preprocessing, which the existing tests already import.
"""

from data_pipeline.preprocessing import HIGH_REJECTION_RATE_WARN


def quality_control(result: dict, n_scored: int) -> dict:
    """
    Per-upload EEG quality control, surfaced in the response so a prediction
    from a mostly-rejected recording cannot look like one from a clean
    recording.

    Not cosmetic. In the 36-subject reference set (2026-09-26), 20 subjects
    had >30% of epochs rejected in at least one condition and 5 hit the ICA
    component cap; C12031144 lost 85.8% of EC and 72.6% of EO epochs. Until
    now a recording like that produced a prediction with nothing attached to
    say so -- the same class of mistake as the dashboard's old chance band:
    presenting a number without what qualifies it.

    Reports what was actually USED, not what was recorded. rejection_rate
    comes from the same epoch_signal() calls that produced the epochs scored
    below, and n_epochs_scored is the capped subset the model actually saw,
    which is smaller again.

    Reports and does NOT decide: README open issue 2 (include / exclude /
    flag problem subjects) is still undecided, so this returns `flagged`
    rather than anything implying a policy such as `usable`.
    """
    conditions = {}
    for task, key in (("EC", "reject_ec"), ("EO", "reject_eo"), ("VCPT", "reject_vcpt")):
        diag = result.get(key)
        if not diag:
            continue
        rate = float(diag["rejection_rate"])
        conditions[task] = {
            "n_epochs_recorded": int(diag["n_events"]),
            "n_epochs_surviving": int(diag["n_kept"]),
            "rejection_rate": round(rate, 4),
            "high_rejection": bool(rate > HIGH_REJECTION_RATE_WARN),
        }

    ica = {}
    for name, key in (("EOEC", "ica_eoec"), ("VCPT", "ica_vcpt")):
        diag = result.get(key)
        if not diag:
            continue
        ica[name] = {
            "n_components_excluded": int(diag["n_excluded"]),
            "n_muscle_flagged": int(diag.get("n_muscle_flagged", 0)),
            "capped": bool(diag["capped"]),
        }

    # high_rejection uses the SAME float comparison as epoch_signal()'s own
    # warning (rate > HIGH_REJECTION_RATE_WARN), so this flag always agrees with
    # what the pipeline logged. The cost: 1 - 70/100 is 0.30000000000000004, so
    # an exactly-30% recording is flagged. The note says "at or above" so it
    # never prints "30.0% ... above 30%", which reads as self-contradictory.
    notes = []
    for task, c in conditions.items():
        if c["high_rejection"]:
            notes.append(
                f"{task}: {c['rejection_rate']:.1%} of epochs rejected "
                f"({c['n_epochs_surviving']}/{c['n_epochs_recorded']} kept), "
                f"at or above the {HIGH_REJECTION_RATE_WARN:.0%} flag threshold."
            )
    # Worded around n_muscle_flagged, not n_components_excluded: the cap is
    # about how many components muscle detection WANTED to remove. The final
    # excluded count is often small because the topography veto runs after
    # the cap, so "hit the cap (1 excluded)" would read as a contradiction.
    for name, i in ica.items():
        if i["capped"]:
            notes.append(
                f"{name}: muscle detection flagged {i['n_muscle_flagged']} ICA "
                "components, more than the cap allows -- the decomposition is "
                "dominated by artifact, and the pipeline's own rule is to flag "
                "this recording for manual QC."
            )
    if result.get("eoec_ambiguous"):
        ratio = result.get("alpha_ratio")
        ratio_txt = f"{ratio:.2f}" if isinstance(ratio, (int, float)) else "unknown"
        notes.append(
            f"EC/EO boundary is ambiguous (occipital alpha ratio {ratio_txt}) -- "
            "which half is eyes-closed is not certain, so EC and EO may be swapped."
        )

    return {
        "conditions": conditions,
        "ica": ica,
        "n_epochs_scored": int(n_scored),
        "flagged": bool(notes),
        "notes": notes,
    }
