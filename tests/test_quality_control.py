"""
Tests for backend.app.quality.quality_control(), the per-upload QC block that
/predict returns (added 2026-10-16).

The contract these pin down:
  * A recording above the rejection threshold, or one where the ICA cap fired,
    or one with an ambiguous EC/EO boundary, is `flagged` with a plain-language
    note -- a prediction from such a recording must not look like one from a
    clean recording.
  * A clean recording is not flagged, and still reports its rejection rates.
  * The block is valid strict JSON. NaN is not, and Starlette's JSONResponse
    serialises with allow_nan=False; that exact class of bug once made every
    /predict request return 500 (2026-08-21).
  * The high-rejection comparison is the SAME float comparison epoch_signal()
    uses, so the API flag can never disagree with what the pipeline logged.

Run:  py -m pytest tests/test_quality_control.py
"""

import json

from backend.app.quality import quality_control
from data_pipeline.preprocessing import HIGH_REJECTION_RATE_WARN


def _reject(n_events, n_kept):
    # Same arithmetic epoch_signal() uses to produce rejection_rate.
    return {"n_events": n_events, "n_kept": n_kept,
            "rejection_rate": 1.0 - (n_kept / n_events), "reject_uv": 250}


def _ica(n_excluded, capped, n_muscle_flagged=0):
    return {"n_excluded": n_excluded, "capped": capped, "n_muscle_flagged": n_muscle_flagged}


def _strict_json(obj):
    return json.loads(json.dumps(obj, allow_nan=False))


def test_clean_recording_is_not_flagged_but_reports_rates():
    qc = quality_control({"reject_ec": _reject(200, 190), "reject_eo": _reject(200, 185),
                          "ica_eoec": _ica(2, False, 3), "eoec_ambiguous": False,
                          "alpha_ratio": 3.1}, n_scored=60)
    assert qc["flagged"] is False and qc["notes"] == []
    assert set(qc["conditions"]) == {"EC", "EO"}
    assert qc["conditions"]["EC"]["n_epochs_surviving"] == 190
    assert qc["n_epochs_scored"] == 60
    _strict_json(qc)


def test_c12031144_shape_is_flagged_on_all_three_grounds():
    # The real values for C12031144 from a live /predict on 2026-10-16.
    qc = quality_control({"reject_ec": _reject(219, 31), "reject_eo": _reject(219, 60),
                          "ica_eoec": _ica(1, True, 8), "eoec_ambiguous": False,
                          "alpha_ratio": 2.0}, n_scored=60)
    assert qc["flagged"] is True
    assert qc["conditions"]["EC"]["high_rejection"] and qc["conditions"]["EO"]["high_rejection"]
    assert qc["ica"]["EOEC"]["capped"] is True
    assert len(qc["notes"]) == 3
    # Worded around what muscle detection flagged, not the final exclude count:
    # "hit the cap (1 excluded)" would read as a contradiction.
    assert any("flagged 8 ICA components" in n for n in qc["notes"])
    _strict_json(qc)


def test_ambiguous_boundary_is_flagged_even_without_alpha_ratio():
    qc = quality_control({"reject_ec": _reject(10, 10), "reject_eo": _reject(10, 10),
                          "eoec_ambiguous": True, "alpha_ratio": None}, n_scored=20)
    assert qc["flagged"] is True
    assert any("ambiguous" in n and "unknown" in n for n in qc["notes"])
    _strict_json(qc)


def test_vcpt_is_reported_when_present_and_absent_otherwise():
    base = {"reject_ec": _reject(200, 190), "reject_eo": _reject(200, 190),
            "ica_eoec": _ica(2, False)}
    assert "VCPT" not in quality_control(base, 60)["conditions"]
    with_vcpt = dict(base, reject_vcpt=_reject(900, 850), ica_vcpt=_ica(1, False))
    qc = quality_control(with_vcpt, 60)
    assert "VCPT" in qc["conditions"] and "VCPT" in qc["ica"]


def test_threshold_agrees_with_the_pipeline_comparison():
    # 1 - 70/100 is 0.30000000000000004 in floating point. epoch_signal() warns
    # on rate > HIGH_REJECTION_RATE_WARN, so it warns here; the API must agree
    # rather than quietly disagree with the audit log.
    rate = _reject(100, 70)["rejection_rate"]
    pipeline_warns = rate > HIGH_REJECTION_RATE_WARN
    qc = quality_control({"reject_ec": _reject(100, 70), "reject_eo": _reject(100, 100)}, 40)
    assert qc["conditions"]["EC"]["high_rejection"] is pipeline_warns
    # ...and the note must not claim "30.0% ... above 30%".
    assert all("at or above" in n for n in qc["notes"] if n.startswith("EC:"))


def test_old_diagnostics_without_n_muscle_flagged_do_not_crash():
    qc = quality_control({"reject_ec": _reject(200, 190), "reject_eo": _reject(200, 190),
                          "ica_eoec": {"n_excluded": 2, "capped": False}}, 60)
    assert qc["ica"]["EOEC"]["n_muscle_flagged"] == 0
