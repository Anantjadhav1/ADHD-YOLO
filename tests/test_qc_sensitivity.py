"""
Tests for training/qc_sensitivity.py (added 2026-10-16).

Pins:
  * the anchor -- with no subjects excluded, the scored AUCs are the headline
    table's (so any difference under a policy is the policy's doing);
  * the policy subject counts on the 84-subject cohort;
  * the top-decile thresholds are measured from the cohort, not the
    pipeline's 30% warning (which sits at EC's median).

Reads tracked files only (docs/qc_audit.csv, runs/*/..._oof_*.csv), so it runs
in CI. Small resample counts keep it fast; the AUC point estimates do not
depend on them.
Run:  py -m pytest tests/test_qc_sensitivity.py
"""

import pandas as pd

from training.qc_sensitivity import QC_AUDIT, policy_masks, sensitivity_table


def test_thresholds_are_measured_tails_not_the_30_percent_warning():
    qc = pd.read_csv(QC_AUDIT)
    _, t = policy_masks(qc)
    assert qc["rej_ec"].median() > 0.30                 # 30% is EC's middle, not its tail
    assert t["rej_ec"] > 0.60 and t["rej_eo"] > 0.40     # measured 63% / 45%


def test_no_exclusion_reproduces_the_headline_aucs_and_policy_counts():
    table, _ = sensitivity_table(pd.read_csv(QC_AUDIT), n_boot=50, n_perm=50)
    none = table[table["policy"] == "none (as reported)"].set_index("arm")["auc"].round(3)
    assert none.to_dict() == {
        "CNN scalogram, frozen backbone": 0.522,
        "CNN topomap, frozen backbone": 0.523,
        "Classical SVM, 551 features with coherence": 0.556,
    }
    n = table.groupby("policy", sort=False)["n"].first().to_dict()
    assert n == {"none (as reported)": 84, "drop ICA-cap subjects": 63,
                 "drop top-decile rejection": 65, "drop either": 52}
