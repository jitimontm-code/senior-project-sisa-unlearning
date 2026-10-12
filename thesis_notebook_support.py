"""Create lightweight thesis-audit artifacts from validated experiment checkpoints.

This stage does not refit the expensive baseline or SISA experiments.  It exposes
data lineage, split integrity, shard structure, aggregation, deletion auditing,
and fidelity details needed by the thesis-facing notebook.
"""
from __future__ import annotations

import ctypes
import json
import os
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from final_pipeline import (
    APP_FEATURES, CFG, DIRS, ROOT, SCHEMA_VERSION, TABLE_SPECS, atomic_csv,
    calibrate, fit_platt, load_matrix, log, positive_probability, save_json,
    update_manifest,
)
from expanded_experiments import (
    MODEL_NAMES, _frozen_original_calibration, _model_path, _raw_ensemble,
    _slug, _strategy_assignments, get_stratified_assignment,
)


def _physical_memory_gib() -> float | None:
    """Return installed RAM on Windows without adding a dependency."""
    if os.name != "nt":
        return None

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong), ("memory_load", ctypes.c_ulong),
            ("total_physical", ctypes.c_ulonglong),
            ("available_physical", ctypes.c_ulonglong),
            ("total_page_file", ctypes.c_ulonglong),
            ("available_page_file", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("available_virtual", ctypes.c_ulonglong),
            ("available_extended_virtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.length = ctypes.sizeof(MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return status.total_physical / 1024**3


def _table_profile(path: Path, id_col: str = "SK_ID_CURR") -> tuple[int, int, int]:
    """Count rows and distinct applicant IDs in bounded-memory chunks."""
    n_columns = len(pd.read_csv(path, nrows=0).columns)
    rows = 0
    unique_ids: set[int] = set()
    for chunk in pd.read_csv(path, usecols=[id_col], chunksize=1_000_000):
        rows += len(chunk)
        unique_ids.update(chunk[id_col].dropna().astype(np.int64).unique().tolist())
    return rows, n_columns, len(unique_ids)


def _dataset_and_preprocessing_audit() -> None:
    sources = [("application", "application_train.csv", "Primary applicant table and TARGET")]
    sources.extend((name, spec[0], "Applicant-linked historical transactions")
                   for name, spec in TABLE_SPECS.items())
    table_rows = []
    for name, filename, role in sources:
        rows, columns, applicants = _table_profile(ROOT / filename)
        table_rows.append({"source": name, "file": filename, "rows": rows,
                           "columns": columns, "unique_applicants": applicants,
                           "role": role})
    atomic_csv(pd.DataFrame(table_rows), DIRS["results"] / "dataset_table_summary.csv")

    marker = json.loads((DIRS["cache"] / f"preprocessing_complete_{CFG.hash}.json").read_text())
    prep_rows = [
        {"stage": "identity", "fit_scope": "not learned", "detail": "SK_ID_CURR retained; one-to-one merge and row order asserted"},
        {"stage": "external aggregation", "fit_scope": "all rows by applicant ID", "detail": "sum/count/min/max/mean; installments also PAYMENT_DELAY and PAYMENT_RATIO"},
        {"stage": "missing-column filter", "fit_scope": "train only", "detail": f"drop when train missing rate > {CFG.missing_drop_threshold:.0%}"},
        {"stage": "numeric imputation", "fit_scope": "train only", "detail": "median + missing indicators"},
        {"stage": "categorical imputation", "fit_scope": "train only", "detail": "most frequent"},
        {"stage": "categorical encoding", "fit_scope": "train only", "detail": "one-hot; unknown categories ignored"},
        {"stage": "output", "fit_scope": marker["fit_scope"], "detail": f"CSR float32; {marker['raw_features']} retained raw and {marker['n_features']} transformed features"},
    ]
    atomic_csv(pd.DataFrame(prep_rows), DIRS["results"] / "preprocessing_audit.csv")

    split = pd.read_csv(DIRS["config"] / f"data_splits_{CFG.hash}.csv")
    checks = [
        {"check": "global_SK_ID_CURR_unique", "value": int(split.SK_ID_CURR.is_unique), "status": "PASS" if split.SK_ID_CURR.is_unique else "FAIL"},
        {"check": "all_rows_have_one_split", "value": int(split.split.notna().all()), "status": "PASS" if split.split.notna().all() else "FAIL"},
        {"check": "expected_split_roles", "value": split.split.nunique(), "status": "PASS" if set(split.split) == {"train","validation","calibration","test"} else "FAIL"},
    ]
    split_sets = {name: set(group.SK_ID_CURR) for name, group in split.groupby("split")}
    names = sorted(split_sets)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap = len(split_sets[left] & split_sets[right])
            checks.append({"check": f"overlap_{left}_{right}", "value": overlap,
                           "status": "PASS" if overlap == 0 else "FAIL"})
    for name in ["train", "validation", "calibration", "test"]:
        X, y, ids = load_matrix(name)
        expected = split.loc[split.split == name].set_index("SK_ID_CURR").TARGET
        aligned = np.array_equal(y, expected.loc[ids].to_numpy())
        checks.extend([
            {"check": f"{name}_matrix_rows", "value": X.shape[0], "status": "PASS" if X.shape[0] == len(ids) else "FAIL"},
            {"check": f"{name}_ID_unique", "value": len(np.unique(ids)), "status": "PASS" if len(np.unique(ids)) == len(ids) else "FAIL"},
            {"check": f"{name}_target_alignment", "value": int(aligned), "status": "PASS" if aligned else "FAIL"},
        ])
    atomic_csv(pd.DataFrame(checks), DIRS["results"] / "split_integrity.csv")


def _model_configuration() -> None:
    rows = [
        {"model": "Logistic Regression", "configuration": "StandardScaler(with_mean=False); C=0.2; class_weight=balanced; max_iter=500", "role": "linear baseline"},
        {"model": "Random Forest", "configuration": "n_estimators=180; max_depth=12; min_samples_leaf=10; max_features=sqrt; class_weight=balanced_subsample", "role": "bagged tree baseline"},
        {"model": "LightGBM", "configuration": "n_estimators=300; learning_rate=0.04; num_leaves=31; subsample=0.85; colsample_bytree=0.85; deterministic", "role": "gradient-boosted tree baseline and main S sweep"},
        {"model": "XGBoost", "configuration": "n_estimators=300; learning_rate=0.04; max_depth=5; min_child_weight=5; subsample=0.85; colsample_bytree=0.85; tree_method=hist", "role": "gradient-boosted tree baseline"},
    ]
    atomic_csv(pd.DataFrame(rows), DIRS["results"] / "baseline_model_config.csv")


def _shard_diagnostics_and_aggregation() -> None:
    assignments, feature_labels, risk = _strategy_assignments()
    emb = np.load(DIRS["cache"] / f"phase2_train_svd8_{CFG.hash}.npy")
    diagnostic_rows = []
    for strategy, assignment in assignments.items():
        for shard, group in assignment.groupby("shard"):
            idx = group.row_index.to_numpy(dtype=int)
            clusters, cluster_counts = np.unique(feature_labels[idx], return_counts=True)
            diagnostic_rows.append({
                "strategy": strategy, "shard": int(shard), "records": len(idx),
                "size_share": len(idx) / len(assignment), "default_rate": group.TARGET.mean(),
                "oof_risk_mean": risk[idx].mean(), "oof_risk_sd": risk[idx].std(),
                "svd_component_1_mean": emb[idx, 0].mean(), "svd_component_1_sd": emb[idx, 0].std(),
                "dominant_feature_cluster": int(clusters[np.argmax(cluster_counts)]),
                "dominant_cluster_share": cluster_counts.max() / len(idx),
            })
    atomic_csv(pd.DataFrame(diagnostic_rows), DIRS["results"] / "shard_diagnostics.csv")

    Xc, yc, _ = load_matrix("calibration")
    meta_fit, meta_cal = train_test_split(np.arange(len(yc)), test_size=.5,
                                         stratify=yc, random_state=CFG.seed)
    coefficient_rows = []
    for strategy in ["specialist", "risk_band"]:
        assignment = assignments[strategy]
        models = [joblib.load(_model_path("LightGBM", CFG.shards, shard, strategy))["model"]
                  for shard in range(CFG.shards)]
        component = np.stack([positive_probability(model, Xc) for model in models], axis=1)
        meta = LogisticRegression(C=.2, class_weight="balanced", max_iter=500,
                                  random_state=CFG.seed).fit(component[meta_fit], yc[meta_fit])
        raw_meta = meta.predict_proba(component[meta_cal])[:, 1]
        calibration = fit_platt(raw_meta, yc[meta_cal])
        for shard, coefficient in enumerate(meta.coef_[0]):
            coefficient_rows.append({"strategy": strategy, "component": f"shard_{shard}",
                                     "coefficient": coefficient, "meta_intercept": meta.intercept_[0],
                                     "meta_fit_rows": len(meta_fit), "platt_fit_rows": len(meta_cal),
                                     "platt_slope": calibration.coef_[0, 0],
                                     "platt_intercept": calibration.intercept_[0]})
    atomic_csv(pd.DataFrame(coefficient_rows), DIRS["results"] / "aggregation_coefficients.csv")

    contribution = [
        {"strategy": "stratified_balanced", "classification": "standard experimental baseline", "assignment_basis": "class-wise shuffled round robin", "intended_structure": "balanced size and target rate", "matched_deletion_pattern": "none/control", "aggregation": "retained-size weighted", "primary_hypothesis": "stable utility and predictable 1/S single-delete workload", "main_limitation": "does not intentionally localize feature/risk cohorts"},
        {"strategy": "random_balanced", "classification": "control", "assignment_basis": "global random balanced assignment", "intended_structure": "balanced size without explicit class constraint", "matched_deletion_pattern": "none/control", "aggregation": "retained-size weighted", "primary_hypothesis": "separate balancing effect from class stratification", "main_limitation": "request locality is accidental"},
        {"strategy": "representative", "classification": "project implementation/adaptation", "assignment_basis": "train-only SVD8 + MiniBatchKMeans; each cluster spread round robin", "intended_structure": "each shard represents multiple feature clusters", "matched_deletion_pattern": "none/control", "aggregation": "retained-size weighted", "primary_hypothesis": "preserve utility through population representation", "main_limitation": "representation depends on embedding and k-means choices"},
        {"strategy": "specialist", "classification": "project-specific experimental design", "assignment_basis": "train-only feature clusters concentrated into equal contiguous shards", "intended_structure": "feature-specialized shards", "matched_deletion_pattern": "feature-localized", "aggregation": "weighted; learned meta-aggregator also tested", "primary_hypothesis": "reduce affected shards for feature-localized erasure", "main_limitation": "specialization may need learned aggregation and may not transfer"},
        {"strategy": "risk_band", "classification": "project-specific experimental design", "assignment_basis": "3-fold train-only OOF LightGBM risk sorted into equal bands", "intended_structure": "risk-specialized shards", "matched_deletion_pattern": "risk-localized", "aggregation": "weighted; learned meta-aggregator also tested", "primary_hypothesis": "reduce affected shards for risk-localized erasure", "main_limitation": "naive averaging mixes different class priors and loses utility"},
    ]
    atomic_csv(pd.DataFrame(contribution), DIRS["results"] / "sharding_contribution_summary.csv")


def _deletion_and_fidelity_audit() -> None:
    Xtr, ytr, idtr = load_matrix("train")
    Xt, yt, idt = load_matrix("test")
    assignment = get_stratified_assignment(ytr, idtr, CFG.shards)
    request = pd.read_csv(DIRS["deletion_requests"] / f"phase2_standard_single_{CFG.hash}.csv").iloc[0]
    delete_row = int(request.original_row_index)
    delete_id = int(request.SK_ID_CURR)
    affected_shard = int(request.original_shard)
    audit_rows, forget_rows, fidelity_rows = [], [], []

    for model_name in MODEL_NAMES:
        original_models = [joblib.load(_model_path(model_name, CFG.shards, shard))["model"]
                           for shard in range(CFG.shards)]
        original_sizes = assignment.groupby("shard").size().sort_index().to_numpy()
        calibrator, threshold = _frozen_original_calibration(original_models, original_sizes)
        exact_models, gold_models = list(original_models), []
        exact_sizes = original_sizes.copy()
        exact_sizes[affected_shard] -= 1
        for shard in range(CFG.shards):
            exact_path = DIRS["models"] / f"phase2_exact_single_{_slug(model_name)}_shard0_{CFG.hash}.joblib"
            gold_path = DIRS["models"] / f"phase2_gold_single_{_slug(model_name)}_shard{shard}_{CFG.hash}.joblib"
            if shard == affected_shard:
                exact_models[shard] = joblib.load(exact_path)["model"]
            gold_models.append(joblib.load(gold_path)["model"])

        original_prob = calibrate(calibrator, _raw_ensemble(original_models, Xtr[delete_row], original_sizes/original_sizes.sum()))[0]
        exact_prob = calibrate(calibrator, _raw_ensemble(exact_models, Xtr[delete_row], exact_sizes/exact_sizes.sum()))[0]
        gold_prob = calibrate(calibrator, _raw_ensemble(gold_models, Xtr[delete_row], exact_sizes/exact_sizes.sum()))[0]
        forget_rows.append({"model": model_name, "SK_ID_CURR": delete_id, "target": int(ytr[delete_row]),
                            "threshold": threshold, "original_probability": original_prob,
                            "exact_probability": exact_prob, "gold_probability": gold_prob,
                            "original_to_exact_change": exact_prob-original_prob,
                            "exact_gold_absolute_difference": abs(exact_prob-gold_prob),
                            "original_decision": int(original_prob >= threshold),
                            "exact_decision": int(exact_prob >= threshold),
                            "gold_decision": int(gold_prob >= threshold)})

        affected = assignment[assignment.shard == affected_shard]
        retained = affected[affected.row_index != delete_row]
        audit_rows.append({"model": model_name, "SK_ID_CURR": delete_id,
                           "assigned_shard": affected_shard, "present_before": int(delete_id in set(affected.SK_ID_CURR)),
                           "present_in_exact_training": int(delete_id in set(retained.SK_ID_CURR)),
                           "present_in_gold_training": int(delete_id in set(assignment.loc[assignment.row_index != delete_row, "SK_ID_CURR"])),
                           "affected_shard_rows_before": len(affected), "affected_shard_rows_after": len(retained),
                           "audit_status": "PASS" if delete_id not in set(retained.SK_ID_CURR) else "FAIL"})

        saved = np.load(DIRS["results"] / f"phase2_exact_gold_single_{_slug(model_name)}_{CFG.hash}.npz")
        exact, gold = saved["exact"], saved["gold"]
        original_frame = pd.read_csv(DIRS["results"] / f"phase2_sisa_{_slug(model_name)}_S{CFG.shards}_{CFG.hash}.csv.gz")
        original_map = original_frame.set_index("SK_ID_CURR").probability
        original = original_map.loc[idt].to_numpy()
        diff = exact-gold
        fidelity_rows.append({"model": model_name, "scenario": "single_record",
                              "exact_gold_mae": np.mean(np.abs(diff)),
                              "exact_gold_rmse": np.sqrt(np.mean(diff**2)),
                              "exact_gold_max_abs": np.max(np.abs(diff)),
                              "exact_gold_correlation": np.corrcoef(exact, gold)[0, 1],
                              "exact_gold_auc_difference": roc_auc_score(yt, exact)-roc_auc_score(yt, gold),
                              "exact_gold_ap_difference": average_precision_score(yt, exact)-average_precision_score(yt, gold),
                              "exact_gold_decision_disagreement": np.mean((exact >= threshold) != (gold >= threshold)),
                              "original_exact_mae": np.mean(np.abs(original-exact)),
                              "original_exact_decision_change": np.mean((original >= threshold) != (exact >= threshold))})

    atomic_csv(pd.DataFrame(audit_rows), DIRS["results"] / "direct_deletion_audit.csv")
    atomic_csv(pd.DataFrame(forget_rows), DIRS["results"] / "forget_set_behavior_four_models.csv")
    atomic_csv(pd.DataFrame(fidelity_rows), DIRS["results"] / "fidelity_detailed_four_models.csv")


def _protocol_environment_and_traceability() -> None:
    protocol = [
        {"component": "data roles", "definition": "train fits preprocessing/models; calibration fits Platt; validation selects thresholds; test reports final metrics"},
        {"component": "randomness", "definition": f"master seed={CFG.seed}; deterministic derived seeds by model/shard/fold"},
        {"component": "SISA comparison", "definition": "same S=4 assignment, model family, retained-size aggregation, calibration and test protocol"},
        {"component": "exact unlearning", "definition": "retrain only affected shard(s) from retained records with compatible seed/configuration"},
        {"component": "gold retraining", "definition": "retrain every shard from retained records with the same seed/configuration"},
        {"component": "approximate unlearning", "definition": "continued boosting heuristic; evaluated only for LightGBM/XGBoost and never called exact"},
        {"component": "runtime", "definition": "wall-clock fit/retrain time on one local machine; workload-derived values explicitly labeled estimated"},
        {"component": "statistical scope", "definition": "paired n=10 locality requests use bootstrap CI + one-sided Wilcoxon; small-n runtime is descriptive"},
    ]
    atomic_csv(pd.DataFrame(protocol), DIRS["results"] / "experimental_protocol.csv")

    environment = {
        "schema_version": SCHEMA_VERSION, "config_hash": CFG.hash,
        "python": platform.python_version(), "platform": platform.platform(),
        "processor": platform.processor(), "logical_cpu_count": os.cpu_count(),
        "installed_ram_gib": _physical_memory_gib(), "configured_threads": CFG.threads,
        "seed": CFG.seed,
    }
    save_json(environment, DIRS["config"] / "environment_detailed.json")

    trace = [
        {"thesis_section": "3.1 Dataset", "notebook_heading": "3.1 Dataset and source tables", "code_or_method": "bounded-memory source profiling; identity-safe merge", "artifact": "dataset_table_summary.csv; split_integrity.csv", "figure": "phase2_class_distribution.png"},
        {"thesis_section": "3.2 Preprocessing", "notebook_heading": "3.2 Leakage-safe preprocessing", "code_or_method": "train-only ColumnTransformer", "artifact": "preprocessing_audit.csv", "figure": "-"},
        {"thesis_section": "3.3 Baselines", "notebook_heading": "3.3 Baseline models", "code_or_method": "make_model; calibration; threshold", "artifact": "baseline_model_config.csv; baseline_performance.csv", "figure": "baseline_and_sisa_auc.png"},
        {"thesis_section": "3.4 SISA architecture", "notebook_heading": "3.4 SISA architecture walkthrough", "code_or_method": "assignment, shard training, aggregation", "artifact": "sisa_four_model_comparison.csv", "figure": "phase2_sisa_four_models.png"},
        {"thesis_section": "3.5 Sharding strategies", "notebook_heading": "3.5 Five sharding strategies", "code_or_method": "visible strategy implementations", "artifact": "sharding_contribution_summary.csv; shard_diagnostics.csv", "figure": "phase2_sharding_utility_locality.png"},
        {"thesis_section": "3.6 Aggregation", "notebook_heading": "3.6 Simple and learned aggregation", "code_or_method": "weighted mean; logistic meta-model; held-out Platt", "artifact": "aggregation_coefficients.csv", "figure": "-"},
        {"thesis_section": "3.7 Deletion generation", "notebook_heading": "3.7 Deletion-request generator", "code_or_method": "distributed, feature-localized, risk-localized", "artifact": "deletion_scale_locality.csv", "figure": "phase2_deletion_scale.png"},
        {"thesis_section": "3.8 Unlearning", "notebook_heading": "3.8 Exact, approximate, and gold definitions", "code_or_method": "affected-shard retrain and all-shard gold", "artifact": "direct_deletion_audit.csv", "figure": "phase2_exact_four_models.png"},
        {"thesis_section": "3.9 Evaluation", "notebook_heading": "3.9 Utility, fidelity, efficiency metrics", "code_or_method": "AUC/AP/KS/Brier; MAE/RMSE/correlation; workload", "artifact": "fidelity_detailed_four_models.csv", "figure": "phase2_exact_four_models.png"},
        {"thesis_section": "3.10 Privacy", "notebook_heading": "3.10 Focused MIA", "code_or_method": "loss-based member/nonmember score", "artifact": "privacy_mia.csv", "figure": "-"},
        {"thesis_section": "3.11 Protocol", "notebook_heading": "3.11 Experimental protocol", "code_or_method": "fixed seed and isolated data roles", "artifact": "experimental_protocol.csv", "figure": "-"},
        {"thesis_section": "3.12 Environment", "notebook_heading": "3.12 Execution environment", "code_or_method": "runtime inventory", "artifact": "environment_detailed.json", "figure": "-"},
        {"thesis_section": "4.1-4.8 Results", "notebook_heading": "Chapter 4 results and interpretation", "code_or_method": "checkpointed analyses", "artifact": "phase2_consolidated_findings.csv; objective_coverage.csv", "figure": "all phase2 figures"},
    ]
    atomic_csv(pd.DataFrame(trace), DIRS["results"] / "traceability_matrix.csv")


def stage_thesis_notebook_support() -> None:
    """Build all transparent audit artifacts without rerunning expensive models."""
    required = [
        "dataset_table_summary.csv", "preprocessing_audit.csv", "split_integrity.csv",
        "baseline_model_config.csv", "shard_diagnostics.csv", "aggregation_coefficients.csv",
        "direct_deletion_audit.csv", "forget_set_behavior_four_models.csv",
        "fidelity_detailed_four_models.csv", "experimental_protocol.csv",
        "traceability_matrix.csv", "sharding_contribution_summary.csv",
    ]
    if all((DIRS["results"] / name).exists() for name in required) and \
            (DIRS["config"] / "environment_detailed.json").exists():
        log("Validated thesis notebook support checkpoints")
        return
    _dataset_and_preprocessing_audit()
    _model_configuration()
    _shard_diagnostics_and_aggregation()
    _deletion_and_fidelity_audit()
    _protocol_environment_and_traceability()
    update_manifest(
        "thesis_notebook_support", "complete",
        ["results/dataset_table_summary.csv", "results/preprocessing_audit.csv",
         "results/split_integrity.csv", "results/shard_diagnostics.csv",
         "results/aggregation_coefficients.csv", "results/direct_deletion_audit.csv",
         "results/forget_set_behavior_four_models.csv", "results/fidelity_detailed_four_models.csv",
         "results/experimental_protocol.csv", "results/traceability_matrix.csv",
         "config/environment_detailed.json"],
        "Audit-only stage derived from corrected checkpoints; no expensive experiment was refitted.",
    )
    log("Thesis notebook support artifacts complete")


if __name__ == "__main__":
    stage_thesis_notebook_support()
