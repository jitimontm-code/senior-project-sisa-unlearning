"""Absolute-final scalability extension and consistency-only artifact derivations.

The only new models trained here are LightGBM SISA shards for S=32,64,128.
Existing S=2,4,8,16 models/results are reused.  Other functions add derived
columns or summaries without changing validated predictions or model settings.
"""
from __future__ import annotations

import gc
import time

import numpy as np
import pandas as pd

from final_pipeline import (
    CFG, DIRS, atomic_csv, calibrate, choose_threshold, fit_platt, load_matrix,
    metrics, positive_probability, update_manifest, log,
)
from expanded_experiments import (
    S_VALUES, _fit_or_load_shards, get_stratified_assignment, ks_statistic,
)


EXTENDED_S_VALUES = [2, 4, 8, 16, 32, 64, 128]
NEW_S_VALUES = [32, 64, 128]


def _component_predictions(models, X) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    component = np.stack([positive_probability(model, X) for model in models], axis=1)
    return component, time.perf_counter() - started


def _timed_evaluation(models, sizes) -> tuple[dict, dict]:
    """Evaluate the frozen protocol and separate prediction from pure aggregation."""
    Xc, yc, _ = load_matrix("calibration")
    Xv, yv, _ = load_matrix("validation")
    Xt, yt, _ = load_matrix("test")
    weights = np.asarray(sizes, dtype=float) / np.sum(sizes)
    prediction_seconds = 0.0
    aggregation_seconds = 0.0

    pc, elapsed = _component_predictions(models, Xc); prediction_seconds += elapsed
    started = time.perf_counter(); raw_c = pc @ weights; aggregation_seconds += time.perf_counter() - started
    calibration = fit_platt(raw_c, yc)
    del pc, raw_c

    pv_component, elapsed = _component_predictions(models, Xv); prediction_seconds += elapsed
    started = time.perf_counter(); raw_v = pv_component @ weights; aggregation_seconds += time.perf_counter() - started
    pv = calibrate(calibration, raw_v)
    threshold = choose_threshold(yv, pv)
    del pv_component, raw_v, pv

    pt_component, elapsed = _component_predictions(models, Xt); prediction_seconds += elapsed
    started = time.perf_counter(); raw_t = pt_component @ weights; aggregation_seconds += time.perf_counter() - started
    pt = calibrate(calibration, raw_t)
    result = metrics(yt, pt, threshold)
    result["ks"] = ks_statistic(yt, pt)
    timing = {"component_prediction_seconds": prediction_seconds,
              "weighted_aggregation_seconds": aggregation_seconds}
    del Xc, Xv, Xt, pt_component, raw_t, pt
    gc.collect()
    return result, timing


def _workload_metrics(assignment: pd.DataFrame, deleted_rows: np.ndarray) -> dict:
    deleted_rows = np.unique(np.asarray(deleted_rows, dtype=int))
    affected = np.sort(assignment.loc[assignment.row_index.isin(deleted_rows), "shard"].unique())
    shard_sizes = assignment.groupby("shard").size()
    affected_original_rows = int(shard_sizes.loc[affected].sum())
    n_original = len(assignment)
    n_deleted = len(deleted_rows)
    actual_retrained_rows = affected_original_rows - n_deleted
    return {
        "deleted_records": n_deleted,
        "affected_shards": len(affected),
        "affected_shard_fraction": len(affected) / assignment.shard.nunique(),
        "affected_data_fraction": affected_original_rows / n_original,
        "actual_retrained_fraction": actual_retrained_rows / (n_original - n_deleted),
    }


def stage_extended_shard_count() -> None:
    """Extend the corrected LightGBM sensitivity study through S=128."""
    _, ytr, idtr = load_matrix("train")
    existing = pd.read_csv(DIRS["results"] / "shard_count_performance.csv").set_index("shards")
    rng = np.random.default_rng(CFG.seed + 3200)
    distributed_rows = np.sort(rng.choice(np.arange(len(idtr)),
                                          max(1, round(len(idtr) * .001)), replace=False))
    request_rows = [{"request": "distributed_0.1%", "shards": "all",
                     "row_index": int(row), "SK_ID_CURR": int(idtr[row])}
                    for row in distributed_rows]
    completed = []

    for n_shards in EXTENDED_S_VALUES:
        checkpoint = DIRS["results"] / f"shard_count_scalability_S{n_shards}.csv"
        if checkpoint.exists():
            completed.append(pd.read_csv(checkpoint).iloc[0].to_dict())
            continue

        assignment = get_stratified_assignment(ytr, idtr, n_shards)
        models, sizes, fit_times = _fit_or_load_shards(
            "LightGBM", n_shards, assignment, strategy="stratified")
        evaluated, timing = _timed_evaluation(models, sizes)

        # Reuse authoritative validated utility for S<=16; assert fresh evaluation compatibility.
        if n_shards in existing.index:
            prior = existing.loc[n_shards]
            assert abs(evaluated["roc_auc"] - prior.roc_auc) < 1e-12
            assert abs(evaluated["average_precision"] - prior.average_precision) < 1e-12
            assert abs(evaluated["brier"] - prior.brier) < 1e-12
            utility = {"roc_auc": prior.roc_auc, "average_precision": prior.average_precision,
                       "ks": prior.ks, "brier": prior.brier}
        else:
            utility = {name: evaluated[name] for name in ["roc_auc", "average_precision", "ks", "brier"]}

        anchor_row = 0
        anchor_shard = int(assignment.loc[assignment.row_index.eq(anchor_row), "shard"].iloc[0])
        anchor_pool = assignment.loc[assignment.shard.eq(anchor_shard), "row_index"].to_numpy(dtype=int)
        localized_rows = np.sort(anchor_pool[:max(1, round(len(idtr) * .001))])
        single = _workload_metrics(assignment, np.asarray([anchor_row]))
        localized = _workload_metrics(assignment, localized_rows)
        distributed = _workload_metrics(assignment, distributed_rows)
        request_rows.extend({"request": "assignment_localized_0.1%", "shards": n_shards,
                             "row_index": int(row), "SK_ID_CURR": int(idtr[row])}
                            for row in localized_rows)

        row = {
            "shards": n_shards, "mean_rows_per_shard": sizes.mean(),
            "min_rows_per_shard": sizes.min(), "max_rows_per_shard": sizes.max(),
            **utility, "sisa_training_seconds": fit_times.sum(),
            **timing,
            "single_record_affected_shard_fraction": single["affected_shard_fraction"],
            "single_record_affected_data_fraction": single["affected_data_fraction"],
            "single_record_actual_retrained_fraction": single["actual_retrained_fraction"],
            "localized_deleted_records": localized["deleted_records"],
            "localized_affected_shards": localized["affected_shards"],
            "localized_affected_data_fraction": localized["affected_data_fraction"],
            "localized_actual_retrained_fraction": localized["actual_retrained_fraction"],
            "distributed_deleted_records": distributed["deleted_records"],
            "distributed_affected_shards": distributed["affected_shards"],
            "distributed_affected_data_fraction": distributed["affected_data_fraction"],
            "distributed_actual_retrained_fraction": distributed["actual_retrained_fraction"],
            "utility_selection_role": "sensitivity analysis only; no test-based optimal-S selection",
        }
        atomic_csv(pd.DataFrame([row]), checkpoint)
        completed.append(row)
        log(f"Checkpointed LightGBM shard-count scalability S={n_shards}: AUC={row['roc_auc']:.4f}")
        del models
        gc.collect()

    result = pd.DataFrame(completed).sort_values("shards").reset_index(drop=True)
    assert result.shards.tolist() == EXTENDED_S_VALUES
    atomic_csv(result, DIRS["results"] / "shard_count_scalability_extended.csv")
    pd.DataFrame(request_rows).to_csv(DIRS["results"] / "shard_count_scalability_requests.csv.gz",
                                      index=False, compression="gzip")
    _scalability_figures(result)
    update_manifest("shard_count_scalability_extended", "complete", [
        "results/shard_count_scalability_extended.csv",
        "results/shard_count_scalability_S32.csv",
        "results/shard_count_scalability_S64.csv",
        "results/shard_count_scalability_S128.csv",
        "figures/shard_count_vs_predictive_utility.png",
        "figures/shard_count_vs_deletion_workload.png"],
        "LightGBM-only sensitivity S=2..128; S<=16 reused; S=32/64/128 newly computed; no optimal-S selection.")


def _scalability_figures(result: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(result.shards, result.roc_auc, marker="o", label="ROC-AUC", color="#256d85")
    ax.plot(result.shards, result.average_precision, marker="s", label="Average Precision", color="#d97706")
    ax.set_xscale("log", base=2); ax.set_xticks(result.shards, result.shards)
    ax.set_xlabel("Number of shards (S)"); ax.set_ylabel("Final-test metric")
    ax.set_title("Shard Count vs Predictive Utility (LightGBM sensitivity)")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(DIRS["figures"] / "shard_count_vs_predictive_utility.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(result.shards, result.single_record_actual_retrained_fraction, marker="o",
            label="Single record: actual retrained fraction", color="#256d85")
    ax.plot(result.shards, result.localized_actual_retrained_fraction, marker="s",
            label="Assignment-localized 0.1%: actual retrained fraction", color="#16a34a")
    ax.plot(result.shards, result.distributed_actual_retrained_fraction, marker="^",
            label="Distributed 0.1%: actual retrained fraction", color="#d97706")
    ax.set_xscale("log", base=2); ax.set_xticks(result.shards, result.shards)
    ax.set_xlabel("Number of shards (S)"); ax.set_ylabel("Actual retrained fraction")
    ax.set_title("Shard Count vs Deletion Workload")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(DIRS["figures"] / "shard_count_vs_deletion_workload.png", dpi=300); plt.close(fig)


def complete_baseline_metrics() -> None:
    baseline = pd.read_csv(DIRS["results"] / "baseline_performance.csv")
    ks_values = []
    for model in baseline.model:
        slug = model.lower().replace(" ", "_")
        pred = pd.read_csv(DIRS["results"] / f"predictions_{slug}_{CFG.hash}.csv.gz")
        test = pred[pred.split.eq("test")]
        ks_values.append(ks_statistic(test.y.to_numpy(), test.probability.to_numpy()))
    baseline["ks"] = ks_values
    columns = ["model", "roc_auc", "average_precision", "ks", "brier", "threshold",
               "f1", "sensitivity", "specificity", "fit_seconds", "log_loss"]
    atomic_csv(baseline[columns], DIRS["results"] / "baseline_performance_complete.csv")


def standardize_workload_artifacts() -> None:
    n_train = load_matrix("train")[0].shape[0]
    path = DIRS["results"] / "deletion_scale_locality.csv"
    deletion = pd.read_csv(path)
    deletion["affected_data_fraction"] = deletion["retraining_workload_fraction"]
    affected_rows = np.rint(deletion.affected_data_fraction * n_train)
    deletion["actual_retrained_fraction"] = ((affected_rows - deletion.deleted_records) /
                                               (n_train - deletion.deleted_records))
    deletion["legacy_workload_definition"] = "retraining_workload_fraction == affected_data_fraction"
    atomic_csv(deletion, path)
    summary = deletion.groupby(["deletion_size", "pattern", "strategy"], as_index=False).agg(
        affected_shards_mean=("affected_shards", "mean"),
        affected_data_fraction_mean=("affected_data_fraction", "mean"),
        affected_data_fraction_sd=("affected_data_fraction", "std"),
        actual_retrained_fraction_mean=("actual_retrained_fraction", "mean"),
        actual_retrained_fraction_sd=("actual_retrained_fraction", "std"))
    atomic_csv(summary, DIRS["results"] / "deletion_scale_locality_summary_explicit.csv")

    policy_path = DIRS["results"] / "policy_deletion_locality_results.csv"
    policy = pd.read_csv(policy_path)
    policy["affected_data_fraction"] = ((policy.retrained_rows + policy.deleted_customers) / n_train)
    policy["actual_retrained_fraction"] = policy["retrained_data_fraction"]
    policy["legacy_workload_definition"] = "retrained_data_fraction == actual_retrained_fraction"
    atomic_csv(policy, policy_path)
    policy_summary = policy.groupby("assignment", as_index=False).agg(
        scenarios=("scenario", "nunique"), affected_shards_mean=("affected_shards", "mean"),
        affected_data_fraction_mean=("affected_data_fraction", "mean"),
        actual_retrained_fraction_mean=("actual_retrained_fraction", "mean"),
        actual_retrained_fraction_sd=("actual_retrained_fraction", "std"),
        locality_mean=("locality_score", "mean"))
    atomic_csv(policy_summary, DIRS["results"] / "policy_deletion_locality_summary.csv")

    pivot = policy.pivot(index=["scenario", "relationship", "policy", "deleted_customers"],
                          columns="assignment",
                          values=["affected_shards", "affected_data_fraction",
                                  "actual_retrained_fraction"]).reset_index()
    pivot.columns = ["_".join(str(x) for x in col if str(x)) if isinstance(col, tuple) else col
                     for col in pivot.columns]
    rename = {
        "affected_shards_Specialist baseline": "specialist_affected_shards",
        "affected_shards_Specialist + policy-aware nested mapping": "policy_aware_affected_shards",
        "affected_data_fraction_Specialist baseline": "specialist_affected_data_fraction",
        "affected_data_fraction_Specialist + policy-aware nested mapping": "policy_aware_affected_data_fraction",
        "actual_retrained_fraction_Specialist baseline": "specialist_actual_retrained_fraction",
        "actual_retrained_fraction_Specialist + policy-aware nested mapping": "policy_aware_actual_retrained_fraction",
    }
    pivot = pivot.rename(columns=rename)
    pivot["workload_improved"] = np.where(
        pivot.policy_aware_actual_retrained_fraction < pivot.specialist_actual_retrained_fraction,
        "yes", "no")
    pivot["estimated_speedup_vs_specialist"] = (pivot.specialist_actual_retrained_fraction /
                                                  pivot.policy_aware_actual_retrained_fraction)
    atomic_csv(pivot, DIRS["results"] / "policy_deletion_per_relationship.csv")

    exact_path = DIRS["results"] / "exact_unlearning_four_models.csv"
    exact = pd.read_csv(exact_path)
    assignment = pd.read_csv(DIRS["shards"] / f"stratified_S{CFG.shards}_{CFG.hash}.csv")
    request = pd.read_csv(DIRS["deletion_requests"] / f"phase2_standard_single_{CFG.hash}.csv").iloc[0]
    affected_size = int((assignment.shard == int(request.original_shard)).sum())
    exact["affected_data_fraction"] = affected_size / n_train
    exact["actual_retrained_fraction"] = ((affected_size - exact.deleted_records) /
                                            (n_train - exact.deleted_records))
    exact["legacy_workload_definition"] = "workload_fraction approximated affected_data_fraction"
    atomic_csv(exact, exact_path)


def update_legacy_scalability_status() -> None:
    path = DIRS["results"] / "legacy_to_final_coverage.csv"
    legacy = pd.read_csv(path)
    mask = legacy.legacy_research_component.eq("Shard-count study")
    legacy.loc[mask, "final_status"] = "restored as corrected focused LightGBM scalability analysis through S=128"
    legacy.loc[mask, "class"] = "B"
    legacy.loc[mask, "class_meaning"] = "replaced by corrected implementation"
    atomic_csv(legacy, path)


def update_final_findings_and_objective_evidence() -> None:
    findings = pd.read_csv(DIRS["results"] / "phase2_consolidated_findings.csv")
    additions = pd.DataFrame([
        {"research_question": "How does fine SISA partitioning affect utility, workload, and overhead?",
         "finding": "S=2,4,8,16,32,64,128 shows non-monotonic utility, decreasing localized retraining fraction, and increasing model/ensemble overhead.",
         "limitation": "LightGBM-only sensitivity analysis; final-test results are descriptive and are not used to select an optimal S."},
        {"research_question": "Can an operational relationship layer configure deletion locality?",
         "finding": "The same 10 strict synthetic relationship requests show mixed per-scenario benefit; mean actual retrained fraction falls, with utility and complexity trade-offs.",
         "limitation": "Synthetic, intentionally optimized case study; not real partnerships or a universal deployment recommendation."},
    ])
    for column in findings.columns:
        if column not in additions:
            additions[column] = ""
    additions = additions[findings.columns]
    atomic_csv(pd.concat([findings, additions], ignore_index=True),
               DIRS["results"] / "final_consolidated_findings.csv")

    objective = pd.read_csv(DIRS["results"] / "objective_coverage.csv")
    evidence = {
        1: "4.2 Sharding Strategy Results; sisa_four_model_comparison.csv; shard_count_scalability_extended.csv",
        2: "4.4–4.6 Fidelity, unlearning, and computational efficiency artifacts",
        3: "4.2–4.3 controlled sharding, paired locality, and supplementary policy-aware evidence",
        4: "4.7 Membership Inference Results; privacy_mia.csv",
        5: "4.8 Synthesis and Discussion; technical suitability and limitations",
    }
    objective["evidence"] = objective.objective.map(evidence)
    atomic_csv(objective, DIRS["results"] / "objective_coverage.csv")

    stats_path = DIRS["results"] / "deletion_locality_statistics.csv"
    stats = pd.read_csv(stats_path)
    stats["metric_definition"] = "paired difference in affected_data_fraction"
    atomic_csv(stats, stats_path)


def write_final_traceability() -> None:
    rows = [
        ("3.1 Dataset", "4.1 Baseline Model Performance", "identity-safe source/split audit", "dataset_table_summary.csv; split_integrity.csv", "phase2_class_distribution.png"),
        ("3.2 Data Preprocessing", "4.1 Baseline Model Performance", "train-only preprocessing audit", "preprocessing_audit.csv", "-"),
        ("3.3 Baseline Model Development", "4.1 Baseline Model Performance", "four fixed model families; frozen calibration/threshold", "baseline_model_config.csv; baseline_performance_complete.csv", "baseline_and_sisa_auc.png"),
        ("3.3.1 Logistic Regression", "4.1 Baseline Model Performance", "linear reference", "baseline_performance_complete.csv", "-"),
        ("3.3.2 Random Forest", "4.1 Baseline Model Performance", "bagged-tree reference", "baseline_performance_complete.csv", "-"),
        ("3.3.3 LightGBM", "4.1 Baseline Model Performance", "boosting reference and scalability representative", "baseline_performance_complete.csv", "-"),
        ("3.3.4 XGBoost", "4.1 Baseline Model Performance", "second boosting family", "baseline_performance_complete.csv", "-"),
        ("3.4 SISA Sharding Strategies", "4.2 Sharding Strategy Results", "fixed SISA architecture and assignment comparison", "sisa_four_model_comparison.csv", "phase2_sisa_four_models.png"),
        ("3.4.1 Stratified", "4.2 Sharding Strategy Results", "class-wise round robin", "sharding_contribution_thesis.csv", "phase2_sharding_utility_locality.png"),
        ("3.4.2 Representative", "4.2 Sharding Strategy Results", "SVD8/cluster representation", "sharding_contribution_thesis.csv", "phase2_sharding_utility_locality.png"),
        ("3.4.3 Specialist", "4.2–4.3 Sharding and Deletion Locality Results", "feature-cluster specialization", "sharding_contribution_thesis.csv; deletion_locality_statistics.csv", "phase2_sharding_utility_locality.png"),
        ("3.4.4 Risk-Band", "4.2–4.3 Sharding and Deletion Locality Results", "OOF-risk bands", "sharding_contribution_thesis.csv; deletion_locality_statistics.csv", "phase2_sharding_utility_locality.png"),
        ("Shard-Count Scalability / Sensitivity Analysis", "4.2 Sharding Strategy Results", "LightGBM S=2,4,8,16,32,64,128; no optimal-S selection", "shard_count_scalability_extended.csv", "shard_count_vs_predictive_utility.png; shard_count_vs_deletion_workload.png"),
        ("3.5 Model Aggregation", "4.2 Sharding Strategy Results", "controlled weighted and practical learned aggregation", "aggregation_coefficients.csv; practical_sharding_architectures.csv", "-"),
        ("3.5.1 Simple Averaging", "4.2 Sharding Strategy Results", "retained-size weighted + Platt", "expanded_controlled_sharding.csv", "-"),
        ("3.5.2 Learned Aggregator", "4.2 Sharding Strategy Results", "logistic meta-model + held-out Platt", "aggregation_coefficients.csv", "-"),
        ("3.6 Deletion Locality Experimental Design", "4.3 Deletion Locality Results", "distributed/feature/risk-localized paired requests", "deletion_scale_locality_summary_explicit.csv", "phase2_deletion_scale.png"),
        ("3.6.1 Supplementary Policy-Aware Relationship Deletion Scenario", "4.3.1 Supplementary Policy-Aware / Company-Aware Relationship Case Study", "synthetic mapping; strict policy; constrained nested Specialist", "policy_deletion_per_relationship.csv; policy_nested_utility.csv", "policy_nested_specialist_results.png"),
        ("3.7 Exact vs Approximate Unlearning", "4.5 Exact vs Approximate Unlearning", "exact affected-shard retraining; slicing; booster heuristic", "slicing_demonstration.csv; approximate_two_boosters.csv", "phase2_slicing_replay.png; phase2_approximate_two_boosters.png"),
        ("3.8 Gold-Standard Fidelity Evaluation", "4.4 Gold-Standard Fidelity Results", "protocol-compatible Gold", "exact_unlearning_four_models.csv", "phase2_exact_four_models.png"),
        ("3.8.1 Direct Deletion Audit", "4.4.1 Direct Deletion Audit", "retained-row identity assertion", "direct_deletion_audit.csv", "-"),
        ("3.8.2 Model Fidelity", "4.4.2 Model Fidelity", "MAE/RMSE/correlation/AUC/AP/disagreement", "fidelity_detailed_four_models.csv", "-"),
        ("3.8.3 Forget-set Behavior", "4.4.3 Forget-set Behavior", "Original/Exact/Gold on deleted applicant", "forget_set_behavior_four_models.csv", "-"),
        ("3.8.4 Cross-model Validation", "4.4.4 Cross-model Validation", "four-family standardized request", "exact_unlearning_four_models.csv", "-"),
        ("3.9 Membership Inference Attack", "4.7 Membership Inference Results", "verified-member loss attack", "privacy_mia.csv", "-"),
        ("3.10 Deletion Workload and Computational Cost", "4.6 Deletion Workload and Computational Efficiency", "affected-data vs actual-retrained fractions; measured runtime", "phase2_runtime_summary.csv; policy_deletion_runtime.csv", "shard_count_vs_deletion_workload.png"),
        ("3.11 Experimental Protocol and Evaluation Criteria", "4.8 Synthesis and Discussion", "fixed data roles/seeds/repetition scope", "experimental_protocol.csv", "-"),
        ("3.12 Tools and Experimental Environment", "4.8 Synthesis and Discussion", "environment/manifest", "environment_detailed.json; experiment_manifest.json", "-"),
        ("4.8.1 Utility–Fidelity–Privacy–Cost Trade-off", "4.8 Synthesis and Discussion", "artifact-backed synthesis", "phase2_consolidated_findings.csv", "-"),
        ("4.8.2 Model-specific Sensitivity", "4.8 Synthesis and Discussion", "four-family and large-S evidence", "sisa_four_model_comparison.csv; shard_count_scalability_extended.csv", "-"),
        ("4.8.3 Sharding Strategy Selection Guideline", "4.8 Synthesis and Discussion", "conditional strategy guidance", "sharding_contribution_thesis.csv", "-"),
        ("4.8.4 Unlearning Method Selection Guideline", "4.8 Synthesis and Discussion", "exact/slicing/approximate scope", "exact_unlearning_four_models.csv; approximate_two_boosters.csv", "-"),
        ("4.8.5 Robustness Across Scenarios and Replications", "4.8 Synthesis and Discussion", "paired n=10 vs descriptive scenarios", "deletion_locality_statistics.csv", "-"),
        ("4.8.6 Practical Decision Framework", "4.8 Synthesis and Discussion", "evidence-driven non-universal guidance", "policy_extension_contribution.csv; objective_coverage.csv", "-"),
    ]
    trace = pd.DataFrame(rows, columns=["thesis_section", "chapter_4_evidence",
                                       "code_or_method", "artifact", "figure"])
    objective = pd.read_csv(DIRS["results"] / "objective_coverage.csv")
    for item in objective.itertuples():
        trace.loc[len(trace)] = [f"Objective {int(item.objective)}", item.evidence,
                                 f"status={item.status}; {item.qualification}",
                                 "objective_coverage.csv", "see mapped result sections"]
    atomic_csv(trace, DIRS["results"] / "traceability_matrix.csv")


def refresh_explicit_workload_figures() -> None:
    import matplotlib.pyplot as plt
    deletion = pd.read_csv(DIRS["results"] / "deletion_scale_locality.csv")
    scale = deletion[(deletion.strategy == "stratified_balanced") &
                     (deletion.pattern == "distributed_random")]
    order = ["single", "0.01%", "0.1%", "1%"]
    summary = scale.groupby("deletion_size", as_index=False).agg(
        affected_data_fraction=("affected_data_fraction", "mean"),
        actual_retrained_fraction=("actual_retrained_fraction", "mean"))
    summary["deletion_size"] = pd.Categorical(summary.deletion_size, order, ordered=True)
    summary = summary.sort_values("deletion_size")
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    ax.plot(summary.deletion_size.astype(str), summary.affected_data_fraction,
            marker="o", label="Affected data fraction", color="#256d85")
    ax.plot(summary.deletion_size.astype(str), summary.actual_retrained_fraction,
            marker="s", linestyle="--", label="Actual retrained fraction", color="#d97706")
    ax.set_ylim(0, 1.05); ax.set_xlabel("Distributed random deletion size")
    ax.set_ylabel("Fraction of training data"); ax.set_title("Deletion scale and explicit workload definitions")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(DIRS["figures"] / "phase2_deletion_scale.png", dpi=300); plt.close(fig)


def run_absolute_finalization() -> None:
    stage_extended_shard_count()
    complete_baseline_metrics()
    standardize_workload_artifacts()
    update_legacy_scalability_status()
    update_final_findings_and_objective_evidence()
    write_final_traceability()
    refresh_explicit_workload_figures()
    log("Absolute-final scalability and consistency artifacts complete")


if __name__ == "__main__":
    run_absolute_finalization()
