"""Corrected supplementary policy-aware nested Specialist case study.

This is deliberately not a fifth primary sharding strategy.  It starts from the
validated Specialist assignment and adds a synthetic organization-defined
relationship layer.  All identity alignment uses the corrected train universe.
"""
from __future__ import annotations

import gc
import time
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from final_pipeline import CFG, DIRS, ROOT, atomic_csv, load_matrix, log, make_model, update_manifest
from expanded_experiments import _evaluate_strategy, _strategy_assignments


POLICY_SEED = CFG.seed + 900
POLICY_SCHEMA_VERSION = "policy-aware-nested-specialist-v2"
SUBGROUPS_PER_ORG_TYPE = 12
MULTI_RELATIONSHIP_FRACTION = 0.18
BALANCE_TOLERANCE = 0.05
MAX_MOVED_FRACTION = 0.10
N_RELATIONSHIP_SCENARIOS = 10
N_RUNTIME_SCENARIOS = 3
RUNTIME_REPEATS = 3


def build_synthetic_relationship_mapping() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create identity-safe synthetic relationships for corrected train applicants.

    ORGANIZATION_TYPE supplies a coarse source category.  Within sufficiently
    large categories, an identity-stable hash creates up to 12 synthetic company
    groups.  Eighteen percent of applicants receive one additional synthetic
    relationship.  No final-test data or target values enter this construction.
    """
    _, _, train_ids = load_matrix("train")
    raw = pd.read_csv(ROOT / "application_train.csv",
                      usecols=["SK_ID_CURR", "ORGANIZATION_TYPE"])
    assert raw.SK_ID_CURR.is_unique
    aligned = pd.DataFrame({"row_index": np.arange(len(train_ids)),
                            "SK_ID_CURR": train_ids}).merge(
        raw, on="SK_ID_CURR", how="left", validate="one_to_one", sort=False)
    assert np.array_equal(aligned.SK_ID_CURR.to_numpy(), train_ids)
    assert aligned.ORGANIZATION_TYPE.notna().all()

    org = aligned.ORGANIZATION_TYPE.astype(str).to_numpy()
    org_counts = pd.Series(org).value_counts()
    use_subgroups = pd.Series(org).map(org_counts).to_numpy() >= 800
    hashed = ((train_ids.astype(np.uint64) * np.uint64(11400714819323198485)
               + np.uint64(POLICY_SEED)) % np.uint64(SUBGROUPS_PER_ORG_TYPE)).astype(int)
    subgroup = np.where(use_subgroups, hashed, 0)
    primary = np.asarray([f"{name}__SyntheticCompany_{part + 1:02d}"
                          for name, part in zip(org, subgroup)], dtype=object)
    companies = np.unique(primary)
    company_index = {company: index for index, company in enumerate(companies)}

    rng = np.random.default_rng(POLICY_SEED)
    has_secondary = rng.random(len(train_ids)) < MULTI_RELATIONSHIP_FRACTION
    secondary = np.full(len(train_ids), None, dtype=object)
    selected = np.flatnonzero(has_secondary)
    draws = rng.integers(0, len(companies) - 1, size=len(selected))
    for row, draw in zip(selected, draws):
        own = company_index[primary[row]]
        other = int(draw + (draw >= own))
        secondary[row] = companies[other]

    primary_frame = aligned[["row_index", "SK_ID_CURR", "ORGANIZATION_TYPE"]].copy()
    primary_frame["relationship"] = primary
    primary_frame["relationship_order"] = 1
    primary_frame["is_primary"] = True
    primary_frame["relationship_count"] = 1 + has_secondary.astype(int)
    primary_frame["strict_delete_eligible"] = ~has_secondary
    secondary_frame = aligned.loc[has_secondary, ["row_index", "SK_ID_CURR", "ORGANIZATION_TYPE"]].copy()
    secondary_frame["relationship"] = secondary[has_secondary]
    secondary_frame["relationship_order"] = 2
    secondary_frame["is_primary"] = False
    secondary_frame["relationship_count"] = 2
    secondary_frame["strict_delete_eligible"] = False
    mapping = pd.concat([primary_frame, secondary_frame], ignore_index=True).sort_values(
        ["row_index", "relationship_order"], kind="stable").reset_index(drop=True)

    # Identity and order assertions required by the corrected SISA universe.
    first = mapping[mapping.relationship_order.eq(1)].sort_values("row_index")
    assert len(first) == len(train_ids)
    assert np.array_equal(first.SK_ID_CURR.to_numpy(), train_ids)
    assert mapping.SK_ID_CURR.isin(set(train_ids)).all()
    assert mapping.groupby("row_index").SK_ID_CURR.nunique().eq(1).all()
    assert mapping.groupby("row_index").size().to_numpy().shape[0] == len(train_ids)
    assert np.array_equal(mapping.groupby("row_index").size().to_numpy(), 1 + has_secondary.astype(int))

    company_size = mapping.groupby("relationship").agg(
        customers=("row_index", "nunique"),
        strict_eligible_customers=("strict_delete_eligible", "sum"),
        source_organization_types=("ORGANIZATION_TYPE", "nunique")).reset_index()
    summary = pd.DataFrame([
        {"measure": "corrected_sisa_train_customers", "value": len(train_ids), "note": "unique SK_ID_CURR in corrected train"},
        {"measure": "relationship_source", "value": "ORGANIZATION_TYPE + deterministic synthetic subgroup + optional secondary relationship", "note": "scenario simulation; not real partnerships"},
        {"measure": "mapping_success_rate", "value": 1.0, "note": "all corrected train IDs mapped"},
        {"measure": "organization_types", "value": len(np.unique(org)), "note": "raw application attribute"},
        {"measure": "synthetic_companies", "value": len(company_size), "note": "unique relationship labels"},
        {"measure": "single_relationship_customers", "value": int((~has_secondary).sum()), "note": f"proportion={(~has_secondary).mean():.6f}"},
        {"measure": "multiple_relationship_customers", "value": int(has_secondary.sum()), "note": f"proportion={has_secondary.mean():.6f}"},
        {"measure": "company_size_min", "value": int(company_size.customers.min()), "note": "includes primary and secondary memberships"},
        {"measure": "company_size_median", "value": float(company_size.customers.median()), "note": "includes primary and secondary memberships"},
        {"measure": "company_size_max", "value": int(company_size.customers.max()), "note": "includes primary and secondary memberships"},
    ])
    mapping.to_csv(DIRS["results"] / "policy_relationship_mapping.csv.gz", index=False, compression="gzip")
    atomic_csv(company_size, DIRS["results"] / "policy_relationship_distribution.csv")
    atomic_csv(summary, DIRS["results"] / "policy_relationship_summary.csv")
    return mapping, company_size


def strict_deletion_rows(mapping: pd.DataFrame, relationship: str) -> np.ndarray:
    """Strict policy deletes only customers whose sole relationship is removed."""
    rows = mapping.loc[(mapping.relationship == relationship) &
                       mapping.strict_delete_eligible, "row_index"].to_numpy(dtype=int)
    assert len(rows) and len(np.unique(rows)) == len(rows)
    return np.sort(rows)


def any_relationship_deletion_rows(mapping: pd.DataFrame, relationship: str) -> np.ndarray:
    """Explanatory alternative: delete every customer linked to the relationship."""
    return np.sort(mapping.loc[mapping.relationship == relationship,
                               "row_index"].unique().astype(int))


def nested_specialist_optimization(base_assignment: np.ndarray, mapping: pd.DataFrame,
                                   priority_relationships: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    """Concentrate strict-eligible relationships subject to hard constraints.

    Only single-relationship customers move.  Therefore a move cannot fragment a
    second relationship.  A complete source-shard block is moved only when the
    strict-eligible spread falls, full membership spread does not increase, and
    both source/target shard sizes remain within ±5% of the equal-size target.
    """
    n = len(base_assignment)
    n_shards = CFG.shards
    nested = base_assignment.copy()
    target_size = n / n_shards
    minimum = int(np.floor(target_size * (1 - BALANCE_TOLERANCE)))
    maximum = int(np.ceil(target_size * (1 + BALANCE_TOLERANCE)))
    load = np.bincount(nested, minlength=n_shards).astype(int)
    maximum_moved = int(np.floor(n * MAX_MOVED_FRACTION))
    moved_total = 0

    full_rows = {name: group.row_index.unique().astype(int)
                 for name, group in mapping.groupby("relationship")}
    eligible_rows = {name: strict_deletion_rows(mapping, name)
                     for name in mapping.loc[mapping.strict_delete_eligible, "relationship"].unique()}
    priority = {name: rank for rank, name in enumerate(priority_relationships)}
    ordered = sorted(eligible_rows, key=lambda name: (
        priority.get(name, len(priority)),
        -len(np.unique(base_assignment[eligible_rows[name]])),
        -len(eligible_rows[name]), str(name)))

    audit = []
    for relationship in ordered:
        eligible = eligible_rows[relationship]
        full = full_rows[relationship]
        base_full_spread = len(np.unique(base_assignment[full]))
        base_eligible_spread = len(np.unique(base_assignment[eligible]))
        before_moves = int(np.sum(nested != base_assignment))
        for _ in range(n_shards - 1):
            counts = np.bincount(nested[eligible], minlength=n_shards)
            present = np.flatnonzero(counts)
            if len(present) <= 1:
                break
            target = int(present[np.argmax(counts[present])])
            changed = False
            for source in present[np.argsort(counts[present])]:
                source = int(source)
                if source == target:
                    continue
                moving = eligible[nested[eligible] == source]
                if moved_total + len(moving) > maximum_moved:
                    continue
                if load[target] + len(moving) > maximum or load[source] - len(moving) < minimum:
                    continue
                old_full_spread = len(np.unique(nested[full]))
                old_eligible_spread = len(np.unique(nested[eligible]))
                nested[moving] = target
                new_full_spread = len(np.unique(nested[full]))
                new_eligible_spread = len(np.unique(nested[eligible]))
                if new_eligible_spread < old_eligible_spread and new_full_spread <= old_full_spread:
                    load[source] -= len(moving)
                    load[target] += len(moving)
                    moved_total += len(moving)
                    changed = True
                    break
                nested[moving] = source
            if not changed:
                break
        audit.append({
            "relationship": relationship, "customers": len(full),
            "strict_eligible_customers": len(eligible),
            "full_spread_before": base_full_spread,
            "full_spread_after": len(np.unique(nested[full])),
            "strict_spread_before": base_eligible_spread,
            "strict_spread_after": len(np.unique(nested[eligible])),
            "customers_moved_for_relationship": int(np.sum(nested != base_assignment)) - before_moves,
            "priority_scenario": relationship in priority,
        })

    audit_frame = pd.DataFrame(audit)
    assert len(nested) == n and np.all((nested >= 0) & (nested < n_shards))
    assert np.bincount(nested, minlength=n_shards).sum() == n
    assert np.all(np.bincount(nested, minlength=n_shards) >= minimum)
    assert np.all(np.bincount(nested, minlength=n_shards) <= maximum)
    assert audit_frame.full_spread_after.le(audit_frame.full_spread_before).all()
    assert audit_frame.strict_spread_after.le(audit_frame.strict_spread_before).all()
    return nested, audit_frame


def _relationship_scenarios(mapping: pd.DataFrame, base_assignment: np.ndarray) -> list[str]:
    eligible = mapping[mapping.strict_delete_eligible].groupby("relationship").row_index.apply(
        lambda values: np.asarray(values, dtype=int))
    candidates = []
    for name, rows in eligible.items():
        if len(rows) >= 200 and len(np.unique(base_assignment[rows])) > 1:
            candidates.append((name, len(rows), len(np.unique(base_assignment[rows]))))
    candidates.sort(key=lambda item: (-item[2], -item[1], str(item[0])))
    assert len(candidates) >= N_RELATIONSHIP_SCENARIOS
    return [item[0] for item in candidates[:N_RELATIONSHIP_SCENARIOS]]


def _locality_results(mapping: pd.DataFrame, base: np.ndarray, nested: np.ndarray,
                      relationships: list[str]) -> pd.DataFrame:
    rows = []
    for scenario, relationship in enumerate(relationships):
        deleted = strict_deletion_rows(mapping, relationship)
        for label, assignment in [("Specialist baseline", base),
                                  ("Specialist + policy-aware nested mapping", nested)]:
            affected = np.unique(assignment[deleted])
            shard_sizes = np.bincount(assignment, minlength=CFG.shards)
            retrained = int(shard_sizes[affected].sum() - len(deleted))
            rows.append({
                "scenario": scenario, "relationship": relationship,
                "policy": "strict", "assignment": label,
                "deleted_customers": len(deleted), "affected_shards": len(affected),
                "affected_shard_fraction": len(affected) / CFG.shards,
                "retrained_rows": retrained,
                "retrained_data_fraction": retrained / (len(assignment) - len(deleted)),
                "locality_score": 1 - (len(affected) - 1) / (CFG.shards - 1),
                "estimated_speedup_from_workload": (len(assignment) - len(deleted)) / max(retrained, 1),
            })
    result = pd.DataFrame(rows)
    atomic_csv(result, DIRS["results"] / "policy_deletion_locality_results.csv")
    summary = result.groupby("assignment", as_index=False).agg(
        scenarios=("scenario", "nunique"), affected_shards_mean=("affected_shards", "mean"),
        affected_shards_sd=("affected_shards", "std"),
        retrained_fraction_mean=("retrained_data_fraction", "mean"),
        retrained_fraction_sd=("retrained_data_fraction", "std"),
        locality_mean=("locality_score", "mean"),
        estimated_speedup_median=("estimated_speedup_from_workload", "median"))
    atomic_csv(summary, DIRS["results"] / "policy_deletion_locality_summary.csv")
    request_rows = []
    _, _, ids = load_matrix("train")
    for scenario, relationship in enumerate(relationships):
        deleted = strict_deletion_rows(mapping, relationship)
        request_rows.extend({"scenario": scenario, "relationship": relationship,
                             "policy": "strict", "row_index": int(row),
                             "SK_ID_CURR": int(ids[row])} for row in deleted)
    pd.DataFrame(request_rows).to_csv(DIRS["results"] / "policy_deletion_requests.csv.gz",
                                      index=False, compression="gzip")
    return result


def _runtime_benchmark(mapping: pd.DataFrame, base: np.ndarray, nested: np.ndarray,
                       relationships: list[str]) -> None:
    X, y, _ = load_matrix("train")
    rows = []
    for scenario, relationship in enumerate(relationships[:N_RUNTIME_SCENARIOS]):
        deleted = strict_deletion_rows(mapping, relationship)
        for repeat in range(RUNTIME_REPEATS):
            for label, assignment in [("Specialist baseline", base),
                                      ("Specialist + policy-aware nested mapping", nested)]:
                affected = np.unique(assignment[deleted])
                started = time.perf_counter()
                for shard in affected:
                    train_rows = np.flatnonzero(assignment == shard)
                    train_rows = train_rows[~np.isin(train_rows, deleted)]
                    model = make_model("LightGBM", POLICY_SEED + repeat * 100 + int(shard))
                    model.fit(X[train_rows], y[train_rows])
                    del model
                seconds = time.perf_counter() - started
                rows.append({"scenario": scenario, "relationship": relationship,
                             "repeat": repeat, "assignment": label,
                             "affected_shards": len(affected),
                             "retraining_seconds": seconds})
                gc.collect()
    runtime = pd.DataFrame(rows)
    atomic_csv(runtime, DIRS["results"] / "policy_deletion_runtime.csv")
    summary = runtime.groupby("assignment", as_index=False).agg(
        measurements=("retraining_seconds", "size"),
        mean_seconds=("retraining_seconds", "mean"),
        sd_seconds=("retraining_seconds", "std"),
        median_seconds=("retraining_seconds", "median"))
    atomic_csv(summary, DIRS["results"] / "policy_deletion_runtime_summary.csv")


def _figures(mapping: pd.DataFrame, balance: pd.DataFrame, locality: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    relationship_count = mapping.drop_duplicates("row_index").relationship_count.value_counts().sort_index()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].bar(relationship_count.index.astype(str), relationship_count.values, color="#256d85")
    axes[0].set_xlabel("Synthetic relationships per customer")
    axes[0].set_ylabel("Corrected train customers")
    axes[0].set_title("Policy relationship multiplicity")
    company_sizes = mapping.groupby("relationship").row_index.nunique()
    axes[1].hist(company_sizes, bins=25, color="#d97706", edgecolor="white")
    axes[1].set_xlabel("Customers per synthetic relationship")
    axes[1].set_ylabel("Relationship count")
    axes[1].set_title("Synthetic relationship-size distribution")
    fig.tight_layout(); fig.savefig(DIRS["figures"] / "policy_relationship_mapping.png", dpi=300); plt.close(fig)

    summary = locality.groupby("assignment").agg(affected=("affected_shards", "mean"),
                                                  workload=("retrained_data_fraction", "mean"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    x = np.arange(CFG.shards); width = .36
    axes[0].bar(x-width/2, balance.base_size, width, label="Specialist", color="#64748b")
    axes[0].bar(x+width/2, balance.nested_size, width, label="Nested", color="#256d85")
    axes[0].set_xticks(x); axes[0].set_xlabel("Shard"); axes[0].set_ylabel("Train records")
    axes[0].set_title("Shard balance before/after"); axes[0].legend()
    labels = ["Specialist", "Policy-aware nested"]
    axes[1].bar(labels, summary.affected.to_numpy(), color=["#64748b", "#16a34a"])
    axes[1].set_ylabel("Mean affected shards"); axes[1].set_ylim(0, CFG.shards + .2)
    axes[1].set_title("Same strict relationship deletions")
    for ax in axes: ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(DIRS["figures"] / "policy_nested_specialist_results.png", dpi=300); plt.close(fig)


def _legacy_coverage_inventory() -> None:
    rows = [
        ("Legacy identity/data pipeline", "B", "replaced by corrected SK_ID_CURR-aligned pipeline and assertions"),
        ("Legacy split/calibration workflow", "B", "replaced by isolated train/validation/calibration/final-test roles"),
        ("LR/RF/LightGBM/XGBoost baselines", "B", "recomputed under corrected common protocol"),
        ("Legacy SISA models", "B", "recomputed with compatible S=4 comparison across four families"),
        ("Shard-count study", "A", "represented as corrected LightGBM S=2,4,8,16 sweep"),
        ("Stratified/Representative/Specialist/Risk-Band", "B", "retained under controlled assignment-only comparison"),
        ("Random-balanced", "A", "retained as control, not a fifth research design"),
        ("Simple and learned aggregation", "B", "controlled and practical layers separated"),
        ("Deletion scale/locality", "B", "replaced by paired distributed/feature/risk-localized requests"),
        ("Exact SISA vs Gold and forget behavior", "B", "revalidated with compatible seeds, assignments and direct audit"),
        ("Legacy slicing", "B", "replaced by deterministic incremental LR R=1/R=4 demonstration"),
        ("Approximate unlearning", "B", "restricted to supported continued boosting for LGBM/XGBoost"),
        ("Legacy MIA", "B", "replaced by verified-member conservative focused attack"),
        ("Runtime/workload/statistics", "B", "replaced by measured runtime and paired locality evidence"),
        ("ORG-1B→ORG-4B company-aware chain", "D", "restored as corrected supplementary synthetic policy-aware nested Specialist case study"),
        ("cluster/partner/vintage/adverse-action branches", "E", "not restored; exploratory proxy labels and/or confounded comparisons"),
        ("duplicate cells, hidden state, test-leaking branches", "C", "intentionally obsolete/invalid and excluded"),
    ]
    frame = pd.DataFrame(rows, columns=["legacy_research_component", "class", "final_status"])
    frame["class_meaning"] = frame["class"].map({
        "A": "already correctly represented", "B": "replaced by corrected implementation",
        "C": "intentionally obsolete/invalid", "D": "supplementary and scientifically meaningful",
        "E": "exploratory and not worth restoring"})
    atomic_csv(frame, DIRS["results"] / "legacy_to_final_coverage.csv")


def _update_traceability() -> None:
    path = DIRS["results"] / "traceability_matrix.csv"
    trace = pd.read_csv(path)
    trace = trace[~trace.thesis_section.astype(str).str.startswith(("3.6.1 Policy", "4.3.1 Policy", "4.8.6 Policy", "Objective "))]
    additions = [
        {"thesis_section": "3.6.1 Policy-aware relationship scenario",
         "notebook_heading": "Supplementary Policy-Aware Relationship Deletion Scenario",
         "code_or_method": "identity-safe synthetic mapping; strict eligibility; constrained nested Specialist optimization",
         "artifact": "policy_relationship_summary.csv; policy_nested_summary.csv",
         "figure": "policy_relationship_mapping.png"},
        {"thesis_section": "4.3.1 Policy-aware case-study result",
         "notebook_heading": "Policy-Aware / Company-Aware Relationship Case Study",
         "code_or_method": "same strict requests on Specialist vs nested assignment",
         "artifact": "policy_deletion_locality_results.csv; policy_nested_utility.csv; policy_deletion_runtime.csv",
         "figure": "policy_nested_specialist_results.png"},
        {"thesis_section": "4.8.6 Policy configurability",
         "notebook_heading": "Practical Decision Framework",
         "code_or_method": "separate statistical sharding and operational locality axes",
         "artifact": "policy_extension_contribution.csv",
         "figure": "-"},
    ]
    objective = pd.read_csv(DIRS["results"] / "objective_coverage.csv")
    objective_headings = {
        1: "4.2 Sharding Strategy Results", 2: "4.4–4.6 Fidelity/unlearning/cost",
        3: "4.2–4.3 Strategy and locality results", 4: "4.7 Membership Inference Results",
        5: "4.8 Synthesis and Discussion"}
    for row in objective.itertuples():
        additions.append({"thesis_section": f"Objective {int(row.objective)}",
                          "notebook_heading": objective_headings[int(row.objective)],
                          "code_or_method": f"status={row.status}; {row.qualification}",
                          "artifact": row.evidence, "figure": "see mapped result sections"})
    atomic_csv(pd.concat([trace, pd.DataFrame(additions)], ignore_index=True), path)


def _update_contribution_tables() -> None:
    controlled = pd.read_csv(DIRS["results"] / "expanded_controlled_sharding.csv").set_index("strategy")
    deletion = pd.read_csv(DIRS["results"] / "deletion_scale_locality_summary.csv")
    def workload(strategy: str, pattern: str) -> float:
        row = deletion[(deletion.deletion_size == "0.1%") &
                       (deletion.strategy == strategy) & (deletion.pattern == pattern)]
        return float(row.workload_mean.iloc[0])
    rows = [
        {"Strategy": "Stratified", "Role in study": "primary standard baseline",
         "Design principle": "balance size and target rate",
         "Actual assignment basis": "class-wise shuffled round robin",
         "Expected shard structure": "similar size and class prior",
         "Matched deletion pattern": "control for feature/risk locality",
         "Aggregation approach": "retained-size weighted + Platt",
         "Primary hypothesis": "stable utility and predictable 1/S single-delete workload",
         "Observed utility effect": f"controlled ROC-AUC={controlled.loc['stratified_balanced','roc_auc']:.6f}",
         "Observed locality/workload effect": f"0.1% feature-localized workload={workload('stratified_balanced','feature_localized'):.3f}; risk-localized={workload('stratified_balanced','risk_localized'):.3f}",
         "Main limitation": "does not intentionally localize feature/risk cohorts"},
        {"Strategy": "Representative", "Role in study": "primary project implementation/adaptation",
         "Design principle": "spread every feature cluster across shards",
         "Actual assignment basis": "train-only SVD8 + MiniBatchKMeans + cluster-wise round robin",
         "Expected shard structure": "population-like mixture per shard",
         "Matched deletion pattern": "none pre-specified",
         "Aggregation approach": "retained-size weighted + Platt",
         "Primary hypothesis": "preserve utility through representation",
         "Observed utility effect": f"controlled ROC-AUC={controlled.loc['representative','roc_auc']:.6f}",
         "Observed locality/workload effect": "no dedicated matched-locality advantage claimed",
         "Main limitation": "depends on embedding/clustering and does not target deletion locality"},
        {"Strategy": "Specialist", "Role in study": "primary project-specific experimental design",
         "Design principle": "concentrate feature-similar applicants",
         "Actual assignment basis": "train-only feature-cluster ordering split into equal contiguous shards",
         "Expected shard structure": "feature-specialized shards",
         "Matched deletion pattern": "feature-localized",
         "Aggregation approach": "weighted controlled; learned practical layer also tested",
         "Primary hypothesis": "reduce affected shards for feature-localized requests",
         "Observed utility effect": f"controlled ROC-AUC={controlled.loc['specialist','roc_auc']:.6f}",
         "Observed locality/workload effect": f"0.1% matched workload={workload('specialist','feature_localized'):.3f}",
         "Main limitation": "structure depends on feature representation; deployment complexity"},
        {"Strategy": "Risk-Band", "Role in study": "primary project-specific experimental design",
         "Design principle": "concentrate similar OOF risk",
         "Actual assignment basis": "3-fold train-only OOF LightGBM risk quantile bands",
         "Expected shard structure": "different risk/class priors by shard",
         "Matched deletion pattern": "risk-localized",
         "Aggregation approach": "weighted controlled; learned practical layer also tested",
         "Primary hypothesis": "reduce affected shards for risk-localized requests",
         "Observed utility effect": f"controlled ROC-AUC={controlled.loc['risk_band','roc_auc']:.6f}; learned aggregation partially recovers utility",
         "Observed locality/workload effect": f"0.1% matched workload={workload('risk_band','risk_localized'):.3f}",
         "Main limitation": "naive averaging fails under heterogeneous shard priors"},
        {"Strategy": "Random-balanced", "Role in study": "optional control",
         "Design principle": "balance size without explicit class/feature/risk structure",
         "Actual assignment basis": "global seeded shuffle and equal split",
         "Expected shard structure": "balanced random shards",
         "Matched deletion pattern": "none/control",
         "Aggregation approach": "retained-size weighted + Platt",
         "Primary hypothesis": "separate balancing from stratification",
         "Observed utility effect": f"controlled ROC-AUC={controlled.loc['random_balanced','roc_auc']:.6f}",
         "Observed locality/workload effect": "locality is incidental rather than designed",
         "Main limitation": "no deliberate alignment with deletion requests"},
    ]
    atomic_csv(pd.DataFrame(rows), DIRS["results"] / "sharding_contribution_thesis.csv")


def stage_policy_aware_extension(run_runtime: bool = True) -> None:
    """Run the only approved supplementary experiment and checkpoint its evidence."""
    required = ["policy_relationship_summary.csv", "policy_nested_summary.csv",
                "policy_deletion_locality_results.csv", "policy_nested_utility.csv",
                "legacy_to_final_coverage.csv"]
    if run_runtime:
        required.append("policy_deletion_runtime.csv")
    version_path = DIRS["config"] / "policy_aware_extension.json"
    version_valid = False
    if version_path.exists():
        import json
        version_valid = json.loads(version_path.read_text(encoding="utf-8")).get("schema_version") == POLICY_SCHEMA_VERSION
    if version_valid and all((DIRS["results"] / name).exists() for name in required):
        _legacy_coverage_inventory()
        _update_traceability()
        _update_contribution_tables()
        log("Loaded validated checkpoint: supplementary policy-aware extension")
        return

    mapping, _ = build_synthetic_relationship_mapping()
    assignments, feature_clusters, _ = _strategy_assignments()
    base_frame = assignments["specialist"].sort_values("row_index")
    base = base_frame.shard.to_numpy(dtype=int)
    relationships = _relationship_scenarios(mapping, base)
    nested, spread = nested_specialist_optimization(base, mapping, relationships)
    assert len(nested) == len(base) == len(feature_clusters)

    _, y, ids = load_matrix("train")
    moved = nested != base
    base_sizes = np.bincount(base, minlength=CFG.shards)
    nested_sizes = np.bincount(nested, minlength=CFG.shards)
    base_cluster_purity = np.mean([
        np.bincount(feature_clusters[base == shard], minlength=CFG.shards).max() / np.sum(base == shard)
        for shard in range(CFG.shards)])
    nested_cluster_purity = np.mean([
        np.bincount(feature_clusters[nested == shard], minlength=CFG.shards).max() / np.sum(nested == shard)
        for shard in range(CFG.shards)])
    balance = pd.DataFrame({"shard": np.arange(CFG.shards), "base_size": base_sizes,
                            "nested_size": nested_sizes,
                            "base_default_rate": [y[base == shard].mean() for shard in range(CFG.shards)],
                            "nested_default_rate": [y[nested == shard].mean() for shard in range(CFG.shards)],
                            "base_dominant_feature_cluster_share": [
                                np.bincount(feature_clusters[base == shard], minlength=CFG.shards).max() / np.sum(base == shard)
                                for shard in range(CFG.shards)],
                            "nested_dominant_feature_cluster_share": [
                                np.bincount(feature_clusters[nested == shard], minlength=CFG.shards).max() / np.sum(nested == shard)
                                for shard in range(CFG.shards)]})
    atomic_csv(balance, DIRS["results"] / "policy_nested_balance.csv")
    atomic_csv(spread, DIRS["results"] / "policy_nested_relationship_spread.csv")
    summary = pd.DataFrame([{
        "extension": "Policy-Aware / Company-Aware Nested Specialist",
        "base_sharding": "Specialist", "relationship_layer": "synthetic company mapping",
        "deletion_policy": "strict", "balance_tolerance": BALANCE_TOLERANCE,
        "maximum_moved_fraction": MAX_MOVED_FRACTION,
        "customers": len(base), "customers_moved": int(moved.sum()),
        "customers_moved_fraction": moved.mean(),
        "specialist_assignment_preservation": 1 - moved.mean(),
        "assignment_adjusted_rand_index": adjusted_rand_score(base, nested),
        "base_mean_dominant_feature_cluster_share": base_cluster_purity,
        "nested_mean_dominant_feature_cluster_share": nested_cluster_purity,
        "dominant_feature_cluster_share_change": nested_cluster_purity - base_cluster_purity,
        "relationships_improved": int((spread.strict_spread_after < spread.strict_spread_before).sum()),
        "relationships_worsened": int((spread.full_spread_after > spread.full_spread_before).sum()),
        "mean_strict_spread_before": spread.strict_spread_before.mean(),
        "mean_strict_spread_after": spread.strict_spread_after.mean(),
        "max_abs_default_rate_change": np.max(np.abs(balance.nested_default_rate-balance.base_default_rate)),
    }])
    atomic_csv(summary, DIRS["results"] / "policy_nested_summary.csv")

    nested_frame = pd.DataFrame({"row_index": np.arange(len(base)), "SK_ID_CURR": ids,
                                 "TARGET": y, "shard": nested})
    nested_frame.to_csv(DIRS["results"] / "policy_nested_assignment.csv.gz", index=False, compression="gzip")
    locality = _locality_results(mapping, base, nested, relationships)

    # One new architecture utility run; the validated Specialist checkpoint is reused.
    controlled = pd.read_csv(DIRS["results"] / "expanded_controlled_sharding.csv")
    base_utility = controlled.loc[controlled.strategy.eq("specialist")].iloc[0]
    met, sizes, fit_times, aggregation = _evaluate_strategy(
        "policy_nested_specialist_v2", nested_frame, practical=False)
    utility = pd.DataFrame([
        {"assignment": "Specialist baseline", "roc_auc": base_utility.roc_auc,
         "average_precision": base_utility.average_precision, "ks": base_utility.ks,
         "brier": base_utility.brier, "aggregation": base_utility.aggregation,
         "total_fit_seconds": base_utility.total_fit_seconds},
        {"assignment": "Specialist + policy-aware nested mapping", "roc_auc": met["roc_auc"],
         "average_precision": met["average_precision"], "ks": met["ks"],
         "brier": met["brier"], "aggregation": aggregation,
         "total_fit_seconds": fit_times.sum()},
    ])
    for metric in ["roc_auc", "average_precision", "ks", "brier"]:
        utility[f"{metric}_delta_vs_specialist"] = utility[metric] - utility.loc[0, metric]
    atomic_csv(utility, DIRS["results"] / "policy_nested_utility.csv")

    if run_runtime:
        _runtime_benchmark(mapping, base, nested, relationships)
    _figures(mapping, balance, locality)
    _legacy_coverage_inventory()
    _update_traceability()
    _update_contribution_tables()
    rules = pd.DataFrame([
        {"policy": "strict", "eligible": "customer has only the removed relationship",
         "multi_relationship_customer": "retain while another relationship remains",
         "role": "main supplementary experiment"},
        {"policy": "any", "eligible": "customer has the removed relationship among one or more",
         "multi_relationship_customer": "delete", "role": "explanatory alternative only; not expanded into grid"},
    ])
    atomic_csv(rules, DIRS["results"] / "policy_deletion_rules.csv")
    extension = pd.DataFrame([{
        "extension": "Policy-Aware / Company-Aware Nested Specialist",
        "base_sharding": "Specialist", "additional_relationship_layer": "synthetic company memberships",
        "deletion_policy": "strict single-relationship eligibility",
        "intended_operational_locality": "concentrate eligible customers for relationship-level erasure",
        "observed_workload_effect": "see policy_deletion_locality_summary.csv",
        "utility_effect": "see policy_nested_utility.csv",
        "limitation": "synthetic, intentionally optimized scenario; governance/fairness/maintenance not evaluated"}])
    atomic_csv(extension, DIRS["results"] / "policy_extension_contribution.csv")
    from final_pipeline import save_json
    save_json({"schema_version": POLICY_SCHEMA_VERSION, "seed": POLICY_SEED,
               "balance_tolerance": BALANCE_TOLERANCE,
               "maximum_moved_fraction": MAX_MOVED_FRACTION,
               "supplementary_not_primary_strategy": True}, version_path)
    update_manifest("supplementary_policy_aware_nested_specialist", "complete", [
        "results/policy_relationship_mapping.csv.gz", "results/policy_nested_assignment.csv.gz",
        "results/policy_nested_summary.csv", "results/policy_nested_utility.csv",
        "results/policy_deletion_locality_results.csv", "results/policy_deletion_runtime.csv",
        "figures/policy_relationship_mapping.png", "figures/policy_nested_specialist_results.png"],
        "Synthetic policy-aware relationship layer nested within corrected Specialist; not a fifth primary strategy.")
    log("Computed during this execution: supplementary policy-aware nested Specialist extension")


if __name__ == "__main__":
    stage_policy_aware_extension(run_runtime=True)
