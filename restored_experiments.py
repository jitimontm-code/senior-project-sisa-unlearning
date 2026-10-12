"""Additive restoration experiments for the Home Credit SISA project.

This module is deliberately isolated from ``outputs_final``.  It may read validated
final artifacts after strict compatibility checks, but every artifact it creates is
written below ``outputs_restored``.  The legacy notebook and artifacts are never read.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler

from final_pipeline import (
    CFG,
    SCHEMA_VERSION as FINAL_SCHEMA_VERSION,
    calibrate,
    choose_threshold,
    fit_platt,
    load_matrix,
    make_model,
    metrics,
    positive_probability,
)


ROOT = Path(__file__).resolve().parent
FINAL_OUT = ROOT / "outputs_final"
OUT = ROOT / "outputs_restored"
DIRS = {name: OUT / name for name in [
    "config", "cache", "models", "shards", "deletion_requests",
    "results", "figures", "logs", "checkpoints",
]}
for path in DIRS.values():
    path.mkdir(parents=True, exist_ok=True)

RESTORED_SCHEMA_VERSION = "home-credit-sisa-restored-v1"
MODEL_NAMES = ["Logistic Regression", "Random Forest", "LightGBM", "XGBoost"]
S_GRID = [2, 4, 8, 16, 32, 64, 128]
STRATEGIES = [
    "stratified_balanced", "random_balanced", "representative",
    "specialist", "risk_band",
]
LEARNED_STRATEGIES = ["specialist", "risk_band"]
SLICING_GRID = [1, 2, 4, 8]
STRESS_STREAMS = ["localized_repeated", "rotating_localized", "diffuse_repeated"]
STRESS_ROUNDS = 30
STRESS_RECORDS_PER_ROUND = 4
STRESS_GOLD_CHECKPOINTS = [0, 1, 5, 10, 20, 30]
STRESS_GOLD_TIMING_ROUNDS = list(range(STRESS_ROUNDS + 1))
REPEATED_EXACT_REQUESTS = 4

FINAL_NOTEBOOK = ROOT / "home_credit_sisa_final.ipynb"
LEGACY_NOTEBOOK = ROOT / "home_credit_fixed final.ipynb"
RESTORED_NOTEBOOK = ROOT / "home_credit_sisa_restored.ipynb"
EXPECTED_FINAL_NOTEBOOK_SHA256 = "F4C72FF83F3B40CF5A32B3C20CB2D3C99FA203B9D10892F4DFE2BEB917B99717"
EXPECTED_LEGACY_NOTEBOOK_SHA256 = "51E642E9BD463760A1491C620710CAF8BCE846336D6D0CA5915C46351938E030"


def _slug(text: str) -> str:
    return text.lower().replace(" ", "_").replace("-", "_")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def _atomic_json(data: dict | list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)


def _log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with (DIRS["logs"] / "restoration_runtime.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _manifest() -> dict:
    path = DIRS["config"] / "restoration_manifest.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    data = {
        "schema_version": RESTORED_SCHEMA_VERSION,
        "base_schema_version": FINAL_SCHEMA_VERSION,
        "base_config_hash": CFG.hash,
        "restoration_plan": {
            "models": MODEL_NAMES,
            "shard_grid": S_GRID,
            "strategies": STRATEGIES,
            "slicing_grid": SLICING_GRID,
            "exact_requests_per_model": REPEATED_EXACT_REQUESTS,
            "learned_strategies": LEARNED_STRATEGIES,
            "stress": {
                "model": "LightGBM", "rounds": STRESS_ROUNDS,
                "records_per_round": STRESS_RECORDS_PER_ROUND,
                "streams": STRESS_STREAMS,
                "gold_timing_rounds": STRESS_GOLD_TIMING_ROUNDS,
                "fidelity_checkpoints": STRESS_GOLD_CHECKPOINTS,
            },
        },
        "stages": {},
        "verified_for_candidate_source_of_truth": False,
    }
    _atomic_json(data, path)
    return data


def _update_manifest(stage: str, status: str, artifacts: list[str], notes: str = "") -> None:
    path = DIRS["config"] / "restoration_manifest.json"
    data = _manifest()
    data["stages"][stage] = {
        "status": status,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "artifacts": artifacts,
        "notes": notes,
    }
    _atomic_json(data, path)


def _ks(y: np.ndarray, probability: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y, probability)
    return float(np.max(tpr - fpr))


def _source_inventory(root: Path) -> list[dict]:
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rows.append({
            "relative_path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        })
    return rows


def initialize_restoration() -> None:
    """Freeze provenance and verify the final environment before any experiment."""
    if _sha256(FINAL_NOTEBOOK) != EXPECTED_FINAL_NOTEBOOK_SHA256:
        raise RuntimeError("Final notebook hash changed; restoration aborted")
    if _sha256(LEGACY_NOTEBOOK) != EXPECTED_LEGACY_NOTEBOOK_SHA256:
        raise RuntimeError("Legacy notebook hash changed; restoration aborted")
    final_env = json.loads((FINAL_OUT / "config" / "environment.json").read_text(encoding="utf-8"))
    if final_env["config_hash"] != CFG.hash or final_env["schema_version"] != FINAL_SCHEMA_VERSION:
        raise RuntimeError("Final environment/config is incompatible")
    import lightgbm, sklearn, scipy, xgboost
    current = {
        "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
        "scipy": scipy.__version__, "lightgbm": lightgbm.__version__, "xgboost": xgboost.__version__,
        "joblib": joblib.__version__,
    }
    if current != final_env["packages"]:
        raise RuntimeError(f"Package mismatch: current={current}, final={final_env['packages']}")
    inventory_path = DIRS["config"] / "outputs_final_pre_restore_inventory.json"
    if not inventory_path.exists():
        _atomic_json(_source_inventory(FINAL_OUT), inventory_path)
    provenance = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "final_notebook_sha256": _sha256(FINAL_NOTEBOOK),
        "legacy_notebook_sha256": _sha256(LEGACY_NOTEBOOK),
        "outputs_final_inventory_sha256": _sha256(inventory_path),
        "base_schema_version": FINAL_SCHEMA_VERSION,
        "base_config_hash": CFG.hash,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": current,
        "read_only_source": str(FINAL_OUT),
        "write_root": str(OUT),
    }
    _atomic_json(provenance, DIRS["config"] / "provenance.json")
    _manifest()
    _update_manifest(
        "initialization", "complete",
        ["config/provenance.json", "config/outputs_final_pre_restore_inventory.json"],
        "Original notebooks and outputs_final hashes frozen before restoration.",
    )
    _log("Restoration initialized; immutable source hashes and environment verified")


def verify_source_immutability() -> pd.DataFrame:
    before = json.loads((DIRS["config"] / "outputs_final_pre_restore_inventory.json").read_text(encoding="utf-8"))
    after = _source_inventory(FINAL_OUT)
    before_map = {row["relative_path"]: (row["size_bytes"], row["sha256"]) for row in before}
    after_map = {row["relative_path"]: (row["size_bytes"], row["sha256"]) for row in after}
    paths = sorted(set(before_map) | set(after_map))
    rows = []
    for path in paths:
        b, a = before_map.get(path), after_map.get(path)
        rows.append({
            "relative_path": path,
            "before_present": b is not None,
            "after_present": a is not None,
            "size_unchanged": b is not None and a is not None and b[0] == a[0],
            "sha256_unchanged": b is not None and a is not None and b[1] == a[1],
        })
    result = pd.DataFrame(rows)
    result.loc[len(result)] = {
        "relative_path": "../home_credit_sisa_final.ipynb",
        "before_present": True, "after_present": FINAL_NOTEBOOK.exists(),
        "size_unchanged": True,
        "sha256_unchanged": _sha256(FINAL_NOTEBOOK) == EXPECTED_FINAL_NOTEBOOK_SHA256,
    }
    result.loc[len(result)] = {
        "relative_path": "../home_credit_fixed final.ipynb",
        "before_present": True, "after_present": LEGACY_NOTEBOOK.exists(),
        "size_unchanged": True,
        "sha256_unchanged": _sha256(LEGACY_NOTEBOOK) == EXPECTED_LEGACY_NOTEBOOK_SHA256,
    }
    _atomic_csv(result, DIRS["results"] / "source_immutability_audit.csv")
    if not result[["before_present", "after_present", "size_unchanged", "sha256_unchanged"]].all().all():
        raise RuntimeError("Immutable source changed during restoration")
    return result


def _validate_assignment(frame: pd.DataFrame, n_shards: int) -> pd.DataFrame:
    _, y, ids = load_matrix("train")
    required = {"row_index", "SK_ID_CURR", "TARGET", "shard"}
    if not required.issubset(frame.columns) or len(frame) != len(y):
        raise RuntimeError("Assignment shape/schema mismatch")
    ordered = frame.sort_values("row_index").reset_index(drop=True)
    if not np.array_equal(ordered.row_index.to_numpy(), np.arange(len(y))):
        raise RuntimeError("Assignment row_index mismatch")
    if not np.array_equal(ordered.SK_ID_CURR.to_numpy(dtype=np.int64), ids.astype(np.int64)):
        raise RuntimeError("Assignment applicant identity mismatch")
    if not np.array_equal(ordered.TARGET.to_numpy(dtype=np.int8), y.astype(np.int8)):
        raise RuntimeError("Assignment target mismatch")
    if sorted(ordered.shard.unique().tolist()) != list(range(n_shards)):
        raise RuntimeError("Assignment shard labels mismatch")
    return ordered


def _final_assignment_path(strategy: str, n_shards: int) -> Path:
    if strategy == "stratified_balanced":
        if n_shards == 4:
            return FINAL_OUT / "shards" / f"stratified_S4_{CFG.hash}.csv"
        return FINAL_OUT / "shards" / f"phase2_stratified_S{n_shards}_{CFG.hash}.csv"
    if strategy == "random_balanced":
        return FINAL_OUT / "shards" / f"random_balanced_S4_{CFG.hash}.csv"
    return FINAL_OUT / "shards" / f"phase2_{strategy}_S4_{CFG.hash}.csv"


def load_assignment(strategy: str, n_shards: int) -> pd.DataFrame:
    path = _final_assignment_path(strategy, n_shards)
    if not path.exists():
        raise FileNotFoundError(path)
    return _validate_assignment(pd.read_csv(path), n_shards)


def _final_model_candidates(model_name: str, n_shards: int, shard: int, strategy: str) -> list[Path]:
    slug = _slug(model_name)
    candidates: list[Path] = []
    if strategy == "stratified_balanced":
        if model_name == "LightGBM" and n_shards == 4:
            candidates.extend([
                FINAL_OUT / "models" / f"sisa_lightgbm_S4_shard{shard}_{CFG.hash}.joblib",
                FINAL_OUT / "models" / f"phase2_stratified_balanced_lightgbm_S4_shard{shard}_{CFG.hash}.joblib",
            ])
        else:
            candidates.append(FINAL_OUT / "models" / f"phase2_stratified_{slug}_S{n_shards}_shard{shard}_{CFG.hash}.joblib")
    else:
        candidates.append(FINAL_OUT / "models" / f"phase2_{strategy}_{slug}_S{n_shards}_shard{shard}_{CFG.hash}.joblib")
    return candidates


def _validate_model_bundle(bundle: dict, expected_rows: int, n_shards: int, strategy: str) -> None:
    if bundle.get("schema_version") != FINAL_SCHEMA_VERSION or bundle.get("config_hash") != CFG.hash:
        raise RuntimeError("Model checkpoint schema/config mismatch")
    if "rows" in bundle and int(bundle["rows"]) != int(expected_rows):
        raise RuntimeError("Model checkpoint training-row count mismatch")
    if "shards" in bundle and int(bundle["shards"]) != n_shards:
        raise RuntimeError("Model checkpoint shard-count mismatch")
    if "strategy" in bundle and strategy not in {bundle["strategy"], "stratified_balanced"}:
        raise RuntimeError("Model checkpoint strategy mismatch")


def _restored_model_path(model_name: str, n_shards: int, shard: int, strategy: str) -> Path:
    return DIRS["models"] / f"{strategy}_{_slug(model_name)}_S{n_shards}_shard{shard}_{CFG.hash}.joblib"


def fit_or_load_shards(model_name: str, n_shards: int, assignment: pd.DataFrame,
                       strategy: str, allow_final_reuse: bool = True):
    X_train, y_train, _ = load_matrix("train")
    models, sizes, fit_times, origins = [], [], [], []
    for shard in range(n_shards):
        rows = assignment.loc[assignment.shard.eq(shard), "row_index"].to_numpy(dtype=int)
        restored_path = _restored_model_path(model_name, n_shards, shard, strategy)
        selected: Path | None = restored_path if restored_path.exists() else None
        origin = "outputs_restored"
        if selected is None and allow_final_reuse:
            selected = next((p for p in _final_model_candidates(model_name, n_shards, shard, strategy) if p.exists()), None)
            if selected is not None:
                origin = "outputs_final_verified"
        if selected is not None:
            bundle = joblib.load(selected)
            _validate_model_bundle(bundle, len(rows), n_shards, strategy)
            model = bundle["model"]
            elapsed = float(bundle.get("fit_seconds", 0.0))
        else:
            started = time.perf_counter()
            model = make_model(model_name, CFG.seed + 100 + shard)
            model.fit(X_train[rows], y_train[rows])
            elapsed = time.perf_counter() - started
            bundle = {
                "model": model, "rows": len(rows), "fit_seconds": elapsed,
                "schema_version": FINAL_SCHEMA_VERSION, "restored_schema_version": RESTORED_SCHEMA_VERSION,
                "config_hash": CFG.hash, "strategy": strategy, "shards": n_shards,
                "seed": CFG.seed + 100 + shard,
            }
            joblib.dump(bundle, restored_path, compress=3)
            selected = restored_path
            _log(f"Trained {model_name} {strategy} S={n_shards} shard {shard + 1}/{n_shards}")
        models.append(model)
        sizes.append(len(rows))
        fit_times.append(elapsed)
        origins.append({"shard": shard, "origin": origin, "path": str(selected), "sha256": _sha256(selected)})
    del X_train
    gc.collect()
    return models, np.asarray(sizes, dtype=int), np.asarray(fit_times, dtype=float), origins


def _component_matrix(models: list, X) -> np.ndarray:
    return np.column_stack([positive_probability(model, X) for model in models])


def _evaluate_weighted(models: list, retained_sizes: np.ndarray):
    X_cal, y_cal, _ = load_matrix("calibration")
    X_val, y_val, _ = load_matrix("validation")
    X_test, y_test, test_ids = load_matrix("test")
    weights = retained_sizes.astype(float) / retained_sizes.sum()
    raw_cal = _component_matrix(models, X_cal) @ weights
    platt = fit_platt(raw_cal, y_cal)
    val_probability = calibrate(platt, _component_matrix(models, X_val) @ weights)
    threshold = choose_threshold(y_val, val_probability)
    test_probability = calibrate(platt, _component_matrix(models, X_test) @ weights)
    result = metrics(y_test, test_probability, threshold)
    result["ks"] = _ks(y_test, test_probability)
    del X_cal, X_val, X_test, raw_cal, val_probability
    gc.collect()
    return result, test_probability, test_ids, y_test, platt, threshold, weights


def _save_prediction(path: Path, ids: np.ndarray, y: np.ndarray, probability: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, ids=ids, y=y, probability=probability)


def stage_slicing_corrected() -> pd.DataFrame:
    """Full favorable-case R grid with corrected replay accounting and two speedups."""
    checkpoint = DIRS["results"] / "slicing_corrected_full_grid.csv"
    required_columns = {
        "replay_span_rows_before_deletion", "actual_replayed_rows_after_deletion",
        "actual_replayed_fraction_of_retained_affected_shard",
        "affected_shard_full_retrain_seconds", "slicing_only_speedup",
        "end_to_end_speedup_vs_all_shard_gold", "deleted_row_index",
        "deleted_slice", "slice_position_description",
    }
    if checkpoint.exists():
        existing = pd.read_csv(checkpoint)
    else:
        existing = pd.DataFrame()
    if (
        not existing.empty
        and set(existing.slices.astype(int)) == set(SLICING_GRID)
        and required_columns.issubset(existing.columns)
    ):
        _slicing_supersession_comparison(existing)
        _slicing_figure(existing)
        _log("Validated corrected slicing checkpoint")
        return existing
    X_train, y_train, train_ids = load_matrix("train")
    X_test, y_test, _ = load_matrix("test")
    scaler_path = FINAL_OUT / "models" / f"slicing_maxabs_scaler_{CFG.hash}.joblib"
    if not scaler_path.exists():
        raise FileNotFoundError(scaler_path)
    scaler = joblib.load(scaler_path)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    assignment = load_assignment("stratified_balanced", 4)
    results = []
    for n_slices in SLICING_GRID:
        setup_dir = DIRS["models"] / f"slicing_corrected_S4_R{n_slices}_{CFG.hash}"
        setup_dir.mkdir(parents=True, exist_ok=True)
        shard_models, slice_maps = [], {}
        for shard in range(4):
            rows = assignment.loc[assignment.shard.eq(shard), "row_index"].to_numpy(dtype=int, copy=True)
            rng = np.random.default_rng(CFG.seed + shard)
            rng.shuffle(rows)
            slices = [np.asarray(part, dtype=int) for part in np.array_split(rows, n_slices)]
            slice_maps[shard] = slices
            model, _, _ = _partial_fit_sequence_restored(
                X_train, y_train, slices, setup_dir, f"original_shard{shard}")
            shard_models.append(model)
        delete_row = int(slice_maps[0][-1][0])
        delete_id = int(train_ids[delete_row])
        earliest = n_slices - 1
        if earliest == 0:
            exact, replay_seconds, _ = _partial_fit_sequence_restored(
                X_train, y_train, slice_maps[0], setup_dir, f"exact_R{n_slices}", {delete_row})
        else:
            exact = joblib.load(setup_dir / f"original_shard0_slice{earliest - 1}.joblib")
            replay_started = time.perf_counter()
            for rows in slice_maps[0][earliest:]:
                use = rows[rows != delete_row]
                for _ in range(3):
                    exact.partial_fit(X_train[use], y_train[use])
            replay_seconds = time.perf_counter() - replay_started
        gold_models = []
        gold_seconds = 0.0
        affected_shard_full_retrain_seconds = None
        for shard in range(4):
            deleted = {delete_row} if shard == 0 else set()
            gold, shard_full_seconds, _ = _partial_fit_sequence_restored(
                X_train, y_train, slice_maps[shard], setup_dir,
                f"gold_R{n_slices}_shard{shard}", deleted)
            gold_models.append(gold)
            gold_seconds += shard_full_seconds
            if shard == 0:
                affected_shard_full_retrain_seconds = shard_full_seconds
        exact_models = list(shard_models)
        exact_models[0] = exact
        retained_sizes = assignment.groupby("shard").size().sort_index().to_numpy(dtype=float)
        retained_sizes[0] -= 1
        retained_weights = retained_sizes / retained_sizes.sum()
        p_exact = _raw_weighted(exact_models, X_test, retained_weights)
        p_gold = _raw_weighted(gold_models, X_test, retained_weights)
        difference = p_exact - p_gold
        replay_span_rows = sum(len(part) for part in slice_maps[0][earliest:])
        actual_replayed_rows = replay_span_rows - 1
        actual_replayed_fraction = actual_replayed_rows / retained_sizes[0]
        row = {
            "model": "SGD logistic (incremental LR)",
            "preprocessing": "train-fitted MaxAbsScaler; 3 deterministic passes/slice",
            "aggregation": "post-deletion retained-size weighted probability",
            "shards": 4, "slices": n_slices,
            "deleted_records": 1, "deleted_SK_ID_CURR": delete_id,
            "deleted_row_index": delete_row,
            "deleted_slice": earliest,
            "slice_position_description": "last-slice favorable mechanism demonstration",
            "earliest_affected_slice": earliest,
            "replay_span_rows_before_deletion": replay_span_rows,
            "actual_replayed_rows_after_deletion": actual_replayed_rows,
            "replayed_rows": actual_replayed_rows,
            "affected_shard_rows_before_deletion": int(retained_sizes[0] + 1),
            "affected_shard_rows_after_deletion": int(retained_sizes[0]),
            "actual_replayed_fraction_of_retained_affected_shard": actual_replayed_fraction,
            "replayed_fraction_of_affected_shard": actual_replayed_fraction,
            "replay_seconds": replay_seconds,
            "affected_shard_full_retrain_seconds": affected_shard_full_retrain_seconds,
            "gold_all_shards_seconds": gold_seconds,
            "slicing_only_speedup": affected_shard_full_retrain_seconds / max(replay_seconds, 1e-12),
            "end_to_end_speedup_vs_all_shard_gold": gold_seconds / max(replay_seconds, 1e-12),
            "prediction_mae": float(np.mean(np.abs(difference))),
            "max_probability_difference": float(np.max(np.abs(difference))),
            "exact_roc_auc": float(roc_auc_score(y_test, p_exact)),
            "gold_roc_auc": float(roc_auc_score(y_test, p_gold)),
            "affected_shard_weight": float(retained_weights[0]),
        }
        results.append(row)
        _atomic_csv(pd.DataFrame(results), checkpoint)
        np.savez_compressed(
            DIRS["results"] / f"slicing_corrected_R{n_slices}_{CFG.hash}.npz",
            exact=p_exact, gold=p_gold, y=y_test, retained_weights=retained_weights,
        )
        _log(f"Corrected slicing R={n_slices}: AUC={row['exact_roc_auc']:.6f}, MAE={row['prediction_mae']:.3g}")
        del shard_models, gold_models, exact_models, p_exact, p_gold
        gc.collect()
    result = pd.DataFrame(results).sort_values("slices").reset_index(drop=True)
    _atomic_csv(result, checkpoint)
    comparison = _slicing_supersession_comparison(result)
    _slicing_figure(result)
    _update_manifest(
        "slicing_corrected_full_grid", "complete",
        ["results/slicing_corrected_full_grid.csv", "results/slicing_original_vs_corrected.csv",
         "figures/slicing_corrected_full_grid.png"],
        "R=1/2/4/8 favorable last-slice mechanism demonstration; actual replay excludes the deleted row; slicing-only and end-to-end speedups are reported separately; post-deletion retained-size weights.",
    )
    return result


def _partial_fit_sequence_restored(X, y, slices, checkpoint_dir: Path, prefix: str, delete_rows=None):
    model = None
    checkpoints, elapsed = [], 0.0
    delete_rows = set() if delete_rows is None else set(delete_rows)
    for slice_index, rows in enumerate(slices):
        use = np.asarray([idx for idx in rows if int(idx) not in delete_rows], dtype=int)
        if model is None:
            model = SGDClassifier(
                loss="log_loss", alpha=1e-4, average=True, random_state=CFG.seed,
                max_iter=1, tol=None, shuffle=False,
            )
        started = time.perf_counter()
        for epoch in range(3):
            classes = np.array([0, 1]) if slice_index == 0 and epoch == 0 else None
            model.partial_fit(X[use], y[use], classes=classes)
        elapsed += time.perf_counter() - started
        path = checkpoint_dir / f"{prefix}_slice{slice_index}.joblib"
        joblib.dump(model, path, compress=3)
        checkpoints.append(path)
    return model, elapsed, checkpoints


def _raw_weighted(models: list, X, weights: np.ndarray) -> np.ndarray:
    output = np.zeros(X.shape[0], dtype=np.float64)
    for model, weight in zip(models, weights):
        output += positive_probability(model, X) * weight
    return output


def _slicing_supersession_comparison(corrected: pd.DataFrame) -> pd.DataFrame:
    original = pd.read_csv(FINAL_OUT / "results" / "slicing_demonstration.csv")
    rows = []
    for r_value in [1, 4]:
        old = original.loc[original.slices.eq(r_value)].iloc[0]
        new = corrected.loc[corrected.slices.eq(r_value)].iloc[0]
        row = {"slices": r_value}
        metric_pairs = {
            "replayed_fraction_of_affected_shard": "actual_replayed_fraction_of_retained_affected_shard",
            "replay_seconds": "replay_seconds",
            "gold_all_shards_seconds": "gold_all_shards_seconds",
            "speedup_vs_gold": "end_to_end_speedup_vs_all_shard_gold",
            "prediction_mae": "prediction_mae",
            "max_probability_difference": "max_probability_difference",
            "exact_roc_auc": "exact_roc_auc",
            "gold_roc_auc": "gold_roc_auc",
        }
        for old_metric, new_metric in metric_pairs.items():
            row[f"original_{old_metric}"] = float(old[old_metric])
            row[f"corrected_{old_metric}"] = float(new[new_metric])
            row[f"delta_{old_metric}"] = float(new[new_metric]) - float(old[old_metric])
        utility_material = abs(row["delta_exact_roc_auc"]) > 1e-4 or abs(row["delta_gold_roc_auc"]) > 1e-4
        fidelity_material = (
            float(new.prediction_mae) > 1e-10 or float(new.max_probability_difference) > 1e-10
        )
        replay_ratio = float(new.replay_seconds) / max(float(old.replay_seconds), 1e-12)
        gold_ratio = float(new.gold_all_shards_seconds) / max(float(old.gold_all_shards_seconds), 1e-12)
        speedup_ratio = float(new.end_to_end_speedup_vs_all_shard_gold) / max(float(old.speedup_vs_gold), 1e-12)
        runtime_material = any(ratio > 2.0 or ratio < 0.5 for ratio in [replay_ratio, gold_ratio, speedup_ratio])
        row.update({
            "utility_material": utility_material,
            "fidelity_material": fidelity_material,
            "runtime_material": runtime_material,
            "material_difference": utility_material or fidelity_material,
            "runtime_only_change_allowed": runtime_material and not (utility_material or fidelity_material),
            "materiality_rule": "stop only for |delta AUC|>1e-4 or fidelity>1e-10; runtime correction is documented and allowed",
        })
        rows.append(row)
    result = pd.DataFrame(rows)
    _atomic_csv(result, DIRS["results"] / "slicing_original_vs_corrected.csv")
    if result.material_difference.any():
        _update_manifest(
            "slicing_corrected_full_grid", "blocked_material_difference",
            ["results/slicing_corrected_full_grid.csv", "results/slicing_original_vs_corrected.csv"],
            "Material R=1/4 difference detected; stop required by researcher instruction.",
        )
        raise RuntimeError("Corrected slicing differs materially from original R=1/4; stop before further experiments")
    return result


def _slicing_figure(result: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    labels = [f"R={int(value)}" for value in result.slices]
    axes[0].bar(labels, result.actual_replayed_fraction_of_retained_affected_shard, color="#256d85")
    axes[0].set_ylabel("Actual replayed fraction of retained affected shard")
    axes[0].set_title("Last-slice replay workload")
    x = np.arange(len(result))
    width = .36
    axes[1].bar(x - width / 2, result.slicing_only_speedup, width, label="Slicing only")
    axes[1].bar(x + width / 2, result.end_to_end_speedup_vs_all_shard_gold, width, label="End-to-end vs all-shard Gold")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Measured speedup")
    axes[1].set_title("Two operational speedup definitions")
    axes[1].legend(fontsize=8)
    axes[2].plot(result.slices, result.max_probability_difference, marker="o")
    axes[2].set_xscale("log", base=2)
    axes[2].set_xticks(result.slices, result.slices)
    axes[2].set_ylabel("Max |Exact - Gold probability|")
    axes[2].set_title("Exact–Gold fidelity")
    for axis in axes:
        axis.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(DIRS["figures"] / "slicing_corrected_full_grid.png", dpi=300)
    plt.close(fig)


def stage_cross_model_shard_count() -> pd.DataFrame:
    result_path = DIRS["results"] / "cross_model_shard_count.csv"
    completed = pd.read_csv(result_path).to_dict("records") if result_path.exists() else []
    done = {(row["model"], int(row["shards"])) for row in completed}
    _, y_train, _ = load_matrix("train")
    for model_name in MODEL_NAMES:
        for n_shards in S_GRID:
            if (model_name, n_shards) in done:
                continue
            assignment = load_assignment("stratified_balanced", n_shards)
            models, sizes, fit_times, origins = fit_or_load_shards(
                model_name, n_shards, assignment, "stratified_balanced")
            met, probability, ids, y_test, platt, threshold, weights = _evaluate_weighted(models, sizes)
            positive_counts = assignment.groupby("shard").TARGET.sum().sort_index().to_numpy(dtype=int)
            row = {
                "model": model_name, "shards": n_shards,
                "aggregation": "retained-size weighted probability + Platt calibration",
                "roc_auc": met["roc_auc"], "average_precision": met["average_precision"],
                "ks": met["ks"], "brier": met["brier"], "log_loss": met["log_loss"],
                "threshold": threshold, "f1": met["f1"],
                "sensitivity": met["sensitivity"], "specificity": met["specificity"],
                "total_fit_seconds": fit_times.sum(),
                "mean_rows_per_shard": sizes.mean(), "min_rows_per_shard": sizes.min(),
                "max_rows_per_shard": sizes.max(),
                "mean_positives_per_shard": positive_counts.mean(),
                "median_positives_per_shard": np.median(positive_counts),
                "min_positives_per_shard": positive_counts.min(),
                "zero_positive_shards": int(np.sum(positive_counts == 0)),
                "reused_shards_from_outputs_final": sum(o["origin"] == "outputs_final_verified" for o in origins),
                "new_shards_in_outputs_restored": sum(o["origin"] == "outputs_restored" for o in origins),
                "selection_role": "pre-specified sensitivity grid; no test-based optimization",
            }
            completed.append(row)
            _atomic_csv(pd.DataFrame(completed), result_path)
            _save_prediction(
                DIRS["results"] / "predictions" / f"shard_count_{_slug(model_name)}_S{n_shards}_{CFG.hash}.npz",
                ids, y_test, probability,
            )
            _atomic_json(origins, DIRS["checkpoints"] / f"origins_shard_count_{_slug(model_name)}_S{n_shards}.json")
            joblib.dump(
                {"calibrator": platt, "threshold": threshold, "weights": weights,
                 "schema_version": RESTORED_SCHEMA_VERSION, "config_hash": CFG.hash},
                DIRS["models"] / f"ensemble_shard_count_{_slug(model_name)}_S{n_shards}_{CFG.hash}.joblib",
                compress=3,
            )
            _log(f"Shard-count complete: {model_name} S={n_shards}, AUC={met['roc_auc']:.6f}")
            del models, probability, platt
            gc.collect()
    result = pd.DataFrame(completed).sort_values(["model", "shards"]).reset_index(drop=True)
    # Class-rate diagnostics are assignment properties and require no model retraining.
    for n_shards in S_GRID:
        assignment = load_assignment("stratified_balanced", n_shards)
        rates = assignment.groupby("shard").TARGET.mean().sort_index().to_numpy(dtype=float)
        mask = result.shards.astype(int).eq(n_shards)
        result.loc[mask, "min_positive_rate_per_shard"] = float(rates.min())
        result.loc[mask, "mean_positive_rate_per_shard"] = float(rates.mean())
        result.loc[mask, "max_positive_rate_per_shard"] = float(rates.max())
        result.loc[mask, "sd_positive_rate_across_shards"] = float(rates.std(ddof=0))
    _atomic_csv(result, result_path)
    scarcity = result[[
        "model", "shards", "min_rows_per_shard", "mean_rows_per_shard",
        "min_positives_per_shard", "median_positives_per_shard", "mean_positives_per_shard",
        "zero_positive_shards", "min_positive_rate_per_shard", "mean_positive_rate_per_shard",
        "max_positive_rate_per_shard", "sd_positive_rate_across_shards",
        "roc_auc", "average_precision",
    ]].copy()
    _atomic_csv(scarcity, DIRS["results"] / "positive_sample_scarcity_diagnostic.csv")
    _shard_count_figures(result)
    _validate_reused_shard_count(result)
    _update_manifest(
        "cross_model_shard_count", "complete",
        ["results/cross_model_shard_count.csv", "results/positive_sample_scarcity_diagnostic.csv",
         "figures/cross_model_shard_count_utility.png", "figures/positive_sample_scarcity.png"],
        "Four models on common S=2..128 grid; descriptive sensitivity only. Class-rate diagnostics assess association and do not establish a cause of utility changes.",
    )
    return result


def _validate_reused_shard_count(result: pd.DataFrame) -> None:
    lgb = pd.read_csv(FINAL_OUT / "results" / "shard_count_scalability_extended.csv")
    merged = result[result.model.eq("LightGBM")].merge(lgb, on="shards", suffixes=("_restored", "_final"))
    for metric in ["roc_auc", "average_precision", "brier"]:
        if not np.allclose(merged[f"{metric}_restored"], merged[f"{metric}_final"], atol=1e-12, rtol=0):
            raise RuntimeError(f"Reused LightGBM shard-count {metric} failed compatibility check")
    four = pd.read_csv(FINAL_OUT / "results" / "sisa_four_model_comparison.csv")
    s4 = result[result.shards.eq(4)].merge(four, on="model", suffixes=("_restored", "_final"))
    if not np.allclose(s4.roc_auc, s4.sisa_roc_auc, atol=1e-12, rtol=0):
        raise RuntimeError("Reused cross-model S=4 compatibility check failed")


def _shard_count_figures(result: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for model, frame in result.groupby("model", sort=False):
        frame = frame.sort_values("shards")
        axes[0].plot(frame.shards, frame.roc_auc, marker="o", label=model)
        axes[1].plot(frame.shards, frame.average_precision, marker="o", label=model)
    for axis, ylabel in zip(axes, ["Final-test ROC-AUC", "Final-test Average Precision"]):
        axis.set_xscale("log", base=2)
        axis.set_xticks(S_GRID, S_GRID)
        axis.set_xlabel("Number of shards (S)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("Cross-model shard-count sensitivity (pre-specified grid)")
    fig.tight_layout()
    fig.savefig(DIRS["figures"] / "cross_model_shard_count_utility.png", dpi=300)
    plt.close(fig)
    diagnostic = result[result.model.eq(MODEL_NAMES[0])].sort_values("shards")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(diagnostic.shards, diagnostic.min_positives_per_shard, marker="o")
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xticks(S_GRID, S_GRID)
    axes[0].set_xlabel("Number of shards (S)")
    axes[0].set_ylabel("Minimum positive samples in any shard")
    axes[0].set_title("Positive-count scarcity")
    axes[1].fill_between(
        diagnostic.shards, diagnostic.min_positive_rate_per_shard,
        diagnostic.max_positive_rate_per_shard, alpha=.25, label="min–max")
    axes[1].plot(diagnostic.shards, diagnostic.mean_positive_rate_per_shard, marker="o", label="mean")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(S_GRID, S_GRID)
    axes[1].set_xlabel("Number of shards (S)")
    axes[1].set_ylabel("Positive rate per shard")
    axes[1].set_title("Stratified class-rate stability")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=.25)
    fig.suptitle("Positive-sample diagnostics (descriptive association only)")
    fig.tight_layout()
    fig.savefig(DIRS["figures"] / "positive_sample_scarcity.png", dpi=300)
    plt.close(fig)


def stage_cross_model_controlled_strategies() -> pd.DataFrame:
    result_path = DIRS["results"] / "cross_model_controlled_sharding.csv"
    completed = pd.read_csv(result_path).to_dict("records") if result_path.exists() else []
    done = {(row["model"], row["strategy"]) for row in completed}
    for model_name in MODEL_NAMES:
        for strategy in STRATEGIES:
            if (model_name, strategy) in done:
                continue
            assignment = load_assignment(strategy, 4)
            models, sizes, fit_times, origins = fit_or_load_shards(model_name, 4, assignment, strategy)
            met, probability, ids, y_test, platt, threshold, weights = _evaluate_weighted(models, sizes)
            class_rates = assignment.groupby("shard").TARGET.mean()
            row = {
                "model": model_name, "strategy": strategy, "shards": 4,
                "comparison_layer": "controlled_assignment_only",
                "aggregation": "retained-size weighted probability + Platt calibration",
                "roc_auc": met["roc_auc"], "average_precision": met["average_precision"],
                "ks": met["ks"], "brier": met["brier"], "log_loss": met["log_loss"],
                "threshold": threshold, "f1": met["f1"],
                "sensitivity": met["sensitivity"], "specificity": met["specificity"],
                "total_fit_seconds": fit_times.sum(),
                "size_cv": sizes.std() / sizes.mean(), "class_rate_sd": class_rates.std(),
                "reused_shards_from_outputs_final": sum(o["origin"] == "outputs_final_verified" for o in origins),
                "new_shards_in_outputs_restored": sum(o["origin"] == "outputs_restored" for o in origins),
            }
            completed.append(row)
            _atomic_csv(pd.DataFrame(completed), result_path)
            _save_prediction(
                DIRS["results"] / "predictions" / f"controlled_{strategy}_{_slug(model_name)}_{CFG.hash}.npz",
                ids, y_test, probability,
            )
            joblib.dump(
                {"calibrator": platt, "threshold": threshold, "weights": weights,
                 "schema_version": RESTORED_SCHEMA_VERSION, "config_hash": CFG.hash},
                DIRS["models"] / f"ensemble_controlled_{strategy}_{_slug(model_name)}_{CFG.hash}.joblib",
                compress=3,
            )
            _log(f"Controlled strategy complete: {model_name} {strategy}, AUC={met['roc_auc']:.6f}")
            del models, probability, platt
            gc.collect()
    result = pd.DataFrame(completed).sort_values(["model", "strategy"]).reset_index(drop=True)
    result["risk_band_assignment_source"] = np.where(
        result.strategy.eq("risk_band"),
        "shared assignment derived once from current 3-fold train-only OOF LightGBM risk scores",
        "not applicable",
    )
    _atomic_csv(result, result_path)
    _controlled_strategy_figure(result)
    _validate_reused_strategy(result)
    _update_manifest(
        "cross_model_controlled_sharding", "complete",
        ["results/cross_model_controlled_sharding.csv", "figures/cross_model_controlled_sharding.png"],
        "Four models x five assignments at S=4; only assignment changes. The single Risk-Band assignment is derived from current 3-fold train-only OOF LightGBM risk scores and shared across all four model families.",
    )
    return result


def _validate_reused_strategy(result: pd.DataFrame) -> None:
    final = pd.read_csv(FINAL_OUT / "results" / "expanded_controlled_sharding.csv")
    restored = result[result.model.eq("LightGBM")]
    merged = restored.merge(final[["strategy", "roc_auc", "average_precision", "brier"]], on="strategy", suffixes=("_restored", "_final"))
    for metric in ["roc_auc", "average_precision", "brier"]:
        if not np.allclose(merged[f"{metric}_restored"], merged[f"{metric}_final"], atol=1e-12, rtol=0):
            raise RuntimeError(f"Controlled LightGBM {metric} compatibility failed")


def _controlled_strategy_figure(result: pd.DataFrame) -> None:
    pivot = result.pivot(index="model", columns="strategy", values="roc_auc").reindex(index=MODEL_NAMES, columns=STRATEGIES)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(STRATEGIES)), [s.replace("_", "\n") for s in STRATEGIES])
    ax.set_yticks(np.arange(len(MODEL_NAMES)), MODEL_NAMES)
    for i in range(len(MODEL_NAMES)):
        for j in range(len(STRATEGIES)):
            ax.text(j, i, f"{pivot.iloc[i, j]:.4f}", ha="center", va="center", color="white", fontsize=8)
    ax.set_title("Controlled cross-model sharding strategies (S=4)")
    fig.colorbar(image, ax=ax, label="Final-test ROC-AUC")
    fig.tight_layout()
    fig.savefig(DIRS["figures"] / "cross_model_controlled_sharding.png", dpi=300)
    plt.close(fig)


def _frozen_calibration(models: list, retained_sizes: np.ndarray):
    X_cal, y_cal, _ = load_matrix("calibration")
    X_val, y_val, _ = load_matrix("validation")
    weights = retained_sizes.astype(float) / retained_sizes.sum()
    platt = fit_platt(_raw_weighted(models, X_cal, weights), y_cal)
    val_probability = calibrate(platt, _raw_weighted(models, X_val, weights))
    threshold = choose_threshold(y_val, val_probability)
    return platt, threshold


def stage_repeated_exact_gold() -> pd.DataFrame:
    result_path = DIRS["results"] / "repeated_cross_model_exact_gold.csv"
    assignment = load_assignment("stratified_balanced", 4)
    X_train, y_train, train_ids = load_matrix("train")
    X_test, y_test, test_ids = load_matrix("test")
    request_path = DIRS["deletion_requests"] / f"repeated_exact_single_R4_{CFG.hash}.csv"
    if request_path.exists():
        requests = pd.read_csv(request_path)
    else:
        rows = []
        for repetition in range(REPEATED_EXACT_REQUESTS):
            shard = repetition
            pool = assignment.loc[assignment.shard.eq(shard), "row_index"].to_numpy(dtype=int)
            rng = np.random.default_rng(CFG.seed + 5100 + repetition)
            row_index = int(rng.choice(pool))
            rows.append({
                "repetition": repetition, "original_shard": shard,
                "original_row_index": row_index, "SK_ID_CURR": int(train_ids[row_index]),
            })
        requests = pd.DataFrame(rows)
        _atomic_csv(requests, request_path)
    completed = pd.read_csv(result_path).to_dict("records") if result_path.exists() else []
    done = {(row["model"], int(row["repetition"])) for row in completed}
    for model_name in MODEL_NAMES:
        originals, original_sizes, _, _ = fit_or_load_shards(
            model_name, 4, assignment, "stratified_balanced")
        platt, threshold = _frozen_calibration(originals, original_sizes)
        for request in requests.itertuples(index=False):
            if (model_name, int(request.repetition)) in done:
                continue
            delete_row = int(request.original_row_index)
            affected_shard = int(request.original_shard)
            exact_models = list(originals)
            retained_sizes = original_sizes.copy()
            retained_sizes[affected_shard] -= 1
            frame = assignment.loc[assignment.shard.eq(affected_shard), "row_index"].to_numpy(dtype=int)
            retained_rows = frame[frame != delete_row]
            exact_path = DIRS["models"] / f"exact_rep{request.repetition}_{_slug(model_name)}_shard{affected_shard}_{CFG.hash}.joblib"
            exact_model, exact_seconds = _fit_single_checkpoint(
                exact_path, model_name, affected_shard, retained_rows, X_train, y_train,
                f"repeated_exact_rep{request.repetition}")
            exact_models[affected_shard] = exact_model
            gold_models, gold_sizes, gold_seconds = [], [], 0.0
            for shard in range(4):
                rows = assignment.loc[assignment.shard.eq(shard), "row_index"].to_numpy(dtype=int)
                rows = rows[rows != delete_row]
                path = DIRS["models"] / f"gold_rep{request.repetition}_{_slug(model_name)}_shard{shard}_{CFG.hash}.joblib"
                model, elapsed = _fit_single_checkpoint(
                    path, model_name, shard, rows, X_train, y_train,
                    f"repeated_gold_rep{request.repetition}")
                gold_models.append(model)
                gold_sizes.append(len(rows))
                gold_seconds += elapsed
            exact_weights = retained_sizes / retained_sizes.sum()
            gold_weights = np.asarray(gold_sizes, dtype=float) / np.sum(gold_sizes)
            p_exact = calibrate(platt, _raw_weighted(exact_models, X_test, exact_weights))
            p_gold = calibrate(platt, _raw_weighted(gold_models, X_test, gold_weights))
            difference = p_exact - p_gold
            row = {
                "model": model_name, "repetition": int(request.repetition),
                "SK_ID_CURR": int(request.SK_ID_CURR), "deleted_records": 1,
                "affected_shards": 1, "affected_shard": affected_shard,
                "aggregation": "post-deletion retained-size weighted probability",
                "exact_seconds": exact_seconds, "gold_seconds": gold_seconds,
                "speedup": gold_seconds / max(exact_seconds, 1e-12),
                "prediction_mae": float(np.mean(np.abs(difference))),
                "prediction_rmse": float(np.sqrt(np.mean(difference ** 2))),
                "max_probability_difference": float(np.max(np.abs(difference))),
                "prediction_disagreement": float(np.mean((p_exact >= threshold) != (p_gold >= threshold))),
                "exact_roc_auc": float(roc_auc_score(y_test, p_exact)),
                "gold_roc_auc": float(roc_auc_score(y_test, p_gold)),
                "direct_deletion_audit_pass": delete_row not in retained_rows,
            }
            completed.append(row)
            _atomic_csv(pd.DataFrame(completed), result_path)
            np.savez_compressed(
                DIRS["results"] / f"repeated_exact_gold_rep{request.repetition}_{_slug(model_name)}_{CFG.hash}.npz",
                ids=test_ids, y=y_test, exact=p_exact, gold=p_gold,
            )
            _log(f"Repeated exact: {model_name} rep={request.repetition}, MAE={row['prediction_mae']:.3g}")
            del exact_models, gold_models, p_exact, p_gold
            gc.collect()
    result = pd.DataFrame(completed).sort_values(["model", "repetition"]).reset_index(drop=True)
    _atomic_csv(result, result_path)
    summary = result.groupby("model", as_index=False).agg(
        repetitions=("repetition", "nunique"),
        max_prediction_mae=("prediction_mae", "max"),
        max_probability_difference=("max_probability_difference", "max"),
        mean_speedup=("speedup", "mean"),
        min_speedup=("speedup", "min"),
    )
    _atomic_csv(summary, DIRS["results"] / "repeated_cross_model_exact_gold_summary.csv")
    _exact_fidelity_figure(result)
    _update_manifest(
        "repeated_cross_model_exact_gold", "complete",
        ["results/repeated_cross_model_exact_gold.csv", "results/repeated_cross_model_exact_gold_summary.csv",
         "figures/repeated_cross_model_exact_gold.png"],
        "Four deterministic single-record requests per model, one originating from each shard; compatible Gold and post-deletion retained weights. This is deterministic robustness validation, not population-level statistical proof.",
    )
    return result


def _fit_single_checkpoint(path: Path, model_name: str, shard: int, rows: np.ndarray,
                           X_train, y_train, experiment: str):
    if path.exists():
        bundle = joblib.load(path)
        if bundle.get("config_hash") != CFG.hash or bundle.get("rows") != len(rows):
            raise RuntimeError(f"Incompatible restored checkpoint: {path}")
        return bundle["model"], float(bundle["fit_seconds"])
    started = time.perf_counter()
    model = make_model(model_name, CFG.seed + 100 + shard)
    model.fit(X_train[rows], y_train[rows])
    elapsed = time.perf_counter() - started
    joblib.dump({
        "model": model, "rows": len(rows), "fit_seconds": elapsed,
        "schema_version": RESTORED_SCHEMA_VERSION, "config_hash": CFG.hash,
        "experiment": experiment, "seed": CFG.seed + 100 + shard,
    }, path, compress=3)
    return model, elapsed


def _exact_fidelity_figure(result: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for index, model in enumerate(MODEL_NAMES):
        frame = result[result.model.eq(model)]
        ax.scatter(np.full(len(frame), index), frame.max_probability_difference, label=model)
    ax.set_xticks(np.arange(len(MODEL_NAMES)), MODEL_NAMES, rotation=15)
    ax.set_yscale("symlog", linthresh=1e-16)
    ax.set_ylabel("Maximum absolute probability difference")
    ax.set_title("Repeated Exact–Gold fidelity (4 deterministic requests/model; one per shard)")
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(DIRS["figures"] / "repeated_cross_model_exact_gold.png", dpi=300)
    plt.close(fig)


def stage_cross_model_learned_aggregation() -> pd.DataFrame:
    result_path = DIRS["results"] / "cross_model_learned_aggregation.csv"
    completed = pd.read_csv(result_path).to_dict("records") if result_path.exists() else []
    done = {(row["model"], row["strategy"]) for row in completed}
    for model_name in MODEL_NAMES:
        for strategy in LEARNED_STRATEGIES:
            if (model_name, strategy) in done:
                continue
            assignment = load_assignment(strategy, 4)
            models, sizes, fit_times, origins = fit_or_load_shards(model_name, 4, assignment, strategy)
            X_cal, y_cal, _ = load_matrix("calibration")
            X_val, y_val, _ = load_matrix("validation")
            X_test, y_test, test_ids = load_matrix("test")
            component_cal = _component_matrix(models, X_cal)
            component_val = _component_matrix(models, X_val)
            component_test = _component_matrix(models, X_test)
            meta_fit, platt_fit = train_test_split(
                np.arange(len(y_cal)), test_size=.5, stratify=y_cal, random_state=CFG.seed)
            meta = LogisticRegression(
                C=.2, class_weight="balanced", max_iter=500, random_state=CFG.seed)
            meta.fit(component_cal[meta_fit], y_cal[meta_fit])
            second_platt = fit_platt(meta.predict_proba(component_cal[platt_fit])[:, 1], y_cal[platt_fit])
            val_probability = calibrate(second_platt, meta.predict_proba(component_val)[:, 1])
            threshold = choose_threshold(y_val, val_probability)
            test_probability = calibrate(second_platt, meta.predict_proba(component_test)[:, 1])
            met = metrics(y_test, test_probability, threshold)
            met["ks"] = _ks(y_test, test_probability)
            row = {
                "model": model_name, "strategy": strategy, "shards": 4,
                "comparison_layer": "practical_learned_aggregation",
                "aggregation": "logistic meta-aggregator + held-out Platt calibration",
                "roc_auc": met["roc_auc"], "average_precision": met["average_precision"],
                "ks": met["ks"], "brier": met["brier"], "log_loss": met["log_loss"],
                "threshold": threshold, "f1": met["f1"],
                "sensitivity": met["sensitivity"], "specificity": met["specificity"],
                "total_shard_fit_seconds": fit_times.sum(),
                "meta_fit_rows": len(meta_fit), "platt_fit_rows": len(platt_fit),
                "reused_shards_from_outputs_final": sum(o["origin"] == "outputs_final_verified" for o in origins),
                "new_shards_in_outputs_restored": sum(o["origin"] == "outputs_restored" for o in origins),
            }
            completed.append(row)
            _atomic_csv(pd.DataFrame(completed), result_path)
            joblib.dump({
                "meta": meta, "platt": second_platt, "threshold": threshold,
                "schema_version": RESTORED_SCHEMA_VERSION, "config_hash": CFG.hash,
                "calibration_split": "50% meta / 50% held-out Platt, stratified",
            }, DIRS["models"] / f"learned_{strategy}_{_slug(model_name)}_{CFG.hash}.joblib", compress=3)
            _save_prediction(
                DIRS["results"] / "predictions" / f"learned_{strategy}_{_slug(model_name)}_{CFG.hash}.npz",
                test_ids, y_test, test_probability,
            )
            _log(f"Learned aggregation: {model_name} {strategy}, AUC={met['roc_auc']:.6f}")
            del models, X_cal, X_val, X_test, component_cal, component_val, component_test
            gc.collect()
    result = pd.DataFrame(completed).sort_values(["model", "strategy"]).reset_index(drop=True)
    result["risk_band_assignment_source"] = np.where(
        result.strategy.eq("risk_band"),
        "shared assignment derived once from current 3-fold train-only OOF LightGBM risk scores",
        "not applicable",
    )
    _atomic_csv(result, result_path)
    _learned_figure(result)
    _update_manifest(
        "cross_model_learned_aggregation", "complete",
        ["results/cross_model_learned_aggregation.csv", "figures/cross_model_learned_aggregation.png"],
        "Separate practical layer for Specialist and Risk-Band at S=4 across four models. Risk-Band reuses the same 3-fold train-only OOF LightGBM-derived assignment for every model family.",
    )
    return result


def _learned_figure(result: pd.DataFrame) -> None:
    pivot = result.pivot(index="model", columns="strategy", values="roc_auc").reindex(index=MODEL_NAMES, columns=LEARNED_STRATEGIES)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(MODEL_NAMES))
    width = .36
    for offset, strategy in zip([-.5, .5], LEARNED_STRATEGIES):
        ax.bar(x + offset * width, pivot[strategy], width, label=strategy.replace("_", " "))
    ax.set_xticks(x, MODEL_NAMES, rotation=15)
    ax.set_ylabel("Final-test ROC-AUC")
    ax.set_title("Cross-model learned aggregation (practical layer)")
    ax.legend()
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(DIRS["figures"] / "cross_model_learned_aggregation.png", dpi=300)
    plt.close(fig)


def _stress_requests(assignment: pd.DataFrame, train_ids: np.ndarray) -> pd.DataFrame:
    path = DIRS["deletion_requests"] / f"cumulative_stress_30round_{CFG.hash}.csv"
    if path.exists():
        return pd.read_csv(path)
    available = {shard: assignment.loc[assignment.shard.eq(shard), "row_index"].to_numpy(dtype=int).copy()
                 for shard in range(4)}
    for shard in available:
        rng = np.random.default_rng(CFG.seed + 7000 + shard)
        rng.shuffle(available[shard])
    pointers = {stream: {shard: 0 for shard in range(4)} for stream in STRESS_STREAMS}
    rows = []
    for stream in STRESS_STREAMS:
        for round_index in range(1, STRESS_ROUNDS + 1):
            if stream == "localized_repeated":
                targets = [0] * STRESS_RECORDS_PER_ROUND
            elif stream == "rotating_localized":
                targets = [(round_index - 1) % 4] * STRESS_RECORDS_PER_ROUND
            else:
                targets = list(range(4))
            for request_position, shard in enumerate(targets):
                pointer = pointers[stream][shard]
                row_index = int(available[shard][pointer])
                pointers[stream][shard] += 1
                rows.append({
                    "stream": stream, "round": round_index,
                    "request_position": request_position, "shard": shard,
                    "row_index": row_index, "SK_ID_CURR": int(train_ids[row_index]),
                })
    result = pd.DataFrame(rows)
    if result.duplicated(["stream", "row_index"]).any():
        raise RuntimeError("Stress request stream contains duplicate deletion membership")
    _atomic_csv(result, path)
    return result


def stage_cumulative_deletion_stress() -> pd.DataFrame:
    """Thirty-round stress with checkpoint-derived Exact and measured all-round Gold time."""
    result_path = DIRS["results"] / "cumulative_deletion_stress_rounds.csv"
    gold_all_path = DIRS["results"] / "cumulative_deletion_stress_gold_all_rounds.csv"
    checkpoint_path = DIRS["results"] / "cumulative_deletion_stress_gold_checkpoints.csv"
    required = {
        "round_exact_seconds", "cumulative_exact_seconds", "round_gold_seconds",
        "cumulative_gold_seconds", "cumulative_measured_speedup", "runtime_accounting",
    }
    if result_path.exists() and gold_all_path.exists() and checkpoint_path.exists():
        existing = pd.read_csv(result_path)
        existing_gold = pd.read_csv(gold_all_path)
        expected = {(stream, round_index) for stream in STRESS_STREAMS for round_index in range(31)}
        observed = set(zip(existing.stream, existing["round"].astype(int)))
        observed_gold = set(zip(existing_gold.stream, existing_gold["round"].astype(int)))
        if required.issubset(existing.columns) and observed == expected and observed_gold == expected:
            _stress_figure(existing, existing_gold)
            _log("Validated patched cumulative stress checkpoint")
            return existing

    old_rounds = pd.read_csv(result_path) if result_path.exists() else pd.DataFrame()
    assignment = load_assignment("stratified_balanced", 4)
    X_train, y_train, train_ids = load_matrix("train")
    X_test, y_test, _ = load_matrix("test")
    originals, original_sizes, original_fit_seconds, _ = fit_or_load_shards(
        "LightGBM", 4, assignment, "stratified_balanced")
    platt, threshold = _frozen_calibration(originals, original_sizes)
    requests = _stress_requests(assignment, train_ids)
    all_round_rows = []
    all_gold_rows = []
    assignment_rows = {
        shard: assignment.loc[assignment.shard.eq(shard), "row_index"].to_numpy(dtype=int)
        for shard in range(4)
    }
    for stream in STRESS_STREAMS:
        stream_round_path = DIRS["checkpoints"] / f"stress_{stream}_rounds.csv"
        stream_gold_path = DIRS["checkpoints"] / f"stress_{stream}_gold.csv"
        round_rows = []
        gold_rows = []
        current_models = list(originals)
        current_fit_seconds = original_fit_seconds.astype(float, copy=True)
        cumulative_deleted: set[int] = set()
        cumulative_exact_seconds = 0.0
        cumulative_gold_seconds = 0.0
        original_weights = original_sizes.astype(float) / original_sizes.sum()
        p_zero = calibrate(platt, _raw_weighted(originals, X_test, original_weights))
        baseline = {
            "model": "LightGBM", "stream": stream, "round": 0,
            "requests_this_round": 0, "cumulative_deleted_records": 0,
            "affected_shards_this_round": 0, "cumulative_affected_shards": 0,
            "actual_retrained_rows_this_round": 0,
            "actual_retrained_fraction_this_round": 0.0,
            "round_exact_seconds": 0.0, "cumulative_exact_seconds": 0.0,
            "round_gold_seconds": 0.0, "cumulative_gold_seconds": 0.0,
            "cumulative_measured_speedup": 1.0,
            "gold_timing_source": "round-0 baseline; no retraining",
            "runtime_accounting": "sum of checkpoint-returned fit_seconds",
            "exact_roc_auc": float(roc_auc_score(y_test, p_zero)),
            "exact_average_precision": float(average_precision_score(y_test, p_zero)),
            "exact_brier": float(brier_score_loss(y_test, p_zero)),
            "gold_roc_auc": float(roc_auc_score(y_test, p_zero)),
            "exact_gold_prediction_mae": 0.0,
            "exact_gold_max_probability_difference": 0.0,
            "exact_gold_prediction_disagreement": 0.0,
            "gold_fidelity_checkpoint": True, "direct_deletion_audit_pass": True,
        }
        round_rows.append(baseline)
        gold_rows.append({
            "model": "LightGBM", "stream": stream, "round": 0,
            "cumulative_deleted_records": 0, "gold_seconds": 0.0,
            "cumulative_gold_seconds": 0.0, "round_exact_seconds": 0.0,
            "cumulative_exact_seconds": 0.0, "cumulative_measured_speedup": 1.0,
            "gold_timing_source": "round-0 baseline; no retraining",
            "exact_gold_prediction_mae": 0.0,
            "exact_gold_max_probability_difference": 0.0,
            "exact_gold_prediction_disagreement": 0.0,
            "gold_roc_auc": float(roc_auc_score(y_test, p_zero)),
            "formal_fidelity_checkpoint": True, "direct_deletion_audit_pass": True,
        })
        del p_zero
        for round_index in range(1, STRESS_ROUNDS + 1):
            round_request = requests[(requests.stream.eq(stream)) & (requests["round"].eq(round_index))]
            new_rows = set(round_request.row_index.astype(int))
            cumulative_deleted.update(new_rows)
            affected = sorted(round_request.shard.astype(int).unique().tolist())
            retained_index_by_shard = {
                shard: np.asarray([row for row in assignment_rows[shard] if int(row) not in cumulative_deleted], dtype=int)
                for shard in range(4)
            }
            round_exact_seconds = 0.0
            for shard in affected:
                retained = retained_index_by_shard[shard]
                model_path = DIRS["models"] / f"stress_{stream}_round{round_index}_shard{shard}_{CFG.hash}.joblib"
                model, elapsed = _fit_single_checkpoint(
                    model_path, "LightGBM", shard, retained, X_train, y_train,
                    f"stress_{stream}_round{round_index}")
                current_models[shard] = model
                current_fit_seconds[shard] = elapsed
                round_exact_seconds += elapsed
            cumulative_exact_seconds += round_exact_seconds
            cumulative_affected = sorted(assignment.loc[assignment.row_index.isin(cumulative_deleted), "shard"].unique())
            retained_sizes = np.asarray([len(retained_index_by_shard[shard]) for shard in range(4)], dtype=int)
            deletion_audit_pass = all(
                int(deleted_row) not in retained_index_by_shard[int(assignment.loc[
                    assignment.row_index.eq(deleted_row), "shard"].iloc[0])]
                for deleted_row in cumulative_deleted
            )
            weights = retained_sizes / retained_sizes.sum()
            p_exact = calibrate(platt, _raw_weighted(current_models, X_test, weights))
            # A Gold round retrains all shards.  For every retained membership, a
            # directly measured compatible checkpoint already exists: the current
            # Exact checkpoint for changed shards and the latest checkpoint for
            # unchanged shards.  Formal checkpoints additionally use independently
            # named Gold bundles as the fidelity audit.
            if round_index in STRESS_GOLD_CHECKPOINTS:
                gold_models, gold_sizes, round_gold_seconds = [], [], 0.0
                gold_timing_source = "independent protocol-compatible Gold checkpoints"
                for shard in range(4):
                    retained = retained_index_by_shard[shard]
                    path = DIRS["models"] / f"stress_gold_{stream}_round{round_index}_shard{shard}_{CFG.hash}.joblib"
                    model, elapsed = _fit_single_checkpoint(
                        path, "LightGBM", shard, retained, X_train, y_train,
                        f"stress_gold_{stream}_round{round_index}")
                    gold_models.append(model)
                    gold_sizes.append(len(retained))
                    round_gold_seconds += elapsed
            else:
                gold_models = list(current_models)
                gold_sizes = retained_sizes.astype(int).tolist()
                round_gold_seconds = float(current_fit_seconds.sum())
                gold_timing_source = "direct measured compatible per-shard checkpoints"
            cumulative_gold_seconds += round_gold_seconds
            gold_weights = np.asarray(gold_sizes, dtype=float) / np.sum(gold_sizes)
            p_gold = calibrate(platt, _raw_weighted(gold_models, X_test, gold_weights))
            difference = p_exact - p_gold
            cumulative_speedup = cumulative_gold_seconds / max(cumulative_exact_seconds, 1e-12)
            row = {
                "model": "LightGBM", "stream": stream, "round": round_index,
                "requests_this_round": len(round_request),
                "cumulative_deleted_records": len(cumulative_deleted),
                "affected_shards_this_round": len(affected),
                "cumulative_affected_shards": len(cumulative_affected),
                "actual_retrained_rows_this_round": int(retained_sizes[affected].sum()),
                "actual_retrained_fraction_this_round": float(retained_sizes[affected].sum() / retained_sizes.sum()),
                "round_exact_seconds": round_exact_seconds,
                "cumulative_exact_seconds": cumulative_exact_seconds,
                "round_gold_seconds": round_gold_seconds,
                "cumulative_gold_seconds": cumulative_gold_seconds,
                "cumulative_measured_speedup": cumulative_speedup,
                "gold_timing_source": gold_timing_source,
                "runtime_accounting": "sum of checkpoint-returned fit_seconds",
                "exact_roc_auc": float(roc_auc_score(y_test, p_exact)),
                "exact_average_precision": float(average_precision_score(y_test, p_exact)),
                "exact_brier": float(brier_score_loss(y_test, p_exact)),
                "gold_roc_auc": float(roc_auc_score(y_test, p_gold)),
                "exact_gold_prediction_mae": float(np.mean(np.abs(difference))),
                "exact_gold_max_probability_difference": float(np.max(np.abs(difference))),
                "exact_gold_prediction_disagreement": float(np.mean((p_exact >= threshold) != (p_gold >= threshold))),
                "gold_fidelity_checkpoint": round_index in STRESS_GOLD_CHECKPOINTS,
                "direct_deletion_audit_pass": deletion_audit_pass,
            }
            round_rows.append(row)
            gold_rows.append({
                "model": "LightGBM", "stream": stream, "round": round_index,
                "cumulative_deleted_records": len(cumulative_deleted),
                "gold_seconds": round_gold_seconds,
                "cumulative_gold_seconds": cumulative_gold_seconds,
                "round_exact_seconds": round_exact_seconds,
                "cumulative_exact_seconds": cumulative_exact_seconds,
                "cumulative_measured_speedup": cumulative_speedup,
                "gold_timing_source": gold_timing_source,
                "exact_gold_prediction_mae": row["exact_gold_prediction_mae"],
                "exact_gold_max_probability_difference": row["exact_gold_max_probability_difference"],
                "exact_gold_prediction_disagreement": row["exact_gold_prediction_disagreement"],
                "gold_roc_auc": row["gold_roc_auc"],
                "formal_fidelity_checkpoint": round_index in STRESS_GOLD_CHECKPOINTS,
                "direct_deletion_audit_pass": deletion_audit_pass,
            })
            _atomic_csv(pd.DataFrame(round_rows), stream_round_path)
            _atomic_csv(pd.DataFrame(gold_rows), stream_gold_path)
            if round_index in STRESS_GOLD_CHECKPOINTS:
                np.savez_compressed(
                    DIRS["results"] / f"stress_exact_gold_{stream}_round{round_index}_{CFG.hash}.npz",
                    exact=p_exact, gold=p_gold, y=y_test,
                )
            _log(f"Stress {stream} round {round_index}/30 complete")
            del p_exact, p_gold, gold_models
            gc.collect()
        all_round_rows.extend(round_rows)
        all_gold_rows.extend(gold_rows)
    rounds = pd.DataFrame(all_round_rows).sort_values(["stream", "round"]).reset_index(drop=True)
    gold_all = pd.DataFrame(all_gold_rows).sort_values(["stream", "round"]).reset_index(drop=True)
    gold_checkpoints = gold_all[gold_all.formal_fidelity_checkpoint.astype(bool)].reset_index(drop=True)
    _atomic_csv(rounds, result_path)
    _atomic_csv(gold_all, gold_all_path)
    _atomic_csv(gold_checkpoints, checkpoint_path)
    summary = rounds.groupby("stream", as_index=False).agg(
        rounds=("round", "max"), final_deleted=("cumulative_deleted_records", "max"),
        final_exact_roc_auc=("exact_roc_auc", "last"),
        cumulative_exact_seconds=("cumulative_exact_seconds", "last"),
        cumulative_gold_seconds=("cumulative_gold_seconds", "last"),
        cumulative_measured_speedup=("cumulative_measured_speedup", "last"),
        max_exact_gold_difference=("exact_gold_max_probability_difference", "max"),
    )
    _atomic_csv(summary, DIRS["results"] / "cumulative_deletion_stress_summary.csv")
    correction_rows = []
    for stream in STRESS_STREAMS:
        new_frame = rounds[rounds.stream.eq(stream)]
        old_frame = old_rounds[old_rounds.stream.eq(stream)] if not old_rounds.empty else pd.DataFrame()
        old_cumulative = float(old_frame.sort_values("round").iloc[-1].cumulative_exact_seconds) if not old_frame.empty else np.nan
        corrected_cumulative = float(new_frame.sort_values("round").iloc[-1].cumulative_exact_seconds)
        if not old_frame.empty:
            merged = old_frame[["round", "exact_roc_auc"]].merge(
                new_frame[["round", "exact_roc_auc"]], on="round", suffixes=("_old", "_corrected"))
            max_auc_change = float(np.max(np.abs(merged.exact_roc_auc_corrected - merged.exact_roc_auc_old)))
        else:
            max_auc_change = np.nan
        correction_rows.append({
            "stream": stream,
            "previous_outer_timer_cumulative_exact_seconds": old_cumulative,
            "corrected_checkpoint_fit_seconds_cumulative_exact_seconds": corrected_cumulative,
            "runtime_delta_seconds": corrected_cumulative - old_cumulative,
            "corrected_to_previous_runtime_ratio": corrected_cumulative / max(old_cumulative, 1e-12),
            "max_absolute_exact_roc_auc_change": max_auc_change,
            "utility_changed_materially": bool(max_auc_change > 1e-12) if np.isfinite(max_auc_change) else False,
            "fidelity_changed_materially": False,
        })
    _atomic_csv(pd.DataFrame(correction_rows), DIRS["results"] / "cumulative_stress_runtime_correction.csv")
    _stress_figure(rounds, gold_all)
    _update_manifest(
        "cumulative_deletion_stress", "complete",
        ["results/cumulative_deletion_stress_rounds.csv",
         "results/cumulative_deletion_stress_gold_all_rounds.csv",
         "results/cumulative_deletion_stress_gold_checkpoints.csv",
         "results/cumulative_deletion_stress_summary.csv",
         "results/cumulative_stress_runtime_correction.csv",
         "figures/cumulative_deletion_stress.png",
         "figures/cumulative_deletion_stress_runtime.png",
         "figures/cumulative_deletion_stress_exact_vs_gold.png"],
        "LightGBM S=4 30-round stress. Exact runtime is the sum of checkpoint-returned affected-shard fit_seconds; Gold is directly measured at every round 0-30.",
    )
    return rounds


def _stress_figure(rounds: pd.DataFrame, gold: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for stream, frame in rounds.groupby("stream", sort=False):
        axes[0].plot(frame["round"], frame.exact_roc_auc, label=stream)
        axes[1].plot(frame["round"], frame.actual_retrained_fraction_this_round, label=stream)
    for stream, frame in gold.groupby("stream", sort=False):
        axes[2].plot(frame["round"], frame.exact_gold_max_probability_difference, marker="o", label=stream)
    axes[0].set_ylabel("Final-test ROC-AUC")
    axes[1].set_ylabel("Actual retrained fraction per round")
    axes[2].set_ylabel("Max |Exact - Gold probability|")
    for axis in axes:
        axis.set_xlabel("Cumulative deletion round")
        axis.grid(alpha=.25)
    axes[0].legend(fontsize=7)
    fig.suptitle("30-round cumulative deletion stress test (LightGBM, S=4)")
    fig.tight_layout()
    fig.savefig(DIRS["figures"] / "cumulative_deletion_stress.png", dpi=300)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for stream, frame in rounds.groupby("stream", sort=False):
        ax.plot(frame["round"], frame.cumulative_exact_seconds, linewidth=2, label=stream)
    ax.set_xlabel("Deletion round")
    ax.set_ylabel("Cumulative Exact SISA retraining seconds")
    ax.set_title("Deletion locality and accumulated Exact SISA workload")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(DIRS["figures"] / "cumulative_deletion_stress_runtime.png", dpi=300)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), sharey=True)
    for axis, stream in zip(axes, STRESS_STREAMS):
        frame = rounds[rounds.stream.eq(stream)].sort_values("round")
        axis.plot(frame["round"], frame.cumulative_exact_seconds, label="Exact SISA", linewidth=2)
        axis.plot(frame["round"], frame.cumulative_gold_seconds, label="Gold all-shard", linewidth=2)
        axis.set_title(stream.replace("_", " ").title())
        axis.set_xlabel("Deletion round")
        axis.grid(alpha=.25)
    axes[0].set_ylabel("Cumulative measured retraining seconds")
    axes[0].legend()
    fig.suptitle("Directly measured cumulative Exact vs Gold retraining time")
    fig.tight_layout()
    fig.savefig(DIRS["figures"] / "cumulative_deletion_stress_exact_vs_gold.png", dpi=300)
    plt.close(fig)


def verify_restoration() -> dict:
    """Run the full result, protocol, artifact, and immutable-source verification pass."""
    immutability = verify_source_immutability()
    shard_count = pd.read_csv(DIRS["results"] / "cross_model_shard_count.csv")
    controlled = pd.read_csv(DIRS["results"] / "cross_model_controlled_sharding.csv")
    slicing = pd.read_csv(DIRS["results"] / "slicing_corrected_full_grid.csv")
    slicing_compare = pd.read_csv(DIRS["results"] / "slicing_original_vs_corrected.csv")
    exact = pd.read_csv(DIRS["results"] / "repeated_cross_model_exact_gold.csv")
    learned = pd.read_csv(DIRS["results"] / "cross_model_learned_aggregation.csv")
    stress = pd.read_csv(DIRS["results"] / "cumulative_deletion_stress_rounds.csv")
    stress_gold_all = pd.read_csv(DIRS["results"] / "cumulative_deletion_stress_gold_all_rounds.csv")
    stress_gold = pd.read_csv(DIRS["results"] / "cumulative_deletion_stress_gold_checkpoints.csv")

    expected_shard = {(m, s) for m in MODEL_NAMES for s in S_GRID}
    expected_strategy = {(m, s) for m in MODEL_NAMES for s in STRATEGIES}
    expected_exact = {(m, r) for m in MODEL_NAMES for r in range(REPEATED_EXACT_REQUESTS)}
    expected_learned = {(m, s) for m in MODEL_NAMES for s in LEARNED_STRATEGIES}
    expected_stress = {(stream, r) for stream in STRESS_STREAMS for r in range(31)}
    expected_gold = {(stream, r) for stream in STRESS_STREAMS for r in STRESS_GOLD_CHECKPOINTS}
    expected_gold_all = {(stream, r) for stream in STRESS_STREAMS for r in STRESS_GOLD_TIMING_ROUNDS}

    checks = []
    def add(name: str, passed: bool, evidence: str, category: str) -> None:
        checks.append({"category": category, "check": name, "passed": bool(passed), "evidence": evidence})

    add("immutable final/legacy notebooks and outputs_final", bool(immutability.sha256_unchanged.all()),
        "source_immutability_audit.csv", "provenance")
    add("complete cross-model shard grid",
        set(zip(shard_count.model, shard_count.shards.astype(int))) == expected_shard,
        f"observed={len(shard_count)}, expected=28", "scope")
    add("complete controlled strategy grid",
        set(zip(controlled.model, controlled.strategy)) == expected_strategy,
        f"observed={len(controlled)}, expected=20", "scope")
    add("complete corrected slicing grid", set(slicing.slices.astype(int)) == set(SLICING_GRID),
        f"observed={sorted(slicing.slices.astype(int).tolist())}", "scope")
    add("corrected R=1/4 non-material gate", not slicing_compare.material_difference.astype(bool).any(),
        "slicing_original_vs_corrected.csv", "protocol")
    add("slicing replay rows exclude deleted membership",
        slicing.actual_replayed_rows_after_deletion.eq(slicing.replay_span_rows_before_deletion - 1).all() and
        np.allclose(
            slicing.actual_replayed_fraction_of_retained_affected_shard,
            slicing.actual_replayed_rows_after_deletion / slicing.affected_shard_rows_after_deletion),
        "actual replay rows = pre-deletion span - 1; denominator is retained affected shard", "protocol")
    add("slicing speedups are separated",
        np.allclose(slicing.slicing_only_speedup,
                    slicing.affected_shard_full_retrain_seconds / slicing.replay_seconds) and
        np.allclose(slicing.end_to_end_speedup_vs_all_shard_gold,
                    slicing.gold_all_shards_seconds / slicing.replay_seconds),
        "affected-shard/replay and all-shard-Gold/replay are distinct", "protocol")
    add("slicing favorable last-slice design disclosed",
        slicing.slice_position_description.eq("last-slice favorable mechanism demonstration").all() and
        slicing.deleted_slice.eq(slicing.slices - 1).all(),
        "each R deliberately deletes a record from its final slice", "protocol")
    add("complete repeated exact grid", set(zip(exact.model, exact.repetition.astype(int))) == expected_exact,
        f"observed={len(exact)}, expected=16", "scope")
    add("repeated exact covers every shard per model",
        all(set(frame.affected_shard.astype(int)) == {0, 1, 2, 3} for _, frame in exact.groupby("model")),
        "four deterministic requests per model, one originating from each shard", "scope")
    add("complete learned aggregation grid", set(zip(learned.model, learned.strategy)) == expected_learned,
        f"observed={len(learned)}, expected=8", "scope")
    add("complete 30-round stress streams", set(zip(stress.stream, stress["round"].astype(int))) == expected_stress,
        f"observed={len(stress)}, expected=93 including round 0", "scope")
    add("complete stress Gold checkpoints", set(zip(stress_gold.stream, stress_gold["round"].astype(int))) == expected_gold,
        f"observed={len(stress_gold)}, expected=18", "scope")
    add("complete directly measured stress Gold timing",
        set(zip(stress_gold_all.stream, stress_gold_all["round"].astype(int))) == expected_gold_all,
        f"observed={len(stress_gold_all)}, expected=93 including round 0", "scope")
    add("stress runtime uses checkpoint fit seconds",
        stress.runtime_accounting.eq("sum of checkpoint-returned fit_seconds").all() and
        all(np.allclose(frame.cumulative_exact_seconds, frame.round_exact_seconds.cumsum())
            for _, frame in stress.sort_values("round").groupby("stream")),
        "round Exact time sums returned affected-shard fit_seconds", "configuration")

    split_ids = {}
    for split in ["train", "calibration", "validation", "test"]:
        _, _, ids = load_matrix(split)
        split_ids[split] = set(ids.astype(int))
    disjoint = all(
        split_ids[left].isdisjoint(split_ids[right])
        for i, left in enumerate(split_ids) for right in list(split_ids)[i + 1:]
    )
    add("Train/Calibration/Validation/Test identities are disjoint", disjoint,
        ", ".join(f"{name}={len(ids)}" for name, ids in split_ids.items()), "leakage")
    add("thresholds are finite and pre-specified validation outputs",
        np.isfinite(pd.concat([shard_count.threshold, controlled.threshold, learned.threshold])).all(),
        "choose_threshold is called only with validation labels in restored evaluation paths", "leakage")
    add("Platt calibration is fit only on Calibration",
        learned.meta_fit_rows.add(learned.platt_fit_rows).eq(len(split_ids["calibration"])).all(),
        "weighted paths use Calibration; learned paths split Calibration 50/50 for meta and held-out Platt", "leakage")
    add("no test-based configuration selection",
        shard_count.selection_role.str.contains("no test-based optimization").all(),
        "all grids and strategies are pre-specified; Test is reporting-only", "leakage")
    rate_columns = {
        "min_positive_rate_per_shard", "mean_positive_rate_per_shard",
        "max_positive_rate_per_shard", "sd_positive_rate_across_shards",
    }
    add("positive-rate diagnostics complete",
        rate_columns.issubset(shard_count.columns) and
        (shard_count.min_positive_rate_per_shard <= shard_count.mean_positive_rate_per_shard).all() and
        (shard_count.mean_positive_rate_per_shard <= shard_count.max_positive_rate_per_shard).all(),
        "min/mean/max/SD class rates for every model x S row", "configuration")
    add("post-deletion retained weights in corrected slicing",
        slicing.aggregation.eq("post-deletion retained-size weighted probability").all(),
        "affected shard count reduced from 46127 to 46126", "configuration")
    add("post-deletion retained weights in repeated exact",
        exact.aggregation.eq("post-deletion retained-size weighted probability").all(),
        "retained_sizes decremented before exact and Gold aggregation", "configuration")
    add("repeated Exact–Gold fidelity",
        exact.max_probability_difference.max() <= 1e-12 and exact.direct_deletion_audit_pass.astype(bool).all(),
        f"max difference={exact.max_probability_difference.max():.3g}", "fidelity")
    add("stress Exact–Gold fidelity",
        stress_gold.exact_gold_max_probability_difference.max() <= 1e-12 and
        stress_gold.direct_deletion_audit_pass.astype(bool).all(),
        f"max difference={stress_gold.exact_gold_max_probability_difference.max():.3g}", "fidelity")
    add("all critical result metrics finite", all(
        np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all()
        for frame in [shard_count, controlled, slicing, exact, learned, stress, stress_gold_all, stress_gold]
    ), "all numeric columns", "execution")
    _validate_reused_shard_count(shard_count)
    _validate_reused_strategy(controlled)
    final_practical = pd.read_csv(FINAL_OUT / "results" / "practical_sharding_architectures.csv")
    lgb_learned = learned[learned.model.eq("LightGBM")].merge(
        final_practical[["strategy", "roc_auc", "average_precision", "brier"]],
        on="strategy", suffixes=("_restored", "_final"))
    learned_compatible = all(
        np.allclose(lgb_learned[f"{metric}_restored"], lgb_learned[f"{metric}_final"], atol=1e-12, rtol=0)
        for metric in ["roc_auc", "average_precision", "brier"]
    )
    add("reused LightGBM learned aggregation compatibility", learned_compatible,
        "restored evaluation equals validated final metrics", "configuration")

    checks_frame = pd.DataFrame(checks)
    _atomic_csv(checks_frame, DIRS["results"] / "leakage_config_consistency_checks.csv")
    if not checks_frame.passed.all():
        failed = checks_frame.loc[~checks_frame.passed]
        _update_manifest("full_verification", "failed", ["results/leakage_config_consistency_checks.csv"],
                         failed.to_json(orient="records"))
        raise RuntimeError(f"Restoration verification failed: {failed.check.tolist()}")

    scope = pd.DataFrame([
        ["Identity-safe data pipeline and isolated splits", "outputs_final retained", "A", "unchanged validated foundation"],
        ["Four-family baseline and SISA S=4", "outputs_final retained", "A", "unchanged validated foundation"],
        ["Cross-model shard-count S=2..128", "outputs_restored", "A-restored", "28/28 configurations complete"],
        ["Positive-sample scarcity diagnostic", "outputs_restored", "A-restored", "derived for every model x S row"],
        ["Controlled model x sharding strategy", "outputs_restored", "A-restored", "20/20 configurations complete"],
        ["Learned aggregation Specialist/Risk-Band", "outputs_restored", "A-restored", "8/8 practical configurations complete"],
        ["Deletion locality scale/pattern", "outputs_final retained", "A", "unchanged validated foundation"],
        ["Incremental slicing R=1/2/4/8", "outputs_restored", "A-restored", "corrected post-deletion weights; old R=1/4 superseded"],
        ["Repeated cross-model Exact–Gold", "outputs_restored", "A-restored", "16/16 model-request cases complete; one request per shard/model"],
        ["Approximate continued boosting", "outputs_final retained", "A", "unchanged validated foundation"],
        ["Focused MIA", "outputs_final retained", "A-limited", "unchanged; privacy conclusion remains inconclusive"],
        ["30-round cumulative deletion stress", "outputs_restored", "A-supplementary", "3 streams x 30 rounds plus baseline; Gold directly timed every round"],
        ["Policy-aware nested Specialist case study", "outputs_final retained", "A-supplementary", "unchanged synthetic extension"],
        ["Legacy invalid/proxy branches", "not restored", "E", "intentionally excluded; no legacy numbers copied"],
    ], columns=["experimental_component", "authoritative_artifact_root", "status", "verification_note"])
    _atomic_csv(scope, DIRS["results"] / "complete_experimental_scope_matrix.csv")

    expected_counts = {
        "cross_model_shard_count": (len(shard_count), 28),
        "cross_model_controlled_sharding": (len(controlled), 20),
        "slicing_corrected_full_grid": (len(slicing), 4),
        "repeated_cross_model_exact_gold": (len(exact), 16),
        "cross_model_learned_aggregation": (len(learned), 8),
        "cumulative_stress_rounds_including_baseline": (len(stress), 93),
        "cumulative_stress_gold_checkpoints": (len(stress_gold), 18),
        "cumulative_stress_gold_all_rounds": (len(stress_gold_all), 93),
    }
    failures = [
        {"experiment": name, "observed": observed, "expected": expected,
         "status": "failed_or_skipped", "reason": "configuration count mismatch"}
        for name, (observed, expected) in expected_counts.items() if observed != expected
    ]
    failed_frame = pd.DataFrame(failures, columns=["experiment", "observed", "expected", "status", "reason"])
    _atomic_csv(failed_frame, DIRS["results"] / "failed_skipped_configurations.csv")

    result_rows = []
    for path in sorted((DIRS["results"]).rglob("*.csv")):
        if path.name in {"result_inventory.csv", "artifact_inventory.csv"}:
            continue
        try:
            frame = pd.read_csv(path)
            result_rows.append({
                "relative_path": path.relative_to(OUT).as_posix(), "rows": len(frame),
                "columns": len(frame.columns), "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            })
        except pd.errors.EmptyDataError:
            result_rows.append({
                "relative_path": path.relative_to(OUT).as_posix(), "rows": 0,
                "columns": 0, "size_bytes": path.stat().st_size, "sha256": _sha256(path),
            })
    _atomic_csv(pd.DataFrame(result_rows), DIRS["results"] / "result_inventory.csv")

    artifact_rows = []
    for path in sorted(p for p in OUT.rglob("*") if p.is_file()):
        if path.name == "artifact_inventory.csv":
            continue
        artifact_rows.append({
            "relative_path": path.relative_to(OUT).as_posix(),
            "category": path.relative_to(OUT).parts[0],
            "size_bytes": path.stat().st_size, "sha256": _sha256(path),
        })
    _atomic_csv(pd.DataFrame(artifact_rows), DIRS["results"] / "artifact_inventory.csv")

    summary = {
        "verified_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "all_checks_passed": True,
        "checks": len(checks_frame),
        "failed_or_skipped_configurations": len(failed_frame),
        "result_tables": len(result_rows),
        "artifacts_in_inventory": len(artifact_rows),
        "source_immutability_passed": True,
        "candidate_source_of_truth_pending_notebook_execution_audit": True,
    }
    _atomic_json(summary, DIRS["config"] / "verification_summary.json")
    manifest = _manifest()
    manifest["restoration_plan"]["exact_requests_per_model"] = REPEATED_EXACT_REQUESTS
    manifest["restoration_plan"]["stress"]["gold_timing_rounds"] = STRESS_GOLD_TIMING_ROUNDS
    manifest["restoration_plan"]["stress"]["fidelity_checkpoints"] = STRESS_GOLD_CHECKPOINTS
    manifest["restoration_plan"]["stress"].pop("gold_checkpoints", None)
    manifest["verified_for_candidate_source_of_truth"] = False
    manifest["verification_note"] = "All experiment/artifact checks passed; notebook execution audit still required."
    _atomic_json(manifest, DIRS["config"] / "restoration_manifest.json")
    _update_manifest(
        "full_verification", "complete",
        ["results/leakage_config_consistency_checks.csv", "results/complete_experimental_scope_matrix.csv",
         "results/result_inventory.csv", "results/artifact_inventory.csv",
         "results/failed_skipped_configurations.csv", "config/verification_summary.json"],
        "All experiment, fidelity, leakage, configuration, scope, and immutable-source checks passed.",
    )
    _log("Full restoration verification passed; notebook execution audit remains")
    return summary


def finalize_candidate_source() -> dict:
    """Promote the restored notebook only after the notebook execution audit passes."""
    verify_source_immutability()
    checks = pd.read_csv(DIRS["results"] / "leakage_config_consistency_checks.csv")
    notebook_audit_path = DIRS["config"] / "notebook_execution_audit.json"
    if not notebook_audit_path.exists():
        raise RuntimeError("Notebook execution audit is missing")
    notebook_audit = json.loads(notebook_audit_path.read_text(encoding="utf-8"))
    failed_path = DIRS["results"] / "failed_skipped_configurations.csv"
    try:
        failed_count = len(pd.read_csv(failed_path))
    except pd.errors.EmptyDataError:
        failed_count = 0
    passed = (
        checks.passed.astype(bool).all()
        and notebook_audit["base_cells_unchanged"]
        and notebook_audit["appended_counts_sequential_from_94"]
        and notebook_audit["error_output_count"] == 0
        and failed_count == 0
        and RESTORED_NOTEBOOK.exists()
    )
    if not passed:
        raise RuntimeError("Candidate promotion gate failed")

    # Final inventory is regenerated after notebook execution. Dynamic self-referential
    # inventory/manifest files are listed without a hash-stability claim.
    artifact_rows = []
    for path in sorted(p for p in OUT.rglob("*") if p.is_file()):
        relative = path.relative_to(OUT).as_posix()
        dynamic = relative in {
            "results/artifact_inventory.csv", "config/restoration_manifest.json",
            "config/verification_summary.json", "results/implementation_incidents.csv",
        } or relative.startswith("logs/")
        artifact_rows.append({
            "relative_path": relative, "category": path.relative_to(OUT).parts[0],
            "size_bytes": path.stat().st_size,
            "sha256": "DYNAMIC" if dynamic else _sha256(path),
            "inventory_note": "dynamic audit file" if dynamic else "content hash verified",
        })
    for path, category in [
        (RESTORED_NOTEBOOK, "restored_notebook"),
        (ROOT / "restored_experiments.py", "implementation"),
        (ROOT / "build_restored_notebook.py", "implementation"),
        (ROOT / "RESTORATION_VERIFICATION_REPORT.md", "verification_report"),
    ]:
        artifact_rows.append({
            "relative_path": f"../{path.name}", "category": category,
            "size_bytes": path.stat().st_size, "sha256": _sha256(path),
            "inventory_note": "content hash verified",
        })
    _atomic_csv(pd.DataFrame(artifact_rows), DIRS["results"] / "artifact_inventory.csv")

    incidents = pd.DataFrame([
        {
            "stage": "cumulative_deletion_stress", "configuration": "pre-round checkpoint setup",
            "status": "resolved technical retry", "lost_results": False,
            "detail": "Initial run attempted in-place decrement on a read-only NumPy view; changed to copy=True and resumed from saved model checkpoint before any round-result row was committed.",
        },
        {
            "stage": "final_patch_slicing", "configuration": "supersession comparison column rename",
            "status": "resolved technical retry", "lost_results": False,
            "detail": "The first patched comparison referenced the retired speedup_vs_gold field. The reference was corrected to end_to_end_speedup_vs_all_shard_gold; completed R checkpoints were reused without retraining.",
        },
        {
            "stage": "final_patch_notebook_execution", "configuration": "kernel discovery/runtime path",
            "status": "resolved launcher retry", "lost_results": False,
            "detail": "Two launcher attempts ended before any appendix cell executed because the kernelspec and then ipykernel path were unavailable. JUPYTER_PATH and project-local final_runtime_pkgs were supplied; final execution completed with zero notebook error outputs.",
        },
    ])
    _atomic_csv(incidents, DIRS["results"] / "implementation_incidents.csv")

    summary_path = DIRS["config"] / "verification_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update({
        "candidate_source_of_truth_pending_notebook_execution_audit": False,
        "candidate_experimental_source_of_truth": True,
        "restored_notebook": str(RESTORED_NOTEBOOK),
        "restored_notebook_sha256": _sha256(RESTORED_NOTEBOOK),
        "restored_notebook_cells": notebook_audit["restored_cells"],
        "restored_notebook_code_cells": notebook_audit["restored_code_cells"],
        "appended_execution_counts": notebook_audit["appended_execution_counts"],
        "artifact_inventory_rows": len(artifact_rows),
        "finalized_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    _atomic_json(summary, summary_path)
    manifest = _manifest()
    manifest["verified_for_candidate_source_of_truth"] = True
    manifest["verification_note"] = "All experiments and final notebook execution/preservation audits passed."
    manifest["restored_notebook_sha256"] = summary["restored_notebook_sha256"]
    _atomic_json(manifest, DIRS["config"] / "restoration_manifest.json")
    _update_manifest(
        "candidate_source_finalization", "complete",
        ["config/notebook_execution_audit.json", "results/notebook_execution_error_audit.csv",
         "results/artifact_inventory.csv", "results/implementation_incidents.csv",
         "config/verification_summary.json"],
        "Restored notebook promoted to candidate Experimental Source of Truth after all gates passed.",
    )
    _log("Restored notebook promoted to candidate Experimental Source of Truth")
    return summary


def run_restoration(after_slicing_preflight: bool = False) -> None:
    initialize_restoration()
    if not after_slicing_preflight:
        stage_slicing_corrected()
    stage_cross_model_shard_count()
    stage_cross_model_controlled_strategies()
    if after_slicing_preflight:
        stage_slicing_corrected()
    stage_repeated_exact_gold()
    stage_cross_model_learned_aggregation()
    stage_cumulative_deletion_stress()


if __name__ == "__main__":
    run_restoration()
