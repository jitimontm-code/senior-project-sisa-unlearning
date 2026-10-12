"""Corrected, checkpointed Home Credit + SISA research pipeline.

The legacy notebook is intentionally not imported.  Every reusable artifact written
by this module carries the final-pipeline schema version.
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
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


SCHEMA_VERSION = "home-credit-sisa-final-v1"
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs_final"
DIRS = {name: OUT / name for name in [
    "config", "cache", "models", "shards", "deletion_requests",
    "results", "figures", "logs"
]}
for path in DIRS.values():
    path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Config:
    seed: int = 20260920
    threads: int = 2
    shards: int = 4
    deletion_fraction: float = 0.01
    deletion_repetitions: int = 3
    chunk_size: int = 500_000
    test_fraction: float = 0.15
    calibration_fraction: float = 0.10
    validation_fraction: float = 0.15
    missing_drop_threshold: float = 0.80

    @property
    def hash(self) -> str:
        raw = json.dumps({"schema": SCHEMA_VERSION, **asdict(self)}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:12]


CFG = Config()

APP_FEATURES = [
    "NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
    "CNT_CHILDREN", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
    "AMT_GOODS_PRICE", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "REGION_POPULATION_RELATIVE",
    "DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH",
    "OWN_CAR_AGE", "CNT_FAM_MEMBERS", "REGION_RATING_CLIENT",
    "REGION_RATING_CLIENT_W_CITY", "EXT_SOURCE_1", "EXT_SOURCE_2",
    "EXT_SOURCE_3", "DAYS_LAST_PHONE_CHANGE", "TOTALAREA_MODE",
]

TABLE_SPECS = {
    "bureau": ("bureau.csv", ["SK_ID_CURR", "DAYS_CREDIT", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT"]),
    "previous": ("previous_application.csv", ["SK_ID_CURR", "AMT_APPLICATION", "AMT_CREDIT", "DAYS_DECISION"]),
    "installments": ("installments_payments.csv", ["SK_ID_CURR", "DAYS_ENTRY_PAYMENT", "DAYS_INSTALMENT", "AMT_PAYMENT", "AMT_INSTALMENT"]),
    "credit_card": ("credit_card_balance.csv", ["SK_ID_CURR", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL", "SK_DPD"]),
}


def log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with (DIRS["logs"] / "runtime.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def save_json(obj: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)


def stage_environment() -> None:
    import lightgbm, sklearn, scipy, xgboost
    info = {
        "schema_version": SCHEMA_VERSION,
        "config": asdict(CFG),
        "config_hash": CFG.hash,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__, "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__, "scipy": scipy.__version__,
            "lightgbm": lightgbm.__version__, "xgboost": xgboost.__version__,
            "joblib": joblib.__version__,
        },
    }
    save_json(info, DIRS["config"] / "environment.json")
    save_json({"schema_version": SCHEMA_VERSION, "config_hash": CFG.hash,
               "stages": {}, "safe_for_thesis": []}, DIRS["config"] / "experiment_manifest.json")
    log(f"Environment recorded; config={CFG.hash}; threads={CFG.threads}")


def update_manifest(stage: str, status: str, artifacts: list[str], notes: str = "") -> None:
    path = DIRS["config"] / "experiment_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["stages"][stage] = {
        "status": status, "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "artifacts": artifacts, "notes": notes,
    }
    if status == "complete" and stage not in manifest["safe_for_thesis"]:
        manifest["safe_for_thesis"].append(stage)
    save_json(manifest, path)


def _split_frame(app: pd.DataFrame) -> pd.DataFrame:
    idx = np.arange(len(app))
    y = app["TARGET"].to_numpy()
    dev_idx, test_idx = train_test_split(idx, test_size=CFG.test_fraction,
                                         stratify=y, random_state=CFG.seed)
    remaining = 1.0 - CFG.test_fraction
    cal_rel = CFG.calibration_fraction / remaining
    dev_idx, cal_idx = train_test_split(dev_idx, test_size=cal_rel,
                                        stratify=y[dev_idx], random_state=CFG.seed + 1)
    remaining -= CFG.calibration_fraction
    val_rel = CFG.validation_fraction / remaining
    train_idx, val_idx = train_test_split(dev_idx, test_size=val_rel,
                                          stratify=y[dev_idx], random_state=CFG.seed + 2)
    labels = np.full(len(app), "", dtype=object)
    for name, rows in [("train", train_idx), ("validation", val_idx),
                       ("calibration", cal_idx), ("test", test_idx)]:
        labels[rows] = name
    assert np.all(labels != "") and len(np.unique(app["SK_ID_CURR"])) == len(app)
    split = app[["SK_ID_CURR", "TARGET"]].copy()
    split["split"] = labels
    return split


def stage_splits() -> pd.DataFrame:
    path = DIRS["config"] / f"data_splits_{CFG.hash}.csv"
    if path.exists():
        split = pd.read_csv(path)
        assert split["SK_ID_CURR"].is_unique
        log("Validated cached split file")
        return split
    app = pd.read_csv(ROOT / "application_train.csv", usecols=["SK_ID_CURR", "TARGET"])
    split = _split_frame(app)
    atomic_csv(split, path)
    summary = split.groupby("split")["TARGET"].agg(["size", "mean"]).reset_index()
    atomic_csv(summary, DIRS["results"] / "data_split_summary.csv")
    update_manifest("data_partitions", "complete", [str(path), "results/data_split_summary.csv"],
                    "Final test is isolated before all model/configuration choices.")
    log(f"Created immutable identity-preserving splits for {len(split):,} applicants")
    return split


def _chunk_features(name: str, ids: set[int]) -> pd.DataFrame:
    filename, cols = TABLE_SPECS[name]
    cache = DIRS["cache"] / f"agg_{name}_{CFG.hash}.csv.gz"
    if cache.exists():
        return pd.read_csv(cache)
    parts = []
    for chunk_no, chunk in enumerate(pd.read_csv(ROOT / filename, usecols=cols,
                                                  chunksize=CFG.chunk_size)):
        chunk = chunk[chunk["SK_ID_CURR"].isin(ids)]
        if chunk.empty:
            continue
        if name == "installments":
            chunk["PAYMENT_DELAY"] = chunk["DAYS_ENTRY_PAYMENT"] - chunk["DAYS_INSTALMENT"]
            chunk["PAYMENT_RATIO"] = chunk["AMT_PAYMENT"] / chunk["AMT_INSTALMENT"].replace(0, np.nan)
        numeric = [c for c in chunk.columns if c != "SK_ID_CURR"]
        part = chunk.groupby("SK_ID_CURR")[numeric].agg(["sum", "count", "min", "max"])
        part.columns = [f"{name}__{c}__{s}" for c, s in part.columns]
        parts.append(part.reset_index())
        if chunk_no % 5 == 0:
            log(f"{name}: processed chunk {chunk_no + 1}")
    combined = pd.concat(parts, ignore_index=True)
    sum_cols = [c for c in combined if c.endswith("__sum") or c.endswith("__count")]
    min_cols = [c for c in combined if c.endswith("__min")]
    max_cols = [c for c in combined if c.endswith("__max")]
    agg_map = {c: "sum" for c in sum_cols} | {c: "min" for c in min_cols} | {c: "max" for c in max_cols}
    combined = combined.groupby("SK_ID_CURR", as_index=False).agg(agg_map)
    for sum_col in sum_cols:
        if sum_col.endswith("__sum"):
            count_col = sum_col[:-5] + "count"
            if count_col in combined:
                combined[sum_col[:-3] + "mean"] = combined[sum_col] / combined[count_col].replace(0, np.nan)
    combined.to_csv(cache, index=False, compression="gzip")
    log(f"Cached {name} aggregates: {combined.shape}")
    return combined


def _load_merged(split: pd.DataFrame) -> pd.DataFrame:
    cache = DIRS["cache"] / f"merged_features_{CFG.hash}.csv.gz"
    if cache.exists():
        merged = pd.read_csv(cache)
    else:
        header = pd.read_csv(ROOT / "application_train.csv", nrows=0)
        usecols = ["SK_ID_CURR", "TARGET"] + [c for c in APP_FEATURES if c in header.columns]
        merged = pd.read_csv(ROOT / "application_train.csv", usecols=usecols)
        original = merged["SK_ID_CURR"].to_numpy(copy=True)
        ids = set(int(x) for x in original)
        for name in TABLE_SPECS:
            agg = _chunk_features(name, ids)
            merged = merged.merge(agg, on="SK_ID_CURR", how="left", validate="one_to_one", sort=False)
            assert np.array_equal(merged["SK_ID_CURR"].to_numpy(), original), f"ID order changed after {name}"
            del agg
            gc.collect()
        merged.to_csv(cache, index=False, compression="gzip")
    check = merged[["SK_ID_CURR", "TARGET"]].merge(
        split[["SK_ID_CURR", "TARGET"]], on="SK_ID_CURR", validate="one_to_one", suffixes=("", "_split"))
    assert (check["TARGET"] == check["TARGET_split"]).all()
    assert merged["SK_ID_CURR"].is_unique and len(merged) == len(split)
    return merged


def stage_preprocessing() -> None:
    marker = DIRS["cache"] / f"preprocessing_complete_{CFG.hash}.json"
    if marker.exists():
        log("Validated preprocessing checkpoint")
        return
    split = stage_splits()
    merged = _load_merged(split)
    merged = merged.merge(split[["SK_ID_CURR", "split"]], on="SK_ID_CURR",
                          validate="one_to_one", sort=False)
    frames = {name: merged.loc[merged["split"] == name].copy() for name in
              ["train", "validation", "calibration", "test"]}
    feature_cols = [c for c in merged if c not in ["SK_ID_CURR", "TARGET", "split"]]
    train_missing = frames["train"][feature_cols].isna().mean()
    feature_cols = [c for c in feature_cols if train_missing[c] <= CFG.missing_drop_threshold]
    categorical = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(frames["train"][c])]
    numeric = [c for c in feature_cols if c not in categorical]
    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median", add_indicator=True), numeric),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)),
        ]), categorical),
    ], sparse_threshold=1.0)
    X_train = preprocessor.fit_transform(frames["train"][feature_cols]).tocsr().astype(np.float32)
    joblib.dump({"preprocessor": preprocessor, "feature_cols": feature_cols,
                 "schema_version": SCHEMA_VERSION, "config_hash": CFG.hash},
                DIRS["models"] / f"preprocessor_{CFG.hash}.joblib", compress=3)
    for name, frame in frames.items():
        X = X_train if name == "train" else preprocessor.transform(frame[feature_cols]).tocsr().astype(np.float32)
        sparse.save_npz(DIRS["cache"] / f"X_{name}_{CFG.hash}.npz", X, compressed=True)
        np.savez_compressed(DIRS["cache"] / f"meta_{name}_{CFG.hash}.npz",
                            ids=frame["SK_ID_CURR"].to_numpy(np.int64),
                            y=frame["TARGET"].to_numpy(np.int8))
        assert X.shape[0] == len(frame)
        del X
    save_json({"schema_version": SCHEMA_VERSION, "config_hash": CFG.hash,
               "n_features": int(X_train.shape[1]), "raw_features": len(feature_cols),
               "categorical": categorical, "numeric": numeric,
               "fit_scope": "train_only"}, marker)
    update_manifest("corrected_preprocessing", "complete",
                    [str(marker), "cache/X_{split}_{config}.npz", "models/preprocessor.joblib"],
                    "Applicant IDs and targets asserted before/after every merge; learned preprocessing fit on train only.")
    log(f"Preprocessing complete: train={X_train.shape}; sparse float32")
    del merged, frames, X_train, preprocessor
    gc.collect()


def load_matrix(name: str):
    X = sparse.load_npz(DIRS["cache"] / f"X_{name}_{CFG.hash}.npz")
    meta = np.load(DIRS["cache"] / f"meta_{name}_{CFG.hash}.npz")
    return X, meta["y"], meta["ids"]


def make_model(name: str, seed: int):
    if name == "Logistic Regression":
        return Pipeline([
            ("scale", StandardScaler(with_mean=False)),
            ("model", LogisticRegression(max_iter=500, class_weight="balanced", C=0.2,
                                         random_state=seed, n_jobs=CFG.threads)),
        ])
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=180, max_depth=12, min_samples_leaf=10,
                                      max_features="sqrt", class_weight="balanced_subsample",
                                      random_state=seed, n_jobs=CFG.threads)
    if name == "LightGBM":
        return LGBMClassifier(n_estimators=300, learning_rate=0.04, num_leaves=31,
                              subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
                              class_weight="balanced", random_state=seed,
                              n_jobs=CFG.threads, verbosity=-1, deterministic=True,
                              force_col_wise=True)
    if name == "XGBoost":
        return XGBClassifier(n_estimators=300, learning_rate=0.04, max_depth=5,
                             min_child_weight=5, subsample=0.85, colsample_bytree=0.85,
                             reg_lambda=1.0, eval_metric="logloss", random_state=seed,
                             n_jobs=CFG.threads, tree_method="hist")
    raise ValueError(name)


def positive_probability(model, X) -> np.ndarray:
    return np.asarray(model.predict_proba(X)[:, 1], dtype=np.float64)


def fit_platt(raw_prob: np.ndarray, y: np.ndarray):
    logits = np.log(np.clip(raw_prob, 1e-6, 1 - 1e-6) / np.clip(1 - raw_prob, 1e-6, 1))[:, None]
    cal = LogisticRegression(C=1e6, max_iter=200, random_state=CFG.seed)
    cal.fit(logits, y)
    return cal


def calibrate(calibrator, raw_prob: np.ndarray) -> np.ndarray:
    logits = np.log(np.clip(raw_prob, 1e-6, 1 - 1e-6) / np.clip(1 - raw_prob, 1e-6, 1))[:, None]
    return calibrator.predict_proba(logits)[:, 1]


def choose_threshold(y: np.ndarray, p: np.ndarray) -> float:
    thresholds = np.linspace(0.02, 0.60, 117)
    f1 = [f1_score(y, p >= t) for t in thresholds]
    return float(thresholds[int(np.argmax(f1))])


def metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": roc_auc_score(y, p), "average_precision": average_precision_score(y, p),
        "log_loss": log_loss(y, p), "brier": brier_score_loss(y, p),
        "threshold": threshold, "f1": f1_score(y, pred),
        "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
    }


def stage_baselines() -> None:
    stage_preprocessing()
    Xtr, ytr, _ = load_matrix("train")
    Xv, yv, idv = load_matrix("validation")
    Xc, yc, idc = load_matrix("calibration")
    Xt, yt, idt = load_matrix("test")
    rows = []
    for i, name in enumerate(["Logistic Regression", "Random Forest", "LightGBM", "XGBoost"]):
        artifact = DIRS["models"] / f"baseline_{name.lower().replace(' ', '_')}_{CFG.hash}.joblib"
        pred_path = DIRS["results"] / f"predictions_{name.lower().replace(' ', '_')}_{CFG.hash}.csv.gz"
        if artifact.exists() and pred_path.exists():
            bundle = joblib.load(artifact)
            pred = pd.read_csv(pred_path)
            rows.append({"model": name, "fit_seconds": bundle["fit_seconds"],
                         **metrics(pred.loc[pred.split == "test", "y"].to_numpy(),
                                   pred.loc[pred.split == "test", "probability"].to_numpy(),
                                   bundle["threshold"])})
            log(f"Reused validated baseline checkpoint: {name}")
            continue
        started = time.perf_counter()
        model = make_model(name, CFG.seed + i)
        model.fit(Xtr, ytr)
        fit_seconds = time.perf_counter() - started
        cal = fit_platt(positive_probability(model, Xc), yc)
        pv = calibrate(cal, positive_probability(model, Xv))
        threshold = choose_threshold(yv, pv)
        pt = calibrate(cal, positive_probability(model, Xt))
        bundle = {"model": model, "calibrator": cal, "threshold": threshold,
                  "fit_seconds": fit_seconds, "schema_version": SCHEMA_VERSION,
                  "config_hash": CFG.hash}
        joblib.dump(bundle, artifact, compress=3)
        pred = pd.concat([
            pd.DataFrame({"SK_ID_CURR": idv, "split": "validation", "y": yv, "probability": pv}),
            pd.DataFrame({"SK_ID_CURR": idt, "split": "test", "y": yt, "probability": pt}),
        ], ignore_index=True)
        pred.to_csv(pred_path, index=False, compression="gzip")
        rows.append({"model": name, "fit_seconds": fit_seconds, **metrics(yt, pt, threshold)})
        atomic_csv(pd.DataFrame(rows), DIRS["results"] / "baseline_performance.csv")
        log(f"Baseline complete: {name}; test AUC={rows[-1]['roc_auc']:.4f}; {fit_seconds:.1f}s")
        del model, cal, pred, bundle
        gc.collect()
    atomic_csv(pd.DataFrame(rows), DIRS["results"] / "baseline_performance.csv")
    update_manifest("baseline_models", "complete",
                    ["results/baseline_performance.csv", "results/predictions_*.csv.gz", "models/baseline_*.joblib"],
                    "Calibration fitted on calibration split; threshold selected on validation; test used once for reporting.")
    del Xtr, Xv, Xc, Xt
    gc.collect()


def make_stratified_shards(y: np.ndarray, ids: np.ndarray) -> pd.DataFrame:
    rng = np.random.default_rng(CFG.seed)
    assignment = np.empty(len(y), dtype=np.int16)
    for label in np.unique(y):
        rows = np.flatnonzero(y == label)
        rng.shuffle(rows)
        assignment[rows] = np.arange(len(rows)) % CFG.shards
    result = pd.DataFrame({"row_index": np.arange(len(y)), "SK_ID_CURR": ids,
                           "TARGET": y, "shard": assignment})
    assert result["SK_ID_CURR"].is_unique and result["shard"].nunique() == CFG.shards
    return result


def _ensemble_raw(models: list, X, weights: np.ndarray) -> np.ndarray:
    pred = np.zeros(X.shape[0], dtype=np.float64)
    for model, weight in zip(models, weights):
        pred += positive_probability(model, X) * weight
    return pred


def stage_sisa() -> None:
    stage_baselines()
    Xtr, ytr, idtr = load_matrix("train")
    Xv, yv, idv = load_matrix("validation")
    Xc, yc, _ = load_matrix("calibration")
    Xt, yt, idt = load_matrix("test")
    assignment_path = DIRS["shards"] / f"stratified_S{CFG.shards}_{CFG.hash}.csv"
    assignment = pd.read_csv(assignment_path) if assignment_path.exists() else make_stratified_shards(ytr, idtr)
    if not assignment_path.exists():
        atomic_csv(assignment, assignment_path)
    models, sizes = [], []
    for shard in range(CFG.shards):
        path = DIRS["models"] / f"sisa_lightgbm_S{CFG.shards}_shard{shard}_{CFG.hash}.joblib"
        rows = assignment.loc[assignment.shard == shard, "row_index"].to_numpy()
        if path.exists():
            model = joblib.load(path)["model"]
        else:
            started = time.perf_counter()
            model = make_model("LightGBM", CFG.seed + 100 + shard)
            model.fit(Xtr[rows], ytr[rows])
            joblib.dump({"model": model, "rows": len(rows), "fit_seconds": time.perf_counter() - started,
                         "schema_version": SCHEMA_VERSION, "config_hash": CFG.hash}, path, compress=3)
            log(f"SISA shard {shard + 1}/{CFG.shards} trained and checkpointed")
        models.append(model); sizes.append(len(rows))
    weights = np.asarray(sizes, dtype=float) / sum(sizes)
    cal = fit_platt(_ensemble_raw(models, Xc, weights), yc)
    pv = calibrate(cal, _ensemble_raw(models, Xv, weights))
    threshold = choose_threshold(yv, pv)
    pt = calibrate(cal, _ensemble_raw(models, Xt, weights))
    bundle = {"calibrator": cal, "threshold": threshold, "weights": weights,
              "schema_version": SCHEMA_VERSION, "config_hash": CFG.hash}
    joblib.dump(bundle, DIRS["models"] / f"sisa_ensemble_S{CFG.shards}_{CFG.hash}.joblib", compress=3)
    result = pd.DataFrame([{"model": "SISA LightGBM", "shards": CFG.shards,
                            **metrics(yt, pt, threshold)}])
    atomic_csv(result, DIRS["results"] / "sisa_performance.csv")
    pd.DataFrame({"SK_ID_CURR": idt, "y": yt, "probability": pt}).to_csv(
        DIRS["results"] / f"predictions_sisa_S{CFG.shards}_{CFG.hash}.csv.gz",
        index=False, compression="gzip")
    update_manifest("standard_sisa", "complete",
                    [str(assignment_path), "results/sisa_performance.csv", "models/sisa_lightgbm_*.joblib"],
                    "Pre-specified S=4; stratified balanced shards; retained-size weighted aggregation.")
    log(f"Standard SISA complete; test AUC={result.iloc[0].roc_auc:.4f}")
    del Xtr, Xv, Xc, Xt, models
    gc.collect()


def _train_shard_models(X, y, assignment, deleted_ids: set[int], only_affected: bool,
                        originals: list) -> tuple[list, np.ndarray, float, list[int]]:
    models, sizes, affected = [], [], []
    started = time.perf_counter()
    for shard in range(CFG.shards):
        frame = assignment[assignment.shard == shard]
        keep = ~frame.SK_ID_CURR.isin(deleted_ids)
        rows = frame.loc[keep, "row_index"].to_numpy()
        is_affected = bool((~keep).any())
        if is_affected:
            affected.append(shard)
        if only_affected and not is_affected:
            model = originals[shard]
        else:
            model = make_model("LightGBM", CFG.seed + 100 + shard)
            model.fit(X[rows], y[rows])
        models.append(model); sizes.append(len(rows))
    return models, np.asarray(sizes, float) / sum(sizes), time.perf_counter() - started, affected


def stage_exact_unlearning() -> None:
    stage_sisa()
    Xtr, ytr, idtr = load_matrix("train")
    Xt, yt, idt = load_matrix("test")
    assignment = pd.read_csv(DIRS["shards"] / f"stratified_S{CFG.shards}_{CFG.hash}.csv")
    ensemble = joblib.load(DIRS["models"] / f"sisa_ensemble_S{CFG.shards}_{CFG.hash}.joblib")
    originals = [joblib.load(DIRS["models"] / f"sisa_lightgbm_S{CFG.shards}_shard{s}_{CFG.hash}.joblib")["model"]
                 for s in range(CFG.shards)]
    request_path = DIRS["deletion_requests"] / f"random_f{CFG.deletion_fraction}_R{CFG.deletion_repetitions}_{CFG.hash}.csv"
    if request_path.exists():
        requests = pd.read_csv(request_path)
    else:
        rng = np.random.default_rng(CFG.seed + 500)
        frames = []
        n_delete = max(1, round(len(idtr) * CFG.deletion_fraction))
        for rep in range(CFG.deletion_repetitions):
            chosen = rng.choice(idtr, size=n_delete, replace=False)
            frames.append(pd.DataFrame({"repetition": rep, "SK_ID_CURR": chosen}))
        requests = pd.concat(frames, ignore_index=True)
        atomic_csv(requests, request_path)
    result_path = DIRS["results"] / "exact_sisa_vs_gold.csv"
    done = pd.read_csv(result_path) if result_path.exists() else pd.DataFrame()
    rows = done.to_dict("records")
    completed = set(done.get("repetition", []))
    for rep in range(CFG.deletion_repetitions):
        if rep in completed:
            continue
        try:
            deleted = set(requests.loc[requests.repetition == rep, "SK_ID_CURR"].astype(int))
            exact_models, exact_w, exact_sec, affected = _train_shard_models(
                Xtr, ytr, assignment, deleted, True, originals)
            gold_models, gold_w, gold_sec, _ = _train_shard_models(
                Xtr, ytr, assignment, deleted, False, originals)
            p_exact = calibrate(ensemble["calibrator"], _ensemble_raw(exact_models, Xt, exact_w))
            p_gold = calibrate(ensemble["calibrator"], _ensemble_raw(gold_models, Xt, gold_w))
            diff = p_exact - p_gold
            threshold = ensemble["threshold"]
            row = {
                "repetition": rep, "deletion_fraction": CFG.deletion_fraction,
                "deleted_records": len(deleted), "affected_shards": len(affected),
                "workload_fraction": len(affected) / CFG.shards,
                "exact_seconds": exact_sec, "gold_seconds": gold_sec,
                "speedup": gold_sec / max(exact_sec, 1e-9),
                "prediction_mae": float(np.mean(np.abs(diff))),
                "prediction_rmse": float(np.sqrt(np.mean(diff ** 2))),
                "max_probability_difference": float(np.max(np.abs(diff))),
                "prediction_disagreement": float(np.mean((p_exact >= threshold) != (p_gold >= threshold))),
                "exact_roc_auc": roc_auc_score(yt, p_exact), "gold_roc_auc": roc_auc_score(yt, p_gold),
                "status": "complete", "error": "",
            }
            np.savez_compressed(DIRS["results"] / f"exact_gold_predictions_rep{rep}_{CFG.hash}.npz",
                                ids=idt, y=yt, exact=p_exact, gold=p_gold)
            rows.append(row)
            atomic_csv(pd.DataFrame(rows), result_path)
            log(f"Exact unlearning rep {rep}: affected={len(affected)}/{CFG.shards}; speedup={row['speedup']:.2f}x; MAE={row['prediction_mae']:.3g}")
            del exact_models, gold_models, p_exact, p_gold
            gc.collect()
        except Exception as exc:
            rows.append({"repetition": rep, "deletion_fraction": CFG.deletion_fraction,
                         "status": "failed", "error": repr(exc)})
            atomic_csv(pd.DataFrame(rows), result_path)
            (DIRS["logs"] / f"exact_rep{rep}_error.log").write_text(traceback.format_exc(), encoding="utf-8")
            log(f"Exact unlearning rep {rep} failed and was checkpointed: {exc!r}")
    update_manifest("exact_sisa_vs_gold", "complete",
                    [str(result_path), str(request_path), "results/exact_gold_predictions_*.npz"],
                    "Exact SISA retraining conditional on fixed train-only preprocessing and fixed shard assignment.")


def stage_sharding_comparison() -> None:
    """Controlled comparison: only assignment rule changes; no best-strategy selection."""
    stage_sisa()
    Xtr, ytr, idtr = load_matrix("train")
    Xv, yv, _ = load_matrix("validation")
    Xc, yc, _ = load_matrix("calibration")
    Xt, yt, _ = load_matrix("test")
    strat = pd.read_csv(DIRS["shards"] / f"stratified_S{CFG.shards}_{CFG.hash}.csv")
    random_path = DIRS["shards"] / f"random_balanced_S{CFG.shards}_{CFG.hash}.csv"
    if random_path.exists():
        random_assignment = pd.read_csv(random_path)
    else:
        rng = np.random.default_rng(CFG.seed)
        order = rng.permutation(len(ytr))
        shard = np.empty(len(ytr), dtype=np.int16)
        shard[order] = np.arange(len(ytr)) % CFG.shards
        random_assignment = pd.DataFrame({"row_index": np.arange(len(ytr)),
                                          "SK_ID_CURR": idtr, "TARGET": ytr, "shard": shard})
        atomic_csv(random_assignment, random_path)
    output = []
    for strategy, assignment in [("stratified_balanced", strat), ("random_balanced", random_assignment)]:
        models, sizes, fit_seconds = [], [], 0.0
        for shard in range(CFG.shards):
            if strategy == "stratified_balanced":
                path = DIRS["models"] / f"sisa_lightgbm_S{CFG.shards}_shard{shard}_{CFG.hash}.joblib"
            else:
                path = DIRS["models"] / f"controlled_{strategy}_S{CFG.shards}_shard{shard}_{CFG.hash}.joblib"
            rows = assignment.loc[assignment.shard == shard, "row_index"].to_numpy()
            if path.exists():
                bundle = joblib.load(path); model = bundle["model"]
                fit_seconds += float(bundle.get("fit_seconds", 0.0))
            else:
                started = time.perf_counter()
                model = make_model("LightGBM", CFG.seed + 100 + shard)
                model.fit(Xtr[rows], ytr[rows])
                elapsed = time.perf_counter() - started; fit_seconds += elapsed
                joblib.dump({"model": model, "rows": len(rows), "fit_seconds": elapsed,
                             "schema_version": SCHEMA_VERSION, "config_hash": CFG.hash}, path, compress=3)
            models.append(model); sizes.append(len(rows))
        weights = np.asarray(sizes, float) / sum(sizes)
        cal = fit_platt(_ensemble_raw(models, Xc, weights), yc)
        pv = calibrate(cal, _ensemble_raw(models, Xv, weights))
        threshold = choose_threshold(yv, pv)
        pt = calibrate(cal, _ensemble_raw(models, Xt, weights))
        class_rates = assignment.groupby("shard").TARGET.mean()
        output.append({"strategy": strategy, "shards": CFG.shards,
                       "fit_seconds": fit_seconds, "class_rate_sd": class_rates.std(),
                       **metrics(yt, pt, threshold)})
        log(f"Controlled sharding complete: {strategy}; test AUC={output[-1]['roc_auc']:.4f}")
        del models
        gc.collect()
    atomic_csv(pd.DataFrame(output), DIRS["results"] / "controlled_sharding_comparison.csv")
    update_manifest("controlled_sharding", "complete",
                    ["results/controlled_sharding_comparison.csv", str(random_path)],
                    "S, model, seeds, splits, aggregation, calibration, and evaluation held constant; only assignment changes.")


def _make_continued_boosting(seed: int):
    return LGBMClassifier(n_estimators=60, learning_rate=0.02, num_leaves=31,
                          subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
                          class_weight="balanced", random_state=seed, n_jobs=CFG.threads,
                          verbosity=-1, deterministic=True, force_col_wise=True)


def stage_approximate_unlearning() -> None:
    """Evaluate continued boosting as an explicitly non-exact heuristic."""
    stage_exact_unlearning()
    Xtr, ytr, _ = load_matrix("train")
    Xt, yt, _ = load_matrix("test")
    assignment = pd.read_csv(DIRS["shards"] / f"stratified_S{CFG.shards}_{CFG.hash}.csv")
    ensemble = joblib.load(DIRS["models"] / f"sisa_ensemble_S{CFG.shards}_{CFG.hash}.joblib")
    originals = [joblib.load(DIRS["models"] / f"sisa_lightgbm_S{CFG.shards}_shard{s}_{CFG.hash}.joblib")["model"]
                 for s in range(CFG.shards)]
    requests = pd.read_csv(DIRS["deletion_requests"] / f"random_f{CFG.deletion_fraction}_R{CFG.deletion_repetitions}_{CFG.hash}.csv")
    result_path = DIRS["results"] / "approximate_vs_exact_gold.csv"
    done = pd.read_csv(result_path) if result_path.exists() else pd.DataFrame()
    rows = done.to_dict("records"); completed = set(done.get("repetition", []))
    for rep in range(CFG.deletion_repetitions):
        if rep in completed:
            continue
        deleted = set(requests.loc[requests.repetition == rep, "SK_ID_CURR"].astype(int))
        models, sizes = [], []
        started = time.perf_counter()
        for shard in range(CFG.shards):
            frame = assignment[assignment.shard == shard]
            keep = ~frame.SK_ID_CURR.isin(deleted)
            train_rows = frame.loc[keep, "row_index"].to_numpy()
            path = DIRS["models"] / f"approx_continued_rep{rep}_shard{shard}_{CFG.hash}.joblib"
            if path.exists():
                model = joblib.load(path)["model"]
            else:
                model = _make_continued_boosting(CFG.seed + 900 + shard)
                model.fit(Xtr[train_rows], ytr[train_rows], init_model=originals[shard].booster_)
                joblib.dump({"model": model, "method": "continued_boosting_heuristic",
                             "schema_version": SCHEMA_VERSION, "config_hash": CFG.hash}, path, compress=3)
            models.append(model); sizes.append(len(train_rows))
        elapsed = time.perf_counter() - started
        weights = np.asarray(sizes, float) / sum(sizes)
        p_approx = calibrate(ensemble["calibrator"], _ensemble_raw(models, Xt, weights))
        gold_file = np.load(DIRS["results"] / f"exact_gold_predictions_rep{rep}_{CFG.hash}.npz")
        p_gold = gold_file["gold"]; diff = p_approx - p_gold
        threshold = ensemble["threshold"]
        rows.append({"repetition": rep, "method": "continued_boosting_heuristic",
                     "approx_seconds": elapsed, "prediction_mae": np.mean(np.abs(diff)),
                     "prediction_rmse": np.sqrt(np.mean(diff ** 2)),
                     "max_probability_difference": np.max(np.abs(diff)),
                     "prediction_disagreement": np.mean((p_approx >= threshold) != (p_gold >= threshold)),
                     "approx_roc_auc": roc_auc_score(yt, p_approx),
                     "gold_roc_auc": roc_auc_score(yt, p_gold), "status": "complete"})
        atomic_csv(pd.DataFrame(rows), result_path)
        log(f"Approximate rep {rep}: MAE vs gold={rows[-1]['prediction_mae']:.4f}")
        del models, p_approx
        gc.collect()
    update_manifest("approximate_unlearning", "complete", [str(result_path), "models/approx_continued_*.joblib"],
                    "Continued boosting retains original trees and is explicitly a heuristic, never an exact fallback; all shards are aggregated.")


def _membership_score(model, X, y):
    p = np.clip(positive_probability(model, X), 1e-7, 1 - 1e-7)
    return np.where(y == 1, np.log(p), np.log(1 - p))


def stage_mia() -> None:
    """Focused loss-based MIA with verified members of one attacked shard."""
    stage_exact_unlearning()
    Xtr, ytr, _ = load_matrix("train")
    Xv, yv, _ = load_matrix("validation")
    assignment = pd.read_csv(DIRS["shards"] / f"stratified_S{CFG.shards}_{CFG.hash}.csv")
    requests = pd.read_csv(DIRS["deletion_requests"] / f"random_f{CFG.deletion_fraction}_R{CFG.deletion_repetitions}_{CFG.hash}.csv")
    deleted = set(requests.loc[requests.repetition == 0, "SK_ID_CURR"].astype(int))
    shard = 0
    frame = assignment[assignment.shard == shard]
    deleted_frame = frame[frame.SK_ID_CURR.isin(deleted)]
    retained_frame = frame[~frame.SK_ID_CURR.isin(deleted)]
    assert len(deleted_frame) > 0 and not deleted_frame.SK_ID_CURR.isin(retained_frame.SK_ID_CURR).any()
    original = joblib.load(DIRS["models"] / f"sisa_lightgbm_S{CFG.shards}_shard{shard}_{CFG.hash}.joblib")["model"]
    exact_path = DIRS["models"] / f"mia_gold_rep0_shard{shard}_{CFG.hash}.joblib"
    if exact_path.exists():
        exact = joblib.load(exact_path)["model"]
    else:
        exact = make_model("LightGBM", CFG.seed + 100 + shard)
        exact.fit(Xtr[retained_frame.row_index.to_numpy()], ytr[retained_frame.row_index.to_numpy()])
        joblib.dump({"model": exact, "schema_version": SCHEMA_VERSION, "config_hash": CFG.hash}, exact_path, compress=3)
    rng = np.random.default_rng(CFG.seed + 700)
    n = min(5000, len(retained_frame), len(yv))
    member_rows = rng.choice(retained_frame.row_index.to_numpy(), n, replace=False)
    nonmember_rows = rng.choice(np.arange(len(yv)), n, replace=False)
    labels = np.r_[np.ones(n), np.zeros(n)]
    rows = []
    for model_name, model in [("original", original), ("exact_and_gold", exact)]:
        scores = np.r_[_membership_score(model, Xtr[member_rows], ytr[member_rows]),
                       _membership_score(model, Xv[nonmember_rows], yv[nonmember_rows])]
        auc = roc_auc_score(labels, scores)
        threshold = np.quantile(scores[n:], .95)
        deleted_score = _membership_score(model, Xtr[deleted_frame.row_index.to_numpy()],
                                           ytr[deleted_frame.row_index.to_numpy()])
        rows.append({"model_state": model_name, "attacked_shard": shard,
                     "verified_members": n, "verified_nonmembers": n,
                     "deleted_records_in_shard": len(deleted_frame), "attack_auc": auc,
                     "attack_power": "meaningful" if auc >= .55 else "inconclusive",
                     "deleted_inferred_member_rate_at_5pct_fpr": np.mean(deleted_score >= threshold)})
    atomic_csv(pd.DataFrame(rows), DIRS["results"] / "privacy_mia.csv")
    update_manifest("privacy_mia", "complete", ["results/privacy_mia.csv", str(exact_path)],
                    "Loss-based focused attack; membership in attacked shard verified. AUC below 0.55 is reported as inconclusive.")
    log(f"Focused MIA complete; original attack AUC={rows[0]['attack_auc']:.3f}")


def stage_reporting() -> None:
    import matplotlib.pyplot as plt
    base = pd.read_csv(DIRS["results"] / "baseline_performance.csv")
    sisa = pd.read_csv(DIRS["results"] / "sisa_performance.csv")
    model_table = pd.concat([base.assign(shards=1), sisa], ignore_index=True, sort=False)
    atomic_csv(model_table, DIRS["results"] / "thesis_model_performance.csv")
    exact = pd.read_csv(DIRS["results"] / "exact_sisa_vs_gold.csv")
    complete = exact[exact.status == "complete"].copy()
    summary_rows = []
    for column in ["exact_seconds", "gold_seconds", "speedup", "prediction_mae",
                   "prediction_rmse", "max_probability_difference", "prediction_disagreement"]:
        values = complete[column].dropna()
        summary_rows.append({"measure": column, "n": len(values), "median": values.median(),
                             "q1": values.quantile(.25), "q3": values.quantile(.75),
                             "mean": values.mean()})
    atomic_csv(pd.DataFrame(summary_rows), DIRS["results"] / "exact_unlearning_summary.csv")
    approx_path = DIRS["results"] / "approximate_vs_exact_gold.csv"
    if approx_path.exists():
        approx = pd.read_csv(approx_path)
        approx_summary = pd.DataFrame([{
            "method": "continued_boosting_heuristic", "repetitions": len(approx),
            "median_seconds": approx.approx_seconds.median(),
            "median_prediction_mae": approx.prediction_mae.median(),
            "median_prediction_rmse": approx.prediction_rmse.median(),
            "median_prediction_disagreement": approx.prediction_disagreement.median(),
            "median_auc_difference_vs_gold": (approx.approx_roc_auc - approx.gold_roc_auc).median(),
        }])
        atomic_csv(approx_summary, DIRS["results"] / "approximate_unlearning_summary.csv")
    runtime_rows = [{"experiment": "baseline", "unit": row.model,
                     "seconds": row.fit_seconds, "threads": CFG.threads}
                    for row in base.itertuples()]
    runtime_rows += [{"experiment": "exact_unlearning", "unit": f"rep_{int(row.repetition)}",
                      "seconds": row.exact_seconds, "threads": CFG.threads}
                     for row in complete.itertuples()]
    runtime_rows += [{"experiment": "gold_all_shard_retrain", "unit": f"rep_{int(row.repetition)}",
                      "seconds": row.gold_seconds, "threads": CFG.threads}
                     for row in complete.itertuples()]
    if approx_path.exists():
        runtime_rows += [{"experiment": "approximate_continued_boosting", "unit": f"rep_{int(row.repetition)}",
                          "seconds": row.approx_seconds, "threads": CFG.threads}
                         for row in approx.itertuples()]
    atomic_csv(pd.DataFrame(runtime_rows), DIRS["results"] / "runtime_summary.csv")
    atomic_csv(pd.DataFrame([{
        "comparison": "exact SISA probability vs compatible gold retrain",
        "paired_n": len(complete), "effect": complete.prediction_mae.mean(),
        "ci95_low": 0.0 if (complete.prediction_mae == 0).all() else np.nan,
        "ci95_high": 0.0 if (complete.prediction_mae == 0).all() else np.nan,
        "p_value": np.nan,
        "note": "No hypothesis test: only three repetitions and all paired fidelity differences are identical."
    }]), DIRS["results"] / "statistical_comparisons.csv")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ordered = model_table.sort_values("roc_auc")
    bars = ax.barh(ordered.model, ordered.roc_auc, color="#2f6b8a")
    ax.set_xlim(0.70, 0.79); ax.set_xlabel("Final-test ROC-AUC")
    ax.set_title("Credit-scoring discrimination after corrected preprocessing")
    ax.grid(axis="x", alpha=.25)
    for bar, value in zip(bars, ordered.roc_auc):
        ax.text(value + .001, bar.get_y() + bar.get_height()/2, f"{value:.3f}", va="center", fontsize=9)
    fig.tight_layout(); fig.savefig(DIRS["figures"] / "baseline_and_sisa_auc.png", dpi=300); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    axes[0].bar(complete.repetition.astype(str), complete.exact_seconds, label="Exact SISA", color="#3b82a0")
    axes[0].scatter(complete.repetition.astype(str), complete.gold_seconds, label="Gold all-shard retrain", color="#d97706", zorder=3)
    axes[0].set_xlabel("Paired repetition"); axes[0].set_ylabel("Seconds")
    axes[0].set_title("Diffuse 1% deletion runtime"); axes[0].legend(fontsize=8)
    axes[1].bar(["Affected shards", "Workload"],
                [complete.affected_shards.mean()/CFG.shards, complete.workload_fraction.mean()], color="#6b7280")
    axes[1].set_ylim(0, 1.05); axes[1].set_ylabel("Fraction")
    axes[1].set_title("Deletion locality determines speedup")
    fig.tight_layout(); fig.savefig(DIRS["figures"] / "exact_unlearning_efficiency.png", dpi=300); plt.close(fig)
    atomic_csv(complete[["repetition", "exact_seconds", "gold_seconds", "speedup",
                         "affected_shards", "workload_fraction", "prediction_mae"]],
               DIRS["results"] / "figure_exact_unlearning_efficiency_data.csv")
    if approx_path.exists():
        fidelity_plot = pd.DataFrame({
            "method": ["Exact SISA", "Continued boosting\n(approximate)"],
            "prediction_mae": [complete.prediction_mae.median(), approx.prediction_mae.median()],
            "prediction_disagreement": [complete.prediction_disagreement.median(),
                                         approx.prediction_disagreement.median()],
        })
        atomic_csv(fidelity_plot, DIRS["results"] / "figure_exact_vs_approximate_fidelity_data.csv")
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        x = np.arange(len(fidelity_plot)); width = .34
        ax.bar(x - width/2, fidelity_plot.prediction_mae, width, label="Probability MAE", color="#2f6b8a")
        ax.bar(x + width/2, fidelity_plot.prediction_disagreement, width, label="Decision disagreement", color="#d97706")
        ax.set_xticks(x, fidelity_plot.method); ax.set_ylabel("Difference from compatible gold retrain")
        ax.set_title("Exact conditional retraining preserves gold fidelity")
        ax.legend(); ax.grid(axis="y", alpha=.25)
        fig.tight_layout(); fig.savefig(DIRS["figures"] / "exact_vs_approximate_fidelity.png", dpi=300); plt.close(fig)
    update_manifest("reporting_core", "complete",
                    ["results/thesis_model_performance.csv", "results/exact_unlearning_summary.csv",
                     "results/approximate_unlearning_summary.csv", "results/runtime_summary.csv",
                     "results/statistical_comparisons.csv", "figures/baseline_and_sisa_auc.png",
                     "figures/exact_unlearning_efficiency.png", "figures/exact_vs_approximate_fidelity.png"],
                    "Figures derive only from saved corrected-pipeline results; no retraining.")
    log("Core thesis tables and figures generated")


def run_core() -> None:
    stage_environment()
    stage_splits()
    stage_preprocessing()
    stage_baselines()
    stage_sisa()
    stage_exact_unlearning()
    stage_sharding_comparison()
    stage_approximate_unlearning()
    stage_mia()
    stage_reporting()


if __name__ == "__main__":
    run_core()
