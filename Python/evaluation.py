"""
evaluation.py – Structured benchmark and ablation protocol.

Evaluation protocols
--------------------
session_blocked
    Mixed k-fold where entire recording sessions are held out.
    Sessions are distributed across folds round-robin within each participant so
    that both participants (and both devices) appear in every fold's train and
    test set.  This estimates generalisation to unseen sessions from known
    participants/devices.

lopo (leave-one-participant-out)
    Train on all data from one participant, test on the other, then reverse.
    Because each participant uses a different device, this jointly measures
    cross-participant and cross-device robustness.

Models
------
Classical : SVM, Random Forest, Gradient Boosting
DL        : CNN, LSTM, CNN-LSTM, TCN  (require TensorFlow; skipped if absent)

Metrics
-------
Primary   : Macro-F1, Balanced Accuracy
Secondary : Accuracy

Ablation
--------
Classical – feature-index masking (drop / isolate one group at a time).
DL        – channel masking (zero out channels for a sensor/axis group).

Feature groups
--------------
sensor_acc, sensor_grav, sensor_gyro, sensor_magnet
axis_x, axis_y, axis_z
time_domain, freq_domain
gravity_aligned, magnitude, correlations
"""

import os
import json
import numpy as np
from collections import defaultdict
from datetime import datetime

import torch
from joblib import Parallel, delayed

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

from Python.posthoc import (
    CNNModel as _CNNModel,
    TCNModel as _TCNModel,
    LSTMModel as _LSTMModel,
    CNNLSTMModel as _CNNLSTMModel,
    ML4QSFlexiblePipeline,
    _train_pytorch,
    _predict_pytorch,
    DEVICE,
)


# ===========================================================================
# Metrics
# ===========================================================================

def compute_metrics(y_true, y_pred):
    """Return a dict with accuracy, macro_f1, and balanced_acc."""
    return {
        "accuracy":     float(accuracy_score(y_true, y_pred)),
        "macro_f1":     float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
    }


# ===========================================================================
# Feature group definitions (classical feature vector)
# ===========================================================================

_TIME_FEATURES = {
    "mean", "std", "max", "min", "median", "skew", "kurtosis",
    "q25", "q75", "q90", "q10", "mean_abs", "rms", "range", "zcr",
}
_FREQ_FEATURES = {
    "spectral_centroid", "spectral_spread",
    "low_band_power", "mid_band_power", "high_band_power",
    "spectral_entropy",
}
_GRAVITY_ALIGNED_PREFIXES = {"acc_vert", "acc_horiz", "gyro_vert", "gyro_horiz"}


def define_feature_groups(feature_names):
    """
    Map each feature index to one or more named groups.

    Returns
    -------
    dict[str, list[int]]
        Group name → sorted list of feature column indices.
    """
    groups = defaultdict(list)

    for i, name in enumerate(feature_names):
        parts = name.split("_")

        # ----- correlation features -----
        if name.startswith("correlation_"):
            groups["correlations"].append(i)
            continue

        # ----- magnitude features: e.g. acc_magnitude_mean -----
        if len(parts) >= 2 and parts[1] == "magnitude":
            groups["magnitude"].append(i)
            groups[f"sensor_{parts[0]}"].append(i)
            feat = "_".join(parts[2:])
            if feat in _TIME_FEATURES:
                groups["time_domain"].append(i)
            elif feat in _FREQ_FEATURES:
                groups["freq_domain"].append(i)
            continue

        # ----- gravity-aligned: e.g. acc_vert_mean, gyro_horiz_spectral_centroid -----
        prefix2 = "_".join(parts[:2]) if len(parts) >= 2 else parts[0]
        if prefix2 in _GRAVITY_ALIGNED_PREFIXES:
            groups["gravity_aligned"].append(i)
            feat = "_".join(parts[2:])
            if feat in _TIME_FEATURES:
                groups["time_domain"].append(i)
            elif feat in _FREQ_FEATURES:
                groups["freq_domain"].append(i)
            continue

        # ----- regular sensor-axis features: e.g. acc_x_mean, gyro_z_spectral_centroid -----
        if len(parts) >= 3:
            sensor = parts[0]           # acc | grav | gyro | magnet
            axis   = parts[1]           # x | y | z
            feat   = "_".join(parts[2:])

            groups[f"sensor_{sensor}"].append(i)
            if axis in ("x", "y", "z"):
                groups[f"axis_{axis}"].append(i)
            if feat in _TIME_FEATURES:
                groups["time_domain"].append(i)
            elif feat in _FREQ_FEATURES:
                groups["freq_domain"].append(i)

    return {k: sorted(v) for k, v in groups.items()}


# ===========================================================================
# Split strategies
# ===========================================================================

def session_blocked_folds(metadata, k=5):
    """
    Mixed k-fold: sessions are distributed round-robin across folds within
    each participant so that both participants appear in every fold's test set.

    Yields
    ------
    (train_indices, test_indices) : np.ndarray, np.ndarray
    """
    participants = np.array([m["participant"] for m in metadata])
    sessions     = np.array([m["session"]     for m in metadata])

    fold_map = {}
    for p in sorted(set(participants)):
        for idx, s in enumerate(sorted(set(sessions[participants == p]))):
            fold_map[(p, s)] = idx % k

    window_folds = np.array([fold_map[(p, s)] for p, s in zip(participants, sessions)])
    all_idx = np.arange(len(metadata))

    for fold_idx in range(k):
        test_mask  = window_folds == fold_idx
        train_mask = ~test_mask
        if not test_mask.any() or not train_mask.any():
            continue
        yield all_idx[train_mask], all_idx[test_mask]


def lopo_folds(metadata):
    """
    Leave-One-Participant-Out: one fold per participant.

    Yields
    ------
    (train_indices, test_indices, test_participant) : np.ndarray, np.ndarray, str
    """
    participants = np.array([m["participant"] for m in metadata])
    all_idx = np.arange(len(metadata))

    for p in sorted(set(participants)):
        test_mask  = participants == p
        train_mask = ~test_mask
        if not test_mask.any() or not train_mask.any():
            continue
        yield all_idx[train_mask], all_idx[test_mask], p


# ===========================================================================
# Classical models
# ===========================================================================

_CLASSICAL_CONFIGS = {
    "rf": (
        RandomForestClassifier,
        {"n_estimators": 200, "random_state": 42, "n_jobs": -1, "class_weight": "balanced"},
    ),
    "svm": (
        SVC,
        {"kernel": "rbf", "C": 10.0, "gamma": "scale", "random_state": 42,
         "class_weight": "balanced"},
    ),
    "gbm": (
        GradientBoostingClassifier,
        {"n_estimators": 150, "learning_rate": 0.1, "max_depth": 5, "random_state": 42},
    ),
}


def _fit_classical(X_tr, y_tr, X_te, model_type, n_jobs_override=None):
    """Fit scaler + model on training data; return (y_pred, model, scaler)."""
    cls, kwargs = _CLASSICAL_CONFIGS[model_type]
    if n_jobs_override is not None and "n_jobs" in kwargs:
        kwargs = {**kwargs, "n_jobs": n_jobs_override}
    scaler = StandardScaler()
    model  = cls(**kwargs)
    model.fit(scaler.fit_transform(X_tr), y_tr)
    return model.predict(scaler.transform(X_te)), model, scaler


def _ablation_condition(cond_name, mask, features, labels, folds_iter, fold_labels, model_types):
    """
    Evaluate one ablation condition across pre-computed folds.
    Called in parallel — uses n_jobs=1 for RF so outer parallelism is not over-subscribed.
    """
    X   = features[:, mask]
    res = {mt: {"per_fold": [], "y_true": [], "y_pred": []} for mt in model_types}

    for (train_idx, test_idx), flabel in zip(folds_iter, fold_labels):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = labels[train_idx], labels[test_idx]
        for mt in model_types:
            y_pred, _, _ = _fit_classical(X_tr, y_tr, X_te, mt, n_jobs_override=1)
            m = compute_metrics(y_te, y_pred)
            res[mt]["per_fold"].append({"fold": flabel, **m})
            res[mt]["y_true"].extend(y_te.tolist())
            res[mt]["y_pred"].extend(y_pred.tolist())

    for mt in model_types:
        fdata = res[mt]["per_fold"]
        res[mt]["mean_macro_f1"]     = float(np.mean([f["macro_f1"]     for f in fdata]))
        res[mt]["mean_balanced_acc"] = float(np.mean([f["balanced_acc"] for f in fdata]))
        res[mt]["mean_accuracy"]     = float(np.mean([f["accuracy"]     for f in fdata]))

    return cond_name, res


def run_classical(
    features,
    labels,
    feature_names,
    metadata,
    strategy="session_blocked",
    model_types=("rf", "svm", "gbm"),
    feature_mask=None,
    verbose=True,
    checkpoint_path=None,
):
    """
    Benchmark classical ML models under one evaluation protocol.

    Parameters
    ----------
    features      : np.ndarray (N, F)
    labels        : np.ndarray (N,)
    feature_names : list[str]  (length F)
    metadata      : list[dict] with keys 'participant', 'session', 'data_source'
    strategy      : 'session_blocked' | 'lopo'
    model_types   : subset of ('rf', 'svm', 'gbm')
    feature_mask  : array of column indices to retain (None = all columns)
    verbose       : print per-fold results

    Returns
    -------
    dict[model_type, dict]
        Keys per model: 'per_fold', 'y_true', 'y_pred',
        'mean_macro_f1', 'mean_balanced_acc', 'mean_accuracy'
    """
    X = features[:, feature_mask] if feature_mask is not None else features
    y = labels

    # Load existing checkpoint to resume from
    results = {mt: {"per_fold": [], "y_true": [], "y_pred": []} for mt in model_types}
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, encoding="utf-8") as fh:
                saved = json.load(fh)
            for mt in model_types:
                if mt in saved:
                    results[mt] = saved[mt]
            completed_keys = {
                (mt, pf["fold"])
                for mt in model_types
                for pf in results[mt]["per_fold"]
            }
            print(f"  [classical checkpoint] resuming — {len(completed_keys)} (model,fold) pairs already done")
        except Exception as exc:
            print(f"  [classical checkpoint] could not load ({exc}); starting fresh")
            results = {mt: {"per_fold": [], "y_true": [], "y_pred": []} for mt in model_types}
    else:
        completed_keys = set()

    if strategy == "session_blocked":
        folds_iter  = list(session_blocked_folds(metadata, k=5))
        fold_labels = [f"fold_{i + 1}" for i in range(len(folds_iter))]
    elif strategy == "lopo":
        raw = list(lopo_folds(metadata))
        folds_iter  = [(tr, te) for tr, te, _ in raw]
        fold_labels = [p for _, _, p in raw]
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose 'session_blocked' or 'lopo'.")

    for (train_idx, test_idx), flabel in zip(folds_iter, fold_labels):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        for mt in model_types:
            if (mt, flabel) in completed_keys:
                if verbose:
                    print(f"  [{strategy}] {mt.upper():4s} {flabel}: skipped (checkpoint)")
                continue
            y_pred, _, _ = _fit_classical(X_tr, y_tr, X_te, mt)
            m = compute_metrics(y_te, y_pred)
            results[mt]["per_fold"].append({"fold": flabel, **m})
            results[mt]["y_true"].extend(y_te.tolist())
            results[mt]["y_pred"].extend(y_pred.tolist())
            if verbose:
                print(f"  [{strategy}] {mt.upper():4s} {flabel}: "
                      f"macro_f1={m['macro_f1']:.4f}  "
                      f"bal_acc={m['balanced_acc']:.4f}  "
                      f"acc={m['accuracy']:.4f}")
            if checkpoint_path:
                _atomic_save_json(results, checkpoint_path)

    for mt in model_types:
        fdata = results[mt]["per_fold"]
        results[mt]["mean_macro_f1"]     = float(np.mean([f["macro_f1"]     for f in fdata]))
        results[mt]["mean_balanced_acc"] = float(np.mean([f["balanced_acc"] for f in fdata]))
        results[mt]["mean_accuracy"]     = float(np.mean([f["accuracy"]     for f in fdata]))

    return results


# ===========================================================================
# Classical feature ablation
# ===========================================================================

def run_ablation(
    features,
    labels,
    feature_names,
    metadata,
    strategy="session_blocked",
    model_types=("rf", "svm", "gbm"),
    n_jobs=-1,
    verbose=False,
    baseline_results=None,
):
    """
    Automated grouped feature ablation for classical models.

    For every feature group two ablation conditions are tested:

    drop_<group>
        Remove that group's features; train on the remaining columns.
        A large performance drop signals that this group is important.

    isolate_<group>
        Use only that group's features.
        High performance indicates the group alone is informative.

    The full-feature baseline ('full') is always included first.

    Parameters
    ----------
    features, labels, feature_names, metadata : as in run_classical()
    strategy      : 'session_blocked' | 'lopo'
    model_types   : classical model types to evaluate
    n_jobs        : parallel workers for conditions (-1 = all CPU cores)
    verbose       : unused (kept for API compatibility)

    Returns
    -------
    ablation_results : dict[condition, dict[model_type, metrics]]
    feature_groups   : dict[group_name, list[int]]  (for reference)
    """
    groups  = define_feature_groups(feature_names)
    all_idx = np.arange(features.shape[1])

    # Build ordered conditions: drop/isolate per group (full handled separately)
    conditions = {}
    for gname, gidx in sorted(groups.items()):
        if not gidx:
            continue
        gset = set(gidx)
        keep = np.array([i for i in all_idx if i not in gset])
        if len(keep) > 0:
            conditions[f"drop_{gname}"] = keep
        iso = np.array(gidx)
        if len(iso) > 0:
            conditions[f"isolate_{gname}"] = iso

    # Pre-compute fold splits once — reused across all conditions
    if strategy == "session_blocked":
        folds_iter  = list(session_blocked_folds(metadata, k=5))
        fold_labels = [f"fold_{i + 1}" for i in range(len(folds_iter))]
    elif strategy == "lopo":
        raw = list(lopo_folds(metadata))
        folds_iter  = [(tr, te) for tr, te, _ in raw]
        fold_labels = [p for _, _, p in raw]
    else:
        raise ValueError(f"Unknown strategy '{strategy}'.")

    total = len(conditions) + (0 if baseline_results else 1)
    print(f"  Running {total} ablation conditions in parallel (n_jobs={n_jobs})")

    # Conditions to actually compute (exclude "full" if baseline already provided)
    compute_conditions = conditions if baseline_results else {"full": all_idx, **conditions}

    raw_pairs: list = list(Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_ablation_condition)(
            cond_name, mask, features, labels, folds_iter, fold_labels, model_types
        )
        for cond_name, mask in compute_conditions.items()
    ) or [])

    ablation_results: dict = {}
    if baseline_results is not None:
        ablation_results["full"] = baseline_results
    for cond_name, res in raw_pairs:
        ablation_results[cond_name] = res
    return ablation_results, groups


# ===========================================================================
# Deep learning models (TensorFlow optional)
# ===========================================================================

# Channel assignments for the 10-channel DL sequences
# (acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z,
#  acc_vert, acc_horiz, gyro_vert, gyro_horiz)
DL_CHANNEL_GROUPS = {
    "sensor_acc":      [0, 1, 2],
    "sensor_gyro":     [3, 4, 5],
    "gravity_aligned": [6, 7, 8, 9],
    "axis_x":          [0, 3],
    "axis_y":          [1, 4],
    "axis_z":          [2, 5],
}


def _build_dl_model(model_type, input_shape, num_classes):
    """Instantiate a PyTorch DL model by type string."""
    _, C = input_shape
    if model_type == "cnn":
        return _CNNModel(C, num_classes)
    if model_type == "lstm":
        return _LSTMModel(C, num_classes)
    if model_type == "cnn_lstm":
        return _CNNLSTMModel(C, num_classes)
    if model_type == "tcn":
        return _TCNModel(C, num_classes)
    raise ValueError(f"Unknown DL model type '{model_type}'. "
                     "Choose from: cnn, lstm, cnn_lstm, tcn")


def run_dl(
    sequences,
    labels,
    metadata,
    strategy="session_blocked",
    model_types=("cnn", "lstm", "cnn_lstm", "tcn"),
    channel_mask=None,
    epochs=50,
    batch_size=32,
    verbose=True,
    checkpoint_path=None,
):
    """
    Benchmark DL models under one evaluation protocol.

    Parameters
    ----------
    sequences    : np.ndarray (N, T, C)
    labels       : np.ndarray (N,)
    metadata     : list[dict] with keys 'participant', 'session'
    strategy     : 'session_blocked' | 'lopo'
    model_types  : subset of ('cnn', 'lstm', 'cnn_lstm', 'tcn')
    channel_mask : list of channel indices to retain (None = all)
    epochs       : max training epochs (early-stopping may reduce this)
    batch_size   : training batch size
    verbose      : print per-fold results

    Returns
    -------
    dict[model_type, dict]  or  {} if TensorFlow is unavailable
    """
    seqs = sequences[:, :, channel_mask] if channel_mask is not None else sequences
    N, T, C = seqs.shape
    input_shape = (T, C)

    le = LabelEncoder()
    le.fit(np.unique(labels))
    num_classes = len(le.classes_)

    # Load existing checkpoint to resume from
    results = {mt: {"per_fold": [], "y_true": [], "y_pred": []} for mt in model_types}
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, encoding="utf-8") as fh:
                saved = json.load(fh)
            for mt in model_types:
                if mt in saved:
                    results[mt] = saved[mt]
            completed_keys = {
                (mt, pf["fold"])
                for mt in model_types
                for pf in results[mt]["per_fold"]
            }
            print(f"  [DL checkpoint] resuming — {len(completed_keys)} (model,fold) pairs already done")
        except Exception as exc:
            print(f"  [DL checkpoint] could not load ({exc}); starting fresh")
            results = {mt: {"per_fold": [], "y_true": [], "y_pred": []} for mt in model_types}
            completed_keys = set()
    else:
        completed_keys = set()

    if strategy == "session_blocked":
        folds_iter  = list(session_blocked_folds(metadata, k=5))
        fold_labels = [f"fold_{i + 1}" for i in range(len(folds_iter))]
    elif strategy == "lopo":
        raw = list(lopo_folds(metadata))
        folds_iter  = [(tr, te) for tr, te, _ in raw]
        fold_labels = [p for _, _, p in raw]
    else:
        raise ValueError(f"Unknown strategy '{strategy}'.")

    for (train_idx, test_idx), flabel in zip(folds_iter, fold_labels):
        X_tr_raw, X_te_raw = seqs[train_idx], seqs[test_idx]
        y_tr, y_te         = labels[train_idx], labels[test_idx]

        # Only compute scaler/split once per fold if any model in the fold needs training
        fold_models_needed = [mt for mt in model_types if (mt, flabel) not in completed_keys]
        if not fold_models_needed:
            if verbose:
                print(f"  [{strategy}] fold {flabel}: all models skipped (checkpoint)")
            continue

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr_raw.reshape(-1, C)).reshape(-1, T, C)
        X_te_s = scaler.transform(X_te_raw.reshape(-1, C)).reshape(-1, T, C)

        y_tr_enc = le.transform(y_tr)
        y_te_enc = le.transform(y_te)

        X_tr_f, X_val, y_tr_f, y_val = train_test_split(
            X_tr_s, y_tr_enc, test_size=0.15, stratify=y_tr_enc, random_state=42
        )

        present      = np.unique(y_tr_f)
        cw_values    = compute_class_weight("balanced", classes=present, y=y_tr_f)
        class_weight = {int(c): float(w) for c, w in zip(present, cw_values)}
        for i in range(num_classes):           # fill any class absent from training fold
            class_weight.setdefault(i, 1.0)

        for mt in fold_models_needed:
            model = _build_dl_model(mt, input_shape, num_classes)
            model, _ = _train_pytorch(
                model, X_tr_f, y_tr_f, X_val, y_val,
                lr=0.001, epochs=epochs, batch_size=batch_size,
                class_weight=class_weight,
            )

            y_pred_enc = _predict_pytorch(model, X_te_s)
            y_pred     = le.inverse_transform(y_pred_enc)
            y_true     = le.inverse_transform(y_te_enc)

            m = compute_metrics(y_true, y_pred)
            results[mt]["per_fold"].append({"fold": flabel, **m})
            results[mt]["y_true"].extend(y_true.tolist())
            results[mt]["y_pred"].extend(y_pred.tolist())

            if verbose:
                print(f"  [{strategy}] {mt.upper():8s} {flabel}: "
                      f"macro_f1={m['macro_f1']:.4f}  "
                      f"bal_acc={m['balanced_acc']:.4f}  "
                      f"acc={m['accuracy']:.4f}")

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if checkpoint_path:
                _atomic_save_json(results, checkpoint_path)

    for mt in model_types:
        fdata = results[mt]["per_fold"]
        if fdata:
            results[mt]["mean_macro_f1"]     = float(np.mean([f["macro_f1"]     for f in fdata]))
            results[mt]["mean_balanced_acc"] = float(np.mean([f["balanced_acc"] for f in fdata]))
            results[mt]["mean_accuracy"]     = float(np.mean([f["accuracy"]     for f in fdata]))

    return results


def run_dl_ablation(
    sequences,
    labels,
    metadata,
    strategy="session_blocked",
    model_types=("cnn", "lstm", "cnn_lstm", "tcn"),
    epochs=50,
    batch_size=32,
    verbose=False,
):
    """
    Channel-based ablation for DL models.

    Only 'drop' conditions are tested (removing a channel group).
    'Isolate' with very few channels tends to be degenerate given the
    window size and is omitted.

    Returns
    -------
    dict[condition, dict[model_type, metrics]]
    """
    N, T, C    = sequences.shape
    all_ch     = list(range(C))
    conditions = {"full": None}

    for gname, ch_idx in sorted(DL_CHANNEL_GROUPS.items()):
        valid = [c for c in ch_idx if c < C]
        if not valid:
            continue
        keep = [c for c in all_ch if c not in valid]
        if keep:
            conditions[f"drop_{gname}"] = keep

    ablation_results = {}
    total = len(conditions)
    for ci, (cond_name, mask) in enumerate(conditions.items(), 1):
        n_ch = C if mask is None else len(mask)
        print(f"  [{ci}/{total}] DL ablation: {cond_name}  ({n_ch} channels)")
        ablation_results[cond_name] = run_dl(
            sequences, labels, metadata,
            strategy=strategy,
            model_types=model_types,
            channel_mask=mask,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
        )

    return ablation_results


# ===========================================================================
# Reporting helpers
# ===========================================================================

def print_benchmark_summary(classical_results, dl_results, strategy):
    """Print a compact comparison table for one evaluation protocol."""
    print(f"\n{'=' * 72}")
    print(f"  BENCHMARK SUMMARY – {strategy.upper()}")
    print(f"{'=' * 72}")
    print(f"  {'Model':<14} {'Macro-F1':>10} {'Bal. Acc':>10} {'Accuracy':>10}")
    print(f"  {'-' * 44}")

    for mt, res in classical_results.items():
        print(f"  {mt.upper():<14} "
              f"{res['mean_macro_f1']:>10.4f} "
              f"{res['mean_balanced_acc']:>10.4f} "
              f"{res['mean_accuracy']:>10.4f}")

    for mt, res in (dl_results or {}).items():
        if res and "mean_macro_f1" in res:
            print(f"  {mt.upper():<14} "
                  f"{res['mean_macro_f1']:>10.4f} "
                  f"{res['mean_balanced_acc']:>10.4f} "
                  f"{res['mean_accuracy']:>10.4f}")

    print(f"{'=' * 72}\n")


def print_ablation_summary(ablation_results, strategy, metric="macro_f1"):
    """
    Print per-condition performance and delta relative to full-feature baseline.

    Non-baseline rows show (value - baseline_value).  A large negative delta
    for 'drop_X' means group X is important.  A large positive delta for
    'isolate_X' means group X alone is sufficient.
    """
    print(f"\n{'=' * 72}")
    print(f"  ABLATION SUMMARY – {strategy.upper()}  (metric: {metric})")
    print(f"  Rows: absolute value for 'full'; delta vs baseline for all others.")
    print(f"{'=' * 72}")

    model_types = list(next(iter(ablation_results.values())).keys()) if ablation_results else []
    header = f"  {'Condition':<34}" + "".join(f"{mt.upper():>10}" for mt in model_types)
    print(header)
    print(f"  {'-' * (34 + 10 * len(model_types))}")

    baseline = ablation_results.get("full", {})

    for cond_name, res in ablation_results.items():
        row = f"  {cond_name:<34}"
        for mt in model_types:
            val = res.get(mt, {}).get(f"mean_{metric}", float("nan"))
            if cond_name == "full":
                row += f"{val:>10.4f}"
            else:
                base = baseline.get(mt, {}).get(f"mean_{metric}", float("nan"))
                delta = val - base
                row += f"{delta:>+10.4f}"
        print(row)

    print(f"{'=' * 72}\n")


def plot_ablation(ablation_results, strategy, metric="macro_f1",
                  model_type="rf", save_path=None):
    """
    Bar chart showing performance delta per ablation condition for one model.
    Drop conditions are shown in red (negative = important group),
    isolate conditions in blue, and the full baseline as a dashed reference.
    """
    baseline_val = ablation_results.get("full", {}).get(model_type, {}).get(
        f"mean_{metric}", float("nan")
    )

    names, deltas, colors = [], [], []
    for cond, res in ablation_results.items():
        if cond == "full":
            continue
        val = res.get(model_type, {}).get(f"mean_{metric}", float("nan"))
        delta = val - baseline_val
        names.append(cond.replace("drop_", "").replace("isolate_", ""))
        deltas.append(delta)
        colors.append("#d32f2f" if cond.startswith("drop") else "#1976d2")

    if not names:
        return

    order = np.argsort(deltas)
    names  = [names[i]  for i in order]
    deltas = [deltas[i] for i in order]
    colors = [colors[i] for i in order]
    labels = [
        f"drop: {n}" if c == "#d32f2f" else f"isolate: {n}"
        for n, c in zip(names, colors)
    ]

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.45)))
    bars = ax.barh(range(len(names)), deltas, color=colors, edgecolor="white")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"Δ {metric} vs full baseline")
    ax.set_title(
        f"Feature-group ablation – {model_type.upper()} [{strategy}]\n"
        f"Baseline {metric}: {baseline_val:.4f}"
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def _json_convert(obj):
    """JSON serialiser that handles numpy types and NaN→null."""
    if isinstance(obj, float) and obj != obj:   # NaN
        return None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _atomic_save_json(data, path):
    """Write *data* to *path* atomically (temp file + os.replace)."""
    import tempfile
    abs_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(abs_path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, default=_json_convert, indent=2)
        os.replace(tmp, abs_path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def save_results(results_dict, out_path):
    """Serialise results to JSON atomically."""
    _atomic_save_json(results_dict, out_path)
    print(f"  Results saved → {out_path}")


# ===========================================================================
# Main entry point
# ===========================================================================

def run_full_evaluation(
    features,
    labels,
    feature_names,
    metadata,
    sequences=None,
    run_dl_models=True,
    run_ablation_classical=True,
    run_ablation_dl=False,
    classical_models=("rf", "svm", "gbm"),
    dl_model_types=("cnn", "lstm", "cnn_lstm", "tcn"),
    dl_epochs=50,
    dl_batch_size=32,
    plots_dir="./Plots",
    save_dir="./Results",
    verbose=True,
    strategies=("session_blocked", "lopo"),
):
    """
    Orchestrate the full evaluation and ablation suite.

    Parameters
    ----------
    features              : np.ndarray (N, F) – hand-crafted feature matrix
    labels                : np.ndarray (N,)
    feature_names         : list[str] of length F
    metadata              : list[dict] with 'participant', 'session', 'data_source'
    sequences             : np.ndarray (N, T, C) raw sequences for DL (None → skip DL)
    run_dl_models         : run DL benchmark (requires TensorFlow)
    run_ablation_classical: run classical feature-group ablation
    run_ablation_dl       : run DL channel ablation (slow; off by default)
    classical_models      : model types for classical benchmark/ablation
    dl_model_types        : model types for DL benchmark/ablation
    dl_epochs             : max epochs per DL fold
    dl_batch_size         : DL batch size
    plots_dir             : directory to save ablation bar charts
    save_dir              : directory to save JSON results
    verbose               : per-fold output for benchmark runs

    Returns
    -------
    dict with nested results for both evaluation protocols and all conditions
    """
    os.makedirs(save_dir,  exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    partial_path = os.path.join(save_dir, "evaluation_partial.json")
    all_results  = {}

    for strategy in strategies:
        print(f"\n{'#' * 72}")
        print(f"#  EVALUATION PROTOCOL: {strategy.upper()}")
        print(f"{'#' * 72}")
        strat_res = {}

        classical_ckpt = os.path.join(save_dir, f"checkpoint_{strategy}_classical.json")
        dl_ckpt        = os.path.join(save_dir, f"checkpoint_{strategy}_dl.json")

        # ---- Classical benchmark ----------------------------------------
        print(f"\n--- Classical ML Benchmark [{strategy}] ---")
        classical_res = run_classical(
            features, labels, feature_names, metadata,
            strategy=strategy,
            model_types=classical_models,
            verbose=verbose,
            checkpoint_path=classical_ckpt,
        )
        strat_res["classical"] = classical_res
        # Save partial after classical in case DL crashes
        all_results[strategy] = strat_res
        _atomic_save_json(all_results, partial_path)

        # ---- DL benchmark -----------------------------------------------
        dl_res = {}
        if run_dl_models and sequences is not None:
            print(f"\n--- DL Benchmark [{strategy}] ---")
            dl_res = run_dl(
                sequences, labels, metadata,
                strategy=strategy,
                model_types=dl_model_types,
                epochs=dl_epochs,
                batch_size=dl_batch_size,
                verbose=verbose,
                checkpoint_path=dl_ckpt,
            )
            strat_res["dl"] = dl_res
            # Save partial after DL in case ablation crashes
            all_results[strategy] = strat_res
            _atomic_save_json(all_results, partial_path)

        print_benchmark_summary(classical_res, dl_res, strategy)

        # ---- Classical feature ablation ---------------------------------
        if run_ablation_classical:
            print(f"\n--- Classical Feature Ablation [{strategy}] ---")
            abl_res, feat_groups = run_ablation(
                features, labels, feature_names, metadata,
                strategy=strategy,
                model_types=classical_models,
                baseline_results=classical_res,
            )
            strat_res["ablation_classical"]    = abl_res
            strat_res["feature_group_indices"] = {k: list(v) for k, v in feat_groups.items()}

            print_ablation_summary(abl_res, strategy, metric="macro_f1")
            print_ablation_summary(abl_res, strategy, metric="balanced_acc")

            # Save ablation bar charts for each classical model
            for mt in classical_models:
                plot_ablation(
                    abl_res, strategy, metric="macro_f1", model_type=mt,
                    save_path=os.path.join(
                        plots_dir, f"ablation_{strategy}_{mt}_macro_f1.png"
                    ),
                )

        # ---- DL channel ablation ----------------------------------------
        if run_ablation_dl and sequences is not None:
            print(f"\n--- DL Channel Ablation [{strategy}] ---")
            dl_abl_res = run_dl_ablation(
                sequences, labels, metadata,
                strategy=strategy,
                model_types=dl_model_types,
                epochs=dl_epochs,
                batch_size=dl_batch_size,
                verbose=False,
            )
            strat_res["ablation_dl"] = dl_abl_res
            print_ablation_summary(dl_abl_res, strategy, metric="macro_f1")

        all_results[strategy] = strat_res
        # Save after every strategy so a crash on the next one doesn't lose this work
        save_results(all_results, partial_path)

    out_path = os.path.join(save_dir, f"evaluation_{timestamp}.json")
    save_results(all_results, out_path)
    return all_results
