"""
training.py - Model training with multiple validation strategies.

Strategies:
  'stratified'  - random stratified split (use only for quick sanity checks)
  'time_based'  - sessions 1-7 train, sessions 8-10 test
  'source_based'- train on original, test on external source
  'loso'        - Leave-One-Subject-Out by participant (recommended for generalisation)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def _scale_and_train(
    X_train, y_train, X_test,
    model_cls, model_kwargs,
    scaler=None,
):
    """Fit scaler on train, apply to test, train model, return (model, scaler, X_test_scaled)."""
    if scaler is None:
        scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model = model_cls(**model_kwargs)
    model.fit(X_train_s, y_train)
    return model, scaler, X_test_s


def _evaluate(model, X_test_scaled, y_test, label, validation_strategy):
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy [{label}]: {acc:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred, labels=sorted(set(y_test)))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=sorted(set(y_test)),
        yticklabels=sorted(set(y_test)),
    )
    plt.title(f"Confusion Matrix [{label}] – {validation_strategy}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()
    return acc, y_pred


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_with_validation(
    features, labels, feature_names, metadata,
    validation_strategy="loso",
    model_type="rf",
):
    """
    Train a classical ML model with the chosen validation strategy.

    Args:
        features:             np.ndarray (N, F)
        labels:               np.ndarray (N,)
        feature_names:        list[str]
        metadata:             list[dict] with keys: participant, session, data_source
        validation_strategy:  'loso' | 'stratified' | 'time_based' | 'source_based'
        model_type:           'rf' | 'svm' | 'gbm'

    Returns:
        model, scaler, results dict
    """
    print(f"\n=== Training [{model_type.upper()}] with [{validation_strategy}] validation ===")

    # ------------------------------------------------------------------
    # Model selection
    # ------------------------------------------------------------------
    model_configs = {
        "rf": (
            RandomForestClassifier,
            {"n_estimators": 200, "random_state": 42, "n_jobs": -1, "class_weight": "balanced"},
        ),
        "svm": (
            SVC,
            {"kernel": "rbf", "C": 10.0, "gamma": "scale", "random_state": 42, "class_weight": "balanced"},
        ),
        "gbm": (
            GradientBoostingClassifier,
            {"n_estimators": 150, "learning_rate": 0.1, "max_depth": 5, "random_state": 42},
        ),
    }
    if model_type not in model_configs:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose from: {list(model_configs)}")
    model_cls, model_kwargs = model_configs[model_type]

    # ------------------------------------------------------------------
    # Split
    # ------------------------------------------------------------------
    if validation_strategy == "loso":
        participants = [m["participant"] for m in metadata]
        unique_p = sorted(set(participants))

        if len(unique_p) < 2:
            print("Warning: only one participant found; falling back to stratified split.")
            return train_with_validation(
                features, labels, feature_names, metadata,
                validation_strategy="stratified", model_type=model_type,
            )

        participants_arr = np.array(participants)
        per_fold_acc = {}
        all_true, all_pred = [], []

        for test_p in unique_p:
            train_mask = participants_arr != test_p
            test_mask = ~train_mask
            X_tr, X_te = features[train_mask], features[test_mask]
            y_tr, y_te = labels[train_mask], labels[test_mask]

            print(f"\n  LOSO fold: test={test_p}  "
                  f"(train={len(y_tr)}, test={len(y_te)})")

            model, scaler, X_te_s = _scale_and_train(X_tr, y_tr, X_te, model_cls, model_kwargs)
            acc, y_pred = _evaluate(
                model, X_te_s, y_te,
                label=f"{model_type.upper()} test={test_p}",
                validation_strategy=validation_strategy,
            )
            per_fold_acc[test_p] = float(acc)
            all_true.extend(y_te.tolist())
            all_pred.extend(y_pred.tolist())

        mean_acc = float(np.mean(list(per_fold_acc.values())))
        print(f"\n  LOSO Mean Accuracy ({model_type.upper()}): {mean_acc:.4f}")
        print(f"  Per-fold: {per_fold_acc}")

        # Return the last fold's model and scaler for downstream use
        return model, scaler, {
            "validation_strategy": "loso",
            "model_type": model_type,
            "per_fold_accuracy": per_fold_acc,
            "mean_accuracy": mean_acc,
            "all_true": all_true,
            "all_pred": all_pred,
        }

    elif validation_strategy == "stratified":
        external_mask = np.array([m["data_source"] == "external" for m in metadata])
        internal_mask = ~external_mask

        X_int, y_int = features[internal_mask], labels[internal_mask]
        X_tr, X_te_int, y_tr, y_te_int = train_test_split(
            X_int, y_int, test_size=0.2, stratify=y_int, random_state=42
        )

        if external_mask.any():
            X_te_ext = features[external_mask]
            y_te_ext = labels[external_mask]
            X_te = np.vstack([X_te_int, X_te_ext])
            y_te = np.hstack([y_te_int, y_te_ext])
            test_sources = (["internal"] * len(y_te_int)
                            + ["external"] * len(y_te_ext))
        else:
            X_te, y_te = X_te_int, y_te_int
            test_sources = ["internal"] * len(y_te)

        print(f"  Train: {len(X_tr)}, Test: {len(X_te)}")
        print(f"  Test sources: internal={test_sources.count('internal')}, "
              f"external={test_sources.count('external')}")

        model, scaler, X_te_s = _scale_and_train(X_tr, y_tr, X_te, model_cls, model_kwargs)
        acc, y_pred = _evaluate(model, X_te_s, y_te, label=model_type.upper(),
                                validation_strategy=validation_strategy)

        return model, scaler, {
            "validation_strategy": "stratified",
            "model_type": model_type,
            "accuracy": float(acc),
        }

    elif validation_strategy == "time_based":
        train_sessions = {1, 2, 3, 4, 5, 6, 7}
        test_sessions = {8, 9, 10, 99}

        train_mask = np.array([m["session"] in train_sessions for m in metadata])
        test_mask = np.array([m["session"] in test_sessions for m in metadata])

        if not test_mask.any():
            print("No test-session data found; falling back to stratified.")
            return train_with_validation(
                features, labels, feature_names, metadata,
                validation_strategy="stratified", model_type=model_type,
            )

        X_tr, y_tr = features[train_mask], labels[train_mask]
        X_te, y_te = features[test_mask], labels[test_mask]
        print(f"  Train: {len(X_tr)}, Test: {len(X_te)}")

        model, scaler, X_te_s = _scale_and_train(X_tr, y_tr, X_te, model_cls, model_kwargs)
        acc, y_pred = _evaluate(model, X_te_s, y_te, label=model_type.upper(),
                                validation_strategy=validation_strategy)

        return model, scaler, {
            "validation_strategy": "time_based",
            "model_type": model_type,
            "accuracy": float(acc),
        }

    elif validation_strategy == "source_based":
        train_mask = np.array([m["data_source"] == "original" for m in metadata])
        test_mask = np.array([m["data_source"] == "external" for m in metadata])

        if not test_mask.any():
            print("No external data; falling back to time_based split.")
            return train_with_validation(
                features, labels, feature_names, metadata,
                validation_strategy="time_based", model_type=model_type,
            )

        X_tr, y_tr = features[train_mask], labels[train_mask]
        X_te, y_te = features[test_mask], labels[test_mask]
        print(f"  Train: {len(X_tr)}, Test: {len(X_te)}")

        model, scaler, X_te_s = _scale_and_train(X_tr, y_tr, X_te, model_cls, model_kwargs)
        acc, y_pred = _evaluate(model, X_te_s, y_te, label=model_type.upper(),
                                validation_strategy=validation_strategy)

        return model, scaler, {
            "validation_strategy": "source_based",
            "model_type": model_type,
            "accuracy": float(acc),
        }

    elif validation_strategy == "mixed_loso":
        # Session-based k-fold where each fold has sessions from BOTH participants
        # in both train and test, so the model always sees some data from each device.
        k = 5
        participants_arr = np.array([m["participant"] for m in metadata])
        sessions_arr = np.array([m["session"] for m in metadata])

        # Assign fold IDs by round-robining sessions within each participant
        fold_map = {}
        for p in sorted(set(participants_arr)):
            p_sessions = sorted(set(sessions_arr[participants_arr == p]))
            for i, s in enumerate(p_sessions):
                fold_map[(p, s)] = i % k

        window_folds = np.array([fold_map[(p, s)] for p, s in zip(participants_arr, sessions_arr)])

        per_fold_acc = {}
        all_true, all_pred = [], []
        last_model, last_scaler = None, None

        for fold_idx in range(k):
            test_mask = window_folds == fold_idx
            train_mask = ~test_mask
            if not test_mask.any() or not train_mask.any():
                continue

            X_tr, X_te = features[train_mask], features[test_mask]
            y_tr, y_te = labels[train_mask], labels[test_mask]
            test_parts = sorted(set(participants_arr[test_mask]))

            print(f"\n  Mixed-LOSO fold {fold_idx + 1}/{k}: "
                  f"participants in test={test_parts}  "
                  f"(train={len(y_tr)}, test={len(y_te)})")

            model, scaler, X_te_s = _scale_and_train(X_tr, y_tr, X_te, model_cls, model_kwargs)
            acc, y_pred = _evaluate(
                model, X_te_s, y_te,
                label=f"{model_type.upper()} fold={fold_idx + 1}",
                validation_strategy=validation_strategy,
            )
            per_fold_acc[f"fold_{fold_idx + 1}"] = float(acc)
            all_true.extend(y_te.tolist())
            all_pred.extend(y_pred.tolist())
            last_model, last_scaler = model, scaler

        mean_acc = float(np.mean(list(per_fold_acc.values())))
        print(f"\n  Mixed-LOSO Mean Accuracy ({model_type.upper()}): {mean_acc:.4f}")
        print(f"  Per-fold: {per_fold_acc}")

        return last_model, last_scaler, {
            "validation_strategy": "mixed_loso",
            "model_type": model_type,
            "per_fold_accuracy": per_fold_acc,
            "mean_accuracy": mean_acc,
            "all_true": all_true,
            "all_pred": all_pred,
        }

    else:
        raise ValueError(
            f"Unknown validation_strategy '{validation_strategy}'. "
            "Choose: 'loso', 'mixed_loso', 'stratified', 'time_based', 'source_based'."
        )
