"""
posthoc.py - ML4QS Deep Learning Pipeline
ML4QSFlexiblePipeline: TCN, LSTM, CNN-LSTM with Optuna tuning and LOSO evaluation.
"""

import os
import json
import pickle
import numpy as np
from datetime import datetime
from pathlib import Path

# TensorFlow/Keras are imported lazily inside _require_keras() so that
# importing this module does not crash when TF is broken or absent.
_keras = None

def _require_keras():
    """Import Keras once and cache it; raise a clear error if unavailable."""
    global _keras
    if _keras is not None:
        return _keras
    try:
        import tensorflow as tf  # noqa: F401
        from tensorflow import keras as _k
        _keras = _k
        return _keras
    except Exception as e:
        raise ImportError(
            "\n\nTensorFlow could not be imported:\n"
            f"  {e}\n\n"
            "To fix, either:\n"
            "  1. Switch the Jupyter kernel to Python 3.13 (which has TF 2.21 installed), or\n"
            "  2. Reinstall TF in this kernel:  pip install --force-reinstall tensorflow\n"
        ) from e

# optuna is only needed for optimize_tcn; import it lazily
def _require_optuna():
    try:
        import optuna as _o
        _o.logging.set_verbosity(_o.logging.WARNING)
        return _o
    except ImportError:
        raise ImportError(
            "optuna is not installed. Run:  pip install optuna\n"
            "It is only needed when optimize=True in evaluate_loso()."
        )

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import matplotlib.pyplot as plt
import seaborn as sns

from Python.integrations import load_gym_dataset
from Python.windows import create_adaptive_windows
from Python.feature_engineering import engineer_robust_features as _engineer_robust_features


# ---------------------------------------------------------------------------
# Activity label map (field dataset labels → human-readable names)
# ---------------------------------------------------------------------------
ACTIVITY_NAMES = {
    "elliptical": "Elliptical",
    "lat_pull": "Lat Pull-Down",
    "row": "Rowing Machine",
}


# ---------------------------------------------------------------------------
# ML4QSFlexiblePipeline
# ---------------------------------------------------------------------------

class ML4QSFlexiblePipeline:
    """
    Flexible pipeline supporting TCN, LSTM, and CNN-LSTM model architectures
    with LOSO (Leave-One-Subject-Out) evaluation and Optuna hyperparameter tuning.
    """

    def __init__(self, sampling_rate=50, window_size=128):
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.label_encoder = LabelEncoder()
        self.best_params = {}

    # ------------------------------------------------------------------
    # Feature engineering (fixes the 'engineer_features' AttributeError)
    # ------------------------------------------------------------------

    def engineer_robust_features(self, windowed_data):
        """Extract hand-crafted features from windowed data."""
        return _engineer_robust_features(windowed_data, self.sampling_rate)

    # Alias so that any caller using the old name does not crash
    def engineer_features(self, windowed_data):
        return self.engineer_robust_features(windowed_data)

    # ------------------------------------------------------------------
    # Sequence preparation for DL models
    # ------------------------------------------------------------------

    def prepare_sequences(self, windowed_data):
        """
        Convert list of window dicts to (N, window_size, C) numpy array.
        Channels: acc_x/y/z, gyro_x/y/z, plus gravity-aligned acc_vert,
        acc_horiz, gyro_vert, gyro_horiz when available (orientation-invariant).
        """
        base_channels = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
        aligned_channels = ["acc_vert", "acc_horiz", "gyro_vert", "gyro_horiz"]
        has_aligned = all(ch in windowed_data[0] for ch in aligned_channels) if windowed_data else False
        channels = base_channels + aligned_channels if has_aligned else base_channels
        sequences, labels, metadata = [], [], []

        for w in windowed_data:
            seq = np.stack([w[ch] for ch in channels], axis=-1)
            n = len(seq)
            if n < self.window_size:
                pad = self.window_size - n
                seq = np.pad(seq, ((0, pad), (0, 0)), mode="reflect")
            else:
                seq = seq[: self.window_size]
            sequences.append(seq)
            labels.append(w["activity"])
            metadata.append(
                {
                    "participant": w["participant"],
                    "session": w["session"],
                    "data_source": w["data_source"],
                }
            )

        return np.array(sequences, dtype=np.float32), np.array(labels), metadata

    # ------------------------------------------------------------------
    # Model builders
    # ------------------------------------------------------------------

    def build_tcn(
        self,
        input_shape,
        num_classes,
        nb_filters=64,
        kernel_size=3,
        nb_stacks=1,
        dilations=None,
        dropout_rate=0.2,
    ):
        """Temporal Convolutional Network with residual blocks."""
        keras = _require_keras()
        kl = keras.layers
        if dilations is None:
            dilations = [1, 2, 4, 8]

        inputs = kl.Input(shape=input_shape)
        x = inputs

        for _ in range(nb_stacks):
            for d in dilations:
                res = x
                x = kl.Conv1D(
                    nb_filters, kernel_size, padding="causal",
                    dilation_rate=d, activation="relu",
                )(x)
                x = kl.BatchNormalization()(x)
                x = kl.Dropout(dropout_rate)(x)
                x = kl.Conv1D(
                    nb_filters, kernel_size, padding="causal",
                    dilation_rate=d, activation="relu",
                )(x)
                x = kl.BatchNormalization()(x)
                # Residual projection if needed
                if res.shape[-1] != nb_filters:
                    res = kl.Conv1D(nb_filters, 1, padding="same")(res)
                x = kl.Add()([x, res])
                x = kl.Dropout(dropout_rate)(x)

        x = kl.GlobalAveragePooling1D()(x)
        x = kl.Dense(64, activation="relu")(x)
        x = kl.Dropout(dropout_rate)(x)
        outputs = kl.Dense(num_classes, activation="softmax")(x)

        return keras.Model(inputs, outputs, name="TCN")

    def build_lstm(self, input_shape, num_classes, units=128, dropout_rate=0.3):
        """Stacked LSTM model."""
        keras = _require_keras()
        kl = keras.layers
        model = keras.Sequential(
            [
                kl.Input(shape=input_shape),
                kl.LSTM(units, return_sequences=True),
                kl.Dropout(dropout_rate),
                kl.LSTM(units // 2, return_sequences=False),
                kl.Dropout(dropout_rate),
                kl.Dense(64, activation="relu"),
                kl.Dropout(dropout_rate),
                kl.Dense(num_classes, activation="softmax"),
            ],
            name="LSTM",
        )
        return model

    def build_cnn_lstm(
        self,
        input_shape,
        num_classes,
        conv_filters=64,
        kernel_size=3,
        lstm_units=64,
        dropout_rate=0.3,
    ):
        """CNN feature extractor + LSTM temporal model."""
        keras = _require_keras()
        kl = keras.layers
        model = keras.Sequential(
            [
                kl.Input(shape=input_shape),
                kl.Conv1D(conv_filters, kernel_size, activation="relu", padding="same"),
                kl.BatchNormalization(),
                kl.Conv1D(conv_filters // 2, kernel_size, activation="relu", padding="same"),
                kl.BatchNormalization(),
                kl.MaxPooling1D(pool_size=2),
                kl.Dropout(dropout_rate),
                kl.LSTM(lstm_units, return_sequences=False),
                kl.Dropout(dropout_rate),
                kl.Dense(64, activation="relu"),
                kl.Dense(num_classes, activation="softmax"),
            ],
            name="CNN_LSTM",
        )
        return model

    # ------------------------------------------------------------------
    # Standard training helper
    # ------------------------------------------------------------------

    def _compile_and_train(
        self,
        model,
        X_tr, y_tr_cat,
        X_val, y_val_cat,
        lr=0.001,
        epochs=50,
        batch_size=32,
        class_weight=None,
    ):
        keras = _require_keras()
        model.compile(
            optimizer=keras.optimizers.Adam(lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=15,
                restore_best_weights=True, mode="min", verbose=0,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5,
                min_lr=1e-6, verbose=0,
            ),
        ]
        history = model.fit(
            X_tr, y_tr_cat,
            validation_data=(X_val, y_val_cat),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1,
        )
        return history

    # ------------------------------------------------------------------
    # Optuna optimisation for TCN
    # ------------------------------------------------------------------

    def optimize_tcn(
        self,
        X_train, y_train_cat,
        X_val, y_val_cat,
        num_classes,
        n_trials=15,
        timeout=300,
    ):
        """
        Run Optuna study to find best TCN hyperparameters.
        Returns (best_params dict, best_val_accuracy).
        """
        keras = _require_keras()
        input_shape = X_train.shape[1:]

        def objective(trial):
            nb_filters = trial.suggest_categorical("nb_filters", [32, 64, 128])
            kernel_size = trial.suggest_categorical("kernel_size", [3, 5])
            nb_stacks = trial.suggest_int("nb_stacks", 1, 2)
            dropout = trial.suggest_float("dropout", 0.1, 0.4)
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_categorical("batch_size", [16, 32])

            model = self.build_tcn(
                input_shape, num_classes,
                nb_filters=nb_filters, kernel_size=kernel_size,
                nb_stacks=nb_stacks, dilations=[1, 2, 4, 8],
                dropout_rate=dropout,
            )
            model.compile(
                optimizer=keras.optimizers.Adam(lr),
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )
            history = model.fit(
                X_train, y_train_cat,
                validation_data=(X_val, y_val_cat),
                epochs=25,
                batch_size=batch_size,
                callbacks=[keras.callbacks.EarlyStopping(
                    patience=5, restore_best_weights=True, verbose=0
                )],
                verbose=0,
            )
            return max(history.history["val_accuracy"])

        optuna = _require_optuna()
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        self.best_params["tcn"] = study.best_params
        return study.best_params, study.best_value

    # ------------------------------------------------------------------
    # LOSO evaluation
    # ------------------------------------------------------------------

    def evaluate_loso(
        self,
        sequences, labels, participants, sessions,
        model_types=("tcn", "lstm", "cnn_lstm"),
        k=5,
        epochs=50,
        batch_size=32,
        optimize=False,
        save_dir="./Models/Field Models",
        plots_dir="./Plots",
    ):
        """
        Mixed session-based k-fold evaluation.
        Sessions are round-robined across folds within each participant,
        so every fold's test set contains data from both participants.
        Returns dict: {fold_idx: {model_type: {accuracy, history, y_true, y_pred}}}
        """
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

        unique_labels = sorted(np.unique(labels))
        num_classes = len(unique_labels)
        unique_participants = sorted(set(participants))
        print(f"\nMixed-LOSO ({k} folds), participants: {unique_participants}")
        print(f"Classes ({num_classes}): {unique_labels}")

        participants_arr = np.array(participants)
        sessions_arr = np.array(sessions)

        # Assign fold IDs by round-robining sessions within each participant
        fold_map = {}
        for p in unique_participants:
            p_sessions = sorted(set(sessions_arr[participants_arr == p]))
            for i, s in enumerate(p_sessions):
                fold_map[(p, s)] = i % k
        window_folds = np.array([fold_map[(p, s)] for p, s in zip(participants_arr, sessions_arr)])

        all_results = {}
        le = LabelEncoder()
        le.fit(unique_labels)

        for fold_idx in range(k):
            test_mask = window_folds == fold_idx
            train_mask = ~test_mask
            if not test_mask.any() or not train_mask.any():
                continue

            test_parts = sorted(set(participants_arr[test_mask]))
            print(f"\n{'='*50}")
            print(f"Mixed-LOSO fold {fold_idx + 1}/{k}: test participants={test_parts}")
            print(f"{'='*50}")

            X_train = sequences[train_mask]
            X_test  = sequences[test_mask]
            y_train = labels[train_mask]
            y_test  = labels[test_mask]
            print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

            n_tr, T, C = X_train.shape
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train.reshape(-1, C)).reshape(n_tr, T, C)
            X_test_s  = scaler.transform(X_test.reshape(-1, C)).reshape(len(X_test), T, C)

            y_tr_enc = le.transform(y_train)
            y_te_enc = le.transform(y_test)

            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train_s, y_tr_enc,
                test_size=0.15, stratify=y_tr_enc, random_state=42,
            )
            keras = _require_keras()
            y_tr_cat  = keras.utils.to_categorical(y_tr,  num_classes)
            y_val_cat = keras.utils.to_categorical(y_val, num_classes)
            y_te_cat  = keras.utils.to_categorical(y_te_enc, num_classes)

            # Class weights to counter majority-class collapse
            cw_values = compute_class_weight("balanced", classes=np.arange(num_classes), y=y_tr)
            class_weight = {i: w for i, w in enumerate(cw_values)}
            print(f"  Class weights: { {le.classes_[i]: f'{w:.2f}' for i, w in class_weight.items()} }")

            input_shape = (T, C)
            fold_results = {}

            for mt in model_types:
                print(f"\nTraining {mt.upper()} (fold {fold_idx + 1})...")

                if mt == "tcn":
                    if optimize:
                        print("  Running Optuna optimisation...")
                        best_p, best_val = self.optimize_tcn(
                            X_tr, y_tr_cat, X_val, y_val_cat, num_classes,
                            n_trials=10, timeout=180,
                        )
                        print(f"  Best params: {best_p}, val_acc={best_val:.4f}")
                        model = self.build_tcn(
                            input_shape, num_classes,
                            nb_filters=best_p.get("nb_filters", 64),
                            kernel_size=best_p.get("kernel_size", 3),
                            nb_stacks=best_p.get("nb_stacks", 1),
                            dropout_rate=best_p.get("dropout", 0.2),
                        )
                        lr = best_p.get("lr", 0.001)
                        bs = best_p.get("batch_size", 32)
                    else:
                        model = self.build_tcn(
                            input_shape, num_classes,
                            nb_filters=64, kernel_size=3,
                            nb_stacks=1, dilations=[1, 2, 4, 8],
                            dropout_rate=0.2,
                        )
                        lr, bs = 0.001, 32
                elif mt == "lstm":
                    model = self.build_lstm(
                        input_shape, num_classes, units=128, dropout_rate=0.3
                    )
                    lr, bs = 0.001, 32
                elif mt == "cnn_lstm":
                    model = self.build_cnn_lstm(
                        input_shape, num_classes,
                        conv_filters=64, kernel_size=3,
                        lstm_units=64, dropout_rate=0.3,
                    )
                    lr, bs = 0.001, 32
                else:
                    raise ValueError(f"Unknown model type: {mt}")

                history = self._compile_and_train(
                    model, X_tr, y_tr_cat, X_val, y_val_cat,
                    lr=lr, epochs=epochs, batch_size=bs,
                    class_weight=class_weight,
                )

                y_pred_proba = model.predict(X_test_s, verbose=0)
                y_pred_enc   = np.argmax(y_pred_proba, axis=1)
                y_pred = le.inverse_transform(y_pred_enc)
                y_true = le.inverse_transform(y_te_enc)

                acc = accuracy_score(y_true, y_pred)
                print(f"  {mt.upper()} Test Accuracy (fold {fold_idx + 1}): {acc:.4f}")

                fold_results[mt] = {
                    "accuracy": float(acc),
                    "y_true": y_true.tolist(),
                    "y_pred": y_pred.tolist(),
                    "history": {
                        "accuracy": history.history["accuracy"],
                        "val_accuracy": history.history["val_accuracy"],
                        "loss": history.history["loss"],
                        "val_loss": history.history["val_loss"],
                    },
                }

                self._plot_training_history(
                    history, model_name=f"field_{mt}_fold{fold_idx + 1}",
                    save_path=os.path.join(plots_dir, f"field_{mt}_training_history_fold{fold_idx + 1}.png"),
                )

            all_results[f"fold_{fold_idx + 1}"] = fold_results

        # Aggregate summary
        print("\n" + "=" * 60)
        print("Mixed-LOSO Summary")
        print("=" * 60)
        summary = {}
        for mt in model_types:
            accs = [all_results[fk][mt]["accuracy"] for fk in all_results]
            mean_acc = float(np.mean(accs))
            summary[mt] = {"per_fold": accs, "mean": mean_acc}
            print(f"  {mt.upper():10s}: per_fold={[f'{a:.4f}' for a in accs]}  ->  Mean = {mean_acc:.4f}")

        # Combined confusion matrices
        for mt in model_types:
            y_true_all, y_pred_all = [], []
            for fk in all_results:
                y_true_all.extend(all_results[fk][mt]["y_true"])
                y_pred_all.extend(all_results[fk][mt]["y_pred"])
            self._plot_confusion_matrix(
                y_true_all, y_pred_all,
                title=f"Gym Data - {mt.upper()} (Mixed-LOSO)",
                save_path=os.path.join(plots_dir, f"gym_confusion_matrix_{mt}.png"),
            )

        return all_results, summary

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _plot_training_history(history, model_name="model", save_path=None):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(history.history["accuracy"], label="train")
        axes[0].plot(history.history["val_accuracy"], label="val")
        axes[0].set_title(f"{model_name} - Accuracy")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy")
        axes[0].legend()
        axes[1].plot(history.history["loss"], label="train")
        axes[1].plot(history.history["val_loss"], label="val")
        axes[1].set_title(f"{model_name} - Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close()

    @staticmethod
    def _plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", save_path=None):
        labels = sorted(set(y_true) | set(y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels,
        )
        plt.title(title)
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close()


# ---------------------------------------------------------------------------
# get_training_summary  (entry point called from Assignment.ipynb cell 2)
# ---------------------------------------------------------------------------

def get_training_summary(rf_model=None, rf_scaler=None):
    """
    Run the complete deep learning pipeline on the Gym Movements Dataset.

    1. Loads raw data and checks for missing values.
    2. Creates adaptive windows.
    3. Trains TCN, LSTM, and CNN-LSTM with LOSO (Leave-One-Subject-Out) split.
    4. Saves results and generates plots.
    5. Compares DL models; RF comparison if rf_model is provided.

    Args:
        rf_model:  Optional RF model from a previous training cell.
        rf_scaler: Optional StandardScaler used with rf_model.
    """
    print("\n" + "=" * 60)
    print("Gym Movements Deep Learning Pipeline (LOSO Evaluation)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\nLoading Gym Movements Dataset...")
    raw_data = load_gym_dataset(Path("./Datasets/cropped"))
    print(f"Total samples: {len(raw_data)}")
    print(f"Participants: {sorted(raw_data['participant'].unique())}")
    print(f"Activities: {sorted(raw_data['activity_label'].unique())}")

    # ------------------------------------------------------------------
    # 2. Missing value handling
    # ------------------------------------------------------------------
    print("\n=== Missing Value Analysis ===")
    sensor_cols = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    missing = raw_data[sensor_cols].isnull().sum()
    print(f"Missing values per channel:\n{missing.to_string()}")
    total_missing = int(missing.sum())
    if total_missing > 0:
        print(f"\nTotal missing: {total_missing} — applying linear interpolation per session")
        raw_data[sensor_cols] = (
            raw_data.groupby(["participant", "session"])[sensor_cols]
            .transform(lambda g: g.interpolate(method="linear").ffill().bfill())
        )
    else:
        print("No missing values found.")

    # ------------------------------------------------------------------
    # 3. Windowing
    # ------------------------------------------------------------------
    print("\nCreating adaptive windows...")
    windowed_data = create_adaptive_windows(
        raw_data,
        min_window_size=64,
        preferred_window_size=128,
        overlap_ratio=0.5,
    )

    # ------------------------------------------------------------------
    # 4. Prepare sequences for DL
    # ------------------------------------------------------------------
    pipeline = ML4QSFlexiblePipeline(sampling_rate=50, window_size=128)
    print("\nPreparing raw sequences for DL models...")
    sequences, labels, metadata = pipeline.prepare_sequences(windowed_data)
    participants = [m["participant"] for m in metadata]
    sessions     = [m["session"]     for m in metadata]
    print(f"Sequences shape: {sequences.shape}  (channels include gravity-aligned: {sequences.shape[2] > 6})")

    # ------------------------------------------------------------------
    # 5. LOSO DL training
    # ------------------------------------------------------------------
    print("\nStarting LOSO evaluation (TCN, LSTM, CNN-LSTM)...")
    os.makedirs("./Models/Gym Models", exist_ok=True)
    os.makedirs("./Plots", exist_ok=True)

    all_results, summary = pipeline.evaluate_loso(
        sequences, labels, participants, sessions,
        model_types=("tcn", "lstm", "cnn_lstm"),
        k=5,
        epochs=50,
        batch_size=32,
        optimize=False,   # set True to run Optuna (slow)
        save_dir="./Models/Gym Models",
        plots_dir="./Plots",
    )

    # ------------------------------------------------------------------
    # 6. Save results (guard against None / missing fields)
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"./Models/Gym Models/dl_loso_results_{timestamp}.json"

    serializable = {}
    for fold_key, fold in all_results.items():
        serializable[fold_key] = {}
        for mt, res in fold.items():
            serializable[fold_key][mt] = {
                "accuracy": res["accuracy"],
                "epochs_trained": len(res["history"]["accuracy"]),
                "final_val_accuracy": res["history"]["val_accuracy"][-1]
                    if res["history"]["val_accuracy"] else None,
            }

    with open(results_path, "w") as f:
        json.dump({"summary": summary, "per_fold": serializable}, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # ------------------------------------------------------------------
    # 7. Print final comparison table
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Model Comparison (mean LOSO accuracy)")
    print("=" * 60)
    for mt, s in summary.items():
        print(f"  {mt.upper():12s}: {s['mean']:.4f}")

    if rf_model is not None:
        print("\n  (RF model from cell 1 is available — re-run training.py with LOSO for RF accuracy)")

    print("\n" + "=" * 60)
    print("Deep Learning Pipeline Complete")
    print("=" * 60)

    return all_results, summary
