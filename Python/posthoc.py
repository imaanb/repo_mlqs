"""
posthoc.py - ML4QS Deep Learning Pipeline
ML4QSFlexiblePipeline: TCN, LSTM, CNN-LSTM with Optuna tuning and LOSO evaluation.
Uses PyTorch for GPU support on native Windows (CUDA-capable).
"""

import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import matplotlib.pyplot as plt
import seaborn as sns

from Python.integrations import load_gym_dataset
from Python.windows import create_adaptive_windows
from Python.feature_engineering import engineer_robust_features as _engineer_robust_features

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


# ===========================================================================
# PyTorch model classes
# ===========================================================================

class _CausalConv1d(nn.Module):
    """Left-only padded Conv1d — output has the same length as the input (causal)."""
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        self._pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self._pad, 0)))


class _TCNResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        self.conv1 = _CausalConv1d(in_ch,   out_ch, kernel_size, dilation)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = _CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.drop2 = nn.Dropout(dropout)
        self.res   = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        res = self.res(x)
        out = self.drop1(F.relu(self.bn1(self.conv1(x))))
        out = F.relu(self.bn2(self.conv2(out)))
        return self.drop2(out + res)


class TCNModel(nn.Module):
    """Temporal Convolutional Network (causal dilated convolutions + residuals)."""
    def __init__(self, in_channels, num_classes, nb_filters=64, kernel_size=3,
                 nb_stacks=1, dilations=None, dropout=0.2):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8]
        blocks, ch = [], in_channels
        for _ in range(nb_stacks):
            for d in dilations:
                blocks.append(_TCNResBlock(ch, nb_filters, kernel_size, d, dropout))
                ch = nb_filters
        self.blocks = nn.Sequential(*blocks)
        self.drop   = nn.Dropout(dropout)
        self.fc1    = nn.Linear(nb_filters, 64)
        self.fc2    = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (B, T, C) → (B, C, T) for Conv1d
        x = self.blocks(x.permute(0, 2, 1))
        x = x.mean(dim=2)                       # GlobalAveragePooling1D
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)                       # raw logits


class LSTMModel(nn.Module):
    """Stacked LSTM classifier."""
    def __init__(self, in_channels, num_classes, units=128, dropout=0.3):
        super().__init__()
        self.lstm1 = nn.LSTM(in_channels, units, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(units, units // 2, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.fc1   = nn.Linear(units // 2, 64)
        self.drop3 = nn.Dropout(dropout)
        self.fc2   = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (B, T, C)
        out, _ = self.lstm1(x)
        out = self.drop1(out)
        out, _ = self.lstm2(out)
        out = self.drop2(out[:, -1, :])          # last timestep
        out = self.drop3(F.relu(self.fc1(out)))
        return self.fc2(out)


class CNNLSTMModel(nn.Module):
    """CNN feature extractor followed by LSTM temporal model."""
    def __init__(self, in_channels, num_classes, conv_filters=64, kernel_size=3,
                 lstm_units=64, dropout=0.3):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels,      conv_filters,      kernel_size, padding=pad)
        self.bn1   = nn.BatchNorm1d(conv_filters)
        self.conv2 = nn.Conv1d(conv_filters, conv_filters // 2, kernel_size, padding=pad)
        self.bn2   = nn.BatchNorm1d(conv_filters // 2)
        self.pool  = nn.MaxPool1d(2)
        self.drop1 = nn.Dropout(dropout)
        self.lstm  = nn.LSTM(conv_filters // 2, lstm_units, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.fc1   = nn.Linear(lstm_units, 64)
        self.fc2   = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)                  # (B, T, C) → (B, C, T)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.drop1(self.pool(x))
        x = x.permute(0, 2, 1)                  # (B, C, T) → (B, T, C) for LSTM
        out, _ = self.lstm(x)
        out = self.drop2(out[:, -1, :])
        out = F.relu(self.fc1(out))
        return self.fc2(out)


class CNNModel(nn.Module):
    """Pure 1-D CNN (used in evaluation.py run_dl)."""
    def __init__(self, in_channels, num_classes, filters=64, kernel_size=3, dropout=0.3):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, filters,      kernel_size, padding=pad)
        self.bn1   = nn.BatchNorm1d(filters)
        self.conv2 = nn.Conv1d(filters,     filters,      kernel_size, padding=pad)
        self.bn2   = nn.BatchNorm1d(filters)
        self.conv3 = nn.Conv1d(filters,     filters // 2, kernel_size, padding=pad)
        self.bn3   = nn.BatchNorm1d(filters // 2)
        self.drop  = nn.Dropout(dropout)
        self.fc1   = nn.Linear(filters // 2, 64)
        self.fc2   = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.drop(x.mean(dim=2))            # GlobalAveragePooling1D
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)


# ===========================================================================
# Training / inference helpers
# ===========================================================================

def _train_pytorch(
    model, X_tr, y_tr, X_val, y_val,
    lr=0.001, epochs=50, batch_size=32,
    class_weight=None, device=None, verbose=False,
):
    """
    Train a PyTorch model with early stopping (patience=15) and
    ReduceLROnPlateau scheduling.

    Parameters
    ----------
    model        : nn.Module
    X_tr, X_val  : np.ndarray (N, T, C)
    y_tr, y_val  : np.ndarray (N,) integer class indices
    class_weight : dict {class_idx: float} or None
    device       : torch.device (defaults to DEVICE)

    Returns
    -------
    (model, history)
        model   : trained nn.Module with best weights restored
        history : dict with keys 'loss', 'val_loss', 'accuracy', 'val_accuracy'
    """
    if device is None:
        device = DEVICE

    model = model.to(device)

    if class_weight is not None:
        n = max(class_weight) + 1
        w = torch.tensor([class_weight[i] for i in range(n)],
                         dtype=torch.float32).to(device)
    else:
        w = None

    criterion = nn.CrossEntropyLoss(weight=w)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    pin = device.type == "cuda"
    train_dl = DataLoader(
        TensorDataset(
            torch.tensor(X_tr, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.long),
        ),
        batch_size=batch_size, shuffle=True, pin_memory=pin,
    )
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)

    history      = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
    best_loss    = float("inf")
    best_state   = None
    patience_ctr = 0

    for epoch in range(epochs):
        model.train()
        t_loss = t_correct = t_total = 0
        for Xb, yb in train_dl:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            t_loss    += loss.item() * len(yb)
            t_correct += (logits.argmax(1) == yb).sum().item()
            t_total   += len(yb)

        model.eval()
        with torch.no_grad():
            v_logits  = model(X_val_t)
            v_loss    = criterion(v_logits, y_val_t).item()
            v_correct = (v_logits.argmax(1) == y_val_t).sum().item()

        t_loss /= t_total
        history["loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["accuracy"].append(t_correct / t_total)
        history["val_accuracy"].append(v_correct / len(y_val))

        if verbose:
            print(f"  epoch {epoch + 1:3d}: loss={t_loss:.4f} "
                  f"val_loss={v_loss:.4f} val_acc={history['val_accuracy'][-1]:.4f}")

        scheduler.step(v_loss)

        if v_loss < best_loss:
            best_loss    = v_loss
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= 15:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def _predict_pytorch(model, X, batch_size=512, device=None):
    """Return (N,) integer class predictions from a PyTorch model."""
    if device is None:
        device = DEVICE
    model.eval().to(device)
    X_t = torch.tensor(X, dtype=torch.float32)
    preds = []
    with torch.no_grad():
        for (Xb,) in DataLoader(TensorDataset(X_t), batch_size=batch_size):
            preds.append(model(Xb.to(device)).argmax(1).cpu())
    return torch.cat(preds).numpy()


# ---------------------------------------------------------------------------
# Activity label map (field dataset labels → human-readable names)
# ---------------------------------------------------------------------------
ACTIVITY_NAMES = {
    "elliptical": "Elliptical",
    "lat_pull":   "Lat Pull-Down",
    "row":        "Rowing Machine",
}


# ===========================================================================
# ML4QSFlexiblePipeline
# ===========================================================================

class ML4QSFlexiblePipeline:
    """
    Flexible pipeline supporting TCN, LSTM, and CNN-LSTM model architectures
    with mixed session-based k-fold evaluation and Optuna hyperparameter tuning.
    Backed by PyTorch — GPU-accelerated on CUDA-capable hardware.
    """

    def __init__(self, sampling_rate=50, window_size=128):
        self.sampling_rate = sampling_rate
        self.window_size   = window_size
        self.label_encoder = LabelEncoder()
        self.best_params   = {}

    # ------------------------------------------------------------------
    # Feature engineering helpers
    # ------------------------------------------------------------------

    def engineer_robust_features(self, windowed_data):
        return _engineer_robust_features(windowed_data, self.sampling_rate)

    def engineer_features(self, windowed_data):
        return self.engineer_robust_features(windowed_data)

    # ------------------------------------------------------------------
    # Sequence preparation for DL models
    # ------------------------------------------------------------------

    def prepare_sequences(self, windowed_data):
        """
        Convert windowed_data list of dicts to (N, T, C) numpy array.
        Channels: acc_x/y/z, gyro_x/y/z + gravity-aligned channels when present.
        """
        base_channels    = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
        aligned_channels = ["acc_vert", "acc_horiz", "gyro_vert", "gyro_horiz"]
        has_aligned = (
            windowed_data and all(ch in windowed_data[0] for ch in aligned_channels)
        )
        channels = base_channels + aligned_channels if has_aligned else base_channels
        sequences, labels, metadata = [], [], []

        for w in windowed_data:
            seq = np.stack([w[ch] for ch in channels], axis=-1)
            n = len(seq)
            if n < self.window_size:
                seq = np.pad(seq, ((0, self.window_size - n), (0, 0)), mode="reflect")
            else:
                seq = seq[: self.window_size]
            sequences.append(seq)
            labels.append(w["activity"])
            metadata.append({
                "participant": w["participant"],
                "session":     w["session"],
                "data_source": w["data_source"],
            })

        return np.array(sequences, dtype=np.float32), np.array(labels), metadata

    # ------------------------------------------------------------------
    # Model builders  (input_shape = (T, C))
    # ------------------------------------------------------------------

    def build_tcn(self, input_shape, num_classes, nb_filters=64, kernel_size=3,
                  nb_stacks=1, dilations=None, dropout_rate=0.2):
        _, C = input_shape
        return TCNModel(C, num_classes, nb_filters=nb_filters, kernel_size=kernel_size,
                        nb_stacks=nb_stacks, dilations=dilations, dropout=dropout_rate)

    def build_lstm(self, input_shape, num_classes, units=128, dropout_rate=0.3):
        _, C = input_shape
        return LSTMModel(C, num_classes, units=units, dropout=dropout_rate)

    def build_cnn_lstm(self, input_shape, num_classes, conv_filters=64, kernel_size=3,
                       lstm_units=64, dropout_rate=0.3):
        _, C = input_shape
        return CNNLSTMModel(C, num_classes, conv_filters=conv_filters,
                            kernel_size=kernel_size, lstm_units=lstm_units,
                            dropout=dropout_rate)

    # ------------------------------------------------------------------
    # Training helper (wraps _train_pytorch)
    # ------------------------------------------------------------------

    def _compile_and_train(self, model, X_tr, y_tr, X_val, y_val,
                           lr=0.001, epochs=50, batch_size=32, class_weight=None):
        return _train_pytorch(model, X_tr, y_tr, X_val, y_val,
                              lr=lr, epochs=epochs, batch_size=batch_size,
                              class_weight=class_weight)

    # ------------------------------------------------------------------
    # Optuna optimisation for TCN
    # ------------------------------------------------------------------

    def optimize_tcn(self, X_train, y_train, X_val, y_val, num_classes,
                     n_trials=15, timeout=300):
        """
        Run an Optuna study to find best TCN hyperparameters.
        Returns (best_params dict, best_val_accuracy).
        """
        input_shape = X_train.shape[1:]

        def objective(trial):
            nb_filters  = trial.suggest_categorical("nb_filters",  [32, 64, 128])
            kernel_size = trial.suggest_categorical("kernel_size",  [3, 5])
            nb_stacks   = trial.suggest_int("nb_stacks", 1, 2)
            dropout     = trial.suggest_float("dropout",  0.1, 0.4)
            lr          = trial.suggest_float("lr",       1e-4, 1e-2, log=True)
            batch_size  = trial.suggest_categorical("batch_size", [16, 32])

            model = self.build_tcn(input_shape, num_classes,
                                   nb_filters=nb_filters, kernel_size=kernel_size,
                                   nb_stacks=nb_stacks, dilations=[1, 2, 4, 8],
                                   dropout_rate=dropout)
            _, history = _train_pytorch(model, X_train, y_train, X_val, y_val,
                                        lr=lr, epochs=25, batch_size=batch_size)
            return max(history["val_accuracy"])

        optuna = _require_optuna()
        study  = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        self.best_params["tcn"] = study.best_params
        return study.best_params, study.best_value

    # ------------------------------------------------------------------
    # LOSO evaluation
    # ------------------------------------------------------------------

    def evaluate_loso(self, sequences, labels, participants, sessions,
                      model_types=("tcn", "lstm", "cnn_lstm"), k=5,
                      epochs=50, batch_size=32, optimize=False,
                      save_dir="./Models/Field Models", plots_dir="./Plots"):
        """
        Mixed session-based k-fold evaluation.
        Sessions are round-robined across folds within each participant,
        so every fold's test set contains data from both participants.

        Returns
        -------
        all_results : dict  {fold_key: {model_type: {accuracy, history, y_true, y_pred}}}
        summary     : dict  {model_type: {per_fold: [...], mean: float}}
        """
        os.makedirs(save_dir,  exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

        unique_labels       = sorted(np.unique(labels))
        num_classes         = len(unique_labels)
        unique_participants = sorted(set(participants))
        print(f"\nMixed-LOSO ({k} folds), participants: {unique_participants}")
        print(f"Classes ({num_classes}): {unique_labels}")
        print(f"Device: {DEVICE}")

        participants_arr = np.array(participants)
        sessions_arr     = np.array(sessions)

        # Assign fold IDs by round-robining sessions within each participant
        fold_map = {}
        for p in unique_participants:
            for i, s in enumerate(sorted(set(sessions_arr[participants_arr == p]))):
                fold_map[(p, s)] = i % k
        window_folds = np.array([fold_map[(p, s)]
                                 for p, s in zip(participants_arr, sessions_arr)])

        le          = LabelEncoder()
        le.fit(unique_labels)
        all_results = {}

        for fold_idx in range(k):
            test_mask  = window_folds == fold_idx
            train_mask = ~test_mask
            if not test_mask.any() or not train_mask.any():
                continue

            test_parts = sorted(set(participants_arr[test_mask]))
            print(f"\n{'=' * 50}")
            print(f"Mixed-LOSO fold {fold_idx + 1}/{k}: test participants={test_parts}")
            print(f"{'=' * 50}")

            X_train = sequences[train_mask]
            X_test  = sequences[test_mask]
            y_train = labels[train_mask]
            y_test  = labels[test_mask]
            print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

            n_tr, T, C = X_train.shape
            scaler    = StandardScaler()
            X_train_s = scaler.fit_transform(X_train.reshape(-1, C)).reshape(n_tr, T, C)
            X_test_s  = scaler.transform(X_test.reshape(-1, C)).reshape(len(X_test), T, C)

            y_tr_enc = le.transform(y_train)
            y_te_enc = le.transform(y_test)

            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train_s, y_tr_enc,
                test_size=0.15, stratify=y_tr_enc, random_state=42,
            )

            present      = np.unique(y_tr)
            cw_values    = compute_class_weight("balanced", classes=present, y=y_tr)
            class_weight = {int(c): float(w) for c, w in zip(present, cw_values)}
            for i in range(num_classes):        # fill any class absent from training fold
                class_weight.setdefault(i, 1.0)
            print(f"  Class weights: "
                  f"{ {le.classes_[i]: f'{w:.2f}' for i, w in class_weight.items()} }")

            input_shape  = (T, C)
            fold_results = {}

            for mt in model_types:
                print(f"\nTraining {mt.upper()} (fold {fold_idx + 1})...")

                if mt == "tcn":
                    if optimize:
                        print("  Running Optuna optimisation...")
                        best_p, best_val = self.optimize_tcn(
                            X_tr, y_tr, X_val, y_val, num_classes,
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
                        lr, bs = best_p.get("lr", 0.001), best_p.get("batch_size", 32)
                    else:
                        model = self.build_tcn(input_shape, num_classes,
                                               nb_filters=64, kernel_size=3,
                                               nb_stacks=1, dilations=[1, 2, 4, 8],
                                               dropout_rate=0.2)
                        lr, bs = 0.001, 32
                elif mt == "lstm":
                    model = self.build_lstm(input_shape, num_classes,
                                            units=128, dropout_rate=0.3)
                    lr, bs = 0.001, 32
                elif mt == "cnn_lstm":
                    model = self.build_cnn_lstm(input_shape, num_classes,
                                                conv_filters=64, kernel_size=3,
                                                lstm_units=64, dropout_rate=0.3)
                    lr, bs = 0.001, 32
                else:
                    raise ValueError(f"Unknown model type: {mt}")

                model, history = _train_pytorch(
                    model, X_tr, y_tr, X_val, y_val,
                    lr=lr, epochs=epochs, batch_size=bs,
                    class_weight=class_weight,
                )

                y_pred_enc = _predict_pytorch(model, X_test_s)
                y_pred     = le.inverse_transform(y_pred_enc)
                y_true     = le.inverse_transform(y_te_enc)

                acc = accuracy_score(y_true, y_pred)
                print(f"  {mt.upper()} Test Accuracy (fold {fold_idx + 1}): {acc:.4f}")

                fold_results[mt] = {
                    "accuracy": float(acc),
                    "y_true":   y_true.tolist(),
                    "y_pred":   y_pred.tolist(),
                    "history":  history,
                }

                self._plot_training_history(
                    history,
                    model_name=f"field_{mt}_fold{fold_idx + 1}",
                    save_path=os.path.join(
                        plots_dir,
                        f"field_{mt}_training_history_fold{fold_idx + 1}.png",
                    ),
                )

                # Free GPU memory
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            all_results[f"fold_{fold_idx + 1}"] = fold_results

        # Aggregate summary
        print("\n" + "=" * 60)
        print("Mixed-LOSO Summary")
        print("=" * 60)
        summary = {}
        for mt in model_types:
            accs     = [all_results[fk][mt]["accuracy"] for fk in all_results]
            mean_acc = float(np.mean(accs))
            summary[mt] = {"per_fold": accs, "mean": mean_acc}
            print(f"  {mt.upper():10s}: per_fold={[f'{a:.4f}' for a in accs]}"
                  f"  ->  Mean = {mean_acc:.4f}")

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
        """history is a plain dict with keys accuracy, val_accuracy, loss, val_loss."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(history["accuracy"],     label="train")
        axes[0].plot(history["val_accuracy"], label="val")
        axes[0].set_title(f"{model_name} - Accuracy")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy")
        axes[0].legend()
        axes[1].plot(history["loss"],     label="train")
        axes[1].plot(history["val_loss"], label="val")
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
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close()


# ---------------------------------------------------------------------------
# get_training_summary  (entry point called from Assignment.ipynb)
# ---------------------------------------------------------------------------

def get_training_summary(rf_model=None, rf_scaler=None):
    """
    Run the complete deep learning pipeline on the Gym Movements Dataset.

    1. Loads raw data and checks for missing values.
    2. Creates adaptive windows.
    3. Trains TCN, LSTM, and CNN-LSTM with mixed LOSO split.
    4. Saves results and generates plots.

    Args:
        rf_model:  Optional RF model from a previous training cell.
        rf_scaler: Optional StandardScaler used with rf_model.
    """
    print("\n" + "=" * 60)
    print("Gym Movements Deep Learning Pipeline (LOSO Evaluation)")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # 1. Load data
    print("\nLoading Gym Movements Dataset...")
    raw_data = load_gym_dataset(Path("./Datasets/cropped"))
    print(f"Total samples: {len(raw_data)}")
    print(f"Participants: {sorted(raw_data['participant'].unique())}")
    print(f"Activities:   {sorted(raw_data['activity_label'].unique())}")

    # 2. Missing value handling
    print("\n=== Missing Value Analysis ===")
    sensor_cols  = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    total_missing = int(raw_data[sensor_cols].isnull().sum().sum())
    print(f"Missing values per channel:\n{raw_data[sensor_cols].isnull().sum().to_string()}")
    if total_missing > 0:
        print(f"\nTotal missing: {total_missing} — applying linear interpolation per session")
        raw_data[sensor_cols] = (
            raw_data.groupby(["participant", "session"])[sensor_cols]
            .transform(lambda g: g.interpolate(method="linear").ffill().bfill())
        )
    else:
        print("No missing values found.")

    # 3. Windowing
    print("\nCreating adaptive windows...")
    windowed_data = create_adaptive_windows(
        raw_data, min_window_size=64, preferred_window_size=128, overlap_ratio=0.5,
    )

    # 4. Prepare sequences for DL
    pipeline = ML4QSFlexiblePipeline(sampling_rate=50, window_size=128)
    print("\nPreparing raw sequences for DL models...")
    sequences, labels, metadata = pipeline.prepare_sequences(windowed_data)
    participants = [m["participant"] for m in metadata]
    sessions     = [m["session"]     for m in metadata]
    print(f"Sequences shape: {sequences.shape}  "
          f"(gravity-aligned channels: {sequences.shape[2] > 6})")

    # 5. LOSO DL training
    print("\nStarting LOSO evaluation (TCN, LSTM, CNN-LSTM)...")
    os.makedirs("./Models/Gym Models", exist_ok=True)
    os.makedirs("./Plots", exist_ok=True)

    all_results, summary = pipeline.evaluate_loso(
        sequences, labels, participants, sessions,
        model_types=("tcn", "lstm", "cnn_lstm"),
        k=5, epochs=50, batch_size=32, optimize=False,
        save_dir="./Models/Gym Models", plots_dir="./Plots",
    )

    # 6. Save results
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"./Models/Gym Models/dl_loso_results_{timestamp}.json"
    serializable = {}
    for fold_key, fold in all_results.items():
        serializable[fold_key] = {}
        for mt, res in fold.items():
            serializable[fold_key][mt] = {
                "accuracy":          res["accuracy"],
                "epochs_trained":    len(res["history"]["accuracy"]),
                "final_val_accuracy": res["history"]["val_accuracy"][-1]
                                      if res["history"]["val_accuracy"] else None,
            }

    with open(results_path, "w") as f:
        json.dump({"summary": summary, "per_fold": serializable}, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # 7. Print final comparison table
    print("\n" + "=" * 60)
    print("Model Comparison (mean LOSO accuracy)")
    print("=" * 60)
    for mt, s in summary.items():
        print(f"  {mt.upper():12s}: {s['mean']:.4f}")
    if rf_model is not None:
        print("\n  (RF model from cell 1 available — re-run training.py with LOSO for RF accuracy)")
    print("\n" + "=" * 60)
    print("Deep Learning Pipeline Complete")
    print("=" * 60)

    return all_results, summary
