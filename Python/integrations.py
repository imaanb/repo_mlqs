"""Data loading and integration-related functions"""

import os
import numpy as np
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Gym Movements Dataset
# ---------------------------------------------------------------------------

GYM_ACTIVITY_MAP = {
    "elliptical": 1,
    "lat_pull": 2,
    "row": 3,
}


def load_gym_dataset(path: Path):
    """
    Load the Gym Movements Dataset.

    Directory structure:
        <path>/<activity>/<participant>/<session>/Accelerometer.csv
        <path>/<activity>/<participant>/<session>/Gyroscope.csv
        <path>/<activity>/<participant>/<session>/labels.csv  (optional)

    Accelerometer / Gyroscope CSV columns: time, seconds_elapsed, z, y, x
      (timestamps in nanoseconds, x/y/z in m/s² or rad/s)

    labels.csv columns: label_start, label_end, label
      (when present, data is clipped to the labelled exercise time range)

    Returns a DataFrame with columns:
        time, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z,
        activity_label, participant, session, data_source
    """
    path = Path(path)
    all_data = []
    session_id = 0

    activities = sorted(d.name for d in path.iterdir() if d.is_dir())

    for activity in activities:
        activity_path = path / activity
        participants = sorted(d.name for d in activity_path.iterdir() if d.is_dir())

        for participant in participants:
            participant_path = activity_path / participant
            sessions = sorted(
                (d.name for d in participant_path.iterdir()
                 if d.is_dir() and d.name.isdigit()),
                key=int,
            )

            for session_num in sessions:
                session_path = participant_path / session_num
                acc_file = session_path / "Accelerometer.csv"
                gyro_file = session_path / "Gyroscope.csv"
                labels_file = session_path / "labels.csv"

                if not acc_file.exists() or not gyro_file.exists():
                    continue

                acc = pd.read_csv(acc_file)[["time", "x", "y", "z"]].rename(
                    columns={"x": "acc_x", "y": "acc_y", "z": "acc_z"}
                )
                gyro = pd.read_csv(gyro_file)[["time", "x", "y", "z"]].rename(
                    columns={"x": "gyro_x", "y": "gyro_y", "z": "gyro_z"}
                )

                acc = acc.sort_values("time").reset_index(drop=True)
                gyro = gyro.sort_values("time").reset_index(drop=True)

                # Merge accelerometer + gyroscope on nearest timestamp (50 ms tolerance)
                merged = pd.merge_asof(
                    acc, gyro, on="time",
                    direction="nearest", tolerance=50_000_000,
                )
                merged = merged.dropna(subset=["gyro_x", "gyro_y", "gyro_z"])

                # Clip to labelled exercise range when labels.csv is present
                if labels_file.exists():
                    ldf = pd.read_csv(labels_file)
                    if len(ldf) > 0:
                        t_start = ldf["label_start"].min()
                        t_end = ldf["label_end"].max()
                        merged = merged[
                            (merged["time"] >= t_start) & (merged["time"] <= t_end)
                        ]

                if len(merged) == 0:
                    continue

                # Convert nanosecond timestamp to relative seconds
                merged["time"] = (merged["time"] - merged["time"].iloc[0]) / 1e9

                merged["activity_label"] = activity
                merged["participant"] = participant
                merged["session"] = session_id
                merged["data_source"] = "original"

                all_data.append(merged[[
                    "time", "acc_x", "acc_y", "acc_z",
                    "gyro_x", "gyro_y", "gyro_z",
                    "activity_label", "participant", "session", "data_source",
                ]])
                session_id += 1

    if not all_data:
        raise RuntimeError(f"No gym dataset sessions found in: {path}")

    raw_data = pd.concat(all_data, ignore_index=True)

    print("Gym Movements Dataset loaded:")
    print(f"  Total samples : {len(raw_data):,}")
    print(f"  Participants  : {sorted(raw_data['participant'].unique())}")
    print(f"  Activities    : {sorted(raw_data['activity_label'].unique())}")
    print(f"  Sessions      : {raw_data['session'].nunique()}")
    print(f"  Columns       : {list(raw_data.columns)}")
    return raw_data


def _get_protocol(session_number):
    """
    Session folder structure:
        01-10 = protocol 2
        11-20 = protocol 1
    """
    return 2 if 1 <= session_number <= 10 else 1


def _standardize_columns(data):
    column_mappings = {
        "Time (s)": "time",
        "Acc_X": "acc_x",
        "Acc_Y": "acc_y",
        "Acc_Z": "acc_z",
        "Gyro_X": "gyro_x",
        "Gyro_Y": "gyro_y",
        "Gyro_Z": "gyro_z",
        "x": "acc_x",
        "y": "acc_y",
        "z": "acc_z",
    }
    return data.rename(columns=column_mappings)


def load_field_dataset(path: Path):
    """
    Load the field dataset collected by two participants.

    Directory structure:
        <path>/<participant>/<session>/processed_sensor_data.csv

    Returns a DataFrame with columns:
        time, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z,
        activity_label, participant, session, protocol, data_source
    """
    all_data = []

    participants = sorted(
        d for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d))
    )

    for participant in participants:
        participant_path = os.path.join(path, participant)
        sessions = sorted(
            (
                d for d in os.listdir(participant_path)
                if os.path.isdir(os.path.join(participant_path, d)) and d.isdigit()
            ),
            key=int,
        )

        for session in sessions:
            session_path = os.path.join(participant_path, session)
            processed_path = os.path.join(session_path, "processed_sensor_data.csv")

            if os.path.exists(processed_path):
                sensor_data = pd.read_csv(processed_path)
                sensor_data = _standardize_columns(sensor_data)
                sensor_data["participant"] = participant
                session_num = int(session)
                sensor_data["session"] = session_num
                sensor_data["protocol"] = _get_protocol(session_num)
                sensor_data["data_source"] = "original"
                all_data.append(sensor_data)

    raw_data = pd.concat(all_data, ignore_index=True)
    return raw_data


# ---------------------------------------------------------------------------
# UCI HAR Dataset
# ---------------------------------------------------------------------------

UCI_ACTIVITY_MAP = {
    1: "Walking",
    2: "Walking Upstairs",
    3: "Walking Downstairs",
    4: "Sitting",
    5: "Standing",
    6: "Laying",
}


def load_uci_dataset(path: Path):
    """
    Load the UCI Human Activity Recognition Dataset (Anguita et al., 2013).

    Directory structure expected:
        <path>/train/X_train.txt  (7352 x 561 pre-extracted features)
        <path>/train/y_train.txt  (activity labels, 1-6)
        <path>/train/subject_train.txt (subject IDs)
        <path>/test/X_test.txt    (2947 x 561)
        <path>/test/y_test.txt
        <path>/test/subject_test.txt
        <path>/features.txt       (561 feature names)

    Returns:
        X_train, y_train, X_test, y_test, feature_names,
        subjects_train, subjects_test
    """
    path = Path(path)

    features_file = path / "features.txt"
    if features_file.exists():
        feature_names = pd.read_csv(
            features_file, header=None, sep=r"\s+", names=["idx", "name"]
        )["name"].tolist()
    else:
        feature_names = [f"feature_{i}" for i in range(561)]

    def _load_split(split):
        X = pd.read_csv(
            path / split / f"X_{split}.txt",
            header=None, sep=r"\s+", dtype=np.float32,
        ).values
        y = pd.read_csv(
            path / split / f"y_{split}.txt",
            header=None, sep=r"\s+",
        ).values.ravel()
        subjects = pd.read_csv(
            path / split / f"subject_{split}.txt",
            header=None, sep=r"\s+",
        ).values.ravel()
        return X, y, subjects

    X_train, y_train, subjects_train = _load_split("train")
    X_test, y_test, subjects_test = _load_split("test")

    print(f"UCI HAR loaded:")
    print(f"  Train: {X_train.shape}, classes={sorted(set(y_train))}")
    print(f"  Test : {X_test.shape},  subjects_train={sorted(set(subjects_train))}")

    return X_train, y_train, X_test, y_test, feature_names, subjects_train, subjects_test
