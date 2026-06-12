import numpy as np


def _pad_window(self, window_dict, target_length):
    """Pad window to target length using reflection padding"""
    for key in ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]:
        current_length = len(window_dict[key])

        if current_length < target_length:
            pad_length = target_length - current_length
            window_dict[key] = np.pad(window_dict[key], (0, pad_length), mode="reflect")

    return window_dict


def _create_window_dict(window, participant, session, window_id):
    activity = (
        window["activity_label"].mode()[0]
        if len(window["activity_label"].mode()) > 0
        else window["activity_label"].iloc[0]
    )
    d = {
        "window_id": window_id,
        "participant": participant,
        "session": session,
        "activity": activity,
        "acc_x": window["acc_x"].values,
        "acc_y": window["acc_y"].values,
        "acc_z": window["acc_z"].values,
        "gyro_x": window["gyro_x"].values,
        "gyro_y": window["gyro_y"].values,
        "gyro_z": window["gyro_z"].values,
        "data_source": (
            window["data_source"].iloc[0] if "data_source" in window else "unknown"
        ),
        "window_length": len(window),
    }
    for col in ["grav_x", "grav_y", "grav_z",
                "magnet_x", "magnet_y", "magnet_z",
                "acc_vert", "acc_horiz", "gyro_vert", "gyro_horiz"]:
        if col in window.columns:
            d[col] = window[col].values
    return d


def create_adaptive_windows(
    raw_data, min_window_size=64, preferred_window_size=128, overlap_ratio=0.5
):
    """
    Create windows with adaptive sizing for variable-length data

    Args:
        min_window_size: Minimum acceptable window size
        preferred_window_size: Preferred window size
        overlap_ratio: Overlap ratio (0.5 = 50% overlap)
    """

    # # Prints
    # print(f"\nCreating adaptive windows...")
    # print(f"Preferred size: {preferred_window_size}, Minimum size: {min_window_size}")

    # Initialise the windows
    windowed_data = []
    window_stats = {"full_windows": 0, "adaptive_windows": 0, "rejected_segments": 0}

    # Group data by participant and session
    grouped = raw_data.groupby(["participant", "session"])

    for (participant, session), group_data in grouped:
        data_length = len(group_data)

        # Determine window size for this segment
        if data_length >= preferred_window_size:
            # Use preferred window size
            window_size = preferred_window_size
            step_size = int(window_size * (1 - overlap_ratio))

            for i in range(0, data_length - window_size + 1, step_size):
                window = group_data.iloc[i : i + window_size]
                window_dict = _create_window_dict(
                    window, participant, session, len(windowed_data)
                )
                windowed_data.append(window_dict)
                window_stats["full_windows"] += 1

        elif data_length >= min_window_size:
            # Use adaptive window size
            window_size = data_length
            window = group_data
            window_dict = _create_window_dict(
                window, participant, session, len(windowed_data)
            )
            # Pad to preferred size for consistency
            window_dict = _pad_window(window_dict, preferred_window_size)
            windowed_data.append(window_dict)
            window_stats["adaptive_windows"] += 1

        else:
            # Segment too short
            window_stats["rejected_segments"] += 1
            print(
                f"  Rejected segment: {participant}/session_{session} (length: {data_length})"
            )

    print(f"\nWindow creation summary:")
    print(f"  Full-size windows: {window_stats['full_windows']}")
    print(f"  Adaptive windows: {window_stats['adaptive_windows']}")
    print(f"  Rejected segments: {window_stats['rejected_segments']}")
    print(f"  Total windows: {len(windowed_data)}")

    return windowed_data
