import numpy as np
from scipy import stats


def _create_feature_names():
    feature_names = []

    # Time and frequency domain features for each sensor
    time_features = [
        "mean",
        "std",
        "max",
        "min",
        "median",
        "skew",
        "kurtosis",
        "q25",
        "q75",
        "q90",
        "q10",
        "mean_abs",
        "rms",
        "range",
        "zcr",
    ]
    freq_features = [
        "spectral_centroid",
        "spectral_spread",
        "low_band_power",
        "mid_band_power",
        "high_band_power",
        "spectral_entropy",
    ]

    for sensor in ["acc", "grav", "gyro", "magnet"]:
        for axis in ["x", "y", "z"]:
            for feat in time_features:
                feature_names.append(f"{sensor}_{axis}_{feat}")
            for feat in freq_features:
                feature_names.append(f"{sensor}_{axis}_{feat}")

    # Magnitude features
    mag_features = ["mean", "std", "max", "min", "skew", "kurtosis"]
    for sensor in ["acc", "grav", "gyro", "magnet"]:
        for feat in mag_features:
            feature_names.append(f"{sensor}_magnitude_{feat}")

    # Correlation features
    corr_pairs = [
        "acc_xy",
        "acc_xz",
        "acc_yz",
        "gyro_xy",
        "gyro_xz",
        "gyro_yz",
        "grav_xy",
        "grav_xz",
        "grav_yz",
        "magnet_xy",
        "magnet_xz",
        "magnet_yz",
        "acc_gyro_magnitude",
        "acc_grav_magnitude",
        "acc_magnet_magnitude",
        "gyro_grav_magnitude",
        "gyro_magnet_magnitude",
        "grav_magnet_magnitude",
    ]
    for pair in corr_pairs:
        feature_names.append(f"correlation_{pair}")
    
    return feature_names


def engineer_robust_features(windowed_data, sampling_rate):
    """
    Extract features that are robust to variable window lengths
    """
    # # PRints
    # print("\n=== Robust Feature Engineering ===")

    feature_list = []
    label_list = []
    metadata_list = []

    for idx, window in enumerate(windowed_data):
        if idx % 100 == 0:
            print(f"  Processing window {idx}/{len(windowed_data)}...")

        window_features = []

        # Length-invariant features
        for sensor_type in ["acc", "grav", "gyro", "magnet"]:
            for axis in ["x", "y", "z"]:
                signal = window[f"{sensor_type}_{axis}"]

                # Time domain features
                window_features.extend(
                    [
                        np.mean(signal),  # Mean value
                        np.std(signal),  # Standard deviation
                        np.median(signal),  # Median absolute deviation (amplitude)
                        np.max(signal),  # Maximum value
                        np.min(signal),  # Minimum value
                        stats.skew(signal),  # Distribution skewness
                        stats.kurtosis(signal),  # Distribution kurtosis
                        np.percentile(signal, 25),  # 25th percentile
                        np.percentile(signal, 75),  # 75th percentile
                        np.percentile(signal, 90),  # 90th percentile
                        np.percentile(signal, 10),  # 10th percentile
                        np.mean(np.abs(signal)),  # Mean absolute value
                        np.sqrt(np.mean(signal**2)),  # Root mean square
                        np.max(signal) - np.min(signal),  # Signal range
                        np.sum(np.diff(np.sign(signal)) != 0)
                        / len(signal),  # Zero crossing rate
                    ]
                )

                # Frequency domain features (resolution-independent)
                fft_vals = np.fft.fft(signal)
                fft_magnitude = np.abs(fft_vals[: len(fft_vals) // 2])
                freqs = np.fft.fftfreq(len(signal), 1 / sampling_rate)[
                    : len(fft_vals) // 2
                ]

                # Normalized frequency features
                total_power = np.sum(fft_magnitude**2)
                if total_power > 0:
                    normalized_fft = fft_magnitude**2 / total_power

                    # Spectral features
                    spectral_centroid = (
                        np.sum(freqs * normalized_fft) / np.sum(normalized_fft)
                        if np.sum(normalized_fft) > 0
                        else 0
                    )
                    spectral_spread = (
                        np.sqrt(
                            np.sum(((freqs - spectral_centroid) ** 2) * normalized_fft)
                            / np.sum(normalized_fft)
                        )
                        if np.sum(normalized_fft) > 0
                        else 0
                    )

                    # Band power ratios
                    low_band = np.sum(normalized_fft[freqs < 5])
                    mid_band = np.sum(normalized_fft[(freqs >= 5) & (freqs < 15)])
                    high_band = np.sum(normalized_fft[freqs >= 15])

                    window_features.extend(
                        [
                            spectral_centroid,
                            spectral_spread,
                            low_band,
                            mid_band,
                            high_band,
                            stats.entropy(normalized_fft + 1e-10),  # Spectral entropy
                        ]
                    )
                else:
                    window_features.extend([0] * 6)

        # Magnitude features
        acc_magnitude = np.sqrt(
            window["acc_x"] ** 2 + window["acc_y"] ** 2 + window["acc_z"] ** 2
        )
        gyro_magnitude = np.sqrt(
            window["gyro_x"] ** 2 + window["gyro_y"] ** 2 + window["gyro_z"] ** 2
        )
        grav_magnitude = np.sqrt(
            window["grav_x"] ** 2 + window["grav_y"] ** 2 + window["grav_z"] ** 2
        )
        magnet_magnitude = np.sqrt(
            window["magnet_x"] ** 2 + window["magnet_y"] ** 2 + window["magnet_z"] ** 2
        )

        for magnitude in [acc_magnitude, gyro_magnitude, grav_magnitude, magnet_magnitude]:
            window_features.extend(
                [
                    np.mean(magnitude),
                    np.std(magnitude),
                    np.max(magnitude),
                    np.min(magnitude),
                    stats.skew(magnitude),
                    stats.kurtosis(magnitude),
                ]
            )

        # Correlation features (length-independent)
        try:
            correlations = [
                np.corrcoef(window["acc_x"], window["acc_y"])[0, 1],
                np.corrcoef(window["acc_x"], window["acc_z"])[0, 1],
                np.corrcoef(window["acc_y"], window["acc_z"])[0, 1],
                np.corrcoef(window["gyro_x"], window["gyro_y"])[0, 1],
                np.corrcoef(window["gyro_x"], window["gyro_z"])[0, 1],
                np.corrcoef(window["gyro_y"], window["gyro_z"])[0, 1],
                np.corrcoef(window["grav_x"], window["grav_y"])[0, 1],
                np.corrcoef(window["grav_x"], window["grav_z"])[0, 1],
                np.corrcoef(window["grav_y"], window["grav_z"])[0, 1],
                np.corrcoef(window["magnet_x"], window["magnet_y"])[0, 1],
                np.corrcoef(window["magnet_x"], window["magnet_z"])[0, 1],
                np.corrcoef(window["magnet_y"], window["magnet_z"])[0, 1],
                np.corrcoef(acc_magnitude, gyro_magnitude)[0, 1],
                np.corrcoef(acc_magnitude, grav_magnitude)[0, 1],
                np.corrcoef(acc_magnitude, magnet_magnitude)[0, 1],
                np.corrcoef(gyro_magnitude, grav_magnitude)[0, 1],
                np.corrcoef(gyro_magnitude, magnet_magnitude)[0, 1],
                np.corrcoef(grav_magnitude, magnet_magnitude)[0, 1],
                np.corrcoef(acc_magnitude, gyro_magnitude)[0, 1],
                np.corrcoef(acc_magnitude, grav_magnitude)[0, 1],
                np.corrcoef(acc_magnitude, magnet_magnitude)[0, 1],
                np.corrcoef(gyro_magnitude, grav_magnitude)[0, 1],
                np.corrcoef(gyro_magnitude, magnet_magnitude)[0, 1],
                np.corrcoef(grav_magnitude, magnet_magnitude)[0, 1],
            ]
            window_features.extend(correlations)
        except:
            window_features.extend([0] * 24)

        # Handle NaN/Inf values
        window_features = [
            0 if np.isnan(x) or np.isinf(x) else x for x in window_features
        ]

        feature_list.append(window_features)
        label_list.append(window["activity"])
        metadata_list.append(
            {
                "participant": window["participant"],
                "session": window["session"],
                "data_source": window["data_source"],
                "window_length": window["window_length"],
            }
        )

    features = np.array(feature_list)
    labels = np.array(label_list)
    metadata = metadata_list

    # print(f"\nFeatures shape: {features.shape}")
    # print(f"Number of features: {features.shape[1]}")

    feature_names = _create_feature_names()

    return features, labels, feature_names, metadata_list
