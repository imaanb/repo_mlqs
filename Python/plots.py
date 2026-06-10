import matplotlib.pylab as plt


def analyze_data_distribution(raw_data):

    print("\n=== Data Distribution Analysis ===")
    # Activity distribution per source
    activity_dist = (
        raw_data.groupby(["data_source", "activity_label"]).size().unstack(fill_value=0)
    )
    print("\nActivity distribution by data source:")
    print(activity_dist)

    # Plot distributions
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    activity_dist.T.plot(kind="bar", ax=axes[0, 0])
    axes[0, 0].set_title("Activity Distribution by Data Source")
    axes[0, 0].set_xlabel("Activity")
    axes[0, 0].set_ylabel("Count")
    sensor_cols = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    for idx, col in enumerate(sensor_cols[:3]):
        ax_idx = (idx + 1) // 2, (idx + 1) % 2
        for source in raw_data["data_source"].unique():
            source_data = raw_data[raw_data["data_source"] == source][col]
            axes[ax_idx].hist(
                source_data, bins=50, alpha=0.5, label=source, density=True
            )
        axes[ax_idx].set_title(f"{col} Distribution")
        axes[ax_idx].set_xlabel("Value")
        axes[ax_idx].set_ylabel("Density")
        axes[ax_idx].legend()

    plt.tight_layout()
    plt.show()

    # Statistical comparison
    print("\n\nSensor statistics by data source:")
    for source in raw_data["data_source"].unique():
        print(f"\n{source.upper()}:")
        source_data = raw_data[raw_data["data_source"] == source]
        print(source_data[sensor_cols].describe())
