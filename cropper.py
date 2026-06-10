"""
crop_sensor.py
--------------
Crops non-exercise artefacts from wearable sensor CSVs.

Consensus mode:
    For a given exercise / participant / round, every sensor x axis pair
    independently detects its crop start and end (in seconds_elapsed).
    The MEDIAN start and MEDIAN end across all pairs is used as the final
    crop point, applied uniformly to every sensor CSV.

Usage
-----
    python crop_sensor.py [options]

Options
-------
  --exercise STR        Exercise folder name (default: row)
  --participant STR     Participant folder name (default: imaan)
  --round INT           Round folder (default: 1)
  --output-dir PATH     Write cropped CSVs here (default: Datasets/cropped/)
  --min-peaks INT       Minimum peaks to require (default: 4)
  --peak-prom FLOAT     Prominence threshold as fraction of range (default: 0.15)
  --window-frac FLOAT   Sliding-window width as fraction of length (default: 0.12)
  --show-crop           Plot every sensor with consensus crop lines overlaid
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter


# ── sensor / axis catalogue ──────────────────────────────────────────────────

SENSORS = {
    "Orientation":  ["qw", "qx", "qy", "qz", "roll", "pitch", "yaw"],
    "Accelerometer": ["x", "y", "z"],
    "Gyroscope":    ["x", "y", "z"],
    "Magnetometer": ["x", "y", "z"],
    "Gravity":     ["x", "y", "z"],
}

DATA_ROOT = Path("Datasets/Gym Movements Dataset")


# ── low-level helpers ────────────────────────────────────────────────────────

def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.dropna(how="all", inplace=True)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(how="all", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _time_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if "second" in c.lower() or "elapsed" in c.lower():
            return c
    return None


def _smooth(signal: np.ndarray, window_frac: float = 0.02) -> np.ndarray:
    n = len(signal)
    wlen = max(5, int(n * window_frac))
    wlen += (wlen % 2 == 0)
    poly = min(3, wlen - 1)
    return savgol_filter(signal, wlen, poly)


def _crop_indices(
    signal: np.ndarray,
    min_peaks: int,
    peak_prom_frac: float,
    window_frac: float,
) -> tuple[int, int]:
    """Return (start_idx, end_idx) of the most densely periodic region."""
    n = len(signal)
    smooth = _smooth(signal)
    sig_range = smooth.max() - smooth.min()
    if sig_range < 1e-9:
        return 0, n - 1

    prominence = peak_prom_frac * sig_range
    peaks_pos, _ = find_peaks( smooth, prominence=prominence)
    peaks_neg, _ = find_peaks(-smooth, prominence=prominence)
    all_peaks = np.sort(np.concatenate([peaks_pos, peaks_neg]))

    if len(all_peaks) < min_peaks:
        return 0, n - 1

    half_w = max(1, int(n * window_frac / 2))
    density = np.zeros(n)
    for p in all_peaks:
        density[max(0, p - half_w) : min(n, p + half_w)] += 1

    sd = _smooth(density, window_frac=0.05)
    threshold = np.median(sd[sd > 0]) * 0.5
    active = sd >= threshold

    best_start, best_end, best_len, cur_start = 0, n - 1, 0, None
    for i in range(n):
        if active[i] and cur_start is None:
            cur_start = i
        elif not active[i] and cur_start is not None:
            length = i - cur_start
            if length > best_len:
                best_len, best_start, best_end = length, cur_start, i - 1
            cur_start = None
    if cur_start is not None and (n - cur_start) > best_len:
        best_start, best_end = cur_start, n - 1

    inside = all_peaks[(all_peaks >= best_start) & (all_peaks <= best_end)]
    if len(inside) < min_peaks:
        return 0, n - 1
    return int(inside[0]), int(inside[-1])


# ── consensus crop ────────────────────────────────────────────────────────────

def compute_consensus_crop(
    exercise: str,
    participant: str,
    round_id: int | str,
    sensors: dict | None = None,
    min_peaks: int = 4,
    peak_prom_frac: float = 0.15,
    window_frac: float = 0.12,
    verbose: bool = True,
) -> tuple[float, float]:
    """
    For every sensor x axis, detect crop boundaries in seconds_elapsed.
    Returns (median_start_sec, median_end_sec).
    """
    if sensors is None:
        sensors = SENSORS

    base = DATA_ROOT / exercise / participant / str(round_id)
    starts, ends = [], []

    for sensor_name, axes in sensors.items():
        csv_path = base / f"{sensor_name}.csv"
        if not csv_path.exists():
            if verbose:
                print(f"  [skip] {csv_path} not found")
            continue

        df = _read_csv(csv_path)
        if df.empty:
            continue

        tc = _time_col(df)
        t = df[tc].to_numpy(dtype=float) if tc else np.arange(len(df))

        for axis in axes:
            if axis not in df.columns:
                continue
            signal = df[axis].to_numpy(dtype=float)
            si, ei = _crop_indices(signal, min_peaks, peak_prom_frac, window_frac)
            t_start, t_end = float(t[si]), float(t[ei])
            starts.append(t_start)
            ends.append(t_end)
            if verbose:
                print(f"  {sensor_name:14s} [{axis}]  "
                      f"start={t_start:.3f}s  end={t_end:.3f}s")

    if not starts:
        raise RuntimeError("No valid axes found — check your data root / sensor names.")

    med_start = float(np.median(starts))
    med_end   = float(np.median(ends))
    if verbose:
        print(f"\n  consensus  start={med_start:.3f}s  end={med_end:.3f}s  "
              f"(from {len(starts)} axis estimates)\n")
    return med_start, med_end


# ── apply crop and save ───────────────────────────────────────────────────────

def apply_and_save(
    exercise: str,
    participant: str,
    round_id: int | str,
    med_start: float,
    med_end: float,
    output_dir: Path,
    sensors: dict | None = None,
    show_crop: bool = False,
) -> None:
    if sensors is None:
        sensors = SENSORS

    base = DATA_ROOT / exercise / participant / str(round_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    for sensor_name in sensors:
        csv_path = base / f"{sensor_name}.csv"
        if not csv_path.exists():
            continue

        df = _read_csv(csv_path)
        if df.empty:
            continue

        tc = _time_col(df)
        t = df[tc].to_numpy(dtype=float) if tc else np.arange(len(df))

        # Nearest row to consensus time points
        start_idx = int(np.argmin(np.abs(t - med_start)))
        end_idx   = int(np.argmin(np.abs(t - med_end)))

        cropped = df.iloc[start_idx : end_idx + 1].reset_index(drop=True)
        out_path = output_dir / f"{exercise}/{participant}/{round_id}/{sensor_name}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cropped.to_csv(out_path, index=False)

        pct = 100 * len(cropped) / len(df)
        print(f"[saved] {sensor_name:14s}  "
              f"{len(df):>6} → {len(cropped):>6} rows ({pct:.1f}%)  "
              f"→ {out_path}")

        if show_crop:
            _plot_sensor(df, sensor_name, t, tc or "index",
                         start_idx, end_idx, med_start, med_end)


# ── plotting ──────────────────────────────────────────────────────────────────

def _plot_sensor(
    df: pd.DataFrame,
    sensor_name: str,
    t: np.ndarray,
    xlabel: str,
    start_idx: int,
    end_idx: int,
    med_start: float,
    med_end: float,
) -> None:
    import matplotlib.pyplot as plt

    data_cols = [c for c in df.columns
                 if not ("time" in c.lower() or "elapsed" in c.lower()
                         or "second" in c.lower())]
    if not data_cols:
        return

    n_axes = len(data_cols)
    fig, axs = plt.subplots(n_axes, 1, figsize=(13, 2.2 * n_axes),
                             sharex=True, squeeze=False)
    fig.suptitle(f"{sensor_name} — consensus crop  "
                 f"[start={med_start:.2f}s, end={med_end:.2f}s]",
                 fontsize=11, y=1.01)

    t_min, t_max = t[0], t[-1]
    palette = ["#4a9eca", "#e8734a", "#5cb85c", "#9b59b6",
               "#e2c640", "#1abc9c", "#e05c5c"]

    for i, col in enumerate(data_cols):
        ax = axs[i][0]
        ax.plot(t, df[col].to_numpy(dtype=float),
                lw=0.7, color=palette[i % len(palette)], label=col)
        ax.axvspan(t_min,   med_start, alpha=0.18, color="#e05c5c", label="cropped" if i == 0 else "")
        ax.axvspan(med_end, t_max,     alpha=0.18, color="#e0a05c")
        ax.axvline(med_start, color="#e05c5c", lw=1.4, ls="--")
        ax.axvline(med_end,   color="#e0a05c", lw=1.4, ls="--")
        ax.set_ylabel(col, fontsize=8)
        ax.legend(loc="upper right", fontsize=7)
        ax.tick_params(labelsize=7)

    axs[-1][0].set_xlabel(xlabel, fontsize=9)
    fig.tight_layout()
    plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Consensus crop of sensor CSVs using median split point.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--exercise",    default="row")
    p.add_argument("--participant", default="richard")
    p.add_argument("--round",       default=1, type=int)
    p.add_argument("--output-dir",  type=Path, default=Path(f"Datasets/cropped"))
    p.add_argument("--min-peaks",   type=int,   default=4)
    p.add_argument("--peak-prom",   type=float, default=0.15)
    p.add_argument("--window-frac", type=float, default=0.12)
    p.add_argument("--show",   action="store_true", default=False)
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    print(f"Computing consensus crop for: {args.exercise} / "
          f"{args.participant} / round {args.round}\n")

    med_start, med_end = compute_consensus_crop(
        exercise=args.exercise,
        participant=args.participant,
        round_id=args.round,
        min_peaks=args.min_peaks,
        peak_prom_frac=args.peak_prom,
        window_frac=args.window_frac,
    )

    apply_and_save(
        exercise=args.exercise,
        participant=args.participant,
        round_id=args.round,
        med_start=med_start,
        med_end=med_end,
        output_dir=args.output_dir,
        show_crop=args.show,
    )


if __name__ == "__main__":
    main()