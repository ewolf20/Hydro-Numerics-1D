#!/usr/bin/env python3
"""Create publication-quality waterfall plots from moving-piston CSV exports.

Expected CSV columns:
    time, x, density, velocity, pressure
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm, colors
from matplotlib.ticker import MultipleLocator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot density, velocity, and pressure waterfall plots from exported CSV.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path to CSV file. If omitted, uses latest moving_piston_no_left_gas_Re*.csv",
    )
    parser.add_argument(
        "--n-snapshots",
        type=int,
        default=12,
        help="Maximum number of time snapshots to display (default: 12)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output PNG DPI (default: 300)",
    )
    parser.add_argument(
        "--stack-spacing",
        type=float,
        default=0.32,
        help=(
            "Vertical spacing factor between snapshots as a fraction of variable span "
            "(default: 0.32). Increase this if time labels touch curves."
        ),
    )
    parser.add_argument(
        "--trim-piston-cells",
        type=int,
        default=6,
        help=(
            "Number of near-piston cells to omit from plotted curves for density/pressure "
            "(default: 5). Set 0 to keep full curve."
        ),
    )
    parser.set_defaults(omit_piston_rise=True)
    parser.add_argument(
        "--keep-piston-rise",
        dest="omit_piston_rise",
        action="store_false",
        help="Keep the sharp near-piston rise (disable trimming of near-piston cells).",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Output prefix (without extension). Defaults to waterfall_Re<...>",
    )
    parser.add_argument(
        "--piston-method",
        choices=("infer", "analytic", "none"),
        default="infer",
        help=(
            "How to get piston position xp(t): "
            "'infer' from density floor, 'analytic' from piston law, "
            "'none' disables right-side clipping."
        ),
    )
    parser.add_argument(
        "--rho-floor",
        type=float,
        default=1.0e-10,
        help="Density floor used in Julia solver export (default: 1e-10)",
    )
    parser.add_argument(
        "--rho-threshold-factor",
        type=float,
        default=1.0e3,
        help="Threshold factor for infer mode: density > rho_floor * factor (default: 1e3)",
    )
    parser.add_argument(
        "--x-left",
        type=float,
        default=0.0,
        help="Left wall position for analytic xpiston(t) (default: 0.0)",
    )
    parser.add_argument(
        "--a0",
        type=float,
        default=5.0,
        help="Piston acceleration parameter a0 for analytic mode (default: 5.0)",
    )
    parser.add_argument(
        "--t-stop",
        type=float,
        default=5.0,
        help="Piston acceleration stop time for analytic mode (default: 5.0)",
    )
    parser.add_argument(
        "--fig-width",
        type=float,
        default=8.0,
        help="Figure width in inches (default: 8.0). Use 4.0 for twice shorter x-axis scale.",
    )
    parser.add_argument(
        "--fig-height",
        type=float,
        default=None,
        help="Figure height in inches (default: 5.0 for regular, 7.0 for shifted). Set to override.",
    )
    parser.set_defaults(draw_piston=True)
    parser.add_argument(
        "--no-draw-piston",
        dest="draw_piston",
        action="store_false",
        help="Disable red piston position markers in waterfall curves.",
    )
    parser.set_defaults(annotate_times=True)
    parser.add_argument(
        "--no-annotate-times",
        dest="annotate_times",
        action="store_false",
        help="Disable per-curve time annotations.",
    )
    parser.set_defaults(make_shifted_set=True)
    parser.add_argument(
        "--no-shifted-set",
        dest="make_shifted_set",
        action="store_false",
        help="Disable additional piston-frame shifted plots.",
    )
    return parser.parse_args()


def find_latest_csv() -> Path:
    candidates = sorted(
        Path(".").glob("moving_piston_no_left_gas_Re*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "No CSV found matching moving_piston_no_left_gas_Re*.csv in current directory."
        )
    return candidates[0]


def find_all_csvs() -> list[Path]:
    candidates = sorted(Path(".").glob("moving_piston_no_left_gas_Re*.csv"))
    if not candidates:
        raise FileNotFoundError(
            "No CSV found matching moving_piston_no_left_gas_Re*.csv in current directory."
        )
    return candidates


def extract_re_label(csv_path: Path) -> str:
    match = re.search(r"_Re(.+)$", csv_path.stem)
    if not match:
        return "unknown"
    return match.group(1).replace("p", ".")


def sample_times(times: np.ndarray, n_snapshots: int) -> np.ndarray:
    """Sample snapshots at approximately equal physical time intervals.

    `times` are available simulation output times (nonuniform due to CFL).
    We construct uniformly spaced target times and select the nearest available
    time for each target.
    """
    if len(times) <= n_snapshots:
        return times

    targets = np.linspace(float(times[0]), float(times[-1]), n_snapshots)
    idx = []
    for t in targets:
        j = int(np.argmin(np.abs(times - t)))
        idx.append(j)

    idx = np.unique(np.array(idx, dtype=int))
    return times[idx]


def compute_offset(values: np.ndarray, spacing: float) -> float:
    span = float(np.nanmax(values) - np.nanmin(values))
    if not np.isfinite(span) or span <= 0.0:
        return 1.0
    return spacing * span


def build_snapshot_cache(df: pd.DataFrame, times: np.ndarray) -> dict:
    """Cache sorted arrays per sampled time to avoid repeated DataFrame filtering."""
    time_list = [float(t) for t in times]
    df_sub = df.loc[
        df["time"].isin(time_list), ["time", "x", "density", "velocity", "pressure"]
    ].copy()
    cache: dict = {}
    for t, grp in df_sub.groupby("time", sort=False):
        g = grp.sort_values("x")
        cache[float(t)] = {
            "x": g["x"].to_numpy(dtype=float),
            "density": g["density"].to_numpy(dtype=float),
            "velocity": g["velocity"].to_numpy(dtype=float),
            "pressure": g["pressure"].to_numpy(dtype=float),
        }
    return cache


def xpiston_analytic(t: float, x_left: float, a0: float, t_stop: float) -> float:
    if t <= 0.0:
        return x_left
    if t <= t_stop:
        return x_left + 0.5 * a0 * t**2
    return x_left + 0.5 * a0 * t_stop**2


def infer_xpiston_from_density_arrays(
    x: np.ndarray, rho: np.ndarray, rho_floor: float, threshold_factor: float
) -> float:
    if len(x) == 0:
        return np.nan
    threshold = rho_floor * threshold_factor
    mask = rho > threshold
    if np.any(mask):
        return float(x[np.argmax(mask)])
    return float(x[0])


def build_piston_positions(cache: dict, times: np.ndarray, args: argparse.Namespace) -> dict:
    if args.piston_method == "none":
        return {float(t): np.nan for t in times}

    xp_by_time: dict = {}
    for t in times:
        if args.piston_method == "analytic":
            xp = xpiston_analytic(float(t), args.x_left, args.a0, args.t_stop)
        else:
            snap = cache[float(t)]
            xp = infer_xpiston_from_density_arrays(
                snap["x"],
                snap["density"],
                rho_floor=args.rho_floor,
                threshold_factor=args.rho_threshold_factor,
            )
        xp_by_time[float(t)] = float(xp)
    return xp_by_time


def plot_waterfall(
    ax,
    cache: dict,
    times: np.ndarray,
    var: str,
    cmap,
    norm,
    xp_by_time: dict,
    draw_piston: bool,
    annotate_times: bool,
    stack_spacing: float,
    shift_by_piston: bool,
    x_left: float,
    omit_piston_rise: bool,
    trim_piston_cells: int,
) -> None:
    ylabels = {
        "density": r"$\rho^*$ (stacked)",
        "velocity": r"$u^*$ (stacked)",
        "pressure": r"$p^*$ (stacked)",
    }
    var_values = np.concatenate([cache[float(t)][var] for t in times])
    offset = compute_offset(var_values, stack_spacing)

    for k, t in enumerate(times):
        snap = cache[float(t)]
        x = snap["x"]
        y_raw = snap[var]
        y0 = k * offset
        xp = float(xp_by_time.get(float(t), np.nan))
        disp = (xp - x_left) if np.isfinite(xp) else 0.0
        x_use = x - disp if shift_by_piston else x

        if np.isfinite(xp):
            valid = x >= xp
            x_plot = x_use[valid]
            y_plot = y_raw[valid] + y0
        else:
            x_plot = x_use
            y_plot = y_raw + y0

        # Optionally hide the first few points right of the piston to suppress the
        # floor-to-equilibrium jump in density/pressure.
        if (
            omit_piston_rise
            and np.isfinite(xp)
            and var in ("density", "pressure")
            and trim_piston_cells > 0
            and len(x_plot) > trim_piston_cells
        ):
            x_plot = x_plot[trim_piston_cells:]
            y_plot = y_plot[trim_piston_cells:]

        if len(x_plot) == 0:
            continue

        ax.plot(x_plot, y_plot, color=cmap(norm(float(t))), lw=2.5)

        if annotate_times:
            y_span = float(np.nanmax(y_raw) - np.nanmin(y_raw))
            y_offset = 0.04 * y_span if y_span > 0.0 else 0.02
            x_target = 0.4 if shift_by_piston else 1.45
            x_annot = min(x_target, float(np.nanmax(x_plot)))
            if x_annot < float(x_plot[-1]):
                y_annot = float(np.interp(x_annot, x_plot, y_plot))
            else:
                y_annot = float(y_plot[-1])
            ax.text(
                x_annot,
                y_annot + y_offset,
                rf"$t^*={float(t):.3f}$",
                fontsize=20,
                va="bottom",
                ha="right",
                color="black",
                alpha=1.0,
                fontweight="bold",
            )

        if draw_piston and np.isfinite(xp):
            x_p_draw = x_left if shift_by_piston else xp
            ax.plot(
                [x_p_draw, x_p_draw],
                [float(np.nanmin(y_raw) + y0), float(np.nanmax(y_raw) + y0)],
                color="red",
                alpha=1.0,
                lw=3,
            )

    ax.set_ylabel(ylabels.get(var, f"{var} (stacked)"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def get_cmap(name="viridis"):
    try:
        import matplotlib as mpl
        return mpl.colormaps[name]
    except (AttributeError, ImportError):
        try:
            import matplotlib.pyplot as plt
            return plt.get_cmap(name)
        except AttributeError:
            from matplotlib import cm
            return cm.get_cmap(name)

def main() -> None:
    args = parse_args()

    if args.csv is not None:
        csv_paths = [args.csv]
    else:
        csv_paths = find_all_csvs()

    # Publication-style defaults
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.size": 16,
            "axes.labelsize": 18,
            "axes.titlesize": 20,
            "legend.fontsize": 21,
            "xtick.labelsize": 21,
            "ytick.labelsize": 21,
            "lines.linewidth": 2.5,
            "axes.linewidth": 1.2,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )

    for csv_path in csv_paths:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        required = {"time", "x", "density", "velocity", "pressure"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")

        times_all = np.sort(df["time"].unique())
        times = sample_times(times_all, max(1, args.n_snapshots))
        snapshot_cache = build_snapshot_cache(df, times)
        xp_by_time = build_piston_positions(snapshot_cache, times, args)

        cmap = get_cmap("viridis")
        if len(times) == 1:
            t0 = float(times[0])
            norm = colors.Normalize(vmin=t0 - 1.0, vmax=t0 + 1.0)
        else:
            norm = colors.Normalize(vmin=float(times[0]), vmax=float(times[-1]))

        re_label = extract_re_label(csv_path)
        if args.output_prefix is None:
            base_prefix = Path(f"waterfall_Re{re_label.replace('.', 'p')}")
        elif len(csv_paths) == 1:
            base_prefix = args.output_prefix
        else:
            base_prefix = Path(f"{args.output_prefix}_{csv_path.stem}")

        saved_outputs = []
        for var in ("density", "velocity", "pressure"):
            fig_height = args.fig_height if args.fig_height is not None else 5.0
            fig, ax = plt.subplots(
                1,
                1,
                figsize=(args.fig_width, fig_height),
                constrained_layout=True,
            )
            plot_waterfall(
                ax,
                snapshot_cache,
                times,
                var,
                cmap,
                norm,
                xp_by_time,
                args.draw_piston,
                args.annotate_times,
                args.stack_spacing,
                False,
                args.x_left,
                args.omit_piston_rise,
                args.trim_piston_cells,
            )
            ax.set_xlabel(r"$x^*$")
            ax.yaxis.set_major_locator(MultipleLocator(2))
            ax.set_xlim(0, 1.5)
            ax.set_ylim(bottom=0)
            # fig.suptitle(
            #     f"Re={re_label}",
            #     y=1.01,
            # )

            var_prefix = Path(f"{base_prefix}_{var}")
            png_path = var_prefix.with_suffix(".png")
            pdf_path = var_prefix.with_suffix(".pdf")
            fig.savefig(png_path, dpi=args.dpi)
            fig.savefig(pdf_path)
            saved_outputs.append((var, png_path, pdf_path))
            plt.close(fig)

            if args.make_shifted_set:
                fig_height_shift = args.fig_height if args.fig_height is not None else 7.0
                fig_s, ax_s = plt.subplots(
                    1,
                    1,
                    figsize=(args.fig_width, fig_height_shift),
                    constrained_layout=True,
                )
                plot_waterfall(
                    ax_s,
                    snapshot_cache,
                    times,
                    var,
                    cmap,
                    norm,
                    xp_by_time,
                    False,
                    args.annotate_times,
                    args.stack_spacing,
                    True,
                    args.x_left,
                    args.omit_piston_rise,
                    args.trim_piston_cells,
                )
                ax_s.set_xlabel(r"$x^*-\Delta x_p^*(t)$")
                ax_s.yaxis.set_major_locator(MultipleLocator(2))
                ax_s.set_xlim(0, 0.4)
                ax_s.set_ylim(bottom=0)

                var_prefix_shift = Path(f"{base_prefix}_{var}_pistonframe")
                png_path_shift = var_prefix_shift.with_suffix(".png")
                pdf_path_shift = var_prefix_shift.with_suffix(".pdf")
                fig_s.savefig(png_path_shift, dpi=args.dpi)
                fig_s.savefig(pdf_path_shift)
                saved_outputs.append((f"{var}-pistonframe", png_path_shift, pdf_path_shift))
                plt.close(fig_s)

        print(f"Input CSV: {csv_path}")
        print(f"Piston mode: {args.piston_method}")
        for var, png_path, pdf_path in saved_outputs:
            print(f"[{var}] Saved PNG: {png_path}")
            print(f"[{var}] Saved PDF: {pdf_path}")


if __name__ == "__main__":
    main()
