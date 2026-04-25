"""Synthetic motion runners for Brownian, contractile, and extensile test data.

This module keeps the notebook thin and makes the full synthetic workflow runnable
from a normal Python file or command line.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation

import analysis_unified as analysis_unified_module


DEFAULT_PARTICLE_COUNT = 300
DEFAULT_FRAME_COUNT = 60
DEFAULT_AVERAGE_WINDOW_FRAMES = 60
DEFAULT_SEED = 7
DEFAULT_BASE_DIR = Path("data")
DEFAULT_OUTPUT_SUFFIX = "60tp_avg20"
DEFAULT_OVERLAY_SCALE = 2.5


def save_motion_overlay_animation(
    result: dict[str, Any],
    output_path: str | Path,
    title: str,
    overlay_scale: float = DEFAULT_OVERLAY_SCALE,
) -> Path:
    """Save a drift-corrected XY overlay animation for a synthetic run."""
    tracks = result["tracks_df"]
    velocities = result["velocity_df"]
    state_local = result["state"]
    frame_ids = sorted(int(frame) for frame in velocities["frame"].unique())
    fps_local = float(state_local["calibration"]["fps"])
    first_frame = velocities[velocities["frame"] == frame_ids[0]].sort_values("particle")

    fig, ax = plt.subplots(figsize=(6.5, 6.0), dpi=150)
    x_min = float(tracks["x_um"].min())
    x_max = float(tracks["x_um"].max())
    y_min = float(tracks["y_um"].min())
    y_max = float(tracks["y_um"].max())
    pad = 1.0
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_title(title)

    scatter = ax.scatter(
        first_frame["x_um"],
        first_frame["y_um"],
        c=first_frame["speed_drift_corrected_um_s"],
        cmap="viridis",
        s=28,
        vmin=float(velocities["speed_drift_corrected_um_s"].min()),
        vmax=float(velocities["speed_drift_corrected_um_s"].max()),
    )
    quiver = ax.quiver(
        first_frame["x_um"],
        first_frame["y_um"],
        first_frame["vx_drift_corrected_um_s"] / fps_local * overlay_scale,
        first_frame["vy_drift_corrected_um_s"] / fps_local * overlay_scale,
        angles="xy",
        scale_units="xy",
        scale=1,
        color="white",
        width=0.004,
    )
    time_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="white",
        bbox=dict(facecolor="black", alpha=0.35, edgecolor="none"),
    )

    def update(frame_index: int):
        frame = frame_ids[frame_index]
        frame_df = velocities[velocities["frame"] == frame].sort_values("particle")
        offsets = frame_df[["x_um", "y_um"]].to_numpy(dtype=float)
        speeds = frame_df["speed_drift_corrected_um_s"].to_numpy(dtype=float)
        u = frame_df["vx_drift_corrected_um_s"].to_numpy(dtype=float) / fps_local * overlay_scale
        v = frame_df["vy_drift_corrected_um_s"].to_numpy(dtype=float) / fps_local * overlay_scale
        scatter.set_offsets(offsets)
        scatter.set_array(speeds)
        quiver.set_offsets(offsets)
        quiver.set_UVC(u, v)
        time_text.set_text(f"frame={frame:d}  t={frame / fps_local:.2f} s")
        return scatter, quiver, time_text

    animation = FuncAnimation(fig, update, frames=len(frame_ids), interval=1000 / fps_local, blit=False)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output_path, writer=FFMpegWriter(fps=fps_local, bitrate=12000), dpi=150)
    plt.close(fig)
    return output_path


def _default_dataset_id(mode: str, suffix: str) -> str:
    return f"synthetic_{mode}_vector_corr_{suffix}"


def _run_one(
    mode: str,
    *,
    dataset_id: str | None = None,
    frame_count: int = DEFAULT_FRAME_COUNT,
    average_window_frames: int = DEFAULT_AVERAGE_WINDOW_FRAMES,
    particle_count: int = DEFAULT_PARTICLE_COUNT,
    seed: int = DEFAULT_SEED,
    base_dir: str | Path = DEFAULT_BASE_DIR,
    save_animation: bool = True,
) -> dict[str, Any]:
    mode = str(mode).strip().lower()
    if mode not in {"brownian", "contractile", "extensile"}:
        raise ValueError(f"Unknown synthetic mode: {mode!r}")

    analysis_module = importlib.reload(analysis_unified_module)
    runner = getattr(analysis_module, f"run_synthetic_{mode}_vector_corr_test")
    dataset_id = dataset_id or _default_dataset_id(mode, DEFAULT_OUTPUT_SUFFIX)
    result = runner(
        dataset_id=dataset_id,
        frame_count=int(frame_count),
        particle_count=int(particle_count),
        seed=int(seed),
        average_window_frames=int(average_window_frames),
        base_dir=base_dir,
    )

    animation_path = None
    if save_animation:
        animation_path = save_motion_overlay_animation(
            result,
            Path(result["state"]["paths"]["plots_dir"]) / "vector_correlations" / "frame_1" / f"{result['state']['dataset_id']}_motion_overlay.mp4",
            f"Synthetic {mode} motion averaged over {int(average_window_frames)}-frame windows with drift-corrected velocity overlay",
        )
        print(f"Saved {mode} animation to {animation_path}")

    return {
        "mode": mode,
        "result": result,
        "animation_path": animation_path,
    }


def run_synthetic_suite(
    *,
    modes: tuple[str, ...] = ("brownian", "contractile", "extensile"),
    frame_count: int = DEFAULT_FRAME_COUNT,
    average_window_frames: int = DEFAULT_AVERAGE_WINDOW_FRAMES,
    particle_count: int = DEFAULT_PARTICLE_COUNT,
    seed: int = DEFAULT_SEED,
    base_dir: str | Path = DEFAULT_BASE_DIR,
    save_animation: bool = True,
) -> dict[str, dict[str, Any]]:
    """Run the requested synthetic modes and return their results."""
    outputs: dict[str, dict[str, Any]] = {}
    for mode in modes:
        outputs[mode] = _run_one(
            mode,
            frame_count=frame_count,
            average_window_frames=average_window_frames,
            particle_count=particle_count,
            seed=seed,
            base_dir=base_dir,
            save_animation=save_animation,
        )
    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for the synthetic runner."""
    parser = argparse.ArgumentParser(description="Run synthetic motion data creation and vector-corr analysis")
    parser.add_argument("--mode", choices=["brownian", "contractile", "extensile", "all"], default="all")
    parser.add_argument("--frame-count", type=int, default=DEFAULT_FRAME_COUNT, help="Raw simulated frame count")
    parser.add_argument("--average-window-frames", type=int, default=DEFAULT_AVERAGE_WINDOW_FRAMES, help="Non-overlapping averaging window size")
    parser.add_argument("--particle-count", type=int, default=DEFAULT_PARTICLE_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--dataset-id", type=str, default=None, help="Override the default dataset id")
    parser.add_argument("--no-animation", action="store_true", help="Skip MP4 animation creation")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Run the synthetic workflow from the command line."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    modes = ("brownian", "contractile", "extensile") if args.mode == "all" else (args.mode,)
    outputs: dict[str, dict[str, Any]] = {}
    for mode in modes:
        dataset_id = args.dataset_id or _default_dataset_id(mode, DEFAULT_OUTPUT_SUFFIX)
        outputs[mode] = _run_one(
            mode,
            dataset_id=dataset_id,
            frame_count=int(args.frame_count),
            average_window_frames=int(args.average_window_frames),
            particle_count=int(args.particle_count),
            seed=int(args.seed),
            base_dir=args.base_dir,
            save_animation=not args.no_animation,
        )
    return outputs


if __name__ == "__main__":
    main()
