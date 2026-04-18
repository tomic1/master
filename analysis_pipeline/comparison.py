from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_hex


ATP_COMPARISON_COLORS = (
    "#08306b",
    "#08519c",
    "#2171b5",
    "#6baed6",
)

PRC_COMPARISON_COLORS = (
    "#fff7bc",
    "#fec44f",
    "#fd8d3c",
    "#e31a1c",
)


@dataclass(frozen=True)
class ComparisonSpec:
    dataset_id: str
    label: str
    color: str
    group: str = ""
    replicate: str = ""
    variation: str = ""
    base_dir: str | None = None
    raw_base_dir: str | None = None


def _resample_palette(anchors: Sequence[str], n: int) -> list[str]:
    if n <= 0:
        return []
    if n <= len(anchors):
        return list(anchors[:n])
    cmap = LinearSegmentedColormap.from_list("comparison_palette", list(anchors))
    samples = np.linspace(0.0, 1.0, n)
    return [to_hex(cmap(value)) for value in samples]


def comparison_palette(kind: str, n: int) -> list[str]:
    kind_norm = str(kind).strip().lower()
    if kind_norm in {"atp", "blue", "b"}:
        return _resample_palette(ATP_COMPARISON_COLORS, n)
    if kind_norm in {"prc", "green", "g"}:
        cmap = cm.get_cmap("YlOrRd")
        if n <= 0:
            return []
        if n == 1:
            return [to_hex(cmap(0.55))]
        samples = np.linspace(0.25, 0.9, n)
        return [to_hex(cmap(value)) for value in samples]
    cmap = cm.get_cmap("tab10")
    if n <= 0:
        return []
    if n == 1:
        return [to_hex(cmap(0.0))]
    samples = np.linspace(0.0, 1.0, n)
    return [to_hex(cmap(value)) for value in samples]


def comparison_legend_kwargs(fig) -> dict[str, float]:
    width, height = fig.get_size_inches()
    base = min(float(width), float(height))
    fontsize = max(10, min(16, int(round(base * 1.4))))
    return {
        "fontsize": fontsize,
        "title_fontsize": fontsize,
        "handlelength": 2.2,
        "labelspacing": 0.6,
        "borderpad": 0.8,
        "handletextpad": 0.7,
        "columnspacing": 1.2,
    }


def comparison_specs_from_config(comparison_cfg: dict[str, Any]) -> list[ComparisonSpec]:
    """Build comparison specs from the comparison section of the analysis config.

    The config accepts either a flat list of dataset entries or grouped entries with a
    ``datasets`` field. Grouped entries inherit the group palette and variation unless a
    child dataset overrides them.
    """
    return comparison_registry_from_config(comparison_cfg)


def comparison_registry_from_config(comparison_cfg: dict[str, Any]) -> list[ComparisonSpec]:
    if not comparison_cfg.get("enabled", False):
        return []

    default_palette = str(comparison_cfg.get("palette", "atp"))
    default_variation = str(comparison_cfg.get("variation", "")).strip()
    registry = comparison_cfg.get("registry")
    if registry:
        return build_comparison_specs(registry, palette=default_palette)

    groups = list(comparison_cfg.get("groups", []))
    if not groups:
        datasets = comparison_cfg.get("datasets")
        if datasets:
            return build_comparison_specs(datasets, palette=default_palette)
        return []

    flat_entries: list[dict[str, Any] | Sequence[Any]] = []

    for group in groups:
        if not isinstance(group, dict):
            flat_entries.append(group)
            continue

        datasets = group.get("datasets")
        if datasets is None:
            flat_entries.append(group)
            continue

        group_name = str(group.get("name", group.get("group", ""))).strip()
        group_palette = str(group.get("palette", default_palette)).strip() or default_palette
        group_variation = str(group.get("variation", default_variation)).strip()
        group_base_dir = group.get("base_dir")
        group_replicate = str(group.get("replicate", "")).strip()

        specs = build_comparison_specs(datasets, palette=group_palette)
        for spec in specs:
            flat_entries.append(
                {
                    "dataset_id": spec.dataset_id,
                    "label": spec.label,
                    "color": spec.color,
                    "group": group_name or spec.group,
                    "replicate": spec.replicate or group_replicate,
                    "variation": spec.variation or group_variation,
                    "base_dir": spec.base_dir if spec.base_dir is not None else group_base_dir,
                }
            )

    return build_comparison_specs(flat_entries, palette=default_palette)


def build_comparison_specs(
    datasets: Iterable[dict[str, Any] | Sequence[Any]],
    *,
    palette: str = "atp",
) -> list[ComparisonSpec]:
    entries = list(datasets)
    colors = comparison_palette(palette, len(entries))
    specs: list[ComparisonSpec] = []
    for idx, entry in enumerate(entries):
        if isinstance(entry, dict):
            dataset_id = str(entry.get("dataset_id", "")).strip()
            if not dataset_id:
                raise ValueError("comparison dataset entries must define dataset_id")
            label = str(entry.get("label", dataset_id)).strip() or dataset_id
            color = str(entry.get("color", colors[idx]))
            specs.append(
                ComparisonSpec(
                    dataset_id=dataset_id,
                    label=label,
                    color=color,
                    group=str(entry.get("group", "")).strip(),
                    replicate=str(entry.get("replicate", "")).strip(),
                    variation=str(entry.get("variation", "")).strip(),
                    base_dir=entry.get("base_dir"),
                    raw_base_dir=entry.get("raw_base_dir"),
                )
            )
            continue

        if len(entry) < 2:
            raise ValueError("comparison dataset tuples must contain at least dataset_id and label")

        dataset_id = str(entry[0]).strip()
        label = str(entry[1]).strip() or dataset_id
        color = str(entry[2]).strip() if len(entry) >= 3 and entry[2] else colors[idx]
        group = str(entry[3]).strip() if len(entry) >= 4 and entry[3] else ""
        variation = str(entry[4]).strip() if len(entry) >= 5 and entry[4] else ""
        base_dir = entry[5] if len(entry) >= 6 else None
        raw_base_dir = entry[6] if len(entry) >= 7 else None

        if not dataset_id:
            raise ValueError("comparison dataset tuples must define a dataset_id")

        specs.append(
            ComparisonSpec(
                dataset_id=dataset_id,
                label=label,
                color=color,
                group=group,
                replicate="",
                variation=variation,
                base_dir=base_dir,
                raw_base_dir=raw_base_dir,
            )
        )

    return specs


def save_comparison_dual_pdf(
    fig,
    output_dir: str | Path,
    stem: str,
    *,
    white_plot_fn: Callable[[Any], None] | None = None,
    dpi: int = 150,
) -> dict[str, Path]:
    """Save a comparison figure as matching black and white PDFs.

    The provided figure is written as the black presentation version. If ``white_plot_fn``
    is given, it is used to redraw the figure on a white background and that version is
    saved without a title.
    """
    paths = comparison_export_paths(output_dir, stem)
    fig.savefig(str(paths["black"]), dpi=dpi, bbox_inches="tight")

    if white_plot_fn is None:
        return paths

    with comparison_style_context("white"):
        fig_white, ax_white = plt.subplots(figsize=fig.get_size_inches(), dpi=dpi)
        white_plot_fn(ax_white)
        ax_white.set_title("")
        fig_white.savefig(str(paths["white"]), dpi=dpi, bbox_inches="tight")
        plt.close(fig_white)

    return paths


def comparison_style_context(background: str = "dark") -> Any:
    background_norm = str(background).strip().lower()
    base_style: list[Any] = ["./science.mplstyle.txt"]
    if background_norm == "white":
        base_style.append(
            {
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "savefig.facecolor": "white",
                "text.color": "black",
                "axes.labelcolor": "black",
                "axes.edgecolor": "black",
                "xtick.color": "black",
                "ytick.color": "black",
                "grid.color": "0.8",
                "legend.edgecolor": "0.5",
            }
        )
    else:
        base_style.append(
            {
                "figure.facecolor": "black",
                "axes.facecolor": "black",
                "savefig.facecolor": "black",
                "text.color": "white",
                "axes.labelcolor": "white",
                "axes.edgecolor": "white",
                "xtick.color": "white",
                "ytick.color": "white",
                "grid.color": "0.4",
                "legend.edgecolor": "0.7",
            }
        )
    return plt.style.context(base_style)


def comparison_output_dir(root: str | Path, group: str, variation: str = "") -> Path:
    root_path = Path(root)
    out_dir = root_path / group
    if variation:
        out_dir = out_dir / variation
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def comparison_export_paths(output_dir: str | Path, stem: str) -> dict[str, Path]:
    output_path = Path(output_dir)
    return {
        "black": output_path / f"{stem}_black.pdf",
        "white": output_path / f"{stem}_white.pdf",
    }


def shared_axis_limit(curves: Iterable[Sequence[Any]], *, axis: int = 0) -> float | None:
    finite_limits: list[float] = []
    for curve in curves:
        if len(curve) <= axis:
            continue
        values = np.asarray(curve[axis], dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        finite_limits.append(float(np.nanmax(values)))
    if not finite_limits:
        return None
    return float(np.nanmin(finite_limits))


def apply_common_limits(
    ax,
    x_values: Sequence[float],
    y_values: Sequence[float],
    *,
    x_pad_frac: float = 0.05,
    y_pad_frac: float = 0.08,
) -> None:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if not np.any(valid):
        return
    x = x[valid]
    y = y[valid]
    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    y_min = float(np.nanmin(y))
    y_max = float(np.nanmax(y))
    if x_max > x_min:
        x_pad = x_pad_frac * (x_max - x_min)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
    if y_max > y_min:
        y_pad = y_pad_frac * (y_max - y_min)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
