"""Draft renderer for PuO2 coordination-defect maps.

The script is based on the cube-density idea from 11.py, but it counts only
coordination defects: Pu atoms without 8 O neighbors and O atoms without 4 Pu
neighbors inside the Pu-O cutoff.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.spatial import cKDTree

from src.io_xyz import read_xyz


DEFECT_COLOR = "#D92D20"
TEXT_COLOR = "#1F2937"
MUTED_COLOR = "#667085"
GRID_EDGE_COLOR = "#7B8798"

DEFECT_CMAP = LinearSegmentedColormap.from_list(
    "defect_density",
    [
        (0.00, "#FFFFFF"),
        (0.15, "#FEE4E2"),
        (0.45, "#FDA29B"),
        (0.75, "#F04438"),
        (1.00, "#7A271A"),
    ],
)


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / path


def coordination_by_atom(atoms: list[str], positions: np.ndarray, cutoff_pu_o: float) -> tuple[np.ndarray, np.ndarray]:
    """Return local Pu-O coordination and ideal coordination for every atom."""
    positions = np.asarray(positions, dtype=float)
    tree = cKDTree(positions)
    coordination = np.zeros(len(atoms), dtype=int)
    ideal = np.zeros(len(atoms), dtype=int)

    for i, atom in enumerate(atoms):
        neighbors = [j for j in tree.query_ball_point(positions[i], r=cutoff_pu_o) if j != i]
        if atom == "Pu":
            coordination[i] = sum(atoms[j] == "O" for j in neighbors)
            ideal[i] = 8
        elif atom == "O":
            coordination[i] = sum(atoms[j] == "Pu" for j in neighbors)
            ideal[i] = 4

    return coordination, ideal


def defect_arrays(atoms: list[str], positions: np.ndarray, cutoff_pu_o: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return defect mask, signed coordination error, and absolute error."""
    coordination, ideal = coordination_by_atom(atoms, positions, cutoff_pu_o)
    signed_error = coordination - ideal
    abs_error = np.abs(signed_error)
    return abs_error > 0, signed_error, abs_error


def binned_defect_counts(
    positions: np.ndarray,
    defect_mask: np.ndarray,
    grid_size: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    positions = np.asarray(positions, dtype=float)
    defect_positions = positions[defect_mask]
    edges = [
        np.linspace(positions[:, axis].min(), positions[:, axis].max(), grid_size + 1)
        for axis in range(3)
    ]
    counts, _ = np.histogramdd(defect_positions, bins=edges)
    return counts, edges


def set_axes_equal(ax) -> None:
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    ranges = np.array([abs(x_limits[1] - x_limits[0]), abs(y_limits[1] - y_limits[0]), abs(z_limits[1] - z_limits[0])])
    centers = np.array([np.mean(x_limits), np.mean(y_limits), np.mean(z_limits)])
    radius = 0.5 * ranges.max()
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])


def plot_defect_cube_map(
    xyz: Path,
    output_png: Path,
    *,
    grid_size: int = 12,
    cutoff_pu_o: float = 3.2,
    min_count_to_draw: float = 1.0,
    view_elev: float = 23.0,
    view_azim: float = -48.0,
    title: str = "Карта дефектов координации",
    subtitle: str = "Пространственное распределение дефектов PuO2",
) -> Path:
    atoms, positions, _ = read_xyz(xyz)
    positions = np.asarray(positions, dtype=float)
    defect_mask, signed_error, abs_error = defect_arrays(atoms, positions, cutoff_pu_o)
    counts, edges = binned_defect_counts(positions, defect_mask, grid_size)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    max_count = float(counts.max()) if counts.size else 0.0
    norm = Normalize(vmin=0.0, vmax=max(1.0, max_count))

    fig = plt.figure(figsize=(13.6, 7.65), dpi=220, facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=view_elev, azim=view_azim)

    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                count = counts[i, j, k]
                if count < min_count_to_draw:
                    continue
                x, y, z = edges[0][i], edges[1][j], edges[2][k]
                dx = edges[0][i + 1] - edges[0][i]
                dy = edges[1][j + 1] - edges[1][j]
                dz = edges[2][k + 1] - edges[2][k]
                color = DEFECT_CMAP(norm(count))
                alpha = 0.03 + 0.52 * norm(count)
                ax.bar3d(
                    x,
                    y,
                    z,
                    dx,
                    dy,
                    dz,
                    color=color,
                    edgecolor=GRID_EDGE_COLOR,
                    linewidth=0.16,
                    alpha=alpha,
                    shade=False,
                )

    total_defects = int(defect_mask.sum())
    pu_defects = int(np.sum(defect_mask & (np.asarray(atoms) == "Pu")))
    o_defects = int(np.sum(defect_mask & (np.asarray(atoms) == "O")))
    mean_abs_error = float(np.mean(abs_error)) if len(abs_error) else 0.0

    fig.text(0.045, 0.948, title, ha="left", va="top", fontsize=24, fontweight="bold", color=TEXT_COLOR)
    fig.text(0.047, 0.902, subtitle, ha="left", va="top", fontsize=13, color=MUTED_COLOR)
    fig.text(
        0.047,
        0.862,
        f"Сетка {grid_size}x{grid_size}x{grid_size}; кубик показывает зону структуры; цвет = число дефектных атомов в зоне; порог Pu-O = {cutoff_pu_o:.2f} A",
        ha="left",
        va="top",
        fontsize=10.5,
        color=MUTED_COLOR,
    )
    metric_items = [
        ("всего дефектов", str(total_defects)),
        ("дефекты Pu", str(pu_defects)),
        ("дефекты O", str(o_defects)),
        ("ср. ошибка коорд.", f"{mean_abs_error:.3f}"),
        ("макс. дефектов/кубик", f"{max_count:.0f}"),
    ]
    for x, (label, value) in zip(np.linspace(0.13, 0.87, len(metric_items)), metric_items, strict=True):
        fig.text(x, 0.080, label, ha="center", va="bottom", fontsize=10.0, color=MUTED_COLOR)
        fig.text(x, 0.050, value, ha="center", va="top", fontsize=17.0, fontweight="bold", color=TEXT_COLOR)

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=DEFECT_CMAP)
    mappable.set_array(counts)
    colorbar = fig.colorbar(mappable, ax=ax, pad=0.006, shrink=0.58)
    colorbar.set_label("число дефектов в кубике", fontsize=10, color=TEXT_COLOR)
    colorbar.ax.tick_params(labelsize=9, colors=TEXT_COLOR)

    set_axes_equal(ax)
    ax.set_axis_off()
    ax.set_position([-0.030, 0.085, 0.900, 0.800])
    fig.subplots_adjust(left=0.00, right=0.965, bottom=0.105, top=0.895)
    fig.savefig(output_png, dpi=220, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)
    return output_png


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a 3D PuO2 coordination-defect map.")
    parser.add_argument("--initial-xyz", default="input/PuO2_324.xyz")
    parser.add_argument("--final-xyz", default="results/06_seeded_stage2_polish_1000K/final.xyz")
    parser.add_argument("--out-dir", default="results/07_crystal_visualization_draft")
    parser.add_argument("--xyz", default=None, help="Optional single XYZ file; if set, only one map is rendered.")
    parser.add_argument("--output", default=None, help="Output PNG for --xyz mode.")
    parser.add_argument("--grid-size", type=int, default=12)
    parser.add_argument("--cutoff-pu-o", type=float, default=3.2)
    parser.add_argument("--min-count-to-draw", type=float, default=1.0)
    args = parser.parse_args()

    if args.xyz is not None:
        output = plot_defect_cube_map(
            resolve_project_path(args.xyz),
            resolve_project_path(args.output or "results/07_crystal_visualization_draft/draft_defect_map.png"),
            grid_size=args.grid_size,
            cutoff_pu_o=args.cutoff_pu_o,
            min_count_to_draw=args.min_count_to_draw,
        )
        print(f"Saved defect map to {output}")
        return

    out_dir = resolve_project_path(args.out_dir)
    outputs = [
        plot_defect_cube_map(
            resolve_project_path(args.initial_xyz),
            out_dir / "draft_initial_defect_map.png",
            grid_size=args.grid_size,
            cutoff_pu_o=args.cutoff_pu_o,
            min_count_to_draw=args.min_count_to_draw,
            title="Начальная конфигурация",
            subtitle="Карта дефектов координации исходной структуры PuO2",
        ),
        plot_defect_cube_map(
            resolve_project_path(args.final_xyz),
            out_dir / "draft_final_defect_map.png",
            grid_size=args.grid_size,
            cutoff_pu_o=args.cutoff_pu_o,
            min_count_to_draw=args.min_count_to_draw,
            title="Конечная конфигурация",
            subtitle="Карта дефектов координации после двухстадийного seeded ML-kMC",
        ),
    ]
    for output in outputs:
        print(f"Saved defect map to {output}")


if __name__ == "__main__":
    main()
