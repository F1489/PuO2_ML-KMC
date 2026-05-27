"""Presentation-quality crystal structure visualizations for PuO2 XYZ files."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Rectangle
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree

from .analysis import close_contact_thresholds_satisfied, coordination_summary
from .io_xyz import read_xyz


PLOT_COLOR = "#1F2937"
MUTED_COLOR = "#667085"
ACCENT_COLOR = "#B54708"
BOND_COLOR = "#7B8798"
ATOM_COLORS = {"Pu": "#AEB6C2", "O": "#B42318"}
PRESENTATION_ATOM_RADII = {"Pu": 0.54, "O": 0.35}
ZOOM_ATOM_RADII = {"Pu": 0.72, "O": 0.46}
PRESENTATION_CAMERA = (18.0, -54.0)

DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "puo2_density",
    ["#F8FAFC", "#D7E8F7", "#77B7D7", "#2F7DA8", "#08306B"],
)
DEFECT_CMAP = LinearSegmentedColormap.from_list(
    "puo2_defect_density",
    ["#FFFFFF", "#FEE4E2", "#FDA29B", "#F04438", "#7A271A"],
)


def _plot_style() -> dict[str, object]:
    return {
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#ffffff",
        "savefig.facecolor": "#ffffff",
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "axes.unicode_minus": False,
        "axes.edgecolor": "#CBD5E1",
        "axes.labelcolor": PLOT_COLOR,
        "xtick.color": PLOT_COLOR,
        "ytick.color": PLOT_COLOR,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
    }


def _positions_by_atom(atoms: list[str], positions: np.ndarray, atom: str) -> np.ndarray:
    mask = np.asarray([item == atom for item in atoms], dtype=bool)
    return np.asarray(positions, dtype=float)[mask]


def _draw_sphere(ax, center: np.ndarray, radius: float, color: str, resolution: int = 18, alpha: float = 0.94) -> None:
    """Draw one smooth 3D atom sphere with deterministic scientific-style lighting."""
    u = np.linspace(0.0, 2.0 * np.pi, resolution)
    v = np.linspace(0.0, np.pi, resolution)
    nx = np.outer(np.cos(u), np.sin(v))
    ny = np.outer(np.sin(u), np.sin(v))
    nz = np.outer(np.ones_like(u), np.cos(v))
    x = center[0] + radius * nx
    y = center[1] + radius * ny
    z = center[2] + radius * nz

    light = np.asarray([-0.35, -0.55, 0.76], dtype=float)
    light /= np.linalg.norm(light)
    diffuse = np.clip(nx * light[0] + ny * light[1] + nz * light[2], 0.0, 1.0)
    shade = 0.48 + 0.52 * diffuse
    specular = 0.16 * np.power(diffuse, 18)
    rgb = np.asarray(to_rgb(color), dtype=float)
    facecolors = np.empty((*shade.shape, 4), dtype=float)
    facecolors[..., :3] = np.clip(rgb * shade[..., None] + specular[..., None], 0.0, 1.0)
    facecolors[..., 3] = alpha

    ax.plot_surface(
        x,
        y,
        z,
        facecolors=facecolors,
        edgecolor="none",
        linewidth=0.0,
        antialiased=True,
        shade=False,
        zorder=3,
    )


def compute_zoom_center(atoms: list[str], positions: np.ndarray, cutoff_pu_o: float = 3.2) -> np.ndarray:
    """Choose a Pu-centered local environment with the best fluorite-like coordination."""
    pu_positions = _positions_by_atom(atoms, positions, "Pu")
    o_positions = _positions_by_atom(atoms, positions, "O")
    if len(pu_positions) == 0:
        return np.mean(positions, axis=0)
    if len(o_positions) == 0:
        return pu_positions[0]

    tree = cKDTree(o_positions)
    best_center = pu_positions[0]
    best_score = -np.inf
    for pu_position in pu_positions:
        neighbors = o_positions[tree.query_ball_point(pu_position, cutoff_pu_o)]
        count_score = -abs(len(neighbors) - 8)
        compactness = -float(np.std(neighbors - pu_position)) if len(neighbors) else -10.0
        score = 10.0 * count_score + compactness
        if score > best_score:
            best_score = score
            best_center = pu_position
    return np.asarray(best_center, dtype=float)


def atoms_in_sphere(atoms: list[str], positions: np.ndarray, center: np.ndarray, radius: float) -> tuple[list[str], np.ndarray]:
    """Return atoms and coordinates inside a spherical zoom region."""
    positions = np.asarray(positions, dtype=float)
    mask = np.linalg.norm(positions - center, axis=1) <= radius
    return [atom for atom, keep in zip(atoms, mask, strict=True) if keep], positions[mask]


def density_in_sphere(n_atoms: int, radius: float) -> float:
    """Return atom number density for a sphere."""
    volume = 4.0 / 3.0 * np.pi * radius**3
    return float(n_atoms / volume)


def _axis_bounds(positions: np.ndarray, scale: float = 0.55, extra_pad: float = 0.0) -> tuple[np.ndarray, float]:
    positions = np.asarray(positions, dtype=float)
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = scale * float(np.max(maxs - mins)) + extra_pad
    return center, max(radius, 1.0)


def _combined_axis_bounds(*position_sets: np.ndarray, scale: float = 0.55, extra_pad: float = 0.0) -> tuple[np.ndarray, float]:
    return _axis_bounds(np.vstack([np.asarray(item, dtype=float) for item in position_sets]), scale, extra_pad)


def _set_equal_axes(
    ax,
    positions: np.ndarray,
    scale: float = 0.55,
    extra_pad: float = 0.0,
    center_radius: tuple[np.ndarray, float] | None = None,
) -> None:
    if len(positions) == 0:
        return
    center, radius = center_radius if center_radius is not None else _axis_bounds(positions, scale, extra_pad)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _project_3d_point_to_figure(fig, ax, point: np.ndarray) -> tuple[float, float]:
    """Project one 3D data coordinate to figure-fraction coordinates."""
    x2, y2, _ = proj3d.proj_transform(float(point[0]), float(point[1]), float(point[2]), ax.get_proj())
    display_xy = ax.transData.transform((x2, y2))
    figure_xy = fig.transFigure.inverted().transform(display_xy)
    return float(figure_xy[0]), float(figure_xy[1])


def _draw_zoom_box(ax, zoom_center: np.ndarray, zoom_box_size: float) -> None:
    half = zoom_box_size / 2.0
    x0, y0, z0 = zoom_center - half
    x1, y1, z1 = zoom_center + half
    vertices = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ]
    )
    faces = [
        [vertices[index] for index in face]
        for face in ([0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [0, 3, 7, 4])
    ]
    box = Poly3DCollection(faces, facecolor="#FDE68A", edgecolor=ACCENT_COLOR, linewidth=0.85, alpha=0.16)
    ax.add_collection3d(box)


def _draw_presentation_structure(
    ax,
    atoms: list[str],
    positions: np.ndarray,
    *,
    show_bonds: bool,
    zoom_center: np.ndarray | None = None,
    atom_radii: dict[str, float] | None = None,
    zoom_box_size: float | None = None,
    bond_linewidth: float = 0.75,
    bond_alpha: float = 0.48,
    bond_cutoff: float = 2.85,
    axis_scale: float = 0.34,
    axis_extra_pad: float = 0.05,
    center_radius: tuple[np.ndarray, float] | None = None,
    sphere_alpha: float = 0.92,
    sphere_resolution: int = 18,
) -> None:
    """Draw a presentation-ready axis-free ball-and-stick cluster."""
    positions = np.asarray(positions, dtype=float)
    atom_radii = PRESENTATION_ATOM_RADII if atom_radii is None else atom_radii

    if show_bonds:
        pu_positions = _positions_by_atom(atoms, positions, "Pu")
        o_positions = _positions_by_atom(atoms, positions, "O")
        if len(pu_positions) and len(o_positions):
            tree = cKDTree(o_positions)
            for pu_position in pu_positions:
                for neighbor_index in tree.query_ball_point(pu_position, bond_cutoff):
                    o_position = o_positions[neighbor_index]
                    ax.plot(
                        [pu_position[0], o_position[0]],
                        [pu_position[1], o_position[1]],
                        [pu_position[2], o_position[2]],
                        color=BOND_COLOR,
                        linewidth=bond_linewidth,
                        alpha=bond_alpha,
                        zorder=1,
                    )

    for atom in ("Pu", "O"):
        for atom_position in _positions_by_atom(atoms, positions, atom):
            _draw_sphere(
                ax,
                atom_position,
                atom_radii[atom],
                ATOM_COLORS[atom],
                resolution=sphere_resolution,
                alpha=sphere_alpha,
            )

    if zoom_center is not None and zoom_box_size is not None:
        _draw_zoom_box(ax, zoom_center, zoom_box_size)

    ax.set_proj_type("ortho")
    ax.view_init(elev=PRESENTATION_CAMERA[0], azim=PRESENTATION_CAMERA[1])
    _set_equal_axes(ax, positions, scale=axis_scale, extra_pad=axis_extra_pad, center_radius=center_radius)
    ax.set_axis_off()


def _atom_density_in_sphere(atom_count: int, radius: float) -> float:
    volume = (4.0 / 3.0) * np.pi * radius**3
    return atom_count / volume


def _metrics_values(
    atoms: list[str],
    positions: np.ndarray,
    energy_per_puo2: float | None = None,
    zoom_atom_count: int | None = None,
    zoom_radius: float | None = None,
) -> dict[str, str]:
    summary = coordination_summary(atoms, positions)
    safety = "да" if close_contact_thresholds_satisfied(atoms, positions) else "нет"
    energy = "н/д" if energy_per_puo2 is None else f"{energy_per_puo2:.4f}"
    values = {
        "E/PuO2, эВ": energy,
        "bulk order": f"{summary['bulk_fluorite_order_score']:.4f}",
        "coord. error": f"{summary['mean_abs_coordination_error']:.4f}",
        "Pu(8O)": f"{100.0 * summary['fraction_pu_with_8_o']:.1f} %",
        "safe contacts": safety,
    }
    if zoom_atom_count is not None and zoom_radius is not None:
        values["плотн. zoom, atom/A^3"] = f"{_atom_density_in_sphere(zoom_atom_count, zoom_radius):.4f}"
    return values


def _add_metric_strip(fig, values: dict[str, str], y: float = 0.060) -> None:
    x_positions = np.linspace(0.13, 0.87, len(values))
    for x, (label, value) in zip(x_positions, values.items(), strict=True):
        fig.text(x, y + 0.026, label, ha="center", va="bottom", fontsize=8.8, color=MUTED_COLOR)
        fig.text(x, y, value, ha="center", va="top", fontsize=13.2, fontweight="bold", color=PLOT_COLOR)


def _add_atom_legend_strip(fig, y: float = 0.925) -> None:
    handles = [
        Line2D([], [], marker="o", linestyle="None", markersize=8, markerfacecolor=ATOM_COLORS["Pu"], markeredgecolor="#6B7280", label="Pu"),
        Line2D([], [], marker="o", linestyle="None", markersize=7, markerfacecolor=ATOM_COLORS["O"], markeredgecolor="#7A271A", label="O"),
        Line2D([], [], color=BOND_COLOR, linewidth=1.3, alpha=0.75, label="связь Pu-O"),
        Line2D([], [], marker="s", linestyle="None", markersize=7, markerfacecolor="#FDE68A", markeredgecolor=ACCENT_COLOR, label="zoom"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.56, y),
        ncol=4,
        frameon=False,
        fontsize=10.0,
        handlelength=1.4,
        handletextpad=0.45,
        columnspacing=1.4,
    )


def _save_figure_outputs(fig, output_png: Path, save_pdf: bool = True, dpi: int = 260) -> None:
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight", pad_inches=0.04, facecolor="#ffffff")
    if save_pdf:
        fig.savefig(output_png.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04, facecolor="#ffffff")


def plot_configuration(
    xyz: str | Path,
    output_png: str | Path,
    title: str,
    metrics_label: str,
    zoom_center: np.ndarray,
    zoom_radius: float = 6.0,
    energy_per_puo2: float | None = None,
    camera: tuple[float, float] = PRESENTATION_CAMERA,
    save_pdf: bool = True,
) -> Path:
    """Save one presentation-style crystal configuration with inset and compact metrics."""
    atoms, positions, _ = read_xyz(xyz)
    zoom_atoms, zoom_positions = atoms_in_sphere(atoms, positions, zoom_center, zoom_radius)
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(_plot_style()):
        fig = plt.figure(figsize=(13.6, 7.65), dpi=220, facecolor="white")
        ax_main = fig.add_axes([0.020, 0.135, 0.655, 0.745], projection="3d")
        ax_zoom = fig.add_axes([0.690, 0.170, 0.305, 0.565], projection="3d")

        _draw_presentation_structure(
            ax_main,
            atoms,
            positions,
            show_bonds=True,
            zoom_center=zoom_center,
            zoom_box_size=4.6,
            bond_linewidth=0.82,
            bond_alpha=0.48,
            axis_scale=0.305,
            axis_extra_pad=0.02,
            sphere_alpha=0.91,
        )
        _draw_presentation_structure(
            ax_zoom,
            zoom_atoms,
            zoom_positions,
            show_bonds=True,
            atom_radii=ZOOM_ATOM_RADII,
            bond_linewidth=1.05,
            bond_alpha=0.62,
            bond_cutoff=3.05,
            axis_scale=0.300,
            axis_extra_pad=0.04,
            sphere_alpha=0.95,
            sphere_resolution=26,
        )
        ax_main.view_init(elev=float(camera[0]), azim=float(camera[1]))
        ax_zoom.view_init(elev=float(camera[0]), azim=float(camera[1]))

        fig.text(0.045, 0.945, title, fontsize=21, fontweight="bold", color=PLOT_COLOR, ha="left", va="top")
        fig.text(0.047, 0.900, metrics_label, fontsize=10.8, color=MUTED_COLOR, ha="left", va="top")
        _add_atom_legend_strip(fig, y=0.932)
        _add_metric_strip(fig, _metrics_values(atoms, positions, energy_per_puo2, len(zoom_atoms), zoom_radius), y=0.055)

        fig.text(0.704, 0.760, f"Увеличенный фрагмент, R = {zoom_radius:.1f} Å", fontsize=11.5, color=PLOT_COLOR, ha="left", va="center")
        fig.add_artist(Rectangle((0.680, 0.145), 0.318, 0.605, transform=fig.transFigure, fill=False, lw=1.05, ec="#475467"))
        fig.canvas.draw()
        zoom_target = _project_3d_point_to_figure(fig, ax_main, zoom_center)
        fig.add_artist(
            ConnectionPatch(
                xyA=(0.680, 0.500),
                coordsA=fig.transFigure,
                xyB=zoom_target,
                coordsB=fig.transFigure,
                color=ACCENT_COLOR,
                linewidth=1.15,
                arrowstyle="->",
                mutation_scale=14,
                shrinkA=2,
                shrinkB=3,
            )
        )

        _save_figure_outputs(fig, output_png, save_pdf=save_pdf, dpi=260)
        plt.close(fig)
    return output_png


def plot_single_crystal_view(
    xyz: str | Path,
    output_png: str | Path,
    title: str,
    metrics_label: str,
    zoom_center: np.ndarray,
    zoom_radius: float = 8.0,
    energy_per_puo2: float | None = None,
) -> Path:
    """Backward-compatible wrapper for the presentation configuration view."""
    return plot_configuration(
        xyz=xyz,
        output_png=output_png,
        title=title,
        metrics_label=metrics_label,
        zoom_center=zoom_center,
        zoom_radius=zoom_radius,
        energy_per_puo2=energy_per_puo2,
    )


def plot_final_presentation_view(
    xyz: str | Path,
    output_png: str | Path,
    title: str,
    metrics_label: str,
    zoom_center: np.ndarray,
    zoom_radius: float = 6.0,
    energy_per_puo2: float | None = None,
) -> Path:
    """Backward-compatible wrapper around plot_configuration()."""
    return plot_configuration(
        xyz=xyz,
        output_png=output_png,
        title=title,
        metrics_label=metrics_label,
        zoom_center=zoom_center,
        zoom_radius=zoom_radius,
        energy_per_puo2=energy_per_puo2,
    )


def plot_comparison_panel(
    initial_xyz: str | Path,
    final_xyz: str | Path,
    output_png: str | Path,
    title: str = "Сравнение начальной и конечной структуры",
    initial_energy_per_puo2: float | None = None,
    final_energy_per_puo2: float | None = None,
) -> Path:
    """Build a clean two-panel comparison with shared camera and scale."""
    atoms_initial, positions_initial, _ = read_xyz(initial_xyz)
    atoms_final, positions_final, _ = read_xyz(final_xyz)
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    shared_bounds = _combined_axis_bounds(positions_initial, positions_final, scale=0.37, extra_pad=0.05)

    initial_summary = coordination_summary(atoms_initial, positions_initial)
    final_summary = coordination_summary(atoms_final, positions_final)
    delta_energy = (
        "н/д"
        if initial_energy_per_puo2 is None or final_energy_per_puo2 is None
        else f"{final_energy_per_puo2 - initial_energy_per_puo2:+.4f} эВ"
    )
    metric_line = (
        f"ΔE/PuO2 = {delta_energy}    "
        f"bulk order: {initial_summary['bulk_fluorite_order_score']:.4f} → {final_summary['bulk_fluorite_order_score']:.4f}    "
        f"coord. error: {initial_summary['mean_abs_coordination_error']:.4f} → {final_summary['mean_abs_coordination_error']:.4f}"
    )

    with plt.rc_context(_plot_style()):
        fig = plt.figure(figsize=(13.6, 7.65), dpi=220, facecolor="white")
        ax_left = fig.add_axes([0.020, 0.130, 0.465, 0.735], projection="3d")
        ax_right = fig.add_axes([0.515, 0.130, 0.465, 0.735], projection="3d")

        for ax, atoms, positions in (
            (ax_left, atoms_initial, positions_initial),
            (ax_right, atoms_final, positions_final),
        ):
            _draw_presentation_structure(
                ax,
                atoms,
                positions,
                show_bonds=True,
                center_radius=shared_bounds,
                bond_linewidth=0.82,
                bond_alpha=0.48,
                sphere_alpha=0.91,
            )

        fig.text(0.045, 0.945, title, fontsize=20, fontweight="bold", color=PLOT_COLOR, ha="left", va="top")
        fig.text(0.045, 0.900, metric_line, fontsize=11.0, color=MUTED_COLOR, ha="left", va="top")
        fig.text(0.055, 0.855, "A", fontsize=16, fontweight="bold", color=PLOT_COLOR, ha="left", va="top")
        fig.text(0.085, 0.855, "Начальная структура", fontsize=13, color=PLOT_COLOR, ha="left", va="top")
        fig.text(0.550, 0.855, "B", fontsize=16, fontweight="bold", color=PLOT_COLOR, ha="left", va="top")
        fig.text(0.580, 0.855, "После ML-kMC", fontsize=13, color=PLOT_COLOR, ha="left", va="top")
        _add_atom_legend_strip(fig, y=0.080)

        _save_figure_outputs(fig, output_png, save_pdf=True, dpi=260)
        plt.close(fig)
    return output_png


def _coordination_by_atom(atoms: list[str], positions: np.ndarray, cutoff_pu_o: float) -> tuple[np.ndarray, np.ndarray]:
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


def _defect_arrays(atoms: list[str], positions: np.ndarray, cutoff_pu_o: float) -> tuple[np.ndarray, np.ndarray]:
    coordination, ideal = _coordination_by_atom(atoms, positions, cutoff_pu_o)
    abs_error = np.abs(coordination - ideal)
    return abs_error > 0, abs_error


def _set_equal_3d_limits(ax, positions: np.ndarray) -> None:
    positions = np.asarray(positions, dtype=float)
    center = np.mean(positions, axis=0)
    radius = 0.5 * float(np.ptp(positions, axis=0).max())
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def plot_defect_cube_map(
    xyz: str | Path,
    output_png: str | Path,
    *,
    grid_size: int = 12,
    cutoff_pu_o: float = 3.2,
    min_count_to_draw: float = 1.0,
    title: str = "Карта дефектов координации",
    subtitle: str = "Пространственное распределение дефектов PuO2",
    save_pdf: bool = True,
) -> Path:
    """Save a 3D cube map of coordination defects."""
    atoms, positions, _ = read_xyz(xyz)
    positions = np.asarray(positions, dtype=float)
    defect_mask, abs_error = _defect_arrays(atoms, positions, cutoff_pu_o)
    defect_positions = positions[defect_mask]
    edges = [
        np.linspace(positions[:, axis].min(), positions[:, axis].max(), grid_size + 1)
        for axis in range(3)
    ]
    counts, _ = np.histogramdd(defect_positions, bins=edges)
    max_count = float(counts.max()) if counts.size else 0.0
    norm = Normalize(vmin=0.0, vmax=max(1.0, max_count))

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(_plot_style()):
        fig = plt.figure(figsize=(13.6, 7.65), dpi=220, facecolor="white")
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=23.0, azim=-48.0)

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
                    ax.bar3d(
                        x,
                        y,
                        z,
                        dx,
                        dy,
                        dz,
                        color=DEFECT_CMAP(norm(count)),
                        edgecolor=BOND_COLOR,
                        linewidth=0.16,
                        alpha=0.03 + 0.52 * norm(count),
                        shade=False,
                    )

        atom_array = np.asarray(atoms)
        total_defects = int(defect_mask.sum())
        pu_defects = int(np.sum(defect_mask & (atom_array == "Pu")))
        o_defects = int(np.sum(defect_mask & (atom_array == "O")))
        mean_abs_error = float(np.mean(abs_error)) if len(abs_error) else 0.0

        fig.text(0.045, 0.948, title, ha="left", va="top", fontsize=24, fontweight="bold", color=PLOT_COLOR)
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
            fig.text(x, 0.050, value, ha="center", va="top", fontsize=17.0, fontweight="bold", color=PLOT_COLOR)

        mappable = plt.cm.ScalarMappable(norm=norm, cmap=DEFECT_CMAP)
        mappable.set_array(counts)
        colorbar = fig.colorbar(mappable, ax=ax, pad=0.006, shrink=0.58)
        colorbar.set_label("число дефектов в кубике", fontsize=10, color=PLOT_COLOR)
        colorbar.ax.tick_params(labelsize=9, colors=PLOT_COLOR)

        _set_equal_3d_limits(ax, positions)
        ax.set_axis_off()
        ax.set_position([-0.030, 0.085, 0.900, 0.800])
        fig.subplots_adjust(left=0.00, right=0.965, bottom=0.105, top=0.895)
        _save_figure_outputs(fig, output_png, save_pdf=save_pdf, dpi=260)
        plt.close(fig)
    return output_png


def plot_density_heatmap(
    xyz: str | Path,
    output_png: str | Path,
    grid_size: int = 12,
    title: str = "2D-проекция плотности финальной структуры PuO2",
    save_pdf: bool = True,
) -> Path:
    """Save an x-y atom-density projection integrated over z."""
    atoms, positions, _ = read_xyz(xyz)
    positions = np.asarray(positions, dtype=float)
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    x_edges = np.linspace(positions[:, 0].min(), positions[:, 0].max(), grid_size + 1)
    y_edges = np.linspace(positions[:, 1].min(), positions[:, 1].max(), grid_size + 1)
    counts, _, _ = np.histogram2d(positions[:, 0], positions[:, 1], bins=[x_edges, y_edges])
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

    with plt.rc_context(_plot_style()):
        fig, ax = plt.subplots(figsize=(13.6, 7.65), dpi=220)
        image = ax.imshow(
            counts.T,
            origin="lower",
            extent=extent,
            cmap=DENSITY_CMAP,
            interpolation="nearest",
            aspect="equal",
        )
        ax.contour(
            0.5 * (x_edges[:-1] + x_edges[1:]),
            0.5 * (y_edges[:-1] + y_edges[1:]),
            counts.T,
            levels=4,
            colors="#0F172A",
            linewidths=0.45,
            alpha=0.30,
        )

        pu_positions = _positions_by_atom(atoms, positions, "Pu")
        o_positions = _positions_by_atom(atoms, positions, "O")
        ax.scatter(pu_positions[:, 0], pu_positions[:, 1], s=12, c=ATOM_COLORS["Pu"], edgecolors="white", linewidths=0.25, alpha=0.72, label="Pu")
        ax.scatter(o_positions[:, 0], o_positions[:, 1], s=7, c=ATOM_COLORS["O"], edgecolors="white", linewidths=0.15, alpha=0.55, label="O")

        ax.set_title(title, loc="left", fontsize=20, fontweight="bold", color=PLOT_COLOR, pad=14)
        ax.set_xlabel("x, Å", fontsize=11)
        ax.set_ylabel("y, Å", fontsize=11)
        ax.grid(color="#FFFFFF", linewidth=0.8, alpha=0.55)
        for spine in ax.spines.values():
            spine.set_color("#94A3B8")

        colorbar = fig.colorbar(image, ax=ax, pad=0.018, shrink=0.86)
        colorbar.set_label("Число атомов в ячейке", fontsize=11, color=PLOT_COLOR)
        colorbar.ax.tick_params(labelsize=9, colors=PLOT_COLOR)
        ax.legend(loc="upper right", frameon=True, framealpha=0.92, facecolor="white", edgecolor="#CBD5E1", fontsize=10)
        fig.text(0.125, 0.075, "Плотность интегрирована вдоль оси z; точки показывают проекцию атомов Pu и O.", fontsize=9.8, color=MUTED_COLOR)
        fig.tight_layout(rect=(0.03, 0.08, 0.98, 0.96))
        _save_figure_outputs(fig, output_png, save_pdf=save_pdf, dpi=260)
        plt.close(fig)
    return output_png


def main() -> None:
    parser = argparse.ArgumentParser(description="Create PuO2 crystal visualization figures.")
    parser.add_argument("--initial-xyz", required=True)
    parser.add_argument("--final-xyz", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--zoom-radius", type=float, default=6.0)
    parser.add_argument("--grid-size", type=int, default=12)
    parser.add_argument("--initial-energy-per-puo2", type=float, default=None)
    parser.add_argument("--final-energy-per-puo2", type=float, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.initial_xyz, out_dir / "initial_crystal.xyz")
    shutil.copyfile(args.final_xyz, out_dir / "final_crystal.xyz")

    atoms_final, positions_final, _ = read_xyz(args.final_xyz)
    zoom_center = compute_zoom_center(atoms_final, positions_final)
    plot_final_presentation_view(
        args.initial_xyz,
        out_dir / "initial_crystal_visualization.png",
        title="Начальная конфигурация",
        metrics_label="Исходная структура PuO2",
        zoom_center=zoom_center,
        zoom_radius=args.zoom_radius,
        energy_per_puo2=args.initial_energy_per_puo2,
    )
    plot_final_presentation_view(
        args.final_xyz,
        out_dir / "final_crystal_visualization.png",
        title="Конечная конфигурация",
        metrics_label="Структура после двухстадийного seeded ML-kMC",
        zoom_center=zoom_center,
        zoom_radius=args.zoom_radius,
        energy_per_puo2=args.final_energy_per_puo2,
    )
    plot_density_heatmap(
        args.final_xyz,
        out_dir / "final_density_heatmap.png",
        grid_size=args.grid_size,
    )
    plot_defect_cube_map(
        args.initial_xyz,
        out_dir / "initial_defect_map.png",
        grid_size=args.grid_size,
        title="Начальная конфигурация",
        subtitle="Карта дефектов координации исходной структуры PuO2",
    )
    plot_defect_cube_map(
        args.final_xyz,
        out_dir / "final_defect_map.png",
        grid_size=args.grid_size,
        title="Конечная конфигурация",
        subtitle="Карта дефектов координации после двухстадийного seeded ML-kMC",
    )
    plot_comparison_panel(
        args.initial_xyz,
        args.final_xyz,
        out_dir / "initial_final_comparison.png",
        initial_energy_per_puo2=args.initial_energy_per_puo2,
        final_energy_per_puo2=args.final_energy_per_puo2,
    )
    print(f"Saved crystal visualizations to {out_dir}")


if __name__ == "__main__":
    main()
