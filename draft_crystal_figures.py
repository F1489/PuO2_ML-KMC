"""Fast standalone draft renderer for PuO2 presentation figures.

This file is intentionally separate from ``src/crystal_visualization.py`` so the
visual style can be iterated quickly before the final code is moved into src.
It renders spherical-looking atoms as depth-sorted 2D circles after an
orthographic 3D projection, which is much faster than matplotlib 3D surfaces.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, ConnectionPatch, Polygon
from scipy.spatial import cKDTree

from src.analysis import close_contact_thresholds_satisfied, coordination_summary
from src.io_xyz import read_xyz


PU_COLOR = "#AEB8C4"
O_COLOR = "#B82218"
BOND_COLOR = "#7B8798"
TEXT_COLOR = "#1F2937"
MUTED_COLOR = "#667085"
ACCENT_COLOR = "#B54708"
ZOOM_FILL = "#FFF3C4"


def positions_by_atom(atoms: list[str], positions: np.ndarray, atom: str) -> np.ndarray:
    mask = np.asarray([item == atom for item in atoms], dtype=bool)
    return np.asarray(positions, dtype=float)[mask]


def rotation_matrix(elev: float = 18.0, azim: float = -54.0) -> np.ndarray:
    elev_rad = np.deg2rad(elev)
    azim_rad = np.deg2rad(azim)
    rz = np.array(
        [
            [np.cos(azim_rad), -np.sin(azim_rad), 0.0],
            [np.sin(azim_rad), np.cos(azim_rad), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(elev_rad), -np.sin(elev_rad)],
            [0.0, np.sin(elev_rad), np.cos(elev_rad)],
        ]
    )
    return rx @ rz


def project_points(positions: np.ndarray, center: np.ndarray, rot: np.ndarray) -> np.ndarray:
    return (np.asarray(positions, dtype=float) - center) @ rot.T


def compute_zoom_center(atoms: list[str], positions: np.ndarray, cutoff_pu_o: float = 3.2) -> np.ndarray:
    pu_positions = positions_by_atom(atoms, positions, "Pu")
    o_positions = positions_by_atom(atoms, positions, "O")
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
    positions = np.asarray(positions, dtype=float)
    keep = np.linalg.norm(positions - center, axis=1) <= radius
    return [atom for atom, is_kept in zip(atoms, keep, strict=True) if is_kept], positions[keep]


def draw_sphere_patch(ax, xy: np.ndarray, radius: float, color: str, zorder: float, alpha: float = 1.0) -> None:
    """Draw a 2D circle with layered shading so it reads as a sphere."""
    rgb = np.asarray(to_rgb(color), dtype=float)
    shadow = np.clip(rgb * 0.52, 0.0, 1.0)
    ax.add_patch(Circle(xy, radius, facecolor=shadow, edgecolor="none", alpha=alpha, zorder=zorder))
    steps = 6
    for i in range(steps):
        t = i / max(steps - 1, 1)
        r = radius * (0.93 - 0.075 * i)
        offset = np.array([-0.22 * radius * t, 0.26 * radius * t])
        shade = 0.66 + 0.34 * t
        color_i = np.clip(rgb * shade + (1.0 - shade) * 0.95, 0.0, 1.0)
        ax.add_patch(Circle(xy + offset, r, facecolor=color_i, edgecolor="none", alpha=alpha, zorder=zorder + 0.01 * i))
    ax.add_patch(Circle(xy + np.array([-0.24 * radius, 0.30 * radius]), radius * 0.22, facecolor="white", edgecolor="none", alpha=0.14 * alpha, zorder=zorder + 0.2))
    ax.add_patch(Circle(xy, radius, facecolor="none", edgecolor=np.clip(rgb * 0.42, 0.0, 1.0), linewidth=0.35, alpha=0.55 * alpha, zorder=zorder + 0.3))


def draw_bonds(ax, atoms: list[str], positions: np.ndarray, projected: np.ndarray, cutoff: float, linewidth: float, alpha: float) -> None:
    pu_indices = [i for i, atom in enumerate(atoms) if atom == "Pu"]
    o_indices = [i for i, atom in enumerate(atoms) if atom == "O"]
    if not pu_indices or not o_indices:
        return
    o_positions = positions[o_indices]
    tree = cKDTree(o_positions)
    for pu_index in pu_indices:
        for local_o_index in tree.query_ball_point(positions[pu_index], cutoff):
            o_index = o_indices[local_o_index]
            ax.plot(
                [projected[pu_index, 0], projected[o_index, 0]],
                [projected[pu_index, 1], projected[o_index, 1]],
                color=BOND_COLOR,
                linewidth=linewidth,
                alpha=alpha,
                zorder=1,
            )


def cube_edges(center: np.ndarray, box_size: float) -> list[tuple[np.ndarray, np.ndarray]]:
    half = box_size / 2.0
    x0, y0, z0 = center - half
    x1, y1, z1 = center + half
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
    pairs = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    return [(vertices[i], vertices[j]) for i, j in pairs]


def draw_projected_structure(
    ax,
    atoms: list[str],
    positions: np.ndarray,
    *,
    rot: np.ndarray,
    center: np.ndarray | None = None,
    limits: tuple[float, float, float, float] | None = None,
    pu_radius: float = 0.34,
    o_radius: float = 0.22,
    bond_cutoff: float = 2.95,
    bond_width: float = 0.75,
    bond_alpha: float = 0.48,
    zoom_center: np.ndarray | None = None,
    zoom_box_size: float | None = None,
) -> np.ndarray:
    center = np.mean(positions, axis=0) if center is None else center
    projected = project_points(positions, center, rot)
    draw_bonds(ax, atoms, positions, projected, cutoff=bond_cutoff, linewidth=bond_width, alpha=bond_alpha)

    for depth_order, atom_index in enumerate(np.argsort(projected[:, 2])):
        atom = atoms[atom_index]
        radius = pu_radius if atom == "Pu" else o_radius
        color = PU_COLOR if atom == "Pu" else O_COLOR
        draw_sphere_patch(ax, projected[atom_index, :2], radius, color, zorder=10 + depth_order, alpha=0.98)

    if zoom_center is not None and zoom_box_size is not None:
        projected_center = project_points(np.asarray([zoom_center]), center, rot)[0, :2]
        face_points = []
        for start, end in cube_edges(zoom_center, zoom_box_size):
            start_p = project_points(np.asarray([start]), center, rot)[0, :2]
            end_p = project_points(np.asarray([end]), center, rot)[0, :2]
            face_points.extend([start_p, end_p])
            ax.plot([start_p[0], end_p[0]], [start_p[1], end_p[1]], color=ACCENT_COLOR, linewidth=1.05, alpha=0.95, zorder=500)
        face_points = np.asarray(face_points)
        hull_min = face_points.min(axis=0)
        hull_max = face_points.max(axis=0)
        polygon = np.array([[hull_min[0], hull_min[1]], [hull_max[0], hull_min[1]], [hull_max[0], hull_max[1]], [hull_min[0], hull_max[1]]])
        ax.add_patch(Polygon(polygon, closed=True, facecolor=ZOOM_FILL, edgecolor="none", alpha=0.22, zorder=4))
        ax.scatter(projected_center[0], projected_center[1], s=18, color=ACCENT_COLOR, zorder=520)

    if limits is None:
        pad = 1.4
        limits = (projected[:, 0].min() - pad, projected[:, 0].max() + pad, projected[:, 1].min() - pad, projected[:, 1].max() + pad)
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    return projected


def atom_density_in_sphere(atom_count: int, radius: float) -> float:
    volume = (4.0 / 3.0) * np.pi * radius**3
    return atom_count / volume


def metrics(
    atoms: list[str],
    positions: np.ndarray,
    energy_per_puo2: float | None,
    zoom_atom_count: int | None = None,
    zoom_radius: float | None = None,
) -> dict[str, str]:
    summary = coordination_summary(atoms, positions)
    values = {
        "E/PuO2, eV": "n/a" if energy_per_puo2 is None else f"{energy_per_puo2:.4f}",
        "bulk order": f"{summary['bulk_fluorite_order_score']:.4f}",
        "coord. error": f"{summary['mean_abs_coordination_error']:.4f}",
        "Pu(8O)": f"{100.0 * summary['fraction_pu_with_8_o']:.1f} %",
        "safe contacts": "да" if close_contact_thresholds_satisfied(atoms, positions) else "нет",
    }
    if zoom_atom_count is not None and zoom_radius is not None:
        values["zoom density, atom/A^3"] = f"{atom_density_in_sphere(zoom_atom_count, zoom_radius):.4f}"
    return values


def add_metric_strip(fig, values: dict[str, str]) -> None:
    for x, (label, value) in zip(np.linspace(0.12, 0.88, len(values)), values.items(), strict=True):
        fig.text(x, 0.085, label, ha="center", va="bottom", fontsize=10.0, color=MUTED_COLOR)
        fig.text(x, 0.055, value, ha="center", va="top", fontsize=17.0, fontweight="bold", color=TEXT_COLOR)


def add_legend(fig, *, bbox_to_anchor: tuple[float, float] = (0.72, 0.935), ncol: int = 4) -> None:
    handles = [
        Line2D([], [], marker="o", linestyle="None", markersize=9, markerfacecolor=PU_COLOR, markeredgecolor="#687487", label="Pu"),
        Line2D([], [], marker="o", linestyle="None", markersize=8, markerfacecolor=O_COLOR, markeredgecolor="#7A271A", label="O"),
        Line2D([], [], color=BOND_COLOR, linewidth=1.5, label="связь Pu-O"),
        Line2D([], [], color=ACCENT_COLOR, linewidth=1.6, label="выделенный фрагмент"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=bbox_to_anchor, ncol=ncol, frameon=False, fontsize=12)


def plot_single(
    xyz: Path,
    output_png: Path,
    title: str,
    subtitle: str,
    zoom_center: np.ndarray,
    zoom_radius: float,
    energy_per_puo2: float | None,
) -> None:
    atoms, positions, _ = read_xyz(xyz)
    zoom_atoms, zoom_positions = atoms_in_sphere(atoms, positions, zoom_center, zoom_radius)
    rot = rotation_matrix()
    center = np.mean(positions, axis=0)
    projected = project_points(positions, center, rot)
    pad = 1.2
    limits = (projected[:, 0].min() - pad, projected[:, 0].max() + pad, projected[:, 1].min() - pad, projected[:, 1].max() + pad)

    fig = plt.figure(figsize=(13.6, 7.65), dpi=220, facecolor="white")
    ax_main = fig.add_axes([0.030, 0.145, 0.625, 0.740])
    ax_zoom = fig.add_axes([0.700, 0.175, 0.290, 0.570])
    draw_projected_structure(
        ax_main,
        atoms,
        positions,
        rot=rot,
        center=center,
        limits=limits,
        pu_radius=0.34,
        o_radius=0.22,
        bond_width=0.82,
        bond_alpha=0.48,
        zoom_center=zoom_center,
        zoom_box_size=4.8,
    )
    zoom_projected = draw_projected_structure(
        ax_zoom,
        zoom_atoms,
        zoom_positions,
        rot=rot,
        center=zoom_center,
        pu_radius=0.58,
        o_radius=0.38,
        bond_width=1.05,
        bond_alpha=0.62,
    )

    projected_zoom_center = project_points(np.asarray([zoom_center]), center, rot)[0, :2]
    fig.add_artist(
        ConnectionPatch(
            xyA=(0.0, 0.56),
            coordsA=ax_zoom.transAxes,
            xyB=(projected_zoom_center[0], projected_zoom_center[1]),
            coordsB=ax_main.transData,
            arrowstyle="->",
            mutation_scale=17,
            linewidth=1.35,
            color=ACCENT_COLOR,
            shrinkA=2,
            shrinkB=5,
        )
    )

    fig.text(0.045, 0.945, title, ha="left", va="top", fontsize=24, fontweight="bold", color=TEXT_COLOR)
    fig.text(0.047, 0.895, subtitle, ha="left", va="top", fontsize=13, color=MUTED_COLOR)
    add_legend(fig, bbox_to_anchor=(0.74, 0.935), ncol=4)
    fig.text(0.715, 0.760, f"Увеличенный фрагмент, R = {zoom_radius:.1f} Å", fontsize=13, color=TEXT_COLOR, ha="left")
    for spine in ax_zoom.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_color("#475467")
    ax_zoom.set_xlim(zoom_projected[:, 0].min() - 0.9, zoom_projected[:, 0].max() + 0.9)
    ax_zoom.set_ylim(zoom_projected[:, 1].min() - 0.9, zoom_projected[:, 1].max() + 0.9)
    ax_zoom.set_xticks([])
    ax_zoom.set_yticks([])
    add_metric_strip(fig, metrics(atoms, positions, energy_per_puo2, len(zoom_atoms), zoom_radius))

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)


def plot_comparison(initial_xyz: Path, final_xyz: Path, output_png: Path, initial_energy: float | None, final_energy: float | None) -> None:
    atoms_i, pos_i, _ = read_xyz(initial_xyz)
    atoms_f, pos_f, _ = read_xyz(final_xyz)
    rot = rotation_matrix()
    center = np.mean(np.vstack([pos_i, pos_f]), axis=0)
    proj_all = project_points(np.vstack([pos_i, pos_f]), center, rot)
    pad = 1.6
    limits = (proj_all[:, 0].min() - pad, proj_all[:, 0].max() + pad, proj_all[:, 1].min() - pad, proj_all[:, 1].max() + pad)

    summary_i = coordination_summary(atoms_i, pos_i)
    summary_f = coordination_summary(atoms_f, pos_f)
    delta = "n/a" if initial_energy is None or final_energy is None else f"{final_energy - initial_energy:+.4f} eV"

    fig = plt.figure(figsize=(13.6, 7.65), dpi=220, facecolor="white")
    ax_i = fig.add_axes([0.035, 0.155, 0.440, 0.690])
    ax_f = fig.add_axes([0.530, 0.155, 0.440, 0.690])
    draw_projected_structure(ax_i, atoms_i, pos_i, rot=rot, center=center, limits=limits, pu_radius=0.33, o_radius=0.215, bond_width=0.82, bond_alpha=0.48)
    draw_projected_structure(ax_f, atoms_f, pos_f, rot=rot, center=center, limits=limits, pu_radius=0.33, o_radius=0.215, bond_width=0.82, bond_alpha=0.48)

    fig.text(0.045, 0.945, "Сравнение начальной и конечной структуры", ha="left", va="top", fontsize=23, fontweight="bold", color=TEXT_COLOR)
    fig.text(
        0.045,
        0.895,
        f"ΔE/PuO2 = {delta}    bulk order: {summary_i['bulk_fluorite_order_score']:.4f} → {summary_f['bulk_fluorite_order_score']:.4f}    coord. error: {summary_i['mean_abs_coordination_error']:.4f} → {summary_f['mean_abs_coordination_error']:.4f}",
        ha="left",
        va="top",
        fontsize=12,
        color=MUTED_COLOR,
    )
    fig.text(0.055, 0.835, "A", fontsize=18, fontweight="bold", color=TEXT_COLOR)
    fig.text(0.088, 0.835, "Начальная структура", fontsize=15, color=TEXT_COLOR)
    fig.text(0.550, 0.835, "B", fontsize=18, fontweight="bold", color=TEXT_COLOR)
    fig.text(0.583, 0.835, "После ML-kMC", fontsize=15, color=TEXT_COLOR)
    add_legend(fig, bbox_to_anchor=(0.52, 0.075), ncol=4)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast draft PuO2 crystal renderer.")
    parser.add_argument("--initial-xyz", default="input/PuO2_324.xyz")
    parser.add_argument("--final-xyz", default="results/06_seeded_stage2_polish_1000K/final.xyz")
    parser.add_argument("--out-dir", default="results/07_crystal_visualization_draft")
    parser.add_argument("--zoom-radius", type=float, default=6.0)
    parser.add_argument("--initial-energy-per-puo2", type=float, default=-46.6442)
    parser.add_argument("--final-energy-per-puo2", type=float, default=-47.5877)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    initial_xyz = Path(args.initial_xyz)
    final_xyz = Path(args.final_xyz)
    out_dir = Path(args.out_dir)
    if not initial_xyz.is_absolute():
        initial_xyz = base_dir / initial_xyz
    if not final_xyz.is_absolute():
        final_xyz = base_dir / final_xyz
    if not out_dir.is_absolute():
        out_dir = base_dir / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(initial_xyz, out_dir / "initial_crystal.xyz")
    shutil.copyfile(final_xyz, out_dir / "final_crystal.xyz")

    atoms_final, positions_final, _ = read_xyz(final_xyz)
    zoom_center = compute_zoom_center(atoms_final, positions_final)
    plot_single(
        initial_xyz,
        out_dir / "draft_initial_crystal_visualization.png",
        "Начальная конфигурация",
        "Исходная структура PuO2",
        zoom_center,
        args.zoom_radius,
        args.initial_energy_per_puo2,
    )
    plot_single(
        final_xyz,
        out_dir / "draft_final_crystal_visualization.png",
        "Конечная конфигурация",
        "Структура после двухстадийного seeded ML-kMC",
        zoom_center,
        args.zoom_radius,
        args.final_energy_per_puo2,
    )
    plot_comparison(
        initial_xyz,
        final_xyz,
        out_dir / "draft_initial_final_comparison.png",
        args.initial_energy_per_puo2,
        args.final_energy_per_puo2,
    )
    print(f"Saved draft figures to {out_dir}")


if __name__ == "__main__":
    main()
