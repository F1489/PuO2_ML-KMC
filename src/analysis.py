"""Structural analysis utilities for PuO2 clusters."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .io_xyz import read_xyz
from .potentials import PairPotentialPuO2


def radial_distribution_function(
    atoms: list[str], positions: np.ndarray, r_max: float, dr: float, pair: tuple[str, str] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a simple finite-cluster RDF-like pair-distance histogram."""
    positions = np.asarray(positions, dtype=float)
    distances: list[float] = []
    pair_sorted = tuple(sorted(pair)) if pair is not None else None
    tree = cKDTree(positions)
    for i, j in tree.query_pairs(r=r_max, output_type="set"):
        if pair_sorted is not None and tuple(sorted((atoms[i], atoms[j]))) != pair_sorted:
            continue
        distances.append(float(np.linalg.norm(positions[j] - positions[i])))
    bins = np.arange(0.0, r_max + dr, dr)
    hist, edges = np.histogram(distances, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    norm = max(len(distances), 1)
    return centers, hist.astype(float) / norm


def min_pair_distances(atoms: list[str], positions: np.ndarray) -> dict[str, float]:
    """Return minimum Pu-O, O-O, and Pu-Pu distances in A."""
    positions = np.asarray(positions, dtype=float)
    mins = {
        "min_distance_pu_o": np.inf,
        "min_distance_o_o": np.inf,
        "min_distance_pu_pu": np.inf,
    }
    for i in range(len(atoms) - 1):
        for j in range(i + 1, len(atoms)):
            distance = float(np.linalg.norm(positions[j] - positions[i]))
            pair = tuple(sorted((atoms[i], atoms[j])))
            if pair == ("O", "Pu"):
                mins["min_distance_pu_o"] = min(mins["min_distance_pu_o"], distance)
            elif pair == ("O", "O"):
                mins["min_distance_o_o"] = min(mins["min_distance_o_o"], distance)
            elif pair == ("Pu", "Pu"):
                mins["min_distance_pu_pu"] = min(mins["min_distance_pu_pu"], distance)
    return {key: (0.0 if not np.isfinite(value) else float(value)) for key, value in mins.items()}


def close_contact_thresholds_satisfied(atoms: list[str], positions: np.ndarray) -> bool:
    """Return True when Pu-O, O-O, and Pu-Pu minimum distances pass protective thresholds."""
    mins = min_pair_distances(atoms, positions)
    return bool(
        mins["min_distance_pu_o"] >= 1.9
        and mins["min_distance_o_o"] >= 2.0
        and mins["min_distance_pu_pu"] >= 2.8
    )


def coordination_numbers(atoms: list[str], positions: np.ndarray, cutoff_pu_o: float = 3.2) -> dict[str, np.ndarray]:
    """Return opposite-species coordination numbers using a Pu-O cutoff in A."""
    positions = np.asarray(positions, dtype=float)
    tree = cKDTree(positions)
    pu_o_counts: list[int] = []
    o_pu_counts: list[int] = []
    for i, atom in enumerate(atoms):
        neighbor_indices = [j for j in tree.query_ball_point(positions[i], r=cutoff_pu_o) if j != i]
        if atom == "Pu":
            count = sum(atoms[j] == "O" for j in neighbor_indices)
            pu_o_counts.append(count)
        else:
            count = sum(atoms[j] == "Pu" for j in neighbor_indices)
            o_pu_counts.append(count)
    return {"Pu_O": np.asarray(pu_o_counts, dtype=int), "O_Pu": np.asarray(o_pu_counts, dtype=int)}


def surface_mask(positions: np.ndarray, surface_shell_thickness: float = 3.2) -> np.ndarray:
    """Return atoms in the outer radial shell of a finite cluster."""
    positions = np.asarray(positions, dtype=float)
    if len(positions) == 0:
        return np.asarray([], dtype=bool)
    center = np.mean(positions, axis=0)
    radii = np.linalg.norm(positions - center, axis=1)
    return radii >= (float(np.max(radii)) - surface_shell_thickness)


def _coordination_arrays_by_atom(atoms: list[str], positions: np.ndarray, cutoff_pu_o: float) -> tuple[np.ndarray, np.ndarray]:
    """Return opposite-species coordination and ideal coordination per atom."""
    positions = np.asarray(positions, dtype=float)
    tree = cKDTree(positions)
    coordination = np.zeros(len(atoms), dtype=int)
    ideal = np.zeros(len(atoms), dtype=int)
    for i, atom in enumerate(atoms):
        neighbor_indices = [j for j in tree.query_ball_point(positions[i], r=cutoff_pu_o) if j != i]
        if atom == "Pu":
            coordination[i] = sum(atoms[j] == "O" for j in neighbor_indices)
            ideal[i] = 8
        else:
            coordination[i] = sum(atoms[j] == "Pu" for j in neighbor_indices)
            ideal[i] = 4
    return coordination, ideal


def local_fluorite_order(atoms: list[str], positions: np.ndarray, cutoff_pu_o: float = 3.2) -> np.ndarray:
    """Return per-atom local fluorite-like coordination score in [0, 1]."""
    coordination, ideal = _coordination_arrays_by_atom(atoms, positions, cutoff_pu_o)
    if len(coordination) == 0:
        return np.asarray([], dtype=float)
    penalties = np.abs(coordination - ideal) / np.maximum(ideal, 1)
    return np.clip(1.0 - penalties, 0.0, 1.0)


def identify_crystalline_core(
    atoms: list[str],
    positions: np.ndarray,
    order_threshold: float = 0.75,
    cutoff_pu_o: float = 3.2,
    exclude_surface: bool = True,
) -> np.ndarray:
    """Return atom indices with high local fluorite-like order."""
    scores = local_fluorite_order(atoms, positions, cutoff_pu_o)
    mask = scores >= order_threshold
    if exclude_surface:
        mask &= ~surface_mask(positions, surface_shell_thickness=cutoff_pu_o)
    return np.flatnonzero(mask)


def identify_growth_front(
    atoms: list[str],
    positions: np.ndarray,
    core_indices: np.ndarray | None = None,
    order_threshold: float = 0.75,
    front_radius: float = 4.2,
    cutoff_pu_o: float = 3.2,
) -> np.ndarray:
    """Return disordered atoms adjacent to a crystalline core."""
    positions = np.asarray(positions, dtype=float)
    core = identify_crystalline_core(atoms, positions, order_threshold, cutoff_pu_o) if core_indices is None else np.asarray(core_indices, dtype=int)
    if len(core) == 0:
        return np.asarray([], dtype=int)
    scores = local_fluorite_order(atoms, positions, cutoff_pu_o)
    tree = cKDTree(positions)
    front = np.zeros(len(atoms), dtype=bool)
    for i in core:
        for j in tree.query_ball_point(positions[i], r=front_radius):
            if scores[j] < order_threshold:
                front[j] = True
    front[core] = False
    return np.flatnonzero(front)


def fluorite_order_score(atoms: list[str], positions: np.ndarray) -> float:
    """Return fraction of atoms with correct or nearly correct fluorite coordination."""
    coords = coordination_numbers(atoms, positions)
    good_pu = np.sum(np.abs(coords["Pu_O"] - 8) <= 1)
    good_o = np.sum(np.abs(coords["O_Pu"] - 4) <= 1)
    return float((good_pu + good_o) / max(len(atoms), 1))


def soft_coordination_order_score(atoms: list[str], positions: np.ndarray, cutoff_pu_o: float = 3.2) -> float:
    """Return a smooth coordination score where near-fluorite environments are partially rewarded."""
    coordination, ideal = _coordination_arrays_by_atom(atoms, positions, cutoff_pu_o)
    if len(coordination) == 0:
        return 0.0
    penalties = np.abs(coordination - ideal) / np.maximum(ideal, 1)
    return float(np.mean(np.clip(1.0 - penalties, 0.0, 1.0)))


def bulk_fluorite_order_score(
    atoms: list[str],
    positions: np.ndarray,
    cutoff_pu_o: float = 3.2,
    surface_shell_thickness: float = 3.2,
) -> float:
    """Return the strict fluorite score for atoms away from the cluster surface."""
    coordination, ideal = _coordination_arrays_by_atom(atoms, positions, cutoff_pu_o)
    bulk = ~surface_mask(positions, surface_shell_thickness)
    if not np.any(bulk):
        return 0.0
    return float(np.mean(np.abs(coordination[bulk] - ideal[bulk]) <= 1))


def rdf_peak_sharpness(
    atoms: list[str],
    positions: np.ndarray,
    pair: tuple[str, str] = ("Pu", "O"),
    r_max: float = 8.0,
    dr: float = 0.1,
) -> float:
    """Return a simple peak-to-background RDF sharpness metric."""
    _, values = radial_distribution_function(atoms, positions, r_max=r_max, dr=dr, pair=pair)
    if len(values) == 0:
        return 0.0
    background = float(np.mean(values))
    if background <= 0.0:
        return 0.0
    return float(np.max(values) / background)


def coordination_summary(atoms: list[str], positions: np.ndarray, cutoff_pu_o: float = 3.2) -> dict[str, float]:
    """Return Russian-report-ready coordination summary metrics."""
    coords = coordination_numbers(atoms, positions, cutoff_pu_o)
    pu_o = coords["Pu_O"]
    o_pu = coords["O_Pu"]
    n_pu = max(len(pu_o), 1)
    n_o = max(len(o_pu), 1)
    atom_coordination, atom_ideal = _coordination_arrays_by_atom(atoms, positions, cutoff_pu_o)
    atom_surface = surface_mask(positions, surface_shell_thickness=cutoff_pu_o)
    atom_bulk = ~atom_surface
    bulk_defects = int(np.sum(atom_coordination[atom_bulk] != atom_ideal[atom_bulk])) if np.any(atom_bulk) else 0
    surface_defects = int(np.sum(atom_coordination[atom_surface] != atom_ideal[atom_surface])) if np.any(atom_surface) else 0
    soft_penalty = np.abs(atom_coordination - atom_ideal)
    return {
        "mean_pu_o_coordination": float(np.mean(pu_o)) if len(pu_o) else 0.0,
        "mean_o_pu_coordination": float(np.mean(o_pu)) if len(o_pu) else 0.0,
        "fraction_pu_with_8_o": float(np.sum(pu_o == 8) / n_pu),
        "fraction_o_with_4_pu": float(np.sum(o_pu == 4) / n_o),
        "fraction_pu_with_7_to_9_o": float(np.sum(np.abs(pu_o - 8) <= 1) / n_pu),
        "fraction_o_with_3_to_5_pu": float(np.sum(np.abs(o_pu - 4) <= 1) / n_o),
        "fraction_pu_with_7_to_9_o_neighbors": float(np.sum(np.abs(pu_o - 8) <= 1) / n_pu),
        "fraction_o_with_3_to_5_pu_neighbors": float(np.sum(np.abs(o_pu - 4) <= 1) / n_o),
        "mean_abs_pu_o_coordination_error": float(np.mean(np.abs(pu_o - 8))) if len(pu_o) else 0.0,
        "mean_abs_o_pu_coordination_error": float(np.mean(np.abs(o_pu - 4))) if len(o_pu) else 0.0,
        "fluorite_order_score": fluorite_order_score(atoms, positions),
        "bulk_fluorite_order_score": bulk_fluorite_order_score(atoms, positions, cutoff_pu_o),
        "soft_coordination_order_score": soft_coordination_order_score(atoms, positions, cutoff_pu_o),
        "mean_abs_coordination_error": float(np.mean(soft_penalty)) if len(soft_penalty) else 0.0,
        "bulk_coordination_defects": bulk_defects,
        "surface_coordination_defects": surface_defects,
        "bulk_atom_fraction": float(np.mean(atom_bulk)) if len(atom_bulk) else 0.0,
        "rdf_pu_o_peak_sharpness": rdf_peak_sharpness(atoms, positions, pair=("Pu", "O")),
        "rdf_pu_pu_peak_sharpness": rdf_peak_sharpness(atoms, positions, pair=("Pu", "Pu")),
        "rdf_o_o_peak_sharpness": rdf_peak_sharpness(atoms, positions, pair=("O", "O")),
    }


def defect_counts(atoms: list[str], positions: np.ndarray, cutoff_pu_o: float = 3.2) -> dict[str, int]:
    """Return simple fluorite coordination defect counts."""
    coords = coordination_numbers(atoms, positions, cutoff_pu_o)
    atom_coordination, atom_ideal = _coordination_arrays_by_atom(atoms, positions, cutoff_pu_o)
    atom_surface = surface_mask(positions, surface_shell_thickness=cutoff_pu_o)
    atom_bulk = ~atom_surface
    pu_defects = int(np.sum(coords["Pu_O"] != 8))
    o_defects = int(np.sum(coords["O_Pu"] != 4))
    soft_error = np.abs(atom_coordination - atom_ideal)
    return {
        "pu_coordination_defects": pu_defects,
        "o_coordination_defects": o_defects,
        "total_coordination_defects": pu_defects + o_defects,
        "bulk_coordination_defects": int(np.sum(atom_coordination[atom_bulk] != atom_ideal[atom_bulk])) if np.any(atom_bulk) else 0,
        "surface_coordination_defects": int(np.sum(atom_coordination[atom_surface] != atom_ideal[atom_surface])) if np.any(atom_surface) else 0,
        "mean_abs_coordination_error": float(np.mean(soft_error)) if len(soft_error) else 0.0,
    }


def compare_initial_final(initial_xyz: str | Path, final_xyz: str | Path, potential: PairPotentialPuO2) -> dict[str, float]:
    """Compare energies and order scores for two XYZ structures."""
    atoms_i, pos_i, _ = read_xyz(initial_xyz)
    atoms_f, pos_f, _ = read_xyz(final_xyz)
    e_i = potential.total_energy(atoms_i, pos_i)
    e_f = potential.total_energy(atoms_f, pos_f)
    return {
        "n_atoms": float(len(atoms_i)),
        "initial_energy": e_i,
        "final_energy": e_f,
        "relative_energy_decrease": float((e_i - e_f) / abs(e_i)) if e_i != 0.0 else 0.0,
        "initial_fluorite_order_score": fluorite_order_score(atoms_i, pos_i),
        "final_fluorite_order_score": fluorite_order_score(atoms_f, pos_f),
    }


def write_summary_json(
    output_path: str | Path,
    atoms_initial: list[str],
    positions_initial: np.ndarray,
    atoms_final: list[str],
    positions_final: np.ndarray,
    potential: PairPotentialPuO2,
    positions_after_repair: np.ndarray | None = None,
) -> dict[str, object]:
    """Write a compact Russian-analysis summary to JSON."""
    e_initial = potential.total_energy(atoms_initial, positions_initial)
    e_final = potential.total_energy(atoms_final, positions_final)
    n_formula_units = max(sum(atom == "Pu" for atom in atoms_initial), 1)
    e_after_repair = potential.total_energy(atoms_initial, positions_after_repair) if positions_after_repair is not None else None
    summary: dict[str, object] = {
        "units": {
            "distance": "angstrom",
            "energy": "eV",
        },
        "n_atoms": len(atoms_initial),
        "n_pu": sum(atom == "Pu" for atom in atoms_initial),
        "n_o": sum(atom == "O" for atom in atoms_initial),
        "initial_energy_eV": e_initial,
        "final_energy_eV": e_final,
        "initial_energy_per_atom_eV": e_initial / len(atoms_initial),
        "final_energy_per_atom_eV": e_final / len(atoms_final),
        "n_formula_units": n_formula_units,
        "original_energy_per_puo2_eV": e_initial / n_formula_units,
        "initial_energy_per_puo2_eV": e_initial / n_formula_units,
        "after_repair_energy_per_puo2_eV": (e_after_repair / n_formula_units) if e_after_repair is not None else None,
        "final_energy_per_puo2_eV": e_final / n_formula_units,
        "delta_repair_energy_per_puo2_eV": ((e_after_repair - e_initial) / n_formula_units) if e_after_repair is not None else None,
        "delta_kmc_energy_per_puo2_eV": ((e_final - e_after_repair) / n_formula_units) if e_after_repair is not None else None,
        "delta_total_energy_per_puo2_eV": (e_final - e_initial) / n_formula_units,
        "delta_energy_per_puo2_eV": (e_final - e_initial) / n_formula_units,
        "relative_energy_decrease": (e_initial - e_final) / abs(e_initial) if e_initial else 0.0,
        "initial_structure": coordination_summary(atoms_initial, positions_initial),
        "after_repair_structure": coordination_summary(atoms_initial, positions_after_repair) if positions_after_repair is not None else None,
        "final_structure": coordination_summary(atoms_final, positions_final),
        "initial_min_pair_distances": min_pair_distances(atoms_initial, positions_initial),
        "after_repair_min_pair_distances": min_pair_distances(atoms_initial, positions_after_repair) if positions_after_repair is not None else None,
        "final_min_pair_distances": min_pair_distances(atoms_final, positions_final),
        "initial_close_contact_thresholds_satisfied": close_contact_thresholds_satisfied(atoms_initial, positions_initial),
        "after_repair_close_contact_thresholds_satisfied": close_contact_thresholds_satisfied(atoms_initial, positions_after_repair) if positions_after_repair is not None else None,
        "final_close_contact_thresholds_satisfied": close_contact_thresholds_satisfied(atoms_final, positions_final),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
