"""Seed construction utilities for seeded PuO2 crystallization runs."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


MIN_PAIR_DISTANCES = {
    ("Pu", "O"): 1.9,
    ("O", "Pu"): 1.9,
    ("O", "O"): 2.0,
    ("Pu", "Pu"): 2.8,
}


def fluorite_basis_sites(center: np.ndarray, radius: float, lattice_constant: float = 5.40) -> tuple[np.ndarray, np.ndarray]:
    """Return ideal Pu and O fluorite sites inside a sphere around center."""
    center = np.asarray(center, dtype=float)
    pu_basis = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ],
        dtype=float,
    )
    o_basis = np.asarray(
        [
            [0.25, 0.25, 0.25],
            [0.25, 0.75, 0.75],
            [0.75, 0.25, 0.75],
            [0.75, 0.75, 0.25],
            [0.75, 0.75, 0.75],
            [0.75, 0.25, 0.25],
            [0.25, 0.75, 0.25],
            [0.25, 0.25, 0.75],
        ],
        dtype=float,
    )
    n_cells = int(np.ceil(radius / lattice_constant)) + 2
    pu_sites: list[np.ndarray] = []
    o_sites: list[np.ndarray] = []
    for i in range(-n_cells, n_cells + 1):
        for j in range(-n_cells, n_cells + 1):
            for k in range(-n_cells, n_cells + 1):
                origin = center + lattice_constant * np.asarray([i, j, k], dtype=float)
                for basis in pu_basis:
                    site = origin + lattice_constant * (basis - 0.5)
                    if np.linalg.norm(site - center) <= radius:
                        pu_sites.append(site)
                for basis in o_basis:
                    site = origin + lattice_constant * (basis - 0.5)
                    if np.linalg.norm(site - center) <= radius:
                        o_sites.append(site)
    return np.asarray(pu_sites, dtype=float), np.asarray(o_sites, dtype=float)


def _assign_nearest_unique(atom_positions: np.ndarray, sites: np.ndarray) -> np.ndarray:
    if len(atom_positions) == 0 or len(sites) == 0:
        return atom_positions.copy()
    available = set(range(len(sites)))
    assigned = atom_positions.copy()
    order = np.argsort(np.linalg.norm(atom_positions - atom_positions.mean(axis=0), axis=1))
    for atom_local_index in order:
        candidates = np.asarray(sorted(available), dtype=int)
        if len(candidates) == 0:
            break
        distances = np.linalg.norm(sites[candidates] - atom_positions[atom_local_index], axis=1)
        picked = int(candidates[int(np.argmin(distances))])
        assigned[atom_local_index] = sites[picked]
        available.remove(picked)
    return assigned


def impose_fluorite_seed(
    atoms: list[str],
    positions: np.ndarray,
    seed_radius: float = 5.0,
    lattice_constant: float = 5.40,
    blend: float = 1.0,
) -> tuple[np.ndarray, set[int]]:
    """Snap atoms in the central seed region toward ideal fluorite sites."""
    positions = np.asarray(positions, dtype=float)
    updated = positions.copy()
    center = positions.mean(axis=0)
    radii = np.linalg.norm(positions - center, axis=1)
    seed_indices = np.flatnonzero(radii <= seed_radius)
    pu_sites, o_sites = fluorite_basis_sites(center, seed_radius, lattice_constant)
    for atom_type, sites in [("Pu", pu_sites), ("O", o_sites)]:
        indices = [int(i) for i in seed_indices if atoms[int(i)] == atom_type]
        if not indices:
            continue
        assigned = _assign_nearest_unique(positions[indices], sites)
        updated[indices] = positions[indices] + float(blend) * (assigned - positions[indices])
    for index in seed_indices:
        atom = atoms[int(index)]
        valid = True
        for j in range(len(atoms)):
            if j == int(index):
                continue
            threshold = MIN_PAIR_DISTANCES.get((atom, atoms[j]), 1.2)
            if float(np.linalg.norm(updated[int(index)] - updated[j])) < threshold:
                valid = False
                break
        if not valid:
            updated[int(index)] = positions[int(index)]
    return updated, set(int(i) for i in seed_indices)


def nearest_fluorite_site_displacements(
    atoms: list[str],
    positions: np.ndarray,
    atom_indices: list[int],
    seed_center: np.ndarray,
    seed_radius: float,
    lattice_constant: float = 5.40,
) -> dict[int, np.ndarray]:
    """Return vectors from selected atoms to nearest ideal fluorite site of the same species."""
    pu_sites, o_sites = fluorite_basis_sites(seed_center, seed_radius, lattice_constant)
    trees = {
        "Pu": cKDTree(pu_sites) if len(pu_sites) else None,
        "O": cKDTree(o_sites) if len(o_sites) else None,
    }
    displacements: dict[int, np.ndarray] = {}
    for index in atom_indices:
        tree = trees.get(atoms[index])
        if tree is None:
            continue
        _, site_index = tree.query(positions[index], k=1)
        site = (pu_sites if atoms[index] == "Pu" else o_sites)[int(site_index)]
        displacements[int(index)] = site - positions[index]
    return displacements
