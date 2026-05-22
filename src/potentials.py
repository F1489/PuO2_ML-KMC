"""MOX-07 pair potential for free PuO2 clusters.

Distances are in angstrom (A), energies in electronvolt (eV). No PBC is used.
The implemented form is:

U_ij(R_ij) = K_E * q_i * q_j / R_ij + A_ij * exp(-B_ij * R_ij) - C_ij / R_ij^6
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree


PairKey = tuple[str, str]


def _pair_key(a: str, b: str) -> PairKey:
    return tuple(sorted((a, b)))  # type: ignore[return-value]


@dataclass
class PairPotentialPuO2:
    """MOX-07 Buckingham+Coulomb pair potential for a free PuO2 cluster."""

    short_range_cutoff: float = 10.0
    cutoff: float | None = None
    small_eps: float = 1.0e-8
    k_coul: float = 14.399645478
    charges: dict[str, float] = field(default_factory=lambda: {"Pu": 2.745, "O": -1.3725})
    params: dict[PairKey, dict[str, float]] = field(
        default_factory=lambda: {
            ("O", "O"): {"A": 50212.0, "B": 5.5200, "C": 74.796},
            ("O", "Pu"): {"A": 871.79, "B": 2.8079, "C": 0.0},
            ("Pu", "Pu"): {"A": 0.0, "B": 0.0, "C": 0.0},
        }
    )

    def __post_init__(self) -> None:
        """Keep old cutoff=... constructor calls as an alias for the short-range cutoff."""
        if self.cutoff is not None:
            self.short_range_cutoff = float(self.cutoff)

    def pair_energy(self, atom_i: str, atom_j: str, r: float) -> float:
        """Return MOX-07 pair energy U_ij(r) in eV for distance r in A."""
        if r < self.small_eps:
            return 0.0
        r_safe = max(float(r), self.small_eps)
        key = _pair_key(atom_i, atom_j)
        if key not in self.params:
            raise ValueError(f"No pair parameters for {key}")
        p = self.params[key]
        coul = self.k_coul * self.charges[atom_i] * self.charges[atom_j] / r_safe
        short_range = 0.0
        if r_safe <= self.short_range_cutoff:
            short_range = p["A"] * np.exp(-p["B"] * r_safe) - p["C"] / (r_safe**6)
        return float(coul + short_range)

    def total_energy(self, atoms: list[str], positions: np.ndarray) -> float:
        """Return total pair energy in eV using all Coulomb pairs and short-range cutoff."""
        positions = np.asarray(positions, dtype=float)
        energy = 0.0
        for i in range(len(atoms) - 1):
            deltas = positions[i + 1 :] - positions[i]
            distances = np.linalg.norm(deltas, axis=1)
            for offset, r in enumerate(distances, start=1):
                energy += self.pair_energy(atoms[i], atoms[i + offset], float(r))
        return float(energy)

    def local_energy(
        self,
        atoms: list[str],
        positions: np.ndarray,
        index: int,
        radius: float | None = None,
        tree: cKDTree | None = None,
    ) -> float:
        """Return interaction energy of one atom with neighbors in eV."""
        positions = np.asarray(positions, dtype=float)
        cutoff = None if radius is None else float(radius)
        energy = 0.0
        if cutoff is None:
            neighbor_indices = range(len(atoms))
        else:
            tree = cKDTree(positions) if tree is None else tree
            neighbor_indices = tree.query_ball_point(positions[index], r=cutoff)
        for j in neighbor_indices:
            if j == index:
                continue
            r = float(np.linalg.norm(positions[j] - positions[index]))
            if cutoff is None or r <= cutoff:
                energy += self.pair_energy(atoms[index], atoms[j], r)
        return float(energy)

    def local_energy_at_position(
        self,
        atoms: list[str],
        positions: np.ndarray,
        index: int,
        point: np.ndarray,
        radius: float | None = None,
        tree: cKDTree | None = None,
    ) -> float:
        """Return interaction energy of one atom placed at point with the fixed environment."""
        positions = np.asarray(positions, dtype=float)
        point = np.asarray(point, dtype=float)
        cutoff = None if radius is None else float(radius)
        if cutoff is None:
            neighbor_indices = range(len(atoms))
        else:
            tree = cKDTree(positions) if tree is None else tree
            neighbor_indices = tree.query_ball_point(point, r=cutoff)
        energy = 0.0
        for j in neighbor_indices:
            if j == index:
                continue
            r = float(np.linalg.norm(positions[j] - point))
            if cutoff is None or r <= cutoff:
                energy += self.pair_energy(atoms[index], atoms[j], r)
        return float(energy)

    def local_energy_and_force_at_position(
        self,
        atoms: list[str],
        positions: np.ndarray,
        index: int,
        point: np.ndarray,
        radius: float | None = None,
        tree: cKDTree | None = None,
    ) -> tuple[float, np.ndarray]:
        """Return local interaction energy and force for one atom at point.

        This combines the two most common feature-extraction queries into one
        neighbor traversal.
        """
        positions = np.asarray(positions, dtype=float)
        point = np.asarray(point, dtype=float)
        cutoff = None if radius is None else float(radius)
        tree = cKDTree(positions) if tree is None else tree
        neighbor_indices = range(len(atoms)) if cutoff is None else tree.query_ball_point(point, r=cutoff)
        atom_i = atoms[index]
        charge_i = self.charges[atom_i]
        energy = 0.0
        force = np.zeros(3, dtype=float)
        for j in neighbor_indices:
            if j == index:
                continue
            vector = point - positions[j]
            r = max(float(np.linalg.norm(vector)), self.small_eps)
            if cutoff is not None and r > cutoff:
                continue
            atom_j = atoms[j]
            key = _pair_key(atom_i, atom_j)
            p = self.params[key]
            charge_product = charge_i * self.charges[atom_j]
            coul = self.k_coul * charge_product / r
            d_u_dr = -self.k_coul * charge_product / (r**2)
            short_range = 0.0
            if r <= self.short_range_cutoff:
                exp_term = np.exp(-p["B"] * r)
                short_range = p["A"] * exp_term - p["C"] / (r**6)
                d_u_dr += -p["A"] * p["B"] * exp_term + 6.0 * p["C"] / (r**7)
            energy += coul + short_range
            force += -d_u_dr * vector / r
        return float(energy), force

    def delta_energy_single_move(
        self, atoms: list[str], positions: np.ndarray, index: int, new_position: np.ndarray
    ) -> float:
        """Return exact Delta E in eV for moving one atom to new_position."""
        positions = np.asarray(positions, dtype=float)
        tree = cKDTree(positions)
        old_local = self.local_energy(atoms, positions, index, tree=tree)
        new_local = self.local_energy_at_position(atoms, positions, index, new_position, tree=tree)
        return float(new_local - old_local)

    def force_on_atom(
        self,
        atoms: list[str],
        positions: np.ndarray,
        index: int,
        radius: float | None = None,
        tree: cKDTree | None = None,
    ) -> np.ndarray:
        """Return negative energy gradient on one atom in eV/A."""
        positions = np.asarray(positions, dtype=float)
        cutoff = None if radius is None else float(radius)
        tree = cKDTree(positions) if tree is None else tree
        force = np.zeros(3, dtype=float)
        neighbor_indices = range(len(atoms)) if cutoff is None else tree.query_ball_point(positions[index], r=cutoff)
        atom_i = atoms[index]
        charge_i = self.charges[atom_i]
        for j in neighbor_indices:
            if j == index:
                continue
            vector = positions[index] - positions[j]
            r = max(float(np.linalg.norm(vector)), self.small_eps)
            if cutoff is not None and r > cutoff:
                continue
            atom_j = atoms[j]
            p = self.params[_pair_key(atom_i, atom_j)]
            d_u_dr = -self.k_coul * charge_i * self.charges[atom_j] / (r**2)
            if r <= self.short_range_cutoff:
                d_u_dr += -p["A"] * p["B"] * np.exp(-p["B"] * r) + 6.0 * p["C"] / (r**7)
            force += -d_u_dr * vector / r
        return force

    def force_on_atom_at_position(
        self,
        atoms: list[str],
        positions: np.ndarray,
        index: int,
        point: np.ndarray,
        radius: float | None = None,
        tree: cKDTree | None = None,
    ) -> np.ndarray:
        """Return force on one atom placed at point while all other atoms stay fixed."""
        positions = np.asarray(positions, dtype=float)
        point = np.asarray(point, dtype=float)
        cutoff = None if radius is None else float(radius)
        tree = cKDTree(positions) if tree is None else tree
        force = np.zeros(3, dtype=float)
        neighbor_indices = range(len(atoms)) if cutoff is None else tree.query_ball_point(point, r=cutoff)
        atom_i = atoms[index]
        charge_i = self.charges[atom_i]
        for j in neighbor_indices:
            if j == index:
                continue
            vector = point - positions[j]
            r = max(float(np.linalg.norm(vector)), self.small_eps)
            if cutoff is not None and r > cutoff:
                continue
            atom_j = atoms[j]
            p = self.params[_pair_key(atom_i, atom_j)]
            d_u_dr = -self.k_coul * charge_i * self.charges[atom_j] / (r**2)
            if r <= self.short_range_cutoff:
                d_u_dr += -p["A"] * p["B"] * np.exp(-p["B"] * r) + 6.0 * p["C"] / (r**7)
            force += -d_u_dr * vector / r
        return force
