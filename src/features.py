"""Feature extraction for local PuO2 kMC events."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .events import Event
from .potentials import PairPotentialPuO2

RADII = (2.0, 3.0, 4.0, 5.0, 6.0)


def get_feature_names(n_neighbors: int = 16) -> list[str]:
    """Return stable feature names used by dataset, training, and kMC."""
    names = [
        "central_is_pu",
        "displacement_length",
        "displacement_x",
        "displacement_y",
        "displacement_z",
        "radius_before",
        "radius_after",
        "radial_displacement",
        "local_energy_before",
        "local_energy_after",
        "local_energy_delta",
        "force_before_x",
        "force_before_y",
        "force_before_z",
        "force_before_norm",
        "force_after_x",
        "force_after_y",
        "force_after_z",
        "force_after_norm",
        "force_projection_before",
    ]
    names += [f"pu_count_before_r_{r:.1f}" for r in RADII]
    names += [f"o_count_before_r_{r:.1f}" for r in RADII]
    names += [f"pu_count_after_r_{r:.1f}" for r in RADII]
    names += [f"o_count_after_r_{r:.1f}" for r in RADII]
    names += [f"nearest_any_before_{i + 1}" for i in range(n_neighbors)]
    names += [f"nearest_pu_before_{i + 1}" for i in range(n_neighbors)]
    names += [f"nearest_o_before_{i + 1}" for i in range(n_neighbors)]
    names += [f"nearest_any_after_{i + 1}" for i in range(n_neighbors)]
    names += [f"nearest_pu_after_{i + 1}" for i in range(n_neighbors)]
    names += [f"nearest_o_after_{i + 1}" for i in range(n_neighbors)]
    names += ["coord_deviation_before", "coord_deviation_after", "coord_deviation_delta"]
    return names


def _padded_sorted(values: np.ndarray, n_neighbors: int, fill_value: float) -> list[float]:
    sorted_values = sorted(float(x) for x in values)
    if len(sorted_values) < n_neighbors:
        sorted_values.extend([fill_value] * (n_neighbors - len(sorted_values)))
    return sorted_values[:n_neighbors]


def _coord_deviation(
    atoms: list[str],
    positions: np.ndarray,
    index: int,
    near_cutoff: float = 3.2,
    tree: cKDTree | None = None,
    query_position: np.ndarray | None = None,
) -> float:
    atom = atoms[index]
    tree = cKDTree(positions) if tree is None else tree
    point = positions[index] if query_position is None else np.asarray(query_position, dtype=float)
    neighbor_indices = [j for j in tree.query_ball_point(point, r=near_cutoff) if j != index]
    if atom == "Pu":
        n_o = sum(atoms[j] == "O" for j in neighbor_indices)
        return float(abs(n_o - 8))
    n_pu = sum(atoms[j] == "Pu" for j in neighbor_indices)
    return float(abs(n_pu - 4))


def _local_environment(
    atoms: list[str],
    positions: np.ndarray,
    index: int,
    point: np.ndarray,
    cutoff: float,
    n_neighbors: int,
    tree: cKDTree,
) -> list[float]:
    neighbor_indices = [j for j in tree.query_ball_point(point, r=cutoff) if j != index]
    if neighbor_indices:
        neighbor_positions = positions[neighbor_indices]
        neighbor_distances = np.linalg.norm(neighbor_positions - point, axis=1)
        neighbor_atoms = np.array([atoms[j] for j in neighbor_indices], dtype=object)
    else:
        neighbor_distances = np.asarray([], dtype=float)
        neighbor_atoms = np.asarray([], dtype=object)

    features: list[float] = []
    for radius in RADII:
        features.append(float(np.sum((neighbor_atoms == "Pu") & (neighbor_distances <= radius))))
    for radius in RADII:
        features.append(float(np.sum((neighbor_atoms == "O") & (neighbor_distances <= radius))))

    within_cutoff = neighbor_distances[neighbor_distances <= cutoff]
    pu_distances = neighbor_distances[(neighbor_atoms == "Pu") & (neighbor_distances <= cutoff)]
    o_distances = neighbor_distances[(neighbor_atoms == "O") & (neighbor_distances <= cutoff)]
    features.extend(_padded_sorted(within_cutoff, n_neighbors, cutoff))
    features.extend(_padded_sorted(pu_distances, n_neighbors, 999.0))
    features.extend(_padded_sorted(o_distances, n_neighbors, 999.0))
    return features


def make_event_features(
    atoms: list[str],
    positions: np.ndarray,
    event: Event,
    cutoff: float = 6.0,
    n_neighbors: int = 16,
    tree: cKDTree | None = None,
    potential: PairPotentialPuO2 | None = None,
    center: np.ndarray | None = None,
) -> np.ndarray:
    """Return a fixed-length local feature vector for one event."""
    positions = np.asarray(positions, dtype=float)
    idx = event.atom_index
    old_position = np.asarray(event.old_position, dtype=float)
    new_position = np.asarray(event.new_position, dtype=float)
    displacement = np.asarray(event.displacement, dtype=float)
    tree = cKDTree(positions) if tree is None else tree
    center = positions.mean(axis=0) if center is None else np.asarray(center, dtype=float)
    old_radial = old_position - center
    old_radius = float(np.linalg.norm(old_radial))
    new_radius = float(np.linalg.norm(new_position - center))
    radial_displacement = 0.0
    if old_radius > 0.0:
        radial_displacement = float(np.dot(displacement, old_radial / old_radius))
    if potential is None:
        local_energy_before = 0.0
        local_energy_after = 0.0
        force_before = np.zeros(3, dtype=float)
        force_after = np.zeros(3, dtype=float)
    else:
        local_energy_before, force_before = potential.local_energy_and_force_at_position(
            atoms, positions, idx, old_position, radius=cutoff, tree=tree
        )
        local_energy_after, force_after = potential.local_energy_and_force_at_position(
            atoms, positions, idx, new_position, radius=cutoff, tree=tree
        )
    displacement_norm = float(np.linalg.norm(displacement))
    force_before_norm = float(np.linalg.norm(force_before))
    force_after_norm = float(np.linalg.norm(force_after))
    force_projection_before = 0.0
    if displacement_norm > 0.0:
        force_projection_before = float(np.dot(force_before, displacement / displacement_norm))

    features: list[float] = [
        1.0 if atoms[idx] == "Pu" else 0.0,
        displacement_norm,
        float(displacement[0]),
        float(displacement[1]),
        float(displacement[2]),
        old_radius,
        new_radius,
        radial_displacement,
        float(local_energy_before),
        float(local_energy_after),
        float(local_energy_after - local_energy_before),
        float(force_before[0]),
        float(force_before[1]),
        float(force_before[2]),
        force_before_norm,
        float(force_after[0]),
        float(force_after[1]),
        float(force_after[2]),
        force_after_norm,
        force_projection_before,
    ]
    features.extend(_local_environment(atoms, positions, idx, old_position, cutoff, n_neighbors, tree))
    features.extend(_local_environment(atoms, positions, idx, new_position, cutoff, n_neighbors, tree))

    before_dev = _coord_deviation(atoms, positions, idx, tree=tree)
    after_dev = _coord_deviation(atoms, positions, idx, tree=tree, query_position=new_position)
    features.extend([before_dev, after_dev, after_dev - before_dev])
    return np.asarray(features, dtype=float)
