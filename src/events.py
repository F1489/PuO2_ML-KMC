"""Off-lattice event generation for PuO2 clusters.

Displacements and positions are in angstrom (A).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .potentials import PairPotentialPuO2

MIN_PAIR_DISTANCES = {
    ("Pu", "O"): 1.9,
    ("O", "Pu"): 1.9,
    ("O", "O"): 2.0,
    ("Pu", "Pu"): 2.8,
}


@dataclass(frozen=True)
class Event:
    """A trial single-atom off-lattice move."""

    atom_index: int
    old_position: np.ndarray
    new_position: np.ndarray
    displacement: np.ndarray
    kind: str
    atom_indices: tuple[int, ...] | None = None
    new_positions_group: np.ndarray | None = None


def event_atom_indices(event: Event) -> tuple[int, ...]:
    """Return all atom indices moved by an event."""
    return event.atom_indices if event.atom_indices is not None else (event.atom_index,)


def event_new_positions(event: Event) -> np.ndarray:
    """Return new positions for all atoms moved by an event."""
    if event.new_positions_group is not None:
        return np.asarray(event.new_positions_group, dtype=float)
    return np.asarray([event.new_position], dtype=float)


def apply_event_to_positions(positions: np.ndarray, event: Event) -> np.ndarray:
    """Return a copy of positions after applying an event."""
    updated = np.asarray(positions, dtype=float).copy()
    updated[list(event_atom_indices(event))] = event_new_positions(event)
    return updated


def event_delta_energy(atoms: list[str], positions: np.ndarray, event: Event, potential: PairPotentialPuO2) -> float:
    """Return exact energy change for a single-atom or collective event."""
    moved_indices = event_atom_indices(event)
    if len(moved_indices) == 1:
        return potential.delta_energy_single_move(atoms, positions, event.atom_index, event.new_position)
    positions = np.asarray(positions, dtype=float)
    moved = set(moved_indices)
    new_by_index = {
        int(index): np.asarray(point, dtype=float)
        for index, point in zip(moved_indices, event_new_positions(event), strict=True)
    }
    delta = 0.0
    for i in range(len(atoms) - 1):
        i_moved = i in moved
        for j in range(i + 1, len(atoms)):
            if not i_moved and j not in moved:
                continue
            old_r = float(np.linalg.norm(positions[j] - positions[i]))
            new_i = new_by_index.get(i, positions[i])
            new_j = new_by_index.get(j, positions[j])
            new_r = float(np.linalg.norm(new_j - new_i))
            delta += potential.pair_energy(atoms[i], atoms[j], new_r) - potential.pair_energy(atoms[i], atoms[j], old_r)
    return float(delta)


def event_min_distance(positions: np.ndarray, event: Event, tree: cKDTree | None = None) -> float:
    """Return the nearest-neighbor distance in A after applying an event."""
    positions = np.asarray(positions, dtype=float)
    if len(positions) <= 1:
        return np.inf
    tree = cKDTree(positions) if tree is None else tree
    moved = set(event_atom_indices(event))
    min_distance = np.inf
    for new_position in event_new_positions(event):
        distances, indices = tree.query(new_position, k=min(len(positions), len(moved) + 2))
        distances = np.atleast_1d(distances)
        indices = np.atleast_1d(indices)
        for distance, index in zip(distances, indices, strict=False):
            if int(index) not in moved:
                min_distance = min(min_distance, float(distance))
                break
    new_positions = event_new_positions(event)
    if len(new_positions) > 1:
        for i in range(len(new_positions)):
            for j in range(i + 1, len(new_positions)):
                min_distance = min(min_distance, float(np.linalg.norm(new_positions[i] - new_positions[j])))
    return min_distance


def is_valid_event(
    atoms: list[str],
    positions: np.ndarray,
    event: Event,
    min_distance: float = 1.2,
    tree: cKDTree | None = None,
) -> bool:
    """Return True if an event does not create an unphysical close contact."""
    if event_min_distance(positions, event, tree=tree) < min_distance:
        return False
    search_radius = max(MIN_PAIR_DISTANCES.values())
    tree = cKDTree(positions) if tree is None else tree
    moved = set(event_atom_indices(event))
    moved_indices = event_atom_indices(event)
    new_positions = event_new_positions(event)
    for moved_index, new_position in zip(moved_indices, new_positions, strict=True):
        moved_atom = atoms[moved_index]
        for j in tree.query_ball_point(new_position, r=search_radius):
            if j in moved:
                continue
            threshold = MIN_PAIR_DISTANCES.get((moved_atom, atoms[j]), min_distance)
            if float(np.linalg.norm(positions[j] - new_position)) < threshold:
                return False
    for i, atom_i in enumerate(moved_indices):
        for j in range(i + 1, len(moved_indices)):
            atom_j = moved_indices[j]
            threshold = MIN_PAIR_DISTANCES.get((atoms[atom_i], atoms[atom_j]), min_distance)
            if float(np.linalg.norm(new_positions[i] - new_positions[j])) < threshold:
                return False
    return True


def _random_displacements(n_events: int, max_displacement: float, rng: np.random.Generator) -> np.ndarray:
    directions = rng.normal(size=(n_events, 3))
    norms = np.linalg.norm(directions, axis=1)
    norms[norms == 0.0] = 1.0
    directions = directions / norms[:, None]
    lengths = rng.uniform(0.0, max_displacement, size=n_events)
    return directions * lengths[:, None]


def random_single_atom_moves(
    atoms: list[str],
    positions: np.ndarray,
    n_events: int,
    max_displacement: float,
    rng: np.random.Generator,
    min_distance: float = 1.2,
    tree: cKDTree | None = None,
) -> list[Event]:
    """Generate random single-atom trial moves with close-contact filtering."""
    return _random_weighted_moves(
        atoms,
        positions,
        n_events,
        max_displacement,
        rng,
        min_distance=min_distance,
        tree=tree,
        weights=None,
        kind="random_bulk",
    )


def _random_weighted_moves(
    atoms: list[str],
    positions: np.ndarray,
    n_events: int,
    max_displacement: float,
    rng: np.random.Generator,
    min_distance: float,
    tree: cKDTree | None,
    weights: np.ndarray | None,
    kind: str,
) -> list[Event]:
    positions = np.asarray(positions, dtype=float)
    tree = cKDTree(positions) if tree is None else tree
    events: list[Event] = []
    attempts = 0
    while len(events) < n_events and attempts < max(20, 10 * n_events):
        attempts += 1
        idx = int(rng.choice(len(positions), p=weights)) if weights is not None else int(rng.integers(0, len(positions)))
        disp = _random_displacements(1, max_displacement, rng)[0]
        event = Event(idx, positions[idx].copy(), positions[idx] + disp, disp.copy(), kind)
        if is_valid_event(atoms, positions, event, min_distance=min_distance, tree=tree):
            events.append(event)
    return events


def random_surface_moves(
    atoms: list[str],
    positions: np.ndarray,
    n_events: int,
    max_displacement: float,
    rng: np.random.Generator,
    min_distance: float = 1.2,
    tree: cKDTree | None = None,
) -> list[Event]:
    """Generate isotropic random moves biased to surface atoms."""
    positions = np.asarray(positions, dtype=float)
    center = positions.mean(axis=0)
    radii = np.linalg.norm(positions - center, axis=1)
    weights = np.maximum(radii - np.percentile(radii, 60), 0.0) + 1.0e-6
    weights = weights / weights.sum()
    return _random_weighted_moves(
        atoms,
        positions,
        n_events,
        max_displacement,
        rng,
        min_distance=min_distance,
        tree=tree,
        weights=weights,
        kind="random_surface",
    )


def surface_biased_moves(
    atoms: list[str],
    positions: np.ndarray,
    n_events: int,
    max_displacement: float,
    rng: np.random.Generator,
    min_distance: float = 1.2,
    tree: cKDTree | None = None,
) -> list[Event]:
    """Generate likely surface-atom moves with close-contact filtering."""
    positions = np.asarray(positions, dtype=float)
    tree = cKDTree(positions) if tree is None else tree
    center = positions.mean(axis=0)
    radii = np.linalg.norm(positions - center, axis=1)
    weights = np.maximum(radii - np.percentile(radii, 30), 0.0) + 1.0e-6
    weights = weights / weights.sum()
    events: list[Event] = []
    attempts = 0
    while len(events) < n_events and attempts < max(20, 10 * n_events):
        attempts += 1
        idx = int(rng.choice(len(positions), p=weights))
        radial = positions[idx] - center
        radial_norm = np.linalg.norm(radial)
        if radial_norm == 0.0:
            radial_norm = 1.0
        inward_bias = -0.35 * max_displacement * radial / radial_norm
        disp = 0.7 * _random_displacements(1, max_displacement, rng)[0] + inward_bias
        event = Event(idx, positions[idx].copy(), positions[idx] + disp, disp.copy(), "surface")
        if is_valid_event(atoms, positions, event, min_distance=min_distance, tree=tree):
            events.append(event)
    return events


def coordination_improving_moves(
    atoms: list[str],
    positions: np.ndarray,
    n_events: int,
    max_displacement: float,
    rng: np.random.Generator,
    min_distance: float = 1.2,
    tree: cKDTree | None = None,
) -> list[Event]:
    """Generate heuristic moves toward opposite-species centroids with filtering."""
    positions = np.asarray(positions, dtype=float)
    tree = cKDTree(positions) if tree is None else tree
    events: list[Event] = []
    attempts = 0
    while len(events) < n_events and attempts < max(20, 10 * n_events):
        attempts += 1
        idx = int(rng.integers(0, len(positions)))
        atom = atoms[idx]
        target_type = "O" if atom == "Pu" else "Pu"
        deltas = positions - positions[idx]
        distances = np.linalg.norm(deltas, axis=1)
        mask = np.array([(a == target_type) for a in atoms]) & (distances > 0.0) & (distances < 5.0)
        if np.any(mask):
            centroid = positions[mask].mean(axis=0)
            direction = centroid - positions[idx]
            norm = np.linalg.norm(direction)
            if norm > 0.0:
                disp = direction / norm * rng.uniform(0.0, max_displacement)
            else:
                disp = _random_displacements(1, max_displacement, rng)[0]
        else:
            disp = _random_displacements(1, max_displacement, rng)[0]
        event = Event(idx, positions[idx].copy(), positions[idx] + disp, disp.copy(), "coordination")
        if is_valid_event(atoms, positions, event, min_distance=min_distance, tree=tree):
            events.append(event)
    return events


def force_relaxation_moves(
    atoms: list[str],
    positions: np.ndarray,
    n_events: int,
    max_displacement: float,
    rng: np.random.Generator,
    potential: PairPotentialPuO2,
    min_distance: float = 1.2,
    tree: cKDTree | None = None,
) -> list[Event]:
    """Generate local relaxation moves along the negative energy gradient."""
    positions = np.asarray(positions, dtype=float)
    tree = cKDTree(positions) if tree is None else tree
    events: list[Event] = []
    attempts = 0
    while len(events) < n_events and attempts < max(20, 10 * n_events):
        attempts += 1
        idx = int(rng.integers(0, len(positions)))
        force = potential.force_on_atom(atoms, positions, idx, tree=tree)
        norm = np.linalg.norm(force)
        if norm == 0.0 or not np.isfinite(norm):
            disp = _random_displacements(1, max_displacement, rng)[0]
        else:
            disp = force / norm * rng.uniform(0.25 * max_displacement, max_displacement)
            disp += 0.15 * _random_displacements(1, max_displacement, rng)[0]
        event = Event(idx, positions[idx].copy(), positions[idx] + disp, disp.copy(), "relaxation")
        if is_valid_event(atoms, positions, event, min_distance=min_distance, tree=tree):
            events.append(event)
    return events


def surface_compression_moves(
    atoms: list[str],
    positions: np.ndarray,
    n_events: int,
    max_displacement: float,
    rng: np.random.Generator,
    min_distance: float = 1.2,
    tree: cKDTree | None = None,
) -> list[Event]:
    """Generate inward surface-compression moves for under-relaxed clusters."""
    positions = np.asarray(positions, dtype=float)
    tree = cKDTree(positions) if tree is None else tree
    center = positions.mean(axis=0)
    radii = np.linalg.norm(positions - center, axis=1)
    weights = np.maximum(radii - np.percentile(radii, 55), 0.0) + 1.0e-6
    weights = weights / weights.sum()
    events: list[Event] = []
    attempts = 0
    while len(events) < n_events and attempts < max(20, 10 * n_events):
        attempts += 1
        idx = int(rng.choice(len(positions), p=weights))
        direction = center - positions[idx]
        norm = np.linalg.norm(direction)
        if norm == 0.0:
            disp = _random_displacements(1, max_displacement, rng)[0]
        else:
            disp = direction / norm * rng.uniform(0.1 * max_displacement, max_displacement)
        event = Event(idx, positions[idx].copy(), positions[idx] + disp, disp.copy(), "surface_compression")
        if is_valid_event(atoms, positions, event, min_distance=min_distance, tree=tree):
            events.append(event)
    return events


def _opposite_coordination_error(atoms: list[str], positions: np.ndarray, cutoff: float = 3.2) -> np.ndarray:
    tree = cKDTree(positions)
    errors = np.zeros(len(atoms), dtype=float)
    for i, atom in enumerate(atoms):
        target = "O" if atom == "Pu" else "Pu"
        ideal = 8 if atom == "Pu" else 4
        neighbors = tree.query_ball_point(positions[i], r=cutoff)
        count = sum(atoms[j] == target for j in neighbors if j != i)
        errors[i] = abs(count - ideal)
    return errors


def snap_to_fluorite_site_moves(
    atoms: list[str],
    positions: np.ndarray,
    n_events: int,
    max_displacement: float,
    rng: np.random.Generator,
    min_distance: float = 1.2,
    tree: cKDTree | None = None,
    errors: np.ndarray | None = None,
) -> list[Event]:
    """Move poorly coordinated atoms toward local fluorite-like opposite-species centroids."""
    positions = np.asarray(positions, dtype=float)
    tree = cKDTree(positions) if tree is None else tree
    errors = _opposite_coordination_error(atoms, positions) if errors is None else np.asarray(errors, dtype=float)
    weights = errors + 0.05
    weights = weights / weights.sum()
    events: list[Event] = []
    attempts = 0
    while len(events) < n_events and attempts < max(30, 15 * n_events):
        attempts += 1
        idx = int(rng.choice(len(positions), p=weights))
        atom = atoms[idx]
        target_type = "O" if atom == "Pu" else "Pu"
        ideal_distance = 2.37 if atom == "Pu" else 2.37
        neighbors = [
            j
            for j in tree.query_ball_point(positions[idx], r=4.2)
            if j != idx and atoms[j] == target_type
        ]
        if not neighbors:
            continue
        centroid = positions[neighbors].mean(axis=0)
        direction = centroid - positions[idx]
        norm = np.linalg.norm(direction)
        if norm == 0.0:
            continue
        target = centroid - direction / norm * ideal_distance
        disp = target - positions[idx]
        disp_norm = np.linalg.norm(disp)
        if disp_norm > max_displacement:
            disp = disp / disp_norm * rng.uniform(0.35 * max_displacement, max_displacement)
        event = Event(idx, positions[idx].copy(), positions[idx] + disp, disp.copy(), "snap_to_fluorite_site")
        if is_valid_event(atoms, positions, event, min_distance=min_distance, tree=tree):
            events.append(event)
    return events


def growth_front_moves(
    atoms: list[str],
    positions: np.ndarray,
    n_events: int,
    max_displacement: float,
    rng: np.random.Generator,
    min_distance: float = 1.2,
    tree: cKDTree | None = None,
    errors: np.ndarray | None = None,
) -> list[Event]:
    """Generate moves on the boundary between locally ordered and disordered atoms."""
    positions = np.asarray(positions, dtype=float)
    tree = cKDTree(positions) if tree is None else tree
    errors = _opposite_coordination_error(atoms, positions) if errors is None else np.asarray(errors, dtype=float)
    core = np.flatnonzero(errors <= 1.0)
    if len(core) == 0:
        return snap_to_fluorite_site_moves(atoms, positions, n_events, max_displacement, rng, min_distance, tree, errors)
    front_mask = np.zeros(len(atoms), dtype=bool)
    for i in core:
        for j in tree.query_ball_point(positions[i], r=4.2):
            if errors[j] > 1.0:
                front_mask[j] = True
    front = np.flatnonzero(front_mask)
    if len(front) == 0:
        return snap_to_fluorite_site_moves(atoms, positions, n_events, max_displacement, rng, min_distance, tree, errors)
    weights = errors[front] + 0.1
    weights = weights / weights.sum()
    events: list[Event] = []
    attempts = 0
    while len(events) < n_events and attempts < max(30, 15 * n_events):
        attempts += 1
        idx = int(rng.choice(front, p=weights))
        nearby_core = [j for j in tree.query_ball_point(positions[idx], r=4.2) if errors[j] <= 1.0]
        if nearby_core:
            target = positions[nearby_core].mean(axis=0)
            direction = target - positions[idx]
            norm = np.linalg.norm(direction)
            disp = direction / norm * rng.uniform(0.1 * max_displacement, max_displacement) if norm > 0 else _random_displacements(1, max_displacement, rng)[0]
        else:
            disp = _random_displacements(1, max_displacement, rng)[0]
        event = Event(idx, positions[idx].copy(), positions[idx] + disp, disp.copy(), "growth_front")
        if is_valid_event(atoms, positions, event, min_distance=min_distance, tree=tree):
            events.append(event)
    return events


def local_cluster_affine_moves(
    atoms: list[str],
    positions: np.ndarray,
    n_events: int,
    max_displacement: float,
    rng: np.random.Generator,
    min_distance: float = 1.2,
    tree: cKDTree | None = None,
    radius: float = 3.6,
    errors: np.ndarray | None = None,
) -> list[Event]:
    """Generate collective local translation/rotation/scale moves for small neighbor clusters."""
    positions = np.asarray(positions, dtype=float)
    tree = cKDTree(positions) if tree is None else tree
    errors = _opposite_coordination_error(atoms, positions) if errors is None else np.asarray(errors, dtype=float)
    weights = errors + 0.2
    weights = weights / weights.sum()
    events: list[Event] = []
    attempts = 0
    while len(events) < n_events and attempts < max(30, 15 * n_events):
        attempts += 1
        center_idx = int(rng.choice(len(positions), p=weights))
        indices = tuple(int(i) for i in tree.query_ball_point(positions[center_idx], r=radius))
        if len(indices) < 2:
            continue
        if len(indices) > 10:
            others = [i for i in indices if i != center_idx]
            picked = rng.choice(others, size=min(9, len(others)), replace=False)
            indices = (center_idx, *(int(i) for i in picked))
        group = positions[list(indices)]
        center = group.mean(axis=0)
        mode = int(rng.integers(0, 3))
        if mode == 0:
            translation = _random_displacements(1, 0.45 * max_displacement, rng)[0]
            new_group = group + translation
        elif mode == 1:
            scale = rng.uniform(0.97, 1.03)
            new_group = center + (group - center) * scale
        else:
            axis = rng.normal(size=3)
            axis_norm = np.linalg.norm(axis)
            if axis_norm == 0.0:
                continue
            axis = axis / axis_norm
            angle = rng.uniform(-0.08, 0.08)
            vectors = group - center
            new_group = (
                center
                + vectors * np.cos(angle)
                + np.cross(axis, vectors) * np.sin(angle)
                + axis * np.dot(vectors, axis)[:, None] * (1.0 - np.cos(angle))
            )
        displacement = new_group[0] - group[0]
        event = Event(
            center_idx,
            positions[center_idx].copy(),
            new_group[list(indices).index(center_idx)].copy(),
            displacement.copy(),
            "local_cluster_affine",
            atom_indices=indices,
            new_positions_group=new_group.copy(),
        )
        if is_valid_event(atoms, positions, event, min_distance=min_distance, tree=tree):
            events.append(event)
    return events


def generate_candidate_events_with_diagnostics(
    atoms: list[str],
    positions: np.ndarray,
    n_events: int,
    max_displacement: float = 0.35,
    rng: np.random.Generator | None = None,
    min_distance: float = 1.2,
    potential: PairPotentialPuO2 | None = None,
    order_biased_events: bool = False,
    frozen_atom_indices: set[int] | None = None,
) -> tuple[list[Event], dict[str, int]]:
    """Generate valid candidate events and return rejection diagnostics."""
    rng = np.random.default_rng() if rng is None else rng
    positions = np.asarray(positions, dtype=float)
    tree = cKDTree(positions)
    events: list[Event] = []
    diagnostics = {"requested": n_events, "generated": 0, "rejected": 0}
    frozen = set() if frozen_atom_indices is None else set(frozen_atom_indices)
    order_errors = _opposite_coordination_error(atoms, positions) if order_biased_events else None
    attempts = 0
    while len(events) < n_events and attempts < 10:
        attempts += 1
        missing = n_events - len(events)
        n_random_bulk = max(0, missing // 10)
        n_random_surface = max(1, missing // 10)
        n_surface = max(1, missing // 5)
        n_coord = max(1, missing // 4)
        n_snap = max(1, missing // 8) if order_biased_events else 0
        n_front = max(1, missing // 8) if order_biased_events else 0
        n_cluster = max(1, missing // 10) if order_biased_events else 0
        n_compress = max(1, missing // 5)
        n_relax = max(
            0,
            missing
            - n_random_bulk
            - n_random_surface
            - n_surface
            - n_coord
            - n_snap
            - n_front
            - n_cluster
            - n_compress,
        )
        before = len(events)
        events.extend(
            random_single_atom_moves(atoms, positions, n_random_bulk, max_displacement, rng, min_distance, tree)
        )
        events.extend(random_surface_moves(atoms, positions, n_random_surface, max_displacement, rng, min_distance, tree))
        events.extend(surface_biased_moves(atoms, positions, n_surface, max_displacement, rng, min_distance, tree))
        events.extend(coordination_improving_moves(atoms, positions, n_coord, max_displacement, rng, min_distance, tree))
        if order_biased_events:
            events.extend(snap_to_fluorite_site_moves(atoms, positions, n_snap, max_displacement, rng, min_distance, tree, order_errors))
            events.extend(growth_front_moves(atoms, positions, n_front, max_displacement, rng, min_distance, tree, order_errors))
            events.extend(local_cluster_affine_moves(atoms, positions, n_cluster, max_displacement, rng, min_distance, tree, errors=order_errors))
        events.extend(surface_compression_moves(atoms, positions, n_compress, max_displacement, rng, min_distance, tree))
        if potential is not None and n_relax > 0:
            events.extend(
                force_relaxation_moves(atoms, positions, n_relax, max_displacement, rng, potential, min_distance, tree)
            )
        else:
            events.extend(random_single_atom_moves(atoms, positions, n_relax, max_displacement, rng, min_distance, tree))
        if frozen:
            events = [event for event in events if not (set(event_atom_indices(event)) & frozen)]
        diagnostics["rejected"] += max(0, missing - (len(events) - before))
    if len(events) < n_events:
        raise RuntimeError(
            f"Could only generate {len(events)} valid events out of {n_events}; "
            "try reducing min_distance or max_displacement"
        )
    rng.shuffle(events)
    events = events[:n_events]
    diagnostics["generated"] = len(events)
    return events, diagnostics


def generate_candidate_events(
    atoms: list[str],
    positions: np.ndarray,
    n_events: int,
    max_displacement: float = 0.35,
    rng: np.random.Generator | None = None,
    min_distance: float = 1.2,
    potential: PairPotentialPuO2 | None = None,
    order_biased_events: bool = False,
    frozen_atom_indices: set[int] | None = None,
) -> list[Event]:
    """Generate a mixed pool of valid random, surface, and coordination-biased moves."""
    events, _ = generate_candidate_events_with_diagnostics(
        atoms,
        positions,
        n_events,
        max_displacement=max_displacement,
        rng=rng,
        min_distance=min_distance,
        potential=potential,
        order_biased_events=order_biased_events,
        frozen_atom_indices=frozen_atom_indices,
    )
    return events
