"""Close-contact repair utilities before production kMC runs."""

from __future__ import annotations

import numpy as np

from .events import MIN_PAIR_DISTANCES
from .potentials import PairPotentialPuO2


def close_contact_penalty(atoms: list[str], positions: np.ndarray) -> float:
    """Return squared violation penalty for pair-specific minimum distances."""
    positions = np.asarray(positions, dtype=float)
    penalty = 0.0
    for i in range(len(atoms) - 1):
        for j in range(i + 1, len(atoms)):
            threshold = MIN_PAIR_DISTANCES.get((atoms[i], atoms[j]), MIN_PAIR_DISTANCES.get((atoms[j], atoms[i]), 0.0))
            if threshold <= 0.0:
                continue
            distance = float(np.linalg.norm(positions[j] - positions[i]))
            if distance < threshold:
                penalty += (threshold - distance) ** 2
    return float(penalty)


def closest_violating_pair(atoms: list[str], positions: np.ndarray) -> tuple[int, int] | None:
    """Return the most violated pair, or None if all pair distances are valid."""
    positions = np.asarray(positions, dtype=float)
    best_pair: tuple[int, int] | None = None
    best_violation = 0.0
    for i in range(len(atoms) - 1):
        for j in range(i + 1, len(atoms)):
            threshold = MIN_PAIR_DISTANCES.get((atoms[i], atoms[j]), MIN_PAIR_DISTANCES.get((atoms[j], atoms[i]), 0.0))
            if threshold <= 0.0:
                continue
            distance = float(np.linalg.norm(positions[j] - positions[i]))
            violation = threshold - distance
            if violation > best_violation:
                best_violation = violation
                best_pair = (i, j)
    return best_pair


def repair_close_contacts(
    atoms: list[str],
    positions: np.ndarray,
    potential: PairPotentialPuO2,
    steps: int = 200,
    max_displacement: float = 0.08,
    seed: int = 42,
    penalty_weight: float = 5000.0,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Move atoms away from too-close pairs using an energy-plus-penalty objective."""
    positions = np.asarray(positions, dtype=float).copy()
    rng = np.random.default_rng(seed)
    history: list[dict[str, float]] = []
    current_energy = potential.total_energy(atoms, positions)
    current_penalty = close_contact_penalty(atoms, positions)
    current_objective = current_energy + penalty_weight * current_penalty

    for step in range(steps):
        pair = closest_violating_pair(atoms, positions)
        if pair is None:
            break
        i, j = pair
        moved = i if rng.random() < 0.5 else j
        other = j if moved == i else i
        direction = positions[moved] - positions[other]
        norm = float(np.linalg.norm(direction))
        if norm <= 1.0e-12:
            direction = rng.normal(size=3)
            norm = float(np.linalg.norm(direction))
        direction = direction / max(norm, 1.0e-12)
        displacement = direction * rng.uniform(0.25 * max_displacement, max_displacement)
        displacement += 0.2 * rng.normal(size=3) * max_displacement

        trial = positions.copy()
        trial[moved] = trial[moved] + displacement
        trial_energy = potential.total_energy(atoms, trial)
        trial_penalty = close_contact_penalty(atoms, trial)
        trial_objective = trial_energy + penalty_weight * trial_penalty
        accepted = trial_objective < current_objective
        if accepted:
            positions = trial
            current_energy = trial_energy
            current_penalty = trial_penalty
            current_objective = trial_objective
        history.append(
            {
                "step": float(step),
                "accepted": float(accepted),
                "energy": float(current_energy),
                "close_contact_penalty": float(current_penalty),
                "objective": float(current_objective),
            }
        )
    return positions, history
