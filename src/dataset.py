"""Dataset generation for ML-accelerated kMC."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback for minimal environments
    tqdm = lambda iterable, **_: iterable

from .analysis import bulk_fluorite_order_score, defect_counts, fluorite_order_score
from .events import apply_event_to_positions, event_delta_energy, generate_candidate_events
from .features import get_feature_names, make_event_features
from .potentials import PairPotentialPuO2


def generate_dataset(
    atoms: list[str],
    positions,
    potential: PairPotentialPuO2,
    n_events: int,
    output_csv: str | Path,
    seed: int = 42,
    max_displacement: float = 0.35,
    min_distance: float = 1.2,
    order_biased_events: bool = False,
    crystallization_energy_tolerance: float = 1.0,
    crystallization_order_threshold: float = 0.01,
) -> pd.DataFrame:
    """Generate event features and exact Delta E labels, then save them to CSV."""
    import numpy as np

    rng = np.random.default_rng(seed)
    feature_names = get_feature_names()
    rows: list[dict[str, object]] = []
    current_order = fluorite_order_score(atoms, positions)
    current_bulk_order = bulk_fluorite_order_score(atoms, positions)
    current_coord_error = defect_counts(atoms, positions)["mean_abs_coordination_error"]
    for _ in tqdm(range(n_events), desc="Generating dataset"):
        event = generate_candidate_events(
            atoms,
            positions,
            1,
            max_displacement=max_displacement,
            rng=rng,
            min_distance=min_distance,
            potential=potential,
            order_biased_events=order_biased_events,
        )[0]
        features = make_event_features(atoms, positions, event, potential=potential)
        delta_e = event_delta_energy(atoms, positions, event, potential)
        trial_positions = apply_event_to_positions(positions, event)
        trial_order = fluorite_order_score(atoms, trial_positions)
        trial_bulk_order = bulk_fluorite_order_score(atoms, trial_positions)
        trial_coord_error = defect_counts(atoms, trial_positions)["mean_abs_coordination_error"]
        delta_order = trial_order - current_order
        delta_bulk_order = trial_bulk_order - current_bulk_order
        delta_coord_error = trial_coord_error - current_coord_error
        row = dict(zip(feature_names, features, strict=True))
        row.update(
            {
                "delta_E": delta_e,
                "delta_fluorite_order": delta_order,
                "delta_bulk_order": delta_bulk_order,
                "delta_coordination_error": delta_coord_error,
                "is_energy_lowering": int(delta_e < 0.0),
                "is_crystallizing_event": int(
                    delta_e < crystallization_energy_tolerance and delta_order > crystallization_order_threshold
                ),
                "atom_type": atoms[event.atom_index],
                "event_kind": event.kind,
            }
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def generate_mixed_dataset(
    states: list[tuple[str, list[str], object]],
    potential: PairPotentialPuO2,
    n_events: int,
    output_csv: str | Path,
    seed: int = 42,
    max_displacement: float = 0.35,
    min_distance: float = 1.2,
    include_order_biased: bool = True,
    crystallization_energy_tolerance: float = 1.0,
    crystallization_order_threshold: float = 0.01,
) -> pd.DataFrame:
    """Generate a single training table from multiple structural states."""
    if not states:
        raise ValueError("At least one structural state is required for mixed dataset generation")

    import numpy as np

    rng = np.random.default_rng(seed)
    event_modes = [False, True] if include_order_biased else [False]
    jobs: list[tuple[str, list[str], object, bool, int]] = []
    for state_name, atoms, positions in states:
        for order_biased in event_modes:
            jobs.append((state_name, atoms, positions, order_biased, int(rng.integers(0, 2**31 - 1))))

    base_count = n_events // len(jobs)
    remainder = n_events % len(jobs)
    frames: list[pd.DataFrame] = []
    for job_index, (state_name, atoms, positions, order_biased, job_seed) in enumerate(jobs):
        count = base_count + (1 if job_index < remainder else 0)
        if count <= 0:
            continue
        frame = generate_dataset(
            atoms,
            positions,
            potential,
            count,
            output_csv=Path(output_csv).with_suffix(f".part_{job_index:03d}.csv"),
            seed=job_seed,
            max_displacement=max_displacement,
            min_distance=min_distance,
            order_biased_events=order_biased,
            crystallization_energy_tolerance=crystallization_energy_tolerance,
            crystallization_order_threshold=crystallization_order_threshold,
        )
        frame["source_state"] = state_name
        frame["order_biased_source"] = int(order_biased)
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    for part_path in output_csv.parent.glob(f"{output_csv.stem}.part_*.csv"):
        part_path.unlink(missing_ok=True)
    return df
