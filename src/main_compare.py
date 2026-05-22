"""Compare ML-kMC relaxation speed for two PuO2 crystallites."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from .analysis import fluorite_order_score
from .dataset import generate_dataset
from .io_xyz import read_xyz
from .kmc import MLKMC
from .ml_model import load_models, train_models
from .potentials import PairPotentialPuO2


def _prepare_models(atoms, positions, out_dir: Path, n_events: int, seed: int, min_distance: float):
    dataset_csv = out_dir / "dataset.csv"
    model_dir = out_dir / "models"
    generate_dataset(atoms, positions, PairPotentialPuO2(), n_events, dataset_csv, seed=seed, min_distance=min_distance)
    train_models(dataset_csv, model_dir, seed=seed)
    return load_models(model_dir)[:2]


def _run_one(
    xyz: str,
    steps: int,
    out_dir: Path,
    n_events: int,
    seed: int,
    model_dir: str | None,
    min_distance: float,
    classifier_rate_weight: float,
) -> dict[str, float]:
    atoms, positions, _ = read_xyz(xyz)
    potential = PairPotentialPuO2()
    initial_energy = potential.total_energy(atoms, positions)
    if model_dir:
        regressor, classifier, _ = load_models(model_dir)
    else:
        regressor, classifier = _prepare_models(atoms, positions, out_dir / "autofit", n_events, seed, min_distance)
    kmc = MLKMC(
        atoms,
        positions,
        potential,
        regressor,
        classifier,
        min_distance=min_distance,
        classifier_rate_weight=classifier_rate_weight,
        seed=seed,
    )
    start = time.perf_counter()
    kmc.run(steps, out_dir=out_dir / "kmc")
    wall_time = time.perf_counter() - start
    final_energy = potential.total_energy(kmc.atoms, kmc.positions)
    return {
        "N atoms": float(len(atoms)),
        "initial energy": initial_energy,
        "final energy": final_energy,
        "relative energy decrease": (initial_energy - final_energy) / abs(initial_energy) if initial_energy else 0.0,
        "wall time": wall_time,
        "steps per second": steps / wall_time if wall_time > 0.0 else 0.0,
        "final fluorite_order_score": fluorite_order_score(kmc.atoms, kmc.positions),
    }


def main() -> None:
    """Run comparison for two XYZ files and print a compact table."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--xyz1", required=True)
    parser.add_argument("--xyz2", required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--n-events", type=int, default=1000)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--out-dir", default="data/output/compare")
    parser.add_argument("--min-distance", type=float, default=1.2)
    parser.add_argument("--classifier-rate-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    rows = [
        (
            "xyz1",
            _run_one(
                args.xyz1,
                args.steps,
                out_dir / "xyz1",
                args.n_events,
                args.seed,
                args.model_dir,
                args.min_distance,
                args.classifier_rate_weight,
            ),
        ),
        (
            "xyz2",
            _run_one(
                args.xyz2,
                args.steps,
                out_dir / "xyz2",
                args.n_events,
                args.seed + 1,
                args.model_dir,
                args.min_distance,
                args.classifier_rate_weight,
            ),
        ),
    ]
    headers = ["case", *rows[0][1].keys()]
    print("\t".join(headers))
    for label, row in rows:
        print("\t".join([label, *[f"{value:.6g}" for value in row.values()]]))


if __name__ == "__main__":
    main()
