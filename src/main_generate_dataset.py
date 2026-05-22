"""CLI for generating an ML-kMC event dataset."""

from __future__ import annotations

import argparse

from .dataset import generate_dataset, generate_mixed_dataset
from .io_xyz import read_xyz
from .potentials import PairPotentialPuO2


def main() -> None:
    """Run dataset generation from command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--xyz", required=True, action="append")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-events", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-displacement", type=float, default=0.35)
    parser.add_argument("--min-distance", type=float, default=1.2)
    parser.add_argument("--order-biased-events", action="store_true")
    parser.add_argument("--mixed", action="store_true")
    parser.add_argument("--no-mixed-order-biased", action="store_true")
    parser.add_argument("--crystallization-energy-tolerance", type=float, default=1.0)
    parser.add_argument("--crystallization-order-threshold", type=float, default=0.01)
    args = parser.parse_args()
    potential = PairPotentialPuO2()
    if args.mixed or len(args.xyz) > 1:
        states = []
        for xyz_path in args.xyz:
            atoms, positions, _ = read_xyz(xyz_path)
            states.append((xyz_path, atoms, positions))
        generate_mixed_dataset(
            states,
            potential,
            args.n_events,
            args.out,
            seed=args.seed,
            max_displacement=args.max_displacement,
            min_distance=args.min_distance,
            include_order_biased=not args.no_mixed_order_biased,
            crystallization_energy_tolerance=args.crystallization_energy_tolerance,
            crystallization_order_threshold=args.crystallization_order_threshold,
        )
    else:
        atoms, positions, _ = read_xyz(args.xyz[0])
        generate_dataset(
            atoms,
            positions,
            potential,
            args.n_events,
            args.out,
            seed=args.seed,
            max_displacement=args.max_displacement,
            min_distance=args.min_distance,
            order_biased_events=args.order_biased_events,
            crystallization_energy_tolerance=args.crystallization_energy_tolerance,
            crystallization_order_threshold=args.crystallization_order_threshold,
        )


if __name__ == "__main__":
    main()
