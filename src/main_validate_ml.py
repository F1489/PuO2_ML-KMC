"""CLI for independent ML validation on newly generated events."""

from __future__ import annotations

import argparse
import json

from .io_xyz import read_xyz
from .potentials import PairPotentialPuO2
from .validation import validate_models_on_independent_events, validate_models_on_state_collection


def main() -> None:
    """Generate independent validation events and report ML quality."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--xyz", required=True, action="append")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-events", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-displacement", type=float, default=0.35)
    parser.add_argument("--min-distance", type=float, default=1.2)
    parser.add_argument("--order-biased-events", action="store_true")
    args = parser.parse_args()

    potential = PairPotentialPuO2()
    if len(args.xyz) > 1:
        states = []
        for xyz_path in args.xyz:
            atoms, positions, _ = read_xyz(xyz_path)
            states.append((xyz_path, atoms, positions))
        metrics = validate_models_on_state_collection(
            states,
            potential,
            args.model_dir,
            args.out_dir,
            n_events=args.n_events,
            seed=args.seed,
            max_displacement=args.max_displacement,
            min_distance=args.min_distance,
            order_biased_events=args.order_biased_events,
        )
    else:
        atoms, positions, _ = read_xyz(args.xyz[0])
        metrics = validate_models_on_independent_events(
            atoms,
            positions,
            potential,
            args.model_dir,
            args.out_dir,
            n_events=args.n_events,
            seed=args.seed,
            max_displacement=args.max_displacement,
            min_distance=args.min_distance,
            order_biased_events=args.order_biased_events,
        )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
