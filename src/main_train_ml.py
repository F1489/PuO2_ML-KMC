"""CLI for training ML models from a generated dataset."""

from __future__ import annotations

import argparse
import json

from .ml_model import train_models


def main() -> None:
    """Train models and print metrics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--classifier-target",
        choices=["is_energy_lowering", "is_crystallizing_event"],
        default="is_energy_lowering",
    )
    parser.add_argument("--no-external-models", action="store_true")
    parser.add_argument("--ensemble-size", type=int, default=3)
    args = parser.parse_args()
    metrics = train_models(
        args.dataset,
        args.model_dir,
        seed=args.seed,
        classifier_target=args.classifier_target,
        use_external_models=not args.no_external_models,
        ensemble_size=args.ensemble_size,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
