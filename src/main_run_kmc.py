"""CLI for running ML-kMC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import pandas as pd

from .analysis import write_summary_json
from .io_xyz import read_xyz
from .kmc import MLKMC
from .ml_model import load_models
from .potentials import PairPotentialPuO2
from .repair import repair_close_contacts
from .visualization import save_all_plots


def main() -> None:
    """Run ML-kMC from command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--xyz", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--temperature", type=float, default=3600.0)
    parser.add_argument("--n-candidates-per-step", type=int, default=64)
    parser.add_argument("--max-displacement", type=float, default=0.35)
    parser.add_argument("--min-distance", type=float, default=1.2)
    parser.add_argument("--exact-check-interval", type=int, default=50)
    parser.add_argument("--classifier-rate-weight", type=float, default=0.5)
    parser.add_argument("--reject-exact-delta-above", type=float, default=25.0)
    parser.add_argument("--max-exact-delta-e-above", type=float, default=None)
    parser.add_argument("--exact-shortlist-size", type=int, default=8)
    parser.add_argument("--uncertainty-shortlist-size", type=int, default=4)
    parser.add_argument("--active-learning-error-threshold", type=float, default=2.0)
    parser.add_argument("--no-adaptive-displacement", action="store_true")
    parser.add_argument("--order-bias-lambda", type=float, default=0.0)
    parser.add_argument("--order-metric", choices=["fluorite", "bulk", "soft"], default="fluorite")
    parser.add_argument("--order-biased-events", action="store_true")
    parser.add_argument("--anneal-final-temperature", type=float, default=None)
    parser.add_argument("--anneal-tau", type=float, default=0.0)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--active-learning-retrain-interval", type=int, default=0)
    parser.add_argument("--base-dataset", default=None)
    parser.add_argument("--save-xyz-interval", type=int, default=0)
    parser.add_argument("--pre-relaxation-steps", type=int, default=0)
    parser.add_argument("--pre-relaxation-max-displacement", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    atoms, positions, _ = read_xyz(args.xyz)
    initial_positions = positions.copy()
    positions_after_repair = positions.copy()
    potential = PairPotentialPuO2()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.pre_relaxation_steps > 0:
        positions, repair_history = repair_close_contacts(
            atoms,
            positions,
            potential,
            steps=args.pre_relaxation_steps,
            max_displacement=args.pre_relaxation_max_displacement,
            seed=args.seed,
        )
        positions_after_repair = positions.copy()
        pd.DataFrame(repair_history).to_csv(out_dir / "pre_relaxation_history.csv", index=False)
    regressor, classifier, _ = load_models(args.model_dir)
    kmc = MLKMC(
        atoms,
        positions,
        potential,
        regressor,
        classifier,
        temperature=args.temperature,
        n_candidates_per_step=args.n_candidates_per_step,
        max_displacement=args.max_displacement,
        min_distance=args.min_distance,
        exact_check_interval=args.exact_check_interval,
        classifier_rate_weight=args.classifier_rate_weight,
        reject_exact_delta_above=args.reject_exact_delta_above,
        max_exact_delta_e_above=args.max_exact_delta_e_above,
        exact_shortlist_size=args.exact_shortlist_size,
        uncertainty_shortlist_size=args.uncertainty_shortlist_size,
        active_learning_error_threshold=args.active_learning_error_threshold,
        adaptive_displacement=not args.no_adaptive_displacement,
        order_bias_lambda=args.order_bias_lambda,
        order_metric=args.order_metric,
        order_biased_events=args.order_biased_events,
        anneal_final_temperature=args.anneal_final_temperature,
        anneal_tau=args.anneal_tau,
        n_jobs=args.n_jobs,
        seed=args.seed,
    )
    start = perf_counter()
    kmc.run(
        args.steps,
        out_dir=out_dir,
        active_learning_retrain_interval=args.active_learning_retrain_interval,
        model_dir=args.model_dir,
        base_dataset_csv=args.base_dataset,
        save_xyz_interval=args.save_xyz_interval,
        seed=args.seed,
    )
    elapsed = perf_counter() - start
    save_all_plots(out_dir / "history.csv", atoms, initial_positions, kmc.atoms, kmc.positions, out_dir)
    summary = write_summary_json(
        out_dir / "summary.json",
        atoms,
        initial_positions,
        kmc.atoms,
        kmc.positions,
        potential,
        positions_after_repair=positions_after_repair,
    )
    summary.update(
        {
            "wall_time_s": elapsed,
            "steps_per_second": args.steps / elapsed if elapsed > 0 else 0.0,
            "n_steps": args.steps,
            "n_candidates_per_step": args.n_candidates_per_step,
            "exact_shortlist_size": args.exact_shortlist_size,
            "uncertainty_shortlist_size": args.uncertainty_shortlist_size,
            "pre_relaxation_steps": args.pre_relaxation_steps,
            "max_exact_delta_e_above": args.max_exact_delta_e_above,
            "order_bias_lambda": args.order_bias_lambda,
            "order_metric": args.order_metric,
            "order_biased_events": args.order_biased_events,
            "anneal_final_temperature": args.anneal_final_temperature,
            "anneal_tau": args.anneal_tau,
            "n_jobs": args.n_jobs,
        }
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
