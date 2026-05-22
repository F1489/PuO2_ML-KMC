"""CLI for iterative kMC-driven active learning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .analysis import write_summary_json
from .io_xyz import read_xyz
from .kmc import MLKMC
from .ml_model import load_models, train_models
from .potentials import PairPotentialPuO2
from .validation import validate_models_on_independent_events


def _balanced_active_sample(active_df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    """Select a bounded, diverse set of hard active-learning events."""
    if "delta_E" in active_df.columns:
        active_df = active_df[active_df["delta_E"].abs() <= 25.0].copy()
    if max_rows <= 0 or len(active_df) <= max_rows:
        return active_df.copy()

    sort_columns = [column for column in ["uncertainty", "good_event_probability"] if column in active_df.columns]
    ranked = active_df.copy()
    if sort_columns:
        ranked = ranked.sort_values(sort_columns, ascending=[False] * len(sort_columns))

    group_columns = [column for column in ["atom_type", "event_kind", "is_energy_lowering"] if column in ranked.columns]
    if not group_columns:
        return ranked.sample(n=max_rows, random_state=seed).reset_index(drop=True)

    n_groups = max(1, ranked[group_columns].drop_duplicates().shape[0])
    per_group = max(1, int(max_rows / n_groups + 0.999))
    selected_parts = [group.head(per_group) for _, group in ranked.groupby(group_columns, sort=False)]
    selected = pd.concat(selected_parts).drop_duplicates()
    selected_indices = selected.index

    if len(selected) < max_rows:
        remainder = ranked.drop(selected_indices, errors="ignore")
        selected = pd.concat([selected, remainder.head(max_rows - len(selected))])
    if len(selected) > max_rows:
        selected = selected.sample(n=max_rows, random_state=seed)
    return selected.reset_index(drop=True)


def main() -> None:
    """Run kMC -> difficult events -> retrain -> validation cycles."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--xyz", required=True)
    parser.add_argument("--base-dataset", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--steps-per-cycle", type=int, default=100)
    parser.add_argument("--validation-events", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=3600.0)
    parser.add_argument("--n-candidates-per-step", type=int, default=64)
    parser.add_argument("--max-displacement", type=float, default=0.35)
    parser.add_argument("--min-distance", type=float, default=1.2)
    parser.add_argument("--reject-exact-delta-above", type=float, default=0.0)
    parser.add_argument("--exact-shortlist-size", type=int, default=8)
    parser.add_argument("--uncertainty-shortlist-size", type=int, default=4)
    parser.add_argument("--active-learning-error-threshold", type=float, default=2.0)
    parser.add_argument("--max-active-events-per-cycle", type=int, default=300)
    parser.add_argument(
        "--classifier-target",
        choices=["is_energy_lowering", "is_crystallizing_event"],
        default="is_energy_lowering",
    )
    parser.add_argument("--no-external-models", action="store_true")
    parser.add_argument("--ensemble-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    model_dir = Path(args.model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    atoms, positions, _ = read_xyz(args.xyz)
    initial_positions = positions.copy()
    potential = PairPotentialPuO2()
    training_csv = out_dir / "active_learning_training.csv"
    if not training_csv.exists():
        pd.read_csv(args.base_dataset).to_csv(training_csv, index=False)
    if not (model_dir / "regressor.joblib").exists():
        train_models(
            training_csv,
            model_dir,
            seed=args.seed,
            classifier_target=args.classifier_target,
            use_external_models=not args.no_external_models,
            ensemble_size=args.ensemble_size,
        )

    cycle_reports: list[dict[str, object]] = []
    for cycle in range(1, args.cycles + 1):
        regressor, classifier, _ = load_models(model_dir)
        cycle_dir = out_dir / f"cycle_{cycle:02d}"
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
            reject_exact_delta_above=args.reject_exact_delta_above,
            exact_shortlist_size=args.exact_shortlist_size,
            uncertainty_shortlist_size=args.uncertainty_shortlist_size,
            active_learning_error_threshold=args.active_learning_error_threshold,
            seed=args.seed + cycle,
        )
        history = kmc.run(
            args.steps_per_cycle,
            out_dir=cycle_dir,
            active_learning_retrain_interval=0,
            model_dir=model_dir,
            base_dataset_csv=training_csv,
            seed=args.seed + cycle,
        )
        atoms = kmc.atoms
        positions = kmc.positions.copy()

        selected_active_events = 0
        if kmc.active_learning_archive_rows:
            active_df = pd.DataFrame(kmc.active_learning_archive_rows)
            selected_df = _balanced_active_sample(active_df, args.max_active_events_per_cycle, args.seed + cycle)
            selected_active_events = int(len(selected_df))
            train_df = pd.concat([pd.read_csv(training_csv), selected_df], ignore_index=True).drop_duplicates()
            train_df.to_csv(training_csv, index=False)
            train_models(
                training_csv,
                model_dir,
                seed=args.seed + cycle,
                classifier_target=args.classifier_target,
                use_external_models=not args.no_external_models,
                ensemble_size=args.ensemble_size,
            )

        validation_dir = cycle_dir / "validation"
        validation_metrics = validate_models_on_independent_events(
            atoms,
            positions,
            potential,
            model_dir,
            validation_dir,
            n_events=args.validation_events,
            seed=args.seed + 1000 + cycle,
            max_displacement=args.max_displacement,
            min_distance=args.min_distance,
        )
        summary = write_summary_json(cycle_dir / "summary.json", atoms, initial_positions, kmc.atoms, kmc.positions, potential)
        cycle_reports.append(
            {
                "cycle": cycle,
                "history_rows": int(len(history)),
                "active_learning_events": int(len(kmc.active_learning_archive_rows)),
                "selected_active_learning_events": selected_active_events,
                "training_rows": int(len(pd.read_csv(training_csv))),
                "validation": validation_metrics,
                "summary": summary,
            }
        )
        (out_dir / "active_learning_report.json").write_text(
            json.dumps(cycle_reports, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    print(json.dumps(cycle_reports, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
