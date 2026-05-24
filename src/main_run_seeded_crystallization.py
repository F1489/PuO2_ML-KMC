"""Two-stage seeded crystallization workflow for PuO2 clusters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import pandas as pd

from .analysis import (
    close_contact_thresholds_satisfied,
    coordination_summary,
    identify_crystalline_core,
    identify_growth_front,
    min_pair_distances,
    write_summary_json,
)
from .io_xyz import read_xyz, write_xyz
from .kmc import MLKMC
from .ml_model import load_models
from .potentials import PairPotentialPuO2
from .repair import repair_close_contacts
from .seeded import impose_fluorite_seed
from .visualization import plot_seeded_stage_comparison, save_all_plots


def _run_stage(
    atoms: list[str],
    positions,
    potential: PairPotentialPuO2,
    model_dir: str | Path,
    out_dir: Path,
    stage_name: str,
    steps: int,
    temperature: float,
    lambda_order: float,
    n_candidates_per_step: int,
    exact_shortlist_size: int,
    uncertainty_shortlist_size: int,
    max_exact_delta_e_above: float | None,
    save_xyz_interval: int,
    n_jobs: int,
    seed: int,
    frozen_atom_indices: set[int] | None = None,
) -> tuple[object, pd.DataFrame, float]:
    regressor, classifier, _ = load_models(model_dir)
    kmc = MLKMC(
        atoms,
        positions,
        potential,
        regressor,
        classifier,
        temperature=temperature,
        n_candidates_per_step=n_candidates_per_step,
        exact_shortlist_size=exact_shortlist_size,
        uncertainty_shortlist_size=uncertainty_shortlist_size,
        exact_check_interval=max(10, min(100, steps // 10 if steps else 10)),
        reject_exact_delta_above=10.0,
        max_exact_delta_e_above=max_exact_delta_e_above,
        order_bias_lambda=lambda_order,
        order_metric="bulk",
        order_biased_events=True,
        frozen_atom_indices=frozen_atom_indices,
        n_jobs=n_jobs,
        seed=seed,
    )
    start = perf_counter()
    history = kmc.run(
        steps,
        out_dir=out_dir / stage_name,
        save_xyz_interval=save_xyz_interval,
        seed=seed,
    )
    elapsed = perf_counter() - start
    return kmc, history, elapsed


def _state_summary(atoms: list[str], positions, potential: PairPotentialPuO2) -> dict[str, object]:
    n_formula_units = max(sum(atom == "Pu" for atom in atoms), 1)
    structure = coordination_summary(atoms, positions)
    return {
        "energy_per_puo2_eV": potential.total_energy(atoms, positions) / n_formula_units,
        "bulk_fluorite_order_score": structure["bulk_fluorite_order_score"],
        "mean_abs_coordination_error": structure["mean_abs_coordination_error"],
        "fraction_pu_with_8_o": structure["fraction_pu_with_8_o"],
        "fraction_o_with_4_pu": structure["fraction_o_with_4_pu"],
        "min_pair_distances": min_pair_distances(atoms, positions),
        "close_contact_thresholds_satisfied": close_contact_thresholds_satisfied(atoms, positions),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xyz", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--steps-stage1", type=int, default=30000)
    parser.add_argument("--steps-stage2", type=int, default=30000)
    parser.add_argument("--lambda-stage1", type=float, default=5.0)
    parser.add_argument("--lambda-stage2", type=float, default=1.0)
    parser.add_argument("--temperature-stage1", type=float, default=3300.0)
    parser.add_argument("--temperature-stage2", type=float, default=2200.0)
    parser.add_argument("--seed-radius", type=float, default=5.0)
    parser.add_argument("--seed-lattice-constant", type=float, default=5.40)
    parser.add_argument("--seed-blend", type=float, default=0.5)
    parser.add_argument("--freeze-seed-stage1", action="store_true")
    parser.add_argument("--pre-relaxation-steps", type=int, default=1000)
    parser.add_argument("--post-seed-repair-steps", type=int, default=1000)
    parser.add_argument("--n-candidates-per-step", type=int, default=256)
    parser.add_argument("--exact-shortlist-size", type=int, default=16)
    parser.add_argument("--uncertainty-shortlist-size", type=int, default=8)
    parser.add_argument("--max-exact-delta-e-above", type=float, default=0.2)
    parser.add_argument("--save-xyz-interval", type=int, default=1000)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=9600)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    atoms, positions, _ = read_xyz(args.xyz)
    original_positions = positions.copy()
    potential = PairPotentialPuO2()

    if args.pre_relaxation_steps > 0:
        positions, repair_history = repair_close_contacts(
            atoms,
            positions,
            potential,
            steps=args.pre_relaxation_steps,
            seed=args.seed,
        )
        pd.DataFrame(repair_history).to_csv(out_dir / "pre_seed_repair_history.csv", index=False)

    seeded_positions, seed_indices = impose_fluorite_seed(
        atoms,
        positions,
        seed_radius=args.seed_radius,
        lattice_constant=args.seed_lattice_constant,
        blend=args.seed_blend,
    )
    write_xyz(out_dir / "seeded_initial.xyz", atoms, seeded_positions, comment="Seeded fluorite core")
    if args.post_seed_repair_steps > 0:
        seeded_positions, post_seed_repair_history = repair_close_contacts(
            atoms,
            seeded_positions,
            potential,
            steps=args.post_seed_repair_steps,
            seed=args.seed + 17,
        )
        pd.DataFrame(post_seed_repair_history).to_csv(out_dir / "post_seed_repair_history.csv", index=False)
    after_repair_positions = seeded_positions.copy()
    write_xyz(out_dir / "after_repair.xyz", atoms, after_repair_positions, comment="After seed and close-contact repair")

    frozen = seed_indices if args.freeze_seed_stage1 else None
    stage1_kmc, stage1_history, stage1_elapsed = _run_stage(
        atoms,
        after_repair_positions,
        potential,
        args.model_dir,
        out_dir,
        "stage1_ordering",
        args.steps_stage1,
        args.temperature_stage1,
        args.lambda_stage1,
        args.n_candidates_per_step,
        args.exact_shortlist_size,
        args.uncertainty_shortlist_size,
        args.max_exact_delta_e_above,
        args.save_xyz_interval,
        args.n_jobs,
        args.seed,
        frozen_atom_indices=frozen,
    )
    write_xyz(out_dir / "after_stage1.xyz", stage1_kmc.atoms, stage1_kmc.positions, comment="After seeded ordering stage")

    stage2_kmc, stage2_history, stage2_elapsed = _run_stage(
        stage1_kmc.atoms,
        stage1_kmc.positions,
        potential,
        args.model_dir,
        out_dir,
        "stage2_annealing",
        args.steps_stage2,
        args.temperature_stage2,
        args.lambda_stage2,
        args.n_candidates_per_step,
        args.exact_shortlist_size,
        args.uncertainty_shortlist_size,
        args.max_exact_delta_e_above,
        args.save_xyz_interval,
        args.n_jobs,
        args.seed + 1,
        frozen_atom_indices=None,
    )
    write_xyz(out_dir / "final.xyz", stage2_kmc.atoms, stage2_kmc.positions, comment="Final seeded crystallization structure")

    stage1_history = stage1_history.copy()
    stage1_history["stage"] = "stage1_ordering"
    stage2_history = stage2_history.copy()
    stage2_history["stage"] = "stage2_annealing"
    if not stage2_history.empty:
        stage2_history["step"] = stage2_history["step"] + len(stage1_history)
    history = pd.concat([stage1_history, stage2_history], ignore_index=True)
    history.to_csv(out_dir / "history.csv", index=False)

    save_all_plots(out_dir / "history.csv", atoms, original_positions, stage2_kmc.atoms, stage2_kmc.positions, out_dir)
    summary = write_summary_json(
        out_dir / "summary.json",
        atoms,
        original_positions,
        stage2_kmc.atoms,
        stage2_kmc.positions,
        potential,
        positions_after_repair=after_repair_positions,
    )
    core = identify_crystalline_core(stage2_kmc.atoms, stage2_kmc.positions)
    front = identify_growth_front(stage2_kmc.atoms, stage2_kmc.positions, core)
    state_comparison = {
        "after_repair": _state_summary(atoms, after_repair_positions, potential),
        "after_stage1": _state_summary(stage1_kmc.atoms, stage1_kmc.positions, potential),
        "final": _state_summary(stage2_kmc.atoms, stage2_kmc.positions, potential),
    }
    state_comparison["deltas"] = {
        "stage1_minus_after_repair_energy_per_puo2_eV": (
            state_comparison["after_stage1"]["energy_per_puo2_eV"]
            - state_comparison["after_repair"]["energy_per_puo2_eV"]
        ),
        "final_minus_after_stage1_energy_per_puo2_eV": (
            state_comparison["final"]["energy_per_puo2_eV"]
            - state_comparison["after_stage1"]["energy_per_puo2_eV"]
        ),
        "final_minus_after_repair_energy_per_puo2_eV": (
            state_comparison["final"]["energy_per_puo2_eV"]
            - state_comparison["after_repair"]["energy_per_puo2_eV"]
        ),
        "final_minus_after_repair_bulk_order_score": (
            state_comparison["final"]["bulk_fluorite_order_score"]
            - state_comparison["after_repair"]["bulk_fluorite_order_score"]
        ),
        "final_minus_after_repair_mean_abs_coordination_error": (
            state_comparison["final"]["mean_abs_coordination_error"]
            - state_comparison["after_repair"]["mean_abs_coordination_error"]
        ),
        "final_minus_after_repair_fraction_pu_with_8_o": (
            state_comparison["final"]["fraction_pu_with_8_o"]
            - state_comparison["after_repair"]["fraction_pu_with_8_o"]
        ),
        "final_minus_after_repair_fraction_o_with_4_pu": (
            state_comparison["final"]["fraction_o_with_4_pu"]
            - state_comparison["after_repair"]["fraction_o_with_4_pu"]
        ),
    }
    summary.update(
        {
            "workflow": "seeded_crystallization",
            "state_comparison_after_repair_after_stage1_final": state_comparison,
            "seed_radius": args.seed_radius,
            "seed_atom_count": len(seed_indices),
            "freeze_seed_stage1": args.freeze_seed_stage1,
            "steps_stage1": args.steps_stage1,
            "steps_stage2": args.steps_stage2,
            "lambda_stage1": args.lambda_stage1,
            "lambda_stage2": args.lambda_stage2,
            "temperature_stage1": args.temperature_stage1,
            "temperature_stage2": args.temperature_stage2,
            "n_candidates_per_step": args.n_candidates_per_step,
            "exact_shortlist_size": args.exact_shortlist_size,
            "pre_relaxation_steps": args.pre_relaxation_steps,
            "post_seed_repair_steps": args.post_seed_repair_steps,
            "max_exact_delta_e_above": args.max_exact_delta_e_above,
            "n_jobs": args.n_jobs,
            "wall_time_stage1_s": stage1_elapsed,
            "wall_time_stage2_s": stage2_elapsed,
            "final_crystalline_core_size": int(len(core)),
            "final_growth_front_size": int(len(front)),
        }
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_seeded_stage_comparison(out_dir)


if __name__ == "__main__":
    main()
