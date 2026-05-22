"""CLI for a non-ML random Metropolis baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .analysis import (
    bulk_fluorite_order_score,
    defect_counts,
    fluorite_order_score,
    min_pair_distances,
    rdf_peak_sharpness,
    soft_coordination_order_score,
    write_summary_json,
)
from .events import generate_candidate_events
from .io_xyz import read_xyz, write_xyz
from .kmc import K_B_EV_PER_K
from .potentials import PairPotentialPuO2
from .repair import repair_close_contacts
from .visualization import save_all_plots


def main() -> None:
    """Run a random local-move Metropolis baseline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--xyz", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--temperature", type=float, default=3600.0)
    parser.add_argument("--n-candidates-per-step", type=int, default=32)
    parser.add_argument("--max-displacement", type=float, default=0.35)
    parser.add_argument("--min-distance", type=float, default=1.2)
    parser.add_argument("--exact-check-interval", type=int, default=50)
    parser.add_argument("--save-xyz-interval", type=int, default=1000)
    parser.add_argument("--pre-relaxation-steps", type=int, default=0)
    parser.add_argument("--pre-relaxation-max-displacement", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    atoms, positions, _ = read_xyz(args.xyz)
    initial_positions = positions.copy()
    positions_after_repair = positions.copy()
    potential = PairPotentialPuO2()
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir = out_dir / "snapshots"
    snapshots_dir.mkdir(exist_ok=True)
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

    history: list[dict[str, object]] = []
    start = perf_counter()
    for step in range(args.steps):
        events = generate_candidate_events(
            atoms,
            positions,
            n_events=args.n_candidates_per_step,
            max_displacement=args.max_displacement,
            rng=rng,
            min_distance=args.min_distance,
            potential=potential,
        )
        event = events[int(rng.integers(0, len(events)))]
        delta_e = potential.delta_energy_single_move(atoms, positions, event.atom_index, event.new_position)
        accept_probability = min(1.0, float(np.exp(-delta_e / (K_B_EV_PER_K * args.temperature)))) if delta_e > 0 else 1.0
        accepted = rng.random() < accept_probability
        if accepted:
            positions[event.atom_index] = event.new_position

        total_energy = np.nan
        defects = {
            "pu_coordination_defects": np.nan,
            "o_coordination_defects": np.nan,
            "total_coordination_defects": np.nan,
            "bulk_coordination_defects": np.nan,
            "surface_coordination_defects": np.nan,
            "mean_abs_coordination_error": np.nan,
        }
        order_score = np.nan
        bulk_order_score = np.nan
        soft_order_score = np.nan
        rdf_sharpness = np.nan
        min_distances = {"min_distance_pu_o": np.nan, "min_distance_o_o": np.nan, "min_distance_pu_pu": np.nan}
        if step == 0 or (step + 1) % args.exact_check_interval == 0:
            total_energy = potential.total_energy(atoms, positions)
            defects = defect_counts(atoms, positions)
            order_score = fluorite_order_score(atoms, positions)
            bulk_order_score = bulk_fluorite_order_score(atoms, positions)
            soft_order_score = soft_coordination_order_score(atoms, positions)
            rdf_sharpness = rdf_peak_sharpness(atoms, positions, pair=("Pu", "O"))
            min_distances = min_pair_distances(atoms, positions)
        n_formula_units = max(sum(atom == "Pu" for atom in atoms), 1)
        history.append(
            {
                "step": step,
                "total_energy": total_energy,
                "energy_per_atom": total_energy / len(atoms) if np.isfinite(total_energy) else np.nan,
                "energy_per_puo2": total_energy / n_formula_units if np.isfinite(total_energy) else np.nan,
                "event_applied": int(accepted),
                "accepted_event_kind": event.kind if accepted else f"rejected_{event.kind}",
                "exact_delta_E_if_checked": delta_e,
                "predicted_delta_E": delta_e,
                "uncertainty": 0.0,
                "good_event_probability": float(delta_e < 0.0),
                "total_rate": np.nan,
                "kmc_time": np.nan,
                "pu_coordination_defects": defects["pu_coordination_defects"],
                "o_coordination_defects": defects["o_coordination_defects"],
                "total_coordination_defects": defects["total_coordination_defects"],
                "bulk_coordination_defects": defects["bulk_coordination_defects"],
                "surface_coordination_defects": defects["surface_coordination_defects"],
                "mean_abs_coordination_error": defects["mean_abs_coordination_error"],
                "fluorite_order_score": order_score,
                "bulk_fluorite_order_score": bulk_order_score,
                "soft_coordination_order_score": soft_order_score,
                "rdf_pu_o_peak_sharpness": rdf_sharpness,
                "min_distance_pu_o": min_distances["min_distance_pu_o"],
                "min_distance_o_o": min_distances["min_distance_o_o"],
                "min_distance_pu_pu": min_distances["min_distance_pu_pu"],
            }
        )
        if args.save_xyz_interval > 0 and ((step + 1) % args.save_xyz_interval == 0):
            write_xyz(snapshots_dir / f"step_{step + 1:06d}.xyz", atoms, positions, comment=f"baseline step {step + 1}")

    elapsed = perf_counter() - start
    history_df = pd.DataFrame(history)
    history_df["wall_time_s"] = elapsed
    history_df["steps_per_second"] = args.steps / elapsed if elapsed > 0 else np.nan
    history_df.to_csv(out_dir / "history.csv", index=False)
    write_xyz(out_dir / "final.xyz", atoms, positions, comment="Final baseline Metropolis structure")
    save_all_plots(out_dir / "history.csv", atoms, initial_positions, atoms, positions, out_dir)
    summary = write_summary_json(
        out_dir / "summary.json",
        atoms,
        initial_positions,
        atoms,
        positions,
        potential,
        positions_after_repair=positions_after_repair,
    )
    summary["wall_time_s"] = elapsed
    summary["steps_per_second"] = args.steps / elapsed if elapsed > 0 else 0.0
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
