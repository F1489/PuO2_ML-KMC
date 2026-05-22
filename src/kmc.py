"""ML-accelerated kinetic Monte Carlo driver."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback for minimal environments
    tqdm = lambda iterable, **_: iterable

from .events import Event, apply_event_to_positions, event_delta_energy, generate_candidate_events_with_diagnostics
from .features import get_feature_names, make_event_features
from .io_xyz import write_xyz
from .ml_model import estimate_uncertainty_rf, load_models, predict_delta_E, predict_good_event_probability, train_models
from .potentials import PairPotentialPuO2
from .analysis import (
    bulk_fluorite_order_score,
    close_contact_thresholds_satisfied,
    defect_counts,
    fluorite_order_score,
    identify_crystalline_core,
    identify_growth_front,
    min_pair_distances,
    rdf_peak_sharpness,
    soft_coordination_order_score,
)
from scipy.spatial import cKDTree

K_B_EV_PER_K = 8.617333262145e-5


class MLKMC:
    """Off-lattice ML-kMC simulation for PuO2 clusters."""

    def __init__(
        self,
        atoms: list[str],
        positions: np.ndarray,
        potential: PairPotentialPuO2,
        regressor,
        classifier,
        temperature: float = 3600.0,
        attempt_frequency: float = 1.0e13,
        base_barrier: float = 0.4,
        n_candidates_per_step: int = 64,
        max_displacement: float = 0.35,
        min_distance: float = 1.2,
        exact_check_interval: int = 50,
        active_learning_uncertainty_threshold: float = 1.0,
        classifier_rate_weight: float = 0.5,
        high_uncertainty_exact: bool = True,
        reject_exact_delta_above: float | None = 25.0,
        max_exact_delta_e_above: float | None = None,
        exact_shortlist_size: int = 8,
        uncertainty_shortlist_size: int = 4,
        active_learning_error_threshold: float = 2.0,
        adaptive_displacement: bool = True,
        order_bias_lambda: float = 0.0,
        order_metric: str = "fluorite",
        order_biased_events: bool = False,
        anneal_final_temperature: float | None = None,
        anneal_tau: float = 0.0,
        frozen_atom_indices: set[int] | None = None,
        seed: int = 42,
    ) -> None:
        self.atoms = list(atoms)
        self.positions = np.asarray(positions, dtype=float).copy()
        self.potential = potential
        self.regressor = regressor
        self.classifier = classifier
        self.temperature = temperature
        self.attempt_frequency = attempt_frequency
        self.base_barrier = base_barrier
        self.n_candidates_per_step = n_candidates_per_step
        self.max_displacement = max_displacement
        self.min_distance = min_distance
        self.exact_check_interval = exact_check_interval
        self.active_learning_uncertainty_threshold = active_learning_uncertainty_threshold
        self.classifier_rate_weight = classifier_rate_weight
        self.high_uncertainty_exact = high_uncertainty_exact
        self.reject_exact_delta_above = reject_exact_delta_above
        self.max_exact_delta_e_above = max_exact_delta_e_above
        self.exact_shortlist_size = exact_shortlist_size
        self.uncertainty_shortlist_size = uncertainty_shortlist_size
        self.active_learning_error_threshold = active_learning_error_threshold
        self.adaptive_displacement = adaptive_displacement
        self.order_bias_lambda = order_bias_lambda
        self.order_metric = order_metric
        self.order_biased_events = order_biased_events
        self.initial_temperature = temperature
        self.anneal_final_temperature = anneal_final_temperature
        self.anneal_tau = anneal_tau
        self.frozen_atom_indices = set() if frozen_atom_indices is None else set(frozen_atom_indices)
        self.current_max_displacement = max_displacement
        self.rng = np.random.default_rng(seed)
        self.kmc_time = 0.0
        self.history: list[dict[str, object]] = []
        self.active_learning_rows: list[dict[str, object]] = []
        self.active_learning_archive_rows: list[dict[str, object]] = []

    def _current_temperature(self, step_index: int) -> float:
        if self.anneal_final_temperature is None or self.anneal_tau <= 0.0:
            return self.temperature
        return float(
            self.anneal_final_temperature
            + (self.initial_temperature - self.anneal_final_temperature) * np.exp(-step_index / self.anneal_tau)
        )

    def _order_score(self, positions: np.ndarray) -> float:
        if self.order_metric == "bulk":
            return bulk_fluorite_order_score(self.atoms, positions)
        if self.order_metric == "soft":
            return soft_coordination_order_score(self.atoms, positions)
        return fluorite_order_score(self.atoms, positions)

    def _delta_order_for_event(self, event: Event, current_order: float) -> float:
        return float(self._order_score(apply_event_to_positions(self.positions, event)) - current_order)

    def _combined_score(self, delta_e: np.ndarray, delta_order: np.ndarray | float = 0.0) -> np.ndarray:
        return np.asarray(delta_e, dtype=float) - self.order_bias_lambda * np.asarray(delta_order, dtype=float)

    def _rates_from_score(self, score: np.ndarray, good_probability: np.ndarray, temperature: float) -> np.ndarray:
        barriers = self.base_barrier + np.maximum(0.0, score)
        rates = self.attempt_frequency * np.exp(-barriers / (K_B_EV_PER_K * temperature))
        if self.classifier_rate_weight > 0.0:
            classifier_factor = (1.0 - self.classifier_rate_weight) + self.classifier_rate_weight * good_probability
            rates = rates * np.clip(classifier_factor, 1.0e-6, 1.0)
        return np.asarray(rates, dtype=float)

    def _rates_from_delta_e(self, delta_e: np.ndarray, good_probability: np.ndarray) -> np.ndarray:
        return self._rates_from_score(delta_e, good_probability, self.temperature)

    def _store_active_learning_event(
        self,
        event: Event,
        exact_delta_e: float,
        exact_delta_order: float,
        predicted_delta_e: float,
        uncertainty: float,
        good_probability: float,
        step_index: int,
    ) -> None:
        feature_values = make_event_features(self.atoms, self.positions, event, potential=self.potential)
        row = dict(zip(get_feature_names(), feature_values, strict=True))
        row.update(
            {
                "delta_E": exact_delta_e,
                "delta_fluorite_order": exact_delta_order,
                "delta_bulk_order": np.nan,
                "delta_coordination_error": np.nan,
                "is_energy_lowering": int(exact_delta_e < 0.0),
                "is_crystallizing_event": int(exact_delta_e < 1.0 and exact_delta_order > 0.01),
                "atom_type": self.atoms[event.atom_index],
                "event_kind": event.kind,
                "uncertainty": uncertainty,
                "good_event_probability": good_probability,
                "step": step_index,
            }
        )
        self.active_learning_rows.append(row)
        self.active_learning_archive_rows.append(row)

    def _choose_event(
        self, events: list[Event], step_index: int
    ) -> tuple[Event, float, float, float, float, float, float | None, float | None, float | None, bool, int]:
        tree = cKDTree(self.positions)
        center = self.positions.mean(axis=0)
        current_temperature = self._current_temperature(step_index)
        current_order = self._order_score(self.positions)
        features = np.vstack(
            [
                make_event_features(
                    self.atoms,
                    self.positions,
                    event,
                    tree=tree,
                    potential=self.potential,
                    center=center,
                )
                for event in events
            ]
        )
        delta_e_ml = predict_delta_E(self.regressor, features)
        uncertainty = estimate_uncertainty_rf(self.regressor, features)
        good_probability = predict_good_event_probability(self.classifier, features)
        rates = self._rates_from_score(delta_e_ml, good_probability, current_temperature)

        shortlist: set[int] = set()
        if self.exact_shortlist_size > 0:
            top_rate = np.argsort(rates)[-min(self.exact_shortlist_size, len(events)) :]
            shortlist.update(int(i) for i in top_rate)
        if self.uncertainty_shortlist_size > 0:
            top_uncertainty = np.argsort(uncertainty)[-min(self.uncertainty_shortlist_size, len(events)) :]
            shortlist.update(int(i) for i in top_uncertainty)
        if step_index == 0 or (step_index + 1) % self.exact_check_interval == 0:
            shortlist.update(int(i) for i in np.argsort(rates)[-min(max(1, self.exact_shortlist_size), len(events)) :])

        exact_delta_by_index: dict[int, float] = {}
        exact_delta_order_by_index: dict[int, float] = {}
        exact_score_by_index: dict[int, float] = {}
        adjusted_rates = np.asarray(rates, dtype=float).copy()
        for index in sorted(shortlist):
            exact_delta_e = event_delta_energy(self.atoms, self.positions, events[index], self.potential)
            exact_delta_order = self._delta_order_for_event(events[index], current_order) if self.order_bias_lambda > 0.0 else 0.0
            exact_score = float(self._combined_score(np.asarray([exact_delta_e]), exact_delta_order)[0])
            exact_delta_by_index[index] = exact_delta_e
            exact_delta_order_by_index[index] = exact_delta_order
            exact_score_by_index[index] = exact_score
            exact_rate = self._rates_from_score(
                np.asarray([exact_score], dtype=float),
                np.asarray([good_probability[index]], dtype=float),
                current_temperature,
            )[0]
            if self.max_exact_delta_e_above is not None and exact_delta_e > self.max_exact_delta_e_above:
                exact_rate = 0.0
            if self.reject_exact_delta_above is not None and exact_score > self.reject_exact_delta_above:
                exact_rate = 0.0
            adjusted_rates[index] = exact_rate
            if (
                uncertainty[index] > self.active_learning_uncertainty_threshold
                or abs(float(delta_e_ml[index]) - exact_delta_e) > self.active_learning_error_threshold
            ):
                self._store_active_learning_event(
                    events[index],
                    exact_delta_e,
                    exact_delta_order,
                    float(delta_e_ml[index]),
                    float(uncertainty[index]),
                    float(good_probability[index]),
                    step_index,
                )
        rates = adjusted_rates
        total_rate = float(np.sum(rates))
        if not np.isfinite(total_rate) or total_rate <= 0.0:
            if exact_delta_by_index:
                selected = min(exact_score_by_index, key=lambda index: exact_score_by_index[index])
                return (
                    events[selected],
                    float(delta_e_ml[selected]),
                    float(uncertainty[selected]),
                    float(good_probability[selected]),
                    1.0,
                    0.0,
                    float(exact_delta_by_index[selected]),
                    float(exact_delta_order_by_index[selected]),
                    float(exact_score_by_index[selected]),
                    True,
                    len(exact_delta_by_index),
                )
            rates = np.full(len(events), 1.0 / len(events), dtype=float)
            total_rate = 1.0
        probabilities = rates / total_rate
        selected = int(self.rng.choice(len(events), p=probabilities))
        dt = float(-np.log(max(self.rng.random(), 1.0e-12)) / total_rate)
        return (
            events[selected],
            float(delta_e_ml[selected]),
            float(uncertainty[selected]),
            float(good_probability[selected]),
            total_rate,
            dt,
            float(exact_delta_by_index[selected]) if selected in exact_delta_by_index else None,
            float(exact_delta_order_by_index[selected]) if selected in exact_delta_by_index else None,
            float(exact_score_by_index[selected]) if selected in exact_delta_by_index else None,
            selected in exact_delta_by_index,
            len(exact_delta_by_index),
        )

    def step(self, step_index: int) -> None:
        """Run one kMC step and append history."""
        events, diagnostics = generate_candidate_events_with_diagnostics(
            self.atoms,
            self.positions,
            self.n_candidates_per_step,
            max_displacement=self.current_max_displacement,
            rng=self.rng,
            min_distance=self.min_distance,
            potential=self.potential,
            order_biased_events=self.order_biased_events,
            frozen_atom_indices=self.frozen_atom_indices,
        )
        (
            event,
            predicted_delta_e,
            uncertainty,
            good_probability,
            total_rate,
            dt,
            exact_delta_e,
            exact_delta_order,
            exact_order_biased_score,
            used_exact_rate,
            exact_checked_candidates,
        ) = self._choose_event(events, step_index)
        event_applied = True
        require_exact_energy_guard = self.max_exact_delta_e_above is not None and self.order_bias_lambda > 0.0
        if exact_delta_e is None and (
            require_exact_energy_guard
            or uncertainty > self.active_learning_uncertainty_threshold
            or step_index == 0
            or (step_index + 1) % self.exact_check_interval == 0
        ):
            current_order = self._order_score(self.positions)
            exact_delta_e = event_delta_energy(self.atoms, self.positions, event, self.potential)
            exact_delta_order = self._delta_order_for_event(event, current_order) if self.order_bias_lambda > 0.0 else 0.0
            exact_order_biased_score = float(self._combined_score(np.asarray([exact_delta_e]), exact_delta_order)[0])
            used_exact_rate = self.high_uncertainty_exact and uncertainty > self.active_learning_uncertainty_threshold
            exact_checked_candidates += 1
            if (
                uncertainty > self.active_learning_uncertainty_threshold
                or abs(predicted_delta_e - exact_delta_e) > self.active_learning_error_threshold
            ):
                self._store_active_learning_event(
                    event, exact_delta_e, exact_delta_order, predicted_delta_e, uncertainty, good_probability, step_index
                )
        reject_value = exact_order_biased_score if exact_order_biased_score is not None else exact_delta_e
        reject_for_score = self.reject_exact_delta_above is not None and reject_value is not None and reject_value > self.reject_exact_delta_above
        reject_for_energy = self.max_exact_delta_e_above is not None and exact_delta_e is not None and exact_delta_e > self.max_exact_delta_e_above
        proposed_positions = apply_event_to_positions(self.positions, event)
        reject_for_close_contact = not close_contact_thresholds_satisfied(self.atoms, proposed_positions)
        if reject_for_score or reject_for_energy or reject_for_close_contact:
            event_applied = False
            if used_exact_rate:
                current_temperature = self._current_temperature(step_index)
                total_rate = float(
                    max(
                        self._rates_from_score(
                            np.asarray([reject_value], dtype=float),
                            np.asarray([good_probability], dtype=float),
                            current_temperature,
                        )[0],
                        1.0e-30,
                    )
                )
                dt = float(-np.log(max(self.rng.random(), 1.0e-12)) / total_rate)

        if event_applied:
            self.positions = proposed_positions
        self.kmc_time += dt
        if self.adaptive_displacement:
            rejection_ratio = diagnostics["rejected"] / max(diagnostics["requested"], 1)
            if rejection_ratio > 0.5 or (exact_delta_e is not None and exact_delta_e > 10.0):
                self.current_max_displacement = max(0.05, 0.85 * self.current_max_displacement)
            elif rejection_ratio < 0.1 and (exact_delta_e is None or exact_delta_e < 2.0):
                self.current_max_displacement = min(self.max_displacement, 1.05 * self.current_max_displacement)
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
        rdf_pu_o_sharpness = np.nan
        min_distances = {"min_distance_pu_o": np.nan, "min_distance_o_o": np.nan, "min_distance_pu_pu": np.nan}
        crystalline_core_size = np.nan
        growth_front_size = np.nan
        if step_index == 0 or (step_index + 1) % self.exact_check_interval == 0:
            total_energy = self.potential.total_energy(self.atoms, self.positions)
            defects = defect_counts(self.atoms, self.positions)
            order_score = fluorite_order_score(self.atoms, self.positions)
            bulk_order_score = bulk_fluorite_order_score(self.atoms, self.positions)
            soft_order_score = soft_coordination_order_score(self.atoms, self.positions)
            rdf_pu_o_sharpness = rdf_peak_sharpness(self.atoms, self.positions, pair=("Pu", "O"))
            min_distances = min_pair_distances(self.atoms, self.positions)
            core = identify_crystalline_core(self.atoms, self.positions)
            front = identify_growth_front(self.atoms, self.positions, core)
            crystalline_core_size = int(len(core))
            growth_front_size = int(len(front))
        n_formula_units = max(sum(atom == "Pu" for atom in self.atoms), 1)
        self.history.append(
            {
                "step": step_index,
                "kmc_time": self.kmc_time,
                "total_energy": total_energy,
                "energy_per_atom": total_energy / len(self.atoms) if np.isfinite(total_energy) else np.nan,
                "energy_per_puo2": total_energy / n_formula_units if np.isfinite(total_energy) else np.nan,
                "pu_coordination_defects": defects["pu_coordination_defects"],
                "o_coordination_defects": defects["o_coordination_defects"],
                "total_coordination_defects": defects["total_coordination_defects"],
                "bulk_coordination_defects": defects["bulk_coordination_defects"],
                "surface_coordination_defects": defects["surface_coordination_defects"],
                "mean_abs_coordination_error": defects["mean_abs_coordination_error"],
                "fluorite_order_score": order_score,
                "bulk_fluorite_order_score": bulk_order_score,
                "soft_coordination_order_score": soft_order_score,
                "rdf_pu_o_peak_sharpness": rdf_pu_o_sharpness,
                "min_distance_pu_o": min_distances["min_distance_pu_o"],
                "min_distance_o_o": min_distances["min_distance_o_o"],
                "min_distance_pu_pu": min_distances["min_distance_pu_pu"],
                "crystalline_core_size": crystalline_core_size,
                "growth_front_size": growth_front_size,
                "accepted_event_kind": event.kind if event_applied else f"rejected_{event.kind}",
                "event_applied": int(event_applied),
                "predicted_delta_E": predicted_delta_e,
                "exact_delta_E_if_checked": exact_delta_e,
                "exact_delta_order_if_checked": exact_delta_order,
                "order_biased_score_if_checked": exact_order_biased_score,
                "max_exact_delta_e_above": self.max_exact_delta_e_above,
                "reject_for_close_contact": int(reject_for_close_contact),
                "used_exact_rate": int(used_exact_rate),
                "exact_checked_candidates": exact_checked_candidates,
                "uncertainty": uncertainty,
                "good_event_probability": good_probability,
                "total_rate": total_rate,
                "candidate_events_requested": diagnostics["requested"],
                "candidate_events_generated": diagnostics["generated"],
                "candidate_events_rejected": diagnostics["rejected"],
                "current_max_displacement": self.current_max_displacement,
                "temperature": self._current_temperature(step_index),
                "order_bias_lambda": self.order_bias_lambda,
            }
        )

    def _retrain_from_active_rows(
        self,
        model_dir: Path,
        base_dataset_csv: str | Path | None,
        retrain_csv: Path,
        seed: int,
    ) -> None:
        """Append active-learning rows to a training CSV and reload models."""
        if not self.active_learning_rows:
            return
        active_df = pd.DataFrame(self.active_learning_rows)
        if retrain_csv.exists():
            old_df = pd.read_csv(retrain_csv)
            train_df = pd.concat([old_df, active_df], ignore_index=True)
        elif base_dataset_csv is not None and Path(base_dataset_csv).exists():
            base_df = pd.read_csv(base_dataset_csv)
            train_df = pd.concat([base_df, active_df], ignore_index=True)
        else:
            train_df = active_df
        train_df = train_df.drop_duplicates()
        retrain_csv.parent.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(retrain_csv, index=False)
        if len(train_df) >= 20 and train_df["is_energy_lowering"].nunique() > 1:
            train_models(retrain_csv, model_dir, seed=seed)
            self.regressor, self.classifier, _ = load_models(model_dir)
            self.active_learning_rows.clear()

    def run(
        self,
        steps: int,
        out_dir: str | Path | None = None,
        active_learning_retrain_interval: int = 0,
        model_dir: str | Path | None = None,
        base_dataset_csv: str | Path | None = None,
        save_xyz_interval: int = 0,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Run kMC for a number of steps and optionally save outputs."""
        out_path = Path(out_dir) if out_dir is not None else None
        snapshots_dir = None
        if out_path is not None and save_xyz_interval > 0:
            snapshots_dir = out_path / "snapshots"
            snapshots_dir.mkdir(parents=True, exist_ok=True)
        for step_index in tqdm(range(steps), desc="Running ML-kMC"):
            self.step(step_index)
            if snapshots_dir is not None and (step_index + 1) % save_xyz_interval == 0:
                write_xyz(snapshots_dir / f"step_{step_index + 1:06d}.xyz", self.atoms, self.positions, comment=f"ML-kMC step {step_index + 1}")
                if out_path is not None:
                    pd.DataFrame(self.history).to_csv(out_path / "history.csv", index=False)
                    pd.DataFrame(self.active_learning_archive_rows).to_csv(out_path / "active_learning_events.csv", index=False)
            if (
                active_learning_retrain_interval > 0
                and model_dir is not None
                and (step_index + 1) % active_learning_retrain_interval == 0
            ):
                retrain_csv = (out_path or Path(model_dir)) / "active_learning_training.csv"
                self._retrain_from_active_rows(Path(model_dir), base_dataset_csv, retrain_csv, seed=seed + step_index)
        history = pd.DataFrame(self.history)
        if out_path is not None:
            out_path.mkdir(parents=True, exist_ok=True)
            history.to_csv(out_path / "history.csv", index=False)
            pd.DataFrame(self.active_learning_archive_rows).to_csv(out_path / "active_learning_events.csv", index=False)
            write_xyz(out_path / "final.xyz", self.atoms, self.positions, comment="Final ML-kMC structure")
        return history
