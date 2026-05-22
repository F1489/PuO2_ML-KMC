"""Independent validation reports for ML Delta E models."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = lambda iterable, **_: iterable

from .events import generate_candidate_events
from .features import get_feature_names, make_event_features
from .ml_model import (
    estimate_uncertainty_rf,
    load_classifier_threshold,
    load_models,
    predict_delta_E,
    predict_good_event_probability,
)
from .potentials import PairPotentialPuO2


ERROR_ANALYSIS_GROUPS = ("atom_type", "event_kind")


def _apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#2D333B",
            "axes.labelcolor": "#1F2328",
            "axes.titleweight": "bold",
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.color": "#1F2328",
            "ytick.color": "#1F2328",
            "legend.frameon": True,
            "legend.framealpha": 0.94,
            "legend.facecolor": "white",
            "legend.edgecolor": "#C9CED6",
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def _style_axes(ax, title: str) -> None:
    ax.set_title(title, fontweight="bold", pad=12)
    ax.grid(True, color="#D7DCE2", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#AEB7C2")
    ax.spines["bottom"].set_color("#AEB7C2")


def _plot_predicted_vs_exact(df: pd.DataFrame, output_dir: Path) -> None:
    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(7.0, 6.4))
    colors = np.where(df["exact_delta_E"] < 0.0, "#245C4F", "#A33D3D")
    ax.scatter(df["exact_delta_E"], df["predicted_delta_E"], s=30, alpha=0.78, c=colors, edgecolor="white", linewidth=0.4)
    low = float(min(df["exact_delta_E"].min(), df["predicted_delta_E"].min()))
    high = float(max(df["exact_delta_E"].max(), df["predicted_delta_E"].max()))
    ax.plot([low, high], [low, high], color="#30343B", linewidth=2, label="идеальное совпадение")
    _style_axes(ax, "Независимая проверка ML-прогноза Delta E")
    ax.set_xlabel("Точное Delta E, эВ")
    ax.set_ylabel("Предсказанное Delta E, эВ")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "validation_predicted_vs_exact_delta_E.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def _plot_residuals(df: pd.DataFrame, output_dir: Path) -> None:
    _apply_plot_style()
    residual = df["predicted_delta_E"] - df["exact_delta_E"]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.hist(residual, bins=30, color="#2F6B9A", alpha=0.82, edgecolor="white")
    ax.axvline(0.0, color="#30343B", linewidth=2)
    _style_axes(ax, "Распределение ошибок прогноза")
    ax.set_xlabel("Ошибка ML - exact, эВ")
    ax.set_ylabel("Число событий")
    fig.tight_layout()
    fig.savefig(output_dir / "validation_residual_hist.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def _plot_uncertainty_vs_error(df: pd.DataFrame, output_dir: Path) -> None:
    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.scatter(df["uncertainty"], np.abs(df["predicted_delta_E"] - df["exact_delta_E"]), s=28, color="#7A3E65", alpha=0.76)
    _style_axes(ax, "Неопределенность против абсолютной ошибки")
    ax.set_xlabel("Оценка неопределенности, эВ")
    ax.set_ylabel("|ошибка|, эВ")
    fig.tight_layout()
    fig.savefig(output_dir / "validation_uncertainty_vs_abs_error.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def _classification_metrics(df: pd.DataFrame) -> dict[str, float]:
    y_true = df["exact_is_energy_lowering"].to_numpy(dtype=int)
    y_pred = df["predicted_is_energy_lowering"].to_numpy(dtype=int)
    return {
        "sign_accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_energy_lowering": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_energy_lowering": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }


def _ranking_metrics(df: pd.DataFrame, fractions: tuple[float, ...] = (0.05, 0.10, 0.20)) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    ranked = df.sort_values("predicted_delta_E", ascending=True).reset_index(drop=True)
    for fraction in fractions:
        top_n = max(1, int(np.ceil(len(ranked) * fraction)))
        top = ranked.head(top_n)
        key = f"top_{int(fraction * 100):02d}pct"
        result[f"{key}_n_events"] = int(top_n)
        result[f"{key}_precision_energy_lowering"] = float(top["exact_is_energy_lowering"].mean())
        result[f"{key}_mean_exact_delta_E"] = float(top["exact_delta_E"].mean())
        result[f"{key}_median_exact_delta_E"] = float(top["exact_delta_E"].median())
    false_bad = df[(df["exact_is_energy_lowering"] == 1) & (df["predicted_is_energy_lowering"] == 0)]
    false_good = df[(df["exact_is_energy_lowering"] == 0) & (df["predicted_is_energy_lowering"] == 1)]
    result.update(
        {
            "false_bad_count": int(len(false_bad)),
            "false_bad_rate": float(len(false_bad) / max(len(df), 1)),
            "false_good_count": int(len(false_good)),
            "false_good_rate": float(len(false_good) / max(len(df), 1)),
        }
    )
    return result


def _regression_metrics(df: pd.DataFrame) -> dict[str, float]:
    y_true = df["exact_delta_E"].to_numpy(dtype=float)
    y_pred = df["predicted_delta_E"].to_numpy(dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "bias": float(np.mean(y_pred - y_true)),
        "mean_abs_error": float(np.mean(np.abs(y_pred - y_true))),
    }


def _summarize_group(group_df: pd.DataFrame) -> dict[str, float | int]:
    result: dict[str, float | int] = {
        "n_events": int(len(group_df)),
        "mean_exact_delta_E": float(group_df["exact_delta_E"].mean()),
        "mean_predicted_delta_E": float(group_df["predicted_delta_E"].mean()),
        "mean_uncertainty": float(group_df["uncertainty"].mean()),
        "max_abs_error": float(group_df["abs_error"].max()),
        "sign_errors": int(group_df["sign_error"].sum()),
        "sign_error_rate": float(group_df["sign_error"].mean()),
    }
    result.update(_regression_metrics(group_df))
    result.update(_classification_metrics(group_df))
    return result


def _save_group_error_table(df: pd.DataFrame, output_dir: Path, group_column: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group_value, group_df in df.groupby(group_column, dropna=False):
        row: dict[str, object] = {group_column: group_value}
        row.update(_summarize_group(group_df))
        rows.append(row)
    table = pd.DataFrame(rows).sort_values(["sign_error_rate", "mae"], ascending=False)
    table.to_csv(output_dir / f"validation_error_by_{group_column}.csv", index=False)
    return table.to_dict(orient="records")


def analyze_validation_errors(df: pd.DataFrame, output_dir: str | Path) -> dict[str, object]:
    """Save validation error diagnostics by atom type, event kind, sign, and uncertainty."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["error"] = df["predicted_delta_E"] - df["exact_delta_E"]
    df["abs_error"] = np.abs(df["error"])
    df["sign_error"] = df["exact_is_energy_lowering"] != df["predicted_is_energy_lowering"]

    overall: dict[str, object] = _summarize_group(df)
    false_bad = df[(df["exact_is_energy_lowering"] == 1) & (df["predicted_is_energy_lowering"] == 0)]
    false_good = df[(df["exact_is_energy_lowering"] == 0) & (df["predicted_is_energy_lowering"] == 1)]
    overall.update(
        {
            "false_bad_count": int(len(false_bad)),
            "false_bad_rate": float(len(false_bad) / max(len(df), 1)),
            "false_good_count": int(len(false_good)),
            "false_good_rate": float(len(false_good) / max(len(df), 1)),
            "mean_abs_error_sign_errors": float(df.loc[df["sign_error"], "abs_error"].mean())
            if bool(df["sign_error"].any())
            else 0.0,
        }
    )

    group_reports = {group: _save_group_error_table(df, output_dir, group) for group in ERROR_ANALYSIS_GROUPS}
    combined_rows: list[dict[str, object]] = []
    for (atom_type, event_kind), group_df in df.groupby(["atom_type", "event_kind"], dropna=False):
        row: dict[str, object] = {"atom_type": atom_type, "event_kind": event_kind}
        row.update(_summarize_group(group_df))
        combined_rows.append(row)
    combined = pd.DataFrame(combined_rows).sort_values(["sign_error_rate", "mae"], ascending=False)
    combined.to_csv(output_dir / "validation_error_by_atom_type_event_kind.csv", index=False)

    uncertainty_bins = pd.qcut(df["uncertainty"], q=min(5, len(df)), duplicates="drop")
    uncertainty_rows: list[dict[str, object]] = []
    for interval, group_df in df.groupby(uncertainty_bins, observed=True):
        row: dict[str, object] = {"uncertainty_bin": str(interval)}
        row.update(_summarize_group(group_df))
        uncertainty_rows.append(row)
    uncertainty_table = pd.DataFrame(uncertainty_rows)
    uncertainty_table.to_csv(output_dir / "validation_error_by_uncertainty_bin.csv", index=False)

    worst_columns = [
        "atom_type",
        "event_kind",
        "exact_delta_E",
        "predicted_delta_E",
        "error",
        "abs_error",
        "exact_is_energy_lowering",
        "predicted_is_energy_lowering",
        "good_event_probability",
        "uncertainty",
    ]
    df.sort_values("abs_error", ascending=False).head(50)[worst_columns].to_csv(
        output_dir / "validation_worst_errors.csv", index=False
    )
    df[df["sign_error"]].sort_values("abs_error", ascending=False).head(50)[worst_columns].to_csv(
        output_dir / "validation_sign_errors.csv", index=False
    )

    report: dict[str, object] = {
        "overall": overall,
        "by_group": group_reports,
        "worst_atom_type_event_kind": combined.head(10).to_dict(orient="records"),
    }
    (output_dir / "validation_error_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def validate_models_on_independent_events(
    atoms: list[str],
    positions: np.ndarray,
    potential: PairPotentialPuO2,
    model_dir: str | Path,
    output_dir: str | Path,
    n_events: int = 1000,
    seed: int = 123,
    max_displacement: float = 0.35,
    min_distance: float = 1.2,
    order_biased_events: bool = False,
) -> dict[str, float]:
    """Validate saved ML models on newly generated independent events."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    regressor, classifier, _ = load_models(model_dir)
    classifier_threshold = load_classifier_threshold(model_dir)
    rng = np.random.default_rng(seed)
    feature_names = get_feature_names()
    rows: list[dict[str, object]] = []
    tree = cKDTree(positions)

    for _ in tqdm(range(n_events), desc="Independent ML validation"):
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
        features = make_event_features(atoms, positions, event, tree=tree, potential=potential)
        exact_delta_e = potential.delta_energy_single_move(atoms, positions, event.atom_index, event.new_position)
        predicted_delta_e = float(predict_delta_E(regressor, features.reshape(1, -1))[0])
        good_probability = float(predict_good_event_probability(classifier, features.reshape(1, -1))[0])
        uncertainty = float(estimate_uncertainty_rf(regressor, features.reshape(1, -1))[0])
        row = dict(zip(feature_names, features, strict=True))
        row.update(
            {
                "exact_delta_E": exact_delta_e,
                "predicted_delta_E": predicted_delta_e,
                "exact_is_energy_lowering": int(exact_delta_e < 0.0),
                "predicted_is_energy_lowering": int(good_probability >= classifier_threshold),
                "good_event_probability": good_probability,
                "uncertainty": uncertainty,
                "atom_type": atoms[event.atom_index],
                "event_kind": event.kind,
            }
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "validation_events.csv", index=False)
    y_true = df["exact_delta_E"].to_numpy(dtype=float)
    y_pred = df["predicted_delta_E"].to_numpy(dtype=float)
    y_true_cls = df["exact_is_energy_lowering"].to_numpy(dtype=int)
    y_pred_cls = df["predicted_is_energy_lowering"].to_numpy(dtype=int)
    metrics = {
        "n_events": int(len(df)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "sign_accuracy": float(accuracy_score(y_true_cls, y_pred_cls)),
        "precision_energy_lowering": float(precision_score(y_true_cls, y_pred_cls, pos_label=1, zero_division=0)),
        "recall_energy_lowering": float(recall_score(y_true_cls, y_pred_cls, pos_label=1, zero_division=0)),
        "mean_uncertainty": float(df["uncertainty"].mean()),
        "mean_abs_error": float(np.mean(np.abs(y_pred - y_true))),
        "classifier_threshold": float(classifier_threshold),
    }
    metrics.update(_ranking_metrics(df))
    (output_dir / "validation_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame([_ranking_metrics(df)]).to_csv(output_dir / "validation_ranking_metrics.csv", index=False)
    analyze_validation_errors(df, output_dir)
    _plot_predicted_vs_exact(df, output_dir)
    _plot_residuals(df, output_dir)
    _plot_uncertainty_vs_error(df, output_dir)
    return metrics


def validate_models_on_state_collection(
    states: list[tuple[str, list[str], np.ndarray]],
    potential: PairPotentialPuO2,
    model_dir: str | Path,
    output_dir: str | Path,
    n_events: int = 1000,
    seed: int = 123,
    max_displacement: float = 0.35,
    min_distance: float = 1.2,
    order_biased_events: bool = False,
) -> dict[str, float]:
    """Validate saved models on independent events sampled from several structures."""
    if not states:
        raise ValueError("At least one structural state is required for validation")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_state = n_events // len(states)
    remainder = n_events % len(states)
    all_frames: list[pd.DataFrame] = []
    state_metrics: dict[str, dict[str, float]] = {}
    for index, (state_name, atoms, positions) in enumerate(states):
        count = per_state + (1 if index < remainder else 0)
        if count <= 0:
            continue
        state_dir = output_dir / f"state_{index:02d}"
        metrics = validate_models_on_independent_events(
            atoms,
            positions,
            potential,
            model_dir,
            state_dir,
            n_events=count,
            seed=seed + index,
            max_displacement=max_displacement,
            min_distance=min_distance,
            order_biased_events=order_biased_events,
        )
        state_metrics[state_name] = metrics
        frame = pd.read_csv(state_dir / "validation_events.csv").copy()
        frame["source_state"] = state_name
        all_frames.append(frame)

    df = pd.concat(all_frames, ignore_index=True)
    df.to_csv(output_dir / "validation_events.csv", index=False)
    y_true = df["exact_delta_E"].to_numpy(dtype=float)
    y_pred = df["predicted_delta_E"].to_numpy(dtype=float)
    y_true_cls = df["exact_is_energy_lowering"].to_numpy(dtype=int)
    y_pred_cls = df["predicted_is_energy_lowering"].to_numpy(dtype=int)
    metrics = {
        "n_events": int(len(df)),
        "n_states": int(len(state_metrics)),
        "classifier_threshold": float(load_classifier_threshold(model_dir)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "sign_accuracy": float(accuracy_score(y_true_cls, y_pred_cls)),
        "precision_energy_lowering": float(precision_score(y_true_cls, y_pred_cls, pos_label=1, zero_division=0)),
        "recall_energy_lowering": float(recall_score(y_true_cls, y_pred_cls, pos_label=1, zero_division=0)),
        "mean_uncertainty": float(df["uncertainty"].mean()),
        "mean_abs_error": float(np.mean(np.abs(y_pred - y_true))),
    }
    metrics.update(_ranking_metrics(df))
    metrics["per_state"] = state_metrics
    (output_dir / "validation_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame([_ranking_metrics(df)]).to_csv(output_dir / "validation_ranking_metrics.csv", index=False)
    analyze_validation_errors(df, output_dir)
    _plot_predicted_vs_exact(df, output_dir)
    _plot_residuals(df, output_dir)
    _plot_uncertainty_vs_error(df, output_dir)
    return metrics
