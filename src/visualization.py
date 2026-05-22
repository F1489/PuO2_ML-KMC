"""Красивые matplotlib-графики для ML-kMC расчетов PuO2.

Энергии указаны в eV, расстояния в Å, время в s. Seaborn не используется.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import coordination_numbers, coordination_summary, radial_distribution_function
from .events import MIN_PAIR_DISTANCES

PLOT_COLORS = {
    "initial": "#2F6B9A",
    "final": "#D08C2E",
    "energy": "#1F6F5B",
    "uncertainty": "#7A3E65",
    "pu": "#6F4AA8",
    "o": "#D1495B",
    "grid": "#D9DEE7",
    "text": "#1F2328",
    "muted": "#65717E",
}

EVENT_KIND_LABELS = {
    "random_bulk": "случайное объемное",
    "random_surface": "случайное поверхностное",
    "relaxation": "релаксация",
    "surface": "поверхность",
    "surface_biased": "поверхностное",
    "coordination": "улучшение координации",
    "snap_to_fluorite_site": "привязка к узлу флюорита",
    "growth_front": "фронт роста",
    "local_cluster_affine": "коллективная перестройка",
    "surface_compression": "сжатие поверхности",
    "force_relaxation": "силовая релаксация",
}


def _apply_style() -> None:
    """Apply a clean publication-style matplotlib theme."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#2D333B",
            "axes.labelcolor": PLOT_COLORS["text"],
            "axes.titleweight": "bold",
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.color": PLOT_COLORS["text"],
            "ytick.color": PLOT_COLORS["text"],
            "legend.frameon": True,
            "legend.framealpha": 0.94,
            "legend.facecolor": "white",
            "legend.edgecolor": "#C9CED6",
            "legend.fontsize": 10,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "lines.solid_capstyle": "round",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def _polish_axes(ax, title: str) -> None:
    """Add Russian title, grid, and clean spines to one axis."""
    ax.set_title(title, pad=12)
    ax.grid(True, color=PLOT_COLORS["grid"], linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    if hasattr(ax, "spines"):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#AEB7C2")
        ax.spines["bottom"].set_color("#AEB7C2")


def _event_label(kind: str) -> str:
    rejected_prefix = "rejected_"
    if kind.startswith(rejected_prefix):
        base = kind[len(rejected_prefix) :]
        return f"отклонено: {EVENT_KIND_LABELS.get(base, base.replace('_', ' '))}"
    return EVENT_KIND_LABELS.get(kind, kind.replace("_", " "))


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def _annotate_last_point(ax, x_value: float, y_value: float, text: str, color: str) -> None:
    ax.annotate(
        text,
        xy=(x_value, y_value),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
        color=color,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": color, "alpha": 0.92},
        arrowprops={"arrowstyle": "->", "color": color, "lw": 1.1},
    )


def _metric_card(ax, title: str, value: str, subtitle: str, color: str) -> None:
    ax.set_facecolor("#F6F8FA")
    for spine in ax.spines.values():
        spine.set_color("#D0D7DE")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.05, 0.76, title, transform=ax.transAxes, fontsize=10, fontweight="bold", color=PLOT_COLORS["text"])
    ax.text(0.05, 0.40, value, transform=ax.transAxes, fontsize=15, fontweight="bold", color=color)
    ax.text(0.05, 0.17, subtitle, transform=ax.transAxes, fontsize=8.7, color=PLOT_COLORS["muted"])


def plot_energy_history(history_csv: str | Path, output_dir: str | Path) -> None:
    """Save Russian energy-vs-step plots."""
    _apply_style()
    df = pd.read_csv(history_csv)
    output_dir = Path(output_dir)
    if "total_energy" in df:
        valid = df.dropna(subset=["total_energy"])
        fig, ax = plt.subplots(figsize=(9.2, 5.4))
        ax.plot(valid["step"], valid["total_energy"], color=PLOT_COLORS["energy"], marker="o", markersize=4, linewidth=2)
        _polish_axes(ax, "Изменение полной потенциальной энергии")
        ax.set_xlabel("Шаг kMC")
        ax.set_ylabel("Полная энергия, эВ")
        _save(fig, output_dir / "energy_vs_step.png")
    if "energy_per_atom" in df:
        valid = df.dropna(subset=["energy_per_atom"])
        fig, ax = plt.subplots(figsize=(9.2, 5.4))
        ax.plot(valid["step"], valid["energy_per_atom"], color=PLOT_COLORS["energy"], marker="o", markersize=4, linewidth=2)
        _polish_axes(ax, "Изменение энергии на атом")
        ax.set_xlabel("Шаг kMC")
        ax.set_ylabel("Энергия на атом, эВ/атом")
        _save(fig, output_dir / "energy_per_atom_vs_step.png")
    if "energy_per_puo2" in df:
        valid = df.dropna(subset=["energy_per_puo2"])
        fig, ax = plt.subplots(figsize=(9.2, 5.4))
        ax.plot(valid["step"], valid["energy_per_puo2"], color="#4C6A92", marker="o", markersize=4, linewidth=2.4)
        if len(valid) >= 2:
            first = valid.iloc[0]
            last = valid.iloc[-1]
            delta = float(last["energy_per_puo2"] - first["energy_per_puo2"])
            ax.fill_between(valid["step"], valid["energy_per_puo2"], float(valid["energy_per_puo2"].max()), color="#4C6A92", alpha=0.08)
            _annotate_last_point(
                ax,
                float(last["step"]),
                float(last["energy_per_puo2"]),
                f"финал: {last['energy_per_puo2']:.4f}\nΔ = {delta:+.4f} эВ",
                "#4C6A92",
            )
        _polish_axes(ax, "Энергия на формульную единицу PuO2")
        ax.set_xlabel("Шаг kMC")
        ax.set_ylabel("Энергия, эВ/PuO2")
        _save(fig, output_dir / "energy_per_puo2_vs_step.png")


def plot_uncertainty_history(history_csv: str | Path, output_dir: str | Path) -> None:
    """Save uncertainty-vs-step plot."""
    _apply_style()
    df = pd.read_csv(history_csv)
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.plot(df["step"], df["uncertainty"], color=PLOT_COLORS["uncertainty"], linewidth=2)
    _polish_axes(ax, "Неопределенность ML-модели")
    ax.set_xlabel("Шаг kMC")
    ax.set_ylabel("Оценка неопределенности, эВ")
    _save(fig, Path(output_dir) / "uncertainty_vs_step.png")


def plot_event_diagnostics(history_csv: str | Path, output_dir: str | Path) -> None:
    """Save event-kind and adaptive-displacement diagnostics."""
    _apply_style()
    df = pd.read_csv(history_csv)
    output_dir = Path(output_dir)
    counts = df["accepted_event_kind"].map(_event_label).value_counts().sort_values()
    if len(counts) > 9:
        top = counts.sort_values(ascending=False).head(8)
        other = pd.Series({"прочие события": counts.drop(top.index).sum()})
        counts = pd.concat([other, top]).sort_values()
    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    ax.barh(counts.index, counts.values, color="#4C6A92")
    _polish_axes(ax, "Какие события выбирал kMC")
    ax.set_xlabel("Число шагов")
    ax.tick_params(axis="y", labelsize=9)
    _save(fig, output_dir / "event_kind_counts.png")

    if "current_max_displacement" in df:
        fig, ax = plt.subplots(figsize=(9.2, 5.4))
        ax.plot(df["step"], df["current_max_displacement"], color="#8A6F2A", linewidth=2)
        _polish_axes(ax, "Адаптация максимального смещения")
        ax.set_xlabel("Шаг kMC")
        ax.set_ylabel("Максимальное смещение, Å")
        _save(fig, output_dir / "adaptive_displacement_vs_step.png")

    if "candidate_events_rejected" in df:
        fig, ax = plt.subplots(figsize=(9.2, 5.4))
        ratio = df["candidate_events_rejected"] / df["candidate_events_requested"].clip(lower=1)
        ax.plot(df["step"], ratio, color="#A33D3D", linewidth=2)
        _polish_axes(ax, "Доля отброшенных кандидатов")
        ax.set_xlabel("Шаг kMC")
        ax.set_ylabel("Отброшено / запрошено")
        _save(fig, output_dir / "rejected_candidates_vs_step.png")


def plot_min_pair_distances_history(history_csv: str | Path, output_dir: str | Path) -> None:
    """Save minimum Pu-O, O-O, and Pu-Pu distances vs kMC step."""
    _apply_style()
    df = pd.read_csv(history_csv)
    columns = ["min_distance_pu_o", "min_distance_o_o", "min_distance_pu_pu"]
    if not set(columns).issubset(df.columns):
        return
    valid = df.dropna(subset=columns, how="all")
    if valid.empty:
        return
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    styles = {
        "min_distance_pu_o": ("Pu-O", "#245C4F", MIN_PAIR_DISTANCES[("Pu", "O")]),
        "min_distance_o_o": ("O-O", "#D1495B", MIN_PAIR_DISTANCES[("O", "O")]),
        "min_distance_pu_pu": ("Pu-Pu", "#7B3F98", MIN_PAIR_DISTANCES[("Pu", "Pu")]),
    }
    for column in columns:
        label, color, threshold = styles[column]
        ax.plot(valid["step"], valid[column], marker="o", linewidth=2, color=color, label=label)
        ax.axhline(threshold, color=color, linewidth=1.1, linestyle="--", alpha=0.55)
    _polish_axes(ax, "Минимальные межатомные расстояния")
    ax.set_xlabel("Шаг kMC")
    ax.set_ylabel("Расстояние, Å")
    ax.legend()
    _save(fig, Path(output_dir) / "min_pair_distances_vs_step.png")


def plot_defects_history(history_csv: str | Path, output_dir: str | Path) -> None:
    """Save a graph of fluorite coordination defects vs kMC step."""
    _apply_style()
    df = pd.read_csv(history_csv)
    needed = {"total_coordination_defects", "pu_coordination_defects", "o_coordination_defects"}
    if not needed.issubset(df.columns):
        return
    valid = df.dropna(subset=["total_coordination_defects"])
    if valid.empty:
        return
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.plot(valid["step"], valid["total_coordination_defects"], marker="o", linewidth=2, color="#30343B", label="всего")
    ax.plot(valid["step"], valid["pu_coordination_defects"], marker="o", linewidth=2, color=PLOT_COLORS["pu"], label="Pu не с 8 O")
    ax.plot(valid["step"], valid["o_coordination_defects"], marker="o", linewidth=2, color=PLOT_COLORS["o"], label="O не с 4 Pu")
    _polish_axes(ax, "Число координационных дефектов")
    ax.set_xlabel("Шаг kMC")
    ax.set_ylabel("Число дефектных атомов")
    ax.legend()
    _save(fig, Path(output_dir) / "defects_vs_step.png")

    extra_needed = {"bulk_coordination_defects", "surface_coordination_defects", "mean_abs_coordination_error"}
    if extra_needed.issubset(df.columns):
        valid = df.dropna(subset=["bulk_coordination_defects"])
        if not valid.empty:
            fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
            axes[0].plot(
                valid["step"],
                valid["bulk_coordination_defects"],
                marker="o",
                linewidth=2,
                color="#245C4F",
                label="объем",
            )
            axes[0].plot(
                valid["step"],
                valid["surface_coordination_defects"],
                marker="o",
                linewidth=2,
                color="#8A6F2A",
                label="поверхность",
            )
            _polish_axes(axes[0], "Дефекты координации: объем и поверхность")
            axes[0].set_xlabel("Шаг kMC")
            axes[0].set_ylabel("Число дефектных атомов")
            axes[0].legend()
            axes[1].plot(
                valid["step"],
                valid["mean_abs_coordination_error"],
                marker="o",
                linewidth=2,
                color="#4C6A92",
            )
            _polish_axes(axes[1], "Средняя ошибка координации")
            axes[1].set_xlabel("Шаг kMC")
            axes[1].set_ylabel("Среднее |координация - идеал|")
            _save(fig, Path(output_dir) / "bulk_surface_defects_vs_step.png")


def plot_delta_e_history(history_csv: str | Path, output_dir: str | Path) -> None:
    """Save predicted and exact Delta E diagnostics vs kMC step."""
    _apply_style()
    df = pd.read_csv(history_csv)
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.plot(df["step"], df["predicted_delta_E"], color="#2F6B9A", linewidth=1.8, label="ML-прогноз")
    exact = df.dropna(subset=["exact_delta_E_if_checked"])
    if not exact.empty:
        ax.scatter(
            exact["step"],
            exact["exact_delta_E_if_checked"],
            s=34,
            color="#D08C2E",
            edgecolor="white",
            linewidth=0.5,
            label="точный пересчет",
            zorder=3,
        )
    ax.axhline(0.0, color="#30343B", linewidth=1.2)
    _polish_axes(ax, "Изменение Delta E выбранных событий")
    ax.set_xlabel("Шаг kMC")
    ax.set_ylabel("Delta E, эВ")
    ax.legend()
    _save(fig, Path(output_dir) / "delta_e_vs_step.png")


def plot_acceptance_history(history_csv: str | Path, output_dir: str | Path) -> None:
    """Save cumulative accepted/rejected event diagnostics."""
    _apply_style()
    df = pd.read_csv(history_csv)
    if "event_applied" not in df:
        return
    steps = df["step"]
    applied = df["event_applied"].astype(float)
    cumulative_acceptance = applied.expanding().mean()
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.plot(steps, cumulative_acceptance, color="#245C4F", linewidth=2)
    _polish_axes(ax, "Накопленная доля принятых событий")
    ax.set_xlabel("Шаг kMC")
    ax.set_ylabel("Доля принятых событий")
    ax.set_ylim(-0.02, 1.02)
    _save(fig, Path(output_dir) / "acceptance_ratio_vs_step.png")


def plot_order_score_history(history_csv: str | Path, output_dir: str | Path) -> None:
    """Save fluorite order score vs kMC step."""
    _apply_style()
    df = pd.read_csv(history_csv)
    if "fluorite_order_score" not in df:
        return
    valid = df.dropna(subset=["fluorite_order_score"])
    if valid.empty:
        return
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.plot(valid["step"], valid["fluorite_order_score"], marker="o", color="#7A3E65", linewidth=2)
    _polish_axes(ax, "Метрика флюоритного порядка")
    ax.set_xlabel("Шаг kMC")
    ax.set_ylabel("Показатель флюоритного порядка")
    ax.set_ylim(-0.02, 1.02)
    _save(fig, Path(output_dir) / "fluorite_order_score_vs_step.png")

    score_columns = [
        column
        for column in ["fluorite_order_score", "bulk_fluorite_order_score", "soft_coordination_order_score"]
        if column in df.columns
    ]
    if len(score_columns) > 1:
        valid = df.dropna(subset=score_columns, how="all")
        if not valid.empty:
            fig, ax = plt.subplots(figsize=(9.2, 5.4))
            labels = {
                "fluorite_order_score": "строгий, все атомы",
                "bulk_fluorite_order_score": "строгий, объем",
                "soft_coordination_order_score": "мягкая координация",
            }
            colors = {
                "fluorite_order_score": "#7A3E65",
                "bulk_fluorite_order_score": "#245C4F",
                "soft_coordination_order_score": "#4C6A92",
            }
            for column in score_columns:
                ax.plot(valid["step"], valid[column], marker="o", linewidth=2.4, color=colors[column], label=labels[column])
                column_valid = valid.dropna(subset=[column])
                if len(column_valid) >= 2:
                    last = column_valid.iloc[-1]
                    _annotate_last_point(ax, float(last["step"]), float(last[column]), f"{last[column]:.3f}", colors[column])
            _polish_axes(ax, "Показатели кристаллизационного порядка")
            ax.set_xlabel("Шаг kMC")
            ax.set_ylabel("Показатель порядка")
            ax.set_ylim(-0.02, 1.02)
            ax.legend()
            _save(fig, Path(output_dir) / "crystallization_order_scores_vs_step.png")

    if "rdf_pu_o_peak_sharpness" in df:
        valid = df.dropna(subset=["rdf_pu_o_peak_sharpness"])
        if not valid.empty:
            fig, ax = plt.subplots(figsize=(9.2, 5.4))
            ax.plot(valid["step"], valid["rdf_pu_o_peak_sharpness"], marker="o", color="#A33D3D", linewidth=2)
            _polish_axes(ax, "Резкость пика RDF для Pu-O")
            ax.set_xlabel("Шаг kMC")
            ax.set_ylabel("Пик / среднее RDF")
            _save(fig, Path(output_dir) / "rdf_pu_o_peak_sharpness_vs_step.png")

    if {"crystalline_core_size", "growth_front_size"}.issubset(df.columns):
        valid = df.dropna(subset=["crystalline_core_size", "growth_front_size"], how="all")
        if not valid.empty:
            fig, ax = plt.subplots(figsize=(9.2, 5.4))
            ax.plot(valid["step"], valid["crystalline_core_size"], marker="o", color="#245C4F", linewidth=2, label="кристаллическое ядро")
            ax.plot(valid["step"], valid["growth_front_size"], marker="o", color="#D08C2E", linewidth=2, label="фронт роста")
            _polish_axes(ax, "Размер кристаллического ядра и фронта роста")
            ax.set_xlabel("Шаг kMC")
            ax.set_ylabel("Число атомов")
            ax.legend()
            _save(fig, Path(output_dir) / "crystalline_core_size_vs_step.png")


def plot_kmc_time_history(history_csv: str | Path, output_dir: str | Path) -> None:
    """Save kMC time and total rate diagnostics."""
    _apply_style()
    df = pd.read_csv(history_csv)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    axes[0].plot(df["step"], df["kmc_time"], color="#4C6A92", linewidth=2)
    _polish_axes(axes[0], "Накопленное kMC-время")
    axes[0].set_xlabel("Шаг kMC")
    axes[0].set_ylabel("Время, с")
    axes[1].plot(df["step"], df["total_rate"], color="#A33D3D", linewidth=2)
    axes[1].set_yscale("log")
    _polish_axes(axes[1], "Суммарная скорость событий")
    axes[1].set_xlabel("Шаг kMC")
    axes[1].set_ylabel("Скорость, с⁻¹")
    _save(fig, Path(output_dir) / "kmc_time_and_rate_vs_step.png")


def plot_rdf_initial_final(
    atoms_initial: list[str],
    positions_initial: np.ndarray,
    atoms_final: list[str],
    positions_final: np.ndarray,
    output_dir: str | Path,
    r_max: float = 8.0,
    dr: float = 0.1,
) -> None:
    """Save total and pair-resolved initial/final RDF plots."""
    _apply_style()
    output_dir = Path(output_dir)
    r_i, g_i = radial_distribution_function(atoms_initial, positions_initial, r_max, dr)
    r_f, g_f = radial_distribution_function(atoms_final, positions_final, r_max, dr)
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.plot(r_i, g_i, label="начальная структура", color=PLOT_COLORS["initial"], linewidth=2)
    ax.plot(r_f, g_f, label="после ML-kMC", color=PLOT_COLORS["final"], linewidth=2)
    _polish_axes(ax, "Радиальная функция распределения")
    ax.set_xlabel("Расстояние r, Å")
    ax.set_ylabel("Нормированное число пар")
    ax.legend(loc="best")
    _save(fig, output_dir / "rdf_initial_final.png")

    for pair in [("Pu", "O"), ("Pu", "Pu"), ("O", "O")]:
        r_i, g_i = radial_distribution_function(atoms_initial, positions_initial, r_max, dr, pair=pair)
        r_f, g_f = radial_distribution_function(atoms_final, positions_final, r_max, dr, pair=pair)
        fig, ax = plt.subplots(figsize=(9.2, 5.4))
        ax.plot(r_i, g_i, label="начальная структура", color=PLOT_COLORS["initial"], linewidth=2)
        ax.plot(r_f, g_f, label="после ML-kMC", color=PLOT_COLORS["final"], linewidth=2)
        _polish_axes(ax, f"RDF для пары {pair[0]}-{pair[1]}")
        ax.set_xlabel("Расстояние r, Å")
        ax.set_ylabel("Нормированное число пар")
        ax.legend(loc="best")
        _save(fig, output_dir / f"rdf_{pair[0].lower()}_{pair[1].lower()}_initial_final.png")


def plot_coordination_hist_initial_final(
    atoms_initial: list[str],
    positions_initial: np.ndarray,
    atoms_final: list[str],
    positions_final: np.ndarray,
    output_dir: str | Path,
    cutoff_pu_o: float = 3.2,
) -> None:
    """Save initial/final coordination histograms."""
    _apply_style()
    c_i = coordination_numbers(atoms_initial, positions_initial, cutoff_pu_o)
    c_f = coordination_numbers(atoms_final, positions_final, cutoff_pu_o)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    bins = np.arange(0, 13) - 0.5
    axes[0].hist(c_i["Pu_O"], bins=bins, alpha=0.68, label="начальная", color=PLOT_COLORS["initial"], edgecolor="white")
    axes[0].hist(c_f["Pu_O"], bins=bins, alpha=0.68, label="после ML-kMC", color=PLOT_COLORS["final"], edgecolor="white")
    _polish_axes(axes[0], "Pu: число O-соседей")
    axes[0].set_xlabel("Координационное число")
    axes[0].set_ylabel("Число атомов")
    axes[0].legend()
    axes[1].hist(c_i["O_Pu"], bins=bins, alpha=0.68, label="начальная", color=PLOT_COLORS["initial"], edgecolor="white")
    axes[1].hist(c_f["O_Pu"], bins=bins, alpha=0.68, label="после ML-kMC", color=PLOT_COLORS["final"], edgecolor="white")
    _polish_axes(axes[1], "O: число Pu-соседей")
    axes[1].set_xlabel("Координационное число")
    axes[1].legend()
    _save(fig, Path(output_dir) / "coordination_hist_initial_final.png")


def plot_coordination_summary_initial_final(
    atoms_initial: list[str],
    positions_initial: np.ndarray,
    atoms_final: list[str],
    positions_final: np.ndarray,
    output_dir: str | Path,
) -> None:
    """Save a bar chart of average coordination and ideal-site fractions."""
    _apply_style()
    initial = coordination_summary(atoms_initial, positions_initial)
    final = coordination_summary(atoms_final, positions_final)
    labels = ["ср. Pu-O", "ср. O-Pu", "Pu с 8 O", "O с 4 Pu", "флюоритный порядок"]
    keys = [
        "mean_pu_o_coordination",
        "mean_o_pu_coordination",
        "fraction_pu_with_8_o",
        "fraction_o_with_4_pu",
        "fluorite_order_score",
    ]
    x = np.arange(len(keys))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.bar(x - width / 2, [initial[k] for k in keys], width, label="начальная", color=PLOT_COLORS["initial"])
    ax.bar(x + width / 2, [final[k] for k in keys], width, label="после ML-kMC", color=PLOT_COLORS["final"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    _polish_axes(ax, "Координационные показатели структуры")
    ax.set_ylabel("Значение")
    ax.legend()
    _save(fig, Path(output_dir) / "coordination_summary_initial_final.png")

    crystal_labels = ["строгий, все", "строгий, объем", "мягкая координация", "резкость RDF Pu-O"]
    crystal_keys = [
        "fluorite_order_score",
        "bulk_fluorite_order_score",
        "soft_coordination_order_score",
        "rdf_pu_o_peak_sharpness",
    ]
    x = np.arange(len(crystal_keys))
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.bar(x - width / 2, [initial[k] for k in crystal_keys], width, label="начальная", color=PLOT_COLORS["initial"])
    ax.bar(x + width / 2, [final[k] for k in crystal_keys], width, label="после ML-kMC", color=PLOT_COLORS["final"])
    ax.set_xticks(x)
    ax.set_xticklabels(crystal_labels)
    _polish_axes(ax, "Индикаторы кристаллизации")
    ax.set_ylabel("Значение")
    ax.legend()
    _save(fig, Path(output_dir) / "crystallization_summary_initial_final.png")


def plot_final_cluster_3d(atoms: list[str], positions: np.ndarray, output_dir: str | Path) -> None:
    """Save a simple 3D visualization of the final cluster."""
    _apply_style()
    positions = np.asarray(positions, dtype=float)
    pu_mask = np.array([atom == "Pu" for atom in atoms])
    fig = plt.figure(figsize=(8.2, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        positions[pu_mask, 0],
        positions[pu_mask, 1],
        positions[pu_mask, 2],
        s=42,
        color=PLOT_COLORS["pu"],
        label="Pu",
        alpha=0.86,
        depthshade=True,
    )
    ax.scatter(
        positions[~pu_mask, 0],
        positions[~pu_mask, 1],
        positions[~pu_mask, 2],
        s=18,
        color=PLOT_COLORS["o"],
        label="O",
        alpha=0.76,
        depthshade=True,
    )
    ax.set_title("Финальная структура кластера PuO2", pad=14, fontweight="bold")
    ax.set_xlabel("x, Å")
    ax.set_ylabel("y, Å")
    ax.set_zlabel("z, Å")
    ax.legend(loc="upper right")
    _save(fig, Path(output_dir) / "final_cluster_3d.png")


def _scatter_cluster_3d(ax, atoms: list[str], positions: np.ndarray, title: str) -> None:
    positions = np.asarray(positions, dtype=float)
    pu_mask = np.array([atom == "Pu" for atom in atoms])
    ax.scatter(
        positions[pu_mask, 0],
        positions[pu_mask, 1],
        positions[pu_mask, 2],
        s=36,
        color=PLOT_COLORS["pu"],
        label="Pu",
        alpha=0.86,
        depthshade=True,
    )
    ax.scatter(
        positions[~pu_mask, 0],
        positions[~pu_mask, 1],
        positions[~pu_mask, 2],
        s=15,
        color=PLOT_COLORS["o"],
        label="O",
        alpha=0.74,
        depthshade=True,
    )
    ax.set_title(title, pad=10, fontweight="bold")
    ax.set_xlabel("x, Å")
    ax.set_ylabel("y, Å")
    ax.set_zlabel("z, Å")


def plot_initial_final_cluster_3d(
    atoms_initial: list[str],
    positions_initial: np.ndarray,
    atoms_final: list[str],
    positions_final: np.ndarray,
    output_dir: str | Path,
) -> None:
    """Save Russian before/after 3D cluster images."""
    _apply_style()
    output_dir = Path(output_dir)

    fig = plt.figure(figsize=(8.2, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    _scatter_cluster_3d(ax, atoms_initial, positions_initial, "Начальная структура PuO2")
    ax.legend(loc="upper right")
    _save(fig, output_dir / "initial_structure.png")

    fig = plt.figure(figsize=(13.2, 6.6))
    ax_initial = fig.add_subplot(121, projection="3d")
    ax_final = fig.add_subplot(122, projection="3d")
    _scatter_cluster_3d(ax_initial, atoms_initial, positions_initial, "До kMC")
    _scatter_cluster_3d(ax_final, atoms_final, positions_final, "После kMC")
    handles, labels = ax_final.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=True, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle("Сравнение структуры PuO2 до и после расчета", fontsize=14, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.10, top=0.86, wspace=0.08)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "initial_final_structure.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_seeded_stage_comparison(output_dir: str | Path) -> None:
    """Save a Russian stage-comparison plot for seeded runs when summary.json contains it."""
    _apply_style()
    output_dir = Path(output_dir)
    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    comparison = summary.get("state_comparison_after_repair_after_stage1_final")
    if not isinstance(comparison, dict):
        return
    states = ["after_repair", "after_stage1", "final"]
    if not all(state in comparison for state in states):
        return
    labels = ["после repair", "после стадии 1", "финал"]
    metrics = [
        ("energy_per_puo2_eV", "E/PuO2, эВ", "Энергия на PuO2"),
        ("bulk_fluorite_order_score", "Доля", "Флюоритный порядок в объеме"),
        ("mean_abs_coordination_error", "Ошибка", "Средняя ошибка координации"),
        ("fraction_pu_with_8_o", "Доля", "Pu с 8 соседями O"),
        ("fraction_o_with_4_pu", "Доля", "O с 4 соседями Pu"),
    ]
    colors = ["#1F6F5B", "#2F6B9A", "#A33D3D", "#6F4AA8", "#D08C2E"]
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.0))
    axes_flat = axes.ravel()
    for ax, (metric, ylabel, title), color in zip(axes_flat, metrics, colors, strict=False):
        values = [float(comparison[state][metric]) for state in states]
        ax.plot(labels, values, marker="o", markersize=7, linewidth=2.4, color=color)
        ax.scatter(labels, values, s=58, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        _polish_axes(ax, title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=0, labelsize=8.8)
        ax.title.set_fontsize(12)
    axes_flat[-1].axis("off")
    deltas = comparison.get("deltas", {})
    lines = [
        "Финал - после repair:",
        f"ΔE/PuO2 = {deltas.get('final_minus_after_repair_energy_per_puo2_eV', 0.0):+.4f} эВ",
        f"bulk order = {deltas.get('final_minus_after_repair_bulk_order_score', 0.0):+.4f}",
        f"coord. error = {deltas.get('final_minus_after_repair_mean_abs_coordination_error', 0.0):+.4f}",
        f"Pu с 8 O = {deltas.get('final_minus_after_repair_fraction_pu_with_8_o', 0.0):+.4f}",
        f"O с 4 Pu = {deltas.get('final_minus_after_repair_fraction_o_with_4_pu', 0.0):+.4f}",
    ]
    axes_flat[-1].text(
        0.02,
        0.92,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=9.8,
        color=PLOT_COLORS["text"],
        bbox={"boxstyle": "round,pad=0.55", "facecolor": "#F6F8FA", "edgecolor": "#D0D7DE"},
    )
    fig.suptitle("Seeded-расчет: после repair → после стадии 1 → финал", fontsize=14, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.10, top=0.86, hspace=0.58, wspace=0.38)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "seeded_stage_comparison.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_result_dashboard(output_dir: str | Path) -> None:
    """Save a presentation-style Russian dashboard with the most important result metrics."""
    _apply_style()
    output_dir = Path(output_dir)
    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    seeded = summary.get("state_comparison_after_repair_after_stage1_final")

    if isinstance(seeded, dict):
        states = ["after_repair", "after_stage1", "final"]
        stage_labels = ["после repair", "стадия 1", "финал"]
        energy = [float(seeded[state]["energy_per_puo2_eV"]) for state in states]
        bulk = [float(seeded[state]["bulk_fluorite_order_score"]) for state in states]
        coord_error = [float(seeded[state]["mean_abs_coordination_error"]) for state in states]
        min_pu_o = float(seeded["final"]["min_pair_distances"]["min_distance_pu_o"])
        delta_energy = float(seeded["deltas"]["final_minus_after_repair_energy_per_puo2_eV"])
        title = "Seeded-расчет: энергия и флюоритоподобный порядок"
        energy_subtitle = "финал - после repair"
    else:
        stage_labels = ["старт", "финал"]
        energy = [float(summary["initial_energy_per_puo2_eV"]), float(summary["final_energy_per_puo2_eV"])]
        initial_structure = summary["initial_structure"]
        final_structure = summary["final_structure"]
        bulk = [float(initial_structure["bulk_fluorite_order_score"]), float(final_structure["bulk_fluorite_order_score"])]
        coord_error = [float(initial_structure["mean_abs_coordination_error"]), float(final_structure["mean_abs_coordination_error"])]
        min_pu_o = float(summary["final_min_pair_distances"]["min_distance_pu_o"])
        delta_energy = float(summary["delta_total_energy_per_puo2_eV"])
        title = "Основной результат ML-kMC: релаксация и локальный порядок"
        energy_subtitle = "финал - старт"

    fig = plt.figure(figsize=(12.8, 7.2))
    gs = fig.add_gridspec(3, 4, height_ratios=[0.92, 2.1, 2.1], hspace=0.55, wspace=0.5)
    card_axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    _metric_card(card_axes[0], "ΔE/PuO2", f"{delta_energy:+.4f} эВ", energy_subtitle, "#1F6F5B")
    _metric_card(card_axes[1], "Порядок", f"{bulk[-1]:.4f}", f"старт: {bulk[0]:.4f}", "#245C4F")
    _metric_card(card_axes[2], "Ошибка коорд.", f"{coord_error[-1]:.4f}", f"старт: {coord_error[0]:.4f}", "#A33D3D")
    _metric_card(card_axes[3], "Min Pu-O", f"{min_pu_o:.4f} Å", "порог: 1.9000 Å", "#6F4AA8")

    ax_energy = fig.add_subplot(gs[1:, :2])
    x = np.arange(len(stage_labels))
    ax_energy.plot(x, energy, marker="o", markersize=8, linewidth=3.0, color="#1F6F5B")
    ax_energy.fill_between(x, energy, max(energy), color="#1F6F5B", alpha=0.10)
    y_span = max(energy) - min(energy)
    label_offset = max(y_span * 0.035, 0.005)
    for xi, yi in zip(x, energy, strict=True):
        va = "bottom" if yi < max(energy) else "top"
        offset = label_offset if va == "bottom" else -label_offset
        ax_energy.text(xi, yi + offset, f"{yi:.4f}", ha="center", va=va, fontsize=8.8, fontweight="bold", color="#1F6F5B")
    ax_energy.set_xticks(x)
    ax_energy.set_xticklabels(stage_labels)
    ax_energy.set_ylabel("E/PuO2, эВ", fontsize=11)
    ax_energy.tick_params(axis="x", labelsize=9)
    _polish_axes(ax_energy, "Энергия")

    ax_order = fig.add_subplot(gs[1, 2:])
    width = 0.34
    ax_order.bar(x - width / 2, bulk, width, color="#245C4F", label="порядок в объеме")
    ax_order.bar(x + width / 2, coord_error, width, color="#D08C2E", label="ошибка координации")
    ax_order.set_xticks(x)
    ax_order.set_xticklabels(stage_labels)
    ax_order.set_ylabel("Значение")
    ax_order.legend(fontsize=9)
    _polish_axes(ax_order, "Порядок и ошибка координации")

    ax_claim = fig.add_subplot(gs[2, 2:])
    ax_claim.axis("off")
    claim = (
        "Вывод:\n"
        "полная рекристаллизация не утверждается;\n"
        "показаны энергетическая релаксация,\n"
        "отжиг локальных дефектов и рост\n"
        "флюоритоподобного порядка."
    )
    ax_claim.text(
        0.02,
        0.92,
        claim,
        va="top",
        ha="left",
        fontsize=10.5,
        color=PLOT_COLORS["text"],
        bbox={"boxstyle": "round,pad=0.75", "facecolor": "#F6F8FA", "edgecolor": "#D0D7DE"},
    )
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.08, top=0.86, hspace=0.60, wspace=0.40)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "result_dashboard.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_all_plots(
    history_csv: str | Path,
    atoms_initial: list[str],
    positions_initial: np.ndarray,
    atoms_final: list[str],
    positions_final: np.ndarray,
    output_dir: str | Path,
) -> None:
    """Save all standard kMC diagnostic plots."""
    plot_energy_history(history_csv, output_dir)
    plot_uncertainty_history(history_csv, output_dir)
    plot_event_diagnostics(history_csv, output_dir)
    plot_min_pair_distances_history(history_csv, output_dir)
    plot_defects_history(history_csv, output_dir)
    plot_delta_e_history(history_csv, output_dir)
    plot_acceptance_history(history_csv, output_dir)
    plot_order_score_history(history_csv, output_dir)
    plot_kmc_time_history(history_csv, output_dir)
    plot_rdf_initial_final(atoms_initial, positions_initial, atoms_final, positions_final, output_dir)
    plot_coordination_hist_initial_final(atoms_initial, positions_initial, atoms_final, positions_final, output_dir)
    plot_coordination_summary_initial_final(atoms_initial, positions_initial, atoms_final, positions_final, output_dir)
    plot_initial_final_cluster_3d(atoms_initial, positions_initial, atoms_final, positions_final, output_dir)
    plot_final_cluster_3d(atoms_final, positions_final, output_dir)
    plot_seeded_stage_comparison(output_dir)
    plot_result_dashboard(output_dir)
