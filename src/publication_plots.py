"""Publication-style figures for PuO2 ML-kMC results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "green": "#2A6F62",
    "blue": "#3E6FA3",
    "orange": "#C9832B",
    "red": "#B44A4A",
    "purple": "#7256A5",
    "gray": "#59636E",
    "light_grid": "#E6E9EF",
    "text": "#222831",
}


def apply_publication_style() -> None:
    """Apply a restrained scientific matplotlib style."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#AAB2BD",
            "axes.labelcolor": COLORS["text"],
            "axes.linewidth": 0.9,
            "axes.titlelocation": "left",
            "axes.titleweight": "bold",
            "axes.titlesize": 12.5,
            "axes.labelsize": 10.5,
            "xtick.color": COLORS["text"],
            "ytick.color": COLORS["text"],
            "xtick.labelsize": 9.3,
            "ytick.labelsize": 9.3,
            "legend.frameon": True,
            "legend.framealpha": 0.96,
            "legend.facecolor": "white",
            "legend.edgecolor": "#D8DDE5",
            "legend.fontsize": 9.0,
            "font.family": "DejaVu Sans",
            "font.size": 10.0,
            "lines.linewidth": 2.15,
            "lines.markersize": 5.8,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.dpi": 320,
        }
    )


def polish(ax, title: str, xlabel: str | None = None, ylabel: str | None = None) -> None:
    ax.set_title(title, pad=10)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.grid(True, color=COLORS["light_grid"], linewidth=0.8, alpha=0.85)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#C3CAD4")
    ax.spines["bottom"].set_color("#C3CAD4")


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=320)
    plt.close(fig)


def _last_valid(history: pd.DataFrame, column: str) -> pd.DataFrame:
    return history.dropna(subset=[column])


def _summary_text(ax, lines: list[str]) -> None:
    ax.axis("off")
    ax.text(
        0.02,
        0.96,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=10.2,
        color=COLORS["text"],
        linespacing=1.45,
        bbox={
            "boxstyle": "round,pad=0.65",
            "facecolor": "#F7F9FC",
            "edgecolor": "#D7DDE7",
            "linewidth": 1.0,
        },
    )


def plot_main_publication_figures(run_dir: str | Path, output_dir: str | Path | None = None) -> None:
    """Create polished summary figures for a normal ML-kMC run directory."""
    apply_publication_style()
    run_dir = Path(run_dir)
    output_dir = run_dir / "publication_figures" if output_dir is None else Path(output_dir)
    history = pd.read_csv(run_dir / "history.csv")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    fig = plt.figure(figsize=(12.5, 7.0))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.35, 1.0, 1.0], height_ratios=[1.0, 1.0], hspace=0.42, wspace=0.34)

    ax_energy = fig.add_subplot(gs[:, 0])
    energy = _last_valid(history, "energy_per_puo2")
    ax_energy.plot(energy["step"], energy["energy_per_puo2"], color=COLORS["green"], marker="o")
    if len(energy) >= 2:
        first = energy.iloc[0]
        last = energy.iloc[-1]
        ax_energy.fill_between(
            energy["step"],
            energy["energy_per_puo2"],
            float(energy["energy_per_puo2"].max()),
            color=COLORS["green"],
            alpha=0.10,
        )
        ax_energy.annotate(
            f"final {last['energy_per_puo2']:.4f} eV/PuO2\nDelta {last['energy_per_puo2'] - first['energy_per_puo2']:+.4f}",
            xy=(float(last["step"]), float(last["energy_per_puo2"])),
            xytext=(-118, 28),
            textcoords="offset points",
            fontsize=9.0,
            color=COLORS["green"],
            arrowprops={"arrowstyle": "->", "color": COLORS["green"], "lw": 1.0},
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": COLORS["green"], "alpha": 0.95},
        )
    polish(ax_energy, "Energy relaxation", "kMC step", "Energy, eV/PuO2")

    ax_order = fig.add_subplot(gs[0, 1])
    order_cols = ["fluorite_order_score", "bulk_fluorite_order_score", "soft_coordination_order_score"]
    order = history.dropna(subset=order_cols, how="all")
    labels = {
        "fluorite_order_score": "global",
        "bulk_fluorite_order_score": "bulk",
        "soft_coordination_order_score": "soft coord.",
    }
    colors = [COLORS["purple"], COLORS["blue"], COLORS["orange"]]
    for column, color in zip(order_cols, colors, strict=True):
        if column in order:
            ax_order.plot(order["step"], order[column], marker="o", color=color, label=labels[column])
    ax_order.set_ylim(0.45, 0.86)
    polish(ax_order, "Fluorite-like order", "kMC step", "score")
    ax_order.legend(loc="lower right")

    ax_dist = fig.add_subplot(gs[0, 2])
    dist_cols = ["min_distance_pu_o", "min_distance_o_o", "min_distance_pu_pu"]
    distances = history.dropna(subset=dist_cols, how="all")
    for column, label, color, threshold in [
        ("min_distance_pu_o", "Pu-O", COLORS["green"], 1.9),
        ("min_distance_o_o", "O-O", COLORS["red"], 2.0),
        ("min_distance_pu_pu", "Pu-Pu", COLORS["purple"], 2.8),
    ]:
        ax_dist.plot(distances["step"], distances[column], marker="o", color=color, label=label)
        ax_dist.axhline(threshold, color=color, linestyle="--", linewidth=1.0, alpha=0.55)
    polish(ax_dist, "Close-contact safety", "kMC step", "minimum distance, A")
    ax_dist.legend(loc="lower right")

    ax_events = fig.add_subplot(gs[1, 1])
    event_counts = history["accepted_event_kind"].value_counts().head(7).sort_values()
    event_counts.plot(kind="barh", ax=ax_events, color=COLORS["blue"], width=0.68)
    polish(ax_events, "Accepted event mix", "count", None)
    ax_events.tick_params(axis="y", labelsize=8.2)

    ax_text = fig.add_subplot(gs[1, 2])
    final_dist = summary["final_min_pair_distances"]
    lines = [
        "Final 5000-step ML-kMC run",
        f"Delta E total: {summary['delta_total_energy_per_puo2_eV']:+.4f} eV/PuO2",
        f"Delta E kMC: {summary['delta_kmc_energy_per_puo2_eV']:+.4f} eV/PuO2",
        f"Acceptance: {history['event_applied'].mean():.3f}",
        f"Bulk order: {summary['initial_structure']['bulk_fluorite_order_score']:.3f} -> {summary['final_structure']['bulk_fluorite_order_score']:.3f}",
        f"Coord. error: {summary['initial_structure']['mean_abs_coordination_error']:.3f} -> {summary['final_structure']['mean_abs_coordination_error']:.3f}",
        f"Min Pu-O/O-O/Pu-Pu: {final_dist['min_distance_pu_o']:.3f} / {final_dist['min_distance_o_o']:.3f} / {final_dist['min_distance_pu_pu']:.3f} A",
    ]
    _summary_text(ax_text, lines)

    fig.suptitle("PuO2 ML-kMC: energy relaxation and partial fluorite-like ordering", fontsize=15.2, fontweight="bold", y=0.985)
    save(fig, output_dir / "main_result_summary.png")

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    exact = history.dropna(subset=["exact_delta_E_if_checked"])
    axes[0].plot(history["step"], history["predicted_delta_E"], color=COLORS["blue"], alpha=0.92, label="ML prediction")
    if not exact.empty:
        axes[0].scatter(exact["step"], exact["exact_delta_E_if_checked"], s=24, color=COLORS["orange"], edgecolor="white", linewidth=0.4, label="exact checked")
    axes[0].axhline(0.0, color="#424A54", linewidth=1.0)
    polish(axes[0], "Selected-event Delta E", "kMC step", "Delta E, eV")
    axes[0].legend()
    axes[1].plot(history["step"], history["uncertainty"], color=COLORS["purple"])
    polish(axes[1], "Model uncertainty", "kMC step", "uncertainty, eV")
    save(fig, output_dir / "model_diagnostics.png")


def plot_seeded_publication_figures(run_dir: str | Path, output_dir: str | Path | None = None) -> None:
    """Create polished summary figures for a seeded crystallization directory."""
    apply_publication_style()
    run_dir = Path(run_dir)
    output_dir = run_dir / "publication_figures" if output_dir is None else Path(output_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    comparison = summary.get("state_comparison_after_repair_after_stage1_final")
    if not isinstance(comparison, dict):
        return

    states = ["after_repair", "after_stage1", "final"]
    labels = ["after repair", "stage 1", "final"]
    energy = [float(comparison[state]["energy_per_puo2_eV"]) for state in states]
    bulk = [float(comparison[state]["bulk_fluorite_order_score"]) for state in states]
    coord_error = [float(comparison[state]["mean_abs_coordination_error"]) for state in states]
    pu8 = [float(comparison[state]["fraction_pu_with_8_o"]) for state in states]
    o4 = [float(comparison[state]["fraction_o_with_4_pu"]) for state in states]
    deltas = comparison["deltas"]

    fig = plt.figure(figsize=(12.0, 6.6))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)
    x = np.arange(len(states))

    ax_energy = fig.add_subplot(gs[:, 0])
    ax_energy.plot(x, energy, marker="o", color=COLORS["green"], linewidth=2.8)
    ax_energy.fill_between(x, energy, max(energy), color=COLORS["green"], alpha=0.10)
    ax_energy.set_xticks(x)
    ax_energy.set_xticklabels(labels)
    polish(ax_energy, "Seeded energy relaxation", None, "Energy, eV/PuO2")

    ax_bulk = fig.add_subplot(gs[0, 1])
    ax_bulk.plot(x, bulk, marker="o", color=COLORS["blue"], linewidth=2.6)
    ax_bulk.set_xticks(x)
    ax_bulk.set_xticklabels(labels)
    polish(ax_bulk, "Bulk fluorite-like order", None, "score")

    ax_coord = fig.add_subplot(gs[0, 2])
    ax_coord.plot(x, coord_error, marker="o", color=COLORS["red"], linewidth=2.6)
    ax_coord.set_xticks(x)
    ax_coord.set_xticklabels(labels)
    polish(ax_coord, "Coordination error", None, "mean abs. error")

    ax_frac = fig.add_subplot(gs[1, 1])
    ax_frac.plot(x, pu8, marker="o", color=COLORS["purple"], linewidth=2.4, label="Pu with 8 O")
    ax_frac.plot(x, o4, marker="o", color=COLORS["orange"], linewidth=2.4, label="O with 4 Pu")
    ax_frac.set_xticks(x)
    ax_frac.set_xticklabels(labels)
    polish(ax_frac, "Ideal local coordinations", None, "fraction")
    ax_frac.legend()

    ax_text = fig.add_subplot(gs[1, 2])
    lines = [
        "Seeded crystallization workflow",
        f"Delta E after repair -> final: {deltas['final_minus_after_repair_energy_per_puo2_eV']:+.4f} eV/PuO2",
        f"Bulk order change: {deltas['final_minus_after_repair_bulk_order_score']:+.4f}",
        f"Coord. error change: {deltas['final_minus_after_repair_mean_abs_coordination_error']:+.4f}",
        f"Pu with 8 O change: {deltas['final_minus_after_repair_fraction_pu_with_8_o']:+.4f}",
        f"O with 4 Pu change: {deltas['final_minus_after_repair_fraction_o_with_4_pu']:+.4f}",
        "Interpretation: local seeded ordering, not complete recrystallization.",
    ]
    _summary_text(ax_text, lines)

    fig.suptitle("PuO2 seeded workflow: local crystallization indicators", fontsize=15.2, fontweight="bold", y=0.985)
    save(fig, output_dir / "seeded_crystallization_summary.png")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate publication-style PuO2 ML-kMC figures.")
    parser.add_argument("run_dir")
    parser.add_argument("--seeded", action="store_true")
    args = parser.parse_args()
    if args.seeded:
        plot_seeded_publication_figures(args.run_dir)
    else:
        plot_main_publication_figures(args.run_dir)


if __name__ == "__main__":
    main()
