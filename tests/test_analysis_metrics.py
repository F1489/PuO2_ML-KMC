import numpy as np

from src.analysis import (
    bulk_fluorite_order_score,
    coordination_summary,
    defect_counts,
    rdf_peak_sharpness,
    soft_coordination_order_score,
    surface_mask,
)


def test_crystallization_metrics_are_finite_and_surface_aware():
    atoms = ["Pu", "O", "O", "Pu", "O", "O"]
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [2.0, 2.0, 0.0],
            [1.0, 1.0, 2.0],
            [6.0, 6.0, 6.0],
        ],
        dtype=float,
    )

    mask = surface_mask(positions, surface_shell_thickness=1.5)
    counts = defect_counts(atoms, positions)
    summary = coordination_summary(atoms, positions)

    assert mask.dtype == bool
    assert mask.shape == (len(atoms),)
    assert counts["total_coordination_defects"] >= counts["bulk_coordination_defects"]
    assert counts["surface_coordination_defects"] >= 0
    assert counts["mean_abs_coordination_error"] >= 0.0
    assert 0.0 <= soft_coordination_order_score(atoms, positions) <= 1.0
    assert 0.0 <= bulk_fluorite_order_score(atoms, positions) <= 1.0
    assert rdf_peak_sharpness(atoms, positions, pair=("Pu", "O")) >= 0.0
    assert "bulk_fluorite_order_score" in summary
    assert "rdf_pu_o_peak_sharpness" in summary
