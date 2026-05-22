import pandas as pd

from src.validation import analyze_validation_errors


def test_analyze_validation_errors_writes_expected_reports(tmp_path):
    df = pd.DataFrame(
        {
            "exact_delta_E": [-3.0, 2.0, -1.5, 4.0],
            "predicted_delta_E": [-2.5, -1.0, 1.0, 3.0],
            "exact_is_energy_lowering": [1, 0, 1, 0],
            "predicted_is_energy_lowering": [1, 1, 0, 0],
            "good_event_probability": [0.8, 0.7, 0.3, 0.2],
            "uncertainty": [0.5, 2.0, 1.5, 0.8],
            "atom_type": ["O", "Pu", "O", "Pu"],
            "event_kind": ["surface", "random", "surface", "coordination"],
        }
    )

    report = analyze_validation_errors(df, tmp_path)

    assert report["overall"]["n_events"] == 4
    assert report["overall"]["sign_errors"] == 2
    assert (tmp_path / "validation_error_summary.json").exists()
    assert (tmp_path / "validation_error_by_atom_type.csv").exists()
    assert (tmp_path / "validation_error_by_event_kind.csv").exists()
    assert (tmp_path / "validation_worst_errors.csv").exists()
    assert (tmp_path / "validation_sign_errors.csv").exists()
