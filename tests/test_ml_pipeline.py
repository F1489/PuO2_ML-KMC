import numpy as np
import pandas as pd

from src.dataset import generate_mixed_dataset
from src.features import get_feature_names
from src.ml_model import (
    estimate_uncertainty_rf,
    load_classifier_threshold,
    load_models,
    predict_delta_E,
    predict_good_event_probability,
    train_models,
)
from src.potentials import PairPotentialPuO2


def test_train_load_predict_and_uncertainty_interfaces(tmp_path):
    rng = np.random.default_rng(10)
    feature_names = get_feature_names()
    X = rng.normal(size=(40, len(feature_names)))
    delta_e = 0.4 * X[:, 0] - 0.2 * X[:, 1] + rng.normal(scale=0.05, size=40)
    df = pd.DataFrame(X, columns=feature_names)
    df["delta_E"] = delta_e
    df["is_energy_lowering"] = (delta_e < 0.0).astype(int)
    df["is_crystallizing_event"] = ((delta_e < 0.2) & (X[:, 2] > 0.0)).astype(int)
    df["atom_type"] = np.where(X[:, 0] > 0.0, "Pu", "O")
    df["event_kind"] = "random_displacement"
    dataset_csv = tmp_path / "dataset.csv"
    model_dir = tmp_path / "models"
    df.to_csv(dataset_csv, index=False)

    metrics = train_models(dataset_csv, model_dir, seed=7, use_external_models=False, ensemble_size=1)
    regressor, classifier, loaded_features = load_models(model_dir)

    assert metrics["best_regressor"]
    assert loaded_features == feature_names
    predictions = predict_delta_E(regressor, X[:3])
    probabilities = predict_good_event_probability(classifier, X[:3])
    uncertainty = estimate_uncertainty_rf(regressor, X[:3])
    assert predictions.shape == (3,)
    assert probabilities.shape == (3,)
    assert uncertainty.shape == (3,)
    assert np.all(np.isfinite(predictions))
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert 0.2 <= load_classifier_threshold(model_dir) <= 0.8
    assert (model_dir / "model_selection_report.json").exists()
    assert (model_dir / "classifier_threshold.json").exists()


def test_generate_mixed_dataset_preserves_runtime_feature_columns(tmp_path):
    atoms = ["Pu", "O", "O", "Pu", "O", "O"]
    positions_a = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [3.0, 3.0, 0.0],
            [5.0, 3.0, 0.0],
            [3.0, 5.0, 0.0],
        ]
    )
    positions_b = positions_a + np.array([0.1, -0.1, 0.05])
    out_csv = tmp_path / "mixed.csv"

    df = generate_mixed_dataset(
        [("state_a", atoms, positions_a), ("state_b", atoms, positions_b)],
        PairPotentialPuO2(),
        n_events=8,
        output_csv=out_csv,
        seed=3,
        include_order_biased=False,
    )

    assert len(df) == 8
    assert out_csv.exists()
    assert set(get_feature_names()).issubset(df.columns)
    assert "source_state" in df.columns
    assert "order_biased_source" in df.columns
    assert set(df["source_state"]) == {"state_a", "state_b"}
