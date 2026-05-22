import numpy as np

from src.events import Event
from src.features import get_feature_names, make_event_features
from src.potentials import PairPotentialPuO2


def test_feature_names_match_event_feature_vector():
    atoms = ["Pu", "O", "O", "Pu"]
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [2.0, 2.0, 0.0],
        ]
    )
    event = Event(
        atom_index=1,
        old_position=positions[1].copy(),
        new_position=np.array([2.2, 0.1, 0.0]),
        displacement=np.array([0.2, 0.1, 0.0]),
        kind="random",
    )

    names = get_feature_names()
    features = make_event_features(atoms, positions, event)

    assert len(names) == len(features)
    assert "nearest_any_before_1" in names
    assert "nearest_any_after_1" in names
    assert "coord_deviation_delta" in names
    assert np.all(np.isfinite(features))

    features_with_potential = make_event_features(atoms, positions, event, potential=PairPotentialPuO2())
    assert len(names) == len(features_with_potential)
    assert features_with_potential[names.index("local_energy_delta")] != 0.0
    assert features_with_potential[names.index("force_before_norm")] >= 0.0
