import numpy as np

from src.events import event_min_distance, generate_candidate_events


def test_events_have_valid_indices_and_displacements():
    atoms = ["Pu", "O", "O", "Pu"]
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [2.0, 2.0, 0.0],
        ]
    )
    events = generate_candidate_events(atoms, positions, n_events=12, max_displacement=0.25, rng=np.random.default_rng(1))
    assert len(events) == 12
    for event in events:
        assert 0 <= event.atom_index < len(atoms)
        np.testing.assert_allclose(event.displacement, event.new_position - event.old_position)
        assert event_min_distance(positions, event) >= 1.2
        trial = positions.copy()
        trial[event.atom_index] = event.new_position
        assert len(trial) == len(positions)
