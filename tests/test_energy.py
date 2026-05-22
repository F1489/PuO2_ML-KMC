import numpy as np

from src.potentials import PairPotentialPuO2


def test_energy_is_finite_and_permutation_symmetric():
    atoms = ["Pu", "O", "O", "Pu"]
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [2.0, 2.0, 0.0],
        ]
    )
    potential = PairPotentialPuO2()
    energy = potential.total_energy(atoms, positions)
    order = [2, 0, 3, 1]
    shuffled_energy = potential.total_energy([atoms[i] for i in order], positions[order])
    assert np.isfinite(energy)
    assert np.isclose(energy, shuffled_energy)


def test_delta_energy_single_move_matches_total_energy_difference():
    atoms = ["Pu", "O", "O", "Pu"]
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [2.0, 2.0, 0.0],
        ]
    )
    potential = PairPotentialPuO2()
    new_position = np.array([2.2, 0.1, 0.0])
    delta = potential.delta_energy_single_move(atoms, positions, 1, new_position)
    trial = positions.copy()
    trial[1] = new_position
    expected = potential.total_energy(atoms, trial) - potential.total_energy(atoms, positions)
    assert np.isclose(delta, expected)
