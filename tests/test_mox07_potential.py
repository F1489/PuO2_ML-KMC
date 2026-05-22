import numpy as np

from src.potentials import PairPotentialPuO2


def test_mox07_default_parameters_are_used():
    potential = PairPotentialPuO2()
    assert potential.k_coul == 14.399645478
    assert potential.charges == {"Pu": 2.745, "O": -1.3725}
    assert potential.params[("O", "O")] == {"A": 50212.0, "B": 5.5200, "C": 74.796}
    assert potential.params[("O", "Pu")] == {"A": 871.79, "B": 2.8079, "C": 0.0}
    assert potential.params[("Pu", "Pu")] == {"A": 0.0, "B": 0.0, "C": 0.0}


def test_mox07_pair_formula_matches_manual_value():
    potential = PairPotentialPuO2(cutoff=10.0)
    r = 2.0
    manual = (
        potential.k_coul * potential.charges["Pu"] * potential.charges["O"] / r
        + 871.79 * np.exp(-2.8079 * r)
    )
    assert np.isclose(potential.pair_energy("Pu", "O", r), manual)
