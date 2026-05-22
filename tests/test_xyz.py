import numpy as np

from src.io_xyz import read_xyz, write_xyz


def test_xyz_roundtrip_preserves_atoms_and_positions(tmp_path):
    atoms = ["Pu", "O", "O"]
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    path = tmp_path / "test.xyz"
    write_xyz(path, atoms, positions, comment="roundtrip")
    read_atoms, read_positions, comment = read_xyz(path)
    assert read_atoms == atoms
    assert comment == "roundtrip"
    assert read_positions.shape == (3, 3)
    np.testing.assert_allclose(read_positions, positions)
