"""XYZ input/output helpers for PuO2 clusters.

Coordinates are stored in angstrom (A). Periodic boundary conditions are not used.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SUPPORTED_ATOMS = {"Pu", "O"}


def read_xyz(path: str | Path) -> tuple[list[str], np.ndarray, str]:
    """Read an XYZ file and return atom symbols, positions in A, and comment."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError(f"XYZ file must contain at least 2 lines: {path}")
    try:
        n_atoms = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError(f"First XYZ line must be an integer atom count: {path}") from exc
    comment = lines[1]
    atom_lines = lines[2:]
    if len(atom_lines) != n_atoms:
        raise ValueError(
            f"XYZ atom count mismatch in {path}: header says {n_atoms}, "
            f"found {len(atom_lines)} coordinate lines"
        )

    atoms: list[str] = []
    positions = np.empty((n_atoms, 3), dtype=float)
    for i, line in enumerate(atom_lines):
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Malformed XYZ atom line {i + 3} in {path}: {line!r}")
        atom = parts[0]
        if atom not in SUPPORTED_ATOMS:
            raise ValueError(f"Unsupported atom type {atom!r}; supported: {sorted(SUPPORTED_ATOMS)}")
        atoms.append(atom)
        try:
            positions[i] = [float(parts[1]), float(parts[2]), float(parts[3])]
        except ValueError as exc:
            raise ValueError(f"Invalid coordinate at line {i + 3} in {path}") from exc
    return atoms, positions, comment


def write_xyz(path: str | Path, atoms: list[str], positions: np.ndarray, comment: str = "") -> None:
    """Write atom symbols and positions in A to an XYZ file."""
    path = Path(path)
    positions = np.asarray(positions, dtype=float)
    if positions.shape != (len(atoms), 3):
        raise ValueError(f"positions must have shape ({len(atoms)}, 3), got {positions.shape}")
    unsupported = sorted(set(atoms) - SUPPORTED_ATOMS)
    if unsupported:
        raise ValueError(f"Unsupported atom types: {unsupported}")

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(len(atoms)), comment]
    lines.extend(
        f"{atom} {xyz[0]:.8f} {xyz[1]:.8f} {xyz[2]:.8f}"
        for atom, xyz in zip(atoms, positions, strict=True)
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
