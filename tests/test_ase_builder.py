import os
from shalom.tools.ase_builder import ASEBuilder


def test_construct_bulk():
    atoms = ASEBuilder.construct_bulk("Fe", "bcc", a=2.87)
    assert len(atoms) == 1
    assert atoms.get_chemical_symbols() == ["Fe"]


def test_construct_surface():
    bulk_atoms = ASEBuilder.construct_bulk("Pt", "fcc", a=3.92)
    slab = ASEBuilder.construct_surface(bulk_atoms, (1, 1, 1), 3, vacuum=10.0)

    # fcc (111) 1x1 3-layer -> 3 atoms
    assert len(slab) == 3
    # Check vacuum
    cell_z = slab.get_cell()[2][2]
    # rough check that cell is larger than the atoms z-span + vacuum
    z_positions = slab.positions[:, 2]
    slab_thickness = z_positions.max() - z_positions.min()
    assert cell_z >= slab_thickness + 10.0


def test_save_poscar(tmp_path):
    atoms = ASEBuilder.construct_bulk("Al", "fcc", a=4.05)
    filepath = ASEBuilder.save_poscar(atoms, filename="POSCAR_test", directory=str(tmp_path))

    assert os.path.exists(filepath)
    with open(filepath, "r") as f:
        content = f.read()
        assert "Al" in content


def test_analyze_structure():
    atoms = ASEBuilder.construct_bulk("Cu", "fcc", a=3.6)
    analysis = ASEBuilder.analyze_structure(atoms)

    assert analysis["num_atoms"] == 1
    assert "cell_volume" in analysis
    assert "minimum_distance" in analysis
