import numpy as np
from rdkit import Chem

from tfa_eads import docking
from tfa_eads.geometry import (bonds, embed_conformers, mol_coords, mol_symbols, read_xyz,
                               rotation_about, rotation_aligning, write_xyz)


def test_xyz_roundtrip(tmp_path):
    syms, xyz = ["C", "O"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.13]])
    write_xyz(tmp_path / "m.xyz", syms, xyz)
    s2, x2 = read_xyz(tmp_path / "m.xyz")
    assert s2 == syms and np.allclose(x2, xyz)


def test_rotations():
    a, b = np.array([1.0, 0, 0]), np.array([0, 0.6, 0.8])
    assert np.allclose(rotation_aligning(a, b) @ a, b)
    assert np.allclose(rotation_aligning(a, -a) @ a, -a)       # antiparallel edge case
    R = rotation_about(np.array([0, 0, 1.0]), np.pi / 2)
    assert np.allclose(R @ [1, 0, 0], [0, 1, 0])


def test_embedding_and_bonds(cfg):
    mol, confs = embed_conformers("CCO", cfg["conformers"])
    assert len(confs) >= 1
    b = bonds(mol_symbols(mol), mol_coords(mol, confs[0][0]))
    assert len(b) == mol.GetNumBonds()       # distance bonds match the graph


def _species(smiles, cfg):
    mol, confs = embed_conformers(smiles, cfg["conformers"], n=5)
    return mol, mol_symbols(mol), mol_coords(mol, confs[0][0])


def test_tfa_anchor_is_carboxylate(cfg):
    gmol, _, gxyz = _species("[O-]C(=O)C(F)(F)F", cfg)
    p, axis = docking.guest_anchor(gmol, gxyz)
    o_idx = [a.GetIdx() for a in gmol.GetAtoms() if a.GetSymbol() == "O"]
    assert np.allclose(p, gxyz[o_idx].mean(axis=0))
    assert abs(np.linalg.norm(axis) - 1) < 1e-6


def test_monatomic_anchor(cfg):
    gmol, _, gxyz = _species("[Cl-]", cfg)
    p, axis = docking.guest_anchor(gmol, gxyz)
    assert axis is None


def test_poses_cover_site_kinds_without_clashes(cfg):
    hmol, hs, hx = _species("C[N+](C)(C)CCO", cfg)     # cation + OH donor + alkyl
    gmol, gs, gx = _species("[O-]C(=O)C(F)(F)F", cfg)
    poses = docking.generate_poses(hmol, hs, hx, gmol, gs, gx, cfg["docking"])
    assert 0 < len(poses) <= cfg["docking"]["max_poses"]
    kinds = {info["site_kind"] for _, info in poses}
    assert {"cation", "hbond", "generic"} <= kinds
    for gpos, _ in poses:
        assert not docking._clash(hs, hx, gs, gpos, cfg["docking"]["clash_scale"])
        assert np.allclose(np.linalg.norm(gpos[0] - gpos[1]), np.linalg.norm(gx[0] - gx[1]))  # rigid


def test_close_contact_filter():
    from tfa_eads.workflow import _too_close
    syms = ["S", "O", "C", "O"]
    far = np.array([[0, 0, 0], [1.45, 0, 0], [4.5, 0, 0], [5.7, 0, 0]], float)
    near = far.copy()
    near[2:] -= [2.0, 0, 0]                 # O...C at 2.45 A
    assert not _too_close(syms, far, 2, 0.78)
    assert _too_close(syms, near, 2, 0.78)
