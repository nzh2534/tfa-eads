"""Geometry helpers: RDKit conformer generation, XYZ I/O, bond perception, rotations."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

HARTREE_TO_KCAL = 627.509474

# Covalent radii (Cordero 2008), Angstrom
COV = {"H": 0.31, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "Si": 1.11,
       "P": 1.07, "S": 1.05, "Cl": 1.02, "Br": 1.20, "I": 1.39, "Se": 1.20, "Na": 1.66,
       "K": 2.03, "Li": 1.28, "Mg": 1.41, "Ca": 1.76, "Zn": 1.22, "Cu": 1.32, "Fe": 1.32}
# Bondi van der Waals radii, Angstrom
VDW = {"H": 1.20, "B": 1.92, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47, "Si": 2.10,
       "P": 1.80, "S": 1.80, "Cl": 1.75, "Br": 1.85, "I": 1.98, "Se": 1.90}


# ---------------------------------------------------------------- XYZ I/O
def write_xyz(path, symbols, coords, comment=""):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fh.write(f"{len(symbols)}\n{comment}\n")
        for s, (x, y, z) in zip(symbols, coords):
            fh.write(f"{s:<3s} {x:14.8f} {y:14.8f} {z:14.8f}\n")


def read_xyz(path):
    lines = Path(path).read_text().splitlines()
    n = int(lines[0].split()[0])
    syms, xyz = [], []
    for ln in lines[2:2 + n]:
        p = ln.split()
        syms.append(p[0].capitalize())
        xyz.append([float(v) for v in p[1:4]])
    return syms, np.array(xyz)


def mol_symbols(mol):
    return [a.GetSymbol() for a in mol.GetAtoms()]


def mol_coords(mol, conf_id=-1):
    return np.array(mol.GetConformer(conf_id).GetPositions())


# ---------------------------------------------------------------- conformers
def n_conformers_for(mol, ccfg) -> int:
    from rdkit.Chem import rdMolDescriptors
    if mol.GetNumHeavyAtoms() > ccfg["large_heavy_atoms"]:
        return ccfg["rdkit_n_large"]
    nrot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    n = ccfg["rdkit_n_min"] + ccfg["rdkit_per_rotor"] * nrot
    return int(min(max(n, ccfg["rdkit_n_min"]), ccfg["rdkit_n_max"]))


def embed_conformers(smiles: str, ccfg: dict, n: int | None = None):
    """ETKDG embedding + MMFF (fallback UFF) relaxation.

    Returns (mol_with_H, [(conf_id, ff_energy_kcal), ...]) sorted by energy,
    near-duplicates (|dE| < 0.05 kcal/mol) removed.
    """
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if n is None:
        n = n_conformers_for(mol, ccfg)
    params = AllChem.ETKDGv3()
    params.randomSeed = ccfg["random_seed"]
    params.pruneRmsThresh = ccfg["embed_rms_prune"]
    params.numThreads = 0
    cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n, params=params))
    if not cids:  # macrocycles / strained systems
        params.useRandomCoords = True
        params.maxIterations = 2000
        cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=max(5, n // 4), params=params))
    if not cids:
        raise RuntimeError("RDKit could not embed any conformer")

    energies = None
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            res = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0, maxIters=2000)
            energies = [e for _, e in res]
        elif AllChem.UFFHasAllMoleculeParams(mol):
            res = AllChem.UFFOptimizeMoleculeConfs(mol, numThreads=0, maxIters=2000)
            energies = [e for _, e in res]
    except Exception:  # noqa: BLE001
        energies = None
    if energies is None:
        energies = [0.0] * len(cids)

    order = sorted(zip(cids, energies), key=lambda t: t[1])
    kept = []
    for cid, e in order:
        if not kept or all(abs(e - ek) > 0.05 for _, ek in kept) or e == 0.0:
            kept.append((cid, e))
    return mol, kept


# ---------------------------------------------------------------- connectivity
def bonds(symbols, coords, scale=1.15):
    """Distance-based bond set {(i, j)} with i < j."""
    coords = np.asarray(coords)
    r = np.array([COV.get(s, 1.5) for s in symbols])
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    cut = scale * (r[:, None] + r[None, :])
    i, j = np.where(np.triu(d < cut, k=1))
    return set(zip(i.tolist(), j.tolist()))


def connectivity_changed(symbols, before, after) -> bool:
    return bonds(symbols, before) != bonds(symbols, after)


def min_intermolecular_distance(coords_a, coords_b) -> float:
    d = np.linalg.norm(np.asarray(coords_a)[:, None, :] - np.asarray(coords_b)[None, :, :], axis=-1)
    return float(d.min())


# ---------------------------------------------------------------- rotations
def rotation_aligning(a, b):
    """Rotation matrix taking unit vector a onto unit vector b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-8:
        if c > 0:
            return np.eye(3)
        perp = np.cross(a, [1, 0, 0])
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(a, [0, 1, 0])
        return rotation_about(perp / np.linalg.norm(perp), np.pi)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def rotation_about(axis, theta):
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(theta) * k + (1 - np.cos(theta)) * (k @ k)


def fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)
