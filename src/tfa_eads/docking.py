"""Generate starting geometries for host + anion complexes.

Strategy (deliberately simple and transparent):
  1. Find candidate binding sites on the host:
       cation  - atoms with positive formal charge (not nitro/N-oxide N+)
       hbond   - polar H atoms (H on N or O)
       generic - spread-out heavy atoms (farthest-point sampled), so even hosts
                 with no polar groups get poses
  2. For each site, sample directions on a sphere and keep the most open ones
     (H-bond sites: only directions roughly along the X-H bond).
  3. Place the anion so its charged end (e.g. the carboxylate O's of TFA)
     points at the site, spin it about the approach axis, and push it outward
     until it no longer clashes with the host.
Each pose is then relaxed with xTB; the lowest-energy valid pose is the
binding geometry. This is a pragmatic "many starts + local optimization"
search, not an exhaustive global one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem

from .geometry import VDW, fibonacci_sphere, rotation_about, rotation_aligning


@dataclass
class Site:
    kind: str           # cation | hbond | generic
    atom: int
    pos: np.ndarray
    parent: int | None = None   # heavy atom bearing the polar H


def guest_anchor(guest_mol: Chem.Mol, coords: np.ndarray):
    """Atoms carrying the anion's charge, e.g. both carboxylate O's of TFA.

    Returns (anchor_point, axis) where axis points from the anchor toward the
    body of the anion (None for monatomic ions).
    """
    neg = [a.GetIdx() for a in guest_mol.GetAtoms() if a.GetFormalCharge() < 0]
    if not neg:
        raise ValueError("guest has no negatively charged atom; expected an anion")
    anchor = set(neg)
    for i in neg:
        for nb in guest_mol.GetAtomWithIdx(i).GetNeighbors():
            for nb2 in nb.GetNeighbors():
                if nb2.GetSymbol() in ("O", "S") and nb2.GetDegree() == 1:
                    anchor.add(nb2.GetIdx())
    anchor = sorted(anchor)
    p = coords[anchor].mean(axis=0)
    if len(coords) == 1:
        return p, None
    axis = coords.mean(axis=0) - p
    if np.linalg.norm(axis) < 1e-3:   # charge spread symmetrically (e.g. nitrate)
        heavy = [a.GetIdx() for a in guest_mol.GetAtoms() if a.GetIdx() not in anchor]
        axis = coords[heavy].mean(axis=0) - p if heavy else np.array([0.0, 0.0, 1.0])
    return p, axis / np.linalg.norm(axis)


def _fps(points: np.ndarray, k: int, start: int = 0) -> list[int]:
    if len(points) == 0:
        return []
    chosen = [start]
    d = np.linalg.norm(points - points[start], axis=1)
    while len(chosen) < min(k, len(points)):
        nxt = int(np.argmax(d))
        chosen.append(nxt)
        d = np.minimum(d, np.linalg.norm(points - points[nxt], axis=1))
    return chosen


def find_sites(host_mol: Chem.Mol, coords: np.ndarray, n_generic: int) -> list[Site]:
    sites = []
    for a in host_mol.GetAtoms():
        i = a.GetIdx()
        if a.GetFormalCharge() > 0 and not any(nb.GetFormalCharge() < 0 for nb in a.GetNeighbors()):
            sites.append(Site("cation", i, coords[i]))
    for a in host_mol.GetAtoms():
        if a.GetSymbol() == "H":
            nb = a.GetNeighbors()[0]
            if nb.GetSymbol() in ("N", "O"):
                sites.append(Site("hbond", a.GetIdx(), coords[a.GetIdx()], parent=nb.GetIdx()))
    used = {s.atom for s in sites} | {s.parent for s in sites if s.parent is not None}
    heavy = [a.GetIdx() for a in host_mol.GetAtoms() if a.GetSymbol() != "H" and a.GetIdx() not in used]
    if heavy and n_generic > 0:
        centroid = coords.mean(axis=0)
        pts = coords[heavy]
        start = int(np.argmax(np.linalg.norm(pts - centroid, axis=1)))
        for j in _fps(pts, n_generic, start):
            sites.append(Site("generic", heavy[j], coords[heavy[j]]))
    return sites


def _open_directions(site: Site, host_coords, dist, n_pick, min_angle_deg=60.0,
                     cone_axis=None, cone_deg=None):
    dirs = fibonacci_sphere(96)
    if cone_axis is not None:
        cone_axis = cone_axis / np.linalg.norm(cone_axis)
        dirs = dirs[dirs @ cone_axis >= np.cos(np.radians(cone_deg))]
        dirs = np.vstack([cone_axis, dirs])
    if len(dirs) == 0:
        return []
    pts = site.pos + dist * dirs
    mask = np.ones(len(host_coords), bool)
    mask[site.atom] = False
    if site.parent is not None:
        mask[site.parent] = False
    others = host_coords[mask]
    if len(others) == 0:
        openness = np.full(len(dirs), dist)
    else:
        openness = np.linalg.norm(pts[:, None, :] - others[None, :, :], axis=-1).min(axis=1)
    picked = []
    for k in np.argsort(-openness):
        if all(np.degrees(np.arccos(np.clip(dirs[k] @ dirs[p], -1, 1))) >= min_angle_deg for p in picked):
            picked.append(k)
        if len(picked) == n_pick:
            break
    return [dirs[k] for k in picked]


def _clash(host_syms, host_xyz, guest_syms, guest_xyz, scale):
    rh = np.array([VDW.get(s, 1.8) for s in host_syms])
    rg = np.array([VDW.get(s, 1.8) for s in guest_syms])
    d = np.linalg.norm(host_xyz[:, None, :] - guest_xyz[None, :, :], axis=-1)
    return bool((d < scale * (rh[:, None] + rg[None, :])).any())


def generate_poses(host_mol, host_syms, host_xyz, guest_mol, guest_syms, guest_xyz, dcfg):
    """Return list of (pose_coords_guest, info_dict); host coordinates are unchanged."""
    rng = np.random.default_rng(dcfg["random_seed"])
    anchor, axis = guest_anchor(guest_mol, guest_xyz)
    sites = find_sites(host_mol, host_xyz, dcfg["n_generic_sites"])

    per_site = dcfg["directions_per_site"] * dcfg["spins_per_direction"]
    if axis is None:
        per_site = dcfg["directions_per_site"]   # spinning a sphere is pointless
    max_sites = max(1, dcfg["max_poses"] // per_site)

    # interleave kinds (cation, hbond, generic, cation, ...) so every kind of
    # site is sampled; within each kind, sites are spread out spatially
    queues = []
    for kind in ("cation", "hbond", "generic"):
        ks = [s for s in sites if s.kind == kind]
        if ks:
            queues.append([ks[j] for j in _fps(np.array([s.pos for s in ks]), len(ks))])
    ordered = []
    while queues and len(ordered) < max_sites:
        for q in list(queues):
            ordered.append(q.pop(0))
            if not q:
                queues.remove(q)
            if len(ordered) == max_sites:
                break

    poses = []
    for s in ordered:
        if s.kind == "cation":
            dist, cone_axis, cone = dcfg["dist_cation"], None, None
        elif s.kind == "hbond":
            dist, cone = dcfg["dist_hbond"], 50.0
            cone_axis = s.pos - host_xyz[s.parent]
        else:
            dist, cone_axis, cone = dcfg["dist_generic"], None, None
        for u in _open_directions(s, host_xyz, dist, dcfg["directions_per_site"],
                                  cone_axis=cone_axis, cone_deg=cone):
            n_spin = 1 if axis is None else dcfg["spins_per_direction"]
            theta0 = rng.uniform(0, 2 * np.pi)
            for k in range(n_spin):
                if axis is None:
                    base = guest_xyz - anchor
                else:
                    R = rotation_about(u, theta0 + 2 * np.pi * k / n_spin) @ rotation_aligning(axis, u)
                    base = (guest_xyz - anchor) @ R.T
                placed = None
                for push in np.arange(0.0, 2.51, 0.25):
                    trial = base + s.pos + (dist + push) * u
                    if not _clash(host_syms, host_xyz, guest_syms, trial, dcfg["clash_scale"]):
                        placed = trial
                        break
                if placed is not None:
                    poses.append((placed, {"site_kind": s.kind, "site_atom": s.atom,
                                           "site_element": host_syms[s.atom]}))
    return poses[: dcfg["max_poses"]]
