"""Quantum ESPRESSO (pw.x) engine for isolated molecules.

QE is a periodic plane-wave code, so each molecule sits in a cubic box of
vacuum. Three things differ from a Gaussian-basis code like ORCA:

* Charged cells. A periodic cell with net charge needs a correction, or the
  energy depends on box size and E_ads is meaningless for charged species.
    - solvation: environ -> the Environ plugin's 'parabolic' 0D correction
    - solvation: none    -> assume_isolated = 'mt' (Martyna-Tuckerman); the box
                            must be >= 2x the molecule's size, so boxes get big.
* Solvation. Core QE has no implicit solvent. Implicit water needs pw.x
  compiled with Environ (https://environ.readthedocs.io) and run with
  `pw.x --environ`. Without it, a -24 polyanion or a +4 cation in vacuum is
  unphysical, so solvation: none is only useful for testing.
* Functional. Defaults to PBE-D3(BJ) on whatever pseudopotentials you supply
  (SSSP recommended). Hybrids are possible in QE but very expensive in large
  vacuum boxes, so there is no separate hybrid single point here.

Plane waves have no basis set superposition error, which is a plus for E_ads.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from rdkit import Chem

from .geometry import read_xyz, write_xyz

RY_TO_HARTREE = 0.5
_ETOT = re.compile(r"!\s+total energy\s+=\s+(-?\d+\.\d+)\s+Ry")
_PT = Chem.GetPeriodicTable()


# ------------------------------------------------------------------ box
def extent(coords) -> float:
    c = np.asarray(coords)
    if len(c) < 2:
        return 0.0
    return float(np.linalg.norm(c[:, None, :] - c[None, :, :], axis=-1).max())


def box_length(ext: float, qcfg: dict) -> float:
    vac = qcfg["vacuum"]
    L = ext + 2 * vac
    if qcfg["solvation"] == "none":   # Martyna-Tuckerman: box >= 2x density extent
        L = max(L, 2 * (ext + 4.0))
    return float(np.ceil(L))


# ------------------------------------------------------------------ pseudos
def find_pseudos(elements, qcfg) -> dict:
    explicit = qcfg.get("pseudopotentials") or {}
    pdir = Path(qcfg["pseudo_dir"]).expanduser()
    out = {}
    for el in sorted(set(elements)):
        if el in explicit:
            out[el] = explicit[el]
            continue
        cands = sorted(f.name for f in pdir.iterdir()
                       if f.is_file() and f.suffix.lower() == ".upf"
                       and re.match(rf"^{el}([._\-]|$)", f.name, re.I))
        if len(cands) != 1:
            raise FileNotFoundError(
                f"pseudopotential for {el}: found {cands or 'none'} in {pdir}; "
                f"set dft.qe.pseudopotentials.{el} in the config")
        out[el] = cands[0]
    return out


# ------------------------------------------------------------------ writing
def _pw_input(calc, symbols, coords, charge, mult, L, pseudos, qcfg, prefix):
    species = sorted(set(symbols))
    sysl = [f"  ibrav = 0", f"  nat = {len(symbols)}", f"  ntyp = {len(species)}",
            f"  ecutwfc = {qcfg['ecutwfc']}", f"  ecutrho = {qcfg['ecutrho']}",
            f"  tot_charge = {float(charge)}"]
    if mult > 1:
        sysl += ["  nspin = 2", f"  tot_magnetization = {mult - 1}"]
    if qcfg["solvation"] == "none":
        sysl.append("  assume_isolated = 'mt'")
    if qcfg.get("vdw_corr"):
        sysl.append(f"  vdw_corr = '{qcfg['vdw_corr']}'")
        if "d3" in qcfg["vdw_corr"].lower():
            sysl.append(f"  dftd3_version = {qcfg['dftd3_version']}")
            sysl.append(f"  dftd3_threebody = {'.true.' if qcfg['dftd3_threebody'] else '.false.'}")
    if qcfg.get("input_dft"):
        sysl.append(f"  input_dft = '{qcfg['input_dft']}'")

    ctrl = [f"  calculation = '{calc}'", f"  prefix = '{prefix}'", "  outdir = './tmp'",
            f"  pseudo_dir = '{Path(qcfg['pseudo_dir']).expanduser().resolve()}'",
            "  tprnfor = .true.", "  disk_io = 'low'"]
    if calc == "relax":
        ctrl += [f"  nstep = {qcfg['nstep']}", f"  forc_conv_thr = {qcfg['forc_conv_thr']:.1e}",
                 f"  etot_conv_thr = {qcfg['etot_conv_thr']:.1e}"]

    lines = ["&CONTROL", *ctrl, "/", "&SYSTEM", *sysl, "/",
             "&ELECTRONS", f"  conv_thr = {qcfg['conv_thr']:.1e}",
             f"  mixing_beta = {qcfg['mixing_beta']}", f"  electron_maxstep = {qcfg['electron_maxstep']}", "/"]
    if calc == "relax":
        lines += ["&IONS", "  ion_dynamics = 'bfgs'", "/"]
    lines.append("ATOMIC_SPECIES")
    for el in species:
        lines.append(f"  {el:<3s} {_PT.GetAtomicWeight(el):10.4f}  {pseudos[el]}")
    lines += ["CELL_PARAMETERS angstrom", f"  {L:.6f} 0.0 0.0", f"  0.0 {L:.6f} 0.0",
              f"  0.0 0.0 {L:.6f}", "ATOMIC_POSITIONS angstrom"]
    for s, (x, y, z) in zip(symbols, coords):
        lines.append(f"  {s:<3s} {x:14.8f} {y:14.8f} {z:14.8f}")
    lines += ["K_POINTS gamma", ""]
    return "\n".join(lines)


def _environ_input(qcfg):
    return (
        "&ENVIRON\n"
        "  verbose = 0\n"
        "  environ_thr = 1.d-1\n"
        f"  environ_type = '{qcfg['environ_type']}'\n"
        "/\n"
        "&BOUNDARY\n"
        "  solvent_mode = 'electronic'\n"
        "/\n"
        "&ELECTROSTATIC\n"
        "  pbc_correction = 'parabolic'\n"
        "  pbc_dim = 0\n"
        "  tol = 1.d-11\n"
        "  mix = 0.6\n"
        "/\n"
    )


_RUN_SH = """#!/bin/bash
# Quantum ESPRESSO geometry optimization. Safe to re-run: a finished job is
# skipped. Override the binary / MPI launcher with QE_PW / QE_LAUNCHER, e.g.
#   export QE_PW=/opt/qe-7.3/bin/pw.x  QE_LAUNCHER="srun"
set -euo pipefail
cd "$(dirname "$0")"
PW="${{QE_PW:-{pw}}}"
LAUNCH="${{QE_LAUNCHER:-{launcher}}}"
if ! grep -q "JOB DONE" relax.out 2>/dev/null; then
  $LAUNCH "$PW" {flags}-in relax.in > relax.out
fi
rm -rf tmp      # wavefunctions are large and not needed afterwards
"""


def write_job(jobdir: Path, symbols, coords, charge, mult, L, dcfg):
    qcfg = dcfg["qe"]
    jobdir.mkdir(parents=True, exist_ok=True)
    coords = np.asarray(coords)
    centered = coords - coords.mean(axis=0) + L / 2.0
    write_xyz(jobdir / "input.xyz", symbols, centered)
    calc = "relax" if len(symbols) > 1 else "scf"
    pseudos = find_pseudos(symbols, qcfg)
    (jobdir / "relax.in").write_text(
        _pw_input(calc, symbols, centered, charge, mult, L, pseudos, qcfg, "job"))
    flags = ""
    if qcfg["solvation"] == "environ":
        (jobdir / "environ.in").write_text(_environ_input(qcfg))
        flags = "--environ "
    launcher = qcfg["launcher"].format(nprocs=dcfg["nprocs"])
    (jobdir / "run.sh").write_text(_RUN_SH.format(pw=qcfg["pw_executable"], launcher=launcher,
                                                  flags=flags))
    (jobdir / "run.sh").chmod(0o755)


# ------------------------------------------------------------------ parsing
def _final_coords(text, n):
    if "Begin final coordinates" not in text:
        return None
    block = text.split("Begin final coordinates")[-1].split("End final coordinates")[0]
    lines = block.splitlines()
    for k, ln in enumerate(lines):
        if ln.strip().startswith("ATOMIC_POSITIONS"):
            if "angstrom" not in ln.lower():
                return None
            rows = [l.split() for l in lines[k + 1:k + 1 + n]]
            return [r[0] for r in rows], np.array([[float(v) for v in r[1:4]] for r in rows])
    return None


def parse(jd: Path, meta: dict) -> dict:
    """Returns dict(status, E_sp [Hartree], G_corr=None). Writes opt.xyz."""
    out = jd / "relax.out"
    if not out.exists():
        return {"status": "not_run"}
    text = out.read_text(errors="replace")
    if "JOB DONE" not in text:
        return {"status": "failed"}
    if "convergence NOT achieved" in text:
        return {"status": "scf_not_converged"}
    if meta["n_atoms"] > 1:
        if "End of BFGS Geometry Optimization" not in text or "bfgs converged" not in text:
            return {"status": "opt_not_converged"}
        fc = _final_coords(text, meta["n_atoms"])
        if fc is None:
            return {"status": "no_final_coordinates"}
        write_xyz(jd / "opt.xyz", fc[0], fc[1])
    else:
        s, x = read_xyz(jd / "input.xyz")
        write_xyz(jd / "opt.xyz", s, x)
    e = _ETOT.findall(text)
    if not e:
        return {"status": "no_energy"}
    return {"status": "ok", "E_sp": float(e[-1]) * RY_TO_HARTREE, "G_corr": None}
