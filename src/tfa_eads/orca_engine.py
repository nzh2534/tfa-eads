"""ORCA engine: Gaussian-basis DFT with implicit solvation (CPCM/SMD).

opt.inp : geometry optimization (default r2SCAN-3c, CPCM water), optional Freq
sp.inp  : single point on the optimized geometry (default wB97M-V/def2-TZVPD, SMD water)
"""
from __future__ import annotations

import math
import re
from pathlib import Path

from .geometry import read_xyz, write_xyz

_FINAL_E = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")
_GCORR = re.compile(r"G-E\(el\)\s+\.\.\.\s+(-?\d+\.\d+)\s+Eh")


# ------------------------------------------------------------------ writing
def _opt_input(dcfg, charge, mult, n_atoms):
    kw = dcfg["opt_keywords"]
    if n_atoms > 1:
        kw += " Opt"
        if dcfg["freq"]:
            kw += " Freq"
    return (f"! {kw}\n%pal nprocs {dcfg['nprocs']} end\n%maxcore {dcfg['maxcore_mb']}\n"
            f"* xyzfile {charge} {mult} input.xyz\n")


def _sp_input(dcfg, charge, mult):
    s = f"! {dcfg['sp_keywords']}\n%pal nprocs {dcfg['nprocs']} end\n%maxcore {dcfg['maxcore_mb']}\n"
    if dcfg["sp_smd"]:
        s += '%cpcm\n  smd true\n  SMDsolvent "water"\nend\n'
    return s + f"* xyzfile {charge} {mult} opt.xyz\n"


_RUN_SH = """#!/bin/bash
# Runs the DFT optimization then the single point. Safe to re-run: finished
# steps are skipped. Parallel ORCA needs its FULL path, e.g.
#   export ORCA_EXE=/opt/orca_6/orca
set -euo pipefail
cd "$(dirname "$0")"
ORCA="${ORCA_EXE:-orca}"
if ! grep -q "ORCA TERMINATED NORMALLY" opt.out 2>/dev/null; then
  "$ORCA" opt.inp > opt.out
fi
[ -f opt.xyz ] || cp input.xyz opt.xyz      # single atoms: nothing to optimize
if ! grep -q "ORCA TERMINATED NORMALLY" sp.out 2>/dev/null; then
  "$ORCA" sp.inp > sp.out
fi
"""


def write_job(jobdir: Path, symbols, coords, charge, mult, L, dcfg):
    jobdir.mkdir(parents=True, exist_ok=True)
    write_xyz(jobdir / "input.xyz", symbols, coords)
    (jobdir / "opt.inp").write_text(_opt_input(dcfg, charge, mult, len(symbols)))
    (jobdir / "sp.inp").write_text(_sp_input(dcfg, charge, mult))
    (jobdir / "run.sh").write_text(_RUN_SH)
    (jobdir / "run.sh").chmod(0o755)


# ------------------------------------------------------------------ parsing
def _translational_gcorr(symbol, temperature):
    """G - E(el) for a closed-shell monatomic ion (ideal gas, 1 atm), Hartree.

    ORCA's Freq is skipped for single atoms, so the translational free energy
    is computed analytically (Sackur-Tetrode) for consistency with ORCA's
    ideal-gas, 1 atm standard state.
    """
    masses = {"F": 18.998, "Cl": 35.453, "Br": 79.904, "I": 126.904}
    kB, h, NA, R = 1.380649e-23, 6.62607015e-34, 6.02214076e23, 8.314462618
    m = masses[symbol] * 1e-3 / NA
    T, P = temperature, 101325.0
    q = (2 * math.pi * m * kB * T / h ** 2) ** 1.5 * kB * T / P
    S = R * (math.log(q) + 2.5)                      # J/mol/K
    H = 2.5 * R * T                                  # J/mol
    return (H - T * S) / 4.3597447222071e-18 / NA    # Hartree


def parse(jd: Path, meta: dict, temperature: float) -> dict:
    rec = {"status": "ok", "E_sp": None, "G_corr": None}
    opt_out, sp_out = jd / "opt.out", jd / "sp.out"
    if not opt_out.exists() or not sp_out.exists():
        rec["status"] = "not_run"
        return rec
    ot, st = opt_out.read_text(errors="replace"), sp_out.read_text(errors="replace")
    if "ORCA TERMINATED NORMALLY" not in ot or "ORCA TERMINATED NORMALLY" not in st:
        rec["status"] = "failed"
        return rec
    if meta["n_atoms"] > 1 and "THE OPTIMIZATION HAS CONVERGED" not in ot:
        rec["status"] = "opt_not_converged"
        return rec
    m = _FINAL_E.findall(st)
    if not m:
        rec["status"] = "no_energy"
        return rec
    rec["E_sp"] = float(m[-1])
    g = _GCORR.findall(ot)
    if g:
        rec["G_corr"] = float(g[-1])
        if "***imaginary mode***" in ot:
            rec["status"] = "ok_imaginary_freq"
    elif meta["n_atoms"] == 1:
        sym = read_xyz(jd / "input.xyz")[0][0]
        rec["G_corr"] = _translational_gcorr(sym, temperature)
    return rec


