"""Configuration: defaults merged with an optional YAML file."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULTS = {
    "input_csv": "seed_set_raw.csv",
    "id_column": "cid",
    "smiles_column": "smiles",
    "charge_column": "charge",       # optional; only used as a consistency check
    "override_column": "smiles_override",  # optional; if filled, used verbatim
    "workdir": "work",

    # rules | dimorphite | as_given
    "protonation": "rules",
    "ph": 7.0,

    "guest": {"name": "TFA", "smiles": "[O-]C(=O)C(F)(F)F"},
    "competitors": [
        {"name": "chloride", "smiles": "[Cl-]"},
    ],

    "conformers": {
        "rdkit_n_min": 20,
        "rdkit_n_max": 300,
        "rdkit_per_rotor": 10,
        "large_heavy_atoms": 80,    # above this, use rdkit_n_large conformers
        "rdkit_n_large": 20,
        "embed_rms_prune": 0.5,
        "n_to_xtb": 10,             # lowest force-field conformers sent to xTB
        "n_keep": 3,                # lowest distinct xTB conformers kept
        "random_seed": 42,
    },

    "docking": {
        "host_conformers": 1,        # dock into this many lowest host conformers
        "max_poses": 24,             # per host conformer, per anion
        "directions_per_site": 2,
        "spins_per_direction": 2,
        "n_generic_sites": 4,        # extra surface sites so every host gets poses
        "n_keep": 3,                 # lowest valid poses kept (sent to DFT)
        "dist_cation": 3.4,          # Angstrom, cationic atom -> anion anchor
        "dist_hbond": 2.2,           # Angstrom, polar H -> anion anchor
        "dist_generic": 3.6,
        "clash_scale": 0.65,         # fraction of summed vdW radii allowed
        "dissociated_cutoff": 4.0,   # min host-guest distance flagged as unbound
        "close_contact_scale": 0.78, # heavy-atom contact below this x vdW sum -> excluded
        "random_seed": 7,
    },

    "xtb": {
        "executable": "xtb",
        "gfn": 2,
        "solvent": "water",          # ALPB implicit solvent; null for gas phase
        "opt_level": "tight",
        "threads_per_job": 1,
        "parallel_jobs": 4,
        "timeout_s": 7200,
    },

    "dft": {
        "engine": "orca",            # orca | qe
        "nprocs": 8,
        "maxcore_mb": 3000,
        "opt_keywords": "r2SCAN-3c CPCM(water) TightSCF",
        "sp_keywords": "wB97M-V def2-TZVPD TightSCF CPCM(water)",
        "sp_smd": True,              # switch CPCM to SMD (water) for the single point
        "freq": False,               # add Freq to the opt step -> free energies
        "temperature": 298.15,
        "n_host_conformers": 3,
        "n_poses": 3,
        "qe": {
            "pw_executable": "pw.x",
            "launcher": "mpirun -np {nprocs}",   # e.g. "srun" on SLURM clusters
            "pseudo_dir": "~/pseudo/SSSP",
            "pseudopotentials": {},      # {element: filename}; else auto-detected
            "ecutwfc": 50,               # Ry; use the max SSSP value over your elements
            "ecutrho": 400,              # Ry; 8x ecutwfc for US/PAW pseudopotentials
            "input_dft": None,           # None = functional from the pseudopotentials (PBE)
            "vdw_corr": "dft-d3",
            "dftd3_version": 4,          # 4 = D3 with Becke-Johnson damping
            "dftd3_threebody": False,    # QE 7 defaults to True; False = standard D3(BJ)
            "solvation": "environ",      # environ (needs pw.x built with Environ) | none
            "environ_type": "water",     # same SCCS parameter set for every species
            "vacuum": 8.0,               # Angstrom of vacuum on each side of the molecule
            "conv_thr": 1e-8,            # Ry
            "forc_conv_thr": 1e-3,       # Ry/Bohr
            "etot_conv_thr": 1e-5,       # Ry
            "mixing_beta": 0.3,
            "electron_maxstep": 200,
            "nstep": 300,
        },
        "slurm": {"time": "24:00:00", "partition": None, "account": None,
                  "max_concurrent": 50},   # array throttle (%N); None = unlimited
    },

    "limits": {
        "warn_heavy_atoms": 100,     # flag as expensive for DFT
        "skip_heavy_atoms": None,    # skip entirely above this (None = never)
    },
}


def _merge(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (new or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path) as fh:
            cfg = _merge(cfg, yaml.safe_load(fh) or {})
    cfg["workdir"] = str(Path(cfg["workdir"]).resolve())
    return cfg
