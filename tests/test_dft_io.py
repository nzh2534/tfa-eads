"""DFT input writing, output parsing and E_ads bookkeeping, with synthetic outputs
(no ORCA or Quantum ESPRESSO needed)."""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from tfa_eads import dft, orca_engine, qe_engine
from tfa_eads.geometry import HARTREE_TO_KCAL, read_xyz, write_xyz

XYZ = (["C", "O"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.13]]))


def test_orca_inputs(cfg, tmp_path):
    orca_engine.write_job(tmp_path, *XYZ, -1, 1, None, cfg["dft"])
    opt = (tmp_path / "opt.inp").read_text()
    sp = (tmp_path / "sp.inp").read_text()
    assert "r2SCAN-3c" in opt and " Opt" in opt and "xyzfile -1 1 input.xyz" in opt
    assert "smd true" in sp and "xyzfile -1 1 opt.xyz" in sp


def test_orca_single_atom_has_no_opt(cfg, tmp_path):
    orca_engine.write_job(tmp_path, ["Cl"], np.zeros((1, 3)), -1, 1, None, cfg["dft"])
    assert " Opt" not in (tmp_path / "opt.inp").read_text()


def test_orca_parse(cfg, tmp_path):
    (tmp_path / "opt.out").write_text(
        "*** THE OPTIMIZATION HAS CONVERGED ***\n"
        "G-E(el)                           ...      0.01234567 Eh      7.75 kcal/mol\n"
        "****ORCA TERMINATED NORMALLY****\n")
    (tmp_path / "sp.out").write_text("FINAL SINGLE POINT ENERGY      -113.123456789\n"
                                     "****ORCA TERMINATED NORMALLY****\n")
    r = orca_engine.parse(tmp_path, {"n_atoms": 2}, 298.15)
    assert r["status"] == "ok" and r["E_sp"] == pytest.approx(-113.123456789)
    assert r["G_corr"] == pytest.approx(0.01234567)


def test_translational_free_energy_chloride():
    # Sackur-Tetrode, 298.15 K, 1 atm: ~ -9.44 kcal/mol for Cl-
    g = orca_engine._translational_gcorr("Cl", 298.15) * HARTREE_TO_KCAL
    assert g == pytest.approx(-9.44, abs=0.05)


def _qe_cfg(cfg, tmp_path, solvation="environ"):
    pdir = tmp_path / "pseudo"
    pdir.mkdir(exist_ok=True)
    for el in ("C", "O", "Cl"):
        (pdir / f"{el}.pbe-test.UPF").write_text("dummy")
    d = cfg["dft"]
    d["engine"] = "qe"
    d["qe"]["pseudo_dir"] = str(pdir)
    d["qe"]["solvation"] = solvation
    return d


def test_qe_inputs_environ(cfg, tmp_path):
    d = _qe_cfg(cfg, tmp_path)
    L = qe_engine.box_length(qe_engine.extent(XYZ[1]), d["qe"])
    jd = tmp_path / "job"
    qe_engine.write_job(jd, *XYZ, -1, 1, L, d)
    txt = (jd / "relax.in").read_text()
    assert "calculation = 'relax'" in txt and "tot_charge = -1.0" in txt
    assert "assume_isolated" not in txt and "dftd3_threebody = .false." in txt
    assert "pbc_correction = 'parabolic'" in (jd / "environ.in").read_text()
    assert "--environ" in (jd / "run.sh").read_text()
    s, x = read_xyz(jd / "input.xyz")
    assert np.allclose(x.mean(axis=0), L / 2)                  # centered in the box


def test_qe_vacuum_uses_martyna_tuckerman_and_bigger_box(cfg, tmp_path):
    d = _qe_cfg(cfg, tmp_path, solvation="none")
    ext = 10.0
    assert qe_engine.box_length(ext, d["qe"]) >= 2 * ext
    qe_engine.write_job(tmp_path / "j", *XYZ, 0, 1, 30.0, d)
    assert "assume_isolated = 'mt'" in (tmp_path / "j" / "relax.in").read_text()
    assert not (tmp_path / "j" / "environ.in").exists()


def test_qe_missing_pseudo_is_clear_error(cfg, tmp_path):
    d = _qe_cfg(cfg, tmp_path)
    with pytest.raises(FileNotFoundError, match="F"):
        qe_engine.write_job(tmp_path / "j", ["F"], np.zeros((1, 3)), -1, 1, 20.0, d)


def test_qe_parse_relax(tmp_path):
    (tmp_path / "relax.out").write_text(
        "!    total energy              =     -59.40000000 Ry\n"
        "!    total energy              =     -59.42457920 Ry\n"
        "     bfgs converged in   5 scf cycles and   3 bfgs steps\n"
        "     End of BFGS Geometry Optimization\n"
        "Begin final coordinates\n\nATOMIC_POSITIONS (angstrom)\n"
        "C   6.0 6.0 5.39\nO   6.0 6.0 6.60\nEnd final coordinates\n   JOB DONE.\n")
    write_xyz(tmp_path / "input.xyz", *XYZ)
    r = qe_engine.parse(tmp_path, {"n_atoms": 2})
    assert r["status"] == "ok" and r["E_sp"] == pytest.approx(-59.4245792 / 2)
    s, x = read_xyz(tmp_path / "opt.xyz")
    assert np.isclose(x[1, 2] - x[0, 2], 1.21)


def test_qe_parse_unconverged(tmp_path):
    (tmp_path / "relax.out").write_text("!    total energy = -1.0 Ry\n"
                                        "The maximum number of steps has been reached.\n   JOB DONE.\n")
    assert qe_engine.parse(tmp_path, {"n_atoms": 2})["status"] == "opt_not_converged"


@pytest.mark.parametrize("engine", ["orca", "qe"])
def test_collect_eads_arithmetic(cfg, tmp_path, engine):
    """E_ads = min(complex) - min(host) - anion, from fake outputs of either engine."""
    wd = Path(cfg["workdir"])
    dd = wd / "dft"
    cfg["competitors"] = []
    import pandas as pd
    wd.mkdir(parents=True)
    pd.DataFrame([{"id": "h1", "status": "ok", "smiles": "C", "charge": 0}]).to_csv(
        wd / "molecules_prepared.csv", index=False)
    energies = {("refs", "TFA"): -100.0, ("hosts/h1", "host_0"): -50.0, ("hosts/h1", "host_1"): -50.01,
                ("hosts/h1", "TFA_0"): -150.02, ("hosts/h1", "TFA_1"): -150.03}
    roles = {"TFA": {"role": "anion", "anion": "TFA"}, "host_0": {"role": "host", "host": "h1"},
             "host_1": {"role": "host", "host": "h1"},
             "TFA_0": {"role": "complex", "host": "h1", "anion": "TFA"},
             "TFA_1": {"role": "complex", "host": "h1", "anion": "TFA"}}
    for (sub, name), e in energies.items():
        jd = dd / sub / name
        jd.mkdir(parents=True)
        write_xyz(jd / "input.xyz", *XYZ)
        (jd / "meta.json").write_text(json.dumps(dict(roles[name], engine=engine, n_atoms=2)))
        if engine == "orca":
            (jd / "opt.out").write_text("THE OPTIMIZATION HAS CONVERGED\nORCA TERMINATED NORMALLY\n")
            shutil.copy(jd / "input.xyz", jd / "opt.xyz")
            (jd / "sp.out").write_text(f"FINAL SINGLE POINT ENERGY {e}\nORCA TERMINATED NORMALLY\n")
        else:
            (jd / "relax.out").write_text(
                f"!    total energy              =     {2 * e} Ry\nbfgs converged\n"
                "End of BFGS Geometry Optimization\nBegin final coordinates\nATOMIC_POSITIONS (angstrom)\n"
                "C 0 0 0\nO 0 0 1.13\nEnd final coordinates\nJOB DONE.\n")
    res = dft.run_dft_collect(cfg)
    expected = (-150.03 - (-50.01) - (-100.0)) * HARTREE_TO_KCAL
    assert res.loc[0, "Eads_dft_TFA"] == pytest.approx(expected, abs=0.01)
    assert res.loc[0, "dft_status"] == "ok"
